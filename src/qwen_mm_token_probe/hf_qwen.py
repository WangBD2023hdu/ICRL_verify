from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedTokenizerBase


@dataclass(frozen=True)
class ModelBundle:
    model_id: str
    model: torch.nn.Module
    processor: Any
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


@dataclass(frozen=True)
class GeneratedTokenStats:
    probabilities: list[float]
    log_probabilities: list[float]
    top_token_ids: list[int]
    top_probabilities: list[float]
    top_log_probabilities: list[float]


def load_model_bundle(
    model_id: str,
    *,
    device_map: str | None = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> ModelBundle:
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if dtype:
        kwargs["dtype"] = dtype

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor does not expose a tokenizer")

    return ModelBundle(
        model_id=model_id,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=_infer_input_device(model),
    )


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    return next(model.parameters()).device


def build_user_messages(
    image: str | Path | Image.Image,
    prompt: str,
    *,
    min_pixels: int = 2048,
    max_pixels: int = 16777216,
) -> list[dict[str, Any]]:
    image_value: str | Image.Image
    if isinstance(image, Image.Image):
        image_value = image
    else:
        image_value = str(Path(image).expanduser().resolve())

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_value,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_prompt_inputs(
    *,
    processor: Any,
    image_path: str | Path,
    prompt: str,
    device: torch.device,
    min_pixels: int = 2048,
    max_pixels: int = 16777216,
    image_patch_size: int = 16,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """Tokenize a multimodal prompt with a generation marker.

    Infinity-Parser2 uses Qwen's vision utility path: PIL RGB image ->
    `apply_chat_template(tokenize=False)` -> `process_vision_info` ->
    `processor(..., do_resize=False)`. Using the same path keeps image patching
    and resize behavior aligned with the model's reference inference code.
    """

    try:
        inputs = prepare_qwen_vl_prompt_inputs(
            processor=processor,
            image_path=image_path,
            prompt=prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_patch_size=image_patch_size,
            enable_thinking=enable_thinking,
        )
    except ImportError as import_error:
        warnings.warn(
            "qwen-vl-utils is not available; falling back to processor.apply_chat_template("
            "tokenize=True). Install qwen-vl-utils>=0.0.14 for Infinity-Parser2-compatible "
            f"image preprocessing. Original import error: {import_error}",
            RuntimeWarning,
        )
        inputs = prepare_legacy_prompt_inputs(
            processor=processor,
            image_path=image_path,
            prompt=prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            enable_thinking=enable_thinking,
        )

    return move_inputs_to_device(inputs, device)


def prepare_qwen_vl_prompt_inputs(
    *,
    processor: Any,
    image_path: str | Path,
    prompt: str,
    min_pixels: int = 2048,
    max_pixels: int = 16777216,
    image_patch_size: int = 16,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    with Image.open(image_path) as raw_image:
        pil_image = raw_image.convert("RGB")

    messages = build_user_messages(
        pil_image,
        prompt,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    batch_messages = [messages]
    chat_template_kwargs = {"enable_thinking": enable_thinking}

    try:
        text = processor.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
    except TypeError:
        text = processor.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    image_inputs, _ = process_vision_info(batch_messages, image_patch_size=image_patch_size)
    inputs = processor(
        text=text,
        images=image_inputs,
        do_resize=False,
        padding=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs


def prepare_legacy_prompt_inputs(
    *,
    processor: Any,
    image_path: str | Path,
    prompt: str,
    min_pixels: int = 2048,
    max_pixels: int = 16777216,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    messages = build_user_messages(
        image_path,
        prompt,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    chat_template_kwargs = {"enable_thinking": enable_thinking}
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **chat_template_kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )


def move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def generate_from_prompt(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_inputs: dict[str, Any],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
) -> tuple[list[int], str]:
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample and temperature is not None:
        generation_kwargs["temperature"] = temperature
    if do_sample and top_p is not None:
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**prompt_inputs, **generation_kwargs)

    generated_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
    score_ids = trim_tail_special_tokens(generated_ids, tokenizer)
    generated_text = tokenizer.decode(
        score_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return score_ids, generated_text


def generate_from_prefilled_tokens(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_inputs: dict[str, Any],
    generated_prefix_token_ids: list[int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
) -> tuple[list[int], str, list[int], str]:
    """Continue generation after already-emitted assistant tokens.

    `generated_prefix_token_ids` are appended to the multimodal prompt as fixed
    context. `model.generate` then produces only the continuation after that
    fixed prefix.
    """

    prefilled_inputs = append_generated_tokens(prompt_inputs, generated_prefix_token_ids)
    prompt_plus_prefix_len = int(prefilled_inputs["input_ids"].shape[-1])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample and temperature is not None:
        generation_kwargs["temperature"] = temperature
    if do_sample and top_p is not None:
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**prefilled_inputs, **generation_kwargs)

    continuation_ids = output_ids[0, prompt_plus_prefix_len:].detach().cpu().tolist()
    continuation_ids = trim_tail_special_tokens(continuation_ids, tokenizer)
    full_ids = generated_prefix_token_ids + continuation_ids
    continuation_text = decode_generated_tokens(tokenizer, continuation_ids).strip()
    full_text = decode_generated_tokens(tokenizer, full_ids).strip()
    return continuation_ids, continuation_text, full_ids, full_text


def next_token_topk(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_inputs: dict[str, Any],
    generated_prefix_token_ids: list[int],
    top_k: int = 10,
    inspect_token_ids: list[int] | None = None,
) -> dict[str, object]:
    prefilled_inputs = append_generated_tokens(prompt_inputs, generated_prefix_token_ids)

    with torch.inference_mode():
        outputs = model(**prefilled_inputs)

    logits = outputs.logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k)

    top_tokens = [
        {
            "rank": rank + 1,
            "token_id": int(token_id),
            "token": display_token(tokenizer, int(token_id)),
            "raw_token": decode_token_piece(tokenizer, int(token_id)),
            "probability": float(prob),
        }
        for rank, (token_id, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()))
    ]

    inspected = []
    for token_id in inspect_token_ids or []:
        inspected.append(
            {
                "token_id": int(token_id),
                "token": display_token(tokenizer, int(token_id)),
                "raw_token": decode_token_piece(tokenizer, int(token_id)),
                "probability": float(probs[int(token_id)].detach().cpu()),
            }
        )

    return {
        "top_tokens": top_tokens,
        "inspected_tokens": inspected,
    }


def decode_generated_tokens(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def trim_tail_special_tokens(
    token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    special_ids = set(tokenizer.all_special_ids or [])
    last = len(token_ids)
    while last > 0 and token_ids[last - 1] in special_ids:
        last -= 1
    return token_ids[:last]


def append_generated_tokens(
    prompt_inputs: dict[str, Any],
    generated_token_ids: list[int],
) -> dict[str, Any]:
    if not generated_token_ids:
        raise ValueError("no generated tokens to score")

    input_ids = prompt_inputs["input_ids"]
    device = input_ids.device
    generated = torch.tensor([generated_token_ids], dtype=input_ids.dtype, device=device)

    scoring_inputs: dict[str, Any] = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            scoring_inputs[key] = value.clone()
        else:
            scoring_inputs[key] = value

    scoring_inputs["input_ids"] = torch.cat([input_ids, generated], dim=-1)

    if "attention_mask" in scoring_inputs:
        attention = scoring_inputs["attention_mask"]
        extension = torch.ones(
            (attention.shape[0], len(generated_token_ids)),
            dtype=attention.dtype,
            device=attention.device,
        )
        scoring_inputs["attention_mask"] = torch.cat([attention, extension], dim=-1)

    # Some processors emit type ids. Text generated after the prompt belongs to
    # the regular text stream, so appending zeros is the least surprising choice.
    for optional_key in ("token_type_ids", "mm_token_type_ids"):
        if optional_key in scoring_inputs:
            token_types = scoring_inputs[optional_key]
            extension = torch.zeros(
                (token_types.shape[0], len(generated_token_ids)),
                dtype=token_types.dtype,
                device=token_types.device,
            )
            scoring_inputs[optional_key] = torch.cat([token_types, extension], dim=-1)

    return scoring_inputs


def token_probabilities_for_generated_ids(
    *,
    model: torch.nn.Module,
    prompt_inputs: dict[str, Any],
    generated_token_ids: list[int],
) -> tuple[list[float], list[float]]:
    stats = token_statistics_for_generated_ids(
        model=model,
        prompt_inputs=prompt_inputs,
        generated_token_ids=generated_token_ids,
    )
    return stats.probabilities, stats.log_probabilities


def token_statistics_for_generated_ids(
    *,
    model: torch.nn.Module,
    prompt_inputs: dict[str, Any],
    generated_token_ids: list[int],
) -> GeneratedTokenStats:
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    scoring_inputs = append_generated_tokens(prompt_inputs, generated_token_ids)
    target_ids = torch.tensor(generated_token_ids, device=scoring_inputs["input_ids"].device)

    with torch.inference_mode():
        outputs = model(**scoring_inputs)

    logits = outputs.logits[0, prompt_len - 1 : prompt_len - 1 + len(generated_token_ids)]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    selected_log_probs = log_probs.gather(1, target_ids[:, None]).squeeze(1)
    selected_probs = selected_log_probs.exp()
    top_log_probs, top_token_ids = torch.max(log_probs, dim=-1)
    top_probs = top_log_probs.exp()

    return GeneratedTokenStats(
        probabilities=selected_probs.detach().cpu().tolist(),
        log_probabilities=selected_log_probs.detach().cpu().tolist(),
        top_token_ids=[int(token_id) for token_id in top_token_ids.detach().cpu().tolist()],
        top_probabilities=top_probs.detach().cpu().tolist(),
        top_log_probabilities=top_log_probs.detach().cpu().tolist(),
    )


def display_token(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    piece = decode_token_piece(tokenizer, token_id)
    return piece.replace("\n", "\\n").replace("\t", "\\t")


def decode_token_piece(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    try:
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
    if piece == "":
        piece = tokenizer.convert_ids_to_tokens([token_id])[0]
    return piece
