from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedTokenizerBase,
)


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


def collate_prompt_inputs(
    prompt_inputs: Sequence[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Left-pad prepared Qwen image-prompt inputs into one generation batch.

    Images have already been opened, resized/patchified, and processed by the
    existing single-sample preparation path. This function only joins those
    tensors: image patch tensors and image-grid rows are concatenated in sample
    order, while sequence-aligned tensors are left-padded to the longest prompt.
    """

    if not prompt_inputs:
        raise ValueError("cannot collate an empty prompt batch")

    supported_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "mm_token_type_ids",
        "pixel_values",
        "image_grid_thw",
    }
    present_keys = set().union(*(set(inputs) for inputs in prompt_inputs))
    unsupported_keys = sorted(present_keys - supported_keys)
    if unsupported_keys:
        raise ValueError(
            "unsupported prepared prompt input key(s) for batching: "
            + ", ".join(unsupported_keys)
        )
    if any("input_ids" not in inputs for inputs in prompt_inputs):
        raise ValueError("every prepared prompt input must contain input_ids")

    def as_single_sequence(inputs: dict[str, Any], key: str, index: int) -> torch.Tensor:
        value = inputs.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"prompt {index} {key} must be a tensor")
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"prompt {index} {key} must have shape [1, sequence_length]")
        return value

    input_rows = [as_single_sequence(inputs, "input_ids", i) for i, inputs in enumerate(prompt_inputs)]
    if any(row.shape[1] == 0 for row in input_rows):
        raise ValueError("prepared prompt input_ids must not be empty")
    reference_device = input_rows[0].device
    if any(row.device != reference_device for row in input_rows):
        raise ValueError("all prepared prompt tensors must be on the same device")

    lengths = [int(row.shape[1]) for row in input_rows]
    max_length = max(lengths)
    batch_input_ids = torch.full(
        (len(input_rows), max_length),
        int(pad_token_id),
        dtype=input_rows[0].dtype,
        device=reference_device,
    )
    batch_attention_mask = torch.zeros(
        (len(input_rows), max_length),
        dtype=torch.long,
        device=reference_device,
    )

    has_attention_masks = ["attention_mask" in inputs for inputs in prompt_inputs]
    if any(has_attention_masks) and not all(has_attention_masks):
        raise ValueError("attention_mask must be present for every prompt or none")

    sequence_optional: dict[str, list[torch.Tensor]] = {}
    for key in ("token_type_ids", "mm_token_type_ids"):
        present = [key in inputs for inputs in prompt_inputs]
        if any(present) and not all(present):
            raise ValueError(f"{key} must be present for every prompt or none")
        if all(present):
            sequence_optional[key] = [
                as_single_sequence(inputs, key, i) for i, inputs in enumerate(prompt_inputs)
            ]

    for i, (inputs, input_row, length) in enumerate(zip(prompt_inputs, input_rows, lengths)):
        offset = max_length - length
        batch_input_ids[i, offset:] = input_row[0]

        if has_attention_masks[i]:
            attention = as_single_sequence(inputs, "attention_mask", i)
            if attention.shape[1] != length:
                raise ValueError(f"prompt {i} attention_mask length does not match input_ids")
            if attention.device != reference_device:
                raise ValueError("all prepared prompt tensors must be on the same device")
            batch_attention_mask[i, offset:] = attention[0].to(dtype=batch_attention_mask.dtype)
        else:
            batch_attention_mask[i, offset:] = 1

    batched: dict[str, torch.Tensor] = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
    }
    for key, rows in sequence_optional.items():
        padded = torch.zeros(
            (len(rows), max_length),
            dtype=rows[0].dtype,
            device=reference_device,
        )
        for i, (row, length) in enumerate(zip(rows, lengths)):
            if row.shape[1] != length:
                raise ValueError(f"prompt {i} {key} length does not match input_ids")
            if row.device != reference_device:
                raise ValueError("all prepared prompt tensors must be on the same device")
            padded[i, max_length - length :] = row[0]
        batched[key] = padded

    image_key_presence = {
        key: [key in inputs for inputs in prompt_inputs]
        for key in ("pixel_values", "image_grid_thw")
    }
    for key, present in image_key_presence.items():
        if any(present) and not all(present):
            raise ValueError(f"{key} must be present for every prompt or none")
    if any(image_key_presence["pixel_values"]) != any(image_key_presence["image_grid_thw"]):
        raise ValueError("pixel_values and image_grid_thw must be provided together")

    if all(image_key_presence["pixel_values"]):
        pixel_values = [inputs["pixel_values"] for inputs in prompt_inputs]
        image_grids = [inputs["image_grid_thw"] for inputs in prompt_inputs]
        if any(not torch.is_tensor(value) or value.ndim < 1 for value in pixel_values):
            raise ValueError("pixel_values must be non-scalar tensors")
        if any(not torch.is_tensor(value) for value in image_grids):
            raise ValueError("image_grid_thw must be tensors")
        normalized_grids = [
            grid.reshape(1, -1) if grid.ndim == 1 else grid
            for grid in image_grids
        ]
        if any(grid.ndim != 2 or grid.shape[1] != 3 for grid in normalized_grids):
            raise ValueError("image_grid_thw tensors must have shape [images, 3]")
        all_image_tensors = pixel_values + normalized_grids
        if any(value.device != reference_device for value in all_image_tensors):
            raise ValueError("all prepared prompt tensors must be on the same device")
        batched["pixel_values"] = torch.cat(pixel_values, dim=0)
        batched["image_grid_thw"] = torch.cat(normalized_grids, dim=0)

    return batched


def _configured_token_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(token_id) for token_id in value}


def _trim_generated_batch_tail(
    token_ids: list[int],
    *,
    eos_token_ids: set[int],
    pad_token_id: int | None,
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            token_ids = token_ids[:index]
            break
    if pad_token_id is not None:
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids = token_ids[:-1]
    return trim_tail_special_tokens(token_ids, tokenizer)


def generate_batch_from_prompts(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_inputs: Sequence[dict[str, Any]],
    max_new_tokens: int,
) -> list[tuple[list[int], str]]:
    """Greedily generate one response per prepared prompt in a single call."""

    generation_config = getattr(model, "generation_config", None)
    pad_token_id = getattr(generation_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("batch generation requires a configured tokenizer/model pad_token_id")

    batch_inputs = collate_prompt_inputs(prompt_inputs, pad_token_id=int(pad_token_id))
    prompt_width = int(batch_inputs["input_ids"].shape[1])
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    with torch.inference_mode():
        output_ids = model.generate(**batch_inputs, **generation_kwargs)

    if not torch.is_tensor(output_ids) or output_ids.ndim != 2:
        raise ValueError("model.generate must return a [batch, sequence_length] token tensor")
    if output_ids.shape[0] != len(prompt_inputs) or output_ids.shape[1] < prompt_width:
        raise ValueError("model.generate returned an unexpected batch or sequence shape")

    configured_eos_ids = getattr(generation_config, "eos_token_id", None)
    if configured_eos_ids is None:
        configured_eos_ids = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = _configured_token_ids(configured_eos_ids)
    decoded: list[tuple[list[int], str]] = []
    for row in output_ids[:, prompt_width:].detach().cpu().tolist():
        response_ids = _trim_generated_batch_tail(
            [int(token_id) for token_id in row],
            eos_token_ids=eos_token_ids,
            pad_token_id=int(pad_token_id),
            tokenizer=tokenizer,
        )
        response_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        decoded.append((response_ids, response_text))
    return decoded


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


def encode_generated_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded["input_ids"]

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


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


def token_statistics_for_text_only_ids(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
    device: torch.device,
) -> tuple[list[int], GeneratedTokenStats, dict[str, int | str | None]]:
    if not token_ids:
        raise ValueError("no text tokens to score")

    prefix_token_id = _text_only_prefix_token_id(tokenizer)
    if prefix_token_id is None:
        if len(token_ids) < 2:
            raise ValueError("at least two text tokens are required when no BOS/EOS prefix exists")
        input_token_ids = token_ids
        target_token_ids = token_ids[1:]
        logit_start = 0
        logit_end = len(target_token_ids)
        prefix_metadata = {"prefix_token_id": None, "prefix_token": None}
    else:
        input_token_ids = [prefix_token_id] + token_ids
        target_token_ids = token_ids
        logit_start = 0
        logit_end = len(target_token_ids)
        prefix_metadata = {
            "prefix_token_id": prefix_token_id,
            "prefix_token": decode_token_piece(tokenizer, prefix_token_id),
        }

    input_ids = torch.tensor([input_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    target_ids = torch.tensor(target_token_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[0, logit_start:logit_end]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    selected_log_probs = log_probs.gather(1, target_ids[:, None]).squeeze(1)
    selected_probs = selected_log_probs.exp()
    top_log_probs, top_ids = torch.max(log_probs, dim=-1)
    top_probs = top_log_probs.exp()

    return (
        target_token_ids,
        GeneratedTokenStats(
            probabilities=selected_probs.detach().cpu().tolist(),
            log_probabilities=selected_log_probs.detach().cpu().tolist(),
            top_token_ids=[int(token_id) for token_id in top_ids.detach().cpu().tolist()],
            top_probabilities=top_probs.detach().cpu().tolist(),
            top_log_probabilities=top_log_probs.detach().cpu().tolist(),
        ),
        prefix_metadata,
    )


def _text_only_prefix_token_id(tokenizer: PreTrainedTokenizerBase) -> int | None:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        return int(bos_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return None


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
