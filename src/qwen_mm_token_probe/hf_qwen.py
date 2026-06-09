from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def build_user_messages(image_path: str | Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path).resolve())},
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
) -> dict[str, Any]:
    """Tokenize a multimodal prompt with a generation marker.

    Newer Transformers multimodal processors accept local image paths under the
    `image` key. If a processor build expects an already-loaded PIL object, the
    fallback keeps the CLI usable across nearby versions.
    """

    messages = build_user_messages(image_path, prompt)
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as path_error:
        try:
            with Image.open(image_path) as raw_image:
                pil_image = raw_image.convert("RGB")
            fallback_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                fallback_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as pil_error:
            raise RuntimeError(
                "failed to prepare multimodal prompt with both local path and "
                f"PIL image. Local-path attempt failed with: {path_error}"
            ) from pil_error

    return move_inputs_to_device(inputs, device)


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
    generated_text = tokenizer.decode(score_ids, skip_special_tokens=True).strip()
    return score_ids, generated_text


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
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    scoring_inputs = append_generated_tokens(prompt_inputs, generated_token_ids)
    target_ids = torch.tensor(generated_token_ids, device=scoring_inputs["input_ids"].device)

    with torch.inference_mode():
        outputs = model(**scoring_inputs)

    logits = outputs.logits[0, prompt_len - 1 : prompt_len - 1 + len(generated_token_ids)]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    selected_log_probs = log_probs.gather(1, target_ids[:, None]).squeeze(1)
    selected_probs = selected_log_probs.exp()

    return (
        selected_probs.detach().cpu().tolist(),
        selected_log_probs.detach().cpu().tolist(),
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
