from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .hf_qwen import (
    decode_token_piece,
    display_token,
    generate_from_prompt,
    load_model_bundle,
    prepare_prompt_inputs,
    token_probabilities_for_generated_ids,
)
from .image_mask import MaskConfig, apply_image_mask, load_rgb_image, save_rgb_image
from .token_grouping import WordScore, group_token_scores


DEFAULT_PRIVILEGED_INFO_TEMPLATE = """{prompt}

[Privileged information]
The following ground-truth answer is provided as privileged information for probability probing. Use it as reference information for the answer.

{privileged_info}

[End privileged information]
"""


@dataclass(frozen=True)
class TokenScore:
    index: int
    token_id: int
    token: str
    raw_token: str
    p_original: float
    p_masked: float
    logp_original: float
    logp_masked: float

    @property
    def delta_p(self) -> float:
        return self.p_original - self.p_masked

    @property
    def delta_logp(self) -> float:
        return self.logp_original - self.logp_masked

    @property
    def compact_token(self) -> str:
        token = self.token
        token = token.replace(" ", "·")
        if len(token) > 12:
            return token[:11] + "…"
        return token

    def to_dict(self) -> dict[str, float | int | str]:
        data = asdict(self)
        data["delta_p"] = self.delta_p
        data["delta_logp"] = self.delta_logp
        return data


@dataclass(frozen=True)
class ResponseProbe:
    label: str
    source_image: str
    generated_text: str
    generated_token_ids: list[int]
    token_scores: list[TokenScore]
    word_scores: list[WordScore]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "source_image": self.source_image,
            "generated_text": self.generated_text,
            "generated_token_ids": self.generated_token_ids,
            "token_scores": [score.to_dict() for score in self.token_scores],
            "word_scores": [score.to_dict() for score in self.word_scores],
        }


@dataclass(frozen=True)
class ProbeResult:
    model_id: str
    prompt: str
    original_response: ResponseProbe
    masked_response: ResponseProbe | None
    original_image_path: Path
    masked_image_path: Path
    mask_metadata: dict[str, object]
    original_condition_label: str = "original image"
    masked_condition_label: str = "masked image"
    privileged_info_metadata: dict[str, object] | None = None

    @property
    def generated_text(self) -> str:
        return self.original_response.generated_text

    @property
    def generated_token_ids(self) -> list[int]:
        return self.original_response.generated_token_ids

    @property
    def token_scores(self) -> list[TokenScore]:
        return self.original_response.token_scores

    @property
    def word_scores(self) -> list[WordScore]:
        return self.original_response.word_scores

    def to_json_payload(self) -> dict[str, object]:
        responses = {
            "original_image_response": self.original_response.to_dict(),
        }
        if self.masked_response is not None:
            responses["masked_image_response"] = self.masked_response.to_dict()

        payload: dict[str, object] = {
            "model_id": self.model_id,
            "prompt": self.prompt,
            "original_image_path": str(self.original_image_path),
            "masked_image_path": str(self.masked_image_path),
            "original_condition_label": self.original_condition_label,
            "masked_condition_label": self.masked_condition_label,
            "privileged_info": self.privileged_info_metadata,
            "mask_metadata": self.mask_metadata,
            "responses": responses,
            "generated_text": self.original_response.generated_text,
            "generated_token_ids": self.original_response.generated_token_ids,
            "token_scores": [score.to_dict() for score in self.token_scores],
            "word_scores": [score.to_dict() for score in self.word_scores],
        }
        if self.masked_response is not None:
            payload.update(
                {
                    "masked_generated_text": self.masked_response.generated_text,
                    "masked_generated_token_ids": self.masked_response.generated_token_ids,
                    "masked_token_scores": [
                        score.to_dict() for score in self.masked_response.token_scores
                    ],
                    "masked_word_scores": [
                        score.to_dict() for score in self.masked_response.word_scores
                    ],
                }
            )
        return payload


def run_probe(
    *,
    model_id: str,
    image_path: str | Path,
    prompt: str,
    output_dir: str | Path,
    mask_config: MaskConfig,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    group_tokens: str = "word",
    device_map: str | None = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    min_pixels: int = 2048,
    max_pixels: int = 16777216,
    image_patch_size: int = 16,
    enable_thinking: bool = False,
    privileged_info_file: str | Path | None = None,
    privileged_info_template: str = DEFAULT_PRIVILEGED_INFO_TEMPLATE,
    skip_masked_generation: bool = False,
) -> ProbeResult:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    original_image = load_rgb_image(image_path)
    masked_image, mask_metadata = apply_image_mask(original_image, mask_config)

    original_out = save_rgb_image(original_image, output_root / "original.png")
    masked_out = save_rgb_image(masked_image, output_root / "masked.png")

    bundle = load_model_bundle(
        model_id,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    masked_prompt = prompt
    masked_condition_label = "masked image"
    privileged_info_metadata = None
    if privileged_info_file is not None:
        privileged_info_path, privileged_info = _load_privileged_info(
            privileged_info_file,
            output_root=output_root,
        )
        masked_prompt = _format_privileged_prompt(
            prompt=prompt,
            privileged_info=privileged_info,
            template=privileged_info_template,
        )
        masked_condition_label = "masked image + privileged info"
        privileged_info_metadata = {
            "path": str(privileged_info_path),
            "num_chars": len(privileged_info),
            "sha256": hashlib.sha256(privileged_info.encode("utf-8")).hexdigest(),
            "applied_to": "masked_condition_prompt",
        }

    original_prompt_inputs = prepare_prompt_inputs(
        processor=bundle.processor,
        image_path=original_out,
        prompt=prompt,
        device=bundle.device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_patch_size=image_patch_size,
        enable_thinking=enable_thinking,
    )
    masked_prompt_inputs = prepare_prompt_inputs(
        processor=bundle.processor,
        image_path=masked_out,
        prompt=masked_prompt,
        device=bundle.device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_patch_size=image_patch_size,
        enable_thinking=enable_thinking,
    )

    original_generated_token_ids, original_generated_text = generate_from_prompt(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        prompt_inputs=original_prompt_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    original_response = _build_response_probe(
        label="original_image_response",
        source_image="original",
        generated_text=original_generated_text,
        generated_token_ids=original_generated_token_ids,
        tokenizer=bundle.tokenizer,
        model=bundle.model,
        original_prompt_inputs=original_prompt_inputs,
        masked_prompt_inputs=masked_prompt_inputs,
        group_tokens=group_tokens,
    )

    masked_response = None
    if not skip_masked_generation:
        masked_generated_token_ids, masked_generated_text = generate_from_prompt(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            prompt_inputs=masked_prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        masked_response = _build_response_probe(
            label="masked_image_response",
            source_image="masked",
            generated_text=masked_generated_text,
            generated_token_ids=masked_generated_token_ids,
            tokenizer=bundle.tokenizer,
            model=bundle.model,
            original_prompt_inputs=original_prompt_inputs,
            masked_prompt_inputs=masked_prompt_inputs,
            group_tokens=group_tokens,
        )

    return ProbeResult(
        model_id=model_id,
        prompt=prompt,
        original_response=original_response,
        masked_response=masked_response,
        original_image_path=original_out,
        masked_image_path=masked_out,
        mask_metadata=mask_metadata.to_dict(),
        masked_condition_label=masked_condition_label,
        privileged_info_metadata=privileged_info_metadata,
    )


def _load_privileged_info(
    path: str | Path,
    *,
    output_root: Path,
) -> tuple[Path, str]:
    info_path = Path(path).expanduser()
    if not info_path.is_absolute():
        info_path = output_root / info_path
    info_path = info_path.resolve()
    if not info_path.exists():
        raise FileNotFoundError(f"privileged info file not found: {info_path}")
    return info_path, info_path.read_text(encoding="utf-8")


def _format_privileged_prompt(
    *,
    prompt: str,
    privileged_info: str,
    template: str,
) -> str:
    try:
        return template.format(prompt=prompt, privileged_info=privileged_info)
    except KeyError as error:
        raise ValueError(
            "privileged info template must contain only {prompt} and {privileged_info}"
        ) from error


def _build_response_probe(
    *,
    label: str,
    source_image: str,
    generated_text: str,
    generated_token_ids: list[int],
    tokenizer,
    model,
    original_prompt_inputs: dict,
    masked_prompt_inputs: dict,
    group_tokens: str,
) -> ResponseProbe:
    if not generated_token_ids:
        raise RuntimeError(f"{label} generated no scoreable text tokens")

    p_original, logp_original = token_probabilities_for_generated_ids(
        model=model,
        prompt_inputs=original_prompt_inputs,
        generated_token_ids=generated_token_ids,
    )
    p_masked, logp_masked = token_probabilities_for_generated_ids(
        model=model,
        prompt_inputs=masked_prompt_inputs,
        generated_token_ids=generated_token_ids,
    )

    token_scores = [
        TokenScore(
            index=i,
            token_id=token_id,
            token=display_token(tokenizer, token_id),
            raw_token=decode_token_piece(tokenizer, token_id),
            p_original=float(p_original[i]),
            p_masked=float(p_masked[i]),
            logp_original=float(logp_original[i]),
            logp_masked=float(logp_masked[i]),
        )
        for i, token_id in enumerate(generated_token_ids)
    ]
    word_scores = group_token_scores(token_scores) if group_tokens == "word" else []
    return ResponseProbe(
        label=label,
        source_image=source_image,
        generated_text=generated_text,
        generated_token_ids=generated_token_ids,
        token_scores=token_scores,
        word_scores=word_scores,
    )
