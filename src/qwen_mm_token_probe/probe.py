from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .hf_qwen import (
    decode_token_piece,
    display_token,
    encode_generated_text,
    generate_from_prompt,
    load_model_bundle,
    prepare_prompt_inputs,
    token_statistics_for_generated_ids,
    token_statistics_for_text_only_ids,
)
from .image_mask import (
    MaskConfig,
    apply_image_mask,
    load_rgb_image,
    resize_rgb_image,
    save_rgb_image,
)
from .token_grouping import WordScore, group_token_scores


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
    top_token_id_original: int
    top_token_original: str
    top_raw_token_original: str
    top_p_original: float
    top_logp_original: float
    top_token_id_masked: int
    top_token_masked: str
    top_raw_token_masked: str
    top_p_masked: float
    top_logp_masked: float

    @property
    def delta_p(self) -> float:
        return self.p_original - self.p_masked

    @property
    def delta_logp(self) -> float:
        return self.logp_original - self.logp_masked

    @property
    def top_token_changed(self) -> bool:
        return self.top_token_id_original != self.top_token_id_masked

    @property
    def target_is_top_original(self) -> bool:
        return self.token_id == self.top_token_id_original

    @property
    def target_is_top_masked(self) -> bool:
        return self.token_id == self.top_token_id_masked

    @property
    def compact_token(self) -> str:
        token = self.token
        token = token.replace(" ", "·")
        if len(token) > 12:
            return token[:11] + "…"
        return token

    def to_dict(self) -> dict[str, bool | float | int | str]:
        data = asdict(self)
        data["delta_p"] = self.delta_p
        data["delta_logp"] = self.delta_logp
        data["top_token_changed"] = self.top_token_changed
        data["target_is_top_original"] = self.target_is_top_original
        data["target_is_top_masked"] = self.target_is_top_masked
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
    original_image_path: Path | None
    masked_image_path: Path | None
    mask_metadata: dict[str, object]
    condition_image_metadata: dict[str, object]
    original_condition_label: str = "original image"
    masked_condition_label: str = "masked image"
    privileged_info_metadata: dict[str, object] | None = None
    fixed_response_metadata: dict[str, object] | None = None
    text_only_forward_metadata: dict[str, object] | None = None

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
            "original_image_path": _optional_path_str(self.original_image_path),
            "masked_image_path": _optional_path_str(self.masked_image_path),
            "original_condition_label": self.original_condition_label,
            "masked_condition_label": self.masked_condition_label,
            "privileged_info": self.privileged_info_metadata,
            "fixed_response": self.fixed_response_metadata,
            "text_only_forward": self.text_only_forward_metadata,
            "mask_metadata": self.mask_metadata,
            "condition_image_metadata": self.condition_image_metadata,
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
    score_response_file: str | Path | None = None,
    compare_text_only_forward: bool = False,
    condition_image_scale: float = 1.0,
    skip_masked_generation: bool = False,
) -> ProbeResult:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    original_image = load_rgb_image(image_path)
    original_out = save_rgb_image(original_image, output_root / "original.png")
    masked_out = None
    mask_metadata = None
    condition_image_metadata: dict[str, object] = {}

    bundle = load_model_bundle(
        model_id,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    masked_prompt = prompt
    masked_condition_label = "text only forward" if compare_text_only_forward else "masked image"
    privileged_info_metadata = None
    if not compare_text_only_forward:
        masked_image, mask_metadata = apply_image_mask(original_image, mask_config)
        condition_image = resize_rgb_image(masked_image, condition_image_scale)
        masked_out = save_rgb_image(condition_image, output_root / "masked.png")
        condition_image_metadata = {
            "scale": condition_image_scale,
            "pre_scale_width": masked_image.size[0],
            "pre_scale_height": masked_image.size[1],
            "output_width": condition_image.size[0],
            "output_height": condition_image.size[1],
            "path": str(masked_out),
        }
        masked_condition_label = _condition_label(
            mask_config=mask_config,
            condition_image_scale=condition_image_scale,
            has_privileged_info=False,
        )
        if privileged_info_file is not None:
            privileged_info_path, privileged_info = _load_privileged_info(privileged_info_file)
            masked_prompt = _append_privileged_info(
                prompt=prompt,
                privileged_info=privileged_info,
            )
            masked_condition_label = _condition_label(
                mask_config=mask_config,
                condition_image_scale=condition_image_scale,
                has_privileged_info=True,
            )
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
    masked_prompt_inputs = None
    if masked_out is not None:
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

    fixed_response_metadata = None
    original_response_label = "original_image_response"
    original_response_source = "original"
    if score_response_file is not None:
        response_path, original_generated_text = _load_text_file(score_response_file)
        original_generated_token_ids = encode_generated_text(
            bundle.tokenizer,
            original_generated_text,
        )
        if not original_generated_token_ids:
            raise RuntimeError(f"score response file has no scoreable text tokens: {response_path}")
        original_response_label = "score_response_file"
        original_response_source = f"response_file:{response_path}"
        fixed_response_metadata = {
            "path": str(response_path),
            "num_chars": len(original_generated_text),
            "num_tokens": len(original_generated_token_ids),
            "sha256": hashlib.sha256(original_generated_text.encode("utf-8")).hexdigest(),
            "applied_to": "original_response",
        }
    else:
        original_generated_token_ids, original_generated_text = generate_from_prompt(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            prompt_inputs=original_prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    text_only_forward_metadata = None
    if compare_text_only_forward:
        original_response, text_only_forward_metadata = _build_response_probe_with_text_only_comparison(
            label=original_response_label,
            source_image=original_response_source,
            generated_text=original_generated_text,
            generated_token_ids=original_generated_token_ids,
            tokenizer=bundle.tokenizer,
            model=bundle.model,
            original_prompt_inputs=original_prompt_inputs,
            group_tokens=group_tokens,
            device=bundle.device,
        )
        masked_condition_label = "text only forward"
    else:
        if masked_prompt_inputs is None:
            raise RuntimeError("masked prompt inputs are missing for image-conditioned comparison")
        original_response = _build_response_probe(
            label=original_response_label,
            source_image=original_response_source,
            generated_text=original_generated_text,
            generated_token_ids=original_generated_token_ids,
            tokenizer=bundle.tokenizer,
            model=bundle.model,
            original_prompt_inputs=original_prompt_inputs,
            masked_prompt_inputs=masked_prompt_inputs,
            group_tokens=group_tokens,
        )

    masked_response = None
    if (
        score_response_file is None
        and not compare_text_only_forward
        and not skip_masked_generation
    ):
        if masked_prompt_inputs is None:
            raise RuntimeError("masked prompt inputs are missing for masked response generation")
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
        mask_metadata={} if mask_metadata is None else mask_metadata.to_dict(),
        condition_image_metadata=condition_image_metadata,
        masked_condition_label=masked_condition_label,
        privileged_info_metadata=privileged_info_metadata,
        fixed_response_metadata=fixed_response_metadata,
        text_only_forward_metadata=text_only_forward_metadata,
    )


def run_text_only_probe(
    *,
    model_id: str,
    response_file: str | Path,
    output_dir: str | Path,
    group_tokens: str = "word",
    device_map: str | None = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> ProbeResult:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(
        model_id,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    response_path, response_text = _load_text_file(response_file)
    token_ids = encode_generated_text(bundle.tokenizer, response_text)
    if not token_ids:
        raise RuntimeError(f"score response file has no scoreable text tokens: {response_path}")

    score_token_ids, stats, prefix_metadata = token_statistics_for_text_only_ids(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        token_ids=token_ids,
        device=bundle.device,
    )
    response = _build_text_only_response_probe(
        label="text_only_response",
        source="response_file",
        text=response_text,
        token_ids=score_token_ids,
        tokenizer=bundle.tokenizer,
        stats=stats,
        group_tokens=group_tokens,
    )
    fixed_response_metadata = {
        "path": str(response_path),
        "num_chars": len(response_text),
        "num_tokens": len(token_ids),
        "num_scored_tokens": len(score_token_ids),
        "sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "applied_to": "text_only_forward",
        "text_only_prefix": prefix_metadata,
    }
    return ProbeResult(
        model_id=model_id,
        prompt="",
        original_response=response,
        masked_response=None,
        original_image_path=None,
        masked_image_path=None,
        mask_metadata={},
        condition_image_metadata={},
        original_condition_label="text only forward",
        masked_condition_label="text only forward copy",
        privileged_info_metadata=None,
        fixed_response_metadata=fixed_response_metadata,
    )


def _optional_path_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _condition_label(
    *,
    mask_config: MaskConfig,
    condition_image_scale: float,
    has_privileged_info: bool,
) -> str:
    scale_changed = condition_image_scale != 1.0
    if mask_config.ratio > 0.0 and scale_changed:
        label = f"masked image + scaled x{condition_image_scale:g}"
    elif mask_config.ratio > 0.0:
        label = "masked image"
    elif scale_changed:
        label = f"scaled image x{condition_image_scale:g}"
    else:
        label = "original image copy"

    if has_privileged_info:
        label += " + privileged info"
    return label


def _load_privileged_info(
    path: str | Path,
) -> tuple[Path, str]:
    return _load_text_file(path)


def _load_text_file(
    path: str | Path,
) -> tuple[Path, str]:
    info_path = Path(path).expanduser()
    info_path = info_path.resolve()
    if not info_path.exists():
        raise FileNotFoundError(f"text file not found: {info_path}")
    return info_path, info_path.read_text(encoding="utf-8")


def _append_privileged_info(
    *,
    prompt: str,
    privileged_info: str,
) -> str:
    return f"{prompt}\n\n{privileged_info}"


def _build_response_probe_with_text_only_comparison(
    *,
    label: str,
    source_image: str,
    generated_text: str,
    generated_token_ids: list[int],
    tokenizer,
    model,
    original_prompt_inputs: dict,
    group_tokens: str,
    device,
) -> tuple[ResponseProbe, dict[str, object]]:
    if not generated_token_ids:
        raise RuntimeError(f"{label} generated no scoreable text tokens")

    original_stats = token_statistics_for_generated_ids(
        model=model,
        prompt_inputs=original_prompt_inputs,
        generated_token_ids=generated_token_ids,
    )
    text_only_token_ids, text_only_stats, prefix_metadata = token_statistics_for_text_only_ids(
        model=model,
        tokenizer=tokenizer,
        token_ids=generated_token_ids,
        device=device,
    )
    if text_only_token_ids != generated_token_ids:
        raise RuntimeError(
            "text-only forward token alignment failed; tokenizer has no BOS/EOS prefix "
            "for scoring the first generated token"
        )

    token_scores = [
        TokenScore(
            index=i,
            token_id=token_id,
            token=display_token(tokenizer, token_id),
            raw_token=decode_token_piece(tokenizer, token_id),
            p_original=float(original_stats.probabilities[i]),
            p_masked=float(text_only_stats.probabilities[i]),
            logp_original=float(original_stats.log_probabilities[i]),
            logp_masked=float(text_only_stats.log_probabilities[i]),
            top_token_id_original=original_stats.top_token_ids[i],
            top_token_original=display_token(tokenizer, original_stats.top_token_ids[i]),
            top_raw_token_original=decode_token_piece(
                tokenizer,
                original_stats.top_token_ids[i],
            ),
            top_p_original=float(original_stats.top_probabilities[i]),
            top_logp_original=float(original_stats.top_log_probabilities[i]),
            top_token_id_masked=text_only_stats.top_token_ids[i],
            top_token_masked=display_token(tokenizer, text_only_stats.top_token_ids[i]),
            top_raw_token_masked=decode_token_piece(
                tokenizer,
                text_only_stats.top_token_ids[i],
            ),
            top_p_masked=float(text_only_stats.top_probabilities[i]),
            top_logp_masked=float(text_only_stats.top_log_probabilities[i]),
        )
        for i, token_id in enumerate(generated_token_ids)
    ]
    word_scores = group_token_scores(token_scores) if group_tokens == "word" else []
    metadata = {
        "applied_to": "comparison_condition",
        "meaning": (
            "p_masked/logp_masked are text-only autoregressive probabilities for "
            "the same generated response tokens."
        ),
        "num_tokens": len(generated_token_ids),
        "text_only_prefix": prefix_metadata,
    }
    return (
        ResponseProbe(
            label=label,
            source_image=source_image,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            token_scores=token_scores,
            word_scores=word_scores,
        ),
        metadata,
    )


def _build_text_only_response_probe(
    *,
    label: str,
    source: str,
    text: str,
    token_ids: list[int],
    tokenizer,
    stats,
    group_tokens: str,
) -> ResponseProbe:
    token_scores = [
        TokenScore(
            index=i,
            token_id=token_id,
            token=display_token(tokenizer, token_id),
            raw_token=decode_token_piece(tokenizer, token_id),
            p_original=float(stats.probabilities[i]),
            p_masked=float(stats.probabilities[i]),
            logp_original=float(stats.log_probabilities[i]),
            logp_masked=float(stats.log_probabilities[i]),
            top_token_id_original=stats.top_token_ids[i],
            top_token_original=display_token(tokenizer, stats.top_token_ids[i]),
            top_raw_token_original=decode_token_piece(tokenizer, stats.top_token_ids[i]),
            top_p_original=float(stats.top_probabilities[i]),
            top_logp_original=float(stats.top_log_probabilities[i]),
            top_token_id_masked=stats.top_token_ids[i],
            top_token_masked=display_token(tokenizer, stats.top_token_ids[i]),
            top_raw_token_masked=decode_token_piece(tokenizer, stats.top_token_ids[i]),
            top_p_masked=float(stats.top_probabilities[i]),
            top_logp_masked=float(stats.top_log_probabilities[i]),
        )
        for i, token_id in enumerate(token_ids)
    ]
    word_scores = group_token_scores(token_scores) if group_tokens == "word" else []
    return ResponseProbe(
        label=label,
        source_image=source,
        generated_text=text,
        generated_token_ids=token_ids,
        token_scores=token_scores,
        word_scores=word_scores,
    )


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

    original_stats = token_statistics_for_generated_ids(
        model=model,
        prompt_inputs=original_prompt_inputs,
        generated_token_ids=generated_token_ids,
    )
    masked_stats = token_statistics_for_generated_ids(
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
            p_original=float(original_stats.probabilities[i]),
            p_masked=float(masked_stats.probabilities[i]),
            logp_original=float(original_stats.log_probabilities[i]),
            logp_masked=float(masked_stats.log_probabilities[i]),
            top_token_id_original=original_stats.top_token_ids[i],
            top_token_original=display_token(tokenizer, original_stats.top_token_ids[i]),
            top_raw_token_original=decode_token_piece(
                tokenizer,
                original_stats.top_token_ids[i],
            ),
            top_p_original=float(original_stats.top_probabilities[i]),
            top_logp_original=float(original_stats.top_log_probabilities[i]),
            top_token_id_masked=masked_stats.top_token_ids[i],
            top_token_masked=display_token(tokenizer, masked_stats.top_token_ids[i]),
            top_raw_token_masked=decode_token_piece(
                tokenizer,
                masked_stats.top_token_ids[i],
            ),
            top_p_masked=float(masked_stats.top_probabilities[i]),
            top_logp_masked=float(masked_stats.top_log_probabilities[i]),
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
