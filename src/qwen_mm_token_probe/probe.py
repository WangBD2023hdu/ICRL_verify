from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .hf_qwen import (
    display_token,
    generate_from_prompt,
    load_model_bundle,
    prepare_prompt_inputs,
    token_probabilities_for_generated_ids,
)
from .image_mask import MaskConfig, apply_image_mask, load_rgb_image, save_rgb_image


@dataclass(frozen=True)
class TokenScore:
    index: int
    token_id: int
    token: str
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
class ProbeResult:
    model_id: str
    prompt: str
    generated_text: str
    generated_token_ids: list[int]
    token_scores: list[TokenScore]
    original_image_path: Path
    masked_image_path: Path
    mask_metadata: dict[str, object]

    def to_json_payload(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "generated_token_ids": self.generated_token_ids,
            "original_image_path": str(self.original_image_path),
            "masked_image_path": str(self.masked_image_path),
            "mask_metadata": self.mask_metadata,
            "token_scores": [score.to_dict() for score in self.token_scores],
        }


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
    device_map: str | None = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
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

    original_prompt_inputs = prepare_prompt_inputs(
        processor=bundle.processor,
        image_path=original_out,
        prompt=prompt,
        device=bundle.device,
    )
    generated_token_ids, generated_text = generate_from_prompt(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        prompt_inputs=original_prompt_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    if not generated_token_ids:
        raise RuntimeError("model generated no scoreable text tokens")

    masked_prompt_inputs = prepare_prompt_inputs(
        processor=bundle.processor,
        image_path=masked_out,
        prompt=prompt,
        device=bundle.device,
    )

    p_original, logp_original = token_probabilities_for_generated_ids(
        model=bundle.model,
        prompt_inputs=original_prompt_inputs,
        generated_token_ids=generated_token_ids,
    )
    p_masked, logp_masked = token_probabilities_for_generated_ids(
        model=bundle.model,
        prompt_inputs=masked_prompt_inputs,
        generated_token_ids=generated_token_ids,
    )

    token_scores = [
        TokenScore(
            index=i,
            token_id=token_id,
            token=display_token(bundle.tokenizer, token_id),
            p_original=float(p_original[i]),
            p_masked=float(p_masked[i]),
            logp_original=float(logp_original[i]),
            logp_masked=float(logp_masked[i]),
        )
        for i, token_id in enumerate(generated_token_ids)
    ]

    return ProbeResult(
        model_id=model_id,
        prompt=prompt,
        generated_text=generated_text,
        generated_token_ids=generated_token_ids,
        token_scores=token_scores,
        original_image_path=original_out,
        masked_image_path=masked_out,
        mask_metadata=mask_metadata.to_dict(),
    )
