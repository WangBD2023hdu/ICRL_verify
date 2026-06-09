from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen-style multimodal inference, then compare generated-token "
            "probabilities under original and randomly masked images."
        )
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="User prompt for the image.")
    parser.add_argument("--output-dir", default="outputs/qwen_probe")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")

    sampling = parser.add_argument_group("generation sampling")
    sampling.add_argument("--do-sample", action="store_true")
    sampling.add_argument("--temperature", type=float, default=None)
    sampling.add_argument("--top-p", type=float, default=None)

    masking = parser.add_argument_group("random image masking")
    masking.add_argument("--mask-ratio", type=float, default=0.35)
    masking.add_argument("--patch-size", type=int, default=32)
    masking.add_argument(
        "--mask-fill",
        choices=["mean", "black", "white", "noise"],
        default="mean",
    )
    masking.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from .image_mask import MaskConfig
    from .probe import run_probe
    from .visualize import (
        write_generated_text,
        write_html_report,
        write_probability_plot,
        write_scores_csv,
        write_scores_json,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_config = MaskConfig(
        ratio=args.mask_ratio,
        patch_size=args.patch_size,
        fill=args.mask_fill,
        seed=args.seed,
    )
    result = run_probe(
        model_id=args.model_id,
        image_path=args.image,
        prompt=args.prompt,
        output_dir=output_dir,
        mask_config=mask_config,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    payload = result.to_json_payload()
    write_generated_text(output_dir / "generated.txt", result.generated_text)
    write_scores_csv(output_dir / "token_probabilities.csv", result.token_scores)
    write_scores_json(output_dir / "token_probabilities.json", payload)
    write_probability_plot(output_dir / "token_probabilities.png", result.token_scores)
    write_html_report(
        output_dir / "token_probabilities.html",
        model_id=result.model_id,
        prompt=result.prompt,
        generated_text=result.generated_text,
        scores=result.token_scores,
        metadata={
            "original_image_path": str(result.original_image_path),
            "masked_image_path": str(result.masked_image_path),
            "mask_metadata": result.mask_metadata,
            "num_generated_tokens": len(result.generated_token_ids),
        },
    )

    print(f"Generated {len(result.generated_token_ids)} tokens.")
    print(f"Output directory: {output_dir}")
    print(f"HTML report: {output_dir / 'token_probabilities.html'}")


if __name__ == "__main__":
    main()
