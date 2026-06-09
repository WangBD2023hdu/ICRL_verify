from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen-style multimodal inference, then compare generated-token "
            "probabilities under original and masked/degraded images."
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

    masking = parser.add_argument_group("image masking and degradation")
    masking.add_argument(
        "--mask-strategy",
        choices=["patch", "word"],
        default="patch",
        help="Select regions by random patches or word-level boxes.",
    )
    masking.add_argument("--mask-ratio", type=float, default=0.35)
    masking.add_argument("--patch-size", type=int, default=32)
    masking.add_argument(
        "--mask-fill",
        choices=["mean", "black", "white", "noise"],
        default="mean",
    )
    masking.add_argument(
        "--mask-effect",
        choices=["replace", "fade", "blur", "noise", "blur_fade"],
        default="replace",
        help="How selected regions are degraded.",
    )
    masking.add_argument(
        "--mask-opacity",
        type=float,
        default=1.0,
        help="Blend strength for the degradation. 1.0 is full strength, 0.0 is no change.",
    )
    masking.add_argument("--blur-radius", type=float, default=1.2)
    masking.add_argument("--noise-std", type=float, default=10.0)
    masking.add_argument(
        "--word-boxes",
        default=None,
        help=(
            "Optional JSON file containing word boxes. Supported shapes include "
            "[x1,y1,x2,y2], {'bbox': [...]}, {'box': [...]}, or {'x','y','w','h'}."
        ),
    )
    masking.add_argument("--word-padding", type=int, default=2)
    masking.add_argument("--word-gap", type=int, default=12)
    masking.add_argument("--word-min-width", type=int, default=4)
    masking.add_argument("--word-min-height", type=int, default=4)
    masking.add_argument(
        "--text-threshold",
        type=int,
        default=None,
        help="Optional grayscale threshold for automatic word-like box detection.",
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
        strategy=args.mask_strategy,
        ratio=args.mask_ratio,
        patch_size=args.patch_size,
        fill=args.mask_fill,
        effect=args.mask_effect,
        opacity=args.mask_opacity,
        blur_radius=args.blur_radius,
        noise_std=args.noise_std,
        seed=args.seed,
        word_boxes_path=args.word_boxes,
        word_padding=args.word_padding,
        word_gap=args.word_gap,
        word_min_width=args.word_min_width,
        word_min_height=args.word_min_height,
        text_threshold=args.text_threshold,
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
