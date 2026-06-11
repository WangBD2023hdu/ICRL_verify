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
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=2048,
        help="Minimum image pixels for Qwen vision preprocessing.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=16777216,
        help="Maximum image pixels for Qwen vision preprocessing.",
    )
    parser.add_argument(
        "--image-patch-size",
        type=int,
        default=16,
        help="Patch size passed to qwen_vl_utils.process_vision_info.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass enable_thinking to the chat template. Disabled by default.",
    )

    sampling = parser.add_argument_group("generation sampling")
    sampling.add_argument("--do-sample", action="store_true")
    sampling.add_argument("--temperature", type=float, default=None)
    sampling.add_argument("--top-p", type=float, default=None)

    analysis = parser.add_argument_group("probability analysis")
    analysis.add_argument(
        "--group-tokens",
        choices=["none", "word"],
        default="word",
        help="Also aggregate subword tokens into word/text-unit scores.",
    )
    analysis.add_argument(
        "--privileged-info-file",
        default=None,
        help=(
            "Optional text file used only for the masked/degraded-image scoring "
            "condition. Relative paths are resolved against --output-dir, so "
            "GT.txt means <output-dir>/GT.txt."
        ),
    )
    analysis.add_argument(
        "--skip-masked-generation",
        action="store_true",
        help=(
            "Do not generate a separate response from the masked/degraded condition; "
            "only score the original-image response under both conditions."
        ),
    )

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
        write_word_html_report,
        write_word_scores_csv,
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
        group_tokens=args.group_tokens,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        image_patch_size=args.image_patch_size,
        enable_thinking=args.enable_thinking,
        privileged_info_file=args.privileged_info_file,
        skip_masked_generation=args.skip_masked_generation,
    )

    payload = result.to_json_payload()
    write_scores_json(output_dir / "token_probabilities.json", payload)

    def write_response_artifacts(
        *,
        response,
        generated_filename: str,
        token_csv_filename: str,
        token_plot_filename: str,
        token_html_filename: str,
        word_csv_filename: str,
        word_html_filename: str,
    ) -> None:
        metadata = {
            "response_label": response.label,
            "response_source_image": response.source_image,
            "original_image_path": str(result.original_image_path),
            "masked_image_path": str(result.masked_image_path),
            "original_condition_label": result.original_condition_label,
            "masked_condition_label": result.masked_condition_label,
            "privileged_info": result.privileged_info_metadata,
            "mask_metadata": result.mask_metadata,
            "num_generated_tokens": len(response.generated_token_ids),
            "num_word_units": len(response.word_scores),
            "score_meaning": (
                "p_original/logp_original score this fixed response under "
                f"{result.original_condition_label}; p_masked/logp_masked score the same "
                f"fixed response under {result.masked_condition_label}."
            ),
        }
        write_generated_text(output_dir / generated_filename, response.generated_text)
        write_scores_csv(output_dir / token_csv_filename, response.token_scores)
        write_probability_plot(
            output_dir / token_plot_filename,
            response.token_scores,
            original_label=result.original_condition_label,
            masked_label=result.masked_condition_label,
        )
        write_html_report(
            output_dir / token_html_filename,
            model_id=result.model_id,
            prompt=result.prompt,
            generated_text=response.generated_text,
            scores=response.token_scores,
            metadata=metadata,
            original_condition_label=result.original_condition_label,
            masked_condition_label=result.masked_condition_label,
        )
        if response.word_scores:
            write_word_scores_csv(output_dir / word_csv_filename, response.word_scores)
            write_word_html_report(
                output_dir / word_html_filename,
                model_id=result.model_id,
                prompt=result.prompt,
                generated_text=response.generated_text,
                scores=response.word_scores,
                metadata=metadata,
                original_condition_label=result.original_condition_label,
                masked_condition_label=result.masked_condition_label,
            )

    write_response_artifacts(
        response=result.original_response,
        generated_filename="generated.txt",
        token_csv_filename="token_probabilities.csv",
        token_plot_filename="token_probabilities.png",
        token_html_filename="token_probabilities.html",
        word_csv_filename="word_probabilities.csv",
        word_html_filename="word_probabilities.html",
    )
    if result.masked_response is not None:
        write_response_artifacts(
            response=result.masked_response,
            generated_filename="masked_generated.txt",
            token_csv_filename="masked_response_token_probabilities.csv",
            token_plot_filename="masked_response_token_probabilities.png",
            token_html_filename="masked_response_token_probabilities.html",
            word_csv_filename="masked_response_word_probabilities.csv",
            word_html_filename="masked_response_word_probabilities.html",
        )

    print(f"Original-image response tokens: {len(result.original_response.generated_token_ids)}")
    if result.masked_response is not None:
        print(f"Masked-image response tokens: {len(result.masked_response.generated_token_ids)}")
    else:
        print("Masked-image response generation: skipped")
    print(f"Output directory: {output_dir}")
    print(f"Original response report: {output_dir / 'token_probabilities.html'}")
    if result.masked_response is not None:
        print(f"Masked response report: {output_dir / 'masked_response_token_probabilities.html'}")
    if result.original_response.word_scores:
        print(f"Original word report: {output_dir / 'word_probabilities.html'}")
    if result.masked_response is not None and result.masked_response.word_scores:
        print(f"Masked word report: {output_dir / 'masked_response_word_probabilities.html'}")


if __name__ == "__main__":
    main()
