from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prefill a multimodal generation with existing output tokens, force "
            "one or more token ids, then continue generation."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", default="outputs/qwen_force_continue")
    parser.add_argument(
        "--prefix-token-csv",
        required=True,
        help="CSV produced by qwen-mm-token-probe, e.g. token_probabilities.csv.",
    )
    parser.add_argument(
        "--prefix-end-index",
        type=int,
        required=True,
        help="Inclusive generated-token index to keep as prefix.",
    )
    parser.add_argument(
        "--force-token-ids",
        required=True,
        help="Comma-separated token ids to force after the prefix, e.g. 91 or 271,91.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=16777216)
    parser.add_argument("--image-patch-size", type=int, default=16)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def read_prefix_token_ids(csv_path: str | Path, end_index: int) -> list[int]:
    rows: list[tuple[int, int]] = []
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(row["index"])
            if index <= end_index:
                rows.append((index, int(row["token_id"])))

    rows.sort(key=lambda item: item[0])
    expected = list(range(rows[0][0], rows[-1][0] + 1)) if rows else []
    actual = [index for index, _ in rows]
    if not rows:
        raise ValueError(f"no token rows found with index <= {end_index}")
    if actual != expected or actual[0] != 0:
        raise ValueError(
            "prefix csv must contain contiguous generated token indexes starting at 0; "
            f"got first={actual[0]}, last={actual[-1]}, count={len(actual)}"
        )
    if rows[-1][0] != end_index:
        raise ValueError(
            f"prefix-end-index {end_index} was not found in {csv_path}; "
            f"last available prefix index is {rows[-1][0]}"
        )
    return [token_id for _, token_id in rows]


def parse_token_ids(value: str) -> list[int]:
    token_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not token_ids:
        raise ValueError("--force-token-ids cannot be empty")
    return token_ids


def main() -> None:
    args = build_parser().parse_args()

    from .hf_qwen import (
        decode_generated_tokens,
        display_token,
        generate_from_prefilled_tokens,
        load_model_bundle,
        next_token_topk,
        prepare_prompt_inputs,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_token_ids = read_prefix_token_ids(args.prefix_token_csv, args.prefix_end_index)
    forced_token_ids = parse_token_ids(args.force_token_ids)

    bundle = load_model_bundle(
        args.model_id,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_inputs = prepare_prompt_inputs(
        processor=bundle.processor,
        image_path=args.image,
        prompt=args.prompt,
        device=bundle.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        image_patch_size=args.image_patch_size,
        enable_thinking=args.enable_thinking,
    )

    next_token_info = next_token_topk(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        prompt_inputs=prompt_inputs,
        generated_prefix_token_ids=prefix_token_ids,
        top_k=args.top_k,
        inspect_token_ids=forced_token_ids,
    )

    forced_prefix_ids = prefix_token_ids + forced_token_ids
    continuation_ids, continuation_text, full_ids, full_text = generate_from_prefilled_tokens(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        prompt_inputs=prompt_inputs,
        generated_prefix_token_ids=forced_prefix_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    prefix_text = decode_generated_tokens(bundle.tokenizer, prefix_token_ids)
    forced_text = decode_generated_tokens(bundle.tokenizer, forced_token_ids)
    forced_prefix_text = decode_generated_tokens(bundle.tokenizer, forced_prefix_ids)

    (output_dir / "prefix.txt").write_text(prefix_text, encoding="utf-8")
    (output_dir / "forced_tokens.txt").write_text(forced_text, encoding="utf-8")
    (output_dir / "forced_prefix.txt").write_text(forced_prefix_text, encoding="utf-8")
    (output_dir / "continuation.txt").write_text(continuation_text, encoding="utf-8")
    (output_dir / "full_forced_output.txt").write_text(full_text, encoding="utf-8")

    payload = {
        "model_id": args.model_id,
        "image": args.image,
        "prefix_token_csv": args.prefix_token_csv,
        "prefix_end_index": args.prefix_end_index,
        "prefix_token_count": len(prefix_token_ids),
        "forced_token_ids": forced_token_ids,
        "forced_tokens": [
            {
                "token_id": token_id,
                "token": display_token(bundle.tokenizer, token_id),
            }
            for token_id in forced_token_ids
        ],
        "continuation_token_ids": continuation_ids,
        "continuation_token_count": len(continuation_ids),
        "full_token_count": len(full_ids),
        "next_token_before_force": next_token_info,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Prefix tokens: {len(prefix_token_ids)}")
    print(f"Forced token ids: {forced_token_ids!r} -> {forced_text!r}")
    print(f"Continuation tokens: {len(continuation_ids)}")
    print(f"Output directory: {output_dir}")
    print(f"Full forced output: {output_dir / 'full_forced_output.txt'}")
    print(f"Next-token top-k: {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
