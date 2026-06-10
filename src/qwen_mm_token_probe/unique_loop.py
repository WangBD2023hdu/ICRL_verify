from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatedly run Qwen-style multimodal inference and save only unique "
            "responses until interrupted."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="User prompt for the image.")
    parser.add_argument("--output-dir", default="outputs/qwen_unique_loop")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=0,
        help="Number of inference rounds. Use 0 for an infinite loop.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between inference rounds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed once before the loop. Leave unset for non-deterministic sampling.",
    )

    sampling = parser.add_argument_group("generation sampling")
    sampling.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stochastic decoding. Enabled by default.",
    )
    sampling.add_argument("--temperature", type=float, default=0.7)
    sampling.add_argument("--top-p", type=float, default=0.95)

    comparison = parser.add_argument_group("deduplication")
    comparison.add_argument(
        "--compare-mode",
        choices=["exact", "strip", "collapse-space"],
        default="strip",
        help="How responses are normalized before duplicate comparison.",
    )
    return parser


def normalize_response(text: str, mode: str) -> str:
    if mode == "exact":
        return text
    if mode == "strip":
        return text.strip()
    if mode == "collapse-space":
        return " ".join(text.split())
    raise ValueError(f"unsupported compare mode: {mode}")


def response_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_seen_hashes(manifest_path: Path) -> set[str]:
    seen: set[str] = set()
    if not manifest_path.exists():
        return seen

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            digest = record.get("hash")
            if isinstance(digest, str):
                seen.add(digest)
    return seen


def write_unique_response(
    *,
    output_dir: Path,
    unique_index: int,
    text: str,
    metadata: dict[str, Any],
) -> Path:
    stem = f"unique_{unique_index:06d}"
    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    txt_path.write_text(text, encoding="utf-8")
    json_path.write_text(
        json.dumps({**metadata, "text": text}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return txt_path


def run_unique_loop(args: argparse.Namespace) -> None:
    import torch

    from .hf_qwen import generate_from_prompt, load_model_bundle, prepare_prompt_inputs

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    runs_path = output_dir / "runs.jsonl"
    seen_hashes = load_seen_hashes(manifest_path)
    unique_count = len(seen_hashes)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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
    )

    config_record = {
        "created_at": utc_now(),
        "model_id": args.model_id,
        "image": str(Path(args.image).expanduser()),
        "output_dir": str(output_dir),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "compare_mode": args.compare_mode,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Output directory: {output_dir}", flush=True)
    print(f"Loaded {len(seen_hashes)} previous unique response hashes.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    iteration = 0
    try:
        while args.num_runs <= 0 or iteration < args.num_runs:
            iteration += 1
            started_at = utc_now()
            generated_ids, generated_text = generate_from_prompt(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                prompt_inputs=prompt_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            normalized = normalize_response(generated_text, args.compare_mode)
            digest = response_hash(normalized)
            is_unique = digest not in seen_hashes

            run_record = {
                "iteration": iteration,
                "started_at": started_at,
                "finished_at": utc_now(),
                "hash": digest,
                "is_unique": is_unique,
                "num_generated_tokens": len(generated_ids),
            }

            if is_unique:
                seen_hashes.add(digest)
                unique_count += 1
                metadata = {
                    **run_record,
                    "unique_index": unique_count,
                    "model_id": args.model_id,
                    "image": str(Path(args.image).expanduser()),
                    "compare_mode": args.compare_mode,
                }
                response_path = write_unique_response(
                    output_dir=output_dir,
                    unique_index=unique_count,
                    text=generated_text,
                    metadata=metadata,
                )
                run_record["saved_path"] = str(response_path)
                append_jsonl(manifest_path, metadata)
                print(
                    f"[{iteration}] unique #{unique_count}: {response_path.name} "
                    f"tokens={len(generated_ids)} hash={digest[:12]}",
                    flush=True,
                )
            else:
                print(
                    f"[{iteration}] duplicate tokens={len(generated_ids)} "
                    f"hash={digest[:12]}",
                    flush=True,
                )

            append_jsonl(runs_path, run_record)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    except KeyboardInterrupt:
        print(
            f"\nStopped by user after {iteration} iterations. "
            f"Unique responses: {unique_count}.",
            flush=True,
        )


def main() -> None:
    run_unique_loop(build_parser().parse_args())


if __name__ == "__main__":
    main()
