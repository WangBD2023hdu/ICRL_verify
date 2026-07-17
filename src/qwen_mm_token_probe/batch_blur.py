from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .prompts import DEFAULT_PDF_OCR_PROMPT


@dataclass(frozen=True)
class BatchItem:
    sample_key: str
    line_number: int
    image_index: int
    record_id: str
    image_path: Path
    prompt: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        return data


@dataclass(frozen=True)
class BatchRunSummary:
    output_dir: Path
    total_items: int
    completed_items: int
    skipped_items: int
    failed_items: int
    interrupted: bool


def load_batch_items(
    jsonl_path: str | Path,
    *,
    default_prompt: str = DEFAULT_PDF_OCR_PROMPT,
    image_field: str = "images",
    id_field: str = "id",
    prompt_field: str | None = None,
    limit: int | None = None,
) -> list[BatchItem]:
    input_path = Path(jsonl_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input JSONL not found: {input_path}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    items: list[BatchItem] = []
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            if image_field not in record:
                raise ValueError(f"line {line_number} is missing image field {image_field!r}")

            image_values = _normalize_image_values(record[image_field], line_number=line_number)
            record_id = str(record.get(id_field, line_number))
            prompt = _record_prompt(
                record,
                default_prompt=default_prompt,
                prompt_field=prompt_field,
                line_number=line_number,
            )
            for image_index, image_value in enumerate(image_values):
                image_path = Path(image_value).expanduser()
                if not image_path.is_absolute():
                    image_path = input_path.parent / image_path
                image_path = image_path.resolve()
                sample_key = _sample_key(
                    line_number=line_number,
                    image_index=image_index,
                    record_id=record_id,
                    image_path=image_path,
                )
                items.append(
                    BatchItem(
                        sample_key=sample_key,
                        line_number=line_number,
                        image_index=image_index,
                        record_id=record_id,
                        image_path=image_path,
                        prompt=prompt,
                    )
                )
                if limit is not None and len(items) >= limit:
                    return items
    return items


def run_blur_batch(
    *,
    model_id: str,
    input_jsonl: str | Path,
    output_dir: str | Path,
    prompt: str = DEFAULT_PDF_OCR_PROMPT,
    prompt_field: str | None = None,
    image_field: str = "images",
    id_field: str = "id",
    blur_radii: Iterable[float] = (1.0, 2.0, 4.0, 8.0),
    max_new_tokens: int = 4096,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int = 7,
    device_map: str | None = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    min_pixels: int = 2048,
    max_pixels: int = 16777216,
    image_patch_size: int = 16,
    enable_thinking: bool = False,
    generate_blurred_responses: bool = False,
    group_tokens: str = "word",
    resume: bool = True,
    fail_fast: bool = False,
    limit: int | None = None,
    report_max_tokens: int = 4096,
    top_affected_tokens: int = 100,
) -> BatchRunSummary:
    radii = _normalize_blur_radii(blur_radii)
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError(
            f"pixel bounds must satisfy 0 < min_pixels <= max_pixels, got "
            f"{min_pixels} and {max_pixels}"
        )
    if image_patch_size <= 0:
        raise ValueError(f"image_patch_size must be positive, got {image_patch_size}")
    if temperature is not None and temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if group_tokens not in {"none", "word"}:
        raise ValueError(f"unsupported token grouping mode: {group_tokens}")
    if report_max_tokens < 0:
        raise ValueError("report_max_tokens must be non-negative")
    if top_affected_tokens <= 0:
        raise ValueError("top_affected_tokens must be positive")

    input_path = Path(input_jsonl).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples_root = output_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    items = load_batch_items(
        input_path,
        default_prompt=prompt,
        image_field=image_field,
        id_field=id_field,
        prompt_field=prompt_field,
        limit=limit,
    )
    if not items:
        raise RuntimeError(f"input JSONL contains no image items: {input_path}")

    runtime_config = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "model_id": model_id,
        "input_jsonl": str(input_path),
        "output_dir": str(output_root),
        "image_field": image_field,
        "id_field": id_field,
        "prompt": prompt,
        "prompt_field": prompt_field,
        "blur_radii": radii,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "device_map": device_map,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "image_patch_size": image_patch_size,
        "enable_thinking": enable_thinking,
        "generate_blurred_responses": generate_blurred_responses,
        "group_tokens": group_tokens,
        "resume": resume,
        "limit": limit,
        "report_max_tokens": report_max_tokens,
        "top_affected_tokens": top_affected_tokens,
        "num_input_items": len(items),
    }
    _write_json_atomic(output_root / "config.json", runtime_config)

    skipped = 0
    failed = 0
    interrupted = False
    fatal_error: BaseException | None = None
    pending_items: list[tuple[int, BatchItem, str]] = []
    for ordinal, item in enumerate(items, start=1):
        sample_dir = samples_root / item.sample_key
        result_path = sample_dir / "result.json"
        fingerprint = _item_fingerprint(
            item=item,
            model_id=model_id,
            radii=radii,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_patch_size=image_patch_size,
            enable_thinking=enable_thinking,
            generate_blurred_responses=generate_blurred_responses,
        )
        if resume and _completed_result_matches(result_path, fingerprint):
            skipped += 1
            print(
                f"[{ordinal}/{len(items)}] resume: {item.sample_key}",
                flush=True,
            )
            continue
        pending_items.append((ordinal, item, fingerprint))

    bundle = None
    if pending_items:
        from .hf_qwen import load_model_bundle

        print(f"Loading model once: {model_id}", flush=True)
        bundle = load_model_bundle(
            model_id,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    for ordinal, item, fingerprint in pending_items:
        sample_dir = samples_root / item.sample_key
        result_path = sample_dir / "result.json"
        if result_path.exists():
            result_path.replace(sample_dir / "result.previous.json")

        print(
            f"[{ordinal}/{len(items)}] OCR + {len(radii)} blur levels: {item.image_path}",
            flush=True,
        )
        try:
            _set_item_seed(seed + ordinal - 1)
            _process_item(
                item=item,
                sample_dir=sample_dir,
                run_fingerprint=fingerprint,
                bundle=bundle,
                model_id=model_id,
                radii=radii,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                image_patch_size=image_patch_size,
                enable_thinking=enable_thinking,
                generate_blurred_responses=generate_blurred_responses,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; rebuilding reports from completed samples.", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 - batch jobs must preserve later samples
            failed += 1
            _append_jsonl(
                output_root / "failures.jsonl",
                {
                    "timestamp": _utc_now(),
                    "sample": item.to_dict(),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{ordinal}/{len(items)}] failed: {type(exc).__name__}: {exc}", flush=True)
            if fail_fast:
                fatal_error = exc
                break

    from .batch_report import rebuild_batch_reports

    report_state = rebuild_batch_reports(
        output_root,
        group_tokens=group_tokens,
        report_max_tokens=report_max_tokens,
        top_affected_tokens=top_affected_tokens,
        allowed_sample_keys={item.sample_key for item in items},
    )
    total_completed = int(report_state["completed_samples"])
    summary = BatchRunSummary(
        output_dir=output_root,
        total_items=len(items),
        completed_items=total_completed,
        skipped_items=skipped,
        failed_items=failed,
        interrupted=interrupted,
    )
    if fatal_error is not None:
        raise fatal_error
    return summary


def _process_item(
    *,
    item: BatchItem,
    sample_dir: Path,
    run_fingerprint: str,
    bundle: Any,
    model_id: str,
    radii: list[float],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    min_pixels: int,
    max_pixels: int,
    image_patch_size: int,
    enable_thinking: bool,
    generate_blurred_responses: bool,
) -> None:
    from PIL import ImageFilter

    from .batch_stats import (
        baseline_token_records,
        blurred_token_records,
        comparison_scores_from_records,
        summarize_scores,
    )
    from .hf_qwen import (
        generate_from_prompt,
        prepare_prompt_inputs,
        token_statistics_for_generated_ids,
    )
    from .image_mask import load_rgb_image, save_rgb_image
    from .token_grouping import group_token_scores

    sample_dir.mkdir(parents=True, exist_ok=True)
    original_image = load_rgb_image(item.image_path)
    original_path = save_rgb_image(original_image, sample_dir / "original.png")
    original_inputs = prepare_prompt_inputs(
        processor=bundle.processor,
        image_path=original_path,
        prompt=item.prompt,
        device=bundle.device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_patch_size=image_patch_size,
        enable_thinking=enable_thinking,
    )
    generated_token_ids, generated_text = generate_from_prompt(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        prompt_inputs=original_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    if not generated_token_ids:
        raise RuntimeError("original image generated no scoreable text tokens")
    (sample_dir / "response.md").write_text(generated_text, encoding="utf-8")

    original_stats = token_statistics_for_generated_ids(
        model=bundle.model,
        prompt_inputs=original_inputs,
        generated_token_ids=generated_token_ids,
    )
    baseline_tokens = baseline_token_records(
        generated_token_ids,
        original_stats,
        bundle.tokenizer,
    )

    blur_levels = []
    for radius in radii:
        radius_name = _radius_name(radius)
        blurred_image = original_image.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred_path = save_rgb_image(blurred_image, sample_dir / f"blur_{radius_name}.png")
        blurred_inputs = prepare_prompt_inputs(
            processor=bundle.processor,
            image_path=blurred_path,
            prompt=item.prompt,
            device=bundle.device,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_patch_size=image_patch_size,
            enable_thinking=enable_thinking,
        )
        blurred_stats = token_statistics_for_generated_ids(
            model=bundle.model,
            prompt_inputs=blurred_inputs,
            generated_token_ids=generated_token_ids,
        )
        blurred_tokens = blurred_token_records(
            generated_token_ids,
            blurred_stats,
            bundle.tokenizer,
        )
        comparison_scores = comparison_scores_from_records(baseline_tokens, blurred_tokens)
        word_scores = group_token_scores(comparison_scores)
        level: dict[str, object] = {
            "blur_radius": radius,
            "image_path": str(blurred_path),
            "tokens": blurred_tokens,
            "summary": summarize_scores(comparison_scores, word_scores=word_scores),
        }

        if generate_blurred_responses:
            blurred_response_ids, blurred_response_text = generate_from_prompt(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                prompt_inputs=blurred_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            response_path = sample_dir / f"blur_{radius_name}_response.md"
            response_path.write_text(blurred_response_text, encoding="utf-8")
            level["generated_response"] = {
                "path": str(response_path),
                "text": blurred_response_text,
                "token_ids": blurred_response_ids,
            }

        blur_levels.append(level)
        del blurred_inputs

    result = {
        "schema_version": 1,
        "completed_at": _utc_now(),
        "run_fingerprint": run_fingerprint,
        "model_id": model_id,
        "sample": item.to_dict(),
        "original": {
            "image_path": str(original_path),
            "width": original_image.size[0],
            "height": original_image.size[1],
            "response_path": str(sample_dir / "response.md"),
            "generated_text": generated_text,
            "generated_token_ids": generated_token_ids,
            "tokens": baseline_tokens,
        },
        "blur_levels": blur_levels,
    }
    _write_json_atomic(sample_dir / "result.json", result)


def _normalize_image_values(value: Any, *, line_number: int) -> list[str]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"line {line_number} has an empty images list")
    normalized = []
    for image_index, item in enumerate(values):
        if isinstance(item, str) and item.strip():
            normalized.append(item)
            continue
        if isinstance(item, dict):
            path_value = item.get("path", item.get("image"))
            if isinstance(path_value, str) and path_value.strip():
                normalized.append(path_value)
                continue
        raise ValueError(
            f"line {line_number} image {image_index} must be a path string "
            "or an object containing 'path'"
        )
    return normalized


def _record_prompt(
    record: dict[str, Any],
    *,
    default_prompt: str,
    prompt_field: str | None,
    line_number: int,
) -> str:
    if prompt_field is None:
        prompt = default_prompt
    else:
        prompt = record.get(prompt_field)
        if not isinstance(prompt, str):
            raise ValueError(
                f"line {line_number} prompt field {prompt_field!r} must be a string"
            )
    if not prompt.strip():
        raise ValueError(f"line {line_number} resolved to an empty prompt")
    return prompt


def _sample_key(
    *,
    line_number: int,
    image_index: int,
    record_id: str,
    image_path: Path,
) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", record_id).strip("-._") or "sample"
    slug = slug[:48]
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:8]
    return f"{line_number:06d}_{image_index:02d}_{slug}_{digest}"


def _normalize_blur_radii(values: Iterable[float]) -> list[float]:
    radii = sorted({float(value) for value in values})
    if not radii:
        raise ValueError("at least one blur radius is required")
    if any(not math.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError(f"blur radii must be finite and positive, got {radii}")
    return radii


def _radius_name(radius: float) -> str:
    return f"r{radius:g}".replace(".", "p")


def _item_fingerprint(
    *,
    item: BatchItem,
    model_id: str,
    radii: list[float],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    seed: int,
    min_pixels: int,
    max_pixels: int,
    image_patch_size: int,
    enable_thinking: bool,
    generate_blurred_responses: bool,
) -> str:
    image_state: dict[str, object] = {"path": str(item.image_path)}
    if item.image_path.exists():
        stat = item.image_path.stat()
        image_state.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = {
        "sample": item.to_dict(),
        "image_state": image_state,
        "model_id": model_id,
        "radii": radii,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "image_patch_size": image_patch_size,
        "enable_thinking": enable_thinking,
        "generate_blurred_responses": generate_blurred_responses,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _completed_result_matches(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("run_fingerprint") == fingerprint and bool(data.get("blur_levels"))


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _set_item_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run OCR over image paths in JSONL, score the original response under "
            "multiple Gaussian blur levels, and build aggregate reports."
        )
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", default="outputs/qwen_blur_batch")
    parser.add_argument("--image-field", default="images")
    parser.add_argument("--id-field", default="id")
    parser.add_argument(
        "--prompt-field",
        default=None,
        help="Optional per-record prompt field. Otherwise the global PDF OCR prompt is used.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None)
    prompt_group.add_argument("--prompt-file", default=None)
    parser.add_argument(
        "--blur-radii",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0],
        help="Positive PIL Gaussian blur radii, for example: 0.5 1 2 4 8.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--generate-blurred-responses",
        action="store_true",
        help="Also generate and save a free-running OCR response at every blur radius.",
    )
    parser.add_argument("--group-tokens", choices=["none", "word"], default="word")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed samples whose input and inference fingerprint still match.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--report-max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens embedded in each interactive HTML report; 0 means all.",
    )
    parser.add_argument("--top-affected-tokens", type=int, default=100)
    return parser


def _load_cli_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
    return DEFAULT_PDF_OCR_PROMPT


def main() -> None:
    args = build_parser().parse_args()
    summary = run_blur_batch(
        model_id=args.model_id,
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        prompt=_load_cli_prompt(args),
        prompt_field=args.prompt_field,
        image_field=args.image_field,
        id_field=args.id_field,
        blur_radii=args.blur_radii,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        image_patch_size=args.image_patch_size,
        enable_thinking=args.enable_thinking,
        generate_blurred_responses=args.generate_blurred_responses,
        group_tokens=args.group_tokens,
        resume=args.resume,
        fail_fast=args.fail_fast,
        limit=args.limit,
        report_max_tokens=args.report_max_tokens,
        top_affected_tokens=args.top_affected_tokens,
    )
    print(
        "Batch complete: "
        f"completed={summary.completed_items}, skipped={summary.skipped_items}, "
        f"failed_this_run={summary.failed_items}",
        flush=True,
    )
    print(f"Report: {summary.output_dir / 'report.html'}", flush=True)
    if summary.interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
