#!/usr/bin/env python3
"""Export already completed V4 confusable pages as training data.

This script is intentionally compile-free.  It scans only authoritative
``pages/*/terminal_result.json`` records from an in-progress or completed
canonical-reflow V4 run, validates the edited page artifacts fail-closed, and
writes an atomic training-data snapshot.  Clean pages, rejected pages, and
partially written pages never enter the exported datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "arxiv_canonical_reflow_v4_completed_snapshot_v1"
# Kept explicit so this compile-free exporter remains standalone.  A later V4
# producer version must be opted into with --accept-pipeline-version.
PIPELINE_VERSION = "arxiv_canonical_reflow_v4_7"
MUTATION_POLICY_VERSION = "canonical_reflow_confusable_v1_aligned"

_SFT_PROMPT = """<image>
Please convert the image document into Markdown format, strictly adhering to the following requirements:

1. Accurately transcribe all visible text without guessing or correcting typos.
2. Preserve headings, paragraphs, lists, inline emphasis, and reading order.
3. Convert formulas to LaTeX.
4. Convert tables to clean HTML without adding metadata attributes.
5. Ignore graphical elements, headers, footers, and page numbers.
6. Return only the Markdown transcription."""

_VERL_PROMPT = (
    "<image>\nPlease transcribe all text in this page image faithfully, "
    "exactly as printed (including any typos)."
)


@dataclass(frozen=True, slots=True)
class CompletedPage:
    result_path: Path
    page_dir: Path
    payload: dict[str, Any]
    image: Path
    pdf: Path


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    result_path: Path
    page: CompletedPage | None
    reason: str | None


def _emit(stage: str, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [{stage}] {message}", flush=True)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_artifact(
    raw: Any,
    *,
    run_root: Path,
    fallback: Path,
) -> Path | None:
    candidates: list[Path] = []
    if isinstance(raw, str) and raw:
        supplied = Path(raw).expanduser()
        candidates.append(supplied if supplied.is_absolute() else run_root / supplied)
    candidates.append(fallback)
    root_resolved = run_root.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if (
            _is_relative_to(resolved, root_resolved)
            and resolved.is_file()
            and resolved.stat().st_size > 0
        ):
            return resolved
    return None


def _png_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _valid_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item) for item in value)
    )


def _validate_changes(
    changes: Any,
    mutation_count: Any,
    *,
    markdown: str,
    image_size: tuple[int, int],
) -> str | None:
    if not isinstance(mutation_count, int) or isinstance(mutation_count, bool):
        return "invalid_mutation_count"
    if mutation_count <= 0:
        return "no_mutations"
    if not isinstance(changes, list) or len(changes) != mutation_count:
        return "mutation_count_or_changes_mismatch"
    image_width, image_height = image_size
    for index, change in enumerate(changes):
        if not isinstance(change, dict):
            return f"invalid_change:{index}:not_object"
        origin = change.get("origin_ans")
        edited = change.get("ocr_ans")
        if (
            not isinstance(origin, str)
            or not origin
            or not isinstance(edited, str)
            or not edited
            or origin == edited
        ):
            return f"invalid_change:{index}:word_pair"
        if edited not in markdown:
            return f"invalid_change:{index}:edited_word_absent_from_gt"
        bbox = change.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in bbox
            )
        ):
            return f"invalid_change:{index}:bbox"
        x0, y0, x1, y1 = (float(value) for value in bbox)
        if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
            return f"invalid_change:{index}:bbox_out_of_image"
    return None


def _validate_terminal_result(
    result_path: Path,
    *,
    run_root: Path,
    accepted_versions: frozenset[str],
) -> ValidationOutcome:
    page_dir = result_path.parent
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return ValidationOutcome(
            result_path=result_path,
            page=None,
            reason=f"terminal_result_unreadable:{type(error).__name__}",
        )
    if not isinstance(payload, dict):
        return ValidationOutcome(result_path, None, "terminal_result_not_object")
    if payload.get("pipeline_version") not in accepted_versions:
        return ValidationOutcome(result_path, None, "pipeline_version_not_accepted")
    if payload.get("status") != "accepted":
        return ValidationOutcome(result_path, None, "terminal_status_not_accepted")
    if payload.get("reason") not in {None, ""}:
        return ValidationOutcome(result_path, None, "accepted_result_has_reason")
    if payload.get("variant") != "confusable_edit":
        return ValidationOutcome(result_path, None, "not_confusable_edit")

    page_id = payload.get("page_id")
    paper_id = payload.get("paper_id")
    clean_page_id = payload.get("clean_page_id")
    if not isinstance(page_id, str) or not page_id or page_id != page_dir.name:
        return ValidationOutcome(result_path, None, "page_id_mismatch")
    if not isinstance(paper_id, str) or not paper_id:
        return ValidationOutcome(result_path, None, "invalid_paper_id")
    if not isinstance(clean_page_id, str) or not clean_page_id:
        return ValidationOutcome(result_path, None, "missing_clean_page_id")
    if payload.get("layout") not in {"one_column", "two_column"}:
        return ValidationOutcome(result_path, None, "invalid_layout")
    if not isinstance(payload.get("has_table"), bool):
        return ValidationOutcome(result_path, None, "invalid_has_table")
    markdown = payload.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        return ValidationOutcome(result_path, None, "empty_markdown")
    if not _valid_string_list(payload.get("block_ids")):
        return ValidationOutcome(result_path, None, "invalid_block_ids")
    if not _valid_string_list(payload.get("source_node_ids")):
        return ValidationOutcome(result_path, None, "invalid_source_node_ids")

    image = _resolve_artifact(
        payload.get("image"),
        run_root=run_root,
        fallback=page_dir / "page.png",
    )
    if image is None:
        return ValidationOutcome(result_path, None, "missing_image")
    image_size = _png_size(image)
    if image_size is None:
        return ValidationOutcome(result_path, None, "invalid_png")
    pdf = _resolve_artifact(
        payload.get("pdf"),
        run_root=run_root,
        fallback=page_dir / "build" / "page.pdf",
    )
    if pdf is None:
        return ValidationOutcome(result_path, None, "missing_pdf")

    ground_truth_path = page_dir / "ground_truth.md"
    try:
        ground_truth = ground_truth_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ValidationOutcome(result_path, None, "missing_ground_truth_file")
    if ground_truth.removesuffix("\n") != markdown:
        return ValidationOutcome(result_path, None, "ground_truth_file_mismatch")

    change_error = _validate_changes(
        payload.get("changes"),
        payload.get("mutation_count"),
        markdown=markdown,
        image_size=image_size,
    )
    if change_error is not None:
        return ValidationOutcome(result_path, None, change_error)

    return ValidationOutcome(
        result_path=result_path,
        page=CompletedPage(
            result_path=result_path,
            page_dir=page_dir,
            payload=payload,
            image=image,
            pdf=pdf,
        ),
        reason=None,
    )


def _relative_artifact(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path, output_dir.resolve())).as_posix()


def _projected_changes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "ocr_ans": change["ocr_ans"],
            "origin_ans": change["origin_ans"],
            "bbox": change["bbox"],
        }
        for change in payload["changes"]
    ]


def _manifest_row(page: CompletedPage, output_dir: Path) -> dict[str, Any]:
    payload = page.payload
    return {
        "pair_id": payload["page_id"],
        "paper_id": payload["paper_id"],
        "image": _relative_artifact(page.image, output_dir),
        "pdf": _relative_artifact(page.pdf, output_dir),
        "markdown": payload["markdown"],
        "layout": payload["layout"],
        "has_table": payload["has_table"],
        "block_ids": payload["block_ids"],
        "source_node_ids": payload["source_node_ids"],
        "verifier_recall": payload.get("verifier_recall"),
        "verifier_precision": payload.get("verifier_precision"),
        "content_fill_ratio": payload.get("content_fill_ratio"),
        "column_fill_ratios": payload.get("column_fill_ratios", []),
        "pack_attempts": payload.get("pack_attempts", 1),
        "variant": "confusable_edit",
        "clean_page_id": payload["clean_page_id"],
        "mutation_count": payload["mutation_count"],
        "changes": payload["changes"],
        "max_mutation_vertical_shift_points": payload.get(
            "max_mutation_vertical_shift_points"
        ),
        "mutation_policy_version": MUTATION_POLICY_VERSION,
        "ground_truth_source": "latex_ast_confusable_edit",
        "pdf_role": "reject_only",
    }


def _sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": _SFT_PROMPT},
            {"role": "assistant", "content": row["markdown"]},
        ],
        "images": [row["image"]],
        "data_source": "chaos_document_ocr",
        "ability": "document_ocr",
        "extra_info": {
            "arxiv_id": row["paper_id"],
            "pair_id": row["pair_id"],
            "layout": row["layout"],
            "has_table": row["has_table"],
            "content_fill_ratio": row["content_fill_ratio"],
            "mutation_count": row["mutation_count"],
        },
    }


def _sft_v1_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "images": [row["image"]],
        "conversations": [
            {"from": "human", "value": _SFT_PROMPT},
            {"from": "gpt", "value": row["markdown"]},
        ],
    }


def _verl_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_source": "chaos_document_ocr",
        "prompt": [{"role": "user", "content": _VERL_PROMPT}],
        "images": [row["image"]],
        "reward_model": {"style": "rule", "ground_truth": row["markdown"]},
        "extra_info": {
            "arxiv_id": row["paper_id"],
            "pair_id": row["pair_id"],
            "changes": _projected_changes(row),
        },
        "ability": "document_ocr",
    }


def export_completed_snapshot(
    run_root: Path,
    *,
    output_dir: Path | None = None,
    workers: int = 32,
    accepted_versions: frozenset[str] | None = None,
    progress_every: int = 1000,
) -> dict[str, Any]:
    started = time.monotonic()
    run_root = run_root.expanduser().resolve()
    pages_root = run_root / "pages"
    if not pages_root.is_dir():
        raise ValueError(f"pages directory does not exist: {pages_root}")
    if output_dir is None:
        output_dir = run_root / "completed_training_data"
    output_dir = output_dir.expanduser().resolve()
    accepted_versions = accepted_versions or frozenset({PIPELINE_VERSION})
    workers = max(1, workers)
    progress_every = max(1, progress_every)

    result_paths = sorted(pages_root.glob("*/terminal_result.json"))
    _emit(
        "start",
        (
            f"run_root={run_root} output_dir={output_dir} "
            f"terminal_results={len(result_paths)} workers={workers} "
            f"accepted_versions={','.join(sorted(accepted_versions))}"
        ),
    )

    def validate(path: Path) -> ValidationOutcome:
        return _validate_terminal_result(
            path,
            run_root=run_root,
            accepted_versions=accepted_versions,
        )

    accepted: list[CompletedPage] = []
    rejected_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    last_progress = started
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = executor.map(validate, result_paths)
        for completed, outcome in enumerate(outcomes, start=1):
            if outcome.page is not None:
                accepted.append(outcome.page)
            else:
                reason = outcome.reason or "unknown"
                reasons[reason] += 1
                rejected_rows.append(
                    {
                        "terminal_result": outcome.result_path.relative_to(
                            run_root
                        ).as_posix(),
                        "reason": reason,
                    }
                )
            now = time.monotonic()
            if (
                completed == len(result_paths)
                or completed % progress_every == 0
                or now - last_progress >= 30.0
            ):
                elapsed = max(now - started, 1e-9)
                rate = completed / elapsed
                remaining = len(result_paths) - completed
                eta = remaining / rate if rate > 0 else 0.0
                _emit(
                    "scan",
                    (
                        f"completed={completed}/{len(result_paths)} "
                        f"percent={completed / max(1, len(result_paths)):.2%} "
                        f"accepted={len(accepted)} skipped={len(rejected_rows)} "
                        f"throughput={rate:.1f}_files/s elapsed={elapsed:.1f}s "
                        f"eta={eta:.1f}s current={outcome.result_path.parent.name}"
                    ),
                )
                last_progress = now

    accepted.sort(key=lambda page: page.payload["page_id"])
    manifest_rows = [_manifest_row(page, output_dir) for page in accepted]
    page_ids = [row["pair_id"] for row in manifest_rows]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("duplicate accepted page_id detected")
    images = [row["image"] for row in manifest_rows]
    if len(images) != len(set(images)):
        raise ValueError("duplicate accepted image detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    _atomic_jsonl(output_dir / "pairs.jsonl", manifest_rows)
    _atomic_jsonl(output_dir / "sft.jsonl", map(_sft_row, manifest_rows))
    _atomic_jsonl(
        output_dir / "sft_v1_compatible.jsonl",
        map(_sft_v1_row, manifest_rows),
    )
    legacy_sft_path = output_dir / f"SFT_edited_{len(manifest_rows)}.jsonl"
    _atomic_jsonl(legacy_sft_path, map(_sft_v1_row, manifest_rows))
    for stale in output_dir.glob("SFT_edited_*.jsonl"):
        if stale != legacy_sft_path:
            stale.unlink()
    _atomic_jsonl(output_dir / "verl.jsonl", map(_verl_row, manifest_rows))
    _atomic_jsonl(output_dir / "skipped_terminal_results.jsonl", rejected_rows)

    elapsed = time.monotonic() - started
    report = {
        "snapshot_version": SNAPSHOT_VERSION,
        "pipeline_versions_accepted": sorted(accepted_versions),
        "status": "passed" if manifest_rows else "empty",
        "partial_snapshot": True,
        "compile_performed": False,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "terminal_results_scanned": len(result_paths),
        "edited_pages_exported": len(manifest_rows),
        "terminal_results_skipped": len(rejected_rows),
        "skip_reasons": dict(reasons.most_common()),
        "mutation_changes_total": sum(
            int(row["mutation_count"]) for row in manifest_rows
        ),
        "table_pages": sum(bool(row["has_table"]) for row in manifest_rows),
        "two_column_pages": sum(row["layout"] == "two_column" for row in manifest_rows),
        "outputs": {
            "manifest": "manifest.jsonl",
            "pairs": "pairs.jsonl",
            "ms_swift_sft": "sft.jsonl",
            "v1_compatible_sft": "sft_v1_compatible.jsonl",
            "v1_compatible_sft_counted": legacy_sft_path.name,
            "verl": "verl.jsonl",
            "skipped": "skipped_terminal_results.jsonl",
        },
        "elapsed_seconds": elapsed,
    }
    _atomic_json(output_dir / "completed_export_report.json", report)
    _emit(
        "finish",
        (
            f"status={report['status']} exported={len(manifest_rows)} "
            f"skipped={len(rejected_rows)} changes={report['mutation_changes_total']} "
            f"table_pages={report['table_pages']} "
            f"two_column_pages={report['two_column_pages']} "
            f"elapsed={elapsed:.1f}s output_dir={output_dir}"
        ),
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Existing V4 output root containing pages/*/terminal_result.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Snapshot directory (default: RUN_ROOT/completed_training_data).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Parallel artifact-validation workers (default: min(32, CPU count)).",
    )
    parser.add_argument(
        "--accept-pipeline-version",
        action="append",
        dest="accepted_versions",
        help=(
            "Accepted V4 pipeline version. Repeat to accept multiple versions. "
            f"Default: {PIPELINE_VERSION}."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print parent-process progress after this many terminal files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = export_completed_snapshot(
            args.run_root,
            output_dir=args.output_dir,
            workers=args.workers,
            accepted_versions=(
                frozenset(args.accepted_versions)
                if args.accepted_versions
                else frozenset({PIPELINE_VERSION})
            ),
            progress_every=args.progress_every,
        )
    except (OSError, ValueError) as error:
        _emit("error", f"{type(error).__name__}: {error}")
        return 1
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
