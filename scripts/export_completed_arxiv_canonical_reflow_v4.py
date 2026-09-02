#!/usr/bin/env python3
"""Export already completed V4 confusable pages as training data.

This script is intentionally compile-free.  It scans only completed confusable
page records from an in-progress or completed canonical-reflow V4 run, checks
that the page image exists and the recorded Markdown exactly matches the page
GT file, then streams minimal SFT and VERL datasets.  VERL keeps the producer's
``ocr_ans``, ``origin_ans``, and ``bbox`` mutation records.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "arxiv_canonical_reflow_v4_completed_snapshot_v2"
# Kept explicit so this compile-free exporter remains standalone.  A later V4
# producer version must be opted into with --accept-pipeline-version.
PIPELINE_VERSION = "arxiv_canonical_reflow_v4_8_direct_edit"

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
    payload: dict[str, Any]
    image: Path


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
    if payload.get("variant") != "confusable_edit":
        return ValidationOutcome(result_path, None, "not_confusable_edit")

    page_id = payload.get("page_id")
    paper_id = payload.get("paper_id")
    if not isinstance(page_id, str) or not page_id or page_id != page_dir.name:
        return ValidationOutcome(result_path, None, "page_id_mismatch")
    if not isinstance(paper_id, str) or not paper_id:
        return ValidationOutcome(result_path, None, "invalid_paper_id")
    markdown = payload.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        return ValidationOutcome(result_path, None, "empty_markdown")
    changes = payload.get("changes")
    if not isinstance(changes, list) or not changes:
        return ValidationOutcome(result_path, None, "missing_mutation_changes")
    if any(
        not isinstance(change, dict)
        or not {"ocr_ans", "origin_ans", "bbox"}.issubset(change)
        for change in changes
    ):
        return ValidationOutcome(result_path, None, "invalid_mutation_changes")

    image = _resolve_artifact(
        payload.get("image"),
        run_root=run_root,
        fallback=page_dir / "page.png",
    )
    if image is None:
        return ValidationOutcome(result_path, None, "missing_image")

    ground_truth_path = page_dir / "ground_truth.md"
    try:
        ground_truth = ground_truth_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ValidationOutcome(result_path, None, "missing_ground_truth_file")
    if ground_truth.removesuffix("\n") != markdown:
        return ValidationOutcome(result_path, None, "ground_truth_file_mismatch")

    return ValidationOutcome(
        result_path=result_path,
        page=CompletedPage(
            result_path=result_path,
            payload=payload,
            image=image,
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


def _training_fields(page: CompletedPage, output_dir: Path) -> dict[str, Any]:
    payload = page.payload
    return {
        "pair_id": payload["page_id"],
        "paper_id": payload["paper_id"],
        "image": _relative_artifact(page.image, output_dir),
        "markdown": payload["markdown"],
        "changes": payload["changes"],
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
        },
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


def _discover_confusable_terminals(
    pages_root: Path,
    *,
    progress_every: int,
) -> tuple[list[Path], int]:
    candidates: list[Path] = []
    discovered = 0
    started = time.monotonic()
    last_progress = started
    _emit("discover", f"pages_root={pages_root} filter=_confusable_s")
    with os.scandir(pages_root) as entries:
        for entry in entries:
            discovered += 1
            if "_confusable_s" in entry.name:
                page_dir = Path(entry.path)
                terminal = page_dir / "terminal_result.json"
                candidates.append(
                    terminal if terminal.is_file() else page_dir / "result.json"
                )
            now = time.monotonic()
            if discovered % progress_every == 0 or now - last_progress >= 30.0:
                _emit(
                    "discover",
                    (
                        f"page_entries={discovered} "
                        f"confusable_candidates={len(candidates)} "
                        f"elapsed={now - started:.1f}s current={entry.name}"
                    ),
                )
                last_progress = now
    _emit(
        "discover",
        (
            f"page_entries={discovered} confusable_candidates={len(candidates)} "
            f"elapsed={time.monotonic() - started:.1f}s"
        ),
    )
    return candidates, discovered


def _remove_obsolete_outputs(output_dir: Path) -> None:
    for name in (
        "manifest.jsonl",
        "pairs.jsonl",
        "sft_v1_compatible.jsonl",
        "skipped_terminal_results.jsonl",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("SFT_edited_*.jsonl"):
        path.unlink()


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

    _emit(
        "start",
        (
            f"run_root={run_root} output_dir={output_dir} workers={workers} "
            f"accepted_versions={','.join(sorted(accepted_versions))}"
        ),
    )
    result_paths, page_entries = _discover_confusable_terminals(
        pages_root,
        progress_every=progress_every,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def validate(path: Path) -> ValidationOutcome:
        return _validate_terminal_result(
            path,
            run_root=run_root,
            accepted_versions=accepted_versions,
        )

    accepted = 0
    skipped = 0
    changes_total = 0
    reasons: Counter[str] = Counter()
    last_progress = time.monotonic()
    sft_path = output_dir / "sft.jsonl"
    verl_path = output_dir / "verl.jsonl"
    sft_temporary = output_dir / f".sft.jsonl.tmp.{os.getpid()}"
    verl_temporary = output_dir / f".verl.jsonl.tmp.{os.getpid()}"
    try:
        with (
            sft_temporary.open("w", encoding="utf-8") as sft_handle,
            verl_temporary.open("w", encoding="utf-8") as verl_handle,
            ThreadPoolExecutor(max_workers=workers) as executor,
        ):
            outcomes = executor.map(validate, result_paths)
            for completed, outcome in enumerate(outcomes, start=1):
                if outcome.page is not None:
                    row = _training_fields(outcome.page, output_dir)
                    sft_handle.write(
                        json.dumps(_sft_row(row), ensure_ascii=False) + "\n"
                    )
                    verl_handle.write(
                        json.dumps(_verl_row(row), ensure_ascii=False) + "\n"
                    )
                    accepted += 1
                    changes_total += len(row["changes"])
                else:
                    skipped += 1
                    reasons[outcome.reason or "unknown"] += 1
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
                        "export",
                        (
                            f"completed={completed}/{len(result_paths)} "
                            f"percent={completed / max(1, len(result_paths)):.2%} "
                            f"exported={accepted} skipped={skipped} "
                            f"changes={changes_total} "
                            f"throughput={rate:.1f}_files/s elapsed={elapsed:.1f}s "
                            f"eta={eta:.1f}s current={outcome.result_path.parent.name}"
                        ),
                    )
                    last_progress = now
        sft_temporary.replace(sft_path)
        verl_temporary.replace(verl_path)
    finally:
        for temporary in (sft_temporary, verl_temporary):
            if temporary.exists():
                temporary.unlink()

    _remove_obsolete_outputs(output_dir)

    elapsed = time.monotonic() - started
    report = {
        "snapshot_version": SNAPSHOT_VERSION,
        "pipeline_versions_accepted": sorted(accepted_versions),
        "status": "passed" if accepted else "empty",
        "partial_snapshot": True,
        "compile_performed": False,
        "validation_scope": "accepted_confusable_image_and_exact_gt_only",
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "page_entries_discovered": page_entries,
        "confusable_candidates": len(result_paths),
        "terminal_results_scanned": len(result_paths),
        "edited_pages_exported": accepted,
        "terminal_results_skipped": skipped,
        "skip_reasons": dict(reasons.most_common()),
        "mutation_changes_total": changes_total,
        "outputs": {
            "ms_swift_sft": "sft.jsonl",
            "verl": "verl.jsonl",
        },
        "elapsed_seconds": elapsed,
    }
    _atomic_json(output_dir / "completed_export_report.json", report)
    _emit(
        "finish",
        (
            f"status={report['status']} exported={accepted} "
            f"skipped={skipped} changes={changes_total} "
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
        help="Existing V4 output root containing accepted pages/*/result.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Snapshot directory (default: RUN_ROOT/completed_training_data).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel image/GT readers (default: min(16, CPU count)).",
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
        help="Print parent-process progress after this many directory/files.",
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
