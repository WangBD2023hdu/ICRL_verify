#!/usr/bin/env python3
"""Run source-first v2, confusable editing, export, and verification in one command.

The stable v10 runner and its four-file execution chain remain frozen.  This
entry point orchestrates only experimental v2 programs.  Clean source-first
artifacts live below ``--work-root/source_first_v2`` while the edited-only SFT
and VERL dataset is written to the independent ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arxiv_source_first_v2.contracts import (
    EXPERIMENTAL_CONTRACT,
    EXPERIMENTAL_SCHEMA_VERSION,
    ContractError,
    assert_stable_files,
    validate_experimental_directory,
    validate_page_ledger_file,
)
from arxiv_source_first_v2.contracts import (
    PIPELINE_VERSION as SOURCE_FIRST_PIPELINE_VERSION,
)

PIPELINE_VERSION = "source_bins_to_confusable_verl_v2_edited_v1"
HEARTBEAT_SECONDS = 30.0
SOURCE_FIRST_SCRIPT = SCRIPT_DIR / "run_arxiv_source_bins_to_verl_v2.py"
MUTATION_SCRIPT = SCRIPT_DIR / "build_arxiv_confusable_from_source_first_v2.py"
VERIFIER_SCRIPT = SCRIPT_DIR / "verify_arxiv_confusable_from_source_first_v2.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_text(seconds: float) -> str:
    value = max(0, round(seconds))
    return f"{value // 3600}h{value % 3600 // 60:02d}m{value % 60:02d}s"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def reusable_source_first_report(source_first_root: Path) -> dict[str, Any] | None:
    """Return a completed source-first report that can be reused unchanged.

    Avoiding a no-op rerun matters because the source-first aggregate report
    contains completion timestamps.  Rewriting it would change the hash pinned
    by the edited dataset even when every clean page artifact is identical.
    """

    root = source_first_root.resolve()
    try:
        validate_experimental_directory(root, create=False)
        report = read_json(root / "validation_report_v2.json")
        state = read_json(root / "batch_state_v2.json")
        ledger = validate_page_ledger_file(
            root / "page_ledger_v2.jsonl", require_explicit_outcomes=True
        )
    except (OSError, ValueError, TypeError, ContractError, json.JSONDecodeError):
        return None
    passed = sum(row.get("source_first_passed") is True for row in ledger)
    required = (
        report.get("schema_version") == EXPERIMENTAL_SCHEMA_VERSION,
        report.get("contract") == EXPERIMENTAL_CONTRACT,
        report.get("pipeline_version") == SOURCE_FIRST_PIPELINE_VERSION,
        report.get("status") == "passed",
        int(report.get("pages_total", -1)) == len(ledger),
        int(report.get("pages_passed", -1)) == passed,
        state.get("schema_version") == EXPERIMENTAL_SCHEMA_VERSION,
        state.get("contract") == EXPERIMENTAL_CONTRACT,
        state.get("pipeline_version") == SOURCE_FIRST_PIPELINE_VERSION,
        state.get("status") == "complete",
        int(state.get("pages_total", -1)) == len(ledger),
        int(state.get("pages_passed", -1)) == passed,
    )
    return report if all(required) else None


def contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_isolation(
    *,
    input_root: Path,
    source_first_root: Path,
    output_dir: Path,
    stable_output_roots: Sequence[Path],
) -> None:
    source = input_root.resolve()
    clean = source_first_root.resolve()
    edited = output_dir.resolve()
    for left_name, left, right_name, right in (
        ("input", source, "source-first", clean),
        ("input", source, "edited", edited),
        ("source-first", clean, "edited", edited),
    ):
        if contains(left, right) or contains(right, left):
            raise ContractError(
                f"{left_name} and {right_name} trees must not overlap: "
                f"{left_name}={left} {right_name}={right}"
            )
    for raw in stable_output_roots:
        stable = raw.expanduser().resolve()
        for label, candidate in (("source-first", clean), ("edited", edited)):
            if contains(stable, candidate) or contains(candidate, stable):
                raise ContractError(
                    f"{label} output overlaps declared stable output: "
                    f"stable={stable} {label}={candidate}"
                )


def require_experimental_script(path: Path) -> Path:
    resolved = path.resolve()
    experimental_root = SCRIPT_DIR.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        resolved.relative_to(experimental_root)
    except ValueError as exc:
        raise ContractError(f"orchestrator may execute experimental scripts only: {resolved}") from exc
    return resolved


def run_visible_stage(
    command: Sequence[str],
    *,
    phase: str,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> None:
    """Stream a child program while guaranteeing a visible parent heartbeat."""

    started = time.monotonic()
    print(
        f"[stage-start] phase={phase} command={json.dumps(list(command), ensure_ascii=False)}",
        flush=True,
    )
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line.rstrip("\n"))
        output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    last_visible = time.monotonic()
    stream_finished = False
    while not stream_finished or process.poll() is None:
        timeout = max(0.05, min(1.0, heartbeat_seconds))
        try:
            item = output_queue.get(timeout=timeout)
        except queue.Empty:
            item = ""
        if item is None:
            stream_finished = True
        elif item:
            print(item, flush=True)
            last_visible = time.monotonic()
        now = time.monotonic()
        if now - last_visible >= heartbeat_seconds:
            print(
                f"[progress] phase={phase} status=running "
                f"elapsed={elapsed_text(now-started)}",
                flush=True,
            )
            last_visible = now
    reader.join(timeout=1.0)
    return_code = process.wait()
    elapsed = time.monotonic() - started
    print(
        f"[stage-done] phase={phase} return_code={return_code} "
        f"elapsed={elapsed_text(elapsed)}",
        flush=True,
    )
    if return_code != 0:
        raise RuntimeError(f"stage failed: phase={phase} return_code={return_code}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--stable-output-root", action="append", type=Path, default=[])
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument(
        "--mutation-workers",
        type=int,
        default=0,
        help="0 reuses --workers; mutation recompilation is paper-parallel",
    )
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--paper-ids", nargs="*", default=[])
    parser.add_argument("--max-pages-per-paper", type=int, default=10000)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--min-eligible-visible-characters", type=int, default=80)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--paper-timeout", type=int, default=2400)
    parser.add_argument(
        "--latex-engines",
        default="pdflatex,xelatex,latex_dvips_ps2pdf",
    )
    parser.add_argument(
        "--figure-policy",
        choices=("drop_then_keep", "drop", "keep"),
        default="drop_then_keep",
    )
    parser.add_argument(
        "--drop-references",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-crawler-unfiltered-license", action="store_true")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--latexmk",
        type=Path,
        default=Path(shutil.which("latexmk") or "/Library/TeX/texbin/latexmk"),
    )
    parser.add_argument(
        "--pdftoppm",
        type=Path,
        default=Path(shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=HEARTBEAT_SECONDS,
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= 256:
        raise ValueError("--workers must be between 1 and 256")
    if args.mutation_workers < 0 or args.mutation_workers > 256:
        raise ValueError("--mutation-workers must be between 0 and 256")
    if args.max_papers < 0:
        raise ValueError("--max-papers must be non-negative")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if not 0.0 < args.heartbeat_seconds <= HEARTBEAT_SECONDS:
        raise ValueError("--heartbeat-seconds must be >0 and <=30")
    for name in (
        "max_pages_per_paper",
        "dpi",
        "min_eligible_visible_characters",
        "compile_timeout",
        "paper_timeout",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def append_common_tool_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--latexmk",
            str(args.latexmk.expanduser().resolve()),
            "--pdftoppm",
            str(args.pdftoppm.expanduser().resolve()),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    started = time.monotonic()
    input_root = args.input_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    source_first_root = work_root / "source_first_v2"
    output_dir = args.output_dir.expanduser().resolve()
    validate_isolation(
        input_root=input_root,
        source_first_root=source_first_root,
        output_dir=output_dir,
        stable_output_roots=args.stable_output_root,
    )
    stable_before = assert_stable_files(REPO_ROOT)
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    source_script = require_experimental_script(SOURCE_FIRST_SCRIPT)
    mutation_script = require_experimental_script(MUTATION_SCRIPT)
    verifier_script = require_experimental_script(VERIFIER_SCRIPT)
    mutation_workers = args.mutation_workers or args.workers
    print(
        f"[start] pipeline={PIPELINE_VERSION} input={input_root} work={work_root} "
        f"source_first={source_first_root} output={output_dir} "
        f"workers={args.workers} mutation_workers={mutation_workers} "
        f"max_papers={args.max_papers} resume={args.resume} dry_run={args.dry_run}",
        flush=True,
    )

    source_command = [
        str(python),
        str(source_script),
        "--input-root",
        str(input_root),
        "--output-dir",
        str(source_first_root),
        "--workers",
        str(args.workers),
        "--max-papers",
        str(args.max_papers),
        "--max-pages-per-paper",
        str(args.max_pages_per_paper),
        "--dpi",
        str(args.dpi),
        "--min-eligible-visible-characters",
        str(args.min_eligible_visible_characters),
        "--compile-timeout",
        str(args.compile_timeout),
        "--paper-timeout",
        str(args.paper_timeout),
        "--latex-engines",
        args.latex_engines,
        "--figure-policy",
        args.figure_policy,
        "--heartbeat-seconds",
        str(args.heartbeat_seconds),
    ]
    append_common_tool_args(source_command, args)
    source_command.append("--drop-references" if args.drop_references else "--no-drop-references")
    # The established v2 source-first runner resumes by default and exposes
    # only the opt-out spelling.  Do not invent an unsupported ``--resume``.
    if not args.resume:
        source_command.append("--no-resume")
    if args.retry_failed:
        source_command.append("--retry-failed")
    if args.allow_crawler_unfiltered_license:
        source_command.append("--allow-crawler-unfiltered-license")
    if args.paper_ids:
        source_command.extend(["--paper-ids", *map(str, args.paper_ids)])
    for stable_root in args.stable_output_root:
        source_command.extend(["--stable-output-root", str(stable_root)])
    if args.dry_run:
        source_command.append("--dry-run")
        run_visible_stage(
            source_command,
            phase="source_first_dry_run",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        print("[finish] status=dry_run edited_output_created=false", flush=True)
        return 0

    work_root.mkdir(parents=True, exist_ok=True)
    source_report = (
        reusable_source_first_report(source_first_root)
        if args.resume and not args.retry_failed
        else None
    )
    if source_report is None:
        run_visible_stage(
            source_command,
            phase="source_first_v2",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        source_report = read_json(source_first_root / "validation_report_v2.json")
    else:
        print(
            f"[checkpoint] phase=source_first_v2 status=reused pages="
            f"{source_report.get('pages_passed', 0)} root={source_first_root}",
            flush=True,
        )
    if source_report.get("status") != "passed":
        raise RuntimeError(
            "source-first v2 report did not pass: "
            f"{source_first_root / 'validation_report_v2.json'}"
        )

    mutation_command = [
        str(python),
        str(mutation_script),
        "--source-first-root",
        str(source_first_root),
        "--output-dir",
        str(output_dir),
        "--server-root",
        args.server_root,
        "--max-papers",
        str(args.max_papers),
        "--workers",
        str(mutation_workers),
        "--seed",
        str(args.seed),
        "--split-seed",
        str(args.split_seed),
        "--val-fraction",
        str(args.val_fraction),
        "--dpi",
        str(args.dpi),
        "--compile-timeout",
        str(args.compile_timeout),
    ]
    append_common_tool_args(mutation_command, args)
    mutation_command.append("--resume" if args.resume else "--no-resume")
    if args.paper_ids:
        mutation_command.extend(["--paper-ids", *map(str, args.paper_ids)])
    for stable_root in args.stable_output_root:
        mutation_command.extend(["--stable-output-root", str(stable_root)])
    run_visible_stage(
        mutation_command,
        phase="confusable_mutation_recompile_export",
        heartbeat_seconds=args.heartbeat_seconds,
    )

    verifier_command = [
        str(python),
        str(verifier_script),
        "--dataset-root",
        str(output_dir),
        "--source-first-root",
        str(source_first_root),
    ]
    run_visible_stage(
        verifier_command,
        phase="independent_verifier",
        heartbeat_seconds=args.heartbeat_seconds,
    )

    validation = read_json(output_dir / "validation_report.json")
    verification = read_json(output_dir / "independent_verifier_report.json")
    stable_after = assert_stable_files(REPO_ROOT)
    final = {
        "pipeline_version": PIPELINE_VERSION,
        "status": (
            "passed"
            if validation.get("status") == "passed"
            and verification.get("status") == "passed"
            else "failed"
        ),
        "input_root": str(input_root),
        "work_root": str(work_root),
        "source_first_root": str(source_first_root),
        "output_dir": str(output_dir),
        "server_root": args.server_root,
        "papers_selected": int(source_report.get("papers_selected", 0)),
        "papers_source_first_success": int(source_report.get("papers_success", 0)),
        "source_first_verified_pages": int(source_report.get("pages_passed", 0)),
        "edit_pairs": int(validation.get("accepted_pairs", 0)),
        "mutation_count_distribution": validation.get(
            "mutation_count_distribution", {}
        ),
        "verl_train": int(validation.get("exports", {}).get("train", 0)),
        "verl_val": int(validation.get("exports", {}).get("val", 0)),
        "independent_verifier": verification.get("status"),
        "stable_guard_before": stable_before,
        "stable_guard_after": stable_after,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(output_dir / "pipeline_report.json", final)
    print(
        f"[finish] status={final['status']} papers={final['papers_source_first_success']}/"
        f"{final['papers_selected']} source_first_pages={final['source_first_verified_pages']} "
        f"edit_pairs={final['edit_pairs']} mutation_distribution="
        f"{final['mutation_count_distribution']} verl_train={final['verl_train']} "
        f"verl_val={final['verl_val']} elapsed={elapsed_text(final['elapsed_seconds'])} "
        f"output={output_dir}",
        flush=True,
    )
    return 0 if final["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
