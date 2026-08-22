#!/usr/bin/env python3
"""Build source-first confusable OCR SFT/VERL data from downloaded arXiv bins.

Input is the directory produced by ``crawl_arxiv_sources.py``: it must contain
``results.jsonl`` and ``papers/<stem>/source_archive.bin``.  Papers are safely
extracted and source-first GT is built in process-isolated workers.  The edit
recompile stage also runs one paper per process, then exports SFT plus VERL
JSONL/Parquet and runs the independent verifier.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_arxiv_latex_recompile_pilot import (  # noqa: E402
    ALLOWED_LICENSES,
    extract_source,
    find_main_tex,
    scan_source,
)


HEARTBEAT_SECONDS = 30.0
PIPELINE_VERSION = "source_bins_to_source_first_confusable_verl_v1"
SAFE_STEM_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    values = list(rows)
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in values),
    )
    return len(values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elapsed_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def paper_stem(row: dict[str, Any]) -> str:
    stem = str(row.get("stem") or f"{row.get('arxiv_id', '')}{row.get('version', '')}")
    if not stem or not SAFE_STEM_RE.fullmatch(stem):
        raise ValueError(f"unsafe or missing paper stem: {stem!r}")
    return stem


def resolve_archive(input_root: Path, row: dict[str, Any]) -> Path:
    stem = paper_stem(row)
    candidates: list[Path] = []
    for key in ("archive", "archive_path"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        candidates.append(path if path.is_absolute() else input_root / path)
    candidates.append(input_root / "papers" / stem / "source_archive.bin")
    existing = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one source archive for {stem}; found={existing}"
        )
    if existing[0].stat().st_size == 0:
        raise ValueError(f"empty source archive: {existing[0]}")
    return existing[0]


def archive_expected_sha256(row: dict[str, Any]) -> str | None:
    direct = row.get("sha256")
    if direct:
        return str(direct)
    download = row.get("download")
    if isinstance(download, dict) and download.get("sha256"):
        return str(download["sha256"])
    return None


def move_incomplete(path: Path, diagnostics_root: Path, label: str) -> Path:
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    destination = diagnostics_root / f"{label}_{timestamp}_{os.getpid()}"
    shutil.move(str(path), str(destination))
    return destination


def terminate_process(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_logged_subprocess(
    command: list[str],
    *,
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        while process.poll() is None:
            if time.monotonic() - started > timeout_seconds:
                timed_out = True
                terminate_process(process)
                break
            time.sleep(1.0)
    return {
        "command": command,
        "return_code": process.poll(),
        "timed_out": timed_out,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path),
        "log_bytes": log_path.stat().st_size,
    }


def source_first_artifacts_valid(source_first_root: Path) -> bool:
    report_path = source_first_root / "validation_report.json"
    pages_path = source_first_root / "pages_passed.jsonl"
    if not report_path.is_file() or not pages_path.is_file():
        return False
    try:
        report = read_json(report_path)
        rows = read_jsonl(pages_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if report.get("status") != "passed" or not rows:
        return False
    for row in rows:
        for key in ("markdown", "image"):
            path = source_first_root / str(row.get(key, ""))
            if not path.is_file() or path.stat().st_size == 0:
                return False
        sidecar = source_first_root / str(row["markdown"])
        if not sidecar.with_suffix(".json").is_file():
            return False
    return True


def build_source_first_paper(task: dict[str, Any]) -> dict[str, Any]:
    """Safely extract one archive and build its verified source-first pages."""

    row = dict(task["row"])
    stem = paper_stem(row)
    archive = Path(str(task["archive"])).resolve()
    recompile_paper = Path(str(task["recompile_root"])) / "papers" / stem
    source_dir = recompile_paper / "source"
    metadata_path = recompile_paper / "metadata.json"
    source_first_root = Path(str(task["source_first_root"])) / stem
    diagnostics_root = recompile_paper / "diagnostics"
    result: dict[str, Any] = {
        **row,
        "stem": stem,
        "status": "failed",
        "stage": "initializing",
        "archive_path": str(archive),
        "source_dir": str(source_dir.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "source_first_root": str(source_first_root.resolve()),
        "pipeline_version": PIPELINE_VERSION,
        "started_at_epoch": time.time(),
    }
    recompile_paper.mkdir(parents=True, exist_ok=True)
    if bool(task["resume"]) and metadata_path.is_file():
        stored = read_json(metadata_path)
        if stored.get("status") == "success" and source_first_artifacts_valid(source_first_root):
            return {**stored, "resume_state": "reused_success"}
        if stored.get("status") in {"failed", "rejected"} and not bool(task["retry_failed"]):
            return {**stored, "resume_state": "reused_failure"}

    expected_sha256 = task.get("expected_sha256")
    actual_sha256 = sha256_file(archive)
    if expected_sha256 and actual_sha256 != expected_sha256:
        result.update(
            status="rejected",
            stage="archive_validation",
            failure_reason="source archive SHA-256 mismatch",
            archive_sha256=actual_sha256,
        )
        atomic_write_json(metadata_path, result)
        return result
    result["archive_sha256"] = actual_sha256

    extraction_path = recompile_paper / "extraction.json"
    if not bool(task["resume"]):
        if extraction_path.exists():
            result["previous_extraction_checkpoint_moved_to"] = str(
                move_incomplete(extraction_path, diagnostics_root, "extraction_checkpoint")
            )
        if source_dir.exists():
            result["previous_source_moved_to"] = str(
                move_incomplete(source_dir, diagnostics_root, "source_no_resume")
            )
        if source_first_root.exists():
            result["previous_source_first_moved_to"] = str(
                move_incomplete(source_first_root, diagnostics_root, "source_first_no_resume")
            )
    if not extraction_path.is_file():
        if source_dir.exists():
            result["incomplete_source_moved_to"] = str(
                move_incomplete(source_dir, diagnostics_root, "source_incomplete")
            )
        try:
            extraction = extract_source(archive, source_dir)
        except Exception as exc:  # noqa: BLE001
            result.update(
                status="failed",
                stage="extraction",
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
            atomic_write_json(metadata_path, result)
            return result
        atomic_write_json(extraction_path, {"status": "passed", **extraction})
    result["extraction"] = read_json(extraction_path)
    result["stage"] = "extracted"

    safety_path = recompile_paper / "safety_scan.json"
    safety = scan_source(source_dir)
    atomic_write_json(safety_path, safety)
    result["safety_scan_path"] = str(safety_path.resolve())
    result["safety_scan"] = safety
    if safety.get("status") != "passed":
        result.update(
            status="rejected",
            stage="safety_scan",
            failure_reason="dangerous source construct detected before compilation",
        )
        atomic_write_json(metadata_path, result)
        return result

    try:
        main_tex, candidates = find_main_tex(source_dir)
    except Exception as exc:  # noqa: BLE001
        result.update(
            status="failed",
            stage="main_tex_selection",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(metadata_path, result)
        return result
    main_relative = main_tex.resolve().relative_to(source_dir.resolve())
    result["main_tex"] = str(main_tex.resolve())
    result["main_tex_candidates"] = candidates

    if source_first_root.exists() and not source_first_artifacts_valid(source_first_root):
        result["incomplete_source_first_moved_to"] = str(
            move_incomplete(source_first_root, diagnostics_root, "source_first_incomplete")
        )
    if not source_first_artifacts_valid(source_first_root):
        runs: list[dict[str, Any]] = []
        for engine_index, engine in enumerate(task["latex_engines"], start=1):
            if source_first_root.exists():
                moved = move_incomplete(
                    source_first_root,
                    diagnostics_root,
                    f"source_first_{engine}_failed",
                )
                result.setdefault("source_first_failed_attempts", []).append(str(moved))
            command = [
                str(task["python"]),
                str(task["source_first_script"]),
                "--source-dir",
                str(source_dir.resolve()),
                "--main-tex",
                main_relative.as_posix(),
                "--output-dir",
                str(source_first_root.resolve()),
                "--paper-id",
                stem,
                "--drop-references",
                "--max-pages",
                str(task["max_pages_per_paper"]),
                "--dpi",
                str(task["dpi"]),
                "--compile-timeout",
                str(task["compile_timeout"]),
                "--engine",
                str(engine),
                "--latexmk",
                str(task["latexmk"]),
                "--pdftoppm",
                str(task["pdftoppm"]),
            ]
            run = run_logged_subprocess(
                command,
                log_path=Path(str(task["log_root"])) / f"{stem}.{engine}.log",
                timeout_seconds=int(task["paper_timeout"]),
            )
            run["engine"] = engine
            run["attempt"] = engine_index
            runs.append(run)
            if run["return_code"] == 0 and source_first_artifacts_valid(source_first_root):
                break
        result["source_first_runs"] = runs
        if not source_first_artifacts_valid(source_first_root):
            last_run = runs[-1]
            tail = Path(last_run["log"]).read_text(encoding="utf-8", errors="replace")[-4000:]
            result.update(
                status="failed",
                stage="source_first_gt",
                failure_reason=(
                    f"all source-first engines failed engines={task['latex_engines']} "
                    f"last_rc={last_run['return_code']} timed_out={last_run['timed_out']} "
                    f"tail={tail}"
                ),
            )
            atomic_write_json(metadata_path, result)
            return result

    report = read_json(source_first_root / "validation_report.json")
    result.update(
        status="success",
        stage="complete",
        compile={
            "status": "success",
            "engine": str(report.get("compile_engine") or "pdflatex"),
            "pdf": report["clean_pdf"],
        },
        pdf_inspection={
            "pdf": report["clean_pdf"],
            "pages": int(report["pages_passed"]) + int(report["pages_rejected"]),
        },
        source_first_report=str((source_first_root / "validation_report.json").resolve()),
        source_first_pages_passed=int(report["pages_passed"]),
        source_first_pages_rejected=int(report["pages_rejected"]),
        completed_at_epoch=time.time(),
    )
    atomic_write_json(metadata_path, result)
    return result


def select_input_rows(
    input_root: Path,
    *,
    paper_ids: set[str],
    max_papers: int,
) -> list[dict[str, Any]]:
    results_path = input_root / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(results_path):
        stem = paper_stem(row)
        if stem in seen:
            raise ValueError(f"duplicate paper stem in crawler results: {stem}")
        seen.add(stem)
        if row.get("status") not in {"passed", "success"}:
            continue
        if row.get("license_name") not in ALLOWED_LICENSES:
            continue
        if paper_ids and stem not in paper_ids and str(row.get("arxiv_id", "")) not in paper_ids:
            continue
        resolve_archive(input_root, row)
        selected.append(row)
        if max_papers and len(selected) >= max_papers:
            break
    if paper_ids:
        found = {paper_stem(row) for row in selected} | {
            str(row.get("arxiv_id", "")) for row in selected
        }
        missing = sorted(paper_ids - found)
        if missing:
            raise ValueError(f"requested paper IDs are unavailable or ineligible: {missing}")
    if not selected:
        raise ValueError("no eligible downloaded source archives were selected")
    return selected


def build_source_first_manifest(
    source_first_root: Path,
    results: list[dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for result in sorted(results, key=lambda row: str(row["stem"])):
        if result.get("status") != "success":
            continue
        paper_root = source_first_root / str(result["stem"])
        for row in read_jsonl(paper_root / "pages_passed.jsonl"):
            data_id = str(row["data_id"])
            if data_id in seen_ids:
                raise ValueError(f"duplicate source-first data_id: {data_id}")
            seen_ids.add(data_id)
            cases.append(
                {
                    "pair_id": data_id,
                    "image": str((paper_root / str(row["image"])).resolve()),
                    "markdown_path": str((paper_root / str(row["markdown"])).resolve()),
                }
            )
    cases.sort(key=lambda row: str(row["pair_id"]))
    if not cases:
        raise RuntimeError("source-first stage produced no verified pages")
    atomic_write_json(output_path, cases)
    return cases


def run_visible_stage(command: list[str], *, phase: str) -> None:
    print(f"[launch] phase={phase} command={' '.join(command)}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    heartbeat = 0
    try:
        while True:
            try:
                process.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                heartbeat += 1
                print(
                    f"[progress] phase={phase} heartbeat={heartbeat} "
                    f"elapsed={elapsed_text(time.monotonic() - started)} process_running=true",
                    flush=True,
                )
    except KeyboardInterrupt:
        terminate_process(process)
        raise
    if process.returncode != 0:
        raise RuntimeError(f"stage failed: phase={phase} return_code={process.returncode}")
    print(
        f"[checkpoint] phase={phase} status=passed "
        f"elapsed={elapsed_text(time.monotonic() - started)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--max-papers", type=int, default=0, help="0 means every eligible downloaded paper")
    parser.add_argument("--paper-ids", nargs="*", default=[])
    parser.add_argument("--max-pages-per-paper", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--paper-timeout", type=int, default=2400)
    parser.add_argument(
        "--latex-engines",
        default="pdflatex,xelatex,latex_dvips_ps2pdf",
        help="comma-separated source-first engine fallback order",
    )
    parser.add_argument("--latexmk", type=Path, default=Path(shutil.which("latexmk") or "/Library/TeX/texbin/latexmk"))
    parser.add_argument("--pdftoppm", type=Path, default=Path(shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--retry-failed", action="store_true")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("--workers must be between 1 and 64")
    if args.max_papers < 0 or args.max_pages_per_paper <= 0:
        raise ValueError("--max-papers must be non-negative and max pages must be positive")
    if args.compile_timeout <= 0 or args.paper_timeout <= 0:
        raise ValueError("compile and paper timeouts must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    latex_engines = [value.strip() for value in args.latex_engines.split(",") if value.strip()]
    supported_engines = {"pdflatex", "xelatex", "latex_dvips_ps2pdf"}
    if not latex_engines or len(latex_engines) != len(set(latex_engines)):
        raise ValueError("--latex-engines must be a non-empty unique list")
    if not set(latex_engines) <= supported_engines:
        raise ValueError(f"unsupported engines: {sorted(set(latex_engines) - supported_engines)}")
    input_root = args.input_root.resolve()
    work_root = args.work_root.resolve()
    output_dir = args.output_dir.resolve()
    latexmk = args.latexmk.expanduser().absolute()
    pdftoppm = args.pdftoppm.expanduser().resolve()
    python = args.python.expanduser().resolve()
    for name, tool in (("latexmk", latexmk), ("pdftoppm", pdftoppm), ("python", python)):
        if not tool.is_file() or not os.access(tool, os.X_OK):
            raise FileNotFoundError(f"required executable unavailable: {name}={tool}")
    work_root.mkdir(parents=True, exist_ok=True)
    recompile_root = work_root / "recompile"
    source_first_root = work_root / "source_first"
    log_root = work_root / "logs" / "source_first"
    manifest_path = work_root / "source_first_cases.json"
    for path in (recompile_root, source_first_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    selected = select_input_rows(
        input_root,
        paper_ids=set(args.paper_ids),
        max_papers=args.max_papers,
    )
    total_bytes = sum(resolve_archive(input_root, row).stat().st_size for row in selected)
    print(
        f"[start] pipeline={PIPELINE_VERSION} papers={len(selected)} bytes={total_bytes} "
        f"workers={args.workers} resume={args.resume} retry_failed={args.retry_failed} "
        f"input={input_root} work={work_root} output={output_dir}",
        flush=True,
    )

    source_first_script = SCRIPT_DIR / "build_source_first_color_page_gt.py"
    tasks: list[dict[str, Any]] = []
    for row in selected:
        tasks.append(
            {
                "row": row,
                "archive": str(resolve_archive(input_root, row)),
                "expected_sha256": archive_expected_sha256(row),
                "recompile_root": str(recompile_root),
                "source_first_root": str(source_first_root),
                "log_root": str(log_root),
                "source_first_script": str(source_first_script),
                "python": str(python),
                "latexmk": str(latexmk),
                "pdftoppm": str(pdftoppm),
                "max_pages_per_paper": args.max_pages_per_paper,
                "dpi": args.dpi,
                "compile_timeout": args.compile_timeout,
                "paper_timeout": args.paper_timeout,
                "latex_engines": latex_engines,
                "resume": args.resume,
                "retry_failed": args.retry_failed,
            }
        )

    started = time.monotonic()
    results_by_stem: dict[str, dict[str, Any]] = {}
    completed = accepted = rejected = errors = pages = bytes_done = 0
    if args.workers > 1:
        try:
            executor: concurrent.futures.Executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            )
            executor_mode = "process"
        except (PermissionError, NotImplementedError, OSError) as exc:
            print(
                f"[warning] process_pool_unavailable={type(exc).__name__}:{exc}; "
                f"fallback=thread workers={args.workers}",
                flush=True,
            )
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
            executor_mode = "thread_fallback"
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor_mode = "single_worker"
    print(
        f"[parallel_start] phase=source_first executor={executor_mode} workers={args.workers}",
        flush=True,
    )
    with executor:
        future_to_task = {
            executor.submit(build_source_first_paper, task): task for task in tasks
        }
        pending = set(future_to_task)
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=HEARTBEAT_SECONDS,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                elapsed = max(time.monotonic() - started, 1e-9)
                throughput = completed / elapsed
                eta = (len(tasks) - completed) / throughput if throughput else math.inf
                print(
                    f"[progress] phase=source_first papers={completed}/{len(tasks)} "
                    f"pct={100 * completed / len(tasks):.1f}% pages={pages} bytes={bytes_done}/{total_bytes} "
                    f"throughput={throughput:.3f}_papers/s elapsed={elapsed_text(elapsed)} "
                    f"eta={'unknown' if not math.isfinite(eta) else elapsed_text(eta)} "
                    f"accepted={accepted} rejected={rejected} errors={errors} running={len(pending)}",
                    flush=True,
                )
                continue
            for future in done:
                task = future_to_task[future]
                stem = paper_stem(task["row"])
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        **task["row"],
                        "stem": stem,
                        "status": "failed",
                        "stage": "worker_exception",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "source_first_pages_passed": 0,
                    }
                results_by_stem[stem] = result
                completed += 1
                bytes_done += Path(str(task["archive"])).stat().st_size
                pages += int(result.get("source_first_pages_passed", 0))
                if result.get("status") == "success":
                    accepted += 1
                elif result.get("status") == "rejected":
                    rejected += 1
                else:
                    errors += 1
                ordered = [results_by_stem[paper_stem(row)] for row in selected if paper_stem(row) in results_by_stem]
                atomic_write_jsonl(recompile_root / "results.jsonl", ordered)
                elapsed = max(time.monotonic() - started, 1e-9)
                throughput = completed / elapsed
                eta = (len(tasks) - completed) / throughput if throughput else 0.0
                print(
                    f"[unit-done] phase=source_first papers={completed}/{len(tasks)} "
                    f"pct={100 * completed / len(tasks):.1f}% current={stem} "
                    f"status={result.get('status')} pages={result.get('source_first_pages_passed', 0)} "
                    f"bytes={bytes_done}/{total_bytes} throughput={throughput:.3f}_papers/s "
                    f"elapsed={elapsed_text(elapsed)} eta={elapsed_text(eta)} "
                    f"accepted={accepted} rejected={rejected} errors={errors}",
                    flush=True,
                )

    ordered_results = [results_by_stem[paper_stem(row)] for row in selected]
    atomic_write_jsonl(recompile_root / "results.jsonl", ordered_results)
    cases = build_source_first_manifest(source_first_root, ordered_results, manifest_path)
    source_summary = {
        "pipeline_version": PIPELINE_VERSION,
        "status": "passed" if accepted > 0 and cases else "failed",
        "papers_selected": len(selected),
        "papers_success": accepted,
        "papers_rejected": rejected,
        "papers_failed": errors,
        "verified_pages": len(cases),
        "workers": args.workers,
        "input_bytes": total_bytes,
        "manifest": str(manifest_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(work_root / "source_first_summary.json", source_summary)
    print(
        f"[checkpoint] phase=source_first status={source_summary['status']} "
        f"papers={accepted}/{len(selected)} pages={len(cases)} rejected={rejected} errors={errors}",
        flush=True,
    )

    mutation_script = SCRIPT_DIR / "build_arxiv_confusable_recompile_pilot.py"
    mutation_command = [
        str(python),
        str(mutation_script),
        "--recompile-root",
        str(recompile_root),
        "--clean-gt-root",
        str(work_root),
        "--source-first-case-manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--server-root",
        args.server_root,
        "--max-papers",
        str(accepted),
        "--workers",
        str(args.workers),
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
        "--latexmk",
        str(latexmk),
        "--pdftoppm",
        str(pdftoppm),
    ]
    if args.resume:
        mutation_command.append("--resume")
    run_visible_stage(mutation_command, phase="mutation_recompile_and_export")

    verifier_script = SCRIPT_DIR / "verify_arxiv_confusable_recompile_pilot.py"
    verifier_command = [
        str(python),
        str(verifier_script),
        "--dataset-root",
        str(output_dir),
        "--source-first-case-manifest",
        str(manifest_path),
    ]
    run_visible_stage(verifier_command, phase="independent_verifier")
    validation = read_json(output_dir / "validation_report.json")
    verification = read_json(output_dir / "independent_verifier_report.json")
    final = {
        "pipeline_version": PIPELINE_VERSION,
        "status": (
            "passed"
            if validation.get("status") == "passed" and verification.get("status") == "passed"
            else "failed"
        ),
        "input_root": str(input_root),
        "work_root": str(work_root),
        "output_dir": str(output_dir),
        "server_root": args.server_root,
        "papers_selected": len(selected),
        "papers_source_first_success": accepted,
        "source_first_verified_pages": len(cases),
        "edit_pairs": int(validation.get("accepted_pairs", 0)),
        "verl_train": int(validation.get("exports", {}).get("train", 0)),
        "verl_val": int(validation.get("exports", {}).get("val", 0)),
        "independent_verifier": verification.get("status"),
    }
    atomic_write_json(output_dir / "pipeline_report.json", final)
    print(
        f"[finish] status={final['status']} papers={accepted}/{len(selected)} "
        f"source_first_pages={len(cases)} edit_pairs={final['edit_pairs']} "
        f"verl_train={final['verl_train']} verl_val={final['verl_val']} output={output_dir}",
        flush=True,
    )
    return 0 if final["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
