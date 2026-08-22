#!/usr/bin/env python3
"""Safely download and recompile a five-paper arXiv LaTeX pilot.

The paper list is deliberately fixed after selection from the local
``outputs/arxiv_chaos_2000`` OAI-derived license manifest.  No source-provided
scripts or latexmk configuration files are executed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


SELECTED_PAPERS = [
    {"arxiv_id": "2208.01224", "version": "v1", "domain": "mathematics"},
    {"arxiv_id": "2402.14521", "version": "v2", "domain": "natural_language_processing"},
    {"arxiv_id": "2402.14031", "version": "v2", "domain": "control_and_machine_learning"},
    {"arxiv_id": "2402.17922", "version": "v3", "domain": "quantum_information"},
    {"arxiv_id": "2606.02095", "version": "v1", "domain": "economics_and_game_theory"},
]

ALLOWED_LICENSES = {"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"}
TEXT_SUFFIXES = {".tex", ".sty", ".cls", ".bib", ".bbl", ".cfg", ".def", ".ltx"}
MAX_ARCHIVE_FILES = 5_000
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_MEMBER_BYTES = 100 * 1024 * 1024
HEARTBEAT_SECONDS = 15.0

DANGEROUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("write18_or_shell_escape", re.compile(r"(?:\\write\s*18|write18|\\ShellEscape)", re.I)),
    ("pipe_input", re.compile(r"\\(?:input|include|openin|openout)\b[^\n]{0,100}[\"']?\|", re.I)),
    (
        "absolute_input_path",
        re.compile(
            r"\\(?:input|include|includegraphics|bibliography|addbibresource)"
            r"(?:\s*\[[^\]]*\])?\s*\{?\s*(?:/|~[/\\]|[A-Za-z]:[/\\])",
            re.I,
        ),
    ),
    (
        "shell_executing_package",
        re.compile(r"\\(?:usepackage|RequirePackage)(?:\s*\[[^\]]*\])?\s*\{[^}]*\b(?:minted|pythontex|shellesc)\b", re.I),
    ),
    ("executable_environment", re.compile(r"\\begin\s*\{(?:minted|pycode|asy)\}", re.I)),
    ("direct_lua", re.compile(r"\\directlua\b", re.I)),
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Progress:
    total: int
    started: float = field(default_factory=time.monotonic)
    completed: int = 0
    success: int = 0
    failed: int = 0
    rejected: int = 0
    downloaded_bytes: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _rate_started: float | None = field(default=None, repr=False)
    _rate_completed_offset: int = field(default=0, repr=False)

    def add_downloaded_bytes(self, value: int) -> None:
        with self._lock:
            self.downloaded_bytes += value

    def record_status(self, status: str) -> None:
        with self._lock:
            self.completed += 1
            if status == "success":
                self.success += 1
            elif status == "rejected":
                self.rejected += 1
            else:
                self.failed += 1

    def reset_rate_baseline(self) -> None:
        """Exclude instantaneous resume accounting from throughput and ETA."""
        with self._lock:
            self._rate_started = time.monotonic()
            self._rate_completed_offset = self.completed

    def emit(self, current: str, stage: str, detail: str = "") -> None:
        with self._lock:
            completed = self.completed
            success = self.success
            failed = self.failed
            rejected = self.rejected
            downloaded_bytes = self.downloaded_bytes
            rate_started = self._rate_started or self.started
            rate_completed_offset = self._rate_completed_offset
        elapsed = max(time.monotonic() - rate_started, 0.001)
        pct = 100.0 * completed / self.total if self.total else 100.0
        throughput_completed = max(0, completed - rate_completed_offset)
        throughput = throughput_completed / elapsed * 60.0
        eta = (self.total - completed) / throughput * 60.0 if throughput > 0 else None
        eta_text = f"{eta:.0f}s" if eta is not None else "unknown"
        suffix = f" detail={detail}" if detail else ""
        print(
            f"[progress] completed={completed}/{self.total} pct={pct:.1f}% "
            f"success={success} fail={failed} reject={rejected} "
            f"bytes={downloaded_bytes} throughput={throughput:.2f}_papers/min "
            f"elapsed={elapsed:.1f}s ETA={eta_text} current={current} stage={stage}{suffix}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/arxiv_chaos_2000"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/arxiv_latex_recompile_pilot_5"))
    parser.add_argument("--latexmk", type=Path, default=Path("/Library/TeX/texbin/latexmk"))
    parser.add_argument("--pdfinfo", type=Path, default=Path("/opt/homebrew/bin/pdfinfo"))
    parser.add_argument("--pdftoppm", type=Path, default=Path("/opt/homebrew/bin/pdftoppm"))
    parser.add_argument("--download-timeout", type=int, default=20)
    parser.add_argument("--compile-timeout", type=int, default=240)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ARXIV_ID_VERSION",
        help="process only this selected stem and merge it into existing rollups; repeatable",
    )
    return parser.parse_args()


def load_selection(dataset_root: Path) -> list[dict[str, Any]]:
    license_path = dataset_root / "licenses.jsonl"
    if not license_path.is_file():
        raise FileNotFoundError(f"missing local license manifest: {license_path}")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with license_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records[(row["arxiv_id"], row["version"])] = row

    selected: list[dict[str, Any]] = []
    for fixed in SELECTED_PAPERS:
        key = (fixed["arxiv_id"], fixed["version"])
        if key not in records:
            raise ValueError(f"selected paper absent from licenses.jsonl: {key}")
        row = records[key]
        if row.get("license_name") not in ALLOWED_LICENSES:
            raise ValueError(f"disallowed or ambiguous license for {key}: {row.get('license_name')!r}")
        stem = f"{row['arxiv_id']}{row['version']}"
        selected.append(
            {
                **fixed,
                "stem": stem,
                "title": row["title"],
                "authors": row["authors"],
                "categories": row["categories"].split(),
                "license_name": row["license_name"],
                "license_url": row["license_url"],
                "submitted": row["submitted"],
                "local_metadata_source": str(license_path),
                "oai_source": row["source"],
                "abstract_url": f"https://arxiv.org/abs/{stem}",
                "source_url": f"https://export.arxiv.org/e-print/{stem}",
            }
        )
    return selected


def download_source(
    url: str,
    destination: Path,
    timeout: int,
    progress: Progress,
    current: str,
    force: bool,
) -> dict[str, Any]:
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        progress.add_downloaded_bytes(destination.stat().st_size)
        progress.emit(current, "download_resume", f"reused={destination.stat().st_size}")
        return {
            "url": url,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "reused": True,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ICLR-val-arxiv-recompile-pilot/1.0 (research reproducibility pilot)",
            "Accept": "application/x-eprint-tar, application/gzip, application/octet-stream",
        },
    )
    progress.emit(current, "download_start", url)
    downloaded = 0
    last_log = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
        content_type = response.headers.get("Content-Type", "")
        total_header = response.headers.get("Content-Length")
        expected = int(total_header) if total_header and total_header.isdigit() else None
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            progress.add_downloaded_bytes(len(chunk))
            if time.monotonic() - last_log >= HEARTBEAT_SECONDS:
                detail = f"downloaded={downloaded} expected={expected or 'unknown'}"
                progress.emit(current, "downloading", detail)
                last_log = time.monotonic()
        handle.flush()
        os.fsync(handle.fileno())
    if downloaded == 0:
        raise ValueError("download returned zero bytes")
    if expected is not None and downloaded != expected:
        partial.unlink(missing_ok=True)
        raise EOFError(
            f"download length mismatch: received={downloaded} expected={expected}"
        )
    prefix = partial.read_bytes()[:512].lower()
    if b"<html" in prefix or b"<!doctype html" in prefix:
        raise ValueError(f"download returned HTML rather than an e-print archive ({content_type})")
    os.replace(partial, destination)
    progress.emit(current, "download_complete", f"bytes={downloaded}")
    return {
        "url": url,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "content_type": content_type,
        "reused": False,
    }


def validate_member_name(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def copy_member(source: BinaryIO, destination: Path) -> int:
    written = 0
    with destination.open("wb") as handle:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_MEMBER_BYTES:
                raise ValueError(f"archive member exceeds limit: {destination}")
            handle.write(chunk)
    return written


def extract_source(archive: Path, source_dir: Path) -> dict[str, Any]:
    if source_dir.is_dir() and any(source_dir.rglob("*")):
        files = [p for p in source_dir.rglob("*") if p.is_file()]
        return {"format": "resumed", "files": len(files), "bytes": sum(p.stat().st_size for p in files)}

    source_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = 0
    extracted_bytes = 0
    try:
        with tarfile.open(archive, mode="r:*") as bundle:
            members = bundle.getmembers()
            if len(members) > MAX_ARCHIVE_FILES:
                raise ValueError(f"archive has too many members: {len(members)}")
            regular_total = sum(member.size for member in members if member.isfile())
            if regular_total > MAX_ARCHIVE_BYTES:
                raise ValueError(f"archive expands beyond byte limit: {regular_total}")
            for member in members:
                relative = validate_member_name(member.name)
                destination = source_dir.joinpath(*relative.parts)
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"links/devices are not allowed in source archive: {member.name!r}")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported archive member type: {member.name!r}")
                if member.size > MAX_MEMBER_BYTES:
                    raise ValueError(f"archive member exceeds limit: {member.name!r} ({member.size})")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read archive member: {member.name!r}")
                with source:
                    extracted_bytes += copy_member(source, destination)
                destination.chmod(0o644)
                extracted_files += 1
        return {"format": "tar", "files": extracted_files, "bytes": extracted_bytes}
    except tarfile.ReadError:
        pass

    raw = archive.read_bytes()
    if raw.startswith(b"%PDF"):
        raise ValueError("arXiv e-print endpoint returned PDF only; no TeX source is available")
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    if len(raw) > MAX_MEMBER_BYTES:
        raise ValueError("single-file source exceeds size limit")
    text = raw.decode("utf-8", errors="replace")
    if "\\documentclass" not in text:
        raise ValueError("source payload is neither a safe tar archive nor a standalone TeX document")
    destination = source_dir / "main.tex"
    destination.write_text(text, encoding="utf-8")
    destination.chmod(0o644)
    return {"format": "single_tex", "files": 1, "bytes": destination.stat().st_size}


def scan_source(source_dir: Path) -> dict[str, Any]:
    scanned: list[str] = []
    ignored_build_files: list[str] = []
    findings: list[dict[str, Any]] = []
    all_files = [path for path in source_dir.rglob("*") if path.is_file()]
    for path in all_files:
        rel = path.relative_to(source_dir).as_posix()
        lower_name = path.name.lower()
        if lower_name in {"latexmkrc", ".latexmkrc", "makefile"} or path.suffix.lower() in {
            ".sh",
            ".py",
            ".pl",
            ".rb",
        }:
            ignored_build_files.append(rel)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned.append(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        for rule, pattern in DANGEROUS_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                snippet = text[max(0, match.start() - 40) : match.end() + 80].replace("\n", " ")
                findings.append({"rule": rule, "file": rel, "line": line, "snippet": snippet[:180]})
    return {
        "status": "rejected" if findings else "passed",
        "scanned_files": scanned,
        "ignored_never_executed_build_files": sorted(ignored_build_files),
        "findings": findings,
        "rules": [name for name, _ in DANGEROUS_PATTERNS],
    }


def quarantine_legacy_relative_outdir(source_dir: Path, paper_dir: Path, stem: str) -> dict[str, Any] | None:
    """Move only the known first-run relative-outdir artifact tree aside.

    An early builder version passed ``outputs/...`` relative to the TeX source
    cwd.  The exact signature below distinguishes those generated aux files
    from submitted source content.  Moving, rather than deleting, preserves a
    complete diagnostic trail.
    """
    legacy_root = source_dir / "outputs" / "arxiv_latex_recompile_pilot_5"
    signature = legacy_root / "papers" / stem / "build"
    if not signature.is_dir():
        return None
    source_resolved = source_dir.resolve()
    legacy_resolved = legacy_root.resolve()
    if not legacy_resolved.is_relative_to(source_resolved):
        raise ValueError(f"refusing to quarantine path outside source tree: {legacy_resolved}")
    files = [path for path in legacy_root.rglob("*") if path.is_file()]
    byte_count = sum(path.stat().st_size for path in files)
    destination = paper_dir / "diagnostics" / "initial_relative_outdir_artifacts"
    if destination.exists():
        destination = destination.with_name(destination.name + "_additional")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_root), str(destination))
    return {
        "source": str(legacy_root),
        "destination": str(destination),
        "files": len(files),
        "bytes": byte_count,
        "reason": "quarantined aux state created by the initial relative outdir bug",
    }


def find_main_tex(source_dir: Path) -> tuple[Path, list[str]]:
    candidates: list[tuple[int, Path]] = []
    for path in source_dir.rglob("*.tex"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\\documentclass" not in text or "\\begin{document}" not in text:
            continue
        name = path.name.lower()
        score = 0
        if name in {"main.tex", "paper.tex", "manuscript.tex", "article.tex", "arxiv.tex"}:
            score += 100
        if any(word in name for word in ("supp", "appendix", "response", "cover")):
            score -= 80
        score -= len(path.relative_to(source_dir).parts) * 3
        score += min(path.stat().st_size // 10_000, 30)
        candidates.append((score, path))
    if not candidates:
        raise ValueError("no TeX file contains both documentclass and begin{document}")
    candidates.sort(key=lambda pair: (-pair[0], pair[1].as_posix()))
    selected = candidates[0][1]
    return selected, [path.relative_to(source_dir).as_posix() for _, path in candidates]


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_logged(
    command: list[str],
    cwd: Path,
    log_path: Path,
    timeout: int,
    progress: Progress,
    current: str,
    stage: str,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = os.environ.copy()
    # latexmk launches its selected TeX engine by basename.  Prefix the exact
    # TeX Live bin directory that supplied latexmk so child engines resolve in
    # non-login shells as well.
    command_path = Path(command[0])
    tex_bins = [
        str(Path("/Library/TeX/texbin")),
        str(command_path.parent),
        str(command_path.resolve().parent),
    ]
    env.update(
        {
            "PATH": os.pathsep.join(dict.fromkeys(tex_bins)) + os.pathsep + env.get("PATH", ""),
            "openout_any": "p",
            "openin_any": "p",
            "shell_escape": "f",
        }
    )
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        last_log = started
        timed_out = False
        while process.poll() is None:
            time.sleep(1)
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                timed_out = True
                terminate_process_group(process)
                break
            if time.monotonic() - last_log >= HEARTBEAT_SECONDS:
                progress.emit(current, stage, f"pid={process.pid} runtime={elapsed:.0f}s log_bytes={log.tell()}")
                last_log = time.monotonic()
        return_code = process.poll()
    duration = time.monotonic() - started
    return {
        "command": command,
        "cwd": str(cwd),
        "log": str(log_path),
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_seconds": round(duration, 3),
        "log_bytes": log_path.stat().st_size,
    }


def compile_source(
    source_dir: Path,
    main_tex: Path,
    paper_dir: Path,
    latexmk: Path,
    timeout: int,
    progress: Progress,
    current: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    main_name = main_tex.name
    cwd = main_tex.parent
    for engine, flag in (
        ("pdflatex", "-pdf"),
        ("xelatex", "-xelatex"),
        ("latex_dvips_ps2pdf", "-pdfps"),
    ):
        # latexmk resolves outdir relative to the source cwd, so always pass an
        # absolute path.  This also makes resume deterministic after invocation
        # from any project working directory.
        build_dir = (paper_dir / "build" / engine).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        log_path = paper_dir / "logs" / f"compile_{engine}.log"
        command = [
            str(latexmk),
            "-norc",
            # Resume may encounter a latexmk database produced by an earlier
            # interrupted/failed attempt.  Force a real rebuild instead of
            # treating that stale state as an up-to-date failed target.
            "-g",
            flag,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-no-shell-escape",
            f"-outdir={build_dir}",
            main_name,
        ]
        progress.emit(current, f"compile_{engine}_start", main_tex.relative_to(source_dir).as_posix())
        attempt = run_logged(command, cwd, log_path, timeout, progress, current, f"compile_{engine}")
        produced = build_dir / f"{main_tex.stem}.pdf"
        attempt["produced_pdf"] = str(produced)
        attempt["produced_pdf_bytes"] = produced.stat().st_size if produced.is_file() else 0
        attempts.append(attempt)
        if attempt["return_code"] == 0 and produced.is_file() and produced.stat().st_size > 0:
            final_pdf = paper_dir / "paper.pdf"
            shutil.copy2(produced, final_pdf)
            return {"status": "success", "engine": engine, "attempts": attempts, "pdf": str(final_pdf)}
        progress.emit(
            current,
            f"compile_{engine}_failed",
            f"return_code={attempt['return_code']} timeout={attempt['timed_out']}",
        )
    return {"status": "failed", "engine": None, "attempts": attempts, "pdf": None}


def inspect_pdf(
    pdf: Path,
    paper_dir: Path,
    pdfinfo: Path,
    pdftoppm: Path,
) -> dict[str, Any]:
    info_path = paper_dir / "pdfinfo.txt"
    info_result = subprocess.run([str(pdfinfo), str(pdf)], capture_output=True, text=True, timeout=30)
    info_path.write_text(info_result.stdout + info_result.stderr, encoding="utf-8")
    if info_result.returncode != 0 or not info_result.stdout.strip():
        raise ValueError(f"pdfinfo failed for {pdf}: return code {info_result.returncode}")
    pages_match = re.search(r"^Pages:\s+(\d+)", info_result.stdout, flags=re.M)
    pages = int(pages_match.group(1)) if pages_match else None
    preview_dir = paper_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_base = preview_dir / "first_page"
    preview_result = subprocess.run(
        [str(pdftoppm), "-f", "1", "-l", "1", "-singlefile", "-png", "-r", "120", str(pdf), str(preview_base)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    preview = preview_base.with_suffix(".png")
    if preview_result.returncode != 0 or not preview.is_file() or preview.stat().st_size == 0:
        raise ValueError(f"pdftoppm failed for {pdf}: {preview_result.stderr.strip()}")
    return {
        "pdf": str(pdf),
        "pdf_bytes": pdf.stat().st_size,
        "pdf_sha256": sha256_file(pdf),
        "pdfinfo": str(info_path),
        "pages": pages,
        "first_page_png": str(preview),
        "first_page_png_bytes": preview.stat().st_size,
    }


def result_paths_nonempty(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("metadata_path", "archive_path"):
        value = result.get(key)
        if value and (not Path(value).is_file() or Path(value).stat().st_size == 0):
            errors.append(f"{key} missing_or_empty: {value}")
    safety_path = result.get("safety_scan_path")
    if safety_path and (not Path(safety_path).is_file() or Path(safety_path).stat().st_size == 0):
        errors.append(f"safety_scan_path missing_or_empty: {safety_path}")
    if result.get("status") == "success":
        for key in ("pdf", "pdfinfo", "first_page_png"):
            value = result.get("pdf_inspection", {}).get(key)
            if not value or not Path(value).is_file() or Path(value).stat().st_size == 0:
                errors.append(f"successful result missing_or_empty {key}: {value}")
    return errors


def write_rollups(output_root: Path, selection: list[dict[str, Any]], results: list[dict[str, Any]], started: float) -> dict[str, Any]:
    results_path = output_root / "results.jsonl"
    tmp = results_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    os.replace(tmp, results_path)

    counts = {name: sum(result["status"] == name for result in results) for name in ("success", "failed", "rejected")}
    validation_errors = [error for result in results for error in result_paths_nonempty(result)]
    summary = {
        "schema_version": 1,
        "created_at": now_iso(),
        "status": "passed" if len(results) == len(selection) and not validation_errors else "failed",
        "selected": len(selection),
        "completed": len(results),
        **counts,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "results_jsonl": str(results_path),
        "selection_json": str(output_root / "selection.json"),
        "validation_errors": validation_errors,
        "compile_policy": {
            "engines": ["pdflatex", "xelatex", "latex_dvips_ps2pdf"],
            "lualatex_prohibited": True,
            "latexmkrc_disabled": True,
            "shell_escape_disabled": True,
            "source_scripts_executed": False,
        },
    }
    atomic_json(output_root / "summary.json", summary)
    table_rows = []
    for result in results:
        engine = result.get("compile", {}).get("engine") or "—"
        pages = result.get("pdf_inspection", {}).get("pages") or "—"
        table_rows.append(
            f"| `{result['stem']}` | {result['domain']} | {result['license_name']} | {result['status']} | {engine} | {pages} |"
        )
    readme = f"""# arXiv LaTeX recompilation pilot (5 papers)

This isolated pilot selected five papers from the local OAI-derived license manifest in
`outputs/arxiv_chaos_2000`. Every selected record has an explicit CC-BY, CC-BY-SA, or
CC0 license. Official e-print source archives were downloaded from `export.arxiv.org`.

## Results

| arXiv id/version | domain | license | result | engine | pages |
|---|---|---|---|---|---:|
{chr(10).join(table_rows)}

Machine-readable details are in `selection.json`, `results.jsonl`, and `summary.json`.
Each `papers/<id><version>/` directory retains the source archive, safely unpacked
source, static safety scan, compile logs, any successful PDF, `pdfinfo` output, first-page
PNG, and per-paper metadata.

## Safety policy

- Archives are extracted without links, devices, absolute paths, or `..` traversal and
  with file/count/expanded-size limits.
- Sources are rejected before compilation when static scanning finds `write18`, pipe
  input, absolute input paths, shell-executing packages/environments, or direct Lua.
- Compilation uses `latexmk -norc`, `-no-shell-escape`, nonstop mode, halt-on-error,
  file-line errors, a process-group timeout, and restricted TeX input/output settings.
- pdfLaTeX is attempted first, then XeLaTeX, then a LaTeX→DVI→dvips→ps2pdf
  route for EPS-era sources. Every attempt is recorded. LuaLaTeX is never used.
- Source-provided scripts, Makefiles, and latexmk configuration are never executed.

Structural validation status: **{summary['status']}**. Compilation failures are retained
as pilot outcomes rather than hidden; consult each log and metadata record.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    return summary


def process_paper(
    paper: dict[str, Any],
    output_root: Path,
    args: argparse.Namespace,
    progress: Progress,
) -> dict[str, Any]:
    stem = paper["stem"]
    paper_dir = output_root / "papers" / stem
    paper_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = paper_dir / "metadata.json"
    archive_path = paper_dir / "source_archive.bin"
    source_dir = paper_dir / "source"
    safety_path = paper_dir / "safety_scan.json"
    result: dict[str, Any] = {
        **paper,
        "paper_dir": str(paper_dir),
        "metadata_path": str(metadata_path),
        "archive_path": str(archive_path),
        "source_dir": str(source_dir),
        "safety_scan_path": str(safety_path),
        "started_at": now_iso(),
        "status": "failed",
        "stage": "initializing",
    }
    atomic_json(metadata_path, result)
    try:
        result["download"] = download_source(
            paper["source_url"],
            archive_path,
            args.download_timeout,
            progress,
            stem,
            args.force_download,
        )
        result["stage"] = "downloaded"
        atomic_json(metadata_path, result)

        progress.emit(stem, "extract_start")
        result["extraction"] = extract_source(archive_path, source_dir)
        quarantined = quarantine_legacy_relative_outdir(source_dir, paper_dir, stem)
        if quarantined:
            result["quarantined_builder_artifacts"] = quarantined
            source_files = [path for path in source_dir.rglob("*") if path.is_file()]
            result["extraction"]["files"] = len(source_files)
            result["extraction"]["bytes"] = sum(path.stat().st_size for path in source_files)
        result["stage"] = "extracted"
        atomic_json(metadata_path, result)

        progress.emit(stem, "safety_scan_start")
        safety = scan_source(source_dir)
        atomic_json(safety_path, safety)
        result["safety_scan"] = safety
        result["stage"] = "scanned"
        atomic_json(metadata_path, result)
        if safety["status"] != "passed":
            result["status"] = "rejected"
            result["failure_reason"] = "dangerous source construct detected before compilation"
            result["completed_at"] = now_iso()
            atomic_json(metadata_path, result)
            return result

        main_tex, candidates = find_main_tex(source_dir)
        result["main_tex"] = str(main_tex)
        result["main_tex_candidates"] = candidates
        result["stage"] = "compiling"
        atomic_json(metadata_path, result)
        result["compile"] = compile_source(
            source_dir,
            main_tex,
            paper_dir,
            args.latexmk,
            args.compile_timeout,
            progress,
            stem,
        )
        if result["compile"]["status"] != "success":
            result["status"] = "failed"
            result["failure_reason"] = "both pdfLaTeX and XeLaTeX compilation attempts failed"
            result["stage"] = "compile_failed"
        else:
            result["pdf_inspection"] = inspect_pdf(
                Path(result["compile"]["pdf"]), paper_dir, args.pdfinfo, args.pdftoppm
            )
            result["status"] = "success"
            result["stage"] = "complete"
    except Exception as exc:  # every failure is persisted as a pilot result
        result["status"] = "failed"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result["stage"] = result.get("stage", "unknown")
    result["completed_at"] = now_iso()
    atomic_json(metadata_path, result)
    return result


def check_tools(args: argparse.Namespace) -> None:
    for name in ("latexmk", "pdfinfo", "pdftoppm"):
        path = getattr(args, name)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"required executable unavailable: {name}={path}")


def main() -> int:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    # Keep the /Library/TeX/texbin symlink spelling: latexmk itself is a Perl
    # script elsewhere in texmf-dist, whereas its child engines live beside
    # the symlink in texbin.
    args.latexmk = args.latexmk.absolute()
    args.pdfinfo = args.pdfinfo.resolve()
    args.pdftoppm = args.pdftoppm.resolve()
    check_tools(args)
    selection = load_selection(args.dataset_root)
    selected_stems = {paper["stem"] for paper in selection}
    unknown = sorted(set(args.only) - selected_stems)
    if unknown:
        raise ValueError(f"--only contains stems outside the fixed selection: {unknown}")
    work_selection = [paper for paper in selection if not args.only or paper["stem"] in set(args.only)]
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "selection.json", selection)
    started = time.monotonic()
    existing_results: dict[str, dict[str, Any]] = {}
    results_path = output_root / "results.jsonl"
    if args.only and results_path.is_file():
        with results_path.open(encoding="utf-8") as handle:
            existing_results = {row["stem"]: row for row in map(json.loads, handle) if row.get("stem")}
    progress = Progress(total=len(work_selection), started=started)
    print(
        f"[start] selected={len(work_selection)} fixed_total={len(selection)} output={output_root} "
        f"compile_timeout={args.compile_timeout}s "
        f"engines=pdflatex,xelatex,latex_dvips_ps2pdf shell_escape=disabled",
        flush=True,
    )
    for index, paper in enumerate(work_selection, start=1):
        stem = paper["stem"]
        progress.emit(stem, "paper_start", f"unit={index}/{len(work_selection)}")
        result = process_paper(paper, output_root, args, progress)
        existing_results[stem] = result
        progress.completed += 1
        if result["status"] == "success":
            progress.success += 1
        elif result["status"] == "rejected":
            progress.rejected += 1
        else:
            progress.failed += 1
        ordered_results = [existing_results[paper["stem"]] for paper in selection if paper["stem"] in existing_results]
        write_rollups(output_root, selection, ordered_results, started)
        progress.emit(stem, "paper_complete", f"status={result['status']}")
    ordered_results = [existing_results[paper["stem"]] for paper in selection if paper["stem"] in existing_results]
    summary = write_rollups(output_root, selection, ordered_results, started)
    print(
        f"[finish] completed={summary['completed']}/{summary['selected']} "
        f"success={summary['success']} fail={summary['failed']} reject={summary['rejected']} "
        f"validation={summary['status']} elapsed={summary['elapsed_seconds']}s output={output_root}",
        flush=True,
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
