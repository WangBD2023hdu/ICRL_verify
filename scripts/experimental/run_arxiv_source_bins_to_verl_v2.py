#!/usr/bin/env python3
"""Run the isolated source-first v2 page builder over downloaded arXiv sources.

The input may be a crawler root (``results.jsonl`` plus ``papers/``), its
``papers/`` directory, a directory containing ``*.bin`` archives, or one/many
already-unpacked source directories.  The runner never writes below the input
tree.  All state and artifacts live below a v2-marked experimental output.

This entry point stops after source-first page generation.  It intentionally
does not invoke the frozen v10 mutation/VERL runner.  Its aggregate artifacts
are ``page_ledger_v2.jsonl``, ``paper_results_v2.jsonl``, and
``validation_report_v2.json``.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
from datetime import datetime, timezone
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
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
STABLE_SCRIPTS = REPO_ROOT / "scripts"
for value in (str(SOURCE_ROOT), str(STABLE_SCRIPTS)):
    if value not in sys.path:
        sys.path.insert(0, value)

from arxiv_source_first_v2.contracts import (  # noqa: E402
    COMPLEX_LAYOUT_BUCKETS,
    EXPERIMENTAL_CONTRACT,
    EXPERIMENTAL_SCHEMA_VERSION,
    PIPELINE_VERSION,
    STABLE_FILE_SHA256,
    STABLE_V10_PIPELINE_VERSION,
    ContractError,
    assert_stable_files,
    normalize_layout_bucket,
    validate_experimental_directory,
    validate_page_ledger_file,
)
from build_arxiv_latex_recompile_pilot import (  # noqa: E402
    ALLOWED_LICENSES,
    extract_source,
    find_main_tex,
    scan_source,
)


HEARTBEAT_SECONDS = 30.0
SAFE_STEM_RE = re.compile(r"^[A-Za-z0-9._-]+$")
TERMINAL_STATUSES = frozenset({"success", "rejected", "failed"})
REPORT_FILENAME = "validation_report_v2.json"
LEDGER_FILENAME = "page_ledger_v2.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_text(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    return f"{value // 3600}h{value % 3600 // 60:02d}m{value % 60:02d}s"


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    values = [dict(row) for row in rows]
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in values
        ),
    )
    return len(values)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL object required at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(value: Any) -> str:
    stem = str(value or "").strip()
    if not stem or not SAFE_STEM_RE.fullmatch(stem):
        raise ValueError(f"unsafe or missing paper stem: {stem!r}")
    return stem


def filesystem_identities(stem: str) -> set[str]:
    """Return identities available without reading a filesystem paper."""

    identities = {stem}
    match = re.fullmatch(r"(.+?)v[1-9][0-9]*", stem)
    if match:
        identities.add(match.group(1))
    return identities


def path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def json_contains_stable_pipeline(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("pipeline_version") == STABLE_V10_PIPELINE_VERSION:
            return True
        return any(json_contains_stable_pipeline(item) for item in value.values())
    if isinstance(value, list):
        return any(json_contains_stable_pipeline(item) for item in value)
    return False


def stable_marker_in_ancestor(output_dir: Path) -> Path | None:
    """Find a direct JSON marker in an ancestor without recursively scanning data."""

    for ancestor in output_dir.parents:
        try:
            candidates = ancestor.glob("*.json")
        except OSError:
            continue
        for path in candidates:
            try:
                if path.stat().st_size > 8 * 1024 * 1024:
                    continue
                value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                continue
            if json_contains_stable_pipeline(value):
                return path
    return None


def validate_output_isolation(
    output_dir: Path,
    input_root: Path,
    stable_output_roots: Sequence[Path],
    *,
    create: bool,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    source = input_root.expanduser().resolve()
    if path_contains(source, output) or path_contains(output, source):
        raise ContractError(
            f"experimental output and input trees must not overlap: input={source} output={output}"
        )
    for raw in stable_output_roots:
        stable_root = raw.expanduser().resolve()
        if path_contains(stable_root, output) or path_contains(output, stable_root):
            raise ContractError(
                "experimental output overlaps declared stable output: "
                f"stable={stable_root} output={output}"
            )
    ancestor_marker = stable_marker_in_ancestor(output)
    if ancestor_marker is not None:
        raise ContractError(
            f"experimental output is nested below a stable v10 output marker: {ancestor_marker}"
        )
    if not create and not output.exists():
        return {
            "status": "dry_run_uncreated",
            "ok": True,
            "root": str(output),
            "marker": None,
            "stable_markers": [],
        }
    return validate_experimental_directory(
        output,
        create=create,
        purpose="source-first-v2-batch-experiment",
        metadata={"input_root": str(source)},
    )


def validate_experimental_builder_path(builder_script: Path) -> Path:
    """Reject attempts to execute one of the frozen production entry points.

    The v2 runner may accept a fixture or a separately versioned experimental
    builder, but it must never be used as a dispatch mechanism for a frozen
    v10 script.  ``resolve`` also closes the symlink spelling of this escape
    hatch.  The check is intentionally exact: helper modules outside the
    frozen execution chain remain usable when they are read-only.
    """

    resolved = builder_script.expanduser().resolve()
    frozen = {
        (REPO_ROOT / relative).resolve(): relative
        for relative in STABLE_FILE_SHA256
    }
    if resolved in frozen:
        raise ContractError(
            "experimental v2 runner cannot execute frozen stable script: "
            f"{frozen[resolved]}"
        )
    return resolved


def directory_inventory(path: Path, *, label: str = "unknown") -> dict[str, Any]:
    total = 0
    files = 0
    digest = hashlib.sha256()
    started = time.monotonic()
    last_progress = started
    for candidate in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                relative = candidate.relative_to(path).as_posix()
                size = candidate.stat().st_size
                total += size
                files += 1
                digest.update(relative.encode("utf-8", errors="surrogateescape"))
                digest.update(b"\0")
                with candidate.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
        except OSError:
            continue
        now = time.monotonic()
        if now - last_progress >= HEARTBEAT_SECONDS:
            print(
                f"[progress] phase=discovery_size current={label} files={files} "
                f"bytes={total} elapsed={elapsed_text(now-started)} "
                "accepted=0 rejected=0 errors=0",
                flush=True,
            )
            last_progress = now
    return {"bytes": total, "files": files, "sha256": digest.hexdigest()}


def input_signature(descriptor: Mapping[str, Any]) -> str:
    payload = {
        "kind": descriptor["kind"],
        "path": descriptor["path"],
        "bytes": descriptor["input_bytes"],
        "mtime_ns": descriptor.get("mtime_ns"),
        "expected_sha256": descriptor.get("expected_sha256"),
        "source_tree_sha256": descriptor.get("source_tree_sha256"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def archive_candidates(input_root: Path, stem: str, row: Mapping[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for key in ("archive", "archive_path"):
        if not row.get(key):
            continue
        path = Path(str(row[key])).expanduser()
        candidates.append(path if path.is_absolute() else input_root / path)
    candidates.extend(
        [
            input_root / "papers" / stem / "source_archive.bin",
            input_root / stem / "source_archive.bin",
        ]
    )
    return sorted({path.resolve() for path in candidates if path.is_file()})


def expected_sha256(row: Mapping[str, Any]) -> str | None:
    if row.get("sha256"):
        return str(row["sha256"])
    download = row.get("download")
    if isinstance(download, Mapping) and download.get("sha256"):
        return str(download["sha256"])
    return None


def make_descriptor(
    stem: str,
    kind: str,
    path: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    expected: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if kind == "archive":
        stat = resolved.stat()
        input_bytes = stat.st_size
        mtime_ns = stat.st_mtime_ns
        tree_sha256 = None
    else:
        inventory = directory_inventory(resolved, label=stem)
        input_bytes = int(inventory["bytes"])
        tree_sha256 = str(inventory["sha256"])
        try:
            mtime_ns = resolved.stat().st_mtime_ns
        except OSError:
            mtime_ns = None
    descriptor = {
        "stem": safe_stem(stem),
        "kind": kind,
        "path": str(resolved),
        "input_bytes": input_bytes,
        "mtime_ns": mtime_ns,
        "expected_sha256": expected,
        "source_tree_sha256": tree_sha256,
        "metadata": dict(metadata or {}),
    }
    descriptor["input_signature"] = input_signature(descriptor)
    return descriptor


def has_tex_source(path: Path) -> bool:
    try:
        return any(candidate.is_file() for candidate in path.rglob("*.tex")) or any(
            candidate.is_file() for candidate in path.rglob("*.ltx")
        )
    except OSError:
        return False


def file_is_main_tex(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "\\documentclass" in text and "\\begin{document}" in text


def has_main_tex_candidate(path: Path) -> bool:
    for suffix in ("*.tex", "*.ltx"):
        for candidate in path.rglob(suffix):
            if not candidate.is_file():
                continue
            if file_is_main_tex(candidate):
                return True
    return False


def discover_from_results(
    input_root: Path,
    *,
    allow_crawler_unfiltered_license: bool = False,
    paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    results_path = input_root / "results.jsonl"
    if not results_path.is_file():
        return []
    descriptors: list[dict[str, Any]] = []
    for row in read_jsonl(results_path):
        if row.get("status") not in {"passed", "success"}:
            continue
        license_name = row.get("license_name")
        crawler_unfiltered = (
            allow_crawler_unfiltered_license
            and license_name == "UNFILTERED"
            and row.get("license_policy") == "accept_all_atom_v1"
        )
        if license_name not in ALLOWED_LICENSES and not crawler_unfiltered:
            continue
        stem = safe_stem(
            row.get("stem") or f"{row.get('arxiv_id', '')}{row.get('version', '')}"
        )
        identities = {stem, str(row.get("arxiv_id") or "")}
        if paper_ids and not paper_ids.intersection(identities):
            # Filter before resolving/stat-ing an archive or constructing its
            # descriptor.  Fixed cohorts must not touch unrelated paper data.
            continue
        archives = archive_candidates(input_root, stem, row)
        if len(archives) != 1:
            # An incomplete crawler directory is valid input.  Missing rows are
            # simply not selected; duplicate archive locations remain unsafe.
            if not archives:
                continue
            raise ValueError(f"multiple archives found for {stem}: {archives}")
        descriptors.append(
            make_descriptor(
                stem,
                "archive",
                archives[0],
                metadata=row,
                expected=expected_sha256(row),
            )
        )
    return descriptors


def discover_from_filesystem(
    input_root: Path,
    *,
    paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    root = input_root.resolve()
    if root.is_file():
        if root.suffix.casefold() != ".bin":
            raise ValueError(f"input file must be a .bin source archive: {root}")
        stem = root.parent.name if root.name == "source_archive.bin" else root.stem
        if paper_ids and not paper_ids.intersection(filesystem_identities(stem)):
            return []
        return [make_descriptor(stem, "archive", root)]

    base = root / "papers" if (root / "papers").is_dir() else root
    all_children = sorted(
        (path for path in base.iterdir() if path.is_dir()), key=lambda p: p.name
    )
    selected_children = (
        [
            child
            for child in all_children
            if paper_ids.intersection(filesystem_identities(child.name))
        ]
        if paper_ids
        else all_children
    )
    base_selected = bool(
        paper_ids and paper_ids.intersection(filesystem_identities(base.name))
    )
    if paper_ids and not selected_children and not base_selected:
        return []
    children = selected_children
    top_level_main = any(
        file_is_main_tex(path)
        for pattern in ("*.tex", "*.ltx")
        for path in base.glob(pattern)
        if path.is_file()
    )
    structured_children = any(
        (child / "source_archive.bin").is_file()
        or ((child / "source").is_dir() and has_tex_source(child / "source"))
        for child in children
    )
    nested_single_paper = (
        not top_level_main
        and base.name != "papers"
        and not structured_children
        and sum(has_main_tex_candidate(child) for child in children) < 2
        and has_main_tex_candidate(base)
    )
    if top_level_main or nested_single_paper:
        if paper_ids and not base_selected:
            return []
        return [make_descriptor(base.name, "source_dir", base)]

    descriptors: list[dict[str, Any]] = []
    for child in children:
        archive = child / "source_archive.bin"
        if archive.is_file() and archive.stat().st_size > 0:
            descriptors.append(make_descriptor(child.name, "archive", archive))
            continue
        source = child / "source"
        if source.is_dir() and has_tex_source(source):
            descriptors.append(make_descriptor(child.name, "source_dir", source))
            continue
        if has_tex_source(child):
            descriptors.append(make_descriptor(child.name, "source_dir", child))
    for archive in sorted(base.glob("*.bin")):
        stem = archive.parent.name if archive.name == "source_archive.bin" else archive.stem
        if paper_ids and not paper_ids.intersection(filesystem_identities(stem)):
            continue
        if archive.stat().st_size <= 0:
            continue
        if not any(row["stem"] == stem for row in descriptors):
            descriptors.append(make_descriptor(stem, "archive", archive))
    return descriptors


def discover_inputs(
    input_root: Path,
    *,
    paper_ids: set[str],
    max_papers: int,
    allow_crawler_unfiltered_license: bool = False,
) -> list[dict[str, Any]]:
    root = input_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_dir() and (root / "results.jsonl").is_file():
        # A crawler manifest is authoritative for status/license filtering.
        # Do not silently bypass it by falling back to a raw filesystem scan.
        descriptors = discover_from_results(
            root,
            allow_crawler_unfiltered_license=allow_crawler_unfiltered_license,
            paper_ids=paper_ids,
        )
    else:
        descriptors = discover_from_filesystem(root, paper_ids=paper_ids)
    by_stem: dict[str, dict[str, Any]] = {}
    for descriptor in descriptors:
        stem = descriptor["stem"]
        if stem in by_stem:
            raise ValueError(f"duplicate discovered paper stem: {stem}")
        metadata = descriptor.get("metadata") or {}
        identities = filesystem_identities(stem) | {
            str(metadata.get("arxiv_id") or "")
        }
        if paper_ids and not paper_ids.intersection(identities):
            continue
        by_stem[stem] = descriptor
    selected = [by_stem[key] for key in sorted(by_stem)]
    if max_papers:
        selected = selected[:max_papers]
    if paper_ids:
        found: set[str] = set()
        for descriptor in selected:
            found.update(filesystem_identities(str(descriptor["stem"])))
            metadata = descriptor.get("metadata") or {}
            if metadata.get("arxiv_id"):
                found.add(str(metadata["arxiv_id"]))
        missing = sorted(paper_ids - found)
        if missing:
            raise ValueError(f"requested paper IDs were not discovered: {missing}")
    if not selected:
        raise ValueError("no downloaded or unpacked TeX source papers were discovered")
    return selected


def move_aside(path: Path, diagnostics_root: Path, label: str) -> Path:
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    destination = diagnostics_root / (
        f"{label}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}."
        f"{os.getpid()}.{time.time_ns()}"
    )
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


def run_logged(command: list[str], log_path: Path, timeout_seconds: int) -> dict[str, Any]:
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
            time.sleep(0.2)
    return {
        "command": command,
        "return_code": process.poll(),
        "timed_out": timed_out,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path),
        "log_bytes": log_path.stat().st_size,
    }


def copy_unpacked_source(source: Path, destination: Path) -> dict[str, Any]:
    symlinks = [path for path in source.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"unpacked source contains symlinks: {symlinks[:3]}")
    shutil.copytree(source, destination)
    files = [path for path in destination.rglob("*") if path.is_file()]
    return {
        "format": "copied_unpacked",
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def load_valid_builder_artifacts(paper_root: Path) -> dict[str, Any] | None:
    report_path = paper_root / REPORT_FILENAME
    ledger_path = paper_root / LEDGER_FILENAME
    if not report_path.is_file() or not ledger_path.is_file():
        return None
    try:
        report = read_json(report_path)
        raw_rows = read_jsonl(ledger_path)
        validate_page_ledger_file(ledger_path, require_explicit_outcomes=True)
    except (OSError, ValueError, ContractError, json.JSONDecodeError):
        return None
    if (
        report.get("schema_version") != EXPERIMENTAL_SCHEMA_VERSION
        or report.get("contract") != EXPERIMENTAL_CONTRACT
        or int(report.get("pages_total", -1)) != len(raw_rows)
        or int(report.get("pages_passed", -1))
        != sum(row.get("source_first_passed") is True for row in raw_rows)
    ):
        return None
    passed = int(report.get("pages_passed", 0))
    if passed and any(
        row.get("source_first_passed") is True
        and row.get("source_first_verifier_exact") is not True
        for row in raw_rows
    ):
        return None
    return {"report": report, "ledger": raw_rows}


def summarize_artifacts(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    report = artifacts["report"]
    ledger = artifacts.get("ledger") or []
    passed_rows = [row for row in ledger if row.get("source_first_passed") is True]
    accepted_two_column = sum(
        normalize_layout_bucket(row.get("layout", row.get("layout_bucket")))
        == "two_column"
        for row in passed_rows
    )
    return {
        "source_first_pages_total": int(report.get("pages_total", 0)),
        "eligible_clean_text_pages": int(report.get("eligible_clean_text_pages", 0)),
        "source_first_pages_passed": int(report.get("pages_passed", 0)),
        "source_first_pages_rejected": int(report.get("pages_rejected", 0)),
        "accepted_complex_layout_pages": int(
            report.get("accepted_complex_layout_pages", 0)
        ),
        "accepted_two_column_pages": accepted_two_column,
        "accepted_two_column_layout_pages": accepted_two_column,
        "accepted_exact_verifier_rate": float(
            report.get("accepted_exact_verifier_rate", 0.0)
        ),
    }


def completed_state_is_reusable(
    state: Mapping[str, Any],
    task: Mapping[str, Any],
) -> bool:
    if (
        state.get("pipeline_version") != PIPELINE_VERSION
        or state.get("contract") != EXPERIMENTAL_CONTRACT
        or state.get("input_signature") != task["descriptor"]["input_signature"]
        or state.get("status") not in TERMINAL_STATUSES
    ):
        return False
    descriptor = task["descriptor"]
    if descriptor["kind"] == "archive" and state.get("archive_sha256") != task.get(
        "observed_archive_sha256"
    ):
        return False
    if state.get("status") in {"success", "rejected"}:
        return load_valid_builder_artifacts(Path(str(task["paper_root"]))) is not None
    return True


def prepare_source(task: Mapping[str, Any], state: dict[str, Any]) -> tuple[Path, Path]:
    descriptor = task["descriptor"]
    stem = descriptor["stem"]
    source_dir = Path(str(task["source_root"])) / stem / "source"
    checkpoint = source_dir.parent / "extraction_v2.json"
    diagnostics = Path(str(task["diagnostics_root"])) / stem
    signature = descriptor["input_signature"]
    if descriptor["kind"] == "archive":
        observed_sha = str(task.get("observed_archive_sha256") or "")
        state["archive_sha256"] = observed_sha
        expected = descriptor.get("expected_sha256")
        if expected and observed_sha != expected:
            raise ValueError(
                f"source archive SHA-256 mismatch expected={expected} observed={observed_sha}"
            )
    valid_checkpoint = False
    if bool(task["resume"]) and source_dir.is_dir() and checkpoint.is_file():
        try:
            stored = read_json(checkpoint)
            valid_checkpoint = (
                stored.get("status") == "passed"
                and stored.get("input_signature") == signature
                and stored.get("pipeline_version") == PIPELINE_VERSION
                and (
                    descriptor["kind"] != "archive"
                    or stored.get("archive_sha256") == state.get("archive_sha256")
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            valid_checkpoint = False
    if not valid_checkpoint:
        if source_dir.parent.exists():
            state.setdefault("moved_aside", []).append(
                str(move_aside(source_dir.parent, diagnostics, "source_incomplete"))
            )
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        input_path = Path(str(descriptor["path"]))
        if descriptor["kind"] == "archive":
            extraction = extract_source(input_path, source_dir)
        else:
            extraction = copy_unpacked_source(input_path, source_dir)
        atomic_write_json(
            checkpoint,
            {
                "status": "passed",
                "pipeline_version": PIPELINE_VERSION,
                "input_signature": signature,
                "archive_sha256": state.get("archive_sha256"),
                "source": str(input_path),
                **extraction,
            },
        )
    safety = scan_source(source_dir)
    atomic_write_json(source_dir.parent / "safety_scan_v2.json", safety)
    if safety.get("status") != "passed":
        raise PermissionError("dangerous source construct detected before compilation")
    main_tex, candidates = find_main_tex(source_dir)
    state["main_tex_candidates"] = candidates
    return source_dir, main_tex.resolve().relative_to(source_dir.resolve())


def figure_attempts(policy: str) -> list[bool]:
    if policy == "drop_then_keep":
        return [True, False]
    return [policy == "drop"]


def build_command(
    task: Mapping[str, Any],
    source_dir: Path,
    main_tex: Path,
    engine: str,
    drop_figures: bool,
) -> list[str]:
    command = [
        str(task["python"]),
        str(task["builder_script"]),
        "--source-dir",
        str(source_dir),
        "--main-tex",
        main_tex.as_posix(),
        "--output-dir",
        str(task["paper_root"]),
        "--paper-id",
        str(task["descriptor"]["stem"]),
        "--max-pages",
        str(task["max_pages_per_paper"]),
        "--dpi",
        str(task["dpi"]),
        "--compile-timeout",
        str(task["compile_timeout"]),
        "--engine",
        engine,
        "--latexmk",
        str(task["latexmk"]),
        "--pdftoppm",
        str(task["pdftoppm"]),
        "--min-eligible-visible-characters",
        str(task["min_eligible_visible_characters"]),
    ]
    if task["drop_references"]:
        command.append("--drop-references")
    if drop_figures:
        command.append("--drop-figures")
    return command


def process_paper(task: dict[str, Any]) -> dict[str, Any]:
    descriptor = task["descriptor"]
    stem = descriptor["stem"]
    if descriptor["kind"] == "archive":
        task["observed_archive_sha256"] = sha256_file(Path(str(descriptor["path"])))
    state_path = Path(str(task["state_root"])) / f"{stem}.json"
    paper_root = Path(str(task["paper_root"]))
    diagnostics = Path(str(task["diagnostics_root"])) / stem
    if bool(task["resume"]) and state_path.is_file():
        try:
            stored = read_json(state_path)
        except (OSError, ValueError, json.JSONDecodeError):
            stored = {}
        if completed_state_is_reusable(stored, task) and not (
            stored.get("status") == "failed" and bool(task["retry_failed"])
        ):
            reused = {**stored, "resume_state": f"reused_{stored['status']}"}
            artifacts = load_valid_builder_artifacts(paper_root)
            if artifacts is not None:
                # Refresh derived layout counters when resuming a state file
                # written by an earlier v2 revision.
                reused.update(summarize_artifacts(artifacts))
            return reused

    state: dict[str, Any] = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "contract": EXPERIMENTAL_CONTRACT,
        "pipeline_version": PIPELINE_VERSION,
        "paper_id": stem,
        "stem": stem,
        "status": "running",
        "stage": "initializing",
        "input_kind": descriptor["kind"],
        "input_path": descriptor["path"],
        "input_signature": descriptor["input_signature"],
        "input_bytes": descriptor["input_bytes"],
        "paper_root": str(paper_root),
        "started_at_utc": utc_now(),
        "started_at_epoch": time.time(),
    }
    if state_path.exists():
        state.setdefault("moved_aside", []).append(
            str(move_aside(state_path, diagnostics, "paper_state_stale"))
        )
    if paper_root.exists():
        state.setdefault("moved_aside", []).append(
            str(move_aside(paper_root, diagnostics, "paper_output_stale"))
        )
    atomic_write_json(state_path, state)
    try:
        state["stage"] = "source_preparation"
        atomic_write_json(state_path, state)
        source_dir, main_tex = prepare_source(task, state)
        state["source_dir"] = str(source_dir)
        state["main_tex"] = main_tex.as_posix()
        state["stage"] = "source_first_gt"
        atomic_write_json(state_path, state)
        attempts: list[dict[str, Any]] = []
        successful: dict[str, Any] | None = None
        best_failed: tuple[int, Path, dict[str, Any]] | None = None
        attempt_number = 0
        for engine in task["latex_engines"]:
            for drop_figures in figure_attempts(str(task["figure_policy"])):
                attempt_number += 1
                if paper_root.exists():
                    moved = move_aside(
                        paper_root,
                        diagnostics,
                        f"attempt_{attempt_number - 1}_not_selected",
                    )
                    if best_failed and best_failed[1] == paper_root:
                        best_failed = (best_failed[0], moved, best_failed[2])
                command = build_command(
                    task, source_dir, main_tex, engine, drop_figures
                )
                run = run_logged(
                    command,
                    Path(str(task["log_root"]))
                    / f"{stem}.{attempt_number:02d}.{engine}."
                    f"{'drop_figures' if drop_figures else 'keep_figures'}.log",
                    int(task["paper_timeout"]),
                )
                run.update(
                    {
                        "attempt": attempt_number,
                        "engine": engine,
                        "drop_figures": drop_figures,
                    }
                )
                artifacts = load_valid_builder_artifacts(paper_root)
                if artifacts is not None:
                    run.update(summarize_artifacts(artifacts))
                    score = int(artifacts["report"].get("pages_passed", 0))
                    if best_failed is None or score > best_failed[0]:
                        best_failed = (score, paper_root, artifacts)
                attempts.append(run)
                if run["return_code"] == 0 and artifacts is not None:
                    successful = artifacts
                    break
            if successful is not None:
                break
        state["builder_attempts"] = attempts
        artifacts = successful
        if artifacts is None and best_failed is not None:
            _, candidate_path, _ = best_failed
            if paper_root.exists() and candidate_path != paper_root:
                move_aside(paper_root, diagnostics, "last_unselected_attempt")
            if candidate_path != paper_root and candidate_path.exists():
                shutil.move(str(candidate_path), str(paper_root))
            artifacts = load_valid_builder_artifacts(paper_root)
        if artifacts is None:
            tail = ""
            if attempts:
                try:
                    tail = Path(str(attempts[-1]["log"])).read_text(
                        encoding="utf-8", errors="replace"
                    )[-4000:]
                except OSError:
                    pass
            raise RuntimeError(
                "all source-first v2 builder variants failed; "
                f"attempts={len(attempts)} last_log_tail={tail}"
            )
        state.update(summarize_artifacts(artifacts))
        state.update(
            status="success" if state["source_first_pages_passed"] > 0 else "rejected",
            stage="complete",
            source_first_report=str((paper_root / REPORT_FILENAME).resolve()),
            source_first_ledger=str((paper_root / LEDGER_FILENAME).resolve()),
        )
    except PermissionError as error:
        state.update(
            status="rejected",
            stage="safety_scan",
            failure_reason=f"{type(error).__name__}: {error}",
            source_first_pages_total=0,
            eligible_clean_text_pages=0,
            source_first_pages_passed=0,
            source_first_pages_rejected=0,
            accepted_complex_layout_pages=0,
            accepted_two_column_pages=0,
            accepted_two_column_layout_pages=0,
        )
    except Exception as error:  # noqa: BLE001 - worker returns auditable failure state
        state.update(
            status="failed",
            failure_reason=f"{type(error).__name__}: {error}",
            source_first_pages_total=int(state.get("source_first_pages_total", 0)),
            eligible_clean_text_pages=int(state.get("eligible_clean_text_pages", 0)),
            source_first_pages_passed=int(state.get("source_first_pages_passed", 0)),
            source_first_pages_rejected=int(state.get("source_first_pages_rejected", 0)),
            accepted_complex_layout_pages=int(
                state.get("accepted_complex_layout_pages", 0)
            ),
            accepted_two_column_pages=int(state.get("accepted_two_column_pages", 0)),
            accepted_two_column_layout_pages=int(
                state.get("accepted_two_column_layout_pages", 0)
            ),
        )
    state["completed_at_utc"] = utc_now()
    state["completed_at_epoch"] = time.time()
    state["elapsed_seconds"] = round(
        state["completed_at_epoch"] - state["started_at_epoch"], 3
    )
    atomic_write_json(state_path, state)
    return state


def result_counters(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counters = collections.Counter()
    for result in results:
        counters["papers_completed"] += 1
        status = str(result.get("status") or "failed")
        counters[f"papers_{status}"] += 1
        counters["bytes_completed"] += int(result.get("input_bytes", 0))
        counters["pages_total"] += int(result.get("source_first_pages_total", 0))
        counters["eligible_pages"] += int(result.get("eligible_clean_text_pages", 0))
        counters["pages_passed"] += int(result.get("source_first_pages_passed", 0))
        counters["pages_rejected"] += int(result.get("source_first_pages_rejected", 0))
        counters["complex_pages_passed"] += int(
            result.get("accepted_complex_layout_pages", 0)
        )
        counters["two_column_pages_passed"] += int(
            result.get(
                "accepted_two_column_pages",
                result.get("accepted_two_column_layout_pages", 0),
            )
        )
    return dict(counters)


def progress_payload(
    results: Iterable[Mapping[str, Any]],
    *,
    total_papers: int,
    total_bytes: int,
    started: float,
) -> dict[str, Any]:
    counters = result_counters(results)
    completed = counters.get("papers_completed", 0)
    elapsed = max(time.monotonic() - started, 1e-9)
    throughput = completed / elapsed
    page_throughput = counters.get("pages_total", 0) / elapsed
    eta = (total_papers - completed) / throughput if throughput else math.inf
    return {
        **counters,
        "papers_total": total_papers,
        "bytes_total": total_bytes,
        "pct": 100.0 * completed / total_papers if total_papers else 100.0,
        "throughput": throughput,
        "page_throughput": page_throughput,
        "elapsed": elapsed,
        "eta": eta,
    }


def print_progress(prefix: str, payload: Mapping[str, Any], current: str) -> None:
    eta = payload["eta"]
    print(
        f"[{prefix}] papers={payload.get('papers_completed', 0)}/{payload['papers_total']} "
        f"pct={payload['pct']:.1f}% current={current} "
        f"pages={payload.get('pages_total', 0)} "
        f"passed={payload.get('pages_passed', 0)} "
        f"rejected={payload.get('pages_rejected', 0)} "
        f"eligible={payload.get('eligible_pages', 0)} "
        f"complex_passed={payload.get('complex_pages_passed', 0)} "
        f"two_column_passed={payload.get('two_column_pages_passed', 0)} "
        f"paper_success={payload.get('papers_success', 0)} "
        f"paper_rejected={payload.get('papers_rejected', 0)} "
        f"errors={payload.get('papers_failed', 0)} "
        f"bytes={payload.get('bytes_completed', 0)}/{payload['bytes_total']} "
        f"throughput={payload['throughput']:.3f}_papers/s "
        f"page_throughput={payload['page_throughput']:.3f}_pages/s "
        f"elapsed={elapsed_text(payload['elapsed'])} "
        f"eta={'unknown' if not math.isfinite(eta) else elapsed_text(eta)}",
        flush=True,
    )


def write_incremental_state(
    output_dir: Path,
    ordered_results: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> None:
    atomic_write_jsonl(output_dir / "paper_results_v2.jsonl", ordered_results)
    atomic_write_json(
        output_dir / "batch_state_v2.json",
        {
            "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
            "contract": EXPERIMENTAL_CONTRACT,
            "pipeline_version": PIPELINE_VERSION,
            "status": "running",
            "updated_at_utc": utc_now(),
            **{
                key: value
                for key, value in payload.items()
                if key not in {"throughput", "page_throughput", "elapsed", "eta"}
            },
        },
    )


def aggregate_ledgers(
    descriptors: Sequence[Mapping[str, Any]],
    results_by_stem: Mapping[str, Mapping[str, Any]],
    papers_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for descriptor in descriptors:
        stem = str(descriptor["stem"])
        result = results_by_stem.get(stem) or {}
        if result.get("status") not in {"success", "rejected"}:
            continue
        path = papers_root / stem / LEDGER_FILENAME
        if not path.is_file():
            continue
        validate_page_ledger_file(path, require_explicit_outcomes=True)
        for row in read_jsonl(path):
            page_id = str(row.get("page_id") or row.get("data_id") or "")
            if not page_id or page_id in seen:
                raise ValueError(f"missing or duplicate aggregate page id: {page_id!r}")
            seen.add(page_id)
            rows.append(row)
    return rows


def aggregate_report(
    descriptors: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    *,
    input_root: Path,
    output_dir: Path,
    workers: int,
    allow_crawler_unfiltered_license: bool,
    stable_guard: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    counters = result_counters(results)
    eligible = sum(bool(row.get("eligible_text_page")) for row in ledger)
    passed_rows = [row for row in ledger if row.get("source_first_passed") is True]
    exact = sum(row.get("source_first_verifier_exact") is True for row in passed_rows)
    complex_passed = sum(
        normalize_layout_bucket(row.get("layout", row.get("layout_bucket")))
        in COMPLEX_LAYOUT_BUCKETS
        for row in passed_rows
    )
    two_column_passed = sum(
        normalize_layout_bucket(row.get("layout", row.get("layout_bucket")))
        == "two_column"
        for row in passed_rows
    )
    source_yield = len(passed_rows) / eligible if eligible else 0.0
    exact_rate: float | None = exact / len(passed_rows) if passed_rows else None
    rejections = collections.Counter(
        str(reason)
        for row in ledger
        for reason in (row.get("rejection_reasons") or [])
    )
    failures = collections.Counter(
        str(row.get("stage") or "unknown")
        for row in results
        if row.get("status") == "failed"
    )
    target = {
        "source_first_yield_gt_0_30": source_yield > 0.30,
        "accepted_complex_layout_pages_gt_0": complex_passed > 0,
        "accepted_two_column_pages_gt_0": two_column_passed > 0,
        "accepted_exact_verifier_rate_1_0": bool(passed_rows) and exact_rate == 1.0,
    }
    target["passed"] = all(target.values())
    return {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "contract": EXPERIMENTAL_CONTRACT,
        "pipeline_version": PIPELINE_VERSION,
        "status": "passed" if passed_rows else "failed",
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "workers": workers,
        "allow_crawler_unfiltered_license": allow_crawler_unfiltered_license,
        "papers_selected": len(descriptors),
        "papers_completed": counters.get("papers_completed", 0),
        "papers_success": counters.get("papers_success", 0),
        "papers_rejected": counters.get("papers_rejected", 0),
        "papers_failed": counters.get("papers_failed", 0),
        "input_bytes": sum(int(row["input_bytes"]) for row in descriptors),
        "pages_total": len(ledger),
        "eligible_clean_text_pages": eligible,
        "pages_passed": len(passed_rows),
        "pages_rejected": len(ledger) - len(passed_rows),
        "accepted_complex_layout_pages": complex_passed,
        "accepted_two_column_pages": two_column_passed,
        "accepted_two_column_layout_pages": two_column_passed,
        "source_first_yield": round(source_yield, 8),
        "accepted_exact_verifier_rate": (
            round(exact_rate, 8) if exact_rate is not None else None
        ),
        "target": target,
        "page_rejection_reasons": dict(rejections.most_common()),
        "paper_failure_stages": dict(failures.most_common()),
        "page_ledger": str((output_dir / LEDGER_FILENAME).resolve()),
        "paper_results": str((output_dir / "paper_results_v2.jsonl").resolve()),
        "stable_guard": dict(stable_guard),
        "pdf_used_for_generation": False,
        "pdf_used_for_verification": True,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "completed_at_utc": utc_now(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="crawler root, papers directory, source .bin, or unpacked TeX source root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new or already v2-marked experimental root; never a stable v10 output",
    )
    parser.add_argument(
        "--stable-output-root",
        action="append",
        type=Path,
        default=[],
        help="stable output tree that must not overlap --output-dir; repeatable",
    )
    parser.add_argument(
        "--allow-crawler-unfiltered-license",
        action="store_true",
        help=(
            "accept crawler rows explicitly marked UNFILTERED/accept_all_atom_v1; "
            "otherwise only CC-BY-4.0, CC-BY-SA-4.0, and CC0-1.0 rows are selected"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) // 2)),
        help="independent paper worker processes",
    )
    parser.add_argument("--max-papers", type=int, default=0, help="0 selects all discovered papers")
    parser.add_argument("--paper-ids", nargs="*", default=[], help="optional stems or arXiv IDs")
    parser.add_argument("--max-pages-per-paper", type=int, default=10000)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--min-eligible-visible-characters", type=int, default=80)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--paper-timeout", type=int, default=2400)
    parser.add_argument(
        "--latex-engines",
        default="pdflatex,xelatex,latex_dvips_ps2pdf",
        help="comma-separated fallback order used independently for each paper",
    )
    parser.add_argument(
        "--figure-policy",
        choices=("drop_then_keep", "drop", "keep"),
        default="drop_then_keep",
        help="try figure-free source first, intact source as fallback, or only one mode",
    )
    parser.add_argument(
        "--drop-references",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove citations/bibliography before page generation",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=HEARTBEAT_SECONDS,
        help="parent-process aggregate progress interval; maximum 30 seconds",
    )
    parser.add_argument(
        "--builder-script",
        type=Path,
        default=SCRIPT_DIR / "build_source_first_span_graph_v2.py",
        help="single-paper source-first v2 builder (overridable for fixture testing)",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
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
        "--dry-run",
        action="store_true",
        help="discover and validate inputs/output isolation without compiling or creating output",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> list[str]:
    if not 1 <= args.workers <= 256:
        raise ValueError("--workers must be between 1 and 256")
    if args.max_papers < 0:
        raise ValueError("--max-papers must be non-negative")
    for name in (
        "max_pages_per_paper",
        "dpi",
        "min_eligible_visible_characters",
        "compile_timeout",
        "paper_timeout",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.heartbeat_seconds <= HEARTBEAT_SECONDS:
        raise ValueError("--heartbeat-seconds must be >0 and <=30")
    engines = [value.strip() for value in args.latex_engines.split(",") if value.strip()]
    supported = {"pdflatex", "xelatex", "latex_dvips_ps2pdf"}
    if not engines or len(engines) != len(set(engines)):
        raise ValueError("--latex-engines must be a non-empty unique list")
    if not set(engines) <= supported:
        raise ValueError(f"unsupported LaTeX engines: {sorted(set(engines) - supported)}")
    return engines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    engines = validate_args(args)
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    print(
        f"[start] phase=discovery input={input_root} output={output_dir} "
        f"max_papers={args.max_papers} dry_run={args.dry_run}",
        flush=True,
    )
    stable_guard = assert_stable_files(REPO_ROOT)
    descriptors = discover_inputs(
        input_root,
        paper_ids=set(args.paper_ids),
        max_papers=args.max_papers,
        allow_crawler_unfiltered_license=args.allow_crawler_unfiltered_license,
    )
    total_bytes = sum(int(row["input_bytes"]) for row in descriptors)
    isolation = validate_output_isolation(
        output_dir,
        input_root,
        args.stable_output_root,
        create=not args.dry_run,
    )
    print(
        f"[start] pipeline={PIPELINE_VERSION} papers={len(descriptors)} "
        f"bytes={total_bytes} workers={args.workers} resume={args.resume} "
        f"retry_failed={args.retry_failed} input={input_root} output={output_dir} "
        f"marker={isolation.get('marker')} dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        for index, descriptor in enumerate(descriptors, 1):
            print(
                f"[unit-done] phase=dry_run papers={index}/{len(descriptors)} "
                f"pct={100*index/len(descriptors):.1f}% current={descriptor['stem']} "
                f"kind={descriptor['kind']} bytes={descriptor['input_bytes']} "
                "accepted=0 rejected=0 errors=0",
                flush=True,
            )
        print(
            f"[finish] status=dry_run papers={len(descriptors)} bytes={total_bytes} output_created=false",
            flush=True,
        )
        return 0

    python = args.python.expanduser().resolve()
    latexmk = args.latexmk.expanduser().resolve()
    pdftoppm = args.pdftoppm.expanduser().resolve()
    builder_script = args.builder_script.expanduser().resolve()
    builder_script = validate_experimental_builder_path(builder_script)
    for name, path, executable in (
        ("python", python, True),
        ("latexmk", latexmk, True),
        ("pdftoppm", pdftoppm, True),
        ("builder_script", builder_script, False),
    ):
        if not path.is_file() or (executable and not os.access(path, os.X_OK)):
            raise FileNotFoundError(f"required tool unavailable: {name}={path}")

    papers_root = output_dir / "papers"
    source_root = output_dir / "extracted_sources"
    state_root = output_dir / "paper_states"
    diagnostics_root = output_dir / "diagnostics"
    log_root = output_dir / "logs"
    for path in (papers_root, source_root, state_root, diagnostics_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for descriptor in descriptors:
        tasks.append(
            {
                "descriptor": descriptor,
                "paper_root": str(papers_root / descriptor["stem"]),
                "source_root": str(source_root),
                "state_root": str(state_root),
                "diagnostics_root": str(diagnostics_root),
                "log_root": str(log_root),
                "builder_script": str(builder_script),
                "python": str(python),
                "latexmk": str(latexmk),
                "pdftoppm": str(pdftoppm),
                "max_pages_per_paper": args.max_pages_per_paper,
                "dpi": args.dpi,
                "min_eligible_visible_characters": args.min_eligible_visible_characters,
                "compile_timeout": args.compile_timeout,
                "paper_timeout": args.paper_timeout,
                "latex_engines": engines,
                "figure_policy": args.figure_policy,
                "drop_references": args.drop_references,
                "resume": args.resume,
                "retry_failed": args.retry_failed,
            }
        )

    started = time.monotonic()
    results_by_stem: dict[str, dict[str, Any]] = {}
    try:
        executor: concurrent.futures.Executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        )
        executor_mode = "process"
    except (PermissionError, NotImplementedError, OSError) as error:
        print(
            f"[warning] process_pool_unavailable={type(error).__name__}:{error} "
            f"fallback=thread workers={args.workers}",
            flush=True,
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        executor_mode = "thread_fallback"
    print(
        f"[parallel_start] executor={executor_mode} workers={args.workers} papers={len(tasks)}",
        flush=True,
    )
    interrupted = False
    try:
        with executor:
            future_to_task = {
                executor.submit(process_paper, task): task for task in tasks
            }
            pending = set(future_to_task)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=args.heartbeat_seconds,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    payload = progress_payload(
                        results_by_stem.values(),
                        total_papers=len(tasks),
                        total_bytes=total_bytes,
                        started=started,
                    )
                    running = ",".join(
                        sorted(
                            str(future_to_task[item]["descriptor"]["stem"])
                            for item in pending
                        )[:4]
                    )
                    print_progress("progress", payload, running or "none")
                    write_incremental_state(
                        output_dir,
                        [
                            results_by_stem[row["stem"]]
                            for row in descriptors
                            if row["stem"] in results_by_stem
                        ],
                        payload,
                    )
                    continue
                for future in done:
                    task = future_to_task[future]
                    stem = str(task["descriptor"]["stem"])
                    try:
                        result = future.result()
                    except Exception as error:  # noqa: BLE001
                        result = {
                            "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
                            "contract": EXPERIMENTAL_CONTRACT,
                            "pipeline_version": PIPELINE_VERSION,
                            "paper_id": stem,
                            "stem": stem,
                            "status": "failed",
                            "stage": "worker_exception",
                            "failure_reason": f"{type(error).__name__}: {error}",
                            "input_bytes": task["descriptor"]["input_bytes"],
                            "source_first_pages_total": 0,
                            "eligible_clean_text_pages": 0,
                            "source_first_pages_passed": 0,
                            "source_first_pages_rejected": 0,
                            "accepted_complex_layout_pages": 0,
                            "accepted_two_column_pages": 0,
                            "accepted_two_column_layout_pages": 0,
                        }
                    results_by_stem[stem] = result
                    ordered = [
                        results_by_stem[row["stem"]]
                        for row in descriptors
                        if row["stem"] in results_by_stem
                    ]
                    payload = progress_payload(
                        ordered,
                        total_papers=len(tasks),
                        total_bytes=total_bytes,
                        started=started,
                    )
                    write_incremental_state(output_dir, ordered, payload)
                    print_progress("unit-done", payload, stem)
    except KeyboardInterrupt:
        interrupted = True
        executor.shutdown(wait=False, cancel_futures=True)
        payload = progress_payload(
            results_by_stem.values(),
            total_papers=len(tasks),
            total_bytes=total_bytes,
            started=started,
        )
        write_incremental_state(
            output_dir,
            [
                results_by_stem[row["stem"]]
                for row in descriptors
                if row["stem"] in results_by_stem
            ],
            payload,
        )
        raise
    finally:
        if interrupted:
            print("[finish] status=interrupted resume=true", flush=True)

    ordered_results = [results_by_stem[row["stem"]] for row in descriptors]
    # Re-check the frozen chain after all worker processes have finished.  The
    # initial guard protects dispatch; the final guard makes an external or
    # accidental mutation during a long batch visible in the result instead
    # of silently claiming an isolated run.
    stable_guard_final = assert_stable_files(REPO_ROOT)
    if not stable_guard_final["ok"]:
        raise ContractError(
            "stable execution chain changed during experimental run: "
            + json.dumps(stable_guard_final["mismatches"], ensure_ascii=False)
        )
    stable_guard = {**stable_guard, "final": stable_guard_final}
    atomic_write_jsonl(output_dir / "paper_results_v2.jsonl", ordered_results)
    ledger = aggregate_ledgers(descriptors, results_by_stem, papers_root)
    atomic_write_jsonl(output_dir / LEDGER_FILENAME, ledger)
    report = aggregate_report(
        descriptors,
        ordered_results,
        ledger,
        input_root=input_root,
        output_dir=output_dir,
        workers=args.workers,
        allow_crawler_unfiltered_license=args.allow_crawler_unfiltered_license,
        stable_guard=stable_guard,
        elapsed_seconds=time.monotonic() - started,
    )
    atomic_write_json(output_dir / REPORT_FILENAME, report)
    final_counters = result_counters(ordered_results)
    atomic_write_json(
        output_dir / "batch_state_v2.json",
        {
            "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
            "contract": EXPERIMENTAL_CONTRACT,
            "pipeline_version": PIPELINE_VERSION,
            "status": "complete",
            "papers_completed": len(ordered_results),
            "papers_total": len(descriptors),
            "papers_success": final_counters.get("papers_success", 0),
            "papers_rejected": final_counters.get("papers_rejected", 0),
            "papers_failed": final_counters.get("papers_failed", 0),
            "bytes_completed": final_counters.get("bytes_completed", 0),
            "bytes_total": sum(int(row["input_bytes"]) for row in descriptors),
            "pages_total": report["pages_total"],
            "eligible_clean_text_pages": report["eligible_clean_text_pages"],
            "pages_passed": report["pages_passed"],
            "pages_rejected": report["pages_rejected"],
            "accepted_complex_layout_pages": report[
                "accepted_complex_layout_pages"
            ],
            "accepted_two_column_pages": report["accepted_two_column_pages"],
            "accepted_two_column_layout_pages": report[
                "accepted_two_column_layout_pages"
            ],
            "source_first_yield": report["source_first_yield"],
            "accepted_exact_verifier_rate": report[
                "accepted_exact_verifier_rate"
            ],
            "target": report["target"],
            "page_ledger": report["page_ledger"],
            "paper_results": report["paper_results"],
            "validation_report": str((output_dir / REPORT_FILENAME).resolve()),
            "updated_at_utc": utc_now(),
        },
    )
    print(
        f"[finish] status={report['status']} papers={report['papers_success']}/"
        f"{report['papers_selected']} pages={report['pages_passed']}/"
        f"{report['eligible_clean_text_pages']} "
        f"yield={100*report['source_first_yield']:.2f}% "
        f"complex={report['accepted_complex_layout_pages']} "
        f"errors={report['papers_failed']} target_passed={report['target']['passed']} "
        f"elapsed={elapsed_text(report['elapsed_seconds'])} output={output_dir}",
        flush=True,
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
