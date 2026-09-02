#!/usr/bin/env python3
"""Build high-yield source-derived canonical arXiv page/Markdown pairs.

This experimental V4 is intentionally separate from the stable original-page
V1/V2/V3 pipelines.  It treats downloaded arXiv projects as a source corpus,
normalizes independently safe AST blocks into a controlled page template, and
compiles one standalone PDF per candidate page.  PDF text is reject-only and
can never create or repair Markdown GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_arxiv_latex_recompile_pilot import (
    extract_source as safely_extract_source_archive,
)
from build_arxiv_latex_recompile_pilot import (
    find_main_tex as find_archive_main_tex,
)
from build_arxiv_latex_recompile_pilot import (
    scan_source as scan_archive_source,
)

from arxiv_canonical_reflow_v4.core import (
    PIPELINE_VERSION,
    CanonicalBlock,
    CanonicalPage,
    blocks_from_document,
    build_page_tex,
    bundle_blocks,
    pack_blocks,
    verify_rendered_text,
)
from arxiv_canonical_reflow_v4.mutation import (
    MUTATION_POLICY_VERSION,
    PageMutation,
    RenderedWord,
    apply_page_mutations,
    choose_page_mutations,
    choose_source_page_mutations,
    markdown_diff_count,
    validate_mutated_word_geometry,
)
from arxiv_source_first_v3.document_ast import parse_document_ast
from arxiv_source_first_v3.semantic_declarations import (
    extract_semantic_environment_definitions,
)
from arxiv_source_first_v3.source_environment_definitions import (
    collect_source_list_environment_definitions,
)
from arxiv_source_first_v3.source_macro_definitions import (
    collect_source_macro_definitions,
)
from arxiv_source_first_v3.source_project import flatten_source_project
from arxiv_source_first_v3.source_sanitizer import sanitize_latex_source

_SFT_PROMPT = """<image>
Please convert the image document into Markdown format, strictly adhering to the following requirements:

1. Accurately transcribe all visible text without guessing or correcting typos.
2. Preserve headings, paragraphs, lists, inline emphasis, and reading order.
3. Convert formulas to LaTeX.
4. Convert tables to clean HTML without adding metadata attributes.
5. Ignore graphical elements, headers, footers, and page numbers.
6. Return only the Markdown transcription."""

_PAGE_MARGIN_PT = 0.72 * 72.0
_SAFE_PAPER_STEM_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_CRAWLER_CACHE_SCHEMA_VERSION = 1
_VERBOSE_OUTPUT = False
_STATUS_LINE_WIDTH = 0


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    output_dir: str
    latexmk: str
    pdftoppm: str
    pdftotext: str
    pdfinfo: str
    compile_timeout: float
    render_timeout: float
    dpi: int
    max_pack_attempts: int
    min_page_chars: int
    target_weight: int
    two_column_rate: float
    target_fill_ratio: float
    min_fill_ratio: float
    work_dir: str | None = None
    minimal_output: bool = True


@dataclass(frozen=True, slots=True)
class MutationConfig:
    seed: int
    minimum_per_page: int
    maximum_per_page: int
    maximum_probability: float
    max_vertical_shift_points: float


@dataclass(frozen=True, slots=True)
class WorkerResult:
    page_id: str
    paper_id: str
    status: str
    reason: str | None
    layout: str
    has_table: bool
    markdown: str
    verifier_recall: float | None
    verifier_precision: float | None
    pdf: str | None
    image: str | None
    block_ids: tuple[str, ...]
    source_node_ids: tuple[str, ...]
    content_fill_ratio: float | None
    column_fill_ratios: tuple[float, ...]
    page_signature: str | None
    elapsed_seconds: float
    rescued: bool = False
    pack_attempts: int = 1
    mutation_count: int = 0
    changes: tuple[dict[str, Any], ...] = ()
    clean_page_id: str | None = None
    max_mutation_vertical_shift_points: float | None = None
    variant: str = "clean"


@dataclass(frozen=True, slots=True)
class BBoxObservation:
    text: str
    words: tuple[RenderedWord, ...]
    content_fill_ratio: float
    column_fill_ratios: tuple[float, ...]
    page_width: float
    page_height: float


@dataclass(frozen=True, slots=True)
class CrawlerArchive:
    paper_id: str
    archive: str
    row: dict[str, Any]
    expected_sha256: str | None
    input_bytes: int
    input_mtime_ns: int


def _status_field(message: str, name: str, default: str = "0") -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", message)
    return match.group(1) if match else default


def _status_line(message: str, *, finish: bool = False) -> None:
    global _STATUS_LINE_WIDTH
    padding = " " * max(0, _STATUS_LINE_WIDTH - len(message))
    print(f"\r{message}{padding}", end="\n" if finish else "", flush=True)
    _STATUS_LINE_WIDTH = 0 if finish else len(message)


def _emit(stage: str, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if _VERBOSE_OUTPUT:
        print(f"[{stamp}] [{stage}] {message}", flush=True)
        return
    if stage == "start":
        _status_line(
            "Start: "
            f"sources={_status_field(message, 'papers')} "
            f"target={_status_field(message, 'target_count', 'all')} "
            f"workers={_status_field(message, 'workers')} "
            f"output={_status_field(message, 'output')}",
            finish=True,
        )
    elif stage in {
        "crawler-stream-unit",
        "crawler-stream-progress",
        "extract-unit",
        "extract-progress",
    }:
        _status_line(
            "Source preparation "
            f"{_status_field(message, 'completed')}, "
            f"candidates={_status_field(message, 'cumulative_candidates', _status_field(message, 'candidates'))}, "
            f"errors={_status_field(message, 'errors')}"
        )
    elif stage == "compile-progress":
        _status_line(
            "Edited-page compilation "
            f"jobs={_status_field(message, 'completed_jobs')}, "
            f"accepted={_status_field(message, 'accepted')}, "
            f"rejected={_status_field(message, 'rejected')}"
        )
    elif stage == "finish":
        _status_line(f"Completed: {message}", finish=True)


class _TargetProgress:
    def __init__(self, target: int, *, initial: int = 0) -> None:
        self.target = target
        self.initial = initial
        self.accepted = initial
        self.rejected = 0
        self.started = time.monotonic()
        self.last_refresh = 0.0

    def update(self, *, accepted: int, rejected: int, force: bool = False) -> None:
        changed = accepted != self.accepted
        self.accepted = accepted
        self.rejected = rejected
        now = time.monotonic()
        if not (force or changed or now - self.last_refresh >= 30):
            return
        elapsed = now - self.started
        rate = max(0, self.accepted - self.initial) / max(elapsed, 1e-9)
        if self.target > 0:
            shown = min(self.accepted, self.target)
            fraction = shown / self.target
            width = 30
            filled = min(width, int(width * fraction))
            bar = "#" * filled + "-" * (width - filled)
            eta = f"{(self.target - shown) / rate:.0f}s" if rate > 0 else "--"
            message = (
                f"[{bar}] {shown}/{self.target} ({fraction:.1%}) "
                f"rejected={self.rejected} rate={rate:.2f}/s eta={eta}"
            )
        else:
            message = (
                f"Accepted={self.accepted} rejected={self.rejected} "
                f"rate={rate:.2f}/s elapsed={elapsed:.0f}s"
            )
        _status_line(message)
        self.last_refresh = now

    def finish(self, *, accepted: int, rejected: int) -> None:
        self.update(
            accepted=accepted,
            rejected=rejected,
            force=True,
        )
        _status_line("", finish=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(
                    f"invalid_jsonl:{path}:{line_number}:{exc}"
                ) from exc
            if not isinstance(value, dict):
                raise PipelineError(f"jsonl_row_is_not_object:{path}:{line_number}")
            rows.append(value)
    return rows


def _crawler_paper_id(row: dict[str, Any]) -> str:
    value = row.get("stem")
    if not value:
        value = f"{row.get('arxiv_id', '')}{row.get('version', '')}"
    paper_id = str(value)
    if not paper_id or not _SAFE_PAPER_STEM_RE.fullmatch(paper_id):
        raise PipelineError(f"unsafe_or_missing_paper_stem:{paper_id!r}")
    return paper_id


def _crawler_expected_sha256(row: dict[str, Any]) -> str | None:
    value = row.get("sha256")
    if value:
        return str(value)
    download = row.get("download")
    if isinstance(download, dict) and download.get("sha256"):
        return str(download["sha256"])
    return None


def _crawler_archive_path(
    crawler_root: Path,
    papers_root: Path,
    row: dict[str, Any],
) -> Path:
    paper_id = _crawler_paper_id(row)
    default = papers_root / paper_id / "source_archive.bin"
    if default.is_file():
        return default.resolve()
    candidates: list[Path] = []
    for key in ("archive", "archive_path"):
        value = row.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = crawler_root / candidate
        if candidate.is_file():
            candidates.append(candidate.resolve())
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise PipelineError(f"source_archive_not_unique:{paper_id}:found={len(unique)}")
    return unique[0]


def _discover_crawler_archives(
    root: Path,
    *,
    limit: int,
    seed: int,
) -> list[CrawlerArchive]:
    """Discover completed crawler archives from a crawler root or its papers/ dir."""

    if (root / "papers").is_dir():
        crawler_root = root
        papers_root = root / "papers"
    else:
        crawler_root = (
            root.parent if (root.parent / "results.jsonl").is_file() else root
        )
        papers_root = root
    if not papers_root.is_dir():
        raise PipelineError(f"crawler_papers_root_missing:{papers_root}")
    results_path = crawler_root / "results.jsonl"
    rows_by_stem: dict[str, dict[str, Any]] = {}
    if results_path.is_file():
        for row in _read_jsonl(results_path):
            paper_id = _crawler_paper_id(row)
            if paper_id in rows_by_stem:
                raise PipelineError(f"duplicate_crawler_paper_stem:{paper_id}")
            rows_by_stem[paper_id] = row

    discovered: list[CrawlerArchive] = []
    seen: set[str] = set()
    for paper_dir in sorted(path for path in papers_root.iterdir() if path.is_dir()):
        archive = paper_dir / "source_archive.bin"
        if not archive.is_file() or archive.stat().st_size <= 0:
            continue
        paper_id = paper_dir.name
        if not _SAFE_PAPER_STEM_RE.fullmatch(paper_id):
            continue
        known_row = rows_by_stem.get(paper_id)
        row = dict(known_row or {"stem": paper_id})
        if known_row is not None and row.get("status") not in {"passed", "success"}:
            continue
        resolved = _crawler_archive_path(crawler_root, papers_root, row)
        stat = resolved.stat()
        discovered.append(
            CrawlerArchive(
                paper_id=paper_id,
                archive=str(resolved),
                row=row,
                expected_sha256=_crawler_expected_sha256(row),
                input_bytes=stat.st_size,
                input_mtime_ns=stat.st_mtime_ns,
            )
        )
        seen.add(paper_id)

    # A result row may point outside the conventional papers/<stem>/ location.
    for paper_id, row in sorted(rows_by_stem.items()):
        if paper_id in seen or row.get("status") not in {"passed", "success"}:
            continue
        try:
            resolved = _crawler_archive_path(crawler_root, papers_root, row)
        except PipelineError:
            continue
        if resolved.name.endswith(".partial") or resolved.stat().st_size <= 0:
            continue
        stat = resolved.stat()
        discovered.append(
            CrawlerArchive(
                paper_id=paper_id,
                archive=str(resolved),
                row=dict(row),
                expected_sha256=_crawler_expected_sha256(row),
                input_bytes=stat.st_size,
                input_mtime_ns=stat.st_mtime_ns,
            )
        )

    discovered.sort(key=lambda item: item.paper_id)
    random.Random(seed).shuffle(discovered)
    if limit > 0:
        discovered = discovered[:limit]
    if not discovered:
        raise PipelineError(f"no_completed_source_archives:{root}")
    return discovered


def _cached_crawler_paper_valid(
    paper_dir: Path,
    archive: CrawlerArchive,
) -> bool:
    metadata_path = paper_dir / "metadata.json"
    source_dir = paper_dir / "source"
    if not metadata_path.is_file() or not source_dir.is_dir():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    main_tex = metadata.get("main_tex")
    return bool(
        metadata.get("crawler_cache_schema_version") == _CRAWLER_CACHE_SCHEMA_VERSION
        and metadata.get("status") == "prepared"
        and metadata.get("archive_bytes") == archive.input_bytes
        and metadata.get("archive_mtime_ns") == archive.input_mtime_ns
        and metadata.get("expected_sha256") == archive.expected_sha256
        and isinstance(main_tex, str)
        and main_tex
        and (source_dir / main_tex).is_file()
    )


def _materialize_crawler_archive(
    archive: CrawlerArchive,
    cache_root: Path,
) -> dict[str, Any]:
    """Safely and atomically normalize one source archive for V4."""

    started = time.monotonic()
    papers_root = cache_root / "papers"
    paper_dir = papers_root / archive.paper_id
    if _cached_crawler_paper_valid(paper_dir, archive):
        return {
            "paper_id": archive.paper_id,
            "status": "prepared",
            "paper_dir": str(paper_dir),
            "archive": archive.archive,
            "input_bytes": archive.input_bytes,
            "resume_state": "reused",
            "elapsed_seconds": time.monotonic() - started,
        }

    staging_root = cache_root / ".staging"
    staging = staging_root / (f"{archive.paper_id}.{os.getpid()}.{time.time_ns()}")
    source_dir = staging / "source"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        archive_path = Path(archive.archive)
        actual_sha256 = _sha256_file(archive_path)
        if (
            archive.expected_sha256 is not None
            and actual_sha256 != archive.expected_sha256
        ):
            raise PipelineError("source_archive_sha256_mismatch")
        extraction = safely_extract_source_archive(archive_path, source_dir)
        safety_scan = scan_archive_source(source_dir)
        if safety_scan.get("status") != "passed":
            raise PipelineError("dangerous_source_construct_detected")
        main_tex, candidates = find_archive_main_tex(source_dir)
        main_relative = main_tex.relative_to(source_dir).as_posix()
        metadata = {
            **archive.row,
            "paper_id": archive.paper_id,
            "stem": archive.paper_id,
            "status": "prepared",
            "main_tex": main_relative,
            "main_tex_candidates": candidates,
            "archive_path": archive.archive,
            "archive_bytes": archive.input_bytes,
            "archive_mtime_ns": archive.input_mtime_ns,
            "archive_sha256": actual_sha256,
            "expected_sha256": archive.expected_sha256,
            "extraction": extraction,
            "safety_scan": safety_scan,
            "crawler_cache_schema_version": _CRAWLER_CACHE_SCHEMA_VERSION,
        }
        _atomic_json(staging / "metadata.json", metadata)
        papers_root.mkdir(parents=True, exist_ok=True)
        if paper_dir.exists():
            diagnostics = cache_root / "diagnostics"
            diagnostics.mkdir(parents=True, exist_ok=True)
            stale = diagnostics / (
                f"{archive.paper_id}.stale.{time.strftime('%Y%m%dT%H%M%S')}"
                f".{time.time_ns()}"
            )
            paper_dir.replace(stale)
        staging.replace(paper_dir)
        return {
            "paper_id": archive.paper_id,
            "status": "prepared",
            "paper_dir": str(paper_dir),
            "archive": archive.archive,
            "input_bytes": archive.input_bytes,
            "archive_sha256": actual_sha256,
            "files": int(extraction.get("files", 0)),
            "expanded_bytes": int(extraction.get("bytes", 0)),
            "resume_state": "created",
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:  # noqa: BLE001 - isolate one downloaded archive
        reason = f"{type(exc).__name__}: {exc}"
        rejected = isinstance(exc, PipelineError) and str(exc) in {
            "source_archive_sha256_mismatch",
            "dangerous_source_construct_detected",
        }
        return {
            "paper_id": archive.paper_id,
            "status": "rejected" if rejected else "failed",
            "paper_dir": None,
            "archive": archive.archive,
            "input_bytes": archive.input_bytes,
            "error": reason,
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    log_path: Path,
) -> tuple[int | None, bool, str]:
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"command={list(command)!r}\n"
        f"timeout={timeout}\n"
        f"timed_out={timed_out}\n"
        f"elapsed_seconds={time.monotonic() - started:.3f}\n\n{output}",
        encoding="utf-8",
    )
    return process.returncode, timed_out, output


def _page_count(pdf: Path, config: WorkerConfig, page_dir: Path) -> int:
    rc, timed_out, output = _run_command(
        [config.pdfinfo, str(pdf)],
        cwd=page_dir,
        timeout=15,
        log_path=page_dir / "pdfinfo.log",
    )
    if timed_out or rc != 0:
        raise PipelineError("pdfinfo_failed")
    for line in output.splitlines():
        if line.startswith("Pages:"):
            value = line.partition(":")[2].strip()
            if value.isdigit():
                return int(value)
    raise PipelineError("pdfinfo_missing_page_count")


def _parse_bbox_output(output: str, *, layout: str) -> BBoxObservation:
    try:
        root = ET.fromstring(output)
    except ET.ParseError as exc:
        raise PipelineError(f"pdftotext_bbox_parse_failed:{exc}") from exc
    pages = [element for element in root.iter() if element.tag.endswith("page")]
    if len(pages) != 1:
        raise PipelineError(f"pdftotext_bbox_page_count:{len(pages)}")
    page = pages[0]
    width = float(page.attrib["width"])
    height = float(page.attrib["height"])
    lines: list[tuple[int, float, float, float, str, tuple[RenderedWord, ...]]] = []
    for line in (element for element in page.iter() if element.tag.endswith("line")):
        word_elements = [word for word in line if word.tag.endswith("word")]
        if not word_elements:
            continue
        x_min = float(line.attrib["xMin"])
        y_min = float(line.attrib["yMin"])
        y_max = float(line.attrib["yMax"])
        column = 0
        if layout == "two_column":
            # A displayed equation can extend across the page midpoint while
            # still belonging to the left minipage.  Column membership is
            # therefore anchored by the line start, not its visual centre.
            column = 0 if x_min < width / 2 else 1
        words = tuple(
            RenderedWord(
                text="".join(word.itertext()),
                column=column,
                x_min=float(word.attrib["xMin"]),
                y_min=float(word.attrib["yMin"]),
                x_max=float(word.attrib["xMax"]),
                y_max=float(word.attrib["yMax"]),
            )
            for word in word_elements
        )
        lines.append(
            (column, y_min, x_min, y_max, " ".join(word.text for word in words), words)
        )
    lines.sort(key=lambda row: (row[0], row[1], row[2]))
    column_count = 2 if layout == "two_column" else 1
    usable_height = max(1.0, height - 2 * _PAGE_MARGIN_PT)
    fills: list[float] = []
    for column in range(column_count):
        bottoms = [row[3] for row in lines if row[0] == column]
        raw = (max(bottoms) - _PAGE_MARGIN_PT) / usable_height if bottoms else 0.0
        fills.append(max(0.0, min(1.0, raw)))
    content_fill = min(fills) if layout == "two_column" else fills[0]
    return BBoxObservation(
        text="\n".join(row[4] for row in lines),
        words=tuple(word for row in lines for word in row[5]),
        content_fill_ratio=content_fill,
        column_fill_ratios=tuple(fills),
        page_width=width,
        page_height=height,
    )


def _bbox_text_for_verification(
    pdf: Path,
    *,
    layout: str,
    config: WorkerConfig,
    page_dir: Path,
) -> tuple[BBoxObservation | None, str | None]:
    rc, timed_out, output = _run_command(
        [config.pdftotext, "-bbox-layout", str(pdf), "-"],
        cwd=page_dir,
        timeout=20,
        log_path=page_dir / "pdftotext.log",
    )
    if timed_out or rc != 0:
        return None, "pdftotext_failed"
    try:
        return _parse_bbox_output(output, layout=layout), None
    except (PipelineError, KeyError, TypeError, ValueError) as exc:
        return None, str(exc)


def _page_signature(page: CanonicalPage, tex: str) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "paper_id": page.paper_id,
        "layout": page.layout,
        "blocks": [
            {
                "block_id": block.block_id,
                "node_id": block.node_id,
                "kind": block.kind,
                "markdown": block.markdown,
                "latex": block.latex,
                "source_char_span": block.source_char_span,
            }
            for block in page.blocks
        ],
        "tex": tex,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _worker_result_from_json(existing: dict[str, Any]) -> WorkerResult:
    payload = {
        key: existing[key]
        for key in WorkerResult.__dataclass_fields__
        if key in existing
    }
    for key in ("block_ids", "source_node_ids", "column_fill_ratios", "changes"):
        if key in payload:
            payload[key] = tuple(payload[key])
    return WorkerResult(**payload)


def _compile_once(page: CanonicalPage, config: WorkerConfig) -> WorkerResult:
    started = time.monotonic()
    tex = build_page_tex(page)
    page_signature = _page_signature(page, tex)
    work_root = Path(config.work_dir or config.output_dir)
    page_dir = work_root / "pages" / page.page_id
    result_path = page_dir / "result.json"
    if result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                existing.get("pipeline_version") == PIPELINE_VERSION
                and existing.get("page_signature") == page_signature
            ):
                cached = _worker_result_from_json(existing)
                artifacts_exist = cached.status != "accepted" or (
                    cached.pdf is not None
                    and cached.image is not None
                    and Path(cached.pdf).is_file()
                    and Path(cached.image).is_file()
                )
                if artifacts_exist:
                    return cached
        except (KeyError, TypeError, ValueError):
            pass

    source_dir = page_dir / "source"
    build_dir = page_dir / "build"
    source_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(source_dir / "page.tex", tex)
    _atomic_text(page_dir / "ground_truth.md", page.markdown + "\n")

    if len(page.markdown.strip()) < config.min_page_chars:
        result = WorkerResult(
            page_id=page.page_id,
            paper_id=page.paper_id,
            status="rejected",
            reason=f"sparse_page:chars={len(page.markdown.strip())}",
            layout=page.layout,
            has_table=page.has_table,
            markdown=page.markdown,
            verifier_recall=None,
            verifier_precision=None,
            pdf=None,
            image=None,
            block_ids=tuple(block.block_id for block in page.blocks),
            source_node_ids=tuple(block.node_id for block in page.blocks),
            content_fill_ratio=None,
            column_fill_ratios=(),
            page_signature=page_signature,
            elapsed_seconds=time.monotonic() - started,
        )
        _atomic_json(
            result_path,
            {"pipeline_version": PIPELINE_VERSION, **asdict(result)},
        )
        return result

    command = [
        config.latexmk,
        "-norc",
        "-g",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-no-shell-escape",
        f"-outdir={build_dir}",
        "page.tex",
    ]
    rc, timed_out, compile_output = _run_command(
        command,
        cwd=source_dir,
        timeout=config.compile_timeout,
        log_path=page_dir / "compile.log",
    )
    pdf = build_dir / "page.pdf"
    reason: str | None = None
    recall: float | None = None
    precision: float | None = None
    content_fill_ratio: float | None = None
    column_fill_ratios: tuple[float, ...] = ()
    image: Path | None = None
    if timed_out:
        reason = "compile_timeout"
    elif rc != 0 or not pdf.is_file():
        reason = f"compile_failed:return_code={rc}"
    else:
        overfull = [
            float(value)
            for value in re.findall(
                r"Overfull \\hbox \(([0-9]+(?:\.[0-9]+)?)pt too wide\)",
                compile_output,
            )
        ]
        if overfull and max(overfull) > 2.5:
            reason = f"visual_overflow:max_points={max(overfull):.3f}"
    if reason is None:
        try:
            pages = _page_count(pdf, config, page_dir)
        except (PipelineError, ValueError) as exc:
            reason = f"{type(exc).__name__}:{exc}"
        else:
            if pages != 1:
                reason = f"canonical_page_count:{pages}"
    if reason is None:
        observation, text_error = _bbox_text_for_verification(
            pdf,
            layout=page.layout,
            config=config,
            page_dir=page_dir,
        )
        if text_error is not None or observation is None:
            reason = text_error or "pdftotext_failed"
        else:
            content_fill_ratio = observation.content_fill_ratio
            column_fill_ratios = observation.column_fill_ratios
            passed, recall, precision = verify_rendered_text(
                page.verifier_text,
                observation.text,
            )
            _atomic_text(page_dir / "pdf_text_reject_only.txt", observation.text)
            if not passed:
                reason = (
                    "rendered_text_mismatch:"
                    f"recall={recall:.6f}:precision={precision:.6f}"
                )
    if reason is None:
        prefix = page_dir / "page"
        render_rc, render_timeout, _ = _run_command(
            [
                config.pdftoppm,
                "-f",
                "1",
                "-singlefile",
                "-png",
                "-r",
                str(config.dpi),
                str(pdf),
                str(prefix),
            ],
            cwd=page_dir,
            timeout=config.render_timeout,
            log_path=page_dir / "render.log",
        )
        image = page_dir / "page.png"
        if render_timeout or render_rc != 0 or not image.is_file():
            reason = "page_render_failed"
            image = None

    result = WorkerResult(
        page_id=page.page_id,
        paper_id=page.paper_id,
        status="accepted" if reason is None else "rejected",
        reason=reason,
        layout=page.layout,
        has_table=page.has_table,
        markdown=page.markdown,
        verifier_recall=recall,
        verifier_precision=precision,
        pdf=str(pdf.resolve()) if pdf.is_file() else None,
        image=str(image.resolve()) if image is not None else None,
        block_ids=tuple(block.block_id for block in page.blocks),
        source_node_ids=tuple(block.node_id for block in page.blocks),
        content_fill_ratio=content_fill_ratio,
        column_fill_ratios=column_fill_ratios,
        page_signature=page_signature,
        elapsed_seconds=time.monotonic() - started,
    )
    _atomic_json(
        result_path,
        {"pipeline_version": PIPELINE_VERSION, **asdict(result)},
    )
    return result


def _mutation_page_id(clean_page_id: str, config: MutationConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:10]
    return f"{clean_page_id}_confusable_s{config.seed}_{digest}"


def _point_bbox_to_pixel_bbox(
    bbox: Sequence[float],
    *,
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> list[int]:
    x_scale = image_width / page_width
    y_scale = image_height / page_height
    return [
        max(0, min(image_width, math.floor(float(bbox[0]) * x_scale))),
        max(0, min(image_height, math.floor(float(bbox[1]) * y_scale))),
        max(0, min(image_width, math.ceil(float(bbox[2]) * x_scale))),
        max(0, min(image_height, math.ceil(float(bbox[3]) * y_scale))),
    ]


def _mutation_rejection(
    clean_result: WorkerResult,
    *,
    page_id: str,
    reason: str,
    started: float,
    mutations: Sequence[PageMutation] = (),
    markdown: str | None = None,
) -> WorkerResult:
    return replace(
        clean_result,
        page_id=page_id,
        status="rejected",
        reason=reason,
        markdown=clean_result.markdown if markdown is None else markdown,
        pdf=None,
        image=None,
        page_signature=None,
        elapsed_seconds=time.monotonic() - started,
        mutation_count=len(mutations),
        changes=(),
        clean_page_id=clean_result.page_id,
        max_mutation_vertical_shift_points=None,
        variant="confusable_edit",
    )


def _mutation_change(
    mutation: PageMutation,
    edited_word: RenderedWord,
    *,
    observation: BBoxObservation,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    bbox_points = (
        edited_word.x_min,
        edited_word.y_min,
        edited_word.x_max,
        edited_word.y_max,
    )
    return {
        "ocr_ans": mutation.mutated_word,
        "origin_ans": mutation.original_word,
        "bbox": _point_bbox_to_pixel_bbox(
            bbox_points,
            page_width=observation.page_width,
            page_height=observation.page_height,
            image_width=image_width,
            image_height=image_height,
        ),
        "from_char": mutation.from_char,
        "to_char": mutation.to_char,
        "char_index_in_word": mutation.char_index_in_word,
        "block_id": mutation.block_id,
        "source_node_id": mutation.node_id,
        "source_files": list(mutation.source_files),
        "source_block_char_span": list(mutation.source_char_span),
        "canonical_markdown_span": [
            mutation.markdown_start,
            mutation.markdown_end,
        ],
        "clean_bbox_points": list(mutation.clean_bbox_points),
        "edited_bbox_points": list(bbox_points),
    }


def _direct_mutation_rejection(
    page: CanonicalPage,
    *,
    page_id: str,
    reason: str,
    started: float,
    mutations: Sequence[PageMutation] = (),
    markdown: str | None = None,
) -> WorkerResult:
    return WorkerResult(
        page_id=page_id,
        paper_id=page.paper_id,
        status="rejected",
        reason=reason,
        layout=page.layout,
        has_table=page.has_table,
        markdown=page.markdown if markdown is None else markdown,
        verifier_recall=None,
        verifier_precision=None,
        pdf=None,
        image=None,
        block_ids=tuple(block.block_id for block in page.blocks),
        source_node_ids=tuple(block.node_id for block in page.blocks),
        content_fill_ratio=None,
        column_fill_ratios=(),
        page_signature=None,
        elapsed_seconds=time.monotonic() - started,
        mutation_count=len(mutations),
        changes=(),
        clean_page_id=None,
        max_mutation_vertical_shift_points=None,
        variant="confusable_edit",
    )


def _direct_result_cache(
    page: CanonicalPage,
    config: WorkerConfig,
) -> WorkerResult | None:
    result_path = Path(config.output_dir) / "pages" / page.page_id / "result.json"
    if not result_path.is_file():
        return None
    try:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        expected_signature = _page_signature(page, build_page_tex(page))
        cached = _worker_result_from_json(existing)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        existing.get("pipeline_version") != PIPELINE_VERSION
        or cached.page_signature != expected_signature
        or cached.status != "accepted"
        or cached.variant != "confusable_edit"
        or cached.image is None
        or not cached.changes
        or not Path(cached.image).is_file()
    ):
        return None
    return cached


def _finish_direct_compile(
    result: WorkerResult,
    config: WorkerConfig,
) -> WorkerResult:
    """Promote only an accepted PNG/GT and discard its compiler workspace."""

    output_root = Path(config.output_dir)
    work_root = Path(config.work_dir or config.output_dir)
    work_page_dir = work_root / "pages" / result.page_id
    final_result = result
    if result.status == "accepted" and result.image is not None:
        final_page_dir = output_root / "pages" / result.page_id
        final_page_dir.mkdir(parents=True, exist_ok=True)
        final_image = final_page_dir / "page.png"
        source_image = Path(result.image)
        if source_image.resolve() != final_image.resolve():
            temporary_image = final_image.with_name(final_image.name + ".tmp")
            shutil.copy2(source_image, temporary_image)
            temporary_image.replace(final_image)
        _atomic_text(final_page_dir / "ground_truth.md", result.markdown + "\n")
        final_result = replace(
            result,
            pdf=None,
            image=str(final_image.resolve()),
        )
        _atomic_json(
            final_page_dir / "result.json",
            {"pipeline_version": PIPELINE_VERSION, **asdict(final_result)},
        )
    else:
        final_result = replace(result, pdf=None, image=None)
    if work_root.resolve() != output_root.resolve() and work_page_dir.exists():
        shutil.rmtree(work_page_dir)
    return final_result


def _direct_mutate_and_compile(
    page: CanonicalPage,
    config: WorkerConfig,
    mutation_config: MutationConfig,
) -> WorkerResult:
    """Mutate source channels first and compile only the edited page."""

    started = time.monotonic()
    page_id = _mutation_page_id(page.page_id, mutation_config)
    mutations = choose_source_page_mutations(
        page,
        seed=mutation_config.seed,
        minimum=mutation_config.minimum_per_page,
        maximum=mutation_config.maximum_per_page,
        maximum_probability=mutation_config.maximum_probability,
    )
    if len(mutations) < mutation_config.minimum_per_page:
        return _direct_mutation_rejection(
            page,
            page_id=page_id,
            reason=(
                "fewer_than_minimum_source_mutations:"
                f"required={mutation_config.minimum_per_page}:found={len(mutations)}"
            ),
            started=started,
        )

    mutated_page = apply_page_mutations(page, mutations, page_id=page_id)
    cached = _direct_result_cache(mutated_page, config)
    if cached is not None:
        return cached
    expected_differences = len(mutations)
    channel_differences = {
        "markdown": markdown_diff_count(page.markdown, mutated_page.markdown),
        "verifier": markdown_diff_count(
            page.verifier_text,
            mutated_page.verifier_text,
        ),
        "latex": markdown_diff_count(
            build_page_tex(page),
            build_page_tex(mutated_page),
        ),
    }
    if any(value != expected_differences for value in channel_differences.values()):
        return _direct_mutation_rejection(
            page,
            page_id=page_id,
            reason=f"mutation_channel_diff_mismatch:{channel_differences}",
            started=started,
            mutations=mutations,
            markdown=mutated_page.markdown,
        )

    edited_result = _compile_once(mutated_page, config)
    if edited_result.status != "accepted" or edited_result.pdf is None:
        result = replace(
            edited_result,
            status="rejected",
            reason=f"edited_{edited_result.reason or 'compile_rejected'}",
            pdf=None,
            image=None,
            elapsed_seconds=time.monotonic() - started,
            mutation_count=len(mutations),
            changes=(),
            clean_page_id=None,
            max_mutation_vertical_shift_points=None,
            variant="confusable_edit",
        )
        return _finish_direct_compile(result, config)

    page_dir = Path(config.work_dir or config.output_dir) / "pages" / page_id
    observation, observation_error = _bbox_text_for_verification(
        Path(edited_result.pdf),
        layout=mutated_page.layout,
        config=config,
        page_dir=page_dir,
    )
    if observation_error is not None or observation is None:
        result = replace(
            edited_result,
            status="rejected",
            reason=f"edited_bbox_failed:{observation_error or 'unknown'}",
            pdf=None,
            image=None,
            elapsed_seconds=time.monotonic() - started,
            mutation_count=len(mutations),
            changes=(),
            clean_page_id=None,
            max_mutation_vertical_shift_points=None,
            variant="confusable_edit",
        )
        return _finish_direct_compile(result, config)

    rendered_indices: dict[str, list[int]] = {}
    for index, word in enumerate(observation.words):
        rendered_indices.setdefault(word.text, []).append(index)
    located: list[PageMutation] = []
    for mutation in mutations:
        indexes = rendered_indices.get(mutation.mutated_word, [])
        if len(indexes) != 1:
            result = replace(
                edited_result,
                status="rejected",
                reason=(
                    "edited_mutation_word_not_unique:"
                    f"word={mutation.mutated_word}:matches={len(indexes)}"
                ),
                pdf=None,
                image=None,
                elapsed_seconds=time.monotonic() - started,
                mutation_count=len(mutations),
                changes=(),
                clean_page_id=None,
                max_mutation_vertical_shift_points=None,
                variant="confusable_edit",
            )
            return _finish_direct_compile(result, config)
        located.append(replace(mutation, rendered_word_index=indexes[0]))

    if edited_result.image is None:
        result = replace(
            edited_result,
            status="rejected",
            reason="edited_page_missing_image",
            pdf=None,
            image=None,
            elapsed_seconds=time.monotonic() - started,
            mutation_count=len(mutations),
            changes=(),
            clean_page_id=None,
            max_mutation_vertical_shift_points=None,
            variant="confusable_edit",
        )
        return _finish_direct_compile(result, config)

    with Image.open(edited_result.image) as image:
        image_width, image_height = image.size
    changes = tuple(
        _mutation_change(
            mutation,
            observation.words[mutation.rendered_word_index],
            observation=observation,
            image_width=image_width,
            image_height=image_height,
        )
        for mutation in located
    )
    result = replace(
        edited_result,
        elapsed_seconds=time.monotonic() - started,
        mutation_count=len(located),
        changes=changes,
        clean_page_id=None,
        max_mutation_vertical_shift_points=None,
        variant="confusable_edit",
    )
    return _finish_direct_compile(result, config)


def _mutate_and_compile(
    clean_result: WorkerResult,
    clean_page: CanonicalPage,
    config: WorkerConfig,
    mutation_config: MutationConfig,
) -> WorkerResult:
    """Create one edited-only page and prove that only declared words changed."""

    started = time.monotonic()
    page_id = _mutation_page_id(clean_result.page_id, mutation_config)
    if clean_result.status != "accepted":
        result = _mutation_rejection(
            clean_result,
            page_id=page_id,
            reason=f"clean_page_not_accepted:{clean_result.reason}",
            started=started,
        )
        _persist_terminal_result(result, config)
        return result
    if clean_result.pdf is None or clean_result.image is None:
        result = _mutation_rejection(
            clean_result,
            page_id=page_id,
            reason="clean_page_missing_artifacts",
            started=started,
        )
        _persist_terminal_result(result, config)
        return result

    clean_page_dir = Path(config.output_dir) / "pages" / clean_result.page_id
    clean_observation, clean_error = _bbox_text_for_verification(
        Path(clean_result.pdf),
        layout=clean_page.layout,
        config=config,
        page_dir=clean_page_dir,
    )
    if clean_error is not None or clean_observation is None:
        result = _mutation_rejection(
            clean_result,
            page_id=page_id,
            reason=f"clean_bbox_failed:{clean_error or 'unknown'}",
            started=started,
        )
        _persist_terminal_result(result, config)
        return result

    mutations = choose_page_mutations(
        clean_page,
        clean_observation.words,
        seed=mutation_config.seed,
        minimum=mutation_config.minimum_per_page,
        maximum=mutation_config.maximum_per_page,
        maximum_probability=mutation_config.maximum_probability,
    )
    if len(mutations) < mutation_config.minimum_per_page:
        result = _mutation_rejection(
            clean_result,
            page_id=page_id,
            reason=(
                "fewer_than_minimum_safe_unique_words:"
                f"required={mutation_config.minimum_per_page}:found={len(mutations)}"
            ),
            started=started,
        )
        _persist_terminal_result(result, config)
        return result

    mutated_page = apply_page_mutations(
        clean_page,
        mutations,
        page_id=page_id,
    )
    expected_differences = len(mutations)
    channel_differences = {
        "markdown": markdown_diff_count(clean_page.markdown, mutated_page.markdown),
        "verifier": markdown_diff_count(
            clean_page.verifier_text,
            mutated_page.verifier_text,
        ),
        "latex": markdown_diff_count(
            build_page_tex(clean_page),
            build_page_tex(mutated_page),
        ),
    }
    if any(value != expected_differences for value in channel_differences.values()):
        result = _mutation_rejection(
            clean_result,
            page_id=page_id,
            reason=f"mutation_channel_diff_mismatch:{channel_differences}",
            started=started,
            mutations=mutations,
            markdown=mutated_page.markdown,
        )
        _persist_terminal_result(result, config)
        return result

    edited_result = _compile_once(mutated_page, config)
    if edited_result.status != "accepted" or edited_result.pdf is None:
        result = replace(
            edited_result,
            status="rejected",
            reason=f"edited_{edited_result.reason or 'compile_rejected'}",
            pdf=None,
            image=None,
            elapsed_seconds=time.monotonic() - started,
            rescued=clean_result.rescued,
            pack_attempts=clean_result.pack_attempts,
            mutation_count=len(mutations),
            changes=(),
            clean_page_id=clean_result.page_id,
            max_mutation_vertical_shift_points=None,
            variant="confusable_edit",
        )
        _persist_terminal_result(result, config)
        return result

    edited_page_dir = Path(config.output_dir) / "pages" / edited_result.page_id
    edited_observation, edited_error = _bbox_text_for_verification(
        Path(edited_result.pdf),
        layout=mutated_page.layout,
        config=config,
        page_dir=edited_page_dir,
    )
    if edited_error is not None or edited_observation is None:
        result = replace(
            edited_result,
            status="rejected",
            reason=f"edited_bbox_failed:{edited_error or 'unknown'}",
            pdf=None,
            image=None,
            elapsed_seconds=time.monotonic() - started,
            rescued=clean_result.rescued,
            pack_attempts=clean_result.pack_attempts,
            mutation_count=len(mutations),
            changes=(),
            clean_page_id=clean_result.page_id,
            variant="confusable_edit",
        )
        _persist_terminal_result(result, config)
        return result

    validation = validate_mutated_word_geometry(
        clean_observation.words,
        edited_observation.words,
        mutations,
        max_vertical_shift_points=mutation_config.max_vertical_shift_points,
    )
    if not validation.passed:
        result = replace(
            edited_result,
            status="rejected",
            reason=f"mutation_validation:{validation.reason}",
            pdf=None,
            image=None,
            elapsed_seconds=time.monotonic() - started,
            rescued=clean_result.rescued,
            pack_attempts=clean_result.pack_attempts,
            mutation_count=len(mutations),
            changes=(),
            clean_page_id=clean_result.page_id,
            max_mutation_vertical_shift_points=(validation.max_vertical_shift_points),
            variant="confusable_edit",
        )
        _persist_terminal_result(result, config)
        return result

    if edited_result.image is None:
        result = replace(
            edited_result,
            status="rejected",
            reason="edited_page_missing_image",
            pdf=None,
            image=None,
            elapsed_seconds=time.monotonic() - started,
            rescued=clean_result.rescued,
            pack_attempts=clean_result.pack_attempts,
            mutation_count=len(mutations),
            changes=(),
            clean_page_id=clean_result.page_id,
            max_mutation_vertical_shift_points=(validation.max_vertical_shift_points),
            variant="confusable_edit",
        )
        _persist_terminal_result(result, config)
        return result

    with Image.open(edited_result.image) as image:
        image_width, image_height = image.size
    changes = tuple(
        _mutation_change(
            mutation,
            edited_observation.words[mutation.rendered_word_index],
            observation=edited_observation,
            image_width=image_width,
            image_height=image_height,
        )
        for mutation in mutations
    )
    result = replace(
        edited_result,
        elapsed_seconds=time.monotonic() - started,
        rescued=clean_result.rescued,
        pack_attempts=clean_result.pack_attempts,
        mutation_count=len(mutations),
        changes=changes,
        clean_page_id=clean_result.page_id,
        max_mutation_vertical_shift_points=validation.max_vertical_shift_points,
        variant="confusable_edit",
    )
    _persist_terminal_result(result, config)
    return result


def _flatten_bundles(
    bundles: Sequence[Sequence[CanonicalBlock]],
    start: int,
    end: int,
) -> tuple[CanonicalBlock, ...]:
    return tuple(block for bundle in bundles[start:end] for block in bundle)


def _dense_layout(
    page: CanonicalPage,
    blocks: Sequence[CanonicalBlock],
    *,
    output_ordinal: int,
    config: WorkerConfig,
) -> str:
    if any(block.has_table for block in blocks):
        return "one_column"
    weight = sum(block.weight for block in blocks)
    eligible = len(blocks) >= 4 and weight >= int(config.target_weight * 0.65)
    if not eligible:
        return "one_column"
    digest = int(
        hashlib.sha256(
            f"{page.paper_id}:{page.ordinal}:{output_ordinal}".encode()
        ).hexdigest()[:8],
        16,
    )
    return (
        "two_column"
        if digest % 10_000 < int(config.two_column_rate * 10_000)
        else "one_column"
    )


def _dense_page(
    page: CanonicalPage,
    bundles: Sequence[Sequence[CanonicalBlock]],
    *,
    start: int,
    end: int,
    output_ordinal: int,
    config: WorkerConfig,
    force_layout: str | None = None,
) -> CanonicalPage:
    blocks = _flatten_bundles(bundles, start, end)
    layout = force_layout or _dense_layout(
        page,
        blocks,
        output_ordinal=output_ordinal,
        config=config,
    )
    identity = "\0".join((layout, *(block.block_id for block in blocks)))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return CanonicalPage(
        page_id=(
            f"{page.paper_id}_dense_{page.ordinal:04d}_{output_ordinal:04d}_{digest}"
        ),
        paper_id=page.paper_id,
        ordinal=output_ordinal,
        layout=layout,
        blocks=blocks,
    )


def _initial_bundle_end(
    bundles: Sequence[Sequence[CanonicalBlock]],
    *,
    start: int,
    target_weight: int,
) -> int:
    end = start
    weight = 0
    while end < len(bundles):
        next_weight = sum(block.weight for block in bundles[end])
        if end > start and weight + next_weight > target_weight:
            break
        weight += next_weight
        end += 1
    return max(start + 1, end)


def _persist_terminal_result(result: WorkerResult, config: WorkerConfig) -> None:
    page_dir = Path(config.output_dir) / "pages" / result.page_id
    if result.status != "accepted" and result.variant == "confusable_edit":
        if page_dir.is_dir():
            shutil.rmtree(page_dir)
        if config.minimal_output:
            return
    if not config.minimal_output:
        _atomic_json(
            page_dir / "terminal_result.json",
            {"pipeline_version": PIPELINE_VERSION, **asdict(result)},
        )
    if (
        result.status == "accepted"
        and result.variant == "confusable_edit"
        and result.image is not None
        and result.changes
    ):
        # A dense source-pool job can take minutes and return many terminal
        # pages at once.  Persist each accepted page from its worker instead of
        # waiting for the whole job (or the whole corpus) to finish.  Unique
        # filenames make these atomic writes safe across processes.
        parts_dir = Path(config.output_dir) / "realtime_training" / "parts"
        sft, verl = _realtime_training_rows(result, parts_dir.parent)
        _atomic_text(
            parts_dir / f"{result.page_id}.sft.jsonl",
            json.dumps(sft, ensure_ascii=False) + "\n",
        )
        _atomic_text(
            parts_dir / f"{result.page_id}.verl.jsonl",
            json.dumps(verl, ensure_ascii=False) + "\n",
        )


def _remove_nonterminal_direct_artifacts(
    attempts: Sequence[WorkerResult],
    *,
    terminal_page_ids: set[str],
    config: WorkerConfig,
) -> None:
    for attempt in attempts:
        if attempt.variant != "confusable_edit" or attempt.page_id in terminal_page_ids:
            continue
        page_dir = Path(config.output_dir) / "pages" / attempt.page_id
        if page_dir.is_dir():
            shutil.rmtree(page_dir)


def _compile_with_rescue(
    page: CanonicalPage,
    config: WorkerConfig,
    depth: int = 0,
    *,
    mutation_config: MutationConfig | None = None,
) -> tuple[WorkerResult, ...]:
    """Compile a source sequence into dense pages with one-bundle backoff.

    The former midpoint bisection emitted two half-filled descendants.  This
    packer instead keeps the last verified prefix, adds adjacent source
    bundles while the page is below the target fill, and removes only the last
    bundle after overflow or verifier failure.  A failed indivisible bundle is
    rejected without sacrificing its verified neighbors.
    """

    del depth  # retained in the signature for callers of the earlier V4 pilot

    def compile_candidate(candidate: CanonicalPage) -> WorkerResult:
        if mutation_config is None:
            return _compile_once(candidate, config)
        return _direct_mutate_and_compile(candidate, config, mutation_config)

    bundles = list(bundle_blocks(page.blocks))
    terminal: list[WorkerResult] = []
    start = 0
    output_ordinal = 1
    while start < len(bundles):
        terminal_start = len(terminal)
        attempt_results: list[WorkerResult] = []
        end = _initial_bundle_end(
            bundles,
            start=start,
            target_weight=config.target_weight,
        )
        current_end = end
        best: tuple[int, WorkerResult] | None = None
        last_result: WorkerResult | None = None
        attempts = 0
        attempted_ends: set[tuple[int, str | None]] = set()

        while attempts < config.max_pack_attempts:
            candidate = _dense_page(
                page,
                bundles,
                start=start,
                end=current_end,
                output_ordinal=output_ordinal,
                config=config,
            )
            attempt_key = (current_end, candidate.layout)
            if attempt_key in attempted_ends:
                break
            attempted_ends.add(attempt_key)
            result = compile_candidate(candidate)
            attempt_results.append(result)
            last_result = result
            attempts += 1

            if result.status == "accepted":
                best = (current_end, result)
                fill = result.content_fill_ratio or 0.0
                if fill >= config.target_fill_ratio or current_end >= len(bundles):
                    break
                current_end += 1
                continue

            if best is not None:
                break
            if (
                result.reason is not None
                and (
                    result.reason.startswith("sparse_page:")
                    or result.reason.startswith("fewer_than_minimum_source_mutations:")
                )
                and current_end < len(bundles)
            ):
                current_end += 1
                continue
            if current_end - start > 1:
                current_end -= 1
                continue
            break

        if best is None and current_end - start > 1:
            # A low attempt cap must never consume a multi-bundle failure as
            # one terminal unit.  Isolate the first indivisible bundle once.
            candidate = _dense_page(
                page,
                bundles,
                start=start,
                end=start + 1,
                output_ordinal=output_ordinal,
                config=config,
            )
            last_result = compile_candidate(candidate)
            attempt_results.append(last_result)
            attempts += 1
            if last_result.status == "accepted":
                best = (start + 1, last_result)

        if best is not None:
            best_end, result = best
            fill = result.content_fill_ratio or 0.0
            if (
                fill < config.min_fill_ratio
                and best_end < len(bundles)
                and last_result is not None
                and last_result.status != "accepted"
            ):
                # If a verified but short prefix is blocked by one bad
                # bundle, prove that bundle independently.  A genuinely bad
                # bundle is rejected and removed from the source pool so the
                # safe prefix can continue packing with later source content.
                obstruction = _dense_page(
                    page,
                    bundles,
                    start=best_end,
                    end=best_end + 1,
                    output_ordinal=output_ordinal,
                    config=config,
                    force_layout="one_column",
                )
                obstruction_result = compile_candidate(obstruction)
                attempt_results.append(obstruction_result)
                attempts += 1
                if obstruction_result.status != "accepted":
                    obstruction_result = replace(
                        obstruction_result,
                        rescued=True,
                        pack_attempts=attempts,
                    )
                    terminal.append(obstruction_result)
                    _persist_terminal_result(obstruction_result, config)
                    _remove_nonterminal_direct_artifacts(
                        attempt_results,
                        terminal_page_ids={obstruction_result.page_id},
                        config=config,
                    )
                    del bundles[best_end]
                    output_ordinal += 1
                    continue
            if fill < config.min_fill_ratio and result.layout == "two_column":
                fallback = _dense_page(
                    page,
                    bundles,
                    start=start,
                    end=best_end,
                    output_ordinal=output_ordinal,
                    config=config,
                    force_layout="one_column",
                )
                fallback_result = compile_candidate(fallback)
                attempt_results.append(fallback_result)
                attempts += 1
                if (
                    fallback_result.status == "accepted"
                    and (fallback_result.content_fill_ratio or 0.0) > fill
                ):
                    result = fallback_result
                    fill = fallback_result.content_fill_ratio or 0.0
            if fill < config.min_fill_ratio:
                result = replace(
                    result,
                    status="rejected",
                    reason=(
                        "underfilled_page:"
                        f"fill={fill:.6f}:minimum={config.min_fill_ratio:.6f}"
                    ),
                )
            result = replace(
                result,
                rescued=attempts > 1,
                pack_attempts=attempts,
            )
            terminal.append(result)
            _persist_terminal_result(result, config)
            start = best_end
        else:
            if last_result is None:
                raise PipelineError("dense_packer_produced_no_attempt")
            result = replace(
                last_result,
                rescued=attempts > 1,
                pack_attempts=attempts,
            )
            terminal.append(result)
            _persist_terminal_result(result, config)
            start += 1
        _remove_nonterminal_direct_artifacts(
            attempt_results,
            terminal_page_ids={item.page_id for item in terminal[terminal_start:]},
            config=config,
        )
        output_ordinal += 1
    return tuple(terminal)


def _safe_sources(
    source_dir: Path, canonical_source: str, main_relative: str
) -> dict[str, str]:
    sources = {main_relative: canonical_source}
    candidates = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tex", ".sty", ".cls"}
    )
    for path in candidates[:1000]:
        if path.stat().st_size > 8 * 1024 * 1024:
            continue
        relative = path.relative_to(source_dir).as_posix()
        if relative == main_relative:
            continue
        sources[relative] = path.read_text(encoding="utf-8", errors="replace")
    return sources


def _main_relative(metadata: dict[str, Any], source_dir: Path) -> str:
    raw = metadata.get("main_tex")
    if not isinstance(raw, str) or not raw:
        candidates: list[tuple[int, int, Path]] = []
        preferred_names = {
            "main.tex": 5,
            "paper.tex": 4,
            "manuscript.tex": 3,
            "article.tex": 2,
        }
        for path in sorted(source_dir.rglob("*.tex")):
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if r"\documentclass" not in text or r"\begin{document}" not in text:
                continue
            candidates.append(
                (
                    preferred_names.get(path.name.casefold(), 0),
                    len(text),
                    path,
                )
            )
        if not candidates:
            raise PipelineError("metadata_missing_main_tex_and_no_standalone_candidate")
        # This selects only a complete standalone document.  Even if an
        # archive carries several drafts, V4 consumes it as source corpus and
        # never claims to reproduce the originally submitted PDF.
        return max(candidates)[2].relative_to(source_dir).as_posix()
    value = Path(raw)
    if value.is_absolute():
        try:
            return value.resolve().relative_to(source_dir.resolve()).as_posix()
        except ValueError:
            candidates = list(source_dir.rglob(value.name))
            if len(candidates) == 1:
                return candidates[0].relative_to(source_dir).as_posix()
            raise PipelineError("absolute_main_tex_not_relocatable")
    return value.as_posix()


def _extract_paper(
    paper_dir: Path,
    *,
    target_weight: int,
    two_column_rate: float,
) -> tuple[tuple[CanonicalPage, ...], dict[str, Any]]:
    paper_id = paper_dir.name
    metadata_path = paper_dir / "metadata.json"
    source_dir = paper_dir / "source"
    if not metadata_path.is_file() or not source_dir.is_dir():
        raise PipelineError("paper_missing_metadata_or_source")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    main_relative = _main_relative(metadata, source_dir)
    _emit("extract-stage", f"paper={paper_id} stage=flatten main={main_relative}")
    flattened = flatten_source_project(source_dir, main_relative)
    _emit(
        "extract-stage",
        f"paper={paper_id} stage=sanitize chars={len(flattened.source)}",
    )
    sanitized = sanitize_latex_source(
        flattened.source,
        source_id=paper_id,
        drop_figures=True,
    )
    if sanitized.audit.status.startswith("rejected"):
        raise PipelineError(
            f"source_sanitizer:{sanitized.audit.status}:{sanitized.audit.message}"
        )
    canonical = sanitized.sanitized_source
    sources = _safe_sources(source_dir, canonical, flattened.main_tex)
    macro_registry = collect_source_macro_definitions(sources)
    list_registry = collect_source_list_environment_definitions(sources)
    semantic_environments, semantic_rejections = (
        extract_semantic_environment_definitions(sources)
    )
    _emit(
        "extract-stage",
        f"paper={paper_id} stage=document_ast sources={len(sources)}",
    )
    document = parse_document_ast(
        canonical,
        source_id=paper_id,
        source_map=sanitized.source_map,
        reference_values={},
        math_macros=macro_registry.by_name,
        semantic_environments=semantic_environments,
        list_environments=list_registry.by_name,
        enable_strict_tables=True,
    )
    _emit(
        "extract-stage",
        f"paper={paper_id} stage=canonical_blocks leaves={len(document.leaf_nodes)}",
    )
    blocks, block_rejections = blocks_from_document(document, paper_id=paper_id)
    pages = pack_blocks(
        blocks,
        paper_id=paper_id,
        target_weight=target_weight,
        two_column_rate=two_column_rate,
    )
    report = {
        "paper_id": paper_id,
        "main_tex": flattened.main_tex,
        "flattened_files": list(flattened.files),
        "sanitizer": sanitized.audit.to_dict(),
        "macro_definitions": len(macro_registry.definitions),
        "macro_rejections": len(macro_registry.rejections),
        "list_environment_definitions": len(list_registry.definitions),
        "list_environment_rejections": len(list_registry.rejections),
        "semantic_environment_definitions": len(semantic_environments),
        "semantic_environment_rejections": len(semantic_rejections),
        "ast_nodes": len(document.nodes),
        "ast_leaf_nodes": len(document.leaf_nodes),
        "ast_rejections": len(document.rejections),
        "blocks_accepted": len(blocks),
        "blocks_rejected": len(block_rejections),
        "table_blocks": sum(block.has_table for block in blocks),
        "page_candidates": len(pages),
        "block_rejections": list(block_rejections),
        "ground_truth_source": "latex_ast_only",
        "pdf_role": "reject_only",
    }
    return pages, report


def _select_papers(root: Path, limit: int, seed: int) -> list[Path]:
    papers = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "metadata.json").is_file()
        and (path / "source").is_dir()
    )
    random.Random(seed).shuffle(papers)
    return papers if limit <= 0 else papers[:limit]


def _prepare_crawler_archives_parallel(
    archives: Sequence[CrawlerArchive],
    *,
    cache_root: Path,
    workers: int,
    global_started: float,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    """Materialize raw crawler bins with bounded process-level concurrency."""

    total = len(archives)
    total_bytes = sum(item.input_bytes for item in archives)
    phase_started = time.monotonic()
    completed = prepared = rejected = errors = 0
    completed_bytes = expanded_bytes = 0
    results_by_id: dict[str, dict[str, Any]] = {}
    selected_by_id = {item.paper_id: item for item in archives}
    iterator = iter(archives)
    max_inflight = min(total, max(1, workers * 2))

    def submit_next(
        executor: ProcessPoolExecutor,
        pending: dict[Any, CrawlerArchive],
    ) -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(_materialize_crawler_archive, item, cache_root)
        pending[future] = item
        return True

    _emit(
        "crawler-prepare-start",
        f"archives={total} input_bytes={total_bytes} workers={workers} "
        f"max_inflight={max_inflight} cache={cache_root}",
    )
    with ProcessPoolExecutor(max_workers=min(workers, max(1, total))) as executor:
        pending: dict[Any, CrawlerArchive] = {}
        while len(pending) < max_inflight and submit_next(executor, pending):
            pass
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                phase_elapsed = time.monotonic() - phase_started
                rate = completed / max(phase_elapsed, 1e-9)
                eta = (total - completed) / max(rate, 1e-9)
                _emit(
                    "crawler-prepare-progress",
                    f"completed={completed}/{total} percent={completed / max(1, total):.1%} "
                    f"pending={len(pending)} bytes={completed_bytes}/{total_bytes} "
                    f"prepared={prepared} rejected={rejected} errors={errors} "
                    f"rate={rate:.3f}_papers/s elapsed={phase_elapsed:.1f}s eta={eta:.1f}s",
                )
                continue
            for future in done:
                item = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - process failure isolation
                    result = {
                        "paper_id": item.paper_id,
                        "status": "failed",
                        "paper_dir": None,
                        "archive": item.archive,
                        "input_bytes": item.input_bytes,
                        "error": f"worker_error:{type(exc).__name__}:{exc}",
                        "elapsed_seconds": 0.0,
                    }
                results_by_id[item.paper_id] = result
                completed += 1
                completed_bytes += item.input_bytes
                expanded_bytes += int(result.get("expanded_bytes", 0))
                prepared += result.get("status") == "prepared"
                rejected += result.get("status") == "rejected"
                errors += result.get("status") == "failed"
                phase_elapsed = time.monotonic() - phase_started
                rate = completed / max(phase_elapsed, 1e-9)
                byte_rate = completed_bytes / max(phase_elapsed, 1e-9)
                eta = (total - completed) / max(rate, 1e-9)
                _emit(
                    "crawler-prepare-unit",
                    f"completed={completed}/{total} percent={completed / max(1, total):.1%} "
                    f"current={item.paper_id} status={result.get('status')} "
                    f"bytes={completed_bytes}/{total_bytes} expanded_bytes={expanded_bytes} "
                    f"throughput={rate:.3f}_papers/s byte_rate={byte_rate / 1048576:.2f}_MiB/s "
                    f"elapsed={phase_elapsed:.1f}s eta={eta:.1f}s prepared={prepared} "
                    f"rejected={rejected} errors={errors}",
                )
                submit_next(executor, pending)

    ordered_results = [results_by_id[item.paper_id] for item in archives]
    paper_dirs = [
        Path(str(result["paper_dir"]))
        for result in ordered_results
        if result.get("status") == "prepared" and result.get("paper_dir")
    ]
    report = {
        "mode": "crawler_archives",
        "crawler_archives_selected": total,
        "crawler_archives_prepared": prepared,
        "crawler_archives_rejected": rejected,
        "crawler_archives_failed": errors,
        "crawler_input_bytes": total_bytes,
        "crawler_completed_bytes": completed_bytes,
        "crawler_expanded_bytes": expanded_bytes,
        "crawler_cache_root": str(cache_root),
        "workers": workers,
        "elapsed_seconds": time.monotonic() - phase_started,
        "global_elapsed_seconds": time.monotonic() - global_started,
        "selected_ids": [item.paper_id for item in archives],
        "prepared_ids": [path.name for path in paper_dirs],
        "archive_stats": {
            item.paper_id: {
                "input_bytes": item.input_bytes,
                "archive": item.archive,
            }
            for item in archives
            if item.paper_id in selected_by_id
        },
    }
    return paper_dirs, ordered_results, report


def _extract_paper_job(
    paper_dir: Path,
    target_weight: int,
    two_column_rate: float,
) -> tuple[tuple[CanonicalPage, ...], dict[str, Any]]:
    started = time.monotonic()
    try:
        pages, report = _extract_paper(
            paper_dir,
            target_weight=target_weight,
            two_column_rate=two_column_rate,
        )
        report = {
            **report,
            "status": "success",
            "elapsed_seconds": time.monotonic() - started,
        }
        return pages, report
    except Exception as exc:  # noqa: BLE001 - isolate one source project
        return (), {
            "paper_id": paper_dir.name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "page_candidates": 0,
            "blocks_accepted": 0,
            "elapsed_seconds": time.monotonic() - started,
        }


def _prepare_extract_crawler_job(
    archive: CrawlerArchive,
    cache_root: Path,
    target_weight: int,
    two_column_rate: float,
) -> tuple[tuple[CanonicalPage, ...], dict[str, Any] | None, dict[str, Any]]:
    """Unpack, extract immutable AST pages, then delete the unpacked copy."""

    job_root = (
        cache_root / "jobs" / f"{archive.paper_id}.{os.getpid()}.{time.time_ns()}"
    )
    preparation: dict[str, Any] = {
        "paper_id": archive.paper_id,
        "status": "failed",
        "paper_dir": None,
        "archive": archive.archive,
        "input_bytes": archive.input_bytes,
    }
    pages: tuple[CanonicalPage, ...] = ()
    report: dict[str, Any] | None = None
    try:
        preparation = _materialize_crawler_archive(archive, job_root)
        if preparation.get("status") != "prepared" or not preparation.get("paper_dir"):
            return (), None, preparation
        paper_dir = Path(str(preparation["paper_dir"]))
        pages, report = _extract_paper_job(
            paper_dir,
            target_weight,
            two_column_rate,
        )
        return pages, report, preparation
    finally:
        cleanup_started = time.monotonic()
        try:
            shutil.rmtree(job_root)
            preparation["cache_cleanup"] = "deleted"
        except FileNotFoundError:
            preparation["cache_cleanup"] = "already_absent"
        except OSError as error:
            preparation["cache_cleanup"] = "failed"
            preparation["cache_cleanup_error"] = f"{type(error).__name__}: {error}"
        preparation["temporary_paper_dir"] = preparation.get("paper_dir")
        preparation["paper_dir"] = None
        preparation["cache_cleanup_elapsed_seconds"] = (
            time.monotonic() - cleanup_started
        )


def _prepare_extract_crawler_parallel(
    archives: Sequence[CrawlerArchive],
    *,
    cache_root: Path,
    workers: int,
    target_weight: int,
    two_column_rate: float,
    global_started: float,
) -> tuple[
    list[CanonicalPage],
    list[dict[str, Any]],
    int,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Stream raw bins through unpack+AST extraction with bounded disk use."""

    total = len(archives)
    total_bytes = sum(item.input_bytes for item in archives)
    phase_started = time.monotonic()
    completed = prepared = rejected = preparation_errors = 0
    extraction_successes = extraction_errors = candidates = 0
    completed_bytes = expanded_bytes = cleaned = cleanup_errors = 0
    pages_by_id: dict[str, tuple[CanonicalPage, ...]] = {}
    extraction_reports: dict[str, dict[str, Any]] = {}
    preparation_results: dict[str, dict[str, Any]] = {}
    iterator = iter(archives)
    max_inflight = min(total, max(1, workers * 2))

    def submit_next(
        executor: ProcessPoolExecutor,
        pending: dict[Any, CrawlerArchive],
    ) -> bool:
        try:
            archive = next(iterator)
        except StopIteration:
            return False
        pending[
            executor.submit(
                _prepare_extract_crawler_job,
                archive,
                cache_root,
                target_weight,
                two_column_rate,
            )
        ] = archive
        return True

    _emit(
        "crawler-stream-start",
        f"archives={total} input_bytes={total_bytes} workers={workers} "
        f"max_inflight={max_inflight} temporary_cache={cache_root}",
    )
    with ProcessPoolExecutor(max_workers=min(workers, max(1, total))) as executor:
        pending: dict[Any, CrawlerArchive] = {}
        while len(pending) < max_inflight and submit_next(executor, pending):
            pass
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                elapsed = time.monotonic() - phase_started
                rate = completed / max(elapsed, 1e-9)
                eta = (total - completed) / max(rate, 1e-9)
                _emit(
                    "crawler-stream-progress",
                    f"completed={completed}/{total} percent="
                    f"{completed / max(1, total):.1%} pending={len(pending)} "
                    f"bytes={completed_bytes}/{total_bytes} "
                    f"prepared={prepared} rejected={rejected} "
                    f"errors={preparation_errors + extraction_errors} "
                    f"candidates={candidates} cache_deleted={cleaned} "
                    f"cleanup_errors={cleanup_errors} rate={rate:.3f}_papers/s "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                )
                continue
            for future in done:
                archive = pending.pop(future)
                try:
                    paper_pages, extraction_report, preparation = future.result()
                except Exception as error:  # noqa: BLE001 - isolate process failure
                    paper_pages = ()
                    extraction_report = None
                    preparation = {
                        "paper_id": archive.paper_id,
                        "status": "failed",
                        "paper_dir": None,
                        "archive": archive.archive,
                        "input_bytes": archive.input_bytes,
                        "error": f"worker_error:{type(error).__name__}:{error}",
                        "cache_cleanup": "unknown_after_worker_error",
                        "elapsed_seconds": 0.0,
                    }
                paper_id = archive.paper_id
                preparation_results[paper_id] = preparation
                pages_by_id[paper_id] = paper_pages
                if extraction_report is not None:
                    extraction_reports[paper_id] = extraction_report
                completed += 1
                completed_bytes += archive.input_bytes
                expanded_bytes += int(preparation.get("expanded_bytes", 0))
                prepared += preparation.get("status") == "prepared"
                rejected += preparation.get("status") == "rejected"
                preparation_errors += preparation.get("status") == "failed"
                extraction_successes += bool(
                    extraction_report and extraction_report.get("status") == "success"
                )
                extraction_errors += bool(
                    extraction_report and extraction_report.get("status") != "success"
                )
                candidates += len(paper_pages)
                cleaned += preparation.get("cache_cleanup") in {
                    "deleted",
                    "already_absent",
                }
                cleanup_errors += preparation.get("cache_cleanup") == "failed"
                elapsed = time.monotonic() - phase_started
                rate = completed / max(elapsed, 1e-9)
                byte_rate = completed_bytes / max(elapsed, 1e-9)
                eta = (total - completed) / max(rate, 1e-9)
                _emit(
                    "crawler-stream-unit",
                    f"completed={completed}/{total} percent="
                    f"{completed / max(1, total):.1%} current={paper_id} "
                    f"prepare={preparation.get('status')} extract="
                    f"{extraction_report.get('status') if extraction_report else 'skipped'} "
                    f"cache_cleanup={preparation.get('cache_cleanup')} "
                    f"paper_candidates={len(paper_pages)} "
                    f"cumulative_candidates={candidates} bytes="
                    f"{completed_bytes}/{total_bytes} expanded_bytes={expanded_bytes} "
                    f"throughput={rate:.3f}_papers/s "
                    f"byte_rate={byte_rate / 1048576:.2f}_MiB/s "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
                    f"accepted={extraction_successes} rejected={rejected} "
                    f"errors={preparation_errors + extraction_errors} "
                    f"cache_deleted={cleaned} cleanup_errors={cleanup_errors}",
                )
                submit_next(executor, pending)

    ordered_ids = [archive.paper_id for archive in archives]
    pages = [page for paper_id in ordered_ids for page in pages_by_id[paper_id]]
    pages.sort(key=lambda page: (page.paper_id, page.ordinal, page.page_id))
    reports = [extraction_reports[paper_id] for paper_id in sorted(extraction_reports)]
    preparation_rows = [preparation_results[paper_id] for paper_id in ordered_ids]
    report = {
        "mode": "crawler_archives",
        "processing_mode": "streaming_ephemeral_unpack_extract",
        "crawler_archives_selected": total,
        "crawler_archives_prepared": prepared,
        "crawler_archives_rejected": rejected,
        "crawler_archives_failed": preparation_errors,
        "crawler_extraction_succeeded": extraction_successes,
        "crawler_extraction_failed": extraction_errors,
        "crawler_input_bytes": total_bytes,
        "crawler_completed_bytes": completed_bytes,
        "crawler_expanded_bytes": expanded_bytes,
        "crawler_cache_root": str(cache_root),
        "crawler_cache_policy": "delete_each_paper_after_ast_extraction",
        "crawler_cache_dirs_deleted": cleaned,
        "crawler_cache_cleanup_errors": cleanup_errors,
        "workers": workers,
        "elapsed_seconds": time.monotonic() - phase_started,
        "global_elapsed_seconds": time.monotonic() - global_started,
        "selected_ids": ordered_ids,
        "prepared_ids": [
            paper_id
            for paper_id in ordered_ids
            if preparation_results[paper_id].get("status") == "prepared"
        ],
        "archive_stats": {
            archive.paper_id: {
                "input_bytes": archive.input_bytes,
                "archive": archive.archive,
            }
            for archive in archives
        },
    }
    _emit(
        "crawler-stream-finish",
        f"archives={total} prepared={prepared} rejected={rejected} "
        f"prepare_errors={preparation_errors} extraction_successes="
        f"{extraction_successes} extraction_errors={extraction_errors} "
        f"candidates={len(pages)} cache_deleted={cleaned} "
        f"cleanup_errors={cleanup_errors} elapsed="
        f"{time.monotonic() - phase_started:.1f}s global_elapsed="
        f"{time.monotonic() - global_started:.1f}s",
    )
    return (
        pages,
        reports,
        extraction_errors,
        preparation_rows,
        report,
    )


def _extract_papers_parallel(
    papers: Sequence[Path],
    *,
    workers: int,
    target_weight: int,
    two_column_rate: float,
    global_started: float,
) -> tuple[list[CanonicalPage], list[dict[str, Any]], int]:
    total = len(papers)
    if not total:
        return [], [], 0
    phase_started = time.monotonic()
    completed = successes = errors = candidates = 0
    pages_by_id: dict[str, tuple[CanonicalPage, ...]] = {}
    reports_by_id: dict[str, dict[str, Any]] = {}
    iterator = iter(papers)
    max_inflight = min(total, max(1, workers * 2))

    def submit_next(
        executor: ProcessPoolExecutor,
        pending: dict[Any, Path],
    ) -> bool:
        try:
            paper_dir = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(
            _extract_paper_job,
            paper_dir,
            target_weight,
            two_column_rate,
        )
        pending[future] = paper_dir
        return True

    _emit(
        "extract-start",
        f"papers={total} workers={workers} max_inflight={max_inflight}",
    )
    with ProcessPoolExecutor(max_workers=min(workers, total)) as executor:
        pending: dict[Any, Path] = {}
        while len(pending) < max_inflight and submit_next(executor, pending):
            pass
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                phase_elapsed = time.monotonic() - phase_started
                rate = completed / max(phase_elapsed, 1e-9)
                eta = (total - completed) / max(rate, 1e-9)
                _emit(
                    "extract-progress",
                    f"completed={completed}/{total} percent={completed / max(1, total):.1%} "
                    f"pending={len(pending)} candidates={candidates} successes={successes} "
                    f"rejected=0 errors={errors} rate={rate:.3f}_papers/s "
                    f"elapsed={phase_elapsed:.1f}s eta={eta:.1f}s",
                )
                continue
            for future in done:
                paper_dir = pending.pop(future)
                try:
                    paper_pages, report = future.result()
                except Exception as exc:  # noqa: BLE001 - process failure isolation
                    paper_pages = ()
                    report = {
                        "paper_id": paper_dir.name,
                        "status": "failed",
                        "error": f"worker_error:{type(exc).__name__}:{exc}",
                        "page_candidates": 0,
                        "blocks_accepted": 0,
                    }
                pages_by_id[paper_dir.name] = paper_pages
                reports_by_id[paper_dir.name] = report
                completed += 1
                successes += report.get("status") == "success"
                errors += report.get("status") != "success"
                candidates += len(paper_pages)
                phase_elapsed = time.monotonic() - phase_started
                rate = completed / max(phase_elapsed, 1e-9)
                eta = (total - completed) / max(rate, 1e-9)
                _emit(
                    "extract-unit",
                    f"completed={completed}/{total} percent={completed / max(1, total):.1%} "
                    f"current={paper_dir.name} status={report.get('status')} "
                    f"paper_candidates={len(paper_pages)} cumulative_candidates={candidates} "
                    f"throughput={rate:.3f}_papers/s elapsed={phase_elapsed:.1f}s eta={eta:.1f}s "
                    f"accepted={successes} rejected=0 errors={errors}",
                )
                submit_next(executor, pending)

    ordered_ids = sorted(reports_by_id)
    pages = [page for paper_id in ordered_ids for page in pages_by_id[paper_id]]
    pages.sort(key=lambda page: (page.paper_id, page.ordinal, page.page_id))
    reports = [reports_by_id[paper_id] for paper_id in ordered_ids]
    _emit(
        "extract-finish",
        f"papers={total} successes={successes} errors={errors} candidates={len(pages)} "
        f"elapsed={time.monotonic() - phase_started:.1f}s "
        f"global_elapsed={time.monotonic() - global_started:.1f}s",
    )
    return pages, reports, errors


def _limit_page_candidates(
    pages: Sequence[CanonicalPage],
    *,
    limit: int,
    seed: int,
) -> list[CanonicalPage]:
    selected = list(pages)
    if limit <= 0 or len(selected) <= limit:
        return selected
    # Avoid the old prefix bias where every pilot page came from the first
    # paper.  The deterministic shuffle samples papers, layouts, and tables
    # from the complete extracted candidate pool.
    random.Random(seed ^ 0xA4C4D15).shuffle(selected)
    return selected[:limit]


def _dense_jobs_from_pages(
    pages: Sequence[CanonicalPage],
) -> tuple[CanonicalPage, ...]:
    """Pool selected source blocks per paper for compile-driven repacking."""

    by_paper: dict[str, list[CanonicalPage]] = {}
    for page in pages:
        by_paper.setdefault(page.paper_id, []).append(page)
    jobs: list[CanonicalPage] = []
    for paper_id in sorted(by_paper):
        selected_pages = sorted(by_paper[paper_id], key=lambda item: item.ordinal)
        seen: set[str] = set()
        blocks: list[CanonicalBlock] = []
        for selected in selected_pages:
            for block in selected.blocks:
                if block.block_id in seen:
                    continue
                seen.add(block.block_id)
                blocks.append(block)
        if not blocks:
            continue
        first_ordinal = selected_pages[0].ordinal
        jobs.append(
            CanonicalPage(
                page_id=f"{paper_id}_dense_source_pool_{first_ordinal:04d}",
                paper_id=paper_id,
                ordinal=first_ordinal,
                layout="one_column",
                blocks=tuple(blocks),
            )
        )
    return tuple(jobs)


def _relative_output_path(value: str | None, output: Path) -> str | None:
    if value is None:
        return None
    return Path(value).resolve().relative_to(output.resolve()).as_posix()


def _realtime_training_rows(
    result: WorkerResult,
    dataset_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if result.image is None:
        raise PipelineError(f"accepted page missing image: {result.page_id}")
    image = Path(
        os.path.relpath(Path(result.image).resolve(), dataset_dir.resolve())
    ).as_posix()
    changes = [
        {
            "ocr_ans": change["ocr_ans"],
            "origin_ans": change["origin_ans"],
            "bbox": change["bbox"],
        }
        for change in result.changes
    ]
    sft = {
        "messages": [
            {"role": "user", "content": _SFT_PROMPT},
            {"role": "assistant", "content": result.markdown},
        ],
        "images": [image],
        "data_source": "chaos_document_ocr",
        "ability": "document_ocr",
        "extra_info": {
            "arxiv_id": result.paper_id,
            "pair_id": result.page_id,
        },
    }
    verl = {
        "data_source": "chaos_document_ocr",
        "prompt": [
            {
                "role": "user",
                "content": "<image>\nPlease transcribe all text in this page image faithfully, exactly as printed (including any typos).",
            }
        ],
        "images": [image],
        "reward_model": {"style": "rule", "ground_truth": result.markdown},
        "extra_info": {
            "arxiv_id": result.paper_id,
            "pair_id": result.page_id,
            "changes": changes,
        },
        "ability": "document_ocr",
    }
    return sft, verl


def _existing_training_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                pair_id = row["extra_info"]["pair_id"]
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(pair_id, str) and pair_id:
                identifiers.add(pair_id)
    return identifiers


class _RealtimeTrainingWriter:
    """Append accepted edited pages as soon as each source-pool job returns."""

    def __init__(self, output: Path, *, target_count: int = 0) -> None:
        self.dataset_dir = output / "realtime_training"
        self.target_count = target_count
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.parts_dir = self.dataset_dir / "parts"
        self.sft_path = self.dataset_dir / "sft.jsonl"
        self.verl_path = self.dataset_dir / "verl.jsonl"
        self.sft_ids = _existing_training_ids(self.sft_path)
        self.verl_ids = _existing_training_ids(self.verl_path)
        self.sft_handle = self.sft_path.open("a", encoding="utf-8", buffering=1)
        self.verl_handle = self.verl_path.open("a", encoding="utf-8", buffering=1)
        self.added_sft = 0
        self.added_verl = 0
        self.recovered_parts = 0
        self._recover_parts()

    @property
    def complete_ids(self) -> set[str]:
        return self.sft_ids & self.verl_ids

    @staticmethod
    def _read_part(path: Path, expected_id: str) -> dict[str, Any] | None:
        try:
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(lines) != 1:
                return None
            row = json.loads(lines[0])
            if row["extra_info"]["pair_id"] != expected_id:
                return None
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return None
        return row

    @staticmethod
    def _append_row(handle: Any, row: dict[str, Any]) -> None:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()

    def _recover_parts(self) -> None:
        if not self.parts_dir.is_dir():
            return
        sft_parts = {
            path.name.removesuffix(".sft.jsonl"): path
            for path in self.parts_dir.glob("*.sft.jsonl")
        }
        verl_parts = {
            path.name.removesuffix(".verl.jsonl"): path
            for path in self.parts_dir.glob("*.verl.jsonl")
        }
        for pair_id in sorted(sft_parts.keys() & verl_parts.keys()):
            sft_path = sft_parts[pair_id]
            verl_path = verl_parts[pair_id]
            sft = self._read_part(sft_path, pair_id)
            verl = self._read_part(verl_path, pair_id)
            if sft is None or verl is None:
                continue
            if pair_id not in self.sft_ids:
                self._append_row(self.sft_handle, sft)
                self.sft_ids.add(pair_id)
                self.recovered_parts += 1
            if pair_id not in self.verl_ids:
                self._append_row(self.verl_handle, verl)
                self.verl_ids.add(pair_id)
                self.recovered_parts += 1
            sft_path.unlink(missing_ok=True)
            verl_path.unlink(missing_ok=True)
        try:
            self.parts_dir.rmdir()
        except (FileNotFoundError, OSError):
            pass

    def _remove_parts(self, pair_id: str) -> None:
        for suffix in (".sft.jsonl", ".verl.jsonl"):
            (self.parts_dir / f"{pair_id}{suffix}").unlink(missing_ok=True)
        try:
            self.parts_dir.rmdir()
        except (FileNotFoundError, OSError):
            pass

    def add(self, result: WorkerResult) -> bool:
        if (
            result.status != "accepted"
            or result.variant != "confusable_edit"
            or result.image is None
            or not result.changes
        ):
            return False
        sft, verl = _realtime_training_rows(result, self.dataset_dir)
        if result.page_id not in self.sft_ids:
            self._append_row(self.sft_handle, sft)
            self.sft_ids.add(result.page_id)
            self.added_sft += 1
        if result.page_id not in self.verl_ids:
            self._append_row(self.verl_handle, verl)
            self.verl_ids.add(result.page_id)
            self.added_verl += 1
        if result.page_id in self.complete_ids:
            self._remove_parts(result.page_id)
        return True

    def checkpoint(
        self,
        *,
        completed_jobs: int,
        total_jobs: int,
        accepted: int,
        rejected: int,
        started: float,
    ) -> None:
        _atomic_json(
            self.dataset_dir / "progress.json",
            {
                "pipeline_version": PIPELINE_VERSION,
                "mode": "direct_edit",
                "target_count": self.target_count,
                "target_reached": (
                    self.target_count > 0 and accepted >= self.target_count
                ),
                "completed_jobs": completed_jobs,
                "total_jobs": total_jobs,
                "accepted": accepted,
                "rejected": rejected,
                "sft_rows": len(self.sft_ids),
                "verl_rows": len(self.verl_ids),
                "elapsed_seconds": time.monotonic() - started,
            },
        )

    def close(self) -> None:
        self.sft_handle.close()
        self.verl_handle.close()


def _discard_unselected_direct_result(result: WorkerResult, output: Path) -> None:
    """Delete a valid concurrent overrun that was not admitted to the target."""

    page_dir = output / "pages" / result.page_id
    if page_dir.is_dir():
        shutil.rmtree(page_dir)
    parts_dir = output / "realtime_training" / "parts"
    for suffix in (".sft.jsonl", ".verl.jsonl"):
        (parts_dir / f"{result.page_id}{suffix}").unlink(missing_ok=True)
    try:
        parts_dir.rmdir()
    except (FileNotFoundError, OSError):
        pass


def _admit_direct_result(
    result: WorkerResult,
    *,
    writer: _RealtimeTrainingWriter,
    accepted_ids: set[str],
    target_count: int,
    output: Path,
) -> str:
    """Commit one valid result, deduplicate it, or discard target overrun."""

    if result.page_id in accepted_ids:
        writer.add(result)
        return "duplicate"
    if target_count > 0 and len(accepted_ids) >= target_count:
        _discard_unselected_direct_result(result, output)
        return "overrun"
    writer.add(result)
    if result.page_id not in writer.complete_ids:
        raise PipelineError(f"realtime pair was not committed: {result.page_id}")
    accepted_ids.add(result.page_id)
    return "admitted"


def _export(
    output: Path,
    results: Sequence[WorkerResult],
    reports: Sequence[dict[str, Any]],
    started: float,
    config: WorkerConfig,
    *,
    clean_results: Sequence[WorkerResult],
    mutation_config: MutationConfig | None,
    mutation_execution: str = "clean_then_edit",
    input_report: dict[str, Any] | None = None,
) -> None:
    accepted = sorted(
        (result for result in results if result.status == "accepted"),
        key=lambda row: row.page_id,
    )
    rejected = sorted(
        (result for result in results if result.status != "accepted"),
        key=lambda row: row.page_id,
    )
    manifest: list[dict[str, Any]] = []
    sft: list[dict[str, Any]] = []
    sft_v1_compatible: list[dict[str, Any]] = []
    verl: list[dict[str, Any]] = []
    for result in accepted:
        image = _relative_output_path(result.image, output)
        pdf = _relative_output_path(result.pdf, output)
        is_mutated = result.variant == "confusable_edit"
        data_source = (
            "chaos_document_ocr" if is_mutated else "arxiv_canonical_reflow_v4"
        )
        projected_changes = [
            {
                "ocr_ans": change["ocr_ans"],
                "origin_ans": change["origin_ans"],
                "bbox": change["bbox"],
            }
            for change in result.changes
        ]
        row = {
            "pair_id": result.page_id,
            "paper_id": result.paper_id,
            "image": image,
            "pdf": pdf,
            "markdown": result.markdown,
            "layout": result.layout,
            "has_table": result.has_table,
            "block_ids": list(result.block_ids),
            "source_node_ids": list(result.source_node_ids),
            "verifier_recall": result.verifier_recall,
            "verifier_precision": result.verifier_precision,
            "content_fill_ratio": result.content_fill_ratio,
            "column_fill_ratios": list(result.column_fill_ratios),
            "pack_attempts": result.pack_attempts,
            "variant": result.variant,
            "clean_page_id": result.clean_page_id,
            "mutation_count": result.mutation_count,
            "changes": list(result.changes),
            "max_mutation_vertical_shift_points": (
                result.max_mutation_vertical_shift_points
            ),
            "mutation_policy_version": (
                MUTATION_POLICY_VERSION if result.mutation_count else None
            ),
            "ground_truth_source": (
                "latex_ast_confusable_edit" if is_mutated else "latex_ast_only"
            ),
            "pdf_role": "reject_only",
        }
        manifest.append(row)
        sft.append(
            {
                "messages": [
                    {"role": "user", "content": _SFT_PROMPT},
                    {"role": "assistant", "content": result.markdown},
                ],
                "images": [image],
                "data_source": data_source,
                "ability": "document_ocr",
                "extra_info": {
                    "arxiv_id": result.paper_id,
                    "pair_id": result.page_id,
                    "layout": result.layout,
                    "has_table": result.has_table,
                    "content_fill_ratio": result.content_fill_ratio,
                    "mutation_count": result.mutation_count,
                },
            }
        )
        sft_v1_compatible.append(
            {
                "images": [image],
                "conversations": [
                    {"from": "human", "value": _SFT_PROMPT},
                    {"from": "gpt", "value": result.markdown},
                ],
            }
        )
        verl.append(
            {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": "<image>\nPlease transcribe all text in this page image faithfully, exactly as printed (including any typos).",
                    }
                ],
                "images": [image],
                "reward_model": {"style": "rule", "ground_truth": result.markdown},
                "extra_info": {
                    "arxiv_id": result.paper_id,
                    "pair_id": result.page_id,
                    "changes": projected_changes,
                },
                "ability": "document_ocr",
            }
        )
    _atomic_jsonl(output / "manifest.jsonl", manifest)
    _atomic_jsonl(output / "pairs.jsonl", manifest)
    _atomic_jsonl(output / "sft.jsonl", sft)
    _atomic_jsonl(
        output / f"SFT_edited_{len(sft_v1_compatible)}.jsonl",
        sft_v1_compatible,
    )
    _atomic_jsonl(output / "verl.jsonl", verl)
    _atomic_jsonl(output / "rejected_pages.jsonl", [asdict(row) for row in rejected])
    _atomic_jsonl(
        output / "clean_stage_results.jsonl",
        [asdict(row) for row in clean_results],
    )

    total_terminal = len(results)
    total_blocks = sum(report.get("blocks_accepted", 0) for report in reports)
    scheduled_block_ids = {
        block_id for result in results for block_id in result.block_ids
    }
    accepted_block_ids = {
        block_id for result in accepted for block_id in result.block_ids
    }
    accepted_fills = [
        result.content_fill_ratio
        for result in accepted
        if result.content_fill_ratio is not None
    ]
    clean_accepted = sum(row.status == "accepted" for row in clean_results)
    clean_rejected = len(clean_results) - clean_accepted
    mutation_rejected = sum(
        row.variant == "confusable_edit" and row.status != "accepted" for row in results
    )
    mutation_distribution = Counter(row.mutation_count for row in accepted)
    rejection_reasons = Counter(row.reason or "unknown" for row in rejected)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "status": "passed",
        "papers_considered": len(reports),
        "papers_with_candidates": sum(
            report.get("page_candidates", 0) > 0 for report in reports
        ),
        "page_candidates_terminal": total_terminal,
        "pages_accepted": len(accepted),
        "pages_rejected": len(rejected),
        "page_acceptance_rate": len(accepted) / max(1, total_terminal),
        "clean_stage_pages": len(clean_results),
        "clean_stage_pages_accepted": clean_accepted,
        "clean_stage_pages_rejected": clean_rejected,
        "clean_stage_acceptance_rate": clean_accepted / max(1, len(clean_results)),
        "mutation_mode": "confusable" if mutation_config is not None else "off",
        "mutation_execution": mutation_execution,
        "mutation_policy_version": (
            MUTATION_POLICY_VERSION if mutation_config is not None else None
        ),
        "mutation_config": (
            asdict(mutation_config) if mutation_config is not None else None
        ),
        "mutation_candidate_pages": (
            total_terminal if mutation_execution == "direct" else clean_accepted
        ),
        "mutation_pages_accepted": len(accepted),
        "mutation_pages_rejected": mutation_rejected,
        "mutation_acceptance_rate": len(accepted)
        / max(
            1,
            total_terminal if mutation_execution == "direct" else clean_accepted,
        ),
        "mutation_count_distribution": {
            str(key): value for key, value in sorted(mutation_distribution.items())
        },
        "mutation_changes_total": sum(row.mutation_count for row in accepted),
        "final_dataset_variant": (
            "confusable_edited_only" if mutation_config is not None else "clean"
        ),
        "clean_pages_in_final_manifest": (
            sum(row.variant == "clean" for row in accepted)
        ),
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "accepted_table_pages": sum(row.has_table for row in accepted),
        "accepted_two_column_pages": sum(
            row.layout == "two_column" for row in accepted
        ),
        "target_fill_ratio": config.target_fill_ratio,
        "minimum_fill_ratio": config.min_fill_ratio,
        "accepted_fill_ratio_min": min(accepted_fills) if accepted_fills else None,
        "accepted_fill_ratio_median": (
            statistics.median(accepted_fills) if accepted_fills else None
        ),
        "accepted_fill_ratio_mean": (
            statistics.fmean(accepted_fills) if accepted_fills else None
        ),
        "accepted_fill_ratio_max": max(accepted_fills) if accepted_fills else None,
        "compile_pack_attempts": sum(result.pack_attempts for result in results),
        "source_blocks_eligible": total_blocks,
        "source_blocks_scheduled": len(scheduled_block_ids),
        "source_blocks_in_accepted_pages": len(accepted_block_ids),
        "scheduled_source_block_yield": len(accepted_block_ids)
        / max(1, len(scheduled_block_ids)),
        "corpus_source_block_yield": len(accepted_block_ids) / max(1, total_blocks),
        "ground_truth_source": (
            "latex_ast_confusable_edit"
            if mutation_config is not None
            else "latex_ast_only"
        ),
        "pdf_role": "reject_only",
        "legacy_v1_v2_v3_page_placement_imported": False,
        "input": input_report or {"mode": "normalized_papers"},
        "elapsed_seconds": time.monotonic() - started,
        "paper_reports": list(reports),
    }
    _atomic_json(output / "pipeline_report.json", report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--papers-root",
        type=Path,
        help="Normalized papers containing metadata.json plus source/.",
    )
    inputs.add_argument(
        "--crawler-root",
        type=Path,
        help=(
            "Raw crawler output root (results.jsonl plus papers/*/source_archive.bin) "
            "or its papers/ directory."
        ),
    )
    parser.add_argument(
        "--crawler-cache-dir",
        "--crawler-work-dir",
        dest="crawler_cache_dir",
        type=Path,
        default=None,
        help=(
            "Temporary safe-unpack workspace for --crawler-root; each paper "
            "copy is deleted immediately after AST extraction."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Separate disposable compile/cache root. Defaults to a sibling "
            "directory named .<output-name>_work; compiler artifacts are never "
            "stored in the final dataset directory in direct mode."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-limit", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Process every completed archive and every extracted page candidate.",
    )
    parser.add_argument(
        "--target-count",
        "--target-samples",
        dest="target_count",
        type=int,
        default=0,
        help=(
            "Stop after exactly this many accepted edited samples exist in both "
            "realtime SFT and VERL files. Zero means no accepted-sample target."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-stage logs instead of the compact progress display.",
    )
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="Retain diagnostic manifests, rejected rows, and extraction reports.",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(128, os.cpu_count() or 1)),
        help="Process workers shared by unpack, AST extraction, compile, and mutation (1-256).",
    )
    parser.add_argument("--target-weight", type=int, default=5200)
    parser.add_argument("--two-column-rate", type=float, default=0.35)
    parser.add_argument(
        "--max-pack-attempts",
        "--max-rescue-depth",
        dest="max_pack_attempts",
        type=int,
        default=24,
        help="Maximum compile/backoff attempts per terminal dense page.",
    )
    parser.add_argument("--target-fill-ratio", type=float, default=0.82)
    parser.add_argument("--min-fill-ratio", type=float, default=0.70)
    parser.add_argument("--compile-timeout", type=float, default=60.0)
    parser.add_argument("--render-timeout", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--min-page-chars", type=int, default=300)
    parser.add_argument(
        "--mutation-mode",
        choices=("confusable", "off"),
        default="confusable",
        help="Export recompiled confusable edits by default; use off for clean diagnostics.",
    )
    parser.add_argument(
        "--mutation-execution",
        choices=("direct", "clean_then_edit"),
        default="direct",
        help=(
            "Compile edited pages directly by default. clean_then_edit retains "
            "the legacy two-stage diagnostic path."
        ),
    )
    parser.add_argument(
        "--mutation-seed",
        type=int,
        default=None,
        help="Defaults to --seed when omitted.",
    )
    parser.add_argument("--min-mutations-per-page", type=int, default=3)
    parser.add_argument("--max-mutations-per-page", type=int, default=4)
    parser.add_argument("--four-mutation-probability", type=float, default=0.6)
    parser.add_argument(
        "--max-mutation-vertical-shift-points",
        type=float,
        default=1.25,
    )
    parser.add_argument("--latexmk", default=shutil.which("latexmk") or "latexmk")
    parser.add_argument("--pdftoppm", default=shutil.which("pdftoppm") or "pdftoppm")
    parser.add_argument("--pdftotext", default=shutil.which("pdftotext") or "pdftotext")
    parser.add_argument("--pdfinfo", default=shutil.which("pdfinfo") or "pdfinfo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global _VERBOSE_OUTPUT

    args = _parser().parse_args(argv)
    _VERBOSE_OUTPUT = args.verbose
    unbounded_input = args.full_corpus or args.target_count > 0
    paper_limit = 0 if unbounded_input else args.paper_limit
    max_pages = 0 if unbounded_input else args.max_pages
    if not 1 <= args.workers <= 256:
        raise SystemExit("--workers must be between 1 and 256")
    if args.paper_limit < 0:
        raise SystemExit("--paper-limit must be non-negative")
    if args.max_pages < 0:
        raise SystemExit("--max-pages must be non-negative")
    if args.target_count < 0:
        raise SystemExit("--target-count must be non-negative")
    if args.target_count > 0 and (
        args.mutation_mode != "confusable" or args.mutation_execution != "direct"
    ):
        raise SystemExit(
            "--target-count requires the default direct confusable-edit mode"
        )
    if args.target_weight < 200:
        raise SystemExit("--target-weight must be at least 200")
    if not 0.0 <= args.two_column_rate <= 1.0:
        raise SystemExit("--two-column-rate must be in [0, 1]")
    if not 0.0 <= args.min_fill_ratio <= args.target_fill_ratio <= 1.0:
        raise SystemExit("fill ratios must satisfy 0 <= min-fill <= target-fill <= 1")
    if args.max_pack_attempts < 1:
        raise SystemExit("--max-pack-attempts must be positive")
    if not (1 <= args.min_mutations_per_page <= args.max_mutations_per_page):
        raise SystemExit(
            "mutation bounds must satisfy 1 <= min-mutations <= max-mutations"
        )
    if not 0.0 <= args.four_mutation_probability <= 1.0:
        raise SystemExit("--four-mutation-probability must be in [0, 1]")
    if args.max_mutation_vertical_shift_points < 0.0:
        raise SystemExit("--max-mutation-vertical-shift-points must be non-negative")
    started = time.monotonic()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    work_root = (
        args.work_dir.expanduser().resolve()
        if args.work_dir is not None
        else output.parent / f".{output.name}_work"
    )
    work_root.mkdir(parents=True, exist_ok=True)
    mutation_seed = args.seed if args.mutation_seed is None else args.mutation_seed
    crawler_prepare_results: list[dict[str, Any]] = []
    if args.crawler_root is not None:
        input_root = args.crawler_root.expanduser().resolve()
        if not input_root.is_dir():
            raise SystemExit(f"crawler root does not exist: {input_root}")
        archives = _discover_crawler_archives(
            input_root,
            limit=paper_limit,
            seed=args.seed,
        )
        selected_count = len(archives)
        input_mode = "crawler_archives"
    else:
        input_root = args.papers_root.expanduser().resolve()
        if not input_root.is_dir():
            raise SystemExit(f"papers root does not exist: {input_root}")
        papers = _select_papers(input_root, paper_limit, args.seed)
        selected_count = len(papers)
        input_mode = "normalized_papers"
    _emit(
        "start",
        f"version={PIPELINE_VERSION} input_mode={input_mode} papers={selected_count} "
        f"workers={args.workers} target_count={args.target_count or 'all'} "
        f"target_weight={args.target_weight} two_column_rate={args.two_column_rate:.3f} "
        f"target_fill={args.target_fill_ratio:.3f} min_fill={args.min_fill_ratio:.3f} "
        f"mutation_mode={args.mutation_mode} "
        f"mutation_execution={args.mutation_execution} mutation_seed={mutation_seed} "
        f"mutations={args.min_mutations_per_page}-{args.max_mutations_per_page} "
        f"input={input_root} output={output} work_dir={work_root}",
    )
    if args.crawler_root is not None:
        cache_root = (
            args.crawler_cache_dir.expanduser().resolve()
            if args.crawler_cache_dir is not None
            else work_root / "crawler_cache"
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        (
            pages,
            reports,
            extraction_errors,
            crawler_prepare_results,
            input_report,
        ) = _prepare_extract_crawler_parallel(
            archives,
            cache_root=cache_root,
            workers=args.workers,
            target_weight=args.target_weight,
            two_column_rate=args.two_column_rate,
            global_started=started,
        )
        input_report.update(
            {
                "input_root": str(input_root),
                "papers_selected": selected_count,
            }
        )
        if args.debug_artifacts:
            _atomic_jsonl(
                output / "crawler_prepare_results.jsonl",
                crawler_prepare_results,
            )
            _atomic_json(output / "crawler_prepare_report.json", input_report)
        papers_prepared = int(input_report["crawler_archives_prepared"])
    else:
        input_report = {
            "mode": "normalized_papers",
            "input_root": str(input_root),
            "papers_selected": selected_count,
            "workers": args.workers,
        }
        pages, reports, extraction_errors = _extract_papers_parallel(
            papers,
            workers=args.workers,
            target_weight=args.target_weight,
            two_column_rate=args.two_column_rate,
            global_started=started,
        )
        papers_prepared = len(papers)
    pages = _limit_page_candidates(pages, limit=max_pages, seed=args.seed)
    dense_jobs = _dense_jobs_from_pages(pages)
    mutation_config = (
        MutationConfig(
            seed=mutation_seed,
            minimum_per_page=args.min_mutations_per_page,
            maximum_per_page=args.max_mutations_per_page,
            maximum_probability=args.four_mutation_probability,
            max_vertical_shift_points=args.max_mutation_vertical_shift_points,
        )
        if args.mutation_mode == "confusable"
        else None
    )
    direct_edit = mutation_config is not None and args.mutation_execution == "direct"
    if args.debug_artifacts:
        _atomic_jsonl(
            output / "page_candidates.jsonl",
            [
                {
                    "page_id": page.page_id,
                    "paper_id": page.paper_id,
                    "layout": page.layout,
                    "has_table": page.has_table,
                    "block_ids": [block.block_id for block in page.blocks],
                    "source_node_ids": [block.node_id for block in page.blocks],
                }
                for page in pages
            ],
        )
        _atomic_jsonl(
            output / "dense_jobs.jsonl",
            [
                {
                    "job_id": job.page_id,
                    "paper_id": job.paper_id,
                    "source_blocks": len(job.blocks),
                    "block_ids": [block.block_id for block in job.blocks],
                }
                for job in dense_jobs
            ],
        )
    _emit(
        "compile-start",
        f"candidate_pages={len(pages)} max_pages={max_pages} "
        f"dense_jobs={len(dense_jobs)} source_blocks="
        f"{sum(len(job.blocks) for job in dense_jobs)} "
        f"extraction_errors={extraction_errors} "
        f"execution={'direct_edit' if direct_edit else 'clean'}",
    )
    config = WorkerConfig(
        output_dir=str(output),
        latexmk=str(Path(args.latexmk).expanduser().resolve()),
        pdftoppm=str(Path(args.pdftoppm).expanduser().resolve()),
        pdftotext=str(Path(args.pdftotext).expanduser().resolve()),
        pdfinfo=str(Path(args.pdfinfo).expanduser().resolve()),
        compile_timeout=args.compile_timeout,
        render_timeout=args.render_timeout,
        dpi=args.dpi,
        max_pack_attempts=args.max_pack_attempts,
        min_page_chars=args.min_page_chars,
        target_weight=args.target_weight,
        two_column_rate=args.two_column_rate,
        target_fill_ratio=args.target_fill_ratio,
        min_fill_ratio=args.min_fill_ratio,
        work_dir=str(work_root / "compile") if direct_edit else None,
        minimal_output=direct_edit and not args.debug_artifacts,
    )
    stage_results: list[WorkerResult] = []
    stage_rejected = completed = discarded_overrun = cancelled_jobs = 0
    realtime_writer = (
        _RealtimeTrainingWriter(output, target_count=args.target_count)
        if direct_edit
        else None
    )
    accepted_ids = (
        set(realtime_writer.complete_ids) if realtime_writer is not None else set()
    )
    initial_accepted = len(accepted_ids)
    if args.target_count > 0 and initial_accepted > args.target_count:
        if realtime_writer is not None:
            realtime_writer.close()
        raise SystemExit(
            "output already contains "
            f"{initial_accepted} complete samples, which exceeds "
            f"--target-count={args.target_count}; use a new output directory"
        )
    stage_accepted = len(accepted_ids)
    target_progress = _TargetProgress(args.target_count, initial=stage_accepted)
    target_progress.update(
        accepted=stage_accepted,
        rejected=stage_rejected,
        force=True,
    )
    compile_iterator = iter(dense_jobs)
    compile_max_inflight = min(len(dense_jobs), max(1, args.workers * 2))

    def target_open() -> bool:
        return args.target_count == 0 or stage_accepted < args.target_count

    def desired_inflight() -> int:
        if args.target_count == 0:
            return compile_max_inflight
        remaining = max(0, args.target_count - stage_accepted)
        return min(compile_max_inflight, remaining)

    def submit_compile(
        executor: ProcessPoolExecutor,
        pending: dict[Any, CanonicalPage],
    ) -> bool:
        if not target_open():
            return False
        try:
            job = next(compile_iterator)
        except StopIteration:
            return False
        pending[
            executor.submit(
                _compile_with_rescue,
                job,
                config,
                mutation_config=mutation_config if direct_edit else None,
            )
        ] = job
        return True

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending: dict[Any, CanonicalPage] = {}
            while len(pending) < desired_inflight() and submit_compile(
                executor, pending
            ):
                pass
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    target_progress.update(
                        accepted=stage_accepted,
                        rejected=stage_rejected,
                        force=True,
                    )
                    if _VERBOSE_OUTPUT:
                        elapsed = time.monotonic() - started
                        _emit(
                            "compile-progress",
                            f"completed_jobs={completed}/{len(dense_jobs)} "
                            f"pending={len(pending)} accepted={stage_accepted} "
                            f"rejected={stage_rejected} elapsed={elapsed:.1f}s",
                        )
                    if realtime_writer is not None:
                        realtime_writer.checkpoint(
                            completed_jobs=completed,
                            total_jobs=len(dense_jobs),
                            accepted=stage_accepted,
                            rejected=stage_rejected,
                            started=started,
                        )
                    continue
                for future in sorted(done, key=lambda item: pending[item].page_id):
                    page = pending.pop(future)
                    try:
                        rows = future.result()
                    except Exception as exc:  # noqa: BLE001 - isolate worker failures
                        failed = WorkerResult(
                            page_id=page.page_id,
                            paper_id=page.paper_id,
                            status="rejected",
                            reason=f"worker_error:{type(exc).__name__}:{exc}",
                            layout=page.layout,
                            has_table=page.has_table,
                            markdown=page.markdown,
                            verifier_recall=None,
                            verifier_precision=None,
                            pdf=None,
                            image=None,
                            block_ids=tuple(block.block_id for block in page.blocks),
                            source_node_ids=tuple(
                                block.node_id for block in page.blocks
                            ),
                            content_fill_ratio=None,
                            column_fill_ratios=(),
                            page_signature=None,
                            elapsed_seconds=0.0,
                        )
                        if direct_edit and mutation_config is not None:
                            failed = replace(
                                failed,
                                page_id=_mutation_page_id(
                                    page.page_id, mutation_config
                                ),
                                variant="confusable_edit",
                            )
                        _persist_terminal_result(failed, config)
                        rows = (failed,)
                    completed += 1
                    for row in rows:
                        if row.status != "accepted":
                            stage_rejected += 1
                            if args.debug_artifacts or not direct_edit:
                                stage_results.append(row)
                            continue

                        if realtime_writer is None:
                            stage_results.append(row)
                            stage_accepted += 1
                            target_progress.update(
                                accepted=stage_accepted,
                                rejected=stage_rejected,
                            )
                            continue

                        admission = _admit_direct_result(
                            row,
                            writer=realtime_writer,
                            accepted_ids=accepted_ids,
                            target_count=args.target_count,
                            output=output,
                        )
                        if admission == "duplicate":
                            continue
                        if admission == "overrun":
                            discarded_overrun += 1
                            continue
                        stage_accepted = len(accepted_ids)
                        if args.debug_artifacts:
                            stage_results.append(row)
                        target_progress.update(
                            accepted=stage_accepted,
                            rejected=stage_rejected,
                        )

                    if realtime_writer is not None:
                        realtime_writer.checkpoint(
                            completed_jobs=completed,
                            total_jobs=len(dense_jobs),
                            accepted=stage_accepted,
                            rejected=stage_rejected,
                            started=started,
                        )
                    elapsed = time.monotonic() - started
                    rate = completed / max(elapsed, 1e-9)
                    eta = (len(dense_jobs) - completed) / max(rate, 1e-9)
                    _emit(
                        "direct-edit-unit" if direct_edit else "compile-unit",
                        f"completed_jobs={completed}/{len(dense_jobs)} "
                        f"job={page.page_id} terminal_pages={len(rows)} attempts="
                        f"{sum(row.pack_attempts for row in rows)} "
                        f"accepted={stage_accepted} rejected={stage_rejected} "
                        f"realtime_rows="
                        f"{len(realtime_writer.sft_ids) if realtime_writer else 0} "
                        f"rate={rate:.3f}_jobs/s elapsed={elapsed:.1f}s "
                        f"eta={eta:.1f}s",
                    )
                    if not target_open():
                        for queued in list(pending):
                            if queued.cancel():
                                pending.pop(queued)
                                cancelled_jobs += 1
                    while len(pending) < desired_inflight() and submit_compile(
                        executor, pending
                    ):
                        pass
    finally:
        target_progress.finish(
            accepted=stage_accepted,
            rejected=stage_rejected,
        )
        if realtime_writer is not None:
            realtime_writer.checkpoint(
                completed_jobs=completed,
                total_jobs=len(dense_jobs),
                accepted=stage_accepted,
                rejected=stage_rejected,
                started=started,
            )
            realtime_writer.close()
        if direct_edit and config.work_dir is not None:
            compile_work = Path(config.work_dir)
            try:
                if compile_work.exists():
                    shutil.rmtree(compile_work)
                _emit(
                    "compile-cleanup",
                    f"status=deleted path={compile_work}",
                )
            except OSError as error:
                _emit(
                    "compile-cleanup",
                    f"status=warning path={compile_work} "
                    f"error={type(error).__name__}:{error}",
                )
        for empty_directory in (
            work_root / "crawler_cache" / "jobs",
            work_root / "crawler_cache",
            work_root,
        ):
            try:
                empty_directory.rmdir()
            except (FileNotFoundError, OSError):
                pass
    clean_results = [] if direct_edit else stage_results
    clean_accepted = 0 if direct_edit else stage_accepted
    clean_rejected = 0 if direct_edit else stage_rejected

    if direct_edit:
        results = stage_results
    elif args.mutation_mode == "confusable":
        block_lookup = {
            block.block_id: block for job in dense_jobs for block in job.blocks
        }
        final_results: list[WorkerResult] = [
            result for result in clean_results if result.status != "accepted"
        ]
        prepared: list[tuple[WorkerResult, CanonicalPage]] = []
        preparation_rejections = 0
        for clean_result in clean_results:
            if clean_result.status != "accepted":
                continue
            try:
                clean_page = CanonicalPage(
                    page_id=clean_result.page_id,
                    paper_id=clean_result.paper_id,
                    ordinal=0,
                    layout=clean_result.layout,
                    blocks=tuple(
                        block_lookup[block_id] for block_id in clean_result.block_ids
                    ),
                )
                if clean_page.markdown != clean_result.markdown:
                    raise PipelineError("clean_page_source_reconstruction_mismatch")
                if tuple(block.node_id for block in clean_page.blocks) != (
                    clean_result.source_node_ids
                ):
                    raise PipelineError("clean_page_node_order_mismatch")
            except (KeyError, PipelineError) as exc:
                preparation_rejections += 1
                rejection = _mutation_rejection(
                    clean_result,
                    page_id=_mutation_page_id(
                        clean_result.page_id,
                        mutation_config,
                    ),
                    reason=f"mutation_preparation:{type(exc).__name__}:{exc}",
                    started=time.monotonic(),
                )
                _persist_terminal_result(rejection, config)
                final_results.append(rejection)
                continue
            prepared.append((clean_result, clean_page))

        _emit(
            "mutation-start",
            f"clean_terminal_pages={len(clean_results)} "
            f"clean_accepted={clean_accepted} clean_rejected={clean_rejected} "
            f"prepared={len(prepared)} preparation_rejected={preparation_rejections} "
            f"workers={args.workers} policy={MUTATION_POLICY_VERSION}",
        )
        mutation_completed = preparation_rejections
        mutation_accepted = 0
        mutation_rejected = preparation_rejections
        mutation_errors = 0
        mutation_changes = 0
        mutation_started = time.monotonic()
        mutation_iterator = iter(prepared)
        mutation_max_inflight = min(len(prepared), max(1, args.workers * 2))

        def submit_mutation(
            executor: ProcessPoolExecutor,
            pending: dict[Any, tuple[WorkerResult, CanonicalPage]],
        ) -> bool:
            try:
                clean_result, clean_page = next(mutation_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _mutate_and_compile,
                clean_result,
                clean_page,
                config,
                mutation_config,
            )
            pending[future] = (clean_result, clean_page)
            return True

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending_mutations: dict[Any, tuple[WorkerResult, CanonicalPage]] = {}
            while len(pending_mutations) < mutation_max_inflight and submit_mutation(
                executor, pending_mutations
            ):
                pass
            while pending_mutations:
                done, _ = wait(
                    pending_mutations,
                    timeout=30,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    phase_elapsed = time.monotonic() - mutation_started
                    rate = mutation_completed / max(phase_elapsed, 1e-9)
                    eta = (clean_accepted - mutation_completed) / max(rate, 1e-9)
                    _emit(
                        "mutation-progress",
                        f"completed={mutation_completed}/{clean_accepted} "
                        f"percent={mutation_completed / max(1, clean_accepted):.1%} "
                        f"pending={len(pending_mutations)} accepted={mutation_accepted} "
                        f"rejected={mutation_rejected} errors={mutation_errors} "
                        f"changes={mutation_changes} rate={rate:.3f}_pages/s "
                        f"elapsed={phase_elapsed:.1f}s eta={eta:.1f}s",
                    )
                    submit_mutation(executor, pending_mutations)
                    continue
                for future in done:
                    clean_result, clean_page = pending_mutations.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 - isolate one edit
                        mutation_errors += 1
                        result = _mutation_rejection(
                            clean_result,
                            page_id=_mutation_page_id(
                                clean_result.page_id,
                                mutation_config,
                            ),
                            reason=f"mutation_worker_error:{type(exc).__name__}:{exc}",
                            started=time.monotonic(),
                        )
                        _persist_terminal_result(result, config)
                    final_results.append(result)
                    mutation_completed += 1
                    mutation_accepted += result.status == "accepted"
                    mutation_rejected += result.status != "accepted"
                    mutation_changes += (
                        result.mutation_count if result.status == "accepted" else 0
                    )
                    phase_elapsed = time.monotonic() - mutation_started
                    rate = mutation_completed / max(phase_elapsed, 1e-9)
                    eta = (clean_accepted - mutation_completed) / max(rate, 1e-9)
                    _emit(
                        "mutation-unit",
                        f"completed={mutation_completed}/{clean_accepted} "
                        f"percent={mutation_completed / max(1, clean_accepted):.1%} "
                        f"page={clean_page.page_id} status={result.status} "
                        f"mutations={result.mutation_count} accepted={mutation_accepted} "
                        f"rejected={mutation_rejected} errors={mutation_errors} "
                        f"changes={mutation_changes} rate={rate:.3f}_pages/s "
                        f"elapsed={phase_elapsed:.1f}s eta={eta:.1f}s",
                    )
        results = final_results
    else:
        results = clean_results

    accepted = (
        stage_accepted
        if direct_edit
        else sum(result.status == "accepted" for result in results)
    )
    rejected = stage_rejected if direct_edit else len(results) - accepted
    if args.debug_artifacts or not direct_edit:
        _export(
            output,
            results,
            reports,
            started,
            config,
            clean_results=clean_results,
            mutation_config=mutation_config,
            mutation_execution=(
                args.mutation_execution if mutation_config is not None else "off"
            ),
            input_report=input_report,
        )
    elapsed = time.monotonic() - started
    target_reached = args.target_count > 0 and accepted >= args.target_count
    if direct_edit and not args.debug_artifacts:
        _atomic_json(
            output / "run_summary.json",
            {
                "pipeline_version": PIPELINE_VERSION,
                "status": (
                    "target_reached"
                    if target_reached
                    else "source_exhausted"
                    if args.target_count > 0
                    else "completed"
                ),
                "target_count": args.target_count,
                "accepted_count": accepted,
                "accepted_before_run": initial_accepted,
                "accepted_added_this_run": accepted - initial_accepted,
                "rejected_count_this_run": rejected,
                "discarded_concurrent_overrun": discarded_overrun,
                "cancelled_jobs": cancelled_jobs,
                "completed_jobs": completed,
                "available_jobs": len(dense_jobs),
                "clean_pages_generated": 0,
                "final_dataset_variant": "confusable_edited_only",
                "sft": "realtime_training/sft.jsonl",
                "verl": "realtime_training/verl.jsonl",
                "pages": "pages",
                "input_mode": input_mode,
                "papers_selected": selected_count,
                "papers_prepared": papers_prepared,
                "elapsed_seconds": elapsed,
            },
        )
    if not _VERBOSE_OUTPUT:
        _emit(
            "finish",
            f"accepted={accepted} target={args.target_count or 'all'} "
            f"rejected={rejected} elapsed={elapsed:.1f}s output={output}",
        )
        return 0
    _emit(
        "finish",
        f"input_mode={input_mode} papers_selected={selected_count} "
        f"papers_prepared={papers_prepared} initial_candidates={len(pages)} "
        f"dense_jobs={len(dense_jobs)} terminal={len(results)} "
        f"mutation_execution={args.mutation_execution} "
        f"clean_accepted={clean_accepted} clean_rejected={clean_rejected} "
        f"final_accepted={accepted} final_rejected={rejected} "
        f"acceptance={accepted / max(1, len(results)):.2%} elapsed={elapsed:.1f}s",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
