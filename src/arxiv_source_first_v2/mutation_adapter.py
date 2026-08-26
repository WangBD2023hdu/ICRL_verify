"""Strict adapter from source-first v2 pages to the frozen v1 edit exporter.

The source-first v2 builder deliberately stops at clean page/Markdown pairs.
This module validates that output and then delegates *all* edit selection,
source mutation, recompilation, page validation, and SFT/VERL serialization to
``scripts/build_arxiv_confusable_recompile_pilot.py``.  Keeping that execution
path frozen is what makes the v2 edit density and training schema identical to
v1: each accepted page has three or four one-character substitutions, with a
four-edit target selected with probability 0.6.

PDF text remains a verifier input.  Markdown is loaded only from the
source-derived v2 sidecars, and edited Markdown is produced by the frozen
source mutation implementation.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_arxiv_confusable_recompile_pilot as frozen_v1  # noqa: I001

from .contracts import (
    EXPERIMENTAL_CONTRACT,
    EXPERIMENTAL_MARKER_FILENAME,
    EXPERIMENTAL_SCHEMA_VERSION,
    PIPELINE_VERSION,
    STABLE_V10_PIPELINE_VERSION,
    ContractError,
    assert_stable_files,
    validate_page_ledger_file,
    write_experimental_marker,
)


AGGREGATE_REPORT = "validation_report_v2.json"
AGGREGATE_LEDGER = "page_ledger_v2.jsonl"
PER_PAPER_REPORT = "validation_report_v2.json"
PER_PAPER_PASSED = "pages_passed.jsonl"
SOURCE_UNITS = "source_units.jsonl"
SOURCE_CLEAN = "source_clean"
RUN_CONFIG_FILENAME = "mutation_adapter_run_config.json"
ADAPTER_POLICY_VERSION = "source_first_v2_anchor_lattice_mutation_input_v1"
V1_FOUR_MUTATION_TARGET_PROBABILITY = 0.6
HEARTBEAT_SECONDS = 30.0
SAFE_PAPER_ID = re.compile(r"^[A-Za-z0-9._-]+$")


Worker = Callable[[dict[str, Any]], tuple[list[dict[str, Any]], dict[str, Any]]]
Exporter = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class V2MutationInputs:
    """Validated and normalized input passed to the frozen mutation worker."""

    rows: tuple[dict[str, Any], ...]
    paper_metadata: dict[str, dict[str, Any]]
    audit: dict[str, Any]
    aggregate_report: dict[str, Any]


@dataclass(frozen=True)
class MutationRunConfig:
    source_first_root: Path
    output_dir: Path
    server_root: str
    max_papers: int = 0
    paper_ids: tuple[str, ...] = ()
    workers: int = 1
    seed: int = 83
    split_seed: int = 42
    val_fraction: float = 0.05
    dpi: int = 144
    compile_timeout: int = 300
    latexmk: Path = Path(shutil.which("latexmk") or "latexmk")
    pdftoppm: Path = Path(shutil.which("pdftoppm") or "pdftoppm")
    resume: bool = True
    stable_output_roots: tuple[Path, ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_text(seconds: float) -> str:
    value = max(0, round(seconds))
    return f"{value // 3600}h{value % 3600 // 60:02d}m{value % 60:02d}s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ContractError(f"JSONL object required at {path}:{line_number}")
        rows.append(value)
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    frozen_v1.atomic_write_json(path, value)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    return frozen_v1.write_jsonl(path, rows)


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_path_isolation(
    *,
    source_first_root: Path,
    output_dir: Path,
    stable_output_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Fail closed if input, experimental output, or stable output overlap."""

    source = source_first_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if source == output or _contains(source, output) or _contains(output, source):
        raise ContractError(
            "source-first input and edit output must be disjoint: "
            f"input={source} output={output}"
        )
    stable_values: list[str] = []
    for raw in stable_output_roots:
        stable = raw.expanduser().resolve()
        stable_values.append(str(stable))
        if stable == output or _contains(stable, output) or _contains(output, stable):
            raise ContractError(
                "v2 edit output overlaps a declared stable output root: "
                f"stable={stable} output={output}"
            )
    return {
        "status": "passed",
        "source_first_root": str(source),
        "output_dir": str(output),
        "stable_output_roots": stable_values,
    }


def _contains_stable_pipeline_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("pipeline_version") == STABLE_V10_PIPELINE_VERSION:
            return True
        return any(_contains_stable_pipeline_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_stable_pipeline_marker(item) for item in value)
    return False


def validate_v2_marker(root: Path) -> dict[str, Any]:
    """Validate the root marker without recursively rereading every page sidecar."""

    directory = root.expanduser().resolve()
    if not directory.is_dir():
        raise ContractError(f"experimental v2 directory does not exist: {directory}")
    marker_path = directory / EXPERIMENTAL_MARKER_FILENAME
    marker = read_json(marker_path)
    expected = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "experiment": "arxiv_source_first_v2",
        "contract": EXPERIMENTAL_CONTRACT,
        "pipeline_version": PIPELINE_VERSION,
    }
    mismatches = {
        key: {"expected": value, "observed": marker.get(key)}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ContractError(
            "experimental marker does not match v2 contract: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    # Stable runners write their identifying pipeline reports at the output
    # root.  Inspect those bounded control files, while avoiding an O(dataset)
    # recursive marker scan before the progress-aware page audit below.
    for name in (
        "pipeline_report.json",
        "validation_report.json",
        "validation_report_v2.json",
        "batch_state.json",
        "batch_state_v2.json",
    ):
        candidate = directory / name
        if candidate.is_file() and _contains_stable_pipeline_marker(
            read_json(candidate)
        ):
            raise ContractError(f"stable v10 pipeline marker detected: {candidate}")
    return marker


def prepare_output_directory(output_dir: Path, *, resume: bool) -> dict[str, Any]:
    """Create or validate an isolated v2-marked edit output directory."""

    output = output_dir.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ContractError(f"edit output is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ContractError(
            f"edit output is non-empty; pass --resume or choose a new path: {output}"
        )
    if not output.exists() or not any(output.iterdir()):
        marker_path = write_experimental_marker(
            output,
            purpose="source-first-v2-frozen-v1-mutation-export",
            metadata={"adapter_policy_version": ADAPTER_POLICY_VERSION},
        )
        return {
            "status": "created",
            "root": str(output),
            "marker": str(marker_path),
            "marker_payload": read_json(marker_path),
        }
    payload = validate_v2_marker(output)
    return {
        "status": "passed",
        "root": str(output),
        "marker": str(output / EXPERIMENTAL_MARKER_FILENAME),
        "marker_payload": payload,
    }


def _safe_relative(root: Path, value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ContractError(f"{label} path is missing")
    relative = Path(text)
    if relative.is_absolute():
        raise ContractError(f"{label} must be relative to {root}: {relative}")
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError(f"{label} escapes paper root: {relative}") from exc
    if not target.is_file() or target.stat().st_size == 0:
        raise ContractError(f"{label} is missing or empty: {target}")
    return target


def _portable_clean_pdf(paper_root: Path, report: Mapping[str, Any]) -> Path:
    recorded = str(report.get("clean_pdf") or "").strip()
    if not recorded:
        raise ContractError("per-paper report has no clean_pdf")
    path = Path(recorded).expanduser()
    if not path.is_absolute():
        path = paper_root / path
    if path.is_file():
        resolved = path.resolve()
        try:
            resolved.relative_to(paper_root.resolve())
        except ValueError:
            pass
        else:
            return resolved
    matches = sorted(
        candidate.resolve()
        for candidate in (paper_root / "build_clean").glob(Path(recorded).name)
        if candidate.is_file()
    )
    if len(matches) != 1:
        raise ContractError(
            f"cannot uniquely rebase clean_pdf {recorded!r} below {paper_root}"
        )
    return matches[0]


def _split_versioned_paper_id(paper_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+?)(v\d+)", paper_id)
    if match is None:
        return paper_id, ""
    return match.group(1), match.group(2)


def _reference_removal_passed(report: Mapping[str, Any]) -> bool:
    removal = report.get("reference_removal")
    if not isinstance(removal, Mapping) or removal.get("status") != "passed":
        return False
    if removal.get("residuals") not in (None, []):
        return False
    files = removal.get("files") or []
    return isinstance(files, list) and all(
        isinstance(row, Mapping) and row.get("residual_markers") in (None, [])
        for row in files
    )


def _figure_removal_passed(report: Mapping[str, Any]) -> bool:
    removal = report.get("figure_removal")
    return (
        report.get("figure_policy") == "drop_figures"
        and isinstance(removal, Mapping)
        and removal.get("status") == "passed"
    )


def _load_unit_inventory(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        unit_id = str(row.get("unit_id") or "")
        if not unit_id:
            raise ContractError(f"source unit {index} has no unit_id: {path}")
        if unit_id in by_id:
            raise ContractError(f"duplicate source unit id {unit_id!r}: {path}")
        by_id[unit_id] = row
    if not rows:
        raise ContractError(f"empty source unit inventory: {path}")
    return rows, by_id


def map_sidecar_source_units(
    sidecar: Mapping[str, Any],
    units_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], str]:
    """Resolve page provenance to exact source units.

    New sidecars carry ``source_unit_ids`` directly.  Older v2 sidecars carry
    fragment IDs such as ``src-0000042-whole``.  For those, the only accepted
    fallback is the unique longest unit-id prefix on a component boundary.
    """

    if "source_unit_ids" in sidecar:
        raw_ids = sidecar.get("source_unit_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ContractError("explicit source_unit_ids must be a non-empty list")
        ids = [str(value) for value in raw_ids]
        if len(ids) != len(set(ids)):
            raise ContractError("explicit source_unit_ids contain duplicates")
        unknown = sorted(set(ids) - set(units_by_id))
        if unknown:
            raise ContractError(f"explicit source_unit_ids are unknown: {unknown}")
        return ids, "explicit_source_unit_ids"

    fragments = sidecar.get("source_fragment_ids")
    if not isinstance(fragments, list) or not fragments:
        raise ContractError(
            "sidecar has neither explicit source_unit_ids nor fragments"
        )
    resolved: list[str] = []
    for raw_fragment in fragments:
        fragment = str(raw_fragment)
        candidates = [
            unit_id
            for unit_id in units_by_id
            if fragment == unit_id or fragment.startswith(unit_id + "-")
        ]
        if not candidates:
            raise ContractError(f"fragment has no source-unit prefix: {fragment}")
        longest = max(map(len, candidates))
        winners = sorted(unit_id for unit_id in candidates if len(unit_id) == longest)
        if len(winners) != 1:
            raise ContractError(
                f"fragment source-unit prefix is ambiguous: {fragment}: {winners}"
            )
        resolved.append(winners[0])
    return list(dict.fromkeys(resolved)), "unique_longest_unit_id_prefix"


def _source_paragraph_ids(
    *,
    source_root: Path,
    unit_ids: Sequence[str],
    units_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    paragraph_ids: list[str] = []
    source_root_resolved = source_root.resolve()
    for unit_id in unit_ids:
        unit = units_by_id[unit_id]
        paragraph_id = str(unit.get("source_paragraph_id") or "")
        source_file = str(unit.get("source_file") or "")
        source_lines = unit.get("source_lines")
        if not paragraph_id:
            raise ContractError(f"selected source unit has no paragraph id: {unit_id}")
        if not source_file or Path(source_file).is_absolute():
            raise ContractError(
                f"selected source unit has unsafe source_file: {unit_id}"
            )
        source_path = (source_root / source_file).resolve()
        try:
            source_path.relative_to(source_root_resolved)
        except ValueError as exc:
            raise ContractError(
                f"selected source unit escapes source root: {unit_id}"
            ) from exc
        if not source_path.is_file():
            raise ContractError(f"selected source file does not exist: {source_path}")
        if (
            not isinstance(source_lines, list)
            or len(source_lines) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in source_lines
            )
            or source_lines[0] < 1
            or source_lines[1] < source_lines[0]
        ):
            raise ContractError(
                f"selected source unit has invalid source_lines: {unit_id}"
            )
        paragraph_ids.append(paragraph_id)
    values = list(dict.fromkeys(paragraph_ids))
    if not values:
        raise ContractError("page provenance resolved no source paragraph ids")
    return values


def _page_rejection(
    *,
    data_id: str,
    paper_id: str,
    page_number: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "data_id": data_id,
        "paper_id": paper_id,
        "page_number": page_number,
        "reasons": [reason],
    }


def load_v2_mutation_inputs(source_first_root: Path) -> V2MutationInputs:
    """Validate a completed v2 root and normalize exact pages for v1 workers."""

    started = time.monotonic()
    root = source_first_root.expanduser().resolve()
    print(f"[input_start] source_first_root={root}", flush=True)
    validate_v2_marker(root)
    report_path = root / AGGREGATE_REPORT
    ledger_path = root / AGGREGATE_LEDGER
    report = read_json(report_path)
    raw_ledger = read_jsonl(ledger_path)
    validate_page_ledger_file(ledger_path, require_explicit_outcomes=True)
    if report.get("schema_version") != EXPERIMENTAL_SCHEMA_VERSION:
        raise ContractError("aggregate report schema_version mismatch")
    if report.get("contract") != EXPERIMENTAL_CONTRACT:
        raise ContractError("aggregate report contract mismatch")
    if report.get("pipeline_version") != PIPELINE_VERSION:
        raise ContractError("aggregate report pipeline_version mismatch")
    if report.get("status") != "passed":
        raise ContractError("aggregate source-first report is not passed")
    if report.get("pdf_used_for_generation") is not False:
        raise ContractError("aggregate report does not prove PDF-free GT generation")
    if report.get("pdf_used_for_verification") is not True:
        raise ContractError("aggregate report does not prove PDF verification")
    if int(report.get("pages_total", -1)) != len(raw_ledger):
        raise ContractError("aggregate report/ledger pages_total mismatch")

    ledger_by_id: dict[str, dict[str, Any]] = {}
    passed_ledger: dict[str, dict[str, Any]] = {}
    for row in raw_ledger:
        data_id = str(row.get("page_id") or row.get("data_id") or "")
        if not data_id or data_id in ledger_by_id:
            raise ContractError(f"missing or duplicate aggregate page id: {data_id!r}")
        ledger_by_id[data_id] = row
        if row.get("source_first_passed") is True:
            if row.get("source_first_verifier_exact") is not True:
                raise ContractError(f"aggregate passed page is not exact: {data_id}")
            passed_ledger[data_id] = row
    if int(report.get("pages_passed", -1)) != len(passed_ledger):
        raise ContractError("aggregate report/ledger pages_passed mismatch")

    passed_files = sorted((root / "papers").glob(f"*/{PER_PAPER_PASSED}"))
    page_rows: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    page_ids_seen: set[str] = set()
    paper_metadata: dict[str, dict[str, Any]] = {}
    discovery_last_progress = time.monotonic()
    for paper_index, passed_path in enumerate(passed_files, 1):
        paper_root = passed_path.parent
        paper_id = paper_root.name
        if not SAFE_PAPER_ID.fullmatch(paper_id):
            raise ContractError(f"unsafe paper directory name: {paper_id!r}")
        paper_report = read_json(paper_root / PER_PAPER_REPORT)
        if (
            paper_report.get("schema_version") != EXPERIMENTAL_SCHEMA_VERSION
            or paper_report.get("contract") != EXPERIMENTAL_CONTRACT
            or paper_report.get("status") != "passed"
            or paper_report.get("paper_id") != paper_id
        ):
            raise ContractError(f"invalid per-paper report contract: {paper_id}")
        rows = read_jsonl(passed_path)
        if int(paper_report.get("pages_passed", -1)) != len(rows):
            raise ContractError(f"per-paper pages_passed count mismatch: {paper_id}")
        source_root = paper_root / SOURCE_CLEAN
        source_units_path = paper_root / SOURCE_UNITS
        main_relative = Path(str(paper_report.get("main_tex") or ""))
        main_path = (source_root / main_relative).resolve()
        try:
            main_path.relative_to(source_root.resolve())
        except ValueError as exc:
            raise ContractError(f"main TeX escapes cleaned source: {paper_id}") from exc
        if (
            not source_root.is_dir()
            or main_relative.is_absolute()
            or not str(main_relative)
            or not main_path.is_file()
        ):
            raise ContractError(f"invalid cleaned source/main TeX: {paper_id}")
        clean_pdf = _portable_clean_pdf(paper_root, paper_report)
        _, units_by_id = _load_unit_inventory(source_units_path)
        paper_metadata[paper_id] = {
            "paper_id": paper_id,
            "paper_root": str(paper_root.resolve()),
            "source_root": str(source_root.resolve()),
            "source_units_path": str(source_units_path.resolve()),
            "main_tex": main_relative.as_posix(),
            "clean_pdf": str(clean_pdf),
            "compile_engine": str(paper_report.get("compile_engine") or "pdflatex"),
            "reference_removal_passed": _reference_removal_passed(paper_report),
            "figure_removal_passed": _figure_removal_passed(paper_report),
            "report": str((paper_root / PER_PAPER_REPORT).resolve()),
            "units_by_id": units_by_id,
        }
        for sidecar in rows:
            data_id = str(sidecar.get("data_id") or "")
            if not data_id or data_id in page_ids_seen:
                raise ContractError(
                    f"missing or duplicate pages_passed data_id: {data_id!r}"
                )
            page_ids_seen.add(data_id)
            page_rows.append((paper_root, sidecar, passed_ledger.get(data_id) or {}))
        now = time.monotonic()
        if (
            paper_index == len(passed_files)
            or paper_index % 25 == 0
            or now - discovery_last_progress >= HEARTBEAT_SECONDS
        ):
            print(
                f"[input_inventory_progress] papers={paper_index}/{len(passed_files)} "
                f"pages={len(page_rows)}/{len(passed_ledger)} accepted=0 rejected=0 "
                f"errors=0 elapsed={elapsed_text(now - started)} current={paper_id}",
                flush=True,
            )
            discovery_last_progress = now

    if page_ids_seen != set(passed_ledger):
        missing = sorted(set(passed_ledger) - page_ids_seen)
        extra = sorted(page_ids_seen - set(passed_ledger))
        raise ContractError(
            f"aggregate/per-paper passed page sets differ: missing={missing[:5]} extra={extra[:5]}"
        )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    mapping_modes: collections.Counter[str] = collections.Counter()
    reason_counts: collections.Counter[str] = collections.Counter()
    last_progress = started
    for index, (paper_root, sidecar, ledger_row) in enumerate(page_rows, 1):
        data_id = str(sidecar.get("data_id") or "")
        paper_id = str(sidecar.get("paper_id") or "")
        try:
            page_number = int(sidecar.get("page_number"))
        except (TypeError, ValueError, OverflowError):
            page_number = -1
        try:
            if paper_id != paper_root.name:
                raise ContractError("paper_id does not match paper directory")
            if page_number < 1:
                raise ContractError("invalid page_number")
            if (
                str(ledger_row.get("paper_id") or "") != paper_id
                or int(ledger_row.get("page_number", -1)) != page_number
            ):
                raise ContractError("aggregate ledger identity mismatch")
            if (
                sidecar.get("schema_version") != EXPERIMENTAL_SCHEMA_VERSION
                or sidecar.get("contract") != EXPERIMENTAL_CONTRACT
                or sidecar.get("status") != "passed"
                or sidecar.get("source_first_passed") is not True
                or sidecar.get("source_first_verifier_exact") is not True
                or sidecar.get("generation_source") != "latex_source"
                or sidecar.get("pdf_role") != "independent_verifier_only"
            ):
                raise ContractError("page source-first contract is not exact")
            verifier = sidecar.get("verifier")
            if (
                not isinstance(verifier, Mapping)
                or verifier.get("status") != "passed"
                or verifier.get("exact_ordered_character_stream_match") is not True
                or verifier.get("exact_ordered_token_match") is not True
            ):
                raise ContractError("page exact verifier evidence is missing")
            metadata_path = _safe_relative(
                paper_root,
                f"pages/page_{page_number:04d}.json",
                label="page sidecar",
            )
            if read_json(metadata_path) != sidecar:
                raise ContractError("pages_passed row differs from page sidecar")
            markdown_path = _safe_relative(
                paper_root, sidecar.get("markdown"), label="source-derived Markdown"
            )
            image_path = _safe_relative(
                paper_root, sidecar.get("image"), label="clean page image"
            )
            expected_sha = str(sidecar.get("markdown_sha256") or "")
            if not expected_sha or sha256_file(markdown_path) != expected_sha:
                raise ContractError("source-derived Markdown SHA-256 mismatch")
            metadata = paper_metadata[paper_id]
            if not metadata["reference_removal_passed"]:
                raise ContractError("reference removal is not strictly passed")
            if not metadata["figure_removal_passed"]:
                raise ContractError("figure removal is not strictly passed")
            unit_ids, mapping_mode = map_sidecar_source_units(
                sidecar, metadata["units_by_id"]
            )
            paragraph_ids = _source_paragraph_ids(
                source_root=Path(metadata["source_root"]),
                unit_ids=unit_ids,
                units_by_id=metadata["units_by_id"],
            )
            explicit_paragraphs = sidecar.get("source_paragraph_ids")
            if explicit_paragraphs is not None and (
                not isinstance(explicit_paragraphs, list)
                or list(map(str, explicit_paragraphs)) != paragraph_ids
            ):
                raise ContractError("explicit source_paragraph_ids mismatch units")
            arxiv_id, version = _split_versioned_paper_id(paper_id)
            normalized = dict(sidecar)
            normalized.update(
                {
                    "arxiv_id": arxiv_id,
                    "version": version,
                    "markdown": str(markdown_path),
                    "image": str(image_path),
                    "source_pdf": metadata["clean_pdf"],
                    "source_units_path": metadata["source_units_path"],
                    "source_root_override": metadata["source_root"],
                    "main_tex_override": metadata["main_tex"],
                    "source_unit_ids": unit_ids,
                    "source_paragraph_ids": paragraph_ids,
                    "source_probe_ids": list(
                        map(str, sidecar.get("source_fragment_ids") or unit_ids)
                    ),
                    "source_paragraph_integration": {
                        "source_paragraph_ids": paragraph_ids
                    },
                    "source_first_input_policy_version": ADAPTER_POLICY_VERSION,
                }
            )
            accepted.append(normalized)
            mapping_modes[mapping_mode] += 1
        except (ContractError, OSError, ValueError) as exc:
            reason = str(exc)
            rejected.append(
                _page_rejection(
                    data_id=data_id,
                    paper_id=paper_id,
                    page_number=page_number,
                    reason=reason,
                )
            )
            reason_counts[reason] += 1
        now = time.monotonic()
        if (
            index == len(page_rows)
            or index % 250 == 0
            or now - last_progress >= HEARTBEAT_SECONDS
        ):
            elapsed = max(now - started, 1e-9)
            print(
                f"[input_progress] pages={index}/{len(page_rows)} "
                f"accepted={len(accepted)} rejected={len(rejected)} errors=0 "
                f"throughput={index / elapsed:.1f}_pages/s "
                f"elapsed={elapsed_text(elapsed)} current={data_id or '-'}",
                flush=True,
            )
            last_progress = now

    accepted.sort(key=lambda row: (str(row["paper_id"]), int(row["page_number"])))
    audit = {
        "policy_version": ADAPTER_POLICY_VERSION,
        "mode": "source_first_v2_exact_page_sidecars",
        "source_first_root": str(root),
        "experimental_marker": EXPERIMENTAL_MARKER_FILENAME,
        "aggregate_report": AGGREGATE_REPORT,
        "aggregate_ledger": AGGREGATE_LEDGER,
        "scanned_pages": len(page_rows),
        "accepted_pages": len(accepted),
        "rejected_pages": len(rejected),
        "reason_counts": dict(sorted(reason_counts.items())),
        "source_unit_mapping_modes": dict(sorted(mapping_modes.items())),
        "rejections": rejected,
        "pdf_used_for_ground_truth": False,
        "pdf_used_for_verification": True,
    }
    return V2MutationInputs(tuple(accepted), paper_metadata, audit, report)


def _resolve_executable(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_file():
        return expanded.resolve()
    discovered = shutil.which(str(path))
    if discovered:
        return Path(discovered).resolve()
    raise FileNotFoundError(f"{label} executable not found: {path}")


def ensure_resume_configuration(
    output_dir: Path,
    *,
    payload: Mapping[str, Any],
    resume: bool,
) -> Path:
    """Pin every mutation/export-affecting argument before checkpoints exist."""

    path = output_dir / RUN_CONFIG_FILENAME
    expected = dict(payload)
    if path.is_file():
        observed = read_json(path)
        if observed != expected:
            raise ContractError(
                "resume configuration differs from the existing v2 edit run; "
                "choose a new --output-dir or restore the original arguments"
            )
        return path
    unexpected = [
        candidate.name
        for candidate in output_dir.iterdir()
        if candidate.name != EXPERIMENTAL_MARKER_FILENAME
    ]
    if resume and unexpected:
        raise ContractError(
            "cannot safely resume v2 edit output without its pinned run config: "
            + ", ".join(sorted(unexpected)[:10])
        )
    atomic_write_json(path, expected)
    return path


def _select_papers(
    rows_by_paper: Mapping[str, list[dict[str, Any]]],
    *,
    requested: Sequence[str],
    max_papers: int,
) -> list[str]:
    aliases: dict[str, str] = {}
    ambiguous_aliases: set[str] = set()
    for paper_id in rows_by_paper:
        arxiv_id, _ = _split_versioned_paper_id(paper_id)
        aliases[paper_id] = paper_id
        if arxiv_id in ambiguous_aliases:
            continue
        if arxiv_id not in aliases:
            aliases[arxiv_id] = paper_id
        elif aliases[arxiv_id] != paper_id:
            aliases.pop(arxiv_id, None)
            ambiguous_aliases.add(arxiv_id)
    if requested:
        missing = [value for value in requested if value not in aliases]
        if missing:
            raise ContractError(
                f"requested papers not found or ambiguous: {sorted(missing)}"
            )
        selected = list(dict.fromkeys(aliases[value] for value in requested))
    else:
        selected = sorted(rows_by_paper)
    return selected if max_papers == 0 else selected[:max_papers]


def validate_v1_mutation_pairs(pair_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Assert the frozen v1 3/4-edit contract before exporting training rows."""

    issues: list[str] = []
    for row in pair_rows:
        pair_id = str(row.get("pair_id") or "")
        count = row.get("mutation_count")
        changes = row.get("changes")
        if count not in {3, 4}:
            issues.append(f"mutation_count_not_3_or_4:{pair_id}:{count}")
        if not isinstance(changes, list) or len(changes) != count:
            issues.append(f"mutation_changes_count_mismatch:{pair_id}")
            continue
        for change in changes:
            if not isinstance(change, Mapping):
                issues.append(f"mutation_change_not_object:{pair_id}")
                continue
            before = str(change.get("origin_ans") or "")
            after = str(change.get("ocr_ans") or "")
            if (
                len(before) != len(after)
                or sum(a != b for a, b in zip(before, after)) != 1
            ):
                issues.append(f"mutation_not_one_character_substitution:{pair_id}")
    return issues


def run_mutation_export(
    config: MutationRunConfig,
    *,
    worker: Worker = frozen_v1.build_paper_process,
    exporter: Exporter = frozen_v1.export_training,
) -> dict[str, Any]:
    """Run frozen v1 mutation/recompile/export over strict source-first v2 pages."""

    started = time.monotonic()
    if config.max_papers < 0:
        raise ValueError("max_papers must be >= 0 (0 means all)")
    if not 1 <= config.workers <= 256:
        raise ValueError("workers must be between 1 and 256")
    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if config.dpi <= 0 or config.compile_timeout <= 0:
        raise ValueError("dpi and compile_timeout must be positive")
    source_root = config.source_first_root.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    isolation = validate_path_isolation(
        source_first_root=source_root,
        output_dir=output_dir,
        stable_output_roots=config.stable_output_roots,
    )
    print(
        f"[start] phase=input_validation workers={config.workers} seed={config.seed} "
        f"resume={config.resume} input={source_root} output={output_dir}",
        flush=True,
    )
    stable_guard_initial = assert_stable_files(REPO_ROOT)
    marker = prepare_output_directory(output_dir, resume=config.resume)
    latexmk = _resolve_executable(config.latexmk, label="latexmk")
    pdftoppm = _resolve_executable(config.pdftoppm, label="pdftoppm")
    inputs = load_v2_mutation_inputs(source_root)

    rows_by_paper: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in inputs.rows:
        rows_by_paper[str(row["paper_id"])].append(dict(row))
    selected_papers = _select_papers(
        rows_by_paper,
        requested=config.paper_ids,
        max_papers=config.max_papers,
    )
    selected_rows = [row for paper in selected_papers for row in rows_by_paper[paper]]
    resume_payload = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "contract": EXPERIMENTAL_CONTRACT,
        "pipeline_version": PIPELINE_VERSION,
        "adapter_policy_version": ADAPTER_POLICY_VERSION,
        "source_first_root": str(source_root),
        "source_first_validation_report_sha256": sha256_file(
            source_root / AGGREGATE_REPORT
        ),
        "source_first_page_ledger_sha256": sha256_file(source_root / AGGREGATE_LEDGER),
        "selected_papers": selected_papers,
        "server_root": config.server_root,
        "seed": config.seed,
        "split_seed": config.split_seed,
        "val_fraction": config.val_fraction,
        "dpi": config.dpi,
        "compile_timeout": config.compile_timeout,
        "latexmk": str(latexmk),
        "pdftoppm": str(pdftoppm),
        "frozen_v1_mutation_script_sha256": stable_guard_initial["files"][
            "scripts/build_arxiv_confusable_recompile_pilot.py"
        ]["observed_sha256"],
    }
    ensure_resume_configuration(
        output_dir, payload=resume_payload, resume=config.resume
    )
    atomic_write_json(output_dir / "strict_input_filter_audit.json", inputs.audit)
    print(
        f"[start] phase=v2_mutation_export papers={len(selected_papers)} "
        f"pages={len(selected_rows)} workers={config.workers} seed={config.seed} "
        f"v1_mutations=3_or_4 v1_four_target_probability=0.6 "
        f"resume={config.resume} input={source_root} output={output_dir}",
        flush=True,
    )

    pair_rows: list[dict[str, Any]] = []
    paper_results: list[dict[str, Any]] = []
    completed_pages = 0
    rejected_pages = 0
    paper_errors = 0

    def payload_for(paper_id: str) -> dict[str, Any]:
        metadata = inputs.paper_metadata[paper_id]
        arxiv_id, version = _split_versioned_paper_id(paper_id)
        return {
            "paper_id": paper_id,
            "strict_rows": sorted(
                rows_by_paper[paper_id], key=lambda row: int(row["page_number"])
            ),
            "recompile": {
                "paper_id": paper_id,
                "arxiv_id": arxiv_id,
                "version": version,
                "compile": {"engine": metadata["compile_engine"]},
            },
            # These roots are not consulted because every row carries strict
            # absolute overrides; retaining them satisfies the frozen worker
            # interface without creating a second mutation implementation.
            "recompile_root": str(source_root),
            "clean_gt_root": str(source_root),
            "output_dir": str(output_dir),
            "seed": config.seed,
            "dpi": config.dpi,
            "timeout_seconds": config.compile_timeout,
            "latexmk": str(latexmk),
            "pdftoppm": str(pdftoppm),
        }

    def consume_result(
        paper_id: str,
        pairs: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        nonlocal completed_pages, rejected_pages
        pages = len(rows_by_paper[paper_id])
        pair_rows.extend(pairs)
        paper_results.append(result)
        completed_pages += pages
        rejected_pages += max(0, pages - len(pairs))

    if config.workers == 1:
        for paper_index, paper_id in enumerate(selected_papers, 1):
            try:
                pairs, result = worker(payload_for(paper_id))
                consume_result(paper_id, pairs, result)
                print(
                    f"[paper_done] paper={paper_index}/{len(selected_papers)} "
                    f"pages={completed_pages}/{len(selected_rows)} accepted={len(pair_rows)} "
                    f"rejected={rejected_pages} errors={paper_errors} current={paper_id} "
                    f"elapsed={elapsed_text(time.monotonic() - started)}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - persist paper failure
                pages = len(rows_by_paper[paper_id])
                completed_pages += pages
                rejected_pages += pages
                paper_errors += 1
                paper_results.append(
                    {
                        "status": "failed",
                        "paper_id": paper_id,
                        "error": str(exc),
                        "pairs": [],
                    }
                )
                print(
                    f"[paper_error] paper={paper_index}/{len(selected_papers)} "
                    f"pages={completed_pages}/{len(selected_rows)} accepted={len(pair_rows)} "
                    f"rejected={rejected_pages} errors={paper_errors} current={paper_id} "
                    f"error={type(exc).__name__}:{exc}",
                    flush=True,
                )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers
        ) as executor:
            futures = {
                executor.submit(worker, payload_for(paper_id)): paper_id
                for paper_id in selected_papers
            }
            pending = set(futures)
            completed_papers = 0
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=HEARTBEAT_SECONDS,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    throughput = completed_pages / elapsed
                    remaining = max(0, len(selected_rows) - completed_pages)
                    eta = remaining / throughput if throughput else 0.0
                    current = futures[next(iter(pending))] if pending else "-"
                    print(
                        f"[progress] phase=mutation_recompile papers={completed_papers}/"
                        f"{len(selected_papers)} pages={completed_pages}/{len(selected_rows)} "
                        f"pct={100 * completed_pages / max(1, len(selected_rows)):.1f}% "
                        f"throughput={throughput:.2f}_pages/s elapsed={elapsed_text(elapsed)} "
                        f"eta={elapsed_text(eta)} accepted={len(pair_rows)} "
                        f"rejected={rejected_pages} errors={paper_errors} current={current}",
                        flush=True,
                    )
                    continue
                for future in done:
                    paper_id = futures[future]
                    completed_papers += 1
                    try:
                        pairs, result = future.result()
                        consume_result(paper_id, pairs, result)
                    except Exception as exc:  # noqa: BLE001
                        pages = len(rows_by_paper[paper_id])
                        completed_pages += pages
                        rejected_pages += pages
                        paper_errors += 1
                        paper_results.append(
                            {
                                "status": "failed",
                                "paper_id": paper_id,
                                "error": str(exc),
                                "pairs": [],
                            }
                        )
                    print(
                        f"[paper_done] paper={completed_papers}/{len(selected_papers)} "
                        f"pages={completed_pages}/{len(selected_rows)} accepted={len(pair_rows)} "
                        f"rejected={rejected_pages} errors={paper_errors} current={paper_id} "
                        f"elapsed={elapsed_text(time.monotonic() - started)}",
                        flush=True,
                    )

    pair_rows.sort(
        key=lambda row: (str(row.get("paper_id", "")), int(row.get("page_number", 0)))
    )
    paper_results.sort(key=lambda row: str(row.get("paper_id", "")))
    atomic_write_jsonl(output_dir / "pairs.jsonl", pair_rows)
    cleanup = frozen_v1.prune_unreferenced_pair_artifacts(output_dir, pair_rows)
    pair_policy_issues = validate_v1_mutation_pairs(pair_rows)
    exports: dict[str, Any] = {}
    if pair_rows and not pair_policy_issues:
        exports = exporter(
            output_dir=output_dir,
            pair_rows=pair_rows,
            server_root=config.server_root,
            split_seed=config.split_seed,
            val_fraction=config.val_fraction,
        )
    subset_issues = frozen_v1.validate_accepted_subset(output_dir, pair_rows, exports)
    stable_guard_final = assert_stable_files(REPO_ROOT)
    mutation_distribution = dict(
        sorted(
            collections.Counter(int(row["mutation_count"]) for row in pair_rows).items()
        )
    )
    issues = pair_policy_issues + subset_issues
    provenance = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "contract": EXPERIMENTAL_CONTRACT,
        "pipeline_version": PIPELINE_VERSION,
        "adapter_policy_version": ADAPTER_POLICY_VERSION,
        "source_first_root": str(source_root),
        "source_first_validation_report_sha256": sha256_file(
            source_root / AGGREGATE_REPORT
        ),
        "source_first_page_ledger_sha256": sha256_file(source_root / AGGREGATE_LEDGER),
        "frozen_v1_mutation_script": "scripts/build_arxiv_confusable_recompile_pilot.py",
        "frozen_v1_mutation_script_sha256": stable_guard_final["files"][
            "scripts/build_arxiv_confusable_recompile_pilot.py"
        ]["observed_sha256"],
        "mutation_policy_version": frozen_v1.MUTATION_POLICY_VERSION,
        "selection_policy_version": frozen_v1.SELECTION_POLICY_VERSION,
        "target_mutations_per_page": [3, 4],
        "four_mutation_target_probability": V1_FOUR_MUTATION_TARGET_PROBABILITY,
        "training_schema_source": "frozen_v1_export_training",
        "pdf_used_for_ground_truth": False,
        "pdf_used_for_verification": True,
    }
    atomic_write_json(output_dir / "source_first_v2_provenance.json", provenance)
    report = {
        "schema_version": frozen_v1.SCHEMA_VERSION,
        "status": "passed" if pair_rows and not issues else "failed",
        "papers_requested": len(selected_papers),
        "papers_with_strict_pages": len(selected_papers),
        "papers_without_strict_pages": [],
        "strict_pages_before_bibliography_filter": len(selected_rows),
        "bibliography_pages_excluded": 0,
        "bibliography_exclusions": [],
        "bibliography_start_pages": {},
        "strict_pages_considered": len(selected_rows),
        "accepted_pairs": len(pair_rows),
        "rejected_pages": rejected_pages,
        "errors": len(issues),
        "error_reasons": issues,
        "paper_processing_errors": paper_errors,
        "paper_processing_failures": [
            row for row in paper_results if row.get("status") == "failed"
        ],
        "mutation_count_distribution": mutation_distribution,
        "mutation_policy_version": frozen_v1.MUTATION_POLICY_VERSION,
        "selection_policy_version": frozen_v1.SELECTION_POLICY_VERSION,
        "strict_input_filter_policy_version": ADAPTER_POLICY_VERSION,
        "strict_input_filter_audit": "strict_input_filter_audit.json",
        "strict_input_pages_scanned": inputs.audit["scanned_pages"],
        "strict_input_pages_accepted": inputs.audit["accepted_pages"],
        "strict_input_pages_rejected": inputs.audit["rejected_pages"],
        "strict_input_rejection_reason_counts": inputs.audit["reason_counts"],
        "bibliography_policy_version": frozen_v1.BIBLIOGRAPHY_POLICY_VERSION,
        "stale_artifact_cleanup": cleanup,
        "confusable_map": {
            key: list(value) for key, value in frozen_v1.CONFUSABLES.items()
        },
        "digits_allowed": False,
        "length_changing_edits_allowed": False,
        "output_mode": "edited_only",
        "clean_assets_copied": False,
        "server_root": config.server_root,
        "exports": exports,
        "paper_results": paper_results,
        "source_first_v2_provenance": provenance,
        "path_isolation": isolation,
        "experimental_output_marker": marker,
        "stable_guard": {"initial": stable_guard_initial, "final": stable_guard_final},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    print(
        f"[finish] status={report['status']} papers={len(selected_papers)} "
        f"pages={len(selected_rows)} accepted={len(pair_rows)} rejected={rejected_pages} "
        f"errors={len(issues)} paper_errors={paper_errors} "
        f"mutation_distribution={mutation_distribution} "
        f"elapsed={elapsed_text(report['elapsed_seconds'])} output={output_dir}",
        flush=True,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source-first-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--max-papers", type=int, default=0, help="0 means all")
    parser.add_argument("--paper-ids", nargs="*", default=[])
    parser.add_argument(
        "--workers", type=int, default=max(1, min(os.cpu_count() or 1, 32))
    )
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument(
        "--latexmk", type=Path, default=Path(shutil.which("latexmk") or "latexmk")
    )
    parser.add_argument(
        "--pdftoppm", type=Path, default=Path(shutil.which("pdftoppm") or "pdftoppm")
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stable-output-root",
        action="append",
        type=Path,
        default=[],
        help="repeatable path that the experimental output must not overlap",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = MutationRunConfig(
        source_first_root=args.source_first_root,
        output_dir=args.output_dir,
        server_root=args.server_root,
        max_papers=args.max_papers,
        paper_ids=tuple(args.paper_ids),
        workers=args.workers,
        seed=args.seed,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        dpi=args.dpi,
        compile_timeout=args.compile_timeout,
        latexmk=args.latexmk,
        pdftoppm=args.pdftoppm,
        resume=args.resume,
        stable_output_roots=tuple(args.stable_output_root),
    )
    try:
        report = run_mutation_export(config)
    except Exception as exc:  # noqa: BLE001 - CLI must return a durable failure status
        print(f"[finish] status=failed error={type(exc).__name__}:{exc}", flush=True)
        return 2
    return (
        0 if report.get("status") == "passed" and report.get("accepted_pairs", 0) else 2
    )


__all__ = [
    "ADAPTER_POLICY_VERSION",
    "V1_FOUR_MUTATION_TARGET_PROBABILITY",
    "MutationRunConfig",
    "V2MutationInputs",
    "build_arg_parser",
    "ensure_resume_configuration",
    "load_v2_mutation_inputs",
    "main",
    "map_sidecar_source_units",
    "prepare_output_directory",
    "run_mutation_export",
    "validate_path_isolation",
    "validate_v1_mutation_pairs",
    "validate_v2_marker",
]
