#!/usr/bin/env python3
"""Fail-closed verifier for edited data derived from source-first v2 pages.

This program deliberately does not call the stable v1 verifier entry point or
its v1 clean-page loader.  It first establishes an independent chain of trust
from the experimental v2 marker through the aggregate ledger, per-paper
report, passed page sidecar, and clean Markdown hash.  Only then does it check
the edited-only v1-compatible SFT/VERL dataset against those clean pages.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pdfplumber
from PIL import Image, ImageChops

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arxiv_source_first_v2.contracts import (
    EXPERIMENT_NAME,
    EXPERIMENTAL_CONTRACT,
    EXPERIMENTAL_MARKER_FILENAME,
    EXPERIMENTAL_SCHEMA_VERSION,
    PIPELINE_VERSION,
    STABLE_FILE_SHA256,
    STABLE_V10_PIPELINE_VERSION,
    assert_stable_files,
    normalize_layout_bucket,
    validate_page_ledger,
)

VERIFIER_VERSION = "source_first_v2_edited_independent_verifier_v1"
V2_MUTATION_INPUT_POLICY_VERSION = "source_first_v2_anchor_lattice_mutation_input_v1"
V2_PAGE_PROVENANCE = "compiled_source_metadata_span_graph"
V2_VERIFIER_CONTRACT_VERSION = 4
HEARTBEAT_SECONDS = 30.0

STABLE_VERIFIER_PATH = REPO_ROOT / "scripts" / "verify_arxiv_confusable_recompile_pilot.py"
STABLE_MUTATION_PATH = REPO_ROOT / "scripts" / "build_arxiv_confusable_recompile_pilot.py"


class VerificationError(ValueError):
    """Raised when a required source-first or edited-data contract fails."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise VerificationError(code)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid_json:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"json_object_required:{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VerificationError(f"cannot_read_jsonl:{path}:{exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"invalid_jsonl:{path}:{line_number}:{exc}") from exc
        if not isinstance(value, dict):
            raise VerificationError(f"jsonl_object_required:{path}:{line_number}")
        rows.append(value)
    return rows


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def safe_relative(root: Path, raw: Any, label: str) -> Path:
    root = root.resolve()
    require(isinstance(raw, str) and bool(raw), f"{label}_path_missing")
    relative = Path(raw)
    require(not relative.is_absolute(), f"{label}_path_not_relative")
    require(".." not in relative.parts, f"{label}_path_traversal")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise VerificationError(f"{label}_outside_dataset") from exc
    require(resolved.is_file(), f"{label}_missing:{raw}")
    return resolved


def rebased_report_file(root: Path, recorded: Any, expected: Path, label: str) -> Path:
    """Resolve a report path while permitting a whole output tree to be moved."""

    expected = expected.resolve()
    require(expected.is_file(), f"{label}_missing:{expected}")
    if recorded:
        recorded_path = Path(str(recorded))
        if recorded_path.is_file():
            require(recorded_path.resolve() == expected, f"{label}_recorded_path_mismatch")
        else:
            require(recorded_path.name == expected.name, f"{label}_recorded_name_mismatch")
    else:
        raise VerificationError(f"{label}_not_recorded")
    return expected


def load_frozen_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerificationError(f"cannot_import_frozen_helper:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class Heartbeat:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "phase": "startup",
            "completed": 0,
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "errors": 0,
            "current": "-",
        }
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)

    def _run(self) -> None:
        while not self.stop_event.wait(HEARTBEAT_SECONDS):
            with self.lock:
                state = dict(self.state)
            elapsed = max(time.monotonic() - self.started, 1e-9)
            completed = int(state["completed"])
            total = int(state["total"])
            throughput = completed / elapsed
            eta = (total - completed) / throughput if total and throughput else None
            print(
                f"[progress] phase={state['phase']} completed={completed}/{total} "
                f"pct={100*completed/max(1,total):.1f}% throughput={throughput:.3f}/s "
                f"elapsed={elapsed:.1f}s eta={eta if eta is not None else 'unknown'} "
                f"accepted={state['accepted']} rejected={state['rejected']} "
                f"errors={state['errors']} current={state['current']}",
                flush=True,
            )

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def validate_stable_guard(guard: Any, label: str, *, require_final: bool) -> None:
    require(isinstance(guard, Mapping), f"{label}_missing")
    assert isinstance(guard, Mapping)
    require(guard.get("status") == "passed" and guard.get("ok") is True, f"{label}_failed")
    require(
        guard.get("stable_pipeline_version") == STABLE_V10_PIPELINE_VERSION,
        f"{label}_stable_pipeline_version_mismatch",
    )
    files = guard.get("files")
    require(isinstance(files, Mapping), f"{label}_files_missing")
    assert isinstance(files, Mapping)
    require(set(files) == set(STABLE_FILE_SHA256), f"{label}_file_set_mismatch")
    for relative, expected in STABLE_FILE_SHA256.items():
        evidence = files.get(relative)
        require(isinstance(evidence, Mapping), f"{label}_file_evidence_missing:{relative}")
        assert isinstance(evidence, Mapping)
        require(evidence.get("status") == "passed", f"{label}_file_failed:{relative}")
        require(
            evidence.get("expected_sha256") == expected
            and evidence.get("observed_sha256") == expected,
            f"{label}_file_hash_mismatch:{relative}",
        )
    require(guard.get("mismatches") == [], f"{label}_mismatches_not_empty")
    if require_final:
        validate_stable_guard(guard.get("final"), f"{label}_final", require_final=False)


def validate_reference_and_figure_provenance(report: Mapping[str, Any], paper_id: str) -> None:
    reference = report.get("reference_removal")
    require(isinstance(reference, Mapping), f"reference_removal_missing:{paper_id}")
    assert isinstance(reference, Mapping)
    require(reference.get("status") == "passed", f"reference_removal_not_passed:{paper_id}")
    require(reference.get("residuals") == [], f"reference_residuals_present:{paper_id}")
    files = reference.get("files")
    require(isinstance(files, list) and bool(files), f"reference_file_audit_missing:{paper_id}")
    for file_report in files:
        require(isinstance(file_report, Mapping), f"reference_file_audit_invalid:{paper_id}")
        require(
            file_report.get("residual_markers") == [],
            f"reference_file_residuals_present:{paper_id}",
        )
    figure = report.get("figure_removal")
    require(report.get("figure_policy") == "drop_figures", f"figure_policy_not_drop:{paper_id}")
    require(isinstance(figure, Mapping), f"figure_removal_missing:{paper_id}")
    assert isinstance(figure, Mapping)
    require(figure.get("status") == "passed", f"figure_removal_not_passed:{paper_id}")
    require(
        isinstance(figure.get("files"), list) and bool(figure.get("files")),
        f"figure_file_audit_missing:{paper_id}",
    )
    for file_report in figure.get("files", []):
        require(isinstance(file_report, Mapping), f"figure_file_audit_invalid:{paper_id}")
        require(file_report.get("status") == "passed", f"figure_file_audit_failed:{paper_id}")


def validate_exact_page_sidecar(
    *,
    source_root: Path,
    paper_id: str,
    paper_report: Mapping[str, Any],
    aggregate_row: Mapping[str, Any],
    source_unit_ids: Sequence[str],
) -> dict[str, Any]:
    page_number = int(aggregate_row["page_number"])
    paper_root = source_root / "papers" / paper_id
    pages_root = paper_root / "pages"
    sidecar_path = pages_root / f"page_{page_number:04d}.json"
    sidecar = read_json(sidecar_path)
    data_id = str(aggregate_row["page_id"])
    required = {
        "page_schema": sidecar.get("schema_version") == EXPERIMENTAL_SCHEMA_VERSION,
        "page_contract": sidecar.get("contract") == EXPERIMENTAL_CONTRACT,
        "page_id": sidecar.get("data_id") == data_id,
        "paper_id": sidecar.get("paper_id") == paper_id,
        "page_number": sidecar.get("page_number") == page_number,
        "status": sidecar.get("status") == "passed",
        "no_rejection_reasons": sidecar.get("rejection_reasons") == [],
        "generation_source": sidecar.get("generation_source") == "latex_source",
        "page_provenance": sidecar.get("page_provenance") == V2_PAGE_PROVENANCE,
        "pdf_role": sidecar.get("pdf_role") == "independent_verifier_only",
        "clean": sidecar.get("clean") is True,
        "eligible": sidecar.get("eligible_text_page") is True,
        "source_first_passed": sidecar.get("source_first_passed") is True,
        "source_first_exact": sidecar.get("source_first_verifier_exact") is True,
        "not_edited": sidecar.get("edit_accepted") is False,
        "not_final_verified": sidecar.get("verifier_exact") is False,
        "fragment_provenance": isinstance(sidecar.get("source_fragment_ids"), list)
        and bool(sidecar.get("source_fragment_ids"))
        and len(sidecar.get("source_fragment_ids"))
        == len(set(map(str, sidecar.get("source_fragment_ids")))),
        "selected_shadow": bool(sidecar.get("selected_shadow_id")),
        "selected_order": bool(sidecar.get("selected_order")),
        "selected_policy": bool(sidecar.get("selected_serialization_policy")),
        "layout": normalize_layout_bucket(sidecar.get("layout_bucket"))
        == normalize_layout_bucket(aggregate_row.get("layout")),
    }
    failures = sorted(name for name, passed in required.items() if not passed)
    require(not failures, f"page_sidecar_contract_failed:{data_id}:{','.join(failures)}")
    for fragment_id in map(str, sidecar["source_fragment_ids"]):
        matching_units = [
            unit_id
            for unit_id in source_unit_ids
            if fragment_id == unit_id or fragment_id.startswith(unit_id + "-")
        ]
        require(
            len(matching_units) == 1,
            f"source_fragment_unit_provenance_mismatch:{data_id}:{fragment_id}",
        )

    markdown_path = safe_relative(paper_root, sidecar.get("markdown"), "clean_markdown")
    image_path = safe_relative(paper_root, sidecar.get("image"), "clean_image")
    require(markdown_path.parent == pages_root.resolve(), f"clean_markdown_wrong_directory:{data_id}")
    require(image_path.parent == pages_root.resolve(), f"clean_image_wrong_directory:{data_id}")
    markdown = markdown_path.read_text(encoding="utf-8")
    markdown_sha256 = sha256_bytes(markdown.encode("utf-8"))
    require(sidecar.get("markdown_sha256") == markdown_sha256, f"clean_markdown_hash_mismatch:{data_id}")
    require(image_path.stat().st_size > 0, f"clean_image_empty:{data_id}")
    with Image.open(image_path) as image:
        image.load()
        clean_image_size = image.size

    verifier = sidecar.get("verifier")
    require(isinstance(verifier, Mapping), f"clean_verifier_missing:{data_id}")
    assert isinstance(verifier, Mapping)
    verifier_required = {
        "status": verifier.get("status") == "passed",
        "contract": verifier.get("contract_version") == V2_VERIFIER_CONTRACT_VERSION,
        "match_mode": verifier.get("match_mode") == "exact_visible_character_stream",
        "token_exact": verifier.get("exact_ordered_token_match") is True,
        "character_exact": verifier.get("exact_ordered_character_stream_match") is True,
        "token_hash": bool(verifier.get("expected_sha256"))
        and verifier.get("expected_sha256") == verifier.get("observed_sha256"),
        "character_hash": bool(verifier.get("expected_character_stream_sha256"))
        and verifier.get("expected_character_stream_sha256")
        == verifier.get("observed_character_stream_sha256"),
        "token_count": verifier.get("expected_tokens") == verifier.get("observed_tokens"),
        "no_expected_mismatch": verifier.get("first_expected_mismatch") is None,
        "no_observed_mismatch": verifier.get("first_observed_mismatch") is None,
        "no_expected_character_mismatch": verifier.get("first_expected_character_mismatch") is None,
        "no_observed_character_mismatch": verifier.get("first_observed_character_mismatch") is None,
    }
    verifier_failures = sorted(name for name, passed in verifier_required.items() if not passed)
    require(
        not verifier_failures,
        f"clean_verifier_not_exact:{data_id}:{','.join(verifier_failures)}",
    )
    projection = verifier.get("experimental_projection")
    require(isinstance(projection, Mapping), f"clean_projection_missing:{data_id}")
    assert isinstance(projection, Mapping)
    require(
        projection.get("pdf_text_used_for_ground_truth") is False,
        f"pdf_used_for_clean_ground_truth:{data_id}",
    )
    visible_flow = projection.get("source_visible_flow")
    require(isinstance(visible_flow, Mapping), f"source_visible_flow_missing:{data_id}")
    assert isinstance(visible_flow, Mapping)
    require(
        visible_flow.get("pdf_text_used_for_ground_truth") is False
        and visible_flow.get("source_only_projection") is True
        and visible_flow.get("all_or_nothing") is True
        and visible_flow.get("edits_rolled_back") is False,
        f"source_visible_flow_provenance_failed:{data_id}",
    )
    source_markdown_hash = visible_flow.get("source_markdown_sha256")
    projected_markdown_hash = visible_flow.get("projected_markdown_sha256")
    require(
        markdown_sha256 in {source_markdown_hash, projected_markdown_hash},
        f"projection_markdown_hash_mismatch:{data_id}",
    )

    matching_attempts = [
        attempt
        for attempt in sidecar.get("shadow_attempts", [])
        if isinstance(attempt, Mapping)
        and attempt.get("status") == "passed"
        and attempt.get("shadow_id") == sidecar.get("selected_shadow_id")
        and attempt.get("markdown") == markdown
        and attempt.get("selected_order") == sidecar.get("selected_order")
        and attempt.get("selected_serialization_policy")
        == sidecar.get("selected_serialization_policy")
        and attempt.get("fragment_ids") == sidecar.get("source_fragment_ids")
        and attempt.get("verifier") == verifier
    ]
    require(len(matching_attempts) == 1, f"selected_shadow_provenance_mismatch:{data_id}")

    clean_pdf = rebased_report_file(
        source_root,
        paper_report.get("clean_pdf"),
        paper_root / "build_clean" / Path(str(paper_report.get("clean_pdf"))).name,
        f"clean_pdf:{paper_id}",
    )
    return {
        **sidecar,
        "markdown_path": markdown_path,
        "image_path": image_path,
        "clean_image_size": clean_image_size,
        "clean_pdf_path": clean_pdf,
        "paper_report": dict(paper_report),
    }


def validate_source_first_root(
    source_root: Path,
    *,
    heartbeat: Heartbeat | None = None,
) -> dict[str, dict[str, Any]]:
    source_root = source_root.resolve()
    require(source_root.is_dir(), f"source_first_root_missing:{source_root}")
    marker = read_json(source_root / EXPERIMENTAL_MARKER_FILENAME)
    expected_marker = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "contract": EXPERIMENTAL_CONTRACT,
        "pipeline_version": PIPELINE_VERSION,
        "stable_v10_pipeline_version": STABLE_V10_PIPELINE_VERSION,
        "stable_file_sha256": STABLE_FILE_SHA256,
    }
    for key, expected in expected_marker.items():
        require(marker.get(key) == expected, f"experimental_marker_mismatch:{key}")

    report = read_json(source_root / "validation_report_v2.json")
    required_report = {
        "schema": report.get("schema_version") == EXPERIMENTAL_SCHEMA_VERSION,
        "contract": report.get("contract") == EXPERIMENTAL_CONTRACT,
        "pipeline": report.get("pipeline_version") == PIPELINE_VERSION,
        "status": report.get("status") == "passed",
        "pdf_not_generator": report.get("pdf_used_for_generation") is False,
        "pdf_is_verifier": report.get("pdf_used_for_verification") is True,
        "exact_rate": report.get("accepted_exact_verifier_rate") == 1.0,
    }
    report_failures = sorted(name for name, passed in required_report.items() if not passed)
    require(not report_failures, f"aggregate_report_failed:{','.join(report_failures)}")
    validate_stable_guard(report.get("stable_guard"), "aggregate_stable_guard", require_final=True)

    ledger_path = source_root / "page_ledger_v2.jsonl"
    result_path = source_root / "paper_results_v2.jsonl"
    rebased_report_file(source_root, report.get("page_ledger"), ledger_path, "aggregate_page_ledger")
    rebased_report_file(source_root, report.get("paper_results"), result_path, "aggregate_paper_results")
    raw_ledger = read_jsonl(ledger_path)
    try:
        ledger = validate_page_ledger(raw_ledger, require_explicit_outcomes=True)
    except Exception as exc:  # ContractError intentionally normalized here.
        raise VerificationError(f"aggregate_ledger_contract_failed:{exc}") from exc
    require(len(ledger) == int(report.get("pages_total", -1)), "aggregate_pages_total_mismatch")
    eligible = sum(bool(row.get("eligible_text_page")) for row in ledger)
    passed_rows = [row for row in ledger if row.get("source_first_passed") is True]
    require(eligible == int(report.get("eligible_clean_text_pages", -1)), "aggregate_eligible_count_mismatch")
    require(len(passed_rows) == int(report.get("pages_passed", -1)), "aggregate_passed_count_mismatch")
    require(
        len(ledger) - len(passed_rows) == int(report.get("pages_rejected", -1)),
        "aggregate_rejected_count_mismatch",
    )
    require(bool(passed_rows), "no_source_first_passed_pages")
    require(
        all(row.get("source_first_verifier_exact") is True for row in passed_rows),
        "aggregate_passed_page_not_exact",
    )

    paper_results = read_jsonl(result_path)
    require(
        len(paper_results) == int(report.get("papers_selected", -1)),
        "aggregate_paper_result_count_mismatch",
    )
    results_by_paper: dict[str, dict[str, Any]] = {}
    for result in paper_results:
        paper_id = str(result.get("paper_id") or "")
        require(bool(paper_id) and paper_id not in results_by_paper, f"duplicate_or_missing_paper_result:{paper_id}")
        require(result.get("schema_version") == EXPERIMENTAL_SCHEMA_VERSION, f"paper_result_schema:{paper_id}")
        require(result.get("contract") == EXPERIMENTAL_CONTRACT, f"paper_result_contract:{paper_id}")
        require(result.get("pipeline_version") == PIPELINE_VERSION, f"paper_result_pipeline:{paper_id}")
        results_by_paper[paper_id] = result

    raw_by_paper: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in raw_ledger:
        raw_by_paper[str(row["paper_id"])].append(row)
    passed_index: dict[str, dict[str, Any]] = {}
    ordered_papers = sorted({str(row["paper_id"]) for row in passed_rows})
    if heartbeat is not None:
        heartbeat.update(phase="source_first_evidence", completed=0, total=len(ordered_papers))
    for paper_index, paper_id in enumerate(ordered_papers, 1):
        result = results_by_paper.get(paper_id)
        require(result is not None, f"paper_result_missing:{paper_id}")
        assert result is not None
        require(result.get("status") == "success" and result.get("stage") == "complete", f"paper_result_not_success:{paper_id}")
        paper_root = source_root / "papers" / paper_id
        paper_report = read_json(paper_root / "validation_report_v2.json")
        per_required = {
            "schema": paper_report.get("schema_version") == EXPERIMENTAL_SCHEMA_VERSION,
            "contract": paper_report.get("contract") == EXPERIMENTAL_CONTRACT,
            "status": paper_report.get("status") == "passed",
            "paper": paper_report.get("paper_id") == paper_id,
            "pdf_not_generator": paper_report.get("pdf_used_for_generation") is False,
            "pdf_is_verifier": paper_report.get("pdf_used_for_verification") is True,
            "exact_rate": paper_report.get("accepted_exact_verifier_rate") == 1.0,
        }
        per_failures = sorted(name for name, passed in per_required.items() if not passed)
        require(not per_failures, f"paper_report_failed:{paper_id}:{','.join(per_failures)}")
        validate_stable_guard(paper_report.get("stable_guard"), f"paper_stable_guard:{paper_id}", require_final=False)
        validate_reference_and_figure_provenance(paper_report, paper_id)
        source_clean = paper_root / "source_clean"
        require(source_clean.is_dir(), f"source_clean_missing:{paper_id}")
        main_tex = Path(str(paper_report.get("main_tex") or ""))
        require(
            bool(str(main_tex))
            and not main_tex.is_absolute()
            and ".." not in main_tex.parts
            and (source_clean / main_tex).is_file(),
            f"source_clean_main_tex_missing:{paper_id}",
        )
        source_units = read_jsonl(paper_root / "source_units.jsonl")
        source_unit_ids = [str(unit.get("unit_id") or "") for unit in source_units]
        require(
            bool(source_unit_ids)
            and all(source_unit_ids)
            and len(source_unit_ids) == len(set(source_unit_ids)),
            f"source_unit_inventory_invalid:{paper_id}",
        )
        for unit in source_units:
            require(
                unit.get("source_file")
                and isinstance(unit.get("source_lines"), list)
                and isinstance(unit.get("markdown"), str)
                and isinstance(unit.get("raw_latex"), str),
                f"source_unit_provenance_invalid:{paper_id}:{unit.get('unit_id')}",
            )

        per_ledger_raw = read_jsonl(paper_root / "page_ledger_v2.jsonl")
        try:
            per_ledger = validate_page_ledger(per_ledger_raw, require_explicit_outcomes=True)
        except Exception as exc:
            raise VerificationError(f"paper_ledger_contract_failed:{paper_id}:{exc}") from exc
        require(
            canonical(per_ledger_raw) == canonical(raw_by_paper[paper_id]),
            f"aggregate_paper_ledger_mismatch:{paper_id}",
        )
        require(len(per_ledger) == int(paper_report.get("pages_total", -1)), f"paper_pages_total_mismatch:{paper_id}")
        per_passed = [row for row in per_ledger if row.get("source_first_passed") is True]
        require(len(per_passed) == int(paper_report.get("pages_passed", -1)), f"paper_passed_count_mismatch:{paper_id}")
        require(
            len(per_ledger) - len(per_passed) == int(paper_report.get("pages_rejected", -1)),
            f"paper_rejected_count_mismatch:{paper_id}",
        )
        passed_manifest = read_jsonl(paper_root / "pages_passed.jsonl")
        manifest_by_id = {str(row.get("data_id") or ""): row for row in passed_manifest}
        require(
            len(manifest_by_id) == len(passed_manifest) == len(per_passed),
            f"pages_passed_manifest_mismatch:{paper_id}",
        )
        for row in per_passed:
            data_id = str(row["page_id"])
            require(data_id in manifest_by_id, f"passed_manifest_row_missing:{data_id}")
            manifest_row = manifest_by_id[data_id]
            for key in (
                "data_id",
                "paper_id",
                "page_number",
                "status",
                "source_first_passed",
                "source_first_verifier_exact",
            ):
                expected_value = data_id if key == "data_id" else row.get(key)
                require(manifest_row.get(key) == expected_value, f"passed_manifest_field_mismatch:{data_id}:{key}")
            reference = validate_exact_page_sidecar(
                source_root=source_root,
                paper_id=paper_id,
                paper_report=paper_report,
                aggregate_row=row,
                source_unit_ids=source_unit_ids,
            )
            require(data_id not in passed_index, f"duplicate_clean_data_id:{data_id}")
            passed_index[data_id] = reference
        if heartbeat is not None:
            heartbeat.update(completed=paper_index, current=paper_id, accepted=len(passed_index))
        print(
            f"[source-paper-done] paper={paper_id} unit={paper_index}/{len(ordered_papers)} "
            f"passed_pages={len(per_passed)} indexed={len(passed_index)}",
            flush=True,
        )
    return passed_index


def validate_dataset_header(
    root: Path,
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_json(root / "validation_report.json")
    required = {
        "schema": report.get("schema_version") == 2,
        "status": report.get("status") == "passed",
        "output_mode": report.get("output_mode") == "edited_only",
        "no_clean_assets": report.get("clean_assets_copied") is False,
        "accepted_pairs": report.get("accepted_pairs") == len(pairs),
        "mutation_policy": report.get("mutation_policy_version") == "chaos_visual_v2",
        "selection_policy": report.get("selection_policy_version")
        == "page_exact_source_paragraph_v6_rendered_line_spread_current_gt_no_bibliography",
        "input_policy": report.get("strict_input_filter_policy_version")
        == V2_MUTATION_INPUT_POLICY_VERSION,
        "bibliography_policy": report.get("bibliography_policy_version")
        == "exclude_bibliography_tail_v1",
        "digits_disallowed": report.get("digits_allowed") is False,
        "length_changes_disallowed": report.get("length_changing_edits_allowed") is False,
        "server_root": isinstance(report.get("server_root"), str)
        and PurePosixPath(str(report.get("server_root"))).is_absolute(),
    }
    failures = sorted(name for name, passed in required.items() if not passed)
    require(not failures, f"edited_dataset_header_failed:{','.join(failures)}")
    audit = read_json(safe_relative(root, report.get("strict_input_filter_audit"), "strict_input_audit"))
    require(audit.get("policy_version") == V2_MUTATION_INPUT_POLICY_VERSION, "strict_input_audit_policy_mismatch")
    for report_key, audit_key in (
        ("strict_input_pages_scanned", "scanned_pages"),
        ("strict_input_pages_accepted", "accepted_pages"),
        ("strict_input_pages_rejected", "rejected_pages"),
    ):
        require(report.get(report_key) == audit.get(audit_key), f"strict_input_audit_count_mismatch:{report_key}")
    require(int(audit.get("accepted_pages", -1)) >= len(pairs), "strict_input_audit_accepted_below_pairs")
    return report, audit


def validate_final_provenance(
    *,
    root: Path,
    source_first_root: Path,
    report: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    """Bind the edited dataset cryptographically to the requested clean v2 run."""

    root = root.resolve()
    source_first_root = source_first_root.resolve()
    require(
        Path(str(audit.get("source_first_root") or "")).expanduser().resolve()
        == source_first_root,
        "strict_input_audit_source_root_mismatch",
    )
    require(
        audit.get("mode") == "source_first_v2_exact_page_sidecars"
        and audit.get("pdf_used_for_ground_truth") is False
        and audit.get("pdf_used_for_verification") is True,
        "strict_input_audit_provenance_mismatch",
    )

    provenance = read_json(root / "source_first_v2_provenance.json")
    require(
        report.get("source_first_v2_provenance") == provenance,
        "source_first_provenance_report_mismatch",
    )
    expected = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "contract": EXPERIMENTAL_CONTRACT,
        "pipeline_version": PIPELINE_VERSION,
        "adapter_policy_version": V2_MUTATION_INPUT_POLICY_VERSION,
        "source_first_root": str(source_first_root),
        "source_first_validation_report_sha256": sha256_file(
            source_first_root / "validation_report_v2.json"
        ),
        "source_first_page_ledger_sha256": sha256_file(
            source_first_root / "page_ledger_v2.jsonl"
        ),
        "frozen_v1_mutation_script": "scripts/build_arxiv_confusable_recompile_pilot.py",
        "frozen_v1_mutation_script_sha256": STABLE_FILE_SHA256[
            "scripts/build_arxiv_confusable_recompile_pilot.py"
        ],
        "mutation_policy_version": "chaos_visual_v2",
        "selection_policy_version": (
            "page_exact_source_paragraph_v6_rendered_line_spread_current_gt_no_bibliography"
        ),
        "target_mutations_per_page": [3, 4],
        "four_mutation_target_probability": 0.6,
        "training_schema_source": "frozen_v1_export_training",
        "pdf_used_for_ground_truth": False,
        "pdf_used_for_verification": True,
    }
    require(provenance == expected, "source_first_provenance_contract_mismatch")

    output_marker = read_json(root / EXPERIMENTAL_MARKER_FILENAME)
    marker_required = {
        "schema_version": output_marker.get("schema_version") == EXPERIMENTAL_SCHEMA_VERSION,
        "experiment": output_marker.get("experiment") == EXPERIMENT_NAME,
        "contract": output_marker.get("contract") == EXPERIMENTAL_CONTRACT,
        "pipeline": output_marker.get("pipeline_version") == PIPELINE_VERSION,
        "stable_pipeline": output_marker.get("stable_v10_pipeline_version")
        == STABLE_V10_PIPELINE_VERSION,
        "stable_hashes": output_marker.get("stable_file_sha256") == STABLE_FILE_SHA256,
        "purpose": output_marker.get("purpose")
        == "source-first-v2-frozen-v1-mutation-export",
        "adapter_policy": (output_marker.get("metadata") or {}).get(
            "adapter_policy_version"
        )
        == V2_MUTATION_INPUT_POLICY_VERSION,
    }
    marker_failures = sorted(name for name, passed in marker_required.items() if not passed)
    require(not marker_failures, f"edited_output_marker_failed:{','.join(marker_failures)}")
    marker_evidence = report.get("experimental_output_marker")
    require(
        isinstance(marker_evidence, Mapping)
        and marker_evidence.get("status") in {"created", "passed"}
        and Path(str(marker_evidence.get("root") or "")).resolve() == root
        and Path(str(marker_evidence.get("marker") or "")).resolve()
        == (root / EXPERIMENTAL_MARKER_FILENAME).resolve()
        and (
            "marker_payload" not in marker_evidence
            or marker_evidence.get("marker_payload") == output_marker
        ),
        "edited_output_marker_report_mismatch",
    )
    stable_guard = report.get("stable_guard")
    require(isinstance(stable_guard, Mapping), "edited_stable_guard_missing")
    assert isinstance(stable_guard, Mapping)
    validate_stable_guard(
        stable_guard.get("initial"), "edited_stable_guard_initial", require_final=False
    )
    validate_stable_guard(
        stable_guard.get("final"), "edited_stable_guard_final", require_final=False
    )
    isolation = report.get("path_isolation")
    require(
        isinstance(isolation, Mapping)
        and isolation.get("status") == "passed"
        and Path(str(isolation.get("source_first_root") or "")).resolve()
        == source_first_root
        and Path(str(isolation.get("output_dir") or "")).resolve() == root,
        "edited_path_isolation_mismatch",
    )


def pair_pdf_path(root: Path, paper_id: str, report: Mapping[str, Any]) -> Path:
    local = root / "papers" / paper_id / "paper_edited.pdf"
    require(local.is_file(), f"edited_pdf_missing:{paper_id}")
    recorded = report.get("edited_pdf")
    require(bool(recorded), f"edited_pdf_not_recorded:{paper_id}")
    recorded_path = Path(str(recorded))
    if recorded_path.is_file():
        require(recorded_path.resolve() == local.resolve(), f"edited_pdf_recorded_path_mismatch:{paper_id}")
    else:
        require(recorded_path.name == local.name, f"edited_pdf_recorded_name_mismatch:{paper_id}")
    require(report.get("edited_pdf_sha256") == sha256_file(local), f"edited_pdf_hash_mismatch:{paper_id}")
    return local


def verify_pair(
    *,
    root: Path,
    pair: Mapping[str, Any],
    clean: Mapping[str, Any],
    paper_result: Mapping[str, Any],
    stable: ModuleType,
) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    required_pair_keys = {
        "pair_id",
        "data_id",
        "paper_id",
        "arxiv_id",
        "version",
        "page_number",
        "edited_image",
        "edited_markdown",
        "metadata",
        "mutation_count",
        "changes",
        "bibliography_policy_version",
        "strict_input_filter_policy_version",
    }
    check(set(pair) == required_pair_keys, "pair_top_level_shape")
    try:
        metadata_path = safe_relative(root, pair.get("metadata"), "metadata")
        metadata = read_json(metadata_path)
    except Exception as exc:  # noqa: BLE001 - preserve a per-pair diagnosis.
        return [f"metadata_unreadable:{exc}"]
    for key in required_pair_keys - {"metadata"}:
        check(metadata.get(key) == pair.get(key), f"metadata_pair_field_mismatch:{key}")
    expected_metadata_keys = (required_pair_keys - {"metadata"}) | {
        "schema_version",
        "status",
        "mutation_policy_version",
        "selection_policy_version",
        "bibliography_content_present",
        "validation",
    }
    check(set(metadata) == expected_metadata_keys, "metadata_top_level_shape")
    check(metadata.get("schema_version") == 2, "metadata_schema_mismatch")
    check(metadata.get("status") == "passed", "metadata_status_not_passed")
    check(metadata.get("mutation_policy_version") == "chaos_visual_v2", "mutation_policy_version_mismatch")
    check(
        metadata.get("selection_policy_version")
        == "page_exact_source_paragraph_v6_rendered_line_spread_current_gt_no_bibliography",
        "selection_policy_version_mismatch",
    )
    check(
        metadata.get("strict_input_filter_policy_version") == V2_MUTATION_INPUT_POLICY_VERSION
        and pair.get("strict_input_filter_policy_version") == V2_MUTATION_INPUT_POLICY_VERSION,
        "strict_input_policy_mismatch",
    )
    check(
        metadata.get("bibliography_policy_version") == "exclude_bibliography_tail_v1"
        and pair.get("bibliography_policy_version") == "exclude_bibliography_tail_v1",
        "bibliography_policy_mismatch",
    )
    check(metadata.get("bibliography_content_present") is False, "bibliography_content_contract_missing")
    changes = pair.get("changes")
    if not isinstance(changes, list):
        return [*errors, "changes_not_list"]
    check(len(changes) in (3, 4), "mutation_count_not_3_or_4")
    check(pair.get("mutation_count") == len(changes), "pair_mutation_count_mismatch")
    check(metadata.get("mutation_count") == len(changes), "metadata_mutation_count_mismatch")

    clean_markdown = Path(str(clean["markdown_path"])).read_text(encoding="utf-8")
    try:
        edited_markdown_path = safe_relative(root, pair.get("edited_markdown"), "edited_markdown")
        edited_markdown = edited_markdown_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return [*errors, f"edited_markdown_unreadable:{exc}"]
    check(len(clean_markdown) == len(edited_markdown), "markdown_length_changed")
    differences = stable.char_differences(clean_markdown, edited_markdown)
    check(len(differences) == len(changes), "markdown_diff_count_mismatch")
    expected_difference_positions: set[int] = set()
    labels: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for change in changes:
        if not isinstance(change, Mapping):
            errors.append("change_not_object")
            continue
        origin = str(change.get("origin_ans") or "")
        edited = str(change.get("ocr_ans") or "")
        confusion = stable.one_char_confusion(origin, edited)
        check(confusion is not None, f"invalid_one_char_confusion:{origin}->{edited}")
        check(not any(char.isdigit() for char in origin + edited), "digit_mutation")
        if confusion is not None:
            check(change.get("from_char") == confusion[0], f"from_char_mismatch:{origin}")
            check(change.get("to_char") == confusion[1], f"to_char_mismatch:{edited}")
        span = change.get("markdown_span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or isinstance(span[0], bool)
            or isinstance(span[1], bool)
            or not isinstance(span[0], int)
            or not isinstance(span[1], int)
        ):
            errors.append(f"invalid_markdown_span:{origin}")
            continue
        start, end = span
        if not (0 <= start < end <= len(clean_markdown)):
            errors.append(f"markdown_span_out_of_bounds:{origin}")
            continue
        spans.append((start, end))
        check(clean_markdown[start:end] == origin, f"clean_markdown_span_mismatch:{origin}")
        check(edited_markdown[start:end] == edited, f"edited_markdown_span_mismatch:{edited}")
        local_diffs = [start + index for index, (left, right) in enumerate(zip(origin, edited)) if left != right]
        check(len(local_diffs) == 1, f"word_diff_not_one_character:{origin}")
        expected_difference_positions.update(local_diffs)
        labels.append((origin, edited))
        bbox = change.get("bbox")
        check(
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
            and bbox[0] < bbox[2]
            and bbox[1] < bbox[3],
            f"invalid_bbox:{edited}",
        )
        check(isinstance(change.get("source_file"), str) and bool(change.get("source_file")), f"source_file_missing:{origin}")
        check(isinstance(change.get("source_line"), int) and change.get("source_line") >= 1, f"source_line_invalid:{origin}")
        check(isinstance(change.get("source_column"), int) and change.get("source_column") >= 0, f"source_column_invalid:{origin}")
    check(len(spans) == len(set(spans)), "duplicate_markdown_span")
    sorted_spans = sorted(spans)
    check(
        all(left[1] <= right[0] for left, right in pairwise(sorted_spans)),
        "overlapping_markdown_spans",
    )
    check(len(labels) == len(set(labels)), "duplicate_change")
    check(
        {index for index, _left, _right in differences} == expected_difference_positions,
        "markdown_exact_diff_positions_mismatch",
    )

    try:
        clean_image_path = Path(str(clean["image_path"]))
        edited_image_path = safe_relative(root, pair.get("edited_image"), "edited_image")
        with Image.open(clean_image_path) as clean_image, Image.open(edited_image_path) as edited_image:
            clean_image.load()
            edited_image.load()
            check(clean_image.size == edited_image.size, "image_size_changed")
            check(clean_image.size == tuple(clean["clean_image_size"]), "clean_image_size_evidence_changed")
            check(
                ImageChops.difference(clean_image.convert("RGB"), edited_image.convert("RGB")).getbbox()
                is not None,
                "images_pixel_identical",
            )
            image_width, image_height = edited_image.size
    except Exception as exc:  # noqa: BLE001
        return [*errors, f"image_validation_failed:{exc}"]

    try:
        clean_pdf = Path(str(clean["clean_pdf_path"]))
        edited_pdf = pair_pdf_path(root, str(pair["paper_id"]), paper_result)
        page_number = int(pair["page_number"])
        with pdfplumber.open(clean_pdf) as clean_document, pdfplumber.open(edited_pdf) as edited_document:
            check(len(clean_document.pages) == len(edited_document.pages), "pdf_page_count_changed")
            if page_number < 1 or page_number > min(len(clean_document.pages), len(edited_document.pages)):
                return [*errors, "page_number_out_of_pdf_range"]
            clean_page = clean_document.pages[page_number - 1]
            edited_page = edited_document.pages[page_number - 1]
            clean_words = stable.pdf_words(clean_page)
            edited_words = stable.pdf_words(edited_page)
            check(len(clean_words) == len(edited_words), "pdf_word_count_changed")
            if len(clean_words) != len(edited_words):
                return errors
            differing_indices = [
                index
                for index, (left, right) in enumerate(zip(clean_words, edited_words))
                if left["text"] != right["text"]
            ]
            observed = collections.Counter(
                (clean_words[index]["text"], edited_words[index]["text"])
                for index in differing_indices
            )
            declared = collections.Counter(labels)
            check(observed == declared, f"pdf_word_differences_mismatch:{dict(observed)}")
            max_vertical_shift = max(
                (
                    abs(left["top"] - right["top"])
                    for left, right in zip(clean_words, edited_words)
                ),
                default=0.0,
            )
            check(max_vertical_shift <= 1.25, f"pdf_vertical_reflow:{max_vertical_shift:.4f}")
            unused_indices = set(differing_indices)
            for change in changes:
                if not isinstance(change, Mapping):
                    continue
                matching = [
                    index
                    for index in unused_indices
                    if clean_words[index]["text"] == change.get("origin_ans")
                    and edited_words[index]["text"] == change.get("ocr_ans")
                ]
                if len(matching) != 1:
                    errors.append(f"pdf_change_not_unique:{change.get('origin_ans')}")
                    continue
                word_index = matching[0]
                unused_indices.remove(word_index)
                expected_bbox = stable.expected_pixel_bbox(
                    edited_words[word_index],
                    page_width=float(edited_page.width),
                    page_height=float(edited_page.height),
                    image_width=image_width,
                    image_height=image_height,
                )
                check(change.get("bbox") == expected_bbox, f"bbox_mismatch:{change.get('ocr_ans')}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pdf_validation_failed:{exc}")

    validation = metadata.get("validation")
    check(isinstance(validation, Mapping), "metadata_validation_missing")
    if isinstance(validation, Mapping):
        check(validation.get("character_substitutions_only") is True, "metadata_character_substitution_claim")
        check(validation.get("markdown_same_length_after_substitution") is True, "metadata_markdown_length_claim")
        check(validation.get("markdown_character_diff_count") == len(changes), "metadata_markdown_diff_claim")
        check(validation.get("pdf_word_count_unchanged") is True, "metadata_pdf_word_count_claim")
        check(validation.get("pdf_word_sequence_expected") is True, "metadata_pdf_sequence_claim")
        check(validation.get("page_count_unchanged") is True, "metadata_pdf_page_count_claim")
    return errors


def canonical_rows(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    return [canonical(dict(row)) for row in rows]


def verify_exports(
    *,
    root: Path,
    pairs: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    default_prompt: str,
    verl_prompt: str,
) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    pair_by_id = {str(pair["pair_id"]): pair for pair in pairs}
    expected_image_to_pair = {
        str(PurePosixPath(str(report["server_root"])) / str(pair["edited_image"])): pair
        for pair in pairs
    }
    exports = report.get("exports")
    if not isinstance(exports, Mapping):
        return ["exports_missing"]
    try:
        sft_path = safe_relative(root, exports.get("sft"), "sft_export")
        sft_rows = read_jsonl(sft_path)
    except Exception as exc:  # noqa: BLE001
        return [f"sft_export_unreadable:{exc}"]
    check(sft_path.name == f"SFT_edited_{len(pairs)}.jsonl", "sft_filename_mismatch")
    check(len(sft_rows) == len(pairs), "sft_count_mismatch")
    seen_sft: set[str] = set()
    for row in sft_rows:
        check(set(row) == {"images", "conversations"}, "sft_top_level_shape")
        images = row.get("images")
        conversations = row.get("conversations")
        if not isinstance(images, list) or len(images) != 1 or images[0] not in expected_image_to_pair:
            errors.append("sft_image_path_mismatch")
            continue
        image = str(images[0])
        pair = expected_image_to_pair[image]
        edited_markdown = (root / str(pair["edited_markdown"])).read_text(encoding="utf-8")
        check(
            conversations
            == [
                {"from": "human", "value": default_prompt},
                {"from": "gpt", "value": edited_markdown},
            ],
            f"sft_conversation_mismatch:{pair['pair_id']}",
        )
        check(image not in seen_sft, f"sft_duplicate_image:{image}")
        seen_sft.add(image)
    check(seen_sft == set(expected_image_to_pair), "sft_pair_coverage_mismatch")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    seen_verl: set[str] = set()
    expected_top = {"data_source", "prompt", "images", "reward_model", "extra_info", "ability"}
    for split in ("train", "val"):
        try:
            jsonl_path = root / "verl_grpo" / f"{split}.jsonl"
            rows = read_jsonl(jsonl_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"verl_jsonl_unreadable:{split}:{exc}")
            rows = []
        split_rows[split] = rows
        check(len(rows) == int(exports.get(split, -1)), f"verl_report_count_mismatch:{split}")
        for row in rows:
            check(set(row) == expected_top, "verl_top_level_shape")
            check(row.get("data_source") == "chaos_document_ocr", "verl_data_source_mismatch")
            check(row.get("ability") == "document_ocr", "verl_ability_mismatch")
            check(row.get("prompt") == [{"role": "user", "content": verl_prompt}], "verl_prompt_mismatch")
            extra = row.get("extra_info")
            if not isinstance(extra, Mapping):
                errors.append("verl_extra_info_missing")
                continue
            check(set(extra) == {"arxiv_id", "pair_id", "changes"}, "verl_extra_info_shape")
            pair_id = str(extra.get("pair_id") or "")
            pair = pair_by_id.get(pair_id)
            if pair is None:
                errors.append(f"verl_unknown_pair:{pair_id}")
                continue
            check(pair_id not in seen_verl, f"verl_duplicate_pair:{pair_id}")
            seen_verl.add(pair_id)
            edited_markdown = (root / str(pair["edited_markdown"])).read_text(encoding="utf-8")
            expected_image = str(PurePosixPath(str(report["server_root"])) / str(pair["edited_image"]))
            check(row.get("images") == [expected_image], f"verl_image_path_mismatch:{pair_id}")
            check(
                row.get("reward_model") == {"style": "rule", "ground_truth": edited_markdown},
                f"verl_ground_truth_mismatch:{pair_id}",
            )
            expected_changes = [
                {
                    "ocr_ans": change["ocr_ans"],
                    "origin_ans": change["origin_ans"],
                    "bbox": change["bbox"],
                }
                for change in pair["changes"]
            ]
            check(extra.get("arxiv_id") == pair.get("arxiv_id"), f"verl_arxiv_id_mismatch:{pair_id}")
            check(extra.get("changes") == expected_changes, f"verl_changes_mismatch:{pair_id}")
    check(seen_verl == set(pair_by_id), "verl_pair_coverage_mismatch")
    train_papers = {
        str(pair_by_id[str(row.get("extra_info", {}).get("pair_id"))]["paper_id"])
        for row in split_rows.get("train", [])
        if str(row.get("extra_info", {}).get("pair_id")) in pair_by_id
    }
    val_papers = {
        str(pair_by_id[str(row.get("extra_info", {}).get("pair_id"))]["paper_id"])
        for row in split_rows.get("val", [])
        if str(row.get("extra_info", {}).get("pair_id")) in pair_by_id
    }
    check(not (train_papers & val_papers), "verl_paper_split_leakage")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        errors.append("pyarrow_missing")
    else:
        for split in ("train", "val"):
            path = root / "verl_grpo" / f"{split}.parquet"
            if not path.is_file():
                errors.append(f"parquet_missing:{split}")
                continue
            try:
                parquet_rows = pq.read_table(path).to_pylist()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"parquet_unreadable:{split}:{exc}")
                continue
            check(
                canonical_rows(parquet_rows) == canonical_rows(split_rows.get(split, [])),
                f"parquet_content_mismatch:{split}",
            )
    return errors


def verify_dataset(
    *,
    root: Path,
    source_first_root: Path,
    clean_index: Mapping[str, Mapping[str, Any]],
    heartbeat: Heartbeat,
) -> tuple[list[dict[str, str]], dict[str, Any], list[dict[str, Any]]]:
    pairs = read_jsonl(root / "pairs.jsonl")
    require(bool(pairs), "edited_dataset_has_no_pairs")
    report, audit = validate_dataset_header(root, pairs)
    validate_final_provenance(
        root=root,
        source_first_root=source_first_root,
        report=report,
        audit=audit,
    )
    stable = load_frozen_module(STABLE_VERIFIER_PATH, "frozen_v1_edited_verifier")
    stable_mutation = load_frozen_module(STABLE_MUTATION_PATH, "frozen_v1_mutation_export")
    paper_results_value = report.get("paper_results")
    require(isinstance(paper_results_value, list), "paper_results_missing")
    paper_results = {
        str(row.get("paper_id") or ""): row
        for row in paper_results_value
        if isinstance(row, Mapping)
    }
    require(len(paper_results) == len(paper_results_value), "paper_results_duplicate_or_invalid")

    errors: list[dict[str, str]] = []
    seen_pairs: set[str] = set()
    seen_data_ids: set[str] = set()
    failed_pairs = 0
    heartbeat.update(phase="edited_pairs", completed=0, total=len(pairs), accepted=0, rejected=0)
    for index, pair in enumerate(pairs, 1):
        pair_id = str(pair.get("pair_id") or "")
        data_id = str(pair.get("data_id") or "")
        pair_errors: list[str] = []
        if not pair_id or pair_id in seen_pairs:
            pair_errors.append("duplicate_or_missing_pair_id")
        seen_pairs.add(pair_id)
        if not data_id or data_id in seen_data_ids:
            pair_errors.append("duplicate_or_missing_data_id")
        seen_data_ids.add(data_id)
        clean = clean_index.get(data_id)
        if clean is None:
            pair_errors.append("clean_v2_reference_missing")
        elif (
            clean.get("paper_id") != pair.get("paper_id")
            or clean.get("page_number") != pair.get("page_number")
        ):
            pair_errors.append("clean_v2_identity_mismatch")
        paper_result = paper_results.get(str(pair.get("paper_id") or ""))
        if paper_result is None:
            pair_errors.append("edited_paper_result_missing")
        else:
            if paper_result.get("status") != "passed":
                pair_errors.append("edited_paper_result_not_passed")
            if paper_result.get("mutation_policy_version") != "chaos_visual_v2":
                pair_errors.append("edited_paper_mutation_policy_mismatch")
            if paper_result.get("strict_input_filter_policy_version") != V2_MUTATION_INPUT_POLICY_VERSION:
                pair_errors.append("edited_paper_input_policy_mismatch")
            if paper_result.get("source_tree_sha256_before") != paper_result.get("source_tree_sha256_after"):
                pair_errors.append("source_tree_changed_during_mutation")
        if clean is not None and paper_result is not None:
            pair_errors.extend(
                verify_pair(
                    root=root,
                    pair=pair,
                    clean=clean,
                    paper_result=paper_result,
                    stable=stable,
                )
            )
        errors.extend({"pair_id": pair_id or "<missing>", "error": error} for error in pair_errors)
        failed_pairs += bool(pair_errors)
        print(
            f"[pair-done] pair={pair_id or '<missing>'} unit={index}/{len(pairs)} "
            f"status={'passed' if not pair_errors else 'failed'} errors={len(pair_errors)}",
            flush=True,
        )
        heartbeat.update(
            completed=index,
            current=pair_id or "<missing>",
            accepted=index - failed_pairs,
            rejected=failed_pairs,
            errors=len(errors),
        )

    heartbeat.update(phase="exports", completed=0, total=1, current="SFT/VERL/Parquet")
    for error in verify_exports(
        root=root,
        pairs=pairs,
        report=report,
        default_prompt=stable_mutation.DEFAULT_PDF_OCR_PROMPT,
        verl_prompt=stable.VERL_PROMPT,
    ):
        errors.append({"pair_id": "dataset", "error": error})
    heartbeat.update(completed=1, errors=len(errors))

    zero_byte = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.stat().st_size == 0
        and path.name not in {"train.jsonl", "val.jsonl"}
        and not any(part in {"build", "source_edited"} for part in path.relative_to(root).parts)
    ]
    if zero_byte:
        errors.append({"pair_id": "dataset", "error": f"zero_byte_files:{zero_byte[:5]}"})
    clean_assets = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith("_clean.png") or path.name.endswith("_clean.md"))
    ]
    if clean_assets:
        errors.append({"pair_id": "dataset", "error": f"clean_assets_present:{clean_assets[:5]}"})
    return errors, report, pairs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-first-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    root = args.dataset_root.expanduser().resolve()
    source_root = args.source_first_root.expanduser().resolve()
    print(
        f"[start] verifier={VERIFIER_VERSION} dataset={root} "
        f"source_first={source_root} heartbeat={HEARTBEAT_SECONDS}s",
        flush=True,
    )
    heartbeat = Heartbeat()
    heartbeat.start()
    errors: list[dict[str, str]] = []
    pairs: list[dict[str, Any]] = []
    try:
        require(root.is_dir(), f"dataset_root_missing:{root}")
        assert_stable_files(REPO_ROOT)
        clean_index = validate_source_first_root(source_root, heartbeat=heartbeat)
        errors, _report, pairs = verify_dataset(
            root=root,
            source_first_root=source_root,
            clean_index=clean_index,
            heartbeat=heartbeat,
        )
    except Exception as exc:  # noqa: BLE001 - the report is the fail-closed boundary.
        errors.append({"pair_id": "dataset", "error": str(exc)})
    finally:
        heartbeat.close()

    mutation_distribution = collections.Counter()
    for pair in pairs:
        value = pair.get("mutation_count")
        if isinstance(value, int) and not isinstance(value, bool):
            mutation_distribution[value] += 1
    verification = {
        "verifier_version": VERIFIER_VERSION,
        "status": "passed" if not errors else "failed",
        "dataset_root": str(root),
        "source_first_root": str(source_root),
        "pairs": len(pairs),
        "unique_pair_ids": len({str(pair.get('pair_id') or '') for pair in pairs}),
        "mutation_distribution": dict(sorted(mutation_distribution.items())),
        "source_first_input_policy_version": V2_MUTATION_INPUT_POLICY_VERSION,
        "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if root.is_dir():
        atomic_write_json(root / "independent_verifier_report.json", verification)
    print(
        f"[finish] status={verification['status']} pairs={len(pairs)} "
        f"errors={len(errors)} mutation_distribution={dict(mutation_distribution)} "
        f"elapsed={verification['elapsed_seconds']}s",
        flush=True,
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
