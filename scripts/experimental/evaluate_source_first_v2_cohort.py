#!/usr/bin/env python3
"""Evaluate a fixed cohort of isolated source-first v2 paper outputs.

This is deliberately a small, read-only aggregator around the per-paper v2
artifacts.  It does not invoke LaTeX, mutate a paper output, or import the
frozen v10 runner.  The denominator is fixed to ledger rows whose
``eligible_text_page`` field is true, so adding a rejected page to one paper
cannot silently change the cohort denominator policy.

Example::

    PYTHONPATH=src python scripts/experimental/evaluate_source_first_v2_cohort.py \
      --paper-output /path/to/paper_a \
      --paper-output /path/to/paper_b \
      --output-dir /path/to/source_first_v2_fixed_cohort

The output directory is an explicitly marked experimental directory and is
never allowed to overlap a paper input or a stable v10 output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

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
    validate_page_ledger,
)


REPORT_FILENAME = "validation_report_v2.json"
LEDGER_FILENAME = "page_ledger_v2.jsonl"
OUTPUT_REPORT_FILENAME = "fixed_cohort_report_v2.json"
OUTPUT_LEDGER_FILENAME = "page_ledger_v2.jsonl"
HEARTBEAT_SECONDS = 30.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_text(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    return f"{value // 3600}h{value % 3600 // 60:02d}m{value % 60:02d}s"


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file atomically, including an fsync before the rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Stream a JSONL write into a temporary file and replace atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return count


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"report must be a JSON object: {path}")
    return value


def read_jsonl_with_progress(
    path: Path,
    *,
    paper_index: int,
    paper_total: int,
    started: float,
) -> list[dict[str, Any]]:
    """Read a paper ledger while keeping a long-running aggregation visible."""

    rows: list[dict[str, Any]] = []
    bytes_read = 0
    last_progress = time.monotonic()
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read page ledger {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            bytes_read += len(line.encode("utf-8"))
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ContractError(f"JSONL object required at {path}:{line_number}")
            rows.append(value)
            now = time.monotonic()
            if now - last_progress >= HEARTBEAT_SECONDS:
                print(
                    f"[progress] phase=read_ledger paper={paper_index}/{paper_total} "
                    f"current={path.name}:{line_number} records={len(rows)} bytes={bytes_read} "
                    f"elapsed={elapsed_text(now-started)} accepted=0 rejected=0 errors=0",
                    flush=True,
                )
                last_progress = now
    return rows


def _safe_bool(value: Any, *, field: str, paper_id: str, page_id: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(
            f"paper {paper_id} page {page_id} field {field!r} must be boolean: {value!r}"
        )
    return value


def _required_report_text(report: Mapping[str, Any], key: str, path: Path) -> str:
    value = report.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"report {path} has missing/empty {key!r}")
    return value


def _validate_report_stable_guard(
    stable_guard: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    """Check the per-paper provenance guard when a full guard is present.

    Tiny synthetic fixtures may carry only ``status``/``ok``; real builder
    reports carry the four-file hash map and are checked against the frozen
    contract here.  This keeps the evaluator read-only while preventing a
    hand-written report from masquerading as a stable-safe artifact.
    """

    if stable_guard.get("stable_pipeline_version") not in {
        None,
        STABLE_V10_PIPELINE_VERSION,
    }:
        raise ContractError(
            f"report {path} stable_guard has unexpected stable pipeline version: "
            f"{stable_guard.get('stable_pipeline_version')!r}"
        )
    files = stable_guard.get("files")
    if files is None:
        return
    if not isinstance(files, Mapping):
        raise ContractError(f"report {path} stable_guard.files must be an object")
    missing = sorted(set(STABLE_FILE_SHA256) - set(files))
    extra = sorted(set(files) - set(STABLE_FILE_SHA256))
    if missing or extra:
        raise ContractError(
            f"report {path} stable_guard.files does not match frozen chain: "
            f"missing={missing} extra={extra}"
        )
    for relative, expected in STABLE_FILE_SHA256.items():
        entry = files.get(relative)
        if not isinstance(entry, Mapping):
            raise ContractError(
                f"report {path} stable_guard.files[{relative!r}] must be an object"
            )
        if (
            entry.get("status") != "passed"
            or entry.get("expected_sha256") != expected
            or entry.get("observed_sha256") != expected
        ):
            raise ContractError(
                f"report {path} stable_guard.files[{relative!r}] is not a "
                f"passed frozen hash record: {entry!r}"
            )


def _check_optional_report_count(
    report: Mapping[str, Any],
    keys: Sequence[str],
    expected: int,
    *,
    label: str,
    path: Path,
) -> None:
    for key in keys:
        if key not in report:
            continue
        value = report[key]
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ContractError(
                f"report {path} {key!r} disagrees with ledger {label}: "
                f"expected {expected}, observed {value!r}"
            )
        return


def _check_optional_report_rate(
    report: Mapping[str, Any],
    keys: Sequence[str],
    expected: float | None,
    *,
    label: str,
    path: Path,
) -> None:
    for key in keys:
        if key not in report:
            continue
        observed = report[key]
        if expected is None:
            # Current builder reports use 0.0 for an empty accepted set,
            # while some older reports use JSON null.  Both encode the same
            # undefined rate; any non-zero value is still inconsistent.
            if observed is not None and not (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and float(observed) == 0.0
            ):
                raise ContractError(
                    f"report {path} {key!r} disagrees with ledger {label}: "
                    f"expected null, observed {observed!r}"
                )
        elif not isinstance(observed, (int, float)) or isinstance(observed, bool) or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ContractError(
                f"report {path} {key!r} disagrees with ledger {label}: "
                f"expected {expected}, observed {observed!r}"
            )
        return


def validate_paper_output(
    paper_output: Path,
    *,
    paper_index: int,
    paper_total: int,
    started: float,
) -> dict[str, Any]:
    """Validate one per-paper report/ledger and return normalized cohort data."""

    paper_output = Path(paper_output).expanduser().resolve()
    if not paper_output.is_dir():
        raise ContractError(f"paper output is not a directory: {paper_output}")
    report_path = paper_output / REPORT_FILENAME
    ledger_path = paper_output / LEDGER_FILENAME
    if not report_path.is_file():
        raise ContractError(f"missing per-paper report: {report_path}")
    if not ledger_path.is_file():
        raise ContractError(f"missing per-paper ledger: {ledger_path}")

    report = read_json_object(report_path)
    paper_id_value = report.get("paper_id")
    if not isinstance(paper_id_value, str) or not paper_id_value.strip():
        raise ContractError(f"report {report_path} has missing/empty paper_id")
    paper_id = paper_id_value.strip()

    contract = _required_report_text(report, "contract", report_path)
    probe_policy = _required_report_text(report, "probe_policy_version", report_path)
    layout_policy = _required_report_text(report, "layout_policy_version", report_path)
    if report.get("schema_version") != EXPERIMENTAL_SCHEMA_VERSION:
        raise ContractError(
            f"report {report_path} has unsupported schema_version: "
            f"{report.get('schema_version')!r}"
        )
    if (
        "pipeline_version" in report
        and report.get("pipeline_version") != PIPELINE_VERSION
    ):
        raise ContractError(
            f"report {report_path} has unsupported pipeline_version: "
            f"{report.get('pipeline_version')!r}"
        )
    if contract != EXPERIMENTAL_CONTRACT:
        raise ContractError(
            f"report {report_path} is not the v2 contract: {contract!r}"
        )

    stable_guard = report.get("stable_guard")
    if not isinstance(stable_guard, Mapping):
        raise ContractError(f"report {report_path} has no stable_guard object")
    if stable_guard.get("status") != "passed" or stable_guard.get("ok") is not True:
        raise ContractError(
            f"report {report_path} stable_guard is not passed: {stable_guard!r}"
        )
    _validate_report_stable_guard(stable_guard, path=report_path)

    raw_rows = read_jsonl_with_progress(
        ledger_path,
        paper_index=paper_index,
        paper_total=paper_total,
        started=started,
    )
    rows = validate_page_ledger(raw_rows, require_explicit_outcomes=True)

    pages_total = report.get("pages_total")
    if isinstance(pages_total, bool) or not isinstance(pages_total, int):
        raise ContractError(f"report {report_path} has invalid pages_total: {pages_total!r}")
    if pages_total != len(rows):
        raise ContractError(
            f"report {report_path} pages_total={pages_total} != ledger rows={len(rows)}"
        )

    seen_page_ids: set[str] = set()
    eligible = passed = complex_passed = two_column_passed = exact_passed = 0
    for index, row in enumerate(rows):
        page_id = str(row["page_id"])
        if page_id in seen_page_ids:
            raise ContractError(f"duplicate page_id in {ledger_path}: {page_id}")
        seen_page_ids.add(page_id)
        row_paper_id = str(row.get("paper_id", ""))
        if row_paper_id != paper_id:
            raise ContractError(
                f"ledger {ledger_path} row {index} paper_id={row_paper_id!r} "
                f"does not match report paper_id={paper_id!r}"
            )
        eligible_page = _safe_bool(
            row.get("eligible_text_page"),
            field="eligible_text_page",
            paper_id=paper_id,
            page_id=page_id,
        )
        source_passed = _safe_bool(
            row.get("source_first_passed"),
            field="source_first_passed",
            paper_id=paper_id,
            page_id=page_id,
        )
        source_exact = _safe_bool(
            row.get("source_first_verifier_exact"),
            field="source_first_verifier_exact",
            paper_id=paper_id,
            page_id=page_id,
        )
        if source_passed and not eligible_page:
            raise ContractError(
                f"paper {paper_id} page {page_id}: source_first_passed requires "
                "eligible_text_page"
            )
        if source_exact and not source_passed:
            raise ContractError(
                f"paper {paper_id} page {page_id}: source_first_verifier_exact "
                "requires source_first_passed"
            )
        if eligible_page:
            eligible += 1
        if source_passed:
            passed += 1
            layout = normalize_layout_bucket(row.get("layout", row.get("layout_bucket")))
            if layout in COMPLEX_LAYOUT_BUCKETS:
                complex_passed += 1
            if layout == "two_column":
                two_column_passed += 1
            if source_exact:
                exact_passed += 1

    _check_optional_report_count(
        report,
        ("eligible_clean_text_pages", "eligible_text_pages"),
        eligible,
        label="eligible_text_page count",
        path=report_path,
    )
    _check_optional_report_count(
        report,
        ("pages_passed", "source_first_passed_pages"),
        passed,
        label="source_first_passed count",
        path=report_path,
    )
    _check_optional_report_count(
        report,
        ("accepted_complex_layout_pages", "accepted_complex_pages"),
        complex_passed,
        label="accepted complex count",
        path=report_path,
    )
    _check_optional_report_count(
        report,
        ("accepted_two_column_pages", "accepted_two_column_layout_pages"),
        two_column_passed,
        label="accepted two-column count",
        path=report_path,
    )
    exact_rate = exact_passed / passed if passed else None
    _check_optional_report_rate(
        report,
        ("accepted_exact_verifier_rate", "accepted_source_first_verifier_exact_rate"),
        exact_rate,
        label="source-first exact rate",
        path=report_path,
    )
    return {
        "paper_id": paper_id,
        "paper_output": str(paper_output),
        "report_path": str(report_path),
        "ledger_path": str(ledger_path),
        "report": report,
        "rows": rows,
        "contract": contract,
        "probe_policy_version": probe_policy,
        "layout_policy_version": layout_policy,
        "stable_guard": dict(stable_guard),
        "pages_total": len(rows),
        "eligible_text_pages": eligible,
        "source_first_passed_pages": passed,
        "accepted_complex_pages": complex_passed,
        "accepted_two_column_pages": two_column_passed,
        "accepted_two_column_layout_pages": two_column_passed,
        "accepted_source_first_verifier_exact_pages": exact_passed,
        "accepted_source_first_verifier_exact_rate": exact_rate,
        "ledger_bytes": ledger_path.stat().st_size,
    }


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _json_contains_stable_pipeline(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("pipeline_version") == STABLE_V10_PIPELINE_VERSION:
            return True
        return any(_json_contains_stable_pipeline(item) for item in value.values())
    if isinstance(value, list):
        return any(_json_contains_stable_pipeline(item) for item in value)
    return False


def _stable_marker_in_ancestor(output_dir: Path) -> Path | None:
    """Find a direct stable report above an evaluator output path."""

    for ancestor in output_dir.parents:
        try:
            candidates = ancestor.glob("*.json")
        except OSError:
            continue
        for candidate in candidates:
            try:
                if candidate.stat().st_size > 8 * 1024 * 1024:
                    continue
                value = json.loads(
                    candidate.read_text(encoding="utf-8", errors="replace")
                )
            except (OSError, json.JSONDecodeError):
                continue
            if _json_contains_stable_pipeline(value):
                return candidate
    return None


def validate_output_isolation(
    output_dir: Path,
    paper_outputs: Sequence[Path],
    stable_output_roots: Sequence[Path],
) -> Path:
    """Validate the evaluator's write boundary before creating any output."""

    output = output_dir.expanduser().resolve()
    for paper_output in paper_outputs:
        paper = paper_output.expanduser().resolve()
        if _path_contains(paper, output) or _path_contains(output, paper):
            raise ContractError(
                "fixed cohort output must not overlap a paper output: "
                f"paper={paper} output={output}"
            )
    for raw in stable_output_roots:
        stable_root = raw.expanduser().resolve()
        if _path_contains(stable_root, output) or _path_contains(output, stable_root):
            raise ContractError(
                "fixed cohort output overlaps declared stable output: "
                f"stable={stable_root} output={output}"
            )
    marker = _stable_marker_in_ancestor(output)
    if marker is not None:
        raise ContractError(
            "fixed cohort output is nested below a stable v10 output marker: "
            f"{marker}"
        )
    return output


def print_cohort_progress(
    *,
    completed: int,
    total: int,
    current: str,
    units: Sequence[Mapping[str, Any]],
    started: float,
) -> None:
    """Emit an aggregate heartbeat with the fields required for long runs."""

    elapsed = max(time.monotonic() - started, 1e-9)
    throughput = completed / elapsed
    eta = (total - completed) / throughput if throughput else math.inf
    rows = sum(int(unit.get("pages_total", 0)) for unit in units)
    eligible = sum(int(unit.get("eligible_text_pages", 0)) for unit in units)
    passed = sum(int(unit.get("source_first_passed_pages", 0)) for unit in units)
    complex_pages = sum(int(unit.get("accepted_complex_pages", 0)) for unit in units)
    two_column = sum(int(unit.get("accepted_two_column_pages", 0)) for unit in units)
    ledger_bytes = sum(int(unit.get("ledger_bytes", 0)) for unit in units)
    rejected = max(eligible - passed, 0)
    print(
        f"[progress] phase=fixed_cohort papers={completed}/{total} "
        f"pct={100*completed/total if total else 100:.1f}% current={current} "
        f"records={rows} bytes={ledger_bytes} eligible={eligible} "
        f"accepted={passed} rejected={rejected} errors=0 complex={complex_pages} "
        f"two_column={two_column} throughput={throughput:.3f}_papers/s "
        f"elapsed={elapsed_text(elapsed)} "
        f"eta={'unknown' if not math.isfinite(eta) else elapsed_text(eta)}",
        flush=True,
    )


def aggregate_fixed_cohort(
    paper_outputs: Sequence[Path],
    output_dir: Path,
    stable_output_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Validate and atomically aggregate a fixed cohort."""

    started = time.monotonic()
    if not paper_outputs:
        raise ContractError("at least one --paper-output is required")
    resolved_inputs = [Path(path).expanduser().resolve() for path in paper_outputs]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ContractError("duplicate --paper-output paths are not allowed")
    resolved_output = validate_output_isolation(
        Path(output_dir), resolved_inputs, stable_output_roots
    )
    stable_guard_current = assert_stable_files(REPO_ROOT)

    print(
        f"[start] phase=fixed_cohort papers={len(resolved_inputs)} "
        f"output={resolved_output} denominator=eligible_text_page "
        f"contract={EXPERIMENTAL_CONTRACT}",
        flush=True,
    )

    units: list[dict[str, Any]] = []
    seen_papers: set[str] = set()
    all_rows: list[dict[str, Any]] = []
    seen_page_ids: set[str] = set()
    contract_key: tuple[str, str, str] | None = None
    stable_guards: list[dict[str, Any]] = []

    for index, paper_output in enumerate(resolved_inputs, 1):
        unit = validate_paper_output(
            paper_output,
            paper_index=index,
            paper_total=len(resolved_inputs),
            started=started,
        )
        paper_id = unit["paper_id"]
        if paper_id in seen_papers:
            raise ContractError(f"paper_id is not unique in fixed cohort: {paper_id}")
        seen_papers.add(paper_id)
        current_contract = (
            unit["contract"],
            unit["probe_policy_version"],
            unit["layout_policy_version"],
        )
        if contract_key is None:
            contract_key = current_contract
        elif current_contract != contract_key:
            raise ContractError(
                "paper outputs do not share contract/probe/layout policy: "
                f"expected={contract_key!r} observed={current_contract!r} "
                f"paper={paper_id}"
            )
        for row in unit["rows"]:
            page_id = str(row["page_id"])
            if page_id in seen_page_ids:
                raise ContractError(f"page_id is not unique in fixed cohort: {page_id}")
            seen_page_ids.add(page_id)
        units.append(unit)
        all_rows.extend(unit["rows"])
        stable_guards.append(unit["stable_guard"])
        elapsed = max(time.monotonic() - started, 1e-9)
        print_cohort_progress(
            completed=index,
            total=len(resolved_inputs),
            current=paper_id,
            units=units,
            started=started,
        )
        print(
            f"[paper_done] paper={paper_id} unit={index}/{len(resolved_inputs)} "
            f"pages={unit['pages_total']} eligible={unit['eligible_text_pages']} "
            f"passed={unit['source_first_passed_pages']} "
            f"complex={unit['accepted_complex_pages']} "
            f"exact={unit['accepted_source_first_verifier_exact_pages']} "
            f"elapsed={elapsed_text(elapsed)}",
            flush=True,
        )

    if contract_key is None:  # defensive; the non-empty check is above
        raise ContractError("fixed cohort has no paper outputs")

    eligible = sum(int(unit["eligible_text_pages"]) for unit in units)
    passed = sum(int(unit["source_first_passed_pages"]) for unit in units)
    complex_passed = sum(int(unit["accepted_complex_pages"]) for unit in units)
    two_column_passed = sum(
        int(unit["accepted_two_column_pages"]) for unit in units
    )
    exact_passed = sum(
        int(unit["accepted_source_first_verifier_exact_pages"]) for unit in units
    )
    source_first_yield = passed / eligible if eligible else 0.0
    exact_rate = exact_passed / passed if passed else None
    target = {
        "source_first_passed_over_eligible_text_page_gt_0_30": source_first_yield > 0.30,
        "accepted_complex_gt_0": complex_passed > 0,
        "accepted_two_column_gt_0": two_column_passed > 0,
        "accepted_source_first_verifier_exact_rate_1_0": exact_rate == 1.0,
    }
    target["passed"] = all(target.values())

    metadata = {
        "purpose": "fixed-cohort-source-first-v2-evaluation",
        "paper_outputs": [unit["paper_output"] for unit in units],
        "stable_output_roots": [
            str(Path(path).expanduser().resolve()) for path in stable_output_roots
        ],
        "denominator": "eligible_text_page",
    }
    # Create/check the output only after every input has passed validation.  A
    # valid v2 marker allows an intentional resume/recompute; stable markers
    # are rejected by the shared contract helper.
    validate_experimental_directory(
        resolved_output,
        create=True,
        purpose="fixed-cohort-source-first-v2-evaluation",
        metadata=metadata,
    )
    atomic_write_jsonl(resolved_output / OUTPUT_LEDGER_FILENAME, all_rows)

    report: dict[str, Any] = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "contract": contract_key[0],
        "probe_policy_version": contract_key[1],
        "layout_policy_version": contract_key[2],
        "pipeline_version": PIPELINE_VERSION,
        "status": "passed" if target["passed"] else "failed",
        "fixed_cohort": True,
        "denominator": "eligible_text_page",
        "papers_selected": len(units),
        "paper_ids": [unit["paper_id"] for unit in units],
        "paper_outputs": [unit["paper_output"] for unit in units],
        "pages_total": len(all_rows),
        "eligible_text_pages": eligible,
        "source_first_passed_pages": passed,
        "accepted_complex_pages": complex_passed,
        "accepted_two_column_pages": two_column_passed,
        "accepted_two_column_layout_pages": two_column_passed,
        "accepted_source_first_verifier_exact_pages": exact_passed,
        "source_first_passed_over_eligible_text_page": round(source_first_yield, 8),
        "source_first_yield": round(source_first_yield, 8),
        "accepted_source_first_verifier_exact_rate": (
            round(exact_rate, 8) if exact_rate is not None else None
        ),
        "stable_guard": {
            "status": "passed",
            "ok": stable_guard_current["ok"] and all(
                guard.get("status") == "passed" and guard.get("ok") is True
                for guard in stable_guards
            ),
            "current": stable_guard_current,
            "papers": stable_guards,
        },
        "target": target,
        "metrics": {
            "eligible_text_page_denominator": eligible,
            "source_first_passed": passed,
            "source_first_yield": round(source_first_yield, 8),
            "accepted_complex": complex_passed,
            "accepted_two_column": two_column_passed,
            "accepted_source_first_verifier_exact_rate": (
                round(exact_rate, 8) if exact_rate is not None else None
            ),
        },
        "source_reports": [
            {
                "paper_id": unit["paper_id"],
                "report": unit["report_path"],
                "ledger": unit["ledger_path"],
                "pages_total": unit["pages_total"],
                "eligible_text_pages": unit["eligible_text_pages"],
                "source_first_passed_pages": unit["source_first_passed_pages"],
                "accepted_complex_pages": unit["accepted_complex_pages"],
                "accepted_source_first_verifier_exact_pages": unit[
                    "accepted_source_first_verifier_exact_pages"
                ],
            }
            for unit in units
        ],
        "created_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(resolved_output / OUTPUT_REPORT_FILENAME, report)
    elapsed = time.monotonic() - started
    print(
        f"[finish] status={report['status']} papers={len(units)} "
        f"pages={passed}/{eligible} source_first_yield={100*source_first_yield:.2f}% "
        f"complex={complex_passed} two_column={two_column_passed} "
        f"exact_rate={exact_rate} target_passed={target['passed']} "
        f"elapsed={elapsed_text(elapsed)} output={resolved_output}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-output",
        action="append",
        type=Path,
        required=True,
        help="Per-paper v2 output directory; repeat for every fixed-cohort paper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New isolated experimental directory for the combined cohort.",
    )
    parser.add_argument(
        "--stable-output-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Stable output tree that must not overlap --output-dir; repeatable. "
            "Ancestor stable markers are rejected automatically."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        aggregate_fixed_cohort(
            args.paper_output,
            args.output_dir,
            stable_output_roots=args.stable_output_root,
        )
    except (ContractError, OSError, ValueError) as exc:
        print(f"[finish] status=failed error={exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
