"""Yield metrics for the experimental source-first v2 page ledger.

The metrics deliberately use page-level denominators and preserve the stage
boundaries that matter for the >30% goal:

``all_clean_pages_yield``
    clean pages / candidate pages;
``eligible_source_first_yield``
    source-first eligible pages / clean pages;
``final_edit_yield``
    accepted edit pages / source-first eligible pages;
``overall_yield``
    accepted edit pages / candidate pages.

The final metric is also reported per layout bucket.  No metric in this file
uses PDF text to manufacture a ground truth; the ledger is an audit record of
the source-driven generation and independent verifier stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import (
    COMPLEX_LAYOUT_BUCKETS,
    LAYOUT_BUCKETS,
    ContractError,
    validate_page_ledger,
    validate_page_ledger_file,
)


YIELD_TARGET = 0.30


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _candidate(row: Mapping[str, Any]) -> bool:
    value = row.get("candidate", True)
    if not isinstance(value, bool):
        raise ContractError(f"normalized page ledger candidate must be boolean: {value!r}")
    return value


def _counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    candidate = clean = source_first = edit = exact = complex_edit = 0
    for row in rows:
        if not _candidate(row):
            continue
        candidate += 1
        is_clean = bool(row["clean"])
        is_source_first = bool(row["source_first_eligible"])
        is_edit = bool(row["edit_accepted"])
        is_exact = bool(row["verifier_exact"])
        if is_clean:
            clean += 1
        if is_source_first:
            source_first += 1
        if is_edit:
            edit += 1
            if str(row["layout"]) in COMPLEX_LAYOUT_BUCKETS:
                complex_edit += 1
            if not is_exact:
                # A final accepted page without an exact independent verifier
                # cannot satisfy the v2 contract.  Keep it in the stage count
                # for diagnosis; the target flag below will fail closed.
                pass
        if is_exact and is_edit:
            exact += 1
    return {
        "candidate_pages": candidate,
        "all_clean_pages": clean,
        "eligible_source_first_pages": source_first,
        "accepted_edit_pages": edit,
        "accepted_verifier_exact_pages": exact,
        "accepted_complex_pages": complex_edit,
    }


def _stage_metrics(counts: Mapping[str, int]) -> dict[str, Any]:
    candidate = int(counts["candidate_pages"])
    clean = int(counts["all_clean_pages"])
    source_first = int(counts["eligible_source_first_pages"])
    edit = int(counts["accepted_edit_pages"])
    exact = int(counts["accepted_verifier_exact_pages"])
    exact_rate: float | None = _ratio(exact, edit) if edit else None
    return {
        **{key: int(value) for key, value in counts.items()},
        "all_clean_pages_yield": _ratio(clean, candidate),
        "eligible_source_first_yield": _ratio(source_first, clean),
        "final_edit_yield": _ratio(edit, source_first),
        "overall_yield": _ratio(edit, candidate),
        "accepted_verifier_exact_rate": exact_rate,
        "target_overall_gt_0_30": _ratio(edit, candidate) > YIELD_TARGET,
        "target_accepted_complex_gt_0": int(counts["accepted_complex_pages"]) > 0,
        "target_accepted_verifier_exact_1_0": exact_rate == 1.0,
    }


def compute_yield_metrics(
    rows: Any,
    *,
    require_explicit_outcomes: bool = False,
) -> dict[str, Any]:
    """Compute stage, overall, and layout-bucket page yields.

    ``rows`` may be a sequence of ledger objects or a mapping containing a
    ``pages``, ``ledger``, or ``rows`` array.  The returned object is JSON
    serializable and includes the normalized bucket denominators, so a report
    can be audited without reconstructing the input ledger.
    """

    normalized = validate_page_ledger(
        rows, require_explicit_outcomes=require_explicit_outcomes
    )
    overall_counts = _counts(normalized)
    buckets: dict[str, Any] = {}
    for bucket in LAYOUT_BUCKETS:
        bucket_rows = [row for row in normalized if row["layout"] == bucket]
        buckets[bucket] = _stage_metrics(_counts(bucket_rows))
    result = _stage_metrics(overall_counts)
    result.update(
        {
            "schema_version": 1,
            "metric_contract": "source_first_v2_page_yield_v1",
            "yield_target": YIELD_TARGET,
            "complex_layout_buckets": sorted(COMPLEX_LAYOUT_BUCKETS),
            "buckets": buckets,
            "accepted_complex_layouts": {
                bucket: buckets[bucket]["accepted_edit_pages"]
                for bucket in COMPLEX_LAYOUT_BUCKETS
            },
            "verifier_exact": {
                "accepted_pages": overall_counts["accepted_verifier_exact_pages"],
                "accepted_pages_total": overall_counts["accepted_edit_pages"],
                "rate": result["accepted_verifier_exact_rate"],
            },
            "target": {
                "overall_yield_gt_0_30": result["target_overall_gt_0_30"],
                "accepted_complex_gt_0": result["target_accepted_complex_gt_0"],
                "accepted_verifier_exact_1_0": result[
                    "target_accepted_verifier_exact_1_0"
                ],
                "passed": (
                    result["target_overall_gt_0_30"]
                    and result["target_accepted_complex_gt_0"]
                    and result["target_accepted_verifier_exact_1_0"]
                ),
            },
        }
    )
    return result


def calculate_yields(rows: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for :func:`compute_yield_metrics`."""

    return compute_yield_metrics(rows, **kwargs)


def compute_yield_metrics_file(
    path: str | Path,
    *,
    require_explicit_outcomes: bool = False,
) -> dict[str, Any]:
    """Load a JSON/JSONL page ledger and compute the v2 yield report."""

    return compute_yield_metrics(
        validate_page_ledger_file(
            path, require_explicit_outcomes=require_explicit_outcomes
        )
    )


def write_yield_metrics(
    rows: Any,
    path: str | Path,
    *,
    require_explicit_outcomes: bool = False,
) -> dict[str, Any]:
    """Write an atomic JSON report and return the report object."""

    report = compute_yield_metrics(rows, require_explicit_outcomes=require_explicit_outcomes)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report


__all__ = [
    "YIELD_TARGET",
    "calculate_yields",
    "compute_yield_metrics",
    "compute_yield_metrics_file",
    "write_yield_metrics",
]
