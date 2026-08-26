"""Contracts and safety boundaries for the experimental source-first v2 pipeline.

This module intentionally contains no import of the stable runner.  The v2
pipeline is allowed to reuse stable implementation helpers, but its output
directories and its data contract are kept separate so that an experiment
cannot silently resume or overwrite a v10 run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


EXPERIMENT_NAME = "arxiv_source_first_v2"
EXPERIMENTAL_MARKER_FILENAME = "EXPERIMENTAL_V2.json"
EXPERIMENTAL_SCHEMA_VERSION = 1
EXPERIMENTAL_CONTRACT = "arxiv_source_first_v2_anchor_lattice"
PIPELINE_VERSION = "source_bins_to_source_first_confusable_verl_v2_anchor_lattice"
# Short aliases make the contract easy to import from an experimental runner;
# they are aliases, not independently mutable version knobs.
V2_SCHEMA_VERSION = EXPERIMENTAL_SCHEMA_VERSION
V2_CONTRACT = EXPERIMENTAL_CONTRACT
V2_PIPELINE_VERSION = PIPELINE_VERSION
EXPERIMENTAL_V2_MARKER = EXPERIMENTAL_MARKER_FILENAME

# This is deliberately a separate constant from PIPELINE_VERSION.  A v2 run
# must never be mistaken for the frozen v10 runner, even if the implementation
# happens to call a stable helper internally.
STABLE_V10_PIPELINE_VERSION = (
    "source_bins_to_source_first_confusable_verl_v10_list_payload_math_minus"
)

LAYOUT_BUCKETS = (
    "single_column",
    "two_column",
    "mixed_full_two_column",
    "other",
    "unknown",
)
COMPLEX_LAYOUT_BUCKETS = frozenset(
    {"two_column", "mixed_full_two_column", "other"}
)

# The four files below are the stable execution chain.  Keep these values
# literal: changing a stable script requires an explicit stable-version
# change, not an implicit update through this experimental package.
STABLE_FILE_SHA256: dict[str, str] = {
    "scripts/run_arxiv_source_bins_to_verl.py": (
        "f754446ae6f7400d656db2bc6e79dac8d7ab9897cfb356984947528978131728"
    ),
    "scripts/build_source_first_color_page_gt.py": (
        "fcb8dd7a0e1b2da0c656f7d8f7fabbb9576ec85f1f17041c7c6bc0bc19276c69"
    ),
    "scripts/build_arxiv_confusable_recompile_pilot.py": (
        "2a0cb793732343e540c9a0c497ed501cb773ee0ecf5cb220cfed398d023af30e"
    ),
    "scripts/verify_arxiv_confusable_recompile_pilot.py": (
        "db7f29eda56578758c4dfe462e6f78bc0495151651b3cca48d671ee27af462d5"
    ),
}

_SAFE_STEM = re.compile(r"^[A-Za-z0-9._-]+$")
_PAGE_STATUSES = frozenset(
    {
        "candidate",
        "clean",
        "eligible",
        "source_first_eligible",
        "passed",
        "pass",
        "success",
        "accepted",
        "complete",
        "edit_accepted",
        "rejected",
        "failed",
        "skipped",
    }
)
_TRUE_WORDS = frozenset({"true", "yes", "passed", "success", "accepted", "clean"})


class ContractError(ValueError):
    """Raised when an experimental input/output boundary is unsafe."""


def _repo_root(root: str | os.PathLike[str] | Path | None) -> Path:
    if root is None:
        return Path(__file__).resolve().parents[2]
    return Path(root).expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stable_files(
    repo_root: str | os.PathLike[str] | Path | None = None,
) -> dict[str, Any]:
    """Read-only verification of the frozen stable execution chain.

    The function only opens and hashes the four declared files.  It never
    writes a marker, updates a checksum, or changes the repository.  A result
    dictionary is returned instead of raising so callers can include the
    diagnosis in a run report.
    """

    root = _repo_root(repo_root)
    files: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, str]] = []
    for relative, expected in STABLE_FILE_SHA256.items():
        path = root / relative
        if not path.is_file():
            files[relative] = {"status": "missing", "expected_sha256": expected}
            mismatches.append(
                {"path": relative, "reason": "missing", "expected_sha256": expected}
            )
            continue
        observed = _sha256_file(path)
        status = "passed" if observed == expected else "mismatch"
        files[relative] = {
            "status": status,
            "expected_sha256": expected,
            "observed_sha256": observed,
        }
        if status != "passed":
            mismatches.append(
                {
                    "path": relative,
                    "reason": "sha256_mismatch",
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                }
            )
    return {
        "status": "passed" if not mismatches else "failed",
        "ok": not mismatches,
        "repo_root": str(root),
        "stable_pipeline_version": STABLE_V10_PIPELINE_VERSION,
        "files": files,
        "mismatches": mismatches,
    }


def assert_stable_files(
    repo_root: str | os.PathLike[str] | Path | None = None,
) -> dict[str, Any]:
    """Verify the frozen files and raise a useful error if any changed."""

    result = verify_stable_files(repo_root)
    if not result["ok"]:
        raise ContractError(
            "stable execution chain changed: "
            + json.dumps(result["mismatches"], ensure_ascii=False)
        )
    return result


def experimental_marker_payload(
    *,
    purpose: str = "source-first-v2-experiment",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the self-describing marker written only to an experimental root."""

    payload: dict[str, Any] = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "contract": EXPERIMENTAL_CONTRACT,
        "pipeline_version": PIPELINE_VERSION,
        "stable_v10_pipeline_version": STABLE_V10_PIPELINE_VERSION,
        "stable_file_sha256": dict(STABLE_FILE_SHA256),
        "purpose": purpose,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def write_experimental_marker(
    root: str | os.PathLike[str] | Path,
    *,
    purpose: str = "source-first-v2-experiment",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Create or replace the v2 marker in an explicitly experimental root.

    This helper is intentionally scoped to the marker filename; callers must
    first pass :func:`validate_experimental_directory` when reusing a root.
    """

    directory = Path(root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / EXPERIMENTAL_MARKER_FILENAME
    temporary = marker.with_name(marker.name + ".tmp")
    temporary.write_text(
        json.dumps(
            experimental_marker_payload(purpose=purpose, metadata=metadata),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)
    return marker


def _json_files_with_stable_marker(directory: Path) -> list[Path]:
    found: list[Path] = []
    # The reports are the common case.  The recursive scan additionally catches
    # a copied v10 report nested under a paper directory.
    try:
        paths = directory.rglob("*.json")
    except OSError:
        return found
    for path in paths:
        if path.name == EXPERIMENTAL_MARKER_FILENAME:
            continue
        try:
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            # A malformed copied report is still a useful safety signal.  A
            # valid v2 marker is parsed above and is excluded from the scan.
            if STABLE_V10_PIPELINE_VERSION in text:
                found.append(path)
            continue

        def has_stable_pipeline_marker(candidate: Any) -> bool:
            if isinstance(candidate, Mapping):
                if candidate.get("pipeline_version") == STABLE_V10_PIPELINE_VERSION:
                    return True
                return any(has_stable_pipeline_marker(item) for item in candidate.values())
            if isinstance(candidate, list):
                return any(has_stable_pipeline_marker(item) for item in candidate)
            return False

        if has_stable_pipeline_marker(value):
            found.append(path)
    return found


def detect_stable_v10_markers(
    root: str | os.PathLike[str] | Path,
) -> list[Path]:
    """Return files under *root* that identify a stable v10 pipeline run."""

    directory = Path(root).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        return []
    return _json_files_with_stable_marker(directory)


def _read_marker(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid experimental marker: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"experimental marker must be a JSON object: {path}")
    return value


def validate_experimental_directory(
    root: str | os.PathLike[str] | Path,
    *,
    create: bool = False,
    purpose: str = "source-first-v2-experiment",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the isolation boundary for a v2 work/output directory.

    A non-existent directory is created only with ``create=True``.  A
    non-empty directory without a valid v2 marker is rejected, as is any
    directory containing a stable v10 pipeline report.  Existing valid v2
    directories are returned unchanged and can safely be resumed.
    """

    directory = Path(root).expanduser().resolve()
    if not directory.exists():
        if not create:
            raise ContractError(f"experimental directory does not exist: {directory}")
        write_experimental_marker(directory, purpose=purpose, metadata=metadata)
        return {
            "status": "created",
            "ok": True,
            "root": str(directory),
            "marker": str(directory / EXPERIMENTAL_MARKER_FILENAME),
            "stable_markers": [],
        }
    if not directory.is_dir():
        raise ContractError(f"experimental path is not a directory: {directory}")

    stable_markers = detect_stable_v10_markers(directory)
    if stable_markers:
        raise ContractError(
            "stable v10 pipeline marker detected under experimental directory: "
            + ", ".join(str(path) for path in stable_markers)
        )

    marker = directory / EXPERIMENTAL_MARKER_FILENAME
    entries = list(directory.iterdir())
    if not marker.exists():
        if entries:
            raise ContractError(
                "refusing non-empty unmarked experimental directory: " + str(directory)
            )
        if create:
            write_experimental_marker(directory, purpose=purpose, metadata=metadata)
            return {
                "status": "created",
                "ok": True,
                "root": str(directory),
                "marker": str(marker),
                "stable_markers": [],
            }
        return {
            "status": "empty_unmarked",
            "ok": True,
            "root": str(directory),
            "marker": None,
            "stable_markers": [],
        }

    payload = _read_marker(marker)
    expected = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "contract": EXPERIMENTAL_CONTRACT,
        "pipeline_version": PIPELINE_VERSION,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ContractError(
            "experimental marker does not match v2 contract: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return {
        "status": "passed",
        "ok": True,
        "root": str(directory),
        "marker": str(marker),
        "marker_payload": payload,
        "stable_markers": [],
    }


def normalize_layout_bucket(value: Any) -> str:
    """Normalize renderer layout labels to the v2 five-bucket vocabulary."""

    if value is None:
        return "unknown"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"single_column", "singlecolumn", "one_column", "onecolumn", "full_width"}:
        return "single_column"
    if text in {"two_column", "twocolumn", "double_column", "doublecolumn", "2_column"}:
        return "two_column"
    if text in {
        "mixed_full_two_column",
        "mixed_full_and_columns",
        "mixed_full_columns",
        "full_and_columns",
        "mixed_twocolumn",
    }:
        return "mixed_full_two_column"
    if text in {"other", "three_column", "multi_column", "sidebar", "complex"}:
        return "other"
    if text in {"", "unknown", "none", "null", "na", "n_a"}:
        return "unknown"
    return "other"


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_WORDS or text == "1":
            return True
        if text in {"false", "no", "failed", "rejected", "0"}:
            return False
    return None


def _first_bool(row: Mapping[str, Any], keys: tuple[str, ...]) -> bool | None:
    for key in keys:
        if key in row:
            value = _bool_value(row[key])
            if value is None:
                raise ContractError(f"page ledger field {key!r} must be boolean: {row[key]!r}")
            return value
    return None


def _nested_verifier_exact(row: Mapping[str, Any]) -> bool | None:
    verifier = row.get("verifier")
    if verifier is None:
        return None
    if not isinstance(verifier, Mapping):
        raise ContractError("page ledger verifier must be an object")
    for key in (
        "exact_ordered_character_stream_match",
        "exact_ordered_token_match",
        "exact",
    ):
        if key in verifier:
            value = _bool_value(verifier[key])
            if value is None:
                raise ContractError(f"verifier field {key!r} must be boolean")
            return value
    return None


def validate_page_ledger(
    rows: Any,
    *,
    require_explicit_outcomes: bool = False,
) -> list[dict[str, Any]]:
    """Validate and normalize page ledger rows used by v2 metrics.

    The returned rows are shallow copies.  The input is never modified.  The
    validator accepts either the v2 names or the stable pipeline's common
    aliases (``status``, ``verifier``, ``layout``), then emits canonical
    ``clean``, ``source_first_eligible``, ``edit_accepted``, and
    ``verifier_exact`` fields for metric computation.
    """

    if isinstance(rows, Mapping):
        rows = rows.get("pages", rows.get("ledger", rows.get("rows")))
    if rows is None or isinstance(rows, (str, bytes)):
        raise ContractError("page ledger must be a sequence of objects")
    try:
        materialized = list(rows)
    except TypeError as exc:
        raise ContractError("page ledger must be iterable") from exc

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(materialized):
        if not isinstance(raw, Mapping):
            raise ContractError(f"page ledger row {index} must be an object")
        row = dict(raw)
        page_id_value = row.get("page_id", row.get("data_id", row.get("pair_id")))
        if page_id_value is None or not str(page_id_value).strip():
            raise ContractError(f"page ledger row {index} has no page_id/data_id/pair_id")
        page_id = str(page_id_value)
        if page_id in seen_ids:
            raise ContractError(f"duplicate page ledger id: {page_id}")
        seen_ids.add(page_id)
        paper_id = row.get("paper_id", row.get("stem"))
        if paper_id is None or not str(paper_id).strip() or not _SAFE_STEM.fullmatch(str(paper_id)):
            raise ContractError(f"page ledger row {page_id} has unsafe/missing paper_id")
        page_number = row.get("page_number", row.get("page"))
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
            raise ContractError(f"page ledger row {page_id} has invalid page_number: {page_number!r}")

        layout = normalize_layout_bucket(row.get("layout", row.get("layout_bucket")))
        status = row.get("status")
        if status is not None and str(status).strip().lower() not in _PAGE_STATUSES:
            # Unknown statuses are unsafe for a strict ledger because their
            # denominator semantics cannot be inferred.
            raise ContractError(f"page ledger row {page_id} has unknown status: {status!r}")

        status_text = str(status).strip().lower() if status is not None else ""
        candidate = _first_bool(row, ("candidate", "candidate_page", "eligible_candidate"))
        if candidate is None:
            candidate = status_text != "skipped"
        clean = _first_bool(row, ("clean", "clean_page", "clean_accepted"))
        if clean is None:
            clean = status_text in {
                "clean",
                "eligible",
                "passed",
                "pass",
                "success",
                "accepted",
                "complete",
                "source_first_eligible",
                "edit_accepted",
            }
        source_first = _first_bool(
            row,
            ("source_first_eligible", "eligible_source_first", "source_first_passed"),
        )
        if source_first is None:
            source_status = row.get("source_first_status", status)
            source_first = str(source_status).strip().lower() in {
                "passed",
                "pass",
                "success",
                "accepted",
                "eligible",
                "source_first_eligible",
                "complete",
                "edit_accepted",
            }
        edit_accepted = _first_bool(
            row,
            ("edit_accepted", "final_edit_accepted", "accepted_edit", "accepted"),
        )
        if edit_accepted is None:
            edit_status = row.get("edit_status", row.get("final_status", status))
            edit_accepted = str(edit_status).strip().lower() in {
                "passed",
                "pass",
                "success",
                "accepted",
                "complete",
                "edit_accepted",
            }
        verifier_exact = _first_bool(
            row,
            ("verifier_exact", "accepted_verifier_exact", "exact_verifier"),
        )
        nested_exact = _nested_verifier_exact(row)
        if verifier_exact is not None and nested_exact is not None and verifier_exact != nested_exact:
            raise ContractError(f"page ledger row {page_id} has conflicting verifier exact flags")
        verifier_exact = verifier_exact if verifier_exact is not None else nested_exact
        if verifier_exact is None:
            verifier_exact = False
        if require_explicit_outcomes and (
            not any(key in row for key in ("clean", "clean_page", "clean_accepted"))
            or not any(key in row for key in ("source_first_eligible", "eligible_source_first", "source_first_passed"))
            or not any(key in row for key in ("edit_accepted", "final_edit_accepted", "accepted_edit", "accepted"))
        ):
            raise ContractError(f"page ledger row {page_id} lacks explicit v2 outcome fields")
        if source_first and not clean:
            raise ContractError(f"page ledger row {page_id}: source-first eligible page is not clean")
        if edit_accepted and not source_first:
            raise ContractError(f"page ledger row {page_id}: accepted edit is not source-first eligible")
        if verifier_exact and not edit_accepted:
            raise ContractError(f"page ledger row {page_id}: exact verifier flag requires accepted edit")

        row.update(
            {
                "page_id": page_id,
                "paper_id": str(paper_id),
                "page_number": page_number,
                "layout": layout,
                "candidate": bool(candidate),
                "clean": bool(clean),
                "source_first_eligible": bool(source_first),
                "edit_accepted": bool(edit_accepted),
                "verifier_exact": bool(verifier_exact),
            }
        )
        normalized.append(row)
    return normalized


def validate_page_ledger_file(
    path: str | os.PathLike[str] | Path,
    *,
    require_explicit_outcomes: bool = False,
) -> list[dict[str, Any]]:
    """Load JSON/JSONL ledger data and pass it through the same validator."""

    ledger_path = Path(path).expanduser().resolve()
    if not ledger_path.is_file():
        raise ContractError(f"page ledger does not exist: {ledger_path}")
    try:
        if ledger_path.suffix.lower() == ".jsonl":
            rows: list[Any] = []
            for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ContractError(f"invalid JSONL at {ledger_path}:{line_number}: {exc}") from exc
        else:
            rows = json.loads(ledger_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read page ledger: {ledger_path}: {exc}") from exc
    return validate_page_ledger(rows, require_explicit_outcomes=require_explicit_outcomes)


__all__ = [
    "COMPLEX_LAYOUT_BUCKETS",
    "ContractError",
    "EXPERIMENTAL_CONTRACT",
    "EXPERIMENTAL_MARKER_FILENAME",
    "EXPERIMENTAL_V2_MARKER",
    "EXPERIMENTAL_SCHEMA_VERSION",
    "EXPERIMENT_NAME",
    "LAYOUT_BUCKETS",
    "PIPELINE_VERSION",
    "STABLE_FILE_SHA256",
    "STABLE_V10_PIPELINE_VERSION",
    "V2_CONTRACT",
    "V2_PIPELINE_VERSION",
    "V2_SCHEMA_VERSION",
    "assert_stable_files",
    "detect_stable_v10_markers",
    "experimental_marker_payload",
    "normalize_layout_bucket",
    "validate_experimental_directory",
    "validate_page_ledger",
    "validate_page_ledger_file",
    "verify_stable_files",
    "write_experimental_marker",
]
