from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from arxiv_source_first_v2.contracts import (
    ContractError,
    EXPERIMENTAL_MARKER_FILENAME,
    PIPELINE_VERSION,
    STABLE_V10_PIPELINE_VERSION,
    detect_stable_v10_markers,
    normalize_layout_bucket,
    validate_experimental_directory,
    validate_page_ledger,
    verify_stable_files,
)
from arxiv_source_first_v2.metrics import compute_yield_metrics


class SourceFirstV2ContractsMetricsTests(unittest.TestCase):
    def test_frozen_stable_chain_is_read_only_and_matches(self) -> None:
        root = Path(__file__).resolve().parents[1]
        before = (root / "scripts/run_arxiv_source_bins_to_verl.py").read_bytes()
        result = verify_stable_files(root)
        self.assertTrue(result["ok"], result)
        self.assertEqual(before, (root / "scripts/run_arxiv_source_bins_to_verl.py").read_bytes())

    def test_new_nonempty_directory_requires_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "partial.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "non-empty unmarked"):
                validate_experimental_directory(root)

    def test_marker_create_resume_and_stable_v10_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "work"
            created = validate_experimental_directory(root, create=True)
            self.assertEqual(created["status"], "created")
            marker = root / EXPERIMENTAL_MARKER_FILENAME
            self.assertTrue(marker.is_file())
            self.assertEqual(detect_stable_v10_markers(root), [])
            self.assertEqual(validate_experimental_directory(root)["status"], "passed")

            stable = root / "nested" / "pipeline_report.json"
            stable.parent.mkdir()
            stable.write_text(
                json.dumps({"pipeline_version": STABLE_V10_PIPELINE_VERSION}),
                encoding="utf-8",
            )
            self.assertEqual(detect_stable_v10_markers(root), [stable.resolve()])
            with self.assertRaisesRegex(ContractError, "stable v10"):
                validate_experimental_directory(root)

    def test_marker_rejects_wrong_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validate_experimental_directory(root, create=True)
            marker = root / EXPERIMENTAL_MARKER_FILENAME
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["pipeline_version"] = "stable-v10"
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "does not match"):
                validate_experimental_directory(root)

    def test_layout_aliases_cover_complex_buckets(self) -> None:
        self.assertEqual(normalize_layout_bucket("single column"), "single_column")
        self.assertEqual(normalize_layout_bucket("twocolumn"), "two_column")
        self.assertEqual(normalize_layout_bucket("mixed_full_and_columns"), "mixed_full_two_column")
        self.assertEqual(normalize_layout_bucket("sidebar"), "other")
        self.assertEqual(normalize_layout_bucket(None), "unknown")

    def test_page_ledger_normalizes_and_rejects_inconsistent_transitions(self) -> None:
        rows = validate_page_ledger(
            [
                {
                    "data_id": "paper_page_0001",
                    "paper_id": "paper",
                    "page_number": 1,
                    "layout": "single column",
                    "clean": True,
                    "source_first_eligible": True,
                    "edit_accepted": True,
                    "verifier": {"exact_ordered_character_stream_match": True},
                }
            ],
            require_explicit_outcomes=True,
        )
        self.assertEqual(rows[0]["page_id"], "paper_page_0001")
        self.assertEqual(rows[0]["layout"], "single_column")
        self.assertTrue(rows[0]["verifier_exact"])

        with self.assertRaisesRegex(ContractError, "not clean"):
            validate_page_ledger(
                [
                    {
                        "page_id": "paper_page_0002",
                        "paper_id": "paper",
                        "page_number": 2,
                        "clean": False,
                        "source_first_eligible": True,
                        "edit_accepted": False,
                    }
                ]
            )

    def test_yields_stage_boundaries_buckets_and_targets(self) -> None:
        rows = [
            {
                "page_id": "p1",
                "paper_id": "paper",
                "page_number": 1,
                "layout": "single_column",
                "candidate": True,
                "clean": True,
                "source_first_eligible": True,
                "edit_accepted": True,
                "verifier_exact": True,
            },
            {
                "page_id": "p2",
                "paper_id": "paper",
                "page_number": 2,
                "layout": "two_column",
                "candidate": True,
                "clean": True,
                "source_first_eligible": True,
                "edit_accepted": True,
                "verifier_exact": True,
            },
            {
                "page_id": "p3",
                "paper_id": "paper",
                "page_number": 3,
                "layout": "mixed_full_and_columns",
                "candidate": True,
                "clean": True,
                "source_first_eligible": False,
                "edit_accepted": False,
                "verifier_exact": False,
            },
            {
                "page_id": "p4",
                "paper_id": "paper",
                "page_number": 4,
                "layout": "unknown-value",
                "candidate": True,
                "clean": False,
                "source_first_eligible": False,
                "edit_accepted": False,
                "verifier_exact": False,
            },
            {
                "page_id": "p5",
                "paper_id": "paper",
                "page_number": 5,
                "layout": "single_column",
                "candidate": False,
                "clean": True,
                "source_first_eligible": True,
                "edit_accepted": True,
                "verifier_exact": True,
            },
        ]
        report = compute_yield_metrics(rows, require_explicit_outcomes=True)
        self.assertEqual(report["candidate_pages"], 4)
        self.assertEqual(report["all_clean_pages"], 3)
        self.assertEqual(report["eligible_source_first_pages"], 2)
        self.assertEqual(report["accepted_edit_pages"], 2)
        self.assertEqual(report["all_clean_pages_yield"], 0.75)
        self.assertEqual(report["eligible_source_first_yield"], round(2 / 3, 8))
        self.assertEqual(report["final_edit_yield"], 1.0)
        self.assertEqual(report["overall_yield"], 0.5)
        self.assertEqual(report["accepted_verifier_exact_rate"], 1.0)
        self.assertEqual(report["accepted_complex_pages"], 1)
        self.assertTrue(report["target"]["passed"])
        self.assertEqual(report["buckets"]["two_column"]["accepted_edit_pages"], 1)
        self.assertEqual(report["buckets"]["mixed_full_two_column"]["all_clean_pages"], 1)
        self.assertEqual(report["buckets"]["other"]["candidate_pages"], 1)
        self.assertEqual(report["buckets"]["unknown"]["candidate_pages"], 0)

    def test_zero_denominators_fail_closed(self) -> None:
        report = compute_yield_metrics([])
        self.assertEqual(report["overall_yield"], 0.0)
        self.assertIsNone(report["accepted_verifier_exact_rate"])
        self.assertFalse(report["target"]["passed"])


if __name__ == "__main__":
    unittest.main()
