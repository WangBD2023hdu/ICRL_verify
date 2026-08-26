from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "experimental" / "evaluate_source_first_v2_cohort.py"
CONTRACT = "arxiv_source_first_v2_anchor_lattice"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class FixedSourceFirstV2CohortTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_paper(self, paper_id: str, passed_pages: set[int], complex_pages: set[int]) -> Path:
        output = self.root / paper_id
        output.mkdir()
        rows: list[dict[str, object]] = []
        for page_number in range(1, 22):
            passed = page_number in passed_pages
            rows.append(
                {
                    "page_id": f"{paper_id}_page_{page_number:04d}_sfspanv2",
                    "paper_id": paper_id,
                    "page_number": page_number,
                    "candidate": True,
                    "clean": True,
                    "eligible_text_page": True,
                    "source_first_passed": passed,
                    "source_first_verifier_exact": passed,
                    "edit_accepted": False,
                    "verifier_exact": False,
                    "layout": "two_column" if page_number in complex_pages else "single_column",
                    "status": "passed" if passed else "rejected",
                    "rejection_reasons": [] if passed else ["fixture_rejection"],
                }
            )
        (output / "page_ledger_v2.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        report = {
            "schema_version": 1,
            "contract": CONTRACT,
            "pipeline_version": "source_bins_to_source_first_confusable_verl_v2_anchor_lattice",
            "probe_policy_version": "source_probe_v2",
            "layout_policy_version": "layout_graph_v2",
            "status": "passed",
            "paper_id": paper_id,
            "pages_total": 21,
            "eligible_clean_text_pages": 21,
            "pages_passed": len(passed_pages),
            "accepted_complex_layout_pages": len(passed_pages & complex_pages),
            "accepted_two_column_pages": len(passed_pages & complex_pages),
            "accepted_exact_verifier_rate": 1.0,
            "stable_guard": {"status": "passed", "ok": True},
        }
        (output / "validation_report_v2.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return output

    def run_command(
        self,
        *paper_outputs: Path,
        stable_output_roots: tuple[Path, ...] = (),
        output: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        destination = output or (self.root / "fixed_cohort_v2")
        command = [sys.executable, str(SCRIPT)]
        for paper_output in paper_outputs:
            command.extend(("--paper-output", str(paper_output)))
        command.extend(("--output-dir", str(destination)))
        for stable_root in stable_output_roots:
            command.extend(("--stable-output-root", str(stable_root)))
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        return completed, destination

    def test_fixed_eligible_text_page_denominator_and_complex_gate(self) -> None:
        # Each of two papers contributes seven accepted source-first pages out
        # of 21 eligible pages; exactly two accepted pages are complex and all
        # accepted pages are exact.  The combined fixed cohort is 14/42.
        first = self.make_paper("fixtureA", {1, 2, 3, 4, 5, 6, 7}, {1})
        second = self.make_paper("fixtureB", {1, 2, 3, 4, 5, 6, 7}, {2})
        completed, destination = self.run_command(first, second)

        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("[start] phase=fixed_cohort", completed.stdout)
        self.assertIn("[paper_done] paper=fixtureA unit=1/2", completed.stdout)
        self.assertIn("[paper_done] paper=fixtureB unit=2/2", completed.stdout)
        self.assertIn("[finish] status=passed", completed.stdout)

        marker = json.loads((destination / "EXPERIMENTAL_V2.json").read_text())
        self.assertEqual(marker["contract"], CONTRACT)
        self.assertEqual(marker["purpose"], "fixed-cohort-source-first-v2-evaluation")
        self.assertTrue((destination / "page_ledger_v2.jsonl").is_file())
        self.assertTrue((destination / "fixed_cohort_report_v2.json").is_file())

        ledger_rows = [
            json.loads(line)
            for line in (destination / "page_ledger_v2.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(ledger_rows), 42)
        self.assertEqual(len({row["page_id"] for row in ledger_rows}), 42)

        report = json.loads((destination / "fixed_cohort_report_v2.json").read_text())
        self.assertEqual(report["papers_selected"], 2)
        self.assertEqual(report["pages_total"], 42)
        self.assertEqual(report["eligible_text_pages"], 42)
        self.assertEqual(report["source_first_passed_pages"], 14)
        self.assertEqual(report["accepted_complex_pages"], 2)
        self.assertEqual(report["accepted_two_column_pages"], 2)
        self.assertEqual(report["accepted_source_first_verifier_exact_pages"], 14)
        self.assertEqual(report["source_first_yield"], round(14 / 42, 8))
        self.assertEqual(report["accepted_source_first_verifier_exact_rate"], 1.0)
        self.assertEqual(
            report["target"],
            {
                "source_first_passed_over_eligible_text_page_gt_0_30": True,
                "accepted_complex_gt_0": True,
                "accepted_two_column_gt_0": True,
                "accepted_source_first_verifier_exact_rate_1_0": True,
                "passed": True,
            },
        )

    def test_policy_mismatch_is_rejected_without_creating_output(self) -> None:
        first = self.make_paper("fixtureA", {1, 2, 3, 4}, {1})
        second = self.make_paper("fixtureB", {1, 2, 3}, {2})
        report_path = second / "validation_report_v2.json"
        report = json.loads(report_path.read_text())
        report["layout_policy_version"] = "different_layout_policy"
        write_json(report_path, report)

        completed, destination = self.run_command(first, second)
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("contract/probe/layout policy", completed.stdout)
        self.assertFalse(destination.exists())

    def test_pages_total_mismatch_is_rejected(self) -> None:
        first = self.make_paper("fixtureA", {1, 2, 3, 4}, {1})
        report_path = first / "validation_report_v2.json"
        report = json.loads(report_path.read_text())
        report["pages_total"] = 20
        write_json(report_path, report)

        completed, destination = self.run_command(first)
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("pages_total=20 != ledger rows=21", completed.stdout)
        self.assertFalse(destination.exists())

    def test_stable_output_root_is_rejected_before_creation(self) -> None:
        first = self.make_paper("fixtureA", {1, 2, 3, 4}, {1})
        stable = self.root / "stable_v10"
        stable.mkdir()
        (stable / "pipeline_report.json").write_text(
            json.dumps({"pipeline_version": "source_bins_to_source_first_confusable_verl_v10_list_payload_math_minus"}),
            encoding="utf-8",
        )
        destination = stable / "nested_cohort"
        completed, _ = self.run_command(
            first,
            stable_output_roots=(stable,),
            output=destination,
        )
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("overlaps declared stable output", completed.stdout)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
