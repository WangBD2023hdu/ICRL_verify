from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "experimental" / "run_arxiv_source_bins_to_verl_v2.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("source_first_v2_batch_runner_test", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


FAKE_BUILDER = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source-dir")
parser.add_argument("--main-tex")
parser.add_argument("--output-dir")
parser.add_argument("--paper-id")
parser.add_argument("--drop-references", action="store_true")
parser.add_argument("--drop-figures", action="store_true")
parser.add_argument("--max-pages", type=int)
parser.add_argument("--dpi", type=int)
parser.add_argument("--compile-timeout", type=int)
parser.add_argument("--engine", choices=("pdflatex", "xelatex", "latex_dvips_ps2pdf"))
parser.add_argument("--latexmk")
parser.add_argument("--pdftoppm")
parser.add_argument("--min-eligible-visible-characters", type=int)
args = parser.parse_args()
output = Path(args.output_dir)
output.mkdir(parents=True)
(output / "fake_invocation.txt").write_text("invoked\n", encoding="utf-8")
complex_layout = args.paper_id.endswith("A")
layouts = ["two_column" if complex_layout else "single_column", "single_column"]
rows = []
for page_number in (1, 2):
    passed = page_number == 1
    rows.append({
        "page_id": f"{args.paper_id}_page_{page_number:04d}_sfspanv2",
        "data_id": f"{args.paper_id}_page_{page_number:04d}_sfspanv2",
        "paper_id": args.paper_id,
        "page_number": page_number,
        "candidate": True,
        "clean": True,
        "eligible_text_page": True,
        "source_first_passed": passed,
        "source_first_verifier_exact": passed,
        "edit_accepted": False,
        "verifier_exact": False,
        "layout": layouts[page_number - 1],
        "status": "passed" if passed else "rejected",
        "rejection_reasons": [] if passed else ["fixture_rejection"],
    })
with (output / "page_ledger_v2.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\n")
report = {
    "schema_version": 1,
    "contract": "arxiv_source_first_v2_anchor_lattice",
    "status": "passed",
    "paper_id": args.paper_id,
    "pages_total": 2,
    "eligible_clean_text_pages": 2,
    "pages_passed": 1,
    "pages_rejected": 1,
    "accepted_complex_layout_pages": 1 if complex_layout else 0,
    "accepted_exact_verifier_rate": 1.0,
}
(output / "validation_report_v2.json").write_text(
    json.dumps(report), encoding="utf-8"
)
'''


class SourceFirstV2BatchRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_inputs(self) -> Path:
        papers = self.root / "input" / "papers"
        papers.mkdir(parents=True)

        archive_paper = papers / "fixtureA"
        archive_paper.mkdir()
        source_file = self.root / "archive_main.tex"
        source_file.write_text(
            "\\documentclass{article}\n\\begin{document}\nArchive A.\n\\end{document}\n",
            encoding="utf-8",
        )
        with tarfile.open(archive_paper / "source_archive.bin", "w") as bundle:
            bundle.add(source_file, arcname="main.tex")

        unpacked = papers / "fixtureB" / "source"
        unpacked.mkdir(parents=True)
        (unpacked / "main.tex").write_text(
            "\\documentclass{article}\n\\begin{document}\nSource B.\n\\end{document}\n",
            encoding="utf-8",
        )
        return papers

    def make_crawler_input(self) -> Path:
        crawler = self.root / "crawler"
        paper = crawler / "papers" / "fixtureA"
        paper.mkdir(parents=True)
        source_file = self.root / "crawler_main.tex"
        source_file.write_text(
            "\\documentclass{article}\n\\begin{document}\nCrawler fixture.\n\\end{document}\n",
            encoding="utf-8",
        )
        archive = paper / "source_archive.bin"
        with tarfile.open(archive, "w") as bundle:
            bundle.add(source_file, arcname="main.tex")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        result = {
            "arxiv_id": "fixture",
            "version": "v1",
            "stem": "fixtureA",
            "status": "passed",
            "license_name": "CC-BY-4.0",
            "archive": "papers/fixtureA/source_archive.bin",
            "bytes": archive.stat().st_size,
            "sha256": digest,
        }
        (crawler / "results.jsonl").write_text(
            json.dumps(result) + "\n", encoding="utf-8"
        )
        return crawler

    def run_fixture(self, papers: Path, output: Path) -> subprocess.CompletedProcess[str]:
        fake_builder = self.root / "fake_builder.py"
        fake_builder.write_text(FAKE_BUILDER, encoding="utf-8")
        command = [
            sys.executable,
            str(RUNNER),
            "--input-root",
            str(papers),
            "--output-dir",
            str(output),
            "--workers",
            "2",
            "--latex-engines",
            "pdflatex",
            "--figure-policy",
            "keep",
            "--heartbeat-seconds",
            "0.1",
            "--builder-script",
            str(fake_builder),
            "--latexmk",
            sys.executable,
            "--pdftoppm",
            sys.executable,
        ]
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )

    def test_multiprocess_fixture_aggregate_and_resume(self) -> None:
        papers = self.make_inputs()
        output = self.root / "experimental_v2"

        first = self.run_fixture(papers, output)
        self.assertEqual(first.returncode, 0, first.stdout)
        self.assertIn("[start]", first.stdout)
        self.assertRegex(first.stdout, r"\[parallel_start\] executor=(process|thread_fallback) workers=2")
        self.assertIn("passed=2", first.stdout)
        self.assertIn("errors=0", first.stdout)
        self.assertTrue((output / "EXPERIMENTAL_V2.json").is_file())

        ledger = read_jsonl(output / "page_ledger_v2.jsonl")
        self.assertEqual(len(ledger), 4)
        report = json.loads((output / "validation_report_v2.json").read_text())
        self.assertEqual(report["papers_selected"], 2)
        self.assertEqual(report["papers_success"], 2)
        self.assertEqual(report["pages_total"], 4)
        self.assertEqual(report["eligible_clean_text_pages"], 4)
        self.assertEqual(report["pages_passed"], 2)
        self.assertEqual(report["accepted_complex_layout_pages"], 1)
        self.assertEqual(report["accepted_two_column_pages"], 1)
        self.assertEqual(report["source_first_yield"], 0.5)
        self.assertEqual(report["accepted_exact_verifier_rate"], 1.0)
        self.assertTrue(report["target"]["passed"])

        invocation_paths = sorted(output.glob("papers/*/fake_invocation.txt"))
        self.assertEqual(len(invocation_paths), 2)
        first_mtimes = {path: path.stat().st_mtime_ns for path in invocation_paths}
        second = self.run_fixture(papers, output)
        self.assertEqual(second.returncode, 0, second.stdout)
        self.assertIn("resume", second.stdout)
        self.assertEqual(
            first_mtimes,
            {path: path.stat().st_mtime_ns for path in invocation_paths},
        )

    def test_dry_run_does_not_create_output(self) -> None:
        papers = self.make_inputs()
        output = self.root / "dry_run_output"
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--input-root",
                str(papers),
                "--output-dir",
                str(output),
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("phase=dry_run", completed.stdout)
        self.assertIn("output_created=false", completed.stdout)
        self.assertFalse(output.exists())

    def test_crawler_results_bin_integration_guards_stable_files_and_marks_output(self) -> None:
        runner = load_runner()
        crawler = self.make_crawler_input()
        output = self.root / "crawler_experimental_v2"
        stable_before = {
            relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in runner.STABLE_FILE_SHA256
        }

        completed = self.run_fixture(crawler, output)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        report = json.loads((output / "validation_report_v2.json").read_text())
        marker = json.loads((output / "EXPERIMENTAL_V2.json").read_text())
        self.assertEqual(report["papers_selected"], 1)
        self.assertEqual(report["papers_success"], 1)
        self.assertTrue(report["stable_guard"]["ok"])
        self.assertEqual(marker["contract"], runner.EXPERIMENTAL_CONTRACT)
        self.assertEqual(marker["pipeline_version"], runner.PIPELINE_VERSION)
        self.assertEqual(marker["stable_file_sha256"], runner.STABLE_FILE_SHA256)
        self.assertEqual(
            stable_before,
            {
                relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
                for relative in runner.STABLE_FILE_SHA256
            },
        )
        batch_state = json.loads((output / "batch_state_v2.json").read_text())
        self.assertEqual(batch_state["status"], "complete")
        self.assertEqual(batch_state["pages_passed"], 1)
        self.assertEqual(batch_state["pages_rejected"], 1)
        self.assertEqual(batch_state["validation_report"], str((output / "validation_report_v2.json").resolve()))

    def test_rejects_output_nested_below_stable_marker(self) -> None:
        runner = load_runner()
        papers = self.make_inputs()
        stable = self.root / "stable_output"
        stable.mkdir()
        (stable / "pipeline_report.json").write_text(
            json.dumps({"pipeline_version": runner.STABLE_V10_PIPELINE_VERSION}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(runner.ContractError, "nested below a stable v10"):
            runner.validate_output_isolation(
                stable / "v2_child",
                papers,
                [],
                create=False,
            )

    def test_rejects_frozen_stable_builder_dispatch(self) -> None:
        runner = load_runner()
        frozen = REPO_ROOT / "scripts" / "run_arxiv_source_bins_to_verl.py"
        with self.assertRaisesRegex(runner.ContractError, "cannot execute frozen stable script"):
            runner.validate_experimental_builder_path(frozen)

    def test_filesystem_paper_ids_filter_before_source_reads_and_inventory(self) -> None:
        runner = load_runner()
        papers = self.root / "filtered" / "papers"
        selected = papers / "2601.00001v2" / "source"
        unselected = papers / "2601.99999v1" / "source"
        selected.mkdir(parents=True)
        unselected.mkdir(parents=True)
        for source, text in (
            (selected, "Selected."),
            (unselected, "Must never be read."),
        ):
            (source / "main.tex").write_text(
                "\\documentclass{article}\n\\begin{document}\n"
                + text
                + "\n\\end{document}\n",
                encoding="utf-8",
            )

        original_has_tex = runner.has_tex_source
        original_has_main = runner.has_main_tex_candidate
        original_inventory = runner.directory_inventory

        def reject_unselected_has_tex(path):
            self.assertNotIn("2601.99999v1", Path(path).parts)
            return original_has_tex(path)

        def reject_unselected_has_main(path):
            self.assertNotIn("2601.99999v1", Path(path).parts)
            return original_has_main(path)

        def reject_unselected_inventory(path, *, label="unknown"):
            self.assertNotIn("2601.99999v1", Path(path).parts)
            return original_inventory(path, label=label)

        with (
            mock.patch.object(runner, "has_tex_source", side_effect=reject_unselected_has_tex),
            mock.patch.object(
                runner, "has_main_tex_candidate", side_effect=reject_unselected_has_main
            ),
            mock.patch.object(
                runner, "directory_inventory", side_effect=reject_unselected_inventory
            ) as inventory,
        ):
            rows = runner.discover_inputs(
                papers,
                paper_ids={"2601.00001"},
                max_papers=0,
            )
        self.assertEqual([row["stem"] for row in rows], ["2601.00001v2"])
        inventory.assert_called_once()
        self.assertEqual(inventory.call_args.kwargs["label"], "2601.00001v2")

    def test_crawler_paper_ids_filter_before_archive_resolution(self) -> None:
        runner = load_runner()
        crawler = self.root / "filtered_crawler"
        crawler.mkdir()
        rows = [
            {
                "arxiv_id": "2601.00001",
                "version": "v2",
                "stem": "2601.00001v2",
                "status": "passed",
                "license_name": "CC-BY-4.0",
            },
            {
                "arxiv_id": "2601.99999",
                "version": "v1",
                "stem": "2601.99999v1",
                "status": "passed",
                "license_name": "CC-BY-4.0",
            },
        ]
        (crawler / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        selected_archive = crawler / "papers" / "2601.00001v2" / "source_archive.bin"
        selected_archive.parent.mkdir(parents=True)
        selected_archive.write_bytes(b"selected archive fixture")

        original_resolver = runner.archive_candidates

        def reject_unselected_resolution(input_root, stem, row):
            self.assertNotEqual(stem, "2601.99999v1")
            return original_resolver(input_root, stem, row)

        with mock.patch.object(
            runner, "archive_candidates", side_effect=reject_unselected_resolution
        ) as resolver:
            selected_rows = runner.discover_inputs(
                crawler,
                paper_ids={"2601.00001"},
                max_papers=0,
            )
        self.assertEqual([row["stem"] for row in selected_rows], ["2601.00001v2"])
        resolver.assert_called_once()

    def test_help_names_supported_inputs_and_aggregate_outputs(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(RUNNER), "--help"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("already-unpacked source directories", completed.stdout)
        self.assertIn("page_ledger_v2.jsonl", completed.stdout)
        self.assertIn("--stable-output-root", completed.stdout)


if __name__ == "__main__":
    unittest.main()
