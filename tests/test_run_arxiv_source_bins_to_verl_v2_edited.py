from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "experimental"
    / "run_arxiv_source_bins_to_verl_v2_edited.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run_v2_edited_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunV2EditedTests(unittest.TestCase):
    def test_isolation_rejects_source_and_edited_overlap(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(module.ContractError, "must not overlap"):
                module.validate_isolation(
                    input_root=root / "input",
                    source_first_root=root / "work" / "source_first_v2",
                    output_dir=root / "work" / "source_first_v2" / "edited",
                    stable_output_roots=[],
                )

    def test_orchestrates_three_experimental_stages_and_writes_report(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            input_root = root / "input"
            input_root.mkdir()
            work_root = root / "work"
            output = root / "edited"
            latexmk = root / "latexmk"
            pdftoppm = root / "pdftoppm"
            latexmk.write_text("fixture", encoding="utf-8")
            pdftoppm.write_text("fixture", encoding="utf-8")
            commands: list[list[str]] = []

            def fake_stage(command, *, phase, heartbeat_seconds):
                commands.append(list(command))
                if phase == "source_first_v2":
                    source_root = work_root / "source_first_v2"
                    source_root.mkdir(parents=True)
                    (source_root / "validation_report_v2.json").write_text(
                        json.dumps(
                            {
                                "status": "passed",
                                "papers_selected": 2,
                                "papers_success": 2,
                                "pages_passed": 7,
                            }
                        ),
                        encoding="utf-8",
                    )
                elif phase == "confusable_mutation_recompile_export":
                    output.mkdir(parents=True)
                    (output / "validation_report.json").write_text(
                        json.dumps(
                            {
                                "status": "passed",
                                "accepted_pairs": 5,
                                "mutation_count_distribution": {"3": 2, "4": 3},
                                "exports": {"train": 4, "val": 1},
                            }
                        ),
                        encoding="utf-8",
                    )
                elif phase == "independent_verifier":
                    (output / "independent_verifier_report.json").write_text(
                        json.dumps({"status": "passed"}), encoding="utf-8"
                    )

            argv = [
                "--input-root",
                str(input_root),
                "--work-root",
                str(work_root),
                "--output-dir",
                str(output),
                "--server-root",
                "/server/v2-edited",
                "--workers",
                "8",
                "--mutation-workers",
                "3",
                "--max-papers",
                "2",
                "--paper-ids",
                "paperA",
                "paperB",
                "--python",
                sys.executable,
                "--latexmk",
                str(latexmk),
                "--pdftoppm",
                str(pdftoppm),
            ]
            with (
                mock.patch.object(module, "run_visible_stage", side_effect=fake_stage),
                mock.patch.object(
                    module,
                    "require_experimental_script",
                    side_effect=lambda path: Path(path).resolve(),
                ),
                mock.patch.object(
                    module,
                    "assert_stable_files",
                    return_value={"ok": True, "status": "passed"},
                ),
            ):
                return_code = module.main(argv)
            self.assertEqual(return_code, 0)
            self.assertEqual(len(commands), 3)
            self.assertIn("--drop-references", commands[0])
            self.assertEqual(commands[1][commands[1].index("--workers") + 1], "3")
            self.assertEqual(commands[1][commands[1].index("--seed") + 1], "83")
            self.assertEqual(commands[1][commands[1].index("--server-root") + 1], "/server/v2-edited")
            report = json.loads((output / "pipeline_report.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["source_first_verified_pages"], 7)
            self.assertEqual(report["edit_pairs"], 5)
            self.assertEqual(report["mutation_count_distribution"], {"3": 2, "4": 3})

    def test_completed_source_first_root_is_reused_without_rewriting_report(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "source_first"
            root.mkdir(parents=True)
            ledger = [
                {
                    "page_id": "paper_page_0001_sfspanv2",
                    "paper_id": "paper",
                    "page_number": 1,
                    "candidate": True,
                    "clean": True,
                    "eligible_text_page": True,
                    "source_first_passed": True,
                    "source_first_verifier_exact": True,
                    "edit_accepted": False,
                    "verifier_exact": False,
                    "layout": "single_column",
                    "status": "passed",
                    "rejection_reasons": [],
                }
            ]
            (root / "page_ledger_v2.jsonl").write_text(
                json.dumps(ledger[0]) + "\n", encoding="utf-8"
            )
            common = {
                "schema_version": module.EXPERIMENTAL_SCHEMA_VERSION,
                "contract": module.EXPERIMENTAL_CONTRACT,
                "pipeline_version": module.SOURCE_FIRST_PIPELINE_VERSION,
                "pages_total": 1,
                "pages_passed": 1,
            }
            (root / "validation_report_v2.json").write_text(
                json.dumps({**common, "status": "passed"}), encoding="utf-8"
            )
            (root / "batch_state_v2.json").write_text(
                json.dumps({**common, "status": "complete"}), encoding="utf-8"
            )
            with mock.patch.object(
                module, "validate_experimental_directory", return_value={"ok": True}
            ):
                report = module.reusable_source_first_report(root)
            self.assertIsNotNone(report)
            self.assertEqual(report["pages_passed"], 1)


if __name__ == "__main__":
    unittest.main()
