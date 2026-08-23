from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_arxiv_source_bins_to_verl.py"
SPEC = importlib.util.spec_from_file_location("run_arxiv_source_bins_to_verl", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SourceBinsToVerlTests(unittest.TestCase):
    def _write_crawl_root(self, root: Path) -> dict[str, object]:
        archive = root / "papers/2601.00001v1/source_archive.bin"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"not-empty-source-archive")
        row: dict[str, object] = {
            "arxiv_id": "2601.00001",
            "version": "v1",
            "stem": "2601.00001v1",
            "status": "passed",
            "license_name": "CC-BY-4.0",
            "archive": "papers/2601.00001v1/source_archive.bin",
            "sha256": MODULE.sha256_file(archive),
        }
        (root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return row

    def _write_valid_source_first(self, root: Path) -> dict[str, object]:
        pages = root / "pages"
        pages.mkdir(parents=True)
        (pages / "page_0001.md").write_text("Alpha beta.\n", encoding="utf-8")
        (pages / "page_0001.png").write_bytes(b"png")
        sidecar: dict[str, object] = {
            "schema_version": MODULE.SOURCE_FIRST_SCHEMA_VERSION,
            "contract": MODULE.SOURCE_FIRST_CONTRACT,
            "probe_policy_version": MODULE.SOURCE_FIRST_PROBE_POLICY_VERSION,
            "shadow_invariant_policy_version": (
                MODULE.SOURCE_FIRST_SHADOW_INVARIANT_POLICY_VERSION
            ),
            "heading_label_policy_version": (
                MODULE.SOURCE_FIRST_HEADING_LABEL_POLICY_VERSION
            ),
            "figure_policy": "drop_figures",
            "status": "passed",
            "markdown": "pages/page_0001.md",
            "image": "pages/page_0001.png",
            "source_probe_ids": ["src-0000001-word-00001"],
            "shadow_invariant": {
                "character_count_equal": True,
                "character_text_equal": True,
                "geometry_equal": False,
                "geometry_role": "diagnostic_only",
            },
            "verifier": {
                "contract_version": MODULE.SOURCE_FIRST_VERIFIER_CONTRACT_VERSION,
                "status": "passed",
                "exact_ordered_token_match": True,
                "exact_ordered_character_stream_match": True,
            },
        }
        (pages / "page_0001.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
        (root / "pages_passed.jsonl").write_text(
            json.dumps(sidecar) + "\n", encoding="utf-8"
        )
        (root / "source_probes.jsonl").write_text(
            json.dumps({"probe_id": sidecar["source_probe_ids"][0]}) + "\n",
            encoding="utf-8",
        )
        (root / "validation_report.json").write_text(
            json.dumps(
                {
                    "schema_version": MODULE.SOURCE_FIRST_SCHEMA_VERSION,
                    "status": "passed",
                    "contract": MODULE.SOURCE_FIRST_CONTRACT,
                    "probe_policy_version": MODULE.SOURCE_FIRST_PROBE_POLICY_VERSION,
                    "shadow_invariant_policy_version": (
                        MODULE.SOURCE_FIRST_SHADOW_INVARIANT_POLICY_VERSION
                    ),
                    "heading_label_policy_version": (
                        MODULE.SOURCE_FIRST_HEADING_LABEL_POLICY_VERSION
                    ),
                    "figure_policy": "drop_figures",
                    "figure_removal": {"status": "passed"},
                    "reference_removal": {"status": "passed"},
                    "localization": {
                        "selected_probe_tier": "paragraph_and_list_tokens"
                    },
                    "verifier_contract_version": (
                        MODULE.SOURCE_FIRST_VERIFIER_CONTRACT_VERSION
                    ),
                    "pages_passed": 1,
                }
            ),
            encoding="utf-8",
        )
        return sidecar

    def test_select_input_rows_accepts_crawler_bin_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_crawl_root(root)
            rows = MODULE.select_input_rows(root, paper_ids=set(), max_papers=0)
            self.assertEqual([row["stem"] for row in rows], ["2601.00001v1"])
            self.assertEqual(
                MODULE.resolve_archive(root, rows[0]),
                (root / "papers/2601.00001v1/source_archive.bin").resolve(),
            )

    def test_select_input_rows_rejects_disallowed_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._write_crawl_root(root)
            row["license_name"] = "arXiv-nonexclusive-distribute"
            (root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no eligible"):
                MODULE.select_input_rows(root, paper_ids=set(), max_papers=0)

    def test_source_first_manifest_contains_absolute_page_and_gt_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper_root = root / "source_first/2601.00001v1"
            pages = paper_root / "pages"
            pages.mkdir(parents=True)
            (pages / "page_0002.md").write_text("Exact source GT.\n", encoding="utf-8")
            (pages / "page_0002.png").write_bytes(b"png")
            page = {
                "data_id": "2601.00001v1_page_0002",
                "markdown": "pages/page_0002.md",
                "image": "pages/page_0002.png",
            }
            (paper_root / "pages_passed.jsonl").write_text(
                json.dumps(page) + "\n", encoding="utf-8"
            )
            manifest_path = root / "source_first_cases.json"
            cases = MODULE.build_source_first_manifest(
                root / "source_first",
                [{"stem": "2601.00001v1", "status": "success"}],
                manifest_path,
            )
            self.assertEqual(cases[0]["pair_id"], page["data_id"])
            self.assertTrue(Path(cases[0]["image"]).is_absolute())
            self.assertTrue(Path(cases[0]["markdown_path"]).is_absolute())
            self.assertEqual(json.loads(manifest_path.read_text()), cases)

    def test_source_first_artifacts_require_word_probe_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._write_valid_source_first(root)
            self.assertTrue(MODULE.source_first_artifacts_valid(root))
            sidecar["contract"] = "source_first_color_v2"
            (root / "pages_passed.jsonl").write_text(
                json.dumps(sidecar) + "\n", encoding="utf-8"
            )
            self.assertFalse(MODULE.source_first_artifacts_valid(root))

    def test_source_first_artifacts_reject_old_contract_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._write_valid_source_first(root)
            report_path = root / "validation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["contract"] = "source_first_color_v4"
            sidecar["contract"] = "source_first_color_v4"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            (root / "pages_passed.jsonl").write_text(
                json.dumps(sidecar) + "\n", encoding="utf-8"
            )
            self.assertFalse(MODULE.source_first_artifacts_valid(root))

            report["contract"] = MODULE.SOURCE_FIRST_CONTRACT
            sidecar["contract"] = MODULE.SOURCE_FIRST_CONTRACT
            report["probe_policy_version"] = "stale-policy"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            (root / "pages_passed.jsonl").write_text(
                json.dumps(sidecar) + "\n", encoding="utf-8"
            )
            self.assertFalse(MODULE.source_first_artifacts_valid(root))

    def test_source_first_artifacts_accept_both_figure_policies_and_probe_fallbacks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._write_valid_source_first(root)
            report_path = root / "validation_report.json"
            for tier in MODULE.SOURCE_FIRST_PROBE_TIERS:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["localization"]["selected_probe_tier"] = tier
                report_path.write_text(json.dumps(report), encoding="utf-8")
                self.assertTrue(MODULE.source_first_artifacts_valid(root), tier)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["figure_policy"] = "keep_figures"
            report["figure_removal"] = {"status": "disabled"}
            sidecar["figure_policy"] = "keep_figures"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            (root / "pages_passed.jsonl").write_text(
                json.dumps(sidecar) + "\n", encoding="utf-8"
            )
            self.assertTrue(MODULE.source_first_artifacts_valid(root))

            report["figure_removal"] = {"status": "passed"}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertFalse(MODULE.source_first_artifacts_valid(root))

    def test_source_first_artifacts_require_known_unique_probe_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._write_valid_source_first(root)
            sidecar["source_probe_ids"] = ["unknown-probe"]
            (root / "pages_passed.jsonl").write_text(
                json.dumps(sidecar) + "\n", encoding="utf-8"
            )
            self.assertFalse(MODULE.source_first_artifacts_valid(root))

            self._write_valid_source_first(root / "fresh")
            probes_path = root / "fresh" / "source_probes.jsonl"
            probe = probes_path.read_text(encoding="utf-8")
            probes_path.write_text(probe + probe, encoding="utf-8")
            self.assertFalse(MODULE.source_first_artifacts_valid(root / "fresh"))

    def test_empty_engine_list_fails_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source_archive.bin"
            archive.write_bytes(b"not-a-tex-archive")
            result = MODULE.build_source_first_paper(
                {
                    "row": {"stem": "2601.00001v1"},
                    "archive": str(archive),
                    "recompile_root": str(root / "recompile"),
                    "source_first_root": str(root / "source_first"),
                    "resume": True,
                    "retry_failed": False,
                    "latex_engines": [],
                }
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "configuration")

    def test_source_first_variants_keep_engine_fallback_adjacent(self) -> None:
        self.assertEqual(
            MODULE.source_first_compile_variants(["pdflatex", "xelatex"]),
            [
                ("pdflatex", True),
                ("pdflatex", False),
                ("xelatex", True),
                ("xelatex", False),
            ],
        )

    def test_changed_archive_does_not_reuse_success_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stem = "2601.00001v1"
            archive = root / "source_archive.bin"
            archive.write_bytes(b"changed-and-invalid-source")
            recompile_paper = root / "recompile" / "papers" / stem
            recompile_paper.mkdir(parents=True)
            self._write_valid_source_first(root / "source_first" / stem)
            old_sha256 = "0" * 64
            (recompile_paper / "metadata.json").write_text(
                json.dumps(
                    {
                        "stem": stem,
                        "status": "success",
                        "pipeline_version": MODULE.PIPELINE_VERSION,
                        "archive_sha256": old_sha256,
                    }
                ),
                encoding="utf-8",
            )
            result = MODULE.build_source_first_paper(
                {
                    "row": {"stem": stem},
                    "archive": str(archive),
                    "recompile_root": str(root / "recompile"),
                    "source_first_root": str(root / "source_first"),
                    "resume": True,
                    "retry_failed": False,
                    "latex_engines": ["pdflatex"],
                    "expected_sha256": None,
                }
            )
            self.assertNotEqual(result.get("resume_state"), "reused_success")
            self.assertEqual(result["archive_sha256"], MODULE.sha256_file(archive))
            self.assertEqual(result["stage"], "extraction")

    def test_old_pipeline_invalidates_same_archive_checkpoint_and_source_first(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stem = "2601.00001v1"
            archive = root / "source_archive.bin"
            archive.write_bytes(b"same-source-archive")
            archive_sha256 = MODULE.sha256_file(archive)
            recompile_paper = root / "recompile" / "papers" / stem
            source_dir = recompile_paper / "source"
            source_dir.mkdir(parents=True)
            (source_dir / "old.tex").write_text("old source", encoding="utf-8")
            old_pipeline = f"{MODULE.PIPELINE_VERSION}_old"
            (recompile_paper / "metadata.json").write_text(
                json.dumps(
                    {
                        "stem": stem,
                        "status": "success",
                        "pipeline_version": old_pipeline,
                        "archive_sha256": archive_sha256,
                    }
                ),
                encoding="utf-8",
            )
            (recompile_paper / "extraction.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "pipeline_version": old_pipeline,
                        "archive_sha256": archive_sha256,
                    }
                ),
                encoding="utf-8",
            )
            source_first_root = root / "source_first" / stem
            self._write_valid_source_first(source_first_root)

            with mock.patch.object(
                MODULE,
                "extract_source",
                side_effect=RuntimeError("forced rebuild"),
            ) as extract_source:
                result = MODULE.build_source_first_paper(
                    {
                        "row": {"stem": stem},
                        "archive": str(archive),
                        "recompile_root": str(root / "recompile"),
                        "source_first_root": str(root / "source_first"),
                        "resume": True,
                        "retry_failed": False,
                        "latex_engines": ["pdflatex"],
                        "expected_sha256": None,
                    }
                )

            extract_source.assert_called_once_with(archive.resolve(), source_dir)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "extraction")
            self.assertEqual(
                result["resume_invalidation_reason"],
                "pipeline_version_mismatch",
            )
            self.assertFalse(source_dir.exists())
            self.assertFalse(source_first_root.exists())
            self.assertFalse((recompile_paper / "extraction.json").exists())
            for key in (
                "stale_extraction_checkpoint_moved_to",
                "stale_source_moved_to",
                "stale_source_first_moved_to",
            ):
                self.assertTrue(Path(result[key]).exists(), key)

    def test_current_pipeline_same_archive_reuses_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stem = "2601.00001v1"
            archive = root / "source_archive.bin"
            archive.write_bytes(b"same-source-archive")
            archive_sha256 = MODULE.sha256_file(archive)
            recompile_paper = root / "recompile" / "papers" / stem
            recompile_paper.mkdir(parents=True)
            stored = {
                "stem": stem,
                "status": "success",
                "pipeline_version": MODULE.PIPELINE_VERSION,
                "archive_sha256": archive_sha256,
            }
            (recompile_paper / "metadata.json").write_text(
                json.dumps(stored), encoding="utf-8"
            )
            self._write_valid_source_first(root / "source_first" / stem)

            with mock.patch.object(MODULE, "extract_source") as extract_source:
                result = MODULE.build_source_first_paper(
                    {
                        "row": {"stem": stem},
                        "archive": str(archive),
                        "recompile_root": str(root / "recompile"),
                        "source_first_root": str(root / "source_first"),
                        "resume": True,
                        "retry_failed": False,
                        "latex_engines": ["pdflatex"],
                        "expected_sha256": None,
                    }
                )

            extract_source.assert_not_called()
            self.assertEqual(result["resume_state"], "reused_success")


if __name__ == "__main__":
    unittest.main()
