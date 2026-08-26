from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "experimental"
    / "verify_arxiv_confusable_from_source_first_v2.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("source_first_v2_edited_verifier_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def stable_guard(*, final: bool = False, hashes: dict[str, str] | None = None) -> dict:
    hashes = MODULE.STABLE_FILE_SHA256 if hashes is None else hashes
    guard = {
        "status": "passed",
        "ok": True,
        "repo_root": str(REPO_ROOT),
        "stable_pipeline_version": MODULE.STABLE_V10_PIPELINE_VERSION,
        "files": {
            path: {
                "status": "passed",
                "expected_sha256": digest,
                "observed_sha256": digest,
            }
            for path, digest in hashes.items()
        },
        "mismatches": [],
    }
    if final:
        guard["final"] = stable_guard(final=False, hashes=hashes)
    return guard


class SourceFirstEvidenceFixture:
    def __init__(self, root: Path, *, stable_hashes: dict[str, str] | None = None) -> None:
        self.root = root
        self.stable_hashes = (
            MODULE.STABLE_FILE_SHA256 if stable_hashes is None else stable_hashes
        )
        self.paper_id = "fixture.v1"
        self.data_id = f"{self.paper_id}_page_0001_sfspanv2"
        self.paper_root = root / "papers" / self.paper_id
        self.markdown = "stone needs urge\n"
        self.markdown_hash = MODULE.sha256_bytes(self.markdown.encode("utf-8"))
        self.ledger_row = {
            "page_id": self.data_id,
            "paper_id": self.paper_id,
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
        self._write()

    def _write(self) -> None:
        self.root.mkdir(parents=True)
        marker = {
            "schema_version": MODULE.EXPERIMENTAL_SCHEMA_VERSION,
            "experiment": MODULE.EXPERIMENT_NAME,
            "contract": MODULE.EXPERIMENTAL_CONTRACT,
            "pipeline_version": MODULE.PIPELINE_VERSION,
            "stable_v10_pipeline_version": MODULE.STABLE_V10_PIPELINE_VERSION,
            "stable_file_sha256": self.stable_hashes,
            "purpose": "fixture",
        }
        write_json(self.root / MODULE.EXPERIMENTAL_MARKER_FILENAME, marker)
        write_jsonl(self.root / "page_ledger_v2.jsonl", [self.ledger_row])
        paper_result = {
            "schema_version": MODULE.EXPERIMENTAL_SCHEMA_VERSION,
            "contract": MODULE.EXPERIMENTAL_CONTRACT,
            "pipeline_version": MODULE.PIPELINE_VERSION,
            "paper_id": self.paper_id,
            "status": "success",
            "stage": "complete",
        }
        write_jsonl(self.root / "paper_results_v2.jsonl", [paper_result])
        aggregate = {
            "schema_version": MODULE.EXPERIMENTAL_SCHEMA_VERSION,
            "contract": MODULE.EXPERIMENTAL_CONTRACT,
            "pipeline_version": MODULE.PIPELINE_VERSION,
            "status": "passed",
            "papers_selected": 1,
            "pages_total": 1,
            "eligible_clean_text_pages": 1,
            "pages_passed": 1,
            "pages_rejected": 0,
            "accepted_exact_verifier_rate": 1.0,
            "page_ledger": str((self.root / "page_ledger_v2.jsonl").resolve()),
            "paper_results": str((self.root / "paper_results_v2.jsonl").resolve()),
            "stable_guard": stable_guard(final=True, hashes=self.stable_hashes),
            "pdf_used_for_generation": False,
            "pdf_used_for_verification": True,
        }
        write_json(self.root / "validation_report_v2.json", aggregate)

        (self.paper_root / "source_clean").mkdir(parents=True)
        (self.paper_root / "source_clean" / "main.tex").write_text(
            "\\documentclass{article}\n\\begin{document}\nstone needs urge\n\\end{document}\n",
            encoding="utf-8",
        )
        write_jsonl(
            self.paper_root / "source_units.jsonl",
            [
                {
                    "unit_id": "src-1",
                    "kind": "paragraph",
                    "source_file": "main.tex",
                    "source_lines": [3, 3],
                    "raw_latex": "stone needs urge",
                    "markdown": "stone needs urge",
                }
            ],
        )
        (self.paper_root / "build_clean").mkdir()
        clean_pdf = self.paper_root / "build_clean" / "main.pdf"
        clean_pdf.write_bytes(b"fixture-pdf")
        pages = self.paper_root / "pages"
        pages.mkdir()
        (pages / "page_0001.md").write_text(self.markdown, encoding="utf-8")
        Image.new("RGB", (100, 100), "white").save(pages / "page_0001.png")
        verifier = {
            "contract_version": MODULE.V2_VERIFIER_CONTRACT_VERSION,
            "status": "passed",
            "match_mode": "exact_visible_character_stream",
            "expected_tokens": 3,
            "observed_tokens": 3,
            "exact_ordered_token_match": True,
            "exact_ordered_character_stream_match": True,
            "expected_sha256": "token-hash",
            "observed_sha256": "token-hash",
            "expected_character_stream_sha256": "character-hash",
            "observed_character_stream_sha256": "character-hash",
            "first_expected_mismatch": None,
            "first_observed_mismatch": None,
            "first_expected_character_mismatch": None,
            "first_observed_character_mismatch": None,
            "experimental_projection": {
                "pdf_text_used_for_ground_truth": False,
                "source_visible_flow": {
                    "pdf_text_used_for_ground_truth": False,
                    "source_only_projection": True,
                    "all_or_nothing": True,
                    "edits_rolled_back": False,
                    "source_markdown_sha256": self.markdown_hash,
                    "projected_markdown_sha256": self.markdown_hash,
                },
            },
        }
        attempt = {
            "status": "passed",
            "shadow_id": "source_atoms",
            "markdown": self.markdown,
            "selected_order": "banded_layout_graph",
            "selected_serialization_policy": "source",
            "fragment_ids": ["src-1-whole"],
            "verifier": verifier,
        }
        sidecar = {
            "schema_version": MODULE.EXPERIMENTAL_SCHEMA_VERSION,
            "contract": MODULE.EXPERIMENTAL_CONTRACT,
            "data_id": self.data_id,
            "paper_id": self.paper_id,
            "page_number": 1,
            "status": "passed",
            "rejection_reasons": [],
            "generation_source": "latex_source",
            "page_provenance": MODULE.V2_PAGE_PROVENANCE,
            "pdf_role": "independent_verifier_only",
            "layout_bucket": "single_column",
            "clean": True,
            "eligible_text_page": True,
            "source_first_passed": True,
            "source_first_verifier_exact": True,
            "edit_accepted": False,
            "verifier_exact": False,
            "selected_shadow_id": "source_atoms",
            "selected_order": "banded_layout_graph",
            "selected_serialization_policy": "source",
            "source_fragment_ids": ["src-1-whole"],
            "shadow_attempts": [attempt],
            "verifier": verifier,
            "markdown": "pages/page_0001.md",
            "image": "pages/page_0001.png",
            "markdown_sha256": self.markdown_hash,
        }
        write_json(pages / "page_0001.json", sidecar)
        write_jsonl(self.paper_root / "pages_passed.jsonl", [sidecar])
        write_jsonl(self.paper_root / "page_ledger_v2.jsonl", [self.ledger_row])
        paper_report = {
            "schema_version": MODULE.EXPERIMENTAL_SCHEMA_VERSION,
            "contract": MODULE.EXPERIMENTAL_CONTRACT,
            "status": "passed",
            "paper_id": self.paper_id,
            "main_tex": "main.tex",
            "clean_pdf": str(clean_pdf.resolve()),
            "pages_total": 1,
            "pages_passed": 1,
            "pages_rejected": 0,
            "accepted_exact_verifier_rate": 1.0,
            "reference_removal": {
                "status": "passed",
                "residuals": [],
                "files": [{"source_file": "main.tex", "residual_markers": []}],
            },
            "figure_policy": "drop_figures",
            "figure_removal": {
                "status": "passed",
                "files": [{"source_file": "main.tex", "status": "passed"}],
            },
            "stable_guard": stable_guard(hashes=self.stable_hashes),
            "pdf_used_for_generation": False,
            "pdf_used_for_verification": True,
        }
        write_json(self.paper_root / "validation_report_v2.json", paper_report)


class FakeDocument:
    def __init__(self, pages: list[object]) -> None:
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class SourceFirstV2EditedVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_source(self) -> SourceFirstEvidenceFixture:
        return SourceFirstEvidenceFixture(self.base / "source_first")

    def test_accepts_complete_source_first_evidence_chain(self) -> None:
        fixture = self.make_source()
        index = MODULE.validate_source_first_root(fixture.root)
        self.assertEqual(set(index), {fixture.data_id})
        self.assertEqual(index[fixture.data_id]["markdown_sha256"], fixture.markdown_hash)

    def test_accepts_legacy_clean_source_first_hashes(self) -> None:
        fixture = SourceFirstEvidenceFixture(
            self.base / "legacy_source_first",
            stable_hashes=MODULE.LEGACY_SOURCE_FIRST_STABLE_FILE_SHA256,
        )
        index = MODULE.validate_source_first_root(fixture.root)
        self.assertEqual(set(index), {fixture.data_id})

    def test_rejects_marker_stable_hash_tampering(self) -> None:
        fixture = self.make_source()
        marker_path = fixture.root / MODULE.EXPERIMENTAL_MARKER_FILENAME
        marker = json.loads(marker_path.read_text())
        marker["stable_file_sha256"][next(iter(MODULE.STABLE_FILE_SHA256))] = "0" * 64
        write_json(marker_path, marker)
        with self.assertRaisesRegex(MODULE.VerificationError, "experimental_marker_mismatch"):
            MODULE.validate_source_first_root(fixture.root)

    def test_rejects_clean_markdown_hash_tampering(self) -> None:
        fixture = self.make_source()
        (fixture.paper_root / "pages" / "page_0001.md").write_text(
            "stone needs urqe\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(MODULE.VerificationError, "clean_markdown_hash_mismatch"):
            MODULE.validate_source_first_root(fixture.root)

    def test_rejects_reference_or_figure_provenance_tampering(self) -> None:
        fixture = self.make_source()
        report_path = fixture.paper_root / "validation_report_v2.json"
        report = json.loads(report_path.read_text())
        report["reference_removal"]["status"] = "disabled"
        write_json(report_path, report)
        with self.assertRaisesRegex(MODULE.VerificationError, "reference_removal_not_passed"):
            MODULE.validate_source_first_root(fixture.root)

        fixture = SourceFirstEvidenceFixture(self.base / "source_first_second")
        report_path = fixture.paper_root / "validation_report_v2.json"
        report = json.loads(report_path.read_text())
        report["figure_policy"] = "keep_figures"
        write_json(report_path, report)
        with self.assertRaisesRegex(MODULE.VerificationError, "figure_policy_not_drop"):
            MODULE.validate_source_first_root(fixture.root)

    def test_rejects_untraceable_source_fragment(self) -> None:
        fixture = self.make_source()
        sidecar_path = fixture.paper_root / "pages" / "page_0001.json"
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["source_fragment_ids"] = ["unknown-fragment"]
        sidecar["shadow_attempts"][0]["fragment_ids"] = ["unknown-fragment"]
        write_json(sidecar_path, sidecar)
        write_jsonl(fixture.paper_root / "pages_passed.jsonl", [sidecar])
        with self.assertRaisesRegex(MODULE.VerificationError, "source_fragment_unit_provenance_mismatch"):
            MODULE.validate_source_first_root(fixture.root)

    def test_dataset_header_requires_v2_mutation_policy_and_audit_counts(self) -> None:
        root = self.base / "dataset"
        root.mkdir()
        audit = {
            "policy_version": MODULE.V2_MUTATION_INPUT_POLICY_VERSION,
            "scanned_pages": 2,
            "accepted_pages": 1,
            "rejected_pages": 1,
        }
        write_json(root / "strict_input_filter_audit.json", audit)
        report = {
            "schema_version": 2,
            "status": "passed",
            "output_mode": "edited_only",
            "clean_assets_copied": False,
            "accepted_pairs": 1,
            "mutation_policy_version": "chaos_visual_v2",
            "selection_policy_version": "page_exact_source_paragraph_v6_rendered_line_spread_current_gt_no_bibliography",
            "strict_input_filter_policy_version": MODULE.V2_MUTATION_INPUT_POLICY_VERSION,
            "bibliography_policy_version": "exclude_bibliography_tail_v1",
            "digits_allowed": False,
            "length_changing_edits_allowed": False,
            "dataset_root": str(root.resolve()),
            "image_path_policy": "absolute_output_dir_v1",
            "strict_input_filter_audit": "strict_input_filter_audit.json",
            "strict_input_pages_scanned": 2,
            "strict_input_pages_accepted": 1,
            "strict_input_pages_rejected": 1,
        }
        write_json(root / "validation_report.json", report)
        pair = {"pair_id": "p1"}
        loaded, _ = MODULE.validate_dataset_header(root, [pair])
        self.assertEqual(loaded["accepted_pairs"], 1)
        report["dataset_root"] = str((self.base / "elsewhere").resolve())
        write_json(root / "validation_report.json", report)
        with self.assertRaisesRegex(MODULE.VerificationError, "dataset_root"):
            MODULE.validate_dataset_header(root, [pair])
        report["dataset_root"] = str(root.resolve())
        report["strict_input_filter_policy_version"] = "v1-policy"
        write_json(root / "validation_report.json", report)
        with self.assertRaisesRegex(MODULE.VerificationError, "input_policy"):
            MODULE.validate_dataset_header(root, [pair])

    def make_pair_fixture(self) -> tuple[Path, dict, dict, dict, object]:
        root = self.base / "edited"
        (root / "metadata").mkdir(parents=True)
        (root / "ground_truths").mkdir()
        (root / "data").mkdir()
        clean_markdown = "stone needs urge\n"
        edited_markdown = "stcne nceds urgc\n"
        clean_md = self.base / "clean.md"
        clean_md.write_text(clean_markdown, encoding="utf-8")
        clean_image = self.base / "clean.png"
        Image.new("RGB", (100, 100), "white").save(clean_image)
        edited_image = root / "data" / "pair_edited.png"
        Image.new("RGB", (100, 100), "black").save(edited_image)
        (root / "ground_truths" / "pair_edited.md").write_text(edited_markdown, encoding="utf-8")
        words = [
            ("stone", "stcne", [1, 1, 11, 11], [0, 5]),
            ("needs", "nceds", [21, 1, 31, 11], [6, 11]),
            ("urge", "urgc", [41, 1, 51, 11], [12, 16]),
        ]
        changes = [
            {
                "ocr_ans": edited,
                "origin_ans": origin,
                "bbox": bbox,
                "from_char": next(left for left, right in zip(origin, edited) if left != right),
                "to_char": next(right for left, right in zip(origin, edited) if left != right),
                "source_file": "main.tex",
                "source_line": 3,
                "source_column": index,
                "markdown_span": span,
            }
            for index, (origin, edited, bbox, span) in enumerate(words)
        ]
        pair = {
            "pair_id": "pair",
            "data_id": "fixture.v1_page_0001_sfspanv2",
            "paper_id": "fixture.v1",
            "arxiv_id": "fixture",
            "version": "v1",
            "page_number": 1,
            "edited_image": "data/pair_edited.png",
            "edited_markdown": "ground_truths/pair_edited.md",
            "metadata": "metadata/pair.json",
            "mutation_count": 3,
            "changes": changes,
            "bibliography_policy_version": "exclude_bibliography_tail_v1",
            "strict_input_filter_policy_version": MODULE.V2_MUTATION_INPUT_POLICY_VERSION,
        }
        metadata = {
            **{key: value for key, value in pair.items() if key != "metadata"},
            "schema_version": 2,
            "status": "passed",
            "mutation_policy_version": "chaos_visual_v2",
            "selection_policy_version": "page_exact_source_paragraph_v6_rendered_line_spread_current_gt_no_bibliography",
            "bibliography_content_present": False,
            "validation": {
                "character_substitutions_only": True,
                "markdown_same_length_after_substitution": True,
                "markdown_character_diff_count": 3,
                "pdf_word_count_unchanged": True,
                "pdf_word_sequence_expected": True,
                "page_count_unchanged": True,
            },
        }
        write_json(root / pair["metadata"], metadata)
        clean = {
            "markdown_path": clean_md,
            "image_path": clean_image,
            "clean_image_size": (100, 100),
            "clean_pdf_path": self.base / "clean.pdf",
        }
        clean["clean_pdf_path"].write_bytes(b"clean")
        paper_result = {
            "edited_pdf": str(root / "papers" / "fixture.v1" / "paper_edited.pdf"),
            "edited_pdf_sha256": "ignored-by-mock",
        }
        clean_page = SimpleNamespace(
            width=100.0,
            height=100.0,
            words=[
                {"text": origin, "x0": bbox[0], "top": bbox[1], "x1": bbox[2], "bottom": bbox[3]}
                for origin, _edited, bbox, _span in words
            ],
        )
        edited_page = SimpleNamespace(
            width=100.0,
            height=100.0,
            words=[
                {"text": edited, "x0": bbox[0], "top": bbox[1], "x1": bbox[2], "bottom": bbox[3]}
                for _origin, edited, bbox, _span in words
            ],
        )

        def one_char_confusion(origin: str, edited: str):
            differences = [(left, right) for left, right in zip(origin, edited) if left != right]
            return differences[0] if len(origin) == len(edited) and len(differences) == 1 else None

        stable = SimpleNamespace(
            char_differences=lambda left, right: [
                (index, a, b) for index, (a, b) in enumerate(zip(left, right)) if a != b
            ]
            if len(left) == len(right)
            else [],
            one_char_confusion=one_char_confusion,
            pdf_words=lambda page: page.words,
            expected_pixel_bbox=lambda word, **_kwargs: [
                int(word["x0"]), int(word["top"]), int(word["x1"]), int(word["bottom"])
            ],
        )
        documents = [FakeDocument([clean_page]), FakeDocument([edited_page])]
        return root, pair, clean, paper_result, (stable, documents)

    def test_pair_verifier_enforces_three_exact_declared_differences(self) -> None:
        root, pair, clean, paper_result, helpers = self.make_pair_fixture()
        stable, documents = helpers
        with mock.patch.object(MODULE, "pair_pdf_path", return_value=self.base / "edited.pdf"), mock.patch.object(
            MODULE.pdfplumber, "open", side_effect=documents
        ):
            errors = MODULE.verify_pair(
                root=root,
                pair=pair,
                clean=clean,
                paper_result=paper_result,
                stable=stable,
            )
        self.assertEqual(errors, [])

    def test_pair_verifier_rejects_undeclared_markdown_change(self) -> None:
        root, pair, clean, paper_result, helpers = self.make_pair_fixture()
        stable, documents = helpers
        path = root / pair["edited_markdown"]
        path.write_text(path.read_text().replace("\n", "!\n"), encoding="utf-8")
        with mock.patch.object(MODULE, "pair_pdf_path", return_value=self.base / "edited.pdf"), mock.patch.object(
            MODULE.pdfplumber, "open", side_effect=documents
        ):
            errors = MODULE.verify_pair(
                root=root,
                pair=pair,
                clean=clean,
                paper_result=paper_result,
                stable=stable,
            )
        self.assertIn("markdown_length_changed", errors)
        self.assertIn("markdown_exact_diff_positions_mismatch", errors)

    def test_export_verifier_checks_full_sft_verl_and_parquet_content(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is unavailable")
        root = self.base / "exports"
        (root / "ground_truths").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "verl_grpo").mkdir()
        markdown = "stcne nceds urgc\n"
        (root / "ground_truths" / "pair.md").write_text(markdown, encoding="utf-8")
        changes = [
            {"ocr_ans": "stcne", "origin_ans": "stone", "bbox": [1, 1, 11, 11]},
            {"ocr_ans": "nceds", "origin_ans": "needs", "bbox": [21, 1, 31, 11]},
            {"ocr_ans": "urgc", "origin_ans": "urge", "bbox": [41, 1, 51, 11]},
        ]
        pair = {
            "pair_id": "pair",
            "paper_id": "fixture.v1",
            "arxiv_id": "fixture",
            "edited_image": "data/pair.png",
            "edited_markdown": "ground_truths/pair.md",
            "changes": changes,
        }
        (root / pair["edited_image"]).write_bytes(b"png")
        image_path = str((root / pair["edited_image"]).resolve())
        sft = {
            "images": [image_path],
            "conversations": [
                {"from": "human", "value": "PROMPT"},
                {"from": "gpt", "value": markdown},
            ],
        }
        verl = {
            "data_source": "chaos_document_ocr",
            "prompt": [{"role": "user", "content": "VERL"}],
            "images": [image_path],
            "reward_model": {"style": "rule", "ground_truth": markdown},
            "extra_info": {
                "arxiv_id": "fixture",
                "pair_id": "pair",
                "changes": changes,
            },
            "ability": "document_ocr",
        }
        write_jsonl(root / "SFT_edited_1.jsonl", [sft])
        write_jsonl(root / "verl_grpo" / "train.jsonl", [verl])
        write_jsonl(root / "verl_grpo" / "val.jsonl", [])
        pq.write_table(pa.Table.from_pylist([verl]), root / "verl_grpo" / "train.parquet")
        pq.write_table(pa.Table.from_pylist([]), root / "verl_grpo" / "val.parquet")
        report = {
            "dataset_root": str(root.resolve()),
            "image_path_policy": "absolute_output_dir_v1",
            "exports": {"sft": "SFT_edited_1.jsonl", "train": 1, "val": 0},
        }
        errors = MODULE.verify_exports(
            root=root,
            pairs=[pair],
            report=report,
            default_prompt="PROMPT",
            verl_prompt="VERL",
        )
        self.assertEqual(errors, [])

        sft["images"] = ["/server/data/data/pair.png"]
        write_jsonl(root / "SFT_edited_1.jsonl", [sft])
        errors = MODULE.verify_exports(
            root=root,
            pairs=[pair],
            report=report,
            default_prompt="PROMPT",
            verl_prompt="VERL",
        )
        self.assertIn("sft_image_path_mismatch", errors)
        sft["images"] = [image_path]

        sft["conversations"][1]["value"] = "tampered"
        write_jsonl(root / "SFT_edited_1.jsonl", [sft])
        errors = MODULE.verify_exports(
            root=root,
            pairs=[pair],
            report=report,
            default_prompt="PROMPT",
            verl_prompt="VERL",
        )
        self.assertIn("sft_conversation_mismatch:pair", errors)


if __name__ == "__main__":
    unittest.main()
