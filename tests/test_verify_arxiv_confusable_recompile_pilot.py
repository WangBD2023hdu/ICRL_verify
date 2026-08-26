from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_arxiv_confusable_recompile_pilot.py"
SPEC = importlib.util.spec_from_file_location("verify_arxiv_confusable_recompile_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VerifyArxivConfusableRecompilePilotTests(unittest.TestCase):
    def test_clean_pdf_path_rebases_after_gt_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "shard_000/papers/paper-v1/synctex_build/main.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"pdf")
            self.assertEqual(
                MODULE.resolve_clean_pdf(
                    root, "paper-v1", "/old/machine/gt/shard_000/papers/paper-v1/synctex_build/main.pdf"
                ),
                pdf.resolve(),
            )

    def test_direct_clean_index_loads_only_requested_pair_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "shard_000/papers/paper-v1/pages/page_0003.json"
            sidecar.parent.mkdir(parents=True)
            sidecar.write_text(json.dumps({"data_id": "paper-v1-page-3"}), encoding="utf-8")
            index = MODULE.load_clean_page_index(
                root,
                [
                    {
                        "paper_id": "paper-v1",
                        "page_number": 3,
                    }
                ],
            )
            self.assertEqual(set(index), {"paper-v1-page-3"})

    def test_verifier_policy_matches_page_exact_builder_contract(self) -> None:
        self.assertEqual(
            MODULE.SELECTION_POLICY_VERSION,
            "page_exact_source_paragraph_v6_rendered_line_spread_current_gt_no_bibliography",
        )
        self.assertEqual(
            MODULE.BIBLIOGRAPHY_POLICY_VERSION,
            "exclude_bibliography_tail_v1",
        )
        self.assertEqual(
            MODULE.STRICT_INPUT_FILTER_POLICY_VERSION,
            "strict_gt_current_contract_v1",
        )

    def test_verifier_finds_bibliography_start_from_all_page_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pages = root / "papers/paper-v1/pages"
            pages.mkdir(parents=True)
            (pages / "page_0001.md").write_text("Body", encoding="utf-8")
            (pages / "page_0002.md").write_text(
                "## Bibliography\n\n[1] Entry", encoding="utf-8"
            )
            (pages / "page_0003.md").write_text("[2] Continued", encoding="utf-8")
            clean_row = {
                "markdown": "papers/paper-v1/pages/page_0001.md",
            }
            self.assertEqual(
                MODULE.bibliography_start_page_for_clean_row(root, clean_row),
                2,
            )

    def test_verifier_ignores_references_in_contents_page(self) -> None:
        contents = "Contents\n\n1 Introduction\n\n2\n\nReferences\n\n9\n\n1"
        self.assertTrue(MODULE.markdown_has_bibliography_heading(contents))
        self.assertTrue(MODULE.markdown_is_table_of_contents_page(contents))

    def test_exports_must_reference_actual_dataset_images(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "data/pair.png"
            markdown = root / "ground_truths/pair.md"
            image.parent.mkdir(parents=True)
            markdown.parent.mkdir(parents=True)
            (root / "verl_grpo").mkdir()
            image.write_bytes(b"png")
            markdown.write_text("mutatcd text\n", encoding="utf-8")
            pair = {
                "pair_id": "pair",
                "arxiv_id": "paper",
                "edited_image": "data/pair.png",
                "edited_markdown": "ground_truths/pair.md",
            }
            sft = {
                "images": [str(image)],
                "conversations": [
                    {"from": "human", "value": "prompt"},
                    {"from": "gpt", "value": "mutatcd text\n"},
                ],
            }
            verl = {
                "data_source": "chaos_document_ocr",
                "prompt": [{"role": "user", "content": MODULE.VERL_PROMPT}],
                "images": [str(image)],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "mutatcd text\n",
                },
                "extra_info": {
                    "arxiv_id": "paper",
                    "pair_id": "pair",
                    "changes": [],
                },
                "ability": "document_ocr",
            }
            (root / "SFT_edited_1.jsonl").write_text(
                json.dumps(sft) + "\n", encoding="utf-8"
            )
            (root / "verl_grpo/train.jsonl").write_text(
                json.dumps(verl) + "\n", encoding="utf-8"
            )
            (root / "verl_grpo/val.jsonl").write_text("", encoding="utf-8")
            pq.write_table(
                pa.Table.from_pylist([verl]), root / "verl_grpo/train.parquet"
            )
            pq.write_table(
                pa.Table.from_pylist([]), root / "verl_grpo/val.parquet"
            )
            report = {
                "exports": {"sft": "SFT_edited_1.jsonl", "train": 1, "val": 0}
            }
            self.assertEqual(MODULE.verify_exports(root, [pair], report), [])

            sft["images"] = ["/server/elsewhere/data/pair.png"]
            (root / "SFT_edited_1.jsonl").write_text(
                json.dumps(sft) + "\n", encoding="utf-8"
            )
            self.assertIn(
                "sft_image_path_mismatch",
                MODULE.verify_exports(root, [pair], report),
            )


if __name__ == "__main__":
    unittest.main()
