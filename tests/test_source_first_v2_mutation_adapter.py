from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from arxiv_source_first_v2.contracts import (
    EXPERIMENTAL_CONTRACT,
    EXPERIMENTAL_SCHEMA_VERSION,
    PIPELINE_VERSION,
    ContractError,
    write_experimental_marker,
)
from arxiv_source_first_v2.mutation_adapter import (
    ADAPTER_POLICY_VERSION,
    V1_FOUR_MUTATION_TARGET_PROBABILITY,
    MutationRunConfig,
    load_v2_mutation_inputs,
    map_sidecar_source_units,
    run_mutation_export,
    validate_path_isolation,
    validate_v1_mutation_pairs,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class SourceFirstV2MutationAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_source_first_root(
        self,
        *,
        explicit_unit_ids: bool = False,
        bad_markdown_hash: bool = False,
    ) -> Path:
        root = self.root / "source_first"
        write_experimental_marker(root)
        paper_id = "2601.00001v2"
        paper_root = root / "papers" / paper_id
        source_root = paper_root / "source_clean"
        source_root.mkdir(parents=True)
        (source_root / "main.tex").write_text(
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "Unique prose words remain safely visible here.\n"
            "\\end{document}\n",
            encoding="utf-8",
        )
        clean_pdf = paper_root / "build_clean" / "main.pdf"
        clean_pdf.parent.mkdir(parents=True)
        clean_pdf.write_bytes(b"fixture-pdf")
        markdown = "Unique prose words remain safely visible here.\n"
        markdown_path = paper_root / "pages" / "page_0001.md"
        markdown_path.parent.mkdir(parents=True)
        markdown_path.write_text(markdown, encoding="utf-8")
        image_path = paper_root / "pages" / "page_0001.png"
        image_path.write_bytes(b"fixture-png")
        units = [
            {
                "unit_id": "src-0000001",
                "kind": "paragraph",
                "source_paragraph_id": "sp-fixture",
                "source_file": "main.tex",
                "source_lines": [3, 3],
                "source_line_numbers": [3],
                "raw_latex": markdown.strip(),
                "markdown": markdown.strip(),
            }
        ]
        write_jsonl(paper_root / "source_units.jsonl", units)
        data_id = f"{paper_id}_page_0001_sfspanv2"
        sidecar = {
            "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
            "contract": EXPERIMENTAL_CONTRACT,
            "data_id": data_id,
            "paper_id": paper_id,
            "page_number": 1,
            "status": "passed",
            "rejection_reasons": [],
            "generation_source": "latex_source",
            "page_provenance": "compiled_source_metadata_span_graph",
            "pdf_role": "independent_verifier_only",
            "layout_bucket": "single_column",
            "clean": True,
            "eligible_text_page": True,
            "source_first_passed": True,
            "source_first_verifier_exact": True,
            "edit_accepted": False,
            "verifier_exact": False,
            "source_fragment_ids": ["src-0000001-whole"],
            "verifier": {
                "contract_version": 4,
                "status": "passed",
                "exact_ordered_token_match": True,
                "exact_ordered_character_stream_match": True,
            },
            "markdown": "pages/page_0001.md",
            "image": "pages/page_0001.png",
            "markdown_sha256": (
                "0" * 64
                if bad_markdown_hash
                else __import__("hashlib").sha256(markdown.encode()).hexdigest()
            ),
        }
        if explicit_unit_ids:
            sidecar["source_unit_ids"] = ["src-0000001"]
        write_json(paper_root / "pages" / "page_0001.json", sidecar)
        write_jsonl(paper_root / "pages_passed.jsonl", [sidecar])
        write_jsonl(
            paper_root / "page_ledger_v2.jsonl",
            [
                {
                    "page_id": data_id,
                    "paper_id": paper_id,
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
            ],
        )
        write_json(
            paper_root / "validation_report_v2.json",
            {
                "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
                "contract": EXPERIMENTAL_CONTRACT,
                "status": "passed",
                "paper_id": paper_id,
                "main_tex": "main.tex",
                "compile_engine": "pdflatex",
                "clean_pdf": str(clean_pdf),
                "pages_total": 1,
                "pages_passed": 1,
                "reference_removal": {
                    "status": "passed",
                    "residuals": [],
                    "files": [],
                },
                "figure_policy": "drop_figures",
                "figure_removal": {"status": "passed"},
            },
        )
        ledger = [
            {
                "page_id": data_id,
                "paper_id": paper_id,
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
        write_jsonl(root / "page_ledger_v2.jsonl", ledger)
        write_jsonl(
            root / "paper_results_v2.jsonl",
            [{"paper_id": paper_id, "status": "success"}],
        )
        write_json(
            root / "validation_report_v2.json",
            {
                "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
                "contract": EXPERIMENTAL_CONTRACT,
                "pipeline_version": PIPELINE_VERSION,
                "status": "passed",
                "pages_total": 1,
                "pages_passed": 1,
                "pdf_used_for_generation": False,
                "pdf_used_for_verification": True,
            },
        )
        return root

    def test_loader_accepts_explicit_ids_and_old_unique_longest_prefix(self) -> None:
        old_root = self.make_source_first_root(explicit_unit_ids=False)
        old = load_v2_mutation_inputs(old_root)
        self.assertEqual(len(old.rows), 1)
        self.assertEqual(
            old.audit["source_unit_mapping_modes"],
            {"unique_longest_unit_id_prefix": 1},
        )
        self.assertEqual(old.rows[0]["source_unit_ids"], ["src-0000001"])
        self.assertEqual(old.rows[0]["source_paragraph_ids"], ["sp-fixture"])
        self.assertEqual(
            old.rows[0]["source_first_input_policy_version"], ADAPTER_POLICY_VERSION
        )

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        new_root = self.make_source_first_root(explicit_unit_ids=True)
        new = load_v2_mutation_inputs(new_root)
        self.assertEqual(
            new.audit["source_unit_mapping_modes"],
            {"explicit_source_unit_ids": 1},
        )

    def test_loader_rejects_hash_mismatch_into_audit(self) -> None:
        source = self.make_source_first_root(bad_markdown_hash=True)
        loaded = load_v2_mutation_inputs(source)
        self.assertEqual(loaded.rows, ())
        self.assertEqual(loaded.audit["accepted_pages"], 0)
        self.assertEqual(loaded.audit["rejected_pages"], 1)
        self.assertIn(
            "source-derived Markdown SHA-256 mismatch", loaded.audit["reason_counts"]
        )

    def test_longest_prefix_is_exact_and_unknown_or_explicit_unknown_fails(
        self,
    ) -> None:
        units = {"src-1": {}, "src-1-long": {}}
        ids, mode = map_sidecar_source_units(
            {"source_fragment_ids": ["src-1-long-whole"]}, units
        )
        self.assertEqual(ids, ["src-1-long"])
        self.assertEqual(mode, "unique_longest_unit_id_prefix")
        with self.assertRaisesRegex(ContractError, "no source-unit prefix"):
            map_sidecar_source_units({"source_fragment_ids": ["missing-whole"]}, units)
        with self.assertRaisesRegex(ContractError, "unknown"):
            map_sidecar_source_units({"source_unit_ids": ["missing"]}, units)

    def test_path_gate_rejects_input_and_declared_stable_overlap(self) -> None:
        source = self.root / "source"
        output = self.root / "output"
        source.mkdir()
        output.mkdir()
        validate_path_isolation(source_first_root=source, output_dir=output)
        with self.assertRaisesRegex(ContractError, "must be disjoint"):
            validate_path_isolation(
                source_first_root=source, output_dir=source / "edited"
            )
        with self.assertRaisesRegex(ContractError, "stable output"):
            validate_path_isolation(
                source_first_root=source,
                output_dir=output,
                stable_output_roots=[self.root],
            )

    def test_policy_gate_allows_only_v1_three_or_four_character_edits(self) -> None:
        self.assertEqual(V1_FOUR_MUTATION_TARGET_PROBABILITY, 0.6)

        def pair(count: int) -> dict:
            return {
                "pair_id": f"p{count}",
                "mutation_count": count,
                "changes": [
                    {"origin_ans": "case", "ocr_ans": "cose"} for _ in range(count)
                ],
            }

        self.assertEqual(validate_v1_mutation_pairs([pair(3), pair(4)]), [])
        self.assertIn(
            "mutation_count_not_3_or_4:p5:5",
            validate_v1_mutation_pairs([pair(5)]),
        )

    def test_mock_worker_exports_exact_v1_sft_and_verl_schema(self) -> None:
        source = self.make_source_first_root()
        output = self.root / "edited"

        def worker(payload: dict) -> tuple[list[dict], dict]:
            paper_id = payload["paper_id"]
            row = payload["strict_rows"][0]
            output_root = Path(payload["output_dir"])
            pair_id = f"{row['data_id']}_confusable_chaosv4_s83"
            paths = {
                "edited_image": f"data/{pair_id}_edited.png",
                "edited_markdown": f"ground_truths/{pair_id}_edited.md",
                "metadata": f"metadata/{pair_id}.json",
            }
            (output_root / "data").mkdir(parents=True, exist_ok=True)
            (output_root / "ground_truths").mkdir(parents=True, exist_ok=True)
            (output_root / "metadata").mkdir(parents=True, exist_ok=True)
            (output_root / paths["edited_image"]).write_bytes(b"png")
            (output_root / paths["edited_markdown"]).write_text(
                "Vnique prose words remain safely visible here.\n", encoding="utf-8"
            )
            changes = [
                {
                    "ocr_ans": mutated,
                    "origin_ans": original,
                    "bbox": [1, 2, 3, 4],
                }
                for original, mutated in (
                    ("Unique", "Vnique"),
                    ("prose", "prase"),
                    ("words", "wordz"),
                )
            ]
            pair_row = {
                "pair_id": pair_id,
                "data_id": row["data_id"],
                "paper_id": paper_id,
                "arxiv_id": row["arxiv_id"],
                "version": row["version"],
                "page_number": 1,
                **paths,
                "mutation_count": 3,
                "changes": changes,
            }
            write_json(output_root / paths["metadata"], pair_row)
            paper_output = output_root / "papers" / paper_id
            paper_output.mkdir(parents=True, exist_ok=True)
            (paper_output / "paper_edited.pdf").write_bytes(b"pdf")
            result = {
                "status": "passed",
                "paper_id": paper_id,
                "pairs": [pair_row],
            }
            write_json(paper_output / "paper_result.json", result)
            return [pair_row], result

        report = run_mutation_export(
            MutationRunConfig(
                source_first_root=source,
                output_dir=output,
                server_root="/server/edited",
                workers=1,
                latexmk=Path(sys.executable),
                pdftoppm=Path(sys.executable),
                resume=True,
            ),
            worker=worker,
        )
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["mutation_count_distribution"], {3: 1})
        self.assertTrue((output / "papers/2601.00001v2/paper_result.json").is_file())
        self.assertTrue((output / "papers/2601.00001v2/paper_edited.pdf").is_file())

        sft_path = output / report["exports"]["sft"]
        sft = json.loads(sft_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(set(sft), {"images", "conversations"})
        self.assertEqual(set(sft["conversations"][0]), {"from", "value"})
        self.assertEqual(sft["conversations"][0]["from"], "human")
        self.assertEqual(sft["conversations"][1]["from"], "gpt")

        train = [
            json.loads(line)
            for line in (output / "verl_grpo/train.jsonl").read_text().splitlines()
            if line
        ]
        self.assertEqual(len(train), 1)
        verl = train[0]
        self.assertEqual(
            set(verl),
            {
                "data_source",
                "prompt",
                "images",
                "reward_model",
                "extra_info",
                "ability",
            },
        )
        self.assertEqual(set(verl["extra_info"]), {"arxiv_id", "pair_id", "changes"})
        self.assertEqual(
            set(verl["extra_info"]["changes"][0]),
            {"ocr_ans", "origin_ans", "bbox"},
        )
        self.assertNotIn("source_first_v2_provenance", verl)
        self.assertEqual(
            report["source_first_v2_provenance"]["target_mutations_per_page"], [3, 4]
        )
        with self.assertRaisesRegex(ContractError, "resume configuration differs"):
            run_mutation_export(
                MutationRunConfig(
                    source_first_root=source,
                    output_dir=output,
                    server_root="/server/edited",
                    workers=1,
                    seed=84,
                    latexmk=Path(sys.executable),
                    pdftoppm=Path(sys.executable),
                    resume=True,
                ),
                worker=worker,
            )


if __name__ == "__main__":
    unittest.main()
