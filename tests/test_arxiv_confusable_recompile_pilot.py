from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_arxiv_confusable_recompile_pilot.py"
SPEC = importlib.util.spec_from_file_location("arxiv_confusable_recompile_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ConfusableRecompilePilotTests(unittest.TestCase):
    def test_recompile_paths_rebase_after_corpus_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "papers/paper-v1/source/nested"
            source.mkdir(parents=True)
            main = source / "main.tex"
            main.write_text("\\documentclass{article}", encoding="utf-8")
            resolved_source, relative_main = MODULE.resolve_recompile_source(
                root,
                "paper-v1",
                {
                    "source_dir": "/old/machine/papers/paper-v1/source",
                    "main_tex": "/old/machine/papers/paper-v1/source/nested/main.tex",
                },
            )
            self.assertEqual(resolved_source, (root / "papers/paper-v1/source").resolve())
            self.assertEqual(relative_main, Path("nested/main.tex"))

    def test_clean_pdf_path_rebases_after_gt_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "shard_003/papers/paper-v1/synctex_build/main.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"pdf")
            self.assertEqual(
                MODULE.resolve_clean_pdf(
                    root, "paper-v1", "/old/machine/gt/shard_003/papers/paper-v1/synctex_build/main.pdf"
                ),
                pdf.resolve(),
            )

    def test_confusable_map_is_lowercase_alpha_one_to_one_and_has_no_digits(self) -> None:
        allowed_pairs = {
            ("a", "o"),
            ("c", "e"),
            ("c", "o"),
            ("e", "c"),
            ("g", "q"),
            ("h", "n"),
            ("i", "l"),
            ("l", "i"),
            ("n", "h"),
            ("o", "a"),
            ("o", "c"),
            ("q", "g"),
            ("s", "z"),
            ("u", "v"),
            ("v", "u"),
            ("z", "s"),
        }
        observed = {
            (source, target)
            for source, targets in MODULE.CONFUSABLES.items()
            for target in targets
        }
        self.assertEqual(observed, allowed_pairs)
        for source, target in observed:
            self.assertEqual(len(source), 1)
            self.assertEqual(len(target), 1)
            self.assertTrue(source.islower() and source.isalpha())
            self.assertTrue(target.islower() and target.isalpha())
            self.assertFalse(source.isdigit() or target.isdigit())

    def test_selection_policy_declares_page_exact_contract(self) -> None:
        self.assertEqual(
            MODULE.SELECTION_POLICY_VERSION,
            "page_exact_source_paragraph_v5_fail_closed_current_gt_no_bibliography",
        )
        self.assertEqual(
            MODULE.BIBLIOGRAPHY_POLICY_VERSION,
            "exclude_bibliography_tail_v1",
        )

    def _write_strict_page(
        self,
        root: Path,
        *,
        data_id: str,
        author_contract_version: int = 5,
    ) -> dict[str, object]:
        page_dir = root / f"papers/{data_id}/pages"
        page_dir.mkdir(parents=True)
        markdown_relative = f"papers/{data_id}/pages/page_0001.md"
        image_relative = f"papers/{data_id}/pages/page_0001.png"
        markdown = "Title\n\nExact body.\n"
        markdown_path = root / markdown_relative
        markdown_path.write_text(markdown, encoding="utf-8")
        (root / image_relative).write_bytes(b"not-empty")
        line_id = "p0001-x0000-test-o01"
        metric = {
            "status": "passed",
            "token_missing": 0,
            "token_extra": 0,
            "fivegram_missing": 0,
            "fivegram_extra": 0,
            "anchor_monotonicity": 1.0,
        }
        row: dict[str, object] = {
            "data_id": data_id,
            "arxiv_id": data_id,
            "version": "",
            "page_number": 1,
            "validation_status": "passed",
            "strict_text_contract_version": 2,
            "strict_text_v2_status": "passed",
            "strict_text_v2_failure_reasons": [],
            "author_superscript_contract_version": author_contract_version,
            "footnote_representation": "html_sup",
            "strict_text_contract": {
                "canonical_order_frozen_before_replacement": True,
                "captions_required": True,
                "headers_footers_page_numbers_required": True,
                "page_edge_hyphen_visible_form_required": True,
                "strict_punctuation_hard_gate": True,
                "strict_footnote_structure_hard_gate": True,
                "strict_author_superscript_hard_gate": True,
                "ignored_graphic": 0,
                "footnote_representation": "html_sup",
                "author_superscript_contract_version": author_contract_version,
                "author_superscript_representation": "html_sup",
            },
            "markdown": markdown_relative,
            "image": image_relative,
            "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "line_inventory": {
                "canonical_line_ids": [line_id],
                "lines": [{"line_id": line_id}],
            },
            "strict_text_claims": {
                "status": "passed",
                "canonical_order_match": True,
                "inventory_count": 1,
                "claimed_unique_count": 1,
                "missing_line_ids": [],
                "duplicate_line_ids": [],
                "unknown_line_ids": [],
                "cross_page_line_ids": [],
                "order_inversions": [],
                "noncontiguous_structural_claims": [],
                "empty_structural_claims": [],
                "flattened_claim_line_ids": [line_id],
            },
            "strict_text_ordered_metrics": metric,
            "strict_text_claimed_line_metrics": metric,
            "strict_punctuation_issues": [],
            "inline_markup_validation": {
                "status": "passed",
                "syntax_issues": [],
                "cid_placeholders": 0,
            },
            "source_integration": {
                "heading_numbering": {
                    "strict": True,
                    "lost": 0,
                    "wrong": 0,
                    "ambiguous": 0,
                }
            },
            "footnotes": {
                "status": "passed",
                "representation": "html_sup",
                "total": 0,
                "structured": 0,
                "fallback": 0,
            },
            "author_superscripts": {
                "contract_version": author_contract_version,
                "status": "not_present",
                "plans": 0,
                "superscripts_emitted": 0,
                "markers": [],
                "unmatched_plans": 0,
            },
            "source_blocks": [],
        }
        markdown_path.with_suffix(".json").write_text(
            json.dumps(row), encoding="utf-8"
        )
        return row

    def test_strict_input_filter_accepts_only_current_contract_and_writes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = self._write_strict_page(root, data_id="current")
            stale = self._write_strict_page(
                root,
                data_id="stale",
                author_contract_version=4,
            )
            manifest = root / "pages_strict_text_v2.jsonl"
            manifest.write_text(
                "\n".join(json.dumps(row) for row in (current, stale)) + "\n",
                encoding="utf-8",
            )
            audit_path = root / "strict_input_filter_audit.json"
            accepted = MODULE.load_strict_rows(root, audit_path=audit_path)
            self.assertEqual([row["data_id"] for row in accepted], ["current"])
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["scanned_pages"], 2)
            self.assertEqual(audit["accepted_pages"], 1)
            self.assertEqual(audit["rejected_pages"], 1)
            self.assertEqual(
                audit["reason_counts"]["author_superscript_contract_version_mismatch"],
                1,
            )

    def test_strict_input_filter_rejects_markdown_hash_or_sidecar_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._write_strict_page(root, data_id="drift")
            markdown_path = root / str(row["markdown"])
            markdown_path.write_text("changed after validation", encoding="utf-8")
            reasons = MODULE.strict_input_rejection_reasons(root, row)
            self.assertIn("markdown_sha256_mismatch", reasons)

    def test_bibliography_heading_detector_is_section_only(self) -> None:
        positives = (
            "References",
            "## References",
            "7 References",
            "## 7. Bibliography",
            "Literature Cited:",
            "# Works Cited",
        )
        for value in positives:
            with self.subTest(value=value):
                self.assertTrue(MODULE.markdown_has_bibliography_heading(value))
        negatives = (
            "We use references from prior work.",
            "See References [4] for details.",
            "## Related Work",
            "[1] A. Author. Paper title.",
        )
        for value in negatives:
            with self.subTest(value=value):
                self.assertFalse(MODULE.markdown_has_bibliography_heading(value))

    def test_bibliography_start_and_continuation_pages_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page_dir = root / "papers/paper-v1/pages"
            page_dir.mkdir(parents=True)
            (page_dir / "page_0001.md").write_text("## Results\n\nBody.", encoding="utf-8")
            (page_dir / "page_0002.md").write_text(
                "Conclusion.\n\nReferences\n\n[1] First entry.",
                encoding="utf-8",
            )
            (page_dir / "page_0003.md").write_text(
                "[2] Continuation entry without repeated heading.",
                encoding="utf-8",
            )
            rows = [
                {
                    "data_id": f"paper-v1-page-{page}",
                    "page_number": page,
                    "markdown": f"papers/paper-v1/pages/page_{page:04d}.md",
                }
                for page in (1, 2, 3)
            ]
            accepted, excluded, start_page = MODULE.filter_bibliography_tail_rows(
                root, "paper-v1", rows
            )
            self.assertEqual(start_page, 2)
            self.assertEqual([row["page_number"] for row in accepted], [1])
            self.assertEqual([row["page_number"] for row in excluded], [2, 3])
            self.assertTrue(
                all(row["reason"] == "bibliography_page_excluded" for row in excluded)
            )

    def test_table_of_contents_reference_entry_is_not_bibliography_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page_dir = root / "papers/paper-v1/pages"
            page_dir.mkdir(parents=True)
            (page_dir / "page_0001.md").write_text(
                "Contents\n\n1 Introduction\n\n2\n\nReferences\n\n8\n\n1",
                encoding="utf-8",
            )
            (page_dir / "page_0002.md").write_text("Body", encoding="utf-8")
            (page_dir / "page_0008.md").write_text(
                "References\n\n[1] A. Author, Paper, 2024.\n\n8",
                encoding="utf-8",
            )
            rows = [
                {
                    "data_id": f"paper-v1-page-{page}",
                    "page_number": page,
                    "markdown": f"papers/paper-v1/pages/page_{page:04d}.md",
                }
                for page in (1, 2, 8)
            ]
            accepted, excluded, start_page = MODULE.filter_bibliography_tail_rows(
                root, "paper-v1", rows
            )
            self.assertEqual(start_page, 8)
            self.assertEqual([row["page_number"] for row in accepted], [1, 2])
            self.assertEqual([row["page_number"] for row in excluded], [8])

    def test_markdown_mutations_are_exact_substitutions(self) -> None:
        clean = "The general method compares geometric objects."
        first_start = clean.index("general")
        second_start = clean.index("geometric")
        mutations = [
            MODULE.Mutation(
                original_word="general",
                mutated_word="qeneral",
                from_char="g",
                to_char="q",
                char_index_in_word=0,
                source_file="main.tex",
                source_word_offset=1,
                source_char_offset=1,
                source_line=1,
                source_column=1,
                page_number=1,
                pdf_word_index=1,
                clean_bbox_points=(0, 0, 1, 1),
                markdown_start=first_start,
                markdown_end=first_start + len("general"),
            ),
            MODULE.Mutation(
                original_word="geometric",
                mutated_word="qeometric",
                from_char="g",
                to_char="q",
                char_index_in_word=0,
                source_file="main.tex",
                source_word_offset=20,
                source_char_offset=20,
                source_line=2,
                source_column=1,
                page_number=1,
                pdf_word_index=4,
                clean_bbox_points=(0, 0, 1, 1),
                markdown_start=second_start,
                markdown_end=second_start + len("geometric"),
            ),
        ]
        edited = MODULE.apply_markdown_mutations(clean, mutations)
        self.assertEqual(edited, "The qeneral method compares qeometric objects.")
        self.assertEqual(len(clean), len(edited))
        self.assertEqual(MODULE.markdown_diff_count(clean, edited), 2)

    def test_chaos_visual_policy_covers_reference_typos(self) -> None:
        examples = {
            "author": "authar",
            "general": "gencral",
            "time": "tlme",
            "verify": "verlfy",
        }
        for origin, edited in examples.items():
            differences = [
                (left, right)
                for left, right in zip(origin, edited)
                if left != right
            ]
            self.assertEqual(len(origin), len(edited))
            self.assertEqual(len(differences), 1)
            source, target = differences[0]
            self.assertIn(target, MODULE.CONFUSABLES[source])

    def test_page_word_validator_requires_only_expected_words_to_change(self) -> None:
        clean = [
            MODULE.PdfWord("general", 1, 0, 1, 10, 20, 20),
            MODULE.PdfWord("method", 1, 1, 25, 10, 40, 20),
        ]
        edited = [
            MODULE.PdfWord("qeneral", 1, 0, 1, 10.2, 20, 20.2),
            MODULE.PdfWord("method", 1, 1, 25, 10.2, 40, 20.2),
        ]
        mutation = MODULE.Mutation(
            original_word="general",
            mutated_word="qeneral",
            from_char="g",
            to_char="q",
            char_index_in_word=0,
            source_file="main.tex",
            source_word_offset=0,
            source_char_offset=0,
            source_line=1,
            source_column=1,
            page_number=1,
            pdf_word_index=0,
            clean_bbox_points=(1, 10, 20, 20),
            markdown_start=0,
            markdown_end=7,
        )
        valid, reason, shift = MODULE.validate_page_words(clean, edited, [mutation])
        self.assertTrue(valid)
        self.assertEqual(reason, "passed")
        self.assertAlmostEqual(shift, 0.2)
        invalid = [edited[0], MODULE.PdfWord("mcthod", 1, 1, 25, 10.2, 40, 20.2)]
        valid, reason, _ = MODULE.validate_page_words(clean, invalid, [mutation])
        self.assertFalse(valid)
        self.assertIn("edited_word_sequence_mismatch", reason)

    def test_document_validator_rejects_unrequested_change_on_other_page(self) -> None:
        clean = {
            1: [MODULE.PdfWord("general", 1, 0, 1, 10, 20, 20)],
            2: [MODULE.PdfWord("method", 2, 0, 1, 10, 20, 20)],
        }
        edited = {
            1: [MODULE.PdfWord("qeneral", 1, 0, 1, 10.1, 20, 20.1)],
            2: [MODULE.PdfWord("mcthod", 2, 0, 1, 10.1, 20, 20.1)],
        }
        mutation = MODULE.Mutation(
            original_word="general",
            mutated_word="qeneral",
            from_char="g",
            to_char="q",
            char_index_in_word=0,
            source_file="main.tex",
            source_word_offset=0,
            source_char_offset=0,
            source_line=1,
            source_column=1,
            page_number=1,
            pdf_word_index=0,
            clean_bbox_points=(1, 10, 20, 20),
            markdown_start=0,
            markdown_end=7,
        )
        valid, reason, _ = MODULE.validate_document_words(clean, edited, {1: [mutation]})
        self.assertFalse(valid)
        self.assertIn("document_page_2:edited_word_sequence_mismatch", reason)

        edited[2] = clean[2]
        valid, reason, shift = MODULE.validate_document_words(clean, edited, {1: [mutation]})
        self.assertTrue(valid)
        self.assertEqual(reason, "passed")
        self.assertAlmostEqual(shift, 0.1)

    def test_mutation_selection_uses_page_local_uniqueness(self) -> None:
        words = ["general", "method", "visual", "source"]
        page_words = [
            MODULE.PdfWord(word, 1, index, 10 + index * 50, 10, 45 + index * 50, 20)
            for index, word in enumerate(words)
        ]
        page_words.append(MODULE.PdfWord("tail", 1, len(words), 230, 10, 250, 20))
        source_by_word = {
            word: [
                MODULE.SourceOccurrence(
                    word=word,
                    source_file="main.tex",
                    word_offset=index * 20,
                    source_line=index + 1,
                    source_column=1,
                )
            ]
            for index, word in enumerate(words)
        }
        markdown = " ".join(words)
        selected = MODULE.choose_mutations_for_page(
            row={"data_id": "page-local-test", "page_number": 1},
            clean_markdown=markdown,
            page_words=page_words,
            source_by_word=source_by_word,
            paper_word_vocabulary=set(words) | {"tail"},
            excluded_source_positions=set(),
            seed=83,
        )
        self.assertIn(len(selected), (3, 4))
        self.assertEqual(
            len({(item.source_file, item.source_word_offset) for item in selected}),
            len(selected),
        )

    def test_mutation_selection_disambiguates_repeated_word_by_page_paragraph(self) -> None:
        words = ["general", "method", "visual", "source"]
        page_words = [
            MODULE.PdfWord(word, 1, index, 10 + index * 50, 10, 45 + index * 50, 20)
            for index, word in enumerate(words)
        ]
        page_words.append(MODULE.PdfWord("tail", 1, len(words), 230, 10, 250, 20))
        source_by_word = {
            word: [
                MODULE.SourceOccurrence(
                    word=word,
                    source_file="main.tex",
                    word_offset=index * 40,
                    source_line=index + 1,
                    source_column=1,
                    paragraph_ids=("page-paragraph",),
                ),
                MODULE.SourceOccurrence(
                    word=word,
                    source_file="main.tex",
                    word_offset=index * 40 + 20,
                    source_line=index + 20,
                    source_column=1,
                    paragraph_ids=("other-paragraph",),
                ),
            ]
            for index, word in enumerate(words)
        }
        selected = MODULE.choose_mutations_for_page(
            row={
                "data_id": "paragraph-disambiguation",
                "page_number": 1,
                "source_paragraph_integration": {
                    "source_paragraph_ids": ["page-paragraph"]
                },
            },
            clean_markdown=" ".join(words),
            page_words=page_words,
            source_by_word=source_by_word,
            paper_word_vocabulary=set(words) | {"tail"},
            excluded_source_positions=set(),
            seed=83,
        )
        self.assertIn(len(selected), (3, 4))
        self.assertTrue(all(item.source_line < 20 for item in selected))

    def test_comment_scanner_preserves_escaped_percent(self) -> None:
        self.assertEqual(MODULE.tex_comment_start(r"visible \% value % hidden"), 17)
        self.assertEqual(MODULE.tex_comment_start(r"visible \% value"), 16)

    def test_verl_prompt_and_requested_change_shape(self) -> None:
        self.assertEqual(
            MODULE.VERL_PROMPT,
            "<image>\nPlease transcribe all text in this page image faithfully, "
            "exactly as printed (including any typos).",
        )
        full_change = {
            "ocr_ans": "qeneral",
            "origin_ans": "general",
            "bbox": [10, 20, 30, 40],
            "from_char": "g",
            "to_char": "q",
        }
        projected = {
            "ocr_ans": full_change["ocr_ans"],
            "origin_ans": full_change["origin_ans"],
            "bbox": full_change["bbox"],
        }
        self.assertEqual(set(projected), {"ocr_ans", "origin_ans", "bbox"})

    def test_atomic_write_json_has_no_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            MODULE.atomic_write_json(path, {"status": "passed"})
            self.assertEqual(json.loads(path.read_text()), {"status": "passed"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_prune_unreferenced_pair_artifacts_removes_only_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            for folder, suffix, key in (
                ("data", ".png", "edited_image"),
                ("ground_truths", ".md", "edited_markdown"),
                ("metadata", ".json", "metadata"),
            ):
                target_dir = root / folder
                target_dir.mkdir(parents=True)
                keep = target_dir / f"keep{suffix}"
                stale = target_dir / f"stale{suffix}"
                keep.write_bytes(b"keep")
                stale.write_bytes(b"stale")
                if not pairs:
                    pairs.append({})
                pairs[0][key] = str(keep.relative_to(root))
            result = MODULE.prune_unreferenced_pair_artifacts(root, pairs)
            self.assertEqual(result["removed"], 3)
            self.assertEqual(result["removed_bytes"], 15)
            self.assertTrue((root / "data/keep.png").is_file())
            self.assertFalse((root / "metadata/stale.json").exists())

    def test_accepted_subset_is_valid_even_when_other_papers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("data/edit.png", "data/edit.md", "metadata/edit.json"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            pairs = [
                {
                    "pair_id": "pair-1",
                    "edited_image": "data/edit.png",
                    "edited_markdown": "data/edit.md",
                    "metadata": "metadata/edit.json",
                }
            ]
            self.assertEqual(
                MODULE.validate_accepted_subset(
                    root, pairs, {"train": 1, "val": 0}
                ),
                [],
            )

    def test_accepted_subset_rejects_missing_artifact_or_export_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            issues = MODULE.validate_accepted_subset(
                Path(directory),
                [
                    {
                        "pair_id": "pair-1",
                        "edited_image": "missing.png",
                        "edited_markdown": "missing.md",
                        "metadata": "missing.json",
                    }
                ],
                {"train": 0, "val": 0},
            )
            self.assertIn("export_count_mismatch", issues)
            self.assertEqual(
                sum(issue.startswith("accepted_artifact_missing:") for issue in issues),
                3,
            )


if __name__ == "__main__":
    unittest.main()
