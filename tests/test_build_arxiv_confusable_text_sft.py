from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_arxiv_confusable_text_sft.py"
SPEC = importlib.util.spec_from_file_location("build_arxiv_confusable_text_sft", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ArxivConfusableTextSftTests(unittest.TestCase):
    def test_english_prompt_and_sample_are_exact_text_copy(self) -> None:
        markdown = "The availobility evidence remains **important**."
        prompt = MODULE.build_prompt(markdown)
        self.assertTrue(prompt.startswith("Please rewrite the document enclosed"))
        self.assertIn("This is not a translation task.", prompt)
        self.assertIn("Change only the number of # characters.", prompt)
        self.assertIn("Preserve the spaces after them exactly.", prompt)
        self.assertEqual(prompt.count("<<<DOCUMENT_START>>>"), 1)
        self.assertEqual(prompt.count("<<<DOCUMENT_END>>>"), 1)
        self.assertEqual(
            prompt.split("<<<DOCUMENT_START>>>\n", 1)[1].rsplit(
                "\n<<<DOCUMENT_END>>>", 1
            )[0],
            markdown,
        )

    def test_latex_conversion_keeps_markup_and_drops_citations_and_bibliography(self) -> None:
        raw = r"""
\section{Introduction}

This is a \textbf{carefully written} paragraph with $x_i$ and prior work
\citep{smith2020}. The next sentence remains visible.

\begin{thebibliography}{9}
\bibitem{smith2020} Hidden reference text.
\end{thebibliography}
"""
        cleaned = MODULE.remove_bibliography_tail(MODULE.strip_tex_comments(raw))
        self.assertNotIn("Hidden reference", cleaned)
        markdown = MODULE.convert_fragment(cleaned)
        self.assertIn("# Introduction", markdown)
        self.assertIn("**carefully written**", markdown)
        self.assertIn("$x_i$", markdown)
        self.assertNotIn("smith2020", markdown)
        self.assertNotIn("citep", markdown)
        with self.assertRaisesRegex(MODULE.RejectedSource, "visible_reference_command:ref"):
            MODULE.convert_fragment(r"Figure~\ref{fig:one} shows the complete result.")

    def test_mutations_are_same_length_recorded_and_outside_math(self) -> None:
        source = " ".join(["availability"] * 100) + " $availability + ongoing$"
        edited, changes = MODULE.mutate_markdown(
            source,
            response_tokens=1_500,
            rng=__import__("random").Random(83),
        )
        self.assertEqual(len(edited), len(source))
        self.assertEqual(len(changes), 10)
        self.assertIn("$availability + ongoing$", edited)
        self.assertEqual(len({change["char_offset"] for change in changes}), len(changes))
        for change in changes:
            start = change["char_offset"]
            end = change["char_end"]
            self.assertEqual(edited[start:end], change["ocr_ans"])
            self.assertEqual(len(change["origin_ans"]), len(change["ocr_ans"]))

    def _text_block(self, markdown: str, *, source_start: int = 0) -> object:
        return MODULE.TextBlock(
            source_file="main.tex",
            source_start=source_start,
            source_end=source_start + len(markdown),
            line_start=1,
            line_end=markdown.count("\n") + 1,
            markdown=markdown,
        )

    def test_rewrite_heading_levels_covers_levels_spaces_and_only_block_starts(self) -> None:
        blocks = []
        expected_blocks = []
        expected_offsets = []
        input_offset = 0
        for level in range(1, 7):
            for spaces in range(4):
                heading = "#" * level + " " * spaces + (
                    '<table data-level="%d"><tr><td>\\LaTeX</td></tr></table>'
                    % level
                )
                markdown = heading + "\n###### must remain on a later line\nplain text"
                blocks.append(self._text_block(markdown, source_start=input_offset))
                expected_offsets.append((input_offset, level, spaces))
                expected_blocks.append(markdown)
                input_offset += len(markdown) + 2

        # A hash later in a block, an inline hash, and a block without any
        # heading are all outside the rewrite contract.
        non_heading = "prefix # inline\n## later line must remain\nHTML <b>ok</b>"
        blocks.append(self._text_block(non_heading, source_start=input_offset))
        expected_blocks.append(non_heading)
        input_offset += len(non_heading) + 2
        no_hashes = "ordinary text with no heading\n\\alpha + 1"
        blocks.append(self._text_block(no_hashes, source_start=input_offset))
        expected_blocks.append(no_hashes)

        markdown = "\n\n".join(block.markdown for block in blocks)
        rewritten, changes = MODULE.rewrite_heading_levels(
            markdown,
            blocks=blocks,
            rng=random.Random(20260907),
        )
        self.assertEqual(len(changes), 24)
        self.assertTrue(all(set(change) == {"input_offset", "from_level", "to_level"} for change in changes))

        by_offset = {change["input_offset"]: change for change in changes}
        self.assertEqual(set(by_offset), {item[0] for item in expected_offsets})
        for block_index, (offset, level, spaces) in enumerate(expected_offsets):
            change = by_offset[offset]
            self.assertEqual(change["from_level"], level)
            self.assertIn(change["to_level"], (1, 2, 3, 4))
            self.assertNotEqual(change["to_level"], level)
            expected_blocks[block_index] = (
                "#" * change["to_level"]
                + " " * spaces
                + expected_blocks[block_index][level + spaces :]
            )

        self.assertEqual(rewritten, "\n\n".join(expected_blocks))
        self.assertIn('<table data-level="6"><tr><td>\\LaTeX</td></tr></table>', rewritten)
        self.assertIn("###### must remain on a later line", rewritten)
        self.assertIn("prefix # inline\n## later line must remain", rewritten)

        repeated, repeated_changes = MODULE.rewrite_heading_levels(
            markdown,
            blocks=blocks,
            rng=random.Random(20260907),
        )
        self.assertEqual((repeated, repeated_changes), (rewritten, changes))

    def test_rewrite_heading_levels_without_heading_only_adds_fence(self) -> None:
        body = "plain text <table><tr><td>HTML</td></tr></table>\n\\alpha + 1"
        block = self._text_block(body)
        rewritten, changes = MODULE.rewrite_heading_levels(
            body,
            blocks=[block],
            rng=random.Random(11),
        )
        self.assertEqual(rewritten, body)
        self.assertEqual(changes, [])
        self.assertEqual(
            MODULE.RESPONSE_PREFIX + rewritten + MODULE.RESPONSE_SUFFIX,
            "```markdown\n" + body + "\n```",
        )

    def test_make_sample_keeps_a_mutations_and_maps_words_into_fenced_b(self) -> None:
        body_words = [
            "availability",
            "methodological",
            "contribution",
            "demographic",
            "ongoing",
            "scientific",
            "evaluation",
            "observation",
            "representation",
            "generalization",
        ]
        blocks = [
            self._text_block("# 1\n" + " ".join(body_words * 4), source_start=10),
            self._text_block("## 2\n" + " ".join(body_words * 4), source_start=200),
            self._text_block("###### 3\n" + " ".join(body_words * 4), source_start=400),
        ]
        markdown = "\n\n".join(block.markdown for block in blocks)
        stem = "2601.00001v1"
        config = MODULE.WorkerConfig(
            fingerprint="heading-rewrite-test",
            seed=83,
            tokenizer="simple",
            tokenizer_local_only=True,
            trust_remote_code=False,
            length_buckets=(MODULE.LengthBucket(1, 10_000, 1.0),),
            min_response_tokens=1,
            max_response_tokens=10_000,
            mutation_word_ratio=0.10,
            min_mutations=3,
            max_mutations=0,
            max_samples_per_paper=0,
            temp_root=None,
            resume=True,
            retry_failed=False,
        )
        counter = MODULE.SimpleTokenCounter()
        row = {"arxiv_id": "2601.00001", "version": "v1", "categories": ["cs.CL"]}
        sample = MODULE.make_sample(
            row=row,
            stem=stem,
            markdown=markdown,
            clean_tokens=counter.count(markdown),
            blocks=blocks,
            counter=counter,
            config=config,
        )

        answer = sample["messages"][1]["content"]
        self.assertTrue(answer.startswith(MODULE.RESPONSE_PREFIX))
        self.assertTrue(answer.endswith(MODULE.RESPONSE_SUFFIX))
        modified_body = answer[len(MODULE.RESPONSE_PREFIX) : -len(MODULE.RESPONSE_SUFFIX)]
        prompt = sample["messages"][0]["content"]
        a = prompt.split("<<<DOCUMENT_START>>>\n", 1)[1].rsplit(
            "\n<<<DOCUMENT_END>>>", 1
        )[0]

        mutation_rng = random.Random(
            MODULE.stable_seed(
                config.seed,
                stem,
                blocks[0].source_file,
                blocks[0].source_start,
                blocks[-1].source_end,
            )
        )
        expected_a, expected_changes = MODULE.mutate_markdown(
            markdown,
            response_tokens=counter.count(markdown),
            rng=mutation_rng,
            mutation_word_ratio=config.mutation_word_ratio,
            min_mutations=config.min_mutations,
            max_mutations=config.max_mutations,
        )
        self.assertEqual(a, expected_a)
        self.assertEqual(len(a), len(markdown))

        heading_rng = random.Random(
            MODULE.stable_seed(
                config.seed,
                stem,
                blocks[0].source_file,
                blocks[0].source_start,
                blocks[-1].source_end,
                MODULE.HEADING_POLICY_VERSION,
            )
        )
        expected_body, expected_heading_changes = MODULE.rewrite_heading_levels(
            expected_a,
            blocks=blocks,
            rng=heading_rng,
        )
        self.assertEqual(modified_body, expected_body)
        self.assertEqual(sample["extra_info"]["heading_changes"], expected_heading_changes)

        actual_changes = sample["extra_info"]["changes"]
        self.assertEqual(len(actual_changes), len(expected_changes))
        heading_deltas = sample["extra_info"]["heading_changes"]
        self.assertGreaterEqual(len(heading_deltas), 3)
        self.assertTrue(
            any(
                change["input_char_offset"] > heading_deltas[1]["input_offset"]
                for change in actual_changes
            )
        )
        for actual, expected in zip(actual_changes, expected_changes):
            self.assertEqual(actual["input_char_offset"], expected["char_offset"])
            self.assertEqual(actual["input_char_end"], expected["char_end"])
            self.assertEqual(actual["origin_ans"], expected["origin_ans"])
            self.assertEqual(actual["ocr_ans"], expected["ocr_ans"])
            self.assertEqual(
                a[actual["input_char_offset"] : actual["input_char_end"]],
                actual["ocr_ans"],
            )
            self.assertEqual(
                answer[actual["char_offset"] : actual["char_end"]],
                actual["ocr_ans"],
            )
            expected_shift = len(MODULE.RESPONSE_PREFIX) + sum(
                heading["to_level"] - heading["from_level"]
                for heading in heading_deltas
                if heading["input_offset"] < actual["input_char_offset"]
            )
            self.assertEqual(actual["char_offset"], actual["input_char_offset"] + expected_shift)
            self.assertEqual(actual["char_end"], actual["input_char_end"] + expected_shift)

    def test_validator_rejects_body_space_newline_and_missing_fence_changes(self) -> None:
        body_words = "availability methodological contribution demographic ongoing scientific"
        blocks = [
            self._text_block("# 1\n" + (body_words + " ") * 8, source_start=10),
            self._text_block("## 2\n" + (body_words + " ") * 8, source_start=200),
        ]
        markdown = "\n\n".join(block.markdown for block in blocks)
        config = MODULE.WorkerConfig(
            fingerprint="validator-heading-rewrite-test",
            seed=83,
            tokenizer="simple",
            tokenizer_local_only=True,
            trust_remote_code=False,
            length_buckets=(MODULE.LengthBucket(1, 10_000, 1.0),),
            min_response_tokens=1,
            max_response_tokens=10_000,
            mutation_word_ratio=0.10,
            min_mutations=3,
            max_mutations=0,
            max_samples_per_paper=0,
            temp_root=None,
            resume=True,
            retry_failed=False,
        )
        counter = MODULE.SimpleTokenCounter()
        sample = MODULE.make_sample(
            row={"arxiv_id": "2601.00001", "version": "v1"},
            stem="2601.00001v1",
            markdown=markdown,
            clean_tokens=counter.count(markdown),
            blocks=blocks,
            counter=counter,
            config=config,
        )
        MODULE.validate_sample(sample, max_response_tokens=config.max_response_tokens)
        answer = sample["messages"][1]["content"]
        body = answer[len(MODULE.RESPONSE_PREFIX) : -len(MODULE.RESPONSE_SUFFIX)]

        body_index = body.index("\n") + 1
        changed_body_char = "X" if body[body_index] != "X" else "Y"
        body_changed = body[:body_index] + changed_body_char + body[body_index + 1 :]
        space_changed = body.replace("# 1", "#  1", 1)
        newline_index = body.index("\n")
        newline_changed = body[:newline_index] + " " + body[newline_index + 1 :]
        missing_fence = body

        for altered_answer in (
            MODULE.RESPONSE_PREFIX + body_changed + MODULE.RESPONSE_SUFFIX,
            MODULE.RESPONSE_PREFIX + space_changed + MODULE.RESPONSE_SUFFIX,
            MODULE.RESPONSE_PREFIX + newline_changed + MODULE.RESPONSE_SUFFIX,
            missing_fence,
        ):
            altered = copy.deepcopy(sample)
            altered["messages"][1]["content"] = altered_answer
            with self.assertRaises(ValueError):
                MODULE.validate_sample(altered, max_response_tokens=config.max_response_tokens)

    def _write_input(self, root: Path) -> None:
        stem = "2601.00001v1"
        source_dir = root / "source"
        source_dir.mkdir(parents=True)
        vocabulary = [
            "availability",
            "methodological",
            "contribution",
            "demographic",
            "ongoing",
            "scientific",
            "evaluation",
            "observation",
            "representation",
            "analysis",
            "learning",
            "generalization",
            "architecture",
            "experiment",
            "measurement",
            "probabilistic",
            "prediction",
            "accuracy",
            "dataset",
            "research",
        ]
        paragraphs = []
        for paragraph_index in range(35):
            words = [vocabulary[(paragraph_index + offset) % len(vocabulary)] for offset in range(45)]
            paragraphs.append(" ".join(words).capitalize() + ".")
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\section{Long Corpus}\n\n"
            + "\n\n".join(paragraphs)
            + "\n\\begin{thebibliography}{9}\n"
            "\\bibitem{x} reference content must be excluded entirely.\n"
            "\\end{thebibliography}\n"
            "\\end{document}\n"
        )
        (source_dir / "main.tex").write_text(tex, encoding="utf-8")
        archive = root / "papers" / stem / "source_archive.bin"
        archive.parent.mkdir(parents=True)
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(source_dir / "main.tex", arcname="main.tex")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        row = {
            "arxiv_id": "2601.00001",
            "version": "v1",
            "stem": stem,
            "status": "passed",
            "license_name": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "archive": f"papers/{stem}/source_archive.bin",
            "sha256": digest,
            "categories": ["cs.CL"],
        }
        (root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def test_pipeline_outputs_available_rows_without_forcing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_dir = root / "output"
            input_root.mkdir()
            self._write_input(input_root)
            args = argparse.Namespace(
                input_root=input_root,
                output_dir=output_dir,
                tokenizer="simple",
                workers=1,
                max_papers=0,
                paper_ids=[],
                max_samples=100,
                max_samples_per_paper=0,
                min_response_tokens=1_000,
                max_response_tokens=1_100,
                mutation_word_ratio=0.10,
                min_mutations=3,
                max_mutations=0,
                shard_size=3,
                write_merged_jsonl=True,
                val_fraction=0.0,
                seed=83,
                split_seed=42,
                temp_root=None,
                allow_all_licenses=False,
                allow_tokenizer_download=False,
                trust_remote_code=False,
                resume=True,
                retry_failed=False,
            )
            summary = MODULE.run_pipeline(args)
            merge = summary["merge"]
            self.assertEqual(summary["status"], "passed")
            self.assertGreater(merge["written_samples"], 0)
            self.assertLess(merge["written_samples"], 100)
            self.assertEqual(merge["written_samples"], merge["available_unique_samples"])
            self.assertFalse(merge["target_reached"])
            self.assertEqual(merge["shortfall"], 100 - merge["written_samples"])
            rows = []
            for path in sorted((output_dir / "train").glob("part-*.jsonl")):
                rows.extend(MODULE.read_jsonl(path))
            self.assertEqual(len(rows), merge["written_samples"])
            merged_rows = list(MODULE.read_jsonl(output_dir / "train.jsonl"))
            self.assertEqual(merged_rows, rows)
            self.assertTrue((output_dir / "val.jsonl").is_file())
            self.assertEqual((output_dir / "val.jsonl").stat().st_size, 0)
            self.assertEqual(
                merge["merged_jsonl"]["train"]["sha256"],
                hashlib.sha256((output_dir / "train.jsonl").read_bytes()).hexdigest(),
            )
            for row in rows:
                self.assertNotIn("images", row)
                answer = row["messages"][1]["content"]
                self.assertTrue(answer.startswith(MODULE.RESPONSE_PREFIX))
                self.assertTrue(answer.endswith(MODULE.RESPONSE_SUFFIX))
                self.assertNotIn("reference content", answer)
                self.assertNotIn("clean_text", row["extra_info"])
                extra = row["extra_info"]
                self.assertEqual(row["ability"], "heading_format_rewrite")
                self.assertIsInstance(extra["heading_changes"], list)
                self.assertEqual(extra["response_text_sha256"], MODULE.sha256_text(answer))
                self.assertEqual(extra["mutation_target"], len(extra["changes"]))
                self.assertGreaterEqual(extra["mutation_word_ratio_achieved"], 0.10)
                self.assertLess(
                    extra["mutation_word_ratio_achieved"],
                    0.10 + 1 / extra["mutation_word_denominator"],
                )
                MODULE.validate_sample(row, max_response_tokens=1_100)

            # A same-config rerun reuses the paper checkpoint and remains exact.
            resumed = MODULE.run_pipeline(args)
            self.assertEqual(resumed["papers"]["reused"], 1)
            self.assertEqual(resumed["merge"]["written_samples"], merge["written_samples"])


if __name__ == "__main__":
    unittest.main()
