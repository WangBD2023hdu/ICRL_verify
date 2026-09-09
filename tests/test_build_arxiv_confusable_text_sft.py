from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from arxiv_canonical_reflow_v4 import weighted_mutation as V5_WEIGHTED_MUTATION

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_arxiv_confusable_text_sft.py"
SPEC = importlib.util.spec_from_file_location("build_arxiv_confusable_text_sft", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RecordingWeightedRng:
    """Small deterministic RNG double for pair-first mutation assertions."""

    def __init__(self, *pairs: tuple[str, str]) -> None:
        self.pairs = list(pairs)
        self.populations: list[tuple[tuple[str, str], ...]] = []
        self.weights: list[tuple[int, ...]] = []
        self.randrange_stops: list[int] = []
        self.position_inputs: list[tuple[int, ...]] = []

    def random(self) -> float:
        return 1.0

    def choices(
        self,
        population: list[tuple[str, str]],
        weights: list[int] | None = None,
        *,
        cum_weights: object = None,
        k: int = 1,
    ) -> list[tuple[str, str]]:
        del cum_weights
        self.populations.append(tuple(population))
        self.weights.append(tuple(weights or ()))
        pair = self.pairs.pop(0) if self.pairs else population[0]
        return [pair for _ in range(k)]

    def randrange(self, stop: int, *args: object) -> int:
        del args
        self.randrange_stops.append(stop)
        return stop - 1

    def shuffle(self, values: list[int]) -> None:
        self.position_inputs.append(tuple(values))
        values.reverse()

    def choice(self, values: list[int]) -> int:
        self.position_inputs.append(tuple(values))
        return values[-1]


class ArxivConfusableTextSftTests(unittest.TestCase):
    def test_english_prompt_and_sample_are_exact_text_copy(self) -> None:
        self.assertEqual(
            MODULE.sha256_text(MODULE.PROMPT_PREFIX + "{A}" + MODULE.PROMPT_SUFFIX),
            "a4e546e28f0f058af71762e285466854e6579ea9f532e9b42916ba88e9902c83",
        )
        markdown = "The availobility evidence remains **important**."
        prompt = MODULE.build_prompt(markdown)
        self.assertTrue(prompt.startswith("Please rewrite the document enclosed"))
        self.assertIn("This is not a translation task.", prompt)
        self.assertIn(
            "randomly choose a different number of # characters from 0 to 4 "
            "(0 means removing all leading # characters).",
            prompt,
        )
        self.assertNotIn(
            "randomly choose a different heading level from 1 to 4.", prompt
        )
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

    def test_no_fence_prompt_changes_only_the_two_approved_sentences(self) -> None:
        expected_prefix = MODULE.PROMPT_PREFIX.replace(
            "these two formatting changes. This is not a translation task.",
            "the following heading-prefix change. This is not a translation task.",
        ).replace(
            "2. Enclose the entire result in a Markdown code fence: start with "
            "```markdown followed by a newline, and end with a newline followed by ```.",
            "2. Output the rewritten document directly. Do not add an outer Markdown code fence.",
        )
        self.assertEqual(MODULE.NO_FENCE_PROMPT_VERSION, "heading_rewrite_boundary_en_v4_no_fence")
        self.assertEqual(MODULE.NO_FENCE_PROMPT_PREFIX, expected_prefix)
        body = (
            "#  heading\n\n"
            "```python\nvalue = 1\n```\n"
            '<table><tr><td>$x_i$ &amp; text</td></tr></table>  '
        )
        prompt = MODULE.build_prompt(body, response_fence="none")
        self.assertEqual(
            prompt,
            MODULE.NO_FENCE_PROMPT_PREFIX + body + MODULE.PROMPT_SUFFIX,
        )
        self.assertEqual(
            prompt.split("<<<DOCUMENT_START>>>\n", 1)[1].rsplit(
                "\n<<<DOCUMENT_END>>>", 1
            )[0],
            body,
        )
        self.assertEqual(MODULE.response_affixes("none"), ("", ""))

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

    def test_weighted_pair_table_is_shared_with_v5(self) -> None:
        self.assertIs(MODULE.PAIR_COUNTS, V5_WEIGHTED_MUTATION.PAIR_COUNTS)
        self.assertIs(MODULE.PAIR_WEIGHTS, V5_WEIGHTED_MUTATION.PAIR_WEIGHTS)
        self.assertEqual(len(MODULE.PAIR_COUNTS), 19)
        self.assertEqual(sum(MODULE.PAIR_WEIGHTS.values()), 1_012)
        self.assertEqual(MODULE.POLICY_FINGERPRINT, V5_WEIGHTED_MUTATION.POLICY_FINGERPRINT)
        self.assertEqual(MODULE.PIPELINE_VERSION, "arxiv_confusable_text_sft_v5_weighted_pairs")
        self.assertNotIn(("0", "e"), MODULE.PAIR_WEIGHTS)
        self.assertEqual(MODULE.PAIR_COUNTS["m"]["n"], 63)
        self.assertEqual(MODULE.PAIR_COUNTS["w"]["v"], 19)
        self.assertNotIn("g", MODULE.PAIR_COUNTS)
        self.assertNotIn(("g", "q"), MODULE.PAIR_WEIGHTS)

    def test_weighted_mutation_uses_pair_weights_not_word_frequency(self) -> None:
        source = " ".join(["many"] * 7 + ["wave"])
        rng = RecordingWeightedRng(("m", "n"), ("m", "n"))
        edited, changes = MODULE.mutate_markdown(
            source,
            response_tokens=1,
            rng=rng,
            mutation_word_ratio=0.25,
            min_mutations=2,
            max_mutations=2,
        )

        self.assertEqual(len(changes), 2)
        self.assertEqual([change["origin_ans"] for change in changes], ["many", "many"])
        self.assertEqual([change["ocr_ans"] for change in changes], ["nany", "nany"])
        self.assertEqual(edited.split().count("nany"), 2)
        self.assertEqual(edited.split().count("many"), 5)
        self.assertTrue(rng.populations)
        observed_weights = {
            pair: weight
            for pair, weight in zip(rng.populations[0], rng.weights[0])
        }
        self.assertEqual(observed_weights[("m", "n")], MODULE.PAIR_WEIGHTS[("m", "n")])
        self.assertEqual(observed_weights[("w", "v")], MODULE.PAIR_WEIGHTS[("w", "v")])
        self.assertEqual(observed_weights[("w", "u")], MODULE.PAIR_WEIGHTS[("w", "u")])
        self.assertEqual(rng.randrange_stops[:2], [7, 6])

    def test_weighted_mutation_selects_each_character_position_and_new_pairs(self) -> None:
        rng = RecordingWeightedRng(("m", "n"))
        edited, changes = MODULE.mutate_markdown(
            "mmmm",
            response_tokens=1,
            rng=rng,
            mutation_word_ratio=1.0,
            min_mutations=1,
            max_mutations=1,
        )
        self.assertEqual(edited, "mmmn")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["origin_ans"], "mmmm")
        self.assertEqual(changes[0]["ocr_ans"], "mmmn")
        self.assertEqual(changes[0]["from_char"], "m")
        self.assertEqual(changes[0]["to_char"], "n")
        self.assertEqual(changes[0]["char_index_in_word"], 3)
        self.assertIn((0, 1, 2, 3), rng.position_inputs)

        pair_rng = RecordingWeightedRng(("m", "n"), ("w", "v"))
        pair_edited, pair_changes = MODULE.mutate_markdown(
            "mmmm wwww gggg",
            response_tokens=1,
            rng=pair_rng,
            mutation_word_ratio=1.0,
            min_mutations=2,
            max_mutations=2,
        )
        self.assertEqual(
            {(change["from_char"], change["to_char"]) for change in pair_changes},
            {("m", "n"), ("w", "v")},
        )
        self.assertNotIn("q", pair_edited)
        self.assertEqual(pair_edited.split(), ["mmmn", "wwwv", "gggg"])

    def test_weighted_mutation_rejects_missing_pairs_and_candidate_exhaustion(self) -> None:
        for source in ("gggg", "zzzz", "123456"):
            with self.assertRaises(MODULE.RejectedSource):
                MODULE.mutate_markdown(
                    source,
                    response_tokens=1,
                    rng=random.Random(7),
                    mutation_word_ratio=1.0,
                    min_mutations=1,
                    max_mutations=1,
                )
        with self.assertRaises(MODULE.RejectedSource):
            MODULE.mutate_markdown(
                "muse",
                response_tokens=1,
                rng=random.Random(7),
                mutation_word_ratio=1.0,
                min_mutations=3,
                max_mutations=3,
            )

    def test_weighted_mutation_protects_digits_and_filters_old_vocabulary_collisions(self) -> None:
        source = "many nany wave vave 2026 model42"
        edited, changes = MODULE.mutate_markdown(
            source,
            response_tokens=1,
            rng=random.Random(17),
            mutation_word_ratio=0.5,
            min_mutations=1,
            max_mutations=0,
        )
        self.assertEqual(len(edited), len(source))
        self.assertEqual(re.findall(r"\d+", edited), re.findall(r"\d+", source))
        vocabulary = {word.lower() for word in re.findall(r"[A-Za-z]{4,}", source)}
        self.assertTrue(changes)
        for change in changes:
            self.assertFalse(any(character.isdigit() for character in change["ocr_ans"]))
            self.assertNotIn(change["ocr_ans"].lower(), vocabulary)

    def _text_block(self, markdown: str, *, source_start: int = 0, kind: str = "text") -> object:
        return MODULE.TextBlock(
            source_file="main.tex",
            source_start=source_start,
            source_end=source_start + len(markdown),
            line_start=1,
            line_end=markdown.count("\n") + 1,
            markdown=markdown,
            kind=kind,
        )

    def test_extracts_source_tables_without_flattening_or_splitting_them(self) -> None:
        raw = r"""\documentclass{article}
\begin{document}
\section{Evaluation}

Visible prose before the source table remains here.
\begin{table}[t]
\centering
\caption{Comparison}
\begin{tabular}{lcc}
\toprule
Method & Score & Formula \\
\midrule
\multirow{2}{*}{\textbf{Baseline}} & 81 & $x_i + 1$ \\

 & \multicolumn{2}{c}{\textit{shared result}} \\
\bottomrule
\end{tabular}
\label{tab:comparison}
\end{table}
Visible prose after the source table remains here.

\begin{tabularx}{\linewidth}{lX}
Research & observation \\
\end{tabularx}
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "main.tex"
            path.write_text(raw, encoding="utf-8")
            blocks, reasons = MODULE.extract_blocks(path, root)
        self.assertFalse(reasons, reasons)
        tables = [block for block in blocks if block.kind == "table"]
        self.assertEqual(len(tables), 2)
        self.assertEqual(tables[0].markdown.count("<table>"), 1)
        self.assertIn('rowspan="2"', tables[0].markdown)
        self.assertIn('colspan="2"', tables[0].markdown)
        self.assertIn("<strong>Baseline</strong>", tables[0].markdown)
        self.assertIn("$x_i + 1$", tables[0].markdown)
        self.assertNotIn("<td></td>", tables[0].markdown)
        self.assertNotIn("Comparison", tables[0].markdown)
        self.assertNotIn("data-", tables[0].markdown)
        self.assertEqual([b.markdown for b in blocks if b.kind == "caption"], ["Comparison"])
        self.assertEqual(
            [block.kind for block in blocks],
            ["text", "text", "caption", "table", "text", "table"],
        )
        for block in tables:
            self.assertIn(r"\begin{tabular", raw.splitlines()[block.line_start - 1])
        self.assertNotIn("Table 1", "\n".join(b.markdown for b in blocks))

    def test_rejected_table_and_figure_contents_never_leak_as_prose(self) -> None:
        raw = r"""Ordinary research text before the excluded blocks.
\begin{table}
\caption{Rejected caption must not survive}
\begin{tabular}{cc}
\unknownmacro{LeakOne} & LeakTwo \\
\end{tabular}
\end{table}
\begin{longtable}{cc}
LeakThree & LeakFour \\
\end{longtable}
\begin{figure}
\begin{tabular}{c}LeakFive\end{tabular}
\end{figure}
Ordinary research text after the excluded blocks.
\begin{table}
\begin{tabular}{c} Unclosed table content must not leak.
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "main.tex"
            path.write_text(raw, encoding="utf-8")
            blocks, reasons = MODULE.extract_blocks(path, root)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(b.kind == "text" for b in blocks))
        self.assertEqual(sum(reasons.values()), 3, reasons)
        self.assertNotIn("Leak", "\n".join(b.markdown for b in blocks))

    def test_mutation_keeps_html_tags_attributes_entities_and_math_exact(self) -> None:
        table = (
            '<table><tbody><tr><td colspan="2">'
            + "availability observation " * 20
            + '&quot; &nbsp; &amp; &#123; &#xAB; $availability$'
            + '</td></tr></tbody></table>'
        )
        edited, changes = MODULE.mutate_markdown(
            table, response_tokens=1500, rng=random.Random(83),
            mutation_word_ratio=0.5, min_mutations=3,
        )
        protected = r"<[^>]+>|&[^;]+;|\$[^$]+\$"
        self.assertEqual(re.findall(protected, table), re.findall(protected, edited))
        self.assertGreater(len(changes), 3)
        self.assertEqual(len(table), len(edited))

    def test_table_html_is_identical_between_a_and_fenced_b(self) -> None:
        table = "<table>\n<tbody><tr><td>###  table heading</td></tr></tbody>\n</table>"
        blocks = [
            self._text_block("#  Heading"),
            self._text_block(table, source_start=100, kind="table"),
            self._text_block("## literal caption", source_start=200, kind="caption"),
        ]
        a = "\n\n".join(b.markdown for b in blocks)
        b, changes = MODULE.rewrite_heading_levels(a, blocks=blocks, rng=random.Random(83))
        self.assertEqual(len(changes), 1)
        self.assertTrue(b.endswith("## literal caption"))
        self.assertEqual(re.search(r"<table>.*</table>", a, re.DOTALL).group(),
                         re.search(r"<table>.*</table>", b, re.DOTALL).group())

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
        self.assertTrue(any(change["to_level"] == 0 for change in changes))

        by_offset = {change["input_offset"]: change for change in changes}
        self.assertEqual(set(by_offset), {item[0] for item in expected_offsets})
        for block_index, (offset, level, spaces) in enumerate(expected_offsets):
            change = by_offset[offset]
            self.assertEqual(change["from_level"], level)
            self.assertIn(change["to_level"], (0, 1, 2, 3, 4))
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

    def test_rewrite_heading_levels_zero_threshold_and_nonzero_choice_pool(self) -> None:
        class ProbeRng:
            def __init__(self) -> None:
                self.random_values: list[float] = []
                self.choice_values: list[tuple[int, ...]] = []

            def random(self) -> float:
                value = 0.0 if not self.random_values else 0.28
                self.random_values.append(value)
                return value

            def choice(self, values: list[int]) -> int:
                candidates = tuple(values)
                self.choice_values.append(candidates)
                return candidates[(len(self.choice_values) - 1) % len(candidates)]

        levels = range(1, 7)
        blocks = [
            self._text_block("#" * level + "   heading", source_start=index * 20)
            for index, level in enumerate(levels)
        ]
        markdown = "\n\n".join(block.markdown for block in blocks)
        rng = ProbeRng()
        rewritten, changes = MODULE.rewrite_heading_levels(
            markdown, blocks=blocks, rng=rng
        )

        self.assertEqual(MODULE.NO_HEADING_PROBABILITY, 0.28)
        self.assertEqual(rng.random_values, [0.0, 0.28, 0.28, 0.28, 0.28, 0.28])
        self.assertEqual(changes[0]["to_level"], 0)
        self.assertEqual(rewritten[: len("   heading")], "   heading")
        self.assertEqual(
            rng.choice_values,
            [(1, 3, 4), (1, 2, 4), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4)],
        )
        self.assertTrue(
            all(
                change["to_level"] in {1, 2, 3, 4}
                and change["to_level"] != change["from_level"]
                for change in changes[1:]
            )
        )
        expected_blocks = [
            "#" * change["to_level"] + "   heading" for change in changes
        ]
        self.assertEqual(rewritten, "\n\n".join(expected_blocks))
        for block_index, change in enumerate(changes):
            expected_prefix = "#" * change["to_level"] + "   heading"
            start = sum(
                len(block.markdown) + 2 for block in blocks[:block_index]
            ) + sum(
                previous["to_level"] - previous["from_level"]
                for previous in changes[:block_index]
            )
            self.assertEqual(rewritten[start : start + len(expected_prefix)], expected_prefix)

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
            seed=2,
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
        self.assertTrue(any(heading["to_level"] == 0 for heading in heading_deltas))
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

        # A removed heading still preserves its following spaces, and the
        # validator must accept the resulting A/B offsets and fenced answer.
        MODULE.validate_sample(sample, max_response_tokens=config.max_response_tokens)

        none_config = replace(config, response_fence="none")
        none_sample = MODULE.make_sample(
            row=row,
            stem=stem,
            markdown=markdown,
            clean_tokens=counter.count(markdown),
            blocks=blocks,
            counter=counter,
            config=none_config,
        )
        none_answer = none_sample["messages"][1]["content"]
        none_prompt = none_sample["messages"][0]["content"]
        none_a = none_prompt.split("<<<DOCUMENT_START>>>\n", 1)[1].rsplit(
            "\n<<<DOCUMENT_END>>>", 1
        )[0]
        self.assertEqual(none_a, a)
        self.assertEqual(none_answer, modified_body)
        self.assertEqual(
            answer,
            MODULE.RESPONSE_PREFIX + none_answer + MODULE.RESPONSE_SUFFIX,
        )
        self.assertFalse(none_answer.startswith(MODULE.RESPONSE_PREFIX))
        self.assertFalse(none_answer.endswith(MODULE.RESPONSE_SUFFIX))
        self.assertEqual(
            none_sample["extra_info"]["heading_changes"],
            sample["extra_info"]["heading_changes"],
        )
        self.assertEqual(none_sample["extra_info"]["response_fence"], "none")
        self.assertEqual(
            none_sample["extra_info"]["prompt_version"],
            MODULE.NO_FENCE_PROMPT_VERSION,
        )
        self.assertNotIn("response_fence", sample["extra_info"])
        self.assertNotEqual(
            none_sample["extra_info"]["sample_id"], sample["extra_info"]["sample_id"]
        )
        self.assertEqual(
            sample["extra_info"]["response_tokens"], counter.count(answer)
        )
        self.assertEqual(
            none_sample["extra_info"]["response_tokens"], counter.count(none_answer)
        )
        self.assertEqual(
            sample["extra_info"]["response_tokens"]
            - none_sample["extra_info"]["response_tokens"],
            counter.count(MODULE.RESPONSE_PREFIX)
            + counter.count(MODULE.RESPONSE_SUFFIX),
        )
        for fenced_change, none_change in zip(
            sample["extra_info"]["changes"], none_sample["extra_info"]["changes"]
        ):
            self.assertEqual(
                fenced_change["input_char_offset"], none_change["input_char_offset"]
            )
            self.assertEqual(
                fenced_change["input_char_end"], none_change["input_char_end"]
            )
            self.assertEqual(
                fenced_change["char_offset"],
                none_change["char_offset"] + len(MODULE.RESPONSE_PREFIX),
            )
            self.assertEqual(
                fenced_change["char_end"],
                none_change["char_end"] + len(MODULE.RESPONSE_PREFIX),
            )
        MODULE.validate_sample(
            none_sample, max_response_tokens=none_config.max_response_tokens
        )

    def test_no_fence_sample_without_headings_is_valid_and_unwrapped(self) -> None:
        words = [
            "availability",
            "methodological",
            "contribution",
            "demographic",
            "ongoing",
            "scientific",
            "evaluation",
            "observation",
        ]
        inner_code = "```python\n# availability stays in this inner fence\nprint(42)\n```"
        table_math = "<table><tr><td>$x_i$</td></tr></table>"
        markdown = "  " + " ".join(words * 20) + "\n\n" + inner_code + "\n" + table_math + "\n  "
        block = self._text_block(markdown)
        config = MODULE.WorkerConfig(
            fingerprint="no-fence-no-heading-test",
            seed=2,
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
            response_fence="none",
        )
        counter = MODULE.SimpleTokenCounter()
        sample = MODULE.make_sample(
            row={"arxiv_id": "2601.00001", "version": "v1"},
            stem="2601.00001v1",
            markdown=markdown,
            clean_tokens=counter.count(markdown),
            blocks=[block],
            counter=counter,
            config=config,
        )
        answer = sample["messages"][1]["content"]
        self.assertEqual(sample["extra_info"]["heading_changes"], [])
        a = sample["messages"][0]["content"].split("<<<DOCUMENT_START>>>\n", 1)[1].rsplit(
            "\n<<<DOCUMENT_END>>>", 1
        )[0]
        self.assertEqual(a, answer)
        self.assertTrue(sample["extra_info"]["changes"])
        self.assertFalse(answer.startswith(MODULE.RESPONSE_PREFIX))
        self.assertFalse(answer.endswith(MODULE.RESPONSE_SUFFIX))
        self.assertIn(inner_code, answer)
        self.assertIn(table_math, answer)
        self.assertTrue(answer.startswith("  "))
        self.assertTrue(answer.endswith("\n  "))
        self.assertEqual(sample["extra_info"]["response_tokens"], counter.count(answer))
        MODULE.validate_sample(sample, max_response_tokens=config.max_response_tokens)

    def test_response_fence_isolated_in_none_fingerprint_but_default_is_legacy(self) -> None:
        buckets = (MODULE.LengthBucket(1, 10_000, 1.0),)
        default_args = self._pipeline_args(
            Path("/tmp/input"), Path("/tmp/output"), response_fence="markdown"
        )
        none_args = self._pipeline_args(
            Path("/tmp/input"), Path("/tmp/output"), response_fence="none"
        )
        legacy_args = copy.deepcopy(default_args)
        delattr(legacy_args, "response_fence")
        default_fingerprint = MODULE.config_fingerprint(default_args, buckets)
        self.assertEqual(default_fingerprint, MODULE.config_fingerprint(legacy_args, buckets))
        self.assertNotEqual(default_fingerprint, MODULE.config_fingerprint(none_args, buckets))

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
        space_index = body.index(" ")
        space_changed = body[:space_index] + "  " + body[space_index + 1 :]
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
        paragraphs.insert(2, r"""\begin{table}
\caption{Research comparison}
\begin{tabular}{lc}
\toprule
Method & Accuracy \\
\midrule
\textbf{Baseline} & $x_i + 1$ \\
Observation & 93.5 \\
\bottomrule
\end{tabular}
\end{table}""")
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
        checkpoint = root / "papers" / stem / "download.json"
        checkpoint.write_text(
            json.dumps({**row, "bytes": archive.stat().st_size, "attempts": 1}) + "\n",
            encoding="utf-8",
        )

    def _write_download_checkpoint(
        self,
        root: Path,
        stem: str,
        *,
        status: str = "passed",
        license_name: str = "CC-BY-4.0",
        archive_state: str = "valid",
        malformed: bool = False,
        arxiv_id: str | None = None,
    ) -> dict[str, object]:
        paper_dir = root / "papers" / stem
        paper_dir.mkdir(parents=True, exist_ok=True)
        archive = paper_dir / "source_archive.bin"
        partial = paper_dir / "source_archive.bin.partial"
        if archive_state == "valid":
            archive.write_bytes((stem + "\nsource archive\n").encode("utf-8"))
        elif archive_state == "empty":
            archive.write_bytes(b"")
        elif archive_state == "partial":
            partial.write_bytes(b"partial source archive")
        elif archive_state != "missing":
            raise AssertionError(f"unknown archive fixture state: {archive_state}")
        row: dict[str, object] = {
            "arxiv_id": arxiv_id or stem.removesuffix("v1"),
            "version": "v1",
            "stem": stem,
            "status": status,
            "license_name": license_name,
            "license_url": "https://example.test/license",
            "archive": f"papers/{stem}/source_archive.bin",
            "categories": ["cs.CL"],
        }
        if archive.is_file():
            row["bytes"] = archive.stat().st_size
            row["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        checkpoint = paper_dir / "download.json"
        checkpoint.write_text(
            "{malformed checkpoint\n" if malformed else json.dumps(row) + "\n",
            encoding="utf-8",
        )
        return row

    def _clone_download_paper(self, root: Path, source_stem: str, target_stem: str) -> None:
        source_dir = root / "papers" / source_stem
        target_dir = root / "papers" / target_stem
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            source_dir / "source_archive.bin",
            target_dir / "source_archive.bin",
        )
        metadata = json.loads((source_dir / "download.json").read_text(encoding="utf-8"))
        metadata.update(
            {
                "arxiv_id": target_stem.removesuffix("v1"),
                "stem": target_stem,
                "archive": f"papers/{target_stem}/source_archive.bin",
            }
        )
        (target_dir / "download.json").write_text(
            json.dumps(metadata) + "\n",
            encoding="utf-8",
        )

    def _pipeline_args(self, input_root: Path, output_dir: Path, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "input_root": input_root,
            "output_dir": output_dir,
            "tokenizer": "simple",
            "workers": 1,
            "max_papers": 0,
            "paper_ids": [],
            "max_samples": 100,
            "max_samples_per_paper": 0,
            "min_response_tokens": 1_000,
            "max_response_tokens": 1_100,
            "mutation_word_ratio": 0.10,
            "min_mutations": 3,
            "max_mutations": 0,
            "shard_size": 3,
            "write_merged_jsonl": True,
            "val_fraction": 0.0,
            "seed": 83,
            "split_seed": 42,
            "temp_root": None,
            "allow_all_licenses": False,
            "allow_tokenizer_download": False,
            "trust_remote_code": False,
            "resume": True,
            "retry_failed": False,
            "response_fence": "markdown",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_select_input_rows_falls_back_to_download_checkpoints_and_counts_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_download_checkpoint(root, "2601.00001v1", status="passed")
            self._write_download_checkpoint(root, "2601.00002v1", status="success")
            self._write_download_checkpoint(root, "2601.00003v1", status="failed")
            self._write_download_checkpoint(
                root, "2601.00004v1", archive_state="partial"
            )
            self._write_download_checkpoint(
                root, "2601.00005v1", archive_state="missing"
            )
            self._write_download_checkpoint(
                root, "2601.00006v1", malformed=True
            )
            self._write_download_checkpoint(
                root, "2601.00007v1", archive_state="empty"
            )
            self._write_download_checkpoint(
                root,
                "2601.00008v1",
                license_name="MIT",
            )

            selected, rejected = MODULE.select_input_rows(
                root,
                max_papers=0,
                paper_ids=set(),
                allow_all_licenses=False,
                workers=1,
            )
            self.assertFalse((root / "results.jsonl").exists())
            self.assertEqual(
                {row["stem"] for row, _archive in selected},
                {"2601.00001v1", "2601.00002v1"},
            )
            self.assertTrue(all(archive.is_file() and archive.stat().st_size for _, archive in selected))
            self.assertEqual(sum(rejected.values()), 6)

            selected_all, rejected_all = MODULE.select_input_rows(
                root,
                max_papers=0,
                paper_ids=set(),
                allow_all_licenses=True,
                workers=1,
            )
            self.assertEqual(
                {row["stem"] for row, _archive in selected_all},
                {"2601.00001v1", "2601.00002v1", "2601.00008v1"},
            )
            self.assertEqual(sum(rejected_all.values()), 5)

    def test_select_input_rows_keeps_results_mode_and_honors_limits_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_input(root)
            self._write_download_checkpoint(root, "2601.00002v1", status="passed")

            selected, rejected = MODULE.select_input_rows(
                root,
                max_papers=0,
                paper_ids=set(),
                allow_all_licenses=False,
                workers=2,
            )
            self.assertEqual([row["stem"] for row, _archive in selected], ["2601.00001v1"])
            self.assertEqual(sum(rejected.values()), 0)

            (root / "results.jsonl").unlink()
            selected_limited, _rejected = MODULE.select_input_rows(
                root,
                max_papers=1,
                paper_ids=set(),
                allow_all_licenses=False,
                workers=1,
            )
            self.assertEqual(len(selected_limited), 1)

            selected_by_stem, _rejected = MODULE.select_input_rows(
                root,
                max_papers=0,
                paper_ids={"2601.00002v1"},
                allow_all_licenses=False,
                workers=1,
            )
            self.assertEqual([row["stem"] for row, _archive in selected_by_stem], ["2601.00002v1"])

            selected_by_id, _rejected = MODULE.select_input_rows(
                root,
                max_papers=0,
                paper_ids={"2601.00001"},
                allow_all_licenses=False,
                workers=1,
            )
            self.assertEqual([row["stem"] for row, _archive in selected_by_id], ["2601.00001v1"])

    def test_pipeline_uses_download_checkpoints_without_results_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_dir = root / "output"
            input_root.mkdir()
            self._write_input(input_root)
            (input_root / "results.jsonl").unlink()

            summary = MODULE.run_pipeline(
                self._pipeline_args(input_root, output_dir)
            )
            self.assertEqual(summary["status"], "passed")
            self.assertGreater(summary["papers"]["selected"], 0)
            self.assertGreater(summary["merge"]["written_samples"], 0)
            self.assertFalse((input_root / "results.jsonl").exists())
            rows = []
            for path in sorted((output_dir / "train").glob("part-*.jsonl")):
                rows.extend(MODULE.read_jsonl(path))
            self.assertEqual(len(rows), summary["merge"]["written_samples"])
            table_rows = [row for row in rows if row["extra_info"]["table_count"]]
            self.assertTrue(table_rows)
            self.assertEqual(len(table_rows), summary["merge"]["table_samples"])
            for row in rows:
                answer = row["messages"][1]["content"]
                self.assertTrue(answer.startswith(MODULE.RESPONSE_PREFIX))
                self.assertTrue(answer.endswith(MODULE.RESPONSE_SUFFIX))
                self.assertIsInstance(row["extra_info"]["heading_changes"], list)
                document_a = row["messages"][0]["content"][len(MODULE.PROMPT_PREFIX):-len(MODULE.PROMPT_SUFFIX)]
                self.assertEqual(
                    re.findall(r"<table>.*?</table>", document_a, re.DOTALL),
                    re.findall(r"<table>.*?</table>", answer, re.DOTALL),
                )
                self.assertNotIn("<html>", answer)
                MODULE.validate_sample(row, max_response_tokens=1_100)

    def test_checkpoint_persists_each_final_row_and_resumes_partial_paper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows_path = Path(directory) / "paper.jsonl"
            metadata_path = rows_path.with_suffix(".json")
            state = {"config_fingerprint": "test-v3", "archive_sha256": "sha"}
            sample = {
                "messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}],
                "extra_info": {"sample_id": "one", "clean_text_sha256": "a", "edited_text_sha256": "b"},
            }
            writer = MODULE.SampleCheckpoint(rows_path, metadata_path, state, resume=True)
            self.assertTrue(writer.write(sample))
            self.assertEqual(list(MODULE.read_jsonl(rows_path)), [sample])
            self.assertEqual(json.loads(metadata_path.read_text())["status"], "running")
            # Simulate termination while the next record is being written.
            with rows_path.open("ab") as stream:
                stream.write(b'{"messages":')
            resumed = MODULE.SampleCheckpoint(rows_path, metadata_path, state, resume=True)
            self.assertFalse(resumed.write(sample))
            self.assertEqual(list(MODULE.read_jsonl(rows_path)), [sample])
            self.assertEqual(resumed.ids, {"one"})

    def test_pipeline_interrupt_keeps_first_sample_and_resume_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root, output_dir = root / "input", root / "output"
            input_root.mkdir()
            self._write_input(input_root)
            args = self._pipeline_args(input_root, output_dir)
            original_make_sample = MODULE.make_sample
            first_sample = None

            def interrupt_next_sample(**kwargs: object) -> dict[str, object]:
                nonlocal first_sample
                if first_sample is not None:
                    checkpoints = list((output_dir / "checkpoints").glob("*/*/*.jsonl"))
                    self.assertEqual(len(checkpoints), 1)
                    self.assertEqual(list(MODULE.read_jsonl(checkpoints[0])), [first_sample])
                    raise KeyboardInterrupt
                first_sample = original_make_sample(**kwargs)
                return first_sample

            with mock.patch.object(MODULE, "make_sample", side_effect=interrupt_next_sample):
                with self.assertRaises(KeyboardInterrupt):
                    MODULE.run_pipeline(args)
            summary = MODULE.run_pipeline(args)
            self.assertEqual(summary["status"], "passed")
            rows = list(MODULE.read_jsonl(output_dir / "train.jsonl"))
            ids = [row["extra_info"]["sample_id"] for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertIn(first_sample["extra_info"]["sample_id"], ids)

    def test_cli_workers_two_processes_download_checkpoint_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_dir = root / "output"
            input_root.mkdir()
            self._write_input(input_root)
            self._clone_download_paper(input_root, "2601.00001v1", "2601.00002v1")
            (input_root / "results.jsonl").unlink()

            command = [
                sys.executable,
                str(SCRIPT),
                "--input-root",
                str(input_root),
                "--output-dir",
                str(output_dir),
                "--tokenizer",
                "simple",
                "--workers",
                "2",
                "--max-papers",
                "2",
                "--max-samples",
                "2",
                "--max-samples-per-paper",
                "1",
                "--min-response-tokens",
                "1000",
                "--max-response-tokens",
                "1100",
                "--mutation-word-ratio",
                "0.10",
                "--min-mutations",
                "3",
                "--shard-size",
                "2",
                "--write-merged-jsonl",
                "--val-fraction",
                "0",
                "--seed",
                "83",
                "--split-seed",
                "42",
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertFalse((input_root / "results.jsonl").exists())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["configuration"]["workers"], 2)
            self.assertEqual(manifest["papers"]["selected"], 2)
            self.assertGreater(manifest["merge"]["written_samples"], 0)
            rows = list(MODULE.read_jsonl(output_dir / "train.jsonl"))
            self.assertEqual(len(rows), manifest["merge"]["written_samples"])
            for row in rows:
                MODULE.validate_sample(row, max_response_tokens=1_100)

    def test_cli_workers_two_no_fence_output_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_dir = root / "output"
            input_root.mkdir()
            self._write_input(input_root)
            command = [
                sys.executable,
                str(SCRIPT),
                "--input-root",
                str(input_root),
                "--output-dir",
                str(output_dir),
                "--tokenizer",
                "simple",
                "--workers",
                "2",
                "--max-papers",
                "1",
                "--max-samples",
                "2",
                "--max-samples-per-paper",
                "1",
                "--min-response-tokens",
                "1",
                "--max-response-tokens",
                "1100",
                "--response-fence",
                "none",
                "--mutation-word-ratio",
                "0.10",
                "--min-mutations",
                "3",
                "--shard-size",
                "2",
                "--write-merged-jsonl",
                "--val-fraction",
                "0",
                "--seed",
                "83",
                "--split-seed",
                "42",
            ]
            first = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                first.returncode,
                0,
                msg=f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}",
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["response_fence"], "none")
            self.assertEqual(manifest["prompt_version"], MODULE.NO_FENCE_PROMPT_VERSION)
            self.assertEqual(manifest["configuration"]["response_fence"], "none")
            first_rows = list(MODULE.read_jsonl(output_dir / "train.jsonl"))
            self.assertGreater(len(first_rows), 0)
            for row in first_rows:
                answer = row["messages"][1]["content"]
                self.assertFalse(answer.startswith(MODULE.RESPONSE_PREFIX))
                self.assertFalse(answer.endswith(MODULE.RESPONSE_SUFFIX))
                self.assertEqual(row["extra_info"]["response_fence"], "none")
                self.assertEqual(
                    row["extra_info"]["prompt_version"],
                    MODULE.NO_FENCE_PROMPT_VERSION,
                )
                MODULE.validate_sample(row, max_response_tokens=1_100)

            second = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                second.returncode,
                0,
                msg=f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}",
            )
            resumed_manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resumed_manifest["papers"]["reused"], 1)
            self.assertEqual(
                resumed_manifest["merge"]["written_samples"], len(first_rows)
            )
            self.assertEqual(list(MODULE.read_jsonl(output_dir / "train.jsonl")), first_rows)

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
                response_fence="markdown",
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
