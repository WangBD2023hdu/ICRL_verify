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
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_arxiv_confusable_text_sft.py"
SPEC = importlib.util.spec_from_file_location("build_arxiv_confusable_text_sft", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ArxivConfusableTextSftTests(unittest.TestCase):
    def test_english_prompt_and_sample_are_exact_text_copy(self) -> None:
        self.assertEqual(
            MODULE.sha256_text(MODULE.PROMPT_PREFIX + "{A}" + MODULE.PROMPT_SUFFIX),
            "d0cb3fc514819601449d1413774e7b86d1618de97a25ba81739b7f60a923520a",
        )
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
