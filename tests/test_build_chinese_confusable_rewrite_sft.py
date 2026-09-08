from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_chinese_confusable_rewrite_sft.py"
ARXIV_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_arxiv_confusable_text_sft.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module("build_chinese_confusable_rewrite_sft", SCRIPT)
ARXIV = load_module("build_arxiv_confusable_text_sft_for_chinese_tests", ARXIV_SCRIPT)


class ChineseConfusableRewriteTests(unittest.TestCase):
    def test_prompt_constants_match_arxiv_and_sample_prompt_is_exact(self) -> None:
        self.assertEqual(MODULE.PROMPT_PREFIX, ARXIV.PROMPT_PREFIX)
        self.assertEqual(MODULE.PROMPT_SUFFIX, ARXIV.PROMPT_SUFFIX)
        self.assertEqual(MODULE.RESPONSE_PREFIX, "```markdown\n")
        self.assertEqual(MODULE.RESPONSE_SUFFIX, "\n```")

        text = "🙂标题\n" + "未" * 120
        config = self._config()
        row = MODULE.make_sample(
            text,
            source={"source_id": "prompt-test", "path": "prompt.md", "license": "test"},
            config=config,
            chunk_index=0,
            count_tokens=len,
        )
        self.assertIsNotNone(row)
        assert row is not None
        prompt = row["messages"][0]["content"]
        answer = row["messages"][1]["content"]
        edited = prompt.split("<<<DOCUMENT_START>>>\n", 1)[1].rsplit(
            "\n<<<DOCUMENT_END>>>", 1
        )[0]
        self.assertEqual(prompt, MODULE.PROMPT_PREFIX + edited + MODULE.PROMPT_SUFFIX)
        self.assertEqual(row["messages"][0]["role"], "user")
        self.assertEqual(row["messages"][1]["role"], "assistant")
        self.assertTrue(answer.startswith(MODULE.RESPONSE_PREFIX))
        self.assertTrue(answer.endswith(MODULE.RESPONSE_SUFFIX))
        self.assertNotIn("images", row)

    def test_protected_mask_covers_code_html_math_tables_and_url_targets(self) -> None:
        text = (
            "prose 未末 and [可见未](https://example.test/path/未?x=末) "
            "https://example.test/raw/未\n"
            "```markdown\n围栏未末\n```\n"
            "`inline未`\n"
            "<table><tr><td>表格未末</td></tr></table>\n"
            "<figure>图未末</figure>\n"
            "<pre>预格式未末</pre>\n"
            "<span title=\"未\">标签末</span>\n"
            "$单未$ $$双末$$ \\(圆未\\) \\[方末\\]\n"
            "$多行未\n数学末$\n"
            "\\begin{align}\n未 &= 末 \\\\\n\\end{align}\n"
            "| 未 | 末 |\n| --- | --- |\n| 表 | 格 |\n"
            "| 单列 |\n| --- |\n| 未 |\n"
        )
        mask = MODULE.protected_mask(text)

        def assert_masked(fragment: str) -> None:
            start = text.index(fragment)
            self.assertTrue(all(mask[start : start + len(fragment)]), fragment)

        for fragment in (
            "```markdown\n围栏未末\n```",
            "`inline未`",
            "<table><tr><td>表格未末</td></tr></table>",
            "<figure>图未末</figure>",
            "<pre>预格式未末</pre>",
            '<span title="未">',
            "</span>",
            "$单未$",
            "$$双末$$",
            r"\(圆未\)",
            r"\[方末\]",
            "$多行未\n数学末$",
            "\\begin{align}\n未 &= 末 \\\\\n\\end{align}",
            "| 未 | 末 |\n| --- | --- |\n| 表 | 格 |",
            "| 单列 |\n| --- |\n| 未 |",
            "https://example.test/path/未?x=末",
            "https://example.test/raw/未",
        ):
            assert_masked(fragment)

        visible = text.index("可见未")
        self.assertFalse(any(mask[visible : visible + len("可见未")]))
        prose = text.index("prose 未末") + len("prose ")
        self.assertFalse(any(mask[prose : prose + len("未末")]))

    def test_mutate_text_uses_all_unprotected_han_denominator_and_unicode_offsets(self) -> None:
        text = "🙂🚀" + "未" * 100 + "中" * 50 + "\n```\n未末\n```"
        mask = MODULE.protected_mask(text)
        expected_prose = sum(
            1 for index, character in enumerate(text)
            if MODULE.HAN_RE.fullmatch(character) and not mask[index]
        )
        edited, changes, stats = MODULE.mutate_text(
            text,
            ratio=0.02,
            rng=random.Random(83),
        )
        self.assertEqual(stats["prose_han_chars"], expected_prose)
        self.assertEqual(stats["requested_mutations"], int(expected_prose * 0.02 + 0.5))
        self.assertEqual(stats["actual_mutations"], len(changes))
        self.assertEqual(len(edited), len(text))
        self.assertEqual(edited.split("\n```\n", 1)[1], "未末\n```")
        self.assertEqual(
            {index for index, (old, new) in enumerate(zip(text, edited)) if old != new},
            {change["input_char_offset"] for change in changes},
        )
        for change in changes:
            start = change["input_char_offset"]
            self.assertEqual(change["input_char_end"], start + 1)
            self.assertEqual(text[start], change["origin_ans"])
            self.assertEqual(edited[start], change["ocr_ans"])
            self.assertFalse(mask[start])
            self.assertEqual(text[:start].encode("utf-8").decode("utf-8"), text[:start])
        self.assertTrue(all(change["input_char_offset"] >= 2 for change in changes))

        again = MODULE.mutate_text(text, ratio=0.02, rng=random.Random(83))
        self.assertEqual((edited, changes, stats), again)

        sparse, sparse_changes, sparse_stats = MODULE.mutate_text(
            "🙂" + "未" + "中" * 99,
            ratio=0.5,
            rng=random.Random(7),
        )
        self.assertEqual(sparse_stats["requested_mutations"], 50)
        self.assertEqual(sparse_stats["actual_mutations"], 1)
        self.assertEqual(len(sparse_changes), 1)
        self.assertEqual(len(sparse), len("🙂" + "未" + "中" * 99))

    def test_rewrite_answer_changes_only_unprotected_line_heading_hashes(self) -> None:
        text = (
            "#zero spaces\n"
            "## one space\n"
            "###  two spaces\n"
            "####   three spaces\n"
            "#####five\n"
            "###### six\n"
            "####### seven is not a heading\n"
            "  # indented is not a heading\n"
            "inline # hash stays\n"
            "```markdown\n# protected code\n```\n"
            "<table>\n## protected table\n</table>\n"
        )
        answer, changes = MODULE.rewrite_answer(text, rng=random.Random(2026))
        self.assertTrue(answer.startswith(MODULE.RESPONSE_PREFIX))
        self.assertTrue(answer.endswith(MODULE.RESPONSE_SUFFIX))
        self.assertEqual(len(changes), 6)
        self.assertTrue(
            all(set(change) == {"input_offset", "from_level", "to_level"} for change in changes)
        )

        body = text
        for change in reversed(changes):
            start = change["input_offset"]
            body = (
                body[:start]
                + "#" * change["to_level"]
                + body[start + change["from_level"] :]
            )
        self.assertEqual(answer, MODULE.RESPONSE_PREFIX + body + MODULE.RESPONSE_SUFFIX)
        for change in changes:
            self.assertIn(change["from_level"], range(1, 7))
            self.assertIn(change["to_level"], range(5))
            self.assertNotEqual(change["from_level"], change["to_level"])
            self.assertEqual(text[change["input_offset"]], "#")

        repeated = MODULE.rewrite_answer(text, rng=random.Random(2026))
        self.assertEqual((answer, changes), repeated)

        no_heading = "普通文本\n```markdown\n# still protected\n```"
        plain_answer, plain_changes = MODULE.rewrite_answer(
            no_heading,
            rng=random.Random(1),
        )
        self.assertEqual(plain_changes, [])
        self.assertEqual(
            plain_answer,
            MODULE.RESPONSE_PREFIX + no_heading + MODULE.RESPONSE_SUFFIX,
        )

    def test_heading_removal_probability_and_remaining_choices(self) -> None:
        self.assertEqual(MODULE.NO_HEADING_PROBABILITY, 0.28)
        for old in range(1, 7):
            text = "#" * old + "   标题\n正文\n<table>\n# 保留表格\n</table>"
            rng = mock.Mock(spec=random.Random)
            rng.random.return_value = 0.279999
            answer, changes = MODULE.rewrite_answer(text, rng=rng)
            self.assertEqual(answer, MODULE.RESPONSE_PREFIX + text[old:] + MODULE.RESPONSE_SUFFIX)
            self.assertEqual(changes, [{"input_offset": 0, "from_level": old, "to_level": 0}])
            rng.choice.assert_not_called()

            choices = [level for level in range(1, 5) if level != old]
            rng.random.return_value = 0.28
            rng.choice.return_value = choices[0]
            _, changes = MODULE.rewrite_answer(text, rng=rng)
            rng.choice.assert_called_once_with(choices)
            self.assertEqual(changes[0]["to_level"], choices[0])

    def test_make_sample_is_deterministic_and_maps_a_offsets_into_b(self) -> None:
        text = (
            "🙂# 标题\n"
            + "未" * 180
            + "\n\n## 第二节\n"
            + "未" * 180
            + "\n```\n保护未末\n```\n"
        )
        source = {"source_id": "unicode-doc", "path": "doc.md", "license": "provided_by_user"}
        config = self._config()
        sample = MODULE.make_sample(
            text,
            source=source,
            config=config,
            chunk_index=3,
            count_tokens=len,
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(
            sample,
            MODULE.make_sample(
                text,
                source=source,
                config=config,
                chunk_index=3,
                count_tokens=len,
            ),
        )
        info = sample["extra_info"]
        prompt = sample["messages"][0]["content"]
        answer = sample["messages"][1]["content"]
        edited = prompt.split("<<<DOCUMENT_START>>>\n", 1)[1].rsplit(
            "\n<<<DOCUMENT_END>>>", 1
        )[0]
        legacy_id = MODULE.digest(
            f"chinese_confusable_rewrite_v1:{config['seed']}:{config['mutation_ratio']}:{MODULE.digest(text)}"
        )
        legacy_a, _, _ = MODULE.mutate_text(
            text, ratio=config["mutation_ratio"],
            rng=random.Random(MODULE.seed_for(legacy_id, "mutation")),
        )
        self.assertEqual(edited, legacy_a)
        self.assertEqual(prompt, MODULE.PROMPT_PREFIX + edited + MODULE.PROMPT_SUFFIX)
        self.assertTrue(answer.startswith(MODULE.RESPONSE_PREFIX))
        self.assertTrue(answer.endswith(MODULE.RESPONSE_SUFFIX))
        self.assertEqual(info["chunk_index"], 3)
        self.assertGreater(info["actual_mutations"], 0)
        self.assertEqual(info["actual_mutations"], len(info["changes"]))
        self.assertEqual(info["response_text_sha256"], MODULE.digest(answer))
        self.assertEqual(info["source_text_sha256"], MODULE.digest(text))

        rebuilt = edited
        for heading in reversed(info["heading_changes"]):
            start = heading["input_offset"]
            rebuilt = (
                rebuilt[:start]
                + "#" * heading["to_level"]
                + rebuilt[start + heading["from_level"] :]
            )
        self.assertEqual(answer, MODULE.RESPONSE_PREFIX + rebuilt + MODULE.RESPONSE_SUFFIX)
        for change in info["changes"]:
            input_start = change["input_char_offset"]
            input_end = change["input_char_end"]
            self.assertEqual(edited[input_start:input_end], change["ocr_ans"])
            self.assertEqual(answer[change["char_offset"] : change["char_end"]], change["ocr_ans"])
            shift = len(MODULE.RESPONSE_PREFIX) + sum(
                heading["to_level"] - heading["from_level"]
                for heading in info["heading_changes"]
                if heading["input_offset"] < input_start
            )
            self.assertEqual(change["char_offset"], input_start + shift)
            self.assertEqual(change["char_end"], input_end + shift)
        self.assertTrue(any(change["input_char_offset"] >= 2 for change in info["changes"]))

    def test_cli_local_md_txt_jsonl_workers_two_and_resume_repairs_trailing_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            md_path = root / "one.md"
            txt_path = root / "two.txt"
            jsonl_path = root / "rows.jsonl"
            output_dir = root / "output"
            md_path.write_text(self._cli_document("一"), encoding="utf-8")
            txt_path.write_text(self._cli_document("二"), encoding="utf-8")
            jsonl_path.write_text(
                json.dumps({"text": self._cli_document("三")}, ensure_ascii=False)
                + "\n"
                + json.dumps({"markdown": self._cli_document("四")}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(md_path),
                str(txt_path),
                str(jsonl_path),
                "--output-dir",
                str(output_dir),
                "--tokenizer",
                "simple",
                "--workers",
                "2",
                "--min-response-tokens",
                "1",
                "--max-response-tokens",
                "7800",
                "--val-fraction",
                "0.5",
                "--seed",
                "83",
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
            train_path = output_dir / "train.jsonl"
            val_path = output_dir / "val.jsonl"
            self.assertTrue(train_path.is_file())
            self.assertTrue(val_path.is_file())
            before_rows = self._read_rows(train_path) + self._read_rows(val_path)
            self.assertGreaterEqual(len(before_rows), 4)
            self.assertEqual(
                len({row["extra_info"]["sample_id"] for row in before_rows}),
                len(before_rows),
            )
            for row in before_rows:
                self.assertEqual(len(row["messages"]), 2)
                self.assertNotIn("images", row)
                self.assertTrue(row["messages"][1]["content"].startswith("```markdown\n"))
                self.assertTrue(row["messages"][1]["content"].endswith("\n```"))

            with train_path.open("ab") as handle:
                handle.write(b'{"truncated":')
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
            after_rows = self._read_rows(train_path) + self._read_rows(val_path)
            self.assertEqual(after_rows, before_rows)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["new_saved"], 0)
            self.assertEqual(summary["duplicates_this_run"], 0)
            self.assertEqual(summary["accepted_saved"], len(after_rows))

    def test_cli_lazy_stop_and_resume_for_1001_jsonl_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "many.jsonl"
            output_dir = root / "many-output"
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for index in range(1_001):
                    text = f"# 文档-{index:04d}\n" + "未" * 72 + "\n"
                    handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

            base_command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(jsonl_path),
                "--output-dir",
                str(output_dir),
                "--tokenizer",
                "simple",
                "--workers",
                "2",
                "--min-response-tokens",
                "1",
                "--max-response-tokens",
                "1000",
                "--val-fraction",
                "0.5",
                "--seed",
                "83",
            ]
            first = subprocess.run(
                [*base_command, "--max-samples", "3"],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                first.returncode,
                0,
                msg=f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}",
            )
            first_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            first_rows = self._read_rows(output_dir / "train.jsonl") + self._read_rows(
                output_dir / "val.jsonl"
            )
            self.assertEqual(len(first_rows), 3)
            self.assertTrue(first_summary["target_reached"])
            self.assertLess(first_summary["documents_discovered"], 1_001)
            self.assertEqual(
                len({row["extra_info"]["source_text_sha256"] for row in first_rows}),
                3,
            )

            second = subprocess.run(
                [*base_command, "--max-samples", "0"],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                second.returncode,
                0,
                msg=f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}",
            )
            second_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            final_rows = self._read_rows(output_dir / "train.jsonl") + self._read_rows(
                output_dir / "val.jsonl"
            )
            self.assertEqual(len(final_rows), 1_001)
            self.assertEqual(second_summary["accepted_saved"], 1_001)
            self.assertEqual(second_summary["documents_discovered"], 1_001)
            self.assertEqual(second_summary["documents_completed_or_error"], 1_001)
            self.assertEqual(second_summary["errors_this_run"], 0)
            self.assertEqual(
                len({row["extra_info"]["sample_id"] for row in final_rows}),
                1_001,
            )
            self.assertEqual(
                len({row["extra_info"]["source_text_sha256"] for row in final_rows}),
                1_001,
            )
            self.assertTrue((output_dir / "train.jsonl").is_file())
            self.assertTrue((output_dir / "val.jsonl").is_file())
            for row in final_rows:
                self.assertEqual(len(row["messages"]), 2)
                self.assertEqual(row["messages"][0]["role"], "user")
                self.assertEqual(row["messages"][1]["role"], "assistant")
                self.assertTrue(row["messages"][1]["content"].startswith("```markdown\n"))
                self.assertTrue(row["messages"][1]["content"].endswith("\n```"))
                self.assertNotIn("images", row)
                self.assertGreater(row["extra_info"]["actual_mutations"], 0)
                self.assertEqual(
                    row["extra_info"]["actual_mutations"],
                    len(row["extra_info"]["changes"]),
                )

    @staticmethod
    def _config() -> dict[str, object]:
        return {
            "mutation_ratio": 0.02,
            "min_response_tokens": 1,
            "max_response_tokens": 7_800,
            "seed": 83,
            "val_fraction": 0.5,
            "tokenizer": "simple",
        }

    @staticmethod
    def _cli_document(label: str) -> str:
        return "# 标题" + label + "\n" + "未" * 2_500 + "\n\n## 第二节" + label + "\n" + "未" * 2_500

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
