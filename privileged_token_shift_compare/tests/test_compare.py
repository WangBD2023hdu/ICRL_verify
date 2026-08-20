from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "compare.py"
SPEC = importlib.util.spec_from_file_location("privileged_token_shift_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


class FakeTokenizer:
    eos_token_id = 99

    _pieces = {
        (1,): "A",
        (2,): "B",
        (3,): "AB",
        (1, 2): "AB",
    }

    def decode(self, ids, **_kwargs):
        return self._pieces.get(tuple(ids), "".join(str(value) for value in ids))


class CompareHelpersTest(unittest.TestCase):
    def test_nested_field_prefers_literal_key(self):
        record = {
            "reward_model.ground_truth": "literal",
            "reward_model": {"ground_truth": "nested"},
        }
        self.assertEqual(compare.get_nested_field(record, "reward_model.ground_truth"), "literal")

    def test_optional_text_normalizes_null_and_lists(self):
        self.assertIsNone(compare.normalize_optional_text(None))
        self.assertIsNone(compare.normalize_optional_text("  "))
        self.assertEqual(compare.normalize_optional_text(["a", "b"]), "a\nb")

    def test_student_message_inserts_images_before_original_text(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            record = {
                "prompt": [{"role": "user", "content": "read this"}],
                "images": ["images/a.png", "images/b.png"],
            }
            messages, images = compare.build_student_messages(
                record,
                prompt_key="prompt",
                image_key="images",
                dataset_dir=base,
                max_pixels=1234,
            )
        content = messages[0]["content"]
        self.assertEqual([item["type"] for item in content], ["image", "image", "text"])
        self.assertEqual(content[0]["max_pixels"], 1234)
        self.assertTrue(content[0]["image"].endswith("/images/a.png"))
        self.assertEqual(content[-1]["text"], "read this")
        self.assertEqual(len(images), 2)

    def test_student_message_preserves_placeholder_position(self):
        with tempfile.TemporaryDirectory() as directory:
            messages, _ = compare.build_student_messages(
                {
                    "prompt": [{"role": "user", "content": "before<image>after"}],
                    "images": ["page.png"],
                },
                prompt_key="prompt",
                image_key="images",
                dataset_dir=Path(directory),
                max_pixels=4096,
            )
        self.assertEqual(
            [item["type"] for item in messages[0]["content"]],
            ["text", "image", "text"],
        )
        self.assertEqual(messages[0]["content"][0]["text"], "before")
        self.assertEqual(messages[0]["content"][2]["text"], "after")

    def test_teacher_message_is_standalone_and_text_only(self):
        messages = compare.build_teacher_messages("answer", "{privileged_text} rewrite")
        self.assertEqual(messages, [{"role": "user", "content": "answer rewrite"}])

    def test_detects_teacher_token_equivalent_to_student_span(self):
        match = compare.find_text_equivalent_candidate(
            FakeTokenizer(),
            response_ids=[1, 2],
            index=0,
            candidates=[{"token_id": 3, "token": "AB", "probability": 0.9, "rank": 1}],
            max_span=4,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["target_span_length"], 2)
        self.assertEqual(match["equivalence_type"], "teacher_merge")

    def test_scanner_selects_only_rows_with_privileged_text(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.jsonl"
            rows = [
                {"id": "a", "reward_model": {"ground_truth": "x"}},
                {"id": "b", "reward_model": {"ground_truth": None}},
                {"id": "c", "reward_model": {"ground_truth": "z"}},
            ]
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            selected, stats = compare.scan_records(
                dataset,
                privileged_key="reward_model.ground_truth",
                requested_ids=set(),
                selection="first",
                sample_count=0,
                offset=0,
                seed=7,
                progress_interval=30,
            )
        self.assertEqual([item.sample_id for item in selected], ["a", "c"])
        self.assertEqual(stats["missing_privileged_rows"], 1)

    def test_report_pages_open_from_file_without_http_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sample_dir = output / "samples" / "sample"
            compare.write_report_shell(output)
            compare.write_sample_artifacts(
                sample_dir,
                {
                    "sample_id": "sample",
                    "slug": "sample",
                    "line_number": 1,
                    "images": [],
                    "response": {"text": "ok"},
                    "summary": {"token_count": 1, "teacher_confidence_filtered_count": 0},
                    "tokens": [],
                },
            )
            compare.rebuild_manifest(output)
            index_js = (output / "assets" / "index.js").read_text(encoding="utf-8")
            sample_js = (output / "assets" / "sample.js").read_text(encoding="utf-8")
            index_html = (output / "index.html").read_text(encoding="utf-8")
            sample_html = (sample_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn('src="manifest.js"', index_html)
            self.assertIn('src="data.js"', sample_html)
            self.assertIn(
                "window.__PRIVILEGED_MANIFEST__",
                (output / "manifest.js").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "window.__PRIVILEGED_SAMPLE__",
                (sample_dir / "data.js").read_text(encoding="utf-8"),
            )
            self.assertNotIn("fetch(", index_js)
            self.assertNotIn("fetch(", sample_js)
            self.assertNotIn("student_top_candidates", sample_html)

    def test_run_arguments_are_json_serializable_after_callback_is_removed(self):
        parser = compare.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--dataset",
                "/tmp/data.jsonl",
                "--model",
                "/tmp/model",
                "--output",
                "/tmp/report",
            ]
        )
        serialized = {key: value for key, value in vars(args).items() if key != "function"}
        json.dumps(serialized)


if __name__ == "__main__":
    unittest.main()
