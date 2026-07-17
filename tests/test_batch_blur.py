from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qwen_mm_token_probe.batch_blur import load_batch_items, run_blur_batch
from qwen_mm_token_probe.batch_report import rebuild_batch_reports
from qwen_mm_token_probe.batch_stats import comparison_scores_from_records, summarize_scores


class BatchInputTests(unittest.TestCase):
    def test_loads_string_list_and_object_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "items.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "one", "images": "a.png"}),
                        json.dumps(
                            {
                                "id": "two",
                                "images": ["b.png", {"path": "nested/c.png"}],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            items = load_batch_items(input_path, default_prompt="OCR")

            self.assertEqual(len(items), 3)
            self.assertEqual(items[0].image_path, (root / "a.png").resolve())
            self.assertEqual(items[2].image_path, (root / "nested/c.png").resolve())
            self.assertEqual(items[1].record_id, "two")
            self.assertNotEqual(items[1].sample_key, items[2].sample_key)

    def test_record_prompt_field_overrides_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "items.jsonl"
            input_path.write_text(
                json.dumps({"images": ["a.png"], "instruction": "record prompt"}) + "\n",
                encoding="utf-8",
            )

            items = load_batch_items(
                input_path,
                default_prompt="global prompt",
                prompt_field="instruction",
            )

            self.assertEqual(items[0].prompt, "record prompt")

    def test_all_resumed_samples_do_not_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "items.jsonl"
            input_path.write_text(
                json.dumps({"id": "one", "images": "a.png"}) + "\n",
                encoding="utf-8",
            )
            output = root / "output"

            with (
                patch(
                    "qwen_mm_token_probe.batch_blur._completed_result_matches",
                    return_value=True,
                ),
                patch(
                    "qwen_mm_token_probe.batch_report.rebuild_batch_reports",
                    return_value={"completed_samples": 1},
                ),
            ):
                summary = run_blur_batch(
                    model_id="model-does-not-need-to-load",
                    input_jsonl=input_path,
                    output_dir=output,
                    prompt="OCR",
                    blur_radii=[1.0],
                )

            self.assertEqual(summary.skipped_items, 1)
            self.assertEqual(summary.completed_items, 1)


class BlurStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = [
            {
                "index": 0,
                "token_id": 10,
                "token": "A",
                "raw_token": "A",
                "p_original": 0.8,
                "logp_original": math.log(0.8),
                "top_token_id_original": 10,
                "top_token_original": "A",
                "top_raw_token_original": "A",
                "top_p_original": 0.8,
                "top_logp_original": math.log(0.8),
            },
            {
                "index": 1,
                "token_id": 11,
                "token": "B",
                "raw_token": " B",
                "p_original": 0.2,
                "logp_original": math.log(0.2),
                "top_token_id_original": 12,
                "top_token_original": "C",
                "top_raw_token_original": "C",
                "top_p_original": 0.7,
                "top_logp_original": math.log(0.7),
            },
        ]
        self.blurred = [
            {
                "index": 0,
                "p_blurred": 0.4,
                "logp_blurred": math.log(0.4),
                "top_token_id_blurred": 12,
                "top_token_blurred": "C",
                "top_raw_token_blurred": "C",
                "top_p_blurred": 0.6,
                "top_logp_blurred": math.log(0.6),
            },
            {
                "index": 1,
                "p_blurred": 0.6,
                "logp_blurred": math.log(0.6),
                "top_token_id_blurred": 11,
                "top_token_blurred": "B",
                "top_raw_token_blurred": "B",
                "top_p_blurred": 0.6,
                "top_logp_blurred": math.log(0.6),
            },
        ]

    def test_tracks_tokens_that_gain_probability_after_blur(self) -> None:
        scores = comparison_scores_from_records(self.baseline, self.blurred)
        summary = summarize_scores(scores)

        self.assertAlmostEqual(summary["probability_gain_rate"], 0.5)
        self.assertEqual(summary["num_probability_gain_tokens"], 1)
        self.assertAlmostEqual(summary["mean_blur_gain_p_gained_tokens"], 0.4)
        self.assertAlmostEqual(
            summary["mean_blur_gain_logp_gained_tokens"],
            math.log(3.0),
        )
        self.assertLess(scores[1].delta_logp, 0.0)

    def test_report_separates_gain_and_drop_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            sample_dir = output / "samples" / "000001_00_one_deadbeef"
            sample_dir.mkdir(parents=True)
            result = {
                "schema_version": 1,
                "model_id": "fake-model",
                "sample": {
                    "sample_key": sample_dir.name,
                    "record_id": "one",
                    "line_number": 1,
                    "image_index": 0,
                    "image_path": str(output / "input.png"),
                    "prompt": "OCR",
                },
                "original": {
                    "image_path": str(sample_dir / "original.png"),
                    "response_path": str(sample_dir / "response.md"),
                    "generated_text": "A B",
                    "generated_token_ids": [10, 11],
                    "tokens": self.baseline,
                },
                "blur_levels": [
                    {
                        "blur_radius": 2.0,
                        "image_path": str(sample_dir / "blur_r2.png"),
                        "tokens": self.blurred,
                    }
                ],
            }
            (sample_dir / "result.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            (output / "config.json").write_text(
                json.dumps({"model_id": "fake-model", "input_jsonl": "items.jsonl"}),
                encoding="utf-8",
            )

            with patch(
                "qwen_mm_token_probe.batch_report._write_aggregate_plot",
                return_value=False,
            ):
                state = rebuild_batch_reports(output)

            sample_html = (sample_dir / "report.html").read_text(encoding="utf-8")
            token_csv = (output / "token_probabilities.csv").read_text(encoding="utf-8")
            self.assertEqual(state["completed_samples"], 1)
            self.assertIn("Largest gains after blur", sample_html)
            self.assertIn("Largest drops after blur", sample_html)
            self.assertIn('src="original.png"', sample_html)
            self.assertIn("probability_increased_after_blur", token_csv.splitlines()[0])
            self.assertIn(",True,", token_csv)


if __name__ == "__main__":
    unittest.main()
