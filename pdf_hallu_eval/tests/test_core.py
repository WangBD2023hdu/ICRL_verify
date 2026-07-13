from __future__ import annotations

import unittest

from pdf_hallu_eval.align import align_text
from pdf_hallu_eval.metrics import compute_page_metrics
from pdf_hallu_eval.normalize import NormalizeConfig, normalize_text


class NormalizeTextTest(unittest.TestCase):
    def test_normalizes_unicode_and_horizontal_whitespace(self) -> None:
        text = "ＡＢＣ\t１２３\r\n  hello   world  "

        normalized = normalize_text(text)

        self.assertEqual(normalized, "ABC 123\nhello world")

    def test_can_flatten_newlines_for_parser_model_comparison(self) -> None:
        text = "A\n\nB\r\nC"

        normalized = normalize_text(text, NormalizeConfig(preserve_newlines=False))

        self.assertEqual(normalized, "A B C")


class AlignTextTest(unittest.TestCase):
    def test_marks_prediction_insertions_as_unsupported_output(self) -> None:
        alignment = align_text("abc", "axbc!")

        self.assertEqual([op.kind for op in alignment.operations], ["match", "insert", "match", "match", "insert"])
        self.assertEqual(alignment.counts.matches, 3)
        self.assertEqual(alignment.counts.substitutions, 0)
        self.assertEqual(alignment.counts.deletions, 0)
        self.assertEqual(alignment.counts.insertions, 2)

    def test_marks_substitutions_and_deletions_separately(self) -> None:
        substitution = align_text("abc", "adc")
        deletion = align_text("abcd", "acd")

        self.assertEqual(substitution.counts.matches, 2)
        self.assertEqual(substitution.counts.substitutions, 1)
        self.assertEqual(deletion.counts.matches, 3)
        self.assertEqual(deletion.counts.deletions, 1)


class PageMetricsTest(unittest.TestCase):
    def test_computes_character_hallucination_and_omission_rates(self) -> None:
        alignment = align_text("abc", "axbc!")

        metrics = compute_page_metrics(alignment)

        self.assertEqual(metrics.reference_chars, 3)
        self.assertEqual(metrics.prediction_chars, 5)
        self.assertAlmostEqual(metrics.cer, 2 / 3)
        self.assertAlmostEqual(metrics.hallucination_rate, 2 / 5)
        self.assertAlmostEqual(metrics.pure_insertion_rate, 2 / 5)
        self.assertAlmostEqual(metrics.omission_rate, 0.0)
        self.assertAlmostEqual(metrics.coverage, 1.0)

    def test_counts_substitutions_as_broad_hallucination_and_omission(self) -> None:
        alignment = align_text("abc", "adc")

        metrics = compute_page_metrics(alignment)

        self.assertAlmostEqual(metrics.hallucination_rate, 1 / 3)
        self.assertAlmostEqual(metrics.pure_insertion_rate, 0.0)
        self.assertAlmostEqual(metrics.omission_rate, 1 / 3)


if __name__ == "__main__":
    unittest.main()
