from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from pdf_hallu_eval.align import align_text
from pdf_hallu_eval.metrics import compute_page_metrics
from pdf_hallu_eval.report_html import write_review_site


class ReportHtmlTest(unittest.TestCase):
    def test_writes_index_and_pdf_review_pages_with_diff_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            image_path = output_root / "images" / "paper_page_0001.png"
            image_path.parent.mkdir()
            image_path.write_bytes(b"fake-png")
            alignment = align_text("abc", "axc")
            metrics = compute_page_metrics(alignment)
            records = [
                {
                    "pdf_id": "paper-12345678",
                    "pdf_name": "paper.pdf",
                    "page_index": 0,
                    "image_path": str(image_path),
                    "parser_text": "abc",
                    "model_text": "axc",
                    "metrics": metrics.to_dict(),
                    "alignment": [
                        {"kind": op.kind, "reference": op.reference, "prediction": op.prediction}
                        for op in alignment.operations
                    ],
                    "status": "ok",
                }
            ]

            write_review_site(output_root, records)

            index = (output_root / "review" / "index.html").read_text(encoding="utf-8")
            pdf_page = (output_root / "review" / "paper-12345678.html").read_text(encoding="utf-8")

            self.assertIn("paper.pdf", index)
            self.assertIn("hallucination_rate", index)
            self.assertIn("paper-12345678.html", index)
            self.assertIn("../images/paper_page_0001.png", pdf_page)
            self.assertIn("Parser Text", pdf_page)
            self.assertIn("Model Text", pdf_page)
            self.assertIn("diff-substitute", pdf_page)
            self.assertIn("abc", pdf_page)
            self.assertIn("axc", pdf_page)


if __name__ == "__main__":
    unittest.main()
