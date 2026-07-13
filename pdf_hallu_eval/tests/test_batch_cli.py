from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from pdf_hallu_eval.batch import BatchConfig, run_batch
from pdf_hallu_eval.cli import build_parser
from pdf_hallu_eval.pdf_parser import ParsedPage
from pdf_hallu_eval.pdf_render import RenderedPage


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe_image(self, image_path: Path) -> str:
        self.calls.append(image_path)
        return "axc"


def fake_extract_text_pages(pdf_path: Path, parser: str = "auto") -> list[ParsedPage]:
    return [ParsedPage(page_index=0, text="abc", parser="fake")]


def fake_render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    *,
    backend: str = "auto",
    dpi: int = 144,
    pdftoppm_path: str | None = None,
) -> list[RenderedPage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "page-1.png"
    image_path.write_bytes(b"fake")
    return [RenderedPage(page_index=0, image_path=image_path, backend="fake")]


class BatchRunnerTest(unittest.TestCase):
    def test_runs_batch_and_writes_review_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            (pdf_dir / "paper.pdf").write_bytes(b"%PDF fake")
            output_dir = root / "out"
            chat_client = FakeChatClient()

            result = run_batch(
                BatchConfig(pdf_dir=pdf_dir, output_dir=output_dir),
                chat_client=chat_client,
                text_extractor=fake_extract_text_pages,
                renderer=fake_render_pdf_pages,
            )

            self.assertEqual(result.total_pages, 1)
            self.assertEqual(len(chat_client.calls), 1)
            self.assertTrue((output_dir / "pages.jsonl").exists())
            self.assertTrue((output_dir / "pdf_summary.csv").exists())
            self.assertTrue((output_dir / "dataset_summary.json").exists())
            self.assertTrue((output_dir / "review" / "index.html").exists())
            self.assertAlmostEqual(result.page_records[0]["metrics"]["hallucination_rate"], 1 / 3)

    def test_resume_skips_existing_page_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            (pdf_dir / "paper.pdf").write_bytes(b"%PDF fake")
            output_dir = root / "out"

            first_client = FakeChatClient()
            run_batch(
                BatchConfig(pdf_dir=pdf_dir, output_dir=output_dir),
                chat_client=first_client,
                text_extractor=fake_extract_text_pages,
                renderer=fake_render_pdf_pages,
            )
            second_client = FakeChatClient()
            second_result = run_batch(
                BatchConfig(pdf_dir=pdf_dir, output_dir=output_dir, resume=True),
                chat_client=second_client,
                text_extractor=fake_extract_text_pages,
                renderer=fake_render_pdf_pages,
            )

            self.assertEqual(len(first_client.calls), 1)
            self.assertEqual(len(second_client.calls), 0)
            self.assertEqual(second_result.total_pages, 1)


class CliTest(unittest.TestCase):
    def test_run_parser_accepts_large_batch_options(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--pdf-dir",
                "/tmp/pdfs",
                "--output-dir",
                "/tmp/out",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "vision-model",
                "--workers",
                "8",
                "--resume",
            ]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.workers, 8)
        self.assertTrue(args.resume)


if __name__ == "__main__":
    unittest.main()
