from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pdf_hallu_eval.batch import BatchConfig, run_batch
from pdf_hallu_eval.cli import build_parser, main
from pdf_hallu_eval.pdf_parser import ParsedPage
from pdf_hallu_eval.pdf_render import RenderedPage


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe_image(self, image_path: Path) -> str:
        self.calls.append(image_path)
        return "axc"


class FailingChatClient:
    def transcribe_image(self, image_path: Path) -> str:
        raise RuntimeError("401 unauthorized")


class EmptyChatClient:
    def transcribe_image(self, image_path: Path) -> str:
        return ""


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
    def test_marks_empty_model_response_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            (pdf_dir / "paper.pdf").write_bytes(b"%PDF fake")

            result = run_batch(
                BatchConfig(pdf_dir=pdf_dir, output_dir=root / "out"),
                chat_client=EmptyChatClient(),
                text_extractor=fake_extract_text_pages,
                renderer=fake_render_pdf_pages,
            )

            self.assertEqual(result.page_records[0]["status"], "model_empty")
            self.assertIn("empty transcription", result.page_records[0]["error"])

    def test_records_model_error_with_original_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            (pdf_dir / "paper.pdf").write_bytes(b"%PDF fake")

            result = run_batch(
                BatchConfig(pdf_dir=pdf_dir, output_dir=root / "out"),
                chat_client=FailingChatClient(),
                text_extractor=fake_extract_text_pages,
                renderer=fake_render_pdf_pages,
            )

            self.assertEqual(result.page_records[0]["status"], "model_error")
            self.assertIn("401 unauthorized", result.page_records[0]["error"])

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
    def test_returns_nonzero_and_prints_page_failures(self) -> None:
        failed_result = SimpleNamespace(
            total_pages=1,
            total_pdfs=1,
            output_dir=Path("/tmp/out"),
            page_records=[
                {
                    "pdf_name": "paper.pdf",
                    "page_index": 0,
                    "status": "model_error",
                    "error": "401 unauthorized",
                }
            ],
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch("pdf_hallu_eval.cli.OpenAIChatClient", return_value=object()),
            patch("pdf_hallu_eval.cli.run_batch", return_value=failed_result),
            patch.dict("os.environ", {"INF_API_KEY": "secret"}),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main(
                [
                    "run",
                    "--pdf-dir",
                    "/tmp/pdfs",
                    "--output-dir",
                    "/tmp/out",
                    "--model",
                    "vision-model",
                    "--api-key-env",
                    "INF_API_KEY",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("Model/render failures: 1 page(s)", stderr.getvalue())
        self.assertIn("401 unauthorized", stderr.getvalue())

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
