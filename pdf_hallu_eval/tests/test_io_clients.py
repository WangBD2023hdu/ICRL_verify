from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
import unittest
from unittest.mock import patch

from pdf_hallu_eval.chat_client import ChatConfig, OpenAIChatClient, build_chat_payload, parse_chat_response
from pdf_hallu_eval.pdf_parser import PDFProcessingError, extract_text_pages
from pdf_hallu_eval.pdf_render import render_pdf_pages
from pdf_hallu_eval.storage import OutputLayout, iter_jsonl, make_pdf_id, write_jsonl


class StorageTest(unittest.TestCase):
    def test_make_pdf_id_is_stable_and_filesystem_safe(self) -> None:
        pdf_path = Path("/tmp/Papers/Some File.pdf")

        first = make_pdf_id(pdf_path)
        second = make_pdf_id(pdf_path)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("some-file-"))
        self.assertNotIn(" ", first)

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"

            write_jsonl(path, [{"a": 1}, {"b": "two"}])

            self.assertEqual(list(iter_jsonl(path)), [{"a": 1}, {"b": "two"}])

    def test_output_layout_returns_stable_page_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = OutputLayout(Path(tmp))

            paths = layout.page_paths("paper-abc123", 2)

            self.assertEqual(paths.parser_text.name, "paper-abc123_page_0003.txt")
            self.assertEqual(paths.model_text.name, "paper-abc123_page_0003.txt")
            self.assertEqual(paths.image.name, "paper-abc123_page_0003.png")
            self.assertIn("parser_text", paths.parser_text.parts)


class ChatClientPayloadTest(unittest.TestCase):
    def test_openai_client_sets_explicit_bearer_authorization_header(self) -> None:
        captured_kwargs = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                captured_kwargs.update(kwargs)

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        config = ChatConfig(
            model="vision-model",
            base_url="https://example.invalid/v1",
            api_key="inf-secret",
            extra_headers={"X-Tenant": "test-tenant"},
        )

        with patch.dict(sys.modules, {"openai": fake_openai_module}):
            OpenAIChatClient(config)

        self.assertEqual(captured_kwargs["api_key"], "inf-secret")
        self.assertEqual(captured_kwargs["default_headers"]["Authorization"], "Bearer inf-secret")
        self.assertEqual(captured_kwargs["default_headers"]["X-Tenant"], "test-tenant")

    def test_builds_openai_compatible_vision_payload(self) -> None:
        config = ChatConfig(model="vision-model", base_url="http://localhost:8000/v1", api_key="token")

        payload = build_chat_payload(
            config=config,
            image_bytes=b"abc",
            prompt="Transcribe exactly.",
            mime_type="image/png",
        )

        self.assertEqual(payload["model"], "vision-model")
        self.assertEqual(payload["temperature"], 0.0)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Transcribe exactly."})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_parses_first_chat_completion_message_content(self) -> None:
        response = {"choices": [{"message": {"content": "hello"}}]}

        self.assertEqual(parse_chat_response(response), "hello")

    def test_openai_chat_client_uses_injected_openai_style_client(self) -> None:
        class FakeMessage:
            content = "transcribed text"

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeResponse()

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeOpenAIClient:
            def __init__(self) -> None:
                self.chat = FakeChat()

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image_path.write_bytes(b"abc")
            fake_client = FakeOpenAIClient()
            client = OpenAIChatClient(
                ChatConfig(model="vision-model", base_url="http://localhost:8000/v1", api_key="token"),
                client=fake_client,
            )

            text = client.transcribe_image(image_path, prompt="Transcribe exactly.")

            self.assertEqual(text, "transcribed text")
            kwargs = fake_client.chat.completions.kwargs
            self.assertEqual(kwargs["model"], "vision-model")
            self.assertEqual(kwargs["messages"][0]["content"][0]["text"], "Transcribe exactly.")
            self.assertTrue(kwargs["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))


class PDFBackendTest(unittest.TestCase):
    def test_parser_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            extract_text_pages(Path("missing.pdf"), parser="bogus")

    def test_renderer_rejects_unknown_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                render_pdf_pages(Path("missing.pdf"), Path(tmp), backend="bogus")

    def test_parser_reports_missing_pdf_cleanly(self) -> None:
        with self.assertRaises(PDFProcessingError):
            extract_text_pages(Path("missing.pdf"), parser="pypdf")


if __name__ == "__main__":
    unittest.main()
