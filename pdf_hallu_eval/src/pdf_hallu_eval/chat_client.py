from __future__ import annotations

from dataclasses import dataclass, field
import base64
import mimetypes
from pathlib import Path
from typing import Any


DEFAULT_TRANSCRIPTION_PROMPT = (
    "Transcribe all visible text on this PDF page exactly.\n"
    "Preserve reading order as much as possible.\n"
    "Do not summarize.\n"
    "Do not infer missing or unreadable text.\n"
    "If a character or word is unreadable, output [UNK].\n"
    "Return only the transcription text."
)


class ChatClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatConfig:
    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 120.0
    retries: int = 3
    prompt: str = DEFAULT_TRANSCRIPTION_PROMPT
    extra_headers: dict[str, str] = field(default_factory=dict)


def build_chat_payload(
    *,
    config: ChatConfig,
    image_bytes: bytes,
    prompt: str | None = None,
    mime_type: str = "image/png",
) -> dict[str, Any]:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or config.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    },
                ],
            }
        ],
    }


def parse_chat_response(payload: Any) -> str:
    if hasattr(payload, "choices"):
        try:
            content = payload.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ChatClientError(f"Unexpected OpenAI chat completion response: {payload!r}") from exc
        return _content_to_text(content)

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatClientError(f"Unexpected chat completion response: {payload!r}") from exc
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(text_parts)
    raise ChatClientError(f"Unsupported message content type: {type(content).__name__}")


class OpenAIChatClient:
    def __init__(self, config: ChatConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client or self._build_client()

    def transcribe_image(self, image_path: Path, prompt: str | None = None) -> str:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_bytes = image_path.read_bytes()
        payload = build_chat_payload(
            config=self.config,
            image_bytes=image_bytes,
            prompt=prompt,
            mime_type=mime_type,
        )
        try:
            response = self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise ChatClientError(f"OpenAI chat completion request failed: {exc}") from exc
        return parse_chat_response(response)

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ChatClientError("Install the openai package to call chat models: pip install openai") from exc
        api_key = self.config.api_key or "EMPTY"
        default_headers = {
            **self.config.extra_headers,
            "Authorization": f"Bearer {api_key}",
        }
        return OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            max_retries=self.config.retries,
            default_headers=default_headers,
        )
