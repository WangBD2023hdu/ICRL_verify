from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class NormalizeConfig:
    preserve_newlines: bool = True
    unicode_form: str = "NFKC"


_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n+")


def normalize_text(text: str | None, config: NormalizeConfig | None = None) -> str:
    """Normalize parser/model text before character alignment."""
    if config is None:
        config = NormalizeConfig()
    if text is None:
        return ""

    normalized = unicodedata.normalize(config.unicode_form, text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]

    if config.preserve_newlines:
        normalized = "\n".join(line for line in lines if line)
        normalized = _MULTI_NEWLINE_RE.sub("\n", normalized)
    else:
        normalized = " ".join(line for line in lines if line)
        normalized = _HORIZONTAL_WHITESPACE_RE.sub(" ", normalized)

    return normalized.strip()

