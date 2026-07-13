from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator


_SAFE_CHARS_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class PagePaths:
    parser_text: Path
    model_text: Path
    image: Path
    alignment: Path


@dataclass(frozen=True)
class OutputLayout:
    root: Path

    def ensure(self) -> None:
        for name in (
            "images",
            "parser_text",
            "model_text",
            "alignments",
            "review",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def page_paths(self, pdf_id: str, page_index: int) -> PagePaths:
        page_name = f"{pdf_id}_page_{page_index + 1:04d}"
        return PagePaths(
            parser_text=self.root / "parser_text" / f"{page_name}.txt",
            model_text=self.root / "model_text" / f"{page_name}.txt",
            image=self.root / "images" / f"{page_name}.png",
            alignment=self.root / "alignments" / f"{page_name}.json",
        )


def make_pdf_id(pdf_path: Path) -> str:
    stem = _SAFE_CHARS_RE.sub("-", pdf_path.stem.lower()).strip("-") or "pdf"
    digest = hashlib.sha1(str(pdf_path.expanduser()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)

