from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Literal


ParserName = Literal["auto", "pymupdf", "pdfplumber", "pypdf"]


class PDFProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedPage:
    page_index: int
    text: str
    parser: str
    status: str = "ok"
    error: str = ""


def extract_text_pages(pdf_path: Path, parser: ParserName = "auto") -> list[ParsedPage]:
    if parser not in {"auto", "pymupdf", "pdfplumber", "pypdf"}:
        raise ValueError(f"Unsupported parser backend: {parser}")
    if not pdf_path.exists():
        raise PDFProcessingError(f"PDF does not exist: {pdf_path}")

    backends = ["pymupdf", "pdfplumber", "pypdf"] if parser == "auto" else [parser]
    errors: list[str] = []
    for backend in backends:
        try:
            if backend == "pymupdf":
                return _extract_with_pymupdf(pdf_path)
            if backend == "pdfplumber":
                return _extract_with_pdfplumber(pdf_path)
            if backend == "pypdf":
                return _extract_with_pypdf(pdf_path)
        except PDFProcessingError as exc:
            errors.append(str(exc))
    raise PDFProcessingError("; ".join(errors) or f"Unable to parse PDF: {pdf_path}")


def _extract_with_pymupdf(pdf_path: Path) -> list[ParsedPage]:
    fitz = _import_module("fitz", "Install pymupdf to use parser=pymupdf.")
    try:
        pages: list[ParsedPage] = []
        with fitz.open(pdf_path) as doc:
            for index, page in enumerate(doc):
                pages.append(ParsedPage(page_index=index, text=page.get_text("text") or "", parser="pymupdf"))
        return pages
    except Exception as exc:
        raise PDFProcessingError(f"pymupdf failed for {pdf_path}: {exc}") from exc


def _extract_with_pdfplumber(pdf_path: Path) -> list[ParsedPage]:
    pdfplumber = _import_module("pdfplumber", "Install pdfplumber to use parser=pdfplumber.")
    try:
        pages: list[ParsedPage] = []
        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages):
                pages.append(ParsedPage(page_index=index, text=page.extract_text() or "", parser="pdfplumber"))
        return pages
    except Exception as exc:
        raise PDFProcessingError(f"pdfplumber failed for {pdf_path}: {exc}") from exc


def _extract_with_pypdf(pdf_path: Path) -> list[ParsedPage]:
    pypdf = _import_module("pypdf", "Install pypdf to use parser=pypdf.")
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        return [
            ParsedPage(page_index=index, text=page.extract_text() or "", parser="pypdf")
            for index, page in enumerate(reader.pages)
        ]
    except Exception as exc:
        raise PDFProcessingError(f"pypdf failed for {pdf_path}: {exc}") from exc


def _import_module(name: str, help_text: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise PDFProcessingError(help_text) from exc

