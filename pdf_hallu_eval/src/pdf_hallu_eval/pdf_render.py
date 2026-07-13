from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import shutil
import subprocess
from typing import Literal

from .pdf_parser import PDFProcessingError


RenderBackend = Literal["auto", "pymupdf", "pdftoppm"]


@dataclass(frozen=True)
class RenderedPage:
    page_index: int
    image_path: Path
    backend: str


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    *,
    backend: RenderBackend = "auto",
    dpi: int = 144,
    pdftoppm_path: str | None = None,
) -> list[RenderedPage]:
    if backend not in {"auto", "pymupdf", "pdftoppm"}:
        raise ValueError(f"Unsupported render backend: {backend}")
    if not pdf_path.exists():
        raise PDFProcessingError(f"PDF does not exist: {pdf_path}")

    backends = ["pymupdf", "pdftoppm"] if backend == "auto" else [backend]
    errors: list[str] = []
    for candidate in backends:
        try:
            if candidate == "pymupdf":
                return _render_with_pymupdf(pdf_path, output_dir, dpi=dpi)
            if candidate == "pdftoppm":
                return _render_with_pdftoppm(pdf_path, output_dir, dpi=dpi, pdftoppm_path=pdftoppm_path)
        except PDFProcessingError as exc:
            errors.append(str(exc))
    raise PDFProcessingError("; ".join(errors) or f"Unable to render PDF: {pdf_path}")


def _render_with_pymupdf(pdf_path: Path, output_dir: Path, *, dpi: int) -> list[RenderedPage]:
    try:
        fitz = importlib.import_module("fitz")
    except ImportError as exc:
        raise PDFProcessingError("Install pymupdf to use render backend=pymupdf.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[RenderedPage] = []
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(pdf_path) as doc:
            for index, page in enumerate(doc):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = output_dir / f"page_{index + 1:04d}.png"
                pixmap.save(image_path)
                pages.append(RenderedPage(page_index=index, image_path=image_path, backend="pymupdf"))
        return pages
    except Exception as exc:
        raise PDFProcessingError(f"pymupdf render failed for {pdf_path}: {exc}") from exc


def _render_with_pdftoppm(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int,
    pdftoppm_path: str | None,
) -> list[RenderedPage]:
    executable = pdftoppm_path or shutil.which("pdftoppm")
    if not executable:
        raise PDFProcessingError("pdftoppm was not found. Install Poppler or pass --pdftoppm-path.")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "page"
    command = [executable, "-png", "-r", str(dpi), str(pdf_path), str(prefix)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise PDFProcessingError(f"pdftoppm render failed for {pdf_path}: {message}") from exc

    generated = sorted(output_dir.glob("page-*.png"))
    return [
        RenderedPage(page_index=index, image_path=image_path, backend="pdftoppm")
        for index, image_path in enumerate(generated)
    ]

