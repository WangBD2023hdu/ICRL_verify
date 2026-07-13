from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Protocol

from .align import align_text
from .chat_client import OpenAIChatClient
from .metrics import PageMetrics, compute_page_metrics, summarize_metrics
from .normalize import NormalizeConfig, normalize_text
from .pdf_parser import ParsedPage, extract_text_pages
from .pdf_render import RenderedPage, render_pdf_pages
from .report_html import write_review_site
from .storage import OutputLayout, make_pdf_id, write_json, write_jsonl, write_text


class ChatClient(Protocol):
    def transcribe_image(self, image_path: Path) -> str:
        ...


TextExtractor = Callable[[Path, str], list[ParsedPage]]
Renderer = Callable[..., list[RenderedPage]]


@dataclass(frozen=True)
class BatchConfig:
    pdf_dir: Path
    output_dir: Path
    parser: str = "auto"
    render_backend: str = "auto"
    dpi: int = 144
    workers: int = 1
    resume: bool = True
    force: bool = False
    limit: int | None = None
    preserve_newlines: bool = False
    dry_run: bool = False
    pdftoppm_path: str | None = None


@dataclass(frozen=True)
class BatchResult:
    output_dir: Path
    total_pdfs: int
    total_pages: int
    page_records: list[dict[str, Any]]


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    return sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())


def run_batch(
    config: BatchConfig,
    *,
    chat_client: ChatClient | OpenAIChatClient | None = None,
    text_extractor: TextExtractor = extract_text_pages,
    renderer: Renderer = render_pdf_pages,
) -> BatchResult:
    if not config.dry_run and chat_client is None:
        raise ValueError("chat_client is required unless dry_run=True")

    layout = OutputLayout(config.output_dir)
    layout.ensure()
    pdf_paths = discover_pdfs(config.pdf_dir)
    if config.limit is not None:
        pdf_paths = pdf_paths[: config.limit]

    def process(pdf_path: Path) -> list[dict[str, Any]]:
        return _process_pdf(
            pdf_path,
            config=config,
            layout=layout,
            chat_client=chat_client,
            text_extractor=text_extractor,
            renderer=renderer,
        )

    if config.workers > 1 and len(pdf_paths) > 1:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            nested_records = list(executor.map(process, pdf_paths))
    else:
        nested_records = [process(path) for path in pdf_paths]

    page_records = [record for records in nested_records for record in records]
    page_records.sort(key=lambda item: (str(item.get("pdf_id", "")), int(item.get("page_index", 0))))
    write_jsonl(config.output_dir / "pages.jsonl", page_records)
    _write_pdf_summary(config.output_dir / "pdf_summary.csv", page_records)
    write_json(config.output_dir / "dataset_summary.json", _dataset_summary(page_records))
    write_review_site(config.output_dir, page_records)
    return BatchResult(
        output_dir=config.output_dir,
        total_pdfs=len(pdf_paths),
        total_pages=len(page_records),
        page_records=page_records,
    )


def _process_pdf(
    pdf_path: Path,
    *,
    config: BatchConfig,
    layout: OutputLayout,
    chat_client: ChatClient | OpenAIChatClient | None,
    text_extractor: TextExtractor,
    renderer: Renderer,
) -> list[dict[str, Any]]:
    pdf_id = make_pdf_id(pdf_path)
    parsed_pages = text_extractor(pdf_path, parser=config.parser)
    with tempfile.TemporaryDirectory(prefix=f"{pdf_id}_", dir=str(layout.root / "images")) as tmp:
        rendered_pages = renderer(
            pdf_path,
            Path(tmp),
            backend=config.render_backend,
            dpi=config.dpi,
            pdftoppm_path=config.pdftoppm_path,
        )
        rendered_by_index = {page.page_index: page for page in rendered_pages}
        records = []
        for parsed_page in parsed_pages:
            records.append(
                _process_page(
                    pdf_path=pdf_path,
                    pdf_id=pdf_id,
                    parsed_page=parsed_page,
                    rendered_page=rendered_by_index.get(parsed_page.page_index),
                    config=config,
                    layout=layout,
                    chat_client=chat_client,
                )
            )
        return records


def _process_page(
    *,
    pdf_path: Path,
    pdf_id: str,
    parsed_page: ParsedPage,
    rendered_page: RenderedPage | None,
    config: BatchConfig,
    layout: OutputLayout,
    chat_client: ChatClient | OpenAIChatClient | None,
) -> dict[str, Any]:
    paths = layout.page_paths(pdf_id, parsed_page.page_index)
    if config.resume and not config.force and paths.alignment.exists():
        return json.loads(paths.alignment.read_text(encoding="utf-8"))

    raw_parser_text = parsed_page.text or ""
    status = "ok"
    error = ""
    if rendered_page is None:
        model_text = ""
        status = "render_missing"
        error = "No rendered image for page."
    else:
        paths.image.parent.mkdir(parents=True, exist_ok=True)
        if rendered_page.image_path.resolve() != paths.image.resolve():
            shutil.copyfile(rendered_page.image_path, paths.image)
        model_text = ""
        if config.dry_run:
            status = "dry_run"
        elif chat_client is not None:
            try:
                model_text = chat_client.transcribe_image(paths.image)
            except Exception as exc:
                status = "model_error"
                error = str(exc)

    normalized_reference = normalize_text(
        raw_parser_text,
        NormalizeConfig(preserve_newlines=config.preserve_newlines),
    )
    normalized_prediction = normalize_text(
        model_text,
        NormalizeConfig(preserve_newlines=config.preserve_newlines),
    )
    if status == "ok" and not normalized_reference:
        status = "parser_empty"

    alignment = align_text(normalized_reference, normalized_prediction)
    metrics = compute_page_metrics(alignment)
    record = {
        "pdf_id": pdf_id,
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "page_index": parsed_page.page_index,
        "parser": parsed_page.parser,
        "status": status,
        "error": error,
        "image_path": str(paths.image),
        "parser_text_path": str(paths.parser_text),
        "model_text_path": str(paths.model_text),
        "parser_text": raw_parser_text,
        "model_text": model_text,
        "normalized_parser_text": normalized_reference,
        "normalized_model_text": normalized_prediction,
        "metrics": metrics.to_dict(),
        "alignment": [
            {"kind": op.kind, "reference": op.reference, "prediction": op.prediction}
            for op in alignment.operations
        ],
    }
    write_text(paths.parser_text, raw_parser_text)
    write_text(paths.model_text, model_text)
    write_json(paths.alignment, record)
    return record


def _write_pdf_summary(path: Path, records: list[dict[str, Any]]) -> None:
    by_pdf: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_pdf.setdefault(str(record["pdf_id"]), []).append(record)

    fieldnames = [
        "pdf_id",
        "pdf_name",
        "pages",
        "reference_chars",
        "prediction_chars",
        "cer",
        "hallucination_rate",
        "pure_insertion_rate",
        "omission_rate",
        "coverage",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pdf_id, pdf_records in sorted(by_pdf.items()):
            summary = _summary_from_records(pdf_records)
            row = {"pdf_id": pdf_id, "pdf_name": pdf_records[0].get("pdf_name", pdf_id), **summary}
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _dataset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summary_from_records(records)
    summary["pdfs"] = len({record.get("pdf_id") for record in records})
    return summary


def _summary_from_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    metrics = [
        PageMetrics(
            reference_chars=int(record["metrics"]["reference_chars"]),
            prediction_chars=int(record["metrics"]["prediction_chars"]),
            matches=int(record["metrics"]["matches"]),
            substitutions=int(record["metrics"]["substitutions"]),
            deletions=int(record["metrics"]["deletions"]),
            insertions=int(record["metrics"]["insertions"]),
            cer=float(record["metrics"]["cer"]),
            hallucination_rate=float(record["metrics"]["hallucination_rate"]),
            pure_insertion_rate=float(record["metrics"]["pure_insertion_rate"]),
            omission_rate=float(record["metrics"]["omission_rate"]),
            coverage=float(record["metrics"]["coverage"]),
        )
        for record in records
    ]
    return summarize_metrics(metrics)
