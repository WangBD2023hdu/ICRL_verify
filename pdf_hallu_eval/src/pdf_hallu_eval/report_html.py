from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any, Iterable


def write_review_site(output_root: Path, page_records: Iterable[dict[str, Any]]) -> None:
    records = list(page_records)
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    by_pdf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pdf[str(record["pdf_id"])].append(record)

    for pdf_id, pdf_records in by_pdf.items():
        pdf_records.sort(key=lambda item: int(item.get("page_index", 0)))
        (review_dir / f"{pdf_id}.html").write_text(
            _render_pdf_page(output_root, pdf_id, pdf_records),
            encoding="utf-8",
        )

    (review_dir / "index.html").write_text(
        _render_index(by_pdf),
        encoding="utf-8",
    )


def _render_index(by_pdf: dict[str, list[dict[str, Any]]]) -> str:
    rows = []
    pdf_summaries = []
    for pdf_id, records in by_pdf.items():
        pdf_name = str(records[0].get("pdf_name", pdf_id))
        metrics = _summarize_record_metrics(records)
        pdf_summaries.append((metrics["hallucination_rate"], pdf_id, pdf_name, metrics))

    for _, pdf_id, pdf_name, metrics in sorted(pdf_summaries, reverse=True):
        rows.append(
            "<tr>"
            f"<td><a href=\"{escape(pdf_id)}.html\">{escape(pdf_name)}</a></td>"
            f"<td>{metrics['pages']}</td>"
            f"<td>{_fmt_rate(metrics['hallucination_rate'])}</td>"
            f"<td>{_fmt_rate(metrics['pure_insertion_rate'])}</td>"
            f"<td>{_fmt_rate(metrics['omission_rate'])}</td>"
            f"<td>{_fmt_rate(metrics['coverage'])}</td>"
            "</tr>"
        )

    return _html_document(
        "PDF Hallucination Review",
        "<main>"
        "<h1>PDF Hallucination Review</h1>"
        "<p class=\"subtle\">Sorted by broad character hallucination rate.</p>"
        "<table>"
        "<thead><tr>"
        "<th>PDF</th><th>pages</th><th>hallucination_rate</th>"
        "<th>pure_insertion_rate</th><th>omission_rate</th><th>coverage</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</main>",
    )


def _render_pdf_page(output_root: Path, pdf_id: str, records: list[dict[str, Any]]) -> str:
    pdf_name = str(records[0].get("pdf_name", pdf_id))
    panels = []
    for record in records:
        metrics = record.get("metrics", {})
        image_src = _relative_image_path(output_root, record.get("image_path", ""))
        page_number = int(record.get("page_index", 0)) + 1
        panels.append(
            "<section class=\"page-panel\">"
            f"<h2>Page {page_number}</h2>"
            "<div class=\"metric-strip\">"
            f"<span>hallucination_rate <strong>{_fmt_rate(metrics.get('hallucination_rate', 0.0))}</strong></span>"
            f"<span>pure_insertion_rate <strong>{_fmt_rate(metrics.get('pure_insertion_rate', 0.0))}</strong></span>"
            f"<span>omission_rate <strong>{_fmt_rate(metrics.get('omission_rate', 0.0))}</strong></span>"
            f"<span>coverage <strong>{_fmt_rate(metrics.get('coverage', 0.0))}</strong></span>"
            f"<span>status <strong>{escape(str(record.get('status', 'ok')))}</strong></span>"
            "</div>"
            "<div class=\"review-grid\">"
            f"<div class=\"page-image\"><img src=\"{escape(image_src)}\" alt=\"Rendered PDF page {page_number}\"></div>"
            "<div class=\"text-col\"><h3>Parser Text</h3>"
            f"<pre>{escape(str(record.get('parser_text', '')))}</pre></div>"
            "<div class=\"text-col\"><h3>Model Text</h3>"
            f"<pre>{escape(str(record.get('model_text', '')))}</pre></div>"
            "</div>"
            "<h3>Character Diff</h3>"
            f"<pre class=\"diff-line\">{_render_diff(record.get('alignment', []))}</pre>"
            "</section>"
        )

    return _html_document(
        f"{pdf_name} - Review",
        "<main>"
        f"<p><a href=\"index.html\">Back to index</a></p>"
        f"<h1>{escape(pdf_name)}</h1>"
        f"{''.join(panels)}"
        "</main>",
    )


def _render_diff(operations: list[dict[str, str]]) -> str:
    spans = []
    for op in operations:
        kind = op.get("kind", "")
        reference = op.get("reference", "")
        prediction = op.get("prediction", "")
        if kind == "match":
            spans.append(f"<span class=\"diff-match\">{escape(prediction)}</span>")
        elif kind == "substitute":
            title = f"reference: {reference}"
            spans.append(f"<span class=\"diff-substitute\" title=\"{escape(title)}\">{escape(prediction)}</span>")
        elif kind == "insert":
            spans.append(f"<span class=\"diff-insert\" title=\"unsupported insertion\">{escape(prediction)}</span>")
        elif kind == "delete":
            spans.append(f"<span class=\"diff-delete\" title=\"omitted from model\">{escape(reference)}</span>")
    return "".join(spans)


def _relative_image_path(output_root: Path, image_path: str | Path) -> str:
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.is_absolute():
        path = output_root / path
    try:
        return str(path.resolve().relative_to((output_root / "review").resolve()))
    except ValueError:
        return str(Path("..") / path.resolve().relative_to(output_root.resolve()))


def _summarize_record_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    totals = {
        "pages": len(records),
        "reference_chars": 0,
        "prediction_chars": 0,
        "matches": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }
    for record in records:
        metrics = record.get("metrics", {})
        for key in ("reference_chars", "prediction_chars", "matches", "substitutions", "deletions", "insertions"):
            totals[key] += int(metrics.get(key, 0))
    reference_chars = totals["reference_chars"]
    prediction_chars = totals["prediction_chars"]
    substitutions = totals["substitutions"]
    deletions = totals["deletions"]
    insertions = totals["insertions"]
    matches = totals["matches"]
    return {
        "pages": totals["pages"],
        "hallucination_rate": _rate(substitutions + insertions, prediction_chars),
        "pure_insertion_rate": _rate(insertions, prediction_chars),
        "omission_rate": _rate(substitutions + deletions, reference_chars),
        "coverage": _rate(matches, reference_chars),
    }


def _html_document(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{escape(title)}</title>\n"
        "<style>\n"
        "body{font-family:Arial,sans-serif;margin:0;background:#f7f8fa;color:#111827;}\n"
        "main{max-width:1440px;margin:0 auto;padding:24px;}\n"
        "table{border-collapse:collapse;width:100%;background:white;}\n"
        "th,td{border:1px solid #d8dee9;padding:8px;text-align:left;font-size:14px;}\n"
        "th{background:#edf2f7;}\n"
        ".subtle{color:#5b6472;}\n"
        ".page-panel{background:white;border:1px solid #d8dee9;border-radius:6px;margin:20px 0;padding:16px;}\n"
        ".metric-strip{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px;}\n"
        ".metric-strip span{background:#eef2f6;border:1px solid #d8dee9;border-radius:4px;padding:4px 8px;font-size:13px;}\n"
        ".review-grid{display:grid;grid-template-columns:minmax(260px,1fr) minmax(260px,1fr) minmax(260px,1fr);gap:12px;align-items:start;}\n"
        ".page-image img{max-width:100%;border:1px solid #d8dee9;background:#fff;}\n"
        "pre{white-space:pre-wrap;word-break:break-word;background:#fbfcfd;border:1px solid #d8dee9;border-radius:4px;padding:10px;line-height:1.5;}\n"
        ".diff-line{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}\n"
        ".diff-match{color:#1f2937;}\n"
        ".diff-insert{background:#ffd6d6;color:#8a1111;}\n"
        ".diff-substitute{background:#ffe1a6;color:#713f12;}\n"
        ".diff-delete{background:#e5e7eb;color:#4b5563;text-decoration:line-through;}\n"
        "@media(max-width:900px){.review-grid{grid-template-columns:1fr;}}\n"
        "</style>\n"
        "</head>\n"
        f"<body>{body}</body>\n"
        "</html>\n"
    )


def _fmt_rate(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "0.00%"


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator

