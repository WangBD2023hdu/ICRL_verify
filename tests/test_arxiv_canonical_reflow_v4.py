from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time

from arxiv_canonical_reflow_v4.core import (
    CanonicalBlock,
    CanonicalPage,
    _render_source_macro_fragment,
    blocks_from_document,
    build_page_tex,
    bundle_blocks,
    pack_blocks,
    render_table_html_to_latex,
    verify_rendered_text,
)
from arxiv_source_first_v3.document_ast import parse_document_ast
from scripts.experimental import build_arxiv_canonical_reflow_v4 as builder


def _write_crawler_archive(path, source: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = source.encode("utf-8")
    with tarfile.open(path, "w:gz") as bundle:
        member = tarfile.TarInfo("main.tex")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _block(
    index: int,
    *,
    kind: str = "paragraph",
    weight: int = 1000,
    has_table: bool = False,
) -> CanonicalBlock:
    return CanonicalBlock(
        block_id=f"b{index}",
        node_id=f"n{index}",
        kind=kind,
        markdown=f"block {index}",
        latex=f"block {index}\\par",
        verifier_text=f"block {index}",
        weight=weight,
        source_char_span=(index, index + 1),
        source_files=("main.tex",),
        has_table=has_table,
    )


def test_strict_source_ast_produces_text_and_clean_table_blocks() -> None:
    source = r"""
\documentclass{article}
\begin{document}
\section*{Introduction}

This is a \textbf{source-derived} paragraph with $x + 1$.

\begin{table}
\caption{Measured results}
\begin{tabular}{lc}
\textbf{Model} & \textbf{Score} \\
Alpha & 91 \\
Beta & 88 \\
\end{tabular}
\end{table}
\end{document}
"""
    document = parse_document_ast(
        source,
        source_id="paper",
        enable_strict_tables=True,
    )
    blocks, rejections = blocks_from_document(document, paper_id="paper")
    assert not rejections
    assert {block.kind for block in blocks} >= {
        "heading",
        "paragraph",
        "caption",
        "table",
    }
    table = next(block for block in blocks if block.kind == "table")
    assert table.markdown.startswith("<table>\n")
    assert "data-table-id" not in table.markdown
    assert "<strong>Model</strong>" in table.markdown
    assert r"\begin{tabularx}" in table.latex


def test_table_renderer_supports_colspan_and_rowspan() -> None:
    value = """<table>
  <thead>
    <tr><th rowspan="2">Model</th><th colspan="2">Scores</th></tr>
    <tr><th>A</th><th>B</th></tr>
  </thead>
  <tbody><tr><td>Base</td><td>1</td><td>2</td></tr></tbody>
</table>"""
    latex = render_table_html_to_latex(value)
    assert r"\multirow{2}{*}" in latex
    assert r"\multicolumn{2}{c}" in latex
    assert r"\begin{tabularx}" in latex


def test_table_renderer_breaks_long_literal_tokens_without_changing_gt() -> None:
    value = "<table><tr><td>CC12C=CC(=O)C=C1CCCC2</td><td>9</td></tr></table>"
    latex = render_table_html_to_latex(value)
    assert r"\seqsplit{CC12C=CC(=O)C=C1CCCC2}" in latex


def test_table_renderer_treats_unmatched_dollar_as_literal_text() -> None:
    value = "<table><tr><td>Price $5</td></tr></table>"
    latex = render_table_html_to_latex(value)
    assert r"Price \$5" in latex


def test_packer_mixes_layouts_without_requiring_tables() -> None:
    blocks = tuple(_block(index) for index in range(8))
    pages = pack_blocks(
        blocks,
        paper_id="paper",
        target_weight=2000,
        two_column_rate=0.5,
    )
    assert len(pages) == 4
    assert all(not page.has_table for page in pages)
    assert {page.layout for page in pages}.issubset({"one_column", "two_column"})
    assert all("block" in page.markdown for page in pages)


def test_bundle_blocks_keeps_heading_caption_and_table_together() -> None:
    blocks = (
        _block(0, kind="heading", weight=100),
        _block(1, kind="caption", weight=100),
        _block(2, kind="table", weight=900, has_table=True),
        _block(3),
    )
    bundles = bundle_blocks(blocks)
    assert [[block.kind for block in bundle] for bundle in bundles] == [
        ["heading", "caption", "table"],
        ["paragraph"],
    ]
    pages = pack_blocks(
        blocks,
        paper_id="paper",
        target_weight=1000,
        two_column_rate=1.0,
    )
    assert [block.kind for block in pages[0].blocks] == [
        "heading",
        "caption",
        "table",
    ]
    assert pages[0].layout == "one_column"


def test_bbox_parser_reports_bottom_fill_and_two_column_minimum() -> None:
    xml = """<doc><page width="612" height="792"><flow><block>
      <line xMin="55" yMin="60" xMax="580" yMax="650"><word xMin="55" yMin="60" xMax="75" yMax="70">Left</word></line>
      <line xMin="330" yMin="60" xMax="560" yMax="580"><word xMin="330" yMin="60" xMax="355" yMax="70">Right</word></line>
    </block></flow></page></doc>"""
    observed = builder._parse_bbox_output(xml, layout="two_column")
    assert observed.text == "Left\nRight"
    assert [word.text for word in observed.words] == ["Left", "Right"]
    assert [word.column for word in observed.words] == [0, 1]
    assert len(observed.column_fill_ratios) == 2
    assert observed.column_fill_ratios[0] > observed.column_fill_ratios[1]
    assert observed.content_fill_ratio == observed.column_fill_ratios[1]
    assert 0.75 < observed.content_fill_ratio < 0.80


def test_cli_default_minimum_fill_ratio_is_seventy_percent() -> None:
    args = builder._parser().parse_args(
        ["--papers-root", "papers", "--output-dir", "output"]
    )
    assert args.min_fill_ratio == 0.70


def test_cli_accepts_raw_crawler_root_and_128_workers() -> None:
    args = builder._parser().parse_args(
        [
            "--crawler-root",
            "crawl",
            "--output-dir",
            "output",
            "--workers",
            "128",
        ]
    )
    assert args.crawler_root.as_posix() == "crawl"
    assert args.papers_root is None
    assert args.workers == 128


def test_raw_crawler_archive_is_discovered_materialized_and_resumed(tmp_path) -> None:
    crawler = tmp_path / "crawl"
    archive = crawler / "papers" / "2601.00001v1" / "source_archive.bin"
    digest = _write_crawler_archive(
        archive,
        r"""\documentclass{article}
\begin{document}
This is source-derived text.
\end{document}
""",
    )
    row = {
        "stem": "2601.00001v1",
        "arxiv_id": "2601.00001",
        "version": "v1",
        "status": "passed",
        "archive": "papers/2601.00001v1/source_archive.bin",
        "bytes": archive.stat().st_size,
        "sha256": digest,
    }
    crawler.mkdir(parents=True, exist_ok=True)
    (crawler / "results.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    partial = crawler / "papers" / "2601.00002v1" / "source_archive.bin.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"unfinished")

    discovered = builder._discover_crawler_archives(crawler, limit=0, seed=7)
    assert [item.paper_id for item in discovered] == ["2601.00001v1"]
    cache = tmp_path / "cache"
    paper_dirs, preparation_rows, preparation_report = (
        builder._prepare_crawler_archives_parallel(
            discovered,
            cache_root=cache,
            workers=128,
            global_started=time.monotonic(),
        )
    )
    assert paper_dirs == [cache / "papers" / "2601.00001v1"]
    assert preparation_report["workers"] == 128
    first = preparation_rows[0]
    assert first["status"] == "prepared"
    assert first["resume_state"] == "created"
    paper_dir = cache / "papers" / "2601.00001v1"
    metadata = json.loads((paper_dir / "metadata.json").read_text())
    assert metadata["main_tex"] == "main.tex"
    assert metadata["archive_sha256"] == digest
    assert (paper_dir / "source" / "main.tex").is_file()

    pages, reports, extraction_errors = builder._extract_papers_parallel(
        paper_dirs,
        workers=128,
        target_weight=5200,
        two_column_rate=0.35,
        global_started=time.monotonic(),
    )
    assert extraction_errors == 0
    assert reports[0]["status"] == "success"
    assert pages

    second = builder._materialize_crawler_archive(discovered[0], cache)
    assert second["status"] == "prepared"
    assert second["resume_state"] == "reused"


def test_crawler_discovery_accepts_direct_papers_directory_without_results(
    tmp_path,
) -> None:
    papers = tmp_path / "papers"
    archive = papers / "fixtureA" / "source_archive.bin"
    _write_crawler_archive(
        archive,
        r"\documentclass{article}\begin{document}Text\end{document}",
    )
    discovered = builder._discover_crawler_archives(papers, limit=0, seed=1)
    assert [item.paper_id for item in discovered] == ["fixtureA"]


def test_crawler_discovery_accepts_completed_archive_not_yet_in_results(
    tmp_path,
) -> None:
    crawler = tmp_path / "crawl"
    listed = crawler / "papers" / "listed" / "source_archive.bin"
    unlisted = crawler / "papers" / "unlisted" / "source_archive.bin"
    listed_sha = _write_crawler_archive(
        listed,
        r"\documentclass{article}\begin{document}Listed\end{document}",
    )
    _write_crawler_archive(
        unlisted,
        r"\documentclass{article}\begin{document}Unlisted\end{document}",
    )
    (crawler / "results.jsonl").write_text(
        json.dumps(
            {
                "stem": "listed",
                "status": "passed",
                "sha256": listed_sha,
                "archive": "papers/listed/source_archive.bin",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    discovered = builder._discover_crawler_archives(crawler, limit=0, seed=1)
    assert {item.paper_id for item in discovered} == {"listed", "unlisted"}


def test_dense_compile_backoff_keeps_maximal_verified_prefix(
    monkeypatch, tmp_path
) -> None:
    page = CanonicalPage(
        page_id="source_pool",
        paper_id="paper",
        ordinal=1,
        layout="one_column",
        blocks=tuple(_block(index) for index in range(10)),
    )
    config = builder.WorkerConfig(
        output_dir=str(tmp_path),
        latexmk="latexmk",
        pdftoppm="pdftoppm",
        pdftotext="pdftotext",
        pdfinfo="pdfinfo",
        compile_timeout=1,
        render_timeout=1,
        dpi=72,
        max_pack_attempts=12,
        min_page_chars=0,
        target_weight=7000,
        two_column_rate=0.0,
        target_fill_ratio=0.85,
        min_fill_ratio=0.60,
    )

    def fake_compile(
        candidate: CanonicalPage, _config: builder.WorkerConfig
    ) -> builder.WorkerResult:
        weight = sum(block.weight for block in candidate.blocks)
        accepted = weight <= 6000
        return builder.WorkerResult(
            page_id=candidate.page_id,
            paper_id=candidate.paper_id,
            status="accepted" if accepted else "rejected",
            reason=None if accepted else "canonical_page_count:2",
            layout=candidate.layout,
            has_table=candidate.has_table,
            markdown=candidate.markdown,
            verifier_recall=1.0 if accepted else None,
            verifier_precision=1.0 if accepted else None,
            pdf="page.pdf" if accepted else None,
            image="page.png" if accepted else None,
            block_ids=tuple(block.block_id for block in candidate.blocks),
            source_node_ids=tuple(block.node_id for block in candidate.blocks),
            content_fill_ratio=weight / 6500 if accepted else None,
            column_fill_ratios=(weight / 6500,) if accepted else (),
            page_signature="signature",
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(builder, "_compile_once", fake_compile)
    results = builder._compile_with_rescue(page, config)
    assert [result.status for result in results] == ["accepted", "accepted"]
    assert [result.block_ids for result in results] == [
        tuple(f"b{index}" for index in range(6)),
        tuple(f"b{index}" for index in range(6, 10)),
    ]
    assert results[0].pack_attempts == 2
    assert all((result.content_fill_ratio or 0) >= 0.60 for result in results)


def test_dense_compile_skips_proven_bad_bundle_and_continues_filling(
    monkeypatch, tmp_path
) -> None:
    page = CanonicalPage(
        page_id="source_pool",
        paper_id="paper",
        ordinal=1,
        layout="one_column",
        blocks=tuple(_block(index) for index in range(5)),
    )
    config = builder.WorkerConfig(
        output_dir=str(tmp_path),
        latexmk="latexmk",
        pdftoppm="pdftoppm",
        pdftotext="pdftotext",
        pdfinfo="pdfinfo",
        compile_timeout=1,
        render_timeout=1,
        dpi=72,
        max_pack_attempts=12,
        min_page_chars=0,
        target_weight=2000,
        two_column_rate=0.0,
        target_fill_ratio=0.80,
        min_fill_ratio=0.60,
    )

    def fake_compile(
        candidate: CanonicalPage, _config: builder.WorkerConfig
    ) -> builder.WorkerResult:
        ids = tuple(block.block_id for block in candidate.blocks)
        accepted = "b2" not in ids
        fill = len(ids) / 5 if accepted else None
        return builder.WorkerResult(
            page_id=candidate.page_id,
            paper_id=candidate.paper_id,
            status="accepted" if accepted else "rejected",
            reason=None if accepted else "rendered_text_mismatch:recall=0.5",
            layout=candidate.layout,
            has_table=candidate.has_table,
            markdown=candidate.markdown,
            verifier_recall=1.0 if accepted else 0.5,
            verifier_precision=1.0 if accepted else 0.5,
            pdf="page.pdf",
            image="page.png",
            block_ids=ids,
            source_node_ids=tuple(block.node_id for block in candidate.blocks),
            content_fill_ratio=fill,
            column_fill_ratios=(fill,) if fill is not None else (),
            page_signature="signature",
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(builder, "_compile_once", fake_compile)
    results = builder._compile_with_rescue(page, config)
    assert [result.status for result in results] == ["rejected", "accepted"]
    assert results[0].block_ids == ("b2",)
    assert results[1].block_ids == ("b0", "b1", "b3", "b4")
    assert results[1].content_fill_ratio == 0.8


def test_page_tex_and_reject_only_verifier() -> None:
    block = CanonicalBlock(
        block_id="b1",
        node_id="n1",
        kind="paragraph",
        markdown="A faithful paragraph.",
        latex=r"A faithful paragraph.\par",
        verifier_text="A faithful paragraph.",
        weight=20,
        source_char_span=(0, 20),
        source_files=("main.tex",),
    )
    page = CanonicalPage(
        page_id="p1",
        paper_id="paper",
        ordinal=1,
        layout="one_column",
        blocks=(block,),
    )
    tex = build_page_tex(page)
    assert r"\pagestyle{empty}" in tex
    assert r"\begin{document}" in tex
    assert "A faithful paragraph" in tex
    passed, recall, precision = verify_rendered_text(
        page.verifier_text,
        "A faithful paragraph.",
    )
    assert passed
    assert recall == 1.0
    assert precision == 1.0
    assert not verify_rendered_text(page.verifier_text, "Different content")[0]


def test_source_macro_markdown_styles_do_not_render_literal_markers() -> None:
    rendered = _render_source_macro_fragment(
        "**C1: strong** and *emphasis* with <sup>2</sup> and $x+1$"
    )
    assert rendered == (
        r"\textbf{C1: strong} and \emph{emphasis} with "
        r"\textsuperscript{2} and $x+1$"
    )
    assert "**" not in rendered
