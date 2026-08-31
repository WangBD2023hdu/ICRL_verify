from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from arxiv_canonical_reflow_v4.core import (
    CanonicalBlock,
    CanonicalPage,
    build_page_tex,
)
from arxiv_canonical_reflow_v4.mutation import (
    CONFUSABLES,
    RenderedWord,
    apply_page_mutations,
    choose_page_mutations,
    markdown_diff_count,
    validate_mutated_word_geometry,
)
from scripts.experimental import build_arxiv_canonical_reflow_v4 as builder


def _block(
    block_id: str,
    markdown: str,
    *,
    kind: str = "paragraph",
) -> CanonicalBlock:
    return CanonicalBlock(
        block_id=block_id,
        node_id=f"node_{block_id}",
        kind=kind,
        markdown=markdown,
        latex=markdown + r"\par",
        verifier_text=markdown,
        weight=len(markdown),
        source_char_span=(0, len(markdown)),
        source_files=("main.tex",),
        has_table=kind == "table",
    )


def _word(
    text: str,
    index: int,
    *,
    column: int = 0,
) -> RenderedWord:
    x_min = 50.0 + (index % 2) * 100.0
    y_min = 60.0 + (index // 2) * 20.0
    return RenderedWord(
        text=text,
        column=column,
        x_min=x_min,
        y_min=y_min,
        x_max=x_min + 45.0,
        y_max=y_min + 10.0,
    )


def _prose_page() -> tuple[CanonicalPage, tuple[RenderedWord, ...]]:
    pairs = (
        ("planet", "follows"),
        ("cinder", "waits"),
        ("signal", "grows"),
        ("hover", "moves"),
        ("quiet", "stands"),
    )
    markdown = "\n".join(" ".join(pair) for pair in pairs)
    page = CanonicalPage(
        page_id="paper_page_1",
        paper_id="paper",
        ordinal=1,
        layout="one_column",
        blocks=(
            _block("heading", "Structural heading", kind="heading"),
            _block("body", markdown),
            _block("caption", "Measured caption", kind="caption"),
            _block("table", "<table><tr><td>Stable</td></tr></table>", kind="table"),
        ),
    )
    rendered = tuple(
        _word(word, index)
        for index, word in enumerate(word for pair in pairs for word in pair)
    )
    return page, rendered


def test_confusable_policy_is_lowercase_equal_length_and_digit_free() -> None:
    assert CONFUSABLES
    for source, targets in CONFUSABLES.items():
        assert len(source) == 1
        assert source.isascii() and source.islower() and not source.isdigit()
        for target in targets:
            assert len(target) == len(source) == 1
            assert target.isascii() and target.islower() and not target.isdigit()
            assert target != source


def test_selection_is_deterministic_paragraph_only_and_synchronized() -> None:
    page, rendered = _prose_page()
    first = choose_page_mutations(
        page,
        rendered,
        seed=19,
        minimum=3,
        maximum=3,
    )
    second = choose_page_mutations(
        page,
        rendered,
        seed=19,
        minimum=3,
        maximum=3,
    )
    assert first == second
    assert len(first) == 3
    assert {mutation.block_id for mutation in first} == {"body"}
    assert len({rendered[item.rendered_word_index].y_min for item in first}) == 3

    edited = apply_page_mutations(page, first, page_id="paper_page_1_edited")
    assert markdown_diff_count(page.markdown, edited.markdown) == 3
    assert markdown_diff_count(page.verifier_text, edited.verifier_text) == 3
    assert markdown_diff_count(build_page_tex(page), build_page_tex(edited)) == 3
    assert edited.blocks[0] == page.blocks[0]
    assert edited.blocks[2:] == page.blocks[2:]


def test_hidden_markdown_and_nonparagraph_text_are_not_mutated() -> None:
    markdown = (
        "planet follows ```cinder waits``` `signal grows` $hover moves$ "
        "https://example.com/quiet"
    )
    page = CanonicalPage(
        page_id="hidden",
        paper_id="paper",
        ordinal=1,
        layout="one_column",
        blocks=(
            _block("heading", "cinder waits", kind="heading"),
            _block("body", markdown),
        ),
    )
    rendered = (
        _word("planet", 0),
        _word("follows", 1),
        _word("cinder", 2),
        _word("waits", 3),
        _word("signal", 4),
        _word("grows", 5),
        _word("hover", 6),
        _word("moves", 7),
        _word("quiet", 8),
        _word("tail", 9),
    )
    selected = choose_page_mutations(
        page,
        rendered,
        seed=3,
        minimum=1,
        maximum=1,
    )
    assert len(selected) == 1
    assert selected[0].original_word == "planet"


def test_mutation_preserves_bold_markup_and_latex_wrapper() -> None:
    block = replace(
        _block("body", "**planet** follows"),
        latex=r"\textbf{planet} follows\par",
        verifier_text="planet follows",
    )
    page = CanonicalPage(
        page_id="bold",
        paper_id="paper",
        ordinal=1,
        layout="one_column",
        blocks=(block,),
    )
    rendered = (_word("planet", 0), _word("follows", 1))
    mutations = choose_page_mutations(
        page,
        rendered,
        seed=13,
        minimum=1,
        maximum=1,
    )
    edited = apply_page_mutations(page, mutations, page_id="bold_edited")
    changed = mutations[0].mutated_word
    assert edited.markdown == f"**{changed}** follows"
    assert edited.blocks[0].latex == rf"\textbf{{{changed}}} follows\par"


def test_geometry_gate_accepts_only_declared_equal_length_edits() -> None:
    page, clean = _prose_page()
    mutations = choose_page_mutations(
        page,
        clean,
        seed=5,
        minimum=3,
        maximum=3,
    )
    edited = list(clean)
    for mutation in mutations:
        original = edited[mutation.rendered_word_index]
        edited[mutation.rendered_word_index] = replace(
            original,
            text=mutation.mutated_word,
            x_max=original.x_max + 0.4,
        )
    validation = validate_mutated_word_geometry(clean, edited, mutations)
    assert validation.passed
    assert validation.max_vertical_shift_points == 0.0

    duplicate = (mutations[0], replace(mutations[1], rendered_word_index=mutations[0].rendered_word_index))
    assert (
        validate_mutated_word_geometry(clean, edited, duplicate).reason
        == "duplicate_rendered_word_index"
    )
    negative = (replace(mutations[0], rendered_word_index=-1),)
    assert (
        validate_mutated_word_geometry(clean, edited, negative).reason
        == "negative_rendered_word_index"
    )
    changed_column = list(edited)
    changed_column[0] = replace(changed_column[0], column=1)
    assert (
        validate_mutated_word_geometry(clean, changed_column, mutations).reason
        == "column_assignment_changed"
    )


def _worker_config(output: Path) -> builder.WorkerConfig:
    return builder.WorkerConfig(
        output_dir=str(output),
        latexmk="latexmk",
        pdftoppm="pdftoppm",
        pdftotext="pdftotext",
        pdfinfo="pdfinfo",
        compile_timeout=1,
        render_timeout=1,
        dpi=72,
        max_pack_attempts=2,
        min_page_chars=0,
        target_weight=1000,
        two_column_rate=0.0,
        target_fill_ratio=0.8,
        min_fill_ratio=0.6,
    )


def test_mutation_worker_recompiles_validates_and_exports_v1_changes(
    monkeypatch,
    tmp_path,
) -> None:
    output = tmp_path / "dataset"
    clean_dir = output / "pages" / "clean"
    clean_dir.mkdir(parents=True)
    clean_pdf = clean_dir / "page.pdf"
    clean_pdf.write_bytes(b"clean")
    clean_image = clean_dir / "page.png"
    Image.new("RGB", (612, 792), "white").save(clean_image)

    page = CanonicalPage(
        page_id="clean",
        paper_id="paper",
        ordinal=1,
        layout="one_column",
        blocks=(_block("body", "planet follows"),),
    )
    clean_result = builder.WorkerResult(
        page_id="clean",
        paper_id="paper",
        status="accepted",
        reason=None,
        layout="one_column",
        has_table=False,
        markdown=page.markdown,
        verifier_recall=1.0,
        verifier_precision=1.0,
        pdf=str(clean_pdf),
        image=str(clean_image),
        block_ids=("body",),
        source_node_ids=("node_body",),
        content_fill_ratio=0.8,
        column_fill_ratios=(0.8,),
        page_signature="clean-signature",
        elapsed_seconds=0.1,
    )
    clean_words = (
        RenderedWord("planet", 0, 50, 60, 95, 70),
        RenderedWord("follows", 0, 110, 60, 155, 70),
    )
    state: dict[str, str] = {}

    def fake_compile(
        mutated_page: CanonicalPage,
        _config: builder.WorkerConfig,
    ) -> builder.WorkerResult:
        edited_dir = output / "pages" / mutated_page.page_id
        edited_dir.mkdir(parents=True, exist_ok=True)
        edited_pdf = edited_dir / "page.pdf"
        edited_pdf.write_bytes(b"edited")
        edited_image = edited_dir / "page.png"
        Image.new("RGB", (612, 792), "white").save(edited_image)
        state["mutated_word"] = mutated_page.markdown.split()[0]
        return replace(
            clean_result,
            page_id=mutated_page.page_id,
            markdown=mutated_page.markdown,
            pdf=str(edited_pdf),
            image=str(edited_image),
            page_signature="edited-signature",
        )

    def fake_bbox(
        pdf: Path,
        *,
        layout: str,
        config: builder.WorkerConfig,
        page_dir: Path,
    ) -> tuple[builder.BBoxObservation, None]:
        del layout, config, page_dir
        words = clean_words
        if pdf.read_bytes() == b"edited":
            words = (
                replace(clean_words[0], text=state["mutated_word"]),
                clean_words[1],
            )
        return (
            builder.BBoxObservation(
                text=" ".join(word.text for word in words),
                words=words,
                content_fill_ratio=0.8,
                column_fill_ratios=(0.8,),
                page_width=612,
                page_height=792,
            ),
            None,
        )

    monkeypatch.setattr(builder, "_compile_once", fake_compile)
    monkeypatch.setattr(builder, "_bbox_text_for_verification", fake_bbox)
    config = _worker_config(output)
    mutation_config = builder.MutationConfig(
        seed=9,
        minimum_per_page=1,
        maximum_per_page=1,
        maximum_probability=0.6,
        max_vertical_shift_points=1.25,
    )
    result = builder._mutate_and_compile(
        clean_result,
        page,
        config,
        mutation_config,
    )
    assert result.status == "accepted"
    assert result.variant == "confusable_edit"
    assert result.mutation_count == 1
    assert result.clean_page_id == "clean"
    assert result.changes[0]["origin_ans"] == "planet"
    assert result.changes[0]["ocr_ans"] == state["mutated_word"]
    assert result.changes[0]["bbox"] == [50, 60, 95, 70]

    builder._export(
        output,
        [result],
        [],
        0.0,
        config,
        clean_results=[clean_result],
        mutation_config=mutation_config,
    )
    manifest = json.loads((output / "manifest.jsonl").read_text().strip())
    assert manifest["variant"] == "confusable_edit"
    assert manifest["mutation_count"] == 1
    verl = json.loads((output / "verl.jsonl").read_text().strip())
    assert verl["data_source"] == "chaos_document_ocr"
    assert set(verl["extra_info"]) == {"arxiv_id", "pair_id", "changes"}
    assert set(verl["extra_info"]["changes"][0]) == {
        "ocr_ans",
        "origin_ans",
        "bbox",
    }
    sft = json.loads((output / "sft.jsonl").read_text().strip())
    assert "changes" not in sft["extra_info"]
    report = json.loads((output / "pipeline_report.json").read_text())
    assert report["clean_pages_in_final_manifest"] == 0
    assert report["final_dataset_variant"] == "confusable_edited_only"
