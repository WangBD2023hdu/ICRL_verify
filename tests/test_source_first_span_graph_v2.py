from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experimental"
    / "build_source_first_span_graph_v2.py"
)
SPEC = importlib.util.spec_from_file_location("build_source_first_span_graph_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SourceFirstSpanGraphV2Tests(unittest.TestCase):
    def test_optional_environment_title_fragment_is_rejected_pre_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory).resolve() / "main.tex"
            source_file.write_text(
                "\\begin{definition}[Replica-Exclusion]\n"
                "Body.\n"
                "\\end{definition}\n",
                encoding="utf-8",
            )
            parser_fragment = MODULE.page_gt.SourceParagraph(
                paragraph_id="false-optional-title",
                kind="paragraph",
                source_file=source_file,
                source_lines=[1],
                raw_latex="[Replica-Exclusion]",
            )
            units, rejected, _wrappers, report = (
                MODULE.build_source_units_with_visible_wrappers(
                    [parser_fragment],
                    references={},
                    macros={},
                )
            )
            self.assertFalse(units)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                rejected[0]["reason"],
                "structural_argument_fragment:environment_optional_argument_fragment",
            )
            provenance = rejected[0]["syntactic_provenance"]
            self.assertEqual(provenance["argument"]["command"], "begin")
            self.assertEqual(provenance["argument"]["delimiter"], "[]")
            self.assertFalse(provenance["pdf_text_used"])
            self.assertEqual(report["structural_argument_fragments_rejected"], 1)
            self.assertEqual(
                report["safe_macro_admission"][
                    "structural_argument_fragments_rejected"
                ],
                1,
            )

    def test_shadow_gate_never_instruments_macro_expansion_atom_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            clean = root / "clean"
            clean.mkdir()
            source_file = clean / "main.tex"
            original = (
                "\\newcommand{\\DAL}{DA-CR}\n"
                "The \\DAL protocol is available.\n"
            )
            source_file.write_text(original, encoding="utf-8")
            unit = MODULE.stable.SourceUnit(
                unit_id="expanded-body",
                kind="paragraph",
                paragraph_id="p1",
                source_file=source_file,
                source_lines=(2,),
                raw_latex=r"The \DAL protocol is available.",
                markdown="The DA-CR protocol is available.",
                rgb=(20, 30, 40),
            )
            source_start = original.index("The ")
            probe = MODULE.stable.SourceProbe(
                probe_id="expanded-body-atom-00001",
                unit_id=unit.unit_id,
                paragraph_id=unit.paragraph_id,
                kind=unit.kind,
                source_file=source_file,
                source_lines=unit.source_lines,
                markdown_fragment="The",
                rgb=(50, 60, 70),
                ordinal=1,
                total=1,
                localization_mode="source_atom",
            )
            locator = MODULE.AtomLocator(
                probe_id=probe.probe_id,
                source_file=source_file,
                source_start=source_start,
                source_end=source_start + 3,
                atom_ordinal=0,
            )
            shadow = root / "shadow"
            shutil.copytree(clean, shadow)
            report = MODULE.instrument_shadow_tree(
                clean,
                shadow,
                [unit],
                [probe],
                {probe.probe_id: locator},
                "pdflatex",
                macro_expansion_mismatches={
                    unit.unit_id: {
                        "reason": (
                            "macro_expanded_markdown_original_atom_mismatch"
                        ),
                        "pdf_text_used": False,
                    }
                },
            )
            self.assertEqual((shadow / "main.tex").read_text(), original)
            self.assertEqual(report["macro_expansion_mismatch_units_rejected"], 1)
            self.assertEqual(report["metadata_only_unit_ids"], [unit.unit_id])
            self.assertFalse(
                report["executable_color_inserted_for_rejected_units"]
            )
            self.assertEqual(
                report["rejected_unit_provenance"][0]["localization_fallback"],
                "synctex_clean_only",
            )

    @unittest.skipUnless(shutil.which("latexmk"), "latexmk is required")
    def test_optional_environment_title_shadow_preserves_semantics_and_page_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            clean = root / "clean"
            clean.mkdir()
            source_file = clean / "main.tex"
            original = (
                "\\documentclass{article}\n"
                "\\newtheorem{definition}{Definition}\n"
                "\\begin{document}\n"
                "\\begin{definition}[Replica-Exclusion]\n"
                "Body.\n"
                "\\end{definition}\n"
                "\\end{document}\n"
            )
            source_file.write_text(original, encoding="utf-8")
            false_unit = MODULE.stable.SourceUnit(
                unit_id="false-optional-title",
                kind="paragraph",
                paragraph_id="p1",
                source_file=source_file,
                source_lines=(4,),
                raw_latex="[Replica-Exclusion]",
                markdown=r"\[Replica-Exclusion\]",
                rgb=(20, 30, 40),
            )
            false_probe = MODULE.stable.SourceProbe(
                probe_id="false-optional-title-whole",
                unit_id=false_unit.unit_id,
                paragraph_id=false_unit.paragraph_id,
                kind=false_unit.kind,
                source_file=source_file,
                source_lines=false_unit.source_lines,
                markdown_fragment=false_unit.markdown,
                rgb=false_unit.rgb,
                ordinal=1,
                total=1,
                localization_mode="whole",
            )
            shadow = root / "shadow"
            shutil.copytree(clean, shadow)
            report = MODULE.instrument_shadow_tree(
                clean,
                shadow,
                [false_unit],
                [false_probe],
                {},
                "pdflatex",
            )
            self.assertEqual((shadow / "main.tex").read_text(), original)
            self.assertEqual(report["syntactic_gate_units_rejected"], 1)
            self.assertEqual(
                report["rejection_reasons"],
                {"environment_optional_argument_fragment": 1},
            )

            latexmk = Path(shutil.which("latexmk") or "latexmk")
            previous = MODULE.color_pilot.LATEXMK
            MODULE.color_pilot.LATEXMK = latexmk
            try:
                clean_pdf = MODULE.color_pilot.run_compile(
                    source_root=clean,
                    main_tex=Path("main.tex"),
                    build_dir=root / "build-clean",
                    log_path=root / "clean.log",
                    label="test-optional-title-clean",
                    timeout_seconds=60,
                    engine="pdflatex",
                )
                shadow_pdf = MODULE.color_pilot.run_compile(
                    source_root=shadow,
                    main_tex=Path("main.tex"),
                    build_dir=root / "build-shadow",
                    log_path=root / "shadow.log",
                    label="test-optional-title-shadow",
                    timeout_seconds=60,
                    engine="pdflatex",
                )
            finally:
                MODULE.color_pilot.LATEXMK = previous
            invariance = MODULE.compare_pdf_logical_invariance(
                clean_pdf, shadow_pdf
            )
            self.assertTrue(invariance["page_count_equal"])
            self.assertTrue(invariance["all_pages_equal"])
            with MODULE.pdfplumber.open(shadow_pdf) as document:
                self.assertIn(
                    "Definition 1 (Replica-Exclusion)",
                    document.pages[0].extract_text(),
                )

    def test_source_list_ir_merges_continuation_and_preserves_source_macro(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "main.tex"
            source_file.write_text(
                "\\newcommand{\\term}{Term}\n"
                "\\begin{itemize}\n"
                "\\item first \\term\n"
                "continued\n"
                "\\item second\n"
                "\\end{itemize}\n"
                "\\begin{description}\n"
                "\\item[Label] described\n"
                "\\end{description}\n",
                encoding="utf-8",
            )
            registry = MODULE.collect_safe_macros([source_file])
            paragraphs = [
                MODULE.page_gt.SourceParagraph(
                    paragraph_id="list-p1",
                    kind="itemize_item",
                    source_file=source_file,
                    source_lines=[3],
                    raw_latex=r"\item first \term",
                    list_environment="itemize",
                    item_depth=1,
                    item_ordinal=1,
                ),
                MODULE.page_gt.SourceParagraph(
                    paragraph_id="list-p1-cont",
                    kind="itemize_item",
                    source_file=source_file,
                    source_lines=[4],
                    raw_latex="continued",
                    list_environment="itemize",
                    item_depth=1,
                    item_ordinal=1,
                ),
                MODULE.page_gt.SourceParagraph(
                    paragraph_id="list-p2",
                    kind="itemize_item",
                    source_file=source_file,
                    source_lines=[5],
                    raw_latex=r"\item second",
                    list_environment="itemize",
                    item_depth=1,
                    item_ordinal=2,
                ),
                MODULE.page_gt.SourceParagraph(
                    paragraph_id="list-p3",
                    kind="description_item",
                    source_file=source_file,
                    source_lines=[8],
                    raw_latex=r"\item[Label] described",
                    list_environment="description",
                    item_depth=1,
                    item_ordinal=1,
                ),
            ]
            units, rejected, _wrappers, report = (
                MODULE.build_source_units_with_visible_wrappers(
                    paragraphs,
                    references={},
                    macros={},
                    safe_macros=registry,
                )
            )
            self.assertFalse(rejected)
            self.assertEqual(len(units), 3)
            self.assertEqual(
                [unit.markdown for unit in units],
                [
                    "- first Term\n  continued",
                    "- second",
                    "- **Label** described",
                ],
            )
            self.assertEqual(units[0].paragraph_id, "list-p1")
            self.assertEqual(units[0].source_lines, (3, 4))
            self.assertEqual(units[0].raw_latex, r"\item first \term" + "\ncontinued")
            self.assertEqual(report["list_admission"]["continuations"], 1)
            self.assertTrue(report["safe_macro_admission"]["original_source_provenance_preserved"])

    def test_source_list_ir_rejects_parent_continuation_over_nested_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "nested.tex"
            source_file.write_text(
                "\\begin{itemize}\n"
                "\\item outer\n"
                "\\begin{enumerate}\n"
                "\\item inner\n"
                "\\end{enumerate}\n"
                "outer continuation\n"
                "\\end{itemize}\n",
                encoding="utf-8",
            )
            paragraphs = [
                MODULE.page_gt.SourceParagraph(
                    paragraph_id="outer",
                    kind="itemize_item",
                    source_file=source_file,
                    source_lines=[2],
                    raw_latex=r"\item outer",
                    list_environment="itemize",
                    item_depth=1,
                    item_ordinal=1,
                ),
                MODULE.page_gt.SourceParagraph(
                    paragraph_id="inner",
                    kind="enumerate_item",
                    source_file=source_file,
                    source_lines=[4],
                    raw_latex=r"\item inner",
                    list_environment="enumerate",
                    item_depth=2,
                    item_ordinal=1,
                ),
                MODULE.page_gt.SourceParagraph(
                    paragraph_id="outer-cont",
                    kind="itemize_item",
                    source_file=source_file,
                    source_lines=[6],
                    raw_latex="outer continuation",
                    list_environment="itemize",
                    item_depth=1,
                    item_ordinal=1,
                ),
            ]
            units, rejected, _wrappers, report = (
                MODULE.build_source_units_with_visible_wrappers(
                    paragraphs,
                    references={},
                    macros={},
                )
            )
            self.assertEqual([unit.paragraph_id for unit in units], ["inner"])
            self.assertTrue(any("overlaps" in str(item.get("reason")) for item in rejected))
            self.assertEqual(report["list_admission"]["rejected_items"], 1)

    def test_source_list_ir_isolates_one_unsafe_description_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory).resolve() / "lists.tex"
            source_file.write_text(
                "\\begin{itemize}\n"
                "\\item safe one\n"
                "\\item safe two\n"
                "\\end{itemize}\n"
                "\\begin{description}\n"
                "\\item[\\dynamiclabel] unsafe\n"
                "\\end{description}\n"
                "\\begin{enumerate}\n"
                "\\item safe three\n"
                "\\end{enumerate}\n",
                encoding="utf-8",
            )

            def paragraph(
                identifier: str,
                line: int,
                raw: str,
                environment: str,
                ordinal: int,
            ) -> object:
                return MODULE.page_gt.SourceParagraph(
                    paragraph_id=identifier,
                    kind=f"{environment}_item",
                    source_file=source_file,
                    source_lines=[line],
                    raw_latex=raw,
                    list_environment=environment,
                    item_depth=1,
                    item_ordinal=ordinal,
                )

            paragraphs = [
                paragraph("safe-1", 2, r"\item safe one", "itemize", 1),
                paragraph("safe-2", 3, r"\item safe two", "itemize", 2),
                paragraph(
                    "unsafe-description",
                    6,
                    r"\item[\dynamiclabel] unsafe",
                    "description",
                    1,
                ),
                paragraph("safe-3", 9, r"\item safe three", "enumerate", 1),
            ]
            units, rejected, report = MODULE.build_source_list_units(
                paragraphs,
                references={},
            )
            self.assertEqual(
                [unit.paragraph_id for unit in units],
                ["safe-1", "safe-2", "safe-3"],
            )
            self.assertEqual(
                [unit.markdown for unit in units],
                ["- safe one", "- safe two", "1. safe three"],
            )
            self.assertEqual(
                {item.get("source_paragraph_id") for item in rejected},
                {"unsafe-description"},
            )
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["instances_total"], 3)
            self.assertEqual(report["instances_accepted"], 2)
            self.assertEqual(report["instances_rejected"], 1)
            self.assertFalse(report["pdf_text_used"])

    def test_safe_macro_pre_admission_expands_already_accepted_body_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "main.tex"
            source.write_text(
                "\\newcommand{\\name}{BigDipper\\xspace}\n"
                "\\newcommand{\\DAL}{\\textmd{DA-CR}\\xspace}\n"
                "The \\DAL protocol uses $\\DAL$.\n",
                encoding="utf-8",
            )
            registry = MODULE.collect_safe_macros([source])
            paragraph = MODULE.page_gt.SourceParagraph(
                paragraph_id="p1",
                kind="paragraph",
                source_file=source,
                source_lines=[3],
                raw_latex=r"The \DAL protocol uses $\DAL$.",
            )
            # The legacy path accepts the formula custom macro unchanged.  The
            # unified admission path must expand it even though the unit was
            # not previously rejected.
            legacy = MODULE.stable.source_paragraph_to_markdown(
                MODULE.page_gt.SourceParagraph(
                    paragraph_id="legacy",
                    kind="paragraph",
                    source_file=source,
                    source_lines=[3],
                    raw_latex=r"The protocol uses $\DAL$.",
                ),
                {},
            )
            self.assertIn(r"\DAL", legacy)
            units, rejected, _wrappers, report = (
                MODULE.build_source_units_with_visible_wrappers(
                    [paragraph],
                    references={},
                    macros={},
                    safe_macros=registry,
                )
            )
            self.assertFalse(rejected)
            self.assertEqual(
                units[0].markdown,
                r"The DA-CR protocol uses $\textmd{DA-CR}$.",
            )
            self.assertEqual(units[0].raw_latex, paragraph.raw_latex)
            self.assertEqual(units[0].source_lines, (3,))
            self.assertEqual(report["safe_macro_recovered_units"], 1)
            admission = report["safe_macro_admission"]
            self.assertEqual(
                (admission["total"], admission["changed"], admission["successful"], admission["rejected"]),
                (1, 1, 1, 0),
            )
            self.assertEqual(admission["provenance"][0]["macros"], ["DAL"])
            self.assertTrue(
                admission["provenance"][0]["original_raw_latex_preserved"]
            )

    def test_safe_macro_pre_admission_expands_existing_and_missing_headings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "main.tex"
            source.write_text(
                "\\newcommand{\\name}{BigDipper\\xspace}\n"
                "\\section{Overview of \\name}\n",
                encoding="utf-8",
            )
            registry = MODULE.collect_safe_macros([source])

            block = MODULE.page_gt.SourceBlock(
                block_id="h1",
                kind="heading",
                source_file=source,
                start_line=2,
                end_line=2,
                raw_latex=r"\section{Overview of \name}",
                markdown=None,
                query_lines=[2],
                heading_command="section",
                heading_level=2,
                heading_starred=False,
                heading_source_title=r"Overview of \name",
            )
            admitted, titles, rejected, admission = (
                MODULE.prepare_heading_blocks_for_safe_admission(
                    [block], registry, references={}
                )
            )
            self.assertFalse(rejected)
            self.assertEqual(titles, {"h1": "Overview of BigDipper"})
            self.assertEqual(
                (admission["total"], admission["changed"], admission["successful"], admission["rejected"]),
                (1, 1, 1, 0),
            )
            existing = MODULE.stable.SourceUnit(
                unit_id="heading-existing",
                kind="heading",
                paragraph_id="h1",
                source_file=source,
                source_lines=(2,),
                raw_latex=block.raw_latex,
                markdown="## 2 Overview of",
                rgb=(1, 2, 3),
                source_command="section",
            )
            normalized, application = MODULE.apply_compiler_heading_labels(
                [existing],
                admitted,
                {(source, 2, "section"): "2"},
                titles,
            )
            self.assertFalse(application["rejections"])
            self.assertEqual(normalized[0].markdown, "## 2 Overview of BigDipper")
            self.assertEqual(normalized[0].raw_latex, block.raw_latex)
            self.assertEqual(normalized[0].source_lines, (2,))

            recovered, heading_report = MODULE.recover_compiler_labeled_headings(
                [],
                admitted,
                {(source, 2, "section"): "2"},
                titles,
            )
            self.assertEqual(recovered[0].markdown, "## 2 Overview of BigDipper")
            self.assertEqual(heading_report["recovered"], 1)
            self.assertEqual(recovered[0].raw_latex, block.raw_latex)
            self.assertEqual(recovered[0].source_lines, (2,))

    def test_unknown_heading_macro_is_rejected_before_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "main.tex"
            source.write_text(
                "\\section{Overview of \\mystery}\n", encoding="utf-8"
            )
            registry = MODULE.collect_safe_macros([source])
            block = MODULE.page_gt.SourceBlock(
                block_id="unknown-heading",
                kind="heading",
                source_file=source,
                start_line=1,
                end_line=1,
                raw_latex=r"\section{Overview of \mystery}",
                markdown=None,
                query_lines=[1],
                heading_command="section",
                heading_level=2,
                heading_starred=False,
                heading_source_title=r"Overview of \mystery",
            )
            admitted, titles, rejected, report = (
                MODULE.prepare_heading_blocks_for_safe_admission(
                    [block], registry, references={}
                )
            )
            self.assertFalse(admitted)
            self.assertFalse(titles)
            self.assertEqual(len(rejected), 1)
            self.assertIn("unknown macros", rejected[0]["reason"])
            self.assertEqual(report["rejected"], 1)

    def test_unknown_body_macro_is_rejected_without_losing_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "main.tex"
            raw = r"This invokes \unknownmacro{content}."
            source.write_text(raw + "\n", encoding="utf-8")
            registry = MODULE.collect_safe_macros([source])
            paragraph = MODULE.page_gt.SourceParagraph(
                paragraph_id="unknown-body",
                kind="paragraph",
                source_file=source,
                source_lines=[1],
                raw_latex=raw,
            )
            units, rejected, _wrappers, report = (
                MODULE.build_source_units_with_visible_wrappers(
                    [paragraph],
                    references={},
                    macros={},
                    safe_macros=registry,
                )
            )
            self.assertFalse(units)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["source_paragraph_id"], "unknown-body")
            admission = report["safe_macro_admission"]
            self.assertEqual(
                (admission["total"], admission["successful"], admission["rejected"]),
                (1, 0, 1),
            )
            self.assertEqual(rejected[0]["source_lines"], [1, 1])
            self.assertEqual(rejected[0]["source_file"], str(source))

    def test_heading_href_keeps_only_visible_argument_via_stable_serializer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "main.tex"
            title = (
                r"Appendix Resources: \href{https://example.test/data}{Online}; "
                r"see Section \ref{sec:source}"
            )
            raw = "\\section{" + title + "}"
            source.write_text(raw + "\n", encoding="utf-8")
            registry = MODULE.collect_safe_macros([source])
            block = MODULE.page_gt.SourceBlock(
                block_id="href-heading",
                kind="heading",
                source_file=source,
                start_line=1,
                end_line=1,
                raw_latex=raw,
                markdown=None,
                query_lines=[1],
                heading_command="section",
                heading_level=2,
                heading_starred=False,
                heading_source_title=title,
            )
            admitted, titles, rejected, _report = (
                MODULE.prepare_heading_blocks_for_safe_admission(
                    [block],
                    registry,
                    references={
                        "sec:source": MODULE.stable.AuxReference(
                            label="sec:source", number="4.2", page="11", kind="section"
                        )
                    },
                )
            )
            self.assertFalse(rejected)
            self.assertEqual(
                titles["href-heading"],
                "Appendix Resources: Online; see Section 4.2",
            )
            self.assertNotIn("example.test", titles["href-heading"])
            self.assertNotIn("sec:source", titles["href-heading"])
            existing = MODULE.stable.SourceUnit(
                unit_id="href-existing",
                kind="heading",
                paragraph_id=block.block_id,
                source_file=source,
                source_lines=(1,),
                raw_latex=raw,
                markdown=(
                    "## A Appendix Resources: "
                    "(https://example.test/dataOnline); see Section sec:source"
                ),
                rgb=(1, 2, 3),
                source_command="section",
            )
            normalized, application = MODULE.apply_compiler_heading_labels(
                [existing],
                admitted,
                {(source, 1, "section"): "A"},
                titles,
            )
            self.assertFalse(application["rejections"])
            self.assertEqual(
                normalized[0].markdown,
                "## A Appendix Resources: Online; see Section 4.2",
            )

    def test_external_verbatim_lines_form_one_source_derived_fenced_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            main = root / "main.tex"
            external = root / "sample.txt"
            main.write_text(r"\verbatiminput{sample.txt}" + "\n", encoding="utf-8")
            external.write_text("alpha\n   \n\t\ngamma\n", encoding="utf-8")
            ir = MODULE.build_external_verbatim_ir(root, [main])
            units, blocks = MODULE.external_verbatim_units(
                ir, color_index_offset=0
            )
            base, modes = MODULE.stable.build_source_probes(
                units, word_probe_kinds=set()
            )
            probes, modes = MODULE.replace_external_verbatim_line_probes(
                base, modes, blocks
            )
            self.assertEqual(len(probes), 2)
            self.assertEqual([probe.ordinal for probe in probes], [1, 4])
            self.assertEqual([probe.total for probe in probes], [4, 4])
            self.assertTrue(
                all(
                    value == MODULE.EXTERNAL_VERBATIM_LOCALIZATION_MODE
                    for value in modes.values()
                )
            )
            rows = {
                probe.probe_id: [
                    {
                        "page_number": 1,
                        "bbox_points": [50, 100 + 12 * index, 150, 110 + 12 * index],
                        "components": [
                            [50, 100 + 12 * index, 150, 110 + 12 * index]
                        ],
                    }
                ]
                for index, probe in enumerate(probes)
            }
            fragments, reasons, _summary = MODULE.build_fragments_for_shadow(
                units,
                probes,
                rows,
                {},
                {},
                {1: 600.0},
                external_blocks=blocks,
            )
            self.assertFalse(reasons)
            self.assertEqual(len(fragments[1]), 1)
            self.assertEqual(
                fragments[1][0].markdown,
                "```text\nalpha\n   \n\t\ngamma\n```",
            )

    def test_max_pages_drops_truncated_probes_without_breaking_whole_units(self) -> None:
        source_file = Path("main.tex")

        def unit(unit_id: str, mode: str) -> object:
            return MODULE.stable.SourceUnit(
                unit_id=unit_id,
                kind="paragraph",
                paragraph_id=unit_id,
                source_file=source_file,
                source_lines=(1,),
                raw_latex="A B",
                markdown="A B",
                rgb=(10, 20, 30) if mode == "plain_word" else (40, 50, 60),
            )

        plain_unit = unit("plain", "plain_word")
        whole_unit = unit("whole", "whole")
        probes = [
            MODULE.stable.SourceProbe(
                probe_id="plain-1",
                unit_id=plain_unit.unit_id,
                paragraph_id=plain_unit.paragraph_id,
                kind=plain_unit.kind,
                source_file=source_file,
                source_lines=(1,),
                markdown_fragment="A",
                rgb=(1, 2, 3),
                ordinal=1,
                total=2,
                localization_mode="plain_word",
            ),
            MODULE.stable.SourceProbe(
                probe_id="plain-2",
                unit_id=plain_unit.unit_id,
                paragraph_id=plain_unit.paragraph_id,
                kind=plain_unit.kind,
                source_file=source_file,
                source_lines=(1,),
                markdown_fragment=" B",
                rgb=(4, 5, 6),
                ordinal=2,
                total=2,
                localization_mode="plain_word",
            ),
            MODULE.stable.SourceProbe(
                probe_id="whole-1",
                unit_id=whole_unit.unit_id,
                paragraph_id=whole_unit.paragraph_id,
                kind=whole_unit.kind,
                source_file=source_file,
                source_lines=(1,),
                markdown_fragment=whole_unit.markdown,
                rgb=(7, 8, 9),
                ordinal=1,
                total=1,
                localization_mode="whole",
            ),
        ]
        rows = {
            "plain-1": [{"page_number": 1, "bbox_points": [40, 100, 80, 112]}],
            "plain-2": [{"page_number": 31, "bbox_points": [40, 120, 80, 132]}],
            "whole-1": [{"page_number": 31, "bbox_points": [40, 140, 80, 152]}],
        }
        fragments, reasons, summary = MODULE.build_fragments_for_shadow(
            [plain_unit, whole_unit],
            probes,
            rows,
            {},
            {},
            {1: 600.0},
        )

        self.assertEqual(fragments[1][0].markdown, "A")
        self.assertEqual(fragments[31][0].unit_id, whole_unit.unit_id)
        self.assertFalse(reasons)
        self.assertEqual(summary["truncated_probes"], 1)
        self.assertEqual(summary["plain_word_mapped"], 1)

    def test_external_verbatim_rows_replace_only_external_synctex_rows(self) -> None:
        def probe(probe_id: str, mode: str) -> object:
            return MODULE.stable.SourceProbe(
                probe_id=probe_id,
                unit_id=(
                    "external"
                    if mode == MODULE.EXTERNAL_VERBATIM_LOCALIZATION_MODE
                    else "body"
                ),
                paragraph_id="p1",
                kind=(
                    "external_verbatim"
                    if mode == MODULE.EXTERNAL_VERBATIM_LOCALIZATION_MODE
                    else "paragraph"
                ),
                source_file=Path("main.tex"),
                source_lines=(1,),
                markdown_fragment=probe_id,
                rgb=(1, 2, 3),
                ordinal=1,
                total=1,
                localization_mode=mode,
            )

        probes = [
            probe("body", "whole"),
            probe("external-line", MODULE.EXTERNAL_VERBATIM_LOCALIZATION_MODE),
        ]
        base_rows = {
            "body": [{"page_number": 1, "bbox_points": [10, 10, 20, 20]}],
            "external-line": [
                {"page_number": 9, "bbox_points": [90, 90, 99, 99]}
            ],
        }
        color_rows = {
            "external-line": [
                {
                    "page_number": 2,
                    "bbox_points": [30, 40, 80, 50],
                    "components": [[30, 40, 80, 50]],
                }
            ]
        }
        merged, summary = MODULE.merge_external_verbatim_alignment(
            probes, base_rows, color_rows
        )
        self.assertEqual(merged["body"], base_rows["body"])
        self.assertEqual(merged["external-line"], color_rows["external-line"])
        self.assertEqual(summary["external_line_probes_mapped_exactly_once"], 1)
        self.assertEqual(summary["coverage"], 1.0)

        suppressed, suppression = (
            MODULE.suppress_untrusted_external_synctex_rows(probes, base_rows)
        )
        self.assertEqual(suppressed["body"], base_rows["body"])
        self.assertEqual(suppressed["external-line"], [])
        self.assertEqual(
            suppression["external_synctex_rows_observed_but_untrusted"], 1
        )
        self.assertTrue(suppression["requires_external_color_gate"])

        with self.assertRaisesRegex(RuntimeError, "incomplete or multi-page"):
            MODULE.merge_external_verbatim_alignment(probes, base_rows, {})

    def test_external_verbatim_exact_gate_rejects_tolerated_nonzero_shift(self) -> None:
        geometry = {
            "status": "passed",
            "page_count_equal": True,
            "character_text_equal": True,
            "geometry_equal": True,
            "max_geometry_shift_points": 0.01,
            "pages_compared": 1,
            "pages": [
                {
                    "character_count_equal": True,
                    "character_text_equal": True,
                    "geometry_equal": True,
                    "max_geometry_shift_points": 0.01,
                }
            ],
        }
        logical = {
            "page_count_equal": True,
            "all_pages_equal": True,
            "pages": [{"logical_content_and_order_equal": True}],
        }
        result = MODULE.exact_pdf_shadow_gate(geometry, logical)
        self.assertEqual(result["status"], "failed")
        self.assertIn("glyph_geometry_not_exact", result["reasons"])

        geometry["max_geometry_shift_points"] = 0.0
        geometry["pages"][0]["max_geometry_shift_points"] = 0.0
        self.assertEqual(
            MODULE.exact_pdf_shadow_gate(geometry, logical)["status"], "passed"
        )

    def test_invariant_unit_hybrid_atomically_uses_preferred_exact_donor(self) -> None:
        source = Path("/tmp/invariant-hybrid-source.tex")
        unit = MODULE.stable.SourceUnit(
            unit_id="u1",
            kind="paragraph",
            paragraph_id="p1",
            source_file=source,
            source_lines=(44, 45),
            raw_latex="alpha beta",
            markdown="alpha beta",
            rgb=(1, 2, 3),
        )

        def probe(ordinal: int, fragment: str):
            return MODULE.stable.SourceProbe(
                probe_id=f"u1-atom-{ordinal:05d}",
                unit_id=unit.unit_id,
                paragraph_id=unit.paragraph_id,
                kind=unit.kind,
                source_file=source,
                source_lines=unit.source_lines,
                markdown_fragment=fragment,
                rgb=(10 + ordinal, 20, 30),
                ordinal=ordinal,
                total=2,
                localization_mode="source_atom",
            )

        probes = [probe(1, "alpha"), probe(2, "beta")]
        locators = {
            probes[0].probe_id: MODULE.AtomLocator(
                probes[0].probe_id, source, 0, 5, 0
            ),
            probes[1].probe_id: MODULE.AtomLocator(
                probes[1].probe_id, source, 6, 10, 1
            ),
        }
        geometry = {
            "status": "passed",
            "page_count_equal": True,
            "character_text_equal": True,
            "geometry_equal": True,
            "max_geometry_shift_points": 0.0,
            "pages_compared": 1,
            "pages": [
                {
                    "character_count_equal": True,
                    "character_text_equal": True,
                    "geometry_equal": True,
                    "max_geometry_shift_points": 0.0,
                }
            ],
        }
        logical = {
            "page_count_equal": True,
            "all_pages_equal": True,
            "pages": [
                {"page_number": 1, "logical_content_and_order_equal": True}
            ],
        }

        def shadow(name: str, rows: dict[str, list[dict[str, object]]]):
            return MODULE.ShadowCandidate(
                shadow_id=name,
                probes=list(probes),
                atom_locators=dict(locators),
                unit_atoms={},
                modes={unit.unit_id: "source_atom"},
                colored_pdf=Path(f"{name}.pdf"),
                geometry=geometry,
                logical_invariance=logical,
                color_rows=rows,
                color_summary=MODULE.alignment_summary_for_rows(
                    probes, rows, locator=name
                ),
            )

        # The base has one valid row but is incomplete.  It must be replaced
        # as one unit, not patched one probe at a time.
        base_rows = {
            probes[0].probe_id: [
                {
                    "page_number": 1,
                    "bbox_points": [60.0, 100.0, 90.0, 110.0],
                    "characters": 5,
                }
            ],
            probes[1].probe_id: [],
        }
        clean_rows = {
            probes[0].probe_id: [
                {
                    "page_number": 1,
                    "bbox_points": [70.0, 120.0, 100.0, 130.0],
                    "characters": 5,
                    "locator": "synctex_clean_source_line",
                }
            ],
            probes[1].probe_id: [
                {
                    "page_number": 1,
                    "bbox_points": [105.0, 120.0, 135.0, 130.0],
                    "characters": 4,
                    "locator": "synctex_clean_source_line",
                }
            ],
        }
        atom_line_rows = {
            probe.probe_id: [
                {
                    "page_number": 1,
                    "bbox_points": [80.0 + 30 * index, 140.0, 100.0 + 30 * index, 150.0],
                }
            ]
            for index, probe in enumerate(probes)
        }
        base = shadow("source_atoms", base_rows)
        clean = shadow("synctex_clean", clean_rows)
        atom_lines = shadow("synctex_atom_lines", atom_line_rows)
        original_base = json.dumps(base.color_rows, sort_keys=True)

        hybrids, report = MODULE.derive_invariant_unit_hybrid_shadows(
            [base, atom_lines, clean], [unit], {1: 600.0}
        )

        self.assertEqual(len(hybrids), 1)
        hybrid = hybrids[0]
        self.assertTrue(
            hybrid.shadow_id.startswith("source_atoms__invariant_unit_hybrid_")
        )
        self.assertEqual(json.dumps(base.color_rows, sort_keys=True), original_base)
        self.assertIs(hybrid.probes[0], base.probes[0])
        self.assertEqual(
            hybrid.color_rows[probes[0].probe_id][0],
            {"page_number": 1, "bbox_points": [70.0, 120.0, 100.0, 130.0]},
        )
        self.assertEqual(
            hybrid.color_rows[probes[1].probe_id][0],
            {"page_number": 1, "bbox_points": [105.0, 120.0, 135.0, 130.0]},
        )
        replacement = report["hybrids"][0]["unit_replacements"][0]
        self.assertEqual(replacement["donor_shadow_id"], "synctex_clean")
        self.assertEqual(replacement["ground_truth_source"], "SourceUnit")
        self.assertEqual(replacement["donor_fields_used"], ["page_number", "bbox_points"])
        self.assertFalse(replacement["pdf_text_used"])
        again, _ = MODULE.derive_invariant_unit_hybrid_shadows(
            [base, atom_lines, clean], [unit], {1: 600.0}
        )
        self.assertEqual(again[0].shadow_id, hybrid.shadow_id)

    def test_invariant_unit_hybrid_rejects_conflict_multilane_and_external(self) -> None:
        source = Path("/tmp/invariant-hybrid-rejections.tex")
        geometry = {
            "status": "passed",
            "page_count_equal": True,
            "character_text_equal": True,
            "geometry_equal": True,
            "max_geometry_shift_points": 0.0,
            "pages_compared": 2,
            "pages": [
                {
                    "character_count_equal": True,
                    "character_text_equal": True,
                    "geometry_equal": True,
                    "max_geometry_shift_points": 0.0,
                },
                {
                    "character_count_equal": True,
                    "character_text_equal": True,
                    "geometry_equal": True,
                    "max_geometry_shift_points": 0.0,
                },
            ],
        }
        logical = {
            "page_count_equal": True,
            "all_pages_equal": True,
            "pages": [
                {"page_number": 1, "logical_content_and_order_equal": True},
                {"page_number": 2, "logical_content_and_order_equal": True},
            ],
        }

        def make_case(mode: str, *, base_page: int, donor_rows):
            unit = MODULE.stable.SourceUnit(
                unit_id="u1",
                kind="paragraph",
                paragraph_id="p1",
                source_file=source,
                source_lines=(1,),
                raw_latex="alpha beta",
                markdown="alpha beta",
                rgb=(1, 2, 3),
            )
            probes = [
                MODULE.stable.SourceProbe(
                    probe_id=f"q{index}",
                    unit_id="u1",
                    paragraph_id="p1",
                    kind="paragraph",
                    source_file=source,
                    source_lines=(1,),
                    markdown_fragment=value,
                    rgb=(index, 2, 3),
                    ordinal=index,
                    total=2,
                    localization_mode=mode,
                )
                for index, value in enumerate(("alpha", "beta"), start=1)
            ]

            def shadow(name, rows):
                return MODULE.ShadowCandidate(
                    shadow_id=name,
                    probes=list(probes),
                    atom_locators={},
                    unit_atoms={},
                    modes={unit.unit_id: mode},
                    colored_pdf=Path(f"{name}.pdf"),
                    geometry=geometry,
                    logical_invariance=logical,
                    color_rows=rows,
                    color_summary=MODULE.alignment_summary_for_rows(
                        probes, rows, locator=name
                    ),
                )

            base_rows = {
                "q1": [
                    {
                        "page_number": base_page,
                        "bbox_points": [60.0, 100.0, 100.0, 110.0],
                    }
                ],
                "q2": [],
            }
            return unit, shadow("source_atoms", base_rows), shadow(
                "synctex_clean", donor_rows
            )

        with self.subTest("page conflict"):
            unit, base, donor = make_case(
                "source_atom",
                base_page=2,
                donor_rows={
                    "q1": [{"page_number": 1, "bbox_points": [60, 100, 100, 110]}],
                    "q2": [{"page_number": 1, "bbox_points": [105, 100, 135, 110]}],
                },
            )
            hybrids, report = MODULE.derive_invariant_unit_hybrid_shadows(
                [base, donor], [unit], {1: 600.0, 2: 600.0}
            )
            self.assertFalse(hybrids)
            self.assertEqual(
                report["rejection_counts"]["base_donor_page_or_lane_conflict"], 1
            )

        with self.subTest("multiple lanes"):
            unit, base, donor = make_case(
                "source_atom",
                base_page=1,
                donor_rows={
                    "q1": [{"page_number": 1, "bbox_points": [60, 100, 100, 110]}],
                    "q2": [{"page_number": 1, "bbox_points": [400, 100, 450, 110]}],
                },
            )
            hybrids, report = MODULE.derive_invariant_unit_hybrid_shadows(
                [base, donor], [unit], {1: 600.0, 2: 600.0}
            )
            self.assertFalse(hybrids)
            self.assertEqual(
                report["rejection_counts"]["unit_rows_span_multiple_lanes"], 1
            )

        with self.subTest("multiple pages"):
            unit, base, donor = make_case(
                "source_atom",
                base_page=1,
                donor_rows={
                    "q1": [{"page_number": 1, "bbox_points": [60, 100, 100, 110]}],
                    "q2": [{"page_number": 2, "bbox_points": [60, 100, 100, 110]}],
                },
            )
            hybrids, report = MODULE.derive_invariant_unit_hybrid_shadows(
                [base, donor], [unit], {1: 600.0, 2: 600.0}
            )
            self.assertFalse(hybrids)
            self.assertEqual(
                report["rejection_counts"]["unit_rows_span_multiple_pages"], 1
            )

        with self.subTest("external verbatim"):
            unit, base, donor = make_case(
                MODULE.EXTERNAL_VERBATIM_LOCALIZATION_MODE,
                base_page=1,
                donor_rows={
                    "q1": [{"page_number": 1, "bbox_points": [60, 100, 100, 110]}],
                    "q2": [{"page_number": 1, "bbox_points": [105, 100, 135, 110]}],
                },
            )
            hybrids, report = MODULE.derive_invariant_unit_hybrid_shadows(
                [base, donor], [unit], {1: 600.0, 2: 600.0}
            )
            self.assertFalse(hybrids)
            self.assertEqual(
                report["rejection_counts"]["external_verbatim_unit_forbidden"], 1
            )

        with self.subTest("donor global geometry not exact"):
            unit, base, donor = make_case(
                "source_atom",
                base_page=1,
                donor_rows={
                    "q1": [{"page_number": 1, "bbox_points": [60, 100, 100, 110]}],
                    "q2": [{"page_number": 1, "bbox_points": [105, 100, 135, 110]}],
                },
            )
            shifted_geometry = json.loads(json.dumps(donor.geometry))
            shifted_geometry["max_geometry_shift_points"] = 0.01
            shifted_geometry["pages"][0]["max_geometry_shift_points"] = 0.01
            donor = MODULE.dataclasses.replace(
                donor, geometry=shifted_geometry
            )
            hybrids, report = MODULE.derive_invariant_unit_hybrid_shadows(
                [base, donor], [unit], {1: 600.0, 2: 600.0}
            )
            self.assertFalse(hybrids)
            self.assertEqual(
                report["rejection_counts"][
                    "donor_global_exact_invariance_failed"
                ],
                1,
            )

    def test_invariant_unit_hybrid_rejects_complete_probe_schema_hash_mismatch(self) -> None:
        source = Path("/tmp/invariant-hybrid-schema.tex")
        unit = MODULE.stable.SourceUnit(
            unit_id="u1",
            kind="paragraph",
            paragraph_id="p1",
            source_file=source,
            source_lines=(1,),
            raw_latex="alpha",
            markdown="alpha",
            rgb=(1, 2, 3),
        )

        def probe(fragment: str):
            return MODULE.stable.SourceProbe(
                probe_id="q1",
                unit_id="u1",
                paragraph_id="p1",
                kind="paragraph",
                source_file=source,
                source_lines=(1,),
                markdown_fragment=fragment,
                rgb=(4, 5, 6),
                ordinal=1,
                total=1,
                localization_mode="source_atom",
            )

        geometry = {
            "status": "passed",
            "page_count_equal": True,
            "character_text_equal": True,
            "geometry_equal": True,
            "max_geometry_shift_points": 0.0,
            "pages_compared": 1,
            "pages": [
                {
                    "character_count_equal": True,
                    "character_text_equal": True,
                    "geometry_equal": True,
                    "max_geometry_shift_points": 0.0,
                }
            ],
        }
        logical = {
            "page_count_equal": True,
            "all_pages_equal": True,
            "pages": [
                {"page_number": 1, "logical_content_and_order_equal": True}
            ],
        }

        def shadow(name: str, value, rows):
            return MODULE.ShadowCandidate(
                shadow_id=name,
                probes=[value],
                atom_locators={},
                unit_atoms={},
                modes={unit.unit_id: "source_atom"},
                colored_pdf=Path(f"{name}.pdf"),
                geometry=geometry,
                logical_invariance=logical,
                color_rows=rows,
                color_summary=MODULE.alignment_summary_for_rows(
                    [value], rows, locator=name
                ),
            )

        base = shadow("source_atoms", probe("alpha"), {"q1": []})
        donor = shadow(
            "synctex_clean",
            probe("ALPHA"),
            {"q1": [{"page_number": 1, "bbox_points": [60, 100, 100, 110]}]},
        )
        hybrids, report = MODULE.derive_invariant_unit_hybrid_shadows(
            [base, donor], [unit], {1: 600.0}
        )
        self.assertFalse(hybrids)
        self.assertEqual(
            report["rejection_counts"]["complete_probe_schema_hash_mismatch"], 1
        )

    def test_ieee_page4_invariant_unit_hybrid_recovers_method_paragraph(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "output"
            / "pdf"
            / "source_first_advanced_ieee_dropfig_v2"
        )
        required = [
            root / "source_units.jsonl",
            root / "validation_report_v2.json",
            root / "build_clean" / "samplepaper.pdf",
            root / "shadows" / "source_atoms" / "color_page_alignment.jsonl",
            root / "shadows" / "synctex_clean" / "synctex_page_alignment.jsonl",
            root
            / "shadows"
            / "synctex_atom_lines"
            / "synctex_page_alignment.jsonl",
        ]
        if not all(path.is_file() for path in required):
            self.skipTest("IEEE invariant-hybrid regression artifact is unavailable")

        clean_root = root / "source_clean"
        units = []
        for line in (root / "source_units.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            value = json.loads(line)
            units.append(
                MODULE.stable.SourceUnit(
                    unit_id=value["unit_id"],
                    kind=value["kind"],
                    paragraph_id=value["source_paragraph_id"],
                    source_file=(clean_root / value["source_file"]).resolve(),
                    source_lines=tuple(value["source_line_numbers"]),
                    raw_latex=value["raw_latex"],
                    markdown=value["markdown"],
                    rgb=tuple(value["rgb"]),
                    source_command=value.get("source_command"),
                )
            )
        base_probes, _ = MODULE.stable.build_source_probes(
            units,
            word_probe_kinds={
                "paragraph",
                "itemize_item",
                "enumerate_item",
                "description_item",
            },
        )
        probes, locators, atoms, modes = MODULE.build_atom_probes(
            units,
            base_probes,
            {},
            include_plain_source_atoms=True,
        )
        validation = json.loads(
            (root / "validation_report_v2.json").read_text(encoding="utf-8")
        )
        attempt_by_id = {
            value["shadow_id"]: value
            for value in validation["shadow_attempts"]
        }
        shadows = []
        for shadow_id, alignment_name in (
            ("source_atoms", "color_page_alignment.jsonl"),
            ("synctex_clean", "synctex_page_alignment.jsonl"),
            ("synctex_atom_lines", "synctex_page_alignment.jsonl"),
        ):
            rows = {
                value["probe_id"]: value["pages"]
                for value in (
                    json.loads(line)
                    for line in (
                        root / "shadows" / shadow_id / alignment_name
                    ).read_text(encoding="utf-8").splitlines()
                )
            }
            attempt = attempt_by_id[shadow_id]
            shadows.append(
                MODULE.ShadowCandidate(
                    shadow_id=shadow_id,
                    probes=list(probes),
                    atom_locators=locators,
                    unit_atoms=atoms,
                    modes=modes,
                    colored_pdf=root / "build_clean" / "samplepaper.pdf",
                    geometry=attempt["geometry"],
                    logical_invariance=attempt["logical_invariance"],
                    color_rows=rows,
                    color_summary=(
                        attempt.get("color_alignment")
                        or attempt["alignment"]
                    ),
                )
            )

        with MODULE.pdfplumber.open(
            root / "build_clean" / "samplepaper.pdf"
        ) as document:
            widths = {
                page_number: float(page.width)
                for page_number, page in enumerate(document.pages, start=1)
            }
            hybrids, audit = MODULE.derive_invariant_unit_hybrid_shadows(
                shadows, units, widths
            )
            self.assertEqual(len(hybrids), 1)
            method_unit = next(
                unit
                for unit in units
                if unit.source_file.name == "method.tex"
                and unit.source_lines == (44, 45)
            )
            replacements = audit["hybrids"][0]["unit_replacements"]
            self.assertEqual(
                [value["unit_id"] for value in replacements],
                [method_unit.unit_id],
            )
            hybrid = hybrids[0]
            fragments, reasons, _summary = MODULE.build_fragments_for_shadow(
                units,
                hybrid.probes,
                hybrid.color_rows,
                hybrid.unit_atoms,
                hybrid.atom_locators,
                widths,
            )
            self.assertFalse(reasons.get(4))
            variants, frontier = MODULE.enumerate_leading_frontier_variants(
                4,
                fragments[4],
                units=units,
                probes=hybrid.probes,
                rows=hybrid.color_rows,
            )
            frozen = MODULE.freeze_page_source_candidates(
                fragments[4],
                page_width=widths[4],
                page_height=float(document.pages[3].height),
                frontier_variants=variants,
                frontier_report=frontier,
            )
            pdf_text, _ = MODULE.stable.pdf_verifier_text(document.pages[3])
            result = MODULE.verify_frozen_page_candidates(
                frozen,
                pdf_text,
                pdf_page=document.pages[3],
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["layout"]["layout_bucket"], "two_column")
        self.assertIn(method_unit.markdown, result["markdown"])
        self.assertTrue(result["verifier"]["exact_ordered_character_stream_match"])

    def test_external_verbatim_fence_info_is_not_treated_as_printed_text(self) -> None:
        markdown = "```text\n# literal **code**\n```"
        result = MODULE.experimental_verifier_result(
            markdown, "# literal **code**"
        )
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["exact_ordered_character_stream_match"])
        self.assertEqual(
            result["experimental_projection"][
                "fenced_code_info_strings_hidden"
            ],
            1,
        )
        self.assertEqual(markdown, "```text\n# literal **code**\n```")

        nested_literal = "````text\n```python\nx = 1\n```\n````"
        projected, hidden = MODULE.project_fenced_code_info_strings(
            nested_literal
        )
        self.assertEqual(hidden, 1)
        self.assertEqual(projected, "````\n```python\nx = 1\n```\n````")
        self.assertEqual(
            MODULE.experimental_verifier_result(
                nested_literal, "```python\nx = 1\n```"
            )["status"],
            "passed",
        )

    def test_external_verbatim_dollars_remain_literal_in_exact_verifier(self) -> None:
        markdown = "```text\ninterpretation-domains($i,$i)\n```"
        result = MODULE.experimental_verifier_result(
            markdown, "interpretation-domains($i,$i)"
        )
        self.assertEqual(result["status"], "passed")
        projection = result["experimental_projection"]["source_visible_flow"]
        self.assertEqual(projection["status"], "projected")
        self.assertEqual(projection["fenced_code_dollars_guarded"], 2)
        self.assertFalse(projection["ground_truth_changed"])
        self.assertEqual(markdown, "```text\ninterpretation-domains($i,$i)\n```")

    @unittest.skipUnless(shutil.which("latexmk"), "latexmk is required")
    def test_real_tiny_external_verbatim_color_shadow_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            clean = root / "clean"
            clean.mkdir()
            main = clean / "main.tex"
            main.write_text(
                "\\documentclass{article}\n"
                "\\usepackage{verbatim}\n"
                "\\begin{document}\n"
                "Before.\n\n"
                "\\verbatiminput{snippet.txt}\n"
                "After.\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (clean / "snippet.txt").write_text(
                "%----------------\nalpha\n   \n\t\n", encoding="utf-8"
            )
            ir = MODULE.build_external_verbatim_ir(clean, [main])
            units, blocks = MODULE.external_verbatim_units(
                ir, color_index_offset=0
            )
            base, modes = MODULE.stable.build_source_probes(
                units, word_probe_kinds=set()
            )
            probes, _modes = MODULE.replace_external_verbatim_line_probes(
                base, modes, blocks
            )
            self.assertEqual([probe.ordinal for probe in probes], [1, 2])
            shadow = root / "shadow"
            instrumentation = MODULE.instrument_external_verbatim_color_tree(
                clean, shadow, blocks, probes, "pdflatex"
            )
            self.assertEqual(instrumentation["lines_instrumented"], 2)
            latexmk = Path(shutil.which("latexmk") or "latexmk")
            previous = MODULE.color_pilot.LATEXMK
            MODULE.color_pilot.LATEXMK = latexmk
            try:
                clean_pdf = MODULE.color_pilot.run_compile(
                    source_root=clean,
                    main_tex=Path("main.tex"),
                    build_dir=root / "build-clean",
                    log_path=root / "clean.log",
                    label="test-external-verbatim-clean",
                    timeout_seconds=60,
                    engine="pdflatex",
                )
                shadow_pdf = MODULE.color_pilot.run_compile(
                    source_root=shadow,
                    main_tex=Path("main.tex"),
                    build_dir=root / "build-shadow",
                    log_path=root / "shadow.log",
                    label="test-external-verbatim-shadow",
                    timeout_seconds=60,
                    engine="pdflatex",
                )
            finally:
                MODULE.color_pilot.LATEXMK = previous
            geometry = MODULE.color_pilot.compare_pdf_geometry(
                clean_pdf, shadow_pdf
            )
            logical = MODULE.compare_pdf_logical_invariance(
                clean_pdf, shadow_pdf
            )
            self.assertEqual(
                MODULE.exact_pdf_shadow_gate(geometry, logical)["status"],
                "passed",
            )
            rows, summary = MODULE.extract_color_runs(shadow_pdf, probes)
            self.assertEqual(summary["probes_mapped"], 2)
            self.assertTrue(
                all(len(rows[probe.probe_id]) == 1 for probe in probes)
            )
            expected_by_probe = {
                record.record_id: "".join(record.visible_text.split())
                for record in ir.records
                if record.visible_text.strip()
            }
            observed_by_probe = {probe.probe_id: "" for probe in probes}
            probe_by_rgb = {probe.rgb: probe for probe in probes}
            with MODULE.pdfplumber.open(shadow_pdf) as document:
                for page in document.pages:
                    for character in page.chars:
                        rgb = MODULE.color_pilot.normalize_pdf_rgb(
                            character.get("non_stroking_color")
                        )
                        probe = probe_by_rgb.get(rgb)
                        if probe is not None:
                            observed_by_probe[probe.probe_id] += str(
                                character.get("text") or ""
                            )
            self.assertEqual(observed_by_probe, expected_by_probe)
            self.assertEqual(
                [rows[probe.probe_id][0]["page_number"] for probe in probes],
                [1, 1],
            )

    def test_bottom_folio_projection_changes_only_pdf_verifier_observation(self) -> None:
        fragment = MODULE.LocatedFragment(
            fragment_id="body",
            unit_id="u1",
            paragraph_id="p1",
            kind="paragraph",
            markdown="Body text",
            probe_ids=("probe",),
            source_file=Path("main.tex"),
            source_start_line=1,
            source_ordinal=1,
            page_number=1,
            bbox=(80, 100, 180, 112),
            components=((80, 100, 180, 112),),
        )
        frozen = MODULE.freeze_page_source_candidates(
            [fragment], page_width=600.0, page_height=800.0
        )

        class Page:
            width = 600.0
            height = 800.0

            @staticmethod
            def extract_words(**_kwargs):
                return [
                    {"text": "Body", "x0": 80, "top": 100, "x1": 115, "bottom": 110},
                    {"text": "text", "x0": 120, "top": 100, "x1": 150, "bottom": 110},
                    {"text": "7", "x0": 297, "top": 700, "x1": 303, "bottom": 710},
                ]

        result = MODULE.verify_frozen_page_candidates(
            frozen, "Body text\n\n7", pdf_page=Page()
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["markdown"].strip(), "Body text")
        projection = result["verifier"]["pdf_layout_projection"]
        self.assertTrue(projection["projection_applied"])
        self.assertFalse(projection["ground_truth_changed"])

    def test_synctex_identity_comments_give_atoms_unique_generated_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean"
            shadow = root / "shadow"
            clean.mkdir()
            source = clean / "part.tex"
            source.write_text("alpha beta\nnext\n", encoding="utf-8")

            def probe(probe_id: str, lines: tuple[int, ...], mode: str):
                return MODULE.stable.SourceProbe(
                    probe_id=probe_id,
                    unit_id="u1",
                    paragraph_id="p1",
                    kind="paragraph",
                    source_file=source,
                    source_lines=lines,
                    markdown_fragment=probe_id,
                    rgb=(1, 2, 3),
                    ordinal=1,
                    total=1,
                    localization_mode=mode,
                )

            probes = [
                probe("atom_alpha", (1,), "source_atom"),
                probe("atom_beta", (1,), "source_atom"),
                probe("whole_next", (2,), "whole"),
            ]
            locators = {
                "atom_alpha": MODULE.AtomLocator(
                    "atom_alpha", source, 0, 5, 0
                ),
                "atom_beta": MODULE.AtomLocator(
                    "atom_beta", source, 6, 10, 1
                ),
            }
            overrides, report = MODULE.instrument_synctex_line_identity_tree(
                clean, shadow, probes, locators
            )

            instrumented = (shadow / "part.tex").read_text(encoding="utf-8")
            self.assertEqual(
                instrumented,
                "%SFV2SYNC:atom_alpha\nalpha %SFV2SYNC:atom_beta\nbeta\nnext\n",
            )
            self.assertEqual(overrides["atom_alpha"][1], (2,))
            self.assertEqual(overrides["atom_beta"][1], (3,))
            self.assertEqual(overrides["whole_next"][1], (4,))
            self.assertEqual(report["markers_inserted"], 2)

    def test_logical_invariance_ignores_only_layout_line_hyphens(self) -> None:
        marker = MODULE.stable.OPTIONAL_LINE_END_HYPHEN
        self.assertEqual(
            MODULE.logical_text_stream(f"con{marker}ditions"),
            MODULE.logical_text_stream("conditions"),
        )
        self.assertNotEqual(
            MODULE.logical_text_stream("co-operate"),
            MODULE.logical_text_stream("cooperate"),
        )

    def test_compiler_heading_meaning_keeps_visible_template_punctuation(self) -> None:
        self.assertEqual(
            MODULE.heading_meaning_visible_label(
                r"macro:->B.\hskip 0.5em\relax"
            ),
            "B.",
        )
        self.assertEqual(
            MODULE.heading_meaning_visible_label(
                r"macro:->1)\hskip 0.5em\relax"
            ),
            "1)",
        )

    def test_units_are_merged_in_tex_execution_order_across_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = root / "main.tex"
            child = root / "child.tex"
            main.write_text("main first\n\\input{child}\nmain last\n", encoding="utf-8")
            child.write_text("child heading\nchild paragraph\n", encoding="utf-8")

            def unit(name: str, kind: str, path: Path, line: int):
                return MODULE.stable.SourceUnit(
                    unit_id=name,
                    kind=kind,
                    paragraph_id=name,
                    source_file=path,
                    source_lines=(line,),
                    raw_latex=name,
                    markdown=name,
                    rgb=(10, 20, 30),
                )

            values = [
                unit("main-last", "paragraph", main, 3),
                unit("child-paragraph", "paragraph", child, 2),
                unit("main-first", "paragraph", main, 1),
                unit("child-heading", "heading", child, 1),
            ]
            execution = MODULE.build_execution_ir(
                main, fls_sources=(main, child)
            )
            ordered, rejected, report = MODULE.order_units_by_execution(
                values, execution
            )
            self.assertFalse(rejected)
            self.assertEqual(
                [value.unit_id for value in ordered],
                ["main-first", "child-heading", "child-paragraph", "main-last"],
            )
            self.assertEqual(report["units_ordered"], 4)

    def test_markup_aware_atom_probes_keep_source_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "main.tex"
            raw = r"A \textbf{bold words}."
            source_file.write_text(raw + "\n", encoding="utf-8")
            unit = MODULE.stable.SourceUnit(
                unit_id="src-1",
                kind="paragraph",
                paragraph_id="p-1",
                source_file=source_file,
                source_lines=(1,),
                raw_latex=raw,
                markdown="A **bold words**.",
                rgb=(10, 20, 30),
            )
            base, _ = MODULE.stable.build_source_probes([unit], word_probe_kinds=set())
            probes, locators, atoms, modes = MODULE.build_atom_probes([unit], base)

            self.assertEqual(modes[unit.unit_id], "source_atom")
            self.assertEqual([probe.markdown_fragment for probe in probes], ["A", "bold", "words", "."])
            self.assertEqual(MODULE.atoms_to_markdown(atoms[unit.unit_id]), unit.markdown)
            for probe in probes:
                locator = locators[probe.probe_id]
                self.assertEqual(
                    source_file.read_text(encoding="utf-8")[locator.source_start : locator.source_end],
                    next(atom.raw_source for atom in atoms[unit.unit_id] if atom.ordinal == locator.atom_ordinal),
                )

    def test_atom_fragment_cross_page_reopens_bold_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "main.tex"
            raw = r"\textbf{bold words}"
            source_file.write_text(raw, encoding="utf-8")
            unit = MODULE.stable.SourceUnit(
                unit_id="src-1",
                kind="paragraph",
                paragraph_id="p-1",
                source_file=source_file,
                source_lines=(1,),
                raw_latex=raw,
                markdown="**bold words**",
                rgb=(10, 20, 30),
            )
            base, _ = MODULE.stable.build_source_probes([unit], word_probe_kinds=set())
            probes, locators, atoms, _ = MODULE.build_atom_probes([unit], base)
            rows = {
                probes[0].probe_id: [
                    {
                        "page_number": 1,
                        "bbox_points": [50, 100, 90, 112],
                        "components": [[50, 100, 90, 112]],
                        "characters": 4,
                    }
                ],
                probes[1].probe_id: [
                    {
                        "page_number": 2,
                        "bbox_points": [50, 80, 100, 92],
                        "components": [[50, 80, 100, 92]],
                        "characters": 5,
                    }
                ],
            }
            fragments, reasons, _ = MODULE.build_fragments_for_shadow(
                [unit], probes, rows, atoms, locators, {1: 600.0, 2: 600.0}
            )
            self.assertFalse(reasons)
            self.assertEqual(fragments[1][0].markdown, "**bold**")
            self.assertEqual(fragments[2][0].markdown, "**words**")

    def test_lane_split_uses_complete_lines_not_individual_word_x(self) -> None:
        source_file = Path("main.tex")

        def probe(index: int):
            return MODULE.stable.SourceProbe(
                probe_id=f"q{index}",
                unit_id="u",
                paragraph_id="p",
                kind="paragraph",
                source_file=source_file,
                source_lines=(1,),
                markdown_fragment=str(index),
                rgb=(index, 0, 0),
                ordinal=index,
                total=4,
                localization_mode="plain_word",
            )

        probes = [probe(index) for index in range(1, 5)]
        single_rows = {
            "q1": [{"page_number": 1, "bbox_points": [40, 100, 90, 112]}],
            "q2": [{"page_number": 1, "bbox_points": [250, 100, 300, 112]}],
            "q3": [{"page_number": 1, "bbox_points": [310, 100, 360, 112]}],
            "q4": [{"page_number": 1, "bbox_points": [510, 100, 560, 112]}],
        }
        self.assertEqual(
            len(MODULE.split_probe_run_by_lane(probes, single_rows, 600)), 1
        )
        two_column_rows = {
            "q1": [{"page_number": 1, "bbox_points": [40, 100, 90, 112]}],
            "q2": [{"page_number": 1, "bbox_points": [180, 100, 260, 112]}],
            "q3": [{"page_number": 1, "bbox_points": [340, 80, 400, 92]}],
            "q4": [{"page_number": 1, "bbox_points": [480, 80, 560, 92]}],
        }
        groups = MODULE.split_probe_run_by_lane(probes, two_column_rows, 600)
        self.assertEqual([[p.probe_id for p in group] for group in groups], [["q1", "q2"], ["q3", "q4"]])

    def test_banded_layout_candidate_supports_full_title_and_two_columns(self) -> None:
        source = Path("main.tex")

        def fragment(name: str, text: str, ordinal: int, bbox: tuple[float, float, float, float]):
            return MODULE.LocatedFragment(
                fragment_id=name,
                unit_id=name,
                paragraph_id=name,
                kind="paragraph",
                markdown=text,
                probe_ids=(name,),
                source_file=source,
                source_start_line=ordinal + 1,
                source_ordinal=ordinal,
                page_number=1,
                bbox=bbox,
                components=(bbox,),
            )

        fragments = [
            fragment("title", "Title", 0, (30, 20, 570, 50)),
            fragment("l1", "Left one", 1, (30, 100, 260, 120)),
            fragment("l2", "Left two", 2, (30, 140, 260, 160)),
            fragment("r1", "Right one", 3, (340, 100, 570, 120)),
            fragment("r2", "Right two", 4, (340, 140, 570, 160)),
        ]
        result = MODULE.choose_exact_page_candidate(
            fragments,
            page_width=600,
            page_height=800,
            pdf_text="Title Left one Left two Right one Right two",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["selected_order"], "banded_layout_graph")
        self.assertEqual(result["layout"]["layout_bucket"], "mixed_full_two_column")

    def test_component_lanes_recover_true_columns_from_full_union_boxes(self) -> None:
        source = Path("main.tex")

        def fragment(
            name: str,
            text: str,
            ordinal: int,
            components: tuple[tuple[float, float, float, float], ...],
        ):
            return MODULE.LocatedFragment(
                fragment_id=name,
                unit_id=name,
                paragraph_id=name,
                kind="paragraph",
                markdown=text,
                probe_ids=(name,),
                source_file=source,
                source_start_line=ordinal + 1,
                source_ordinal=ordinal,
                page_number=1,
                bbox=MODULE.union_bbox(components),
                components=components,
            )

        # ``bridge`` is one source paragraph flowing from the bottom of the
        # left column to the top of the right.  Its union bbox spans the page,
        # but no rendered line crosses the gutter.
        fragments = [
            fragment(
                "right",
                "Right after",
                1,
                ((312, 100, 320, 112), (340, 100, 570, 112)),
            ),
            fragment(
                "left",
                "Left before",
                2,
                ((30, 100, 250, 112), (270, 100, 300, 112)),
            ),
            fragment(
                "bridge",
                "Left end then right start",
                3,
                ((30, 700, 260, 712), (340, 60, 570, 72)),
            ),
        ]
        ordered, report = MODULE.component_lane_reading_order(fragments, 600)
        self.assertEqual(
            [fragment.fragment_id for fragment in ordered],
            ["left", "bridge", "right"],
        )
        self.assertEqual(report["layout_bucket"], "two_column")
        result = MODULE.choose_exact_page_candidate(
            fragments,
            page_width=600,
            page_height=800,
            pdf_text="Left before Left end then right start Right after",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["selected_order"], "component_lane_reading_order")

    def test_source_typography_lattice_handles_small_caps_and_ieee_lists(self) -> None:
        source = Path("main.tex")

        def fragment(name: str, kind: str, markdown: str, y: float):
            return MODULE.LocatedFragment(
                fragment_id=name,
                unit_id=name,
                paragraph_id=name,
                kind=kind,
                markdown=markdown,
                probe_ids=(name,),
                source_file=source,
                source_start_line=int(y),
                source_ordinal=int(y),
                page_number=1,
                bbox=(30, y, 560, y + 12),
                components=((30, y, 560, y + 12),),
            )

        result = MODULE.choose_exact_page_candidate(
            [
                fragment("heading", "heading", "## I. Results", 20),
                fragment("item", "enumerate_item", "1. Alpha", 50),
            ],
            page_width=600,
            page_height=800,
            pdf_text="I. RESULTS 1) Alpha",
        )
        self.assertEqual(result["status"], "passed")
        self.assertIn("h2:upper", result["selected_serialization_policy"])
        self.assertIn("enum=paren", result["selected_serialization_policy"])

    def test_heading_lattice_restores_only_missing_numeric_terminal_dots(self) -> None:
        self.assertEqual(
            MODULE.heading_markdown_variant(
                "## 1 Results",
                uppercase=False,
                trailing_colon=False,
                terminal_label_dot=True,
            ),
            "## 1. Results",
        )
        self.assertEqual(
            MODULE.heading_markdown_variant(
                "### 3.1 Details",
                uppercase=False,
                trailing_colon=False,
                terminal_label_dot=True,
            ),
            "### 3.1. Details",
        )
        for markdown in ("## I. Results", "### A. Methods", "## 1) Scope"):
            with self.subTest(markdown=markdown):
                self.assertEqual(
                    MODULE.heading_markdown_variant(
                        markdown,
                        uppercase=False,
                        trailing_colon=False,
                        terminal_label_dot=True,
                    ),
                    markdown,
                )

        def fragment(name: str, markdown: str, level: int):
            return MODULE.LocatedFragment(
                fragment_id=name,
                unit_id=name,
                paragraph_id=name,
                kind="heading",
                markdown=markdown,
                probe_ids=(name,),
                source_file=Path("main.tex"),
                source_start_line=level,
                source_ordinal=level,
                page_number=1,
                bbox=(30, 30.0 * level, 560, 30.0 * level + 12),
                components=((30, 30.0 * level, 560, 30.0 * level + 12),),
            )

        candidates = MODULE.source_serialization_candidates(
            [fragment("section", "## 1 Results", 2), fragment("subsection", "### 3.1 Details", 3)]
        )
        markdown_values = {markdown for _, markdown in candidates}
        self.assertIn("## 1 Results\n\n### 3.1 Details\n", markdown_values)
        self.assertIn("## 1. Results\n\n### 3.1. Details\n", markdown_values)
        self.assertTrue(
            any(
                "h2:source_label_dot" in policy
                and "h3:source_label_dot" in policy
                for policy, markdown in candidates
                if markdown == "## 1. Results\n\n### 3.1. Details\n"
            )
        )

        ieee_candidates = MODULE.source_serialization_candidates(
            [fragment("roman", "## I. Results", 2), fragment("alpha", "### A. Methods", 3)]
        )
        self.assertFalse(any("label_dot" in policy for policy, _ in ieee_candidates))
        self.assertTrue(
            all("I.." not in markdown and "A.." not in markdown for _, markdown in ieee_candidates)
        )

    def test_aux_table_number_and_caption_style_are_source_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aux = Path(directory) / "main.aux"
            aux.write_text(
                r"\@writefile{lot}{\contentsline {table}{\numberline {I}{\ignorespaces Visible caption}}{3}{}}"
                + "\n",
                encoding="utf-8",
            )
            numbers, report = MODULE.parse_aux_table_numbers([aux])
            self.assertEqual(numbers, {"visible caption": ["I"]})
            self.assertEqual(report["entries_parsed"], 1)
        value = (
            "Table I\n\nVisible caption\n\n"
            "<table><thead><tr><th><strong>Region</strong></th></tr></thead></table>"
        )
        rendered = MODULE.table_markdown_variant(
            value, True, True, True, True
        )
        self.assertTrue(rendered.startswith("TABLE I\n\nVISIBLE CAPTION."))
        self.assertNotIn("strong", rendered)

    def test_structural_table_markdown_keeps_caption_outside_clean_html(self) -> None:
        block = MODULE.page_gt.SourceBlock(
            block_id="b1",
            kind="table",
            source_file=Path("main.tex"),
            start_line=3,
            end_line=8,
            raw_latex=r"\begin{table}\caption{Visible}\begin{tabular}{c}A\\\end{tabular}\end{table}",
            markdown="Visible\n\n<table>\n<tr><td>A</td></tr>\n</table>",
            query_lines=[8],
            caption_markdown="Visible",
            table_html="<table>\n<tr><td>A</td></tr>\n</table>",
            table_parse_status="parsed",
        )
        units, rejected = MODULE.structural_units([block], color_index_offset=20)
        self.assertFalse(rejected)
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0].markdown.startswith("Visible\n\n<table>"))
        self.assertNotIn("data-table-id", units[0].markdown)
        self.assertNotIn("data-source", units[0].markdown)

        labeled = MODULE.page_gt.SourceBlock(
            block_id="b2",
            kind="table",
            source_file=Path("main.tex"),
            start_line=10,
            end_line=14,
            raw_latex=(
                r"\begin{table}\caption{Macro caption}\label{tab:one}"
                r"\begin{tabular}{c}A\\\end{tabular}\end{table}"
            ),
            markdown="Macro caption\n\n<table><tr><td>A</td></tr></table>",
            query_lines=[14],
            caption_markdown="Macro caption",
            table_html="<table><tr><td>A</td></tr></table>",
            table_parse_status="parsed",
        )
        refs = {
            "tab:one": MODULE.stable.AuxReference(
                label="tab:one", number="3", page="9", kind="table"
            )
        }
        labeled_units, labeled_rejected = MODULE.structural_units(
            [labeled], color_index_offset=30, references=refs
        )
        self.assertFalse(labeled_rejected)
        self.assertTrue(labeled_units[0].markdown.startswith("Table 3\n\n"))

    def test_theorem_heading_ir_is_synctex_only_and_freezes_aux_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            clean = root / "clean"
            clean.mkdir()
            source_file = clean / "main.tex"
            source = (
                "\\newtheorem{theorem}{Theorem}\n"
                "\\begin{theorem}[Sharp bound]\n"
                "\\label{thm:sharp}\n"
                "Body text.\n"
                "\\end{theorem}\n"
            )
            source_file.write_text(source, encoding="utf-8")
            theorem_ir = MODULE.build_theorem_ir_from_sources(
                {source_file: source}, {"thm:sharp": ("2.4",)}
            )
            units, variants, rejected, report = MODULE.build_theorem_heading_units(
                theorem_ir,
                color_index_offset=7,
            )
            self.assertFalse(rejected)
            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].kind, "theorem_heading")
            self.assertEqual(units[0].source_lines, (2,))
            self.assertEqual(units[0].raw_latex, r"\begin{theorem}[Sharp bound]")
            self.assertEqual(len(variants[units[0].unit_id]), 5)
            self.assertTrue(
                all("2.4" in markdown for _, markdown in variants[units[0].unit_id])
            )
            self.assertEqual(report["localization"], "synctex_metadata_only")
            self.assertFalse(report["pdf_text_used"])

            shadow = root / "shadow"
            shutil.copytree(clean, shadow)
            probes, _modes = MODULE.stable.build_source_probes(units)
            safety = MODULE.instrument_shadow_tree(
                clean,
                shadow,
                units,
                probes,
                {},
                "pdflatex",
            )
            self.assertEqual(safety["metadata_only_units"], 1)
            self.assertIn(units[0].unit_id, safety["metadata_only_unit_ids"])
            self.assertEqual(
                (shadow / "main.tex").read_text(encoding="utf-8"), source
            )

    def test_display_equation_tail_ir_is_source_aux_lattice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory).resolve() / "main.tex"
            raw = (
                "\\begin{equation}\n"
                "x = y\n"
                "\\label{eq:xy}\n"
                "\\end{equation}"
            )
            source_file.write_text(raw + "\n", encoding="utf-8")
            block = MODULE.page_gt.SourceBlock(
                block_id="eq-1",
                kind="display_math",
                source_file=source_file,
                start_line=1,
                end_line=4,
                raw_latex=raw,
                markdown="$$\nx = y\n$$",
                query_lines=[4],
            )
            units, rejected = MODULE.structural_units(
                [block], color_index_offset=3
            )
            self.assertFalse(rejected)
            updated, variants, report = MODULE.apply_display_equation_tail_ir(
                units,
                [block],
                {"eq:xy": ("3.2",)},
            )
            self.assertEqual(updated[0].markdown, "$$\nx = y\n$$\n(3.2)")
            self.assertEqual(
                [markdown for _, markdown in variants[updated[0].unit_id]],
                ["$$\nx = y\n$$\n(3.2)", "$$\nx = y\n$$\n3.2"],
            )
            self.assertEqual(report["units_with_tail_candidates"], 1)
            self.assertTrue(report["formula_markdown_preserved_before_tail"])
            self.assertFalse(report["pdf_text_used"])

            fragment = MODULE.LocatedFragment(
                fragment_id="equation",
                unit_id=updated[0].unit_id,
                paragraph_id=updated[0].paragraph_id,
                kind="display_math",
                markdown=updated[0].markdown,
                probe_ids=("equation-whole",),
                source_file=source_file,
                source_start_line=1,
                source_ordinal=1,
                page_number=1,
                bbox=(100, 100, 400, 150),
                components=((100, 100, 400, 150),),
                structural_markdown_candidates=variants[updated[0].unit_id],
            )
            candidates = MODULE.source_serialization_candidates([fragment])
            self.assertEqual(len(candidates), 2)
            self.assertEqual(
                {markdown for _, markdown in candidates},
                {"$$\nx = y\n$$\n(3.2)\n", "$$\nx = y\n$$\n3.2\n"},
            )
            self.assertTrue(all(";struct=" in policy for policy, _ in candidates))

    def test_aux_label_number_candidates_preserve_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aux = Path(directory) / "main.aux"
            aux.write_text(
                "\\newlabel{eq:one}{{3.2}{4}}\n"
                "\\newlabel{eq:one}{{3.2}{4}}\n"
                "\\newlabel{eq:bad}{{5}{7}}\n"
                "\\newlabel{eq:bad}{{6}{8}}\n",
                encoding="utf-8",
            )
            values, report = MODULE.parse_aux_label_number_candidates([aux])
            self.assertEqual(values["eq:one"], ("3.2",))
            self.assertEqual(values["eq:bad"], ("5", "6"))
            self.assertEqual(report["ambiguous_labels"], 1)
            self.assertEqual(report["duplicate_identical_records"], 1)
            self.assertFalse(report["pdf_text_used"])

    def test_longtable_locator_is_inserted_after_column_preamble(self) -> None:
        source = (
            "\\begin{longtable}{p{0.2\\linewidth}p{0.7\\linewidth}}\n"
            "\\caption{Visible caption}\\\\\n"
            "A & B \\\\\n"
            "\\end{longtable}\n"
        )
        unit = MODULE.stable.SourceUnit(
            unit_id="table-1",
            kind="table",
            paragraph_id="block-1",
            source_file=Path("main.tex"),
            source_lines=(1, 2, 3, 4),
            raw_latex=source.strip(),
            markdown="<table><tr><td>A</td><td>B</td></tr></table>",
            rgb=(10, 20, 30),
        )
        rendered = MODULE.instrument_structural_ranges(source, [unit], "pdflatex")
        self.assertIn("\\caption{\\pdfliteral", rendered)
        self.assertIn(
            "Visible caption" + MODULE.scoped_color_suffix("pdflatex") + "}",
            rendered,
        )
        self.assertNotIn("\\begin{longtable}\\pdfliteral", rendered)

    def test_safe_takeaway_wrapper_recovers_source_gt_and_argument_locators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "main.tex"
            definition = r"\newcommand{\takeaway}[1]{\textbf{Takeaway:} #1}"
            call = r"\takeaway{Keep \emph{this} result.}"
            source = definition + "\n" + call + "\n"
            source_file.write_text(source, encoding="utf-8")
            macros, definition_report = MODULE.collect_safe_visible_wrapper_macros(
                [source_file]
            )
            self.assertEqual(definition_report["safe_macros"], ["takeaway"])
            paragraph = MODULE.page_gt.SourceParagraph(
                paragraph_id="takeaway-p",
                kind="paragraph",
                source_file=source_file,
                source_lines=[2],
                raw_latex=call,
            )
            units, rejected, wrappers, unit_report = (
                MODULE.build_source_units_with_visible_wrappers(
                    [paragraph], references={}, macros=macros
                )
            )
            self.assertFalse(rejected)
            self.assertEqual(unit_report["recovered_units"], 1)
            self.assertEqual(units[0].markdown, "**Takeaway:** Keep *this* result.")

            base, _ = MODULE.stable.build_source_probes(
                units, word_probe_kinds=set()
            )
            probes, locators, atoms, modes = MODULE.build_atom_probes(
                units, base, wrappers
            )
            self.assertEqual(modes[units[0].unit_id], "source_wrapper_atom")
            self.assertEqual(MODULE.atoms_to_markdown(atoms[units[0].unit_id]), units[0].markdown)
            self.assertEqual(
                [source[item.source_start:item.source_end] for item in locators.values()],
                ["Keep", "this", "result."],
            )
            rendered = MODULE.instrument_atom_ranges(
                source, probes, locators, "pdflatex"
            )
            argument_open = rendered.index(r"\takeaway{") + len(r"\takeaway{")
            first_color = rendered.index(r"\pdfliteral direct", argument_open)
            self.assertGreaterEqual(first_color, argument_open)
            self.assertNotIn(r"\pdfliteral direct", rendered[:argument_open])

    def test_real_takeaway_tcolorbox_is_layout_only_source_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "main.tex"
            source_file.write_text(
                "\\newcommand{\\takeaway}[1]{%\n"
                "  \\begin{tcolorbox}[colback=gray!5, colframe=gray!75, "
                "boxrule=1.5pt, left=2pt, right=2pt, top=1pt, bottom=1pt]\n"
                "    \\textit{#1}\n"
                "  \\end{tcolorbox}}\n",
                encoding="utf-8",
            )
            macros, report = MODULE.collect_safe_visible_wrapper_macros([source_file])
            self.assertEqual(report["safe_macros"], ["takeaway"])
            self.assertEqual(macros["takeaway"].body.strip(), r"\textit{#1}")
            invocation, reason = MODULE.parse_visible_wrapper_invocation(
                r"\takeaway{\textbf{Takeaway RQ1:} Source conclusion.}",
                macros,
            )
            self.assertIsNone(reason)
            assert invocation is not None
            self.assertEqual(
                MODULE.atoms_to_markdown(
                    MODULE.build_source_atoms(invocation.expanded_source)
                ).strip(),
                "***Takeaway RQ1:** Source conclusion.*",
            )
            unsafe, unsafe_reason = MODULE.unwrap_layout_only_visible_wrapper(
                r"\begin{tcolorbox}[title=Visible]#1\end{tcolorbox}"
            )
            self.assertIsNone(unsafe)
            self.assertEqual(
                unsafe_reason, "tcolorbox_key_may_contribute_visible_content"
            )

    def test_wrapper_prefix_is_assigned_to_first_page_and_style_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "main.tex"
            definition = r"\newcommand{\takeaway}[1]{\textbf{Takeaway: #1}}"
            call = r"\takeaway{first second}"
            source_file.write_text(definition + "\n" + call + "\n", encoding="utf-8")
            macros, _ = MODULE.collect_safe_visible_wrapper_macros([source_file])
            paragraph = MODULE.page_gt.SourceParagraph(
                paragraph_id="takeaway-p",
                kind="paragraph",
                source_file=source_file,
                source_lines=[2],
                raw_latex=call,
            )
            units, rejected, wrappers, _ = MODULE.build_source_units_with_visible_wrappers(
                [paragraph], references={}, macros=macros
            )
            self.assertFalse(rejected)
            base, _ = MODULE.stable.build_source_probes(units, word_probe_kinds=set())
            probes, locators, atoms, _ = MODULE.build_atom_probes(units, base, wrappers)
            rows = {
                probes[0].probe_id: [
                    {"page_number": 1, "bbox_points": [40, 100, 80, 112]}
                ],
                probes[1].probe_id: [
                    {"page_number": 2, "bbox_points": [40, 80, 90, 92]}
                ],
            }
            fragments, reasons, _ = MODULE.build_fragments_for_shadow(
                units,
                probes,
                rows,
                atoms,
                locators,
                {1: 600.0, 2: 600.0},
            )
            self.assertFalse(reasons)
            self.assertEqual(fragments[1][0].markdown, "**Takeaway: first**")
            self.assertEqual(fragments[2][0].markdown, "**second**")

    def test_unsafe_or_competing_wrapper_definition_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "main.tex"
            source_file.write_text(
                r"\newcommand{\takeaway}[1]{\colorbox{yellow}{#1}}"
                "\n"
                r"\renewcommand{\takeaway}[1]{#1}"
                "\n"
                r"\takeaway{Do not guess}"
                "\n",
                encoding="utf-8",
            )
            macros, report = MODULE.collect_safe_visible_wrapper_macros([source_file])
            self.assertNotIn("takeaway", macros)
            self.assertIn("takeaway", report["blocked_macros"])
            paragraph = MODULE.page_gt.SourceParagraph(
                paragraph_id="unsafe-p",
                kind="paragraph",
                source_file=source_file,
                source_lines=[3],
                raw_latex=r"\takeaway{Do not guess}",
            )
            units, rejected, wrappers, unit_report = (
                MODULE.build_source_units_with_visible_wrappers(
                    [paragraph], references={}, macros=macros
                )
            )
            self.assertFalse(units)
            self.assertFalse(wrappers)
            self.assertEqual(unit_report["recovered_units"], 0)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                rejected[0]["visible_wrapper_recovery_reason"],
                "no_safe_visible_wrapper_definition",
            )

    def test_pdf_text_changes_verdict_but_never_fragment_markdown(self) -> None:
        fragment = MODULE.LocatedFragment(
            fragment_id="f",
            unit_id="u",
            paragraph_id="p",
            kind="paragraph",
            markdown="Source truth",
            probe_ids=("q",),
            source_file=Path("main.tex"),
            source_start_line=1,
            source_ordinal=1,
            page_number=1,
            bbox=(30, 30, 300, 60),
            components=((30, 30, 300, 60),),
        )
        passed = MODULE.choose_exact_page_candidate(
            [fragment], page_width=600, page_height=800, pdf_text="Source truth"
        )
        rejected = MODULE.choose_exact_page_candidate(
            [fragment], page_width=600, page_height=800, pdf_text="Different text"
        )
        self.assertEqual(passed["markdown"], "Source truth\n")
        self.assertEqual(rejected["status"], "rejected")
        attempts = rejected["attempts"]
        self.assertTrue(attempts)
        expected_hash = MODULE.hashlib.sha256(b"Source truth\n").hexdigest()
        self.assertTrue(all("markdown" not in attempt for attempt in attempts))
        self.assertTrue(
            all(attempt["markdown_sha256"] == expected_hash for attempt in attempts)
        )

    def test_math_brace_projection_is_strict_and_does_not_change_gt(self) -> None:
        markdown = r"Grid $T \in \{100, 200\}$ remains."
        pdf_text = "Grid T ∈ {100, 200} remains."
        self.assertEqual(MODULE.stable.verifier_result(markdown, pdf_text)["status"], "failed")
        projected, replacements = MODULE.project_math_visible_braces(markdown)
        self.assertEqual(replacements, 2)
        self.assertEqual(markdown, r"Grid $T \in \{100, 200\}$ remains.")
        self.assertIn("｛100, 200｝", projected)
        result = MODULE.experimental_verifier_result(markdown, pdf_text)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            result["experimental_projection"]["math_visible_brace_replacements"],
            2,
        )
        # Projection does not relax case or punctuation outside math.
        self.assertEqual(
            MODULE.experimental_verifier_result(markdown, pdf_text.lower())["status"],
            "failed",
        )

    def test_math_style_projection_is_verifier_only_and_preserves_gt(self) -> None:
        markdown = r"Layer $\textmd{DA-CR}$ contains $\mathcal{X}$."
        result = MODULE.experimental_verifier_result(
            markdown, "Layer DA-CR contains X."
        )
        self.assertEqual(result["status"], "passed")
        projection = result["experimental_projection"]["source_visible_flow"]
        self.assertEqual(projection["status"], "projected")
        self.assertEqual(projection["style_wrappers_removed"], 2)
        self.assertFalse(projection["ground_truth_changed"])
        self.assertFalse(projection["pdf_text_used_for_ground_truth"])
        self.assertEqual(
            markdown, r"Layer $\textmd{DA-CR}$ contains $\mathcal{X}$."
        )

    def test_frontier_candidate_limits_fail_closed(self) -> None:
        unit = MODULE.stable.SourceUnit(
            unit_id="whole",
            kind="paragraph",
            paragraph_id="p",
            source_file=Path("main.tex"),
            source_lines=(1,),
            raw_latex=" ".join(f"word{index}" for index in range(140)),
            markdown=" ".join(f"word{index}" for index in range(140)),
            rgb=(1, 2, 3),
        )
        suffixes, reason = MODULE.safe_whole_unit_suffixes(unit)
        self.assertFalse(suffixes)
        self.assertIn("whole_frontier_cut_limit_exceeded", str(reason))

        fragment = MODULE.LocatedFragment(
            fragment_id="f",
            unit_id="u",
            paragraph_id="p",
            kind="paragraph",
            markdown="Source text",
            probe_ids=("q",),
            source_file=Path("main.tex"),
            source_start_line=1,
            source_ordinal=1,
            page_number=1,
            bbox=(30, 30, 300, 60),
            components=((30, 30, 300, 60),),
        )
        previous_limit = MODULE.MAX_PAGE_SOURCE_CANDIDATES
        MODULE.MAX_PAGE_SOURCE_CANDIDATES = 1
        try:
            frozen = MODULE.freeze_page_source_candidates(
                [fragment],
                page_width=600,
                page_height=800,
                frontier_variants=[
                    {
                        "text": "prefix",
                        "join": "new_paragraph",
                        "provenance": {"kind": "test", "cut_index": 1},
                    }
                ],
            )
        finally:
            MODULE.MAX_PAGE_SOURCE_CANDIDATES = previous_limit
        self.assertEqual(frozen["status"], "failed")
        self.assertIn("page_candidate_limit_exceeded", frozen["reason"])
        self.assertFalse(frozen["candidates"])

    def test_table_v7_page2_and_page4_frontiers_are_unique_exact_offline(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "output"
            / "pdf"
            / "source_first_span_graph_v2_ieee_2605_30809_table_v7"
        )
        if not (root / "source_units.jsonl").is_file():
            self.skipTest("table_v7 offline diagnostic artifact is unavailable")

        def rows(path: Path):
            return [json.loads(line) for line in path.read_text().splitlines()]

        units = []
        for value in rows(root / "source_units.jsonl"):
            units.append(
                MODULE.stable.SourceUnit(
                    unit_id=value["unit_id"],
                    kind=value["kind"],
                    paragraph_id=value["source_paragraph_id"],
                    source_file=(root / "source_clean" / value["source_file"]).resolve(),
                    source_lines=tuple(value["source_line_numbers"]),
                    raw_latex=value["raw_latex"],
                    markdown=value["markdown"],
                    rgb=tuple(value["rgb"]),
                    source_command=value.get("source_command"),
                )
            )
        shadow = root / "shadows" / "legacy_words"
        probes = []
        for value in rows(shadow / "source_probes.jsonl"):
            span = value.get("source_token_span")
            probes.append(
                MODULE.stable.SourceProbe(
                    probe_id=value["probe_id"],
                    unit_id=value["source_unit_id"],
                    paragraph_id=value["source_paragraph_id"],
                    kind=value["kind"],
                    source_file=(root / "source_clean" / value["source_file"]).resolve(),
                    source_lines=tuple(value["source_line_numbers"]),
                    markdown_fragment=value["markdown_fragment"],
                    rgb=tuple(value["rgb"]),
                    ordinal=int(value["ordinal"]),
                    total=int(value["total"]),
                    localization_mode=value["localization_mode"],
                    token_span=(
                        int(span["line"]),
                        int(span["start_column"]),
                        int(span["end_column"]),
                    )
                    if span
                    else None,
                )
            )
        color_rows = {
            value["probe_id"]: value["pages"]
            for value in rows(shadow / "color_page_alignment.jsonl")
        }
        with MODULE.pdfplumber.open(root / "build_clean" / "samplepaper.pdf") as document:
            widths = {
                page_number: float(document.pages[page_number - 1].width)
                for page_number in range(1, len(document.pages) + 1)
            }
            fragments, reasons, _ = MODULE.build_fragments_for_shadow(
                units, probes, color_rows, {}, {}, widths
            )
            for page_number, expected_kind, expected_cut in (
                (2, "plain_word_token_suffix", 3),
                (4, "whole_unit_suffix", 246),
            ):
                self.assertFalse(reasons.get(page_number))
                variants, frontier_report = MODULE.enumerate_leading_frontier_variants(
                    page_number,
                    fragments[page_number],
                    units=units,
                    probes=probes,
                    rows=color_rows,
                )
                self.assertEqual(frontier_report["carrier_kind"], expected_kind)
                frozen = MODULE.freeze_page_source_candidates(
                    fragments[page_number],
                    page_width=widths[page_number],
                    page_height=float(document.pages[page_number - 1].height),
                    frontier_variants=variants,
                    frontier_report=frontier_report,
                )
                self.assertEqual(frozen["status"], "passed")
                self.assertLessEqual(
                    frozen["candidate_count"], MODULE.MAX_PAGE_SOURCE_CANDIDATES
                )
                pdf_text, _ = MODULE.stable.pdf_verifier_text(
                    document.pages[page_number - 1]
                )
                result = MODULE.verify_frozen_page_candidates(frozen, pdf_text)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["selected_frontier"]["kind"], expected_kind)
                self.assertEqual(result["selected_frontier"]["cut_index"], expected_cut)
                self.assertTrue(result["verifier"]["exact_ordered_character_stream_match"])
                self.assertTrue(all("markdown" not in row for row in result["attempts"]))
                serialized = json.dumps(
                    result, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self.assertLess(len(serialized), 1_000_000)
            self.assertEqual(
                result["verifier"]["experimental_projection"][
                    "math_visible_brace_replacements"
                ],
                6,
            )

    def test_2606_00856_retry3_pages_remain_unique_exact_with_compiler_labels(self) -> None:
        artifact_parent = Path(__file__).resolve().parents[1] / "output" / "pdf"
        current = artifact_parent / "source_first_span_graph_v2_pilot_2606_00856_frontier_v8"
        retry3 = artifact_parent / "source_first_span_graph_v2_pilot_2606_00856_retry3"
        if not (current / "source_units.jsonl").is_file() or not (
            retry3 / "pages" / "page_0002.md"
        ).is_file():
            self.skipTest("2606.00856 offline regression artifacts are unavailable")

        def rows(path: Path):
            return [json.loads(line) for line in path.read_text().splitlines()]

        units = []
        for value in rows(current / "source_units.jsonl"):
            units.append(
                MODULE.stable.SourceUnit(
                    unit_id=value["unit_id"],
                    kind=value["kind"],
                    paragraph_id=value["source_paragraph_id"],
                    source_file=(
                        current / "source_clean" / value["source_file"]
                    ).resolve(),
                    source_lines=tuple(value["source_line_numbers"]),
                    raw_latex=value["raw_latex"],
                    markdown=value["markdown"],
                    rgb=tuple(value["rgb"]),
                    source_command=value.get("source_command"),
                )
            )
        self.assertTrue(
            any(
                unit.kind == "heading" and unit.markdown == "## 2 Motivation and Origins"
                for unit in units
            ),
            "fixture must exercise compiler labels whose terminal dot is absent",
        )

        shadow = current / "shadows" / "legacy_words"
        probes = []
        for value in rows(shadow / "source_probes.jsonl"):
            span = value.get("source_token_span")
            probes.append(
                MODULE.stable.SourceProbe(
                    probe_id=value["probe_id"],
                    unit_id=value["source_unit_id"],
                    paragraph_id=value["source_paragraph_id"],
                    kind=value["kind"],
                    source_file=(
                        current / "source_clean" / value["source_file"]
                    ).resolve(),
                    source_lines=tuple(value["source_line_numbers"]),
                    markdown_fragment=value["markdown_fragment"],
                    rgb=tuple(value["rgb"]),
                    ordinal=int(value["ordinal"]),
                    total=int(value["total"]),
                    localization_mode=value["localization_mode"],
                    token_span=(
                        int(span["line"]),
                        int(span["start_column"]),
                        int(span["end_column"]),
                    )
                    if span
                    else None,
                )
            )
        color_rows = {
            value["probe_id"]: value["pages"]
            for value in rows(shadow / "color_page_alignment.jsonl")
        }
        pdf_path = current / "build_clean" / "paper.pdf"
        with MODULE.pdfplumber.open(pdf_path) as document:
            widths = {
                page_number: float(document.pages[page_number - 1].width)
                for page_number in range(1, len(document.pages) + 1)
            }
            fragments, reasons, _ = MODULE.build_fragments_for_shadow(
                units, probes, color_rows, {}, {}, widths
            )
            for page_number in (2, 6, 9, 11, 12):
                with self.subTest(page_number=page_number):
                    self.assertFalse(reasons.get(page_number))
                    variants, frontier_report = MODULE.enumerate_leading_frontier_variants(
                        page_number,
                        fragments[page_number],
                        units=units,
                        probes=probes,
                        rows=color_rows,
                    )
                    frozen = MODULE.freeze_page_source_candidates(
                        fragments[page_number],
                        page_width=widths[page_number],
                        page_height=float(document.pages[page_number - 1].height),
                        frontier_variants=variants,
                        frontier_report=frontier_report,
                    )
                    self.assertEqual(frozen["status"], "passed")
                    self.assertLessEqual(
                        frozen["candidate_count"], MODULE.MAX_PAGE_SOURCE_CANDIDATES
                    )
                    pdf_text, _ = MODULE.stable.pdf_verifier_text(
                        document.pages[page_number - 1]
                    )
                    result = MODULE.verify_frozen_page_candidates(frozen, pdf_text)
                    expected = (
                        retry3 / "pages" / f"page_{page_number:04d}.md"
                    ).read_text()
                    self.assertEqual(result["status"], "passed")
                    self.assertEqual(result["markdown"], expected)
                    self.assertIn(
                        "label_dot", result["selected_serialization_policy"]
                    )
                    self.assertTrue(
                        result["verifier"]["exact_ordered_character_stream_match"]
                    )
                    self.assertTrue(
                        all("markdown" not in row for row in result["attempts"])
                    )


if __name__ == "__main__":
    unittest.main()
