from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_source_first_color_page_gt.py"
)
SPEC = importlib.util.spec_from_file_location("build_source_first_color_page_gt", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SourceFirstColorPageGtTests(unittest.TestCase):
    def test_engine_specific_zero_dimension_color_switches(self) -> None:
        rgb = (10, 20, 30)
        self.assertIn("\\pdfliteral direct", MODULE.pdf_literal_color(rgb, "pdflatex"))
        self.assertIn(
            "\\special{pdf:literal direct",
            MODULE.pdf_literal_color(rgb, "xelatex"),
        )
        self.assertIn(
            "\\special{color push rgb",
            MODULE.pdf_literal_color(rgb, "latex_dvips_ps2pdf"),
        )
        self.assertEqual(
            MODULE.pdf_literal_restore("latex_dvips_ps2pdf"),
            "\\special{color pop}",
        )

    def test_compile_support_adds_color_and_empty_page_style(self) -> None:
        source = "\\documentclass{article}\n\\begin{document}\nText.\n\\end{document}\n"
        rendered = MODULE.inject_compile_support(source)
        self.assertIn("\\begin{document}\n\\pagestyle{empty}", rendered)

    def test_figure_removal_preserves_lines_and_drops_own_references(self) -> None:
        source = (
            "Before Figure~\\ref{fig:x}.\n"
            "\\begin{figure}[t]\n"
            "\\includegraphics{plot}\n"
            "\\caption{A plot.}\\label{fig:x}\n"
            "\\end{figure}\n"
            "After.\n"
            "% \\begin{figure} commented \\end{figure}\n"
        )
        rendered, report = MODULE.strip_ignored_figures(source)
        self.assertEqual(rendered.count("\n"), source.count("\n"))
        self.assertNotIn("includegraphics", rendered)
        self.assertNotIn("\\ref{fig:x}", rendered)
        self.assertIn("Before Figure~", rendered)
        self.assertIn("% \\begin{figure} commented", rendered)
        self.assertEqual(report["figure_environments_removed"], 1)
        self.assertEqual(report["figure_references_removed"], 1)

    def test_source_paragraph_markdown_preserves_inline_markup(self) -> None:
        paragraph = MODULE.page_gt.SourceParagraph(
            paragraph_id="sp-test",
            kind="paragraph",
            source_file=Path("main.tex"),
            source_lines=[10],
            raw_latex="A \\textbf{bold} and \\emph{soft} result with $x+y$.",
        )
        markdown = MODULE.source_paragraph_to_markdown(paragraph)
        self.assertEqual(markdown, "A **bold** and *soft* result with $x+y$.")

    def test_tex_text_punctuation_matches_compiled_glyphs_outside_math(self) -> None:
        paragraph = MODULE.page_gt.SourceParagraph(
            paragraph_id="sp-punctuation",
            kind="paragraph",
            source_file=Path("main.tex"),
            source_lines=[10],
            raw_latex="The agent's ``claim'' -- unlike $x'$ --- changed.",
        )
        self.assertEqual(
            MODULE.source_paragraph_to_markdown(paragraph),
            "The agent’s “claim” – unlike $x'$ — changed.",
        )

    def test_source_deterministic_commands_do_not_need_pdf_text(self) -> None:
        paragraph = MODULE.page_gt.SourceParagraph(
            paragraph_id="sp-deterministic",
            kind="paragraph",
            source_file=Path("main.tex"),
            source_lines=[11],
            raw_latex=(
                "A\\linebreak[4] linked \\href{https://x}{\\textbf{word}} "
                "and \\textsuperscript{*} marker."
            ),
        )
        self.assertEqual(
            MODULE.source_paragraph_to_markdown(paragraph),
            r"A linked **word** and <sup>\*</sup> marker.",
        )

    def test_source_layout_and_box_commands_keep_only_visible_source_text(self) -> None:
        paragraph = MODULE.page_gt.SourceParagraph(
            paragraph_id="sp-box",
            kind="paragraph",
            source_file=Path("main.tex"),
            source_lines=[11],
            raw_latex=(
                "\\medskip \\noindent \\fbox{Boxed} "
                "\\parbox{2cm}{paragraph text} "
                "\\resizebox{!}{1em}{scaled text}\\vspace{3pt}\\pagestyle{empty}"
            ),
        )
        self.assertEqual(
            MODULE.source_paragraph_to_markdown(paragraph),
            "Boxed paragraph text scaled text",
        )

    def test_text_thinspace_is_not_treated_as_unknown_math(self) -> None:
        paragraph = MODULE.page_gt.SourceParagraph(
            paragraph_id="sp-thinspace",
            kind="paragraph",
            source_file=Path("main.tex"),
            source_lines=[12],
            raw_latex=r"Text\,spacing and $x\,y$.",
        )
        self.assertEqual(
            MODULE.source_paragraph_to_markdown(paragraph),
            r"Text spacing and $x\,y$.",
        )

    def test_source_paragraph_rejects_compiler_dependent_macros(self) -> None:
        paragraph = MODULE.page_gt.SourceParagraph(
            paragraph_id="sp-test",
            kind="paragraph",
            source_file=Path("main.tex"),
            source_lines=[10],
            raw_latex="See Section~\\ref{sec:x}.",
        )
        with self.assertRaisesRegex(ValueError, "compiler-dependent"):
            MODULE.source_paragraph_to_markdown(paragraph)

    def test_aux_references_are_resolved_without_pdf_text(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            aux = Path(directory) / "main.aux"
            aux.write_text(
                "\\newlabel{sec:method}{{3.2}{7}{Method}{subsection.3.2}{}}\n"
                "\\newlabel{sec:method@cref}{{[subsection][2][3]3.2}{[1][7][]7}{}{}{}}\n"
                "\\newlabel{eq:loss}{{5}{8}{Method}{equation.5}{}}\n",
                encoding="utf-8",
            )
            references = MODULE.parse_aux_references([aux])
            self.assertEqual(references["sec:method"].kind, "subsection")
            paragraph = MODULE.page_gt.SourceParagraph(
                paragraph_id="sp-ref",
                kind="paragraph",
                source_file=Path("main.tex"),
                source_lines=[12],
                raw_latex=(
                    "See \\Cref{sec:method}, page~\\pageref{sec:method}, "
                    "and Eq.~\\eqref{eq:loss}."
                ),
            )
            self.assertEqual(
                MODULE.source_paragraph_to_markdown(paragraph, references),
                "See Section 3.2, page 7, and Eq. (5).",
            )

    def test_source_list_item_uses_source_structure(self) -> None:
        paragraph = MODULE.page_gt.SourceParagraph(
            paragraph_id="sp-item",
            kind="enumerate_item",
            source_file=Path("main.tex"),
            source_lines=[20],
            raw_latex="\\item First **literal** item.",
            list_environment="enumerate",
            item_depth=1,
            item_ordinal=2,
        )
        self.assertEqual(
            MODULE.source_paragraph_to_markdown(paragraph),
            r"2. First \*\*literal\*\* item.",
        )

    def test_aux_heading_numbers_are_source_compiler_metadata(self) -> None:
        with self.subTest("numbered section"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                aux = Path(directory) / "main.aux"
                aux.write_text(
                    "\\@writefile{toc}{\\contentsline {section}{\\numberline {3}Methods}{7}{section.3}}\n",
                    encoding="utf-8",
                )
                values = MODULE.parse_aux_heading_numbers(aux)
                self.assertEqual(values[("section", "methods")], ["3"])

    def test_aux_heading_metadata_records_class_unnumbered_commands(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            main_aux = Path(directory) / "main.aux"
            included_aux = Path(directory) / "included.aux"
            main_aux.write_text(
                "\\@writefile{toc}{\\contentsline {paragraph}{The insight.}{2}{section*.2}}\n",
                encoding="utf-8",
            )
            included_aux.write_text(
                "\\@writefile{toc}{\\contentsline {subsection}{\\numberline {A.1}Model}{3}{subsection.A.1}}\n"
                "\\@writefile{toc}{\\contentsline {paragraph}{Ambiguous}{3}{paragraph.1}}\n",
                encoding="utf-8",
            )
            values = MODULE.parse_aux_heading_numbers([main_aux, included_aux])
            self.assertEqual(values[("paragraph", "the insight.")], [None])
            self.assertEqual(values[("subsection", "model")], ["A.1"])
            self.assertNotIn(("paragraph", "ambiguous"), values)

            block = MODULE.page_gt.SourceBlock(
                block_id="b-heading",
                kind="heading",
                source_file=Path("main.tex"),
                start_line=20,
                end_line=20,
                raw_latex="The insight.",
                markdown="",
                query_lines=[20],
                heading_level=5,
                heading_command="paragraph",
                heading_starred=False,
                heading_source_title="The insight.",
            )
            units, rejected = MODULE.build_heading_units(
                [block], [main_aux, included_aux], color_index_offset=0
            )
            self.assertEqual(rejected, [])
            self.assertEqual(units[0].markdown, "##### The insight.")

    def test_unique_titleformat_label_supplies_visible_number_punctuation(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            style = root / "publication.sty"
            style.write_text(
                "\\titleformat{\\section}\n"
                "  {\\bfseries}{\\thesection.}{1em}{}\n"
                "\\titleformat{\\subsection}{\\bfseries}{Section~\\thesubsection:}{1em}{}\n",
                encoding="utf-8",
            )
            labels, report = MODULE.parse_unique_titleformat_labels([style])
            self.assertEqual(labels["section"], "{number}.")
            self.assertEqual(labels["subsection"], "Section {number}:")
            self.assertEqual(report["definitions_parsed"], 2)

            aux = root / "main.aux"
            aux.write_text(
                "\\@writefile{toc}{\\contentsline {section}{\\numberline {7}Model}{3}{section.7}}\n",
                encoding="utf-8",
            )
            block = MODULE.page_gt.SourceBlock(
                block_id="b-section",
                kind="heading",
                source_file=Path("main.tex"),
                start_line=30,
                end_line=30,
                raw_latex="Model",
                markdown="",
                query_lines=[30],
                heading_level=2,
                heading_command="section",
                heading_starred=False,
                heading_source_title="Model",
            )
            units, rejected = MODULE.build_heading_units(
                [block],
                aux,
                color_index_offset=0,
                heading_label_templates=labels,
            )
            self.assertEqual(rejected, [])
            self.assertEqual(units[0].markdown, "## 7. Model")

    def test_instrumentation_preserves_original_text_and_wraps_complete_unit(self) -> None:
        unit = MODULE.SourceUnit(
            unit_id="src-0000001",
            kind="paragraph",
            paragraph_id="sp-test",
            source_file=Path("main.tex"),
            source_lines=(2, 3),
            raw_latex="First line\ncontinues.",
            markdown="First line continues.",
            rgb=(10, 20, 30),
            source_command=None,
        )
        source = "before\nFirst line\ncontinues.\nafter\n"
        rendered = MODULE.instrument_source_file(source, [unit])
        color = MODULE.pdf_literal_color((10, 20, 30))
        black = MODULE.pdf_literal_color((0, 0, 0))
        self.assertIn("\\leavevmode" + color + "First line", rendered)
        self.assertIn("continues." + black + "\n", rendered)
        stripped = rendered.replace("\\leavevmode" + color, "").replace(
            black, ""
        )
        self.assertEqual(stripped, source)

    def test_heading_color_is_scoped_inside_title_argument(self) -> None:
        unit = MODULE.SourceUnit(
            unit_id="src-heading",
            kind="heading",
            paragraph_id="b1",
            source_file=Path("main.tex"),
            source_lines=(1,),
            raw_latex="Methods",
            markdown="## 2 Methods",
            rgb=(40, 50, 60),
            source_command="section",
        )
        rendered = MODULE.instrument_source_file("\\section{Methods}\\label{x}\n", [unit])
        self.assertEqual(
            rendered,
            "\\section{{"
            + MODULE.pdf_literal_color((40, 50, 60))
            + "Methods"
            + MODULE.pdf_literal_color((0, 0, 0))
            + "}}\\label{x}\n",
        )

    def test_run_in_heading_line_overlap_is_rejected_fail_closed(self) -> None:
        paragraph = MODULE.SourceUnit(
            unit_id="src-paragraph",
            kind="paragraph",
            paragraph_id="sp-body",
            source_file=Path("main.tex"),
            source_lines=(8, 9),
            raw_latex="Body.",
            markdown="Body.",
            rgb=(10, 20, 30),
        )
        heading = MODULE.SourceUnit(
            unit_id="src-heading",
            kind="heading",
            paragraph_id="block-heading",
            source_file=Path("main.tex"),
            source_lines=(8,),
            raw_latex="Title",
            markdown="#### Title",
            rgb=(40, 50, 60),
            source_command="paragraph",
        )
        accepted, rejected = MODULE.reject_line_overlaps([paragraph, heading])
        self.assertEqual(accepted, [])
        self.assertEqual({row["source_unit_id"] for row in rejected}, {
            "src-paragraph",
            "src-heading",
        })

    def test_list_color_starts_after_item_command(self) -> None:
        unit = MODULE.SourceUnit(
            unit_id="src-item",
            kind="itemize_item",
            paragraph_id="sp-item",
            source_file=Path("main.tex"),
            source_lines=(1,),
            raw_latex="\\item Visible text.",
            markdown="- Visible text.",
            rgb=(70, 80, 90),
            source_command=None,
        )
        rendered = MODULE.instrument_source_file("\\item Visible text.\n", [unit])
        self.assertEqual(
            rendered,
            "\\item"
            + MODULE.pdf_literal_color((70, 80, 90))
            + " Visible text."
            + MODULE.pdf_literal_color((0, 0, 0))
            + "\n",
        )

    def test_whole_list_color_starts_inside_leading_text_wrappers(self) -> None:
        source = "\\item \\underline{\\textit{Robustness}} body.\n"
        unit = MODULE.SourceUnit(
            unit_id="src-item-wrapper",
            kind="itemize_item",
            paragraph_id="sp-item-wrapper",
            source_file=Path("main.tex"),
            source_lines=(1,),
            raw_latex=source.rstrip("\n"),
            markdown="- *Robustness* body.",
            rgb=(70, 80, 90),
            source_command=None,
        )
        rendered = MODULE.instrument_source_file(source, [unit])
        self.assertEqual(
            rendered,
            "\\item \\underline{\\textit{"
            + MODULE.pdf_literal_color((70, 80, 90))
            + "Robustness}} body."
            + MODULE.pdf_literal_color((0, 0, 0))
            + "\n",
        )

    def test_noindent_remains_before_paragraph_color(self) -> None:
        unit = MODULE.SourceUnit(
            unit_id="src-noindent",
            kind="paragraph",
            paragraph_id="sp-noindent",
            source_file=Path("main.tex"),
            source_lines=(1,),
            raw_latex="\\noindent Text.",
            markdown="Text.",
            rgb=(11, 22, 33),
            source_command=None,
        )
        rendered = MODULE.instrument_source_file("\\noindent Text.\n", [unit])
        self.assertEqual(
            rendered,
            "\\noindent"
            + MODULE.pdf_literal_color((11, 22, 33))
            + " Text."
            + MODULE.pdf_literal_color((0, 0, 0))
            + "\n",
        )

    def test_verifier_is_exact_and_order_sensitive(self) -> None:
        passed = MODULE.verifier_result("Alpha beta.\n", "Alpha beta.")
        failed = MODULE.verifier_result("Alpha beta.\n", "beta Alpha.")
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(failed["status"], "failed")

    def test_verifier_strips_markup_but_preserves_visible_punctuation(self) -> None:
        passed = MODULE.verifier_result(
            "## 3 “Code is **Law**”\n\nValue $x_i$. <sup>1</sup>\n",
            "3 “Code is Law” Value xi. 1",
        )
        missing_quotes = MODULE.verifier_result(
            "## 3 Code is Law\n",
            "3 “Code is Law”",
        )
        wrong_case = MODULE.verifier_result("Alpha.\n", "alpha.")
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(missing_quotes["match_mode"], "punctuation_or_case_mismatch")
        self.assertEqual(missing_quotes["status"], "failed")
        self.assertEqual(wrong_case["status"], "failed")

    def test_verifier_never_discards_unmatched_or_unknown_markup_like_text(self) -> None:
        cases = [
            ("A * literal.\n", "A literal."),
            ("A ` literal.\n", "A literal."),
            ("A [link](url).\n", "A link."),
            ("A <x> literal.\n", "A literal."),
        ]
        for markdown, pdf_text in cases:
            with self.subTest(markdown=markdown):
                self.assertEqual(
                    MODULE.verifier_result(markdown, pdf_text)["status"],
                    "failed",
                )

    def test_verifier_unescapes_source_literal_markdown_characters(self) -> None:
        result = MODULE.verifier_result(
            r"A \*literal\* \_value\_ \<x\> \#1." + "\n",
            "A *literal* _value_ <x> #1.",
        )
        self.assertEqual(result["status"], "passed")

    def test_verifier_protects_code_contents_from_html_and_emphasis_parsing(self) -> None:
        passed = MODULE.verifier_result(
            "A `<sup>*</sup>` literal.\n",
            "A <sup>*</sup> literal.",
        )
        missing = MODULE.verifier_result(
            "A `<sup>*</sup>` literal.\n",
            "A literal.",
        )
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(missing["status"], "failed")

    def test_verifier_uses_tex_math_minus_without_changing_text_hyphens(self) -> None:
        math_result = MODULE.verifier_result("Value $x-y$.\n", "Value x−y.")
        text_result = MODULE.verifier_result("high-low\n", "high−low")
        self.assertEqual(math_result["status"], "passed")
        self.assertEqual(text_result["status"], "failed")

    def test_plain_word_probes_preserve_source_and_markdown(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "main.tex"
            source = "before\nAlpha beta\ncontinues here.\nafter\n"
            source_path.write_text(source, encoding="utf-8")
            unit = MODULE.SourceUnit(
                unit_id="src-0000001",
                kind="paragraph",
                paragraph_id="sp-plain",
                source_file=source_path,
                source_lines=(2, 3),
                raw_latex="Alpha beta\ncontinues here.",
                markdown="Alpha beta continues here.",
                rgb=(10, 20, 30),
            )
            probes, modes = MODULE.build_source_probes([unit])
            self.assertEqual(modes, {unit.unit_id: "plain_word"})
            self.assertEqual(len(probes), 4)
            self.assertEqual(
                "".join(probe.markdown_fragment for probe in probes),
                unit.markdown,
            )
            self.assertEqual(
                [probe.token_span for probe in probes],
                [(2, 0, 5), (2, 6, 10), (3, 0, 9), (3, 10, 15)],
            )
            self.assertEqual(len({probe.rgb for probe in probes}), len(probes))
            self.assertNotIn(unit.rgb, {probe.rgb for probe in probes})

            rendered = MODULE.instrument_source_file(
                source,
                [unit],
                probes=probes,
            )
            stripped = rendered
            for probe in probes:
                stripped = stripped.replace(MODULE.pdf_literal_color(probe.rgb), "")
            stripped = stripped.replace(MODULE.pdf_literal_restore(), "")
            stripped = stripped.replace("\\leavevmode", "")
            self.assertEqual(stripped, source)

    def test_word_probes_preserve_inline_markup(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "main.tex"
            source_path.write_text(
                "A \\textbf{bold word} with $x+y$.\n",
                encoding="utf-8",
            )
            unit = MODULE.SourceUnit(
                unit_id="src-0000001",
                kind="paragraph",
                paragraph_id="sp-markup",
                source_file=source_path,
                source_lines=(1,),
                raw_latex="A \\textbf{bold word} with $x+y$.",
                markdown="A **bold word** with $x+y$.",
                rgb=(10, 20, 30),
            )
            probes, modes = MODULE.build_source_probes([unit])
            self.assertEqual(modes, {unit.unit_id: "plain_word"})
            self.assertEqual(len(probes), 5)
            self.assertEqual(
                "".join(probe.markdown_fragment for probe in probes),
                unit.markdown,
            )

    def test_word_probe_falls_back_for_whitespace_bearing_inline_math(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "main.tex"
            source_path.write_text("A formula $x + y$ here.\n", encoding="utf-8")
            unit = MODULE.SourceUnit(
                unit_id="src-0000001",
                kind="paragraph",
                paragraph_id="sp-spaced-math",
                source_file=source_path,
                source_lines=(1,),
                raw_latex="A formula $x + y$ here.",
                markdown="A formula $x + y$ here.",
                rgb=(10, 20, 30),
            )
            probes, modes = MODULE.build_source_probes([unit])
            self.assertEqual(modes, {unit.unit_id: "whole"})
            self.assertEqual(len(probes), 1)
            self.assertEqual(probes[0].localization_mode, "whole")

    def test_list_item_word_probes_keep_source_derived_marker(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "main.tex"
            source_path.write_text("\\item Alpha bold words.\n", encoding="utf-8")
            unit = MODULE.SourceUnit(
                unit_id="src-0000001",
                kind="itemize_item",
                paragraph_id="sp-item",
                source_file=source_path,
                source_lines=(1,),
                raw_latex="\\item Alpha bold words.",
                markdown="- Alpha bold words.",
                rgb=(10, 20, 30),
            )
            probes, modes = MODULE.build_source_probes([unit])
            self.assertEqual(modes, {unit.unit_id: "plain_word"})
            self.assertEqual(len(probes), 3)
            self.assertTrue(probes[0].markdown_fragment.startswith("- Alpha"))
            self.assertEqual(
                "".join(probe.markdown_fragment for probe in probes),
                unit.markdown,
            )
            rendered = MODULE.instrument_source_file(
                source_path.read_text(encoding="utf-8"),
                [unit],
                probes=probes,
            )
            self.assertTrue(rendered.startswith("\\item \\leavevmode"))
            fallback_probes, fallback_modes = MODULE.build_source_probes(
                [unit], word_probe_kinds={"paragraph"}
            )
            self.assertEqual(fallback_modes, {unit.unit_id: "whole"})
            self.assertEqual(fallback_probes[0].localization_mode, "whole")

    def test_plain_word_fragments_split_one_source_paragraph_across_pages(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "main.tex"
            source_path.write_text("Alpha beta continues here.\n", encoding="utf-8")
            unit = MODULE.SourceUnit(
                unit_id="src-0000001",
                kind="paragraph",
                paragraph_id="sp-cross-page",
                source_file=source_path,
                source_lines=(1,),
                raw_latex="Alpha beta continues here.",
                markdown="Alpha beta continues here.",
                rgb=(10, 20, 30),
            )
            probes, _ = MODULE.build_source_probes([unit])
            rows = {}
            for index, probe in enumerate(probes):
                page_number = 1 if index < 2 else 2
                rows[probe.probe_id] = [
                    {
                        "page_number": page_number,
                        "bbox_points": [10.0 + index, 20.0, 11.0 + index, 21.0],
                        "characters": len(probe.markdown_fragment.strip()),
                    }
                ]
            placements, reasons, summary = MODULE.build_page_fragments(
                [unit], probes, rows
            )
            self.assertEqual(dict(reasons), {})
            self.assertEqual(summary["plain_units_split_across_pages"], 1)
            self.assertEqual(MODULE.compose_page_markdown(placements[1]), "Alpha beta\n")
            self.assertEqual(
                MODULE.compose_page_markdown(placements[2]),
                "continues here.\n",
            )

    def test_run_in_heading_uses_source_order_for_tight_visual_tie(self) -> None:
        source = Path("main.tex")
        body = MODULE.PageFragment(
            fragment_id="body",
            unit_id="src-body",
            paragraph_id="sp-body",
            kind="paragraph",
            markdown="Body text.",
            probe_ids=("body-probe",),
            source_file=source,
            source_start_line=21,
        )
        heading = MODULE.PageFragment(
            fragment_id="heading",
            unit_id="src-heading",
            paragraph_id="b-heading",
            kind="heading",
            markdown="#### The insight.",
            probe_ids=("heading-probe",),
            source_file=source,
            source_start_line=20,
        )
        ordered, error = MODULE.order_page_units(
            [
                (body, {"bbox_points": [72.0, 100.0, 500.0, 125.0]}),
                (heading, {"bbox_points": [72.0, 102.0, 150.0, 113.0]}),
            ],
            612.0,
        )
        self.assertIsNone(error)
        self.assertEqual([fragment.fragment_id for fragment, _ in ordered], [
            "heading",
            "body",
        ])

    def test_nonmonotonic_word_probe_pages_are_rejected(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "main.tex"
            source_path.write_text("Alpha beta.\n", encoding="utf-8")
            unit = MODULE.SourceUnit(
                unit_id="src-0000001",
                kind="paragraph",
                paragraph_id="sp-order",
                source_file=source_path,
                source_lines=(1,),
                raw_latex="Alpha beta.",
                markdown="Alpha beta.",
                rgb=(10, 20, 30),
            )
            probes, _ = MODULE.build_source_probes([unit])
            rows = {
                probes[0].probe_id: [
                    {"page_number": 2, "bbox_points": [1, 1, 2, 2], "characters": 5}
                ],
                probes[1].probe_id: [
                    {"page_number": 1, "bbox_points": [1, 1, 2, 2], "characters": 5}
                ],
            }
            placements, reasons, summary = MODULE.build_page_fragments(
                [unit], probes, rows
            )
            self.assertEqual(dict(placements), {})
            self.assertEqual(
                dict(reasons),
                {
                    1: {"source_word_probe_page_order_mismatch"},
                    2: {"source_word_probe_page_order_mismatch"},
                },
            )
            self.assertEqual(summary["plain_units_page_order_mismatch"], 1)

    def test_verifier_accepts_only_exact_word_boundary_merge(self) -> None:
        merged = MODULE.verifier_result("x cell\n", "xcell")
        changed = MODULE.verifier_result("x cell\n", "x sell")
        self.assertEqual(merged["status"], "passed")
        self.assertFalse(merged["exact_ordered_token_match"])
        self.assertTrue(merged["exact_ordered_character_stream_match"])
        self.assertEqual(merged["match_mode"], "exact_visible_character_stream")
        self.assertEqual(changed["status"], "failed")

    def test_verifier_treats_visual_line_end_hyphen_as_source_optional(self) -> None:
        semantic = MODULE.verifier_result(
            "confident-wrong result.\n",
            "confident" + MODULE.OPTIONAL_LINE_END_HYPHEN + "wrong result.",
        )
        discretionary = MODULE.verifier_result(
            "probabilistic result.\n",
            "probabilis" + MODULE.OPTIONAL_LINE_END_HYPHEN + "tic result.",
        )
        self.assertEqual(semantic["status"], "passed")
        self.assertEqual(discretionary["status"], "passed")


if __name__ == "__main__":
    unittest.main()
