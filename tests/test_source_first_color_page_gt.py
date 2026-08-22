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
            "A linked **word** and <sup>*</sup> marker.",
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
            "2. First **literal** item.",
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


if __name__ == "__main__":
    unittest.main()
