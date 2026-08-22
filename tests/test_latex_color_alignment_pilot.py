from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_latex_color_alignment_pilot.py"
SPEC = importlib.util.spec_from_file_location("build_latex_color_alignment_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LatexColorAlignmentPilotTests(unittest.TestCase):
    def test_deterministic_colors_are_unique(self) -> None:
        colors = [MODULE.deterministic_rgb(index) for index in range(5000)]
        self.assertEqual(len(colors), len(set(colors)))
        self.assertTrue(all(all(0 <= channel <= 255 for channel in color) for color in colors))

    def test_comments_math_and_nonvisible_arguments_are_not_colored(self) -> None:
        source = (
            "Visible prose. % hidden comment words\n"
            "Math $x + y$ and reference \\ref{sec:hidden}.\n"
        )
        colored, tokens = MODULE.colorize_source_range(
            source,
            source_file="main.tex",
            start_line=1,
            end_line=2,
        )
        token_text = [token.text for token in tokens]
        self.assertEqual(token_text, ["Visible", "prose", ".", "Math", "and", "reference", "."])
        self.assertIn("% hidden comment words", colored)
        self.assertIn("$x + y$", colored)
        self.assertIn("\\ref{sec:hidden}", colored)
        self.assertNotIn("color[RGB]{", colored[colored.index("% hidden") : colored.index("\n")])

    def test_visible_macro_arguments_are_colored_but_command_is_preserved(self) -> None:
        source = "\\emph{Important words} remain.\n"
        colored, tokens = MODULE.colorize_source_range(
            source,
            source_file="main.tex",
            start_line=1,
            end_line=1,
        )
        self.assertTrue(colored.startswith("\\emph{"))
        self.assertEqual([token.text for token in tokens], ["Important", "words", "remain", "."])

    def test_pdf_rgb_normalization(self) -> None:
        self.assertEqual(MODULE.normalize_pdf_rgb((1.0, 0.5, 0.0)), (255, 128, 0))
        self.assertEqual(MODULE.normalize_pdf_rgb((12, 34, 56)), (12, 34, 56))
        self.assertEqual(MODULE.normalize_pdf_rgb(0.5), (128, 128, 128))

    def test_paragraph_mode_uses_one_color_group_per_source_paragraph(self) -> None:
        source = "First line\ncontinues here.\n\nSecond paragraph.\n"
        rendered, blocks = MODULE.colorize_paragraphs(
            source,
            source_file="intro.tex",
            start_line=1,
            end_line=4,
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual([block.source_line for block in blocks], [1, 4])
        self.assertEqual(rendered.count("\\color[RGB]"), 2)
        self.assertEqual(rendered.count("\\par}"), 2)
        self.assertIn("First line\ncontinues here.", rendered)

    def test_reference_removal_drops_citations_but_preserves_comments(self) -> None:
        source = (
            "Claim~\\citep[see][p.~2]{alpha,beta}; \\citet{gamma} agrees.\n"
            "% commented \\cite{keep-this-comment}\n"
        )
        rendered, stats = MODULE.strip_reference_content(source)
        self.assertNotIn("alpha", rendered)
        self.assertNotIn("beta", rendered)
        self.assertNotIn("gamma", rendered)
        self.assertIn("% commented \\cite{keep-this-comment}", rendered)
        self.assertEqual(sum(stats["citation_commands"].values()), 2)
        self.assertEqual(MODULE.visible_reference_markers(rendered), [])

    def test_unknown_custom_citation_command_is_fail_closed(self) -> None:
        markers = MODULE.visible_reference_markers(
            "A result \\briefCite{alpha}.\\citestyle{authoryear}\n"
        )
        self.assertEqual(markers, ["unknown-citation-command:\\briefCite"])

    def test_reference_removal_drops_bibliography_output_but_keeps_cross_refs(self) -> None:
        source = (
            "See Figure~\\ref{fig:x}.\n"
            "\\section*{References}\n"
            "\\begin{thebibliography}{9}\n"
            "\\bibitem{alpha} A. Author.\n"
            "\\end{thebibliography}\n"
            "\\bibliographystyle{plain}\n"
            "\\bibliography{paper}\n"
            "\\input{cached.bbl}\n"
        )
        rendered, stats = MODULE.strip_reference_content(source)
        self.assertIn("\\ref{fig:x}", rendered)
        self.assertNotIn("A. Author", rendered)
        self.assertNotIn("cached.bbl", rendered)
        self.assertEqual(stats["reference_headings"], 1)
        self.assertEqual(sum(stats["reference_environments"].values()), 1)
        self.assertEqual(sum(stats["bibliography_commands"].values()), 2)
        self.assertEqual(stats["input_bbl_commands"], 1)
        self.assertEqual(MODULE.visible_reference_markers(rendered), [])


if __name__ == "__main__":
    unittest.main()
