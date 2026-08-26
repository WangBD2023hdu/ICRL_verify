from __future__ import annotations

import unittest

from arxiv_source_first_v2.source_ir import (
    OPAQUE_MARKER_PREFIX,
    SourceAtom,
    atoms_to_markdown,
    build_source_atoms,
    reconstruct_page_markdown,
)


class SourceIrTest(unittest.TestCase):
    def test_bold_and_nested_emphasis_are_source_atoms(self) -> None:
        source = r"The \textbf{very \emph{important}} result."
        atoms = build_source_atoms(source)

        self.assertEqual(
            atoms_to_markdown(atoms),
            "The **very *important*** result.",
        )
        important = next(atom for atom in atoms if atom.raw_source == "important")
        self.assertEqual(important.kind, "text")
        self.assertEqual(important.style_stack, ("strong", "em"))
        self.assertEqual(source[important.source_start : important.source_end], "important")

    def test_inline_math_keeps_source_latex_and_normalizes_layout(self) -> None:
        source = r"Minimize \(L(\theta)\) with $\alpha_i^2$ and $" + "\n  x + y\n$."
        atoms = build_source_atoms(source)
        math = [atom for atom in atoms if atom.kind == "math"]

        self.assertEqual([atom.markdown_fragment for atom in math], [r"$L(\theta)$", r"$\alpha_i^2$", "$x + y$"])
        self.assertEqual(atoms_to_markdown(atoms), r"Minimize $L(\theta)$ with $\alpha_i^2$ and $x + y$.")

    def test_source_base_offset_is_applied_to_each_atom(self) -> None:
        source = r"A \textbf{bold} word."
        base = 10_000
        atoms = build_source_atoms(source, source_base_offset=base)

        for atom in atoms:
            self.assertEqual(
                source[atom.source_start - base : atom.source_end - base],
                atom.raw_source,
            )
        bold = next(atom for atom in atoms if atom.raw_source == "bold")
        self.assertEqual(bold.source_span, (base + source.index("bold"), base + source.index("bold") + 4))

    def test_cross_page_style_is_reopened_and_balanced(self) -> None:
        source = r"A \textbf{bold words} after."
        atoms = build_source_atoms(source)
        bold_words = [atom for atom in atoms if atom.raw_source in {"bold", "words"}]
        self.assertEqual(len(bold_words), 2)

        first_page = reconstruct_page_markdown(atoms, [bold_words[0].ordinal])
        second_page = reconstruct_page_markdown(atoms, [bold_words[1].ordinal])
        self.assertEqual(first_page, "**bold**")
        self.assertEqual(second_page, "**words**")
        self.assertEqual(first_page.count("**") % 2, 0)
        self.assertEqual(second_page.count("**") % 2, 0)

    def test_escaped_characters_and_code_style_are_source_derived(self) -> None:
        source = r"Use \& and \_ plus \texttt{a_b}."
        atoms = build_source_atoms(source)
        self.assertEqual(atoms_to_markdown(atoms), r"Use & and \_ plus `a\_b`.")
        self.assertEqual(next(atom for atom in atoms if atom.raw_source == "a_b").style_stack, ("code",))

    def test_opaque_macro_is_explicit_and_never_uses_pdf_text(self) -> None:
        source = r"Prior \citep{smith2020} gives \verb|x_y|."
        atoms = build_source_atoms(source)
        opaque = [atom for atom in atoms if atom.kind == "opaque"]
        self.assertEqual([atom.raw_source for atom in opaque], [r"\citep{smith2020}", r"\verb|x_y|"])
        rendered = atoms_to_markdown(atoms)
        self.assertIn(OPAQUE_MARKER_PREFIX + r"\citep{smith2020}", rendered)
        self.assertIn(OPAQUE_MARKER_PREFIX + r"\verb|x_y|", rendered)
        self.assertNotIn("[12]", rendered)


if __name__ == "__main__":
    unittest.main()
