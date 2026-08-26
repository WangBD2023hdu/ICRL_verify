from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from arxiv_source_first_v2.structural_ir import (
    StructuralIRError,
    build_theorem_ir,
    build_theorem_ir_from_sources,
    collect_theorem_definitions_from_sources,
    mask_tex_comments,
    render_static_visible_text,
    resolve_display_equation_tail,
)


class SourceFirstV2StructuralIRTests(unittest.TestCase):
    def test_static_within_and_shared_counter_declarations_ignore_comments(self) -> None:
        source = r"""
% \newtheorem{claim}{Fake Claim}
\newtheorem{theorem}{Theorem}[section] % \newtheorem{theorem}{Conflict}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{prop}{Proposition}
"""
        registry = collect_theorem_definitions_from_sources({"main.tex": source})
        self.assertFalse(registry.rejections)
        self.assertEqual(
            [item.environment for item in registry.definitions],
            ["theorem", "lemma", "prop"],
        )
        theorem = registry.by_environment["theorem"]
        self.assertEqual(theorem.within_counter, "section")
        self.assertIsNone(theorem.shared_counter)
        self.assertEqual(theorem.numbering_policy, "within_counter")
        lemma = registry.by_environment["lemma"]
        self.assertEqual(lemma.shared_counter, "theorem")
        self.assertEqual(lemma.numbering_policy, "shared_counter")
        self.assertEqual(registry.by_environment["prop"].visible_name_plain, "Proposition")
        self.assertTrue(
            source[slice(*theorem.declaration_span)].startswith(r"\newtheorem")
        )
        self.assertFalse(registry.as_report()["pdf_text_used"])

    def test_llncs_static_declarations_are_admitted_without_counter_guessing(self) -> None:
        source = r"""
\def\spnewtheorem{\@ifstar{\@sthm}{\@Sthm}}
\spnewtheorem{theorem}{Theorem}[section]{\bfseries}{\itshape}
\spnewtheorem{theorem}{Theorem}{\bfseries}{\itshape}
\spnewtheorem*{claim}{Claim}{\itshape}{\rmfamily}
\def\spn@wtheorem#1#2#3#4{\@spothm{#1}[theorem]{#2}{#3}{#4}}
\spn@wtheorem{corollary}{Corollary}{\bfseries}{\itshape}
\spn@wtheorem{definition}{Definition}{\bfseries}{\itshape}
\spn@wtheorem{example}{Example}{\itshape}{\rmfamily}
\spn@wtheorem{lemma}{Lemma}{\bfseries}{\itshape}
"""
        registry = collect_theorem_definitions_from_sources(
            {Path("source_clean/llncs.cls"): source}
        )
        self.assertFalse(registry.rejections)
        self.assertEqual(
            set(registry.by_environment),
            {"theorem", "claim", "corollary", "definition", "example", "lemma"},
        )

        theorem = registry.by_environment["theorem"]
        self.assertEqual(theorem.visible_name_plain, "Theorem")
        self.assertEqual(theorem.numbering_policy, "compiler_aux_only")
        self.assertEqual(theorem.counter_semantics, "compiler_aux_only")
        self.assertIsNone(theorem.shared_counter)
        self.assertIsNone(theorem.within_counter)
        self.assertEqual(len(theorem.equivalent_declaration_sites), 2)
        self.assertTrue(
            all(
                item.source_file == Path("source_clean/llncs.cls")
                for item in theorem.equivalent_declaration_sites
            )
        )

        definition = registry.by_environment["definition"]
        self.assertEqual(definition.declaration_command, "spn@wtheorem")
        self.assertEqual(definition.numbering_policy, "compiler_aux_only")
        self.assertTrue(definition.numbered)

        claim = registry.by_environment["claim"]
        self.assertEqual(claim.declaration_command, "spnewtheorem*")
        self.assertFalse(claim.numbered)
        self.assertEqual(claim.numbering_policy, "unnumbered")
        report = registry.as_report()
        self.assertEqual(
            report["policy"], "static_newtheorem_llncs_source_registry_v2"
        )
        self.assertTrue(all(not item["pdf_text_used"] for item in report["definitions"]))

    def test_llncs_heading_uses_only_static_caption_and_unique_aux_number(self) -> None:
        source = r"""
\spn@wtheorem{definition}{Definition}{\bfseries}{\itshape}
\begin{definition}[Finite witness]
\label{def:witness}
The source body.
\end{definition}
"""
        ir = build_theorem_ir_from_sources(
            {"main.tex": source}, {"def:witness": "4.3"}
        )
        self.assertFalse(ir.rejections)
        self.assertEqual(len(ir.blocks), 1)
        block = ir.blocks[0]
        self.assertEqual(block.definition.numbering_policy, "compiler_aux_only")
        self.assertEqual(block.aux_number, "4.3")
        self.assertEqual(block.definition.visible_name_source, "Definition")
        self.assertIn(
            "Definition 4.3 (Finite witness)",
            {item.visible_text for item in block.heading_candidates},
        )
        self.assertTrue(
            all(
                item.as_json()["generation_sources"]
                == ["latex_source", "compiler_aux"]
                for item in block.heading_candidates
            )
        )

    def test_llncs_conflicts_dynamic_arguments_and_unnumbered_blocks_fail_closed(self) -> None:
        source = r"""
\spn@wtheorem{definition}{Definition}{\bfseries}{\itshape}
\spn@wtheorem{definition}{Concept}{\bfseries}{\itshape}
\spn@wtheorem{example}{\dynamiccaption}{\itshape}{\rmfamily}
\spnewtheorem{lemma}{Lemma}{\fontmacro{#1}}{\itshape}
\spn@wtheorem{\dynamicenv}{Corollary}{\bfseries}{\itshape}
\def\installproperty{\spn@wtheorem{property}{Property}{\itshape}{\rmfamily}}
\spnewtheorem*{claim}{Claim}{\itshape}{\rmfamily}
\begin{claim}\label{claim:x}x\end{claim}
"""
        registry = collect_theorem_definitions_from_sources({"llncs.cls": source})
        self.assertNotIn("definition", registry.by_environment)
        self.assertNotIn("example", registry.by_environment)
        self.assertNotIn("lemma", registry.by_environment)
        self.assertNotIn("property", registry.by_environment)
        self.assertIn("claim", registry.by_environment)
        codes = [item.code for item in registry.rejections]
        self.assertEqual(codes.count("conflicting_or_redefined_environment"), 2)
        self.assertIn("unsafe_visible_name", codes)
        self.assertIn("dynamic_style_argument", codes)
        self.assertIn("dynamic_environment_name", codes)
        self.assertIn("dynamic_declaration_context", codes)

        ir = build_theorem_ir_from_sources(
            {"llncs.cls": source},
            {"claim:x": "99"},
            registry=registry,
        )
        self.assertFalse(ir.blocks)
        self.assertIn(
            "unnumbered_theorem_has_no_aux_number",
            [item.code for item in ir.rejections],
        )

    def test_conflicts_redefinitions_dynamic_names_and_unknown_titles_fail_closed(self) -> None:
        source = r"""
\newtheorem{theorem}{Theorem}
\newtheorem{theorem}{Different theorem}
\renewtheorem{lemma}{Lemma}
\newtheorem{\csname dynamic\endcsname}{Claim}
\newtheorem{remark}{A \mystery Remark}
"""
        registry = collect_theorem_definitions_from_sources({Path("defs.tex"): source})
        self.assertNotIn("theorem", registry.by_environment)
        self.assertNotIn("lemma", registry.by_environment)
        self.assertNotIn("remark", registry.by_environment)
        codes = [item.code for item in registry.rejections]
        self.assertEqual(codes.count("conflicting_or_redefined_environment"), 2)
        self.assertIn("unsupported_redefinition", codes)
        self.assertIn("dynamic_environment_name", codes)
        self.assertIn("unsafe_visible_name", codes)
        self.assertTrue(all(not item.as_json()["pdf_text_used"] for item in registry.rejections))

    def test_balanced_theorem_unique_label_aux_and_finite_heading_candidates(self) -> None:
        source = r"""\newtheorem{theorem}{Theorem}[section]
\begin{theorem}[A \textbf{sharp} bound]
% \label{fake:comment}
\label{thm:sharp}
Every admissible object has the desired property.
\end{theorem}
"""
        ir = build_theorem_ir_from_sources(
            {"main.tex": source}, {"thm:sharp": "2.7"}
        )
        self.assertFalse(ir.rejections)
        self.assertEqual(len(ir.blocks), 1)
        block = ir.blocks[0]
        self.assertEqual(block.environment, "theorem")
        self.assertEqual(block.label, "thm:sharp")
        self.assertEqual(block.aux_number, "2.7")
        self.assertEqual(block.optional_title_markdown, "A **sharp** bound")
        self.assertEqual(source[slice(*block.block_span)], block.raw_latex)
        self.assertEqual(source[slice(*block.begin_span)], r"\begin{theorem}")
        self.assertEqual(source[slice(*block.end_span)], r"\end{theorem}")
        self.assertEqual(source[slice(*block.label_span)], "thm:sharp")
        self.assertEqual(len(block.heading_candidates), 5)
        visible = {item.visible_text for item in block.heading_candidates}
        self.assertIn("Theorem 2.7 (A **sharp** bound)", visible)
        self.assertIn("Theorem 2.7: A **sharp** bound", visible)
        self.assertTrue(
            all(item.markdown == item.visible_text for item in block.heading_candidates)
        )
        self.assertTrue(
            all(not item.as_json()["pdf_text_used"] for item in block.heading_candidates)
        )

    def test_zero_multiple_labels_and_ambiguous_aux_are_explicit_rejections(self) -> None:
        source = r"""\newtheorem{lemma}{Lemma}
\begin{lemma}No label.\end{lemma}
\begin{lemma}\label{a}\label{b}Two labels.\end{lemma}
\begin{lemma}\label{c}Ambiguous AUX.\end{lemma}
"""
        ir = build_theorem_ir_from_sources(
            {"lemmas.tex": source}, {"a": "1", "b": "2", "c": ["3", "4"]}
        )
        self.assertFalse(ir.blocks)
        codes = [item.code for item in ir.rejections]
        self.assertIn("missing_unique_label", codes)
        self.assertIn("multiple_labels_in_block", codes)
        self.assertIn("ambiguous_aux_number", codes)

    def test_nested_mismatched_and_unbalanced_theorem_blocks_fail_closed(self) -> None:
        nested = r"""\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\begin{theorem}\label{outer}
\begin{lemma}\label{inner}x\end{lemma}
\end{theorem}
"""
        nested_ir = build_theorem_ir_from_sources(
            {"nested.tex": nested}, {"outer": "1", "inner": "2"}
        )
        self.assertFalse(nested_ir.blocks)
        self.assertIn(
            "nested_theorem_environment",
            [item.code for item in nested_ir.rejections],
        )

        unbalanced = r"""\newtheorem{claim}{Claim}
\begin{claim}\label{open}Never closed.
"""
        unbalanced_ir = build_theorem_ir_from_sources(
            {"bad.tex": unbalanced}, {"open": "1"}
        )
        self.assertFalse(unbalanced_ir.blocks)
        self.assertIn(
            "unbalanced_theorem_environment",
            [item.code for item in unbalanced_ir.rejections],
        )

        mismatched = r"""\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\begin{theorem}\label{x}x\end{lemma}
"""
        mismatch_ir = build_theorem_ir_from_sources(
            {"mismatch.tex": mismatched}, {"x": "1"}
        )
        self.assertFalse(mismatch_ir.blocks)
        self.assertIn(
            "mismatched_theorem_end",
            [item.code for item in mismatch_ir.rejections],
        )

    def test_unknown_macro_in_optional_title_rejects_only_that_block(self) -> None:
        source = r"""\newtheorem{example}{Example}
\begin{example}[Known title]\label{ok}x\end{example}
\begin{example}[Unknown \projectmacro]\label{bad}y\end{example}
"""
        ir = build_theorem_ir_from_sources(
            {"examples.tex": source}, {"ok": "1", "bad": "2"}
        )
        self.assertEqual(len(ir.blocks), 1)
        self.assertEqual(ir.blocks[0].label, "ok")
        rejection = next(item for item in ir.rejections if item.code == "unsafe_optional_title")
        self.assertIn("unknown_macro_in_visible_text:projectmacro", rejection.message)

    def test_equation_aux_number_preserves_formula_markdown(self) -> None:
        raw = r"""\begin{equation}
x = y
\label{eq:xy}
\end{equation}"""
        formula = "$$\nx = y\n$$"
        resolution = resolve_display_equation_tail(
            raw,
            formula,
            source_file="equations.tex",
            aux_label_numbers={"eq:xy": "3.2"},
            source_offset=100,
            start_line=20,
        )
        self.assertEqual(resolution.status, "accepted")
        self.assertEqual(resolution.formula_markdown, formula)
        self.assertEqual(resolution.number_source, "compiler_aux")
        label_start = 100 + raw.index("eq:xy")
        self.assertEqual(
            resolution.number_source_span,
            (label_start, label_start + len("eq:xy")),
        )
        self.assertEqual(
            [item.tail_text for item in resolution.candidates], ["(3.2)", "3.2"]
        )
        self.assertTrue(
            all(item.formula_markdown == formula for item in resolution.candidates)
        )
        self.assertTrue(
            all(item.markdown.startswith(formula + "\n") for item in resolution.candidates)
        )
        self.assertEqual(resolution.block_span, (100, 100 + len(raw)))

    def test_explicit_equation_tag_wins_without_aux_and_comments_are_ignored(self) -> None:
        raw = r"""\begin{align}
x &= y \tag{A.1} % \tag{fake}
\end{align}"""
        formula = "$$\nx &= y\n$$"
        resolution = resolve_display_equation_tail(
            raw,
            formula,
            source_file="tagged.tex",
            aux_label_numbers={},
        )
        self.assertEqual(resolution.status, "accepted")
        self.assertEqual(resolution.number_source, "explicit_tag")
        self.assertIsNone(resolution.label)
        self.assertEqual(resolution.number, "A.1")
        self.assertEqual(
            [item.tail_text for item in resolution.candidates], ["(A.1)", "A.1"]
        )
        self.assertEqual(resolution.formula_markdown, formula)

        symbolic = resolve_display_equation_tail(
            r"\begin{equation}x\tag{*}\end{equation}",
            "$$x$$",
            source_file="tagged.tex",
            aux_label_numbers={},
        )
        self.assertEqual(symbolic.status, "accepted")
        self.assertEqual(symbolic.number, r"\*")
        self.assertEqual(
            [item.tail_text for item in symbolic.candidates],
            [r"(\*)", r"\*"],
        )

    def test_equation_multiple_labels_tags_and_unknown_tag_macro_reject(self) -> None:
        multiple_labels = resolve_display_equation_tail(
            r"\begin{equation}\label{a}\label{b}x\end{equation}",
            "$$x$$",
            source_file="eq.tex",
            aux_label_numbers={"a": "1", "b": "2"},
        )
        self.assertEqual(multiple_labels.status, "rejected")
        self.assertEqual(multiple_labels.rejections[0].code, "multiple_labels_in_block")

        multiple_tags = resolve_display_equation_tail(
            r"\begin{equation}x\tag{1}\tag{2}\end{equation}",
            "$$x$$",
            source_file="eq.tex",
            aux_label_numbers={},
        )
        self.assertEqual(multiple_tags.status, "rejected")
        self.assertEqual(multiple_tags.rejections[0].code, "multiple_equation_tags")

        dynamic_tag = resolve_display_equation_tail(
            r"\begin{equation}x\tag{\countervalue}\end{equation}",
            "$$x$$",
            source_file="eq.tex",
            aux_label_numbers={},
        )
        self.assertEqual(dynamic_tag.status, "rejected")
        self.assertEqual(dynamic_tag.rejections[0].code, "unsafe_equation_tag")

    def test_file_api_and_comment_mask_keep_length_and_offsets(self) -> None:
        value = "A \\% visible % hidden\nB\n"
        masked = mask_tex_comments(value)
        self.assertEqual(len(masked), len(value))
        self.assertIn(r"\% visible", masked)
        self.assertNotIn("hidden", masked)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "main.tex"
            path.write_text(
                r"""\newtheorem{definition}{Definition}
\begin{definition}\label{def:x}x\end{definition}
""",
                encoding="utf-8",
            )
            ir = build_theorem_ir([path], {"def:x": "4"})
            self.assertEqual(len(ir.blocks), 1)
            self.assertEqual(ir.blocks[0].definition.source_file, path)

    def test_static_visible_text_rejects_unknown_macro(self) -> None:
        rendered = render_static_visible_text(r"A \emph{finite} $\alpha$ case")
        self.assertEqual(rendered.markdown, r"A *finite* $\alpha$ case")
        with self.assertRaisesRegex(StructuralIRError, "unknown_macro"):
            render_static_visible_text(r"A \unknown{title}")


if __name__ == "__main__":
    unittest.main()
