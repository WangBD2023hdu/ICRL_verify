from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import re
import sys
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "arxiv_inline_markup.py"
SPEC = importlib.util.spec_from_file_location("arxiv_inline_markup", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ArxivInlineMarkupTest(unittest.TestCase):
    def render(self, latex: str, pdf: str, **kwargs: object) -> str:
        plan = MODULE.parse_inline_plan(latex)
        result = MODULE.apply_inline_plan(plan, pdf, **kwargs)
        self.assertIsNotNone(result)
        assert result is not None
        return result.markdown

    def test_finditer_is_bounded_while_materializing_pathological_regex(self) -> None:
        compiled = re.compile(r"^(a+)+$")
        regex = MODULE.InlineRegex(
            pattern=compiled.pattern,
            compiled=compiled,
            group_names={},
            max_wildcard=0,
        )
        previous = MODULE.INLINE_REGEX_TIMEOUT_SECONDS
        MODULE.INLINE_REGEX_TIMEOUT_SECONDS = 0.01
        started = time.monotonic()
        try:
            matches = regex.finditer("a" * 20_000 + "b")
        finally:
            MODULE.INLINE_REGEX_TIMEOUT_SECONDS = previous
        self.assertEqual(matches, ())
        self.assertLess(time.monotonic() - started, 1.0)

    def test_nested_strong_and_emphasis(self) -> None:
        latex = r"The \textbf{very \emph{important}} result."
        plan = MODULE.parse_inline_plan(latex)
        self.assertEqual(plan.feature_counts["strong"], 1)
        self.assertEqual(plan.feature_counts["em"], 1)
        self.assertEqual(
            self.render(latex, "The very important result."),
            "The **very *important*** result.",
        )

    def test_citation_and_reference_keep_pdf_text(self) -> None:
        latex = r"Prior work \citep{smith2020} gives Eq.~\eqref{eq:x}."
        plan = MODULE.parse_inline_plan(latex)
        result = MODULE.apply_inline_plan(plan, "Prior work [12] gives Eq. (3).")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.markdown, "Prior work [12] gives Eq. (3).")
        self.assertEqual(plan.feature_counts["citation"], 1)
        self.assertEqual(plan.feature_counts["reference"], 1)

    def test_footnote_body_is_never_inlined(self) -> None:
        latex = r"Claim\footnote{Secret \textbf{note}.} holds."
        rendered = self.render(latex, "Claim1 holds.")
        self.assertEqual(rendered, "Claim<sup>1</sup> holds.")
        self.assertNotIn("Secret", rendered)
        plan = MODULE.parse_inline_plan(latex)
        self.assertNotIn("strong", plan.feature_counts)
        self.assertEqual(plan.feature_counts["footnote"], 1)

    def test_footnote_callout_only_converts_ascii_digits(self) -> None:
        latex = r"Claim\footnote{Body} holds."
        cases = [
            ("Claim1 holds.", "Claim<sup>1</sup> holds."),
            ("Claim12 holds.", "Claim<sup>12</sup> holds."),
            ("Claim† holds.", "Claim† holds."),
            ("Claim¹ holds.", "Claim¹ holds."),
            ("Claim holds.", "Claim holds."),
        ]
        for pdf, expected in cases:
            with self.subTest(pdf=pdf):
                self.assertEqual(self.render(latex, pdf), expected)

    def test_footnote_body_numbers_are_not_callouts(self) -> None:
        self.assertEqual(
            MODULE.render_footnote_body(
                r"The \textbf{12} retained cases.",
                "The 12 retained cases.",
            ),
            "The **12** retained cases.",
        )

    def test_extract_footnote_source_balances_nested_body(self) -> None:
        plan = MODULE.parse_inline_plan(
            r"Text\footnote[7]{Outer {nested \textbf{body}} and $x$.} after."
        )
        node = next(
            item
            for item in MODULE.iter_inline_nodes(plan.root)
            if item.opaque_role == "footnote"
        )
        source = MODULE.extract_footnote_source(node)
        self.assertIsNotNone(source)
        assert source is not None
        self.assertEqual(source.optional_arguments, ("7",))
        self.assertEqual(source.body_raw, r"Outer {nested \textbf{body}} and $x$.")
        self.assertEqual(MODULE.extract_footnote_body(node), source.body_raw)

    def test_extract_footnote_source_fails_closed(self) -> None:
        plan = MODULE.parse_inline_plan(r"X\footnote{safe}Y")
        node = next(
            item
            for item in MODULE.iter_inline_nodes(plan.root)
            if item.opaque_role == "footnote"
        )
        unsafe_raw_values = [
            r"\footnote{unclosed",
            r"\footnote{one}{two}",
            r"\footnoteevil{body}",
            r"\footnote[broken{body}",
        ]
        for raw in unsafe_raw_values:
            with self.subTest(raw=raw):
                unsafe = dataclasses.replace(node, raw=raw)
                self.assertIsNone(MODULE.extract_footnote_source(unsafe))
        cite = next(
            item
            for item in MODULE.iter_inline_nodes(
                MODULE.parse_inline_plan(r"See \cite{key}.").root
            )
            if item.kind == "opaque"
        )
        self.assertIsNone(MODULE.extract_footnote_source(cite))

    def test_render_footnote_body_restores_inline_source_features(self) -> None:
        fixtures = [
            (
                r"The \textbf{value} is $\alpha$.",
                "The value is α.",
                r"The **value** is $\alpha$.",
            ),
            (
                r"An \emph{open} and \textit{closed} set.",
                "An open and closed set.",
                "An *open* and *closed* set.",
            ),
            (
                r"The {\em floor function} maps $x$.",
                "The floor function maps x.",
                r"The *floor function* maps $x$.",
            ),
        ]
        for body_raw, pdf_body, expected in fixtures:
            with self.subTest(body_raw=body_raw):
                self.assertEqual(
                    MODULE.render_footnote_body(body_raw, pdf_body), expected
                )

    def test_render_footnote_body_keeps_pdf_cite_and_ref_expansions(self) -> None:
        raw = (
            r"See \cite{secret-cite}, Ref.~\ref{secret-ref}, "
            r"and Eq.~\eqref{secret-eq}."
        )
        pdf = "See [7], Ref. 2, and Eq. (4)."
        rendered = MODULE.render_footnote_body(raw, pdf)
        self.assertEqual(rendered, pdf)
        assert rendered is not None
        self.assertNotIn("secret", rendered)

    def test_render_footnote_body_returns_none_on_unsafe_or_mismatch(self) -> None:
        cases = [
            (r"Broken \textbf{body", "Broken body"),
            (r"Broken $x", "Broken x"),
            (r"The {\em floor function", "The floor function"),
            (r"Expected body.", "Different body."),
        ]
        for body_raw, pdf in cases:
            with self.subTest(body_raw=body_raw):
                self.assertIsNone(MODULE.render_footnote_body(body_raw, pdf))

    def test_both_math_delimiters_restore_source_latex(self) -> None:
        latex = r"Minimize \(L(\theta)\) with $\alpha_i^2$ and \texttt{Adam}."
        pdf = "Minimize L(θ) with αᵢ² and Adam."
        self.assertEqual(
            self.render(latex, pdf),
            r"Minimize $L(\theta)$ with $\alpha_i^2$ and `Adam`.",
        )

    def test_math_keeps_trailing_tex_control_space(self) -> None:
        latex = r"Dots $\ldots\ $ and \(\cdots\ \)."
        self.assertEqual(
            self.render(latex, "Dots … and ⋯."),
            r"Dots $\ldots\ $ and $\cdots\ $.",
        )

    def test_math_drops_only_layout_whitespace(self) -> None:
        latex = r"""Value $
    x + y
  $ and \(
  z^2
\)."""
        self.assertEqual(
            self.render(latex, "Value (x + y) and z²."),
            "Value $x + y$ and $z^2$.",
        )

    def test_escaped_characters_and_verb(self) -> None:
        latex = r"Use \& and \_ plus \verb|a_b|."
        self.assertEqual(
            self.render(latex, "Use & and _ plus a_b."),
            "Use & and _ plus `a_b`.",
        )
        plan = MODULE.parse_inline_plan(latex)
        self.assertEqual(plan.feature_counts["escape"], 2)
        self.assertEqual(plan.feature_counts["code"], 1)

    def test_unknown_command_is_opaque_and_keeps_pdf(self) -> None:
        latex = r"Use \custom{alpha {beta}} now."
        plan = MODULE.parse_inline_plan(latex)
        self.assertEqual(plan.feature_counts["unknown_command"], 1)
        self.assertEqual(self.render(latex, "Use ALPHA-BETA now."), "Use ALPHA-BETA now.")

    def test_discretionary_pdf_hyphenation_restores_continuous_source_word(self) -> None:
        latex = r"The \textit{ORGANIZATION} label is retained."
        self.assertEqual(
            self.render(latex, "The ORGA- NIZATION label is retained."),
            "The *ORGANIZATION* label is retained.",
        )

    def test_multiple_discretionary_breaks_in_separate_words_are_removed(self) -> None:
        latex = r"The \textbf{ORGANIZATION} records the LOCATION."
        self.assertEqual(
            self.render(latex, "The ORGA- NIZATION records the LO- CATION."),
            "The **ORGANIZATION** records the LOCATION.",
        )

    def test_multiword_and_true_source_hyphens_are_not_removed(self) -> None:
        latex = r"The \emph{DATA ORGANIZATION} uses a well-known LOCATION."
        self.assertEqual(
            self.render(
                latex,
                "The DATA ORGA- NIZATION uses a well-known LO- CATION.",
            ),
            "The *DATA ORGANIZATION* uses a well-known LOCATION.",
        )
        # No whitespace means this is PDF-visible punctuation, not the narrow
        # discretionary line-break form repaired above.
        self.assertEqual(
            self.render("The ORGANIZATION.", "The ORGA-NIZATION."),
            "The ORGA-NIZATION.",
        )

    def test_formula_adjacent_missing_hyphen_is_restored(self) -> None:
        latex = r"The $k$-bonacci sequence."
        self.assertEqual(
            self.render(latex, "The kbonacci sequence."),
            r"The $k$-bonacci sequence.",
        )

    def test_formula_adjacent_pdf_hyphen_is_preserved(self) -> None:
        latex = r"The $k$-bonacci sequence."
        for pdf_hyphen in ("-", "‐"):
            with self.subTest(pdf_hyphen=pdf_hyphen):
                self.assertEqual(
                    self.render(latex, f"The k{pdf_hyphen}bonacci sequence."),
                    rf"The $k${pdf_hyphen}bonacci sequence.",
                )

    def test_plain_literal_hyphen_is_preserved_or_repaired(self) -> None:
        self.assertEqual(
            self.render("A well-known fact.", "A well-known fact."),
            "A well-known fact.",
        )
        self.assertEqual(
            self.render("A well-known fact.", "A wellknown fact."),
            "A well-known fact.",
        )

    def test_hyphen_repair_does_not_invent_other_punctuation_or_leak_opaque(self) -> None:
        punctuation = MODULE.parse_inline_plan(r"The $k$-bonacci, sequence.")
        self.assertIsNone(
            MODULE.apply_inline_plan(punctuation, "The kbonacci sequence.")
        )
        opaque = r"See \custom{hidden-hyphen} after $k$-bonacci."
        self.assertEqual(
            self.render(opaque, "See TOKEN after kbonacci."),
            r"See TOKEN after $k$-bonacci.",
        )

    def test_regex_has_named_and_bounded_wildcards(self) -> None:
        plan = MODULE.parse_inline_plan(r"A $x$ and \cite{k}.")
        regex = MODULE.build_inline_regex(plan, max_wildcard=32)
        self.assertIn("?P<inline_", regex.pattern)
        self.assertIn(r"[\s\S]{0,32}?", regex.pattern)
        self.assertNotIn(".*", regex.pattern)
        self.assertNotIn(".+", regex.pattern)

    def test_label_is_zero_width_and_search_returns_span(self) -> None:
        plan = MODULE.parse_inline_plan(r"A\label{a} result")
        regex = MODULE.build_inline_regex(plan, max_wildcard=20)
        self.assertIn(r"[\s\S]{0,0}?", regex.pattern)
        result = MODULE.apply_inline_plan(
            plan, "prefix A result suffix", max_wildcard=20, fullmatch=False
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.markdown, "A result")
        self.assertEqual(result.matched_text, "A result")
        self.assertEqual(result.span, (7, 15))

    def test_summary_reports_features_and_anchors(self) -> None:
        plan = MODULE.parse_inline_plan(r"The \textbf{loss} $L$ is used.")
        summary = MODULE.summarize_inline_plan(plan)
        self.assertEqual(summary["wildcards"], 1)
        self.assertEqual(summary["features"]["strong"], 1)
        self.assertGreaterEqual(summary["anchors"]["count"], 3)
        self.assertGreater(summary["anchors"]["characters"], 0)

    def test_focus_plan_drops_distant_sentence_tail_but_keeps_targets(self) -> None:
        full = MODULE.parse_inline_plan(
            r"A copy of $\sigma(\theta)$ uses $\theta$, refining the estimator "
            r"towards the optimal \cite{a,b}."
        )
        focused = MODULE.focus_inline_plan(full, context_characters=120)
        self.assertEqual(focused.feature_counts["math"], 2)
        self.assertNotIn("citation", focused.feature_counts)
        result = MODULE.apply_inline_plan(
            focused,
            "A copy of sigma(theta) uses theta, refining the estimator towards the optimal [3].",
            fullmatch=False,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.markdown.rstrip(),
            r"A copy of $\sigma(\theta)$ uses $\theta$, refining the estimator towards the optimal",
        )

    def test_source_render_requires_no_opaque_commands(self) -> None:
        plan = MODULE.focus_inline_plan(
            MODULE.parse_inline_plan(r"The \textbf{loss} is $L(\theta)$.")
        )
        self.assertEqual(
            MODULE.render_inline_source(plan),
            r"The **loss** is $L(\theta)$.",
        )
        opaque = MODULE.parse_inline_plan(r"See \cite{hidden} and $x$.")
        with self.assertRaises(ValueError):
            MODULE.render_inline_source(opaque)

    def test_malformed_inputs_fail_closed(self) -> None:
        broken = [
            r"A $x",
            r"A \(x",
            r"A \textbf{oops",
            r"A \textit without braces",
            r"A \verb|oops",
            r"A \cite",
            r"A \cite[see]",
            "dangling \\",
            "unexpected }",
        ]
        for value in broken:
            with self.subTest(value=value):
                with self.assertRaises(MODULE.InlineParseError):
                    MODULE.parse_inline_plan(value)

    def test_alignment_failure_returns_none(self) -> None:
        plan = MODULE.parse_inline_plan("Alpha beta")
        self.assertIsNone(MODULE.apply_inline_plan(plan, "unrelated text"))


if __name__ == "__main__":
    unittest.main()
