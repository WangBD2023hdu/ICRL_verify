from __future__ import annotations

import unicodedata
import unittest

from arxiv_source_first_v2.verifier_projection import (
    MATH_VISIBLE_STYLE_WRAPPERS,
    FolioProjectionPolicy,
    project_bottom_margin_folio,
    project_fenced_code_dollar_guards,
    project_math_visible_flow,
)


def word(text: str, x0: float, top: float, x1: float, bottom: float) -> dict[str, object]:
    return {"text": text, "x0": x0, "top": top, "x1": x1, "bottom": bottom}


class WordPage:
    def __init__(self, words: list[dict[str, object]], *, width: float = 600, height: float = 800):
        self.width = width
        self.height = height
        self._words = words

    def extract_words(self, **_kwargs: object) -> list[dict[str, object]]:
        return list(self._words)


class CharPage:
    def __init__(self, chars: list[dict[str, object]], *, width: float = 600, height: float = 800):
        self.width = width
        self.height = height
        self.chars = chars


class VerifierProjectionTest(unittest.TestCase):
    def page(self, *extra: dict[str, object]) -> WordPage:
        return WordPage(
            [
                word("Source", 50, 80, 100, 92),
                word("truth", 106, 80, 144, 92),
                *extra,
            ]
        )

    def test_unique_bottom_centered_numeric_folio_is_removed_pdf_side_only(self) -> None:
        markdown = r"Source **truth** with $T \in \{1\}$."
        result = project_bottom_margin_folio(
            markdown,
            "Source truth with T ∈ {1}. 7",
            page=self.page(word("with", 150, 80, 175, 92), word("7", 297, 760, 303, 772)),
        )

        self.assertTrue(result.projection_applied)
        self.assertEqual(result.source_markdown, markdown)
        self.assertEqual(result.verifier_text, "Source truth with T ∈ {1}.")
        self.assertEqual(result.verifier_stream, "SourcetruthwithT∈{1}.")
        self.assertFalse(result.provenance["ground_truth_changed"])
        self.assertFalse(result.provenance["pdf_text_used_for_ground_truth"])
        self.assertEqual(result.provenance["removed_folio"]["bbox"], [297.0, 760.0, 303.0, 772.0])

    def test_multiple_bottom_numbers_fail_closed(self) -> None:
        result = project_bottom_margin_folio(
            "Body",
            "Body 7 8",
            page=self.page(word("7", 297, 750, 303, 762), word("8", 297, 770, 303, 782)),
        )
        self.assertFalse(result.projection_applied)
        self.assertEqual(result.verifier_text, "Body 7 8")
        self.assertEqual(result.provenance["reason"], "multiple_bottom_folio_candidates")

    def test_non_centered_bottom_number_fails_closed(self) -> None:
        result = project_bottom_margin_folio(
            "Body",
            "Body 7",
            page=self.page(word("7", 540, 760, 546, 772)),
        )
        self.assertFalse(result.projection_applied)
        self.assertEqual(result.provenance["reason"], "bottom_folio_not_horizontally_centered")

    def test_folio_may_be_centered_on_a_two_sided_body_text_block(self) -> None:
        page = WordPage(
            [
                word("A", 125, 80, 145, 92),
                word("wide", 150, 80, 190, 92),
                word("body", 200, 80, 240, 92),
                word("line", 450, 80, 495, 92),
                # Body envelope centre is 310; physical page centre is 300.
                word("2", 307, 760, 313, 772),
            ]
        )
        result = project_bottom_margin_folio("A wide body line", "A wide body line 2", page=page)
        self.assertTrue(result.projection_applied)
        self.assertEqual(
            result.provenance["geometry"]["horizontal_center_reference"],
            "body_text_envelope",
        )

    def test_centered_number_in_body_region_is_never_removed(self) -> None:
        result = project_bottom_margin_folio(
            "Equation 7",
            "Equation 7",
            page=self.page(word("7", 297, 400, 303, 412)),
        )
        self.assertFalse(result.projection_applied)
        self.assertEqual(result.verifier_text, "Equation 7")
        self.assertEqual(result.provenance["reason"], "folio_like_text_in_body_region")

    def test_page_label_or_other_margin_text_is_not_a_pure_isolated_folio(self) -> None:
        result = project_bottom_margin_folio(
            "Body",
            "Body Page 7",
            page=self.page(
                word("Page", 260, 760, 292, 772),
                word("7", 297, 760, 303, 772),
            ),
        )
        self.assertFalse(result.projection_applied)
        self.assertEqual(result.provenance["reason"], "bottom_folio_not_isolated_on_line")

    def test_geometry_candidate_must_be_terminal_in_existing_verifier_order(self) -> None:
        result = project_bottom_margin_folio(
            "Body suffix",
            "Body 7 suffix",
            page=self.page(word("7", 297, 760, 303, 772)),
        )
        self.assertFalse(result.projection_applied)
        self.assertEqual(result.provenance["reason"], "bottom_folio_not_terminal_in_verifier_text")

    def test_roman_folio_is_opt_in_and_canonical(self) -> None:
        page = self.page(word("iv", 294, 760, 306, 772))
        disabled = project_bottom_margin_folio("Body", "Body iv", page=page)
        enabled = project_bottom_margin_folio(
            "Body",
            "Body iv",
            page=page,
            policy=FolioProjectionPolicy(allow_roman=True),
        )
        invalid = project_bottom_margin_folio(
            "Body",
            "Body iiv",
            page=self.page(word("iiv", 292, 760, 308, 772)),
            policy=FolioProjectionPolicy(allow_roman=True),
        )

        self.assertFalse(disabled.projection_applied)
        self.assertTrue(enabled.projection_applied)
        self.assertEqual(enabled.provenance["removed_folio"]["kind"], "roman")
        self.assertFalse(invalid.projection_applied)

    def test_char_box_fixture_is_supported_without_extract_words(self) -> None:
        chars = [
            word("B", 50, 80, 57, 92),
            word("o", 57, 80, 63, 92),
            word("d", 63, 80, 69, 92),
            word("y", 69, 80, 75, 92),
            word("8", 297, 760, 303, 772),
        ]
        result = project_bottom_margin_folio("Body", "Body 8", page=CharPage(chars))
        self.assertTrue(result.projection_applied)
        self.assertEqual(result.verifier_text, "Body")

    def test_custom_stream_projection_keeps_existing_hyphen_policy_composable(self) -> None:
        sentinel = "\uE001"
        markdown = r"Grid $T \in \{1\}$ inter-national."
        observed = f"Grid T ∈ {{1}} inter{sentinel}national. 9"

        result = project_bottom_margin_folio(
            markdown,
            observed,
            page=self.page(word("9", 297, 760, 303, 772)),
            stream_projector=lambda value: value.replace(" ", ""),
        )

        self.assertTrue(result.projection_applied)
        self.assertIn(sentinel, result.verifier_stream)
        self.assertEqual(result.source_markdown, markdown)
        self.assertIn(r"\{1\}", result.source_markdown)

    def test_explicit_layout_words_interface_does_not_require_pdfplumber_page(self) -> None:
        result = project_bottom_margin_folio(
            "Body",
            "Body 12",
            layout_words=[word("Body", 50, 80, 80, 92), word("12", 294, 760, 306, 772)],
            page_width=600,
            page_height=800,
        )
        self.assertTrue(result.projection_applied)
        self.assertEqual(result.verifier_text, "Body")


class MathVisibleFlowProjectionTest(unittest.TestCase):
    def test_textmd_and_mathcal_are_transparent_inside_math_only(self) -> None:
        markdown = r"Name $\textmd{DA-CR}$ and set $\mathcal{X}$."
        result = project_math_visible_flow(markdown)

        self.assertTrue(result.projection_applied)
        self.assertEqual(result.source_markdown, markdown)
        self.assertEqual(result.projected_markdown, r"Name $DA－CR$ and set $X$.")
        self.assertEqual(
            unicodedata.normalize("NFKC", result.projected_markdown),
            r"Name $DA-CR$ and set $X$.",
        )
        self.assertFalse(result.provenance["ground_truth_changed"])
        self.assertFalse(result.provenance["pdf_text_used_for_ground_truth"])
        self.assertEqual(result.provenance["style_wrappers_removed"], 2)
        self.assertEqual(result.provenance["text_mode_hyphens_guarded"], 1)
        self.assertEqual(
            result.provenance["style_wrapper_counts"],
            {"mathcal": 1, "textmd": 1},
        )

    def test_every_allowlisted_style_wrapper_has_one_visible_argument(self) -> None:
        for command in sorted(MATH_VISIBLE_STYLE_WRAPPERS):
            with self.subTest(command=command):
                markdown = rf"$\{command}{{Visible}}$"
                result = project_math_visible_flow(markdown)
                self.assertEqual(result.status, "projected")
                self.assertEqual(result.source_markdown, markdown)
                self.assertEqual(result.projected_markdown, "$Visible$")
                self.assertEqual(
                    result.provenance["style_wrapper_counts"], {command: 1}
                )

    def test_wrappers_project_in_all_four_supported_math_delimiters(self) -> None:
        markdown = (
            r"$\mathbf{a}$ $$\mathit{b}$$ "
            r"\(\textbf{c}\) \[\mathbb{d}\]"
        )
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.projected_markdown, r"$a$ $$b$$ \(c\) \[d\]")
        self.assertEqual(result.provenance["math_regions_seen"], 4)
        self.assertEqual(
            result.provenance["math_delimiter_counts"],
            {"$": 1, "$$": 1, r"\(": 1, r"\[": 1},
        )

    def test_escaped_math_delimiters_are_literal_and_backslash_parity_is_exact(self) -> None:
        markdown = (
            r"Price \$5; literal \\(not math \\); real \(\mathbf{x}\); "
            r"even slash parity \\$\mathcal{Y}$."
        )
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "projected")
        self.assertEqual(
            result.projected_markdown,
            (
                r"Price \$5; literal \\(not math \\); real \(x\); "
                r"even slash parity \\$Y$."
            ),
        )
        self.assertEqual(result.provenance["math_regions_seen"], 2)

    def test_escaped_dollar_and_braces_inside_math_do_not_close_groups(self) -> None:
        markdown = r"$\textmd{cost=\$5, set=\{A\}}$"
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "projected")
        self.assertEqual(result.projected_markdown, r"$cost=\$5, set=\{A\}$")
        self.assertEqual(result.provenance["style_wrappers_removed"], 1)

    def test_style_commands_outside_math_are_byte_unchanged(self) -> None:
        markdown = r"Outside \textbf{bold} and \mathcal{X}."
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "unchanged")
        self.assertFalse(result.projection_applied)
        self.assertEqual(result.source_markdown, markdown)
        self.assertEqual(result.projected_markdown, markdown)
        self.assertEqual(result.provenance["math_regions_seen"], 0)

    def test_text_mode_escaped_underscore_survives_math_canonicalization(self) -> None:
        markdown = r"$\texttt{file\_name}$ versus $\mathtt{x_y}$"
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.projected_markdown, r"$file＿name$ versus $x_y$")
        self.assertEqual(result.provenance["text_mode_underscores_guarded"], 1)
        self.assertEqual(
            unicodedata.normalize("NFKC", result.projected_markdown),
            r"$file_name$ versus $x_y$",
        )

    def test_nested_wrappers_and_balanced_ordinary_groups_are_preserved(self) -> None:
        markdown = r"$q^{\textbf{A_{\mathfrak{B}}}}+\frac{\mathbf{x}}{y}$"
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "projected")
        self.assertEqual(result.projected_markdown, r"$q^{A_{B}}+\frac{x}{y}$")
        self.assertEqual(result.provenance["style_wrappers_removed"], 3)

    def test_layout_environment_commands_tabs_and_rows_are_layout_only(self) -> None:
        markdown = (
            r"\[\begin{alignedat}[t]{2}"
            r"\mathbf{x}&=1\\[2pt]y&=\mathcal{Z}"
            r"\end{alignedat}\]"
        )
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "projected")
        self.assertEqual(result.source_markdown, markdown)
        self.assertEqual(result.projected_markdown, r"\[x=1y=Z\]")
        self.assertEqual(result.provenance["layout_environments_removed"], 1)
        self.assertEqual(
            result.provenance["layout_environment_counts"], {"alignedat": 1}
        )
        self.assertEqual(result.provenance["layout_alignment_tabs_removed"], 2)
        self.assertEqual(result.provenance["layout_row_breaks_removed"], 1)

    def test_nested_supported_layout_environments_must_pair_exactly(self) -> None:
        markdown = (
            r"$\begin{cases}a&\begin{split}b\\c\end{split}"
            r"\end{cases}$"
        )
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.projected_markdown, "$abc$")
        self.assertEqual(result.provenance["layout_environments_removed"], 2)

    def test_missing_wrapper_argument_rejects_and_rolls_back_all_edits(self) -> None:
        markdown = r"$\mathbf{ok}+\mathcal X$"
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.projection_applied)
        self.assertEqual(result.source_markdown, markdown)
        self.assertEqual(result.projected_markdown, markdown)
        self.assertEqual(
            result.provenance["reason"],
            "style_wrapper_missing_braced_argument:mathcal",
        )

    def test_unbalanced_brace_and_math_delimiter_fail_closed(self) -> None:
        for markdown, reason in (
            (r"$\mathbf{X$", "unbalanced_math_brace"),
            (r"$\mathbf{X}", "unclosed_math_delimiter"),
            (r"\) stray", "math_closing_delimiter_without_opener"),
            (r"\(x\]\)", "mismatched_math_delimiter"),
            (r"$$x$y$$", "nested_math_delimiter"),
        ):
            with self.subTest(markdown=markdown):
                result = project_math_visible_flow(markdown)
                self.assertEqual(result.status, "rejected")
                self.assertEqual(result.projected_markdown, markdown)
                self.assertEqual(result.provenance["reason"], reason)

    def test_unknown_or_mismatched_layout_environment_fails_closed(self) -> None:
        for markdown, reason in (
            (
                r"$\begin{matrix}x\end{matrix}$",
                "unsupported_math_layout_environment:matrix",
            ),
            (
                r"$\begin{aligned}x\end{cases}$",
                "mismatched_layout_environment:aligned!=cases",
            ),
            (
                r"$\begin{aligned}x$",
                "unclosed_math_layout_environment:aligned",
            ),
        ):
            with self.subTest(markdown=markdown):
                result = project_math_visible_flow(markdown)
                self.assertEqual(result.status, "rejected")
                self.assertEqual(result.projected_markdown, markdown)
                self.assertEqual(result.provenance["reason"], reason)

    def test_unknown_nontransforming_math_command_is_retained_not_swallowed(self) -> None:
        markdown = r"$\frac{\mathbf{x}}{y}+\unknown{z}$"
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "projected")
        self.assertEqual(result.projected_markdown, r"$\frac{x}{y}+\unknown{z}$")

    def test_removed_wrapper_preserves_adjacent_control_word_boundaries(self) -> None:
        for markdown, expected in (
            (
                r"$\unknown\textsf{heading}$",
                r"$\unknown{}heading$",
            ),
            (
                r"$\textsf{\unknown}heading$",
                r"$\unknown{}heading$",
            ),
        ):
            with self.subTest(markdown=markdown):
                result = project_math_visible_flow(markdown)
                self.assertEqual(result.status, "projected")
                self.assertEqual(result.source_markdown, markdown)
                self.assertEqual(result.projected_markdown, expected)
                self.assertEqual(
                    result.provenance["control_word_boundaries_inserted"], 1
                )
                self.assertFalse(result.provenance["ground_truth_changed"])
                self.assertFalse(
                    result.provenance["pdf_text_used_for_ground_truth"]
                )

    def test_unambiguous_operator_allowlist_is_static_and_token_exact(self) -> None:
        markdown = r"$a\land b\lor c\neg d+\landmark$"
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "projected")
        self.assertEqual(result.source_markdown, markdown)
        self.assertEqual(
            result.projected_markdown,
            r"$a∧ b∨ c¬ d+\landmark$",
        )
        self.assertEqual(result.provenance["operator_commands_projected"], 3)
        self.assertEqual(
            result.provenance["operator_command_counts"],
            {"land": 1, "lor": 1, "neg": 1},
        )
        # Prefix-like unknown control words are deliberately not guessed.
        self.assertIn(r"\landmark", result.projected_markdown)

    def test_llncs_math_heading_projection_keeps_gt_and_matches_printed_flow(self) -> None:
        markdown = (
            r"### B.2 $\textsf{F b} \land "
            r"\textsf{G }\neg\textsf{h}$"
        )
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "projected")
        self.assertEqual(result.source_markdown, markdown)
        self.assertEqual(result.projected_markdown, "### B.2 $F b ∧ G ¬h$")
        self.assertNotIn(r"\negh", result.projected_markdown)
        self.assertEqual(result.provenance["style_wrappers_removed"], 3)
        self.assertEqual(
            result.provenance["operator_command_counts"],
            {"land": 1, "neg": 1},
        )
        self.assertFalse(result.provenance["ground_truth_changed"])
        self.assertFalse(result.provenance["pdf_text_used_for_ground_truth"])


class FencedCodeDollarGuardTest(unittest.TestCase):
    def test_only_fenced_code_body_dollars_are_guarded_and_nfkc_restores_them(self) -> None:
        markdown = (
            "Before $x$.\n\n"
            "```text\n"
            "price=$i and $$ literally\n"
            "```\n\n"
            "After $y$."
        )
        result = project_fenced_code_dollar_guards(markdown)

        expected = markdown.replace(
            "price=$i and $$ literally", "price=＄i and ＄＄ literally"
        )
        self.assertEqual(result.status, "projected")
        self.assertEqual(result.source_markdown, markdown)
        self.assertEqual(result.projected_markdown, expected)
        self.assertEqual(result.provenance["fenced_code_blocks_seen"], 1)
        self.assertEqual(result.provenance["fenced_code_dollars_guarded"], 3)
        self.assertEqual(unicodedata.normalize("NFKC", expected), markdown)

    def test_fenced_code_guards_escaped_and_unescaped_dollars_equally(self) -> None:
        markdown = "```latex\n$raw \\$escaped $$pair\n```"
        result = project_fenced_code_dollar_guards(markdown)

        self.assertEqual(result.status, "projected")
        self.assertEqual(
            result.projected_markdown,
            "```latex\n＄raw \\＄escaped ＄＄pair\n```",
        )
        self.assertEqual(result.provenance["fenced_code_dollars_guarded"], 4)
        self.assertEqual(
            unicodedata.normalize("NFKC", result.projected_markdown), markdown
        )

    def test_combined_projection_skips_fence_and_projects_real_math(self) -> None:
        markdown = (
            "~~~latex\n$\\mathbf{not_math}$\n~~~\n"
            r"Real $\mathbf{x}$."
        )
        result = project_math_visible_flow(markdown)

        self.assertEqual(
            result.projected_markdown,
            "~~~latex\n＄\\mathbf{not_math}＄\n~~~\nReal $x$.",
        )
        self.assertEqual(result.provenance["fenced_code_dollars_guarded"], 2)
        self.assertEqual(result.provenance["style_wrappers_removed"], 1)

    def test_unpaired_fence_fails_closed_even_after_an_earlier_valid_block(self) -> None:
        markdown = "```\n$first\n```\n\n~~~\n$second\n"
        result = project_math_visible_flow(markdown)

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.projection_applied)
        self.assertEqual(result.source_markdown, markdown)
        self.assertEqual(result.projected_markdown, markdown)
        self.assertEqual(result.provenance["reason"], "unclosed_fenced_code_block")


if __name__ == "__main__":
    unittest.main()
