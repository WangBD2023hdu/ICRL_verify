from __future__ import annotations

import unittest

from arxiv_source_first_v2.list_ir import (
    ListIRSafetyError,
    ListSourceParagraph,
    assign_list_instance_ids,
    parse_latex_list,
    serialize_source_list,
)


def plain(value: str) -> str:
    return value.replace("\\textbf{", "").replace("}", "")


class SourceFirstV2ListIRTests(unittest.TestCase):
    def test_first_item_and_fixed_enumerate_ordinal(self) -> None:
        records = parse_latex_list(
            r"""\begin{enumerate}
\item First \textbf{entry}
\item Second
\end{enumerate}""",
            source_file="intro.tex",
        )
        result = serialize_source_list(records, render_inline=plain)
        self.assertEqual(result.markdown, "1. First entry\n2. Second")
        self.assertEqual([item.ordinal for item in result.items], [1, 2])
        self.assertEqual(result.items[0].marker, "1.")
        self.assertFalse(result.items[0].provenance[0]["pdf_text_used"])

    def test_same_item_continuation_never_gets_a_second_marker(self) -> None:
        records = (
            ListSourceParagraph(
                "p1", "itemize_item", "a.tex", (10,), r"\item first", "itemize", 1, 1, "L1"
            ),
            ListSourceParagraph(
                "p2", "itemize_item", "a.tex", (11,), "same item continues", "itemize", 1, 1, "L1"
            ),
            ListSourceParagraph(
                "p3", "itemize_item", "a.tex", (12,), r"\item next", "itemize", 1, 2, "L1"
            ),
        )
        result = serialize_source_list(records, render_inline=lambda value: value)
        self.assertEqual(result.markdown, "- first\n  same item continues\n- next")
        self.assertEqual(result.items[0].continuation_count, 1)
        self.assertEqual(result.items[0].paragraph_ids, ("p1", "p2"))

    def test_empty_description_label_fails_closed(self) -> None:
        with self.assertRaises(ListIRSafetyError):
            parse_latex_list(
                r"""\begin{description}
\item[Static label] explanation
\item[] invalid
\end{description}""",
                source_file="definitions.tex",
            )

    def test_description_label(self) -> None:
        records = parse_latex_list(
            r"""\begin{description}
\item[Static label] explanation
\end{description}""",
            source_file="definitions.tex",
        )
        result = serialize_source_list(records, render_inline=lambda value: value)
        self.assertEqual(result.markdown, "- **Static label** explanation")
        self.assertEqual(result.items[0].description_label, "Static label")
        self.assertEqual(result.items[0].source_lines, (2,))

    def test_description_label_preserves_source_colon_only(self) -> None:
        records = parse_latex_list(
            r"""\begin{description}
\item[Without colon] first
\item[With colon:] second
\end{description}""",
            source_file="definitions.tex",
        )
        result = serialize_source_list(records, render_inline=lambda value: value)
        self.assertEqual(
            result.markdown,
            "- **Without colon** first\n- **With colon:** second",
        )

    def test_missing_list_id_separates_two_ordinal_one_lists(self) -> None:
        records = (
            ListSourceParagraph("p1", "itemize_item", "a.tex", (1,), r"\item first", "itemize", 1, 1),
            ListSourceParagraph("p2", "itemize_item", "a.tex", (2,), "first continuation", "itemize", 1, 1),
            ListSourceParagraph("p3", "itemize_item", "a.tex", (3,), r"\item second", "itemize", 1, 2),
            # A literal new item at ordinal one is a reset/new list instance.
            ListSourceParagraph("p4", "itemize_item", "a.tex", (10,), r"\item new list", "itemize", 1, 1),
            ListSourceParagraph("p5", "itemize_item", "a.tex", (11,), "new list continuation", "itemize", 1, 1),
        )
        assigned = assign_list_instance_ids(records)
        self.assertEqual(
            [record.list_id for record in assigned],
            ["auto-list-000001", "auto-list-000001", "auto-list-000001", "auto-list-000002", "auto-list-000002"],
        )
        result = serialize_source_list(records, render_inline=lambda value: value)
        self.assertEqual(
            result.markdown,
            "- first\n  first continuation\n- second\n- new list\n  new list continuation",
        )

    def test_nested_depth_and_item_keys(self) -> None:
        records = parse_latex_list(
            r"""\begin{itemize}
\item outer
\begin{enumerate}
\item inner
\end{enumerate}
\item after
\end{itemize}""",
            source_file="nested.tex",
        )
        result = serialize_source_list(records, render_inline=lambda value: value)
        self.assertEqual(result.markdown, "- outer\n  1. inner\n- after")
        self.assertEqual([item.depth for item in result.items], [1, 2, 1])
        self.assertEqual(len({item.item_key for item in result.items}), 3)

    def test_parent_continuation_after_nested_child_keeps_execution_order(self) -> None:
        records = (
            ListSourceParagraph("outer", "itemize_item", "nested.tex", (1,), r"\item outer", "itemize", 1, 1),
            ListSourceParagraph("inner", "enumerate_item", "nested.tex", (3,), r"\item inner", "enumerate", 2, 1),
            ListSourceParagraph("outer-cont", "itemize_item", "nested.tex", (5,), "outer continuation", "itemize", 1, 1),
        )
        result = serialize_source_list(records, render_inline=lambda value: value)
        self.assertEqual(result.markdown, "- outer\n  1. inner\n  outer continuation")
        self.assertEqual(result.items[0].paragraph_ids, ("outer", "outer-cont"))

    def test_unbalanced_unknown_and_dynamic_constructs_fail_closed(self) -> None:
        with self.assertRaises(ListIRSafetyError):
            parse_latex_list(r"\begin{itemize}\item x", source_file="bad.tex")
        with self.assertRaises(ListIRSafetyError):
            parse_latex_list(r"\begin{mystery}\item x\end{mystery}", source_file="bad.tex")
        with self.assertRaises(ListIRSafetyError):
            parse_latex_list(r"\begin{\foo}\item x\end{itemize}", source_file="bad.tex")
        with self.assertRaises(ListIRSafetyError):
            parse_latex_list(r"\begin{itemize}\csname item\endcsname x\end{itemize}", source_file="bad.tex")

    def test_noncontiguous_continuation_and_skip_depth_are_rejected(self) -> None:
        records = (
            ListSourceParagraph("a", "itemize_item", "x.tex", (1,), r"\item a", "itemize", 1, 1, "L"),
            ListSourceParagraph("b", "itemize_item", "x.tex", (2,), r"\item b", "itemize", 1, 2, "L"),
            ListSourceParagraph("a2", "itemize_item", "x.tex", (3,), "a continued", "itemize", 1, 1, "L"),
        )
        with self.assertRaises(ListIRSafetyError):
            serialize_source_list(records, render_inline=lambda value: value)
        skipped = (
            ListSourceParagraph("a", "itemize_item", "x.tex", (1,), r"\item a", "itemize", 1, 1, "L"),
            ListSourceParagraph("b", "itemize_item", "x.tex", (2,), r"\item b", "itemize", 3, 1, "N"),
        )
        with self.assertRaises(ListIRSafetyError):
            serialize_source_list(skipped, render_inline=lambda value: value)


if __name__ == "__main__":
    unittest.main()
