from __future__ import annotations

import unittest

from arxiv_source_first_v2.layout_graph import (
    GlyphComponent,
    LayoutConflictError,
    SourceFragment,
    build_layout_graph,
    fragments_from_glyph_components,
)


def fragment(
    name: str,
    ordinal: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    page: int = 1,
) -> SourceFragment:
    return SourceFragment(name, ordinal, (x0, y0, x1, y1), page_number=page)


class SourceFirstV2LayoutGraphTests(unittest.TestCase):
    def test_full_title_then_unequal_two_columns(self) -> None:
        result = build_layout_graph(
            [
                fragment("title", 0, 42, 20, 958, 64),
                fragment("left-1", 1, 42, 110, 326, 130),
                fragment("left-2", 2, 42, 142, 326, 162),
                fragment("right-1", 3, 474, 110, 958, 130),
                fragment("right-2", 4, 474, 142, 958, 162),
            ],
            page_width=1000,
            page_height=1400,
            strict=True,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.layout_kind, "two_column")
        self.assertEqual(
            result.ordered_fragment_ids,
            ("title", "left-1", "left-2", "right-1", "right-2"),
        )
        self.assertEqual([band.lane for band in result.bands], ["full", "columns"])

    def test_columns_spanning_block_columns_and_footer(self) -> None:
        result = build_layout_graph(
            [
                fragment("title", 0, 50, 20, 950, 60),
                fragment("left-before", 1, 40, 100, 330, 120),
                fragment("right-before", 2, 470, 100, 960, 120),
                fragment("left-before-2", 3, 40, 132, 330, 152),
                fragment("right-before-2", 4, 470, 132, 960, 152),
                fragment("spanning-block", 5, 24, 210, 976, 270),
                fragment("left-after", 6, 40, 320, 330, 340),
                fragment("right-after", 7, 470, 320, 960, 340),
                fragment("footer", 8, 120, 1280, 880, 1300),
            ],
            page_width=1000,
            page_height=1350,
            strict=True,
        )
        self.assertEqual(
            result.ordered_fragment_ids,
            (
                "title",
                "left-before",
                "left-before-2",
                "right-before",
                "right-before-2",
                "spanning-block",
                "left-after",
                "right-after",
                "footer",
            ),
        )
        self.assertEqual(
            [band.lane for band in result.bands],
            ["full", "columns", "full", "columns", "full"],
        )

    def test_left_column_end_precedes_right_column_top(self) -> None:
        result = build_layout_graph(
            [
                fragment("left-end", 10, 20, 500, 350, 530),
                fragment("right-top", 11, 520, 100, 980, 130),
                fragment("left-middle", 9, 20, 460, 350, 490),
                fragment("right-bottom", 12, 520, 540, 980, 570),
            ],
            page_width=1000,
            strict=True,
        )
        self.assertEqual(
            result.ordered_fragment_ids,
            ("left-middle", "left-end", "right-top", "right-bottom"),
        )

    def test_source_ordinal_breaks_near_baseline_tie(self) -> None:
        result = build_layout_graph(
            [
                fragment("left-source-2", 2, 20, 100.0, 350, 120),
                fragment("left-source-1", 1, 20, 100.5, 350, 120.5),
                fragment("right", 3, 520, 100, 980, 120),
            ],
            page_width=1000,
            strict=True,
        )
        self.assertEqual(result.ordered_fragment_ids[:2], ("left-source-1", "left-source-2"))

    def test_geometry_conflict_reports_and_raises(self) -> None:
        values = [
            fragment("left", 0, 20, 100, 350, 160),
            fragment("right", 1, 520, 100, 980, 160),
            fragment("left-2", 3, 20, 180, 350, 205),
            fragment("right-2", 4, 520, 180, 980, 205),
            fragment("spanning", 2, 20, 130, 980, 200),
        ]
        result = build_layout_graph(values, page_width=1000)
        self.assertFalse(result.accepted)
        self.assertIn("column_overlaps_full_block", {item.code for item in result.errors})
        with self.assertRaises(LayoutConflictError):
            build_layout_graph(values, page_width=1000, strict=True)

    def test_glyph_components_aggregate_without_text(self) -> None:
        fragments = fragments_from_glyph_components(
            [
                GlyphComponent("a-1", "a", 4, (20, 30, 30, 40)),
                GlyphComponent("a-2", "a", 4, (31, 30, 42, 40)),
                GlyphComponent("b-1", "b", 5, (520, 30, 530, 40)),
            ]
        )
        self.assertEqual(tuple(item.fragment_id for item in fragments), ("a", "b"))
        self.assertEqual(fragments[0].bbox.x0, 20)
        self.assertEqual(fragments[0].bbox.x1, 42)

    def test_mixed_pages_are_explicit_diagnostic(self) -> None:
        result = build_layout_graph(
            [
                fragment("page-one", 0, 0, 0, 100, 10),
                fragment("page-two", 1, 0, 0, 100, 10, page=2),
            ],
            page_width=100,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.diagnostics[0].code, "mixed_page_input")


if __name__ == "__main__":
    unittest.main()
