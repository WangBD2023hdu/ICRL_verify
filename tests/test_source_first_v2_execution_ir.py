from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arxiv_source_first_v2.execution_ir import (
    AmbiguousExecutionError,
    IncludeCycleError,
    SourceUnitRef,
    build_execution_ir,
    build_execution_ir_from_fls,
    parse_fls_executed_sources,
)


class ExecutionIrTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path.resolve()

    def build(self, main: Path, *sources: Path):
        return build_execution_ir(main, fls_sources=(main, *sources))

    def test_nested_input_and_include_follow_tex_execution_order(self) -> None:
        main = self.write("main.tex", "main-before\n\\input{parts/one}\nmain-after\n")
        one = self.write(
            "parts/one.tex",
            "one-before\n\\include{nested/two.tex}\none-after\n",
        )
        # Resolution is relative to the file containing the include, hence
        # ``nested`` lives below ``parts``.
        two = self.write("parts/nested/two.tex", "two-body\n")
        ir = self.build(main, one, two)

        positions = [
            ir.resolve(main, line=1).require_unique(),
            ir.resolve(one, line=1).require_unique(),
            ir.resolve(two, line=1).require_unique(),
            ir.resolve(one, line=3).require_unique(),
            ir.resolve(main, line=3).require_unique(),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(ir.executed_sources, (main, one, two))

    def test_comments_do_not_create_include_edges(self) -> None:
        main = self.write(
            "main.tex",
            "% \\input{ghost}\n"
            "visible % \\include{also-ghost}\n"
            "escaped \\% stays visible \\input{real}\n",
        )
        real = self.write("real.tex", "real body\n")
        ir = self.build(main, real)

        self.assertEqual(ir.executed_sources, (main, real))
        self.assertFalse(any("ghost" in item.message for item in ir.diagnostics))

    def test_fls_is_an_allowlist_for_literal_source_edges(self) -> None:
        main = self.write(
            "main.tex",
            "before\n\\input{not-executed}\n\\input{executed.tex}\nafter\n",
        )
        skipped = self.write("not-executed.tex", "skip me\n")
        executed = self.write("executed.tex", "keep me\n")
        ir = self.build(main, executed)

        self.assertEqual(ir.executed_sources, (main, executed))
        self.assertEqual(ir.resolve(skipped, line=1).status, "not_executed")
        self.assertIn(
            "include_not_in_fls_allowlist",
            [item.code for item in ir.diagnostics],
        )

    def test_repeated_include_is_explicitly_ambiguous_and_sort_fails_closed(self) -> None:
        main = self.write(
            "main.tex",
            "before\n\\input{shared}\nmiddle\n\\input{shared.tex}\nafter\n",
        )
        shared = self.write("shared.tex", "shared body\n")
        ir = self.build(main, shared)

        resolution = ir.resolve(shared, line=1)
        self.assertEqual(resolution.status, "ambiguous")
        self.assertEqual(len(resolution.ordinals), 2)
        self.assertEqual(
            len({ordinal.occurrence_id for ordinal in resolution.ordinals}),
            2,
        )
        self.assertIsNone(resolution.ordinal)
        with self.assertRaises(AmbiguousExecutionError):
            resolution.require_unique()
        with self.assertRaises(AmbiguousExecutionError):
            ir.order_source_units(
                [SourceUnitRef("paragraph", shared, line=1)]
            )
        self.assertIn(
            "repeated_source_execution",
            [item.code for item in ir.diagnostics],
        )

    def test_cross_type_units_share_one_execution_order(self) -> None:
        main = self.write(
            "main.tex",
            "front matter\nheading\n\\input{table}\nparagraph\n",
        )
        table = self.write("table.tex", "table body\n")
        ir = self.build(main, table)
        units = [
            SourceUnitRef("paragraph", main, line=4, payload="p"),
            SourceUnitRef("table", table, line=1, payload="t"),
            SourceUnitRef("frontmatter", main, line=1, payload="f"),
            SourceUnitRef("heading", main, line=2, payload="h"),
        ]

        ordered = ir.order_source_units(units)
        self.assertEqual([item.unit.kind for item in ordered], [
            "frontmatter",
            "heading",
            "table",
            "paragraph",
        ])
        self.assertEqual([item.unit.payload for item in ordered], ["f", "h", "t", "p"])

    def test_cycle_is_detected(self) -> None:
        main = self.write("main.tex", "\\input{a}\n")
        a = self.write("a.tex", "\\include{main}\n")
        with self.assertRaises(IncludeCycleError):
            self.build(main, a)

    def test_fls_parser_uses_pwd_and_wrapper_builds_ir(self) -> None:
        main = self.write("main.tex", "\\input{child}\n")
        child = self.write("child.tex", "child\n")
        fls = self.write(
            "build/main.fls",
            f"PWD {self.root}\nINPUT main.tex\nINPUT child.tex\nINPUT child.tex\n",
        )

        self.assertEqual(parse_fls_executed_sources(fls), (main, child))
        ir = build_execution_ir_from_fls(main, fls)
        self.assertEqual(ir.executed_sources, (main, child))

    def test_byte_offsets_are_bytes_not_unicode_codepoints(self) -> None:
        main = self.write("main.tex", "é heading\nparagraph\n")
        ir = self.build(main)
        by_line = ir.resolve(main, line=2).require_unique()
        byte_offset = len("é heading\n".encode("utf-8"))
        by_byte = ir.resolve(main, byte_offset=byte_offset).require_unique()
        self.assertEqual(by_line, by_byte)


if __name__ == "__main__":
    unittest.main()
