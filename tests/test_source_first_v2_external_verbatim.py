from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arxiv_source_first_v2.external_verbatim import (
    ExternalVerbatimSafetyError,
    build_external_verbatim_ir,
    render_fenced_code,
)


class ExternalVerbatimIRTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.main = self.root / "main.tex"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_literal_input_emits_one_source_record_per_external_line(self) -> None:
        snippets = self.root / "snippets"
        snippets.mkdir()
        external = snippets / "demo.txt"
        external.write_text("alpha\n\nUnicode: 你好 % literal comment\n", encoding="utf-8")
        self.main.write_text(
            "before\n\\verbatiminput{snippets/demo.txt}\nafter\n",
            encoding="utf-8",
        )

        ir = build_external_verbatim_ir(self.root, [self.main])

        self.assertEqual(len(ir.blocks), 1)
        block = ir.blocks[0]
        self.assertEqual(block.execution_source_line, 2)
        self.assertEqual(block.external_source, external.resolve())
        self.assertEqual(
            [record.external_source_line for record in block.records], [1, 2, 3]
        )
        self.assertEqual(
            [record.visible_text for record in block.records],
            ["alpha", "", "Unicode: 你好 % literal comment"],
        )
        self.assertEqual(
            block.fenced_markdown,
            "```text\nalpha\n\nUnicode: 你好 % literal comment\n```",
        )
        provenance = block.records[2].provenance
        self.assertEqual(provenance["generation_source"], "latex_source")
        self.assertFalse(provenance["pdf_text_used"])
        self.assertEqual(provenance["execution"]["source_file"], "main.tex")
        self.assertEqual(provenance["execution"]["source_line"], 2)
        self.assertEqual(provenance["external"]["source_file"], "snippets/demo.txt")
        self.assertEqual(provenance["external"]["source_line"], 3)

    def test_tex_comment_is_ignored_but_external_percent_is_literal(self) -> None:
        (self.root / "real.txt").write_text("% printed comment\n", encoding="utf-8")
        self.main.write_text(
            "% \\verbatiminput{missing.txt}\n"
            "text \\% not-a-comment\n"
            "\\verbatiminput{real.txt}% trailing TeX comment\n",
            encoding="utf-8",
        )

        ir = build_external_verbatim_ir(self.root, [self.main])

        self.assertEqual(len(ir.blocks), 1)
        self.assertEqual(ir.records[0].visible_text, "% printed comment")

    def test_static_line_range_and_gobble_options_preserve_original_line_numbers(self) -> None:
        (self.root / "part.txt").write_text(
            "skip\n  first\n  second\n  third\nskip\n", encoding="utf-8"
        )
        self.main.write_text(
            "\\verbatiminput[firstline=2,lastline=4,gobble=2]{part.txt}\n",
            encoding="utf-8",
        )

        ir = build_external_verbatim_ir(self.root, [self.main])

        self.assertEqual(
            [
                (record.external_source_line, record.raw_text, record.visible_text)
                for record in ir.records
            ],
            [
                (2, "  first", "first"),
                (3, "  second", "second"),
                (4, "  third", "third"),
            ],
        )
        self.assertEqual(
            dict(ir.blocks[0].options),
            {"firstline": "2", "lastline": "4", "gobble": "2"},
        )

    def test_render_fence_expands_for_literal_backticks(self) -> None:
        (self.root / "ticks.txt").write_text("```literal\n", encoding="utf-8")
        self.main.write_text("\\verbatiminput{ticks.txt}\n", encoding="utf-8")
        records = build_external_verbatim_ir(self.root, [self.main]).records

        self.assertEqual(render_fenced_code(records), "````text\n```literal\n````")

    def test_path_escape_and_dynamic_path_fail_closed(self) -> None:
        outside = self.root.parent / "external-verbatim-secret.txt"
        outside.write_text("secret\n", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)

        for invocation in (
            "\\verbatiminput{../external-verbatim-secret.txt}\n",
            "\\verbatiminput{\\jobname.txt}\n",
            "\\verbatiminput[firstline=\\input{evil}]{safe.txt}\n",
        ):
            with self.subTest(invocation=invocation):
                self.main.write_text(invocation, encoding="utf-8")
                with self.assertRaises(ExternalVerbatimSafetyError):
                    build_external_verbatim_ir(self.root, [self.main])

    def test_non_strict_mode_records_rejection_without_reading_unsafe_target(self) -> None:
        self.main.write_text("\\verbatiminput{../secret.txt}\n", encoding="utf-8")

        ir = build_external_verbatim_ir(self.root, [self.main], strict=False)

        self.assertEqual(ir.blocks, ())
        self.assertEqual(len(ir.rejections), 1)
        self.assertEqual(ir.rejections[0].code, "unsafe_path")
        self.assertEqual(ir.rejections[0].execution_source_line, 1)

    def test_unicode_literal_path_and_symlink_escape(self) -> None:
        unicode_file = self.root / "片段.txt"
        unicode_file.write_text("内容\n", encoding="utf-8")
        self.main.write_text("\\verbatiminput{片段.txt}\n", encoding="utf-8")
        ir = build_external_verbatim_ir(self.root, [self.main])
        self.assertEqual(ir.records[0].visible_text, "内容")
        self.assertEqual(
            ir.records[0].provenance["external"]["source_file"], "片段.txt"
        )

        outside = self.root.parent / "external-verbatim-link-target.txt"
        outside.write_text("outside\n", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        link = self.root / "link.txt"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symbolic links are unavailable")
        self.main.write_text("\\verbatiminput{link.txt}\n", encoding="utf-8")
        with self.assertRaises(ExternalVerbatimSafetyError):
            build_external_verbatim_ir(self.root, [self.main])

    def test_empty_external_file_is_a_valid_empty_block(self) -> None:
        (self.root / "empty.txt").write_bytes(b"")
        self.main.write_text("\\verbatiminput{empty.txt}\n", encoding="utf-8")

        ir = build_external_verbatim_ir(self.root, [self.main])

        self.assertEqual(len(ir.blocks), 1)
        self.assertEqual(ir.records, ())
        self.assertEqual(ir.blocks[0].fenced_markdown, "```text\n\n```")


if __name__ == "__main__":
    unittest.main()
