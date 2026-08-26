from __future__ import annotations

import gzip
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from arxiv_source_first_v2.synctex_ir import (
    SCALED_POINTS_PER_PDF_POINT,
    alignment_for_probes,
    parse_synctex,
    source_lines_for_span,
)


@dataclass(frozen=True)
class Probe:
    probe_id: str
    source_file: Path
    source_lines: tuple[int, ...]


@dataclass(frozen=True)
class Locator:
    source_file: Path
    source_start: int
    source_end: int


class SyncTeXIrTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "part.tex"
        self.source.write_text("alpha beta\ngamma\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_synctex(self, value: str, *, gzip_output: bool = False) -> Path:
        suffix = ".synctex.gz" if gzip_output else ".synctex"
        path = self.root / ("main" + suffix)
        if gzip_output:
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(value)
        else:
            path.write_text(value, encoding="utf-8")
        return path

    def fixture(self) -> str:
        scale = int(round(SCALED_POINTS_PER_PDF_POINT))
        return (
            "SyncTeX Version:1\n"
            f"Input:7:{self.source}\n"
            "Output:pdf\nContent:\n"
            "{1\n"
            f"x7,1:{scale * 10},{scale * 20}\n"
            f"k7,1:{scale * 12},{scale * 20}:{scale}\n"
            "}\n"
            "{2\n"
            f"x7,2:{scale * 30},{scale * 40}\n"
            "}\n"
            "Postamble:\n"
        )

    def test_parser_reads_gzip_and_indexes_source_lines(self) -> None:
        index = parse_synctex(
            self.write_synctex(self.fixture(), gzip_output=True),
            source_root=self.root,
        )
        first = index.points_for_line(self.source, 1)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0].page_number, 1)
        self.assertAlmostEqual(first[0].x, 10.0, places=3)
        self.assertEqual(index.pages, (1, 2))
        self.assertFalse(index.as_json(self.root)["pdf_text_used"])

    def test_source_span_line_mapping_is_one_based_and_end_exclusive(self) -> None:
        source = self.source.read_text(encoding="utf-8")
        self.assertEqual(source_lines_for_span(source, 0, 5), (1,))
        self.assertEqual(source_lines_for_span(source, 6, 16), (1, 2))
        self.assertEqual(source_lines_for_span(source, 11, 16), (2,))

    def test_probe_alignment_uses_atom_offsets_and_never_pdf_text(self) -> None:
        index = parse_synctex(
            self.write_synctex(self.fixture()),
            source_root=self.root,
        )
        probe = Probe("p1", self.source, (99,))
        rows, summary = alignment_for_probes(
            index,
            [probe],
            {"p1": Locator(self.source, 11, 16)},
        )
        self.assertEqual([row["page_number"] for row in rows["p1"]], [2])
        self.assertEqual(rows["p1"][0]["locator"], "synctex_clean_source_line")
        self.assertEqual(summary["coverage"], 1.0)
        self.assertFalse(summary["pdf_text_used"])

    def test_explicit_generated_line_override_wins_over_clean_offsets(self) -> None:
        index = parse_synctex(
            self.write_synctex(self.fixture()),
            source_root=self.root,
        )
        probe = Probe("p1", self.source, (1,))
        rows, _summary = alignment_for_probes(
            index,
            [probe],
            {"p1": Locator(self.source, 0, 5)},
            line_overrides={"p1": (self.source, (2,))},
        )
        self.assertEqual([row["page_number"] for row in rows["p1"]], [2])


if __name__ == "__main__":
    unittest.main()
