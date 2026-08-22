from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_arxiv_source_bins_to_verl.py"
SPEC = importlib.util.spec_from_file_location("run_arxiv_source_bins_to_verl", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SourceBinsToVerlTests(unittest.TestCase):
    def _write_crawl_root(self, root: Path) -> dict[str, object]:
        archive = root / "papers/2601.00001v1/source_archive.bin"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"not-empty-source-archive")
        row: dict[str, object] = {
            "arxiv_id": "2601.00001",
            "version": "v1",
            "stem": "2601.00001v1",
            "status": "passed",
            "license_name": "CC-BY-4.0",
            "archive": "papers/2601.00001v1/source_archive.bin",
            "sha256": MODULE.sha256_file(archive),
        }
        (root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return row

    def test_select_input_rows_accepts_crawler_bin_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_crawl_root(root)
            rows = MODULE.select_input_rows(root, paper_ids=set(), max_papers=0)
            self.assertEqual([row["stem"] for row in rows], ["2601.00001v1"])
            self.assertEqual(
                MODULE.resolve_archive(root, rows[0]),
                (root / "papers/2601.00001v1/source_archive.bin").resolve(),
            )

    def test_select_input_rows_rejects_disallowed_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._write_crawl_root(root)
            row["license_name"] = "arXiv-nonexclusive-distribute"
            (root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no eligible"):
                MODULE.select_input_rows(root, paper_ids=set(), max_papers=0)

    def test_source_first_manifest_contains_absolute_page_and_gt_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper_root = root / "source_first/2601.00001v1"
            pages = paper_root / "pages"
            pages.mkdir(parents=True)
            (pages / "page_0002.md").write_text("Exact source GT.\n", encoding="utf-8")
            (pages / "page_0002.png").write_bytes(b"png")
            page = {
                "data_id": "2601.00001v1_page_0002",
                "markdown": "pages/page_0002.md",
                "image": "pages/page_0002.png",
            }
            (paper_root / "pages_passed.jsonl").write_text(
                json.dumps(page) + "\n", encoding="utf-8"
            )
            manifest_path = root / "source_first_cases.json"
            cases = MODULE.build_source_first_manifest(
                root / "source_first",
                [{"stem": "2601.00001v1", "status": "success"}],
                manifest_path,
            )
            self.assertEqual(cases[0]["pair_id"], page["data_id"])
            self.assertTrue(Path(cases[0]["image"]).is_absolute())
            self.assertTrue(Path(cases[0]["markdown_path"]).is_absolute())
            self.assertEqual(json.loads(manifest_path.read_text()), cases)


if __name__ == "__main__":
    unittest.main()
