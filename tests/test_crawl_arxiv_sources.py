from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "crawl_arxiv_sources.py"
SPEC = importlib.util.spec_from_file_location("crawl_arxiv_sources", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CrawlArxivSourcesTests(unittest.TestCase):
    @staticmethod
    def _single_record_payload(arxiv_id: str, submitted: str) -> bytes:
        return f'''<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
 xmlns:a="http://arxiv.org/OAI/arXivRaw/">
 <ListRecords>
  <record><metadata><a:arXivRaw>
   <a:id>{arxiv_id}</a:id><a:title>Allowed</a:title><a:authors>A</a:authors>
   <a:categories>cs.CL</a:categories>
   <a:license>https://creativecommons.org/licenses/by/4.0/</a:license>
   <a:version version="v1"><a:date>{submitted}</a:date></a:version>
  </a:arXivRaw></metadata></record>
 </ListRecords>
</OAI-PMH>'''.encode()

    def test_crawl_oai_shards_dates_and_reuses_window_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output_root=Path(directory),
                from_date="2026-01-01",
                until_date="2026-01-04",
                oai_window_days=2,
                oai_pages=10,
                set_spec="",
                resume=True,
                user_agent="test",
                timeout=1,
                retries=1,
                oai_delay_seconds=0,
            )
            payloads = [
                self._single_record_payload("2601.00001", "2026-01-01"),
                self._single_record_payload("2601.00002", "2026-01-03"),
            ]
            with mock.patch.object(
                MODULE,
                "request_bytes",
                side_effect=payloads,
            ) as request:
                rows, report = MODULE.crawl_oai(args)
            self.assertEqual([row["stem"] for row in rows], ["2601.00001v1", "2601.00002v1"])
            self.assertEqual(request.call_count, 2)
            self.assertIn("from=2026-01-01", request.call_args_list[0].args[0])
            self.assertIn("until=2026-01-02", request.call_args_list[0].args[0])
            self.assertIn("from=2026-01-03", request.call_args_list[1].args[0])
            self.assertIn("until=2026-01-04", request.call_args_list[1].args[0])
            self.assertEqual(report["date_windows"], 2)
            self.assertEqual(report["windows_started"], 2)
            self.assertEqual(report["pages_processed"], 2)

            with mock.patch.object(
                MODULE,
                "request_bytes",
                side_effect=AssertionError("cache was not reused"),
            ):
                cached_rows, cached_report = MODULE.crawl_oai(args)
            self.assertEqual(cached_rows, rows)
            self.assertEqual(cached_report["pages_processed"], 2)

    def test_oai_date_windows_are_inclusive_and_non_overlapping(self) -> None:
        self.assertEqual(
            MODULE.oai_date_windows("2026-01-01", "2026-01-10", 4),
            [
                ("2026-01-01", "2026-01-04"),
                ("2026-01-05", "2026-01-08"),
                ("2026-01-09", "2026-01-10"),
            ],
        )

    def test_oai_date_windows_reject_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.oai_date_windows("2026-01-01", "2026-01-10", 0)
        with self.assertRaises(ValueError):
            MODULE.oai_date_windows("2026-01-10", "2026-01-01", 7)

    def test_safe_stem_handles_legacy_identifier(self) -> None:
        self.assertEqual(MODULE.safe_stem("math/0301001", "v2"), "math_0301001v2")

    def test_parse_oai_keeps_only_allowed_license(self) -> None:
        payload = b'''<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
 xmlns:a="http://arxiv.org/OAI/arXivRaw/">
 <ListRecords>
  <record><metadata><a:arXivRaw>
   <a:id>2601.00001</a:id><a:title>Allowed</a:title><a:authors>A</a:authors>
   <a:categories>cs.CL cs.AI</a:categories>
   <a:license>https://creativecommons.org/licenses/by/4.0/</a:license>
   <a:version version="v1"><a:date>2026-01-01</a:date></a:version>
  </a:arXivRaw></metadata></record>
  <record><metadata><a:arXivRaw>
   <a:id>2601.00002</a:id><a:title>Blocked</a:title><a:authors>B</a:authors>
   <a:categories>cs.CL</a:categories>
   <a:license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</a:license>
   <a:version version="v1"><a:date>2026-01-01</a:date></a:version>
  </a:arXivRaw></metadata></record>
  <resumptionToken>next-token</resumptionToken>
 </ListRecords>
</OAI-PMH>'''
        rows, token, counts = MODULE.parse_oai_payload(payload)
        self.assertEqual([row["stem"] for row in rows], ["2601.00001v1"])
        self.assertEqual(token, "next-token")
        self.assertEqual(counts, {"records": 2, "eligible": 1, "disallowed": 1, "malformed": 0})

    def test_selection_is_deterministic_and_respects_exclusions(self) -> None:
        rows = [
            {"stem": f"paper-{index}", "license_name": "CC-BY-4.0"}
            for index in range(10)
        ]
        first = MODULE.deterministic_selection(rows, count=4, seed=7, excluded={"paper-1"})
        second = MODULE.deterministic_selection(rows, count=4, seed=7, excluded={"paper-1"})
        self.assertEqual(first, second)
        self.assertNotIn("paper-1", {row["stem"] for row in first})

    def test_balanced_category_selection_is_even_and_deterministic(self) -> None:
        rows = [
            {
                "stem": f"{category}-{index}",
                "primary_category": category,
                "license_name": "CC-BY-4.0",
            }
            for category in ("cs.CL", "cs.CV", "cs.LG")
            for index in range(10)
        ]
        first = MODULE.deterministic_selection(
            rows,
            count=11,
            seed=7,
            excluded=set(),
            categories=["cs.CL", "cs.CV", "cs.LG"],
        )
        second = MODULE.deterministic_selection(
            rows,
            count=11,
            seed=7,
            excluded=set(),
            categories=["cs.CL", "cs.CV", "cs.LG"],
        )
        self.assertEqual(first, second)
        counts = {}
        for row in first:
            category = row["primary_category"]
            counts[category] = counts.get(category, 0) + 1
        self.assertEqual(sorted(counts.values()), [3, 4, 4])

    def test_balanced_selection_redistributes_exhausted_category(self) -> None:
        rows = [
            {"stem": "rare-0", "primary_category": "rare"},
            *[
                {"stem": f"large-{index}", "primary_category": "large"}
                for index in range(10)
            ],
        ]
        selected = MODULE.deterministic_selection(
            rows,
            count=6,
            seed=3,
            excluded=set(),
            categories=["rare", "large"],
        )
        self.assertEqual(
            sum(row["primary_category"] == "rare" for row in selected), 1
        )

    def test_archive_validation_rejects_html_error_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            path.write_bytes(b"<!doctype html><title>rate limited</title>")
            self.assertEqual(
                MODULE.archive_looks_valid(path),
                (False, "server_returned_markup_instead_of_source"),
            )
            path.write_bytes(b"\x1f\x8bsource archive")
            self.assertEqual(MODULE.archive_looks_valid(path), (True, "passed"))


if __name__ == "__main__":
    unittest.main()
