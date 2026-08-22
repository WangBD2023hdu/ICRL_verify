from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_arxiv_latex_recompile_pilot as PILOT  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "arxiv_latex_recompile_dataset",
    SCRIPTS / "build_arxiv_latex_recompile_dataset.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ArxivLatexRecompileDatasetTests(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_unique(self) -> None:
        records = [
            {"stem": "a1", "primary_category": "a"},
            {"stem": "a2", "primary_category": "a"},
            {"stem": "b1", "primary_category": "b"},
            {"stem": "b2", "primary_category": "b"},
        ]
        first = MODULE.stratified_selection(records, count=4, seed=73)
        second = MODULE.stratified_selection(records, count=4, seed=73)
        self.assertEqual(first, second)
        self.assertEqual(len({row["stem"] for row in first}), 4)
        self.assertEqual([row["selection_rank"] for row in first], list(range(4)))

    def test_load_excluded_stems_accepts_batch_roots_and_result_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch"
            batch.mkdir()
            first = batch / "results.jsonl"
            first.write_text(
                json.dumps({"stem": "2401.00001v1", "status": "success"}) + "\n"
                + json.dumps({"stem": "2401.00002v2", "status": "failed"}) + "\n",
                encoding="utf-8",
            )
            second = root / "other.jsonl"
            second.write_text(
                json.dumps({"stem": "2401.00003v1", "status": "rejected"}) + "\n",
                encoding="utf-8",
            )
            stems, sources = MODULE.load_excluded_stems([batch, second])
            self.assertEqual(
                stems,
                {"2401.00001v1", "2401.00002v2", "2401.00003v1"},
            )
            self.assertEqual(sources, [str(first.resolve()), str(second.resolve())])

    def test_rate_limiter_serializes_threaded_request_starts(self) -> None:
        progress = MODULE.Progress(total=3, started=time.monotonic())
        limiter = MODULE.RequestRateLimiter(0.025, progress)
        starts: list[float] = []
        lock = threading.Lock()

        def run(index: int) -> None:
            limiter.wait(f"paper-{index}", 1)
            with lock:
                starts.append(time.monotonic())

        threads = [threading.Thread(target=run, args=(index,)) for index in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        starts.sort()
        self.assertEqual(len(starts), 3)
        self.assertGreaterEqual(starts[1] - starts[0], 0.022)
        self.assertGreaterEqual(starts[2] - starts[1], 0.022)

    def test_progress_updates_are_thread_safe(self) -> None:
        progress = MODULE.Progress(total=200)

        def update() -> None:
            for _ in range(50):
                progress.add_downloaded_bytes(2)
            progress.record_status("success")

        threads = [threading.Thread(target=update) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(progress.downloaded_bytes, 400)
        self.assertEqual(progress.completed, 4)
        self.assertEqual(progress.success, 4)
        self.assertEqual(progress.failed, 0)

    def test_resume_baseline_does_not_change_counts(self) -> None:
        progress = MODULE.Progress(total=10)
        progress.record_status("success")
        progress.record_status("failed")
        progress.reset_rate_baseline()
        self.assertEqual(progress.completed, 2)
        self.assertEqual(progress.success, 1)
        self.assertEqual(progress.failed, 1)
        self.assertEqual(progress._rate_completed_offset, 2)

    def test_truncated_content_length_is_not_published_as_archive(self) -> None:
        class Response(io.BytesIO):
            def __init__(self) -> None:
                super().__init__(b"abc")
                self.headers = {
                    "Content-Type": "application/gzip",
                    "Content-Length": "10",
                }

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source_archive.bin"
            progress = MODULE.Progress(total=1)
            with mock.patch.object(PILOT.urllib.request, "urlopen", return_value=Response()):
                with self.assertRaisesRegex(EOFError, "download length mismatch"):
                    PILOT.download_source(
                        "https://example.invalid/e-print/test",
                        destination,
                        10,
                        progress,
                        "testv1",
                        False,
                    )
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".partial").exists())


if __name__ == "__main__":
    unittest.main()
