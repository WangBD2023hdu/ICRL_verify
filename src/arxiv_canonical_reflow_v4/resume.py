"""Small durable compile checkpoints; no PDFs, images, or duplicate GT.

Each worker owns one job file. Only the parent appends the source-completion
index. An accepted page consumes source blocks only after both training rows
have been committed, so target overruns and interrupted exports remain work.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core import CanonicalBlock, CanonicalPage

SCHEMA_VERSION = 1
_DENSE_ID = re.compile(
    r"^(?P<paper>.+)_dense_(?P<pool>\d+)_(?P<ordinal>\d+)_(?P<digest>[0-9a-f]{12})"
    r"(?P<mutation>_confusable_s.*)$"
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _durable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class JobResumeSpec:
    path: str
    saved_ids: tuple[str, ...]


class CompileCheckpoint:
    def __init__(self, page: CanonicalPage, spec: JobResumeSpec) -> None:
        self.page = page
        self.path = Path(spec.path)
        self.saved_ids = set(spec.saved_ids)
        self.signature = _digest(asdict(page))
        self.records: list[dict[str, Any]] = []
        self.next_ordinal = 1
        self.legacy_recovered = 0
        self.skipped_blocks = 0
        self.saved_pages_skipped = 0
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("signature") == self.signature:
                self.records = data["records"]
                self.next_ordinal = data["next_ordinal"]
                self.saved_pages_skipped = data.get("saved_pages_skipped", 0)

    def _consumed(self, saved_ids: set[str]) -> set[str]:
        return {
            block_id
            for record in self.records
            if record["status"] == "rejected" or record["page_id"] in saved_ids
            for block_id in record["blocks"]
        }

    def _save(self) -> None:
        _durable_json(
            self.path,
            {
                "signature": self.signature,
                "next_ordinal": self.next_ordinal,
                "records": self.records,
                "saved_pages_skipped": self.saved_pages_skipped,
            },
        )

    def record(
        self,
        *,
        page_id: str,
        status: str,
        blocks: Sequence[CanonicalBlock],
        next_ordinal: int,
    ) -> None:
        self.records.append(
            {
                "page_id": page_id,
                "status": status,
                "blocks": [block.block_id for block in blocks],
            }
        )
        self.next_ordinal = max(self.next_ordinal, next_ordinal)
        self._save()

    def _legacy_spans(
        self,
        bundles: Sequence[Sequence[CanonicalBlock]],
        *,
        start_only: int | None = None,
        ordinal: int | None = None,
    ) -> dict[str, tuple[int, int]]:
        recorded = {record["page_id"] for record in self.records}
        wanted: dict[str, str] = {}
        for pair_id in self.saved_ids - recorded:
            match = _DENSE_ID.fullmatch(pair_id)
            if match is not None and (
                ordinal is None or int(match["ordinal"]) == ordinal
            ):
                wanted[match["digest"]] = pair_id
        found: dict[str, tuple[int, int]] = {}
        if not wanted:
            return found
        starts = range(len(bundles)) if start_only is None else (start_only,)
        # Recover only exact native block-ID hashes, never fuzzy PDF/GT text.
        # Source pools are bounded (normally four candidate pages).
        for start in starts:
            hashes = [
                hashlib.sha256(layout.encode())
                for layout in (
                    "one_column",
                    "two_column",
                )
            ]
            for end in range(start + 1, len(bundles) + 1):
                for block in bundles[end - 1]:
                    for digest in hashes:
                        digest.update(b"\0" + block.block_id.encode())
                for digest in hashes:
                    pair_id = wanted.get(digest.hexdigest()[:12])
                    if pair_id is not None:
                        found[pair_id] = (start, end)
            if len(found) == len(wanted):
                break
        return found

    def restore(
        self,
        bundles: Sequence[Sequence[CanonicalBlock]],
    ) -> list[Sequence[CanonicalBlock]]:
        consumed = self._consumed(self.saved_ids)
        self.saved_pages_skipped = len(
            {r["page_id"] for r in self.records if r["page_id"] in self.saved_ids}
        )
        remaining = [b for b in bundles if not all(x.block_id in consumed for x in b)]
        self.skipped_blocks = len(consumed)
        known = self.saved_ids - {r["page_id"] for r in self.records}
        spans = self._legacy_spans(remaining)
        # If an old page crossed a removed obstruction, its hash may not match
        # a contiguous span yet. Replay that job's unresolved path instead of
        # guessing or dropping its whole paper; skip_saved_prefix retries once
        # the packer has removed the obstruction.
        if known and spans.keys() == known:
            for pair_id, (start, end) in spans.items():
                match = _DENSE_ID.fullmatch(pair_id)
                assert match is not None
                self.records.append(
                    {
                        "page_id": pair_id,
                        "status": "accepted",
                        "blocks": [
                            b.block_id
                            for bundle in remaining[start:end]
                            for b in bundle
                        ],
                    }
                )
                self.next_ordinal = max(self.next_ordinal, int(match["ordinal"]) + 1)
            self.legacy_recovered += len(spans)
            self.saved_pages_skipped += len(spans)
            consumed = self._consumed(self.saved_ids)
            remaining = [
                b for b in bundles if not all(x.block_id in consumed for x in b)
            ]
            self.skipped_blocks = len(consumed)
        self._save()
        return remaining

    def skip_saved_prefix(
        self,
        bundles: Sequence[Sequence[CanonicalBlock]],
        start: int,
        ordinal: int,
    ) -> int | None:
        spans = self._legacy_spans(bundles, start_only=start, ordinal=ordinal)
        if not spans:
            return None
        pair_id, (_, end) = next(iter(spans.items()))
        blocks = [b for bundle in bundles[start:end] for b in bundle]
        self.saved_pages_skipped += 1
        self.record(
            page_id=pair_id, status="accepted", blocks=blocks, next_ordinal=ordinal + 1
        )
        self.legacy_recovered += 1
        self.skipped_blocks += len(blocks)
        return end

    def completion_ids(self, saved_ids: set[str]) -> set[str] | None:
        consumed = self._consumed(saved_ids)
        if not all(block.block_id in consumed for block in self.page.blocks):
            return None
        return {r["page_id"] for r in self.records if r["page_id"] in saved_ids}


class ResumeIndex:
    """Parent-side index, read once on startup; excludes target/worker counts."""

    def __init__(
        self,
        output: Path,
        settings: dict[str, Any],
        saved_ids: set[str],
        mutation_suffix: str,
    ) -> None:
        namespace = _digest({"schema": SCHEMA_VERSION, **settings})[:20]
        self.root = output / "realtime_training" / "resume" / namespace
        self.source_path = self.root / "sources.jsonl"
        self.saved_ids = saved_ids
        self.ids_by_job: dict[tuple[str, int], set[str]] = {}
        for pair_id in saved_ids:
            match = _DENSE_ID.fullmatch(pair_id)
            if match is not None and match["mutation"] == mutation_suffix:
                self.ids_by_job.setdefault(
                    (match["paper"], int(match["pool"])),
                    set(),
                ).add(pair_id)
        self.sources: dict[str, dict[str, Any]] = {}
        if self.source_path.is_file():
            # A killed append can leave just the final line incomplete.
            with self.source_path.open("rb+") as handle:
                while line := handle.readline():
                    if not line.endswith(b"\n"):
                        handle.seek(-len(line), os.SEEK_CUR)
                        handle.truncate()
                        break
                    row = json.loads(line)
                    self.sources[row["paper_id"]] = row

    def job_spec(self, page: CanonicalPage) -> JobResumeSpec:
        key = _digest(page.page_id)
        return JobResumeSpec(
            str(self.root / "jobs" / key[:2] / f"{key}.json"),
            tuple(sorted(self.ids_by_job.get((page.paper_id, page.ordinal), ()))),
        )

    def source_complete(self, paper_id: str, fingerprint: str) -> bool:
        row = self.sources.get(paper_id)
        return bool(
            row is not None
            and row["fingerprint"] == fingerprint
            and set(row["accepted_ids"]) <= self.saved_ids
        )

    def mark_source_complete(
        self, paper_id: str, fingerprint: str, accepted_ids: set[str]
    ) -> None:
        row = {
            "paper_id": paper_id,
            "fingerprint": fingerprint,
            "accepted_ids": sorted(accepted_ids),
        }
        self.source_path.parent.mkdir(parents=True, exist_ok=True)
        with self.source_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.sources[paper_id] = row
