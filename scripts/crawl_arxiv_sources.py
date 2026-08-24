#!/usr/bin/env python3
"""Standalone, resumable downloader for license-eligible arXiv source packages.

The script performs only data acquisition.  It does not extract archives or
compile LaTeX.  It first crawls arXiv's OAI ``arXivRaw`` metadata, keeps only
CC-BY-4.0, CC-BY-SA-4.0, and CC0-1.0 records, deterministically selects the
requested number of papers, and then downloads version-pinned e-print source
archives from ``export.arxiv.org``.

Example:

    python scripts/crawl_arxiv_sources.py \
      --output-root outputs/arxiv_sources_2000 \
      --from-date 2025-08-01 --until-date 2026-05-31 \
      --num-papers 2000 --oai-pages 30

Re-running the same command resumes from per-paper checkpoints.  Use repeated
``--exclude-results-root`` arguments to avoid papers attempted by older runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


OAI_ENDPOINT = "https://export.arxiv.org/oai2"
EPRINT_ENDPOINT = "https://export.arxiv.org/e-print"
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}
ALLOWED_LICENSES = {
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "https://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "http://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA-4.0",
    "https://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA-4.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}
HEARTBEAT_SECONDS = 30.0
SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(arxiv_id: str, version: str) -> str:
    value = SAFE_STEM_RE.sub("_", f"{arxiv_id}{version}").strip("._-")
    if not value or value in {".", ".."}:
        raise ValueError(f"unsafe arXiv identifier: {arxiv_id!r} {version!r}")
    return value


def parse_oai_payload(payload: bytes) -> tuple[list[dict[str, Any]], str | None, dict[str, int]]:
    root = ET.fromstring(payload)
    error = root.find(".//oai:error", OAI_NS)
    if error is not None:
        raise RuntimeError(
            f"OAI error code={error.attrib.get('code', 'unknown')}: "
            f"{(error.text or '').strip()}"
        )
    rows: list[dict[str, Any]] = []
    counts = Counter(records=0, eligible=0, disallowed=0, malformed=0)
    for record in root.findall(".//oai:record", OAI_NS):
        counts["records"] += 1
        raw = record.find(".//raw:arXivRaw", OAI_NS)
        if raw is None:
            counts["malformed"] += 1
            continue
        license_url = (raw.findtext("raw:license", default="", namespaces=OAI_NS) or "").strip()
        license_name = ALLOWED_LICENSES.get(license_url)
        if not license_name:
            counts["disallowed"] += 1
            continue
        arxiv_id = (raw.findtext("raw:id", default="", namespaces=OAI_NS) or "").strip()
        versions = raw.findall("raw:version", OAI_NS)
        categories = " ".join(
            (raw.findtext("raw:categories", default="", namespaces=OAI_NS) or "").split()
        )
        if not arxiv_id or not versions or not categories:
            counts["malformed"] += 1
            continue
        version_node = versions[-1]
        version = str(version_node.attrib.get("version", "")).strip()
        if not re.fullmatch(r"v[1-9][0-9]*", version):
            counts["malformed"] += 1
            continue
        row = {
            "arxiv_id": arxiv_id,
            "version": version,
            "stem": safe_stem(arxiv_id, version),
            "title": " ".join(
                (raw.findtext("raw:title", default="", namespaces=OAI_NS) or "").split()
            ),
            "authors": " ".join(
                (raw.findtext("raw:authors", default="", namespaces=OAI_NS) or "").split()
            ),
            "categories": categories,
            "primary_category": categories.split()[0],
            "license_url": license_url,
            "license_name": license_name,
            "submitted": (
                version_node.findtext("raw:date", default="", namespaces=OAI_NS) or ""
            ).strip(),
            "source_url": f"{EPRINT_ENDPOINT}/{arxiv_id}{version}",
        }
        rows.append(row)
        counts["eligible"] += 1
    token = root.findtext(".//oai:resumptionToken", default=None, namespaces=OAI_NS)
    return rows, token.strip() if token and token.strip() else None, dict(counts)


def request_bytes(url: str, *, user_agent: str, timeout: int, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": user_agent})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def oai_date_windows(
    from_date: str,
    until_date: str,
    window_days: int,
) -> list[tuple[str, str]]:
    """Return inclusive, non-overlapping OAI date windows."""

    if window_days < 1:
        raise ValueError("window_days must be positive")
    first = date.fromisoformat(from_date)
    last = date.fromisoformat(until_date)
    if first > last:
        raise ValueError("from_date must not be after until_date")
    windows: list[tuple[str, str]] = []
    current = first
    while current <= last:
        end = min(last, current + timedelta(days=window_days - 1))
        windows.append((current.isoformat(), end.isoformat()))
        current = end + timedelta(days=1)
    return windows


def crawl_oai(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oai_dir = args.output_root / "oai"
    oai_dir.mkdir(parents=True, exist_ok=True)
    windows = oai_date_windows(
        args.from_date,
        args.until_date,
        args.oai_window_days,
    )
    catalog: dict[str, dict[str, Any]] = {}
    totals: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    started = time.monotonic()
    print(
        f"[start] phase=oai pages={args.oai_pages} from={args.from_date} "
        f"until={args.until_date} windows={len(windows)} "
        f"window_days={args.oai_window_days} set={args.set_spec or 'all'} "
        f"output={oai_dir}",
        flush=True,
    )
    page_number = 0
    page_limit_reached = False
    for window_index, (window_from, window_until) in enumerate(windows, start=1):
        token: str | None = None
        window_page = 0
        while page_number < args.oai_pages:
            page_number += 1
            window_page += 1
            path = oai_dir / (
                f"window_{window_index:04d}_{window_from}_{window_until}_"
                f"page_{window_page:04d}.xml"
            )
            if args.resume and path.is_file() and path.stat().st_size:
                payload = path.read_bytes()
                state = "reused"
            else:
                if window_page == 1:
                    query = {
                        "verb": "ListRecords",
                        "metadataPrefix": "arXivRaw",
                        "from": window_from,
                        "until": window_until,
                    }
                    if args.set_spec:
                        query["set"] = args.set_spec
                else:
                    if not token:
                        break
                    query = {"verb": "ListRecords", "resumptionToken": token}
                url = f"{OAI_ENDPOINT}?{urlencode(query)}"
                request_done = threading.Event()

                def oai_request_heartbeat() -> None:
                    waited = 0.0
                    while not request_done.wait(HEARTBEAT_SECONDS):
                        waited += HEARTBEAT_SECONDS
                        print(
                            f"[progress] phase=oai_download "
                            f"page={page_number}/{args.oai_pages} "
                            f"window={window_index}/{len(windows)} "
                            f"window_page={window_page} range={window_from}:{window_until} "
                            f"waited={waited:.0f}s eligible_unique={len(catalog)} "
                            f"accepted={len(catalog)} "
                            f"rejected={totals['disallowed']+totals['malformed']} "
                            f"errors=0",
                            flush=True,
                        )

                request_reporter = threading.Thread(
                    target=oai_request_heartbeat,
                    daemon=True,
                )
                request_reporter.start()
                try:
                    payload = request_bytes(
                        url,
                        user_agent=args.user_agent,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                finally:
                    request_done.set()
                    request_reporter.join(timeout=1.0)
                parse_oai_payload(payload)
                atomic_write_bytes(path, payload)
                state = "downloaded"
            rows, token, counts = parse_oai_payload(payload)
            for key, value in counts.items():
                totals[key] += value
            for row in rows:
                catalog[str(row["stem"])] = row
            files.append(
                {
                    "path": str(path),
                    "window": window_index,
                    "window_from": window_from,
                    "window_until": window_until,
                    "window_page": window_page,
                    "bytes": len(payload),
                    **counts,
                }
            )
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"[unit-done] phase=oai page={page_number}/{args.oai_pages} "
                f"window={window_index}/{len(windows)} window_page={window_page} "
                f"range={window_from}:{window_until} state={state} "
                f"records={counts['records']} eligible={counts['eligible']} "
                f"eligible_unique={len(catalog)} "
                f"bytes={sum(item['bytes'] for item in files)} "
                f"pct={100*page_number/args.oai_pages:.1f}% "
                f"throughput={page_number/elapsed:.2f}_pages/s "
                f"elapsed={elapsed:.1f}s "
                f"eta={elapsed/page_number*(args.oai_pages-page_number):.1f}s "
                f"accepted={len(catalog)} "
                f"rejected={totals['disallowed']+totals['malformed']} "
                f"errors=0 current={path}",
                flush=True,
            )
            if not token:
                break
            if state == "downloaded" and args.oai_delay_seconds > 0:
                time.sleep(args.oai_delay_seconds)
        if page_number >= args.oai_pages:
            page_limit_reached = True
            break
    rows = [catalog[key] for key in sorted(catalog)]
    report = {
        "records": totals["records"],
        "eligible_unique": len(rows),
        "disallowed": totals["disallowed"],
        "malformed": totals["malformed"],
        "date_windows": len(windows),
        "windows_started": len({int(item["window"]) for item in files}),
        "pages_processed": page_number,
        "page_limit_reached": page_limit_reached,
        "files": files,
    }
    atomic_jsonl(args.output_root / "licenses.jsonl", rows)
    return rows, report


def load_excluded_stems(roots: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for value in roots:
        root = value.resolve()
        candidates = [root] if root.is_file() else [
            root / "results.jsonl",
            root / "selection.jsonl",
            root / "selection.json",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix == ".jsonl":
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            else:
                value_json = json.loads(path.read_text(encoding="utf-8"))
                rows = value_json if isinstance(value_json, list) else []
            for row in rows:
                stem = row.get("stem")
                if stem:
                    excluded.add(str(stem))
    return excluded


def deterministic_selection(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int,
    excluded: set[str],
    categories: Iterable[str] = (),
) -> list[dict[str, Any]]:
    eligible = [dict(row) for row in rows if str(row["stem"]) not in excluded]
    requested_categories = list(dict.fromkeys(str(value) for value in categories if value))
    if not requested_categories:
        eligible.sort(key=lambda row: str(row["stem"]))
        random.Random(seed).shuffle(eligible)
        if len(eligible) < count:
            raise ValueError(
                f"only {len(eligible)} eligible unexcluded papers; requested {count}"
            )
        return eligible[:count]

    requested_set = set(requested_categories)
    groups: dict[str, list[dict[str, Any]]] = {
        category: [] for category in requested_categories
    }
    for row in eligible:
        category = str(row.get("primary_category", ""))
        if category in requested_set:
            groups[category].append(row)
    missing = [category for category in requested_categories if not groups[category]]
    if missing:
        raise ValueError(f"requested categories have no eligible records: {missing}")
    if sum(len(values) for values in groups.values()) < count:
        capacities = {key: len(value) for key, value in groups.items()}
        raise ValueError(
            f"requested categories contain only {sum(capacities.values())} eligible papers; "
            f"requested {count}; capacities={capacities}"
        )
    for category, values in groups.items():
        values.sort(key=lambda row: str(row["stem"]))
        random.Random(f"{seed}:{category}").shuffle(values)

    # Round-robin allocation keeps non-exhausted categories within one sample
    # of each other.  If a small category is exhausted, its remaining quota is
    # deterministically redistributed across categories that still have data.
    offsets = {category: 0 for category in requested_categories}
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for category in requested_categories:
            offset = offsets[category]
            if offset >= len(groups[category]):
                continue
            selected.append(groups[category][offset])
            offsets[category] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError("balanced category selection exhausted unexpectedly")
    return selected


@dataclass
class RateLimiter:
    interval_seconds: float

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._last_started: float | None = None

    def wait(self) -> None:
        with self._lock:
            if self._last_started is not None:
                remaining = self.interval_seconds - (time.monotonic() - self._last_started)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_started = time.monotonic()


def archive_looks_valid(path: Path) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, "archive_missing_or_empty"
    with path.open("rb") as stream:
        prefix = stream.read(512).lstrip().lower()
    if prefix.startswith((b"<html", b"<!doctype html", b"<?xml")):
        return False, "server_returned_markup_instead_of_source"
    return True, "passed"


def download_source(
    row: dict[str, Any], args: argparse.Namespace, limiter: RateLimiter
) -> dict[str, Any]:
    stem = str(row["stem"])
    paper_dir = args.output_root / "papers" / stem
    archive = paper_dir / "source_archive.bin"
    checkpoint = paper_dir / "download.json"
    if args.resume and checkpoint.is_file() and archive.is_file():
        stored = read_json(checkpoint)
        valid, _ = archive_looks_valid(archive)
        if (
            stored.get("status") == "passed"
            and stored.get("source_url") == row["source_url"]
            and int(stored.get("bytes", -1)) == archive.stat().st_size
            and stored.get("sha256") == sha256_file(archive)
            and valid
        ):
            return {**stored, "resume_state": "reused"}

    paper_dir.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".bin.partial")
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            limiter.wait()
            request = Request(str(row["source_url"]), headers={"User-Agent": args.user_agent})
            digest = hashlib.sha256()
            total = 0
            with urlopen(request, timeout=args.timeout) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > args.max_archive_mib * 1024 * 1024:
                        raise ValueError(f"archive exceeds {args.max_archive_mib} MiB")
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            valid, reason = archive_looks_valid(temporary)
            if not valid:
                raise ValueError(reason)
            os.replace(temporary, archive)
            result = {
                **row,
                "status": "passed",
                "source_url": row["source_url"],
                "archive": str(archive.relative_to(args.output_root)),
                "bytes": total,
                "sha256": digest.hexdigest(),
                "attempts": attempt,
                "completed_at": utc_now(),
            }
            atomic_json(checkpoint, result)
            return result
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < args.retries:
                time.sleep(min(3 * (2 ** (attempt - 1)), 60))
    result = {
        **row,
        "status": "failed",
        "source_url": row["source_url"],
        "error": f"{type(last_error).__name__}: {last_error}",
        "attempts": args.retries,
        "completed_at": utc_now(),
    }
    atomic_json(checkpoint, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--from-date", default="2025-08-01")
    value.add_argument("--until-date", default=date.today().isoformat())
    value.add_argument("--set", dest="set_spec", default="", help="OAI setSpec; empty means all")
    value.add_argument(
        "--categories",
        nargs="+",
        default=[],
        metavar="CATEGORY",
        help=(
            "exact primary arXiv categories to sample evenly, for example "
            "--categories cs.CL cs.CV cs.LG math.OC quant-ph; fetch with an empty "
            "--set (the default) when categories span multiple OAI sets"
        ),
    )
    value.add_argument("--oai-pages", type=int, default=30)
    value.add_argument(
        "--oai-window-days",
        type=int,
        default=7,
        help=(
            "split the initial OAI date range into inclusive windows; small "
            "windows avoid export.arxiv.org timing out on multi-year ListRecords queries"
        ),
    )
    value.add_argument("--oai-delay-seconds", type=float, default=3.0)
    value.add_argument("--num-papers", type=int, default=2000)
    value.add_argument("--seed", type=int, default=73)
    value.add_argument("--workers", type=int, default=3)
    value.add_argument("--request-interval-seconds", type=float, default=3.1)
    value.add_argument("--timeout", type=int, default=180)
    value.add_argument("--retries", type=int, default=5)
    value.add_argument("--max-archive-mib", type=int, default=250)
    value.add_argument(
        "--user-agent",
        default="arxiv-source-crawler/1.0 (research dataset; contact: local-user)",
    )
    value.add_argument("--exclude-results-root", type=Path, action="append", default=[])
    value.add_argument("--metadata-only", action="store_true")
    value.add_argument("--no-resume", dest="resume", action="store_false")
    value.set_defaults(resume=True)
    return value


def main() -> int:
    args = parser().parse_args()
    date.fromisoformat(args.from_date)
    date.fromisoformat(args.until_date)
    if args.from_date > args.until_date:
        raise ValueError("--from-date must not be after --until-date")
    if (
        args.oai_pages < 1
        or args.oai_window_days < 1
        or args.num_papers < 1
        or args.retries < 1
    ):
        raise ValueError(
            "oai-pages, oai-window-days, num-papers, and retries must be positive"
        )
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if args.request_interval_seconds < 3.0:
        raise ValueError("request interval must be at least 3 seconds")
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    catalog, oai_report = crawl_oai(args)
    excluded = load_excluded_stems(args.exclude_results_root)
    selection = deterministic_selection(
        catalog,
        count=args.num_papers,
        seed=args.seed,
        excluded=excluded,
        categories=args.categories,
    )
    atomic_jsonl(args.output_root / "selection.jsonl", selection)
    selection_categories = dict(
        sorted(Counter(str(row["primary_category"]) for row in selection).items())
    )
    if args.metadata_only:
        summary = {
            "status": "metadata_only",
            "created_at": utc_now(),
            "oai": oai_report,
            "excluded": len(excluded),
            "selected": len(selection),
            "requested_categories": args.categories,
            "selection_categories": selection_categories,
        }
        atomic_json(args.output_root / "crawl_summary.json", summary)
        print(
            f"[finish] status=metadata_only selected={len(selection)} accepted={len(selection)} "
            f"rejected=0 errors=0 output={args.output_root}",
            flush=True,
        )
        return 0

    limiter = RateLimiter(args.request_interval_seconds)
    started = time.monotonic()
    completed = accepted = failed = bytes_total = 0
    state_lock = threading.Lock()
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
            with state_lock:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = completed / elapsed
                eta = (len(selection) - completed) / rate if rate else 0.0
                print(
                    f"[progress] phase=source_download completed={completed}/{len(selection)} "
                    f"pct={100*completed/len(selection):.1f}% bytes={bytes_total} "
                    f"throughput={rate:.3f}_papers/s elapsed={elapsed:.1f}s eta={eta:.1f}s "
                    f"accepted={accepted} rejected={failed} errors={failed}",
                    flush=True,
                )

    reporter = threading.Thread(target=heartbeat, daemon=True)
    reporter.start()
    results_by_stem: dict[str, dict[str, Any]] = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_source, row, args, limiter): row
                for row in selection
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with state_lock:
                    completed += 1
                    if result["status"] == "passed":
                        accepted += 1
                        bytes_total += int(result["bytes"])
                    else:
                        failed += 1
                    results_by_stem[str(result["stem"])] = result
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = completed / elapsed
                    eta = (len(selection) - completed) / rate if rate else 0.0
                    print(
                        f"[unit-done] phase=source_download completed={completed}/{len(selection)} "
                        f"pct={100*completed/len(selection):.1f}% current={result['stem']} "
                        f"status={result['status']} bytes={int(result.get('bytes', 0))} "
                        f"bytes_total={bytes_total} throughput={rate:.3f}_papers/s "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s accepted={accepted} "
                        f"rejected={failed} errors={failed}",
                        flush=True,
                    )
    finally:
        stop_heartbeat.set()
        reporter.join(timeout=1.0)

    results = [results_by_stem[str(row["stem"])] for row in selection]
    atomic_jsonl(args.output_root / "results.jsonl", results)
    summary = {
        "status": "passed" if accepted > 0 else "failed",
        "created_at": utc_now(),
        "oai": oai_report,
        "selected": len(selection),
        "requested_categories": args.categories,
        "selection_categories": selection_categories,
        "accepted": accepted,
        "failed": failed,
        "bytes": bytes_total,
        "licenses": dict(Counter(row["license_name"] for row in results if row["status"] == "passed")),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_json(args.output_root / "crawl_summary.json", summary)
    print(
        f"[finish] status={summary['status']} selected={len(selection)} accepted={accepted} "
        f"rejected={failed} errors={failed} bytes={bytes_total} "
        f"elapsed={summary['elapsed_seconds']}s output={args.output_root}",
        flush=True,
    )
    return 0 if accepted > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
