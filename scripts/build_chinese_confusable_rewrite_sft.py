#!/usr/bin/env python3
"""Chinese glyph-confusion rewrite SFT, streamed directly to MS-Swift JSONL.

No LaTeX compiler, PDF, image, or language model is used. Public-source mode
fetches only Markdown files, pinned to Git commit IDs. Local mode accepts
Markdown/TXT and JSONL with a `text` or `markdown` field. Paths are relative to
the launch directory. Only tokenizers (not model weights) are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, islice
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

VERSION = "chinese_confusable_rewrite_v2_heading_zero"
MUTATION_SEED_VERSION = "chinese_confusable_rewrite_v1"
PROMPT_VERSION = "heading_rewrite_boundary_en_v3"
NO_HEADING_PROBABILITY = 0.28
# Deliberately the exact prompt already approved for the arXiv rewrite task.
# A regression test compares these strings with the existing script.
PROMPT_PREFIX = (
    "Please rewrite the document enclosed by the boundary markers using only "
    "these two formatting changes. This is not a translation task.\n"
    "1. For each text block beginning with an existing heading prefix of 1 to 6 "
    "# characters, randomly choose a different number of # characters from "
    "0 to 4 (0 means removing all leading # characters). "
    "Change only the number of # characters. Preserve the spaces after them "
    "exactly. Do not add heading prefixes to non-heading text or change prefixes "
    "inside figures, tables, formulas, or code.\n"
    "2. Enclose the entire result in a Markdown code fence: start with "
    "```markdown followed by a newline, and end with a newline followed by ```.\n"
    "Preserve every other character exactly, including spelling errors, "
    "numbers, whitespace, line breaks, HTML tables, and LaTeX formulas. Do not "
    "correct, add, omit, or explain any content. Do not output the boundary "
    "markers.\n\n"
    "<<<DOCUMENT_START>>>\n"
)
PROMPT_SUFFIX = "\n<<<DOCUMENT_END>>>"
RESPONSE_PREFIX = "```markdown\n"
RESPONSE_SUFFIX = "\n```"
HEADING_RE = re.compile(r"^(#{1,6})(?!#)", re.MULTILINE)
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002ebef]")

# Explicit, auditable shape pairs; not homophones and not the unfiltered
# SpellGCN shape list. No digits or automatic arbitrary-Han substitutions.
PAIR_GROUPS = (
    "未末", "土士", "己已", "日目", "牛午", "乌鸟", "晴睛", "清情",
    "待侍", "棉绵", "铜桐", "辨辩", "候侯", "微徽", "拆折", "大太",
    "人入", "处外", "石右", "贝见", "王玉", "问间", "休体", "住往",
    "很狠", "根跟", "抬治", "洒酒", "密蜜", "拔拨", "烧浇", "捡检",
    "慢漫", "板版", "堆推", "池地", "持特", "抱泡", "柏拍", "镜境",
)
DEFAULT_PAIRS = {
    char: tuple(sorted({other for group in PAIR_GROUPS if char in group
                        for other in group if other != char}))
    for group in PAIR_GROUPS for char in group
}
PUBLIC_SOURCES = {
    "d2l-zh": {
        "repo": "d2l-ai/d2l-zh", "prefix": "chapter_",
        "attribution": "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola and contributors",
        "license": "CC-BY-SA-4.0 and MIT-0 (project declaration; see source-specific notices)",
        "license_url": "https://github.com/d2l-ai/d2l-zh/blob/master/config.ini",
    },
    "oi-wiki": {
        "repo": "OI-wiki/OI-wiki", "prefix": "docs/",
        "attribution": "OI Wiki Team and contributors",
        "license": "CC-BY-SA-4.0 + SATA (non-code, except where otherwise noted)",
        "license_url": "https://github.com/OI-wiki/OI-wiki/blob/master/README.md",
    },
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seed_for(*values) -> int:
    return int(digest(json.dumps(values, ensure_ascii=False, sort_keys=True))[:16], 16)


def protected_mask(text: str) -> bytearray:
    """Protect source syntax without reserializing or normalizing the document.

    Tables/formulas/code are copied whole. Links' visible prose may mutate;
    destinations, HTML attributes, reference definitions and directives may not.
    """
    mask = bytearray(len(text))

    def mark(start, end):
        mask[start:end] = b"\1" * (end - start)

    # Fences are scanned first, including indented fences used by OI Wiki.
    fence = None
    offset = 0
    lines = text.splitlines(keepends=True)
    line_starts = []
    for line in lines:
        line_starts.append(offset)
        match = re.match(r"[ \t]*(`{3,}|~{3,})", line)
        if fence is not None:
            mark(offset, offset + len(line))
            if (match and match[1][0] == fence[0] and len(match[1]) >= fence[1]
                    and not line[match.end():].strip()):
                fence = None
        elif match:
            fence = (match[1][0], len(match[1]))
            mark(offset, offset + len(line))
        elif line.startswith(("    ", "\t")):
            mark(offset, offset + len(line))
        offset += len(line)

    patterns = (
        r"<!--.*?(?:-->|\Z)",
        r"<(table|figure|pre|code|script|style|math)\b[^>]*>.*?</\1\s*>",
        r"</?[A-Za-z][^>]*>|<https?://[^>]+>",
        r"(?<!\\)\$\$.*?(?<!\\)\$\$",
        r"\\\[.*?\\\]", r"\\\(.*?\\\)",
        r"(?<![\\$])\$(?!\$)(?:\\.|[^$])*?(?<!\\)\$",
        r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)",
        r"!?\[[^\]\n]*\]\((?:\\.|[^()\n]|\([^()\n]*\))*\)",
        r"https?://[^\s<>]+",
    )
    for index, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            start, end = match.span()
            if index == 8:  # Only the link destination; keep its label visible.
                start = text.index("](", start, end) + 1
            mark(start, end)
    for match in re.finditer(
            r"\\begin\{(equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|"
            r"displaymath|math|tabular\*?|tabularx|longtable)\}.*?\\end\{\1\}",
            text, re.DOTALL):
        mark(*match.span())
    # YAML front matter and non-visible document directives are not prose.
    front = re.match(r"\A---\r?\n.*?\r?\n(?:---|\.\.\.)\s*(?:\r?\n|\Z)", text, re.DOTALL)
    if front:
        mark(*front.span())
    for match in re.finditer(r"^(?:\s*\[[^\]\n]+\]:[^\n]*|:[\w-]+:[^\n]*|\{:[^\n]*\})$", text, re.MULTILINE):
        mark(*match.span())
    # A Markdown table consists of a header, delimiter and following pipe rows.
    for index, line in enumerate(lines):
        cells = line.strip().strip("|").split("|")
        if (index and "|" in line and "|" in lines[index - 1]
                and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)):
            first = index - 1
            last = index + 1
            while last < len(lines) and "|" in lines[last] and lines[last].strip():
                last += 1
            mark(line_starts[first], line_starts[last] if last < len(lines) else len(text))
    return mask


def mutate_text(text, *, ratio, rng, pairs=DEFAULT_PAIRS):
    mask = protected_mask(text)
    prose = [m.start() for m in HAN_RE.finditer(text) if not mask[m.start()]]
    by_pair = defaultdict(list)
    for offset in prose:
        for replacement in pairs.get(text[offset], ()):
            by_pair[(text[offset], replacement)].append(offset)
    target = math.floor(len(prose) * ratio + 0.5)
    selected = set()
    changes = []
    # Pair-first selection avoids high-frequency characters swallowing the map.
    active = sorted(by_pair)
    while active and len(changes) < target:
        pair = rng.choice(active)
        candidates = by_pair[pair]
        while candidates:
            index = rng.randrange(len(candidates))
            offset = candidates[index]
            candidates[index] = candidates[-1]
            candidates.pop()
            if offset not in selected:
                break
        else:
            active.remove(pair)
            continue
        selected.add(offset)
        changes.append({"origin_ans": pair[0], "ocr_ans": pair[1],
                        "input_char_offset": offset, "input_char_end": offset + 1})
    chars = list(text)
    for change in changes:
        chars[change["input_char_offset"]] = change["ocr_ans"]
    changes.sort(key=lambda c: c["input_char_offset"])
    stats = {"prose_han_chars": len(prose), "requested_mutations": target,
             "actual_mutations": len(changes), "mutation_ratio_requested": ratio,
             "mutation_ratio_achieved": len(changes) / len(prose) if prose else 0.0}
    return "".join(chars), changes, stats


def rewrite_answer(text, *, rng):
    mask = protected_mask(text)
    changes = []
    for match in HEADING_RE.finditer(text):
        if mask[match.start()]:
            continue
        old = len(match[1])
        # Keep removal at 28% even when the old level is excluded.
        new = (0 if rng.random() < NO_HEADING_PROBABILITY
               else rng.choice([level for level in range(1, 5) if level != old]))
        changes.append({"input_offset": match.start(), "from_level": old,
                        "to_level": new})
    body = text
    for change in reversed(changes):
        pos = change["input_offset"]
        body = body[:pos] + "#" * change["to_level"] + body[pos + change["from_level"]:]
    return RESPONSE_PREFIX + body + RESPONSE_SUFFIX, changes


def make_sample(text, *, source, config, chunk_index, count_tokens):
    if "<<<DOCUMENT_START>>>" in text or "<<<DOCUMENT_END>>>" in text:
        return None
    source_hash = digest(text)
    sample_id = digest(f"{VERSION}:{config['seed']}:{config['mutation_ratio']}:{source_hash}")
    # B's new heading policy must not change A's character mutations.
    mutation_id = digest(f"{MUTATION_SEED_VERSION}:{config['seed']}:{config['mutation_ratio']}:{source_hash}")
    edited, changes, stats = mutate_text(
        text, ratio=config["mutation_ratio"], rng=random.Random(seed_for(mutation_id, "mutation")))
    if not changes:
        return None
    response, heading_changes = rewrite_answer(edited, rng=random.Random(seed_for(sample_id, "heading")))
    tokens = count_tokens(response)
    if not config["min_response_tokens"] <= tokens <= config["max_response_tokens"]:
        return None
    for change in changes:
        delta = sum(h["to_level"] - h["from_level"] for h in heading_changes
                    if h["input_offset"] < change["input_char_offset"])
        change["char_offset"] = len(RESPONSE_PREFIX) + change["input_char_offset"] + delta
        change["char_end"] = change["char_offset"] + 1
        assert response[change["char_offset"]:change["char_end"]] == change["ocr_ans"]
    # Exact reconstruction checks the only permitted A -> B differences.
    rebuilt = edited
    for h in reversed(heading_changes):
        pos = h["input_offset"]
        rebuilt = rebuilt[:pos] + "#" * h["to_level"] + rebuilt[pos + h["from_level"]:]
    assert response == RESPONSE_PREFIX + rebuilt + RESPONSE_SUFFIX
    return {
        "messages": [{"role": "user", "content": PROMPT_PREFIX + edited + PROMPT_SUFFIX},
                     {"role": "assistant", "content": response}],
        "data_source": "chinese_confusable_text_rewrite", "ability": "heading_format_rewrite",
        "extra_info": {
            "sample_id": sample_id, "pipeline_version": VERSION, "prompt_version": PROMPT_VERSION,
            "source": source, "chunk_index": chunk_index, "source_text_sha256": source_hash,
            "edited_text_sha256": digest(edited), "response_text_sha256": digest(response),
            "response_tokens": tokens, "tokenizer": config.get("tokenizer", "test"),
            "changes": changes, "heading_changes": heading_changes, **stats,
        },
    }


def split_document(text, *, count_tokens, max_tokens, seed):
    """Contiguous, nonoverlapping windows; never split inside protected markup.

    No padding, duplicated context, trimming, or cross-document concatenation.
    A single protected block too long for the limit is skipped as one block.
    """
    mask = protected_mask(text)
    cuts = {0, len(text)}
    for match in re.finditer(r"\n|[。！？；]", text):
        pos = match.end()
        if pos == len(text) or not (mask[pos - 1] and mask[pos]):
            cuts.add(pos)
    # Preserve a whole table/code/formula if it is between long text paragraphs.
    for pos in range(1, len(text)):
        if mask[pos] != mask[pos - 1]:
            cuts.add(pos)
    cuts = sorted(cuts)
    rng = random.Random(seed)
    start_index = 0
    while start_index < len(cuts) - 1:
        start = cuts[start_index]
        # Random lengths only select contiguous source; no repeated variants.
        target = rng.randint(max(1, int(max_tokens * 0.55)), max_tokens)
        low, high = start_index + 1, len(cuts) - 1
        best = start_index
        while low <= high:
            mid = (low + high) // 2
            if count_tokens(text[start:cuts[mid]]) <= target:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best == start_index:
            # Do not repeatedly retry an indivisible oversized unit.
            yield start, cuts[start_index + 1], None
            start_index += 1
        else:
            end = cuts[best]
            yield start, end, text[start:end]
            start_index = best


def load_counter(identifier):
    if identifier == "simple":
        return len  # Explicitly test-only, never label it as Qwen token counts.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from tokenizers import Tokenizer
    path = Path(identifier)
    if path.is_dir():
        path = path / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    return lambda value: len(tokenizer.encode(value, add_special_tokens=False).ids)


def request_bytes(url):
    headers = {"User-Agent": "chinese-rewrite-sft/1.0"}
    if os.environ.get("GITHUB_TOKEN") and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=20) as response:
                return response.read()
        except (URLError, TimeoutError) as exc:
            if isinstance(exc, HTTPError) and exc.code < 500 and exc.code != 429:
                raise
            if attempt == 2:
                raise
            print(f"[download-retry] attempt={attempt + 1}/3 url={url} error={type(exc).__name__}", flush=True)
            time.sleep(attempt + 1)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def public_manifest(names, state_dir):
    path = state_dir / "public_sources.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    jobs = []
    for name in names:
        spec = PUBLIC_SOURCES[name]
        print(f"[discover] source={name} listing Markdown paths (no PDFs/images downloaded)", flush=True)
        tree = json.loads(request_bytes(f"https://api.github.com/repos/{spec['repo']}/git/trees/master?recursive=1"))
        if tree.get("truncated"):
            raise ValueError(f"GitHub tree was truncated: {name}")
        commit = tree["sha"]
        for item in tree["tree"]:
            rel = item["path"]
            if (item["type"] != "blob" or not rel.startswith(spec["prefix"])
                    or not rel.endswith(".md") or rel.endswith("_origin.md")):
                continue
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{commit}/{quote(rel)}"
            source = {"source_id": f"{name}:{commit}:{rel}", "corpus": name,
                      "url": f"https://github.com/{spec['repo']}/blob/{commit}/{quote(rel)}",
                      "path": rel, "commit": commit, "license": spec["license"],
                      "license_url": spec["license_url"].replace("/master/", f"/{commit}/"),
                      "attribution": spec["attribution"]}
            jobs.append({"kind": "public", "url": url, "source": source})
        print(f"[discover] source={name} complete cumulative_documents={len(jobs)} commit={commit}", flush=True)
    atomic_json(path, jobs)
    return jobs


def local_jobs(inputs, *, excluded_dir=None):
    files = []
    started = last_log = time.monotonic()
    discovered = 0
    excluded = Path(excluded_dir).resolve() if excluded_dir is not None else None
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.suffix.lower() in {".md", ".txt", ".jsonl"} and p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(value)
    total_bytes = 0
    files = sorted(p for p in set(files) if excluded is None or not p.resolve().is_relative_to(excluded))
    for index, path in enumerate(files):
        # Source identity changes if an unfinished local input grows/changes.
        stat = path.stat()
        base_id = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
        if path.suffix.lower() == ".jsonl":
            with path.open("rb") as handle:
                line_number = 0
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    line_number += 1
                    if not line.strip():
                        continue
                    discovered += 1
                    if time.monotonic() - last_log >= 20:
                        elapsed = time.monotonic() - started
                        read = handle.tell()
                        print(f"[discover] file={index + 1}/{len(files)} records={discovered} "
                              f"bytes={total_bytes + read} file_bytes={read}/{stat.st_size} "
                              f"percent={100 * read / stat.st_size:.1f}% rows_per_s={discovered / elapsed:.1f} "
                              f"elapsed={elapsed:.1f}s eta=unknown current={path}", flush=True)
                        last_log = time.monotonic()
                    yield {"kind": "jsonl", "path": str(path), "offset": offset, "size": len(line),
                           "source": {"source_id": f"{base_id}:{line_number}", "path": str(path),
                                      "line": line_number, "license": "provided_by_user"}}
        else:
            yield {"kind": "file", "path": str(path),
                   "source": {"source_id": base_id, "path": str(path), "license": "provided_by_user"}}
        total_bytes += stat.st_size
        print(f"[discover] file={index + 1}/{len(files)} bytes={total_bytes} current={path}", flush=True)


_EVENTS = None
_COUNTER = None
_STOP = None
_DOWNLOAD_SLOTS = None


def initialize_worker(events, stop, download_slots, tokenizer):
    global _EVENTS, _COUNTER, _STOP, _DOWNLOAD_SLOTS
    _EVENTS, _STOP, _DOWNLOAD_SLOTS = events, stop, download_slots
    # At a user stop/target, pending (not yet persisted) events may be discarded;
    # do not let worker exit wait on a full queue. Complete documents are only
    # checkpointed after the parent has consumed all their sample events.
    events.cancel_join_thread()
    _COUNTER = load_counter(tokenizer)


def emit(event):
    while not _STOP.is_set():
        try:
            _EVENTS.put(event, timeout=0.5)
            return True
        except queue.Full:
            pass
    return False


def process_document(job, config):
    source = dict(job["source"])
    key = source["source_id"]
    read_bytes = 0
    rejected = Counter()
    emitted = 0
    try:
        if _STOP.is_set():
            return
        if job["kind"] == "public":
            with _DOWNLOAD_SLOTS:
                raw = request_bytes(job["url"])
        elif job["kind"] == "jsonl":
            with Path(job["path"]).open("rb") as handle:
                handle.seek(job["offset"])
                raw = handle.read(job["size"])
        else:
            raw = Path(job["path"]).read_bytes()
        read_bytes = len(raw)
        if job["kind"] == "jsonl":
            data = json.loads(raw)
            text = data.get("markdown", data.get("text"))
            if not isinstance(text, str):
                raise ValueError("JSONL row requires a string markdown or text field")
            source.update({k: data[k] for k in ("url", "title", "license", "license_url", "attribution") if k in data})
        else:
            text = raw.decode("utf-8")  # Preserve CRLF and all original whitespace.
        source["document_sha256"] = digest(text)
        # Keep headroom for both fences and possible tokenization changes.
        budget = max(1, int(config["max_response_tokens"] * 0.92) - 16)
        for chunk_index, (start, end, chunk) in enumerate(split_document(
                text, count_tokens=_COUNTER, max_tokens=budget,
                seed=seed_for(config["seed"], source["document_sha256"], "windows"))):
            if _STOP.is_set():
                return
            if chunk is None:
                rejected["oversized_indivisible_block"] += 1
                continue
            chunk_source = {**source, "document_char_start": start, "document_char_end": end}
            row = make_sample(chunk, source=chunk_source, config=config,
                              chunk_index=chunk_index, count_tokens=_COUNTER)
            if row is None:
                rejected["no_safe_mutation_or_response_length"] += 1
            else:
                if not emit({"type": "sample", "document": key, "row": row}):
                    return
                emitted += 1
        emit({"type": "done", "document": key, "bytes": read_bytes,
              "emitted": emitted, "rejections": dict(rejected)})
    except Exception as exc:  # noqa: BLE001 - report the failed unit; continue the corpus
        emit({"type": "error", "document": key, "bytes": read_bytes,
              "error": f"{type(exc).__name__}: {exc}"})


def durable_append(handle, value):
    # One parent writer, one complete encoded line per write. An interrupted
    # trailing write is removed on resume; complete lines are never discarded.
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    handle.flush()
    os.fsync(handle.fileno())


def read_completed(path):
    """Repair only an incomplete trailing JSONL write, then stream full rows."""
    if not path.exists():
        return
    last_log = time.monotonic()
    with path.open("r+b") as handle:
        count = 0
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                return
            if not line.endswith(b"\n"):
                handle.truncate(start)
                handle.flush()
                os.fsync(handle.fileno())
                print(f"[resume] removed incomplete trailing write path={path} offset={start}", flush=True)
                return
            yield json.loads(line)
            count += 1
            if time.monotonic() - last_log >= 20:
                print(f"[resume] file={path.name} rows={count} bytes={handle.tell()}/{path.stat().st_size}", flush=True)
                last_log = time.monotonic()


def run(args):
    started = time.monotonic()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = output / ".state"
    state.mkdir(exist_ok=True)
    config = {"version": VERSION, "tokenizer": args.tokenizer,
              "mutation_ratio": args.mutation_ratio, "min_response_tokens": args.min_response_tokens,
              "max_response_tokens": args.max_response_tokens, "seed": args.seed,
              "val_fraction": args.val_fraction, "inputs": args.input,
              "public_sources": args.public_sources,
              "pairs_sha256": digest(json.dumps(DEFAULT_PAIRS, ensure_ascii=False, sort_keys=True))}
    print(f"[start] output={output} workers={args.workers} mutation_ratio={args.mutation_ratio} "
          f"response_tokens={args.min_response_tokens}..{args.max_response_tokens} "
          f"max_samples={args.max_samples or 'all'} streaming=train.jsonl,val.jsonl", flush=True)
    if args.tokenizer == "simple":
        print("[warning] tokenizer=simple is character counting for TESTS ONLY, not training token limits", flush=True)
    load_counter(args.tokenizer)("中文 tokenizer 检查")
    config_path = state / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Generation settings changed; use a new --output-dir")
    else:
        atomic_json(config_path, config)
    seen, hashes = set(), set()
    split_counts = Counter()
    saved_bytes = 0
    for split in ("train", "val"):
        path = output / f"{split}.jsonl"
        for row in read_completed(path):
            info = row["extra_info"]
            seen.add(info["sample_id"])
            hashes.add(info["source_text_sha256"])
            split_counts[split] += 1
        saved_bytes += path.stat().st_size if path.exists() else 0
    done = {row["document"] for row in read_completed(state / "completed_documents.jsonl")}
    public_jobs = (public_manifest(args.public_sources, state) if args.public_sources else [])
    random.Random(args.seed).shuffle(public_jobs)
    # Do not scan a large local JSONL to the end before generating its first row.
    pending_jobs = chain(public_jobs, local_jobs(args.input, excluded_dir=output))
    if args.max_documents:
        pending_jobs = islice(pending_jobs, args.max_documents)
    total = None if args.input else min(len(public_jobs), args.max_documents or len(public_jobs))
    discovered = prior_done = processed = 0
    discovery_finished = False
    errors = duplicates = received = processed_bytes = 0
    rejections = Counter()
    mutation_pairs = Counter()
    actual_han = actual_mutations = 0
    new_saved = 0
    last_progress = time.monotonic()

    def progress(current, phase="progress"):
        nonlocal last_progress
        elapsed = time.monotonic() - started
        rate = (processed - prior_done) / elapsed if elapsed else 0
        eta = (total - processed) / rate if rate and total is not None else None
        percent = f"{100 * processed / total if total else 100:.1f}%" if total is not None else "unknown"
        print(f"[{phase}] documents={processed}/{total if total is not None else 'unknown'} "
              f"discovered={discovered} percent={percent} "
              f"accepted_saved={len(seen)} new_saved={new_saved} received={received} "
              f"train={split_counts['train']} val={split_counts['val']} "
              f"rejected={sum(rejections.values())} duplicate={duplicates} errors={errors} "
              f"input_bytes={processed_bytes} output_bytes={saved_bytes} "
              f"samples_per_s={new_saved / elapsed:.2f} documents_per_s={rate:.2f} "
              f"elapsed={elapsed:.1f}s eta={f'{eta:.1f}s' if eta is not None else 'unknown'} "
              f"current={current}", flush=True)
        last_progress = time.monotonic()

    ctx = mp.get_context("spawn")
    events = ctx.Queue(maxsize=max(2, args.workers * 2))
    stop = ctx.Event()
    download_slots = ctx.BoundedSemaphore(min(args.workers, 8))
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                   initializer=initialize_worker,
                                   initargs=(events, stop, download_slots, args.tokenizer))
    pending = {}
    interrupted = False
    target_reached = bool(args.max_samples and len(seen) >= args.max_samples)
    handles = {s: (output / f"{s}.jsonl").open("ab") for s in ("train", "val")}
    checkpoint = (state / "completed_documents.jsonl").open("ab")
    error_log = (state / "errors.jsonl").open("ab")

    def submit():
        nonlocal total, discovered, processed, prior_done, discovery_finished
        while len(pending) < args.workers * 2 and not target_reached and not discovery_finished:
            job = next(pending_jobs, None)
            if job is None:
                total = discovered
                discovery_finished = True
                break
            discovered += 1
            key = job["source"]["source_id"]
            if key in done:
                processed += 1
                prior_done += 1
                if time.monotonic() - last_progress >= 20:
                    progress(key, "resume")
                continue
            pending[key] = executor.submit(process_document, job, config)

    try:
        submit()
        progress("starting")
        while pending and not target_reached:
            try:
                event = events.get(timeout=1)
            except queue.Empty:
                # Surface initializer/process crashes; do not wait forever for a
                # done event that a dead process can never send.
                for key, future in list(pending.items()):
                    if future.done() and future.exception() is not None:
                        raise RuntimeError(f"worker failed: {key}") from future.exception()
                if time.monotonic() - last_progress >= 20:
                    progress(next(iter(pending)))
                continue
            kind, key = event["type"], event["document"]
            if kind == "sample":
                received += 1
                row = event["row"]
                info = row["extra_info"]
                if info["sample_id"] in seen or info["source_text_sha256"] in hashes:
                    duplicates += 1
                    continue
                # Split by the whole original document, never by its windows.
                fraction = int(info["source"]["document_sha256"][:16], 16) / 2**64
                split = "val" if fraction < args.val_fraction else "train"
                durable_append(handles[split], row)
                seen.add(info["sample_id"])
                hashes.add(info["source_text_sha256"])
                split_counts[split] += 1
                new_saved += 1
                saved_bytes += len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
                actual_han += info["prose_han_chars"]
                actual_mutations += info["actual_mutations"]
                mutation_pairs.update(f"{c['origin_ans']}->{c['ocr_ans']}" for c in info["changes"])
                target_reached = bool(args.max_samples and len(seen) >= args.max_samples)
                progress(f"{key}:sample={info['sample_id'][:12]}", "saved")
            else:
                processed += 1
                processed_bytes += event["bytes"]
                pending.pop(key)
                if kind == "done":
                    rejections.update(event["rejections"])
                    durable_append(checkpoint, event)
                else:
                    errors += 1
                    durable_append(error_log, event)
                progress(key, "document-complete" if kind == "done" else "document-error")
                submit()
    except KeyboardInterrupt:
        interrupted = True
        print("[interrupt] stopping workers; already saved train/val rows remain usable", flush=True)
    finally:
        stop.set()
        for future in pending.values():
            future.cancel()
        # Drain blocked queue writers while workers observe the stop flag.
        while any(not f.done() for f in pending.values()):
            try:
                events.get(timeout=0.2)
            except queue.Empty:
                pass
            if time.monotonic() - last_progress >= 20:
                progress("stopping-workers")
        executor.shutdown(wait=True, cancel_futures=True)
        for handle in [*handles.values(), checkpoint, error_log]:
            handle.close()
        events.close()
    summary = {"status": "interrupted" if interrupted else "completed", "version": VERSION,
               "documents_selected": total, "documents_discovered": discovered,
               "discovery_finished": discovery_finished, "documents_completed_or_error": processed,
               "accepted_saved": len(seen), "new_saved": new_saved, "splits": dict(split_counts),
               "rejections_this_run": dict(rejections), "errors_this_run": errors,
               "duplicates_this_run": duplicates, "target_reached": target_reached,
               "mutation_pairs_this_run": dict(mutation_pairs),
               "mutation_ratio_achieved_this_run": actual_mutations / actual_han if actual_han else None,
               "elapsed_seconds": time.monotonic() - started,
               "train": str(output / "train.jsonl"), "val": str(output / "val.jsonl")}
    atomic_json(output / "summary.json", summary)
    progress("finished", "finish")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", default=[], help="Local MD/TXT/JSONL files or directories")
    parser.add_argument("--public-sources", nargs="+", choices=sorted(PUBLIC_SOURCES), default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True, help="Local model folder or tokenizer.json; simple is TEST ONLY")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--mutation-ratio", type=float, default=0.02)
    parser.add_argument("--min-response-tokens", type=int, default=1000)
    parser.add_argument("--max-response-tokens", type=int, default=7800)
    parser.add_argument("--max-samples", type=int, default=0, help="Total final rows including resumed rows; 0 = all available")
    parser.add_argument("--max-documents", type=int, default=0, help="Deterministic source subset for smoke tests; 0 = all")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    args = parser.parse_args(argv)
    if not args.input and not args.public_sources:
        args.public_sources = ["d2l-zh", "oi-wiki"]
    if args.workers < 1 or not 0 < args.mutation_ratio <= 1 or not 0 <= args.val_fraction < 1:
        parser.error("workers >= 1, 0 < mutation-ratio <= 1, 0 <= val-fraction < 1 required")
    if not 1 <= args.min_response_tokens <= args.max_response_tokens or min(args.max_samples, args.max_documents) < 0:
        parser.error("invalid length/count limits")
    summary = run(args)
    return 130 if summary["status"] == "interrupted" else (1 if summary["errors_this_run"] else 0)


if __name__ == "__main__":
    sys.exit(main())
