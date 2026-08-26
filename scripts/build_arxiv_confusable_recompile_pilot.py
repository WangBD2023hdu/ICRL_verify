#!/usr/bin/env python3
"""Build source-recompiled arXiv OCR pairs with 3--4 letter confusions/page.

The clean page/Markdown inputs must already have passed strict-text-v2.  Each
accepted pair applies one character substitution in each of three or four
distinct visible words, recompiles the copied LaTeX source, and keeps the
typos in the edited Markdown ground truth.  No source archive is modified.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Sequence

import pdfplumber
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_mm_token_probe.prompts import DEFAULT_PDF_OCR_PROMPT  # noqa: E402


DEFAULT_RECOMPILE_ROOT = Path("outputs/arxiv_latex_recompile_10")
DEFAULT_CLEAN_GT_ROOT = Path("outputs/arxiv_page_markdown_gt_10_caption_separate")
DEFAULT_OUTPUT = Path("outputs/arxiv_confusable_recompile_10")
LATEXMK = Path(shutil.which("latexmk") or "/Library/TeX/texbin/latexmk")
PDFTOPPM = Path(shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm")
SCHEMA_VERSION = 2
MUTATION_POLICY_VERSION = "chaos_visual_v2"
SELECTION_POLICY_VERSION = (
    "page_exact_source_paragraph_v6_rendered_line_spread_current_gt_no_bibliography"
)
BIBLIOGRAPHY_POLICY_VERSION = "exclude_bibliography_tail_v1"
STRICT_INPUT_FILTER_POLICY_VERSION = "strict_gt_current_contract_v1"
STRICT_INPUT_STRICT_TEXT_CONTRACT_VERSION = 2
STRICT_INPUT_AUTHOR_SUPERSCRIPT_CONTRACT_VERSION = 5
STRICT_INPUT_FOOTNOTE_REPRESENTATION = "html_sup"
SOURCE_FIRST_INPUT_POLICY_VERSION = "source_first_color_v6_literal_markdown_v5"
SOURCE_FIRST_SCHEMA_VERSION = 6
SOURCE_FIRST_CONTRACT = "source_first_color_v6"
SOURCE_FIRST_VERIFIER_CONTRACT_VERSION = 4
SOURCE_FIRST_PROBE_POLICY_VERSION = (
    "paragraph_list_payload_then_paragraph_then_whole_v2"
)
SOURCE_FIRST_SHADOW_INVARIANT_POLICY_VERSION = "exact_page_character_sequence_v1"
SOURCE_FIRST_HEADING_LABEL_POLICY_VERSION = "aux_number_unique_titleformat_label_v1"
PAIR_POLICY_TAG = "chaosv4"
HEARTBEAT_SECONDS = 15.0
WORD_RE = re.compile(r"[A-Za-z]{4,}")
BIBLIOGRAPHY_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"(?:(?:appendix\s+)?[A-Z0-9]+(?:\.[A-Z0-9]+)*[.)]?\s+)?"
    r"(?:references|bibliography|works\s+cited|literature\s+cited)\s*:?[\s*_]*$",
    re.IGNORECASE,
)
VERL_PROMPT = (
    "<image>\nPlease transcribe all text in this page image faithfully, "
    "exactly as printed (including any typos)."
)

# Lower-case alphabetic, one-codepoint-to-one-codepoint substitutions only.
# Digits, Unicode homoglyphs, and length-changing confusions are forbidden.
CONFUSABLES: dict[str, tuple[str, ...]] = {
    "a": ("o",),
    "c": ("e", "o"),
    "e": ("c",),
    "g": ("q",),
    "h": ("n",),
    "i": ("l",),
    "l": ("i",),
    "n": ("h",),
    "o": ("a", "c"),
    "q": ("g",),
    "s": ("z",),
    "u": ("v",),
    "v": ("u",),
    "z": ("s",),
}


@dataclasses.dataclass(frozen=True)
class SourceOccurrence:
    word: str
    source_file: str
    word_offset: int
    source_line: int
    source_column: int
    paragraph_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class PdfWord:
    text: str
    page_number: int
    word_index: int
    x0: float
    top: float
    x1: float
    bottom: float


@dataclasses.dataclass(frozen=True)
class Mutation:
    original_word: str
    mutated_word: str
    from_char: str
    to_char: str
    char_index_in_word: int
    source_file: str
    source_word_offset: int
    source_char_offset: int
    source_line: int
    source_column: int
    page_number: int
    pdf_word_index: int
    clean_bbox_points: tuple[float, float, float, float]
    markdown_start: int
    markdown_end: int

    def as_selection_json(self) -> dict[str, Any]:
        return {
            "origin_ans": self.original_word,
            "ocr_ans": self.mutated_word,
            "from_char": self.from_char,
            "to_char": self.to_char,
            "char_index_in_word": self.char_index_in_word,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "source_column": self.source_column,
            "page_number": self.page_number,
            "pdf_word_index": self.pdf_word_index,
            "clean_bbox_points": [round(value, 3) for value in self.clean_bbox_points],
            "markdown_span": [self.markdown_start, self.markdown_end],
        }


class Progress:
    def __init__(self, total_papers: int, total_pages: int) -> None:
        self.total_papers = total_papers
        self.total_pages = total_pages
        self.started = time.monotonic()
        self.pages_completed = 0
        self.accepted = 0
        self.rejected = 0
        self.errors = 0

    def emit(self, phase: str, current: str = "", detail: str = "") -> None:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        pct = 100.0 * self.pages_completed / max(1, self.total_pages)
        throughput = self.pages_completed / elapsed
        remaining = max(0, self.total_pages - self.pages_completed)
        eta = remaining / throughput if throughput > 0 else math.inf
        eta_text = "unknown" if not math.isfinite(eta) else elapsed_text(eta)
        print(
            f"[progress] phase={phase} current={current or '-'} "
            f"pages={self.pages_completed}/{self.total_pages} pct={pct:.1f}% "
            f"accepted={self.accepted} rejected={self.rejected} errors={self.errors} "
            f"throughput={throughput:.3f} pages/s elapsed={elapsed_text(elapsed)} "
            f"eta={eta_text} {detail}".rstrip(),
            flush=True,
        )


def elapsed_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}m{seconds % 60:02d}s"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for count, row in enumerate(rows, start=1):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def stable_seed(seed: int, label: str) -> int:
    payload = f"{seed}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def tex_comment_start(line: str) -> int:
    for index, character in enumerate(line):
        if character != "%":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return index
    return len(line)


def source_occurrences(source_root: Path, paragraph_rows: Sequence[dict[str, Any]]) -> list[SourceOccurrence]:
    """Return candidate source words from executed, visible paragraph ranges."""

    by_file: dict[str, dict[tuple[int, int], set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    for row in paragraph_rows:
        relative = str(row.get("source_file", ""))
        source_lines = row.get("source_lines") or []
        if relative and len(source_lines) == 2:
            paragraph_id = str(row.get("source_paragraph_id", ""))
            if paragraph_id:
                by_file[relative][
                    (int(source_lines[0]), int(source_lines[1]))
                ].add(paragraph_id)
    occurrence_values: dict[tuple[str, int], tuple[str, int, int]] = {}
    occurrence_paragraph_ids: dict[tuple[str, int], set[str]] = collections.defaultdict(set)
    for relative, ranges_with_ids in sorted(by_file.items()):
        path = source_root / relative
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        line_offsets: list[int] = []
        offset = 0
        for line in lines:
            line_offsets.append(offset)
            offset += len(line)
        for (start_line, end_line), paragraph_ids in sorted(ranges_with_ids.items()):
            for line_number in range(start_line, min(end_line, len(lines)) + 1):
                line = lines[line_number - 1]
                visible = line[: tex_comment_start(line)]
                command_spans = [
                    (match.start(), match.end())
                    for match in re.finditer(r"\\[A-Za-z@]+", visible)
                ]
                dollar_positions = [
                    index
                    for index, character in enumerate(visible)
                    if character == "$" and (index == 0 or visible[index - 1] != "\\")
                ]
                for match in WORD_RE.finditer(visible):
                    word = match.group(0)
                    if not any(character in CONFUSABLES for character in word):
                        continue
                    if any(left <= match.start() < right for left, right in command_spans):
                        continue
                    # Initial pilot mutates prose only; inline math is handled
                    # by a separate policy in order not to corrupt TeX syntax.
                    if sum(position < match.start() for position in dollar_positions) % 2:
                        continue
                    absolute = line_offsets[line_number - 1] + match.start()
                    key = (relative, absolute)
                    occurrence_values[key] = (word, line_number, match.start() + 1)
                    occurrence_paragraph_ids[key].update(paragraph_ids)
    occurrences = [
        SourceOccurrence(
            word=value[0],
            source_file=key[0],
            word_offset=key[1],
            source_line=value[1],
            source_column=value[2],
            paragraph_ids=tuple(sorted(occurrence_paragraph_ids[key])),
        )
        for key, value in occurrence_values.items()
    ]
    return sorted(
        occurrences, key=lambda item: (item.source_file, item.word_offset)
    )


def extract_pdf_words(
    pdf_path: Path,
    *,
    progress: Progress | None = None,
    current: str = "",
    phase: str = "pdf_extract",
) -> tuple[dict[int, list[PdfWord]], dict[str, int]]:
    by_page: dict[int, list[PdfWord]] = {}
    counts: collections.Counter[str] = collections.Counter()
    state: dict[str, int] = {"page": 0, "total": 0, "words": 0}
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
            if progress is not None:
                progress.emit(
                    phase,
                    current,
                    f"pdf_page={state['page']}/{state['total']} words={state['words']} "
                    f"file={pdf_path}",
                )

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    if progress is not None:
        heartbeat_thread.start()
    try:
        with pdfplumber.open(pdf_path) as document:
            state["total"] = len(document.pages)
            for page_number, page in enumerate(document.pages, start=1):
                state["page"] = page_number
                raw_words = page.extract_words(
                    x_tolerance=1,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                    split_at_punctuation=True,
                )
                page_words: list[PdfWord] = []
                for word_index, word in enumerate(raw_words):
                    text = str(word.get("text", ""))
                    page_words.append(
                        PdfWord(
                            text=text,
                            page_number=page_number,
                            word_index=word_index,
                            x0=float(word["x0"]),
                            top=float(word["top"]),
                            x1=float(word["x1"]),
                            bottom=float(word["bottom"]),
                        )
                    )
                    if WORD_RE.fullmatch(text):
                        counts[text] += 1
                by_page[page_number] = page_words
                state["words"] += len(page_words)
    finally:
        stop_heartbeat.set()
        if heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=1.0)
    if progress is not None:
        progress.emit(
            f"{phase}_done",
            current,
            f"pdf_pages={state['total']} words={state['words']} file={pdf_path}",
        )
    return by_page, dict(counts)


def whole_word_spans(text: str, word: str) -> list[tuple[int, int]]:
    pattern = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def choose_mutations_for_page(
    *,
    row: dict[str, Any],
    clean_markdown: str,
    page_words: Sequence[PdfWord],
    source_by_word: dict[str, list[SourceOccurrence]],
    paper_word_vocabulary: set[str],
    excluded_source_positions: set[tuple[str, int]],
    seed: int,
) -> list[Mutation]:
    candidates: list[tuple[SourceOccurrence, PdfWord, int, int]] = []
    allowed_paragraph_ids = set(
        row.get("source_paragraph_integration", {}).get("source_paragraph_ids", [])
    )
    page_counts = collections.Counter(
        word.text for word in page_words if WORD_RE.fullmatch(word.text)
    )
    for pdf_word in page_words:
        word = pdf_word.text
        # A page-local unique occurrence is sufficient because the source
        # occurrence is also unique and the edited PDF is validated globally
        # after recompilation.  The old document-global uniqueness condition
        # discarded safe words merely because the same spelling occurred on a
        # different page.
        if not WORD_RE.fullmatch(word) or page_counts.get(word) != 1:
            continue
        source_matches = source_by_word.get(word, [])
        if allowed_paragraph_ids:
            source_matches = [
                occurrence
                for occurrence in source_matches
                if allowed_paragraph_ids.intersection(occurrence.paragraph_ids)
            ]
        if len(source_matches) != 1:
            continue
        source_match = source_matches[0]
        if (source_match.source_file, source_match.word_offset) in excluded_source_positions:
            continue
        markdown_spans = whole_word_spans(clean_markdown, word)
        if len(markdown_spans) != 1:
            continue
        # Avoid line-end words: small width changes then cannot trigger TeX
        # reflow in the overwhelming majority of cases.
        same_line_followers = [
            other
            for other in page_words
            if abs(other.top - pdf_word.top) <= 1.0 and other.x0 > pdf_word.x1
        ]
        if not same_line_followers:
            continue
        candidates.append(
            (source_match, pdf_word, markdown_spans[0][0], markdown_spans[0][1])
        )
    rng = random.Random(stable_seed(seed, str(row["data_id"])))
    rng.shuffle(candidates)
    requested = 4 if rng.random() < 0.6 else 3
    selected: list[Mutation] = []
    selected_words: set[str] = set()
    selected_mutated_words: set[str] = set()
    selected_visual_lines: list[tuple[int, float]] = []
    for source, pdf_word, md_start, md_end in candidates:
        if source.word in selected_words:
            continue
        eligible_positions = [
            index for index, character in enumerate(source.word) if character in CONFUSABLES
        ]
        rng.shuffle(eligible_positions)
        mutation: Mutation | None = None
        for character_index in eligible_positions:
            from_char = source.word[character_index]
            targets = list(CONFUSABLES[from_char])
            rng.shuffle(targets)
            for to_char in targets:
                mutated_word = (
                    source.word[:character_index]
                    + to_char
                    + source.word[character_index + 1 :]
                )
                if (
                    mutated_word in paper_word_vocabulary
                    or mutated_word in selected_mutated_words
                    or len(mutated_word) != len(source.word)
                ):
                    continue
                mutation = Mutation(
                    original_word=source.word,
                    mutated_word=mutated_word,
                    from_char=from_char,
                    to_char=to_char,
                    char_index_in_word=character_index,
                    source_file=source.source_file,
                    source_word_offset=source.word_offset,
                    source_char_offset=source.word_offset + character_index,
                    source_line=source.source_line,
                    source_column=source.source_column + character_index,
                    page_number=int(row["page_number"]),
                    pdf_word_index=pdf_word.word_index,
                    clean_bbox_points=(pdf_word.x0, pdf_word.top, pdf_word.x1, pdf_word.bottom),
                    markdown_start=md_start,
                    markdown_end=md_end,
                )
                break
            if mutation is not None:
                break
        if mutation is None:
            continue
        # A TeX author often writes an entire paragraph on one source line, so
        # source-line uniqueness can make a visually dense page impossible to
        # mutate.  Spread edits over distinct rendered lines instead.  This is
        # the layout property that matters, and the post-compile document gate
        # still rejects any resulting reflow or word-sequence change.
        if any(
            page_number == mutation.page_number
            and abs(top - pdf_word.top) <= 1.0
            for page_number, top in selected_visual_lines
        ):
            continue
        selected.append(mutation)
        selected_words.add(mutation.original_word)
        selected_mutated_words.add(mutation.mutated_word)
        selected_visual_lines.append((mutation.page_number, pdf_word.top))
        if len(selected) >= requested:
            break
    if len(selected) < 3:
        return []
    return selected[:requested]


def apply_source_mutations(source_root: Path, mutations: Sequence[Mutation]) -> None:
    by_file: dict[str, list[Mutation]] = collections.defaultdict(list)
    for mutation in mutations:
        by_file[mutation.source_file].append(mutation)
    for relative, file_mutations in sorted(by_file.items()):
        path = source_root / relative
        text = path.read_text(encoding="utf-8")
        for mutation in sorted(
            file_mutations, key=lambda item: item.source_char_offset, reverse=True
        ):
            offset = mutation.source_char_offset
            actual = text[offset : offset + 1]
            if actual != mutation.from_char:
                raise ValueError(
                    f"source character mismatch {relative}:{offset}: "
                    f"expected={mutation.from_char!r} actual={actual!r}"
                )
            text = text[:offset] + mutation.to_char + text[offset + 1 :]
        atomic_write_text(path, text)


def apply_markdown_mutations(clean: str, mutations: Sequence[Mutation]) -> str:
    edited = clean
    for mutation in sorted(mutations, key=lambda item: item.markdown_start, reverse=True):
        actual = edited[mutation.markdown_start : mutation.markdown_end]
        if actual != mutation.original_word:
            raise ValueError(
                f"Markdown span mismatch: expected={mutation.original_word!r} actual={actual!r}"
            )
        edited = (
            edited[: mutation.markdown_start]
            + mutation.mutated_word
            + edited[mutation.markdown_end :]
        )
    return edited


def tool_env() -> dict[str, str]:
    env = os.environ.copy()
    tex_bin = str(LATEXMK.parent)
    env["PATH"] = tex_bin + os.pathsep + env.get("PATH", "")
    return env


def run_logged_with_heartbeat(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
    progress: Progress,
    current: str,
) -> tuple[int, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=tool_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        next_heartbeat = started + HEARTBEAT_SECONDS
        while process.poll() is None:
            now = time.monotonic()
            if now - started > timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                break
            if now >= next_heartbeat:
                progress.emit(
                    "compile",
                    current,
                    f"compile_elapsed={elapsed_text(now-started)}",
                )
                next_heartbeat = now + HEARTBEAT_SECONDS
            time.sleep(0.25)
        return_code = process.returncode if process.returncode is not None else -9
    return return_code, timed_out, time.monotonic() - started


def compile_edited_source(
    *,
    source_root: Path,
    main_relative: Path,
    engine: str,
    build_dir: Path,
    progress: Progress,
    paper_id: str,
    timeout_seconds: int,
) -> tuple[Path, dict[str, Any]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    main_path = source_root / main_relative
    if engine == "xelatex":
        flag = "-xelatex"
    elif engine == "latex_dvips_ps2pdf":
        flag = "-pdfps"
    else:
        flag = "-pdf"
    command = [
        str(LATEXMK),
        "-norc",
        "-g",
        flag,
        "-synctex=1",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-no-shell-escape",
        f"-outdir={build_dir.resolve()}",
        main_path.name,
    ]
    log_path = build_dir / "compile_edited.log"
    return_code, timed_out, duration = run_logged_with_heartbeat(
        command,
        cwd=main_path.parent,
        log_path=log_path,
        timeout_seconds=timeout_seconds,
        progress=progress,
        current=paper_id,
    )
    pdf_path = build_dir / f"{main_path.stem}.pdf"
    compile_info = {
        "status": "passed"
        if return_code == 0 and not timed_out and pdf_path.is_file()
        else "failed",
        "engine": engine,
        "command": command,
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_seconds": round(duration, 3),
        "log": str(log_path),
        "pdf": str(pdf_path),
    }
    if compile_info["status"] != "passed":
        raise RuntimeError(
            f"edited compile failed paper={paper_id} rc={return_code} "
            f"timeout={timed_out} log={log_path}"
        )
    return pdf_path, compile_info


def render_page(pdf_path: Path, page_number: int, output_png: Path, dpi: int) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    prefix = output_png.with_suffix("")
    command = [
        str(PDFTOPPM),
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-singlefile",
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(prefix),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if completed.returncode != 0 or not output_png.is_file() or output_png.stat().st_size == 0:
        raise RuntimeError(
            f"pdftoppm failed page={page_number} rc={completed.returncode}: "
            f"{completed.stderr.strip()}"
        )


def validate_page_words(
    clean_words: Sequence[PdfWord],
    edited_words: Sequence[PdfWord],
    mutations: Sequence[Mutation],
) -> tuple[bool, str, float]:
    if len(clean_words) != len(edited_words):
        return False, f"word_count_changed:{len(clean_words)}->{len(edited_words)}", math.inf
    expected = [word.text for word in clean_words]
    for mutation in mutations:
        index = mutation.pdf_word_index
        if index >= len(expected) or expected[index] != mutation.original_word:
            return False, f"clean_word_index_mismatch:{mutation.original_word}", math.inf
        expected[index] = mutation.mutated_word
    actual = [word.text for word in edited_words]
    if actual != expected:
        mismatch = next(
            (
                index
                for index, (expected_word, actual_word) in enumerate(zip(expected, actual))
                if expected_word != actual_word
            ),
            -1,
        )
        return False, f"edited_word_sequence_mismatch:index={mismatch}", math.inf
    max_vertical_shift = max(
        (abs(clean.top - edited.top) for clean, edited in zip(clean_words, edited_words)),
        default=0.0,
    )
    if max_vertical_shift > 1.25:
        return False, f"line_reflow:max_vertical_shift={max_vertical_shift:.3f}", max_vertical_shift
    return True, "passed", max_vertical_shift


def validate_document_words(
    clean_words_by_page: dict[int, list[PdfWord]],
    edited_words_by_page: dict[int, list[PdfWord]],
    page_mutations: dict[int, list[Mutation]],
) -> tuple[bool, str, float]:
    """Require the entire edited PDF to differ only at requested words."""

    if set(clean_words_by_page) != set(edited_words_by_page):
        return False, "document_page_set_changed", math.inf
    maximum_shift = 0.0
    for page_number in sorted(clean_words_by_page):
        valid, reason, shift = validate_page_words(
            clean_words_by_page[page_number],
            edited_words_by_page[page_number],
            page_mutations.get(page_number, []),
        )
        if not valid:
            return False, f"document_page_{page_number}:{reason}", shift
        maximum_shift = max(maximum_shift, shift)
    return True, "passed", maximum_shift


def point_bbox_to_pixel_bbox(
    bbox: Sequence[float],
    *,
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> list[int]:
    x_scale = image_width / page_width
    y_scale = image_height / page_height
    return [
        max(0, min(image_width, math.floor(float(bbox[0]) * x_scale))),
        max(0, min(image_height, math.floor(float(bbox[1]) * y_scale))),
        max(0, min(image_width, math.ceil(float(bbox[2]) * x_scale))),
        max(0, min(image_height, math.ceil(float(bbox[3]) * y_scale))),
    ]


def markdown_diff_count(clean: str, edited: str) -> int:
    if len(clean) != len(edited):
        return -1
    return sum(left != right for left, right in zip(clean, edited))


def output_artifact_path(output_dir: Path, relative: str) -> str:
    """Return the absolute path of a materialized artifact inside output_dir."""

    root = output_dir.expanduser().resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output artifact escapes output directory: {relative}") from exc
    if not target.is_file():
        raise FileNotFoundError(f"output artifact does not exist: {target}")
    return str(target)


def split_by_paper(
    pair_rows: Sequence[dict[str, Any]], seed: int, val_fraction: float = 0.05
) -> dict[str, str]:
    papers = sorted({str(row["paper_id"]) for row in pair_rows})
    shuffled = list(papers)
    random.Random(seed).shuffle(shuffled)
    val_count = (
        min(len(shuffled) - 1, max(1, round(len(shuffled) * val_fraction)))
        if len(shuffled) > 1
        else 0
    )
    val_papers = set(shuffled[:val_count])
    return {paper: ("val" if paper in val_papers else "train") for paper in papers}


def export_training(
    *,
    output_dir: Path,
    pair_rows: list[dict[str, Any]],
    split_seed: int,
    val_fraction: float,
) -> dict[str, Any]:
    sft_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    paper_splits = split_by_paper(pair_rows, split_seed, val_fraction)
    sft_name = f"SFT_edited_{len(pair_rows)}.jsonl"
    for row in pair_rows:
        edited_markdown = (output_dir / row["edited_markdown"]).read_text(encoding="utf-8")
        image_path = output_artifact_path(output_dir, row["edited_image"])
        sft_rows.append(
            {
                "images": [image_path],
                "conversations": [
                    {"from": "human", "value": DEFAULT_PDF_OCR_PROMPT},
                    {"from": "gpt", "value": edited_markdown},
                ],
            }
        )
        changes = [
            {
                "ocr_ans": change["ocr_ans"],
                "origin_ans": change["origin_ans"],
                "bbox": change["bbox"],
            }
            for change in row["changes"]
        ]
        verl_row = {
            "data_source": "chaos_document_ocr",
            "prompt": [{"role": "user", "content": VERL_PROMPT}],
            "images": [image_path],
            "reward_model": {"style": "rule", "ground_truth": edited_markdown},
            "extra_info": {
                "arxiv_id": row["arxiv_id"],
                "pair_id": row["pair_id"],
                "changes": changes,
            },
            "ability": "document_ocr",
        }
        grouped[paper_splits[row["paper_id"]]].append(verl_row)
    write_jsonl(output_dir / sft_name, sft_rows)
    write_jsonl(output_dir / "verl_grpo" / "train.jsonl", grouped["train"])
    write_jsonl(output_dir / "verl_grpo" / "val.jsonl", grouped["val"])
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for VERL parquet export") from exc
    for split in ("train", "val"):
        pq.write_table(
            pa.Table.from_pylist(grouped[split]),
            output_dir / "verl_grpo" / f"{split}.parquet",
            compression="zstd",
        )
    return {
        "sft": sft_name,
        "train": len(grouped["train"]),
        "val": len(grouped["val"]),
        "paper_splits": paper_splits,
    }


def load_recompile_results(root: Path) -> dict[str, dict[str, Any]]:
    return {str(row["stem"]): row for row in read_jsonl(root / "results.jsonl")}


def resolve_recompile_source(
    recompile_root: Path,
    paper_id: str,
    recompile: dict[str, Any],
) -> tuple[Path, Path]:
    """Resolve source/main paths after a corpus is moved to another machine."""

    recorded_source = Path(str(recompile["source_dir"]))
    recorded_main = Path(str(recompile["main_tex"]))
    if recorded_source.is_dir() and recorded_main.is_file():
        return recorded_source.resolve(), recorded_main.resolve().relative_to(recorded_source.resolve())
    source_root = recompile_root / "papers" / paper_id / "source"
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"rebased source directory does not exist: {source_root} "
            f"(recorded={recorded_source})"
        )
    try:
        relative_main = recorded_main.relative_to(recorded_source)
    except ValueError:
        relative_main = Path(recorded_main.name)
    candidate = source_root / relative_main
    if candidate.is_file():
        return source_root.resolve(), relative_main
    matches = sorted(source_root.rglob(recorded_main.name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one rebased main TeX for {paper_id}; found {len(matches)} "
            f"matching {recorded_main.name!r}"
        )
    return source_root.resolve(), matches[0].resolve().relative_to(source_root.resolve())


def resolve_clean_pdf(clean_gt_root: Path, paper_id: str, recorded: str) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path.resolve()
    filename = path.name
    matches = list(
        clean_gt_root.glob(f"shard_*/papers/{paper_id}/synctex_build/{filename}")
    )
    direct = clean_gt_root / "papers" / paper_id / "synctex_build" / filename
    if direct.is_file():
        matches.append(direct)
    matches = sorted(set(item.resolve() for item in matches if item.is_file()))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one rebased clean PDF for {paper_id}; found {len(matches)} "
            f"matching {filename!r} under {clean_gt_root}"
        )
    return matches[0]


def prepare_output(output_dir: Path, resume: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"output directory is not empty; use a new path or --resume: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def remove_stale_pair_artifacts(
    output_dir: Path, rows: Sequence[dict[str, Any]]
) -> int:
    """Remove artifacts owned by an invalidated paper checkpoint."""

    output_root = output_dir.resolve()
    removed = 0
    for row in rows:
        for key in ("edited_image", "edited_markdown", "metadata"):
            relative = row.get(key)
            if not relative:
                continue
            target = (output_dir / str(relative)).resolve()
            try:
                target.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(f"stale artifact escapes output root: {target}") from exc
            if target.is_file():
                target.unlink()
                removed += 1
    return removed


def prune_unreferenced_pair_artifacts(
    output_dir: Path, pair_rows: Sequence[dict[str, Any]]
) -> dict[str, int]:
    """Delete stale page artifacts that are not referenced by the final manifest."""

    specifications = {
        "data": ("edited_image", "*.png"),
        "ground_truths": ("edited_markdown", "*.md"),
        "metadata": ("metadata", "*.json"),
    }
    counts = {"scanned": 0, "removed": 0, "removed_bytes": 0}
    for directory_name, (row_key, pattern) in specifications.items():
        expected = {
            str(Path(str(row[row_key])))
            for row in pair_rows
            if row.get(row_key)
        }
        directory = output_dir / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(pattern)):
            counts["scanned"] += 1
            relative = str(path.relative_to(output_dir))
            if relative in expected:
                continue
            counts["removed_bytes"] += path.stat().st_size
            path.unlink()
            counts["removed"] += 1
    return counts


def validate_accepted_subset(
    output_dir: Path,
    pair_rows: Sequence[dict[str, Any]],
    exports: dict[str, Any],
) -> list[str]:
    """Validate the materialized accepted subset independently of rejected papers.

    A paper-level compile/reflow exception is a safe rejection, not corruption of
    already accepted examples.  The independent verifier still performs the
    expensive PDF/image/Markdown checks after this builder finishes; this gate
    only ensures that the builder produced a complete, self-consistent subset.
    """

    issues: list[str] = []
    if not pair_rows:
        issues.append("accepted_subset_empty")
        return issues
    pair_ids = [str(row.get("pair_id", "")) for row in pair_rows]
    if any(not pair_id for pair_id in pair_ids):
        issues.append("accepted_pair_id_missing")
    if len(set(pair_ids)) != len(pair_ids):
        issues.append("accepted_pair_id_duplicate")
    for row in pair_rows:
        for key in ("edited_image", "edited_markdown", "metadata"):
            relative = row.get(key)
            path = output_dir / str(relative) if relative else None
            if path is None or not path.is_file() or path.stat().st_size == 0:
                issues.append(f"accepted_artifact_missing:{row.get('pair_id')}:{key}")
    if int(exports.get("train", -1)) + int(exports.get("val", -1)) != len(pair_rows):
        issues.append("export_count_mismatch")
    return issues


def resolve_clean_artifact(clean_gt_root: Path, relative: str) -> Path:
    direct = clean_gt_root / relative
    if direct.is_file():
        return direct
    matches = sorted(clean_gt_root.glob(f"shard_*/{relative}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one clean artifact for {relative!r}; found {len(matches)}"
        )
    return matches[0]


def markdown_has_bibliography_heading(markdown: str) -> bool:
    """Return whether a rendered page starts a bibliography/reference section.

    Inline citations and prose mentions of references are intentionally not
    matched.  Once a heading is found, the caller excludes that page and the
    remaining pages of the paper because bibliography entries commonly flow
    onto continuation pages without repeating the heading.
    """

    return any(
        BIBLIOGRAPHY_HEADING_RE.fullmatch(line)
        for line in markdown.splitlines()
        if line.strip()
    )


def markdown_is_table_of_contents_page(markdown: str) -> bool:
    """Detect contents pages so their ``References`` entry is not a section start."""

    nonblank = [line.strip() for line in markdown.splitlines() if line.strip()]
    if any(
        re.fullmatch(r"(?:#{1,6}\s*)?(?:table\s+of\s+)?contents\s*:?[\s*_]*", line, re.I)
        for line in nonblank
    ):
        return True
    isolated_page_numbers = sum(
        bool(re.fullmatch(r"(?:[ivxlcdm]+|\d{1,4})", line, re.I))
        for line in nonblank
    )
    # A continuation page of a table of contents may omit the word
    # ``Contents`` but still consists of section labels alternating with page
    # numbers.  Real bibliography pages normally have only the footer number.
    return isolated_page_numbers >= 2


def find_bibliography_start_page(
    clean_gt_root: Path,
    paper_id: str,
    strict_rows: Sequence[dict[str, Any]],
) -> int | None:
    """Find the first bibliography page by scanning every emitted page Markdown."""

    page_dirs: set[Path] = set()
    for row in strict_rows:
        markdown_path = resolve_clean_artifact(clean_gt_root, str(row["markdown"]))
        page_dirs.add(markdown_path.parent)
    if len(page_dirs) != 1:
        raise ValueError(
            f"expected one clean Markdown page directory for {paper_id}; "
            f"found={sorted(str(path) for path in page_dirs)}"
        )
    page_dir = next(iter(page_dirs))
    starts: list[int] = []
    for markdown_path in sorted(page_dir.glob("page_*.md")):
        match = re.fullmatch(r"page_(\d+)", markdown_path.stem)
        if match is None:
            continue
        markdown = markdown_path.read_text(encoding="utf-8")
        if (
            markdown_has_bibliography_heading(markdown)
            and not markdown_is_table_of_contents_page(markdown)
        ):
            starts.append(int(match.group(1)))
    return min(starts) if starts else None


def filter_bibliography_tail_rows(
    clean_gt_root: Path,
    paper_id: str,
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None]:
    """Exclude the bibliography start page and every later page in a paper."""

    start_page = find_bibliography_start_page(clean_gt_root, paper_id, rows)
    if start_page is None:
        return list(rows), [], None
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        page_number = int(row["page_number"])
        if page_number >= start_page:
            excluded.append(
                {
                    "data_id": str(row["data_id"]),
                    "paper_id": paper_id,
                    "page_number": page_number,
                    "bibliography_start_page": start_page,
                    "reason": "bibliography_page_excluded",
                }
            )
        else:
            accepted.append(row)
    return accepted, excluded, start_page


def strict_input_rejection_reasons(
    clean_gt_root: Path,
    row: dict[str, Any],
) -> list[str]:
    """Return fail-closed reasons why a clean page cannot seed edit data.

    The merged strict manifest is only an index, never an authority by itself.
    This gate reopens the page sidecar and Markdown, requires the current GT
    contracts, and rejects every uncertain structure instead of attempting a
    heuristic repair in the edit-data stage.
    """

    reasons: list[str] = []

    def require(condition: bool, reason: str) -> None:
        if not condition:
            reasons.append(reason)

    def integer(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def number(value: Any, default: float = -1.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return default

    require(row.get("validation_status") == "passed", "validation_status_not_passed")
    require(
        row.get("strict_text_contract_version")
        == STRICT_INPUT_STRICT_TEXT_CONTRACT_VERSION,
        "strict_text_contract_version_mismatch",
    )
    require(row.get("strict_text_v2_status") == "passed", "strict_text_v2_not_passed")
    require(
        row.get("strict_text_v2_failure_reasons") == [],
        "strict_text_v2_failure_reasons_not_empty",
    )
    require(
        row.get("author_superscript_contract_version")
        == STRICT_INPUT_AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
        "author_superscript_contract_version_mismatch",
    )
    require(
        row.get("footnote_representation") == STRICT_INPUT_FOOTNOTE_REPRESENTATION,
        "footnote_representation_mismatch",
    )

    contract = row.get("strict_text_contract")
    require(isinstance(contract, dict), "strict_text_contract_missing")
    if isinstance(contract, dict):
        for key in (
            "canonical_order_frozen_before_replacement",
            "captions_required",
            "headers_footers_page_numbers_required",
            "page_edge_hyphen_visible_form_required",
            "strict_punctuation_hard_gate",
            "strict_footnote_structure_hard_gate",
            "strict_author_superscript_hard_gate",
        ):
            require(contract.get(key) is True, f"strict_contract_gate_missing:{key}")
        require(contract.get("ignored_graphic") == 0, "ignored_graphic_not_zero")
        require(
            contract.get("footnote_representation")
            == STRICT_INPUT_FOOTNOTE_REPRESENTATION,
            "strict_contract_footnote_representation_mismatch",
        )
        require(
            contract.get("author_superscript_contract_version")
            == STRICT_INPUT_AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
            "strict_contract_author_superscript_version_mismatch",
        )
        require(
            contract.get("author_superscript_representation") == "html_sup",
            "strict_contract_author_superscript_representation_mismatch",
        )

    markdown_relative = row.get("markdown")
    image_relative = row.get("image")
    if not isinstance(markdown_relative, str) or not markdown_relative.strip():
        reasons.append("markdown_path_missing")
        return sorted(set(reasons))
    try:
        markdown_path = resolve_clean_artifact(clean_gt_root, markdown_relative)
    except (FileNotFoundError, ValueError):
        reasons.append("markdown_file_missing_or_ambiguous")
        return sorted(set(reasons))
    sidecar_path = markdown_path.with_suffix(".json")
    require(sidecar_path.is_file(), "sidecar_file_missing")
    if not sidecar_path.is_file():
        return sorted(set(reasons))
    try:
        sidecar = read_json(sidecar_path)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        reasons.append("sidecar_invalid_json")
        return sorted(set(reasons))
    require(sidecar == row, "strict_manifest_sidecar_mismatch")
    require(
        sidecar.get("data_id") == row.get("data_id"),
        "sidecar_data_id_mismatch",
    )
    markdown_sha256 = sha256_file(markdown_path)
    require(
        row.get("markdown_sha256") == markdown_sha256,
        "markdown_sha256_mismatch",
    )
    if not isinstance(image_relative, str) or not image_relative.strip():
        reasons.append("image_path_missing")
    else:
        try:
            image_path = resolve_clean_artifact(clean_gt_root, image_relative)
        except (FileNotFoundError, ValueError):
            reasons.append("image_file_missing_or_ambiguous")
        else:
            require(image_path.stat().st_size > 0, "image_file_empty")

    claims = sidecar.get("strict_text_claims")
    require(isinstance(claims, dict), "strict_text_claims_missing")
    if isinstance(claims, dict):
        require(claims.get("status") == "passed", "strict_text_claims_not_passed")
        require(
            claims.get("canonical_order_match") is True,
            "canonical_order_match_not_true",
        )
        for key in (
            "missing_line_ids",
            "duplicate_line_ids",
            "unknown_line_ids",
            "cross_page_line_ids",
            "order_inversions",
            "noncontiguous_structural_claims",
            "empty_structural_claims",
        ):
            require(claims.get(key) == [], f"strict_text_claims_not_empty:{key}")
        require(
            claims.get("inventory_count") == claims.get("claimed_unique_count"),
            "strict_text_claim_count_mismatch",
        )
        inventory = sidecar.get("line_inventory")
        require(isinstance(inventory, dict), "line_inventory_missing")
        if isinstance(inventory, dict):
            lines = inventory.get("lines")
            require(isinstance(lines, list), "line_inventory_lines_missing")
            if isinstance(lines, list):
                line_ids = [
                    line.get("line_id") if isinstance(line, dict) else None
                    for line in lines
                ]
                require(None not in line_ids, "line_inventory_invalid_line")
                require(len(set(line_ids)) == len(line_ids), "line_inventory_duplicate_ids")
                require(
                    inventory.get("canonical_line_ids") == line_ids,
                    "line_inventory_canonical_ids_mismatch",
                )
                require(
                    claims.get("flattened_claim_line_ids") == line_ids,
                    "flattened_claim_line_ids_mismatch",
                )

    for metric_name in (
        "strict_text_ordered_metrics",
        "strict_text_claimed_line_metrics",
    ):
        metric = sidecar.get(metric_name)
        require(isinstance(metric, dict), f"{metric_name}_missing")
        if isinstance(metric, dict):
            require(metric.get("status") == "passed", f"{metric_name}_not_passed")
            for prefix in ("token", "fivegram"):
                require(
                    integer(metric.get(f"{prefix}_missing")) == 0,
                    f"{metric_name}_{prefix}_missing",
                )
                require(
                    integer(metric.get(f"{prefix}_extra")) == 0,
                    f"{metric_name}_{prefix}_extra",
                )
            require(
                number(metric.get("anchor_monotonicity")) == 1.0,
                f"{metric_name}_order_not_exact",
            )

    require(sidecar.get("strict_punctuation_issues") == [], "punctuation_issues_present")
    inline_validation = sidecar.get("inline_markup_validation")
    require(isinstance(inline_validation, dict), "inline_markup_validation_missing")
    if isinstance(inline_validation, dict):
        require(
            inline_validation.get("status") == "passed",
            "inline_markup_validation_not_passed",
        )
        require(
            inline_validation.get("syntax_issues") == [],
            "inline_markup_syntax_issues_present",
        )
        require(
            integer(inline_validation.get("cid_placeholders")) == 0,
            "inline_markup_cid_placeholders_present",
        )

    source_integration = sidecar.get("source_integration")
    require(isinstance(source_integration, dict), "source_integration_missing")
    if isinstance(source_integration, dict):
        numbering = source_integration.get("heading_numbering")
        require(isinstance(numbering, dict), "heading_numbering_audit_missing")
        if isinstance(numbering, dict):
            require(numbering.get("strict") is True, "heading_numbering_not_strict")
            for key in ("lost", "wrong", "ambiguous"):
                require(
                    integer(numbering.get(key)) == 0,
                    f"heading_numbering_{key}",
                )

    footnotes = sidecar.get("footnotes")
    require(isinstance(footnotes, dict), "footnote_audit_missing")
    if isinstance(footnotes, dict):
        require(footnotes.get("status") == "passed", "footnote_audit_not_passed")
        require(
            footnotes.get("representation") == STRICT_INPUT_FOOTNOTE_REPRESENTATION,
            "footnote_audit_representation_mismatch",
        )
        require(integer(footnotes.get("fallback")) == 0, "footnote_fallback_present")
        require(
            integer(footnotes.get("total")) == integer(footnotes.get("structured"), -2),
            "footnote_total_structured_mismatch",
        )

    author_superscripts = sidecar.get("author_superscripts")
    require(isinstance(author_superscripts, dict), "author_superscript_audit_missing")
    if isinstance(author_superscripts, dict):
        require(
            author_superscripts.get("contract_version")
            == STRICT_INPUT_AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
            "author_superscript_audit_version_mismatch",
        )
        plans = integer(author_superscripts.get("plans"), 0)
        emitted = integer(author_superscripts.get("superscripts_emitted"))
        markers = author_superscripts.get("markers")
        require(isinstance(markers, list), "author_superscript_markers_missing")
        if isinstance(markers, list):
            require(emitted == len(markers), "author_superscript_emitted_count_mismatch")
        if integer(sidecar.get("page_number"), 0) == 1 and plans:
            require(
                author_superscripts.get("status") == "passed",
                "author_superscript_plans_unresolved",
            )
            require(emitted == plans, "author_superscript_plan_count_mismatch")
        require(
            integer(author_superscripts.get("unmatched_plans"), 0) == 0,
            "author_superscript_unmatched_plans",
        )

    source_blocks = sidecar.get("source_blocks")
    require(isinstance(source_blocks, list), "source_blocks_missing")
    if isinstance(source_blocks, list):
        for block in source_blocks:
            if not isinstance(block, dict):
                reasons.append("source_block_invalid")
                continue
            block_id = str(block.get("block_id", "unknown"))
            kind = block.get("kind")
            if kind == "table":
                require(
                    block.get("table_parse_status") == "parsed",
                    f"table_not_parsed:{block_id}",
                )
                require(
                    bool(str(block.get("table_html") or "").strip()),
                    f"table_html_missing:{block_id}",
                )
            if kind == "display_math":
                require(
                    block.get("formula_number_status")
                    not in {"ambiguous", "wrong", "unsafe_to_tag"},
                    f"formula_number_unresolved:{block_id}",
                )

    markdown = markdown_path.read_text(encoding="utf-8")
    for pattern, reason in (
        (r"\(cid:\d+\)", "pdf_cid_placeholder_present"),
        (r"@@(?:INLINE|MATH)[A-Za-z0-9_]*@@", "internal_placeholder_present"),
        (r"ZZ(?:STRUCT|INLINE)[A-Za-z0-9]+ZZ", "strict_sentinel_present"),
        (r"data-(?:table-id|source|parse-status)=", "internal_html_attribute_present"),
        (r"(?m)^\s*\|.*\|\s*$", "markdown_pipe_table_present"),
        (r"\\(?:cite\w*|ref|eqref|pageref|footnote)\s*\{", "raw_latex_command_present"),
        (r"\[\^[0-9]{1,4}\](?::)?", "legacy_markdown_footnote_present"),
    ):
        require(re.search(pattern, markdown) is None, reason)

    return sorted(set(reasons))


def load_strict_rows(
    clean_gt_root: Path,
    *,
    audit_path: Path | None = None,
) -> list[dict[str, Any]]:
    manifest = clean_gt_root / "pages_strict_text_v2.jsonl"
    if manifest.is_file():
        candidates = read_jsonl(manifest)
        print(
            f"[strict_manifest] mode=merged rows={len(candidates)} path={manifest}",
            flush=True,
        )
        mode = "merged"
    else:
        print(
            f"[strict_manifest] mode=shard_sidecars root={clean_gt_root}",
            flush=True,
        )
        candidates = [
            read_json(path)
            for path in sorted(clean_gt_root.glob("shard_*/papers/*/pages/page_*.json"))
        ]
        mode = "shard_sidecars"

    started = time.monotonic()
    data_id_counts = collections.Counter(str(row.get("data_id", "")) for row in candidates)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: collections.Counter[str] = collections.Counter()
    last_progress = started
    print(
        f"[strict_input_filter_start] pages={len(candidates)} "
        f"policy={STRICT_INPUT_FILTER_POLICY_VERSION} "
        f"strict_contract={STRICT_INPUT_STRICT_TEXT_CONTRACT_VERSION} "
        f"author_contract={STRICT_INPUT_AUTHOR_SUPERSCRIPT_CONTRACT_VERSION}",
        flush=True,
    )
    for index, row in enumerate(candidates, start=1):
        data_id = str(row.get("data_id", ""))
        reasons = strict_input_rejection_reasons(clean_gt_root, row)
        if not data_id:
            reasons.append("data_id_missing")
        elif data_id_counts[data_id] != 1:
            reasons.append("data_id_duplicate")
        reasons = sorted(set(reasons))
        if reasons:
            rejected.append({"data_id": data_id or None, "reasons": reasons})
            reason_counts.update(reasons)
        else:
            accepted.append(row)
        now = time.monotonic()
        if index % 500 == 0 or index == len(candidates) or now - last_progress >= 30.0:
            elapsed = max(now - started, 1e-9)
            print(
                f"[strict_input_filter_progress] completed={index}/{len(candidates)} "
                f"pct={100.0 * index / max(1, len(candidates)):.1f}% "
                f"accepted={len(accepted)} rejected={len(rejected)} "
                f"throughput={index / elapsed:.1f}_pages/s "
                f"elapsed={elapsed_text(elapsed)} current={data_id or '-'}",
                flush=True,
            )
            last_progress = now
    rows = sorted(
        accepted,
        key=lambda row: (
            str(row.get("arxiv_id", "")),
            str(row.get("version", "")),
            int(row.get("page_number", 0)),
        ),
    )
    print(
        f"[strict_input_filter_done] scanned={len(candidates)} accepted={len(rows)} "
        f"rejected={len(rejected)} reason_counts={dict(sorted(reason_counts.items()))} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    audit = {
        "policy_version": STRICT_INPUT_FILTER_POLICY_VERSION,
        "mode": mode,
        "clean_gt_root": str(clean_gt_root),
        "strict_text_contract_version": STRICT_INPUT_STRICT_TEXT_CONTRACT_VERSION,
        "author_superscript_contract_version": (
            STRICT_INPUT_AUTHOR_SUPERSCRIPT_CONTRACT_VERSION
        ),
        "footnote_representation": STRICT_INPUT_FOOTNOTE_REPRESENTATION,
        "scanned_pages": len(candidates),
        "accepted_pages": len(rows),
        "rejected_pages": len(rejected),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rejections": rejected,
    }
    if audit_path is not None:
        atomic_write_json(audit_path, audit)
        print(f"[strict_input_filter_audit] path={audit_path}", flush=True)
    return rows


def split_versioned_paper_id(paper_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+?)(v\d+)", paper_id)
    if match is None:
        return paper_id, ""
    return match.group(1), match.group(2)


def strict_input_policy_for_rows(rows: Sequence[dict[str, Any]]) -> str:
    policies = {
        str(row.get("source_first_input_policy_version") or STRICT_INPUT_FILTER_POLICY_VERSION)
        for row in rows
    }
    if len(policies) != 1:
        raise ValueError(f"mixed strict input policies within paper: {sorted(policies)}")
    return next(iter(policies))


def source_first_resume_fingerprint(
    rows: Sequence[dict[str, Any]],
) -> str | None:
    """Hash every source-first artifact that can affect edit generation.

    Older strict manifests do not carry these paths and therefore deliberately
    return ``None``: they may still be processed, but their paper checkpoint is
    never reused without a complete content identity proof.
    """

    if not rows:
        return None
    required = (
        "markdown",
        "source_pdf",
        "source_units_path",
        "source_root_override",
        "main_tex_override",
    )
    if any(not row.get(key) for row in rows for key in required):
        return None
    source_roots = {
        Path(str(row["source_root_override"])).resolve() for row in rows
    }
    source_units_paths = {
        Path(str(row["source_units_path"])).resolve() for row in rows
    }
    source_pdfs = {Path(str(row["source_pdf"])).resolve() for row in rows}
    main_tex_paths = {str(row["main_tex_override"]) for row in rows}
    if not (
        len(source_roots)
        == len(source_units_paths)
        == len(source_pdfs)
        == len(main_tex_paths)
        == 1
    ):
        return None
    source_root = next(iter(source_roots))
    source_units_path = next(iter(source_units_paths))
    source_pdf = next(iter(source_pdfs))
    main_tex = next(iter(main_tex_paths))
    if not (
        source_root.is_dir()
        and (source_root / main_tex).is_file()
        and source_units_path.is_file()
        and source_pdf.is_file()
    ):
        return None
    row_payloads: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item.get("data_id", ""))):
        markdown_path = Path(str(row["markdown"])).resolve()
        if not markdown_path.is_file():
            return None
        row_payloads.append(
            {
                "data_id": str(row.get("data_id", "")),
                "page_number": int(row.get("page_number", 0)),
                "markdown_sha256": sha256_file(markdown_path),
                "source_paragraph_ids": row.get("source_paragraph_ids"),
                "source_probe_ids": row.get("source_probe_ids"),
                "source_first_input_policy_version": row.get(
                    "source_first_input_policy_version"
                ),
            }
        )
    payload = {
        "rows": row_payloads,
        "source_tree_sha256": tree_hash(source_root),
        "source_units_sha256": sha256_file(source_units_path),
        "source_pdf_sha256": sha256_file(source_pdf),
        "main_tex": main_tex,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_source_first_case_rows(
    manifest_path: Path,
    *,
    audit_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load independently verified pages whose Markdown comes from LaTeX."""

    manifest_path = manifest_path.resolve()
    candidates = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, list):
        raise ValueError("source-first case manifest must be a JSON array")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: collections.Counter[str] = collections.Counter()
    seen_data_ids: collections.Counter[str] = collections.Counter()
    started = time.monotonic()
    print(
        f"[source_first_input_start] pages={len(candidates)} "
        f"policy={SOURCE_FIRST_INPUT_POLICY_VERSION} manifest={manifest_path}",
        flush=True,
    )

    for index, candidate in enumerate(candidates, start=1):
        reasons: list[str] = []

        def require(condition: bool, reason: str) -> None:
            if not condition:
                reasons.append(reason)

        if not isinstance(candidate, dict):
            rejected.append({"data_id": None, "reasons": ["manifest_row_invalid"]})
            reason_counts.update(["manifest_row_invalid"])
            continue
        markdown_value = candidate.get("markdown_path")
        image_value = candidate.get("image")
        markdown_path = Path(str(markdown_value)) if markdown_value else Path()
        image_path = Path(str(image_value)) if image_value else Path()
        if markdown_value and not markdown_path.is_absolute():
            markdown_path = manifest_path.parent / markdown_path
        if image_value and not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        markdown_path = markdown_path.resolve()
        image_path = image_path.resolve()
        require(bool(markdown_value) and markdown_path.is_file(), "markdown_file_missing")
        require(bool(image_value) and image_path.is_file(), "image_file_missing")
        sidecar_path = markdown_path.with_suffix(".json")
        require(sidecar_path.is_file(), "sidecar_file_missing")
        if reasons:
            data_id = str(candidate.get("pair_id") or "")
            rejected.append({"data_id": data_id or None, "reasons": sorted(set(reasons))})
            reason_counts.update(set(reasons))
            continue

        sidecar = read_json(sidecar_path)
        if markdown_path.is_file():
            require(
                sidecar.get("markdown_sha256") == sha256_file(markdown_path),
                "markdown_sha256_mismatch",
            )
        result_root = markdown_path.parent.parent
        report_path = result_root / "validation_report.json"
        source_units_path = result_root / "source_units.jsonl"
        source_probes_path = result_root / "source_probes.jsonl"
        source_root = result_root / "source_clean"
        require(report_path.is_file(), "validation_report_missing")
        require(source_units_path.is_file(), "source_units_missing")
        require(source_probes_path.is_file(), "source_probes_missing")
        require(source_root.is_dir(), "source_clean_missing")
        report = read_json(report_path) if report_path.is_file() else {}
        source_probes = read_jsonl(source_probes_path) if source_probes_path.is_file() else []
        known_probe_ids = [str(row.get("probe_id") or "") for row in source_probes]
        verifier = sidecar.get("verifier")
        shadow_invariant = sidecar.get("shadow_invariant")
        figure_policy = report.get("figure_policy")
        figure_status = (report.get("figure_removal") or {}).get("status")
        valid_figure_policy = (
            (figure_policy == "drop_figures" and figure_status == "passed")
            or (figure_policy == "keep_figures" and figure_status == "disabled")
        )
        require(sidecar.get("schema_version") == SOURCE_FIRST_SCHEMA_VERSION, "page_schema_mismatch")
        require(sidecar.get("contract") == SOURCE_FIRST_CONTRACT, "page_contract_mismatch")
        require(
            sidecar.get("probe_policy_version") == SOURCE_FIRST_PROBE_POLICY_VERSION,
            "page_probe_policy_mismatch",
        )
        require(
            sidecar.get("shadow_invariant_policy_version")
            == SOURCE_FIRST_SHADOW_INVARIANT_POLICY_VERSION,
            "page_shadow_invariant_policy_mismatch",
        )
        require(
            sidecar.get("heading_label_policy_version")
            == SOURCE_FIRST_HEADING_LABEL_POLICY_VERSION,
            "page_heading_label_policy_mismatch",
        )
        require(isinstance(shadow_invariant, dict), "page_shadow_invariant_missing")
        if isinstance(shadow_invariant, dict):
            require(
                shadow_invariant.get("character_count_equal") is True
                and shadow_invariant.get("character_text_equal") is True,
                "page_shadow_text_not_identical",
            )
            require(
                shadow_invariant.get("geometry_role") == "diagnostic_only",
                "page_shadow_geometry_role_mismatch",
            )
        require(sidecar.get("figure_policy") == figure_policy, "page_figure_policy_mismatch")
        require(sidecar.get("status") == "passed", "page_status_not_passed")
        require(sidecar.get("generation_source") == "latex_source", "generation_not_latex_source")
        require(sidecar.get("page_provenance") == "compiled_vector_color", "page_provenance_mismatch")
        require(sidecar.get("pdf_role") == "independent_verifier_only", "pdf_role_mismatch")
        require(isinstance(verifier, dict), "verifier_missing")
        if isinstance(verifier, dict):
            require(verifier.get("status") == "passed", "verifier_not_passed")
            require(
                verifier.get("contract_version")
                == SOURCE_FIRST_VERIFIER_CONTRACT_VERSION,
                "verifier_contract_version_mismatch",
            )
            require(
                verifier.get("exact_ordered_character_stream_match") is True,
                "ordered_content_match_not_exact",
            )
        require(report.get("schema_version") == SOURCE_FIRST_SCHEMA_VERSION, "source_first_schema_mismatch")
        require(report.get("contract") == SOURCE_FIRST_CONTRACT, "source_first_contract_mismatch")
        require(
            report.get("probe_policy_version") == SOURCE_FIRST_PROBE_POLICY_VERSION,
            "source_first_probe_policy_mismatch",
        )
        require(
            report.get("shadow_invariant_policy_version")
            == SOURCE_FIRST_SHADOW_INVARIANT_POLICY_VERSION,
            "source_first_shadow_invariant_policy_mismatch",
        )
        require(
            report.get("heading_label_policy_version")
            == SOURCE_FIRST_HEADING_LABEL_POLICY_VERSION,
            "source_first_heading_label_policy_mismatch",
        )
        require(valid_figure_policy, "source_first_figure_policy_mismatch")
        require(
            (report.get("reference_removal") or {}).get("status") == "passed",
            "source_first_reference_removal_mismatch",
        )
        require(report.get("status") == "passed", "source_first_report_not_passed")
        require(report.get("pdf_used_for_generation") is False, "pdf_used_for_generation")
        require(report.get("pdf_used_for_verification") is True, "pdf_not_used_for_verification")
        require(str(sidecar.get("data_id", "")) == str(candidate.get("pair_id", "")), "manifest_data_id_mismatch")
        require((result_root / str(sidecar.get("markdown", ""))).resolve() == markdown_path, "sidecar_markdown_path_mismatch")
        require((result_root / str(sidecar.get("image", ""))).resolve() == image_path, "sidecar_image_path_mismatch")
        main_relative = Path(str(report.get("main_tex", "")))
        require(bool(str(report.get("main_tex", ""))), "main_tex_missing")
        require((source_root / main_relative).is_file(), "main_tex_file_missing")
        clean_pdf_value = report.get("clean_pdf")
        clean_pdf = Path(str(clean_pdf_value)) if clean_pdf_value else Path()
        if clean_pdf_value and not clean_pdf.is_absolute():
            clean_pdf = result_root / clean_pdf
        clean_pdf = clean_pdf.resolve()
        require(bool(clean_pdf_value) and clean_pdf.is_file(), "clean_pdf_missing")
        source_paragraph_ids = sidecar.get("source_paragraph_ids")
        require(isinstance(source_paragraph_ids, list) and bool(source_paragraph_ids), "source_paragraph_ids_missing")
        source_probe_ids = sidecar.get("source_probe_ids")
        require(isinstance(source_probe_ids, list) and bool(source_probe_ids), "source_probe_ids_missing")
        require(
            bool(known_probe_ids)
            and all(known_probe_ids)
            and len(known_probe_ids) == len(set(known_probe_ids)),
            "source_probe_inventory_invalid",
        )
        if isinstance(source_probe_ids, list):
            require(
                set(map(str, source_probe_ids)) <= set(known_probe_ids),
                "source_probe_ids_unknown",
            )

        data_id = str(sidecar.get("data_id", ""))
        paper_id = str(sidecar.get("paper_id", ""))
        require(bool(data_id), "data_id_missing")
        require(bool(paper_id), "paper_id_missing")
        if reasons:
            reasons = sorted(set(reasons))
            rejected.append({"data_id": data_id or None, "reasons": reasons})
            reason_counts.update(reasons)
        else:
            arxiv_id, version = split_versioned_paper_id(paper_id)
            normalized = dict(sidecar)
            normalized.update(
                {
                    "arxiv_id": arxiv_id,
                    "version": version,
                    "markdown": str(markdown_path),
                    "image": str(image_path),
                    "source_pdf": str(clean_pdf),
                    "source_units_path": str(source_units_path.resolve()),
                    "source_root_override": str(source_root.resolve()),
                    "main_tex_override": main_relative.as_posix(),
                    "source_paragraph_integration": {
                        "source_paragraph_ids": list(source_paragraph_ids)
                    },
                    "source_first_input_policy_version": SOURCE_FIRST_INPUT_POLICY_VERSION,
                }
            )
            accepted.append(normalized)
            seen_data_ids[data_id] += 1
        print(
            f"[source_first_input_progress] completed={index}/{len(candidates)} "
            f"accepted={len(accepted)} rejected={len(rejected)} current={data_id or '-'}",
            flush=True,
        )

    duplicate_ids = {data_id for data_id, count in seen_data_ids.items() if count != 1}
    if duplicate_ids:
        accepted = [row for row in accepted if str(row["data_id"]) not in duplicate_ids]
        for data_id in sorted(duplicate_ids):
            rejected.append({"data_id": data_id, "reasons": ["data_id_duplicate"]})
            reason_counts.update(["data_id_duplicate"])
    accepted.sort(key=lambda row: (str(row["paper_id"]), int(row["page_number"])))
    audit = {
        "policy_version": SOURCE_FIRST_INPUT_POLICY_VERSION,
        "mode": "source_first_case_manifest",
        "manifest": str(manifest_path),
        "scanned_pages": len(candidates),
        "accepted_pages": len(accepted),
        "rejected_pages": len(rejected),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rejections": rejected,
    }
    if audit_path is not None:
        atomic_write_json(audit_path, audit)
    print(
        f"[source_first_input_done] scanned={len(candidates)} accepted={len(accepted)} "
        f"rejected={len(rejected)} elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return accepted


def build_paper(
    *,
    paper_id: str,
    strict_rows: list[dict[str, Any]],
    recompile: dict[str, Any],
    recompile_root: Path,
    clean_gt_root: Path,
    output_dir: Path,
    seed: int,
    dpi: int,
    timeout_seconds: int,
    progress: Progress,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paper_output = output_dir / "papers" / paper_id
    checkpoint = paper_output / "paper_result.json"
    expected_data_ids = sorted(str(row["data_id"]) for row in strict_rows)
    strict_input_policy_version = strict_input_policy_for_rows(strict_rows)
    source_first_input_sha256 = source_first_resume_fingerprint(strict_rows)
    if checkpoint.is_file():
        stored = read_json(checkpoint)
        stored_rows = stored.get("pairs", [])
        if (
            stored.get("status") == "passed"
            and stored.get("mutation_policy_version") == MUTATION_POLICY_VERSION
            and stored.get("selection_policy_version") == SELECTION_POLICY_VERSION
            and stored.get("strict_input_filter_policy_version")
            == strict_input_policy_version
            and stored.get("bibliography_policy_version")
            == BIBLIOGRAPHY_POLICY_VERSION
            and stored.get("strict_data_ids_considered") == expected_data_ids
            and source_first_input_sha256 is not None
            and stored.get("source_first_input_sha256")
            == source_first_input_sha256
            and all(
            (output_dir / row["edited_image"]).is_file()
            and (output_dir / row["edited_markdown"]).is_file()
            for row in stored_rows
            )
        ):
            local_edited_pdf = paper_output / "paper_edited.pdf"
            if local_edited_pdf.is_file():
                stored["edited_pdf"] = str(local_edited_pdf.resolve())
                atomic_write_json(checkpoint, stored)
            progress.accepted += len(stored_rows)
            progress.rejected += len(stored.get("page_rejections", []))
            progress.pages_completed += len(strict_rows)
            progress.emit("resume_paper", paper_id, f"pairs={len(stored_rows)}")
            return stored_rows, stored
        removed = remove_stale_pair_artifacts(output_dir, stored_rows)
        if removed:
            progress.emit(
                "invalidate_paper_checkpoint",
                paper_id,
                f"removed_stale_artifacts={removed}",
            )

    first_row = strict_rows[0]
    if first_row.get("source_root_override"):
        clean_source_root = Path(str(first_row["source_root_override"])).resolve()
        main_relative = Path(str(first_row["main_tex_override"]))
        if not clean_source_root.is_dir():
            raise FileNotFoundError(clean_source_root)
        if not (clean_source_root / main_relative).is_file():
            raise FileNotFoundError(clean_source_root / main_relative)
        for row in strict_rows:
            if Path(str(row.get("source_root_override", ""))).resolve() != clean_source_root:
                raise ValueError(f"source root differs within paper: {paper_id}")
            if Path(str(row.get("main_tex_override", ""))) != main_relative:
                raise ValueError(f"main TeX differs within paper: {paper_id}")
    else:
        clean_source_root, main_relative = resolve_recompile_source(
            recompile_root, paper_id, recompile
        )
    original_tree_sha256 = tree_hash(clean_source_root)
    source_edited = paper_output / "source_edited"
    if source_edited.exists():
        shutil.rmtree(source_edited)
    source_edited.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(clean_source_root, source_edited)
    clean_pdf = resolve_clean_pdf(
        clean_gt_root, paper_id, str(strict_rows[0]["source_pdf"])
    )
    clean_words_by_page, pdf_global_counts = extract_pdf_words(
        clean_pdf,
        progress=progress,
        current=paper_id,
        phase="clean_pdf_extract",
    )
    paper_vocabulary = set(pdf_global_counts)
    if first_row.get("source_units_path"):
        paragraph_path = Path(str(first_row["source_units_path"])).resolve()
        if not paragraph_path.is_file():
            raise FileNotFoundError(paragraph_path)
        if any(
            Path(str(row.get("source_units_path", ""))).resolve() != paragraph_path
            for row in strict_rows
        ):
            raise ValueError(f"source units differ within paper: {paper_id}")
    else:
        paragraph_path = resolve_clean_artifact(
            clean_gt_root, f"papers/{paper_id}/source_paragraphs.jsonl"
        )
    paragraph_rows = read_jsonl(paragraph_path)
    occurrences = source_occurrences(clean_source_root, paragraph_rows)
    source_by_word: dict[str, list[SourceOccurrence]] = collections.defaultdict(list)
    for occurrence in occurrences:
        source_by_word[occurrence.word].append(occurrence)

    page_mutations: dict[int, list[Mutation]] = {}
    selected_source_positions: set[tuple[str, int]] = set()
    preflight_rejections: list[dict[str, Any]] = []
    for row in strict_rows:
        page_number = int(row["page_number"])
        clean_markdown_path = resolve_clean_artifact(clean_gt_root, str(row["markdown"]))
        clean_markdown = clean_markdown_path.read_text(encoding="utf-8")
        mutations = choose_mutations_for_page(
            row=row,
            clean_markdown=clean_markdown,
            page_words=clean_words_by_page[page_number],
            source_by_word=source_by_word,
            paper_word_vocabulary=paper_vocabulary,
            excluded_source_positions=selected_source_positions,
            seed=seed,
        )
        if len(mutations) < 3:
            preflight_rejections.append(
                {"data_id": row["data_id"], "reason": "fewer_than_3_safe_unique_words"}
            )
            progress.rejected += 1
            progress.pages_completed += 1
            progress.emit("preflight_reject", str(row["data_id"]))
            continue
        page_mutations[page_number] = mutations
        selected_source_positions.update(
            (mutation.source_file, mutation.source_word_offset)
            for mutation in mutations
        )

    all_mutations = [
        mutation
        for page_number in sorted(page_mutations)
        for mutation in page_mutations[page_number]
    ]
    if not all_mutations:
        result = {
            "status": "no_eligible_pages",
            "paper_id": paper_id,
            "mutation_policy_version": MUTATION_POLICY_VERSION,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "strict_input_filter_policy_version": strict_input_policy_version,
            "bibliography_policy_version": BIBLIOGRAPHY_POLICY_VERSION,
            "strict_data_ids_considered": expected_data_ids,
            "source_first_input_sha256": source_first_input_sha256,
            "preflight_rejections": preflight_rejections,
            "pairs": [],
        }
        atomic_write_json(checkpoint, result)
        return [], result
    apply_source_mutations(source_edited, all_mutations)
    compile_dir = paper_output / "build"
    if compile_dir.exists():
        shutil.rmtree(compile_dir)
    edited_pdf, compile_info = compile_edited_source(
        source_root=source_edited,
        main_relative=main_relative,
        engine=str(recompile.get("compile", {}).get("engine") or "pdflatex"),
        build_dir=compile_dir,
        progress=progress,
        paper_id=paper_id,
        timeout_seconds=timeout_seconds,
    )
    final_pdf = paper_output / "paper_edited.pdf"
    shutil.copy2(edited_pdf, final_pdf)

    with pdfplumber.open(clean_pdf) as clean_document, pdfplumber.open(edited_pdf) as edited_document:
        if len(clean_document.pages) != len(edited_document.pages):
            raise ValueError(
                f"page count changed for {paper_id}: "
                f"{len(clean_document.pages)}->{len(edited_document.pages)}"
            )
        edited_words_by_page, _ = extract_pdf_words(
            edited_pdf,
            progress=progress,
            current=paper_id,
            phase="edited_pdf_extract",
        )
        pair_rows: list[dict[str, Any]] = []
        page_rejections = list(preflight_rejections)
        document_valid, document_reason, document_max_vertical_shift = validate_document_words(
            clean_words_by_page,
            edited_words_by_page,
            page_mutations,
        )
        document_validation = {
            "status": "passed" if document_valid else "failed",
            "reason": document_reason,
            "pages_checked": len(clean_words_by_page),
            "expected_mutated_pages": len(page_mutations),
            "max_vertical_shift_points": (
                round(document_max_vertical_shift, 4)
                if math.isfinite(document_max_vertical_shift)
                else None
            ),
        }
        if not document_valid:
            # The exported training unit is one rendered page, not the full
            # edited PDF.  Keep the document-wide comparison as diagnostics,
            # but accept only pages whose own word sequence differs by exactly
            # the declared 3--4 substitutions.  This preserves the hard page
            # contract without discarding unrelated valid pages because a
            # repeated source word changed elsewhere in the document.
            progress.emit(
                "document_warning",
                paper_id,
                f"pages={len(page_mutations)} reason={document_reason}",
            )
        strict_by_page = {int(row["page_number"]): row for row in strict_rows}
        for page_number, mutations in sorted(page_mutations.items()):
            row = strict_by_page[page_number]
            clean_words = clean_words_by_page[page_number]
            edited_words = edited_words_by_page[page_number]
            valid, reason, max_vertical_shift = validate_page_words(
                clean_words, edited_words, mutations
            )
            clean_markdown_path = resolve_clean_artifact(clean_gt_root, str(row["markdown"]))
            clean_markdown = clean_markdown_path.read_text(encoding="utf-8")
            edited_markdown = apply_markdown_mutations(clean_markdown, mutations)
            if markdown_diff_count(clean_markdown, edited_markdown) != len(mutations):
                valid = False
                reason = "markdown_diff_not_exactly_mutation_count"
            if not valid:
                page_rejections.append({"data_id": row["data_id"], "reason": reason})
                progress.rejected += 1
                progress.pages_completed += 1
                progress.emit("page_reject", str(row["data_id"]), reason)
                continue

            pair_id = f"{row['data_id']}_confusable_{PAIR_POLICY_TAG}_s{seed}"
            edited_image_rel = f"data/{pair_id}_edited.png"
            edited_markdown_rel = f"ground_truths/{pair_id}_edited.md"
            metadata_rel = f"metadata/{pair_id}.json"
            edited_image_target = output_dir / edited_image_rel
            render_page(edited_pdf, page_number, edited_image_target, dpi)
            atomic_write_text(output_dir / edited_markdown_rel, edited_markdown)

            with Image.open(edited_image_target) as image:
                image_width, image_height = image.size
            edited_page = edited_document.pages[page_number - 1]
            changes: list[dict[str, Any]] = []
            for mutation in mutations:
                edited_word = edited_words[mutation.pdf_word_index]
                bbox_points = [edited_word.x0, edited_word.top, edited_word.x1, edited_word.bottom]
                bbox_pixels = point_bbox_to_pixel_bbox(
                    bbox_points,
                    page_width=float(edited_page.width),
                    page_height=float(edited_page.height),
                    image_width=image_width,
                    image_height=image_height,
                )
                changes.append(
                    {
                        "ocr_ans": mutation.mutated_word,
                        "origin_ans": mutation.original_word,
                        "bbox": bbox_pixels,
                        "from_char": mutation.from_char,
                        "to_char": mutation.to_char,
                        "source_file": mutation.source_file,
                        "source_line": mutation.source_line,
                        "source_column": mutation.source_column,
                        "markdown_span": [mutation.markdown_start, mutation.markdown_end],
                    }
                )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "mutation_policy_version": MUTATION_POLICY_VERSION,
                "selection_policy_version": SELECTION_POLICY_VERSION,
                "strict_input_filter_policy_version": strict_input_policy_version,
                "bibliography_policy_version": BIBLIOGRAPHY_POLICY_VERSION,
                "pair_id": pair_id,
                "data_id": row["data_id"],
                "paper_id": paper_id,
                "arxiv_id": row["arxiv_id"],
                "version": row.get("version"),
                "page_number": page_number,
                "edited_image": edited_image_rel,
                "edited_markdown": edited_markdown_rel,
                "mutation_count": len(changes),
                "changes": changes,
                "bibliography_content_present": False,
                "validation": {
                    "character_substitutions_only": True,
                    "markdown_same_length_after_substitution": len(clean_markdown)
                    == len(edited_markdown),
                    "markdown_character_diff_count": markdown_diff_count(
                        clean_markdown, edited_markdown
                    ),
                    "pdf_word_count_unchanged": len(clean_words) == len(edited_words),
                    "pdf_word_sequence_expected": True,
                    "max_vertical_shift_points": round(max_vertical_shift, 4),
                    "page_count_unchanged": True,
                },
            }
            atomic_write_json(output_dir / metadata_rel, metadata)
            pair_row = {
                "pair_id": pair_id,
                "data_id": row["data_id"],
                "paper_id": paper_id,
                "arxiv_id": row["arxiv_id"],
                "version": row.get("version"),
                "page_number": page_number,
                "edited_image": edited_image_rel,
                "edited_markdown": edited_markdown_rel,
                "metadata": metadata_rel,
                "mutation_count": len(changes),
                "changes": changes,
                "bibliography_policy_version": BIBLIOGRAPHY_POLICY_VERSION,
                "strict_input_filter_policy_version": strict_input_policy_version,
            }
            pair_rows.append(pair_row)
            progress.accepted += 1
            progress.pages_completed += 1
            progress.emit(
                "page_accept",
                str(row["data_id"]),
                f"mutations={len(changes)}",
            )

    original_tree_after = tree_hash(clean_source_root)
    if original_tree_after != original_tree_sha256:
        raise RuntimeError(f"original source tree changed unexpectedly: {paper_id}")
    result = {
        "status": "passed",
        "paper_id": paper_id,
        "mutation_policy_version": MUTATION_POLICY_VERSION,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "strict_input_filter_policy_version": strict_input_policy_version,
        "bibliography_policy_version": BIBLIOGRAPHY_POLICY_VERSION,
        "strict_data_ids_considered": expected_data_ids,
        "source_first_input_sha256": source_first_input_sha256,
        "source_tree_sha256_before": original_tree_sha256,
        "source_tree_sha256_after": original_tree_after,
        "edited_source_tree_sha256": tree_hash(source_edited),
        "edited_pdf": str(final_pdf),
        "edited_pdf_sha256": sha256_file(final_pdf),
        "compile": compile_info,
        "document_validation": document_validation,
        "preflight_rejections": preflight_rejections,
        "page_rejections": page_rejections,
        "pairs": pair_rows,
    }
    atomic_write_json(checkpoint, result)
    return pair_rows, result


def build_paper_process(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Process-isolated paper worker used by the bulk server runner."""

    global LATEXMK, PDFTOPPM
    LATEXMK = Path(str(payload["latexmk"]))
    PDFTOPPM = Path(str(payload["pdftoppm"]))
    strict_rows = list(payload["strict_rows"])
    local_progress = Progress(1, len(strict_rows))
    return build_paper(
        paper_id=str(payload["paper_id"]),
        strict_rows=strict_rows,
        recompile=dict(payload["recompile"]),
        recompile_root=Path(str(payload["recompile_root"])),
        clean_gt_root=Path(str(payload["clean_gt_root"])),
        output_dir=Path(str(payload["output_dir"])),
        seed=int(payload["seed"]),
        dpi=int(payload["dpi"]),
        timeout_seconds=int(payload["timeout_seconds"]),
        progress=local_progress,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompile-root", type=Path, default=DEFAULT_RECOMPILE_ROOT)
    parser.add_argument("--clean-gt-root", type=Path, default=DEFAULT_CLEAN_GT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument("--paper-ids", nargs="*", default=[])
    parser.add_argument(
        "--source-first-case-manifest",
        type=Path,
        help="JSON array of independently verified source-first page cases",
    )
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="paper-level process workers; each worker compiles one paper at a time",
    )
    parser.add_argument(
        "--latexmk",
        type=Path,
        default=Path(os.environ.get("LATEXMK", str(LATEXMK))),
        help="latexmk executable; defaults to LATEXMK or PATH discovery",
    )
    parser.add_argument(
        "--pdftoppm",
        type=Path,
        default=Path(os.environ.get("PDFTOPPM", str(PDFTOPPM))),
        help="pdftoppm executable; defaults to PDFTOPPM or PATH discovery",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    global LATEXMK, PDFTOPPM
    args = parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("--workers must be between 1 and 64")
    LATEXMK = args.latexmk.expanduser().absolute()
    PDFTOPPM = args.pdftoppm.expanduser().resolve()
    for tool in (LATEXMK, PDFTOPPM):
        if not tool.is_file():
            raise FileNotFoundError(tool)
    recompile_root = args.recompile_root.resolve()
    clean_gt_root = args.clean_gt_root.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output(output_dir, args.resume)
    recompile_results = load_recompile_results(recompile_root)
    strict_input_audit_path = output_dir / "strict_input_filter_audit.json"
    if args.source_first_case_manifest:
        strict_rows = load_source_first_case_rows(
            args.source_first_case_manifest,
            audit_path=strict_input_audit_path,
        )
    else:
        strict_rows = load_strict_rows(
            clean_gt_root,
            audit_path=strict_input_audit_path,
        )
    strict_input_audit = read_json(strict_input_audit_path)
    unfiltered_strict_by_paper: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in strict_rows:
        paper_id = f"{row['arxiv_id']}{row.get('version', '')}"
        unfiltered_strict_by_paper[paper_id].append(row)
    strict_by_paper: dict[str, list[dict[str, Any]]] = {}
    bibliography_exclusions_by_paper: dict[str, list[dict[str, Any]]] = {}
    bibliography_start_pages: dict[str, int] = {}
    bibliography_filter_started = time.monotonic()
    print(
        f"[bibliography_filter_start] papers={len(unfiltered_strict_by_paper)} "
        f"strict_pages={len(strict_rows)} policy={BIBLIOGRAPHY_POLICY_VERSION}",
        flush=True,
    )
    for filter_index, (paper_id, paper_rows) in enumerate(
        sorted(unfiltered_strict_by_paper.items()), start=1
    ):
        eligible_rows, excluded_rows, start_page = filter_bibliography_tail_rows(
            clean_gt_root,
            paper_id,
            sorted(paper_rows, key=lambda row: int(row["page_number"])),
        )
        strict_by_paper[paper_id] = eligible_rows
        if excluded_rows:
            bibliography_exclusions_by_paper[paper_id] = excluded_rows
        if start_page is not None:
            bibliography_start_pages[paper_id] = start_page
        if filter_index % 250 == 0 or filter_index == len(unfiltered_strict_by_paper):
            excluded_count = sum(
                len(rows) for rows in bibliography_exclusions_by_paper.values()
            )
            elapsed = max(time.monotonic() - bibliography_filter_started, 1e-9)
            print(
                f"[bibliography_filter_progress] papers={filter_index}/"
                f"{len(unfiltered_strict_by_paper)} excluded={excluded_count} "
                f"throughput={filter_index / elapsed:.1f}_papers/s "
                f"elapsed={elapsed_text(elapsed)} current={paper_id}",
                flush=True,
            )
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    requested = set(args.paper_ids)
    if args.source_first_case_manifest and not requested:
        requested = set(strict_by_paper)
    selected_papers = [
        paper
        for paper in recompile_results
        if not requested or paper in requested or recompile_results[paper].get("arxiv_id") in requested
    ][: args.max_papers]
    if requested:
        found = set(selected_papers) | {
            str(recompile_results[paper].get("arxiv_id")) for paper in selected_papers
        }
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"requested papers not found: {missing}")
    selected_bibliography_exclusions = [
        row
        for paper in selected_papers
        for row in bibliography_exclusions_by_paper.get(paper, [])
    ]
    eligible_papers = [paper for paper in selected_papers if strict_by_paper.get(paper)]
    strict_pages_before_bibliography_filter = sum(
        len(unfiltered_strict_by_paper.get(paper, [])) for paper in selected_papers
    )
    progress = Progress(len(selected_papers), sum(len(strict_by_paper[paper]) for paper in eligible_papers))
    print(
        f"[start] papers_requested={len(selected_papers)} papers_with_strict_pages={len(eligible_papers)} "
        f"strict_pages_before_bibliography_filter={strict_pages_before_bibliography_filter} "
        f"bibliography_pages_excluded={len(selected_bibliography_exclusions)} "
        f"strict_pages={progress.total_pages} seed={args.seed} dpi={args.dpi} "
        f"output={output_dir}",
        flush=True,
    )
    all_pairs: list[dict[str, Any]] = []
    paper_results: list[dict[str, Any]] = []
    skipped_papers = [
        paper
        for paper in selected_papers
        if not unfiltered_strict_by_paper.get(paper)
    ]
    bibliography_only_papers = [
        paper
        for paper in selected_papers
        if unfiltered_strict_by_paper.get(paper) and not strict_by_paper.get(paper)
    ]
    if args.workers == 1:
        for index, paper_id in enumerate(eligible_papers, start=1):
            paper_rows = sorted(
                strict_by_paper[paper_id], key=lambda row: int(row["page_number"])
            )
            progress.emit(
                "paper_start",
                paper_id,
                f"paper={index}/{len(eligible_papers)} strict_pages={len(paper_rows)}",
            )
            pages_before_paper = progress.pages_completed
            try:
                pairs, paper_result = build_paper(
                    paper_id=paper_id,
                    strict_rows=paper_rows,
                    recompile=recompile_results[paper_id],
                    recompile_root=recompile_root,
                    clean_gt_root=clean_gt_root,
                    output_dir=output_dir,
                    seed=args.seed,
                    dpi=args.dpi,
                    timeout_seconds=args.compile_timeout,
                    progress=progress,
                )
                all_pairs.extend(pairs)
                paper_results.append(paper_result)
                progress.emit("paper_done", paper_id, f"pairs={len(pairs)}")
            except Exception as exc:  # noqa: BLE001 - persist exact paper failure
                progress.errors += 1
                completed_for_paper = progress.pages_completed - pages_before_paper
                remaining_for_paper = max(0, len(paper_rows) - completed_for_paper)
                progress.rejected += remaining_for_paper
                progress.pages_completed += remaining_for_paper
                paper_results.append(
                    {"status": "failed", "paper_id": paper_id, "error": str(exc), "pairs": []}
                )
                progress.emit("paper_error", paper_id, f"error={exc}")
    else:
        print(
            f"[parallel_start] phase=mutation_recompile workers={args.workers} "
            f"papers={len(eligible_papers)} pages={progress.total_pages}",
            flush=True,
        )
        try:
            executor: concurrent.futures.Executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            )
            executor_mode = "process"
        except (PermissionError, NotImplementedError, OSError) as exc:
            print(
                f"[warning] process_pool_unavailable={type(exc).__name__}:{exc}; "
                f"fallback=thread workers={args.workers}",
                flush=True,
            )
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
            executor_mode = "thread_fallback"
        print(
            f"[parallel_executor] phase=mutation_recompile mode={executor_mode} "
            f"workers={args.workers}",
            flush=True,
        )
        with executor:
            future_to_paper: dict[concurrent.futures.Future[Any], str] = {}
            for paper_id in eligible_papers:
                paper_rows = sorted(
                    strict_by_paper[paper_id], key=lambda row: int(row["page_number"])
                )
                payload = {
                    "paper_id": paper_id,
                    "strict_rows": paper_rows,
                    "recompile": recompile_results[paper_id],
                    "recompile_root": str(recompile_root),
                    "clean_gt_root": str(clean_gt_root),
                    "output_dir": str(output_dir),
                    "seed": args.seed,
                    "dpi": args.dpi,
                    "timeout_seconds": args.compile_timeout,
                    "latexmk": str(LATEXMK),
                    "pdftoppm": str(PDFTOPPM),
                }
                future_to_paper[executor.submit(build_paper_process, payload)] = paper_id
            pending = set(future_to_paper)
            completed_papers = 0
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=30.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    running = sorted(future_to_paper[future] for future in pending)
                    progress.emit(
                        "parallel_heartbeat",
                        running[0] if running else "-",
                        f"papers={completed_papers}/{len(eligible_papers)} "
                        f"running={len(running)} workers={args.workers}",
                    )
                    continue
                for future in done:
                    paper_id = future_to_paper[future]
                    paper_pages = len(strict_by_paper[paper_id])
                    completed_papers += 1
                    try:
                        pairs, paper_result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        progress.errors += 1
                        progress.rejected += paper_pages
                        progress.pages_completed += paper_pages
                        paper_results.append(
                            {"status": "failed", "paper_id": paper_id, "error": str(exc), "pairs": []}
                        )
                        progress.emit(
                            "paper_error",
                            paper_id,
                            f"paper={completed_papers}/{len(eligible_papers)} error={exc}",
                        )
                        continue
                    all_pairs.extend(pairs)
                    paper_results.append(paper_result)
                    progress.accepted += len(pairs)
                    progress.rejected += max(0, paper_pages - len(pairs))
                    progress.pages_completed += paper_pages
                    progress.emit(
                        "paper_done",
                        paper_id,
                        f"paper={completed_papers}/{len(eligible_papers)} pairs={len(pairs)}",
                    )
    all_pairs.sort(key=lambda row: (str(row["paper_id"]), int(row["page_number"])))
    paper_results.sort(key=lambda row: str(row.get("paper_id", "")))
    write_jsonl(output_dir / "pairs.jsonl", all_pairs)
    stale_artifact_cleanup = prune_unreferenced_pair_artifacts(output_dir, all_pairs)
    print(
        f"[stale_artifact_cleanup] scanned={stale_artifact_cleanup['scanned']} "
        f"removed={stale_artifact_cleanup['removed']} "
        f"removed_bytes={stale_artifact_cleanup['removed_bytes']}",
        flush=True,
    )
    exports: dict[str, Any] = {}
    if all_pairs:
        exports = export_training(
            output_dir=output_dir,
            pair_rows=all_pairs,
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
        )
    mutation_distribution = dict(
        sorted(collections.Counter(row["mutation_count"] for row in all_pairs).items())
    )
    accepted_subset_issues = validate_accepted_subset(output_dir, all_pairs, exports)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not accepted_subset_issues else "failed",
        "papers_requested": len(selected_papers),
        "papers_with_strict_pages": len(eligible_papers),
        "papers_without_strict_pages": skipped_papers,
        "papers_removed_by_bibliography_filter": bibliography_only_papers,
        "strict_pages_before_bibliography_filter": strict_pages_before_bibliography_filter,
        "bibliography_pages_excluded": len(selected_bibliography_exclusions),
        "bibliography_exclusions": selected_bibliography_exclusions,
        "bibliography_start_pages": {
            paper: bibliography_start_pages[paper]
            for paper in selected_papers
            if paper in bibliography_start_pages
        },
        "strict_pages_considered": progress.total_pages,
        "accepted_pairs": len(all_pairs),
        "rejected_pages": progress.rejected + len(selected_bibliography_exclusions),
        "errors": len(accepted_subset_issues),
        "error_reasons": accepted_subset_issues,
        "paper_processing_errors": progress.errors,
        "paper_processing_failures": [
            row for row in paper_results if row.get("status") == "failed"
        ],
        "mutation_count_distribution": mutation_distribution,
        "mutation_policy_version": MUTATION_POLICY_VERSION,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "strict_input_filter_policy_version": strict_input_audit["policy_version"],
        "strict_input_filter_audit": "strict_input_filter_audit.json",
        "strict_input_pages_scanned": strict_input_audit["scanned_pages"],
        "strict_input_pages_accepted": strict_input_audit["accepted_pages"],
        "strict_input_pages_rejected": strict_input_audit["rejected_pages"],
        "strict_input_rejection_reason_counts": strict_input_audit["reason_counts"],
        "bibliography_policy_version": BIBLIOGRAPHY_POLICY_VERSION,
        "stale_artifact_cleanup": stale_artifact_cleanup,
        "confusable_map": {key: list(value) for key, value in CONFUSABLES.items()},
        "digits_allowed": False,
        "length_changing_edits_allowed": False,
        "output_mode": "edited_only",
        "clean_assets_copied": False,
        "dataset_root": str(output_dir),
        "image_path_policy": "absolute_output_dir_v1",
        "exports": exports,
        "paper_results": paper_results,
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    print(
        f"[final] status={report['status']} papers={len(eligible_papers)} "
        f"strict_pages={progress.total_pages} accepted={len(all_pairs)} "
        f"rejected={progress.rejected + len(selected_bibliography_exclusions)} "
        f"bibliography_excluded={len(selected_bibliography_exclusions)} "
        f"errors={len(accepted_subset_issues)} "
        f"paper_errors={progress.errors} "
        f"mutation_distribution={mutation_distribution} output={output_dir}",
        flush=True,
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
