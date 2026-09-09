#!/usr/bin/env python3
"""Build long, text-only confusable-rewrite SFT data from arXiv sources.

The input is the directory produced by ``crawl_arxiv_sources.py`` and contains
``papers/<stem>/source_archive.bin`` plus either ``results.jsonl`` or per-paper
``download.json`` checkpoints. The script safely
extracts each source archive, conservatively converts visible LaTeX prose to
Markdown, creates deterministic same-length character confusions, and writes
MS-Swift ``messages`` JSONL shards. The input document A remains unchanged;
the answer B changes only existing heading levels, with an optional outer
Markdown fence (enabled by default).

This is intentionally a text-rewrite auxiliary task, not page OCR: no PDF is read
and no image field is emitted.  A target such as one million rows is a ceiling,
never a quota.  If the eligible source corpus yields fewer unique samples, all
eligible samples are written and the shortfall is reported.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import tarfile
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arxiv_confusable_pairs import PAIR_COUNTS, PAIR_WEIGHTS, POLICY_FINGERPRINT
from arxiv_source_first_v3.table_ast import (
    TABLE_AST_VERSION,
    TableAstError,
    parse_strict_table,
)

SCHEMA_VERSION = 2
PIPELINE_VERSION = "arxiv_confusable_text_sft_v5_weighted_pairs"
PROMPT_VERSION = "heading_rewrite_boundary_en_v3"
HEADING_POLICY_VERSION = "block_start_heading_levels_0_to_4_zero28_v2"
NO_HEADING_PROBABILITY = 0.28
MUTATION_POLICY_VERSION = "chaos_text_empirical_1012_pair_first_v3"
DEFAULT_MUTATION_WORD_RATIO = 0.10
DEFAULT_MIN_MUTATIONS = 3
DEFAULT_MAX_MUTATIONS = 0
HEARTBEAT_SECONDS = 30.0

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
NO_FENCE_PROMPT_VERSION = "heading_rewrite_boundary_en_v4_no_fence"
# Only these two prompt edits were approved for the optional no-fence mode.
NO_FENCE_PROMPT_PREFIX = PROMPT_PREFIX.replace(
    "these two formatting changes. This is not a translation task.",
    "the following heading-prefix change. This is not a translation task.",
).replace(
    "2. Enclose the entire result in a Markdown code fence: start with "
    "```markdown followed by a newline, and end with a newline followed by ```.",
    "2. Output the rewritten document directly. Do not add an outer Markdown code fence.",
)
RESPONSE_PREFIX = "```markdown\n"
RESPONSE_SUFFIX = "\n```"
HEADING_PREFIX_RE = re.compile(r"^(#{1,6})(?!#)")
TABLE_ENVIRONMENTS = {"tabular", "tabular*", "tabularx"}
TABLE_FLOAT_ENVIRONMENTS = {"table", "table*"}
ENVIRONMENT_TOKEN_RE = re.compile(r"\\(begin|end)\s*\{\s*([A-Za-z*]+)\s*\}")

ALLOWED_LICENSES = {"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"}
SAFE_STEM_RE = re.compile(r"^[A-Za-z0-9._-]+$")
WORD_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]{4,}(?![A-Za-z])")
CONTROL_SEQUENCE_RE = re.compile(r"\\[A-Za-z@]+\*?")

MAX_ARCHIVE_FILES = 5_000
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_MEMBER_BYTES = 100 * 1024 * 1024
MAX_TEX_FILES_PER_PAPER = 500
MAX_TEX_FILE_BYTES = 20 * 1024 * 1024

EXCLUDED_ENVIRONMENTS = {
    "thebibliography",
    "bibliography",
    "references",
    "figure",
    "figure*",
    "table",
    "table*",
    "tabular",
    "tabular*",
    "tabularx",
    "longtable",
    "algorithm",
    "algorithm*",
    "algorithmic",
    "algorithm2e",
    "listing",
    "lstlisting",
    "minted",
    "verbatim",
    "Verbatim",
    "tikzpicture",
    "picture",
    "equation",
    "equation*",
    "align",
    "align*",
    "alignat",
    "alignat*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "math",
}
ALLOWED_STRUCTURAL_ENVIRONMENTS = {
    "document",
    "abstract",
    "itemize",
    "enumerate",
    "description",
    "quote",
    "quotation",
    "center",
    "flushleft",
    "flushright",
    "proof",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "remark",
    "example",
}
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:references|bibliography|works cited|literature cited)\s*:?[\s*]*$",
    re.IGNORECASE,
)
DEFINITION_COMMAND_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|def|gdef|xdef|edef|newenvironment|"
    r"renewenvironment|DeclareMathOperator|usepackage|RequirePackage|documentclass)\b"
)

STYLE_COMMANDS = {
    "textbf": ("**", "**"),
    "mathbf": ("**", "**"),
    "emph": ("*", "*"),
    "textit": ("*", "*"),
    "textsl": ("*", "*"),
    "texttt": ("`", "`"),
    "textsc": ("", ""),
    "textrm": ("", ""),
    "textsf": ("", ""),
    "mbox": ("", ""),
    "hbox": ("", ""),
    "makebox": ("", ""),
    "underline": ("", ""),
}
HEADING_COMMANDS = {
    "part": "#",
    "chapter": "#",
    "section": "#",
    "subsection": "##",
    "subsubsection": "###",
    "paragraph": "####",
    "subparagraph": "#####",
}
CITE_COMMANDS = {
    "cite",
    "citep",
    "citet",
    "citealp",
    "citealt",
    "citeauthor",
    "citeyear",
    "citeyearpar",
    "parencite",
    "textcite",
    "autocite",
    "footcite",
    "nocite",
}
DROPPABLE_CITE_COMMANDS = {
    "cite",
    "citep",
    "parencite",
    "autocite",
    "footcite",
    "nocite",
}
REFERENCE_COMMANDS = {
    "ref",
    "pageref",
    "eqref",
    "autoref",
    "cref",
    "Cref",
    "vref",
}
REJECT_ARGUMENT_COMMANDS = REFERENCE_COMMANDS | (CITE_COMMANDS - DROPPABLE_CITE_COMMANDS)
DROP_ARGUMENT_COMMANDS = DROPPABLE_CITE_COMMANDS | {
    "label",
    "index",
    "glossary",
    "bibliography",
    "bibliographystyle",
    "addbibresource",
}
REJECT_COMMANDS = {
    "footnote",
    "footnotetext",
    "thanks",
    "url",
    "path",
    "includegraphics",
    "input",
    "include",
    "subfile",
    "lstinline",
    "verb",
    "write",
    "openout",
    "openin",
    "read",
    "title",
    "author",
    "date",
    "affiliation",
    "institute",
    "email",
    "address",
}
NO_ARGUMENT_TEXT = {
    "LaTeX": "LaTeX",
    "TeX": "TeX",
    "BibTeX": "BibTeX",
    "ldots": "…",
    "dots": "…",
    "textbackslash": "\\",
    "textasciitilde": "~",
    "textasciicircum": "^",
    "copyright": "©",
}
LAYOUT_COMMANDS = {
    "noindent",
    "indent",
    "smallskip",
    "medskip",
    "bigskip",
    "hfill",
    "vfill",
    "quad",
    "qquad",
    "centering",
    "raggedright",
    "raggedleft",
    "normalfont",
    "rmfamily",
    "sffamily",
    "ttfamily",
    "upshape",
    "itshape",
    "slshape",
    "scshape",
    "bfseries",
    "mdseries",
    "tiny",
    "scriptsize",
    "footnotesize",
    "small",
    "normalsize",
    "large",
    "Large",
    "LARGE",
    "huge",
    "Huge",
    "protect",
}


class RejectedSource(ValueError):
    """A source fragment cannot be converted without guessing."""


@dataclass(frozen=True)
class TextBlock:
    source_file: str
    source_start: int
    source_end: int
    line_start: int
    line_end: int
    markdown: str
    kind: str = "text"


@dataclass(frozen=True)
class LengthBucket:
    minimum: int
    maximum: int
    weight: float


@dataclass(frozen=True)
class WorkerConfig:
    fingerprint: str
    seed: int
    tokenizer: str
    tokenizer_local_only: bool
    trust_remote_code: bool
    length_buckets: tuple[LengthBucket, ...]
    min_response_tokens: int
    max_response_tokens: int
    mutation_word_ratio: float
    min_mutations: int
    max_mutations: int
    max_samples_per_paper: int
    temp_root: str | None
    resume: bool
    retry_failed: bool
    response_fence: str = "markdown"


@dataclass
class PaperResult:
    stem: str
    status: str
    checkpoint: str | None
    samples: int
    candidate_chunks: int
    tex_files: int
    blocks: int
    extracted_bytes: int
    rejection_reasons: dict[str, int]
    error: str | None = None
    reused: bool = False


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def elapsed_text(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def stable_digest(*values: object) -> bytes:
    encoded = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def stable_seed(*values: object) -> int:
    return int.from_bytes(stable_digest(*values)[:8], "big")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for count, row in enumerate(rows, start=1):
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield row


def paper_stem(row: dict[str, Any]) -> str:
    stem = str(row.get("stem") or f"{row.get('arxiv_id', '')}{row.get('version', '')}")
    if not stem or not SAFE_STEM_RE.fullmatch(stem):
        raise ValueError(f"unsafe or missing paper stem: {stem!r}")
    return stem


def resolve_archive(input_root: Path, row: dict[str, Any]) -> Path:
    stem = paper_stem(row)
    candidates: list[Path] = []
    for key in ("archive", "archive_path"):
        value = row.get(key)
        if value:
            candidate = Path(str(value))
            candidates.append(candidate if candidate.is_absolute() else input_root / candidate)
    candidates.append(input_root / "papers" / stem / "source_archive.bin")
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise FileNotFoundError(f"source archive is missing or empty for {stem}")


def validate_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def copy_member(source: BinaryIO, destination: Path) -> int:
    written = 0
    with destination.open("wb") as stream:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_MEMBER_BYTES:
                raise ValueError(f"archive member exceeds limit: {destination}")
            stream.write(chunk)
    return written


def extract_source(archive: Path, destination: Path) -> tuple[int, int, str]:
    destination.mkdir(parents=True, exist_ok=True)
    files = 0
    total_bytes = 0
    try:
        with tarfile.open(archive, mode="r:*") as bundle:
            members = bundle.getmembers()
            if len(members) > MAX_ARCHIVE_FILES:
                raise ValueError(f"archive has too many members: {len(members)}")
            expanded = sum(member.size for member in members if member.isfile())
            if expanded > MAX_ARCHIVE_BYTES:
                raise ValueError(f"archive expands beyond byte limit: {expanded}")
            for member in members:
                relative = validate_member_name(member.name)
                output = destination.joinpath(*relative.parts)
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"links/devices are not allowed: {member.name!r}")
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported archive member: {member.name!r}")
                if member.size > MAX_MEMBER_BYTES:
                    raise ValueError(f"archive member exceeds limit: {member.name!r}")
                output.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name!r}")
                with source:
                    total_bytes += copy_member(source, output)
                output.chmod(0o644)
                files += 1
            return files, total_bytes, "tar"
    except tarfile.ReadError:
        pass

    raw = archive.read_bytes()
    if raw.startswith(b"%PDF"):
        raise ValueError("PDF-only arXiv payload has no TeX source")
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    if len(raw) > MAX_MEMBER_BYTES:
        raise ValueError("single-file source exceeds size limit")
    text = raw.decode("utf-8", errors="replace")
    if "\\documentclass" not in text:
        raise ValueError("payload is neither a safe tar archive nor standalone TeX")
    output = destination / "main.tex"
    output.write_text(text, encoding="utf-8")
    output.chmod(0o644)
    return 1, output.stat().st_size, "single_tex"


def tex_comment_start(line: str) -> int | None:
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
    return None


def strip_tex_comments(text: str) -> str:
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        newline = "\n" if line.endswith(("\n", "\r")) else ""
        body = line.rstrip("\r\n")
        index = tex_comment_start(body)
        output.append((body if index is None else body[:index]) + newline)
    return "".join(output)


def _blank_match(match: re.Match[str]) -> str:
    return "\n" * match.group(0).count("\n")


def remove_excluded_environments(text: str) -> str:
    for environment in sorted(EXCLUDED_ENVIRONMENTS, key=len, reverse=True):
        name = re.escape(environment)
        pattern = re.compile(
            rf"\\begin\s*\{{{name}\}}.*?\\end\s*\{{{name}\}}",
            re.DOTALL,
        )
        previous = None
        while previous != text:
            previous = text
            text = pattern.sub(_blank_match, text)
    return text


def remove_bibliography_tail(text: str) -> str:
    marker = re.search(
        r"\\begin\s*\{thebibliography\}|\\printbibliography\b|"
        r"\\bibliography\s*\{|"
        r"\\(?:section|chapter)\*?\s*\{\s*(?:references|bibliography|works cited)\s*\}",
        text,
        re.IGNORECASE,
    )
    if not marker:
        return text
    return text[: marker.start()] + "\n" * text[marker.start() :].count("\n")


def document_body(text: str) -> tuple[str, int]:
    line_offset = 0
    begin = re.search(r"\\begin\s*\{document\}", text)
    if begin:
        line_offset = text.count("\n", 0, begin.end())
        text = text[begin.end() :]
    end = re.search(r"\\end\s*\{document\}", text)
    if end:
        text = text[: end.start()]
    return text, line_offset


class LatexMarkdownParser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)
        self.position = 0
        self.list_stack: list[str] = []

    def parse(self, stop: str | None = None) -> str:
        output: list[str] = []
        while self.position < self.length:
            character = self.source[self.position]
            if stop and character == stop:
                self.position += 1
                return "".join(output)
            if character == "}":
                raise RejectedSource("unbalanced_closing_brace")
            if character == "{":
                self.position += 1
                output.append(self.parse(stop="}"))
                continue
            if character == "$":
                output.append(self._math_dollar())
                continue
            if character == "\\":
                if self.source.startswith("\\(", self.position):
                    output.append(self._math_pair("\\(", "\\)", "$", "$"))
                    continue
                if self.source.startswith("\\[", self.position):
                    output.append(self._math_pair("\\[", "\\]", "$$", "$$"))
                    continue
                output.append(self._command())
                continue
            output.append(" " if character == "~" else character)
            self.position += 1
        if stop:
            raise RejectedSource("unclosed_group")
        return "".join(output)

    def _math_dollar(self) -> str:
        delimiter = "$$" if self.source.startswith("$$", self.position) else "$"
        start = self.position
        cursor = start + len(delimiter)
        while cursor < self.length:
            if self.source.startswith(delimiter, cursor):
                backslashes = 0
                left = cursor - 1
                while left >= 0 and self.source[left] == "\\":
                    backslashes += 1
                    left -= 1
                if backslashes % 2 == 0:
                    body = self.source[start + len(delimiter) : cursor].strip()
                    if not body or "\n\n" in body:
                        raise RejectedSource("unsafe_math_fragment")
                    self.position = cursor + len(delimiter)
                    return f"{delimiter}{body}{delimiter}"
            cursor += 1
        raise RejectedSource("unclosed_math")

    def _math_pair(self, opening: str, closing: str, left: str, right: str) -> str:
        start = self.position + len(opening)
        end = self.source.find(closing, start)
        if end < 0:
            raise RejectedSource("unclosed_math")
        body = self.source[start:end].strip()
        if not body or "\n\n" in body:
            raise RejectedSource("unsafe_math_fragment")
        self.position = end + len(closing)
        return f"{left}{body}{right}"

    def _consume_space(self) -> None:
        while self.position < self.length and self.source[self.position].isspace():
            self.position += 1

    def _optional_raw(self) -> str | None:
        self._consume_space()
        if self.position >= self.length or self.source[self.position] != "[":
            return None
        start = self.position + 1
        depth = 1
        cursor = start
        while cursor < self.length:
            if self.source[cursor] == "[":
                depth += 1
            elif self.source[cursor] == "]":
                depth -= 1
                if depth == 0:
                    value = self.source[start:cursor]
                    self.position = cursor + 1
                    return value
            cursor += 1
        raise RejectedSource("unclosed_optional_argument")

    def _required_raw(self) -> str:
        self._consume_space()
        if self.position >= self.length or self.source[self.position] != "{":
            raise RejectedSource("missing_required_argument")
        start = self.position + 1
        depth = 1
        cursor = start
        while cursor < self.length:
            character = self.source[cursor]
            if character == "\\":
                cursor += 2
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    value = self.source[start:cursor]
                    self.position = cursor + 1
                    return value
            cursor += 1
        raise RejectedSource("unclosed_required_argument")

    @staticmethod
    def _render_fragment(raw: str) -> str:
        return LatexMarkdownParser(raw).parse()

    def _command_name(self) -> str:
        assert self.source[self.position] == "\\"
        self.position += 1
        if self.position >= self.length:
            raise RejectedSource("trailing_backslash")
        if not (self.source[self.position].isalpha() or self.source[self.position] == "@"):
            value = self.source[self.position]
            self.position += 1
            return value
        start = self.position
        while self.position < self.length and (
            self.source[self.position].isalpha() or self.source[self.position] == "@"
        ):
            self.position += 1
        name = self.source[start:self.position]
        if self.position < self.length and self.source[self.position] == "*":
            self.position += 1
        return name

    def _accent(self, name: str) -> str:
        accents = {
            "'": "\u0301",
            "`": "\u0300",
            "^": "\u0302",
            '"': "\u0308",
            "~": "\u0303",
            "=": "\u0304",
            ".": "\u0307",
            "u": "\u0306",
            "v": "\u030c",
            "H": "\u030b",
            "c": "\u0327",
            "k": "\u0328",
            "b": "\u0331",
            "d": "\u0323",
        }
        self._consume_space()
        if self.position >= self.length:
            raise RejectedSource("missing_accent_argument")
        braced = self.source[self.position] == "{"
        raw = self._required_raw() if braced else self.source[self.position]
        if not raw:
            raise RejectedSource("empty_accent")
        if not braced:
            self.position += 1
        rendered = self._render_fragment(raw)
        if len(rendered) != 1:
            raise RejectedSource("complex_accent")
        return unicodedata.normalize("NFC", rendered + accents[name])

    def _command(self) -> str:
        name = self._command_name()
        escaped = {"%": "%", "&": "&", "_": "\\_", "#": "#", "$": "$", "{": "{", "}": "}", " ": " ", "\\": "\n"}
        if name in escaped:
            return escaped[name]
        if name in {"'", "`", "^", '"', "~", "=", ".", "u", "v", "H", "c", "k", "b", "d"}:
            return self._accent(name)
        if name in STYLE_COMMANDS:
            self._optional_raw()
            body = self._render_fragment(self._required_raw()).strip()
            if not body:
                raise RejectedSource("empty_style_argument")
            left, right = STYLE_COMMANDS[name]
            if name == "texttt" and "`" in body:
                raise RejectedSource("nested_code_delimiter")
            return f"{left}{body}{right}"
        if name in HEADING_COMMANDS:
            self._optional_raw()
            title = self._render_fragment(self._required_raw()).strip()
            if not title:
                raise RejectedSource("empty_heading")
            return f"\n\n{HEADING_COMMANDS[name]} {title}\n\n"
        if name in REJECT_ARGUMENT_COMMANDS:
            raise RejectedSource(f"visible_reference_command:{name}")
        if name in DROP_ARGUMENT_COMMANDS:
            while self._optional_raw() is not None:
                pass
            self._required_raw()
            return ""
        if name in REJECT_COMMANDS:
            raise RejectedSource(f"rejected_command:{name}")
        if name == "href":
            self._optional_raw()
            self._required_raw()
            return self._render_fragment(self._required_raw())
        if name in {"textsuperscript", "textsubscript"}:
            body = self._render_fragment(self._required_raw()).strip()
            tag = "sup" if name == "textsuperscript" else "sub"
            return f"<{tag}>{body}</{tag}>"
        if name in {"texorpdfstring", "ifthenelse"}:
            first = self._render_fragment(self._required_raw())
            self._required_raw()
            if name == "ifthenelse":
                return self._render_fragment(self._required_raw())
            return first
        if name in NO_ARGUMENT_TEXT:
            return NO_ARGUMENT_TEXT[name]
        if name in LAYOUT_COMMANDS:
            return ""
        if name in {"par", "newline", "linebreak"}:
            self._optional_raw()
            return "\n\n" if name == "par" else "\n"
        if name == "item":
            label = self._optional_raw()
            if label:
                rendered = self._render_fragment(label).strip()
                return f"\n\n- **{rendered}:** "
            prefix = "1. " if self.list_stack and self.list_stack[-1] == "enumerate" else "- "
            return f"\n\n{prefix}"
        if name in {"begin", "end"}:
            environment = self._required_raw().strip()
            if environment not in ALLOWED_STRUCTURAL_ENVIRONMENTS:
                raise RejectedSource(f"unknown_environment:{environment}")
            if environment in {"itemize", "enumerate", "description"}:
                if name == "begin":
                    self.list_stack.append(environment)
                elif self.list_stack and self.list_stack[-1] == environment:
                    self.list_stack.pop()
                else:
                    raise RejectedSource("unbalanced_list_environment")
            return "\n\n"
        if name in {",", ";", ":", "!", "/"}:
            return ""
        raise RejectedSource(f"unknown_command:{name}")


def normalize_markdown(value: str) -> str:
    value = value.replace("\r", "\n").replace("\x00", "")
    lines: list[str] = []
    for line in value.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"\(\s*[,;]\s*\)", "", line)
        lines.append(line)
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value


def convert_fragment(raw: str) -> str:
    if DEFINITION_COMMAND_RE.search(raw):
        raise RejectedSource("definition_or_preamble_command")
    markdown = normalize_markdown(LatexMarkdownParser(raw).parse())
    if not markdown:
        raise RejectedSource("empty_after_conversion")
    if REFERENCE_HEADING_RE.fullmatch(markdown):
        raise RejectedSource("reference_heading")
    if "<<<DOCUMENT_" in markdown:
        raise RejectedSource("boundary_collision")
    if len(WORD_RE.findall(markdown)) < 3 and not markdown.startswith("#"):
        raise RejectedSource("too_few_words")
    return markdown


def excluded_regions(
    text: str, names: set[str],
) -> Iterator[tuple[str, int, int, bool]]:
    """Yield outer environment spans, without exposing nested excluded content."""
    stack: list[str] = []
    start = 0
    name = ""
    balanced = True
    for token in ENVIRONMENT_TOKEN_RE.finditer(text):
        command, environment = token.groups()
        if not stack:
            if command != "begin" or environment not in names:
                continue
            start, name, balanced = token.start(), environment, True
            stack.append(environment)
        elif command == "begin":
            stack.append(environment)
        elif environment == stack[-1]:
            stack.pop()
            if not stack:
                yield name, start, token.end(), balanced
        elif environment == name:
            # Drop a malformed outer block as a whole, not its cell text.
            stack.clear()
            yield name, start, token.end(), False
        else:
            balanced = False
    if stack:
        yield name, start, len(text), False


def source_table_blocks(
    fragment: str, *, source_file: str,
) -> tuple[list[tuple[int, int, str, str]], Counter[str]]:
    """Convert complete source tables; keep captions outside the HTML table."""
    output: list[tuple[int, int, str, str]] = []
    reasons: Counter[str] = Counter()
    spans = list(excluded_regions(fragment, TABLE_ENVIRONMENTS | {"longtable"}))
    if not spans:
        reasons["table:no_supported_tabular"] += 1
        return output, reasons
    for environment, start, end, balanced in spans:
        if not balanced or environment not in TABLE_ENVIRONMENTS:
            reasons["table:unsupported_or_unbalanced_environment"] += 1
            continue
        try:
            table = parse_strict_table(
                fragment, start=start, end=end, source_id=source_file,
            )
        except TableAstError as exc:
            reasons[f"table:{exc}"] += 1
            continue
        output.append((start, end, table.html, "table"))
    # A caption alone must not survive an entirely rejected table float.
    if not output:
        return output, reasons
    for match in re.finditer(r"\\caption\*?(?![A-Za-z@])", fragment):
        if any(start <= match.start() < end for _, start, end, _ in spans):
            continue
        parser = LatexMarkdownParser(fragment)
        parser.position = match.end()
        try:
            parser._optional_raw()
            raw = parser._required_raw()
            caption = normalize_markdown(LatexMarkdownParser(raw).parse())
            if not caption or "<<<DOCUMENT_" in caption:
                raise RejectedSource("empty_or_unsafe_caption")
        except RejectedSource as exc:
            reasons[f"table_caption:{exc}"] += 1
            continue
        output.append((match.start(), parser.position, caption, "caption"))
    return sorted(output), reasons


def extract_blocks(path: Path, source_root: Path) -> tuple[list[TextBlock], Counter[str]]:
    reasons: Counter[str] = Counter()
    if path.stat().st_size > MAX_TEX_FILE_BYTES:
        reasons["tex_file_too_large"] += 1
        return [], reasons
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "\ufffd" in raw:
        reasons["decode_replacement_character"] += 1
        return [], reasons
    text, line_offset = document_body(remove_bibliography_tail(strip_tex_comments(raw)))
    blocks: list[TextBlock] = []
    relative = path.relative_to(source_root).as_posix()

    def add(start: int, end: int, markdown: str, kind: str = "text") -> None:
        blocks.append(
            TextBlock(
                source_file=relative,
                source_start=start,
                source_end=end,
                line_start=line_offset + text.count("\n", 0, start) + 1,
                line_end=line_offset + text.count("\n", 0, end) + 1,
                markdown=markdown,
                kind=kind,
            )
        )

    def add_prose(start: int, end: int) -> None:
        for match in re.finditer(r"\S(?:.*?)(?=\n[ \t]*\n|\Z)", text[start:end], re.DOTALL):
            fragment = match.group(0)
            if len(fragment) > 2_000_000:
                reasons["source_fragment_too_large"] += 1
                continue
            try:
                markdown = convert_fragment(fragment)
            except RejectedSource as exc:
                reasons[str(exc)] += 1
                continue
            add(start + match.start(), start + match.end(), markdown)

    cursor = 0
    for environment, start, end, balanced in excluded_regions(text, EXCLUDED_ENVIRONMENTS):
        add_prose(cursor, start)
        if environment in TABLE_ENVIRONMENTS | TABLE_FLOAT_ENVIRONMENTS | {"longtable"}:
            if not balanced:
                reasons["table:unbalanced_environment"] += 1
            else:
                tables, table_reasons = source_table_blocks(text[start:end], source_file=relative)
                reasons.update(table_reasons)
                for local_start, local_end, markdown, kind in tables:
                    add(start + local_start, start + local_end, markdown, kind)
        cursor = end
    add_prose(cursor, len(text))
    blocks.sort(key=lambda block: block.source_start)
    return blocks, reasons


class TokenCounter:
    def count(self, value: str) -> int:
        raise NotImplementedError


class SimpleTokenCounter(TokenCounter):
    """Deterministic test-only tokenizer; not suitable for final 8k limits."""

    def count(self, value: str) -> int:
        return len(re.findall(r"[A-Za-z0-9_]+|[^\w\s]", value, re.UNICODE))


class HuggingFaceTokenCounter(TokenCounter):
    def __init__(self, identifier: str, *, local_only: bool, trust_remote_code: bool) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "exact token limits require transformers; install it or use --tokenizer simple only for tests"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            identifier,
            local_files_only=local_only,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

    def count(self, value: str) -> int:
        encoded = self.tokenizer.encode(value, add_special_tokens=False)
        return len(encoded)


_TOKEN_COUNTER_CACHE: dict[tuple[str, bool, bool], TokenCounter] = {}


def get_token_counter(config: WorkerConfig) -> TokenCounter:
    key = (config.tokenizer, config.tokenizer_local_only, config.trust_remote_code)
    if key not in _TOKEN_COUNTER_CACHE:
        if config.tokenizer == "simple":
            _TOKEN_COUNTER_CACHE[key] = SimpleTokenCounter()
        else:
            _TOKEN_COUNTER_CACHE[key] = HuggingFaceTokenCounter(
                config.tokenizer,
                local_only=config.tokenizer_local_only,
                trust_remote_code=config.trust_remote_code,
            )
    return _TOKEN_COUNTER_CACHE[key]


def choose_bucket(rng: random.Random, buckets: Sequence[LengthBucket]) -> LengthBucket:
    threshold = rng.random() * sum(bucket.weight for bucket in buckets)
    cumulative = 0.0
    for bucket in buckets:
        cumulative += bucket.weight
        if threshold <= cumulative:
            return bucket
    return buckets[-1]


def chunk_windows(
    blocks: Sequence[TextBlock],
    *,
    counter: TokenCounter,
    config: WorkerConfig,
    stem: str,
) -> list[tuple[str, int, list[TextBlock]]]:
    by_file: dict[str, list[TextBlock]] = defaultdict(list)
    for block in blocks:
        by_file[block.source_file].append(block)
    windows: list[tuple[str, int, list[TextBlock]]] = []
    seen_text: set[str] = set()
    for source_file in sorted(by_file):
        file_blocks = sorted(by_file[source_file], key=lambda block: block.source_start)
        token_counts = [counter.count(block.markdown) for block in file_blocks]
        separator_tokens = counter.count("\n\n")
        for start in range(len(file_blocks)):
            rng = random.Random(stable_seed(config.seed, stem, source_file, start))
            bucket = choose_bucket(rng, config.length_buckets)
            target = rng.randint(bucket.minimum, bucket.maximum)
            selected: list[TextBlock] = []
            approximate_tokens = 0
            index = start
            while index < len(file_blocks) and approximate_tokens < target:
                if selected:
                    approximate_tokens += separator_tokens
                approximate_tokens += token_counts[index]
                selected.append(file_blocks[index])
                index += 1
            if not selected:
                continue
            text = "\n\n".join(block.markdown for block in selected)
            tokens = counter.count(text)
            while tokens < target and index < len(file_blocks):
                selected.append(file_blocks[index])
                index += 1
                text = "\n\n".join(block.markdown for block in selected)
                tokens = counter.count(text)
            while tokens > config.max_response_tokens and selected:
                selected.pop()
                text = "\n\n".join(block.markdown for block in selected)
                tokens = counter.count(text) if selected else 0
            # Near the end of a source file the chosen long bucket may be
            # impossible.  Keep the longest globally valid tail instead of
            # discarding real, unique corpus merely to force the histogram.
            if tokens < config.min_response_tokens:
                continue
            digest = sha256_text(text)
            if digest in seen_text:
                continue
            seen_text.add(digest)
            windows.append((text, tokens, selected))
    if config.max_samples_per_paper and len(windows) > config.max_samples_per_paper:
        rng = random.Random(stable_seed(config.seed, stem, "paper-window-cap"))
        rng.shuffle(windows)
        windows = windows[: config.max_samples_per_paper]
    return windows


def protected_spans(markdown: str) -> list[tuple[int, int]]:
    patterns = [
        re.compile(r"```.*?```", re.DOTALL),
        re.compile(r"`[^`\n]*`"),
        re.compile(r"\$\$.*?\$\$", re.DOTALL),
        re.compile(r"(?<!\\)\$(?!\$).*?(?<!\\)\$", re.DOTALL),
        re.compile(r"<[^>]+>"),
        re.compile(r"&(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);"),
        re.compile(r"\]\([^)]*\)"),
        re.compile(r"https?://\S+|\b\S+@\S+\.\S+"),
    ]
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        spans.extend((match.start(), match.end()) for match in pattern.finditer(markdown))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def span_is_protected(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < protected_end and end > protected_start for protected_start, protected_end in spans)


def mutation_count_for_words(
    word_count: int,
    *,
    mutation_word_ratio: float,
    min_mutations: int,
    max_mutations: int,
) -> int:
    """Return an auditable occurrence-level mutation target.

    ``word_count`` includes ordinary ASCII word occurrences outside protected
    Markdown spans.  Repeated words count repeatedly: this policy targets about
    ten percent of visible word occurrences, rather than ten percent of unique
    vocabulary types.
    """

    if word_count < 0:
        raise ValueError("word_count cannot be negative")
    if not 0 < mutation_word_ratio <= 1:
        raise ValueError("mutation_word_ratio must be in (0, 1]")
    if min_mutations < 1:
        raise ValueError("min_mutations must be positive")
    if max_mutations < 0 or (max_mutations and max_mutations < min_mutations):
        raise ValueError("max_mutations must be 0 or at least min_mutations")
    desired = max(min_mutations, math.ceil(word_count * mutation_word_ratio))
    return min(desired, max_mutations) if max_mutations else desired


def mutation_word_denominator(
    markdown: str,
    spans: Sequence[tuple[int, int]] | None = None,
) -> int:
    protected = protected_spans(markdown) if spans is None else spans
    return sum(
        not span_is_protected(match.start(), match.end(), protected)
        for match in WORD_RE.finditer(markdown)
    )


def mutate_markdown(
    markdown: str,
    *,
    response_tokens: int,
    rng: random.Random,
    mutation_word_ratio: float = DEFAULT_MUTATION_WORD_RATIO,
    min_mutations: int = DEFAULT_MIN_MUTATIONS,
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
) -> tuple[str, list[dict[str, Any]]]:
    """Use V5 pair weights while retaining rewrite's occurrence-level budget.

    Draw the character pair first, then a matching word occurrence, then a
    valid position. Missing/exhausted pairs are removed, not replaced with
    unrelated substitutions. Pair weights are not multiplied by word counts.
    """
    del response_tokens  # Retained for call-site compatibility and audit context.
    spans = protected_spans(markdown)
    vocabulary = {match.group(0).lower() for match in WORD_RE.finditer(markdown)}
    occurrences = [
        match
        for match in WORD_RE.finditer(markdown)
        if not span_is_protected(match.start(), match.end(), spans)
    ]
    desired = mutation_count_for_words(
        len(occurrences),
        mutation_word_ratio=mutation_word_ratio,
        min_mutations=min_mutations,
        max_mutations=max_mutations,
    )
    by_pair: dict[tuple[str, str], list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for word_index, match in enumerate(occurrences):
        original = match.group(0)
        positions: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, character in enumerate(original):
            for replacement in PAIR_COUNTS.get(character, {}):
                mutated = original[:index] + replacement + original[index + 1 :]
                if mutated.lower() not in vocabulary:
                    positions[character, replacement].append(index)
        for pair, indexes in positions.items():
            by_pair[pair].append((word_index, tuple(indexes)))

    available = [pair for pair in PAIR_WEIGHTS if pair in by_pair]
    selected_indexes: set[int] = set()
    selected: list[tuple[int, int, str, str, int, str, str]] = []
    while available and len(selected) < desired:
        pair = rng.choices(available, weights=[PAIR_WEIGHTS[pair] for pair in available])[0]
        options = by_pair[pair]
        # Random removal gives uniform word sampling without repeatedly
        # copying all candidates; stale entries from other pairs are skipped.
        while options:
            option_index = rng.randrange(len(options))
            word_index, positions_tuple = options[option_index]
            options[option_index] = options[-1]
            options.pop()
            if word_index not in selected_indexes:
                break
        else:
            available.remove(pair)
            continue
        selected_indexes.add(word_index)
        match = occurrences[word_index]
        original = match.group(0)
        char_index = rng.choice(positions_tuple)
        replacement = pair[1]
        mutated = original[:char_index] + replacement + original[char_index + 1 :]
        selected.append(
            (
                match.start(),
                match.end(),
                original,
                mutated,
                char_index,
                original[char_index],
                replacement,
            )
        )
    if len(selected) != desired:
        raise RejectedSource(
            "fewer_than_required_mutations:"
            f"{len(selected)}<{desired}:words={len(occurrences)}:"
            f"ratio={mutation_word_ratio:g}"
        )

    edited = markdown
    for start, end, _original, mutated, _index, _old, _new in sorted(selected, reverse=True):
        edited = edited[:start] + mutated + edited[end:]
    if len(edited) != len(markdown):
        raise AssertionError("same-length mutation policy changed character count")
    changes: list[dict[str, Any]] = []
    for start, end, original, mutated, char_index, old, new in sorted(selected):
        if edited[start:end] != mutated:
            raise AssertionError("mutation offset does not locate edited word")
        changes.append(
            {
                "origin_ans": original,
                "ocr_ans": mutated,
                "from_char": old,
                "to_char": new,
                "char_index_in_word": char_index,
                "char_offset": start,
                "char_end": end,
            }
        )
    return edited, changes


def prompt_version_for(response_fence: str) -> str:
    return NO_FENCE_PROMPT_VERSION if response_fence == "none" else PROMPT_VERSION


def response_affixes(response_fence: str) -> tuple[str, str]:
    if response_fence == "none":
        return "", ""
    if response_fence == "markdown":
        return RESPONSE_PREFIX, RESPONSE_SUFFIX
    raise ValueError(f"unsupported response fence: {response_fence}")


def build_prompt(markdown: str, *, response_fence: str = "markdown") -> str:
    prefix = NO_FENCE_PROMPT_PREFIX if response_fence == "none" else PROMPT_PREFIX
    return prefix + markdown + PROMPT_SUFFIX


def rewrite_heading_levels(
    markdown: str,
    *,
    blocks: Sequence[TextBlock],
    rng: random.Random,
) -> tuple[str, list[dict[str, int]]]:
    """Change only the leading hashes of existing source text-block headings.

    Character mutations keep A the same length as the joined source blocks,
    so these offsets remain exact even when words in a heading were mutated.
    Heading metadata uses offsets in A; word metadata separately maps A and B.
    """
    changes: list[dict[str, int]] = []
    offset = 0
    for block in blocks:
        match = HEADING_PREFIX_RE.match(block.markdown) if block.kind == "text" else None
        if match:
            old_level = len(match[1])
            # Draw removal separately so excluding the old level does not
            # inflate its probability above 28%.
            new_level = (
                0 if rng.random() < NO_HEADING_PROBABILITY
                else rng.choice([level for level in range(1, 5) if level != old_level])
            )
            changes.append(
                {"input_offset": offset, "from_level": old_level, "to_level": new_level}
            )
        offset += len(block.markdown) + 2  # Windows join blocks with two newlines.
    rewritten = markdown
    for change in reversed(changes):
        start = change["input_offset"]
        rewritten = (
            rewritten[:start] + "#" * change["to_level"]
            + rewritten[start + change["from_level"]:]
        )
    return rewritten, changes


def validate_sample(row: dict[str, Any], *, max_response_tokens: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("sample must contain exactly user and assistant messages")
    user = messages[0]
    assistant = messages[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("invalid message roles")
    prompt = user.get("content")
    answer = assistant.get("content")
    if not isinstance(prompt, str) or not isinstance(answer, str) or not answer:
        raise ValueError("prompt and answer must be non-empty strings")
    if prompt.count("<<<DOCUMENT_START>>>") != 1 or prompt.count("<<<DOCUMENT_END>>>") != 1:
        raise ValueError("boundary markers must occur exactly once")
    extracted = prompt.split("<<<DOCUMENT_START>>>\n", 1)[1].rsplit("\n<<<DOCUMENT_END>>>", 1)[0]
    if "images" in row or "<image>" in prompt:
        raise ValueError("text-only rewrite sample must not contain image fields or markers")
    extra = row.get("extra_info")
    if not isinstance(extra, dict):
        raise ValueError("missing extra_info")
    response_prefix, response_suffix = response_affixes(extra.get("response_fence", "markdown"))
    headings = extra.get("heading_changes")
    if not isinstance(headings, list):
        raise ValueError("missing heading changes")
    previous_end = 0
    for heading in headings:
        start = heading["input_offset"]
        old_level = heading["from_level"]
        new_level = heading["to_level"]
        if (
            start < previous_end or not 1 <= old_level <= 6
            or new_level not in (0, 1, 2, 3, 4) or new_level == old_level
        ):
            raise ValueError("invalid heading level change")
        match = HEADING_PREFIX_RE.match(extracted[start:])
        if match is None or len(match[1]) != old_level:
            raise ValueError("heading offset mismatch")
        previous_end = start + old_level
    expected_body = extracted
    for heading in reversed(headings):
        start = heading["input_offset"]
        expected_body = (
            expected_body[:start] + "#" * heading["to_level"]
            + expected_body[start + heading["from_level"]:]
        )
    if answer != response_prefix + expected_body + response_suffix:
        raise ValueError("answer differs beyond heading levels and Markdown fence")
    response_tokens = extra.get("response_tokens")
    if not isinstance(response_tokens, int) or response_tokens <= 0 or response_tokens > max_response_tokens:
        raise ValueError("invalid response token count")
    changes = extra.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("missing mutation changes")
    mutation_word_count = extra.get("mutation_word_denominator")
    mutation_word_ratio = extra.get("mutation_word_ratio_requested")
    mutation_word_ratio_achieved = extra.get("mutation_word_ratio_achieved")
    mutation_target = extra.get("mutation_target")
    if not isinstance(mutation_word_count, int) or mutation_word_count <= 0:
        raise ValueError("invalid mutation word denominator")
    if not isinstance(mutation_word_ratio, (int, float)) or not 0 < mutation_word_ratio <= 1:
        raise ValueError("invalid requested mutation word ratio")
    if not isinstance(mutation_target, int) or mutation_target != len(changes):
        raise ValueError("mutation target does not match recorded changes")
    expected_achieved = len(changes) / mutation_word_count
    if not isinstance(mutation_word_ratio_achieved, (int, float)) or not math.isclose(
        mutation_word_ratio_achieved, expected_achieved, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("invalid achieved mutation word ratio")
    seen_offsets: set[int] = set()
    for change in changes:
        start = int(change["char_offset"])
        end = int(change["char_end"])
        mutated = str(change["ocr_ans"])
        original = str(change["origin_ans"])
        if answer[start:end] != mutated:
            raise ValueError("mutation offset mismatch")
        input_start = int(change["input_char_offset"])
        input_end = int(change["input_char_end"])
        if extracted[input_start:input_end] != mutated:
            raise ValueError("input mutation offset mismatch")
        if len(original) != len(mutated):
            raise ValueError("mutation changed word length")
        if start in seen_offsets:
            raise ValueError("multiple mutations share one word occurrence")
        seen_offsets.add(start)


def make_sample(
    *,
    row: dict[str, Any],
    stem: str,
    markdown: str,
    clean_tokens: int,
    blocks: Sequence[TextBlock],
    counter: TokenCounter,
    config: WorkerConfig,
) -> dict[str, Any]:
    source_file = blocks[0].source_file
    rng = random.Random(
        stable_seed(config.seed, stem, source_file, blocks[0].source_start, blocks[-1].source_end)
    )
    mutation_word_count = mutation_word_denominator(markdown)
    mutation_target = mutation_count_for_words(
        mutation_word_count,
        mutation_word_ratio=config.mutation_word_ratio,
        min_mutations=config.min_mutations,
        max_mutations=config.max_mutations,
    )
    edited, changes = mutate_markdown(
        markdown,
        response_tokens=clean_tokens,
        rng=rng,
        mutation_word_ratio=config.mutation_word_ratio,
        min_mutations=config.min_mutations,
        max_mutations=config.max_mutations,
    )
    # Do not consume the character-mutation RNG or change its input: A and its
    # word mutations must be identical to the original copy-task pipeline.
    heading_rng = random.Random(
        stable_seed(
            config.seed, stem, source_file, blocks[0].source_start,
            blocks[-1].source_end, HEADING_POLICY_VERSION,
        )
    )
    rewritten, heading_changes = rewrite_heading_levels(edited, blocks=blocks, rng=heading_rng)
    response_prefix, response_suffix = response_affixes(config.response_fence)
    answer = response_prefix + rewritten + response_suffix
    response_tokens = counter.count(answer)
    if response_tokens < config.min_response_tokens:
        raise RejectedSource("edited_response_below_min_tokens")
    if response_tokens > config.max_response_tokens:
        raise RejectedSource("edited_response_above_max_tokens")
    signature = ";".join(
        f"{change['char_offset']}:{change['origin_ans']}:{change['ocr_ans']}" for change in changes
    )
    heading_signature = ";".join(
        f"{change['input_offset']}:{change['from_level']}:{change['to_level']}"
        for change in heading_changes
    )
    sample_digest = stable_digest(
        source_file, blocks[0].source_start, blocks[-1].source_end,
        signature, PIPELINE_VERSION, heading_signature,
    ).hex()[:20]
    if config.response_fence == "none":
        sample_digest = stable_digest(sample_digest, NO_FENCE_PROMPT_VERSION).hex()[:20]
    sample_id = f"{stem}_{sample_digest}"
    for change in changes:
        start, end = change["char_offset"], change["char_end"]
        shift = len(response_prefix) + sum(
            heading["to_level"] - heading["from_level"]
            for heading in heading_changes if heading["input_offset"] < start
        )
        change.update(
            input_char_offset=start, input_char_end=end,
            char_offset=start + shift, char_end=end + shift,
        )
    source_spans = [
        {
            "source_file": block.source_file,
            "normalized_source_span": [block.source_start, block.source_end],
            "source_lines": [block.line_start, block.line_end],
            "kind": block.kind,
        }
        for block in blocks
    ]
    sample = {
        "messages": [
            {"role": "user", "content": build_prompt(edited, response_fence=config.response_fence)},
            {"role": "assistant", "content": answer},
        ],
        "data_source": "arxiv_confusable_text_copy",
        "ability": "heading_format_rewrite",
        "extra_info": {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "prompt_version": prompt_version_for(config.response_fence),
            "mutation_policy_version": MUTATION_POLICY_VERSION,
            "mutation_pair_policy_fingerprint": POLICY_FINGERPRINT,
            "heading_policy_version": HEADING_POLICY_VERSION,
            "heading_changes": heading_changes,
            "sample_id": sample_id,
            "arxiv_id": row.get("arxiv_id"),
            "version": row.get("version"),
            "paper_id": stem,
            "categories": row.get("categories"),
            "license_name": row.get("license_name"),
            "license_url": row.get("license_url"),
            "response_tokens": response_tokens,
            "mutation_word_denominator": mutation_word_count,
            "mutation_word_ratio_requested": config.mutation_word_ratio,
            "mutation_word_ratio_achieved": len(changes) / mutation_word_count,
            "mutation_target": mutation_target,
            "min_mutations": config.min_mutations,
            "max_mutations": config.max_mutations,
            "clean_text_sha256": sha256_text(markdown),
            "edited_text_sha256": sha256_text(edited),
            "response_text_sha256": sha256_text(answer),
            "source_spans": source_spans,
            "table_count": sum(block.kind == "table" for block in blocks),
            "table_ast_version": TABLE_AST_VERSION,
            "changes": changes,
        },
    }
    if config.response_fence == "none":
        sample["extra_info"]["response_fence"] = "none"
    validate_sample(sample, max_response_tokens=config.max_response_tokens)
    return sample


def checkpoint_paths(checkpoint_root: Path, stem: str) -> tuple[Path, Path]:
    prefix = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:2]
    root = checkpoint_root / prefix
    return root / f"{stem}.jsonl", root / f"{stem}.json"


def reusable_checkpoint(
    rows_path: Path,
    metadata_path: Path,
    *,
    archive_sha256: str,
    config: WorkerConfig,
) -> dict[str, Any] | None:
    if not config.resume or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("config_fingerprint") != config.fingerprint:
        return None
    if metadata.get("archive_sha256") != archive_sha256:
        return None
    if metadata.get("status") == "running":
        return None
    if metadata.get("status") not in {"success", "rejected"}:
        if not config.retry_failed:
            return metadata
        return None
    samples = int(metadata.get("samples", -1))
    if samples < 0 or (samples > 0 and not rows_path.is_file()):
        return None
    if samples == 0 and rows_path.exists() and rows_path.stat().st_size:
        return None
    return metadata


class SampleCheckpoint:
    """One paper, one writer: each row is a durable, directly usable SFT record."""

    def __init__(self, path: Path, metadata_path: Path, state: dict[str, Any], *, resume: bool) -> None:
        self.path = path
        self.ids: set[str] = set()
        self.clean_hashes: set[str] = set()
        self.edited_hashes: set[str] = set()
        previous: dict[str, Any] = {}
        if resume and metadata_path.is_file():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        matching = all(previous.get(key) == state[key] for key in ("config_fingerprint", "archive_sha256"))
        if not (resume and matching and path.is_file()):
            atomic_write_jsonl(path, [])
        # A terminated write may leave only the last record incomplete. Keep
        # all earlier completed samples and regenerate the unfinished one.
        with path.open("r+b") as stream:
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    stream.seek(-len(line), os.SEEK_CUR)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                extra = json.loads(line)["extra_info"]
                self.ids.add(extra["sample_id"])
                self.clean_hashes.add(extra["clean_text_sha256"])
                self.edited_hashes.add(extra["edited_text_sha256"])
        atomic_write_json(metadata_path, {**state, "status": "running", "samples": len(self.ids)})

    def write(self, sample: dict[str, Any]) -> bool:
        extra = sample["extra_info"]
        if extra["sample_id"] in self.ids:
            return False
        encoded = (json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self.path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self.ids.add(extra["sample_id"])
        self.clean_hashes.add(extra["clean_text_sha256"])
        self.edited_hashes.add(extra["edited_text_sha256"])
        return True


def process_paper(task: dict[str, Any]) -> PaperResult:
    row = dict(task["row"])
    stem = paper_stem(row)
    archive = Path(task["archive"])
    rows_path = Path(task["rows_path"])
    metadata_path = Path(task["metadata_path"])
    config = WorkerConfig(
        fingerprint=task["config"]["fingerprint"],
        seed=int(task["config"]["seed"]),
        tokenizer=str(task["config"]["tokenizer"]),
        tokenizer_local_only=bool(task["config"]["tokenizer_local_only"]),
        trust_remote_code=bool(task["config"]["trust_remote_code"]),
        length_buckets=tuple(LengthBucket(**bucket) for bucket in task["config"]["length_buckets"]),
        min_response_tokens=int(task["config"]["min_response_tokens"]),
        max_response_tokens=int(task["config"]["max_response_tokens"]),
        mutation_word_ratio=float(task["config"]["mutation_word_ratio"]),
        min_mutations=int(task["config"]["min_mutations"]),
        max_mutations=int(task["config"]["max_mutations"]),
        max_samples_per_paper=int(task["config"]["max_samples_per_paper"]),
        temp_root=task["config"].get("temp_root"),
        resume=bool(task["config"]["resume"]),
        retry_failed=bool(task["config"]["retry_failed"]),
        response_fence=task["config"].get("response_fence", "markdown"),
    )
    started = time.monotonic()
    archive_sha256: str | None = None
    checkpoint: SampleCheckpoint | None = None
    try:
        archive_sha256 = sha256_file(archive)
        expected = row.get("sha256")
        if not expected and isinstance(row.get("download"), dict):
            expected = row["download"].get("sha256")
        if expected and str(expected) != archive_sha256:
            raise ValueError("archive SHA256 differs from crawler manifest")
        reusable = reusable_checkpoint(
            rows_path,
            metadata_path,
            archive_sha256=archive_sha256,
            config=config,
        )
        if reusable is not None:
            return PaperResult(
                stem=stem,
                status=str(reusable.get("status", "failed")),
                checkpoint=str(rows_path) if int(reusable.get("samples", 0)) else None,
                samples=int(reusable.get("samples", 0)),
                candidate_chunks=int(reusable.get("candidate_chunks", 0)),
                tex_files=int(reusable.get("tex_files", 0)),
                blocks=int(reusable.get("blocks", 0)),
                extracted_bytes=int(reusable.get("extracted_bytes", 0)),
                rejection_reasons=dict(reusable.get("rejection_reasons", {})),
                error=reusable.get("error"),
                reused=True,
            )

        temporary_parent = Path(config.temp_root) if config.temp_root else None
        if temporary_parent:
            temporary_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"arxiv-text-{stem}-", dir=temporary_parent) as directory:
            source_root = Path(directory) / "source"
            _files, extracted_bytes, _archive_format = extract_source(archive, source_root)
            tex_paths = [
                path
                for path in sorted(source_root.rglob("*.tex"))
                if not re.fullmatch(
                    r"(?:references?|bibliography|thebibliography|refs?)",
                    path.stem,
                    re.IGNORECASE,
                )
            ]
            if not tex_paths:
                raise RejectedSource("no_tex_files")
            if len(tex_paths) > MAX_TEX_FILES_PER_PAPER:
                raise RejectedSource(f"too_many_tex_files:{len(tex_paths)}")
            blocks: list[TextBlock] = []
            rejection_reasons: Counter[str] = Counter()
            for path in tex_paths:
                extracted, reasons = extract_blocks(path, source_root)
                blocks.extend(extracted)
                rejection_reasons.update(reasons)
            if not blocks:
                raise RejectedSource("no_safe_markdown_blocks")
            counter = get_token_counter(config)
            windows = chunk_windows(blocks, counter=counter, config=config, stem=stem)
            checkpoint = SampleCheckpoint(
                rows_path, metadata_path,
                {"config_fingerprint": config.fingerprint, "archive_sha256": archive_sha256},
                resume=config.resume,
            )
            # Shuffle candidates before generating them, rather than retaining
            # completed SFT records in memory until the entire paper finishes.
            random.Random(stable_seed(config.seed, stem, "checkpoint-order")).shuffle(windows)
            for markdown, tokens, selected_blocks in windows:
                try:
                    sample = make_sample(
                        row=row,
                        stem=stem,
                        markdown=markdown,
                        clean_tokens=tokens,
                        blocks=selected_blocks,
                        counter=counter,
                        config=config,
                    )
                except RejectedSource as exc:
                    rejection_reasons[str(exc)] += 1
                    continue
                extra = sample["extra_info"]
                clean_hash = str(extra["clean_text_sha256"])
                edited_hash = str(extra["edited_text_sha256"])
                if extra["sample_id"] in checkpoint.ids:
                    continue
                if clean_hash in checkpoint.clean_hashes or edited_hash in checkpoint.edited_hashes:
                    rejection_reasons["duplicate_within_paper"] += 1
                    continue
                checkpoint.write(sample)
            sample_count = len(checkpoint.ids)
            status = "success" if sample_count else "rejected"
            metadata = {
                "status": status,
                "pipeline_version": PIPELINE_VERSION,
                "config_fingerprint": config.fingerprint,
                "stem": stem,
                "archive": str(archive),
                "archive_sha256": archive_sha256,
                "samples": sample_count,
                "candidate_chunks": len(windows),
                "tex_files": len(tex_paths),
                "blocks": len(blocks),
                "extracted_bytes": extracted_bytes,
                "rejection_reasons": dict(rejection_reasons),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
            }
            atomic_write_json(metadata_path, metadata)
            return PaperResult(
                stem=stem,
                status=status,
                checkpoint=str(rows_path) if sample_count else None,
                samples=sample_count,
                candidate_chunks=len(windows),
                tex_files=len(tex_paths),
                blocks=len(blocks),
                extracted_bytes=extracted_bytes,
                rejection_reasons=dict(rejection_reasons),
            )
    except RejectedSource as exc:
        error = str(exc)
        status = "rejected"
    except Exception as exc:  # noqa: BLE001 - one bad archive must not stop the corpus
        error = f"{type(exc).__name__}: {exc}"
        status = "failed"
    sample_count = len(checkpoint.ids) if checkpoint else 0
    metadata = {
        "status": status,
        "pipeline_version": PIPELINE_VERSION,
        "config_fingerprint": config.fingerprint,
        "stem": stem,
        "archive": str(archive),
        "archive_sha256": archive_sha256,
        "samples": sample_count,
        "candidate_chunks": 0,
        "tex_files": 0,
        "blocks": 0,
        "extracted_bytes": 0,
        "rejection_reasons": {error: 1} if status == "rejected" else {},
        "error": error,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_at": utc_now(),
    }
    atomic_write_json(metadata_path, metadata)
    if not rows_path.exists():
        atomic_write_jsonl(rows_path, [])
    return PaperResult(
        stem=stem,
        status=status,
        checkpoint=str(rows_path) if sample_count else None,
        samples=sample_count,
        candidate_chunks=0,
        tex_files=0,
        blocks=0,
        extracted_bytes=0,
        rejection_reasons=dict(metadata["rejection_reasons"]),
        error=error,
    )


class ShardWriter:
    def __init__(self, root: Path, shard_size: int) -> None:
        self.root = root
        self.shard_size = shard_size
        self.shard_index = 0
        self.rows_in_shard = 0
        self.total_rows = 0
        self.total_bytes = 0
        self._stream: Any = None
        self._temporary: Path | None = None
        self.parts: list[dict[str, Any]] = []

    def _open(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / f"part-{self.shard_index:05d}.jsonl"
        self._temporary = final.with_suffix(".jsonl.tmp")
        self._stream = self._temporary.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        if self._stream is None:
            self._open()
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._stream.write(encoded)
        self.rows_in_shard += 1
        self.total_rows += 1
        self.total_bytes += len(encoded.encode("utf-8"))
        if self.rows_in_shard >= self.shard_size:
            self._close_part()

    def _close_part(self) -> None:
        if self._stream is None or self._temporary is None:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        final = self.root / f"part-{self.shard_index:05d}.jsonl"
        os.replace(self._temporary, final)
        self.parts.append(
            {
                "path": str(final),
                "rows": self.rows_in_shard,
                "bytes": final.stat().st_size,
                "sha256": sha256_file(final),
            }
        )
        self.shard_index += 1
        self.rows_in_shard = 0
        self._stream = None
        self._temporary = None

    def close(self) -> None:
        self._close_part()
        for stale in self.root.glob("part-*.jsonl"):
            match = re.fullmatch(r"part-(\d{5})\.jsonl", stale.name)
            if match and int(match.group(1)) >= self.shard_index:
                stale.unlink()


def concatenate_jsonl_parts(
    output_dir: Path,
    *,
    split: str,
    parts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically concatenate final shards while reporting byte progress."""

    output = output_dir / f"{split}.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    total_bytes = sum(int(part["bytes"]) for part in parts)
    expected_rows = sum(int(part["rows"]) for part in parts)
    copied_bytes = 0
    copied_rows = 0
    started = time.monotonic()
    last_log = started
    print(
        f"[start] phase=concatenate split={split} parts={len(parts)} "
        f"rows={expected_rows} bytes={total_bytes} output={output}",
        flush=True,
    )
    with temporary.open("wb") as destination:
        for part_index, part in enumerate(parts, start=1):
            source_path = Path(str(part["path"]))
            with source_path.open("rb") as source:
                while True:
                    chunk = source.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
                    copied_bytes += len(chunk)
                    copied_rows += chunk.count(b"\n")
                    now = time.monotonic()
                    if now - last_log >= HEARTBEAT_SECONDS:
                        elapsed = max(now - started, 1e-9)
                        rate = copied_bytes / elapsed
                        eta = (total_bytes - copied_bytes) / rate if rate else math.inf
                        pct = 100 * copied_bytes / total_bytes if total_bytes else 100.0
                        print(
                            f"[progress] phase=concatenate split={split} "
                            f"parts={part_index}/{len(parts)} rows={copied_rows}/{expected_rows} "
                            f"bytes={copied_bytes}/{total_bytes} pct={pct:.2f}% "
                            f"throughput={rate / 1024**2:.2f}_MiB/s "
                            f"elapsed={elapsed_text(elapsed)} eta={elapsed_text(eta)} "
                            f"current={source_path.name}",
                            flush=True,
                        )
                        last_log = now
            print(
                f"[unit-done] phase=concatenate split={split} "
                f"part={part_index}/{len(parts)} current={source_path.name} "
                f"rows={copied_rows}/{expected_rows} bytes={copied_bytes}/{total_bytes}",
                flush=True,
            )
        destination.flush()
        os.fsync(destination.fileno())
    if copied_bytes != total_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"concatenated byte mismatch for {split}: {copied_bytes}!={total_bytes}"
        )
    if copied_rows != expected_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"concatenated row mismatch for {split}: {copied_rows}!={expected_rows}"
        )
    os.replace(temporary, output)
    result = {
        "path": str(output),
        "rows": copied_rows,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "parts": len(parts),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    print(
        f"[finish] phase=concatenate split={split} parts={len(parts)} "
        f"rows={copied_rows} bytes={result['bytes']} output={output}",
        flush=True,
    )
    return result


def stable_split(stem: str, *, split_seed: int, val_fraction: float) -> str:
    score = int.from_bytes(stable_digest(split_seed, stem)[:8], "big") / 2**64
    return "val" if score < val_fraction else "train"


def balanced_paper_quotas(
    results: Sequence[PaperResult], *, max_samples: int, seed: int
) -> dict[str, int]:
    capacities = {result.stem: result.samples for result in results}
    total = sum(capacities.values())
    if not max_samples or total <= max_samples:
        return capacities
    low = 0
    high = max(capacities.values(), default=0)
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(capacity, middle) for capacity in capacities.values()) <= max_samples:
            low = middle
        else:
            high = middle - 1
    quotas = {stem: min(capacity, low) for stem, capacity in capacities.items()}
    remainder = max_samples - sum(quotas.values())
    expandable = [stem for stem, capacity in capacities.items() if quotas[stem] < capacity]
    expandable.sort(key=lambda stem: stable_digest(seed, stem, "quota-remainder"))
    for stem in expandable[:remainder]:
        quotas[stem] += 1
    return quotas


def merge_checkpoints(
    results: Sequence[PaperResult],
    *,
    output_dir: Path,
    max_samples: int,
    shard_size: int,
    split_seed: int,
    val_fraction: float,
    seed: int,
    max_response_tokens: int,
) -> dict[str, Any]:
    started = time.monotonic()
    successful = [result for result in results if result.checkpoint and result.samples]
    successful.sort(key=lambda result: stable_digest(seed, result.stem, "merge-order"))
    paper_quotas = balanced_paper_quotas(successful, max_samples=max_samples, seed=seed)
    writers = {
        "train": ShardWriter(output_dir / "train", shard_size),
        "val": ShardWriter(output_dir / "val", shard_size),
    }
    seen_clean: set[str] = set()
    seen_edited: set[str] = set()
    available_unique = 0
    global_duplicates = 0
    written = 0
    processed_rows = 0
    last_log = started
    token_histogram: Counter[str] = Counter()
    mutation_histogram: Counter[int] = Counter()
    table_histogram: Counter[int] = Counter()
    paper_counts: Counter[str] = Counter()

    selected_clean: set[str] = set()
    selected_edited: set[str] = set()

    def write_selected(row: dict[str, Any], stem: str) -> None:
        nonlocal written
        extra = row["extra_info"]
        split = stable_split(stem, split_seed=split_seed, val_fraction=val_fraction)
        writers[split].write(row)
        written += 1
        paper_counts[stem] += 1
        selected_clean.add(str(extra["clean_text_sha256"]))
        selected_edited.add(str(extra["edited_text_sha256"]))
        tokens = int(extra["response_tokens"])
        if tokens < 2_000:
            token_histogram["1000-1999"] += 1
        elif tokens < 4_000:
            token_histogram["2000-3999"] += 1
        elif tokens < 6_000:
            token_histogram["4000-5999"] += 1
        else:
            token_histogram["6000-7800"] += 1
        mutation_histogram[len(extra["changes"])] += 1
        table_histogram[int(extra.get("table_count", 0))] += 1

    for paper_index, result in enumerate(successful, start=1):
        assert result.checkpoint is not None
        for row in read_jsonl(Path(result.checkpoint)):
            processed_rows += 1
            validate_sample(row, max_response_tokens=max_response_tokens)
            extra = row["extra_info"]
            clean_hash = str(extra["clean_text_sha256"])
            edited_hash = str(extra["edited_text_sha256"])
            if clean_hash in seen_clean or edited_hash in seen_edited:
                global_duplicates += 1
                continue
            seen_clean.add(clean_hash)
            seen_edited.add(edited_hash)
            available_unique += 1
            if paper_counts[result.stem] >= paper_quotas[result.stem]:
                continue
            write_selected(row, result.stem)
            now = time.monotonic()
            if now - last_log >= HEARTBEAT_SECONDS:
                elapsed = max(now - started, 1e-9)
                rate = processed_rows / elapsed
                print(
                    f"[progress] phase=merge papers={paper_index}/{len(successful)} "
                    f"rows_scanned={processed_rows} available_unique={available_unique} "
                    f"written={written} duplicates={global_duplicates} bytes={sum(writer.total_bytes for writer in writers.values())} "
                    f"throughput={rate:.2f}_rows/s elapsed={elapsed_text(elapsed)} current={result.stem}",
                    flush=True,
                )
                last_log = now
        print(
            f"[unit-done] phase=merge paper={result.stem} papers={paper_index}/{len(successful)} "
            f"rows_scanned={processed_rows} available_unique={available_unique} written={written}",
            flush=True,
        )

    # Cross-paper duplicates can leave a few balanced quotas unfilled.  A
    # deterministic second pass fills only that genuine shortfall from other
    # unique candidates; it never duplicates a selected clean or edited text.
    fill_rows_scanned = 0
    if max_samples and written < min(max_samples, available_unique):
        fill_seen_clean: set[str] = set()
        fill_seen_edited: set[str] = set()
        for paper_index, result in enumerate(successful, start=1):
            assert result.checkpoint is not None
            for row in read_jsonl(Path(result.checkpoint)):
                fill_rows_scanned += 1
                extra = row["extra_info"]
                clean_hash = str(extra["clean_text_sha256"])
                edited_hash = str(extra["edited_text_sha256"])
                if clean_hash in fill_seen_clean or edited_hash in fill_seen_edited:
                    continue
                fill_seen_clean.add(clean_hash)
                fill_seen_edited.add(edited_hash)
                if clean_hash in selected_clean or edited_hash in selected_edited:
                    continue
                write_selected(row, result.stem)
                if written >= max_samples:
                    break
            print(
                f"[unit-done] phase=merge_fill paper={result.stem} "
                f"papers={paper_index}/{len(successful)} rows_scanned={fill_rows_scanned} written={written}",
                flush=True,
            )
            if written >= max_samples:
                break
    for writer in writers.values():
        writer.close()
    return {
        "available_unique_samples": available_unique,
        "written_samples": written,
        "target_ceiling": max_samples,
        "target_reached": bool(max_samples and available_unique >= max_samples),
        "shortfall": max(0, max_samples - available_unique) if max_samples else 0,
        "global_duplicates_rejected": global_duplicates,
        "rows_scanned": processed_rows,
        "fill_rows_scanned": fill_rows_scanned,
        "balanced_quota_papers": len(paper_quotas),
        "unique_papers_written": len(paper_counts),
        "samples_per_paper_min": min(paper_counts.values()) if paper_counts else 0,
        "samples_per_paper_max": max(paper_counts.values()) if paper_counts else 0,
        "train": {"rows": writers["train"].total_rows, "bytes": writers["train"].total_bytes, "parts": writers["train"].parts},
        "val": {"rows": writers["val"].total_rows, "bytes": writers["val"].total_bytes, "parts": writers["val"].parts},
        "response_token_histogram": dict(token_histogram),
        "mutation_count_histogram": {str(key): value for key, value in sorted(mutation_histogram.items())},
        "table_samples": sum(count for tables, count in table_histogram.items() if tables),
        "tables_written": sum(tables * count for tables, count in table_histogram.items()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _read_download_checkpoint(
    task: tuple[Path, Path],
) -> tuple[dict[str, Any] | None, Path | None, str | None, int]:
    input_root, checkpoint = task
    try:
        payload = checkpoint.read_bytes()
    except FileNotFoundError:
        return None, None, "missing_download_checkpoint", 0
    except OSError:
        return None, None, "unreadable_download_checkpoint", 0
    try:
        row = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None, "invalid_download_checkpoint", len(payload)
    if not isinstance(row, dict):
        return None, None, "invalid_download_checkpoint", len(payload)
    row.setdefault("stem", checkpoint.parent.name)
    try:
        paper_stem(row)
    except ValueError:
        return None, None, "unsafe_or_missing_stem", len(payload)
    if row.get("status") not in {"passed", "success"}:
        return None, None, "crawler_status_not_passed", len(payload)
    try:
        archive = resolve_archive(input_root, row)
    except (OSError, ValueError):
        return None, None, "archive_missing_or_empty", len(payload)
    return row, archive, None, len(payload)


def _discover_download_checkpoints(
    input_root: Path, *, workers: int,
) -> tuple[list[tuple[dict[str, Any], Path]], Counter[str]]:
    """Read completed downloads without waiting for the final crawler manifest."""
    papers_root = input_root / "papers"
    if not papers_root.is_dir():
        raise FileNotFoundError(
            f"neither {input_root / 'results.jsonl'} nor {papers_root} exists"
        )
    started = last_log = time.monotonic()
    print(
        f"[start] phase=input_discovery mode=download_checkpoints "
        f"directory={papers_root} workers={workers}",
        flush=True,
    )
    checkpoints: list[Path] = []
    with os.scandir(papers_root) as entries:
        for entry in entries:
            if entry.is_dir():
                checkpoints.append(Path(entry.path) / "download.json")
            if time.monotonic() - last_log >= HEARTBEAT_SECONDS:
                last_log = time.monotonic()
                print(
                    f"[progress] phase=input_discovery scanned={len(checkpoints)} "
                    f"total=unknown current={entry.name} "
                    f"elapsed={elapsed_text(last_log - started)}",
                    flush=True,
                )
    checkpoints.sort()
    total = len(checkpoints)
    print(f"[start] phase=input_checkpoints total={total} workers={workers}", flush=True)
    tasks = [(input_root, path) for path in checkpoints]
    executor = (
        concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, total))
        if workers > 1 and total > 1 else None
    )
    results = (
        executor.map(_read_download_checkpoint, tasks, chunksize=16)
        if executor is not None else map(_read_download_checkpoint, tasks)
    )
    ready: list[tuple[dict[str, Any], Path]] = []
    rejected: Counter[str] = Counter()
    metadata_bytes = 0
    completed = 0
    try:
        for completed, (row, archive, reason, size) in enumerate(results, start=1):
            metadata_bytes += size
            if reason is not None:
                rejected[reason] += 1
            elif row is not None and archive is not None:
                ready.append((row, archive))
            now = time.monotonic()
            if completed % 1000 == 0 or completed == total or now - last_log >= HEARTBEAT_SECONDS:
                last_log = now
                elapsed = max(now - started, 1e-9)
                rate = completed / elapsed
                errors = sum(rejected[key] for key in (
                    "invalid_download_checkpoint", "unreadable_download_checkpoint",
                ))
                print(
                    f"[unit-done] phase=input_checkpoints completed={completed}/{total} "
                    f"pct={100 * completed / max(total, 1):.2f}% "
                    f"current={checkpoints[completed - 1].parent.name} "
                    f"ready={len(ready)} rejected={sum(rejected.values())} errors={errors} "
                    f"metadata_bytes={metadata_bytes} throughput={rate:.3f}_papers/s "
                    f"elapsed={elapsed_text(elapsed)} eta={elapsed_text((total - completed) / rate)}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    print(
        f"[finish] phase=input_discovery completed={completed}/{total} "
        f"ready={len(ready)} rejected={sum(rejected.values())} "
        f"metadata_bytes={metadata_bytes} elapsed={elapsed_text(time.monotonic() - started)}",
        flush=True,
    )
    return ready, rejected


def select_input_rows(
    input_root: Path,
    *,
    max_papers: int,
    paper_ids: set[str],
    allow_all_licenses: bool,
    workers: int = 1,
) -> tuple[list[tuple[dict[str, Any], Path]], Counter[str]]:
    results_path = input_root / "results.jsonl"
    selected: list[tuple[dict[str, Any], Path]] = []
    rejected: Counter[str] = Counter()
    if results_path.is_file():
        candidates = ((row, None) for row in read_jsonl(results_path))
    else:
        downloaded, rejected = _discover_download_checkpoints(input_root, workers=workers)
        candidates = iter(downloaded)
    seen: set[str] = set()
    for row, discovered_archive in candidates:
        try:
            stem = paper_stem(row)
        except ValueError:
            rejected["unsafe_or_missing_stem"] += 1
            continue
        if stem in seen:
            rejected["duplicate_stem"] += 1
            continue
        seen.add(stem)
        if row.get("status") not in {"passed", "success"}:
            rejected["crawler_status_not_passed"] += 1
            continue
        if not allow_all_licenses and row.get("license_name") not in ALLOWED_LICENSES:
            rejected["license_not_allowed"] += 1
            continue
        if paper_ids and stem not in paper_ids and str(row.get("arxiv_id", "")) not in paper_ids:
            rejected["not_requested"] += 1
            continue
        try:
            archive = discovered_archive or resolve_archive(input_root, row)
        except (OSError, ValueError):
            rejected["archive_missing_or_empty"] += 1
            continue
        selected.append((row, archive))
        if max_papers and len(selected) >= max_papers:
            break
    if not selected:
        detail = f"no eligible downloaded source archives were selected; reasons={dict(rejected)}"
        if rejected["license_not_allowed"]:
            detail += "; use --allow-all-licenses only if you intend to include those licenses"
        raise ValueError(detail)
    return selected, rejected


def config_fingerprint(args: argparse.Namespace, buckets: Sequence[LengthBucket]) -> str:
    response_fence = getattr(args, "response_fence", "markdown")
    value = {
        "pipeline_version": PIPELINE_VERSION,
        "table_ast_version": TABLE_AST_VERSION,
        "prompt_version": prompt_version_for(response_fence),
        "mutation_policy_version": MUTATION_POLICY_VERSION,
        "mutation_pair_policy_fingerprint": POLICY_FINGERPRINT,
        "heading_policy_version": HEADING_POLICY_VERSION,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "tokenizer_local_only": not args.allow_tokenizer_download,
        "trust_remote_code": args.trust_remote_code,
        "length_buckets": [asdict(bucket) for bucket in buckets],
        "min_response_tokens": args.min_response_tokens,
        "max_response_tokens": args.max_response_tokens,
        "mutation_word_ratio": args.mutation_word_ratio,
        "min_mutations": args.min_mutations,
        "max_mutations": args.max_mutations,
        "max_samples_per_paper": args.max_samples_per_paper,
    }
    # Keep existing default-mode checkpoints usable; isolate only the new mode.
    if response_fence == "none":
        value["response_fence"] = "none"
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))[:20]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    response_fence = getattr(args, "response_fence", "markdown")
    response_affixes(response_fence)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_papers < 0 or args.max_samples < 0 or args.max_samples_per_paper < 0:
        raise ValueError("sample and paper ceilings cannot be negative")
    if args.min_response_tokens < 1 or args.max_response_tokens < args.min_response_tokens:
        raise ValueError("invalid response token limits")
    if not 0 < args.mutation_word_ratio <= 1:
        raise ValueError("--mutation-word-ratio must be in (0, 1]")
    if args.min_mutations < 1:
        raise ValueError("--min-mutations must be positive")
    if args.max_mutations < 0 or (
        args.max_mutations and args.max_mutations < args.min_mutations
    ):
        raise ValueError("--max-mutations must be 0 or at least --min-mutations")
    if not 0 <= args.val_fraction < 1:
        raise ValueError("--val-fraction must be in [0, 1)")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    buckets = (
        LengthBucket(1_000, 1_999, 0.10),
        LengthBucket(2_000, 3_999, 0.25),
        LengthBucket(4_000, 5_999, 0.35),
        LengthBucket(6_000, args.max_response_tokens, 0.30),
    )
    if args.min_response_tokens != 1_000:
        buckets = tuple(
            bucket for bucket in buckets if bucket.maximum >= args.min_response_tokens
        )
        buckets = tuple(
            LengthBucket(max(bucket.minimum, args.min_response_tokens), bucket.maximum, bucket.weight)
            for bucket in buckets
            if max(bucket.minimum, args.min_response_tokens) <= bucket.maximum
        )
    if args.max_response_tokens < 6_000:
        clipped: list[LengthBucket] = []
        for bucket in buckets:
            maximum = min(bucket.maximum, args.max_response_tokens)
            if bucket.minimum <= maximum:
                clipped.append(LengthBucket(bucket.minimum, maximum, bucket.weight))
        buckets = tuple(clipped)
    if not buckets:
        raise ValueError("token limits do not leave any configured length bucket")

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, census_rejections = select_input_rows(
        input_root,
        max_papers=args.max_papers,
        paper_ids=set(args.paper_ids),
        allow_all_licenses=args.allow_all_licenses,
        workers=args.workers,
    )
    rng = random.Random(args.seed)
    rng.shuffle(selected)
    fingerprint = config_fingerprint(args, buckets)
    checkpoint_root = output_dir / "checkpoints" / fingerprint
    worker_config = {
        "fingerprint": fingerprint,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "tokenizer_local_only": not args.allow_tokenizer_download,
        "trust_remote_code": args.trust_remote_code,
        "length_buckets": [asdict(bucket) for bucket in buckets],
        "min_response_tokens": args.min_response_tokens,
        "max_response_tokens": args.max_response_tokens,
        "mutation_word_ratio": args.mutation_word_ratio,
        "min_mutations": args.min_mutations,
        "max_mutations": args.max_mutations,
        "max_samples_per_paper": args.max_samples_per_paper,
        "temp_root": str(args.temp_root.resolve()) if args.temp_root else None,
        "resume": args.resume,
        "retry_failed": args.retry_failed,
        "response_fence": response_fence,
    }
    preflight_config = WorkerConfig(
        fingerprint=fingerprint,
        seed=args.seed,
        tokenizer=args.tokenizer,
        tokenizer_local_only=not args.allow_tokenizer_download,
        trust_remote_code=args.trust_remote_code,
        length_buckets=tuple(buckets),
        min_response_tokens=args.min_response_tokens,
        max_response_tokens=args.max_response_tokens,
        mutation_word_ratio=args.mutation_word_ratio,
        min_mutations=args.min_mutations,
        max_mutations=args.max_mutations,
        max_samples_per_paper=args.max_samples_per_paper,
        temp_root=str(args.temp_root.resolve()) if args.temp_root else None,
        resume=args.resume,
        retry_failed=args.retry_failed,
        response_fence=response_fence,
    )
    preflight_tokens = get_token_counter(preflight_config).count("tokenizer preflight")
    if preflight_tokens <= 0:
        raise RuntimeError("tokenizer preflight returned no tokens")
    _TOKEN_COUNTER_CACHE.clear()
    print(
        f"[check] kind=tokenizer name={args.tokenizer} status=passed "
        f"probe_tokens={preflight_tokens} local_only={not args.allow_tokenizer_download}",
        flush=True,
    )
    tasks: list[dict[str, Any]] = []
    for row, archive in selected:
        stem = paper_stem(row)
        rows_path, metadata_path = checkpoint_paths(checkpoint_root, stem)
        tasks.append(
            {
                "row": row,
                "archive": str(archive),
                "rows_path": str(rows_path),
                "metadata_path": str(metadata_path),
                "config": worker_config,
            }
        )

    print(
        f"[start] phase=paper_processing input_root={input_root} output_dir={output_dir} "
        f"papers={len(tasks)} workers={args.workers} tokenizer={args.tokenizer} "
        f"response_tokens={args.min_response_tokens}-{args.max_response_tokens} "
        f"mutation_word_ratio={args.mutation_word_ratio:g} "
        f"heading_policy={HEADING_POLICY_VERSION} response_fence={response_fence} "
        f"mutation_limits={args.min_mutations}-{args.max_mutations or 'unlimited'} "
        f"target_ceiling={args.max_samples or 'unlimited'} resume={args.resume} "
        f"config_fingerprint={fingerprint}",
        flush=True,
    )
    atomic_write_json(
        output_dir / "build_state.json",
        {
            "status": "building",
            "started_at": utc_now(),
            "config_fingerprint": fingerprint,
            "papers_selected": len(tasks),
            "census_rejections": dict(census_rejections),
        },
    )

    completed = 0
    success = 0
    rejected = 0
    failed = 0
    sample_candidates = 0
    extracted_bytes = 0
    all_rejections: Counter[str] = Counter()
    results: list[PaperResult] = []
    processing_started = time.monotonic()

    def record(result: PaperResult) -> None:
        nonlocal completed, success, rejected, failed, sample_candidates, extracted_bytes
        completed += 1
        success += result.status == "success"
        rejected += result.status == "rejected"
        failed += result.status == "failed"
        sample_candidates += result.samples
        extracted_bytes += result.extracted_bytes
        all_rejections.update(result.rejection_reasons)
        results.append(result)
        elapsed = max(time.monotonic() - processing_started, 1e-9)
        rate = completed / elapsed
        eta = (len(tasks) - completed) / rate if rate else math.inf
        print(
            f"[unit-done] phase=paper_processing completed={completed}/{len(tasks)} "
            f"pct={100 * completed / len(tasks):.2f}% current={result.stem} status={result.status} "
            f"samples={result.samples} cumulative_samples={sample_candidates} "
            f"accepted_papers={success} rejected_papers={rejected} errors={failed} "
            f"bytes={extracted_bytes} throughput={rate:.3f}_papers/s "
            f"elapsed={elapsed_text(elapsed)} eta={elapsed_text(eta)} reused={result.reused}",
            flush=True,
        )

    if args.workers == 1:
        for task in tasks:
            record(process_paper(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {executor.submit(process_paper, task): paper_stem(task["row"]) for task in tasks}
            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    timeout=HEARTBEAT_SECONDS,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    elapsed = max(time.monotonic() - processing_started, 1e-9)
                    rate = completed / elapsed
                    eta = (len(tasks) - completed) / rate if rate else math.inf
                    print(
                        f"[progress] phase=paper_processing completed={completed}/{len(tasks)} "
                        f"pct={100 * completed / len(tasks):.2f}% active={len(pending)} "
                        f"cumulative_samples={sample_candidates} accepted_papers={success} "
                        f"rejected_papers={rejected} errors={failed} bytes={extracted_bytes} "
                        f"throughput={rate:.3f}_papers/s elapsed={elapsed_text(elapsed)} eta={elapsed_text(eta)}",
                        flush=True,
                    )
                    continue
                for future in done:
                    stem = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 - executor-level failure
                        result = PaperResult(
                            stem=stem,
                            status="failed",
                            checkpoint=None,
                            samples=0,
                            candidate_chunks=0,
                            tex_files=0,
                            blocks=0,
                            extracted_bytes=0,
                            rejection_reasons={},
                            error=f"executor:{type(exc).__name__}: {exc}",
                        )
                    record(result)

    result_rows = [asdict(result) for result in sorted(results, key=lambda result: result.stem)]
    atomic_write_jsonl(output_dir / "paper_results.jsonl", result_rows)
    print(
        f"[checkpoint] phase=paper_processing papers={completed}/{len(tasks)} success={success} "
        f"rejected={rejected} errors={failed} candidate_samples={sample_candidates} "
        f"elapsed={elapsed_text(time.monotonic() - processing_started)}",
        flush=True,
    )

    print(
        f"[start] phase=merge checkpoints={success} candidate_samples={sample_candidates} "
        f"target_ceiling={args.max_samples or 'unlimited'}",
        flush=True,
    )
    merge = merge_checkpoints(
        results,
        output_dir=output_dir,
        max_samples=args.max_samples,
        shard_size=args.shard_size,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_response_tokens=args.max_response_tokens,
    )
    if args.write_merged_jsonl:
        merge["merged_jsonl"] = {
            split: concatenate_jsonl_parts(
                output_dir,
                split=split,
                parts=merge[split]["parts"],
            )
            for split in ("train", "val")
        }
    summary = {
        "status": "passed" if merge["written_samples"] else "failed",
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": prompt_version_for(response_fence),
        "response_fence": response_fence,
        "mutation_policy_version": MUTATION_POLICY_VERSION,
        "mutation_pair_policy_fingerprint": POLICY_FINGERPRINT,
        "heading_policy_version": HEADING_POLICY_VERSION,
        "created_at": utc_now(),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "config_fingerprint": fingerprint,
        "configuration": {
            **worker_config,
            "workers": args.workers,
            "max_samples": args.max_samples,
            "shard_size": args.shard_size,
            "split_seed": args.split_seed,
            "val_fraction": args.val_fraction,
            "allow_all_licenses": args.allow_all_licenses,
            "write_merged_jsonl": args.write_merged_jsonl,
        },
        "papers": {
            "selected": len(tasks),
            "success": success,
            "rejected": rejected,
            "failed": failed,
            "reused": sum(result.reused for result in results),
        },
        "census_rejections": dict(census_rejections),
        "source_rejection_reasons": dict(all_rejections.most_common()),
        "merge": merge,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(output_dir / "manifest.json", summary)
    atomic_write_json(output_dir / "stats.json", {"papers": summary["papers"], "merge": merge, "source_rejection_reasons": summary["source_rejection_reasons"]})
    atomic_write_json(
        output_dir / "build_state.json",
        {
            "status": summary["status"],
            "completed_at": utc_now(),
            "config_fingerprint": fingerprint,
            "manifest": str(output_dir / "manifest.json"),
            "written_samples": merge["written_samples"],
            "available_unique_samples": merge["available_unique_samples"],
        },
    )
    print(
        f"[finish] status={summary['status']} papers={len(tasks)} paper_success={success} "
        f"paper_rejected={rejected} paper_errors={failed} "
        f"available_unique={merge['available_unique_samples']} written={merge['written_samples']} "
        f"target_ceiling={args.max_samples or 'unlimited'} shortfall={merge['shortfall']} "
        f"train={merge['train']['rows']} val={merge['val']['rows']} "
        f"table_samples={merge['table_samples']} tables={merge['tables_written']} "
        f"bytes={merge['train']['bytes'] + merge['val']['bytes']} "
        f"elapsed={elapsed_text(time.monotonic() - started)} manifest={output_dir / 'manifest.json'}",
        flush=True,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, required=True,
        help="Crawler root containing results.jsonl or papers/*/download.json and source_archive.bin",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="local Hugging Face tokenizer/model path; use 'simple' only for tests",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--max-papers", type=int, default=0, help="0 processes every eligible paper")
    parser.add_argument("--paper-ids", nargs="*", default=[])
    parser.add_argument("--max-samples", type=int, default=1_000_000, help="ceiling only; 0 writes every eligible unique sample")
    parser.add_argument("--max-samples-per-paper", type=int, default=0, help="0 keeps every unique window produced by the fixed policy")
    parser.add_argument("--min-response-tokens", type=int, default=1_000)
    parser.add_argument("--max-response-tokens", type=int, default=7_800)
    parser.add_argument(
        "--response-fence", choices=("markdown", "none"), default="markdown",
        help="Outer answer wrapper: markdown keeps V5 behavior; none uses the approved no-fence prompt",
    )
    parser.add_argument(
        "--mutation-word-ratio",
        type=float,
        default=DEFAULT_MUTATION_WORD_RATIO,
        help=(
            "fraction of visible non-protected word occurrences to mutate; "
            "the default 0.10 targets about ten percent per sample"
        ),
    )
    parser.add_argument("--min-mutations", type=int, default=DEFAULT_MIN_MUTATIONS)
    parser.add_argument(
        "--max-mutations",
        type=int,
        default=DEFAULT_MAX_MUTATIONS,
        help="optional per-sample ceiling; 0 keeps the ratio-derived count uncapped",
    )
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument(
        "--write-merged-jsonl",
        action="store_true",
        help="also atomically concatenate final shards into train.jsonl and val.jsonl",
    )
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--allow-all-licenses", action="store_true")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--retry-failed", action="store_true")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_pipeline(args)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
