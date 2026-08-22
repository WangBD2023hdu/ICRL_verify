#!/usr/bin/env python3
"""Build page-level Markdown GT from recompiled arXiv LaTeX sources.

The builder deliberately uses a hybrid representation:

* visible prose and reading order come from the compiled PDF text layer;
* display and high-confidence inline mathematics come from the LaTeX source;
* bold, italic, and code spans are restored from source onto PDF prose;
* every table is serialized as structural HTML inside Markdown;
* SyncTeX maps source blocks to the page and approximate PDF bounding box.

The input is the output produced by ``build_arxiv_latex_recompile_pilot.py``.
Only papers with a successful compile and a passed source safety scan are used.
The program never executes source-provided scripts or latexmk configuration.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import gzip
import hashlib
import html
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:
    import pdfplumber
except ImportError as exc:  # pragma: no cover - exercised by CLI preflight
    raise SystemExit(
        "pdfplumber is required; run with the Codex bundled Python or the agents environment"
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from arxiv_inline_markup import (  # noqa: E402
    InlineParseError,
    InlinePlan,
    InlineRenderResult,
    apply_inline_plan,
    build_inline_regex,
    build_text_anchor_regex,
    extract_footnote_source,
    focus_inline_plan,
    iter_inline_nodes,
    parse_inline_plan,
    render_footnote_body,
    render_inline_match,
    render_inline_source,
    summarize_inline_plan,
)


SCHEMA_VERSION = 2
SOURCE_PARAGRAPH_CONTRACT_VERSION = 6
AUTHOR_SUPERSCRIPT_CONTRACT_VERSION = 5
FOOTNOTE_REPRESENTATION = "html_sup"
HEARTBEAT_SECONDS = 15.0
DEFAULT_INPUT = Path("outputs/arxiv_latex_recompile_pilot_5")
DEFAULT_OUTPUT = Path("outputs/arxiv_page_markdown_gt_pilot")
TEX_BIN = Path("/Library/TeX/texbin")
PDFTOTEXT = Path("/opt/homebrew/bin/pdftotext")
PDFTOPPM = Path("/opt/homebrew/bin/pdftoppm")

DISPLAY_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "alignat",
    "alignat*",
    "flalign",
    "flalign*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "eqnarray",
    "eqnarray*",
}
TABLE_CONTAINER_ENVS = {"table", "table*"}
STANDALONE_TABULAR_ENVS = {"tabular", "tabular*", "tabularx", "tabulary", "longtable"}
TABLE_ENVS = TABLE_CONTAINER_ENVS | STANDALONE_TABULAR_ENVS
TABULAR_ENVS = STANDALONE_TABULAR_ENVS | {"array"}
SECTION_LEVELS = {
    "part": 1,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}
HEADING_NUMBER_PREFIX_PATTERN = re.compile(
    r"^\s*(?P<prefix>(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.|[A-Z]\.))(?=\s+)"
)
INLINE_TARGET_PATTERN = re.compile(
    r"(?<!\\)\$(?!\$)|\\\(|\\(?:textbf|emph|textit|texttt|verb|footnote)\b"
)
INLINE_IGNORED_ENVS = (
    DISPLAY_ENVS
    | TABLE_ENVS
    | TABULAR_ENVS
    | {
        "figure",
        "figure*",
        "algorithm",
        "algorithmic",
        "itemize",
        "enumerate",
        "description",
        "thebibliography",
        "verbatim",
        "Verbatim",
        "lstlisting",
        "minted",
        "tikzpicture",
        "picture",
    }
)
INLINE_NONPROSE_COMMAND = re.compile(
    r"^\s*\\(?:documentclass|usepackage|RequirePackage|newcommand|renewcommand|providecommand|"
    r"DeclareMathOperator|def|gdef|edef|xdef|let|input|include|bibliography|bibliographystyle|"
    r"title|author|date|name|address|affiliation|email|thanks|keywords|maketitle|"
    r"part|chapter|section|subsection|subsubsection|paragraph|subparagraph|caption|label)\*?\b"
)
RUNIN_HEADING_PREFIX = re.compile(
    r"^\s*\\(?:paragraph|subparagraph)\*?\s*\{"
)

SOURCE_PARAGRAPH_LIST_ENVS = {"itemize", "enumerate", "description"}
SOURCE_PARAGRAPH_IGNORED_ENVS = (
    DISPLAY_ENVS
    | TABLE_ENVS
    | {
        "figure",
        "figure*",
        "algorithm",
        "algorithmic",
        "thebibliography",
        "verbatim",
        "Verbatim",
        "lstlisting",
        "minted",
        "tikzpicture",
        "picture",
    }
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def elapsed_string(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:d}m{seconds:02d}s"


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_tex_comment(line: str) -> str:
    for index, character in enumerate(line):
        if character != "%":
            continue
        slash_count = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        if slash_count % 2 == 0:
            return line[:index]
    return line


def normalize_space(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip()


ALPHABETIC_HYPHENATED_TERM_PATTERN = re.compile(
    r"(?<![A-Za-z\\])(?P<term>[A-Za-z]{2,}(?:-\s*[A-Za-z]{2,})+)(?![A-Za-z])"
)
INTRA_PAGE_HYPHENATION_RESIDUE_PATTERN = re.compile(
    r"(?<![A-Za-z])(?P<left>[A-Za-z]{2,})-\s+(?P<right>[A-Za-z]{2,})(?![A-Za-z])",
    flags=re.IGNORECASE,
)


def extract_source_hyphenated_terms(source_root: Path) -> list[str]:
    """Extract deterministic lexical ``alpha-alpha`` terms from TeX source.

    TeX command names are removed before matching, while their textual
    arguments remain eligible.  Each alphabetic component must contain at
    least two letters; this deliberately excludes mathematical ``x-y`` and
    lexical constructions such as ``k-bonacci`` from the conservative gate.
    """

    terms: set[str] = set()
    for source_file in sorted(source_root.rglob("*.tex")):
        try:
            raw_lines = source_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        text = "\n".join(strip_tex_comment(line) for line in raw_lines)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\$\$[\s\S]*?\$\$", " ", text)
        # Keep the two alternatives disjoint.  ``[^$]`` also matched a
        # backslash, so an unmatched dollar followed by many TeX commands
        # caused exponential backtracking between ``\\.`` and ``[^$]`` on
        # real arXiv sources.  Excluding both delimiters makes this scan
        # linear while preserving the intended escaped-character handling.
        text = re.sub(r"(?<!\\)\$(?:\\.|[^\\$])*(?<!\\)\$", " ", text)
        text = re.sub(r"\\\([\s\S]*?\\\)", " ", text)
        text = re.sub(r"\\\[[\s\S]*?\\\]", " ", text)
        for environment in sorted(DISPLAY_ENVS, key=len, reverse=True):
            text = re.sub(
                r"\\begin\s*\{" + re.escape(environment) + r"\}"
                r"[\s\S]*?\\end\s*\{" + re.escape(environment) + r"\}",
                " ",
                text,
            )
        text = re.sub(r"\\[A-Za-z@]+", " ", text)
        for match in ALPHABETIC_HYPHENATED_TERM_PATTERN.finditer(text):
            term = re.sub(r"-\s*", "-", match.group("term")).casefold()
            terms.add(term)
    return sorted(terms)


def punctuation_prose_projection(markdown: str) -> str:
    """Mask math/code while retaining visible prose and HTML cell text."""

    value = unicodedata.normalize("NFKC", markdown)
    value = re.sub(r"```[\s\S]*?```", " ", value)
    value = re.sub(r"`[^`\n]*`", " ", value)
    value = re.sub(r"\$\$[\s\S]*?\$\$", " ", value)
    value = re.sub(r"(?<!\\)\$(?:\\.|[^\\$])*(?<!\\)\$", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    return value.casefold()


def strict_punctuation_issues(
    markdown: str, source_hyphenated_terms: Sequence[str]
) -> list[str]:
    """Return deterministic strict failures for prose hyphen corruption."""

    prose = punctuation_prose_projection(markdown)
    issues: set[str] = set()
    for match in INTRA_PAGE_HYPHENATION_RESIDUE_PATTERN.finditer(prose):
        term = f"{match.group('left')}-{match.group('right')}".casefold()
        issues.add(f"intra_page_hyphenation_residue:{term}")
    for raw_term in sorted(set(source_hyphenated_terms)):
        term = unicodedata.normalize("NFKC", str(raw_term)).casefold()
        parts = term.split("-")
        if len(parts) < 2 or any(
            len(part) < 2 or not part.isascii() or not part.isalpha()
            for part in parts
        ):
            continue
        collapsed = "".join(parts)
        collapsed_count = len(
            re.findall(
                rf"(?<![A-Za-z]){re.escape(collapsed)}(?![A-Za-z])", prose
            )
        )
        hyphen_pattern = r"\s*-\s*".join(re.escape(part) for part in parts)
        hyphenated_count = len(
            re.findall(
                rf"(?<![A-Za-z]){hyphen_pattern}(?![A-Za-z])", prose
            )
        )
        if collapsed_count > hyphenated_count:
            issues.add(f"source_hyphen_collapsed:{term}")
    return sorted(issues)


def aggregate_strict_punctuation_issues(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    exact: collections.Counter[str] = collections.Counter()
    issue_types: collections.Counter[str] = collections.Counter()
    pages_with_issues = 0
    for row in rows:
        issues = list(row.get("strict_punctuation_issues", []))
        if issues:
            pages_with_issues += 1
        exact.update(str(issue) for issue in issues)
        issue_types.update(str(issue).split(":", 1)[0] for issue in issues)
    return {
        "pages_with_issues": pages_with_issues,
        "total_issues": sum(exact.values()),
        "by_type": dict(sorted(issue_types.items())),
        "by_issue": dict(sorted(exact.items())),
    }


def extract_balanced(text: str, open_index: int) -> tuple[str, int] | None:
    if open_index >= len(text) or text[open_index] != "{":
        return None
    depth = 0
    escaped = False
    for index in range(open_index, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index], index + 1
    return None


def extract_command_argument(text: str, command: str, start: int = 0) -> tuple[str, int, int] | None:
    pattern = re.compile(r"\\" + re.escape(command) + r"\*?(?:\s*\[[^\]]*\])?\s*")
    match = pattern.search(text, start)
    if not match:
        return None
    brace_index = match.end()
    if brace_index >= len(text) or text[brace_index] != "{":
        return None
    result = extract_balanced(text, brace_index)
    if result is None:
        return None
    value, end = result
    return value, match.start(), end


def runin_heading_trailing_prose(value: str) -> str | None:
    """Return prose following a leading paragraph/subparagraph heading.

    LaTeX ``\\paragraph`` headings are commonly run into the first prose line.
    Treating the whole source line as a structural heading drops both its
    paragraph identity and every inline formula in the trailing prose.
    """

    match = RUNIN_HEADING_PREFIX.match(value)
    if match is None:
        return None
    argument = extract_balanced(value, match.end() - 1)
    if argument is None:
        return None
    return value[argument[1] :].strip()


def replace_simple_braced_commands(value: str) -> str:
    commands = [
        "textbf",
        "mathbf",
        "bfseries",
        "emph",
        "textit",
        "textrm",
        "textsf",
        "texttt",
        "mathrm",
        "operatorname",
        "mbox",
        "hbox",
    ]
    for _ in range(8):
        previous = value
        for command in commands:
            value = re.sub(r"\\" + command + r"\s*\{([^{}]*)\}", r"\1", value)
        if value == previous:
            break
    return value


def latex_to_plain(value: str) -> str:
    value = re.sub(r"\\label\s*\{[^{}]*\}", "", value)
    value = re.sub(r"\\(?:ref|eqref|pageref)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)*\{([^{}]*)\}", r"[\1]", value)
    value = replace_simple_braced_commands(value)
    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
        "~": " ",
        "---": "-",
        "--": "-",
        "``": '"',
        "''": '"',
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    value = re.sub(r"\\(?:small|footnotesize|scriptsize|normalsize|large|Large|LARGE|centering)\b", "", value)
    value = re.sub(r"\\[a-zA-Z@]+\*?(?:\s*\[[^\]]*\])?", "", value)
    value = value.replace("{", "").replace("}", "")
    return normalize_space(value)


AUTHOR_SUPERSCRIPT_UNSUPPORTED_PATTERN = re.compile(
    r"\\footnotetext\b"
)
AUTHOR_SUPERSCRIPT_SAFE_MARKER_PATTERN = re.compile(
    r"[0-9A-Za-z,.*+\-\u2020\u2021]+(?: [0-9A-Za-z,.*+\-\u2020\u2021]+)*"
)


@dataclasses.dataclass
class AuthorSuperscriptPlan:
    """Source-authoritative superscript markers from one ``\\author`` block."""

    source_file: Path
    start_line: int
    end_line: int
    raw_latex: str
    pieces: list[tuple[str, str]]

    @property
    def plain_text(self) -> str:
        return normalize_space("".join(value for _, value in self.pieces))

    @property
    def markdown_text(self) -> str:
        return normalize_space(
            "".join(
                f"<sup>{html.escape(value, quote=False)}</sup>"
                if kind == "sup"
                else value
                for kind, value in self.pieces
            )
        )

    @property
    def markers(self) -> list[str]:
        return [value for kind, value in self.pieces if kind == "sup"]

    def as_json(self, source_root: Path) -> dict[str, Any]:
        try:
            source_file = self.source_file.relative_to(source_root).as_posix()
        except ValueError:
            source_file = str(self.source_file)
        return {
            "source_file": source_file,
            "source_lines": [self.start_line, self.end_line],
            "raw_latex": self.raw_latex,
            "plain_text": self.plain_text,
            "markdown_text": self.markdown_text,
            "markers": list(self.markers),
            "contract_version": AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
        }


def author_marker_to_plain(raw: str) -> str | None:
    """Return a safe visible author marker without simulating TeX counters."""

    value = raw
    replacements = {
        r"\dagger": "\u2020",
        r"\dag": "\u2020",
        r"\ddagger": "\u2021",
        r"\ddag": "\u2021",
        r"\ast": "*",
        r"\star": "*",
        r"\textdagger": "\u2020",
        r"\textdaggerdbl": "\u2021",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    value = latex_to_plain(value).replace(" ", "")
    if not value or AUTHOR_SUPERSCRIPT_SAFE_MARKER_PATTERN.fullmatch(value) is None:
        return None
    return value


def strip_author_metadata_commands(raw: str) -> str:
    """Remove source arguments that TeX renders outside the author name."""

    value = raw
    for command in ("thanks", "footnote", "email", "orcidlink"):
        cursor = 0
        while True:
            argument = extract_command_argument(value, command, cursor)
            if argument is None:
                break
            _, start, end = argument
            value = value[:start] + " " + value[end:]
            cursor = start + 1
    return re.sub(r"\\footnotemark(?:\s*\[[^\]]*\])?", " ", value)


def author_plan_from_raw(
    raw: str,
    *,
    source_file: Path,
    start_line: int,
    end_line: int,
) -> AuthorSuperscriptPlan | None:
    """Parse simple author markers while failing closed on footnote machinery."""

    if AUTHOR_SUPERSCRIPT_UNSUPPORTED_PATTERN.search(raw):
        return None
    markers: list[str] = []

    def marker_replacement(match: re.Match[str]) -> str:
        marker = author_marker_to_plain(match.group("marker"))
        if marker is None:
            raise ValueError("unsupported author superscript marker")
        index = len(markers)
        markers.append(marker)
        return f"ZZZAUTHORSUP{index:04d}STARTZZZ{marker}ZZZAUTHORSUP{index:04d}ENDZZZ"

    value = raw
    value = re.sub(r"\\raisebox\s*\{[^{}]*\}\s*", "", value)
    value = re.sub(
        r"\\parbox(?:\s*\[[^\]]*\])?\s*\{[^{}]*\}\s*\{(.*)\}",
        r"\1",
        value,
    )
    # ``\\thanks``/``\\footnote`` bodies are inserted elsewhere by TeX and
    # must never leak into the author line.  Their generated symbol remains
    # PDF-authoritative; direct affiliation superscripts around them can still
    # be recovered from source.
    value = strip_author_metadata_commands(value)
    # A separate footnote marker is compiler-generated and not inferred here.
    # Removing it lets an adjacent, explicitly written affiliation marker be
    # matched without claiming that the footnote mark itself is known.
    # Layout-only spacing math sometimes separates names inside ``\author``.
    # Remove only known spacing commands; other remaining math still fails
    # closed below.
    value = re.sub(r"\$\s*\\(?:,|;|!|quad|qquad)\s*\$", " ", value)
    patterns = (
        re.compile(r"\$\s*\^\s*\{(?P<marker>[^{}$]+)\}\s*\$"),
        re.compile(r"\$\s*\^(?P<marker>\\[A-Za-z]+|[0-9A-Za-z*\u2020\u2021])\s*\$"),
        re.compile(r"\\textsuperscript\s*\{(?P<marker>[^{}]+)\}"),
        # LLNCS, EasyChair, and related classes use an explicit ``\inst``
        # argument for the visible affiliation marker.  The marker is already
        # source-declared, so converting it does not require simulating a TeX
        # counter.
        re.compile(r"\\inst\s*\{(?P<marker>[^{}]+)\}"),
    )
    try:
        for pattern in patterns:
            value = pattern.sub(marker_replacement, value)
    except ValueError:
        return None
    if not markers:
        return None
    # Remaining math can affect visible author text and is intentionally not
    # guessed.  Standard author separators and line breaks are layout only.
    if "$" in value or r"\(" in value or r"\)" in value:
        return None
    value = re.sub(r"\\\\(?:\s*\[[^\]]*\])?", " ", value)
    value = re.sub(r"\\(?:quad|qquad|And|AND|and)\b", " ", value)
    rendered = latex_to_plain(value)
    sentinel = re.compile(
        r"ZZZAUTHORSUP(?P<index>\d{4})STARTZZZ"
        r"(?P<marker>.*?)"
        r"ZZZAUTHORSUP(?P=index)ENDZZZ"
    )
    pieces: list[tuple[str, str]] = []
    cursor = 0
    seen: list[str] = []
    for match in sentinel.finditer(rendered):
        if match.start() > cursor:
            pieces.append(("literal", rendered[cursor : match.start()]))
        marker = match.group("marker")
        pieces.append(("sup", marker))
        seen.append(marker)
        cursor = match.end()
    if cursor < len(rendered):
        pieces.append(("literal", rendered[cursor:]))
    if seen != markers or not any(kind == "literal" and value.strip() for kind, value in pieces):
        return None
    return AuthorSuperscriptPlan(
        source_file=source_file,
        start_line=start_line,
        end_line=end_line,
        raw_latex=raw,
        pieces=pieces,
    )


def iter_two_argument_commands(
    text: str,
    command: str,
) -> Iterator[tuple[int, int, str, str]]:
    """Yield balanced two-argument commands without evaluating TeX."""

    pattern = re.compile(rf"\\{re.escape(command)}\s*")
    cursor = 0
    while True:
        match = pattern.search(text, cursor)
        if match is None or match.end() >= len(text) or text[match.end()] != "{":
            break
        first = extract_balanced(text, match.end())
        if first is None:
            break
        first_value, first_end = first
        second_start = first_end
        while second_start < len(text) and text[second_start].isspace():
            second_start += 1
        if second_start >= len(text) or text[second_start] != "{":
            cursor = first_end
            continue
        second = extract_balanced(text, second_start)
        if second is None:
            break
        second_value, end = second
        yield match.start(), end, first_value, second_value
        cursor = end


def parse_icml_author_superscript_plans(
    scan_text: str,
    source_file: Path,
) -> list[AuthorSuperscriptPlan]:
    """Mirror ICML's source-declared affiliation-key numbering."""

    authors = list(iter_two_argument_commands(scan_text, "icmlauthor"))
    affiliations = list(iter_two_argument_commands(scan_text, "icmlaffiliation"))
    if not authors or not affiliations:
        return []
    affiliation_bodies = {key.strip(): body for _, _, key, body in affiliations if key.strip()}
    if not affiliation_bodies:
        return []

    key_numbers: dict[str, int] = {}
    for _, _, _, raw_keys in authors:
        for key in (part.strip() for part in raw_keys.split(",")):
            if key in affiliation_bodies and key not in key_numbers:
                key_numbers[key] = len(key_numbers) + 1
    if not key_numbers:
        return []

    plans: list[AuthorSuperscriptPlan] = []
    for start, end, raw_name, raw_keys in authors:
        name = latex_to_plain(strip_author_metadata_commands(raw_name))
        keys = [part.strip() for part in raw_keys.split(",") if part.strip()]
        marker_parts = (["*"] if "equal" in keys else []) + [
            str(key_numbers[key]) for key in keys if key in key_numbers
        ]
        marker = " ".join(marker_parts)
        if name and marker:
            plans.append(
                AuthorSuperscriptPlan(
                    source_file=source_file,
                    start_line=scan_text.count("\n", 0, start) + 1,
                    end_line=scan_text.count("\n", 0, end) + 1,
                    raw_latex=scan_text[start:end],
                    pieces=[("literal", name), ("sup", marker)],
                )
            )

    affiliation_positions = {key: (start, end, body) for start, end, key, body in affiliations}
    corresponding = list(iter_two_argument_commands(scan_text, "icmlcorrespondingauthor"))
    corresponding_text = []
    for _, _, raw_name, raw_contact in corresponding:
        name = latex_to_plain(raw_name)
        contact = latex_to_plain(raw_contact)
        if name and contact:
            corresponding_text.append(f"{name} <{contact}>")
    for key, number in sorted(key_numbers.items(), key=lambda item: item[1]):
        start, end, body = affiliation_positions[key]
        visible = latex_to_plain(body)
        if visible:
            pieces: list[tuple[str, str]] = [("sup", str(number)), ("literal", visible)]
            raw_latex = scan_text[start:end]
            if number == max(key_numbers.values()) and corresponding_text:
                pieces.append(
                    (
                        "literal",
                        " Correspondence to: " + ", ".join(corresponding_text) + ".",
                    )
                )
                raw_latex += " " + " ".join(
                    scan_text[corresponding_start:corresponding_end]
                    for corresponding_start, corresponding_end, _, _ in corresponding
                )
            plans.append(
                AuthorSuperscriptPlan(
                    source_file=source_file,
                    start_line=scan_text.count("\n", 0, start) + 1,
                    end_line=scan_text.count("\n", 0, end) + 1,
                    raw_latex=raw_latex,
                    pieces=pieces,
                )
            )
    return plans


def find_icml_affiliation_group_repair(
    candidates: Sequence["PageNode"],
    plans: Sequence[AuthorSuperscriptPlan],
) -> tuple[list[AuthorSuperscriptPlan], list["PageNode"], str] | None:
    """Recover an ICML affiliation footnote whose baselines were interleaved.

    Superscript markers and ordinary text often become separate PDF text lines.
    Sorting those lines by ``top`` can yield ``1The 2Cisco University ...``.
    We only repair when source-declared affiliation anchors are unique and the
    shortest contiguous PDF-line window has exactly the same normalized
    character multiset as the complete source group.  Thus source controls the
    order while the compiled page still proves that no visible text is added or
    removed.
    """

    affiliations = [
        plan
        for plan in plans
        if "\\icmlaffiliation" in plan.raw_latex
        and plan.pieces
        and plan.pieces[0][0] == "sup"
    ]
    if len(affiliations) < 2:
        return None
    candidate_keys = [author_alignment_projection(node.text)[0] for node in candidates]
    anchor_indices: list[int] = []
    for plan in affiliations:
        marker_key = author_alignment_projection(plan.markers[0])[0]
        literal = next((value for kind, value in plan.pieces if kind == "literal"), "")
        first_word_match = re.search(r"[^\W_]+", literal, flags=re.UNICODE)
        literal_key = (
            author_alignment_projection(first_word_match.group(0))[0]
            if first_word_match is not None
            else ""
        )
        if not marker_key or len(literal_key) < 3:
            return None
        anchor = marker_key + literal_key
        hits = [index for index, key in enumerate(candidate_keys) if anchor in key]
        if len(hits) != 1:
            return None
        anchor_indices.append(hits[0])

    source_markdown = " ".join(plan.markdown_text for plan in affiliations)
    source_key = author_alignment_projection(" ".join(plan.plain_text for plan in affiliations))[0]
    source_counter = collections.Counter(source_key)
    start = min(anchor_indices)
    minimum_end = max(anchor_indices)
    for end in range(minimum_end, min(len(candidates), start + 16)):
        window = list(candidates[start : end + 1])
        window_key = "".join(author_alignment_projection(node.text)[0] for node in window)
        if len(window_key) == len(source_key) and collections.Counter(window_key) == source_counter:
            return affiliations, window, source_markdown
    return None


def parse_revtex_author_superscript_plans(
    scan_text: str,
    source_file: Path,
) -> list[AuthorSuperscriptPlan]:
    """Recover REVTeX's automatic author/affiliation numbers from source order."""

    if re.search(r"\\documentclass(?:\s*\[[^\]]*\])?\s*\{[^{}]*revtex", scan_text) is None:
        return []
    command_pattern = re.compile(r"\\(?P<command>author|affiliation)\s*")
    commands: list[tuple[str, int, int, str]] = []
    cursor = 0
    while True:
        match = command_pattern.search(scan_text, cursor)
        if match is None or match.end() >= len(scan_text) or scan_text[match.end()] != "{":
            break
        argument = extract_balanced(scan_text, match.end())
        if argument is None:
            break
        raw, end = argument
        commands.append((match.group("command"), match.start(), end, raw))
        cursor = end

    grouped: list[tuple[int, int, str, list[tuple[int, int, str]]]] = []
    current: tuple[int, int, str, list[tuple[int, int, str]]] | None = None
    for command, start, end, raw in commands:
        if command == "author":
            if current is not None:
                grouped.append(current)
            current = (start, end, raw, [])
        elif current is not None:
            current[3].append((start, end, raw))
    if current is not None:
        grouped.append(current)
    if not grouped or not any(affiliations for _, _, _, affiliations in grouped):
        return []

    affiliation_numbers: dict[str, int] = {}
    plans: list[AuthorSuperscriptPlan] = []
    affiliation_plans: dict[str, AuthorSuperscriptPlan] = {}
    for author_start, author_end, raw_author, affiliations in grouped:
        marker_numbers: list[str] = []
        for affiliation_start, affiliation_end, raw_affiliation in affiliations:
            visible_affiliation = latex_to_plain(raw_affiliation)
            key = author_alignment_projection(visible_affiliation)[0]
            if not key:
                continue
            if key not in affiliation_numbers:
                affiliation_numbers[key] = len(affiliation_numbers) + 1
                affiliation_plans[key] = AuthorSuperscriptPlan(
                    source_file=source_file,
                    start_line=scan_text.count("\n", 0, affiliation_start) + 1,
                    end_line=scan_text.count("\n", 0, affiliation_end) + 1,
                    raw_latex=scan_text[affiliation_start:affiliation_end],
                    pieces=[("sup", str(affiliation_numbers[key])), ("literal", visible_affiliation)],
                )
            marker_numbers.append(str(affiliation_numbers[key]))
        visible_author = latex_to_plain(strip_author_metadata_commands(raw_author))
        if visible_author and marker_numbers:
            plans.append(
                AuthorSuperscriptPlan(
                    source_file=source_file,
                    start_line=scan_text.count("\n", 0, author_start) + 1,
                    end_line=scan_text.count("\n", 0, author_end) + 1,
                    raw_latex=scan_text[author_start:author_end],
                    pieces=[("literal", visible_author), ("sup", ",".join(marker_numbers))],
                )
            )
    plans.extend(affiliation_plans.values())
    return plans


def parse_author_superscript_plans(
    executed_sources: Sequence[Path],
) -> list[AuthorSuperscriptPlan]:
    """Extract active-looking ``\\author`` blocks from compiled TeX inputs."""

    plans: list[AuthorSuperscriptPlan] = []
    for source_file in sorted({path.resolve() for path in executed_sources}):
        if source_file.suffix.casefold() != ".tex":
            continue
        try:
            raw_lines = source_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        text = "\n".join(strip_tex_comment(line) for line in raw_lines)
        end_document = re.search(r"\\end\s*\{document\}", text)
        scan_text = text[: end_document.start()] if end_document else text
        cursor = 0
        author_command = re.compile(
            r"\\(?P<command>author|affil)(?P<starred>\*)?"
            r"(?:\s*\[(?P<optional>[^\]]*)\])?\s*"
        )
        while True:
            command_match = author_command.search(scan_text, cursor)
            if command_match is None or command_match.end() >= len(scan_text):
                break
            if scan_text[command_match.end()] != "{":
                cursor = command_match.end()
                continue
            argument = extract_balanced(scan_text, command_match.end())
            if argument is None:
                break
            raw, end = argument
            start = command_match.start()
            plan = author_plan_from_raw(
                raw,
                source_file=source_file,
                start_line=scan_text.count("\n", 0, start) + 1,
                end_line=scan_text.count("\n", 0, end) + 1,
            )
            line_plans: list[AuthorSuperscriptPlan] = []
            raw_start_line = scan_text.count("\n", 0, command_match.end()) + 1
            for line_offset, raw_line in enumerate(raw.splitlines()):
                line_plan = author_plan_from_raw(
                    raw_line,
                    source_file=source_file,
                    start_line=raw_start_line + line_offset,
                    end_line=raw_start_line + line_offset,
                )
                if line_plan is not None:
                    line_plans.append(line_plan)
            optional = command_match.group("optional")
            if line_plans:
                plans.extend(line_plans)
            else:
                # Journal templates commonly use both ``\author[1]{...}`` and
                # ``\author*[1]{...}``.  In either form the optional argument
                # is the source-declared affiliation marker; the star controls
                # separate template behavior and is not required for the
                # number itself to be a superscript.
                if plan is None and optional:
                    raw_marker = optional + ("*" if command_match.group("starred") else "")
                    marker = author_marker_to_plain(raw_marker)
                    visible = latex_to_plain(strip_author_metadata_commands(raw))
                    if marker and visible:
                        pieces = (
                            [("sup", marker), ("literal", visible)]
                            if command_match.group("command") == "affil"
                            else [("literal", visible), ("sup", marker)]
                        )
                        plan = AuthorSuperscriptPlan(
                            source_file=source_file,
                            start_line=scan_text.count("\n", 0, start) + 1,
                            end_line=scan_text.count("\n", 0, end) + 1,
                            raw_latex=raw,
                            pieces=pieces,
                        )
                if plan is not None:
                    plans.append(plan)
            cursor = end
        plans.extend(parse_icml_author_superscript_plans(scan_text, source_file))
        plans.extend(parse_revtex_author_superscript_plans(scan_text, source_file))
    deduplicated: dict[tuple[str, tuple[str, ...]], AuthorSuperscriptPlan] = {}
    for plan in plans:
        deduplicated[(author_alignment_projection(plan.plain_text)[0], tuple(plan.markers))] = plan
    return list(deduplicated.values())


def author_alignment_projection(value: str) -> tuple[str, list[int]]:
    """Normalize author text and retain each normalized character's origin."""

    normalized: list[str] = []
    origins: list[int] = []
    for index, character in enumerate(value):
        for projected in unicodedata.normalize("NFKD", character).casefold():
            if unicodedata.combining(projected):
                continue
            if projected in {"\u2217", "\u204e", "\u22c6"}:
                projected = "*"
            if projected.isalnum() or projected in "*\u2020\u2021":
                normalized.append(projected)
                origins.append(index)
    return "".join(normalized), origins


def apply_author_superscript_plans(
    nodes: list["PageNode"],
    plans: Sequence[AuthorSuperscriptPlan],
    *,
    page_number: int,
    page_width: float,
) -> tuple[list["PageNode"], dict[str, Any]]:
    """Inject source-confirmed ``<sup>`` tags into PDF-authoritative lines."""

    base_audit: dict[str, Any] = {
        "contract_version": AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
        "status": "not_applicable" if page_number != 1 else "not_present",
        "plans": len(plans),
        "superscripts_emitted": 0,
        "markers": [],
        "matched_line_ids": [],
        "marked_lines": [],
    }
    if page_number != 1 or not plans:
        return nodes, base_audit
    ordered, _ = order_page_nodes(nodes, page_width)
    candidates = [
        node
        for node in ordered
        if node.kind == "text" and node.line_id
    ]
    joined_parts: list[str] = []
    line_spans: list[tuple[PageNode, int, int]] = []
    joined_cursor = 0
    for index, node in enumerate(candidates):
        start = joined_cursor
        joined_parts.append(node.text)
        joined_cursor += len(node.text)
        line_spans.append((node, start, joined_cursor))
        if index + 1 < len(candidates):
            joined_parts.append(" ")
            joined_cursor += 1
    joined = "".join(joined_parts)
    candidate_key, candidate_origins = author_alignment_projection(joined)
    icml_group_repair = find_icml_affiliation_group_repair(candidates, plans)
    repaired_plan_ids = (
        {id(plan) for plan in icml_group_repair[0]}
        if icml_group_repair is not None
        else set()
    )
    matches: list[tuple[AuthorSuperscriptPlan, int, int]] = []
    for plan in plans:
        if id(plan) in repaired_plan_ids:
            continue
        source_key = "".join(
            author_alignment_projection(value)[0] for _, value in plan.pieces
        )
        if len(source_key) < 5:
            continue
        starts = [
            match.start()
            for match in re.finditer(re.escape(source_key), candidate_key)
        ]
        if len(starts) == 1:
            matches.append((plan, starts[0], starts[0] + len(source_key)))
    if not matches and icml_group_repair is None:
        base_audit["status"] = "unmatched"
        base_audit["matched_plans"] = 0
        return nodes, base_audit
    repaired_plans = icml_group_repair[0] if icml_group_repair is not None else []
    all_plans_matched = len(matches) + len(repaired_plans) == len(plans)
    replacements: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
    emitted_markers: list[str] = []
    matched_nodes: set[int] = set()
    for plan, match_key_start, _ in matches:
        source_key_cursor = match_key_start
        for kind, value in plan.pieces:
            piece_key, _ = author_alignment_projection(value)
            key_start = source_key_cursor
            source_key_cursor += len(piece_key)
            if kind != "sup":
                continue
            key_end = source_key_cursor
            if key_start >= key_end or key_end > len(candidate_origins):
                base_audit["status"] = "mapping_failed"
                return nodes, base_audit
            char_start = candidate_origins[key_start]
            char_end = candidate_origins[key_end - 1] + 1
            owner = next(
                (
                    (node, line_start, line_end)
                    for node, line_start, line_end in line_spans
                    if line_start <= char_start < char_end <= line_end
                ),
                None,
            )
            if owner is None:
                base_audit["status"] = "marker_crossed_line"
                return nodes, base_audit
            node, line_start, _ = owner
            relative_start = char_start - line_start
            relative_end = char_end - line_start
            visible = node.text[relative_start:relative_end]
            if author_alignment_projection(visible)[0] != author_alignment_projection(value)[0]:
                base_audit["status"] = "marker_mismatch"
                return nodes, base_audit
            replacements[id(node)].append((relative_start, relative_end, value))
            emitted_markers.append(value)
            matched_nodes.add(id(node))
    flattened_ranges = [
        (id(node), start, end)
        for node in candidates
        for start, end, _ in replacements.get(id(node), [])
    ]
    if len(flattened_ranges) != len(set(flattened_ranges)):
        base_audit["status"] = "overlapping_markers"
        return nodes, base_audit
    marked_lines: list[str] = []
    for node in candidates:
        text = node.text
        for start, end, marker in sorted(replacements.get(id(node), []), reverse=True):
            text = (
                text[:start]
                + f"<sup>{html.escape(marker, quote=False)}</sup>"
                + text[end:]
            )
        node.text = text
        if replacements.get(id(node)):
            marked_lines.append(text)
    repaired_line_ids: list[str] = []
    if icml_group_repair is not None:
        _, group_nodes, group_markdown = icml_group_repair
        group_node_ids = {id(node) for node in group_nodes}
        claimed_line_ids = [
            line_id
            for node in group_nodes
            for line_id in node.claimed_line_ids
        ]
        derived_line_ids = [
            line_id
            for node in group_nodes
            for line_id in node.derived_line_ids
        ]
        repaired_line_ids.extend(claimed_line_ids)
        first = group_nodes[0]
        merged = PageNode(
            kind="text",
            text=group_markdown,
            bbox=[
                min(node.bbox[0] for node in group_nodes),
                min(node.bbox[1] for node in group_nodes),
                max(node.bbox[2] for node in group_nodes),
                max(node.bbox[3] for node in group_nodes),
            ],
            font_size=statistics.median(node.font_size for node in group_nodes),
            lane=first.lane,
            line_id=first.line_id,
            origin_page=first.origin_page,
            origin_order=min(
                node.origin_order
                for node in group_nodes
                if node.origin_order is not None
            ),
            claimed_line_ids=claimed_line_ids,
            derived_line_ids=derived_line_ids,
        )
        rebuilt_nodes: list[PageNode] = []
        inserted = False
        for node in nodes:
            if id(node) not in group_node_ids:
                rebuilt_nodes.append(node)
            elif not inserted:
                rebuilt_nodes.append(merged)
                inserted = True
        nodes = rebuilt_nodes
        marked_lines.append(group_markdown)
        emitted_markers.extend(
            marker for plan in repaired_plans for marker in plan.markers
        )
    base_audit.update(
        {
            "status": "passed" if all_plans_matched else "partial",
            "matched_plans": len(matches) + len(repaired_plans),
            "unmatched_plans": len(plans) - len(matches) - len(repaired_plans),
            "source_blocks": [
                {
                    "source_file": str(plan.source_file),
                    "source_lines": [plan.start_line, plan.end_line],
                }
                for plan in [
                    *(plan for plan, _, _ in matches),
                    *repaired_plans,
                ]
            ],
            "superscripts_emitted": len(emitted_markers),
            "markers": emitted_markers,
            "matched_line_ids": sorted(
                {
                    *(str(node.line_id) for node in candidates if id(node) in matched_nodes),
                    *(str(line_id) for line_id in repaired_line_ids),
                }
            ),
            "marked_lines": marked_lines,
            "icml_affiliation_group_repaired": icml_group_repair is not None,
        }
    )
    return nodes, base_audit


@dataclasses.dataclass
class SourceBlock:
    block_id: str
    kind: str
    source_file: Path
    start_line: int
    end_line: int
    raw_latex: str
    markdown: str
    query_lines: list[int]
    heading_level: int | None = None
    heading_command: str | None = None
    heading_starred: bool = False
    heading_source_title: str | None = None
    pdf_visible_heading: str | None = None
    visible_number_prefix: str | None = None
    heading_number_status: str | None = None
    heading_structure_status: str | None = None
    heading_matched_line_count: int = 0
    pdf_visible_caption: str | None = None
    visible_caption_prefix: str | None = None
    caption_number_status: str | None = None
    caption_markdown: str | None = None
    table_html: str | None = None
    table_parse_status: str | None = None
    pdf_visible_formula_number: str | None = None
    formula_number_status: str | None = None
    page: int | None = None
    bbox: list[float] | None = None
    mapping_candidates: int = 0
    mapping_status: str = "pending"

    def as_json(self, source_root: Path) -> dict[str, Any]:
        try:
            relative_source = self.source_file.relative_to(source_root).as_posix()
        except ValueError:
            relative_source = str(self.source_file)
        value = {
            "block_id": self.block_id,
            "kind": self.kind,
            "source_file": relative_source,
            "source_lines": [self.start_line, self.end_line],
            "query_lines": self.query_lines,
            "page": self.page,
            "bbox": self.bbox,
            "mapping_candidates": self.mapping_candidates,
            "mapping_status": self.mapping_status,
            "markdown": self.markdown,
            "raw_latex": self.raw_latex,
        }
        if self.kind == "heading":
            value.update(
                {
                    "heading_command": self.heading_command,
                    "heading_starred": self.heading_starred,
                    "heading_source_title": self.heading_source_title,
                    "pdf_visible_heading": self.pdf_visible_heading,
                    "visible_number_prefix": self.visible_number_prefix,
                    "heading_number_status": self.heading_number_status,
                    "heading_structure_status": self.heading_structure_status,
                    "heading_matched_line_count": self.heading_matched_line_count,
                    "heading_level": self.heading_level,
                }
            )
        elif self.kind == "table":
            value.update(
                {
                    "pdf_visible_caption": self.pdf_visible_caption,
                    "visible_caption_prefix": self.visible_caption_prefix,
                    "caption_number_status": self.caption_number_status,
                    "caption_markdown": self.caption_markdown,
                    "table_html": self.table_html,
                    "table_parse_status": self.table_parse_status,
                }
            )
        elif self.kind == "display_math":
            value.update(
                {
                    "pdf_visible_formula_number": self.pdf_visible_formula_number,
                    "formula_number_status": self.formula_number_status,
                }
            )
        return value


@dataclasses.dataclass
class SourceParagraph:
    """One TeX paragraph or list item used only as a boundary annotation.

    Visible wording still comes from the compiled PDF.  This object never
    replaces PDF text; it only lets the writer decide which adjacent PDF lines
    belong to the same source paragraph.
    """

    paragraph_id: str
    kind: str
    source_file: Path
    source_lines: list[int]
    raw_latex: str
    list_environment: str | None = None
    item_depth: int | None = None
    item_ordinal: int | None = None

    @property
    def start_line(self) -> int:
        return min(self.source_lines)

    @property
    def end_line(self) -> int:
        return max(self.source_lines)

    def as_json(self, source_root: Path) -> dict[str, Any]:
        try:
            relative_source = self.source_file.relative_to(source_root).as_posix()
        except ValueError:
            relative_source = str(self.source_file)
        return {
            "source_paragraph_id": self.paragraph_id,
            "kind": self.kind,
            "source_file": relative_source,
            "source_lines": [self.start_line, self.end_line],
            "source_line_numbers": list(self.source_lines),
            "raw_latex": self.raw_latex,
            "list_environment": self.list_environment,
            "item_depth": self.item_depth,
            "item_ordinal": self.item_ordinal,
        }


@dataclasses.dataclass(frozen=True)
class SourceParagraphPoint:
    """A glyph-level SyncTeX point carrying a source paragraph identity."""

    page: int
    x: float
    y: float
    paragraph_id: str
    source_file: Path
    source_line: int


@dataclasses.dataclass
class FootnoteSpec:
    """One source footnote plus its PDF-authoritative visible marker/body."""

    note_id: str
    node_id: int
    source_raw: str
    body_raw: str
    optional_arguments: list[str] = dataclasses.field(default_factory=list)
    marker: str | None = None
    callout_line_ids: list[str] = dataclasses.field(default_factory=list)
    definition_line_ids: list[str] = dataclasses.field(default_factory=list)
    rendered_body: str | None = None
    status: str = "pending"
    failure_reason: str | None = None
    content_validation_status: str = "pending"
    body_feature_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    content_validation_issues: list[str] = dataclasses.field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "node_id": self.node_id,
            "source_raw": self.source_raw,
            "body_raw": self.body_raw,
            "optional_arguments": list(self.optional_arguments),
            "marker": self.marker,
            "callout_line_ids": list(self.callout_line_ids),
            "definition_line_ids": list(self.definition_line_ids),
            "rendered_body": self.rendered_body,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "content_validation_status": self.content_validation_status,
            "body_feature_counts": dict(sorted(self.body_feature_counts.items())),
            "content_validation_issues": list(self.content_validation_issues),
            "representation": FOOTNOTE_REPRESENTATION,
        }


@dataclasses.dataclass
class InlineSourceBlock:
    block_id: str
    source_file: Path
    start_line: int
    end_line: int
    raw_latex: str
    plan: InlinePlan
    query_lines: list[int]
    page: int | None = None
    bbox: list[float] | None = None
    candidate_pages: list[int] = dataclasses.field(default_factory=list)
    mapping_candidates: int = 0
    mapping_status: str = "pending"
    match_status: str = "pending"
    match_reason: str | None = None
    matched_line_count: int = 0
    matched_bbox: list[float] | None = None
    matched_pdf_text: str | None = None
    enriched_markdown: str | None = None
    match_score: float | None = None
    absorbed_pdf_line_ids: list[str] = dataclasses.field(default_factory=list)
    footnotes: list[FootnoteSpec] = dataclasses.field(default_factory=list)

    @property
    def target_feature_counts(self) -> dict[str, int]:
        return {
            key: int(self.plan.feature_counts.get(key, 0))
            for key in ("math", "strong", "em", "code", "footnote")
            if int(self.plan.feature_counts.get(key, 0)) > 0
        }

    @property
    def target_feature_total(self) -> int:
        return sum(self.target_feature_counts.values())

    def as_json(self, source_root: Path) -> dict[str, Any]:
        try:
            relative_source = self.source_file.relative_to(source_root).as_posix()
        except ValueError:
            relative_source = str(self.source_file)
        return {
            "block_id": self.block_id,
            "kind": "inline_prose",
            "source_file": relative_source,
            "source_lines": [self.start_line, self.end_line],
            "query_lines": self.query_lines,
            "page": self.page,
            "bbox": self.bbox,
            "candidate_pages": self.candidate_pages,
            "mapping_candidates": self.mapping_candidates,
            "mapping_status": self.mapping_status,
            "match_status": self.match_status,
            "match_reason": self.match_reason,
            "matched_line_count": self.matched_line_count,
            "matched_bbox": self.matched_bbox,
            "matched_pdf_text": self.matched_pdf_text,
            "enriched_markdown": self.enriched_markdown,
            "match_score": self.match_score,
            "absorbed_pdf_line_ids": list(self.absorbed_pdf_line_ids),
            "target_feature_counts": self.target_feature_counts,
            "target_feature_total": self.target_feature_total,
            "plan": summarize_inline_plan(self.plan),
            "footnotes": [footnote.as_json() for footnote in self.footnotes],
            "raw_latex": self.raw_latex,
        }


@dataclasses.dataclass(frozen=True)
class SimpleMathMacro:
    name: str
    argument_count: int
    body: str


def collect_simple_math_macros(source_root: Path) -> dict[str, SimpleMathMacro]:
    """Collect unambiguous, brace-defined math macros from an arXiv source tree."""

    definitions: dict[str, set[tuple[int, str]]] = collections.defaultdict(set)
    command_pattern = re.compile(
        r"\\(?:newcommand|renewcommand|providecommand)\*?\s*"
        r"(?:\{\s*\\(?P<braced>[A-Za-z@]+)\s*\}|\\(?P<plain>[A-Za-z@]+))"
    )
    operator_pattern = re.compile(
        r"\\DeclareMathOperator\*?\s*\{\s*\\(?P<name>[A-Za-z@]+)\s*\}\s*\{"
    )
    for source_file in sorted(
        path
        for path in source_root.rglob("*")
        if path.suffix.casefold() in {".tex", ".sty"} and path.is_file()
    ):
        try:
            raw = source_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = "\n".join(strip_tex_comment(line) for line in raw.splitlines())
        for match in command_pattern.finditer(text):
            name = match.group("braced") or match.group("plain")
            cursor = match.end()
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            argument_count = 0
            optional = re.match(r"\[(\d+)\]", text[cursor:])
            if optional:
                argument_count = int(optional.group(1))
                cursor += optional.end()
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                # Optional/default macro arguments require TeX execution and
                # are deliberately not approximated here.
                if cursor < len(text) and text[cursor] == "[":
                    continue
            body = extract_balanced(text, cursor)
            if body is None or argument_count > 4:
                continue
            definitions[name].add((argument_count, body[0]))
        for match in operator_pattern.finditer(text):
            body = extract_balanced(text, match.end() - 1)
            if body is not None:
                definitions[match.group("name")].add(
                    (0, r"\operatorname{" + body[0] + "}")
                )
    return {
        name: SimpleMathMacro(name, next(iter(values))[0], next(iter(values))[1])
        for name, values in definitions.items()
        if len(values) == 1
    }


def expand_simple_math_macros(
    value: str, macros: dict[str, SimpleMathMacro], *, maximum_passes: int = 8
) -> str:
    """Expand only macros whose complete braced definition is known locally."""

    current = value
    command_pattern = re.compile(r"\\([A-Za-z@]+)")
    for _ in range(maximum_passes):
        changed = False
        pieces: list[str] = []
        cursor = 0
        for match in command_pattern.finditer(current):
            if match.start() < cursor:
                continue
            macro = macros.get(match.group(1))
            if macro is None:
                continue
            argument_cursor = match.end()
            arguments: list[str] = []
            valid = True
            for _argument_index in range(macro.argument_count):
                while argument_cursor < len(current) and current[argument_cursor].isspace():
                    argument_cursor += 1
                argument = extract_balanced(current, argument_cursor)
                if argument is None:
                    valid = False
                    break
                arguments.append(argument[0])
                argument_cursor = argument[1]
            if not valid:
                continue
            replacement = macro.body
            for argument_index, argument in enumerate(arguments, start=1):
                replacement = replacement.replace(f"#{argument_index}", argument)
            pieces.append(current[cursor : match.start()])
            pieces.append(replacement)
            cursor = argument_cursor
            changed = True
        if not changed:
            return current
        pieces.append(current[cursor:])
        updated = "".join(pieces)
        if updated == current:
            return current
        current = updated
    return current


def expand_inline_plan_math(
    plan: InlinePlan, macros: dict[str, SimpleMathMacro]
) -> InlinePlan:
    if not macros:
        return plan

    def visit(node: Any) -> Any:
        children = tuple(visit(child) for child in node.children)
        value = (
            expand_simple_math_macros(node.value, macros)
            if node.kind == "math"
            else node.value
        )
        return dataclasses.replace(node, value=value, children=children)

    return dataclasses.replace(plan, root=visit(plan.root))


def clean_display_math(
    raw: str,
    environment: str,
    macros: dict[str, SimpleMathMacro] | None = None,
) -> str:
    value = raw
    value = re.sub(r"\\begin\s*\{" + re.escape(environment) + r"\}(?:\s*\[[^\]]*\])?", "", value, count=1)
    value = re.sub(r"\\end\s*\{" + re.escape(environment) + r"\}\s*$", "", value, count=1)
    value = re.sub(r"\\label\s*\{[^{}]*\}", "", value)
    value = re.sub(r"\\(?:nonumber|notag)\b", "", value)
    value = value.strip()
    value = expand_simple_math_macros(value, macros or {})
    if environment.startswith(("align", "flalign", "gather", "multline", "eqnarray")):
        value = "\\begin{aligned}\n" + value + "\n\\end{aligned}"
    return "$$\n" + value + "\n$$"


def split_unescaped(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    brace_depth = 0
    math_mode = False
    index = 0
    while index < len(value):
        character = value[index]
        if brace_depth == 0 and not math_mode and value.startswith(delimiter, index):
            parts.append(value[start:index])
            start = index + len(delimiter)
            index = start
            continue
        if character == "\\":
            index += 2
            continue
        if character == "$":
            math_mode = not math_mode
            index += 1
            continue
        if not math_mode:
            if character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth = max(0, brace_depth - 1)
        index += 1
    parts.append(value[start:])
    return parts


def latex_inline_to_html(value: str) -> str:
    value = normalize_space(value)
    math_parts: list[str] = []

    def protect_math(match: re.Match[str]) -> str:
        math_parts.append(match.group(0))
        return f"@@MATH{len(math_parts) - 1}@@"

    value = re.sub(r"\$[^$]*\$|\\\([^)]*\\\)", protect_math, value)
    value = re.sub(r"\\multicolumn\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\multirow\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\textbf\s*\{([^{}]*)\}", r"@@STRONG@@\1@@/STRONG@@", value)
    value = re.sub(r"\\(?:emph|textit)\s*\{([^{}]*)\}", r"@@EM@@\1@@/EM@@", value)
    value = latex_to_plain(value)
    value = html.escape(value, quote=False)
    value = value.replace("@@STRONG@@", "<strong>").replace("@@/STRONG@@", "</strong>")
    value = value.replace("@@EM@@", "<em>").replace("@@/EM@@", "</em>")
    value = value.replace("@@BR@@", "<br>")
    for index, math_value in enumerate(math_parts):
        value = value.replace(f"@@MATH{index}@@", html.escape(math_value, quote=False))
    return value.strip()


@dataclasses.dataclass
class TableCell:
    value: str
    colspan: int = 1
    rowspan: int = 1


@dataclasses.dataclass(frozen=True)
class TableRenderResult:
    """Source-derived table parts kept separate in the Markdown GT.

    ``caption_markdown`` is deliberately outside ``table_html``.  Parse
    provenance belongs in the JSON sidecar, never in HTML ``data-*``
    attributes that would otherwise become training targets.
    """

    caption_markdown: str | None
    table_html: str
    parse_status: str


def parse_table_cell(raw: str) -> TableCell:
    raw = raw.strip()
    colspan = 1
    rowspan = 1
    multicolumn = re.search(r"\\multicolumn\s*\{(\d+)\}\s*\{[^{}]*\}\s*\{", raw)
    if multicolumn:
        colspan = max(1, int(multicolumn.group(1)))
        argument = extract_balanced(raw, multicolumn.end() - 1)
        if argument:
            raw = raw[: multicolumn.start()] + argument[0] + raw[argument[1] :]
    multirow = re.search(r"\\multirow\s*(?:\[[^\]]*\])?\s*\{(\d+)\}\s*\{[^{}]*\}\s*\{", raw)
    if multirow:
        rowspan = max(1, int(multirow.group(1)))
        argument = extract_balanced(raw, multirow.end() - 1)
        if argument:
            raw = raw[: multirow.start()] + argument[0] + raw[argument[1] :]
    raw = re.sub(r"\\(?:cellcolor|rowcolor)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}", "", raw)
    return TableCell(latex_inline_to_html(raw), colspan=colspan, rowspan=rowspan)


def find_environment(raw: str, names: set[str]) -> tuple[str, str, int, int] | None:
    begin_pattern = re.compile(r"\\begin\s*\{(" + "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True)) + r")\}")
    match = begin_pattern.search(raw)
    if not match:
        return None
    name = match.group(1)
    depth = 1
    token_pattern = re.compile(r"\\(begin|end)\s*\{" + re.escape(name) + r"\}")
    for token in token_pattern.finditer(raw, match.end()):
        if token.group(1) == "begin":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return name, raw[match.end() : token.start()], match.start(), token.end()
    return None


def replace_nested_tabulars(content: str) -> str:
    """Collapse tabulars nested inside cells to protected ``<br>`` text.

    Nested one-column tabulars are commonly used only to create line breaks in
    a cell.  Their row terminators must not be mistaken for rows of the outer
    table.
    """
    token_pattern = re.compile(
        r"\\(begin|end)\s*\{("
        + "|".join(re.escape(name) for name in sorted(TABULAR_ENVS, key=len, reverse=True))
        + r")\}"
    )
    while True:
        stack: list[re.Match[str]] = []
        pair: tuple[re.Match[str], re.Match[str]] | None = None
        for token in token_pattern.finditer(content):
            if token.group(1) == "begin":
                stack.append(token)
                continue
            if stack:
                begin = stack.pop()
                pair = (begin, token)
                break
        if pair is None:
            return content
        begin, end = pair
        cursor = begin.end()
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        optional = re.match(r"\[[^\]]*\]", content[cursor:])
        if optional:
            cursor += optional.end()
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        required_arguments = 2 if begin.group(2) in {"tabularx", "tabulary"} else 1
        for _ in range(required_arguments):
            argument = extract_balanced(content, cursor)
            if argument is None:
                break
            _, cursor = argument
            while cursor < len(content) and content[cursor].isspace():
                cursor += 1
        inner = content[cursor : end.start()]
        rows = [normalize_space(row) for row in split_unescaped(inner, r"\\")]
        rendered_rows: list[str] = []
        for row in rows:
            if not row:
                continue
            cells = [normalize_space(cell) for cell in split_unescaped(row, "&")]
            rendered_rows.append(" ".join(cell for cell in cells if cell))
        replacement = "@@BR@@".join(rendered_rows)
        content = content[: begin.start()] + replacement + content[end.end() :]


def render_table(raw: str, table_id: str | None = None) -> TableRenderResult:
    del table_id  # Stable public call shape; identifiers live in sidecar JSON.
    caption_result = extract_command_argument(raw, "caption")
    caption = latex_inline_to_html(caption_result[0]) if caption_result else ""
    caption_markdown = caption or None
    found = find_environment(raw, TABULAR_ENVS)
    if not found:
        escaped = html.escape(raw.strip())
        return TableRenderResult(
            caption_markdown=caption_markdown,
            table_html=(
                "<table>\n"
                f"  <tbody><tr><td><pre>{escaped}</pre></td></tr></tbody>\n"
                "</table>"
            ),
            parse_status="raw_latex",
        )
    tabular_environment, content, _, _ = found
    # The environment matcher stops immediately after ``\begin{tabular}``;
    # remove its column specification (and width for tabularx/tabulary).
    leading_arguments = 2 if tabular_environment in {"tabularx", "tabulary"} else 1
    content_cursor = 0
    for _ in range(leading_arguments):
        while content_cursor < len(content) and content[content_cursor].isspace():
            content_cursor += 1
        optional = re.match(r"\[[^\]]*\]", content[content_cursor:])
        if optional:
            content_cursor += optional.end()
            while content_cursor < len(content) and content[content_cursor].isspace():
                content_cursor += 1
        argument = extract_balanced(content, content_cursor)
        if argument is None:
            break
        _, content_cursor = argument
    content = replace_nested_tabulars(content[content_cursor:])
    # Horizontal rules describe structure but are not rows.  Only midrule is
    # used as the explicit boundary between thead and tbody; cmidrule/cline
    # often appear inside multi-line headers.
    content = re.sub(r"\\(?:toprule|bottomrule|hline)\b", "", content)
    content = re.sub(
        r"\\midrule\s*(?:\([^)]*\))?\s*(?:\{[^{}]*\})?",
        lambda _: r"\\@@HEADER_BREAK@@\\",
        content,
    )
    content = re.sub(
        r"\\(?:cmidrule|cline)\s*(?:\([^)]*\))?\s*(?:\{[^{}]*\})?",
        "",
        content,
    )
    content = re.sub(r"\\addlinespace(?:\[[^\]]*\])?", "", content)
    raw_rows = split_unescaped(content, r"\\")
    rows: list[list[TableCell]] = []
    header_break_after: int | None = None
    for row_raw in raw_rows:
        if "@@HEADER_BREAK@@" in row_raw:
            if rows and header_break_after is None:
                header_break_after = len(rows)
            row_raw = row_raw.replace("@@HEADER_BREAK@@", "")
        row_raw = re.sub(r"\\(?:noalign|rule)\s*\{[^{}]*\}", "", row_raw)
        row_raw = row_raw.strip()
        if not row_raw:
            continue
        cells = [parse_table_cell(cell) for cell in split_unescaped(row_raw, "&")]
        if any(cell.value or cell.colspan > 1 or cell.rowspan > 1 for cell in cells):
            rows.append(cells)
    if not rows:
        escaped = html.escape(content.strip())
        return TableRenderResult(
            caption_markdown=caption_markdown,
            table_html=(
                "<table>\n"
                f"  <tbody><tr><td><pre>{escaped}</pre></td></tr></tbody>\n"
                "</table>"
            ),
            parse_status="empty",
        )
    if header_break_after is None:
        # Multi-level headers commonly put a multirow in the first row and use
        # only cmidrule/cline below it.  The largest first-row rowspan gives a
        # better thead boundary than blindly selecting one row.
        header_break_after = min(len(rows), max((cell.rowspan for cell in rows[0]), default=1))
    active_rowspans: dict[int, int] = {}
    rendered_rows: list[list[tuple[int, TableCell]]] = []
    for cells in rows:
        for column in list(active_rowspans):
            active_rowspans[column] -= 1
            if active_rowspans[column] <= 0:
                del active_rowspans[column]
        column = 0
        rendered: list[tuple[int, TableCell]] = []
        for cell in cells:
            if column in active_rowspans and not cell.value and cell.colspan == 1 and cell.rowspan == 1:
                # Some LaTeX tables include an explicit empty ``&`` placeholder
                # below a multirow; consume it instead of shifting later cells.
                column += 1
                continue
            while column in active_rowspans:
                column += 1
            rendered.append((column, cell))
            if cell.rowspan > 1:
                for occupied in range(column, column + cell.colspan):
                    active_rowspans[occupied] = max(active_rowspans.get(occupied, 0), cell.rowspan)
            column += cell.colspan
        rendered_rows.append(rendered)
    lines = ["<table>"]
    sections = [("thead", rendered_rows[:header_break_after]), ("tbody", rendered_rows[header_break_after:])]
    for section_name, section_rows in sections:
        if not section_rows:
            continue
        lines.append(f"  <{section_name}>")
        for row in section_rows:
            lines.append("    <tr>")
            tag = "th" if section_name == "thead" else "td"
            for _, cell in row:
                attributes = []
                if cell.colspan > 1:
                    attributes.append(f'colspan="{cell.colspan}"')
                if cell.rowspan > 1:
                    attributes.append(f'rowspan="{cell.rowspan}"')
                attribute_text = (" " + " ".join(attributes)) if attributes else ""
                lines.append(f"      <{tag}{attribute_text}>{cell.value}</{tag}>")
            lines.append("    </tr>")
        lines.append(f"  </{section_name}>")
    lines.append("</table>")
    return TableRenderResult(
        caption_markdown=caption_markdown,
        table_html="\n".join(lines),
        parse_status="parsed",
    )


def table_to_html(raw: str, table_id: str | None = None) -> str:
    """Compatibility wrapper returning only clean, attribute-free HTML."""

    return render_table(raw, table_id).table_html


def compose_table_markdown(
    caption_markdown: str | None, table_html: str | None
) -> str:
    """Serialize a caption paragraph followed by a standalone HTML table."""

    return "\n\n".join(
        part.strip() for part in (caption_markdown, table_html) if part and part.strip()
    )


def parse_source_blocks(
    source_root: Path,
    math_macros: dict[str, SimpleMathMacro] | None = None,
) -> list[SourceBlock]:
    blocks: list[SourceBlock] = []
    serial = 0
    for source_file in sorted(source_root.rglob("*.tex")):
        try:
            text = source_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        raw_lines = text.splitlines()
        lines = [strip_tex_comment(line) for line in raw_lines]
        begin_document = next(
            (
                line_index
                for line_index, line in enumerate(lines)
                if re.search(r"\\begin\s*\{document\}", line)
            ),
            None,
        )
        if begin_document is not None:
            scan_start = begin_document + 1
            scan_end = next(
                (
                    line_index
                    for line_index in range(scan_start, len(lines))
                    if re.search(r"\\end\s*\{document\}", lines[line_index])
                ),
                len(lines),
            )
        elif any(re.search(r"\\documentclass\b", line) for line in lines):
            # A main file without an active document body cannot contribute
            # rendered structural blocks.  Included fragment files usually
            # have no documentclass and remain eligible in full.
            continue
        else:
            scan_start = 0
            scan_end = len(lines)
        index = scan_start
        while index < scan_end:
            line = lines[index]
            section_match = re.search(
                r"\\(?P<command>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)"
                r"(?P<starred>\*)?\s*\{",
                line,
            )
            if section_match:
                accumulated = "\n".join(lines[index : min(scan_end, index + 20)])
                argument = extract_balanced(accumulated, section_match.end() - 1)
                if argument:
                    serial += 1
                    title = latex_to_plain(argument[0])
                    command = section_match.group("command")
                    starred = bool(section_match.group("starred"))
                    level = SECTION_LEVELS[command]
                    blocks.append(
                        SourceBlock(
                            block_id=f"b{serial:06d}",
                            kind="heading",
                            source_file=source_file.resolve(),
                            start_line=index + 1,
                            end_line=index + argument[0].count("\n") + 1,
                            raw_latex=argument[0],
                            markdown="#" * level + " " + title,
                            query_lines=[index + 1],
                            heading_level=level,
                            heading_command=command,
                            heading_starred=starred,
                            heading_source_title=title,
                            heading_number_status="pending",
                            heading_structure_status="pending",
                        )
                    )
            begin_matches = list(re.finditer(r"\\begin\s*\{([^{}]+)\}", line))
            handled_end = index
            for begin_match in begin_matches:
                environment = begin_match.group(1)
                if environment not in DISPLAY_ENVS and environment not in TABLE_ENVS:
                    continue
                depth = 0
                end_index: int | None = None
                tabular_end_lines: list[int] = []
                caption_lines: list[int] = []
                for cursor in range(index, scan_end):
                    cursor_line = lines[cursor]
                    depth += len(re.findall(r"\\begin\s*\{" + re.escape(environment) + r"\}", cursor_line))
                    depth -= len(re.findall(r"\\end\s*\{" + re.escape(environment) + r"\}", cursor_line))
                    if environment in TABLE_ENVS:
                        if re.search(r"\\caption\*?(?:\[[^\]]*\])?\s*\{", cursor_line):
                            caption_lines.append(cursor + 1)
                        if any(re.search(r"\\end\s*\{" + re.escape(name) + r"\}", cursor_line) for name in TABULAR_ENVS):
                            tabular_end_lines.append(cursor + 1)
                    if depth == 0:
                        end_index = cursor
                        break
                if end_index is None:
                    continue
                raw = "\n".join(lines[index : end_index + 1]).strip()
                serial += 1
                if environment in DISPLAY_ENVS:
                    markdown = clean_display_math(raw, environment, math_macros)
                    query_lines = [end_index + 1]
                    kind = "display_math"
                    table_render = None
                else:
                    table_render = render_table(raw, f"b{serial:06d}")
                    markdown = compose_table_markdown(
                        table_render.caption_markdown, table_render.table_html
                    )
                    query_lines = tabular_end_lines + caption_lines
                    if not query_lines:
                        query_lines = [end_index + 1]
                    kind = "table"
                blocks.append(
                    SourceBlock(
                        block_id=f"b{serial:06d}",
                        kind=kind,
                        source_file=source_file.resolve(),
                        start_line=index + 1,
                        end_line=end_index + 1,
                        raw_latex=raw,
                        markdown=markdown,
                        query_lines=sorted(set(query_lines)),
                        caption_markdown=(
                            table_render.caption_markdown if table_render else None
                        ),
                        table_html=table_render.table_html if table_render else None,
                        table_parse_status=(
                            table_render.parse_status if table_render else None
                        ),
                    )
                )
                handled_end = max(handled_end, end_index)
                break
            index = max(index + 1, handled_end + 1)
    # Display math nested inside tables should be represented by the table only.
    table_ranges = [
        (block.source_file, block.start_line, block.end_line)
        for block in blocks
        if block.kind == "table"
    ]
    filtered: list[SourceBlock] = []
    for block in blocks:
        if block.kind == "display_math" and any(
            source == block.source_file and start <= block.start_line <= end
            for source, start, end in table_ranges
        ):
            continue
        filtered.append(block)
    return filtered


def parse_source_paragraphs(
    source_root: Path,
    structural_blocks: Sequence[SourceBlock],
    executed_sources: Sequence[Path],
) -> list[SourceParagraph]:
    """Parse ordinary TeX paragraphs and list items from compiled inputs.

    The parser is deliberately structural rather than a TeX renderer.  It
    recognizes TeX paragraph boundaries (blank lines/``\\par``), list-item
    boundaries, and protected structural environments.  Later SyncTeX glyph
    evidence decides whether a PDF line may receive one of these IDs.
    """

    protected: dict[Path, set[int]] = collections.defaultdict(set)
    for block in structural_blocks:
        protected[block.source_file.resolve()].update(
            range(block.start_line, block.end_line + 1)
        )

    paragraphs: list[SourceParagraph] = []
    root = source_root.resolve()
    for source_file in sorted({path.resolve() for path in executed_sources}):
        try:
            source_file.relative_to(root)
        except ValueError:
            continue
        if source_file.suffix.casefold() != ".tex" or not source_file.is_file():
            continue
        try:
            raw_lines = source_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        lines = [strip_tex_comment(line) for line in raw_lines]
        has_document_boundary = any(
            re.search(r"\\begin\s*\{document\}", line) for line in lines
        )
        inside_document = not has_document_boundary
        ignored_stack: list[str] = []
        list_stack: list[dict[str, Any]] = []
        buffer: list[tuple[int, str]] = []
        buffer_kind = "paragraph"
        buffer_list_environment: str | None = None
        buffer_item_depth: int | None = None
        buffer_item_ordinal: int | None = None

        def active_item() -> tuple[str, int, int] | None:
            for depth_index in range(len(list_stack) - 1, -1, -1):
                value = list_stack[depth_index]
                ordinal = value.get("active_ordinal")
                if isinstance(ordinal, int):
                    return str(value["environment"]), depth_index + 1, ordinal
            return None

        def begin_buffer_if_needed() -> None:
            nonlocal buffer_kind, buffer_list_environment
            nonlocal buffer_item_depth, buffer_item_ordinal
            if buffer:
                return
            item = active_item()
            if item is None:
                buffer_kind = "paragraph"
                buffer_list_environment = None
                buffer_item_depth = None
                buffer_item_ordinal = None
            else:
                environment, depth, ordinal = item
                buffer_kind = f"{environment}_item"
                buffer_list_environment = environment
                buffer_item_depth = depth
                buffer_item_ordinal = ordinal

        def flush_buffer() -> None:
            nonlocal buffer
            if not buffer:
                return
            selected = list(buffer)
            buffer = []
            raw = "\n".join(value for _, value in selected).strip()
            if not raw:
                return
            # Control-only lines have no visible paragraph to annotate.  Math
            # and structural environments have already been protected above.
            probe = re.sub(
                r"^\s*\\item(?:\s*\[[^\]]*\])?\s*", "", raw, count=1
            )
            if not re.search(r"[A-Za-z0-9]", latex_to_plain(probe)):
                return
            line_numbers = list(dict.fromkeys(number for number, _ in selected))
            try:
                relative = source_file.relative_to(root).as_posix()
            except ValueError:
                relative = str(source_file)
            identity = json.dumps(
                {
                    "source": relative,
                    "kind": buffer_kind,
                    "lines": line_numbers,
                    "list_environment": buffer_list_environment,
                    "item_depth": buffer_item_depth,
                    "item_ordinal": buffer_item_ordinal,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            paragraphs.append(
                SourceParagraph(
                    paragraph_id=f"sp-{digest}",
                    kind=buffer_kind,
                    source_file=source_file,
                    source_lines=line_numbers,
                    raw_latex=raw,
                    list_environment=buffer_list_environment,
                    item_depth=buffer_item_depth,
                    item_ordinal=buffer_item_ordinal,
                )
            )

        def append_fragment(line_number: int, value: str) -> None:
            if not value.strip():
                return
            begin_buffer_if_needed()
            buffer.append((line_number, value))

        for line_number, original_line in enumerate(lines, start=1):
            line = original_line
            if re.search(r"\\begin\s*\{document\}", line):
                flush_buffer()
                inside_document = True
                continue
            if re.search(r"\\end\s*\{document\}", line):
                flush_buffer()
                inside_document = False
                continue
            if not inside_document:
                flush_buffer()
                continue
            if line_number in protected.get(source_file, set()):
                trailing = runin_heading_trailing_prose(line)
                flush_buffer()
                if not trailing:
                    continue
                line = trailing

            environment_tokens = list(
                re.finditer(r"\\(begin|end)\s*\{([^{}]+)\}", line)
            )
            begins_ignored = [
                match.group(2)
                for match in environment_tokens
                if match.group(1) == "begin"
                and match.group(2) in SOURCE_PARAGRAPH_IGNORED_ENVS
            ]
            ends_ignored = [
                match.group(2)
                for match in environment_tokens
                if match.group(1) == "end"
                and match.group(2) in SOURCE_PARAGRAPH_IGNORED_ENVS
            ]
            if ignored_stack or begins_ignored:
                flush_buffer()
                ignored_stack.extend(begins_ignored)
                for environment in ends_ignored:
                    if environment in ignored_stack:
                        reverse_index = len(ignored_stack) - 1 - ignored_stack[::-1].index(
                            environment
                        )
                        del ignored_stack[reverse_index]
                continue

            list_begins = [
                match.group(2)
                for match in environment_tokens
                if match.group(1) == "begin"
                and match.group(2) in SOURCE_PARAGRAPH_LIST_ENVS
            ]
            list_ends = [
                match.group(2)
                for match in environment_tokens
                if match.group(1) == "end"
                and match.group(2) in SOURCE_PARAGRAPH_LIST_ENVS
            ]
            if list_begins:
                flush_buffer()
                for environment in list_begins:
                    list_stack.append(
                        {
                            "environment": environment,
                            "next_ordinal": 0,
                            "active_ordinal": None,
                        }
                    )
                line = re.sub(
                    r"\\begin\s*\{(?:itemize|enumerate|description)\}", "", line
                )
            if list_ends:
                # Content before an end marker belongs to the active item.
                prefix = re.split(
                    r"\\end\s*\{(?:itemize|enumerate|description)\}",
                    line,
                    maxsplit=1,
                )[0]
                append_fragment(line_number, prefix)
                flush_buffer()
                for environment in list_ends:
                    for stack_index in range(len(list_stack) - 1, -1, -1):
                        if list_stack[stack_index]["environment"] == environment:
                            del list_stack[stack_index:]
                            break
                continue

            item_match = re.search(r"\\item(?:\s*\[[^\]]*\])?", line)
            if item_match:
                flush_buffer()
                if not list_stack:
                    # A custom list implementation can expose ``\\item``
                    # without a recognized environment.  Keep it isolated.
                    list_stack.append(
                        {
                            "environment": "itemize",
                            "next_ordinal": 0,
                            "active_ordinal": None,
                        }
                    )
                list_stack[-1]["next_ordinal"] += 1
                list_stack[-1]["active_ordinal"] = list_stack[-1]["next_ordinal"]
                append_fragment(line_number, line)
                continue

            # Other environment boundaries delimit prose but do not become
            # part of the visible paragraph identity.
            if environment_tokens:
                stripped = re.sub(r"\\(?:begin|end)\s*\{[^{}]+\}", "", line)
                if stripped != line:
                    flush_buffer()
                    line = stripped
            if not line.strip():
                flush_buffer()
                continue
            if INLINE_NONPROSE_COMMAND.search(line):
                flush_buffer()
                continue
            pieces = re.split(r"\\par\b", line)
            for piece_index, piece in enumerate(pieces):
                append_fragment(line_number, piece)
                if piece_index + 1 < len(pieces):
                    flush_buffer()
        flush_buffer()
    return paragraphs


def split_tex_prose_units(value: str) -> list[tuple[str, int, int]]:
    """Split prose into sentence-sized units without cutting TeX groups/math.

    Whole source paragraphs frequently cross PDF page boundaries.  Inline
    enrichment therefore uses smaller units, while balanced braces and math
    delimiters keep citations, footnotes, styled spans, and formulas intact.
    Returned offsets refer to ``value`` and include sentence punctuation.
    """

    units: list[tuple[str, int, int]] = []
    start = 0
    brace_depth = 0
    dollar_math = False
    paren_math = False
    index = 0
    while index < len(value):
        if value.startswith(r"\(", index) and not dollar_math:
            paren_math = True
            index += 2
            continue
        if value.startswith(r"\)", index) and paren_math:
            paren_math = False
            index += 2
            continue
        character = value[index]
        if character == "\\":
            # Skip a control symbol or control-word name.  Its arguments are
            # still scanned normally so brace depth remains accurate.
            index += 1
            if index < len(value) and (value[index].isalpha() or value[index] == "@"):
                while index < len(value) and (value[index].isalpha() or value[index] == "@"):
                    index += 1
            elif index < len(value):
                index += 1
            continue
        if character == "$" and not paren_math:
            if index + 1 < len(value) and value[index + 1] == "$":
                # A display delimiter is not prose.  Keep it in one unit so
                # the inline parser can reject it explicitly and audibly.
                dollar_math = not dollar_math
                index += 2
                continue
            dollar_math = not dollar_math
            index += 1
            continue
        if not dollar_math and not paren_math:
            if character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth = max(0, brace_depth - 1)
            elif character in ".?!" and brace_depth == 0:
                next_index = index + 1
                if next_index == len(value) or value[next_index].isspace():
                    fragment = value[start:next_index]
                    if fragment.strip():
                        units.append((fragment, start, next_index))
                    start = next_index
        index += 1
    if value[start:].strip():
        units.append((value[start:], start, len(value)))
    return units


def footnote_specs_from_plan(plan: InlinePlan, block_id: str) -> list[FootnoteSpec]:
    """Extract source footnote bodies without rendering source-only text."""

    specs: list[FootnoteSpec] = []
    for node in iter_inline_nodes(plan.root):
        source = extract_footnote_source(node)
        if source is None:
            continue
        specs.append(
            FootnoteSpec(
                note_id=f"{block_id}-fn{len(specs) + 1:02d}",
                node_id=node.node_id,
                source_raw=source.raw,
                body_raw=source.body_raw,
                optional_arguments=list(source.optional_arguments),
            )
        )
    return specs


def parse_inline_source_blocks(
    source_root: Path,
    structural_blocks: Sequence[SourceBlock],
    math_macros: dict[str, SimpleMathMacro] | None = None,
) -> tuple[list[InlineSourceBlock], list[dict[str, Any]]]:
    """Extract prose fragments that contain inline math or text styling.

    Pure prose remains sourced from the PDF.  These blocks are only formatting
    plans: later alignment keeps the PDF's rendered citations, references, and
    footnote marks while restoring supported inline LaTeX and Markdown style.
    """

    protected: dict[Path, list[tuple[int, int]]] = collections.defaultdict(list)
    for block in structural_blocks:
        protected[block.source_file.resolve()].append((block.start_line, block.end_line))

    blocks: list[InlineSourceBlock] = []
    rejections: list[dict[str, Any]] = []
    serial = 0
    for source_file in sorted(source_root.rglob("*.tex")):
        resolved_source = source_file.resolve()
        try:
            raw_lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            rejections.append({"source_file": str(source_file), "reason": f"read_error: {exc}"})
            continue
        lines = [strip_tex_comment(line) for line in raw_lines]
        ranges = list(protected.get(resolved_source, []))
        # Author blocks are metadata/front matter, not prose.  Their inline
        # math (for example ``Name$^{1}$``) is handled by the dedicated author
        # superscript pass.  Protect the complete balanced command argument so
        # continuation lines cannot later be restored as ``$^{...}$`` and
        # wrap the generated ``<sup>`` tag a second time.
        source_text = "\n".join(lines)
        author_cursor = 0
        author_command = re.compile(
            r"\\(?:author|affil)(?P<starred>\*)?"
            r"(?:\s*\[(?P<optional>[^\]]*)\])?\s*"
        )
        while True:
            author_match = author_command.search(source_text, author_cursor)
            if author_match is None or author_match.end() >= len(source_text):
                break
            if source_text[author_match.end()] != "{":
                author_cursor = author_match.end()
                continue
            author_argument = extract_balanced(source_text, author_match.end())
            if author_argument is None:
                break
            _, author_end = author_argument
            ranges.append(
                (
                    source_text.count("\n", 0, author_match.start()) + 1,
                    source_text.count("\n", 0, author_end) + 1,
                )
            )
            author_cursor = author_end
        has_document_boundary = any(re.search(r"\\begin\s*\{document\}", line) for line in lines)
        inside_document = not has_document_boundary
        ignored_stack: list[str] = []
        buffer: list[tuple[int, str]] = []

        def line_is_protected(line_number: int) -> bool:
            return any(start <= line_number <= end for start, end in ranges)

        def flush_buffer() -> None:
            nonlocal serial, buffer
            if not buffer:
                return
            buffer_start_line = buffer[0][0]
            raw_paragraph = "\n".join(value for _, value in buffer)
            selected_lines = list(buffer)
            buffer = []
            if not raw_paragraph.strip() or not INLINE_TARGET_PATTERN.search(raw_paragraph):
                return
            for raw_unit, unit_start, unit_end in split_tex_prose_units(raw_paragraph):
                raw = raw_unit.strip()
                if not raw or not INLINE_TARGET_PATTERN.search(raw):
                    continue
                start_line = buffer_start_line + raw_paragraph[:unit_start].count("\n")
                end_line = buffer_start_line + raw_paragraph[:unit_end].count("\n")
                if raw_unit.endswith("\n") and end_line > start_line:
                    end_line -= 1
                if len(raw) > 12000 or end_line - start_line > 80:
                    rejections.append(
                        {
                            "source_file": str(resolved_source),
                            "source_lines": [start_line, end_line],
                            "reason": "inline_prose_fragment_too_large",
                        }
                    )
                    continue
                try:
                    full_plan = expand_inline_plan_math(
                        parse_inline_plan(raw), math_macros or {}
                    )
                    # A footnote is a target in its own right, and its body is
                    # intentionally opaque to the parent prose alignment.  Do
                    # not let focus_inline_plan trim the opaque footnote node
                    # from a sentence whose other target occurs earlier.
                    contains_footnote = any(
                        node.kind == "opaque" and node.opaque_role == "footnote"
                        for node in iter_inline_nodes(full_plan.root)
                    )
                    plan = full_plan if contains_footnote else focus_inline_plan(full_plan)
                except InlineParseError as exc:
                    rejections.append(
                        {
                            "source_file": str(resolved_source),
                            "source_lines": [start_line, end_line],
                            "reason": f"inline_parse_error: {exc}",
                            "raw_latex": raw,
                        }
                    )
                    continue
                target_total = sum(
                    int(plan.feature_counts.get(key, 0))
                    for key in ("math", "strong", "em", "code", "footnote")
                )
                if target_total == 0:
                    continue
                target_lines = [
                    start_line + line_offset
                    for line_offset, value in enumerate(raw.splitlines())
                    if INLINE_TARGET_PATTERN.search(value)
                ]
                query_lines = sorted(
                    {
                        start_line,
                        end_line,
                        target_lines[0] if target_lines else start_line,
                        target_lines[-1] if target_lines else end_line,
                    }
                )
                serial += 1
                block_id = f"i{serial:06d}"
                blocks.append(
                    InlineSourceBlock(
                        block_id=block_id,
                        source_file=resolved_source,
                        start_line=start_line,
                        end_line=end_line,
                        raw_latex=raw,
                        plan=plan,
                        query_lines=query_lines,
                        footnotes=footnote_specs_from_plan(plan, block_id),
                    )
                )

        for index, line in enumerate(lines, start=1):
            if re.search(r"\\begin\s*\{document\}", line):
                flush_buffer()
                inside_document = True
                continue
            if re.search(r"\\end\s*\{document\}", line):
                flush_buffer()
                inside_document = False
                continue
            environment_tokens = list(re.finditer(r"\\(begin|end)\s*\{([^{}]+)\}", line))
            begins_ignored = [
                match.group(2)
                for match in environment_tokens
                if match.group(1) == "begin" and match.group(2) in INLINE_IGNORED_ENVS
            ]
            ends_ignored = [
                match.group(2)
                for match in environment_tokens
                if match.group(1) == "end" and match.group(2) in INLINE_IGNORED_ENVS
            ]
            if ignored_stack or begins_ignored or ends_ignored:
                flush_buffer()
                for environment in begins_ignored:
                    ignored_stack.append(environment)
                for environment in ends_ignored:
                    if environment in ignored_stack:
                        reverse_index = len(ignored_stack) - 1 - ignored_stack[::-1].index(environment)
                        del ignored_stack[reverse_index]
                continue
            if not inside_document:
                flush_buffer()
                continue
            if line_is_protected(index):
                trailing = runin_heading_trailing_prose(line)
                flush_buffer()
                if not trailing:
                    continue
                line = trailing
            if not line.strip() or INLINE_NONPROSE_COMMAND.search(line):
                flush_buffer()
                continue
            buffer.append((index, line))
        flush_buffer()
    return blocks, rejections


def run_with_heartbeat(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: float,
    label: str,
) -> tuple[int, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("wb") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        last_report = started
        while process.poll() is None:
            now = time.monotonic()
            if now - started > timeout_seconds:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            if now - last_report >= HEARTBEAT_SECONDS:
                print(f"[progress] {label} elapsed={elapsed_string(now - started)}", flush=True)
                last_report = now
            time.sleep(0.25)
        return_code = process.wait()
    return return_code, timed_out, time.monotonic() - started


def tool_env() -> dict[str, str]:
    env = dict(os.environ)
    path_items = [str(TEX_BIN), "/opt/homebrew/bin", "/usr/bin", "/bin"]
    existing = env.get("PATH", "")
    if existing:
        path_items.append(existing)
    env["PATH"] = os.pathsep.join(path_items)
    return env


def natbib_retry_reason(log_text: str) -> str | None:
    """Return the narrowly allowed reason for a numerical-natbib retry.

    Some arXiv sources use numerical ``thebibliography`` entries while loading
    natbib in author-year mode. XeLaTeX can finish all pages and write both the
    XDV and SyncTeX files before the final AUX read reports that incompatibility.
    Only this known condition may trigger a generated wrapper that explicitly
    selects natbib's numerical mode; arbitrary compile errors remain fatal.
    """

    compact = re.sub(r"\s+", "", log_text)
    if (
        "PackagenatbibError:Bibliographynotcompatiblewithauthor-yearcitations." in compact
        and "Outputwrittenon" in compact
        and ".xdv" in compact
        and "SyncTeXwrittenon" in compact
    ):
        return "natbib_author_year_aux_error_after_complete_xdv"
    return None


def pdf_text_sha256(pdf_path: Path) -> str:
    completed = subprocess.run(
        [str(PDFTOTEXT), str(pdf_path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"pdftotext failed for {pdf_path}: rc={completed.returncode} {detail}")
    return hashlib.sha256(completed.stdout).hexdigest()


def compile_with_synctex(paper: dict[str, Any], build_dir: Path, resume: bool) -> tuple[Path, Path, dict[str, Any]]:
    source_root = Path(paper["source_dir"]).resolve()
    main_tex = Path(paper["main_tex"]).resolve()
    build_dir = build_dir.resolve()
    stem = main_tex.stem
    pdf_path = build_dir / f"{stem}.pdf"
    synctex_path = build_dir / f"{stem}.synctex.gz"
    if resume and pdf_path.is_file() and pdf_path.stat().st_size > 0 and synctex_path.is_file() and synctex_path.stat().st_size > 0:
        stored_compile: dict[str, Any] | None = None
        compile_info_path = build_dir / "compile_info.json"
        if compile_info_path.is_file():
            try:
                stored_compile = json.loads(compile_info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored_compile = None
        if stored_compile is None:
            prior_summary_path = build_dir.parent / "paper_summary.json"
            if prior_summary_path.is_file():
                try:
                    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
                    if isinstance(prior_summary.get("compile"), dict):
                        stored_compile = prior_summary["compile"]
                except (OSError, json.JSONDecodeError):
                    stored_compile = None
        # A failed TeX process can leave non-empty but truncated PDF/SyncTeX
        # files behind.  Never treat mere existence as a resumable compile:
        # require successful provenance and parse both artifacts before reuse.
        accepted_statuses = {"compiled", "compiled_natbib_numbers_retry"}
        cache_is_valid = bool(
            stored_compile and stored_compile.get("status") in accepted_statuses
        )
        cached_pages = 0
        if cache_is_valid:
            try:
                with pdfplumber.open(pdf_path) as cached_document:
                    cached_pages = len(cached_document.pages)
                with gzip.open(synctex_path, "rb") as synctex_stream:
                    synctex_stream.read(1)
            except Exception:
                cache_is_valid = False
        expected_pages = int(paper.get("pdf_inspection", {}).get("pages", 0) or 0)
        if cache_is_valid and expected_pages and cached_pages != expected_pages:
            cache_is_valid = False
        if cache_is_valid:
            compile_info = dict(stored_compile)
            compile_info["resume_status"] = "reused_validated"
            compile_info["validated_pages"] = cached_pages
            atomic_write_json(compile_info_path, compile_info)
            return pdf_path, synctex_path, compile_info
    build_dir.mkdir(parents=True, exist_ok=True)
    for child in build_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    engine = paper.get("compile", {}).get("engine", "pdflatex")
    if engine == "xelatex":
        engine_flag = "-xelatex"
    elif engine == "latex_dvips_ps2pdf":
        engine_flag = "-pdfps"
    else:
        engine_flag = "-pdf"
    latexmk = TEX_BIN / "latexmk"
    command = [
        str(latexmk),
        "-norc",
        "-g",
        engine_flag,
        "-synctex=1",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-no-shell-escape",
        f"-outdir={build_dir}",
        main_tex.name,
    ]
    log_path = build_dir / "compile_synctex.log"
    return_code, timed_out, duration = run_with_heartbeat(
        command,
        # Match the successful upstream recompilation working directory.
        # arXiv archives frequently place their main file in a nested folder
        # (for example ``source/Arxiv/main.tex``); invoking only
        # ``main_tex.name`` from the extraction root makes latexmk report a
        # false "Could not find file 'main.tex'" failure.
        cwd=main_tex.parent,
        env=tool_env(),
        log_path=log_path,
        timeout_seconds=300,
        label=f"compile {paper['stem']}",
    )
    retry_info: dict[str, Any] | None = None
    compile_failed = return_code != 0 or timed_out or not pdf_path.is_file() or not synctex_path.is_file()
    if compile_failed and engine == "xelatex" and not timed_out:
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        reason = natbib_retry_reason(log_text)
        if reason:
            wrapper_path = build_dir / "_natbib_numbers_wrapper.tex"
            atomic_write_text(
                wrapper_path,
                "% Auto-generated compatibility wrapper; the arXiv source is not modified.\n"
                "\\PassOptionsToPackage{numbers}{natbib}\n"
                f"\\input{{{main_tex.as_posix()}}}\n",
            )
            retry_log = build_dir / "compile_synctex_natbib_numbers_retry.log"
            retry_command = [
                str(latexmk),
                "-norc",
                "-g",
                "-xelatex",
                f"-jobname={stem}",
                "-synctex=1",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "-no-shell-escape",
                f"-outdir={build_dir}",
                str(wrapper_path),
            ]
            retry_rc, retry_timeout, retry_duration = run_with_heartbeat(
                retry_command,
                cwd=main_tex.parent,
                env=tool_env(),
                log_path=retry_log,
                timeout_seconds=300,
                label=f"natbib numerical retry {paper['stem']}",
            )
            retry_pages = 0
            if retry_rc == 0 and not retry_timeout and pdf_path.is_file() and pdf_path.stat().st_size > 0:
                with pdfplumber.open(pdf_path) as retry_document:
                    retry_pages = len(retry_document.pages)
            expected_pages = int(paper.get("pdf_inspection", {}).get("pages", 0) or 0)
            reference_pdf_value = paper.get("pdf_inspection", {}).get("pdf") or paper.get("compile", {}).get("pdf")
            reference_pdf = Path(reference_pdf_value).resolve() if reference_pdf_value else None
            reference_text_sha256 = (
                pdf_text_sha256(reference_pdf)
                if reference_pdf is not None and reference_pdf.is_file() and reference_pdf.stat().st_size > 0
                else None
            )
            retry_text_sha256 = pdf_text_sha256(pdf_path) if retry_pages else None
            page_count_matches = retry_pages > 0 and (expected_pages == 0 or retry_pages == expected_pages)
            text_matches_reference = bool(reference_text_sha256) and retry_text_sha256 == reference_text_sha256
            if (
                retry_rc == 0
                and not retry_timeout
                and synctex_path.is_file()
                and synctex_path.stat().st_size > 0
                and page_count_matches
                and text_matches_reference
            ):
                compile_failed = False
                retry_info = {
                    "status": "accepted",
                    "reason": reason,
                    "command": retry_command,
                    "return_code": retry_rc,
                    "timed_out": retry_timeout,
                    "duration_seconds": round(retry_duration, 3),
                    "log": str(retry_log),
                    "wrapper": str(wrapper_path),
                    "pages": retry_pages,
                    "expected_pages": expected_pages or None,
                    "pdf_text_sha256": retry_text_sha256,
                    "reference_pdf": str(reference_pdf),
                    "reference_pdf_text_sha256": reference_text_sha256,
                    "text_matches_reference": text_matches_reference,
                }
    if compile_failed:
        raise RuntimeError(
            f"SyncTeX compile failed for {paper['stem']}: rc={return_code} timeout={timed_out}; log={log_path}"
        )
    compile_info = {
        "status": "compiled_natbib_numbers_retry" if retry_info else "compiled",
        "engine": engine,
        "command": command,
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_seconds": round(duration, 3),
        "log": str(log_path),
        "pdf": str(pdf_path),
        "synctex": str(synctex_path),
        "natbib_numbers_retry": retry_info,
    }
    atomic_write_json(build_dir / "compile_info.json", compile_info)
    return pdf_path, synctex_path, compile_info


def synctex_inputs(synctex_path: Path) -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    with gzip.open(synctex_path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = re.match(r"Input:\d+:(.*)", line.rstrip("\n"))
            if not match:
                continue
            exact = match.group(1)
            normalized = Path(exact.replace("/./", "/")).resolve()
            mapping[normalized] = exact
    return mapping


def compiled_tex_sources(source_root: Path, synctex_path: Path) -> list[Path]:
    """Return only TeX files that the successful compilation actually read."""

    root = source_root.resolve()
    values: list[Path] = []
    for source_file in synctex_inputs(synctex_path):
        try:
            source_file.relative_to(root)
        except ValueError:
            continue
        if source_file.suffix.casefold() == ".tex" and source_file.is_file():
            values.append(source_file)
    return sorted(set(values))


def parse_synctex_paragraph_points(
    synctex_path: Path,
    paragraphs: Sequence[SourceParagraph],
) -> dict[int, list[SourceParagraphPoint]]:
    """Read glyph-level SyncTeX provenance without invoking one CLI per line.

    ``x`` records correspond to shipped text positions.  Container boxes are
    intentionally ignored because their source line often belongs to the next
    list item or environment boundary and would blur paragraph ownership.
    Coordinates are converted from TeX scaled points to PDF big points.
    """

    line_candidates: dict[tuple[Path, int], set[str]] = collections.defaultdict(set)
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    for paragraph in paragraphs:
        for line_number in paragraph.source_lines:
            line_candidates[(paragraph.source_file.resolve(), line_number)].add(
                paragraph.paragraph_id
            )
    unique_line_owner = {
        key: next(iter(values))
        for key, values in line_candidates.items()
        if len(values) == 1
    }
    if not unique_line_owner:
        return {}

    tagged_inputs: dict[int, Path] = {}
    points: dict[int, list[SourceParagraphPoint]] = collections.defaultdict(list)
    magnification = 1000.0
    unit = 1.0
    x_offset = 0.0
    y_offset = 0.0
    page: int | None = None
    in_content = False
    scanned = 0
    accepted = 0
    started = time.monotonic()
    last_progress = started
    input_pattern = re.compile(r"Input:(\d+):(.*)")
    glyph_pattern = re.compile(
        r"^x(?P<tag>\d+),(?P<line>\d+):(?P<x>-?\d+),(?P<y>-?\d+)"
    )
    with gzip.open(synctex_path, "rt", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            scanned += 1
            line = raw_line.rstrip("\n")
            now = time.monotonic()
            if now - last_progress >= HEARTBEAT_SECONDS:
                print(
                    f"[progress] phase=synctex-paragraph-points records={scanned} "
                    f"accepted={accepted} pages={len(points)} "
                    f"elapsed={elapsed_string(now-started)}",
                    flush=True,
                )
                last_progress = now
            if match := input_pattern.fullmatch(line):
                # Included inputs may first be declared after ``Content:``.
                # Keep accepting Input records throughout the stream so their
                # glyph records can be tied back to real source paragraphs.
                tagged_inputs[int(match.group(1))] = Path(
                    match.group(2).replace("/./", "/")
                ).resolve()
                continue
            if not in_content:
                if line.startswith("Content:"):
                    in_content = True
                    continue
                if line.startswith("Magnification:"):
                    magnification = float(line.split(":", 1)[1] or 1000)
                elif line.startswith("Unit:"):
                    unit = float(line.split(":", 1)[1] or 1)
                elif line.startswith("X Offset:"):
                    x_offset = float(line.split(":", 1)[1] or 0)
                elif line.startswith("Y Offset:"):
                    y_offset = float(line.split(":", 1)[1] or 0)
                continue
            if match := re.fullmatch(r"\{(\d+)", line):
                page = int(match.group(1))
                continue
            if line == "}":
                page = None
                continue
            if page is None or not (match := glyph_pattern.match(line)):
                continue
            source_file = tagged_inputs.get(int(match.group("tag")))
            source_line = int(match.group("line"))
            if source_file is None:
                continue
            paragraph_id = unique_line_owner.get((source_file, source_line))
            if paragraph_id is None:
                continue
            # TeX uses scaled points and PDF uses big points (72 bp/in).
            factor = unit * (magnification / 1000.0) * (72.0 / 72.27) / 65536.0
            x = (float(match.group("x")) + x_offset) * factor
            y = (float(match.group("y")) + y_offset) * factor
            paragraph = paragraph_by_id[paragraph_id]
            points[page].append(
                SourceParagraphPoint(
                    page=page,
                    x=x,
                    y=y,
                    paragraph_id=paragraph_id,
                    source_file=paragraph.source_file,
                    source_line=source_line,
                )
            )
            accepted += 1
    for values in points.values():
        values.sort(key=lambda point: (point.y, point.x, point.paragraph_id))
    print(
        f"[done] phase=synctex-paragraph-points records={scanned} "
        f"accepted={accepted} pages={len(points)} "
        f"elapsed={elapsed_string(time.monotonic()-started)}",
        flush=True,
    )
    return dict(points)


def parse_synctex_output(value: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in value.splitlines():
        if line.startswith("Output:") and current:
            if "Page" in current:
                records.append(current)
            current = {}
        match = re.match(r"(Page|x|y|h|v|W|H):(.+)", line)
        if not match:
            continue
        key, raw = match.groups()
        try:
            current[key] = int(raw) if key == "Page" else float(raw)
        except ValueError:
            continue
    if current and "Page" in current:
        records.append(current)
    for record in records:
        if all(key in record for key in ("h", "v", "W", "H")):
            x0 = record["h"]
            y1 = record["v"]
            record["bbox"] = [x0, y1 - record["H"], x0 + record["W"], y1]
    return records


def valid_bbox(bbox: Sequence[float]) -> bool:
    return (
        len(bbox) == 4
        and all(math.isfinite(value) for value in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def union_bboxes(bboxes: Sequence[Sequence[float]]) -> list[float]:
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def select_display_math_bbox(
    page_candidates: Sequence[dict[str, Any]],
) -> list[float] | None:
    reasonable = [
        candidate
        for candidate in page_candidates
        if 2 <= float(candidate.get("H", 0)) <= 100
        and 1 < float(candidate.get("W", 0)) <= 1000
    ]
    bbox = union_bboxes(
        [candidate["bbox"] for candidate in (reasonable or page_candidates)]
    )
    return None if bbox[3] - bbox[1] > 350 else bbox


def map_source_blocks(
    blocks: list[SourceBlock],
    source_root: Path,
    pdf_path: Path,
    synctex_path: Path,
    page_count: int,
) -> None:
    inputs = synctex_inputs(synctex_path)
    executable = TEX_BIN / "synctex"
    for block_index, block in enumerate(blocks, start=1):
        exact_source = inputs.get(block.source_file.resolve())
        if exact_source is None:
            # SyncTeX often records main source as /source/./main.tex.
            matches = [value for path, value in inputs.items() if path.name == block.source_file.name]
            exact_source = matches[0] if len(matches) == 1 else str(block.source_file)
        candidates: list[dict[str, Any]] = []
        for line_number in block.query_lines:
            spec = f"{line_number}:1:{exact_source}"
            completed = subprocess.run(
                [str(executable), "view", "-i", spec, "-o", str(pdf_path)],
                cwd=source_root,
                env=tool_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
                check=False,
            )
            candidates.extend(parse_synctex_output(completed.stdout))
        candidates = [
            candidate
            for candidate in candidates
            if 1 <= candidate.get("Page", 0) <= page_count
            and valid_bbox(candidate.get("bbox", []))
            and candidate.get("W", 0) > 1
            and candidate.get("H", 0) > 1
        ]
        block.mapping_candidates = len(candidates)
        if not candidates:
            block.mapping_status = "unmapped"
            continue
        page_histogram = collections.Counter(candidate["Page"] for candidate in candidates)
        page = page_histogram.most_common(1)[0][0]
        page_candidates = [candidate for candidate in candidates if candidate["Page"] == page]
        if block.kind == "heading":
            reasonable = [candidate for candidate in page_candidates if 2 <= candidate.get("H", 0) <= 80]
            best = max(reasonable or page_candidates, key=lambda item: bbox_area(item["bbox"]))
            bbox = list(best["bbox"])
        elif block.kind == "display_math":
            # A SyncTeX query on the final line of a multiline align/gather
            # environment returns one box per shipped row.  Selecting only the
            # largest box leaves the remaining PDF glyph fragments behind and
            # duplicates the formula after source restoration.  Union all
            # reasonable row boxes on the selected page instead.
            selected_display_bbox = select_display_math_bbox(page_candidates)
            if selected_display_bbox is None:
                block.mapping_status = "ambiguous_display_bbox"
                continue
            bbox = selected_display_bbox
        elif block.kind == "table" and len(block.query_lines) > 1:
            selected_by_line = sorted(page_candidates, key=lambda item: bbox_area(item["bbox"]), reverse=True)[:2]
            bbox = union_bboxes([item["bbox"] for item in selected_by_line])
        else:
            best = max(page_candidates, key=lambda item: bbox_area(item["bbox"]))
            bbox = list(best["bbox"])
        block.page = int(page)
        block.bbox = [round(float(value), 3) for value in bbox]
        block.mapping_status = "mapped"
        if block_index % 50 == 0:
            print(f"[progress] mapped_source_blocks={block_index}/{len(blocks)}", flush=True)


def map_inline_source_blocks(
    blocks: list[InlineSourceBlock],
    source_root: Path,
    pdf_path: Path,
    synctex_path: Path,
    page_count: int,
) -> None:
    inputs = synctex_inputs(synctex_path)
    executable = TEX_BIN / "synctex"
    for block_index, block in enumerate(blocks, start=1):
        exact_source = inputs.get(block.source_file.resolve())
        if exact_source is None:
            matches = [value for path, value in inputs.items() if path.name == block.source_file.name]
            exact_source = matches[0] if len(matches) == 1 else str(block.source_file)
        candidates: list[dict[str, Any]] = []
        for line_number in block.query_lines:
            completed = subprocess.run(
                [
                    str(executable),
                    "view",
                    "-i",
                    f"{line_number}:1:{exact_source}",
                    "-o",
                    str(pdf_path),
                ],
                cwd=source_root,
                env=tool_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
                check=False,
            )
            candidates.extend(parse_synctex_output(completed.stdout))
        candidates = [
            candidate
            for candidate in candidates
            if 1 <= candidate.get("Page", 0) <= page_count
            and valid_bbox(candidate.get("bbox", []))
            and candidate.get("W", 0) > 1
            and candidate.get("H", 0) > 1
        ]
        block.mapping_candidates = len(candidates)
        if not candidates:
            block.mapping_status = "unmapped"
            continue
        page_histogram = collections.Counter(int(candidate["Page"]) for candidate in candidates)
        block.candidate_pages = sorted(page_histogram)
        ranked_pages = page_histogram.most_common()
        if len(ranked_pages) > 1 and ranked_pages[0][1] == ranked_pages[1][1]:
            block.mapping_status = "ambiguous_page"
            continue
        page = ranked_pages[0][0]
        page_candidates = [candidate for candidate in candidates if int(candidate["Page"]) == page]
        reasonable = [
            candidate
            for candidate in page_candidates
            if 2 <= float(candidate.get("H", 0)) <= 100
            and float(candidate.get("W", 0)) <= 0.98 * 1000
        ]
        best = max(reasonable or page_candidates, key=lambda item: bbox_area(item["bbox"]))
        block.page = page
        block.bbox = [round(float(value), 3) for value in best["bbox"]]
        block.mapping_status = "mapped"
        if block_index % 50 == 0:
            print(
                f"[progress] mapped_inline_blocks={block_index}/{len(blocks)}",
                flush=True,
            )


def disambiguate_inline_source_pages(
    blocks: list[InlineSourceBlock], document: Any
) -> None:
    """Resolve multi-page SyncTeX candidates by deterministic PDF alignment.

    A long TeX source line can legitimately map to several PDF pages.  SyncTeX
    alone cannot tell which sentence appears on which page, so sentence-sized
    inline plans are matched against each candidate page's text lines.  Ties
    remain unresolved instead of being guessed.
    """

    page_cache: dict[int, tuple[list[PageNode], str, float]] = {}

    def page_material(page_number: int) -> tuple[list[PageNode], str, float]:
        if page_number not in page_cache:
            page = document.pages[page_number - 1]
            nodes = words_to_line_nodes(page)
            _, layout = order_page_nodes(nodes, float(page.width))
            page_cache[page_number] = (nodes, layout, float(page.width))
        return page_cache[page_number]

    targets = [block for block in blocks if len(block.candidate_pages) > 1]
    for target_index, block in enumerate(targets, start=1):
        matches: list[InlineSourceBlock] = []
        for page_number in block.candidate_pages:
            base_nodes, layout, page_width = page_material(page_number)
            probe = dataclasses.replace(
                block,
                page=page_number,
                bbox=None,
                match_status="pending",
                match_reason=None,
                matched_line_count=0,
                matched_bbox=None,
                matched_pdf_text=None,
                enriched_markdown=None,
                match_score=None,
            )
            probe_nodes = [
                dataclasses.replace(node, inline_source_ids=list(node.inline_source_ids))
                for node in base_nodes
            ]
            apply_inline_source_blocks(
                probe_nodes,
                [probe],
                page_width,
                layout_hint=layout,
            )
            if probe.match_status == "matched":
                matches.append(probe)
        matches.sort(key=lambda item: float(item.match_score or 0), reverse=True)
        if not matches:
            block.page = None
            block.bbox = None
            block.mapping_status = "unresolved_multi_page_pdf_alignment"
        elif len(matches) > 1 and float(matches[0].match_score or 0) - float(
            matches[1].match_score or 0
        ) < 0.03:
            block.page = None
            block.bbox = None
            block.mapping_status = "ambiguous_multi_page_pdf_alignment"
        else:
            selected = matches[0]
            block.page = selected.page
            block.bbox = selected.matched_bbox
            block.mapping_status = "mapped_pdf_disambiguated"
        if target_index % 25 == 0:
            print(
                f"[progress] disambiguated_inline_blocks={target_index}/{len(targets)}",
                flush=True,
            )


@dataclasses.dataclass
class PageNode:
    kind: str
    text: str
    bbox: list[float]
    font_size: float
    lane: str = ""
    source_block_id: str | None = None
    inline_source_ids: list[str] = dataclasses.field(default_factory=list)
    line_id: str | None = None
    origin_page: int | None = None
    origin_order: int | None = None
    claimed_line_ids: list[str] = dataclasses.field(default_factory=list)
    derived_line_ids: list[str] = dataclasses.field(default_factory=list)
    source_paragraph_id: str | None = None
    source_paragraph_slice_id: str | None = None

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def bottom(self) -> float:
        return self.bbox[3]


def words_to_line_nodes(page: Any) -> list[PageNode]:
    words = page.extract_words(
        x_tolerance=1.0,
        y_tolerance=3.0,
        keep_blank_chars=False,
        use_text_flow=False,
    )
    words = [word for word in words if normalize_space(str(word.get("text", "")))]
    clusters: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        target: list[dict[str, Any]] | None = None
        for cluster in reversed(clusters[-8:]):
            cluster_top = statistics.median(float(item["top"]) for item in cluster)
            if abs(float(word["top"]) - cluster_top) <= 4.2:
                target = cluster
                break
            if cluster_top < float(word["top"]) - 4:
                break
        if target is None:
            target = []
            clusters.append(target)
        target.append(word)
    nodes: list[PageNode] = []
    split_gap = max(18.0, float(page.width) * 0.035)
    midpoint = float(page.width) / 2
    gutter = float(page.width) * 0.012
    characters = list(page.chars)
    for cluster in clusters:
        segments: list[list[dict[str, Any]]] = [[]]
        previous_x1: float | None = None
        for word in sorted(cluster, key=lambda item: float(item["x0"])):
            gap = float(word["x0"]) - previous_x1 if previous_x1 is not None else 0
            crosses_midpoint = (
                previous_x1 is not None
                and previous_x1 <= midpoint + gutter
                and float(word["x0"]) >= midpoint - gutter
                and gap >= max(7.0, float(page.width) * 0.012)
            )
            if previous_x1 is not None and (gap > split_gap or crosses_midpoint):
                segments.append([])
            segments[-1].append(word)
            previous_x1 = float(word["x1"])
        for segment in segments:
            if not segment:
                continue
            text = normalize_space(" ".join(str(word["text"]) for word in segment))
            if not text:
                continue
            bbox = [
                min(float(word["x0"]) for word in segment),
                min(float(word["top"]) for word in segment),
                max(float(word["x1"]) for word in segment),
                max(float(word["bottom"]) for word in segment),
            ]
            overlapping_sizes = [
                float(character.get("size", 0) or 0)
                for character in characters
                if float(character.get("x1", 0)) > bbox[0]
                and float(character.get("x0", 0)) < bbox[2]
                and float(character.get("bottom", 0)) > bbox[1]
                and float(character.get("top", 0)) < bbox[3]
            ]
            font_size = statistics.median(overlapping_sizes) if overlapping_sizes else 0.0
            nodes.append(PageNode("text", text, bbox, font_size))
    page_number = int(getattr(page, "page_number", 1) or 1)
    occurrence_counts: collections.Counter[str] = collections.Counter()
    for extraction_order, node in enumerate(nodes):
        identity_payload = json.dumps(
            {
                "page": page_number,
                "bbox": [round(float(value), 3) for value in node.bbox],
                "text": unicodedata.normalize("NFKC", node.text),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:12]
        occurrence_counts[digest] += 1
        line_id = (
            f"p{page_number:04d}-x{extraction_order:04d}-{digest}-"
            f"o{occurrence_counts[digest]:02d}"
        )
        node.line_id = line_id
        node.origin_page = page_number
        node.origin_order = extraction_order
        node.claimed_line_ids = [line_id]
    return nodes


def annotate_source_paragraph_ids(
    nodes: Sequence[PageNode],
    points: Sequence[SourceParagraphPoint],
    paragraphs_by_id: dict[str, SourceParagraph],
) -> dict[str, Any]:
    """Attach one high-confidence source paragraph ID to each PDF text line."""

    sorted_points = sorted(points, key=lambda point: (point.y, point.x))
    y_values = [point.y for point in sorted_points]
    assignments: list[dict[str, Any]] = []
    mapped = ambiguous = 0
    for node in nodes:
        lower = bisect.bisect_left(y_values, node.top - 2.5)
        upper = bisect.bisect_right(y_values, node.bottom + 2.5)
        candidates = [
            point
            for point in sorted_points[lower:upper]
            if node.bbox[0] - 6.0 <= point.x <= node.bbox[2] + 6.0
        ]
        counts = collections.Counter(point.paragraph_id for point in candidates)
        ranked = counts.most_common()
        status = "unmapped"
        confidence = 0.0
        lexical_support = 0.0
        selected: str | None = None
        if ranked:
            selected, top_count = ranked[0]
            total = sum(counts.values())
            second_count = ranked[1][1] if len(ranked) > 1 else 0
            confidence = top_count / max(1, total)
            visible_tokens = collections.Counter(normalize_tokens(node.text))
            paragraph = paragraphs_by_id.get(selected)
            source_tokens = collections.Counter(
                normalize_tokens(latex_to_plain(paragraph.raw_latex))
                if paragraph is not None
                else []
            )
            lexical_matches = sum(
                min(count, source_tokens[token])
                for token, count in visible_tokens.items()
            )
            lexical_support = lexical_matches / max(1, sum(visible_tokens.values()))
            if (
                top_count >= 2
                and confidence >= 0.65
                and top_count - second_count >= 1
                and sum(visible_tokens.values()) >= 2
                and lexical_support >= 0.60
                and selected in paragraphs_by_id
            ):
                status = "mapped"
                node.source_paragraph_id = selected
                node.source_paragraph_slice_id = (
                    f"{selected}@p{int(node.origin_page or 0):04d}"
                )
                mapped += 1
            elif len(ranked) > 1:
                status = "ambiguous"
                selected = None
                ambiguous += 1
            else:
                selected = None
        assignments.append(
            {
                "line_id": node.line_id,
                "source_paragraph_id": selected,
                "source_paragraph_slice_id": node.source_paragraph_slice_id,
                "status": status,
                "confidence": round(confidence, 6),
                "lexical_support": round(lexical_support, 6),
                "candidate_counts": dict(sorted(counts.items())),
            }
        )
    edge_suffix_mapped = 0
    if nodes:
        first_line_top = min(node.top for node in nodes)
        for node, assignment in zip(nodes, assignments):
            if node.source_paragraph_id is not None:
                continue
            # SyncTeX can attach every glyph of a very long one-line TeX
            # paragraph to the previous physical page even when its final word
            # flows onto the next page. Recover only a unique source-paragraph
            # suffix at the top of the compiled page. This restores the
            # paragraph boundary without copying invisible text from page N-1.
            if node.top > first_line_top + 3.0:
                continue
            if not re.search(r"[.!?;:]\s*$", node.text):
                continue
            visible_tokens = normalize_tokens(node.text)
            if not visible_tokens or len(visible_tokens) > 12:
                continue
            suffix_matches: list[str] = []
            for paragraph_id, paragraph in paragraphs_by_id.items():
                source_tokens = normalize_tokens(latex_to_plain(paragraph.raw_latex))
                if (
                    len(source_tokens) >= len(visible_tokens)
                    and source_tokens[-len(visible_tokens) :] == visible_tokens
                ):
                    suffix_matches.append(paragraph_id)
            if len(suffix_matches) != 1:
                continue
            selected = suffix_matches[0]
            node.source_paragraph_id = selected
            node.source_paragraph_slice_id = (
                f"{selected}@p{int(node.origin_page or 0):04d}"
            )
            assignment.update(
                {
                    "source_paragraph_id": selected,
                    "source_paragraph_slice_id": node.source_paragraph_slice_id,
                    "status": "mapped_unique_page_start_source_suffix",
                    "confidence": 1.0,
                    "lexical_support": 1.0,
                    "candidate_counts": {selected: 1},
                }
            )
            mapped += 1
            edge_suffix_mapped += 1

    mapped_ids = sorted(
        {
            node.source_paragraph_id
            for node in nodes
            if node.source_paragraph_id is not None
        }
    )
    return {
        "contract_version": SOURCE_PARAGRAPH_CONTRACT_VERSION,
        "status": "mapped" if mapped else "fallback_geometry",
        "lines_total": len(nodes),
        "lines_mapped": mapped,
        "lines_fallback_geometry": len(nodes) - mapped,
        "lines_ambiguous": ambiguous,
        "page_start_source_suffix_lines_mapped": edge_suffix_mapped,
        "paragraphs_mapped": len(mapped_ids),
        "source_paragraph_ids": mapped_ids,
        "line_assignments": assignments,
    }


def intersection_ratio(line_bbox: Sequence[float], block_bbox: Sequence[float]) -> float:
    x0 = max(line_bbox[0], block_bbox[0])
    y0 = max(line_bbox[1], block_bbox[1])
    x1 = min(line_bbox[2], block_bbox[2])
    y1 = min(line_bbox[3], block_bbox[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return ((x1 - x0) * (y1 - y0)) / max(1.0, bbox_area(line_bbox))


def padded_bbox(bbox: Sequence[float], x_padding: float, y_padding: float) -> list[float]:
    return [bbox[0] - x_padding, bbox[1] - y_padding, bbox[2] + x_padding, bbox[3] + y_padding]


def visible_block_label(block: SourceBlock) -> str:
    if block.kind == "heading":
        return (
            block.heading_source_title
            or re.sub(r"^#{1,6}\s+", "", block.markdown).strip()
        )
    if block.kind == "table":
        if block.caption_markdown:
            return html.unescape(re.sub(r"<[^>]+>", " ", block.caption_markdown))
    return ""


TABLE_CAPTION_PREFIX_PATTERN = re.compile(
    r"^\s*(?P<prefix>Table\s+(?:[A-Za-z]|[IVXLCDM]+|\d+)(?:[.:-]\d+)*)"
    r"(?=\s*[:.\-]?\s+)",
    flags=re.IGNORECASE,
)
FORMULA_NUMBER_PATTERN = re.compile(
    # Automatic equation numbers are overwhelmingly numeric (optionally
    # prefixed by an appendix letter).  Do not accept a bare symbolic ``(x)``
    # at the end of a formula: that is ordinary mathematics, not a safe tag.
    r"\((?P<number>(?:[A-Za-z][.:-])?\d+(?:[.:-]\d+)*)\)\s*$"
)


def preserve_visible_table_number(block: SourceBlock, visible_text: str) -> str:
    """Keep a Table prefix only when it is visible in the compiled page."""

    block.pdf_visible_caption = normalize_space(visible_text)
    match = TABLE_CAPTION_PREFIX_PATTERN.match(block.pdf_visible_caption)
    if match is None:
        block.caption_number_status = "visible_unnumbered"
        block.markdown = compose_table_markdown(
            block.caption_markdown, block.table_html
        )
        return block.markdown
    prefix = normalize_space(match.group("prefix"))
    block.visible_caption_prefix = prefix
    if not block.caption_markdown:
        block.caption_number_status = "missing_structural_caption"
        return block.markdown
    caption = block.caption_markdown.strip()
    if normalize_space(re.sub(r"<[^>]+>", " ", html.unescape(caption))).casefold().startswith(
        prefix.casefold()
    ):
        block.caption_number_status = "preserved"
        block.markdown = compose_table_markdown(caption, block.table_html)
        return block.markdown
    numbered_caption = f"{html.escape(prefix)}: {caption}"
    block.caption_markdown = numbered_caption
    block.markdown = compose_table_markdown(numbered_caption, block.table_html)
    block.caption_number_status = "preserved"
    return block.markdown


def preserve_visible_formula_number(
    block: SourceBlock, visible_lines: Sequence[PageNode]
) -> str:
    """Keep a safe compiled equation number as a LaTeX ``\\tag``."""

    numbers = {
        match.group("number")
        for node in visible_lines
        if (match := FORMULA_NUMBER_PATTERN.search(normalize_space(node.text)))
    }
    if not numbers:
        block.formula_number_status = "absent"
        return block.markdown
    if len(numbers) != 1:
        block.formula_number_status = "ambiguous"
        return block.markdown
    number = next(iter(numbers))
    block.pdf_visible_formula_number = number
    existing = re.search(r"\\tag\s*\{([^{}]+)\}", block.markdown)
    if existing:
        block.formula_number_status = (
            "preserved" if normalize_space(existing.group(1)) == number else "wrong"
        )
        return block.markdown
    closing = block.markdown.rfind("$$")
    if closing <= 0:
        block.formula_number_status = "unsafe_to_tag"
        return block.markdown
    block.markdown = block.markdown[:closing].rstrip() + f"\n\\tag{{{number}}}\n$$"
    block.formula_number_status = "preserved"
    return block.markdown


def match_visible_lines(
    nodes: list[PageNode],
    target: str,
    page_width: float,
    max_lines: int = 3,
    *,
    require_unique: bool = False,
    allow_trailing_prose: bool = False,
) -> list[int]:
    target_sequence = normalize_tokens(target)
    target_tokens = collections.Counter(target_sequence)
    target_total = sum(target_tokens.values())
    if target_total < 1:
        return []
    global_order = sorted(
        range(len(nodes)), key=lambda index: (nodes[index].top, nodes[index].bbox[0])
    )
    # Scan each visual lane independently as well as the legacy whole-page
    # order.  On a two-column page, a line from the opposite column otherwise
    # interrupts every wrapped heading window.
    orders = [global_order]
    for lane in ("full", "left", "right"):
        lane_order = [
            index for index in global_order if assign_lane(nodes[index], page_width) == lane
        ]
        if lane_order:
            orders.append(lane_order)
    best_score = 0.0
    best_indices: list[int] = []
    scored_candidates: list[tuple[float, tuple[int, ...]]] = []
    seen_candidates: set[tuple[int, ...]] = set()
    for ordered_indices in orders:
        for position in range(len(ordered_indices)):
            selected: list[int] = []
            for length in range(1, max_lines + 1):
                if position + length > len(ordered_indices):
                    break
                index = ordered_indices[position + length - 1]
                if selected:
                    previous = nodes[selected[-1]]
                    current = nodes[index]
                    same_column = assign_lane(previous, page_width) == assign_lane(
                        current, page_width
                    )
                    if current.top - previous.bottom > 22 or not same_column:
                        break
                selected.append(index)
                candidate_tokens = collections.Counter(
                    token
                    for selected_index in selected
                    for token in normalize_tokens(nodes[selected_index].text)
                )
                matched = sum(
                    min(count, candidate_tokens[token])
                    for token, count in target_tokens.items()
                )
                recall = matched / target_total
                excess = max(0, sum(candidate_tokens.values()) - target_total) / max(
                    1, target_total
                )
                score = recall - 0.08 * excess
                if allow_trailing_prose:
                    candidate_sequence = [
                        token
                        for selected_index in selected
                        for token in normalize_tokens(nodes[selected_index].text)
                    ]
                    # A run-in paragraph heading is followed immediately by
                    # its prose and an inline replacement may collapse both
                    # into one node.  Accept only a literal title prefix (with
                    # at most one compiled numbering token before it); do not
                    # let the valid trailing prose count as excess noise.
                    if any(
                        candidate_sequence[offset : offset + target_total]
                        == target_sequence
                        for offset in range(min(2, len(candidate_sequence)))
                    ):
                        score = 1.0
                candidate_key = tuple(selected)
                if candidate_key not in seen_candidates:
                    seen_candidates.add(candidate_key)
                    scored_candidates.append((score, candidate_key))
                if score > best_score:
                    best_score = score
                    best_indices = list(selected)
    if best_score < 0.58:
        return []
    if require_unique:
        near_best = [
            set(indices)
            for score, indices in scored_candidates
            if score >= max(0.58, best_score - 0.02)
        ]
        location_clusters: list[set[int]] = []
        for candidate in near_best:
            overlapping = [
                cluster for cluster in location_clusters if cluster.intersection(candidate)
            ]
            if not overlapping:
                location_clusters.append(set(candidate))
                continue
            merged = set(candidate)
            for cluster in overlapping:
                merged.update(cluster)
                location_clusters.remove(cluster)
            location_clusters.append(merged)
        if len(location_clusters) != 1:
            return []
    return best_indices


def heading_title_relation(visible_title: str, source_title: str) -> str:
    """Classify a conservative token-prefix relationship between two titles."""

    visible_tokens = normalize_tokens(visible_title)
    source_tokens = normalize_tokens(source_title)
    if not visible_tokens or not source_tokens:
        return "mismatch"
    if visible_tokens == source_tokens:
        return "exact"
    if len(visible_tokens) > len(source_tokens) and visible_tokens[: len(source_tokens)] == source_tokens:
        return "source_prefix_of_visible"
    if len(source_tokens) > len(visible_tokens) and source_tokens[: len(visible_tokens)] == visible_tokens:
        return "visible_prefix_of_source"
    return "mismatch"


def heading_trailing_text(visible_title: str, source_title: str) -> str:
    """Return run-in prose after an exact source-title token prefix."""

    source_tokens = normalize_tokens(source_title)
    token_spans = list(re.finditer(r"[A-Za-z0-9]+", visible_title))
    if not source_tokens or len(token_spans) <= len(source_tokens):
        return ""
    visible_prefix = [
        match.group(0).casefold() for match in token_spans[: len(source_tokens)]
    ]
    if visible_prefix != source_tokens:
        return ""
    trailing = visible_title[token_spans[len(source_tokens) - 1].end() :].lstrip()
    plain_source = re.sub(r"[$*_`]", "", source_title).rstrip()
    punctuation_match = re.search(r"([^\w\s]+)$", plain_source)
    if punctuation_match and trailing.startswith(punctuation_match.group(1)):
        trailing = trailing[len(punctuation_match.group(1)) :].lstrip()
    return trailing


def analyze_pdf_visible_heading(
    block: SourceBlock, visible_heading: str
) -> dict[str, str | None]:
    """Audit a matched PDF heading without recreating LaTeX counters.

    Only an actual prefix from the compiled PDF text layer can be emitted.
    Starred headings never have a prefix stripped or synthesized.
    """

    source_title = block.heading_source_title or visible_block_label(block)
    visible_heading = normalize_space(visible_heading)
    prefix_match = HEADING_NUMBER_PREFIX_PATTERN.match(visible_heading)
    if block.heading_starred:
        relation = heading_title_relation(visible_heading, source_title)
        if relation not in {"exact", "source_prefix_of_visible"}:
            return {
                "status": "ambiguous",
                "prefix": None,
                "trailing_text": None,
                "relation": relation,
            }
        return {
            "status": "unnumbered",
            "prefix": None,
            "trailing_text": (
                heading_trailing_text(visible_heading, source_title)
                if relation == "source_prefix_of_visible"
                else ""
            ),
            "relation": relation,
        }

    candidate_title = visible_heading
    prefix: str | None = None
    if prefix_match:
        prefix = prefix_match.group("prefix")
        candidate_title = visible_heading[prefix_match.end() :].strip()
    relation = heading_title_relation(candidate_title, source_title)
    # A partial visible title means the matched window did not cover the whole
    # heading; removing it could leave a duplicate continuation behind.
    if relation in {"mismatch", "visible_prefix_of_source"}:
        return {
            "status": "ambiguous",
            "prefix": prefix,
            "trailing_text": None,
            "relation": relation,
        }
    return {
        "status": "preserved" if prefix else "unnumbered",
        "prefix": prefix,
        "trailing_text": (
            heading_trailing_text(candidate_title, source_title)
            if relation == "source_prefix_of_visible"
            else ""
        ),
        "relation": relation,
    }


def heading_numbering_summary(page_blocks: Sequence[SourceBlock]) -> dict[str, Any]:
    statuses = collections.Counter(
        block.heading_number_status or "ambiguous"
        for block in page_blocks
        if block.kind == "heading"
    )
    summary = {
        key: int(statuses.get(key, 0))
        for key in ("preserved", "lost", "wrong", "ambiguous", "unnumbered")
    }
    summary["total"] = sum(summary.values())
    summary["strict"] = not any(
        summary[key] for key in ("lost", "wrong", "ambiguous")
    )
    return summary


def compact_alignment_probe(value: str) -> str:
    """Keep only normalized letters/digits for a cheap regex prefilter."""

    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def ordered_anchor_probe_matches(anchors: Sequence[str], candidate: str) -> bool:
    """Reject impossible windows before running the full alignment regex.

    Long regex plans contain bounded math/opaque wildcards and discretionary
    hyphen alternatives.  Running them on every line window is needlessly
    expensive.  Every literal anchor of four or more normalized characters
    must occur in source order for the full regex to have any chance of
    matching, so this linear probe is lossless for those candidates.
    """

    required = [compact_alignment_probe(anchor) for anchor in anchors]
    required = [anchor for anchor in required if len(anchor) >= 4]
    if not required:
        return True
    compact_candidate = compact_alignment_probe(candidate)
    cursor = 0
    for anchor in required:
        position = compact_candidate.find(anchor, cursor)
        if position < 0:
            return False
        cursor = position + len(anchor)
    return True


def build_pdf_authoritative_inline_regex(
    block: InlineSourceBlock, *, max_wildcard: int
) -> Any:
    """Require every footnote capture to be a visible ASCII PDF number."""

    regex = build_inline_regex(block.plan, max_wildcard=max_wildcard)
    if not block.footnotes:
        return regex
    pattern = regex.pattern
    for footnote in block.footnotes:
        group_name = regex.group_names.get(footnote.node_id)
        if not group_name:
            raise ValueError(f"footnote group missing: {footnote.note_id}")
        wildcard = re.compile(
            r"\(\?P<"
            + re.escape(group_name)
            + r">\[\\s\\S\]\{0,\d+\}\?\)"
        )
        pattern, substitutions = wildcard.subn(
            f"(?P<{group_name}>[0-9]{{1,4}})", pattern, count=1
        )
        if substitutions != 1:
            raise ValueError(f"footnote wildcard not found: {footnote.note_id}")
    return dataclasses.replace(regex, pattern=pattern, compiled=re.compile(pattern))


def join_text_lines_with_owners(
    lines: Sequence[PageNode],
) -> tuple[str, list[str | None]]:
    """Mirror join_text_lines while tracking the owner of each character."""

    characters: list[str] = []
    owners: list[str | None] = []
    for index, node in enumerate(lines):
        if index:
            if characters and characters[-1] == "-" and node.text and node.text[0].islower():
                characters.pop()
                owners.pop()
            else:
                characters.append(" ")
                owners.append(None)
        characters.extend(node.text)
        owners.extend([node.line_id] * len(node.text))
    # Line text is normalized when extracted, so joining adds at most one
    # separator.  Keep this assertion auditable instead of silently changing
    # offsets used for footnote-marker provenance.
    value = "".join(characters)
    normalized = normalize_space(value)
    if normalized != value:
        remapped_characters: list[str] = []
        remapped_owners: list[str | None] = []
        for match in re.finditer(r"\S+", value):
            if remapped_characters:
                remapped_characters.append(" ")
                remapped_owners.append(None)
            remapped_characters.extend(match.group(0))
            remapped_owners.extend(owners[match.start() : match.end()])
        value = "".join(remapped_characters)
        owners = remapped_owners
    return value, owners


def captured_footnote_callouts(
    block: InlineSourceBlock,
    match: re.Match[str],
    regex: Any,
    character_owners: Sequence[str | None],
) -> list[dict[str, Any]] | None:
    captures: list[dict[str, Any]] = []
    for footnote in block.footnotes:
        group_name = regex.group_names.get(footnote.node_id)
        if not group_name:
            return None
        marker = match.groupdict().get(group_name) or ""
        if re.fullmatch(r"[0-9]{1,4}", marker, flags=re.ASCII) is None:
            return None
        start, end = match.span(group_name)
        line_ids = list(
            dict.fromkeys(
                owner
                for owner in character_owners[start:end]
                if isinstance(owner, str) and owner
            )
        )
        if not line_ids:
            return None
        captures.append(
            {
                "note_id": footnote.note_id,
                "marker": marker,
                "callout_line_ids": line_ids,
            }
        )
    return captures


def render_html_sup_callouts(
    rendered: str, captures: Sequence[dict[str, Any]]
) -> str | None:
    """Convert inline-module footnote placeholders to explicit HTML superscripts."""

    marker_counts = collections.Counter(str(capture["marker"]) for capture in captures)
    for marker, count in marker_counts.items():
        placeholder = f"[^{marker}]"
        html_sup = f"<sup>{marker}</sup>"
        if rendered.count(placeholder) + rendered.count(html_sup) != count:
            return None
        rendered = rendered.replace(placeholder, html_sup)
    if re.search(r"\[\^[0-9]{1,4}\](?::)?", rendered):
        return None
    return rendered


def apply_inline_source_blocks(
    line_nodes: list[PageNode],
    page_blocks: Sequence[InlineSourceBlock],
    page_width: float,
    layout_hint: str | None = None,
) -> tuple[list[PageNode], dict[str, Any]]:
    """Restore inline source markup on exact PDF-text windows.

    PDF text remains the baseline.  A block is only applied when its bounded
    alignment regex matches a nearby, same-lane sequence of PDF lines.  Failed
    blocks leave the original lines untouched.
    """

    nodes = list(line_nodes)
    matched_blocks = 0
    fallback_blocks = 0
    resolved_features: collections.Counter[str] = collections.Counter()
    unresolved_features: collections.Counter[str] = collections.Counter()
    ordered_blocks = sorted(
        page_blocks,
        key=lambda item: (
            float(item.bbox[1]) if item.bbox else math.inf,
            str(item.source_file),
            item.start_line,
        ),
    )
    last_progress = time.monotonic()
    for block_index, block in enumerate(ordered_blocks, start=1):
        now = time.monotonic()
        if now - last_progress >= HEARTBEAT_SECONDS:
            print(
                f"[progress] inline_alignment_blocks={block_index-1}/{len(ordered_blocks)} "
                f"matched={matched_blocks} fallback={fallback_blocks}",
                flush=True,
            )
            last_progress = now
        target_counts = block.target_feature_counts
        anchor_text = " ".join(block.plan.anchors)
        anchor_tokens = normalize_tokens(anchor_text)
        source_math_tokens = {
            token
            for inline_node in iter_inline_nodes(block.plan.root)
            if inline_node.kind == "math"
            for token in normalize_tokens(latex_to_plain(inline_node.value))
        }
        body_font_sizes = [
            node.font_size
            for node in nodes
            if node.kind == "text" and len(normalize_tokens(node.text)) >= 6
        ]
        body_font_size = (
            statistics.median(body_font_sizes) if body_font_sizes else 0.0
        )
        potential_satellite_ids = {
            id(node)
            for node in nodes
            if source_math_tokens
            and node.kind == "text"
            and 0 < len(normalize_tokens(node.text)) <= 3
            and set(normalize_tokens(node.text)).issubset(source_math_tokens)
            and body_font_size > 0
            and node.font_size <= body_font_size * 0.9
        }
        if len(anchor_tokens) < 3 or block.plan.anchor_characters < 12:
            block.match_status = "fallback_pdf"
            block.match_reason = "insufficient_literal_anchor"
            fallback_blocks += 1
            unresolved_features.update(target_counts)
            continue
        for node in nodes:
            node.lane = (
                "single"
                if layout_hint == "single_column"
                else assign_lane(node, page_width)
            )
        hinted_lane = ""
        hinted_top: float | None = None
        if block.bbox and valid_bbox(block.bbox):
            hinted_top = float(block.bbox[1])
            hinted_lane = (
                "single"
                if layout_hint == "single_column"
                else assign_lane(PageNode("hint", "", list(block.bbox), 0), page_width)
            )
        lane_values = ["single"] if layout_hint == "single_column" else ["left", "right", "full"]
        if hinted_lane in {"left", "right"} and layout_hint != "single_column":
            lane_values.sort(key=lambda lane: lane != hinted_lane)
        candidates: list[dict[str, Any]] = []
        try:
            alignment_regex = build_pdf_authoritative_inline_regex(
                block, max_wildcard=256
            )
        except (ValueError, re.error) as exc:
            block.match_status = "fallback_pdf"
            block.match_reason = f"invalid_footnote_alignment_regex:{exc}"
            fallback_blocks += 1
            unresolved_features.update(target_counts)
            continue
        for lane in lane_values:
            lane_nodes = [
                node
                for node in nodes
                if node.kind == "text"
                and node.lane == lane
            ]
            lane_nodes.sort(key=lambda node: (node.top, node.bbox[0]))
            for start in range(len(lane_nodes)):
                start_distance = abs(lane_nodes[start].top - hinted_top) if hinted_top is not None else 0.0
                for length in range(1, min(24, len(lane_nodes) - start) + 1):
                    window = lane_nodes[start : start + length]
                    semantic_window = [
                        node for node in window if id(node) not in potential_satellite_ids
                    ]
                    if not semantic_window:
                        continue
                    paragraph_ids = {
                        node.source_paragraph_id
                        for node in semantic_window
                        if node.source_paragraph_id is not None
                    }
                    if len(paragraph_ids) > 1:
                        # A formatting replacement may not collapse two source
                        # paragraphs into one PDF node.
                        break
                    candidate_text, character_owners = join_text_lines_with_owners(
                        semantic_window
                    )
                    if len(candidate_text) > 6000:
                        break
                    if not ordered_anchor_probe_matches(
                        block.plan.anchors, candidate_text
                    ):
                        continue
                    # TeX layout commands and environments can leave harmless
                    # leading/trailing whitespace nodes in the inline plan
                    # (for example ``\hspace*{\fill}`` or ``center``).  Requiring
                    # those spaces to come from an adjacent PDF line used to
                    # make the matcher claim the preceding/following paragraph.
                    # Supply one ownerless virtual boundary space on each side,
                    # then clip it out of both the visible match and rendered
                    # Markdown.  The real PDF line claims therefore remain the
                    # smallest source-paragraph-local window.
                    alignment_text = " " + candidate_text + " "
                    alignment_owners: list[str | None] = [
                        None,
                        *character_owners,
                        None,
                    ]
                    match = alignment_regex.search(alignment_text)
                    if match is None or not match.group(0).strip():
                        continue
                    footnote_captures = captured_footnote_callouts(
                        block,
                        match,
                        alignment_regex,
                        alignment_owners,
                    )
                    if block.footnotes and footnote_captures is None:
                        continue
                    rendered_markdown = render_inline_match(
                        block.plan, match, alignment_regex
                    )
                    if block.footnotes:
                        rendered_markdown = render_html_sup_callouts(
                            rendered_markdown, footnote_captures or []
                        )
                        if rendered_markdown is None:
                            continue
                    visible_start = max(0, match.start() - 1)
                    visible_end = min(len(candidate_text), match.end() - 1)
                    matched_visible_text = candidate_text[visible_start:visible_end]
                    if not matched_visible_text.strip():
                        continue
                    if match.start() == 0 and rendered_markdown[:1].isspace():
                        rendered_markdown = rendered_markdown[1:]
                    if (
                        match.end() == len(alignment_text)
                        and rendered_markdown[-1:].isspace()
                    ):
                        rendered_markdown = rendered_markdown[:-1]
                    result = InlineRenderResult(
                        markdown=rendered_markdown,
                        matched_text=matched_visible_text,
                        span=(visible_start, visible_end),
                        regex=alignment_regex,
                    )
                    matched_tokens = normalize_tokens(result.matched_text)
                    if not matched_tokens:
                        continue
                    token_anchor_precision = len(anchor_tokens) / len(matched_tokens)
                    character_anchor_precision = block.plan.anchor_characters / max(
                        1, len(result.matched_text)
                    )
                    anchor_precision = min(
                        1.0,
                        max(token_anchor_precision, character_anchor_precision),
                    )
                    if anchor_precision < 0.25:
                        continue
                    match_bbox = union_bboxes([node.bbox for node in window])
                    spatial_distance = (
                        abs(match_bbox[1] - hinted_top) if hinted_top is not None else 0.0
                    )
                    spatial_score = max(0.0, 1.0 - spatial_distance / 180.0)
                    score = 0.75 * anchor_precision + 0.25 * spatial_score - 0.002 * (length - 1)
                    candidates.append(
                        {
                            "score": score,
                            "window": window,
                            "satellites": [
                                node
                                for node in window
                                if id(node) in potential_satellite_ids
                            ],
                            "candidate_text": candidate_text,
                            "result": result,
                            "bbox": match_bbox,
                            "spatial_distance": spatial_distance,
                            "footnote_captures": footnote_captures or [],
                        }
                    )
                    # Prefer the smallest exact line window; larger windows
                    # only add duplicate prefix/suffix around the same match.
                    break
        if not candidates:
            block.match_status = "fallback_pdf"
            block.match_reason = "no_high_confidence_pdf_alignment"
            fallback_blocks += 1
            unresolved_features.update(target_counts)
            continue
        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                len(item["window"]),
                float(item["spatial_distance"]),
            )
        )
        selected = candidates[0]
        window = list(selected["window"])
        window_ids = {id(node) for node in window}
        absorbed_satellites: list[PageNode] = list(selected.get("satellites", []))
        if source_math_tokens:
            for node in nodes:
                if (
                    id(node) in window_ids
                    or id(node) not in potential_satellite_ids
                    or node.kind != "text"
                ):
                    continue
                # PDF text extraction frequently emits delayed sub/superscript
                # glyphs as separate short lines.  Consume such a glyph only
                # when it overlaps the exact prose window whose source math is
                # being restored; nearby unrelated small print is untouched.
                if any(
                    max(0.0, min(node.bottom, owner.bottom) - max(node.top, owner.top))
                    / max(0.1, node.bottom - node.top)
                    >= 0.3
                    for owner in window
                ):
                    absorbed_satellites.append(node)
        unique_claim_nodes: list[PageNode] = []
        seen_claim_nodes: set[int] = set()
        for claim_node in [*window, *absorbed_satellites]:
            if id(claim_node) in seen_claim_nodes:
                continue
            seen_claim_nodes.add(id(claim_node))
            unique_claim_nodes.append(claim_node)
        claim_window = sorted(
            unique_claim_nodes,
            key=lambda node: (
                node.origin_order if node.origin_order is not None else math.inf,
                node.top,
                node.bbox[0],
            ),
        )
        result = selected["result"]
        candidate_text = str(selected["candidate_text"])
        enriched_text = (
            candidate_text[: result.span[0]]
            + result.markdown
            + candidate_text[result.span[1] :]
        )
        replacement = PageNode(
            kind="text",
            text=enriched_text,
            bbox=[
                round(float(value), 3)
                for value in union_bboxes([node.bbox for node in claim_window])
            ],
            font_size=statistics.median(node.font_size for node in window),
            lane=window[0].lane,
            inline_source_ids=sorted(
                {
                    block.block_id,
                    *(source_id for node in window for source_id in node.inline_source_ids),
                }
            ),
            line_id=next((node.line_id for node in window if node.line_id), None),
            origin_page=next(
                (node.origin_page for node in window if node.origin_page is not None),
                None,
            ),
            origin_order=min(
                (
                    node.origin_order
                    for node in window
                    if node.origin_order is not None
                ),
                default=None,
            ),
            claimed_line_ids=[
                line_id
                for node in claim_window
                for line_id in node.claimed_line_ids
            ],
            derived_line_ids=sorted(
                {
                    line_id
                    for node in claim_window
                    for line_id in node.derived_line_ids
                }
            ),
            source_paragraph_id=next(
                (
                    node.source_paragraph_id
                    for node in window
                    if node.source_paragraph_id is not None
                ),
                None,
            ),
            source_paragraph_slice_id=next(
                (
                    node.source_paragraph_slice_id
                    for node in window
                    if node.source_paragraph_slice_id is not None
                ),
                None,
            ),
        )
        claimed = {id(node) for node in claim_window}
        nodes = [node for node in nodes if id(node) not in claimed]
        nodes.append(replacement)
        block.match_status = "matched"
        block.match_reason = None
        block.matched_line_count = len(claim_window)
        block.matched_bbox = list(replacement.bbox)
        block.matched_pdf_text = result.matched_text
        block.enriched_markdown = result.markdown
        block.match_score = round(float(selected["score"]), 6)
        block.absorbed_pdf_line_ids = [
            line_id
            for node in absorbed_satellites
            for line_id in node.claimed_line_ids
        ]
        captures_by_note = {
            str(capture["note_id"]): capture
            for capture in selected.get("footnote_captures", [])
        }
        for footnote in block.footnotes:
            capture = captures_by_note.get(footnote.note_id)
            if capture is None:
                footnote.status = "fallback"
                footnote.failure_reason = "callout_capture_missing"
                continue
            footnote.marker = str(capture["marker"])
            footnote.callout_line_ids = list(capture["callout_line_ids"])
            footnote.status = "callout_matched"
            footnote.failure_reason = None
        matched_blocks += 1
        resolved_features.update(target_counts)
        now = time.monotonic()
        if now - last_progress >= HEARTBEAT_SECONDS:
            print(
                f"[progress] inline_alignment_blocks={block_index}/{len(ordered_blocks)} "
                f"matched={matched_blocks} fallback={fallback_blocks}",
                flush=True,
            )
            last_progress = now
    return nodes, {
        "blocks_total": len(page_blocks),
        "blocks_matched": matched_blocks,
        "blocks_fallback_pdf": fallback_blocks,
        "features_total": sum(block.target_feature_total for block in page_blocks),
        "features_resolved": dict(sorted(resolved_features.items())),
        "features_resolved_total": sum(resolved_features.values()),
        "features_unresolved": dict(sorted(unresolved_features.items())),
        "features_unresolved_total": sum(unresolved_features.values()),
    }


def inline_integration_summary(
    page_blocks: Sequence[InlineSourceBlock],
) -> dict[str, Any]:
    resolved: collections.Counter[str] = collections.Counter()
    unresolved: collections.Counter[str] = collections.Counter()
    for block in page_blocks:
        if block.match_status == "matched":
            resolved.update(block.target_feature_counts)
        else:
            unresolved.update(block.target_feature_counts)
    return {
        "blocks_total": len(page_blocks),
        "blocks_matched": sum(block.match_status == "matched" for block in page_blocks),
        "blocks_fallback_pdf": sum(block.match_status != "matched" for block in page_blocks),
        "features_total": sum(block.target_feature_total for block in page_blocks),
        "features_resolved": dict(sorted(resolved.items())),
        "features_resolved_total": sum(resolved.values()),
        "features_unresolved": dict(sorted(unresolved.items())),
        "features_unresolved_total": sum(unresolved.values()),
    }


def reconcile_emitted_inline_blocks(
    page_blocks: Sequence[InlineSourceBlock],
    emitted_inline_source_ids: set[str],
) -> dict[str, int]:
    """Demote line matches removed by later structural integration.

    Inline formatting is first attached to PDF line nodes.  A subsequent
    source-first table/display-math replacement may legitimately remove one of
    those nodes.  Carrying source IDs on the nodes lets us distinguish a real
    final emission from an intermediate match without inserting audit markers
    into the Markdown itself.
    """

    checked = emitted = removed = 0
    for block in page_blocks:
        if block.match_status != "matched":
            continue
        checked += 1
        if block.block_id in emitted_inline_source_ids:
            emitted += 1
            continue
        block.match_status = "fallback_pdf"
        block.match_reason = "line_match_removed_by_structural_integration"
        removed += 1
    return {
        "line_matches_checked_for_emission": checked,
        "line_matches_emitted": emitted,
        "line_matches_removed_by_structural_integration": removed,
    }


def _inline_markup_spans(markdown: str) -> list[tuple[int, int]]:
    """Return spans that a later fallback must never overwrite."""

    pattern = re.compile(
        r"\$\$[\s\S]*?\$\$"
        r"|(?<!\$)\$(?!\$)(?:\\.|[^$\n])*?\$(?!\$)"
        r"|\*\*[^\n]*?\*\*"
        r"|(?<!\*)\*(?!\*)[^\n*]+?(?<!\*)\*(?!\*)"
        r"|`[^`\n]+`"
        r"|<sup>[0-9]{1,4}</sup>"
    )
    return [match.span() for match in pattern.finditer(markdown)]


def _spans_overlap(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def reconcile_final_inline_markup(
    markdown: str,
    page_blocks: Sequence[InlineSourceBlock],
) -> dict[str, int]:
    """Require every claimed match to exist in the final Markdown text.

    This is deliberately stricter than trusting an earlier regex match.  It
    catches any later fallback that replaced a broad prose span and thereby
    erased another block's inline math or emphasis.
    """

    normalized_markdown = normalize_space(markdown)
    checked = present = missing = 0
    for block in page_blocks:
        if block.match_status != "matched":
            continue
        checked += 1
        expected = normalize_space(block.enriched_markdown or "")
        if expected and expected in normalized_markdown:
            present += 1
            continue
        block.match_status = "fallback_pdf"
        block.match_reason = "matched_markup_missing_from_final_markdown"
        missing += 1
    return {
        "final_markup_claims_checked": checked,
        "final_markup_claims_present": present,
        "final_markup_claims_missing": missing,
    }


def _markdown_character_is_escaped(value: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and value[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def markdown_inline_syntax_issues(markdown: str) -> list[str]:
    """Check the delimiters emitted by deterministic inline enrichment.

    The scanner distinguishes TeX-escaped dollars from Markdown delimiters and
    skips complete display-math blocks.  This specifically prevents a source
    control-space such as ``$\\ldots\\ $`` from silently becoming an unclosed
    ``$\\ldots\\$`` sequence in a published GT page.
    """

    issues: list[str] = []
    position = 0
    length = len(markdown)
    while position < length:
        if markdown[position] != "$" or (
            position > 0 and _markdown_character_is_escaped(markdown, position)
        ):
            position += 1
            continue
        if markdown.startswith("$$", position):
            closing = position + 2
            while closing < length - 1:
                if markdown.startswith("$$", closing) and not _markdown_character_is_escaped(
                    markdown, closing
                ):
                    break
                closing += 1
            if closing >= length - 1:
                issues.append(f"unclosed_display_math_at_offset_{position}")
                break
            position = closing + 2
            continue
        closing = position + 1
        while closing < length:
            if markdown[closing] == "$" and not _markdown_character_is_escaped(markdown, closing):
                break
            closing += 1
        if closing >= length:
            issues.append(f"unclosed_inline_math_at_offset_{position}")
            break
        position = closing + 1
    return issues


def apply_inline_blocks_to_markdown(
    markdown: str,
    page_blocks: Sequence[InlineSourceBlock],
) -> tuple[str, dict[str, Any]]:
    """Retry unresolved blocks against unique, unstructured prose chunks.

    Superscripts and subscripts can fragment PDF geometry into separate line
    nodes even when the final reading-order Markdown is correct.  This second
    deterministic path operates only on ordinary prose separated by blank
    lines.  Display math, HTML tables, and Markdown headings are protected,
    and a block is accepted only when it has exactly one match on the page.
    """

    parts = re.split(r"(\n{2,})", markdown)

    def eligible(value: str) -> bool:
        stripped = value.strip()
        return bool(
            stripped
            and not stripped.startswith("#")
            and "<table" not in stripped
            and "</table>" not in stripped
            and not (stripped.startswith("$$") and stripped.endswith("$$"))
        )

    matched = 0
    ambiguous = 0
    for block in page_blocks:
        if block.match_status == "matched":
            continue
        # Footnote callouts require exact PDF line provenance so their page
        # bottom definitions can be claimed atomically before Markdown is
        # written.  A prose-only writer retry cannot provide that provenance.
        if block.footnotes:
            continue
        regex = build_inline_regex(block.plan, max_wildcard=512)
        candidates: list[tuple[int, re.Match[str]]] = []
        for part_index, part in enumerate(parts):
            if not eligible(part):
                continue
            protected = _inline_markup_spans(part)
            candidates.extend(
                (part_index, match)
                for match in regex.finditer(part)
                if not _spans_overlap(match.start(), match.end(), protected)
            )
        if len(candidates) != 1:
            if len(candidates) > 1:
                ambiguous += 1
                block.match_reason = "ambiguous_markdown_alignment"
            continue
        part_index, match = candidates[0]
        rendered = render_inline_match(block.plan, match, regex)
        original = parts[part_index]
        parts[part_index] = original[: match.start()] + rendered + original[match.end() :]
        block.match_status = "matched"
        block.match_reason = "unique_markdown_prose_alignment"
        block.matched_line_count = match.group(0).count("\n") + 1
        block.matched_bbox = block.bbox
        block.matched_pdf_text = match.group(0)
        block.enriched_markdown = rendered
        block.match_score = 1.0
        matched += 1
    return "".join(parts), {
        "matched_by_unique_markdown_alignment": matched,
        "ambiguous_markdown_alignments": ambiguous,
    }


def _boundary_anchor_fragment(value: str, *, from_start: bool, limit: int = 64) -> str:
    value = normalize_space(value)
    words = value.split()
    if len(words) > 4:
        value = " ".join(words[:4] if from_start else words[-4:])
    if len(value) <= limit:
        return value
    if from_start:
        fragment = value[:limit]
        boundary = fragment.rfind(" ")
        return fragment[:boundary] if boundary > 12 else fragment
    fragment = value[-limit:]
    boundary = fragment.find(" ")
    return fragment[boundary + 1 :] if 0 <= boundary < len(fragment) - 12 else fragment


def apply_source_first_inline_blocks_to_markdown(
    markdown: str,
    page_blocks: Sequence[InlineSourceBlock],
) -> tuple[str, dict[str, Any]]:
    """Replace a uniquely bounded corrupt PDF span with safe source markup.

    This path is limited to focused plans with no opaque commands.  Therefore
    citations, references, footnotes, labels, and custom prose macros can never
    be guessed from source.  It is useful when PDF glyph ordering inserts
    superscripts or formula symbols inside an otherwise ordinary word.
    """

    matched = 0
    ambiguous = 0
    for block in page_blocks:
        if block.match_status == "matched":
            continue
        if int(block.plan.feature_counts.get("opaque", 0)):
            continue
        anchors = [normalize_space(value) for value in block.plan.anchors if normalize_space(value)]
        if not anchors:
            continue
        first_anchor = _boundary_anchor_fragment(anchors[0], from_start=True)
        last_anchor = _boundary_anchor_fragment(anchors[-1], from_start=False)
        if len(re.sub(r"\W", "", first_anchor)) < 4 or len(
            re.sub(r"\W", "", last_anchor)
        ) < 4:
            continue
        rendered = render_inline_source(block.plan).strip()
        if not rendered:
            continue
        maximum_candidate_span = min(2500, max(400, int(len(rendered) * 3.0 + 150)))
        first_regex = build_text_anchor_regex(first_anchor)
        last_regex = build_text_anchor_regex(last_anchor)
        protected = [
            match.span()
            for match in re.finditer(
                r"\$\$[\s\S]*?\$\$|<table\b[\s\S]*?</table>",
                markdown,
                flags=re.IGNORECASE,
            )
        ]
        protected.extend(_inline_markup_spans(markdown))

        def overlaps_protected(start: int, end: int) -> bool:
            return _spans_overlap(start, end, protected)

        starts = [
            match
            for match in first_regex.finditer(markdown)
            if not overlaps_protected(match.start(), match.end())
        ]
        ends = [
            match
            for match in last_regex.finditer(markdown)
            if not overlaps_protected(match.start(), match.end())
        ]
        candidates: list[tuple[int, int]] = []
        if first_anchor == last_anchor:
            candidates = [(match.start(), match.end()) for match in starts]
        else:
            for start_match in starts:
                for end_match in ends:
                    if end_match.end() < start_match.end():
                        continue
                    if end_match.end() - start_match.start() > maximum_candidate_span:
                        continue
                    start = start_match.start()
                    end = end_match.end()
                    candidate = markdown[start:end]
                    if overlaps_protected(start, end):
                        continue
                    if re.search(r"(?m)^#{1,6}\s", candidate):
                        continue
                    candidates.append((start, end))
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            if len(candidates) > 1:
                ambiguous += 1
                block.match_reason = "ambiguous_source_first_anchor_alignment"
            continue
        start, end = candidates[0]
        original = markdown[start:end]
        markdown = markdown[:start] + rendered + markdown[end:]
        block.match_status = "matched"
        block.match_reason = "unique_source_first_anchor_alignment"
        block.matched_line_count = original.count("\n") + 1
        block.matched_bbox = block.bbox
        block.matched_pdf_text = original
        block.enriched_markdown = rendered
        block.match_score = 0.95
        matched += 1
    return markdown, {
        "matched_by_unique_source_first_alignment": matched,
        "ambiguous_source_first_alignments": ambiguous,
    }


def integrate_source_blocks(
    line_nodes: list[PageNode], page_blocks: list[SourceBlock], page_width: float
) -> tuple[list[PageNode], list[PageNode], dict[str, Any]]:
    removed: set[int] = set()
    structured_nodes: list[PageNode] = []
    matched_headings = 0
    ambiguous_headings = 0
    matched_captions = 0
    for block in page_blocks:
        bbox = list(block.bbox or [0, 0, 0, 0])
        heading_trailing_node: PageNode | None = None
        block_claim_indices: set[int] = set()
        if block.kind in {"heading", "table"}:
            matched_indices = match_visible_lines(
                line_nodes,
                visible_block_label(block),
                page_width,
                max_lines=4 if block.kind == "heading" else 3,
                require_unique=block.kind == "heading",
                allow_trailing_prose=(
                    block.kind == "heading"
                    and block.heading_command in {"paragraph", "subparagraph"}
                ),
            )
            if matched_indices:
                matched_nodes = sorted(
                    (line_nodes[index] for index in matched_indices),
                    key=lambda node: (node.top, node.bbox[0]),
                )
                matched_bbox = union_bboxes([node.bbox for node in matched_nodes])
                if block.kind == "heading":
                    visible_heading = join_text_lines(matched_nodes)
                    analysis = analyze_pdf_visible_heading(block, visible_heading)
                    block.pdf_visible_heading = visible_heading
                    block.visible_number_prefix = analysis["prefix"]
                    block.heading_number_status = str(analysis["status"])
                    block.heading_matched_line_count = len(matched_nodes)
                    bbox = matched_bbox
                    block.bbox = [round(value, 3) for value in matched_bbox]
                    if block.heading_number_status == "ambiguous":
                        # Keep every original PDF line when the visible prefix
                        # or title extent cannot be verified safely.
                        block.heading_structure_status = "fallback_ambiguous"
                        ambiguous_headings += 1
                        continue
                    source_title = block.heading_source_title or visible_block_label(block)
                    visible_prefix = block.visible_number_prefix
                    heading_label = (
                        f"{visible_prefix} {source_title}"
                        if visible_prefix
                        else source_title
                    )
                    block.markdown = (
                        "#" * int(block.heading_level or 2) + " " + heading_label
                    )
                    if visible_prefix:
                        emitted_prefix_match = HEADING_NUMBER_PREFIX_PATTERN.match(
                            heading_label
                        )
                        if not emitted_prefix_match:
                            block.heading_number_status = "lost"
                        elif emitted_prefix_match.group("prefix") != visible_prefix:
                            block.heading_number_status = "wrong"
                    block.heading_structure_status = "structured"
                    matched_headings += 1
                    trailing_text = str(analysis.get("trailing_text") or "")
                    if trailing_text:
                        inline_source_ids = sorted(
                            {
                                source_id
                                for node in matched_nodes
                                for source_id in node.inline_source_ids
                            }
                        )
                        source_paragraph_ids = {
                            node.source_paragraph_id
                            for node in matched_nodes
                            if node.source_paragraph_id is not None
                        }
                        source_paragraph_slice_ids = {
                            node.source_paragraph_slice_id
                            for node in matched_nodes
                            if node.source_paragraph_slice_id is not None
                        }
                        heading_trailing_node = PageNode(
                            kind="text",
                            text=trailing_text,
                            bbox=list(matched_bbox),
                            font_size=statistics.median(
                                node.font_size for node in matched_nodes
                            ),
                            inline_source_ids=inline_source_ids,
                            source_paragraph_id=(
                                next(iter(source_paragraph_ids))
                                if len(source_paragraph_ids) == 1
                                else None
                            ),
                            source_paragraph_slice_id=(
                                next(iter(source_paragraph_slice_ids))
                                if len(source_paragraph_slice_ids) == 1
                                else None
                            ),
                        )
                else:
                    bbox = union_bboxes([bbox, matched_bbox]) if valid_bbox(bbox) else matched_bbox
                    block.bbox = [round(value, 3) for value in bbox]
                    preserve_visible_table_number(block, join_text_lines(matched_nodes))
                    matched_captions += 1
                block_claim_indices.update(matched_indices)
            elif block.kind == "heading":
                # A broad/ambiguous SyncTeX heading box is dangerous: retain
                # the visible PDF text rather than deleting a whole column.
                block.heading_number_status = "ambiguous"
                block.heading_structure_status = "fallback_unmatched"
                block.heading_matched_line_count = 0
                ambiguous_headings += 1
                continue
        if block.kind == "display_math":
            removal_bbox = padded_bbox(bbox, 14.0, 5.0)
        elif block.kind == "table":
            removal_bbox = padded_bbox(bbox, 4.0, 4.0)
        else:
            removal_bbox = bbox
        if block.kind != "heading":
            block_claim_indices.update(
                index
                for index, node in enumerate(line_nodes)
                if intersection_ratio(node.bbox, removal_bbox) >= 0.14
            )
        claimed_nodes = [line_nodes[index] for index in sorted(block_claim_indices)]
        if block.kind == "display_math":
            preserve_visible_formula_number(block, claimed_nodes)
        claimed_line_ids = [
            line_id
            for node in claimed_nodes
            for line_id in node.claimed_line_ids
        ]
        removed.update(block_claim_indices)
        structured_nodes.append(
            PageNode(
                kind=block.kind,
                text=block.markdown,
                bbox=bbox,
                font_size=0,
                source_block_id=block.block_id,
                line_id=claimed_line_ids[0] if claimed_line_ids else None,
                origin_page=next(
                    (
                        node.origin_page
                        for node in claimed_nodes
                        if node.origin_page is not None
                    ),
                    None,
                ),
                origin_order=min(
                    (
                        node.origin_order
                        for node in claimed_nodes
                        if node.origin_order is not None
                    ),
                    default=None,
                ),
                claimed_line_ids=claimed_line_ids,
            )
        )
        if heading_trailing_node is not None:
            heading_trailing_node.derived_line_ids = list(claimed_line_ids)
            structured_nodes.append(heading_trailing_node)
    retained = [node for index, node in enumerate(line_nodes) if index not in removed]
    numbering = heading_numbering_summary(page_blocks)
    return retained, structured_nodes, {
        "matched_headings": matched_headings,
        "ambiguous_headings": ambiguous_headings,
        "matched_table_captions": matched_captions,
        "removed_pdf_lines": len(removed),
        "heading_numbering": numbering,
        "table_caption_numbering": dict(
            collections.Counter(
                block.caption_number_status or "pending"
                for block in page_blocks
                if block.kind == "table"
            )
        ),
        "display_formula_numbering": dict(
            collections.Counter(
                block.formula_number_status or "pending"
                for block in page_blocks
                if block.kind == "display_math"
            )
        ),
    }


def assign_lane(node: PageNode, page_width: float) -> str:
    midpoint = page_width / 2
    gutter = page_width * 0.035
    width = node.bbox[2] - node.bbox[0]
    if width >= page_width * 0.58 or (node.bbox[0] < midpoint - gutter and node.bbox[2] > midpoint + gutter):
        return "full"
    if node.bbox[2] <= midpoint + gutter:
        return "left"
    if node.bbox[0] >= midpoint - gutter:
        return "right"
    return "full"


def order_page_nodes(
    nodes: list[PageNode], page_width: float, layout_hint: str | None = None
) -> tuple[list[PageNode], str]:
    min_top = min((node.top for node in nodes), default=0.0)
    max_bottom = max((node.bottom for node in nodes), default=0.0)
    midpoint = page_width / 2
    for node in nodes:
        centered_edge_number = (
            bool(re.fullmatch(r"\s*(?:\d+|[ivxlcdm]+)\s*", node.text, re.I))
            and abs(((node.bbox[0] + node.bbox[2]) / 2) - midpoint)
            <= page_width * 0.08
            and (
                node.top <= min_top + 24.0
                or node.bottom >= max_bottom - 24.0
            )
        )
        # A centered folio is a page-wide header/footer even though its tiny
        # glyph bbox falls geometrically inside one column. Treating it as a
        # left-column line yields the incorrect order left -> folio -> right.
        node.lane = "full" if centered_edge_number else assign_lane(node, page_width)
    minimum_column_width = page_width * 0.18
    left_candidates = [
        node
        for node in nodes
        if node.lane == "left"
        and node.kind == "text"
        and node.bbox[2] - node.bbox[0] >= minimum_column_width
    ]
    right_candidates = [
        node
        for node in nodes
        if node.lane == "right"
        and node.kind == "text"
        and node.bbox[2] - node.bbox[0] >= minimum_column_width
    ]
    unused_right = set(range(len(right_candidates)))
    paired_baselines = 0
    for left in sorted(left_candidates, key=lambda node: node.top):
        choices = [
            (abs(left.top - right_candidates[index].top), index)
            for index in unused_right
            if abs(left.top - right_candidates[index].top) <= 4.0
        ]
        if choices:
            _, selected = min(choices)
            unused_right.remove(selected)
            paired_baselines += 1
    text_segment_count = sum(node.kind == "text" for node in nodes)
    paired_threshold = max(5, math.ceil(text_segment_count * 0.12))
    layout = layout_hint or ("two_column" if paired_baselines >= paired_threshold else "single_column")
    if layout == "single_column":
        return sorted(nodes, key=lambda node: (node.top, node.bbox[0])), layout
    full_nodes = sorted((node for node in nodes if node.lane == "full"), key=lambda node: (node.top, node.bbox[0]))
    column_nodes = [node for node in nodes if node.lane != "full"]
    ordered: list[PageNode] = []
    cursor = -math.inf
    for full_node in full_nodes:
        band = [node for node in column_nodes if cursor <= node.top < full_node.top]
        ordered.extend(sorted((node for node in band if node.lane == "left"), key=lambda node: (node.top, node.bbox[0])))
        ordered.extend(sorted((node for node in band if node.lane == "right"), key=lambda node: (node.top, node.bbox[0])))
        ordered.append(full_node)
        cursor = max(cursor, full_node.bottom)
    band = [node for node in column_nodes if node.top >= cursor]
    ordered.extend(sorted((node for node in band if node.lane == "left"), key=lambda node: (node.top, node.bbox[0])))
    ordered.extend(sorted((node for node in band if node.lane == "right"), key=lambda node: (node.top, node.bbox[0])))
    included = {id(node) for node in ordered}
    ordered.extend(sorted((node for node in nodes if id(node) not in included), key=lambda node: (node.top, node.bbox[0])))
    return ordered, layout


def _footnote_content_issues(body_raw: str, rendered_body: str) -> tuple[dict[str, int], list[str]]:
    issues: list[str] = []
    try:
        body_plan = parse_inline_plan(body_raw)
        features = {
            key: int(value)
            for key, value in body_plan.feature_counts.items()
            if key in {"math", "strong", "em", "code", "citation", "reference"}
            and int(value) > 0
        }
    except InlineParseError:
        return {}, ["body_parse_failed"]
    if re.search(r"\\(?:cite\w*|ref|eqref|pageref|autoref|cref|Cref|footnote)\s*\{", rendered_body):
        issues.append("raw_tex_command_leaked")
    for command, key in re.findall(
        r"\\(cite\w*|ref|eqref|pageref|autoref|cref|Cref)\s*\{([^{}]+)\}",
        body_raw,
    ):
        del command
        for source_key in (part.strip() for part in key.split(",")):
            # A key may legitimately equal visible prose (for example the
            # citation key ``Agronomof`` beside the author name Agronomof).
            # Treat only TeX-like braced serialization as key leakage; the
            # raw-command check above handles the normal failure mode.
            if source_key and f"{{{source_key}}}" in rendered_body:
                issues.append("source_key_leaked")
    issues.extend(markdown_inline_syntax_issues(rendered_body))
    return features, sorted(set(issues))


def _restore_footnote_callouts(
    nodes: Sequence[PageNode], page_blocks: Sequence[InlineSourceBlock]
) -> None:
    for block in page_blocks:
        if not block.footnotes:
            continue
        for footnote in block.footnotes:
            if not footnote.marker:
                continue
            reference = f"<sup>{footnote.marker}</sup>"
            for node in nodes:
                if block.block_id in node.inline_source_ids:
                    node.text = node.text.replace(reference, footnote.marker)
            if block.enriched_markdown:
                block.enriched_markdown = block.enriched_markdown.replace(
                    reference, footnote.marker
                )


def integrate_footnote_definitions(
    ordered_nodes: list[PageNode],
    page_blocks: Sequence[InlineSourceBlock],
    inventory: dict[str, Any],
    page_number: int,
) -> tuple[list[PageNode], dict[str, Any]]:
    """Atomically turn matched PDF callouts and bottom lines into footnotes."""

    specifications = [
        footnote for block in page_blocks for footnote in block.footnotes
    ]
    definition_node_id = f"footnotes-page-{page_number:04d}"

    def fallback(reason: str) -> tuple[list[PageNode], dict[str, Any]]:
        _restore_footnote_callouts(ordered_nodes, page_blocks)
        for footnote in specifications:
            footnote.status = "fallback"
            footnote.failure_reason = reason
            footnote.definition_line_ids = []
            footnote.rendered_body = None
            footnote.content_validation_status = "failed"
            footnote.content_validation_issues = [reason]
        return ordered_nodes, {
            "status": "failed",
            "representation": FOOTNOTE_REPRESENTATION,
            "definition_node_id": definition_node_id,
            "total": len(specifications),
            "structured": 0,
            "fallback": len(specifications),
            "failure_reason": reason,
        }

    if not specifications:
        return ordered_nodes, {
            "status": "passed",
            "representation": FOOTNOTE_REPRESENTATION,
            "definition_node_id": None,
            "total": 0,
            "structured": 0,
            "fallback": 0,
            "failure_reason": None,
        }
    if any(
        footnote.status != "callout_matched"
        or re.fullmatch(r"[0-9]{1,4}", footnote.marker or "", flags=re.ASCII) is None
        or not footnote.callout_line_ids
        for footnote in specifications
    ):
        return fallback("callout_not_uniquely_matched")
    markers = [str(footnote.marker) for footnote in specifications]
    if len(set(markers)) != len(markers):
        return fallback("duplicate_footnote_marker")

    inventory_lines = {
        str(entry["line_id"]): entry for entry in inventory.get("lines", [])
    }
    canonical_rank = {
        str(line_id): index
        for index, line_id in enumerate(inventory.get("canonical_line_ids", []))
    }
    try:
        latest_callout_rank = max(
            canonical_rank[line_id]
            for footnote in specifications
            for line_id in footnote.callout_line_ids
        )
    except (KeyError, ValueError):
        return fallback("callout_line_provenance_invalid")

    excluded_line_ids = {
        line_id
        for line_id, entry in inventory_lines.items()
        if any(
            bool(entry.get("must_preserve", {}).get(key))
            for key in ("header", "footer", "page_number", "caption")
        )
    }
    marker_alternation = "|".join(
        re.escape(marker) for marker in sorted(set(markers), key=len, reverse=True)
    )
    definition_start = re.compile(
        rf"^\s*(?P<marker>{marker_alternation})(?![0-9])\s*(?P<body>\S[\s\S]*)$",
        flags=re.ASCII,
    )

    def eligible_definition_node(node: PageNode) -> bool:
        return bool(
            node.kind == "text"
            and node.source_block_id is None
            and node.claimed_line_ids
            # A page number can sit just above the generic 94% edge threshold
            # used by the legacy inventory.  A number-only line cannot be a
            # supported glued-marker definition and must remain independent.
            and re.fullmatch(r"\s*(?:[0-9]+|[ivxlcdm]+)\s*", node.text, re.I)
            is None
            and not (set(node.claimed_line_ids) & excluded_line_ids)
            and all(
                line_id in canonical_rank
                and canonical_rank[line_id] > latest_callout_rank
                for line_id in node.claimed_line_ids
            )
        )

    starts: list[tuple[int, re.Match[str]]] = []
    for node_index, node in enumerate(ordered_nodes):
        if not eligible_definition_node(node):
            continue
        match = definition_start.fullmatch(node.text)
        if match is not None:
            starts.append((node_index, match))
    if not starts:
        return fallback("definition_marker_not_found")

    candidate_groups: list[dict[str, Any]] = []
    start_indices = {index for index, _ in starts}
    for start_position, (start_index, start_match) in enumerate(starts):
        next_start = (
            starts[start_position + 1][0]
            if start_position + 1 < len(starts)
            else len(ordered_nodes)
        )
        group_nodes: list[PageNode] = []
        for node_index in range(start_index, next_start):
            node = ordered_nodes[node_index]
            if node_index != start_index and node_index in start_indices:
                break
            if not eligible_definition_node(node):
                break
            group_nodes.append(node)
        if not group_nodes:
            continue
        visible_body = join_inventory_texts(
            [start_match.group("body")]
            + [node.text for node in group_nodes[1:]]
        )
        candidate_groups.append(
            {
                "marker": start_match.group("marker"),
                "start_index": start_index,
                "nodes": group_nodes,
                "visible_body": visible_body,
            }
        )

    selected_groups: list[dict[str, Any]] = []
    for footnote in specifications:
        matches: list[dict[str, Any]] = []
        for candidate in candidate_groups:
            if candidate["marker"] != footnote.marker:
                continue
            rendered = render_footnote_body(
                footnote.body_raw,
                str(candidate["visible_body"]),
                max_wildcard=512,
            )
            if rendered is None:
                continue
            features, issues = _footnote_content_issues(footnote.body_raw, rendered)
            if issues:
                continue
            value = dict(candidate)
            value["rendered_body"] = rendered
            value["body_feature_counts"] = features
            value["content_validation_issues"] = issues
            matches.append(value)
        if len(matches) != 1:
            return fallback(
                "definition_alignment_ambiguous"
                if len(matches) > 1
                else "definition_body_alignment_failed"
            )
        selected_groups.append(matches[0])

    selected_node_ids = [
        id(node) for group in selected_groups for node in group["nodes"]
    ]
    if len(set(selected_node_ids)) != len(selected_node_ids):
        return fallback("definition_groups_overlap")
    ordered_specifications = sorted(
        specifications,
        key=lambda footnote: min(
            canonical_rank[line_id] for line_id in footnote.callout_line_ids
        ),
    )
    group_by_marker = {str(group["marker"]): group for group in selected_groups}
    selected_groups = [group_by_marker[str(note.marker)] for note in ordered_specifications]
    group_starts = [int(group["start_index"]) for group in selected_groups]
    if group_starts != sorted(group_starts):
        return fallback("definition_order_mismatch")
    definition_indices = sorted(
        index
        for group in selected_groups
        for index, node in enumerate(ordered_nodes)
        if id(node) in {id(value) for value in group["nodes"]}
    )
    if definition_indices != list(range(min(definition_indices), max(definition_indices) + 1)):
        return fallback("definition_groups_not_contiguous")

    definition_lines: list[str] = []
    claimed_line_ids: list[str] = []
    definition_nodes: list[PageNode] = []
    for footnote in ordered_specifications:
        group = group_by_marker[str(footnote.marker)]
        group_nodes = list(group["nodes"])
        definition_nodes.extend(group_nodes)
        footnote.definition_line_ids = [
            line_id for node in group_nodes for line_id in node.claimed_line_ids
        ]
        footnote.rendered_body = str(group["rendered_body"])
        footnote.body_feature_counts = dict(group["body_feature_counts"])
        footnote.content_validation_issues = list(group["content_validation_issues"])
        footnote.content_validation_status = "passed"
        footnote.status = "structured"
        footnote.failure_reason = None
        claimed_line_ids.extend(footnote.definition_line_ids)
        definition_lines.append(
            f"<sup>{footnote.marker}</sup> {footnote.rendered_body}"
        )

    definition_node = PageNode(
        kind="footnote_definitions",
        text="\n\n".join(definition_lines),
        bbox=union_bboxes([node.bbox for node in definition_nodes]),
        font_size=statistics.median(node.font_size for node in definition_nodes),
        lane=definition_nodes[0].lane,
        source_block_id=definition_node_id,
        origin_page=page_number,
        origin_order=min(
            node.origin_order
            for node in definition_nodes
            if node.origin_order is not None
        ),
        claimed_line_ids=claimed_line_ids,
    )
    selected_ids = {id(node) for node in definition_nodes}
    first_index = min(definition_indices)
    integrated: list[PageNode] = []
    for node_index, node in enumerate(ordered_nodes):
        if node_index == first_index:
            integrated.append(definition_node)
        if id(node) not in selected_ids:
            integrated.append(node)
    return integrated, {
        "status": "passed",
        "representation": FOOTNOTE_REPRESENTATION,
        "definition_node_id": definition_node_id,
        "total": len(specifications),
        "structured": len(specifications),
        "fallback": 0,
        "failure_reason": None,
    }


def finalize_footnote_audit(
    markdown: str,
    page_blocks: Sequence[InlineSourceBlock],
    integration: dict[str, Any],
) -> dict[str, Any]:
    notes: list[dict[str, Any]] = []
    for block in page_blocks:
        for footnote in block.footnotes:
            value = footnote.as_json()
            value["block_id"] = block.block_id
            marker = footnote.marker or ""
            if marker:
                markup = f"<sup>{marker}</sup>"
                # Count the callout inside its source-owned paragraph rather
                # than across the whole page.  Author affiliation markers may
                # legitimately use the same visible superscript number.
                callout_occurrences = (
                    (block.enriched_markdown or "").count(markup)
                    if block.match_status == "matched"
                    else 0
                )
                definition_occurrences = len(
                    re.findall(
                        rf"(?m)^<sup>{re.escape(marker)}</sup>\s+", markdown
                    )
                )
                value["definition_markdown_occurrences"] = definition_occurrences
                value["callout_markdown_occurrences"] = callout_occurrences
                value["total_sup_occurrences"] = (
                    callout_occurrences + definition_occurrences
                )
            else:
                value["callout_markdown_occurrences"] = 0
                value["definition_markdown_occurrences"] = 0
                value["total_sup_occurrences"] = 0
            notes.append(value)
    audit = dict(integration)
    audit["notes"] = notes
    return audit


def join_text_lines(lines: list[PageNode]) -> str:
    if not lines:
        return ""
    value = lines[0].text
    for previous, current in zip(lines, lines[1:]):
        current_text = current.text
        if value.endswith("-") and current_text and current_text[0].islower():
            value = value[:-1] + current_text
        else:
            value += " " + current_text
    return normalize_space(value)


def escape_plain_heading_marker(value: str) -> str:
    """Prevent PDF prose beginning with ``#`` from becoming a heading."""

    match = re.match(r"^(\s*)#{1,6}(?=\s|$)", value)
    if not match:
        return value
    marker_start = len(match.group(1))
    return value[:marker_start] + "\\" + value[marker_start:]


def nodes_to_markdown(
    nodes: list[PageNode],
    page_number: int,
    layout: str | None = None,
) -> str:
    title_node_ids: set[int] = set()
    merged_title = ""
    if page_number == 1:
        title_candidates = [
            node
            for node in nodes
            if node.kind == "text" and node.top < 220 and 6 <= len(node.text) <= 240
        ]
        if title_candidates:
            maximum_size = max(node.font_size for node in title_candidates)
            selected_title_nodes = [
                node
                for node in title_candidates
                if node.font_size >= maximum_size - 0.35
            ]
            selected_title_nodes.sort(key=lambda node: (node.top, node.bbox[0]))
            title_node_ids = {id(node) for node in selected_title_nodes}
            merged_title = join_text_lines(selected_title_nodes)
    chunks: list[str] = []
    paragraph: list[PageNode] = []
    title_emitted = False

    def flush() -> None:
        nonlocal paragraph
        text = join_text_lines(paragraph)
        if text:
            if text.casefold() == "abstract":
                chunks.append("## Abstract")
            else:
                chunks.append(escape_plain_heading_marker(text))
        paragraph = []

    for node in nodes:
        if id(node) in title_node_ids:
            flush()
            if not title_emitted and merged_title:
                chunks.append("# " + merged_title)
                title_emitted = True
            continue
        if node.kind != "text":
            flush()
            chunks.append(node.text.strip())
            continue
        if not paragraph:
            paragraph.append(node)
            continue
        previous = paragraph[-1]
        gap = node.top - previous.bottom
        # A short final line in an otherwise page-wide paragraph can be
        # classified as ``left`` purely because its glyph bbox does not cross
        # the page midpoint.  Lane changes are meaningful paragraph evidence
        # only on an actual multi-column page; on a single-column page they
        # must not split captions or prose.
        same_lane = layout == "single_column" or node.lane == previous.lane
        similar_font = abs(node.font_size - previous.font_size) <= max(1.2, previous.font_size * 0.18)
        indented_new_paragraph = node.bbox[0] - previous.bbox[0] > 12 and previous.text.rstrip().endswith((".", ":", ";", "?", "!"))
        same_source_paragraph = bool(
            node.source_paragraph_slice_id
            and previous.source_paragraph_slice_id
            and node.source_paragraph_slice_id
            == previous.source_paragraph_slice_id
        )
        different_source_paragraph = bool(
            node.source_paragraph_slice_id
            and previous.source_paragraph_slice_id
            and node.source_paragraph_slice_id
            != previous.source_paragraph_slice_id
        )
        if same_source_paragraph:
            paragraph.append(node)
        elif different_source_paragraph:
            flush()
            paragraph.append(node)
        elif same_lane and similar_font and gap <= max(5.5, previous.font_size * 0.75) and not indented_new_paragraph:
            paragraph.append(node)
        else:
            flush()
            paragraph.append(node)
    flush()
    return "\n\n".join(chunk for chunk in chunks if chunk).strip() + "\n"


def normalize_tokens(value: str) -> list[str]:
    value = re.sub(r"<sup>([^<>]+)</sup>", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\\[a-zA-Z@]+", " ", value)
    # PDF text layers preserve visual line-end hyphenation (``rela- tion``),
    # while the Markdown writer intentionally repairs it (``relation``).
    value = re.sub(r"\b([A-Za-z]{2,})-\s+([a-z]{2,})\b", r"\1\2", value)
    return re.findall(r"[A-Za-z0-9]+", value.casefold())


def token_recall(source_text: str, generated_text: str) -> float:
    source = collections.Counter(normalize_tokens(source_text))
    generated = collections.Counter(normalize_tokens(generated_text))
    total = sum(source.values())
    if total == 0:
        return 1.0
    matched = sum(min(count, generated[token]) for token, count in source.items())
    return matched / total


STRICT_TEXT_CONTRACT_VERSION = 2
STRICT_TOKEN_THRESHOLD = 0.995
STRICT_FIVEGRAM_THRESHOLD = 0.99
STRICT_ANCHOR_MONOTONICITY = 1.0


def strict_text_tokens(value: str) -> list[str]:
    """Canonical visible tokens for ordered page-text validation."""

    value = html.unescape(unicodedata.normalize("NFKC", value)).casefold()
    value = re.sub(r"<sup>([^<>]+)</sup>", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\\(?:begin|end)\s*\{[^{}]+\}", " ", value)
    value = re.sub(r"\\(?:tag|label)\s*\{[^{}]+\}", " ", value)
    value = re.sub(r"\\[a-zA-Z@]+", " ", value)
    value = re.sub(r"\b([a-z]{2,})-\s+([a-z]{2,})\b", r"\1\2", value)
    value = re.sub(r"\$\s*([0-9a-z])\s*\$\s*s\b", r"\1s", value)
    return re.findall(r"[a-z0-9]+", value)


def rolling_ngram_hashes(tokens: Sequence[str], width: int = 5) -> list[int]:
    if width <= 0 or len(tokens) < width:
        return []
    mask = (1 << 64) - 1
    base = 1_000_003
    values = [
        int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        for token in tokens
    ]
    highest_power = pow(base, width - 1, 1 << 64)
    rolling = 0
    for value in values[:width]:
        rolling = ((rolling * base) + value) & mask
    output = [rolling]
    for old, new in zip(values, values[width:]):
        rolling = (rolling - ((old * highest_power) & mask)) & mask
        rolling = ((rolling * base) + new) & mask
        output.append(rolling)
    return output


def longest_increasing_subsequence_length(values: Sequence[int]) -> int:
    tails: list[int] = []
    for value in values:
        low, high = 0, len(tails)
        while low < high:
            middle = (low + high) // 2
            if tails[middle] < value:
                low = middle + 1
            else:
                high = middle
        if low == len(tails):
            tails.append(value)
        else:
            tails[low] = value
    return len(tails)


def ordered_text_metrics(expected_text: str, candidate_text: str) -> dict[str, Any]:
    expected_tokens = strict_text_tokens(expected_text)
    candidate_tokens = strict_text_tokens(candidate_text)
    expected_counts = collections.Counter(expected_tokens)
    candidate_counts = collections.Counter(candidate_tokens)
    token_matches = sum(
        min(count, candidate_counts[token]) for token, count in expected_counts.items()
    )
    token_recall_value = token_matches / len(expected_tokens) if expected_tokens else 1.0
    token_precision_value = token_matches / len(candidate_tokens) if candidate_tokens else (
        1.0 if not expected_tokens else 0.0
    )
    expected_grams = rolling_ngram_hashes(expected_tokens)
    candidate_grams = rolling_ngram_hashes(candidate_tokens)
    expected_gram_counts = collections.Counter(expected_grams)
    candidate_gram_counts = collections.Counter(candidate_grams)
    gram_matches = sum(
        min(count, candidate_gram_counts[value])
        for value, count in expected_gram_counts.items()
    )
    gram_recall = gram_matches / len(expected_grams) if expected_grams else 1.0
    gram_precision = gram_matches / len(candidate_grams) if candidate_grams else (
        1.0 if not expected_grams else 0.0
    )
    expected_positions = {
        value: index
        for index, value in enumerate(expected_grams)
        if expected_gram_counts[value] == 1 and candidate_gram_counts[value] == 1
    }
    candidate_positions = {
        value: index
        for index, value in enumerate(candidate_grams)
        if candidate_gram_counts[value] == 1 and expected_gram_counts[value] == 1
    }
    anchor_candidate_positions = [
        candidate_positions[value]
        for value, _ in sorted(expected_positions.items(), key=lambda item: item[1])
    ]
    anchor_lis = longest_increasing_subsequence_length(anchor_candidate_positions)
    anchor_count = len(anchor_candidate_positions)
    anchor_monotonicity = anchor_lis / anchor_count if anchor_count else 1.0
    exact_short_sequence = (
        len(expected_tokens) < 5 and expected_tokens == candidate_tokens
    )
    anchor_evidence = anchor_count > 0 or exact_short_sequence
    passed = (
        token_recall_value >= STRICT_TOKEN_THRESHOLD
        and token_precision_value >= STRICT_TOKEN_THRESHOLD
        and gram_recall >= STRICT_FIVEGRAM_THRESHOLD
        and gram_precision >= STRICT_FIVEGRAM_THRESHOLD
        and anchor_monotonicity == STRICT_ANCHOR_MONOTONICITY
        and anchor_evidence
    )
    return {
        "status": "passed" if passed else "failed",
        "token_expected": len(expected_tokens),
        "token_candidate": len(candidate_tokens),
        "token_matched": token_matches,
        "token_missing": max(0, len(expected_tokens) - token_matches),
        "token_extra": max(0, len(candidate_tokens) - token_matches),
        "token_recall": round(token_recall_value, 6),
        "token_precision": round(token_precision_value, 6),
        "fivegram_expected": len(expected_grams),
        "fivegram_candidate": len(candidate_grams),
        "fivegram_matched": gram_matches,
        "fivegram_missing": max(0, len(expected_grams) - gram_matches),
        "fivegram_extra": max(0, len(candidate_grams) - gram_matches),
        "fivegram_recall": round(gram_recall, 6),
        "fivegram_precision": round(gram_precision, 6),
        "unique_anchor_count": anchor_count,
        "unique_anchor_lis": anchor_lis,
        "anchor_monotonicity": round(anchor_monotonicity, 6),
        "anchor_evidence": anchor_evidence,
        "thresholds": {
            "token_recall": STRICT_TOKEN_THRESHOLD,
            "token_precision": STRICT_TOKEN_THRESHOLD,
            "fivegram_recall": STRICT_FIVEGRAM_THRESHOLD,
            "fivegram_precision": STRICT_FIVEGRAM_THRESHOLD,
            "anchor_monotonicity": STRICT_ANCHOR_MONOTONICITY,
        },
    }


def line_inventory_hash(inventory: Sequence[dict[str, Any]]) -> str:
    serialized = json.dumps(
        list(inventory), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def freeze_line_inventory(
    canonical_nodes: Sequence[PageNode],
    page_height: float,
    page_width: float | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for canonical_index, node in enumerate(canonical_nodes):
        if not node.line_id:
            raise ValueError("raw PDF line missing stable line_id")
        edge = node.top <= max(50.0, page_height * 0.06) or node.bottom >= page_height * 0.94
        page_number_text = bool(re.fullmatch(r"\s*(?:\d+|[ivxlcdm]+)\s*", node.text, re.I))
        centered_bottom_folio = (
            page_width is not None
            and abs(((node.bbox[0] + node.bbox[2]) / 2) - page_width / 2)
            <= page_width * 0.08
            and node.bottom >= page_height * 0.88
        )
        caption = bool(re.match(r"\s*(?:table|fig(?:ure)?)\s+[A-Za-z0-9]", node.text, re.I))
        entries.append(
            {
                "line_id": node.line_id,
                "canonical_index": canonical_index,
                "extraction_order": node.origin_order,
                "origin_page": node.origin_page,
                "lane": node.lane,
                "source_paragraph_id": node.source_paragraph_id,
                "source_paragraph_slice_id": node.source_paragraph_slice_id,
                "bbox": [round(float(value), 3) for value in node.bbox],
                "text": node.text,
                "text_sha256": hashlib.sha256(node.text.encode("utf-8")).hexdigest(),
                "must_preserve": {
                    "header": node.top <= max(50.0, page_height * 0.06),
                    "footer": node.bottom >= page_height * 0.94,
                    "page_number": page_number_text
                    and (edge or centered_bottom_folio),
                    "caption": caption,
                    "page_edge_hyphen": node.text.rstrip().endswith("-")
                    and node.bottom >= page_height * 0.82,
                },
            }
        )
    if entries and entries[-1]["text"].rstrip().endswith("-"):
        entries[-1]["must_preserve"]["page_edge_hyphen"] = True
    return {
        "count": len(entries),
        "sha256": line_inventory_hash(entries),
        "canonical_line_ids": [entry["line_id"] for entry in entries],
        "lines": entries,
    }


def audit_line_claims(
    inventory: dict[str, Any], emitted_nodes: Sequence[PageNode], page_number: int
) -> dict[str, Any]:
    entries = list(inventory.get("lines", []))
    canonical_ids = [str(entry["line_id"]) for entry in entries]
    canonical_index = {line_id: index for index, line_id in enumerate(canonical_ids)}
    origin_pages = {str(entry["line_id"]): entry.get("origin_page") for entry in entries}
    claim_counts: collections.Counter[str] = collections.Counter()
    claims: list[dict[str, Any]] = []
    structural_noncontiguous: list[str] = []
    output_claim_positions: list[int] = []
    flattened_claim_line_ids: list[str] = []
    for output_index, node in enumerate(emitted_nodes):
        line_ids = list(node.claimed_line_ids)
        if not line_ids and not node.derived_line_ids:
            continue
        claim_counts.update(line_ids)
        flattened_claim_line_ids.extend(line_ids)
        positions = sorted(
            canonical_index[line_id]
            for line_id in line_ids
            if line_id in canonical_index
        )
        output_claim_positions.extend(
            canonical_index[line_id]
            for line_id in line_ids
            if line_id in canonical_index
        )
        if (
            node.source_block_id
            and len(positions) > 1
            and any(right != left + 1 for left, right in zip(positions, positions[1:]))
        ):
            structural_noncontiguous.append(node.source_block_id)
        claims.append(
            {
                "claim_id": f"c{output_index:04d}",
                "output_index": output_index,
                "kind": node.kind,
                "source_block_id": node.source_block_id,
                "line_ids": line_ids,
                "derived_line_ids": list(node.derived_line_ids),
                "canonical_indices": positions,
                "inline_source_ids": list(node.inline_source_ids),
                "source_paragraph_id": node.source_paragraph_id,
                "source_paragraph_slice_id": node.source_paragraph_slice_id,
            }
        )
    missing = [line_id for line_id in canonical_ids if claim_counts[line_id] == 0]
    duplicate = [line_id for line_id in canonical_ids if claim_counts[line_id] > 1]
    unknown = sorted(line_id for line_id in claim_counts if line_id not in canonical_index)
    cross_page = sorted(
        line_id
        for line_id in claim_counts
        if line_id in origin_pages and origin_pages[line_id] != page_number
    )
    inversions = [
        [left, right]
        for left, right in zip(output_claim_positions, output_claim_positions[1:])
        if right < left
    ]
    canonical_order_match = flattened_claim_line_ids == canonical_ids
    empty_structural = [
        node.source_block_id
        for node in emitted_nodes
        if node.source_block_id and not node.claimed_line_ids
    ]
    status = "passed"
    if (
        missing
        or duplicate
        or unknown
        or cross_page
        or inversions
        or not canonical_order_match
        or structural_noncontiguous
        or empty_structural
    ):
        status = "failed"
    return {
        "status": status,
        "inventory_count": len(canonical_ids),
        "claimed_unique_count": sum(claim_counts[line_id] == 1 for line_id in canonical_ids),
        "missing_line_ids": missing,
        "duplicate_line_ids": duplicate,
        "unknown_line_ids": unknown,
        "cross_page_line_ids": cross_page,
        "order_inversions": inversions,
        "flattened_claim_line_ids": flattened_claim_line_ids,
        "canonical_order_match": canonical_order_match,
        "noncontiguous_structural_claims": sorted(set(structural_noncontiguous)),
        "empty_structural_claims": sorted(set(value for value in empty_structural if value)),
        "claims": claims,
    }


def project_structural_nodes(
    markdown: str, ordered_nodes: Sequence[PageNode]
) -> tuple[str, str, list[dict[str, Any]]]:
    expected_parts: list[str] = []
    candidate = markdown
    structural_claims: list[dict[str, Any]] = []
    for node in ordered_nodes:
        if node.source_block_id and node.kind != "text":
            sentinel = "ZZSTRUCT" + re.sub(r"[^A-Za-z0-9]", "", node.source_block_id) + "ZZ"
            markup = node.text.strip()
            occurrences = candidate.count(markup) if markup else 0
            structural_claims.append(
                {
                    "block_id": node.source_block_id,
                    "kind": node.kind,
                    "sentinel": sentinel,
                    "markdown_occurrences": occurrences,
                    "line_ids": list(node.claimed_line_ids),
                }
            )
            if occurrences:
                candidate = candidate.replace(markup, sentinel, 1)
            expected_parts.append(sentinel)
        else:
            expected_parts.append(node.text)
    return "\n".join(expected_parts), candidate, structural_claims


def join_inventory_texts(values: Sequence[str]) -> str:
    """Join claimed PDF lines with the same page-local rule as the writer."""

    if not values:
        return ""
    joined = values[0]
    for value in values[1:]:
        if joined.endswith("-") and value and value[0].islower():
            joined = joined[:-1] + value
        else:
            joined += " " + value
    return normalize_space(joined)


def contiguous_token_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    if not needle:
        return True
    width = len(needle)
    return any(list(haystack[index : index + width]) == list(needle) for index in range(len(haystack) - width + 1))


def project_claimed_line_text(
    inventory: dict[str, Any],
    ordered_nodes: Sequence[PageNode],
    inline_blocks: Sequence[InlineSourceBlock],
) -> tuple[str, str, list[str]]:
    """Tie every emitted node's visible text back to its claimed PDF lines.

    The writer-level projection alone is insufficient: a node could claim the
    ID of a missing line while emitting unrelated text.  This projection uses
    the immutable inventory as the expected side.  Confirmed inline source
    substitutions are represented by the same sentinel on both sides.
    """

    inventory_by_id = {
        str(entry["line_id"]): entry for entry in inventory.get("lines", [])
    }
    inline_by_id = {block.block_id: block for block in inline_blocks}
    expected_parts: list[str] = []
    candidate_parts: list[str] = []
    failures: list[str] = []
    for output_index, node in enumerate(ordered_nodes):
        if node.source_block_id and node.kind != "text":
            sentinel = (
                "ZZSTRUCT"
                + re.sub(r"[^A-Za-z0-9]", "", node.source_block_id)
                + "ZZ"
            )
            expected_parts.append(sentinel)
            candidate_parts.append(sentinel)
            continue
        if node.claimed_line_ids:
            unknown = [
                line_id
                for line_id in node.claimed_line_ids
                if line_id not in inventory_by_id
            ]
            if unknown:
                failures.append(f"unknown_claim_text:{output_index}")
                continue
            absorbed_line_ids = {
                line_id
                for source_id in node.inline_source_ids
                for line_id in (
                    inline_by_id[source_id].absorbed_pdf_line_ids
                    if source_id in inline_by_id
                    else []
                )
            }
            expected = join_inventory_texts(
                [
                    str(inventory_by_id[line_id]["text"])
                    for line_id in node.claimed_line_ids
                    if line_id not in absorbed_line_ids
                ]
            )
            candidate = node.text
            for source_id in node.inline_source_ids:
                block = inline_by_id.get(source_id)
                if block is None or block.match_status != "matched":
                    failures.append(f"inline_projection_block_missing:{source_id}")
                    continue
                pdf_span = block.matched_pdf_text or ""
                markdown_span = block.enriched_markdown or ""
                sentinel = "ZZINLINE" + re.sub(r"[^A-Za-z0-9]", "", source_id) + "ZZ"
                if not pdf_span or expected.count(pdf_span) != 1:
                    failures.append(f"inline_pdf_span_not_exact_once:{source_id}")
                else:
                    expected = expected.replace(pdf_span, sentinel, 1)
                if not markdown_span or candidate.count(markdown_span) != 1:
                    failures.append(f"inline_markup_span_not_exact_once:{source_id}")
                else:
                    candidate = candidate.replace(markdown_span, sentinel, 1)
            expected_parts.append(expected)
            candidate_parts.append(candidate)
            continue
        if node.derived_line_ids:
            absorbed_line_ids = {
                line_id
                for source_id in node.inline_source_ids
                for line_id in (
                    inline_by_id[source_id].absorbed_pdf_line_ids
                    if source_id in inline_by_id
                    else []
                )
            }
            source_text = join_inventory_texts(
                [
                    str(inventory_by_id[line_id]["text"])
                    for line_id in node.derived_line_ids
                    if line_id in inventory_by_id and line_id not in absorbed_line_ids
                ]
            )
            candidate_text = node.text
            for source_id in node.inline_source_ids:
                block = inline_by_id.get(source_id)
                if block is None or block.match_status != "matched":
                    failures.append(f"derived_inline_block_missing:{source_id}")
                    continue
                pdf_span = block.matched_pdf_text or ""
                markdown_span = block.enriched_markdown or ""
                sentinel = (
                    "ZZINLINE" + re.sub(r"[^A-Za-z0-9]", "", source_id) + "ZZ"
                )
                if not pdf_span or source_text.count(pdf_span) != 1:
                    failures.append(f"derived_inline_pdf_span_not_exact_once:{source_id}")
                else:
                    source_text = source_text.replace(pdf_span, sentinel, 1)
                if not markdown_span or candidate_text.count(markdown_span) != 1:
                    failures.append(
                        f"derived_inline_markup_span_not_exact_once:{source_id}"
                    )
                else:
                    candidate_text = candidate_text.replace(
                        markdown_span, sentinel, 1
                    )
            if not contiguous_token_subsequence(
                strict_text_tokens(candidate_text), strict_text_tokens(source_text)
            ):
                failures.append(f"derived_text_not_in_source:{output_index}")
            expected_parts.append(node.text)
            candidate_parts.append(node.text)
            continue
        failures.append(f"output_node_without_provenance:{output_index}")
        expected_parts.append(node.text)
        candidate_parts.append(node.text)
    return "\n".join(expected_parts), "\n".join(candidate_parts), sorted(set(failures))


UNSTRUCTURED_MATH_MARKER = re.compile(
    r"\(cid:\d+\)|\|=|⇐⇒|[=∑∫≤≥→∞∈∀∃¬∧∨⊂⊆⊨⊭⊢Γηλμσπϕι˓︀︁]"
)


def unstructured_math_fragment_issues(
    ordered_nodes: Sequence[PageNode],
) -> list[str]:
    """Reject visible PDF math that was not replaced from an exact source span.

    The PDF text layer may split a formula into many delayed sub/superscript
    lines.  Exact line coverage alone would happily emit those fragments and
    call the page strict.  A strict page therefore requires every math-looking
    text node to belong to an accepted inline source replacement or a source
    structural node.  Small overlapping satellite glyphs are also rejected,
    even when their text is only an ASCII subscript such as ``DM`` or ``i``.
    """

    text_nodes = [node for node in ordered_nodes if node.kind == "text"]
    body_fonts = [
        node.font_size
        for node in text_nodes
        if len(normalize_tokens(node.text)) >= 6
    ]
    body_font = statistics.median(body_fonts) if body_fonts else 0.0
    issues: list[str] = []
    for node in text_nodes:
        if node.inline_source_ids:
            continue
        line_id = node.line_id or (
            node.claimed_line_ids[0] if node.claimed_line_ids else "unknown"
        )
        if UNSTRUCTURED_MATH_MARKER.search(node.text):
            issues.append(f"unstructured_math_text:{line_id}")
            continue
        tokens = normalize_tokens(node.text)
        if (
            not body_font
            or not tokens
            or len(tokens) > 3
            or node.font_size > body_font * 0.9
        ):
            continue
        overlaps_baseline = any(
            other is not node
            and max(0.0, min(node.bottom, other.bottom) - max(node.top, other.top))
            / max(0.1, node.bottom - node.top)
            >= 0.3
            for other in text_nodes
        )
        if overlaps_baseline:
            issues.append(f"unstructured_math_satellite:{line_id}")
    return sorted(set(issues))


def build_strict_text_v2_contract(
    *,
    markdown: str,
    ordered_nodes: Sequence[PageNode],
    inventory: dict[str, Any],
    page_number: int,
    layout: str,
    page_blocks: Sequence[SourceBlock],
    inline_blocks: Sequence[InlineSourceBlock],
    punctuation_issues: Sequence[str] = (),
    footnote_audit: dict[str, Any] | None = None,
    author_superscript_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_rank = {
        str(line_id): index
        for index, line_id in enumerate(inventory.get("canonical_line_ids", []))
    }
    # PDF formula/table text layers often enumerate glyph fragments in x/y
    # extraction order rather than semantic reading order.  A structured node
    # owns the whole contiguous region, so normalize only its internal claim
    # list to the already-frozen canonical order.  This does not change the
    # claimed set and cannot hide a missing, duplicate, or non-contiguous line.
    for node in ordered_nodes:
        if node.source_block_id and node.kind != "text":
            node.claimed_line_ids = sorted(
                node.claimed_line_ids,
                key=lambda line_id: canonical_rank.get(line_id, math.inf),
            )
    expected_projection, candidate_projection, structural_claims = project_structural_nodes(
        markdown, ordered_nodes
    )
    claimed_expected, claimed_candidate, claim_text_failures = project_claimed_line_text(
        inventory, ordered_nodes, inline_blocks
    )
    claims = audit_line_claims(inventory, ordered_nodes, page_number)
    ordered_metrics = ordered_text_metrics(expected_projection, candidate_projection)
    claimed_line_text_metrics = ordered_text_metrics(
        claimed_expected, claimed_candidate
    )
    math_fragment_issues = unstructured_math_fragment_issues(ordered_nodes)
    inline_claims: list[dict[str, Any]] = []
    for block in inline_blocks:
        if block.match_status != "matched":
            continue
        node_line_ids = sorted(
            {
                line_id
                for node in ordered_nodes
                if block.block_id in node.inline_source_ids
                for line_id in [*node.claimed_line_ids, *node.derived_line_ids]
            }
        )
        enriched = block.enriched_markdown or ""
        inline_claims.append(
            {
                "block_id": block.block_id,
                "line_ids": node_line_ids,
                "provenance_status": "propagated" if node_line_ids else "missing",
                "markdown_occurrences": markdown.count(enriched) if enriched else 0,
            }
        )
    must_preserve = {
        key: [
            entry["line_id"]
            for entry in inventory.get("lines", [])
            if entry.get("must_preserve", {}).get(key)
        ]
        for key in ("caption", "header", "footer", "page_number", "page_edge_hyphen")
    }
    failures: list[str] = []
    failures.extend(str(issue) for issue in punctuation_issues)
    if claims["status"] != "passed":
        failures.append("line_claim_audit_failed")
    if ordered_metrics["status"] != "passed":
        failures.append("ordered_text_metrics_failed")
    if claimed_line_text_metrics["status"] != "passed":
        failures.append("claimed_line_text_metrics_failed")
    failures.extend(claim_text_failures)
    failures.extend(math_fragment_issues)
    if markdown_inline_syntax_issues(markdown):
        failures.append("inline_markdown_syntax_invalid")
    if re.search(r"\(cid:\d+\)", markdown):
        failures.append("pdf_cid_placeholder_present")
    for structural in structural_claims:
        if structural["markdown_occurrences"] != 1:
            failures.append(f"structural_sentinel_not_exact_once:{structural['block_id']}")
    for inline_claim in inline_claims:
        if inline_claim["provenance_status"] != "propagated":
            failures.append(f"inline_line_claim_missing:{inline_claim['block_id']}")
        if inline_claim["markdown_occurrences"] != 1:
            failures.append(f"inline_sentinel_not_exact_once:{inline_claim['block_id']}")
    normalized_footnote_audit = footnote_audit or {
        "status": "passed",
        "representation": FOOTNOTE_REPRESENTATION,
        "total": 0,
        "structured": 0,
        "fallback": 0,
        "notes": [],
    }
    normalized_author_superscript_audit = author_superscript_audit or {
        "contract_version": AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
        "status": "not_applicable" if page_number != 1 else "not_present",
        "plans": 0,
        "superscripts_emitted": 0,
        "markers": [],
        "matched_line_ids": [],
        "marked_lines": [],
    }
    footnote_notes = normalized_footnote_audit.get("notes", [])
    if normalized_footnote_audit.get("representation") != FOOTNOTE_REPRESENTATION:
        failures.append("footnote_representation_invalid")
    if re.search(r"\[\^[0-9]{1,4}\](?::)?", markdown):
        failures.append("legacy_markdown_footnote_syntax_present")
    superscript_values = re.findall(r"<sup>([^<>]+)</sup>", markdown)
    if (
        markdown.count("<sup>") != len(superscript_values)
        or markdown.count("</sup>") != len(superscript_values)
    ):
        failures.append("sup_html_invalid")
    for value in superscript_values:
        if AUTHOR_SUPERSCRIPT_SAFE_MARKER_PATTERN.fullmatch(value) is None:
            failures.append("sup_html_value_unsafe")
    if not isinstance(footnote_notes, list):
        failures.append("footnote_audit_notes_invalid")
        footnote_notes = []
    if int(normalized_footnote_audit.get("total", 0)) != len(footnote_notes):
        failures.append("footnote_audit_total_mismatch")
    author_markers = normalized_author_superscript_audit.get("markers", [])
    if not isinstance(author_markers, list):
        failures.append("author_superscript_markers_invalid")
        author_markers = []
    author_emitted = int(
        normalized_author_superscript_audit.get("superscripts_emitted", 0) or 0
    )
    author_status = normalized_author_superscript_audit.get("status")
    author_plans = int(normalized_author_superscript_audit.get("plans", 0) or 0)
    if normalized_author_superscript_audit.get("contract_version") != AUTHOR_SUPERSCRIPT_CONTRACT_VERSION:
        failures.append("author_superscript_contract_version_invalid")
    if page_number == 1 and author_plans and author_status != "passed":
        failures.append("author_superscript_unresolved")
    if author_emitted != len(author_markers):
        failures.append("author_superscript_count_mismatch")
    for marker in author_markers:
        if not isinstance(marker, str) or AUTHOR_SUPERSCRIPT_SAFE_MARKER_PATTERN.fullmatch(marker) is None:
            failures.append("author_superscript_marker_invalid")
    marked_lines = normalized_author_superscript_audit.get("marked_lines", [])
    if not isinstance(marked_lines, list):
        failures.append("author_superscript_marked_lines_invalid")
        marked_lines = []
    for marked_line in marked_lines:
        if not isinstance(marked_line, str) or markdown.count(marked_line) != 1:
            failures.append("author_superscript_marked_line_not_exact_once")
    matched_author_line_ids = normalized_author_superscript_audit.get(
        "matched_line_ids", []
    )
    if not isinstance(matched_author_line_ids, list) or any(
        line_id not in canonical_rank for line_id in matched_author_line_ids
    ):
        failures.append("author_superscript_line_claim_invalid")
    if len(superscript_values) != 2 * len(footnote_notes) + author_emitted:
        failures.append("page_sup_count_mismatch")
    if normalized_footnote_audit.get("status") != "passed" or int(
        normalized_footnote_audit.get("fallback", 0)
    ):
        failures.append("footnote_structuring_unresolved")
    markers: list[str] = []
    for note in footnote_notes:
        if not isinstance(note, dict):
            failures.append("footnote_audit_note_invalid")
            continue
        note_id = str(note.get("note_id", "missing"))
        marker = str(note.get("marker", ""))
        markers.append(marker)
        if re.fullmatch(r"[0-9]{1,4}", marker, flags=re.ASCII) is None:
            failures.append(f"footnote_marker_invalid:{note_id}")
        if note.get("status") != "structured":
            failures.append(f"footnote_not_structured:{note_id}")
        if note.get("content_validation_status") != "passed" or note.get(
            "content_validation_issues"
        ):
            failures.append(f"footnote_body_validation_failed:{note_id}")
        callout_line_ids = note.get("callout_line_ids", [])
        definition_line_ids = note.get("definition_line_ids", [])
        if not isinstance(callout_line_ids, list) or not callout_line_ids:
            failures.append(f"footnote_callout_claim_missing:{note_id}")
        if not isinstance(definition_line_ids, list) or not definition_line_ids:
            failures.append(f"footnote_definition_claim_missing:{note_id}")
        if set(callout_line_ids) & set(definition_line_ids):
            failures.append(f"footnote_callout_definition_claim_overlap:{note_id}")
        if note.get("callout_markdown_occurrences") != 1:
            failures.append(f"footnote_callout_not_exact_once:{note_id}")
        if note.get("definition_markdown_occurrences") != 1:
            failures.append(f"footnote_definition_not_exact_once:{note_id}")
        if note.get("total_sup_occurrences") != 2:
            failures.append(f"footnote_sup_not_exactly_twice:{note_id}")
        if note.get("representation") != FOOTNOTE_REPRESENTATION:
            failures.append(f"footnote_note_representation_invalid:{note_id}")
    if len(set(markers)) != len(markers):
        failures.append("footnote_markers_not_unique")
    for block in inline_blocks:
        if block.target_feature_total and block.match_status != "matched":
            failures.append(f"inline_enrichment_unresolved:{block.block_id}")
    claimed_ids = {
        line_id for claim in claims["claims"] for line_id in claim["line_ids"]
    }
    absorbed_line_owners: collections.Counter[str] = collections.Counter(
        line_id
        for block in inline_blocks
        for line_id in block.absorbed_pdf_line_ids
    )
    for line_id, owner_count in absorbed_line_owners.items():
        if owner_count != 1:
            failures.append(f"inline_absorbed_line_owner_count:{line_id}")
        if line_id not in claimed_ids:
            failures.append(f"inline_absorbed_line_unclaimed:{line_id}")
    caption_line_ids = set(must_preserve["caption"])
    if not caption_line_ids.issubset(claimed_ids):
        failures.append("required_caption_unclaimed")
    claim_by_line_id = {
        line_id: claim
        for claim in claims["claims"]
        for line_id in claim["line_ids"]
    }
    for preservation_kind in ("header", "footer", "page_number", "page_edge_hyphen"):
        for line_id in must_preserve[preservation_kind]:
            claim = claim_by_line_id.get(line_id)
            if claim and claim.get("source_block_id"):
                failures.append(
                    f"must_preserve_consumed_by_structure:{preservation_kind}:{line_id}"
                )
    inventory_by_id = {
        str(entry["line_id"]): entry for entry in inventory.get("lines", [])
    }
    node_by_claimed_line_id = {
        line_id: node
        for node in ordered_nodes
        for line_id in node.claimed_line_ids
    }
    for line_id in must_preserve["caption"]:
        claim = claim_by_line_id.get(line_id)
        if not claim or not claim.get("source_block_id"):
            continue
        original = str(inventory_by_id.get(line_id, {}).get("text", ""))
        if not (
            claim.get("kind") == "table"
            and re.match(r"\s*table\s+", original, re.IGNORECASE)
        ):
            failures.append(f"caption_consumed_by_wrong_structure:{line_id}")
    for line_id in must_preserve["page_edge_hyphen"]:
        node = node_by_claimed_line_id.get(line_id)
        original = str(inventory_by_id.get(line_id, {}).get("text", "")).rstrip()
        visible_suffix = original.rsplit(None, 1)[-1] if original else ""
        if not node or not visible_suffix.endswith("-") or visible_suffix not in node.text:
            failures.append(f"page_edge_hyphen_changed:{line_id}")
    for block in page_blocks:
        if block.kind == "heading" and block.heading_number_status in {
            "lost",
            "wrong",
            "ambiguous",
        }:
            failures.append(f"heading_number_not_preserved:{block.block_id}")
        elif block.kind == "table":
            if block.table_parse_status != "parsed":
                failures.append(f"table_not_structurally_parsed:{block.block_id}")
            if not block.table_html or not block.table_html.lstrip().startswith("<table>"):
                failures.append(f"table_root_not_plain:{block.block_id}")
            if re.search(r"<table\s+[^>]*>", block.markdown, re.IGNORECASE):
                failures.append(f"table_root_has_attributes:{block.block_id}")
            if re.search(r"<caption\b", block.markdown, re.IGNORECASE):
                failures.append(f"table_caption_embedded_in_html:{block.block_id}")
            if not block.caption_markdown and block.pdf_visible_caption:
                failures.append(f"table_caption_missing:{block.block_id}")
            if block.caption_markdown and block.caption_number_status not in {
                "preserved",
                "visible_unnumbered",
            }:
                failures.append(f"table_number_not_preserved:{block.block_id}")
            if block.pdf_visible_caption:
                emitted_caption_tokens = strict_text_tokens(
                    block.caption_markdown or ""
                )
                if strict_text_tokens(block.pdf_visible_caption) != emitted_caption_tokens:
                    failures.append(f"table_caption_text_mismatch:{block.block_id}")
        elif block.kind == "display_math" and block.formula_number_status in {
            "ambiguous",
            "wrong",
            "unsafe_to_tag",
        }:
            failures.append(f"formula_number_not_preserved:{block.block_id}")
    failures = sorted(set(failures))
    return {
        "strict_text_contract_version": STRICT_TEXT_CONTRACT_VERSION,
        "strict_text_v2_status": "passed" if not failures else "failed",
        "contract": {
            "canonical_order_frozen_before_replacement": True,
            "original_line_claim": "exactly_once",
            "structural_claim": "exactly_once_contiguous",
            "inline_claim": "exactly_once_with_line_provenance",
            "graphics_structure_ignored": True,
            "ignored_graphic": 0,
            "captions_required": True,
            "headers_footers_page_numbers_required": True,
            "page_edge_hyphen_visible_form_required": True,
            "strict_punctuation_hard_gate": True,
            "strict_footnote_structure_hard_gate": True,
            "footnote_representation": FOOTNOTE_REPRESENTATION,
            "strict_author_superscript_hard_gate": True,
            "author_superscript_contract_version": AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
            "author_superscript_representation": "html_sup",
        },
        "line_inventory": inventory,
        "canonical_line_ids": list(inventory.get("canonical_line_ids", [])),
        "canonical_layout": layout,
        "coverage_counts": {
            key: ordered_metrics[key]
            for key in (
                "token_expected",
                "token_candidate",
                "token_matched",
                "token_missing",
                "token_extra",
                "fivegram_expected",
                "fivegram_candidate",
                "fivegram_matched",
                "fivegram_missing",
                "fivegram_extra",
            )
        },
        "claims": claims,
        "structural_claims": structural_claims,
        "inline_claims": inline_claims,
        "ordered_metrics": ordered_metrics,
        "claimed_line_text_metrics": claimed_line_text_metrics,
        "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "must_preserve": must_preserve,
        "punctuation_issues": sorted(set(str(issue) for issue in punctuation_issues)),
        "unstructured_math_fragment_issues": math_fragment_issues,
        "footnotes": normalized_footnote_audit,
        "author_superscripts": normalized_author_superscript_audit,
        "failure_reasons": failures,
    }


def render_page_png(pdf_path: Path, page_number: int, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = output_path.with_suffix("")
    command = [
        str(PDFTOPPM),
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-r",
        str(dpi),
        "-png",
        "-singlefile",
        str(pdf_path),
        str(prefix),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
    if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"pdftoppm failed for page {page_number}: {completed.stderr.decode(errors='replace')}")


def load_page_checkpoint(
    paper_output: Path,
    *,
    paper: dict[str, Any],
    page_number: int,
) -> dict[str, Any] | None:
    """Return a complete page sidecar when all emitted artifacts still agree."""

    pages_dir = paper_output / "pages"
    stem = f"page_{page_number:04d}"
    json_path = pages_dir / f"{stem}.json"
    markdown_path = pages_dir / f"{stem}.md"
    image_path = pages_dir / f"{stem}.png"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (json_path, markdown_path, image_path)):
        return None
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_id = f"{paper['stem']}_page_{page_number:04d}"
    if metadata.get("data_id") != expected_id:
        return None
    if metadata.get("strict_text_contract_version") != STRICT_TEXT_CONTRACT_VERSION:
        return None
    if (
        metadata.get("source_paragraph_contract_version")
        != SOURCE_PARAGRAPH_CONTRACT_VERSION
    ):
        return None
    if metadata.get("footnote_representation") != FOOTNOTE_REPRESENTATION:
        return None
    if (
        page_number == 1
        and metadata.get("author_superscript_contract_version")
        != AUTHOR_SUPERSCRIPT_CONTRACT_VERSION
    ):
        return None
    expected_relative = Path("papers") / paper["stem"] / "pages" / stem
    if metadata.get("image") != expected_relative.with_suffix(".png").as_posix():
        return None
    if metadata.get("markdown") != expected_relative.with_suffix(".md").as_posix():
        return None
    markdown = markdown_path.read_text(encoding="utf-8")
    if metadata.get("markdown_sha256") != hashlib.sha256(markdown.encode("utf-8")).hexdigest():
        return None
    return metadata


def load_complete_paper_checkpoint(
    paper_output: Path,
    *,
    paper: dict[str, Any],
    page_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Fast-resume a fully emitted paper before expensive source mapping.

    Every page still passes the ordinary checkpoint contract and the paper's
    source-provenance manifests must exist.  Partially emitted papers always
    fall back to the normal compile/map/page path.
    """

    summary_path = paper_output / "paper_summary.json"
    provenance_paths = [
        paper_output / "source_blocks.jsonl",
        paper_output / "source_paragraphs.jsonl",
        paper_output / "inline_source_blocks.jsonl",
        paper_output / "inline_source_rejections.jsonl",
        paper_output / "author_superscript_plans.jsonl",
    ]
    if not summary_path.is_file() or not all(path.is_file() for path in provenance_paths):
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("stem") != paper.get("stem"):
        return None
    if int(summary.get("pages_emitted", -1)) != page_limit:
        return None
    if summary.get("compile", {}).get("status") == "failed":
        return None
    pages: list[dict[str, Any]] = []
    for page_number in range(1, page_limit + 1):
        checkpoint = load_page_checkpoint(
            paper_output,
            paper=paper,
            page_number=page_number,
        )
        if checkpoint is None:
            return None
        pages.append(checkpoint)
    return pages, summary


def build_page(
    page: Any,
    *,
    paper: dict[str, Any],
    page_number: int,
    source_blocks: list[SourceBlock],
    inline_blocks: list[InlineSourceBlock],
    paper_output: Path,
    pdf_path: Path,
    dpi: int,
    source_hyphenated_terms: Sequence[str] = (),
    source_paragraph_points: Sequence[SourceParagraphPoint] = (),
    source_paragraphs_by_id: dict[str, SourceParagraph] | None = None,
    author_superscript_plans: Sequence[AuthorSuperscriptPlan] = (),
) -> dict[str, Any]:
    line_nodes = words_to_line_nodes(page)
    source_paragraph_integration = annotate_source_paragraph_ids(
        line_nodes,
        source_paragraph_points,
        source_paragraphs_by_id or {},
    )
    canonical_nodes, layout_hint = order_page_nodes(line_nodes, float(page.width))
    line_inventory = freeze_line_inventory(
        canonical_nodes,
        float(page.height),
        float(page.width),
    )
    page_inline_blocks = [
        block for block in inline_blocks if block.page == page_number and block.bbox
    ]
    enriched_lines, _line_inline_integration = apply_inline_source_blocks(
        line_nodes,
        page_inline_blocks,
        float(page.width),
        layout_hint=layout_hint,
    )
    enriched_lines, author_superscript_audit = apply_author_superscript_plans(
        enriched_lines,
        author_superscript_plans,
        page_number=page_number,
        page_width=float(page.width),
    )
    page_blocks = [block for block in source_blocks if block.page == page_number and block.bbox]
    retained_lines, structured_nodes, integration = integrate_source_blocks(
        enriched_lines, page_blocks, float(page.width)
    )
    ordered_nodes, layout = order_page_nodes(
        retained_lines + structured_nodes,
        float(page.width),
        layout_hint=layout_hint,
    )
    ordered_nodes, footnote_integration = integrate_footnote_definitions(
        ordered_nodes,
        page_inline_blocks,
        line_inventory,
        page_number,
    )
    emitted_inline_source_ids = {
        source_id
        for node in ordered_nodes
        for source_id in node.inline_source_ids
    }
    line_emission_reconciliation = reconcile_emitted_inline_blocks(
        page_inline_blocks,
        emitted_inline_source_ids,
    )
    markdown = nodes_to_markdown(ordered_nodes, page_number, layout=layout)
    markdown, markdown_inline_integration = apply_inline_blocks_to_markdown(
        markdown, page_inline_blocks
    )
    markdown, source_first_inline_integration = apply_source_first_inline_blocks_to_markdown(
        markdown, page_inline_blocks
    )
    pre_retry_reconciliation = reconcile_final_inline_markup(
        markdown, page_inline_blocks
    )
    markdown, retry_markdown_inline_integration = apply_inline_blocks_to_markdown(
        markdown, page_inline_blocks
    )
    markdown, retry_source_first_inline_integration = apply_source_first_inline_blocks_to_markdown(
        markdown, page_inline_blocks
    )
    final_markup_reconciliation = reconcile_final_inline_markup(
        markdown, page_inline_blocks
    )
    footnote_audit = finalize_footnote_audit(
        markdown, page_inline_blocks, footnote_integration
    )
    inline_syntax_issues = markdown_inline_syntax_issues(markdown)
    cid_placeholders = len(re.findall(r"\(cid:\d+\)", markdown))
    inline_integration = inline_integration_summary(page_inline_blocks)
    inline_integration.update(line_emission_reconciliation)
    inline_integration.update(markdown_inline_integration)
    inline_integration.update(source_first_inline_integration)
    inline_integration.update(
        {
            "pre_retry_final_markup_claims_checked": pre_retry_reconciliation[
                "final_markup_claims_checked"
            ],
            "pre_retry_final_markup_claims_present": pre_retry_reconciliation[
                "final_markup_claims_present"
            ],
            "pre_retry_final_markup_claims_missing": pre_retry_reconciliation[
                "final_markup_claims_missing"
            ],
            "retry_matched_by_unique_markdown_alignment": retry_markdown_inline_integration[
                "matched_by_unique_markdown_alignment"
            ],
            "retry_ambiguous_markdown_alignments": retry_markdown_inline_integration[
                "ambiguous_markdown_alignments"
            ],
            "retry_matched_by_unique_source_first_alignment": retry_source_first_inline_integration[
                "matched_by_unique_source_first_alignment"
            ],
            "retry_ambiguous_source_first_alignments": retry_source_first_inline_integration[
                "ambiguous_source_first_alignments"
            ],
            **final_markup_reconciliation,
        }
    )
    punctuation_issues = strict_punctuation_issues(
        markdown, source_hyphenated_terms
    )
    strict_text_v2 = build_strict_text_v2_contract(
        markdown=markdown,
        ordered_nodes=ordered_nodes,
        inventory=line_inventory,
        page_number=page_number,
        layout=layout,
        page_blocks=page_blocks,
        inline_blocks=page_inline_blocks,
        punctuation_issues=punctuation_issues,
        footnote_audit=footnote_audit,
        author_superscript_audit=author_superscript_audit,
    )
    relative_base = Path("papers") / paper["stem"] / "pages" / f"page_{page_number:04d}"
    markdown_path = paper_output / "pages" / f"page_{page_number:04d}.md"
    json_path = paper_output / "pages" / f"page_{page_number:04d}.json"
    image_path = paper_output / "pages" / f"page_{page_number:04d}.png"
    atomic_write_text(markdown_path, markdown)
    render_page_png(pdf_path, page_number, image_path, dpi)
    pdf_visible_text = " ".join(node.text for node in line_nodes)
    # Compare against PDF lines in the exact reading order used by the writer;
    # y/x extraction order interleaves columns and can falsely join a left-line
    # hyphen with the first word of the right column.
    retained_visible_text = " ".join(
        node.text for node in ordered_nodes if node.source_block_id is None
    )
    full_coverage = token_recall(pdf_visible_text, markdown)
    retained_coverage = token_recall(retained_visible_text, markdown)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "data_id": f"{paper['stem']}_page_{page_number:04d}",
        "arxiv_id": paper["arxiv_id"],
        "version": paper["version"],
        "title": paper["title"],
        "license_name": paper["license_name"],
        "page_number": page_number,
        "page_size_points": {"width": round(float(page.width), 3), "height": round(float(page.height), 3)},
        "layout": layout,
        "image": relative_base.with_suffix(".png").as_posix(),
        "markdown": relative_base.with_suffix(".md").as_posix(),
        "source_pdf": str(pdf_path),
        "source_blocks": [block.as_json(Path(paper["source_dir"]).resolve()) for block in page_blocks],
        "inline_source_blocks": [
            block.as_json(Path(paper["source_dir"]).resolve())
            for block in page_inline_blocks
        ],
        "block_counts": dict(collections.Counter(block.kind for block in page_blocks)),
        "pdf_text_line_count": len(line_nodes),
        "retained_text_line_count": len(retained_lines),
        "pdf_text_token_recall": round(full_coverage, 6),
        "retained_pdf_text_token_recall": round(retained_coverage, 6),
        "source_integration": integration,
        "source_paragraph_contract_version": SOURCE_PARAGRAPH_CONTRACT_VERSION,
        "source_paragraph_integration": source_paragraph_integration,
        "inline_source_integration": inline_integration,
        "footnotes": footnote_audit,
        "footnote_representation": FOOTNOTE_REPRESENTATION,
        "author_superscripts": author_superscript_audit,
        "author_superscript_contract_version": AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
        "table_representation": "html",
        "formula_representation": "latex_math_markdown",
        "inline_markup_validation": {
            "status": "passed"
            if not inline_syntax_issues and cid_placeholders == 0
            else "failed",
            "syntax_issues": inline_syntax_issues,
            "cid_placeholders": cid_placeholders,
        },
        "strict_text_contract_version": STRICT_TEXT_CONTRACT_VERSION,
        "strict_text_contract": strict_text_v2["contract"],
        "strict_punctuation_issues": strict_text_v2["punctuation_issues"],
        "source_hyphenated_terms_count": len(source_hyphenated_terms),
        "source_hyphenated_terms_sha256": hashlib.sha256(
            "\n".join(source_hyphenated_terms).encode("utf-8")
        ).hexdigest(),
        "line_inventory": strict_text_v2["line_inventory"],
        "line_inventory_hash": line_inventory["sha256"],
        "line_inventory_count": line_inventory["count"],
        "canonical_line_ids": strict_text_v2["canonical_line_ids"],
        "canonical_layout": strict_text_v2["canonical_layout"],
        "strict_text_coverage_counts": strict_text_v2["coverage_counts"],
        "strict_text_claims": strict_text_v2["claims"],
        "strict_text_structural_claims": strict_text_v2["structural_claims"],
        "strict_text_inline_claims": strict_text_v2["inline_claims"],
        "strict_text_ordered_metrics": strict_text_v2["ordered_metrics"],
        "strict_text_claimed_line_metrics": strict_text_v2[
            "claimed_line_text_metrics"
        ],
        "markdown_sha256": strict_text_v2["markdown_sha256"],
        "must_preserve": strict_text_v2["must_preserve"],
        "strict_text_v2_failure_reasons": strict_text_v2["failure_reasons"],
        "strict_text_v2_status": strict_text_v2["strict_text_v2_status"],
        "validation_status": (
            "passed"
            if markdown.strip()
            and retained_coverage >= 0.98
            and inline_integration["features_unresolved_total"] == 0
            and not inline_syntax_issues
            and cid_placeholders == 0
            and footnote_audit["fallback"] == 0
            and integration["heading_numbering"]["strict"]
            else "warning"
        ),
    }
    atomic_write_json(json_path, metadata)
    return metadata


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return values


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paper-ids", nargs="*", default=[])
    parser.add_argument("--max-papers", type=int, default=2)
    parser.add_argument("--max-pages-per-paper", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-page-warnings",
        action="store_true",
        help=(
            "return success when every target page was emitted without execution errors; "
            "warning pages remain excluded from strict manifests"
        ),
    )
    return parser.parse_args()


def validate_preflight(args: argparse.Namespace) -> tuple[Path, Path]:
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    if not (input_root / "results.jsonl").is_file():
        raise FileNotFoundError(input_root / "results.jsonl")
    for tool in (TEX_BIN / "latexmk", TEX_BIN / "synctex", PDFTOPPM, PDFTOTEXT):
        if not tool.is_file():
            raise FileNotFoundError(tool)
    if args.max_papers <= 0 or args.max_pages_per_paper <= 0 or args.dpi <= 0:
        raise ValueError("max-papers, max-pages-per-paper, and dpi must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_root, output_dir


def main() -> int:
    args = parse_args()
    input_root, output_dir = validate_preflight(args)
    all_papers = read_jsonl(input_root / "results.jsonl")
    requested = set(args.paper_ids)
    eligible = [
        paper
        for paper in all_papers
        if paper.get("status") == "success"
        and paper.get("safety_scan", {}).get("status") == "passed"
        and (not requested or paper.get("stem") in requested or paper.get("arxiv_id") in requested)
    ]
    if requested:
        found = {paper["stem"] for paper in eligible} | {paper["arxiv_id"] for paper in eligible}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"requested papers not eligible/found: {missing}")
    papers = eligible[: args.max_papers]
    if not papers:
        raise RuntimeError("no eligible papers")
    started = time.monotonic()
    print(
        f"[start] papers={len(papers)} max_pages_per_paper={args.max_pages_per_paper} "
        f"dpi={args.dpi} output={output_dir}",
        flush=True,
    )
    generation_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "paper_ids": [paper["stem"] for paper in papers],
        "max_pages_per_paper": args.max_pages_per_paper,
        "dpi": args.dpi,
        "table_representation": "html",
        "formula_representation": "latex_math_markdown",
        "inline_representation": "pdf_baseline_with_latex_markup_enrichment",
        "footnote_representation": FOOTNOTE_REPRESENTATION,
        "author_superscript_representation": "html_sup_source_authoritative",
        "author_superscript_contract_version": AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
        "mapping": "synctex_plus_pdf_text_layer",
    }
    atomic_write_json(output_dir / "generation_config.json", generation_config)
    page_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    accepted = rejected = errors = 0
    resumed_pages = 0
    total_target = sum(
        min(args.max_pages_per_paper, int(paper.get("pdf_inspection", {}).get("pages", args.max_pages_per_paper)))
        for paper in papers
    )
    for paper_index, paper in enumerate(papers, start=1):
        paper_started = time.monotonic()
        print(f"[paper_start] {paper_index}/{len(papers)} id={paper['stem']}", flush=True)
        paper_output = output_dir / "papers" / paper["stem"]
        build_dir = paper_output / "synctex_build"
        try:
            expected_page_limit = min(
                args.max_pages_per_paper,
                int(
                    paper.get("pdf_inspection", {}).get(
                        "pages", args.max_pages_per_paper
                    )
                ),
            )
            fast_checkpoint = (
                load_complete_paper_checkpoint(
                    paper_output,
                    paper=paper,
                    page_limit=expected_page_limit,
                )
                if args.resume
                else None
            )
            if fast_checkpoint is not None:
                checkpoint_pages, checkpoint_summary = fast_checkpoint
                page_rows.extend(checkpoint_pages)
                paper_rows.append(checkpoint_summary)
                accepted += len(checkpoint_pages)
                resumed_pages += len(checkpoint_pages)
                completed = accepted + rejected + errors
                elapsed = max(time.monotonic() - started, 1e-9)
                throughput = completed / elapsed
                remaining = max(0, total_target - completed)
                eta = remaining / throughput if throughput else 0
                print(
                    f"[paper_resume_fast] {paper_index}/{len(papers)} id={paper['stem']} "
                    f"pages={len(checkpoint_pages)} global={completed}/{total_target} "
                    f"pct={100*completed/max(1,total_target):.1f}% resumed={resumed_pages} "
                    f"accepted={accepted} rejected={rejected} errors={errors} "
                    f"throughput={throughput:.3f} pages/s elapsed={elapsed_string(elapsed)} "
                    f"eta={elapsed_string(eta)}",
                    flush=True,
                )
                continue
            pdf_path, synctex_path, compile_info = compile_with_synctex(paper, build_dir, args.resume)
            source_root = Path(paper["source_dir"]).resolve()
            source_hyphenated_terms = extract_source_hyphenated_terms(source_root)
            math_macros = collect_simple_math_macros(source_root)
            blocks = parse_source_blocks(source_root, math_macros)
            executed_sources = compiled_tex_sources(source_root, synctex_path)
            author_superscript_plans = parse_author_superscript_plans(
                executed_sources
            )
            source_paragraphs = parse_source_paragraphs(
                source_root,
                blocks,
                executed_sources,
            )
            source_paragraphs_by_id = {
                paragraph.paragraph_id: paragraph
                for paragraph in source_paragraphs
            }
            source_paragraph_points = parse_synctex_paragraph_points(
                synctex_path,
                source_paragraphs,
            )
            inline_blocks, inline_rejections = parse_inline_source_blocks(
                source_root, blocks, math_macros
            )
            with pdfplumber.open(pdf_path) as pdf:
                map_source_blocks(blocks, source_root, pdf_path, synctex_path, len(pdf.pages))
                map_inline_source_blocks(
                    inline_blocks,
                    source_root,
                    pdf_path,
                    synctex_path,
                    len(pdf.pages),
                )
                disambiguate_inline_source_pages(inline_blocks, pdf)
                page_limit = min(args.max_pages_per_paper, len(pdf.pages))
                for page_number in range(1, page_limit + 1):
                    checkpoint = (
                        load_page_checkpoint(
                            paper_output,
                            paper=paper,
                            page_number=page_number,
                        )
                        if args.resume
                        else None
                    )
                    if checkpoint is not None:
                        page_rows.append(checkpoint)
                        accepted += 1
                        resumed_pages += 1
                        completed = accepted + rejected + errors
                        elapsed = max(time.monotonic() - started, 1e-9)
                        throughput = completed / elapsed
                        remaining = max(0, total_target - completed)
                        eta = remaining / throughput if throughput else 0
                        print(
                            f"[page_resume] paper={paper['stem']} page={page_number}/{page_limit} "
                            f"global={completed}/{total_target} pct={100*completed/max(1,total_target):.1f}% "
                            f"resumed={resumed_pages} accepted={accepted} rejected={rejected} errors={errors} "
                            f"throughput={throughput:.3f} pages/s elapsed={elapsed_string(elapsed)} "
                            f"eta={elapsed_string(eta)}",
                            flush=True,
                        )
                        continue
                    try:
                        row = build_page(
                            pdf.pages[page_number - 1],
                            paper=paper,
                            page_number=page_number,
                            source_blocks=blocks,
                            inline_blocks=inline_blocks,
                            paper_output=paper_output,
                            pdf_path=pdf_path,
                            dpi=args.dpi,
                            source_hyphenated_terms=source_hyphenated_terms,
                            source_paragraph_points=source_paragraph_points.get(
                                page_number, []
                            ),
                            source_paragraphs_by_id=source_paragraphs_by_id,
                            author_superscript_plans=author_superscript_plans,
                        )
                        page_rows.append(row)
                        accepted += 1
                    except Exception as exc:
                        errors += 1
                        print(f"[page_error] paper={paper['stem']} page={page_number} error={exc}", flush=True)
                        continue
                    completed = accepted + rejected + errors
                    elapsed = time.monotonic() - started
                    throughput = completed / elapsed if elapsed else 0
                    remaining = max(0, total_target - completed)
                    eta = remaining / throughput if throughput else 0
                    print(
                        f"[page_done] paper={paper['stem']} page={page_number}/{page_limit} "
                        f"global={completed}/{total_target} pct={100*completed/max(1,total_target):.1f}% "
                        f"accepted={accepted} rejected={rejected} errors={errors} "
                        f"throughput={throughput:.3f} pages/s elapsed={elapsed_string(elapsed)} eta={elapsed_string(eta)}",
                        flush=True,
                    )
            mapped = sum(block.mapping_status == "mapped" for block in blocks)
            unmapped = sum(block.mapping_status != "mapped" for block in blocks)
            emitted_inline_blocks = [
                block
                for block in inline_blocks
                if block.page is not None and block.page <= page_limit
            ]
            inline_feature_counts: collections.Counter[str] = collections.Counter()
            resolved_inline_feature_counts: collections.Counter[str] = collections.Counter()
            for block in emitted_inline_blocks:
                inline_feature_counts.update(block.target_feature_counts)
                if block.match_status == "matched":
                    resolved_inline_feature_counts.update(block.target_feature_counts)
            write_jsonl(paper_output / "source_blocks.jsonl", (block.as_json(source_root) for block in blocks))
            write_jsonl(
                paper_output / "source_paragraphs.jsonl",
                (
                    paragraph.as_json(source_root)
                    for paragraph in source_paragraphs
                ),
            )
            write_jsonl(
                paper_output / "inline_source_blocks.jsonl",
                (block.as_json(source_root) for block in inline_blocks),
            )
            write_jsonl(
                paper_output / "inline_source_rejections.jsonl",
                inline_rejections,
            )
            write_jsonl(
                paper_output / "author_superscript_plans.jsonl",
                (
                    plan.as_json(source_root)
                    for plan in author_superscript_plans
                ),
            )
            paper_row = {
                "stem": paper["stem"],
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "compile": compile_info,
                "pages_emitted": sum(row["arxiv_id"] == paper["arxiv_id"] for row in page_rows),
                "source_blocks": len(blocks),
                "source_paragraphs": len(source_paragraphs),
                "source_paragraph_points": sum(
                    len(values) for values in source_paragraph_points.values()
                ),
                "mapped_source_blocks": mapped,
                "unmapped_source_blocks": unmapped,
                "inline_source_blocks": len(inline_blocks),
                "mapped_inline_source_blocks": sum(
                    block.mapping_status in {"mapped", "mapped_pdf_disambiguated"}
                    for block in inline_blocks
                ),
                "emitted_inline_source_blocks": len(emitted_inline_blocks),
                "matched_inline_source_blocks": sum(
                    block.match_status == "matched" for block in emitted_inline_blocks
                ),
                "fallback_inline_source_blocks": sum(
                    block.match_status == "fallback_pdf" for block in emitted_inline_blocks
                ),
                "inline_source_rejections": len(inline_rejections),
                "source_hyphenated_terms": len(source_hyphenated_terms),
                "author_superscript_plans": len(author_superscript_plans),
                "author_superscript_markers": sum(
                    len(plan.markers) for plan in author_superscript_plans
                ),
                "inline_features": dict(sorted(inline_feature_counts.items())),
                "resolved_inline_features": dict(
                    sorted(resolved_inline_feature_counts.items())
                ),
                "elapsed_seconds": round(time.monotonic() - paper_started, 3),
            }
            atomic_write_json(paper_output / "paper_summary.json", paper_row)
            paper_rows.append(paper_row)
            print(
                f"[paper_done] {paper_index}/{len(papers)} id={paper['stem']} "
                f"pages={paper_row['pages_emitted']} blocks={len(blocks)} mapped={mapped} unmapped={unmapped} "
                f"inline={len(emitted_inline_blocks)} matched_inline={paper_row['matched_inline_source_blocks']} "
                f"inline_features={sum(inline_feature_counts.values())} "
                f"elapsed={elapsed_string(time.monotonic()-paper_started)}",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            paper_rows.append({"stem": paper["stem"], "status": "failed", "error": str(exc)})
            print(f"[paper_error] {paper_index}/{len(papers)} id={paper['stem']} error={exc}", flush=True)
    write_jsonl(output_dir / "pages.jsonl", page_rows)
    write_jsonl(output_dir / "papers.jsonl", paper_rows)
    strict_page_rows = [
        row for row in page_rows if row.get("validation_status") == "passed"
    ]
    write_jsonl(output_dir / "pages_strict.jsonl", strict_page_rows)
    strict_text_v2_rows = [
        row
        for row in page_rows
        if row.get("strict_text_v2_status") == "passed"
        and row.get("strict_punctuation_issues") == []
    ]
    write_jsonl(output_dir / "pages_strict_text_v2.jsonl", strict_text_v2_rows)
    table_blocks = sum(row.get("block_counts", {}).get("table", 0) for row in page_rows)
    math_blocks = sum(row.get("block_counts", {}).get("display_math", 0) for row in page_rows)
    heading_blocks = sum(row.get("block_counts", {}).get("heading", 0) for row in page_rows)
    full_recalls = [row["pdf_text_token_recall"] for row in page_rows]
    retained_recalls = [row["retained_pdf_text_token_recall"] for row in page_rows]
    inline_blocks_total = sum(
        int(row.get("inline_source_integration", {}).get("blocks_total", 0))
        for row in page_rows
    )
    inline_blocks_matched = sum(
        int(row.get("inline_source_integration", {}).get("blocks_matched", 0))
        for row in page_rows
    )
    inline_blocks_fallback = sum(
        int(row.get("inline_source_integration", {}).get("blocks_fallback_pdf", 0))
        for row in page_rows
    )
    inline_features_total = sum(
        int(row.get("inline_source_integration", {}).get("features_total", 0))
        for row in page_rows
    )
    inline_features_resolved = sum(
        int(row.get("inline_source_integration", {}).get("features_resolved_total", 0))
        for row in page_rows
    )
    inline_features_unresolved = sum(
        int(row.get("inline_source_integration", {}).get("features_unresolved_total", 0))
        for row in page_rows
    )
    footnotes_total = sum(
        int(row.get("footnotes", {}).get("total", 0)) for row in page_rows
    )
    footnotes_structured = sum(
        int(row.get("footnotes", {}).get("structured", 0)) for row in page_rows
    )
    footnotes_fallback = sum(
        int(row.get("footnotes", {}).get("fallback", 0)) for row in page_rows
    )
    heading_numbering = {
        key: sum(
            int(
                row.get("source_integration", {})
                .get("heading_numbering", {})
                .get(key, 0)
            )
            for row in page_rows
        )
        for key in ("preserved", "lost", "wrong", "ambiguous", "unnumbered")
    }
    heading_numbering["total"] = sum(heading_numbering.values())
    heading_numbering["strict_pages"] = sum(
        bool(
            row.get("source_integration", {})
            .get("heading_numbering", {})
            .get("strict", False)
        )
        for row in page_rows
    )
    strict_text_v2_failures: collections.Counter[str] = collections.Counter(
        reason
        for row in page_rows
        for reason in row.get("strict_text_v2_failure_reasons", [])
    )
    strict_punctuation_summary = aggregate_strict_punctuation_issues(page_rows)
    strict_text_v2_metrics = {
        key: [
            float(row.get("strict_text_ordered_metrics", {}).get(key, 0.0))
            for row in page_rows
        ]
        for key in (
            "token_recall",
            "token_precision",
            "fivegram_recall",
            "fivegram_precision",
            "anchor_monotonicity",
        )
    }
    strict_text_v2_claim_metrics = {
        key: [
            float(row.get("strict_text_claimed_line_metrics", {}).get(key, 0.0))
            for row in page_rows
        ]
        for key in (
            "token_recall",
            "token_precision",
            "fivegram_recall",
            "fivegram_precision",
            "anchor_monotonicity",
        )
    }
    validation_errors: list[str] = []
    for row in page_rows:
        for key in ("image", "markdown"):
            target = output_dir / row[key]
            if not target.is_file() or target.stat().st_size == 0:
                validation_errors.append(f"missing/empty {key}: {row[key]}")
        markdown_text = (output_dir / row["markdown"]).read_text(encoding="utf-8")
        if "|---" in markdown_text:
            validation_errors.append(f"pipe table forbidden: {row['markdown']}")
        if row.get("block_counts", {}).get("table", 0) and "<table" not in markdown_text:
            validation_errors.append(f"mapped table missing HTML: {row['markdown']}")
        if re.search(r"<table\s+[^>]*>", markdown_text, re.IGNORECASE):
            validation_errors.append(f"table root attributes forbidden: {row['markdown']}")
        if re.search(r"<caption\b", markdown_text, re.IGNORECASE):
            validation_errors.append(f"table caption must be outside HTML: {row['markdown']}")
        for source_block in row.get("source_blocks", []):
            if (
                source_block.get("kind") == "table"
                and source_block.get("table_parse_status") != "parsed"
            ):
                validation_errors.append(
                    f"table was not structurally parsed: {row['markdown']} "
                    f"block={source_block.get('block_id')}"
                )
        if "@@INLINE" in markdown_text:
            validation_errors.append(f"inline placeholder leaked: {row['markdown']}")
        syntax_issues = markdown_inline_syntax_issues(markdown_text)
        if syntax_issues:
            validation_errors.append(
                f"invalid inline Markdown syntax: {row['markdown']} issues={syntax_issues}"
            )
        cid_placeholders = len(re.findall(r"\(cid:\d+\)", markdown_text))
        if cid_placeholders:
            validation_errors.append(
                f"PDF cid placeholders remain: {row['markdown']} count={cid_placeholders}"
            )
        inline_audit = row.get("inline_source_integration", {})
        if int(inline_audit.get("features_unresolved_total", 0)):
            validation_errors.append(
                f"unresolved inline features: {row['markdown']} "
                f"count={inline_audit.get('features_unresolved_total')}"
            )
        footnote_audit = row.get("footnotes", {})
        if int(footnote_audit.get("fallback", 0)):
            validation_errors.append(
                f"unresolved footnotes: {row['markdown']} "
                f"count={footnote_audit.get('fallback')}"
            )
        heading_audit = row.get("source_integration", {}).get(
            "heading_numbering", {}
        )
        if not heading_audit.get("strict", False):
            validation_errors.append(
                f"heading numbering audit failed: {row['markdown']} "
                f"lost={heading_audit.get('lost', 0)} "
                f"wrong={heading_audit.get('wrong', 0)} "
                f"ambiguous={heading_audit.get('ambiguous', 0)}"
            )
        if row.get("validation_status") != "passed":
            validation_errors.append(
                f"page validation warning: {row['markdown']} retained_recall={row.get('retained_pdf_text_token_recall')}"
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "passed" if page_rows and not validation_errors and errors == 0 else "failed",
        "papers_requested": len(papers),
        "papers_completed": sum("error" not in row for row in paper_rows),
        "pages": len(page_rows),
        "accepted": accepted,
        "resumed_pages": resumed_pages,
        "strict_pages": len(strict_page_rows),
        "rejected": rejected,
        "errors": errors,
        "structured_blocks": {
            "headings": heading_blocks,
            "display_math": math_blocks,
            "html_tables": table_blocks,
        },
        "heading_numbering": heading_numbering,
        "inline_enrichment": {
            "blocks_total": inline_blocks_total,
            "blocks_matched": inline_blocks_matched,
            "blocks_fallback_pdf": inline_blocks_fallback,
            "features_total": inline_features_total,
            "features_resolved": inline_features_resolved,
            "features_unresolved": inline_features_unresolved,
        },
        "footnotes": {
            "representation": FOOTNOTE_REPRESENTATION,
            "total": footnotes_total,
            "structured": footnotes_structured,
            "fallback": footnotes_fallback,
        },
        "strict_text_v2": {
            "contract_version": STRICT_TEXT_CONTRACT_VERSION,
            "pages_evaluated": len(page_rows),
            "strict_pages": len(strict_text_v2_rows),
            "failed_pages": len(page_rows) - len(strict_text_v2_rows),
            "failure_reasons": dict(sorted(strict_text_v2_failures.items())),
            "punctuation_hard_gate": strict_punctuation_summary,
            "thresholds": {
                "token_recall": STRICT_TOKEN_THRESHOLD,
                "token_precision": STRICT_TOKEN_THRESHOLD,
                "fivegram_recall": STRICT_FIVEGRAM_THRESHOLD,
                "fivegram_precision": STRICT_FIVEGRAM_THRESHOLD,
                "anchor_monotonicity": STRICT_ANCHOR_MONOTONICITY,
            },
            "ordered_metrics": {
                key: {
                    "minimum": round(min(values), 6) if values else None,
                    "mean": round(statistics.mean(values), 6) if values else None,
                }
                for key, values in strict_text_v2_metrics.items()
            },
            "claimed_line_text_metrics": {
                key: {
                    "minimum": round(min(values), 6) if values else None,
                    "mean": round(statistics.mean(values), 6) if values else None,
                }
                for key, values in strict_text_v2_claim_metrics.items()
            },
            "graphics_structure_ignored": True,
            "ignored_graphic": 0,
            "captions_required": True,
        },
        "pdf_text_token_recall": {
            "minimum": round(min(full_recalls), 6) if full_recalls else None,
            "mean": round(statistics.mean(full_recalls), 6) if full_recalls else None,
        },
        "retained_pdf_text_token_recall": {
            "minimum": round(min(retained_recalls), 6) if retained_recalls else None,
            "mean": round(statistics.mean(retained_recalls), 6) if retained_recalls else None,
        },
        "table_representation": "html_only",
        "validation_errors": validation_errors,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    readme = f"""# arXiv page-level Markdown GT pilot

This pilot emits one PNG, Markdown file, and audit JSON per compiled PDF page.

- Pages: {len(page_rows)}
- Strict pages (inline markup, CID, and heading-number audits passed): {len(strict_page_rows)}
- Strict-text-v2 pages (ordered text and exact line claims passed): {len(strict_text_v2_rows)}
- Papers: {len(paper_rows)}
- Heading numbering: preserved={heading_numbering['preserved']}, unnumbered={heading_numbering['unnumbered']}, lost={heading_numbering['lost']}, wrong={heading_numbering['wrong']}, ambiguous={heading_numbering['ambiguous']}
- Display-math blocks mapped from LaTeX: {math_blocks}
- HTML table blocks mapped from LaTeX: {table_blocks}
- Inline prose blocks matched: {inline_blocks_matched}/{inline_blocks_total}
- Inline math/style features resolved: {inline_features_resolved}/{inline_features_total}
- Validation: {report['status']}

Tables are always embedded as structural HTML (`<table>`, `<thead>`, `<tbody>`,
`<tr>`, `<th>`, `<td>`).  Markdown pipe tables are forbidden by validation.
The JSON sidecar records source file/line spans, SyncTeX page mapping, and bbox.

## Run

```bash
/opt/homebrew/bin/conda run --no-capture-output -n agents python \\
  scripts/build_arxiv_page_markdown_gt.py \\
  --input-root outputs/arxiv_latex_recompile_pilot_5 \\
  --output-dir outputs/arxiv_page_markdown_gt_pilot \\
  --max-papers 5 --max-pages-per-paper 6 --dpi 144 --resume
```

Each page is emitted as `page_NNNN.png`, `page_NNNN.md`, and
`page_NNNN.json`.  Dataset-level manifests are `pages.jsonl`, `papers.jsonl`,
`pages_strict.jsonl`, `pages_strict_text_v2.jsonl`, and
`validation_report.json`.  `pages_strict.jsonl` preserves the legacy gate;
`pages_strict_text_v2.jsonl` additionally requires exact line claims and the
ordered text contract.

## Representation and limits

- Reading order and prose come from the compiled PDF text layer.
- Headings, display mathematics, and tables come from LaTeX source and are
  placed with SyncTeX.
- Heading levels come from the LaTeX section command, while visible Arabic,
  Roman, or letter prefixes are retained from the matched compiled-PDF line.
  Unmatched or ambiguous headings remain PDF text and fail the strict audit;
  numbering is never synthesized from a counter guess.
- Display formulas remain LaTeX inside `$$ ... $$`.  High-confidence inline
  formulas are restored as `$...$`; `\\textbf`, `\\emph`/`\\textit`, and
  `\\texttt`/`\\verb` become Markdown strong, emphasis, and code spans.
- Citations, references, and unknown macros retain the exact text observed in
  the compiled PDF.  Numeric footnote callouts are emitted as `<sup>N</sup>`
  only when their page-bottom PDF bodies jointly and uniquely align with the
  source; definitions begin with the same HTML superscript, and source keys
  never leak.
- Inline enrichment is accepted only after bounded source-to-PDF alignment.
  Failed target features leave PDF text unchanged and make validation fail,
  rather than being silently counted as structured GT.
- Table captions are emitted as separate Markdown paragraphs before a plain
  attribute-free `<table>`.  A visible `Table N` prefix is retained only when
  it is observed on the compiled page; it is never synthesized.  Parse status
  and source provenance live in JSON sidecars, not in the Markdown GT.
- Tables that cannot be parsed into cells carry a failing `table_parse_status`
  in the JSON sidecar rather than silently becoming accepted GT.
- Figures and plots remain visible in the page PNG but are not serialized as
  structured Markdown objects.  Their visible text is retained, captions are
  required, and the v2 contract records `ignored_graphic=0`.
"""
    atomic_write_text(output_dir / "README.md", readme)
    print(
        f"[final] status={report['status']} papers={len(paper_rows)} pages={len(page_rows)} "
        f"math={math_blocks} html_tables={table_blocks} errors={errors} "
        f"heading_numbers={heading_numbering['preserved']}/{heading_numbering['total']} "
        f"heading_lost={heading_numbering['lost']} heading_wrong={heading_numbering['wrong']} "
        f"heading_ambiguous={heading_numbering['ambiguous']} "
        f"inline={inline_blocks_matched}/{inline_blocks_total} "
        f"inline_features={inline_features_resolved}/{inline_features_total} "
        f"strict_text_v2={len(strict_text_v2_rows)}/{len(page_rows)} "
        f"elapsed={elapsed_string(time.monotonic()-started)} output={output_dir}",
        flush=True,
    )
    if report["status"] == "passed":
        return 0
    if args.allow_page_warnings and len(page_rows) == total_target and errors == 0:
        print(
            f"[finish_policy] allow_page_warnings=true emitted={len(page_rows)}/{total_target} "
            f"strict_text_v2={len(strict_text_v2_rows)} warnings_excluded=true return_code=0",
            flush=True,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
