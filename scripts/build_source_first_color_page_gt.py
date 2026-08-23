#!/usr/bin/env python3
"""Build page-level Markdown from LaTeX source with color-only page provenance.

Ordinary prose is serialized from LaTeX source.  Color probes in a shadow
compile determine page membership and reading order; word probes can localize
source-derived fragments when a paragraph crosses a page boundary.  Text
extracted from the clean PDF is used only by an independent verifier; it never
supplies Markdown content.  The contract is intentionally conservative: pages
with unresolved source constructs, unclaimed visible text, or verifier
disagreement are rejected.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Iterable, Sequence
import unicodedata

import pdfplumber


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import arxiv_inline_markup as inline_markup  # noqa: E402
import build_arxiv_page_markdown_gt as page_gt  # noqa: E402
import build_latex_color_alignment_pilot as color_pilot  # noqa: E402


SCHEMA_VERSION = 6
CONTRACT = "source_first_color_v6"
VERIFIER_CONTRACT_VERSION = 4
PROBE_POLICY_VERSION = "paragraph_list_payload_then_paragraph_then_whole_v2"
SHADOW_INVARIANT_POLICY_VERSION = "exact_page_character_sequence_v1"
HEADING_LABEL_POLICY_VERSION = "aux_number_unique_titleformat_label_v1"
WORD_PROBE_FORBIDDEN = re.compile(r"(?<!\\)[%&#]")
NONWHITESPACE_TOKEN = re.compile(r"\S+")
FIGURE_ENVIRONMENTS = {"figure", "figure*"}
FIGURE_ENVIRONMENT_TOKEN = re.compile(
    r"\\(?P<action>begin|end)\s*\{(?P<environment>figure\*?)\}"
)
FIGURE_REFERENCE_COMMAND = re.compile(
    r"\\(?P<command>ref|pageref|autoref|cref|Cref)\s*\{(?P<labels>[^{}]+)\}"
)
OPTIONAL_LINE_END_HYPHEN = "\uFFF4"


@dataclasses.dataclass(frozen=True)
class SourceUnit:
    unit_id: str
    kind: str
    paragraph_id: str
    source_file: Path
    source_lines: tuple[int, ...]
    raw_latex: str
    markdown: str
    rgb: tuple[int, int, int]
    source_command: str | None = None

    @property
    def start_line(self) -> int:
        return min(self.source_lines)

    @property
    def end_line(self) -> int:
        return max(self.source_lines)

    def as_json(self, source_root: Path) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind,
            "source_paragraph_id": self.paragraph_id,
            "source_file": self.source_file.relative_to(source_root).as_posix(),
            "source_lines": [self.start_line, self.end_line],
            "source_line_numbers": list(self.source_lines),
            "raw_latex": self.raw_latex,
            "markdown": self.markdown,
            "rgb": list(self.rgb),
            "hex": "#" + "".join(f"{channel:02x}" for channel in self.rgb),
            "source_command": self.source_command,
        }


@dataclasses.dataclass(frozen=True)
class SourceProbe:
    """One color-only locator for a source-derived semantic unit.

    ``whole`` probes preserve the previous paragraph/heading localization.
    ``plain_word`` probes are used only when every whitespace-delimited source
    token aligns exactly with one source-derived Markdown token.  A probe may
    safely wrap compact inline math or a formatting command such as
    ``\\textbf{...}``; PDF characters establish page and geometry only, while
    ``markdown_fragment`` always comes from the already source-rendered parent
    :class:`SourceUnit`.
    """

    probe_id: str
    unit_id: str
    paragraph_id: str
    kind: str
    source_file: Path
    source_lines: tuple[int, ...]
    markdown_fragment: str
    rgb: tuple[int, int, int]
    ordinal: int
    total: int
    localization_mode: str
    token_span: tuple[int, int, int] | None = None

    def as_json(self, source_root: Path) -> dict[str, Any]:
        value: dict[str, Any] = {
            "probe_id": self.probe_id,
            "source_unit_id": self.unit_id,
            "source_paragraph_id": self.paragraph_id,
            "kind": self.kind,
            "source_file": self.source_file.relative_to(source_root).as_posix(),
            "source_line_numbers": list(self.source_lines),
            "markdown_fragment": self.markdown_fragment,
            "rgb": list(self.rgb),
            "hex": "#" + "".join(f"{channel:02x}" for channel in self.rgb),
            "ordinal": self.ordinal,
            "total": self.total,
            "localization_mode": self.localization_mode,
        }
        if self.token_span is not None:
            line, start, end = self.token_span
            value["source_token_span"] = {
                "line": line,
                "start_column": start + 1,
                "end_column": end + 1,
            }
        return value


@dataclasses.dataclass(frozen=True)
class PageFragment:
    """A source-only Markdown fragment localized to exactly one page."""

    fragment_id: str
    unit_id: str
    paragraph_id: str
    kind: str
    markdown: str
    probe_ids: tuple[str, ...]
    source_file: Path
    source_start_line: int


@dataclasses.dataclass(frozen=True)
class AuxReference:
    """One compiler-resolved LaTeX cross-reference from ``.aux`` metadata."""

    label: str
    number: str
    page: str
    kind: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--main-tex", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--drop-references", action="store_true")
    parser.add_argument(
        "--drop-figures",
        action="store_true",
        help="remove figure/figure* environments before both clean and shadow compilation",
    )
    parser.add_argument("--max-pages", type=int, default=10000)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument(
        "--engine",
        choices=("pdflatex", "xelatex", "latex_dvips_ps2pdf"),
        default="pdflatex",
    )
    parser.add_argument(
        "--latexmk",
        type=Path,
        default=color_pilot.LATEXMK,
        help="latexmk executable for clean and colored source compilation",
    )
    parser.add_argument(
        "--pdftoppm",
        type=Path,
        default=page_gt.PDFTOPPM,
        help="pdftoppm executable for full-page PNG rendering",
    )
    return parser.parse_args()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tex_token_is_visible(source: str, offset: int) -> bool:
    line_start = source.rfind("\n", 0, offset) + 1
    line_end = source.find("\n", offset)
    if line_end < 0:
        line_end = len(source)
    line = source[line_start:line_end]
    return offset - line_start < color_pilot.tex_comment_start(line)


def strip_ignored_figures(source: str) -> tuple[str, dict[str, Any]]:
    """Remove visible figure floats while preserving source line numbers.

    Figures, plots and flowcharts are outside this dataset contract.  The
    removal happens before *both* clean and colored compilation, so generated
    images and source GT remain the same document.  References whose every
    label belongs to a removed figure are also removed; mixed-label commands
    remain untouched and will fail closed later if unresolved.
    """

    ranges: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []
    for match in FIGURE_ENVIRONMENT_TOKEN.finditer(source):
        if not _tex_token_is_visible(source, match.start()):
            continue
        action = match.group("action")
        environment = match.group("environment")
        if action == "begin":
            stack.append((environment, match.start()))
            continue
        if not stack:
            raise ValueError(f"unmatched \\end{{{environment}}}")
        opened_environment, start = stack.pop()
        if opened_environment != environment:
            raise ValueError(
                f"mismatched figure environments: {opened_environment} -> {environment}"
            )
        if not stack:
            ranges.append((start, match.end()))
    if stack:
        raise ValueError(f"unclosed figure environment: {stack[-1][0]}")

    removed_labels: set[str] = set()
    for start, end in ranges:
        fragment = source[start:end]
        for label_match in re.finditer(r"\\label\s*\{([^{}]+)\}", fragment):
            if _tex_token_is_visible(fragment, label_match.start()):
                removed_labels.add(label_match.group(1).strip())
    transformed = source
    for start, end in reversed(ranges):
        transformed = (
            transformed[:start]
            + color_pilot.tex_blank_preserving_lines(transformed[start:end])
            + transformed[end:]
        )

    references_removed = 0
    mixed_references = 0

    def replace_reference(match: re.Match[str]) -> str:
        nonlocal references_removed, mixed_references
        if not _tex_token_is_visible(transformed, match.start()):
            return match.group(0)
        labels = {
            label.strip() for label in match.group("labels").split(",") if label.strip()
        }
        if labels and labels <= removed_labels:
            references_removed += 1
            return color_pilot.tex_blank_preserving_lines(match.group(0))
        if labels & removed_labels:
            mixed_references += 1
        return match.group(0)

    transformed = FIGURE_REFERENCE_COMMAND.sub(replace_reference, transformed)
    return transformed, {
        "status": "passed",
        "figure_environments_removed": len(ranges),
        "figure_labels_removed": len(removed_labels),
        "figure_references_removed": references_removed,
        "mixed_figure_references_retained": mixed_references,
    }


def strip_ignored_figures_tree(source_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    totals: collections.Counter[str] = collections.Counter()
    candidates = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".tex", ".ltx"}
    )
    for index, path in enumerate(candidates, start=1):
        original = path.read_text(encoding="utf-8", errors="replace")
        transformed, report = strip_ignored_figures(original)
        if transformed != original:
            atomic_write_text(path, transformed)
        for key in (
            "figure_environments_removed",
            "figure_labels_removed",
            "figure_references_removed",
            "mixed_figure_references_retained",
        ):
            totals[key] += int(report[key])
        totals["files_changed"] += int(transformed != original)
        files.append(
            {
                "source_file": path.relative_to(source_root).as_posix(),
                "changed": transformed != original,
                **report,
            }
        )
        print(
            f"[figure_strip] file={index}/{len(candidates)} "
            f"removed={report['figure_environments_removed']} "
            f"references={report['figure_references_removed']} "
            f"current={path.relative_to(source_root)}",
            flush=True,
        )
    return {"status": "passed", **dict(sorted(totals.items())), "files": files}


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for count, row in enumerate(rows, start=1):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def inject_compile_support(source: str) -> str:
    """Suppress compiler-generated headers/page numbers in both shadow trees."""

    begin = re.search(r"\\begin\s*\{document\}", source)
    if begin is None:
        raise ValueError("main TeX file has no \\begin{document}")
    prefix = source[: begin.start()]
    suffix = source[begin.start() :]
    begin_in_suffix = re.search(r"\\begin\s*\{document\}", suffix)
    assert begin_in_suffix is not None
    insertion = begin_in_suffix.end()
    return suffix[:0] + prefix + suffix[:insertion] + "\n\\pagestyle{empty}\n" + suffix[insertion:]


def pdf_literal_color(
    rgb: tuple[int, int, int],
    engine: str = "pdflatex",
) -> str:
    """Emit a zero-dimensional engine-specific fill-color switch."""

    values = " ".join(f"{channel / 255.0:.6f}" for channel in rgb)
    if engine == "xelatex":
        return "\\special{pdf:literal direct " + values + " rg}"
    if engine == "latex_dvips_ps2pdf":
        return "\\special{color push rgb " + values + "}"
    return "\\pdfliteral direct {" + values + " rg}"


def pdf_literal_restore(engine: str = "pdflatex") -> str:
    if engine == "latex_dvips_ps2pdf":
        return "\\special{color pop}"
    return pdf_literal_color((0, 0, 0), engine)


def compiled_project_sources(source_root: Path, build_dir: Path) -> list[Path]:
    values: set[Path] = set()
    root = source_root.resolve()
    for fls_path in build_dir.glob("*.fls"):
        for line in fls_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("INPUT "):
                continue
            candidate = Path(line[6:].replace("/./", "/"))
            if not candidate.is_absolute():
                candidate = source_root / candidate
            candidate = candidate.resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.suffix.casefold() in {".tex", ".ltx", ".sty", ".cls"} and candidate.is_file():
                values.add(candidate)
    return sorted(values)


def compiled_tex_sources(source_root: Path, build_dir: Path) -> list[Path]:
    return [
        path
        for path in compiled_project_sources(source_root, build_dir)
        if path.suffix.casefold() in {".tex", ".ltx"}
    ]


def replace_balanced_command(
    value: str,
    command: str,
    *,
    argument_count: int,
    visible_argument: int,
    wrapper: tuple[str, str] = ("", ""),
) -> str:
    """Replace deterministic braced commands without flattening nested content."""

    pattern = re.compile(r"\\" + re.escape(command) + r"\*?")
    cursor = 0
    output: list[str] = []
    while match := pattern.search(value, cursor):
        argument_cursor = match.end()
        arguments: list[str] = []
        valid = True
        for _ in range(argument_count):
            while argument_cursor < len(value) and value[argument_cursor].isspace():
                argument_cursor += 1
            argument = page_gt.extract_balanced(value, argument_cursor)
            if argument is None:
                valid = False
                break
            arguments.append(argument[0])
            argument_cursor = argument[1]
        if not valid:
            output.append(value[cursor : match.end()])
            cursor = match.end()
            continue
        output.append(value[cursor : match.start()])
        output.append(wrapper[0] + arguments[visible_argument] + wrapper[1])
        cursor = argument_cursor
    output.append(value[cursor:])
    return "".join(output)


def normalize_source_deterministic_commands(raw: str) -> str:
    """Expand only commands whose visible result is source-determined."""

    value = raw
    value = re.sub(
        r"\\(?:linebreak|nolinebreak|pagebreak|nopagebreak)(?:\s*\[[^\]]*\])?",
        " ",
        value,
    )
    value = re.sub(
        r"\\(?:allowbreak|break|smallskip|medskip|bigskip|newpage|clearpage|vfill)\b",
        " ",
        value,
    )
    value = re.sub(
        r"\\(?:vspace|hspace)\*?\s*\{[^{}]*\}",
        " ",
        value,
    )
    value = re.sub(
        r"\\(?:thispagestyle|pagestyle)\s*\{[^{}]*\}",
        " ",
        value,
    )
    value = re.sub(
        r"\\(?:tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)\b",
        " ",
        value,
    )
    for command in (
        "mbox",
        "hbox",
        "textrm",
        "textnormal",
        "textsc",
        "underline",
        "uline",
        "fbox",
    ):
        value = replace_balanced_command(
            value,
            command,
            argument_count=1,
            visible_argument=0,
        )
    value = replace_balanced_command(
        value,
        "parbox",
        argument_count=2,
        visible_argument=1,
    )
    value = replace_balanced_command(
        value,
        "resizebox",
        argument_count=3,
        visible_argument=2,
    )
    value = replace_balanced_command(
        value,
        "href",
        argument_count=2,
        visible_argument=1,
    )
    value = replace_balanced_command(
        value,
        "textcolor",
        argument_count=2,
        visible_argument=1,
    )
    value = replace_balanced_command(
        value,
        "foreignlanguage",
        argument_count=2,
        visible_argument=1,
    )
    value = replace_balanced_command(
        value,
        "textsuperscript",
        argument_count=1,
        visible_argument=0,
        wrapper=("<sup>", "</sup>"),
    )
    value = replace_balanced_command(
        value,
        "url",
        argument_count=1,
        visible_argument=0,
    )
    return value


def tex_text_punctuation_to_unicode(value: str) -> str:
    """Convert TeX text punctuation while leaving inline math untouched."""

    output: list[str] = []
    index = 0
    in_dollar_math = False
    in_paren_math = False
    while index < len(value):
        if value.startswith(r"\(", index):
            in_paren_math = True
            output.append(r"\(")
            index += 2
            continue
        if value.startswith(r"\)", index):
            in_paren_math = False
            output.append(r"\)")
            index += 2
            continue
        character = value[index]
        escaped = index > 0 and value[index - 1] == "\\"
        if character == "$" and not escaped and not in_paren_math:
            in_dollar_math = not in_dollar_math
            output.append(character)
            index += 1
            continue
        in_math = in_dollar_math or in_paren_math
        if not in_math and value.startswith("---", index):
            output.append("—")
            index += 3
        elif not in_math and value.startswith("--", index):
            output.append("–")
            index += 2
        elif not in_math and value.startswith("``", index):
            output.append("“")
            index += 2
        elif not in_math and value.startswith("''", index):
            output.append("”")
            index += 2
        elif not in_math and character == "`" and not escaped:
            output.append("‘")
            index += 1
        elif not in_math and character == "'" and not escaped:
            output.append("’")
            index += 1
        elif not in_math and value.startswith(r"\,", index):
            output.append(" ")
            index += 2
        else:
            output.append(character)
            index += 1
    return "".join(output)


def _aux_fields(payload: str) -> list[str]:
    fields: list[str] = []
    cursor = 0
    while cursor < len(payload):
        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
        if cursor >= len(payload) or payload[cursor] != "{":
            break
        field = page_gt.extract_balanced(payload, cursor)
        if field is None:
            break
        fields.append(field[0])
        cursor = field[1]
    return fields


def parse_aux_references(aux_paths: Iterable[Path]) -> dict[str, AuxReference]:
    """Read reference numbers/types from compiler metadata, never PDF text."""

    references: dict[str, AuxReference] = {}
    cref_kinds: dict[str, str] = {}
    for aux_path in aux_paths:
        if not aux_path.is_file():
            continue
        for line in aux_path.read_text(encoding="utf-8", errors="replace").splitlines():
            marker = re.search(r"\\newlabel\s*", line)
            if marker is None:
                continue
            cursor = marker.end()
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            label_group = page_gt.extract_balanced(line, cursor)
            if label_group is None:
                continue
            label, cursor = label_group
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            payload_group = page_gt.extract_balanced(line, cursor)
            if payload_group is None:
                continue
            fields = _aux_fields(payload_group[0])
            if label.endswith("@cref"):
                if fields:
                    kind_match = re.match(r"\[([^\]]+)\]", fields[0])
                    if kind_match:
                        cref_kinds[label[:-5]] = kind_match.group(1)
                continue
            if len(fields) < 2:
                continue
            anchor = fields[3] if len(fields) > 3 else ""
            anchor_kind = anchor.split(".", 1)[0] or None
            references[label] = AuxReference(
                label=label,
                number=page_gt.latex_to_plain(fields[0]),
                page=page_gt.latex_to_plain(fields[1]),
                kind=anchor_kind,
            )
    for label, kind in cref_kinds.items():
        reference = references.get(label)
        if reference is not None:
            references[label] = dataclasses.replace(reference, kind=kind)
    return references


def _reference_type_name(kind: str | None, command: str) -> str:
    normalized = (kind or "reference").casefold()
    if normalized in {"subsection", "subsubsection", "paragraph", "subparagraph"}:
        normalized = "section"
    elif normalized in {"figure", "subfigure"}:
        normalized = "figure"
    elif normalized in {"table", "subtable"}:
        normalized = "table"
    elif normalized in {"equation", "eq"}:
        normalized = "equation"
    if command == "autoref":
        return {
            "section": "section",
            "chapter": "chapter",
            "figure": "Figure",
            "table": "Table",
            "equation": "Equation",
        }.get(normalized, normalized)
    return normalized.capitalize() if command == "Cref" else normalized


def resolve_source_references(
    raw: str,
    references: dict[str, AuxReference],
) -> str:
    """Resolve only single-label references using the clean compile's aux data."""

    pattern = re.compile(r"\\(?P<command>pageref|autoref|eqref|Cref|cref|ref)\*?")
    cursor = 0
    output: list[str] = []
    while match := pattern.search(raw, cursor):
        argument_cursor = match.end()
        while argument_cursor < len(raw) and raw[argument_cursor].isspace():
            argument_cursor += 1
        argument = page_gt.extract_balanced(raw, argument_cursor)
        if argument is None:
            output.append(raw[cursor : match.end()])
            cursor = match.end()
            continue
        label = argument[0].strip()
        reference = references.get(label) if "," not in label else None
        if reference is None:
            output.append(raw[cursor : argument[1]])
            cursor = argument[1]
            continue
        command = match.group("command")
        if command == "pageref":
            visible = reference.page
        elif command == "eqref":
            visible = f"({reference.number})"
        elif command == "ref":
            visible = reference.number
        else:
            visible = f"{_reference_type_name(reference.kind, command)} {reference.number}"
        output.append(raw[cursor : match.start()])
        output.append(visible)
        cursor = argument[1]
    output.append(raw[cursor:])
    return "".join(output)


def source_paragraph_to_markdown(
    paragraph: page_gt.SourceParagraph,
    references: dict[str, AuxReference] | None = None,
) -> str:
    """Serialize one ordinary paragraph entirely from its LaTeX source."""

    if paragraph.kind not in {"paragraph", "itemize_item", "enumerate_item"}:
        raise ValueError(f"unsupported paragraph kind: {paragraph.kind}")
    raw = "\n".join(
        page_gt.strip_tex_comment(line) for line in paragraph.raw_latex.splitlines()
    ).strip()
    raw = normalize_source_deterministic_commands(raw)
    raw = resolve_source_references(raw, references or {})
    raw = tex_text_punctuation_to_unicode(raw)
    raw = raw.lstrip()
    raw = re.sub(r"^(?:\\(?:noindent|leavevmode)\b\s*)+", "", raw)
    if paragraph.kind.endswith("_item"):
        item_match = re.match(r"^\\item(?:\s*\[[^\]]*\])?\s*", raw)
        if item_match is None:
            raise ValueError("list continuation requires structural merge")
        raw = raw[item_match.end() :]
    raw = re.sub(r"\\par\s*$", "", raw).strip()
    if not raw:
        raise ValueError("empty paragraph")
    plan = inline_markup.parse_inline_plan(raw)
    if int(plan.feature_counts.get("opaque", 0)):
        raise ValueError("paragraph contains compiler-dependent or unknown macros")
    markdown = inline_markup.render_inline_source(plan).strip()
    if not markdown:
        raise ValueError("source serializer produced empty Markdown")
    if paragraph.kind == "itemize_item":
        return "- " + markdown
    if paragraph.kind == "enumerate_item":
        if not paragraph.item_ordinal:
            raise ValueError("enumerate item has no source ordinal")
        return f"{paragraph.item_ordinal}. " + markdown
    return markdown


def build_source_units(
    paragraphs: Sequence[page_gt.SourceParagraph],
    *,
    references: dict[str, AuxReference] | None = None,
    color_index_offset: int = 0,
) -> tuple[list[SourceUnit], list[dict[str, Any]]]:
    units: list[SourceUnit] = []
    rejections: list[dict[str, Any]] = []
    for paragraph in paragraphs:
        try:
            markdown = source_paragraph_to_markdown(paragraph, references)
        except (ValueError, inline_markup.InlineParseError) as error:
            rejections.append(
                {
                    "source_paragraph_id": paragraph.paragraph_id,
                    "source_file": str(paragraph.source_file),
                    "source_lines": [paragraph.start_line, paragraph.end_line],
                    "reason": str(error),
                }
            )
            continue
        index = color_index_offset + len(units)
        units.append(
            SourceUnit(
                unit_id=f"src-{index + 1:07d}",
                kind=paragraph.kind,
                paragraph_id=paragraph.paragraph_id,
                source_file=paragraph.source_file,
                source_lines=tuple(paragraph.source_lines),
                raw_latex=paragraph.raw_latex,
                markdown=markdown,
                rgb=color_pilot.deterministic_rgb(index),
                source_command=None,
            )
        )
    return units, rejections


def _plain_word_probes(
    unit: SourceUnit,
    *,
    color_index_start: int,
) -> list[SourceProbe]:
    """Return word probes only when source/Markdown spans are unambiguous.

    Exact token-wise source/Markdown alignment is the safety gate.  This keeps
    source markup (bold, emphasis, code and compact inline math) while allowing
    the color locator to follow the rendered token across a page boundary.
    Alignment characters, macro parameters, comments and whitespace-bearing
    inline math remain whole-unit fallbacks.
    """

    if unit.kind not in {"paragraph", "itemize_item", "enumerate_item"}:
        return []
    if WORD_PROBE_FORBIDDEN.search(unit.raw_latex):
        return []
    raw_dollars = [
        match.start()
        for match in re.finditer(r"(?<!\\)\$", unit.raw_latex)
    ]
    if len(raw_dollars) % 2 or "$$" in unit.raw_latex:
        return []
    if any(
        re.search(r"\s", unit.raw_latex[left + 1 : right])
        for left, right in zip(raw_dollars[::2], raw_dollars[1::2])
    ):
        return []
    if r"\[" in unit.raw_latex or r"\]" in unit.raw_latex:
        return []
    paren_math = list(
        re.finditer(r"\\\((.*?)\\\)", unit.raw_latex, flags=re.DOTALL)
    )
    if (
        unit.raw_latex.count(r"\(") != len(paren_math)
        or unit.raw_latex.count(r"\)") != len(paren_math)
    ):
        return []
    if any(re.search(r"\s", match.group(1)) for match in paren_math):
        return []
    expected_lines = tuple(range(unit.start_line, unit.end_line + 1))
    if unit.source_lines != expected_lines:
        return []
    source = unit.source_file.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines(keepends=True)
    if unit.start_line < 1 or unit.end_line > len(lines):
        return []
    line_starts: list[int] = []
    cursor = 0
    for line in lines:
        line_starts.append(cursor)
        cursor += len(line)
    range_start = line_starts[unit.start_line - 1]
    range_end = (
        line_starts[unit.end_line]
        if unit.end_line < len(line_starts)
        else len(source)
    )
    selected = source[range_start:range_end]
    raw = unit.raw_latex
    relative = selected.find(raw)
    if relative < 0 or selected.find(raw, relative + 1) >= 0:
        return []
    raw_start = range_start + relative
    raw_visible_offset = 0
    markdown_visible_offset = 0
    if unit.kind.endswith("_item"):
        item_match = re.match(r"^\s*\\item(?:\s*\[[^\]]*\])?\s*", raw)
        if item_match is None:
            return []
        markdown_prefix = "- " if unit.kind == "itemize_item" else re.match(
            r"^\d+\.\s+", unit.markdown
        )
        if unit.kind == "itemize_item":
            if not unit.markdown.startswith(markdown_prefix):
                return []
            markdown_visible_offset = len(markdown_prefix)
        else:
            if markdown_prefix is None:
                return []
            markdown_visible_offset = markdown_prefix.end()
        raw_visible_offset = item_match.end()
    source_visible = raw[raw_visible_offset:]
    markdown_visible = unit.markdown[markdown_visible_offset:]
    source_tokens = list(NONWHITESPACE_TOKEN.finditer(source_visible))
    markdown_tokens = list(NONWHITESPACE_TOKEN.finditer(markdown_visible))
    if len(source_tokens) < 2 or len(source_tokens) != len(markdown_tokens):
        return []
    for source_token, markdown_token in zip(source_tokens, markdown_tokens):
        if page_gt.normalize_tokens(source_token.group(0)) != page_gt.normalize_tokens(
            markdown_token.group(0)
        ):
            return []

    probes: list[SourceProbe] = []
    total = len(source_tokens)
    for ordinal, (source_token, markdown_token) in enumerate(
        zip(source_tokens, markdown_tokens), start=1
    ):
        absolute_start = raw_start + raw_visible_offset + source_token.start()
        absolute_end = raw_start + raw_visible_offset + source_token.end()
        line_index = bisect.bisect_right(line_starts, absolute_start) - 1
        if line_index < 0 or absolute_end > line_starts[line_index] + len(lines[line_index]):
            return []
        start_column = absolute_start - line_starts[line_index]
        end_column = absolute_end - line_starts[line_index]
        markdown_start = (
            0
            if ordinal == 1
            else markdown_visible_offset + markdown_token.start()
        )
        markdown_end = (
            len(unit.markdown)
            if ordinal == total
            else markdown_visible_offset + markdown_tokens[ordinal].start()
        )
        probes.append(
            SourceProbe(
                probe_id=f"{unit.unit_id}-word-{ordinal:05d}",
                unit_id=unit.unit_id,
                paragraph_id=unit.paragraph_id,
                kind=unit.kind,
                source_file=unit.source_file,
                source_lines=unit.source_lines,
                markdown_fragment=unit.markdown[markdown_start:markdown_end],
                rgb=color_pilot.deterministic_rgb(color_index_start + ordinal - 1),
                ordinal=ordinal,
                total=total,
                localization_mode="plain_word",
                token_span=(line_index + 1, start_column, end_column),
            )
        )
    if "".join(probe.markdown_fragment for probe in probes) != unit.markdown:
        return []
    return probes


def build_source_probes(
    units: Sequence[SourceUnit],
    *,
    word_probe_kinds: set[str] | None = None,
) -> tuple[list[SourceProbe], dict[str, str]]:
    """Build color locators while preserving each parent unit as content truth."""

    probes: list[SourceProbe] = []
    modes: dict[str, str] = {}
    # ``reject_line_overlaps`` can remove units from the middle of the original
    # color sequence.  Starting word-probe colors at ``len(units)`` would then
    # collide with a surviving later unit.  Reserve every whole-unit color and
    # allocate each word-probe color from the remaining deterministic palette.
    used_colors = {unit.rgb for unit in units}
    next_color_index = 0

    def allocate_probe_color() -> tuple[int, int, int]:
        nonlocal next_color_index
        while True:
            rgb = color_pilot.deterministic_rgb(next_color_index)
            next_color_index += 1
            if rgb not in used_colors:
                used_colors.add(rgb)
                return rgb

    for unit in units:
        allow_word_probe = (
            word_probe_kinds is None or unit.kind in word_probe_kinds
        )
        word_probes = (
            _plain_word_probes(unit, color_index_start=0)
            if allow_word_probe
            else []
        )
        if word_probes:
            probes.extend(
                dataclasses.replace(probe, rgb=allocate_probe_color())
                for probe in word_probes
            )
            modes[unit.unit_id] = "plain_word"
            continue
        probes.append(
            SourceProbe(
                probe_id=f"{unit.unit_id}-whole",
                unit_id=unit.unit_id,
                paragraph_id=unit.paragraph_id,
                kind=unit.kind,
                source_file=unit.source_file,
                source_lines=unit.source_lines,
                markdown_fragment=unit.markdown,
                rgb=unit.rgb,
                ordinal=1,
                total=1,
                localization_mode="whole",
            )
        )
        modes[unit.unit_id] = "whole"
    return probes, modes


def _titleformat_label_template(command: str, label: str) -> str | None:
    """Return a source-exact visible label around ``{number}``, or ``None``."""

    marker = "\\the" + command
    if label.count(marker) != 1:
        return None
    value = label
    value = re.sub(
        r"\\(?:quad|qquad|enspace|thinspace|space|nobreakspace|relax)\b",
        "",
        value,
    )
    value = value.replace("~", " ")
    value = tex_text_punctuation_to_unicode(value)
    value = value.replace(marker, "{number}")
    if "\\" in value or "{" in value.replace("{number}", "") or "}" in value.replace(
        "{number}", ""
    ):
        return None
    return value.strip()


def parse_unique_titleformat_labels(
    source_files: Iterable[Path],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Parse only unambiguous ``titlesec`` label arguments from executed source."""

    observed: dict[str, list[str]] = collections.defaultdict(list)
    definitions = 0
    rejected = 0
    for source_file in source_files:
        source = "\n".join(
            page_gt.strip_tex_comment(line)
            for line in source_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        )
        cursor = 0
        pattern = re.compile(r"\\titleformat\s*")
        while match := pattern.search(source, cursor):
            argument_cursor = match.end()
            command_group = page_gt.extract_balanced(source, argument_cursor)
            if command_group is None:
                rejected += 1
                cursor = match.end()
                continue
            command_match = re.fullmatch(
                r"\s*\\(section|subsection|subsubsection|paragraph|subparagraph)\s*",
                command_group[0],
            )
            argument_cursor = command_group[1]
            while argument_cursor < len(source) and source[argument_cursor].isspace():
                argument_cursor += 1
            if argument_cursor < len(source) and source[argument_cursor] == "[":
                argument_cursor = color_pilot.balanced_group_end(
                    source, argument_cursor, "[", "]"
                )
            arguments: list[str] = []
            for _ in range(4):
                while argument_cursor < len(source) and source[argument_cursor].isspace():
                    argument_cursor += 1
                group = page_gt.extract_balanced(source, argument_cursor)
                if group is None:
                    break
                arguments.append(group[0])
                argument_cursor = group[1]
            cursor = max(argument_cursor, match.end())
            if command_match is None or len(arguments) != 4:
                rejected += 1
                continue
            command = command_match.group(1)
            template = _titleformat_label_template(command, arguments[1])
            if template is None:
                rejected += 1
                continue
            observed[command].append(template)
            definitions += 1
    accepted: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for command, templates in sorted(observed.items()):
        unique = sorted(set(templates))
        if len(unique) == 1:
            accepted[command] = unique[0]
        else:
            ambiguous[command] = unique
    return accepted, {
        "policy_version": HEADING_LABEL_POLICY_VERSION,
        "definitions_parsed": definitions,
        "definitions_rejected": rejected,
        "labels": dict(sorted(accepted.items())),
        "ambiguous": ambiguous,
    }


def parse_aux_heading_numbers(
    aux_paths: Path | Iterable[Path],
) -> dict[tuple[str, str], list[str | None]]:
    """Return compiler-resolved heading-number visibility from aux streams.

    Some document classes render an unstarred command such as ``\\paragraph``
    without a visible number.  Its ``\\contentsline`` is present but has no
    ``\\numberline``.  Recording ``None`` distinguishes that compiler-resolved
    unnumbered form from a heading that is genuinely missing from metadata.
    """

    values: dict[tuple[str, str], list[str | None]] = collections.defaultdict(list)
    paths = [aux_paths] if isinstance(aux_paths, Path) else list(aux_paths)
    for aux_path in paths:
        if not aux_path.is_file():
            continue
        for line in aux_path.read_text(encoding="utf-8", errors="replace").splitlines():
            marker = re.search(r"\\contentsline\s*", line)
            if marker is None:
                continue
            cursor = marker.end()
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            kind_group = page_gt.extract_balanced(line, cursor)
            if kind_group is None:
                continue
            kind, cursor = kind_group
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            title_group = page_gt.extract_balanced(line, cursor)
            if title_group is None:
                continue
            title_raw = title_group[0]
            number: str | None = None
            title_start = 0
            number_match = re.match(r"\s*\\numberline\s*", title_raw)
            if number_match is not None:
                number_group = page_gt.extract_balanced(title_raw, number_match.end())
                if number_group is None:
                    continue
                number, title_start = number_group
                number = page_gt.latex_to_plain(number)
            else:
                trailing_fields: list[str] = []
                trailing_cursor = title_group[1]
                for _ in range(2):
                    while (
                        trailing_cursor < len(line)
                        and line[trailing_cursor].isspace()
                    ):
                        trailing_cursor += 1
                    trailing_group = page_gt.extract_balanced(line, trailing_cursor)
                    if trailing_group is None:
                        break
                    trailing_fields.append(trailing_group[0])
                    trailing_cursor = trailing_group[1]
                anchor = trailing_fields[1].strip() if len(trailing_fields) == 2 else ""
                if re.fullmatch(
                    r"(?:section|subsection|subsubsection|paragraph|subparagraph)\*\.\d+",
                    anchor,
                ) is None:
                    continue
            title = page_gt.latex_to_plain(
                tex_text_punctuation_to_unicode(title_raw[title_start:])
            )
            key = (kind.strip(), page_gt.normalize_space(title).casefold())
            values[key].append(number)
    return values


def build_heading_units(
    blocks: Sequence[page_gt.SourceBlock],
    aux_paths: Path | Iterable[Path],
    *,
    color_index_offset: int,
    heading_label_templates: dict[str, str] | None = None,
) -> tuple[list[SourceUnit], list[dict[str, Any]]]:
    number_queues = parse_aux_heading_numbers(aux_paths)
    units: list[SourceUnit] = []
    rejections: list[dict[str, Any]] = []
    for block in blocks:
        if block.kind != "heading" or not block.heading_command or not block.heading_source_title:
            continue
        title_visible = page_gt.latex_to_plain(
            tex_text_punctuation_to_unicode(block.heading_source_title)
        )
        if not title_visible:
            rejections.append({"source_block_id": block.block_id, "reason": "empty heading title"})
            continue
        title = inline_markup.escape_markdown_text(title_visible)
        number: str | None = None
        if not block.heading_starred:
            key = (
                block.heading_command,
                page_gt.normalize_space(title_visible).casefold(),
            )
            queue = number_queues.get(key, [])
            if not queue:
                rejections.append(
                    {"source_block_id": block.block_id, "reason": "heading number missing from aux"}
                )
                continue
            number = queue.pop(0)
        level = int(block.heading_level or 2)
        if number:
            template = (heading_label_templates or {}).get(
                block.heading_command, "{number}"
            )
            visible_number = template.format(number=number)
        else:
            visible_number = ""
        markdown = "#" * level + " " + ((visible_number + " ") if visible_number else "") + title
        index = color_index_offset + len(units)
        units.append(
            SourceUnit(
                unit_id=f"src-{index + 1:07d}",
                kind="heading",
                paragraph_id=block.block_id,
                source_file=block.source_file,
                source_lines=tuple(range(block.start_line, block.end_line + 1)),
                raw_latex=block.raw_latex,
                markdown=markdown,
                rgb=color_pilot.deterministic_rgb(index),
                source_command=block.heading_command,
            )
        )
    return units, rejections


def reject_line_overlaps(
    units: Sequence[SourceUnit],
) -> tuple[list[SourceUnit], list[dict[str, Any]]]:
    """Fail closed when line-level provenance cannot separate two source units.

    Run-in headings such as ``\\paragraph{Title} body`` can share a source line
    with the following prose.  Until source offsets are carried end-to-end, it
    is safer to remove both units and let the page verifier reject the page than
    to assign either unit an ambiguous color range.
    """

    owners: dict[tuple[Path, int], list[SourceUnit]] = collections.defaultdict(list)
    for unit in units:
        resolved = unit.source_file.resolve()
        for line_number in unit.source_lines:
            owners[(resolved, line_number)].append(unit)
    rejected_ids = {
        unit.unit_id
        for line_units in owners.values()
        if len(line_units) > 1
        for unit in line_units
    }
    accepted = [unit for unit in units if unit.unit_id not in rejected_ids]
    rejections = [
        {
            "source_unit_id": unit.unit_id,
            "source_paragraph_id": unit.paragraph_id,
            "source_file": str(unit.source_file),
            "source_lines": [unit.start_line, unit.end_line],
            "reason": "ambiguous line overlap between run-in source units",
        }
        for unit in units
        if unit.unit_id in rejected_ids
    ]
    return accepted, rejections


LIST_LEADING_VISIBLE_WRAPPER = re.compile(
    r"\\(?:underline|textbf|textit|emph|texttt|textnormal|textrm|textsf|"
    r"textsl|textup|textsc|mbox|hbox)\s*\{"
)


def list_payload_color_offset(line: str, item_end: int) -> int:
    """Place a whole-item probe inside deterministic leading text wrappers."""

    cursor = item_end
    while cursor < len(line) and line[cursor].isspace():
        cursor += 1
    wrappers = 0
    while match := LIST_LEADING_VISIBLE_WRAPPER.match(line, cursor):
        wrappers += 1
        cursor = match.end()
        while cursor < len(line) and line[cursor].isspace():
            cursor += 1
    return cursor if wrappers else item_end


def instrument_source_file(
    source: str,
    units: Sequence[SourceUnit],
    engine: str = "pdflatex",
    probes: Sequence[SourceProbe] | None = None,
) -> str:
    """Switch colors at complete paragraph boundaries without TeX grouping.

    A brace group plus an injected ``\\par`` is safe for isolated pilots but
    can alter template glue or paragraph hooks when repeated across a paper.
    Declaration switches are zero-width whatsits; restoring black at the
    existing source boundary preserves the original paragraph construction.
    """

    lines = source.splitlines(keepends=True)
    word_probes = [
        probe
        for probe in (probes or ())
        if probe.localization_mode == "plain_word"
    ]
    word_unit_ids = {probe.unit_id for probe in word_probes}
    prefixes: dict[int, str] = {}
    suffixes: dict[int, str] = {}
    occupied: set[int] = set()
    for unit in sorted(units, key=lambda value: value.start_line):
        if unit.unit_id in word_unit_ids:
            continue
        start = unit.start_line - 1
        end = unit.end_line - 1
        if start < 0 or end >= len(lines):
            raise ValueError(f"unit line range outside source: {unit.unit_id}")
        claimed = set(range(start, end + 1))
        if occupied & claimed:
            raise ValueError(f"overlapping source units near {unit.unit_id}")
        occupied.update(claimed)
        color_switch = pdf_literal_color(unit.rgb, engine)
        black_switch = pdf_literal_restore(engine)
        if unit.kind == "heading":
            if start != end or not unit.source_command:
                raise ValueError(f"unsupported multiline heading instrumentation: {unit.unit_id}")
            command_pattern = re.compile(
                r"\\" + re.escape(unit.source_command) + r"\*?\s*(?:\[[^\]]*\]\s*)?\{"
            )
            match = command_pattern.search(lines[start])
            if match is None:
                raise ValueError(f"heading command not found: {unit.unit_id}")
            argument = page_gt.extract_balanced(lines[start], match.end() - 1)
            if argument is None:
                raise ValueError(f"unbalanced heading command: {unit.unit_id}")
            title, argument_end = argument
            colored_title = "{" + color_switch + title + black_switch + "}"
            lines[start] = (
                lines[start][: match.end()]
                + colored_title
                + lines[start][argument_end - 1 :]
            )
            continue
        if unit.kind in {"itemize_item", "enumerate_item"}:
            item_match = re.search(r"\\item(?:\s*\[[^\]]*\])?", lines[start])
            if item_match is None:
                raise ValueError(f"list item command not found: {unit.unit_id}")
            insertion = list_payload_color_offset(lines[start], item_match.end())
            lines[start] = (
                lines[start][:insertion]
                + color_switch
                + lines[start][insertion:]
            )
            suffixes[end] = suffixes.get(end, "") + black_switch
            continue
        mode_match = re.match(r"(?P<prefix>\s*\\(?:noindent|indent)\b)", lines[start])
        if mode_match is not None:
            boundary = mode_match.end()
            lines[start] = lines[start][:boundary] + color_switch + lines[start][boundary:]
        else:
            # Enter horizontal mode exactly where TeX would enter it for the
            # first visible paragraph token.  A color whatsit left in vertical
            # mode can become a page-break object and move later floats.
            prefixes[start] = prefixes.get(start, "") + "\\leavevmode" + color_switch
        suffixes[end] = suffixes.get(end, "") + black_switch

    token_edits: dict[int, list[tuple[int, int, str, str]]] = collections.defaultdict(list)
    token_spans: set[tuple[int, int, int]] = set()
    for probe in word_probes:
        if probe.token_span is None:
            raise ValueError(f"word probe has no token span: {probe.probe_id}")
        line_number, start_column, end_column = probe.token_span
        line_index = line_number - 1
        if line_index < 0 or line_index >= len(lines):
            raise ValueError(f"probe line outside source: {probe.probe_id}")
        span = (line_index, start_column, end_column)
        if span in token_spans:
            raise ValueError(f"duplicate source token probe span: {probe.probe_id}")
        token_spans.add(span)
        if not 0 <= start_column < end_column <= len(lines[line_index]):
            raise ValueError(f"probe columns outside source: {probe.probe_id}")
        prefix = pdf_literal_color(probe.rgb, engine)
        if probe.ordinal == 1:
            prefix = "\\leavevmode" + prefix
        token_edits[line_index].append(
            (start_column, end_column, prefix, pdf_literal_restore(engine))
        )
    for line_index, edits in token_edits.items():
        rendered = lines[line_index]
        previous_start = len(rendered) + 1
        for start_column, end_column, prefix, suffix in sorted(edits, reverse=True):
            if end_column > previous_start:
                raise ValueError(f"overlapping word probes on source line {line_index + 1}")
            rendered = (
                rendered[:start_column]
                + prefix
                + rendered[start_column:end_column]
                + suffix
                + rendered[end_column:]
            )
            previous_start = start_column
        lines[line_index] = rendered
    output: list[str] = []
    for index, line in enumerate(lines):
        prefix = prefixes.get(index, "")
        suffix = suffixes.get(index, "")
        if suffix and line.endswith("\r\n"):
            output.append(prefix + line[:-2] + suffix + "\r\n")
        elif suffix and line.endswith("\n"):
            output.append(prefix + line[:-1] + suffix + "\n")
        else:
            output.append(prefix + line + suffix)
    return "".join(output)


def instrument_source_tree(
    clean_root: Path,
    colored_root: Path,
    units: Sequence[SourceUnit],
    probes: Sequence[SourceProbe],
    engine: str = "pdflatex",
) -> None:
    by_file: dict[Path, list[SourceUnit]] = collections.defaultdict(list)
    for unit in units:
        by_file[unit.source_file.resolve()].append(unit)
    probes_by_file: dict[Path, list[SourceProbe]] = collections.defaultdict(list)
    for probe in probes:
        probes_by_file[probe.source_file.resolve()].append(probe)
    for clean_path, file_units in by_file.items():
        relative = clean_path.relative_to(clean_root.resolve())
        colored_path = colored_root / relative
        rendered = instrument_source_file(
            colored_path.read_text(encoding="utf-8", errors="replace"),
            file_units,
            engine,
            probes_by_file.get(clean_path, []),
        )
        atomic_write_text(colored_path, rendered)


def extract_color_geometry(
    colored_pdf: Path,
    probes: Sequence[SourceProbe],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Map source IDs to pages/bboxes without reading PDF character text."""

    probe_by_rgb = {probe.rgb: probe for probe in probes}
    if len(probe_by_rgb) != len(probes):
        raise ValueError("source probe RGB values must be unique")
    observed: dict[str, dict[int, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    with pdfplumber.open(colored_pdf) as document:
        total_pages = len(document.pages)
        for page_number, page in enumerate(document.pages, start=1):
            matched = 0
            for char in page.chars:
                rgb = color_pilot.normalize_pdf_rgb(char.get("non_stroking_color"))
                probe = probe_by_rgb.get(rgb)
                if probe is None:
                    continue
                observed[probe.probe_id][page_number].append(char)
                matched += 1
            print(
                f"[color_page] page={page_number}/{total_pages} "
                f"matched_characters={matched}",
                flush=True,
            )
    rows: dict[str, list[dict[str, Any]]] = {}
    mapped = multi_page = 0
    mode_counts: collections.Counter[str] = collections.Counter()
    for probe in probes:
        mode_counts[probe.localization_mode] += 1
        pages: list[dict[str, Any]] = []
        for page_number, chars in sorted(observed.get(probe.probe_id, {}).items()):
            pages.append(
                {
                    "page_number": page_number,
                    "bbox_points": [
                        round(min(float(char["x0"]) for char in chars), 3),
                        round(min(float(char["top"]) for char in chars), 3),
                        round(max(float(char["x1"]) for char in chars), 3),
                        round(max(float(char["bottom"]) for char in chars), 3),
                    ],
                    "characters": len(chars),
                }
            )
        rows[probe.probe_id] = pages
        if pages:
            mapped += 1
        if len(pages) > 1:
            multi_page += 1
    return rows, {
        "probes_total": len(probes),
        "probes_mapped": mapped,
        "probes_unmapped": len(probes) - mapped,
        "probes_spanning_multiple_pages": multi_page,
        "probe_mode_counts": dict(sorted(mode_counts.items())),
        "coverage": round(mapped / max(1, len(probes)), 6),
    }


def build_page_fragments(
    units: Sequence[SourceUnit],
    probes: Sequence[SourceProbe],
    color_rows: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[int, list[tuple[PageFragment, dict[str, Any]]]],
    dict[int, set[str]],
    dict[str, Any],
]:
    """Convert color-only probe placements into source Markdown fragments."""

    probes_by_unit: dict[str, list[SourceProbe]] = collections.defaultdict(list)
    for probe in probes:
        probes_by_unit[probe.unit_id].append(probe)
    placements_by_page: dict[int, list[tuple[PageFragment, dict[str, Any]]]] = (
        collections.defaultdict(list)
    )
    reasons_by_page: dict[int, set[str]] = collections.defaultdict(set)
    summary: collections.Counter[str] = collections.Counter()
    for unit in units:
        unit_probes = sorted(probes_by_unit[unit.unit_id], key=lambda probe: probe.ordinal)
        if not unit_probes:
            summary["units_without_probes"] += 1
            continue
        mode = unit_probes[0].localization_mode
        if any(probe.localization_mode != mode for probe in unit_probes):
            raise ValueError(f"mixed localization modes for {unit.unit_id}")
        if mode == "whole":
            pages = color_rows.get(unit_probes[0].probe_id, [])
            if len(pages) > 1:
                for page in pages:
                    reasons_by_page[int(page["page_number"])].add(
                        "cross_page_source_paragraph"
                    )
                summary["whole_units_cross_page"] += 1
                continue
            if not pages:
                summary["whole_units_unmapped"] += 1
                continue
            page = pages[0]
            page_number = int(page["page_number"])
            fragment = PageFragment(
                fragment_id=f"{unit.unit_id}-page-{page_number:04d}",
                unit_id=unit.unit_id,
                paragraph_id=unit.paragraph_id,
                kind=unit.kind,
                markdown=unit.markdown,
                probe_ids=(unit_probes[0].probe_id,),
                source_file=unit.source_file,
                source_start_line=unit.start_line,
            )
            placements_by_page[page_number].append((fragment, page))
            summary["whole_units_mapped"] += 1
            continue

        if mode != "plain_word":
            raise ValueError(f"unknown source localization mode: {mode}")
        invalid = False
        known_pages: set[int] = set()
        page_sequence: list[int] = []
        for probe in unit_probes:
            pages = color_rows.get(probe.probe_id, [])
            if len(pages) != 1:
                invalid = True
                for page in pages:
                    known_pages.add(int(page["page_number"]))
                continue
            page_number = int(pages[0]["page_number"])
            known_pages.add(page_number)
            page_sequence.append(page_number)
        if invalid or len(page_sequence) != len(unit_probes):
            for page_number in known_pages:
                reasons_by_page[page_number].add("source_word_probe_incomplete")
            summary["plain_units_incomplete"] += 1
            continue
        if any(right < left for left, right in zip(page_sequence, page_sequence[1:])):
            for page_number in known_pages:
                reasons_by_page[page_number].add("source_word_probe_page_order_mismatch")
            summary["plain_units_page_order_mismatch"] += 1
            continue
        cursor = 0
        while cursor < len(unit_probes):
            page_number = page_sequence[cursor]
            end = cursor + 1
            while end < len(unit_probes) and page_sequence[end] == page_number:
                end += 1
            selected = unit_probes[cursor:end]
            selected_pages = [color_rows[probe.probe_id][0] for probe in selected]
            bboxes = [page["bbox_points"] for page in selected_pages]
            placement = {
                "page_number": page_number,
                "bbox_points": [
                    round(min(float(bbox[0]) for bbox in bboxes), 3),
                    round(min(float(bbox[1]) for bbox in bboxes), 3),
                    round(max(float(bbox[2]) for bbox in bboxes), 3),
                    round(max(float(bbox[3]) for bbox in bboxes), 3),
                ],
                "characters": sum(int(page["characters"]) for page in selected_pages),
                "probe_ids": [probe.probe_id for probe in selected],
                "probe_ordinals": [selected[0].ordinal, selected[-1].ordinal],
            }
            markdown = "".join(probe.markdown_fragment for probe in selected).strip()
            if not markdown:
                reasons_by_page[page_number].add("empty_source_word_fragment")
            else:
                fragment = PageFragment(
                    fragment_id=(
                        f"{unit.unit_id}-words-{selected[0].ordinal:05d}-"
                        f"{selected[-1].ordinal:05d}"
                    ),
                    unit_id=unit.unit_id,
                    paragraph_id=unit.paragraph_id,
                    kind=unit.kind,
                    markdown=markdown,
                    probe_ids=tuple(probe.probe_id for probe in selected),
                    source_file=unit.source_file,
                    source_start_line=unit.start_line,
                )
                placements_by_page[page_number].append((fragment, placement))
            cursor = end
        summary["plain_units_mapped"] += 1
        if len(known_pages) > 1:
            summary["plain_units_split_across_pages"] += 1
    return placements_by_page, reasons_by_page, dict(sorted(summary.items()))


def compose_page_markdown(
    ordered: Sequence[tuple[PageFragment, dict[str, Any]]],
) -> str:
    """Join page fragments while preserving source paragraph identity."""

    output = ""
    previous: PageFragment | None = None
    for fragment, _ in ordered:
        if previous is None:
            output = fragment.markdown
        elif fragment.paragraph_id == previous.paragraph_id:
            output = output.rstrip() + " " + fragment.markdown.lstrip()
        else:
            output = output.rstrip() + "\n\n" + fragment.markdown.lstrip()
        previous = fragment
    return output.strip() + ("\n" if output.strip() else "")


def order_page_units(
    placements: Sequence[tuple[PageFragment, dict[str, Any]]],
    page_width: float,
) -> tuple[list[tuple[PageFragment, dict[str, Any]]], str | None]:
    midpoint = page_width / 2.0
    gutter = max(8.0, page_width * 0.025)
    left: list[tuple[PageFragment, dict[str, Any]]] = []
    right: list[tuple[PageFragment, dict[str, Any]]] = []
    full: list[tuple[PageFragment, dict[str, Any]]] = []
    for placement in placements:
        bbox = placement[1]["bbox_points"]
        if bbox[2] <= midpoint + gutter:
            left.append(placement)
        elif bbox[0] >= midpoint - gutter:
            right.append(placement)
        else:
            full.append(placement)
    def ordered(
        values: Sequence[tuple[PageFragment, dict[str, Any]]],
    ) -> list[tuple[PageFragment, dict[str, Any]]]:
        geometry_order = sorted(
            values,
            key=lambda item: (
                float(item[1]["bbox_points"][1]),
                float(item[1]["bbox_points"][0]),
            ),
        )
        output: list[tuple[PageFragment, dict[str, Any]]] = []
        cursor = 0
        while cursor < len(geometry_order):
            anchor_top = float(geometry_order[cursor][1]["bbox_points"][1])
            end = cursor + 1
            while (
                end < len(geometry_order)
                and float(geometry_order[end][1]["bbox_points"][1]) - anchor_top
                <= 4.0
            ):
                end += 1
            cluster = geometry_order[cursor:end]
            source_files = {
                fragment.source_file.resolve() for fragment, _ in cluster
            }
            # Run-in headings often share the first visual line with the
            # following paragraph.  Their larger font can report a slightly
            # lower PDF top coordinate.  Only for this tight same-file visual
            # cluster, use source line order as the deterministic tiebreaker.
            if len(source_files) == 1 and any(
                fragment.kind == "heading" for fragment, _ in cluster
            ):
                cluster = sorted(
                    cluster,
                    key=lambda item: (
                        item[0].source_start_line,
                        0 if item[0].kind == "heading" else 1,
                        float(item[1]["bbox_points"][1]),
                        float(item[1]["bbox_points"][0]),
                    ),
                )
            output.extend(cluster)
            cursor = end
        return output

    if left and right and full:
        return [], "mixed_full_and_columns"
    if left and right:
        return ordered(left) + ordered(right), None
    return ordered([*left, *right, *full]), None


def pdf_verifier_text(page: Any) -> tuple[str, str]:
    nodes = page_gt.words_to_line_nodes(page)
    ordered, layout = page_gt.order_page_nodes(nodes, float(page.width))
    if not ordered:
        return "", layout
    value = ordered[0].text
    for current in ordered[1:]:
        current_text = current.text
        if value.endswith("-") and current_text and current_text[0].islower():
            value = value[:-1] + OPTIONAL_LINE_END_HYPHEN + current_text
        else:
            value += " " + current_text
    return page_gt.normalize_space(value), layout


MATH_VISIBLE_COMMANDS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "varepsilon": "ε",
    "theta": "θ",
    "vartheta": "ϑ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "ϕ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "cdot": "·",
    "times": "×",
    "pm": "±",
    "mp": "∓",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "neq": "≠",
    "approx": "≈",
    "infty": "∞",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "cup": "∪",
    "cap": "∩",
    "sum": "∑",
    "prod": "∏",
}
MATH_INVISIBLE_COMMANDS = {
    ",",
    ";",
    ":",
    "!",
    "quad",
    "qquad",
    "displaystyle",
    "textstyle",
    "scriptstyle",
    "scriptscriptstyle",
    "left",
    "right",
}


def math_source_visible_text(value: str) -> str:
    """Best-effort visible text for a restricted inline LaTeX math span.

    Unknown commands are retained as an impossible sentinel so the strict
    verifier fails closed instead of silently dropping a visible symbol.
    """

    for _ in range(8):
        previous = value
        for command in (
            "text",
            "textrm",
            "textnormal",
            "mathrm",
            "mathbf",
            "mathit",
            "mathsf",
            "mathtt",
            "operatorname",
            "mbox",
            "hbox",
        ):
            value = replace_balanced_command(
                value,
                command,
                argument_count=1,
                visible_argument=0,
            )
        if value == previous:
            break
    replacements = {
        r"\%": "%",
        r"\_": "_",
        r"\&": "&",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)

    def command_replacement(match: re.Match[str]) -> str:
        command = match.group(1)
        if command in MATH_VISIBLE_COMMANDS:
            return MATH_VISIBLE_COMMANDS[command]
        if command in MATH_INVISIBLE_COMMANDS:
            return ""
        return f"⟦{command}⟧"

    value = re.sub(r"\\([A-Za-z]+|[,;:!])", command_replacement, value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("_", "").replace("^", "")
    value = value.replace("-", "−")
    return value


def markdown_visible_text(value: str) -> str:
    """Remove Markdown syntax while preserving characters intended to print."""

    value = html.unescape(value)
    math_spans: list[str] = []
    code_spans: list[str] = []
    escaped_literals: list[str] = []

    def stash_math(match: re.Match[str]) -> str:
        math_spans.append(math_source_visible_text(match.group(1)))
        return f"\uFFF0{len(math_spans) - 1}\uFFF1"

    value = re.sub(r"(?<!\\)\$(?!\$)(.*?)(?<!\\)\$", stash_math, value, flags=re.DOTALL)
    value = re.sub(r"\\\((.*?)\\\)", stash_math, value, flags=re.DOTALL)

    def stash_code_span(match: re.Match[str]) -> str:
        body = match.group("body")
        if (
            len(body) >= 2
            and body.startswith(" ")
            and body.endswith(" ")
            and body.strip(" ")
        ):
            body = body[1:-1]
        code_spans.append(body)
        return f"\uFFD0{len(code_spans) - 1}\uFFD1"

    code_span = re.compile(
        r"(?<!`)(?P<fence>`+)(?!`)(?P<body>.*?)(?<!`)\1(?!`)",
        flags=re.DOTALL,
    )
    value = code_span.sub(stash_code_span, value)

    def stash_escaped_literal(match: re.Match[str]) -> str:
        escaped_literals.append(match.group(1))
        return f"\uFFE0{len(escaped_literals) - 1}\uFFE1"

    value = re.sub(
        r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])",
        stash_escaped_literal,
        value,
    )
    value = re.sub(r"<sup>(.*?)</sup>", r"\1", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(
        r"</?(?:table|thead|tbody|tr|th|td)>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", value)
    value = re.sub(r"(?m)^\s*[-+]\s+", "• ", value)
    value = re.sub(r"(?m)^\s*\*\s+", "• ", value)

    star_runs = list(re.finditer(r"\*+", value))
    opening_stack: list[int] = []
    paired_indices: set[int] = set()
    for run in star_runs:
        previous = value[run.start() - 1] if run.start() else ""
        following = value[run.end()] if run.end() < len(value) else ""
        can_close = bool(previous and not previous.isspace())
        can_open = bool(following and not following.isspace())
        run_indices = list(range(run.start(), run.end()))
        if can_close:
            while run_indices and opening_stack:
                paired_indices.add(opening_stack.pop())
                paired_indices.add(run_indices.pop(0))
        if can_open:
            opening_stack.extend(reversed(run_indices))
    if paired_indices:
        value = "".join(
            character
            for index, character in enumerate(value)
            if index not in paired_indices
        )
    for index, span in enumerate(math_spans):
        value = value.replace(f"\uFFF0{index}\uFFF1", span)
    for index, span in enumerate(code_spans):
        value = value.replace(f"\uFFD0{index}\uFFD1", span)
    for index, literal in enumerate(escaped_literals):
        value = value.replace(f"\uFFE0{index}\uFFE1", literal)
    return value


def exact_visible_character_stream(value: str, *, markdown: bool) -> str:
    """Canonicalize layout artifacts, never visible punctuation or case."""

    if markdown:
        value = markdown_visible_text(value)
    value = unicodedata.normalize("NFKC", value).replace("\u00ad", "")
    value = re.sub(r"\s+", "", value)
    return value


def visible_streams_match(expected: str, observed: str) -> bool:
    """Match with each visually printed line-end hyphen optional once."""

    states = {0}
    for character in observed:
        next_states: set[int] = set()
        if character == OPTIONAL_LINE_END_HYPHEN:
            for index in states:
                next_states.add(index)
                if index < len(expected) and expected[index] == "-":
                    next_states.add(index + 1)
        else:
            for index in states:
                if index < len(expected) and expected[index] == character:
                    next_states.add(index + 1)
        if not next_states:
            return False
        states = next_states
    return len(expected) in states


def verifier_result(markdown: str, pdf_text: str) -> dict[str, Any]:
    expected = page_gt.normalize_tokens(markdown)
    observed = page_gt.normalize_tokens(pdf_text)
    prefix = 0
    for left, right in zip(expected, observed):
        if left != right:
            break
        prefix += 1
    token_exact = expected == observed
    expected_characters = exact_visible_character_stream(markdown, markdown=True)
    observed_characters = exact_visible_character_stream(pdf_text, markdown=False)
    character_stream_exact = visible_streams_match(
        expected_characters, observed_characters
    )
    observed_hash_characters = (
        expected_characters
        if character_stream_exact
        else observed_characters.replace(OPTIONAL_LINE_END_HYPHEN, "")
    )
    if character_stream_exact:
        # Poppler occasionally merges two adjacent visual words.  Whitespace
        # is layout-only, but punctuation and case remain exact in this stream.
        match_mode = "exact_visible_character_stream"
    elif token_exact:
        match_mode = "punctuation_or_case_mismatch"
    elif not expected and observed:
        match_mode = "missing_source_content"
    elif expected and not observed:
        match_mode = "missing_pdf_content"
    elif len(observed) < len(expected) and expected[: len(observed)] == observed:
        match_mode = "source_fragment_overflows_page"
    elif len(expected) < len(observed) and observed[: len(expected)] == expected:
        match_mode = "unclaimed_pdf_suffix"
    elif len(expected) < len(observed) and observed[-len(expected) :] == expected:
        match_mode = "unclaimed_pdf_prefix"
    else:
        match_mode = "content_or_order_mismatch"
    passed = character_stream_exact
    return {
        "contract_version": VERIFIER_CONTRACT_VERSION,
        "status": "passed" if passed else "failed",
        "match_mode": match_mode,
        "expected_tokens": len(expected),
        "observed_tokens": len(observed),
        "exact_ordered_token_match": token_exact,
        "exact_ordered_character_stream_match": character_stream_exact,
        "matching_prefix_tokens": prefix,
        "expected_sha256": hashlib.sha256("\n".join(expected).encode("utf-8")).hexdigest(),
        "observed_sha256": hashlib.sha256("\n".join(observed).encode("utf-8")).hexdigest(),
        "expected_character_stream_sha256": hashlib.sha256(
            expected_characters.encode("utf-8")
        ).hexdigest(),
        "observed_character_stream_sha256": hashlib.sha256(
            observed_hash_characters.encode("utf-8")
        ).hexdigest(),
        "optional_line_end_hyphens": observed_characters.count(
            OPTIONAL_LINE_END_HYPHEN
        ),
        "first_expected_mismatch": expected[prefix] if prefix < len(expected) else None,
        "first_observed_mismatch": observed[prefix] if prefix < len(observed) else None,
        "first_expected_character_mismatch": next(
            (
                expected_characters[index]
                for index in range(
                    min(len(expected_characters), len(observed_hash_characters))
                )
                if expected_characters[index] != observed_hash_characters[index]
            ),
            None,
        ),
        "first_observed_character_mismatch": next(
            (
                observed_hash_characters[index]
                for index in range(
                    min(len(expected_characters), len(observed_hash_characters))
                )
                if expected_characters[index] != observed_hash_characters[index]
            ),
            None,
        ),
    }


def main() -> int:
    args = parse_args()
    color_pilot.LATEXMK = args.latexmk.expanduser().absolute()
    page_gt.PDFTOPPM = args.pdftoppm.expanduser().resolve()
    for name, tool in (
        ("latexmk", color_pilot.LATEXMK),
        ("pdftoppm", page_gt.PDFTOPPM),
    ):
        if not tool.is_file() or not os.access(tool, os.X_OK):
            raise FileNotFoundError(f"required executable unavailable: {name}={tool}")
    started = time.monotonic()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not (source_dir / args.main_tex).is_file():
        raise FileNotFoundError(source_dir / args.main_tex)
    print(
        f"[start] contract={CONTRACT} paper={args.paper_id} source={source_dir} "
        f"main={args.main_tex} output={output_dir}",
        flush=True,
    )
    clean_root = output_dir / "source_clean"
    colored_root = output_dir / "source_colored"
    output_dir.mkdir(parents=True)
    shutil.copytree(source_dir, clean_root)
    if args.drop_references:
        reference_report = color_pilot.strip_references_tree(clean_root)
    else:
        reference_report = {"status": "disabled", "residuals": []}
    if args.drop_figures:
        figure_report = strip_ignored_figures_tree(clean_root)
        figure_policy = "drop_figures"
    else:
        figure_report = {"status": "disabled"}
        figure_policy = "keep_figures"
    main_path = clean_root / args.main_tex
    atomic_write_text(
        main_path,
        inject_compile_support(main_path.read_text(encoding="utf-8", errors="replace")),
    )
    clean_build = output_dir / "build_clean"
    clean_pdf = color_pilot.run_compile(
        source_root=clean_root,
        main_tex=args.main_tex,
        build_dir=clean_build,
        log_path=output_dir / "logs" / "clean.log",
        label="source-first-clean",
        timeout_seconds=args.compile_timeout,
        engine=args.engine,
    )
    executed_project_sources = compiled_project_sources(clean_root, clean_build)
    executed = [
        path
        for path in executed_project_sources
        if path.suffix.casefold() in {".tex", ".ltx"}
    ]
    if not executed:
        raise RuntimeError("clean compile exposed no executed TeX sources")
    math_macros = page_gt.collect_simple_math_macros(clean_root)
    structural_blocks = page_gt.parse_source_blocks(clean_root, math_macros)
    paragraphs = page_gt.parse_source_paragraphs(clean_root, structural_blocks, executed)
    aux_paths = sorted(clean_build.glob("*.aux"))
    references = parse_aux_references(aux_paths)
    heading_label_templates, heading_label_report = parse_unique_titleformat_labels(
        executed_project_sources
    )
    units, source_rejections = build_source_units(paragraphs, references=references)
    heading_units, heading_rejections = build_heading_units(
        structural_blocks,
        aux_paths,
        color_index_offset=len(units),
        heading_label_templates=heading_label_templates,
    )
    units.extend(heading_units)
    source_rejections.extend(heading_rejections)
    units, overlap_rejections = reject_line_overlaps(units)
    source_rejections.extend(overlap_rejections)
    if not units:
        raise RuntimeError("no source-renderable ordinary paragraphs")
    probe_tiers: list[tuple[str, set[str]]] = [
        ("paragraph_and_list_tokens", {"paragraph", "itemize_item", "enumerate_item"}),
        ("paragraph_tokens", {"paragraph"}),
        ("whole_units", set()),
    ]
    probe_attempts: list[dict[str, Any]] = []
    probes: list[SourceProbe] = []
    localization_modes: dict[str, str] = {}
    mode_counts: collections.Counter[str] = collections.Counter()
    geometry: dict[str, Any] | None = None
    colored_pdf: Path | None = None
    probe_tier = ""
    for attempt, (tier_name, word_probe_kinds) in enumerate(probe_tiers, start=1):
        tier_probes, tier_modes = build_source_probes(
            units,
            word_probe_kinds=word_probe_kinds,
        )
        tier_counts = collections.Counter(tier_modes.values())
        print(
            f"[source_units] tier={tier_name} attempt={attempt}/{len(probe_tiers)} "
            f"paragraphs={len(paragraphs)} accepted={len(units)} "
            f"headings={len(heading_units)} overlaps_rejected={len(overlap_rejections)} "
            f"rejected={len(source_rejections)} probes={len(tier_probes)} "
            f"plain_word_units={tier_counts.get('plain_word', 0)} "
            f"whole_units={tier_counts.get('whole', 0)} files={len(executed)}",
            flush=True,
        )
        if colored_root.exists():
            shutil.rmtree(colored_root)
        shutil.copytree(clean_root, colored_root)
        instrument_source_tree(clean_root, colored_root, units, tier_probes, args.engine)
        suffix = "" if attempt == 1 else f"_{tier_name}"
        colored_build = output_dir / f"build_colored{suffix}"
        try:
            candidate_pdf = color_pilot.run_compile(
                source_root=colored_root,
                main_tex=args.main_tex,
                build_dir=colored_build,
                log_path=output_dir / "logs" / f"colored{suffix}.log",
                label=f"source-first-colored-{tier_name}",
                timeout_seconds=args.compile_timeout,
                engine=args.engine,
            )
            candidate_geometry = color_pilot.compare_pdf_geometry(
                clean_pdf, candidate_pdf
            )
            probe_attempts.append(
                {
                    "tier": tier_name,
                    "status": "compiled",
                    "page_count_equal": candidate_geometry["page_count_equal"],
                    "geometry_status": candidate_geometry["status"],
                    "probes": len(tier_probes),
                    "unit_modes": dict(sorted(tier_counts.items())),
                }
            )
            print(
                f"[geometry] tier={tier_name} status={candidate_geometry['status']} "
                f"page_count_equal={candidate_geometry['page_count_equal']} "
                f"pages={candidate_geometry['pages_compared']} "
                f"max_shift_points={candidate_geometry['max_geometry_shift_points']}",
                flush=True,
            )
            if not candidate_geometry["page_count_equal"]:
                continue
        except (RuntimeError, TimeoutError, FileNotFoundError) as error:
            probe_attempts.append(
                {
                    "tier": tier_name,
                    "status": "compile_failed",
                    "error": f"{type(error).__name__}: {error}",
                    "probes": len(tier_probes),
                    "unit_modes": dict(sorted(tier_counts.items())),
                }
            )
            print(
                f"[probe_fallback] tier={tier_name} reason=compile_failed "
                f"error={type(error).__name__}:{error}",
                flush=True,
            )
            continue
        probes = tier_probes
        localization_modes = tier_modes
        mode_counts = tier_counts
        geometry = candidate_geometry
        colored_pdf = candidate_pdf
        probe_tier = tier_name
        break
    if geometry is None or colored_pdf is None:
        raise RuntimeError(
            "all source probe tiers changed page count or failed compilation: "
            + json.dumps(probe_attempts, ensure_ascii=False)
        )
    geometry_by_page = {
        int(page["page_number"]): page for page in geometry["pages"]
    }
    shadow_text_rejected_pages = {
        page_number
        for page_number, page in geometry_by_page.items()
        if not page["character_count_equal"] or not page["character_text_equal"]
    }
    geometry_diagnostic_pages = {
        page_number
        for page_number, page in geometry_by_page.items()
        if not page["geometry_equal"]
    }
    if shadow_text_rejected_pages:
        print(
            f"[shadow_text_filter] rejected_pages={len(shadow_text_rejected_pages)} "
            f"pages={sorted(shadow_text_rejected_pages)}",
            flush=True,
        )
    if geometry_diagnostic_pages:
        print(
            f"[geometry_diagnostic] drift_pages={len(geometry_diagnostic_pages)} "
            f"pages={sorted(geometry_diagnostic_pages)}",
            flush=True,
        )
    color_rows, color_summary = extract_color_geometry(colored_pdf, probes)
    placements_by_page, localization_reasons, localization_summary = build_page_fragments(
        units, probes, color_rows
    )
    page_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    pages_dir = output_dir / "pages"
    with pdfplumber.open(clean_pdf) as document:
        page_limit = min(len(document.pages), args.max_pages)
        for page_number in range(1, page_limit + 1):
            page = document.pages[page_number - 1]
            reasons: list[str] = []
            placements = placements_by_page.get(page_number, [])
            shadow_geometry = geometry_by_page.get(page_number)
            if shadow_geometry is None or page_number in shadow_text_rejected_pages:
                reasons.append("color_shadow_page_text_mismatch")
            if not placements:
                reasons.append("no_colored_source_paragraphs")
            reasons.extend(sorted(localization_reasons.get(page_number, set())))
            ordered, order_error = order_page_units(placements, float(page.width))
            if order_error:
                reasons.append(order_error)
            markdown = compose_page_markdown(ordered)
            pdf_text, pdf_layout = pdf_verifier_text(page)
            verifier = verifier_result(markdown, pdf_text)
            if verifier["status"] != "passed":
                reasons.append("pdf_content_or_order_mismatch")
            status = "passed" if not reasons else "rejected"
            row = {
                "schema_version": SCHEMA_VERSION,
                "contract": CONTRACT,
                "probe_policy_version": PROBE_POLICY_VERSION,
                "shadow_invariant_policy_version": SHADOW_INVARIANT_POLICY_VERSION,
                "heading_label_policy_version": HEADING_LABEL_POLICY_VERSION,
                "figure_policy": figure_policy,
                "data_id": f"{args.paper_id}_page_{page_number:04d}",
                "paper_id": args.paper_id,
                "page_number": page_number,
                "status": status,
                "rejection_reasons": sorted(set(reasons)),
                "generation_source": "latex_source",
                "page_provenance": "compiled_vector_color",
                "pdf_role": "independent_verifier_only",
                "layout": pdf_layout,
                "source_unit_ids": list(
                    dict.fromkeys(fragment.unit_id for fragment, _ in ordered)
                ),
                "source_paragraph_ids": list(
                    dict.fromkeys(fragment.paragraph_id for fragment, _ in ordered)
                ),
                "source_probe_ids": [
                    probe_id
                    for fragment, _ in ordered
                    for probe_id in fragment.probe_ids
                ],
                "color_placements": [
                    {
                        "fragment_id": fragment.fragment_id,
                        "unit_id": fragment.unit_id,
                        "probe_ids": list(fragment.probe_ids),
                        **placement,
                    }
                    for fragment, placement in ordered
                ],
                "verifier": verifier,
                "shadow_invariant": {
                    "character_count_equal": (
                        shadow_geometry.get("character_count_equal")
                        if shadow_geometry is not None
                        else False
                    ),
                    "character_text_equal": (
                        shadow_geometry.get("character_text_equal")
                        if shadow_geometry is not None
                        else False
                    ),
                    "geometry_equal": (
                        shadow_geometry.get("geometry_equal")
                        if shadow_geometry is not None
                        else False
                    ),
                    "max_geometry_shift_points": (
                        shadow_geometry.get("max_geometry_shift_points")
                        if shadow_geometry is not None
                        else None
                    ),
                    "geometry_role": "diagnostic_only",
                },
            }
            if status == "passed":
                stem = f"page_{page_number:04d}"
                markdown_path = pages_dir / f"{stem}.md"
                image_path = pages_dir / f"{stem}.png"
                metadata_path = pages_dir / f"{stem}.json"
                atomic_write_text(markdown_path, markdown)
                page_gt.render_page_png(clean_pdf, page_number, image_path, args.dpi)
                row.update(
                    {
                        "markdown": markdown_path.relative_to(output_dir).as_posix(),
                        "image": image_path.relative_to(output_dir).as_posix(),
                        "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                    }
                )
                atomic_write_json(metadata_path, row)
                page_rows.append(row)
            else:
                rejected_rows.append(row)
            completed = page_number
            elapsed = max(time.monotonic() - started, 1e-9)
            throughput = completed / elapsed
            eta = (page_limit - completed) / throughput if throughput else 0.0
            print(
                f"[page_done] page={page_number}/{page_limit} status={status} "
                f"accepted={len(page_rows)} rejected={len(rejected_rows)} "
                f"pct={100*completed/max(1,page_limit):.1f}% throughput={throughput:.3f} pages/s "
                f"elapsed={color_pilot.elapsed_text(elapsed)} eta={color_pilot.elapsed_text(eta)}",
                flush=True,
            )
    write_jsonl(output_dir / "source_units.jsonl", (unit.as_json(clean_root) for unit in units))
    write_jsonl(output_dir / "source_probes.jsonl", (probe.as_json(clean_root) for probe in probes))
    write_jsonl(output_dir / "source_rejections.jsonl", source_rejections)
    write_jsonl(output_dir / "color_page_alignment.jsonl", (
        {"probe_id": probe_id, "pages": pages} for probe_id, pages in color_rows.items()
    ))
    write_jsonl(output_dir / "pages_passed.jsonl", page_rows)
    write_jsonl(output_dir / "pages_rejected.jsonl", rejected_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "probe_policy_version": PROBE_POLICY_VERSION,
        "shadow_invariant_policy_version": SHADOW_INVARIANT_POLICY_VERSION,
        "heading_label_policy_version": HEADING_LABEL_POLICY_VERSION,
        "figure_policy": figure_policy,
        "status": "passed" if page_rows else "failed",
        "paper_id": args.paper_id,
        "source_dir": str(source_dir),
        "main_tex": args.main_tex.as_posix(),
        "compile_engine": args.engine,
        "clean_pdf": str(clean_pdf),
        "colored_pdf": str(colored_pdf),
        "reference_removal": reference_report,
        "figure_removal": figure_report,
        "geometry_validation": geometry,
        "shadow_page_invariant": {
            "policy_version": SHADOW_INVARIANT_POLICY_VERSION,
            "text_mismatch_pages": sorted(shadow_text_rejected_pages),
            "geometry_diagnostic_pages": sorted(geometry_diagnostic_pages),
        },
        "heading_label_resolution": heading_label_report,
        "color_alignment": color_summary,
        "localization": {
            "selected_probe_tier": probe_tier,
            "probe_attempts": probe_attempts,
            "unit_modes": dict(sorted(mode_counts.items())),
            **localization_summary,
        },
        "verifier_contract_version": VERIFIER_CONTRACT_VERSION,
        "source_paragraphs_total": len(paragraphs),
        "source_units_renderable": len(units),
        "source_units_rejected": len(source_rejections),
        "aux_references_resolved": len(references),
        "pages_passed": len(page_rows),
        "pages_rejected": len(rejected_rows),
        "pdf_used_for_generation": False,
        "pdf_used_for_verification": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    print(
        f"[finish] status={report['status']} passed={len(page_rows)} "
        f"rejected={len(rejected_rows)} source_units={len(units)} "
        f"elapsed={color_pilot.elapsed_text(time.monotonic()-started)} output={output_dir}",
        flush=True,
    )
    return 0 if page_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
