#!/usr/bin/env python3
"""Build page-level Markdown from LaTeX source with color-only page provenance.

Ordinary prose is serialized from LaTeX source.  Unique paragraph colors in a
shadow compile determine page membership and reading order.  Text extracted
from the clean PDF is used only by an independent verifier; it never supplies
Markdown content.  The first contract is intentionally conservative: pages
with cross-page paragraphs, unresolved source constructs, unclaimed visible
text, or verifier disagreement are rejected.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Iterable, Sequence

import pdfplumber


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import arxiv_inline_markup as inline_markup  # noqa: E402
import build_arxiv_page_markdown_gt as page_gt  # noqa: E402
import build_latex_color_alignment_pilot as color_pilot  # noqa: E402


SCHEMA_VERSION = 2
CONTRACT = "source_first_color_v2"


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


def compiled_tex_sources(source_root: Path, build_dir: Path) -> list[Path]:
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
            if candidate.suffix.casefold() == ".tex" and candidate.is_file():
                values.add(candidate)
    return sorted(values)


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
    value = re.sub(r"\\(?:allowbreak|break|smallskip|medskip|bigskip)\b", " ", value)
    for command in (
        "mbox",
        "hbox",
        "textrm",
        "textnormal",
        "textsc",
        "underline",
        "uline",
    ):
        value = replace_balanced_command(
            value,
            command,
            argument_count=1,
            visible_argument=0,
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


def parse_aux_heading_numbers(aux_path: Path) -> dict[tuple[str, str], list[str]]:
    """Return compiler-assigned heading numbers from the LaTeX aux stream."""

    values: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    if not aux_path.is_file():
        return values
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
        number_match = re.match(r"\s*\\numberline\s*", title_raw)
        if number_match is None:
            continue
        number_group = page_gt.extract_balanced(title_raw, number_match.end())
        if number_group is None:
            continue
        number, title_start = number_group
        title = page_gt.latex_to_plain(title_raw[title_start:])
        key = (kind.strip(), page_gt.normalize_space(title).casefold())
        values[key].append(page_gt.latex_to_plain(number))
    return values


def build_heading_units(
    blocks: Sequence[page_gt.SourceBlock],
    aux_path: Path,
    *,
    color_index_offset: int,
) -> tuple[list[SourceUnit], list[dict[str, Any]]]:
    number_queues = parse_aux_heading_numbers(aux_path)
    units: list[SourceUnit] = []
    rejections: list[dict[str, Any]] = []
    for block in blocks:
        if block.kind != "heading" or not block.heading_command or not block.heading_source_title:
            continue
        title = page_gt.latex_to_plain(block.heading_source_title)
        if not title:
            rejections.append({"source_block_id": block.block_id, "reason": "empty heading title"})
            continue
        number: str | None = None
        if not block.heading_starred:
            key = (block.heading_command, page_gt.normalize_space(title).casefold())
            queue = number_queues.get(key, [])
            if not queue:
                rejections.append(
                    {"source_block_id": block.block_id, "reason": "heading number missing from aux"}
                )
                continue
            number = queue.pop(0)
        level = int(block.heading_level or 2)
        markdown = "#" * level + " " + ((number + " ") if number else "") + title
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


def instrument_source_file(
    source: str,
    units: Sequence[SourceUnit],
    engine: str = "pdflatex",
) -> str:
    """Switch colors at complete paragraph boundaries without TeX grouping.

    A brace group plus an injected ``\\par`` is safe for isolated pilots but
    can alter template glue or paragraph hooks when repeated across a paper.
    Declaration switches are zero-width whatsits; restoring black at the
    existing source boundary preserves the original paragraph construction.
    """

    lines = source.splitlines(keepends=True)
    prefixes: dict[int, str] = {}
    suffixes: dict[int, str] = {}
    occupied: set[int] = set()
    for unit in sorted(units, key=lambda value: value.start_line):
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
            lines[start] = (
                lines[start][: item_match.end()]
                + color_switch
                + lines[start][item_match.end() :]
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
    engine: str = "pdflatex",
) -> None:
    by_file: dict[Path, list[SourceUnit]] = collections.defaultdict(list)
    for unit in units:
        by_file[unit.source_file.resolve()].append(unit)
    for clean_path, file_units in by_file.items():
        relative = clean_path.relative_to(clean_root.resolve())
        colored_path = colored_root / relative
        rendered = instrument_source_file(
            colored_path.read_text(encoding="utf-8", errors="replace"),
            file_units,
            engine,
        )
        atomic_write_text(colored_path, rendered)


def extract_color_geometry(
    colored_pdf: Path,
    units: Sequence[SourceUnit],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Map source IDs to pages/bboxes without reading PDF character text."""

    unit_by_rgb = {unit.rgb: unit for unit in units}
    observed: dict[str, dict[int, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    with pdfplumber.open(colored_pdf) as document:
        total_pages = len(document.pages)
        for page_number, page in enumerate(document.pages, start=1):
            matched = 0
            for char in page.chars:
                rgb = color_pilot.normalize_pdf_rgb(char.get("non_stroking_color"))
                unit = unit_by_rgb.get(rgb)
                if unit is None:
                    continue
                observed[unit.unit_id][page_number].append(char)
                matched += 1
            print(
                f"[color_page] page={page_number}/{total_pages} "
                f"matched_characters={matched}",
                flush=True,
            )
    rows: dict[str, list[dict[str, Any]]] = {}
    mapped = multi_page = 0
    for unit in units:
        pages: list[dict[str, Any]] = []
        for page_number, chars in sorted(observed.get(unit.unit_id, {}).items()):
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
        rows[unit.unit_id] = pages
        if pages:
            mapped += 1
        if len(pages) > 1:
            multi_page += 1
    return rows, {
        "units_total": len(units),
        "units_mapped": mapped,
        "units_unmapped": len(units) - mapped,
        "units_spanning_multiple_pages": multi_page,
        "coverage": round(mapped / max(1, len(units)), 6),
    }


def order_page_units(
    placements: Sequence[tuple[SourceUnit, dict[str, Any]]],
    page_width: float,
) -> tuple[list[tuple[SourceUnit, dict[str, Any]]], str | None]:
    midpoint = page_width / 2.0
    gutter = max(8.0, page_width * 0.025)
    left: list[tuple[SourceUnit, dict[str, Any]]] = []
    right: list[tuple[SourceUnit, dict[str, Any]]] = []
    full: list[tuple[SourceUnit, dict[str, Any]]] = []
    for placement in placements:
        bbox = placement[1]["bbox_points"]
        if bbox[2] <= midpoint + gutter:
            left.append(placement)
        elif bbox[0] >= midpoint - gutter:
            right.append(placement)
        else:
            full.append(placement)
    key = lambda item: (float(item[1]["bbox_points"][1]), float(item[1]["bbox_points"][0]))
    if left and right and full:
        return [], "mixed_full_and_columns"
    if left and right:
        return sorted(left, key=key) + sorted(right, key=key), None
    return sorted([*left, *right, *full], key=key), None


def pdf_verifier_text(page: Any) -> tuple[str, str]:
    nodes = page_gt.words_to_line_nodes(page)
    ordered, layout = page_gt.order_page_nodes(nodes, float(page.width))
    return page_gt.join_text_lines(ordered), layout


def verifier_result(markdown: str, pdf_text: str) -> dict[str, Any]:
    expected = page_gt.normalize_tokens(markdown)
    observed = page_gt.normalize_tokens(pdf_text)
    prefix = 0
    for left, right in zip(expected, observed):
        if left != right:
            break
        prefix += 1
    exact = expected == observed
    return {
        "status": "passed" if exact else "failed",
        "expected_tokens": len(expected),
        "observed_tokens": len(observed),
        "exact_ordered_token_match": exact,
        "matching_prefix_tokens": prefix,
        "expected_sha256": hashlib.sha256("\n".join(expected).encode("utf-8")).hexdigest(),
        "observed_sha256": hashlib.sha256("\n".join(observed).encode("utf-8")).hexdigest(),
        "first_expected_mismatch": expected[prefix] if prefix < len(expected) else None,
        "first_observed_mismatch": observed[prefix] if prefix < len(observed) else None,
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
    executed = compiled_tex_sources(clean_root, clean_build)
    if not executed:
        raise RuntimeError("clean compile exposed no executed TeX sources")
    math_macros = page_gt.collect_simple_math_macros(clean_root)
    structural_blocks = page_gt.parse_source_blocks(clean_root, math_macros)
    paragraphs = page_gt.parse_source_paragraphs(clean_root, structural_blocks, executed)
    aux_paths = sorted(clean_build.glob("*.aux"))
    references = parse_aux_references(aux_paths)
    units, source_rejections = build_source_units(paragraphs, references=references)
    heading_units, heading_rejections = build_heading_units(
        structural_blocks,
        clean_build / args.main_tex.with_suffix(".aux").name,
        color_index_offset=len(units),
    )
    units.extend(heading_units)
    source_rejections.extend(heading_rejections)
    units, overlap_rejections = reject_line_overlaps(units)
    source_rejections.extend(overlap_rejections)
    if not units:
        raise RuntimeError("no source-renderable ordinary paragraphs")
    print(
        f"[source_units] paragraphs={len(paragraphs)} accepted={len(units)} "
        f"headings={len(heading_units)} overlaps_rejected={len(overlap_rejections)} "
        f"rejected={len(source_rejections)} files={len(executed)}",
        flush=True,
    )
    shutil.copytree(clean_root, colored_root)
    instrument_source_tree(clean_root, colored_root, units, args.engine)
    colored_build = output_dir / "build_colored"
    colored_pdf = color_pilot.run_compile(
        source_root=colored_root,
        main_tex=args.main_tex,
        build_dir=colored_build,
        log_path=output_dir / "logs" / "colored.log",
        label="source-first-colored",
        timeout_seconds=args.compile_timeout,
        engine=args.engine,
    )
    geometry = color_pilot.compare_pdf_geometry(clean_pdf, colored_pdf)
    print(
        f"[geometry] status={geometry['status']} pages={geometry['pages_compared']} "
        f"max_shift_points={geometry['max_geometry_shift_points']}",
        flush=True,
    )
    if not geometry["page_count_equal"]:
        raise RuntimeError("color instrumentation changed PDF page count")
    geometry_rejected_pages = {
        int(page["page_number"])
        for page in geometry["pages"]
        if not page["character_text_equal"] or not page["geometry_equal"]
    }
    if geometry_rejected_pages:
        print(
            f"[geometry_filter] rejected_pages={len(geometry_rejected_pages)} "
            f"pages={sorted(geometry_rejected_pages)}",
            flush=True,
        )
    color_rows, color_summary = extract_color_geometry(colored_pdf, units)
    unit_by_id = {unit.unit_id: unit for unit in units}
    placements_by_page: dict[int, list[tuple[SourceUnit, dict[str, Any]]]] = collections.defaultdict(list)
    multi_page_pages: set[int] = set()
    for unit_id, pages in color_rows.items():
        if len(pages) > 1:
            multi_page_pages.update(int(page["page_number"]) for page in pages)
        for page in pages:
            placements_by_page[int(page["page_number"])].append((unit_by_id[unit_id], page))
    page_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    pages_dir = output_dir / "pages"
    with pdfplumber.open(clean_pdf) as document:
        page_limit = min(len(document.pages), args.max_pages)
        for page_number in range(1, page_limit + 1):
            page = document.pages[page_number - 1]
            reasons: list[str] = []
            placements = placements_by_page.get(page_number, [])
            if page_number in geometry_rejected_pages:
                reasons.append("color_geometry_mismatch")
            if not placements:
                reasons.append("no_colored_source_paragraphs")
            if page_number in multi_page_pages:
                reasons.append("cross_page_source_paragraph")
            ordered, order_error = order_page_units(placements, float(page.width))
            if order_error:
                reasons.append(order_error)
            markdown = "\n\n".join(unit.markdown for unit, _ in ordered).strip() + "\n"
            pdf_text, pdf_layout = pdf_verifier_text(page)
            verifier = verifier_result(markdown, pdf_text)
            if verifier["status"] != "passed":
                reasons.append("pdf_content_or_order_mismatch")
            status = "passed" if not reasons else "rejected"
            row = {
                "schema_version": SCHEMA_VERSION,
                "contract": CONTRACT,
                "data_id": f"{args.paper_id}_page_{page_number:04d}",
                "paper_id": args.paper_id,
                "page_number": page_number,
                "status": status,
                "rejection_reasons": sorted(set(reasons)),
                "generation_source": "latex_source",
                "page_provenance": "compiled_vector_color",
                "pdf_role": "independent_verifier_only",
                "layout": pdf_layout,
                "source_unit_ids": [unit.unit_id for unit, _ in ordered],
                "source_paragraph_ids": [unit.paragraph_id for unit, _ in ordered],
                "color_placements": [
                    {"unit_id": unit.unit_id, **placement} for unit, placement in ordered
                ],
                "verifier": verifier,
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
    write_jsonl(output_dir / "source_rejections.jsonl", source_rejections)
    write_jsonl(output_dir / "color_page_alignment.jsonl", (
        {"unit_id": unit_id, "pages": pages} for unit_id, pages in color_rows.items()
    ))
    write_jsonl(output_dir / "pages_passed.jsonl", page_rows)
    write_jsonl(output_dir / "pages_rejected.jsonl", rejected_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": "passed" if page_rows else "failed",
        "paper_id": args.paper_id,
        "source_dir": str(source_dir),
        "main_tex": args.main_tex.as_posix(),
        "compile_engine": args.engine,
        "clean_pdf": str(clean_pdf),
        "colored_pdf": str(colored_pdf),
        "reference_removal": reference_report,
        "geometry_validation": geometry,
        "color_alignment": color_summary,
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
