"""Strict source-derived LaTeX tabular AST and HTML serialization.

This module deliberately implements a small, fail-closed table language.  It
never reads PDF text and it never falls back to embedding raw LaTeX.  A table
is admitted only when every row, cell, span, and visible cell token can be
reconstructed from immutable LaTeX source.  Compiler traces may subsequently
place the resulting block on a page, but cannot change its HTML or text.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .ast_ir import (
    MathMacroDefinition,
    SourceDocumentIR,
    SourceIrError,
    parse_source_ir,
)

TABLE_AST_VERSION = "source_first_v3_table_ast_v2"

_SUPPORTED_ENVIRONMENTS = frozenset({"tabular", "tabular*", "tabularx", "array"})
_SUPPORTED_CELL_ENVIRONMENTS = frozenset({"enumerate", "itemize"})
_STYLE_TAGS = {
    "strong": "strong",
    "em": "em",
    "body_em": "em",
    "code": "code",
    "sup": "sup",
}
_BEGIN = re.compile(r"\\begin\s*\{\s*([A-Za-z*]+)\s*\}")
_ENVIRONMENT_TOKEN = re.compile(
    r"\\(begin|end)\s*\{\s*([A-Za-z*]+)\s*\}"
)
_PAR_COMMAND = re.compile(r"\\par(?![A-Za-z@])")
_ITEM_COMMAND = re.compile(r"\\item(?![A-Za-z@])")
_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*")
_LITERAL_COLUMN_SPEC = re.compile(r"[A-Za-z0-9*{}@.<>|!+\-/:;=, \t\\]+")
_LITERAL_DIMENSION_OR_STAR = re.compile(
    r"(?:\*|[+\-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em))"
)


class TableAstError(ValueError):
    """The table cannot be serialized without guessing."""


@dataclass(frozen=True, slots=True)
class TableCellAst:
    html: str
    visible_text: str
    verifier_fragments: tuple[tuple[str, bool], ...]
    colspan: int
    rowspan: int
    source_char_span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TableRowAst:
    cells: tuple[TableCellAst, ...]
    source_char_span: tuple[int, int]
    header: bool


@dataclass(frozen=True, slots=True)
class StrictTableAst:
    environment_name: str
    rows: tuple[TableRowAst, ...]
    html: str
    visible_text: str
    verifier_fragments: tuple[tuple[str, bool], ...]
    anchor_char_span: tuple[int, int]
    source_char_span: tuple[int, int]
    column_count: int

    def to_metadata(self) -> tuple[tuple[str, str], ...]:
        return (
            ("table_ast_version", TABLE_AST_VERSION),
            ("table_html", self.html),
            ("table_visible_text", self.visible_text),
            (
                "table_verifier_fragments_json",
                json.dumps(
                    [
                        {"text": text, "verifiable": verifiable}
                        for text, verifiable in self.verifier_fragments
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
            ("table_anchor_char_start", str(self.anchor_char_span[0])),
            ("table_anchor_char_end", str(self.anchor_char_span[1])),
            ("table_row_count", str(len(self.rows))),
            ("table_column_count", str(self.column_count)),
        )


def is_serializable_table_metadata(metadata: Mapping[str, str]) -> bool:
    """Return whether metadata carries one complete strict table contract."""

    return (
        metadata.get("table_ast_version") == TABLE_AST_VERSION
        and bool(metadata.get("table_html", "").startswith("<table>\n"))
        and bool(metadata.get("table_visible_text", "").strip())
        and bool(metadata.get("table_verifier_fragments_json", "").strip())
        and metadata.get("table_anchor_char_start", "").isdigit()
        and metadata.get("table_anchor_char_end", "").isdigit()
    )


def load_table_verifier_fragments(
    metadata: Mapping[str, str],
) -> tuple[tuple[str, bool], ...]:
    """Load the immutable source-derived verifier runs from table metadata."""

    if not is_serializable_table_metadata(metadata):
        raise TableAstError("table metadata is not serializable")
    try:
        raw = json.loads(metadata["table_verifier_fragments_json"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TableAstError("table verifier metadata is malformed") from exc
    if not isinstance(raw, list) or not raw:
        raise TableAstError("table verifier metadata has no fragments")
    output: list[tuple[str, bool]] = []
    for row in raw:
        if (
            not isinstance(row, dict)
            or set(row) != {"text", "verifiable"}
            or not isinstance(row["text"], str)
            or not row["text"]
            or not isinstance(row["verifiable"], bool)
        ):
            raise TableAstError("table verifier fragment is unsafe")
        output.append((row["text"], row["verifiable"]))
    return tuple(output)


def _is_unescaped(source: str, position: int) -> bool:
    slashes = 0
    cursor = position - 1
    while cursor >= 0 and source[cursor] == "\\":
        slashes += 1
        cursor -= 1
    return slashes % 2 == 0


def _skip_comment(source: str, cursor: int, limit: int) -> int:
    newline = source.find("\n", cursor, limit)
    return limit if newline < 0 else newline + 1


def _skip_ignored(source: str, cursor: int, limit: int) -> int:
    while cursor < limit:
        if source[cursor].isspace():
            cursor += 1
            continue
        if source[cursor] == "%" and _is_unescaped(source, cursor):
            cursor = _skip_comment(source, cursor, limit)
            continue
        break
    return cursor


def _balanced_end(
    source: str,
    opening: int,
    limit: int,
    left: str = "{",
    right: str = "}",
) -> int:
    if opening >= limit or source[opening] != left:
        raise TableAstError(f"expected balanced {left}{right} argument")
    depth = 0
    cursor = opening
    while cursor < limit:
        char = source[cursor]
        if char == "%" and _is_unescaped(source, cursor):
            cursor = _skip_comment(source, cursor, limit)
            continue
        if char == "\\":
            cursor += 2
            continue
        if char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    raise TableAstError(f"unterminated balanced {left}{right} argument")


def _required_group(source: str, cursor: int, limit: int) -> tuple[int, int, int]:
    cursor = _skip_ignored(source, cursor, limit)
    end = _balanced_end(source, cursor, limit)
    return cursor + 1, end - 1, end


def _optional_group(source: str, cursor: int, limit: int) -> tuple[int, int, int] | None:
    cursor = _skip_ignored(source, cursor, limit)
    if cursor >= limit or source[cursor] != "[":
        return None
    end = _balanced_end(source, cursor, limit, "[", "]")
    return cursor + 1, end - 1, end


def _parenthesized_group(
    source: str, cursor: int, limit: int
) -> tuple[int, int, int] | None:
    """Read booktabs' literal ``(lr)`` cmidrule trimming option."""

    cursor = _skip_ignored(source, cursor, limit)
    if cursor >= limit or source[cursor] != "(":
        return None
    end = _balanced_end(source, cursor, limit, "(", ")")
    value = source[cursor + 1 : end - 1].strip()
    if not value or re.fullmatch(r"[lr]+", value) is None:
        raise TableAstError("cmidrule trimming option is not a safe literal")
    return cursor + 1, end - 1, end


def _command_end(source: str, cursor: int, limit: int) -> tuple[str, int]:
    if cursor >= limit or source[cursor] != "\\":
        raise TableAstError("expected TeX command")
    end = cursor + 1
    if end >= limit:
        raise TableAstError("trailing TeX escape")
    if source[end].isalpha() or source[end] == "@":
        end += 1
        while end < limit and (source[end].isalpha() or source[end] == "@"):
            end += 1
        if end < limit and source[end] == "*":
            end += 1
    else:
        end += 1
    return source[cursor + 1 : end], end


def _find_environment_body(
    source: str,
    start: int,
    end: int,
) -> tuple[str, int, int]:
    match = _BEGIN.match(source, start, end)
    if match is None:
        raise TableAstError("table source does not begin with a literal environment")
    environment = match.group(1)
    if environment not in _SUPPORTED_ENVIRONMENTS:
        raise TableAstError(f"unsupported table environment: {environment}")
    cursor = match.end()
    optional = _optional_group(source, cursor, end)
    if optional is not None:
        cursor = optional[2]
    required = 2 if environment in {"tabular*", "tabularx"} else 1
    arguments: list[str] = []
    for _ in range(required):
        body_start, body_end, cursor = _required_group(source, cursor, end)
        arguments.append(source[body_start:body_end].strip())
    column_spec = arguments[-1]
    if not column_spec or _LITERAL_COLUMN_SPEC.fullmatch(column_spec) is None:
        raise TableAstError("table column specification is not a safe literal")
    body_start = cursor
    closing_pattern = re.compile(
        r"\\end\s*\{\s*" + re.escape(environment) + r"\s*\}"
    )
    closing = tuple(closing_pattern.finditer(source, body_start, end))
    if len(closing) != 1:
        raise TableAstError("table environment has nested or ambiguous closing tokens")
    stack: list[str] = []
    for token in _ENVIRONMENT_TOKEN.finditer(source, body_start, closing[0].start()):
        action, name = token.groups()
        if name not in _SUPPORTED_CELL_ENVIRONMENTS:
            raise TableAstError(
                f"unsupported nested table cell environment: {name}"
            )
        if action == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            raise TableAstError("unbalanced nested table cell environment")
    if stack:
        raise TableAstError("unterminated nested table cell environment")
    if source[closing[0].end() : end].strip():
        raise TableAstError("visible or control source follows the table environment")
    return environment, body_start, closing[0].start()


def _split_top_level(
    source: str,
    start: int,
    end: int,
    *,
    delimiter: str,
) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    token_start = start
    brace_depth = 0
    bracket_depth = 0
    dollar_math = False
    parenthesized_math = False
    environment_stack: list[str] = []
    cursor = start
    while cursor < end:
        char = source[cursor]
        if char == "%" and _is_unescaped(source, cursor):
            cursor = _skip_comment(source, cursor, end)
            continue
        if source.startswith(r"\(", cursor) and brace_depth == 0:
            parenthesized_math = True
            cursor += 2
            continue
        if source.startswith(r"\)", cursor) and parenthesized_math:
            parenthesized_math = False
            cursor += 2
            continue
        if char == "$" and _is_unescaped(source, cursor) and not parenthesized_math:
            dollar_math = not dollar_math
            cursor += 1
            continue
        if not dollar_math and not parenthesized_math:
            if brace_depth == 0 and bracket_depth == 0 and char == "\\":
                environment_token = _ENVIRONMENT_TOKEN.match(source, cursor, end)
                if environment_token is not None:
                    action, name = environment_token.groups()
                    if name not in _SUPPORTED_CELL_ENVIRONMENTS:
                        raise TableAstError(
                            f"unsupported nested table cell environment: {name}"
                        )
                    if action == "begin":
                        environment_stack.append(name)
                    elif not environment_stack or environment_stack.pop() != name:
                        raise TableAstError(
                            "unbalanced nested table cell environment"
                        )
                    cursor = environment_token.end()
                    continue
            if char == "{" and _is_unescaped(source, cursor):
                brace_depth += 1
            elif char == "}" and _is_unescaped(source, cursor):
                brace_depth -= 1
                if brace_depth < 0:
                    raise TableAstError("unbalanced table cell braces")
            elif char == "[" and _is_unescaped(source, cursor):
                bracket_depth += 1
            elif char == "]" and _is_unescaped(source, cursor):
                bracket_depth -= 1
                if bracket_depth < 0:
                    raise TableAstError("unbalanced table cell brackets")
        if (
            brace_depth == 0
            and bracket_depth == 0
            and not dollar_math
            and not parenthesized_math
            and not environment_stack
        ):
            if delimiter == "&" and char == "&" and _is_unescaped(source, cursor):
                spans.append((token_start, cursor))
                token_start = cursor + 1
                cursor += 1
                continue
            if delimiter == r"\\" and source.startswith(r"\\", cursor):
                spans.append((token_start, cursor))
                cursor += 2
                optional = _optional_group(source, cursor, end)
                if optional is not None:
                    spacing = source[optional[0] : optional[1]].strip()
                    if _LITERAL_DIMENSION_OR_STAR.fullmatch(spacing) is None:
                        raise TableAstError("row-break spacing is not a literal dimension")
                    cursor = optional[2]
                token_start = cursor
                continue
        cursor += 1
    if (
        brace_depth
        or bracket_depth
        or dollar_math
        or parenthesized_math
        or environment_stack
    ):
        raise TableAstError("unterminated table cell group or inline math")
    spans.append((token_start, end))
    return tuple(spans)


def _trim(source: str, start: int, end: int) -> tuple[int, int]:
    start = _skip_ignored(source, start, end)
    while end > start and source[end - 1].isspace():
        end -= 1
    return start, end


def _consume_rule_prefix(
    source: str,
    start: int,
    end: int,
) -> tuple[int, tuple[str, ...]]:
    rules: list[str] = []
    cursor = start
    while True:
        cursor = _skip_ignored(source, cursor, end)
        if cursor >= end or source[cursor] != "\\":
            return cursor, tuple(rules)
        name, command_end = _command_end(source, cursor, end)
        if name in {"toprule", "midrule", "bottomrule", "hline"}:
            rules.append(name)
            cursor = command_end
            if name == "midrule":
                optional = _optional_group(source, cursor, end)
                if optional is not None:
                    cursor = optional[2]
            continue
        if name in {"cmidrule", "cline"}:
            cursor = command_end
            optional = _optional_group(source, cursor, end)
            if optional is not None:
                cursor = optional[2]
            if name == "cmidrule":
                trimming = _parenthesized_group(source, cursor, end)
                if trimming is not None:
                    cursor = trimming[2]
            _arg_start, _arg_end, cursor = _required_group(source, cursor, end)
            rules.append(name)
            continue
        if name == "addlinespace":
            cursor = command_end
            optional = _optional_group(source, cursor, end)
            if optional is not None:
                cursor = optional[2]
            rules.append(name)
            continue
        return cursor, tuple(rules)


def _unwrap_span_commands(
    source: str,
    start: int,
    end: int,
) -> tuple[int, int, int, int]:
    colspan = 1
    rowspan = 1
    cursor_start, cursor_end = _trim(source, start, end)
    for _ in range(2):
        if cursor_start >= cursor_end or source[cursor_start] != "\\":
            break
        name, command_end = _command_end(source, cursor_start, cursor_end)
        if name == "multicolumn":
            count_start, count_end, cursor = _required_group(
                source, command_end, cursor_end
            )
            count = source[count_start:count_end].strip()
            spec_start, spec_end, cursor = _required_group(source, cursor, cursor_end)
            spec = source[spec_start:spec_end].strip()
            body_start, body_end, cursor = _required_group(source, cursor, cursor_end)
            if (
                _POSITIVE_INTEGER.fullmatch(count) is None
                or _LITERAL_COLUMN_SPEC.fullmatch(spec) is None
                or _skip_ignored(source, cursor, cursor_end) != cursor_end
            ):
                raise TableAstError("unsafe or nonliteral multicolumn cell")
            colspan = int(count)
            cursor_start, cursor_end = _trim(source, body_start, body_end)
            continue
        if name == "multirow":
            cursor = command_end
            optional = _optional_group(source, cursor, cursor_end)
            if optional is not None:
                cursor = optional[2]
            count_start, count_end, cursor = _required_group(source, cursor, cursor_end)
            width_start, width_end, cursor = _required_group(source, cursor, cursor_end)
            body_start, body_end, cursor = _required_group(source, cursor, cursor_end)
            count = source[count_start:count_end].strip()
            width = source[width_start:width_end].strip()
            if (
                _POSITIVE_INTEGER.fullmatch(count) is None
                or _LITERAL_DIMENSION_OR_STAR.fullmatch(width) is None
                or _skip_ignored(source, cursor, cursor_end) != cursor_end
            ):
                raise TableAstError("unsafe or nonliteral multirow cell")
            rowspan = int(count)
            cursor_start, cursor_end = _trim(source, body_start, body_end)
            continue
        break
    return cursor_start, cursor_end, colspan, rowspan


def _render_inline_html(
    ir: SourceDocumentIR,
) -> tuple[str, str, tuple[tuple[str, bool], ...]]:
    if ir.opaque_atoms or ir.footnotes:
        raise TableAstError("table cell contains unsupported inline LaTeX")
    output: list[str] = []
    visible: list[str] = []
    active: tuple[str, ...] = ()
    verifier_fragments: list[tuple[str, bool]] = []
    verifiable_run: list[str] = []

    def flush_verifiable() -> None:
        text = "".join(verifiable_run).strip()
        verifiable_run.clear()
        if text:
            verifier_fragments.append((text, True))

    def normalized_styles(styles: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for style in styles:
            if style == "smallcaps":
                continue
            if style not in _STYLE_TAGS:
                raise TableAstError(f"unsupported table cell style: {style}")
            tag = _STYLE_TAGS[style]
            if not normalized or normalized[-1] != tag:
                normalized.append(tag)
        return tuple(normalized)

    for atom in ir.atoms:
        target = normalized_styles(atom.style_stack)
        common = 0
        while common < len(active) and common < len(target) and active[common] == target[common]:
            common += 1
        for tag in reversed(active[common:]):
            output.append(f"</{tag}>")
        for tag in target[common:]:
            output.append(f"<{tag}>")
        if atom.is_whitespace:
            output.append(" ")
            visible.append(" ")
            if verifiable_run:
                verifiable_run.append(" ")
        elif atom.kind == "math":
            output.append(html.escape(atom.markdown_fragment, quote=False))
            visible.append(atom.visible_text)
            flush_verifiable()
            if atom.visible_text:
                verifier_fragments.append((atom.visible_text, False))
        elif atom.raw_source.startswith(r"\\") and atom.visible_text == "\n":
            output.append("<br>")
            visible.append(" ")
            flush_verifiable()
            verifier_fragments.append(("\n", False))
        elif atom.kind == "source_macro":
            # Keep the same source-derived Markdown serialization used by
            # ordinary page GT. In particular, a macro such as
            # ``LinuxFL\ensuremath{^{+}}`` must retain its inline formula
            # inside the HTML cell instead of exposing raw TeX as prose.
            output.append(html.escape(atom.markdown_fragment, quote=False))
            visible.append(atom.visible_text)
            for text, verifiable in atom.verifier_fragments or (
                (atom.visible_text, True),
            ):
                if verifiable:
                    verifiable_run.append(text)
                else:
                    flush_verifiable()
                    if text:
                        verifier_fragments.append((text, False))
        else:
            output.append(html.escape(atom.visible_text, quote=False))
            visible.append(atom.visible_text)
            if atom.kind in {"text", "reference"}:
                verifiable_run.append(atom.visible_text)
            else:
                flush_verifiable()
                if atom.visible_text:
                    verifier_fragments.append((atom.visible_text, False))
        active = target
    for tag in reversed(active):
        output.append(f"</{tag}>")
    html_value = re.sub(r"[ \t\r\f\v]+", " ", "".join(output)).strip()
    visible_value = " ".join("".join(visible).split())
    flush_verifiable()
    return html_value, visible_value, tuple(verifier_fragments)


def _render_inline_source(
    source: str,
    start: int,
    end: int,
    *,
    source_id: str,
    reference_values: Mapping[str, str],
    math_macros: Mapping[str, MathMacroDefinition],
) -> tuple[str, str, tuple[tuple[str, bool], ...]]:
    start, end = _trim(source, start, end)
    if start == end:
        return "", "", ()
    prefix_bytes = len(source[:start].encode("utf-8"))
    try:
        ir = parse_source_ir(
            source[start:end],
            source_id=source_id,
            source_char_base=start,
            source_byte_base=prefix_bytes,
            reference_values=reference_values,
            math_macros=math_macros,
        )
        return _render_inline_html(ir)
    except SourceIrError as exc:
        raise TableAstError(f"table cell source IR failed: {exc}") from exc


def _matching_cell_environment(
    source: str,
    begin: re.Match[str],
    limit: int,
) -> tuple[str, int, int, int]:
    """Return one balanced, source-literal list environment."""

    environment = begin.group(2)
    if begin.group(1) != "begin" or environment not in _SUPPORTED_CELL_ENVIRONMENTS:
        raise TableAstError("expected a supported table-cell list environment")
    cursor = begin.end()
    optional = _optional_group(source, cursor, limit)
    list_attributes = ""
    marker_seen = False
    if optional is not None:
        option_text = source[optional[0] : optional[1]].strip()
        # Table-cell lists are admitted only when options affect spacing or
        # select one standard ordered-list marker. Arbitrary macros here can
        # produce visible labels that source-only HTML cannot prove.
        allowed_parts = {
            "nosep",
            "noitemsep",
            "leftmargin=*",
        }
        marker_types = {
            r"label=\arabic*.": "",
            r"label=\alph*.": ' type="a"',
            r"label=\Alph*.": ' type="A"',
            r"label=\roman*.": ' type="i"',
            r"label=\Roman*.": ' type="I"',
            r"label=(\alph*)": ' type="a"',
            r"label=(\Alph*)": ' type="A"',
            r"label=(\roman*)": ' type="i"',
            r"label=(\Roman*)": ' type="I"',
        }
        for part in (value.strip() for value in option_text.split(",")):
            if not part:
                continue
            if part in marker_types:
                if environment != "enumerate" or marker_seen:
                    raise TableAstError("ambiguous table-cell list marker options")
                list_attributes = marker_types[part]
                marker_seen = True
                continue
            if part in allowed_parts or re.fullmatch(
                r"(?:itemsep|topsep|parsep|labelsep)="
                r"[+\-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
                r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em)",
                part,
            ):
                continue
            raise TableAstError("unsupported table-cell list options")
        cursor = optional[2]
    stack = [environment]
    search = cursor
    while search < limit:
        token = _ENVIRONMENT_TOKEN.search(source, search, limit)
        if token is None:
            break
        action, name = token.groups()
        if name not in _SUPPORTED_CELL_ENVIRONMENTS:
            raise TableAstError(
                f"unsupported nested table cell environment: {name}"
            )
        if action == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            raise TableAstError("unbalanced nested table cell environment")
        if not stack:
            return list_attributes, cursor, token.start(), token.end()
        search = token.end()
    raise TableAstError("unterminated nested table cell environment")


def _split_list_items(
    source: str,
    start: int,
    end: int,
) -> tuple[tuple[int, int], ...]:
    item_starts: list[int] = []
    environment_stack: list[str] = []
    brace_depth = 0
    bracket_depth = 0
    dollar_math = False
    parenthesized_math = False
    cursor = start
    while cursor < end:
        char = source[cursor]
        if char == "%" and _is_unescaped(source, cursor):
            cursor = _skip_comment(source, cursor, end)
            continue
        if source.startswith(r"\(", cursor) and brace_depth == 0:
            parenthesized_math = True
            cursor += 2
            continue
        if source.startswith(r"\)", cursor) and parenthesized_math:
            parenthesized_math = False
            cursor += 2
            continue
        if char == "$" and _is_unescaped(source, cursor) and not parenthesized_math:
            dollar_math = not dollar_math
            cursor += 1
            continue
        if not dollar_math and not parenthesized_math:
            if brace_depth == 0 and bracket_depth == 0 and char == "\\":
                token = _ENVIRONMENT_TOKEN.match(source, cursor, end)
                if token is not None:
                    action, name = token.groups()
                    if name not in _SUPPORTED_CELL_ENVIRONMENTS:
                        raise TableAstError(
                            f"unsupported nested table cell environment: {name}"
                        )
                    if action == "begin":
                        environment_stack.append(name)
                    elif not environment_stack or environment_stack.pop() != name:
                        raise TableAstError(
                            "unbalanced nested table cell environment"
                        )
                    cursor = token.end()
                    continue
                item = _ITEM_COMMAND.match(source, cursor, end)
                if item is not None and not environment_stack:
                    optional = _optional_group(source, item.end(), end)
                    if optional is not None:
                        raise TableAstError(
                            "custom table-cell list item labels are unsupported"
                        )
                    item_starts.append(item.end())
                    cursor = item.end()
                    continue
            if char == "{" and _is_unescaped(source, cursor):
                brace_depth += 1
            elif char == "}" and _is_unescaped(source, cursor):
                brace_depth -= 1
                if brace_depth < 0:
                    raise TableAstError("unbalanced table-cell list braces")
            elif char == "[" and _is_unescaped(source, cursor):
                bracket_depth += 1
            elif char == "]" and _is_unescaped(source, cursor):
                bracket_depth -= 1
                if bracket_depth < 0:
                    raise TableAstError("unbalanced table-cell list brackets")
        cursor += 1
    if (
        brace_depth
        or bracket_depth
        or dollar_math
        or parenthesized_math
        or environment_stack
    ):
        raise TableAstError("unterminated table-cell list content")
    if not item_starts:
        raise TableAstError("table-cell list has no literal items")
    prefix_end = source.rfind(r"\item", start, item_starts[0])
    if prefix_end < start or _skip_ignored(source, start, prefix_end) != prefix_end:
        raise TableAstError("visible content precedes first table-cell list item")
    return tuple(
        (item_start, item_starts[index + 1] - len(r"\item"))
        if index + 1 < len(item_starts)
        else (item_start, end)
        for index, item_start in enumerate(item_starts)
    )


def _render_cell_content(
    source: str,
    start: int,
    end: int,
    *,
    source_id: str,
    reference_values: Mapping[str, str],
    math_macros: Mapping[str, MathMacroDefinition],
) -> tuple[str, str, tuple[tuple[str, bool], ...]]:
    html_parts: list[str] = []
    visible_parts: list[str] = []
    verifier_fragments: list[tuple[str, bool]] = []

    def append_inline(part_start: int, part_end: int) -> None:
        cell_html, visible, fragments = _render_inline_source(
            source,
            part_start,
            part_end,
            source_id=source_id,
            reference_values=reference_values,
            math_macros=math_macros,
        )
        if cell_html:
            html_parts.append(cell_html)
        if visible:
            visible_parts.append(visible)
        verifier_fragments.extend(fragments)

    segment_start = start
    brace_depth = 0
    bracket_depth = 0
    dollar_math = False
    parenthesized_math = False
    cursor = start
    while cursor < end:
        char = source[cursor]
        if char == "%" and _is_unescaped(source, cursor):
            cursor = _skip_comment(source, cursor, end)
            continue
        if source.startswith(r"\(", cursor) and brace_depth == 0:
            parenthesized_math = True
            cursor += 2
            continue
        if source.startswith(r"\)", cursor) and parenthesized_math:
            parenthesized_math = False
            cursor += 2
            continue
        if char == "$" and _is_unescaped(source, cursor) and not parenthesized_math:
            dollar_math = not dollar_math
            cursor += 1
            continue
        if not dollar_math and not parenthesized_math:
            if brace_depth == 0 and bracket_depth == 0 and char == "\\":
                environment = _ENVIRONMENT_TOKEN.match(source, cursor, end)
                if environment is not None:
                    if environment.group(1) != "begin":
                        raise TableAstError(
                            "unexpected table-cell environment closing token"
                        )
                    append_inline(segment_start, cursor)
                    attributes, body_start, body_end, environment_end = (
                        _matching_cell_environment(source, environment, end)
                    )
                    item_html: list[str] = []
                    item_visible: list[str] = []
                    item_fragments: list[tuple[str, bool]] = []
                    for item_start, item_end in _split_list_items(
                        source, body_start, body_end
                    ):
                        rendered, visible, fragments = _render_cell_content(
                            source,
                            item_start,
                            item_end,
                            source_id=source_id,
                            reference_values=reference_values,
                            math_macros=math_macros,
                        )
                        if not visible:
                            raise TableAstError("table-cell list item has no visible text")
                        item_html.append(f"<li>{rendered}</li>")
                        item_visible.append(visible)
                        item_fragments.extend(fragments)
                    tag = "ol" if environment.group(2) == "enumerate" else "ul"
                    html_parts.append(
                        f"<{tag}{attributes}>" + "".join(item_html) + f"</{tag}>"
                    )
                    visible_parts.extend(item_visible)
                    verifier_fragments.extend(item_fragments)
                    cursor = environment_end
                    segment_start = cursor
                    continue
                paragraph = _PAR_COMMAND.match(source, cursor, end)
                if paragraph is not None:
                    append_inline(segment_start, cursor)
                    html_parts.append("<br><br>")
                    verifier_fragments.append(("\n", False))
                    cursor = paragraph.end()
                    segment_start = cursor
                    continue
            if char == "{" and _is_unescaped(source, cursor):
                brace_depth += 1
            elif char == "}" and _is_unescaped(source, cursor):
                brace_depth -= 1
                if brace_depth < 0:
                    raise TableAstError("unbalanced table cell braces")
            elif char == "[" and _is_unescaped(source, cursor):
                bracket_depth += 1
            elif char == "]" and _is_unescaped(source, cursor):
                bracket_depth -= 1
                if bracket_depth < 0:
                    raise TableAstError("unbalanced table cell brackets")
        cursor += 1
    if brace_depth or bracket_depth or dollar_math or parenthesized_math:
        raise TableAstError("unterminated table cell group or inline math")
    append_inline(segment_start, end)
    return (
        "\n".join(html_parts).strip(),
        " ".join(visible_parts),
        tuple(verifier_fragments),
    )


def _parse_cell(
    source: str,
    start: int,
    end: int,
    *,
    source_id: str,
    reference_values: Mapping[str, str],
    math_macros: Mapping[str, MathMacroDefinition],
) -> TableCellAst:
    body_start, body_end, colspan, rowspan = _unwrap_span_commands(source, start, end)
    if body_start == body_end:
        return TableCellAst("", "", (), colspan, rowspan, (start, end))
    cell_html, visible, verifier_fragments = _render_cell_content(
        source,
        body_start,
        body_end,
        source_id=source_id,
        reference_values=reference_values,
        math_macros=math_macros,
    )
    return TableCellAst(
        cell_html,
        visible,
        verifier_fragments,
        colspan,
        rowspan,
        (start, end),
    )


def _validate_grid(rows: list[list[TableCellAst]]) -> int:
    coverages: list[set[int]] = []
    active_until: dict[int, int] = {}
    max_column = 0
    for row_index, cells in enumerate(rows):
        occupied = {
            column for column, until in active_until.items() if until >= row_index
        }
        column = 0
        for cell in cells:
            if (
                column in occupied
                and not cell.visible_text
                and cell.colspan == 1
                and cell.rowspan == 1
            ):
                column += 1
                continue
            while column in occupied:
                column += 1
            cell_columns = set(range(column, column + cell.colspan))
            if occupied.intersection(cell_columns):
                raise TableAstError("table cell spans overlap")
            occupied.update(cell_columns)
            if cell.rowspan > 1:
                for target in cell_columns:
                    active_until[target] = row_index + cell.rowspan - 1
            column += cell.colspan
        if not occupied:
            raise TableAstError("table row has no cells")
        max_column = max(max_column, max(occupied) + 1)
        coverages.append(occupied)
    expected = set(range(max_column))
    if any(coverage != expected for coverage in coverages):
        raise TableAstError("table rows do not form one complete rectangular grid")
    return max_column


def parse_strict_table(
    source: str,
    *,
    start: int,
    end: int,
    source_id: str,
    reference_values: Mapping[str, str] | None = None,
    math_macros: Mapping[str, MathMacroDefinition] | None = None,
) -> StrictTableAst:
    """Parse one exact canonical-source tabular span or raise fail-closed."""

    if not (0 <= start < end <= len(source)):
        raise TableAstError("table source span is invalid")
    environment, body_start, body_end = _find_environment_body(source, start, end)
    row_spans = _split_top_level(source, body_start, body_end, delimiter=r"\\")
    rows: list[list[TableCellAst]] = []
    row_source_spans: list[tuple[int, int]] = []
    header_break: int | None = None
    saw_toprule = False
    for raw_start, raw_end in row_spans:
        row_start, row_end = _trim(source, raw_start, raw_end)
        row_start, rules = _consume_rule_prefix(source, row_start, row_end)
        saw_toprule = saw_toprule or "toprule" in rules
        if "midrule" in rules:
            # The first midrule following a top rule proves the header/body
            # boundary.  Later midrules are ordinary body separators and do
            # not make that already-proven boundary ambiguous.
            if header_break is None and saw_toprule and rows:
                header_break = len(rows)
        row_start, row_end = _trim(source, row_start, row_end)
        if row_start == row_end:
            continue
        cell_spans = _split_top_level(source, row_start, row_end, delimiter="&")
        cells = [
            _parse_cell(
                source,
                cell_start,
                cell_end,
                source_id=source_id,
                reference_values=reference_values or {},
                math_macros=math_macros or {},
            )
            for cell_start, cell_end in cell_spans
        ]
        rows.append(cells)
        row_source_spans.append((row_start, row_end))
    if not rows or not any(cell.visible_text for row in rows for cell in row):
        raise TableAstError("table has no visible source-derived cell content")
    if header_break == len(rows):
        raise TableAstError("table header boundary has no body rows")
    column_count = _validate_grid(rows)
    row_asts = tuple(
        TableRowAst(
            cells=tuple(cells),
            source_char_span=row_source_spans[index],
            header=header_break is not None and index < header_break,
        )
        for index, cells in enumerate(rows)
    )
    lines = ["<table>"]
    sections: list[tuple[str, tuple[TableRowAst, ...]]] = []
    if header_break is not None:
        sections.append(("thead", row_asts[:header_break]))
        sections.append(("tbody", row_asts[header_break:]))
    else:
        sections.append(("tbody", row_asts))
    for section_name, section_rows in sections:
        if not section_rows:
            continue
        lines.append(f"  <{section_name}>")
        for row in section_rows:
            lines.append("    <tr>")
            tag = "th" if section_name == "thead" else "td"
            for cell in row.cells:
                attributes: list[str] = []
                if cell.colspan > 1:
                    attributes.append(f'colspan="{cell.colspan}"')
                if cell.rowspan > 1:
                    attributes.append(f'rowspan="{cell.rowspan}"')
                suffix = "" if not attributes else " " + " ".join(attributes)
                lines.append(f"      <{tag}{suffix}>{cell.html}</{tag}>")
            lines.append("    </tr>")
        lines.append(f"  </{section_name}>")
    lines.append("</table>")
    visible_cells = [
        cell for row in row_asts for cell in row.cells if cell.visible_text
    ]
    anchor = visible_cells[0].source_char_span
    return StrictTableAst(
        environment_name=environment,
        rows=row_asts,
        html="\n".join(lines),
        visible_text=" ".join(cell.visible_text for cell in visible_cells),
        verifier_fragments=tuple(
            fragment
            for cell in visible_cells
            for fragment in cell.verifier_fragments
        ),
        anchor_char_span=anchor,
        source_char_span=(start, end),
        column_count=column_count,
    )


__all__ = [
    "TABLE_AST_VERSION",
    "StrictTableAst",
    "TableAstError",
    "TableCellAst",
    "TableRowAst",
    "is_serializable_table_metadata",
    "load_table_verifier_fragments",
    "parse_strict_table",
]
