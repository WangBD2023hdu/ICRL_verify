"""Source-only sanitization for the independent source-first v3 pipeline.

The sanitizer removes bibliography-producing constructs, citation and
cross-reference commands, and (optionally) figure-like environments from one
canonical LaTeX source.  It
does not inspect PDF text and it does not import any v1/v2 implementation.

Edits are compact rather than length preserving.  Every retained output
piece therefore receives an exact canonical-output to pre-sanitize source-map
segment.  Macro definitions, comments, verbatim-like regions, and complete
``.sty``/``.cls`` inputs are protected and are never rewritten.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import re
from typing import Any

from pylatexenc.latexwalker import (
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
    LatexWalkerParseError,
    get_default_latex_context_db,
)
from pylatexenc.macrospec import MacroSpec, MacroStandardArgsParser

from .document_ast import (
    CanonicalSourceMap,
    DocumentAstError,
    DocumentSourceSpan,
    FlattenedSourceSegment,
    locate_document_body,
)


SOURCE_SANITIZER_VERSION = "source_first_v3_source_sanitizer_v1"


class SourceSanitizerError(ValueError):
    """The sanitizer input or policy is invalid."""


@dataclass(frozen=True, slots=True)
class SanitizationEdit:
    """One AST-recognized source edit and its exact applied removal span."""

    edit_id: str
    kind: str
    construct_name: str
    source_span: DocumentSourceSpan
    removal_span: DocumentSourceSpan
    raw_source: str
    removed_source_sha256: str
    whitespace_adjustment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "kind": self.kind,
            "construct_name": self.construct_name,
            "source_char_span": list(self.source_span.char_span),
            "source_byte_span": list(self.source_span.byte_span),
            "source_line_span": list(self.source_span.line_span),
            "removal_char_span": list(self.removal_span.char_span),
            "removal_byte_span": list(self.removal_span.byte_span),
            "removal_line_span": list(self.removal_span.line_span),
            "removed_source_sha256": self.removed_source_sha256,
            "whitespace_adjustment": self.whitespace_adjustment,
        }


@dataclass(frozen=True, slots=True)
class SanitizationAudit:
    """Deterministic summary of one source sanitization pass."""

    version: str
    status: str
    source_id: str
    original_sha256: str
    sanitized_sha256: str
    original_chars: int
    sanitized_chars: int
    original_bytes: int
    sanitized_bytes: int
    drop_figures: bool
    parse_mode: str
    edit_counts: tuple[tuple[str, int], ...]
    applied_removal_spans: tuple[tuple[int, int], ...]
    message: str | None = None

    @property
    def total_edits(self) -> int:
        return sum(count for _, count in self.edit_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "source_id": self.source_id,
            "original_sha256": self.original_sha256,
            "sanitized_sha256": self.sanitized_sha256,
            "original_chars": self.original_chars,
            "sanitized_chars": self.sanitized_chars,
            "original_bytes": self.original_bytes,
            "sanitized_bytes": self.sanitized_bytes,
            "drop_figures": self.drop_figures,
            "parse_mode": self.parse_mode,
            "total_edits": self.total_edits,
            "edit_counts": dict(self.edit_counts),
            "applied_removal_spans": [list(span) for span in self.applied_removal_spans],
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class SanitizedSource:
    """Sanitized canonical source with exact pre-sanitize provenance."""

    source_id: str
    original_sha256: str
    sanitized_source: str
    source_map: CanonicalSourceMap
    edits: tuple[SanitizationEdit, ...]
    audit: SanitizationAudit

    @property
    def source(self) -> str:
        """Alias used by downstream canonical-source consumers."""

        return self.sanitized_source

    @property
    def changed(self) -> bool:
        return self.audit.original_sha256 != self.audit.sanitized_sha256

    def to_dict(self, *, include_source: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_id": self.source_id,
            "original_sha256": self.original_sha256,
            "sanitized_sha256": self.audit.sanitized_sha256,
            "changed": self.changed,
            "audit": self.audit.to_dict(),
            "edits": [edit.to_dict() for edit in self.edits],
            "source_map": [
                {
                    "canonical_char_span": [
                        segment.canonical_char_start,
                        segment.canonical_char_end,
                    ],
                    "source_path": segment.source_path,
                    "source_char_start": segment.source_char_start,
                    "source_byte_start": segment.source_byte_start,
                    "source_line_start": segment.source_line_start,
                    "source_column_start": segment.source_column_start,
                }
                for segment in self.source_map.segments
            ],
        }
        if include_source:
            payload["sanitized_source"] = self.sanitized_source
        return payload


_CITATION_COMMANDS = frozenset(
    {
        "Cite",
        "Citealp",
        "Citealt",
        "Citeauthor",
        "Citep",
        "Citet",
        "Textcite",
        "autocite",
        "autocite*",
        "cite",
        "citealp",
        "citealt",
        "citeauthor",
        "citeauthor*",
        "citep",
        "citep*",
        "citet",
        "citet*",
        "citeyear",
        "citeyearpar",
        "footcite",
        "footcitetext",
        "fullcite",
        "nocite",
        "parencite",
        "parencite*",
        "smartcite",
        "supercite",
        "textcite",
        "textcite*",
    }
)

# pylatexenc exposes the star as an argument, not part of ``macroname``.  The
# starred spellings remain in the allowlist for audit clarity but are reduced
# to their base names when constructing parser specs.
_CITATION_MACRO_NAMES = frozenset(name.rstrip("*") for name in _CITATION_COMMANDS)
_CROSS_REFERENCE_COMMANDS = frozenset(
    {
        "Cpageref",
        "Cref",
        "Crefrange",
        "Vref",
        "autoref",
        "cpageref",
        "cref",
        "crefrange",
        "eqref",
        "nameref",
        "pageref",
        "ref",
        "vref",
    }
)
_BIBLIOGRAPHY_COMMANDS = frozenset(
    {"bibliography", "bibliographystyle", "printbibliography"}
)
_BIBLIOGRAPHY_ENVIRONMENTS = frozenset({"thebibliography"})
_FIGURE_ENVIRONMENTS = frozenset(
    {
        "figure",
        "figure*",
        "wrapfigure",
        "wrapfig",
    }
)
_VERBATIM_ENVIRONMENTS = frozenset(
    {
        "BVerbatim",
        "LVerbatim",
        "SaveVerbatim",
        "Verbatim",
        "comment",
        "lstlisting",
        "minted",
        "verbatim",
        "verbatim*",
    }
)
_VERBATIM_MACROS = frozenset({"lstinline", "mintinline", "verb", "verb*"})

# Complete command nodes are protected.  They may contain citation-looking
# text in their replacement body, but that text is a definition, not a
# document-level invocation.
_DEFINITION_COMMANDS = frozenset(
    {
        "AtBeginDocument",
        "AtEndDocument",
        "DeclareDocumentCommand",
        "DeclareMathOperator",
        "DeclareMathOperator*",
        "DeclarePairedDelimiter",
        "DeclareRobustCommand",
        "NewDocumentCommand",
        "NewDocumentEnvironment",
        "ProvideDocumentCommand",
        "RenewDocumentCommand",
        "RenewDocumentEnvironment",
        "def",
        "edef",
        "gdef",
        "globaldefs",
        "let",
        "long",
        "newcommand",
        "newenvironment",
        "providecommand",
        "renewcommand",
        "renewenvironment",
        "xdef",
    }
)


def _make_latex_context():
    context = get_default_latex_context_db()
    citation_specs = [
        MacroSpec(name, MacroStandardArgsParser("*[[{"))
        for name in sorted(_CITATION_MACRO_NAMES)
    ]
    reference_specs = [
        MacroSpec(
            name,
            MacroStandardArgsParser("*{{" if name.lower().endswith("range") else "*{"),
        )
        for name in sorted(_CROSS_REFERENCE_COMMANDS)
    ]
    other_specs = [
        MacroSpec("bibliography", MacroStandardArgsParser("{")),
        MacroSpec("bibliographystyle", MacroStandardArgsParser("{")),
        MacroSpec("printbibliography", MacroStandardArgsParser("[")),
    ]
    context.add_context_category(
        "source-first-v3-sanitizer",
        macros=citation_specs + reference_specs + other_specs,
        prepend=True,
    )
    return context


class _PositionIndex:
    def __init__(self, source: str) -> None:
        self.source = source
        self.byte_prefix = [0]
        self.line_starts = [0]
        for index, char in enumerate(source):
            self.byte_prefix.append(self.byte_prefix[-1] + len(char.encode("utf-8")))
            if char == "\n":
                self.line_starts.append(index + 1)

    def _line_column(self, position: int) -> tuple[int, int]:
        # The number of starts <= position is the one-based line number.
        low, high = 0, len(self.line_starts)
        while low < high:
            middle = (low + high) // 2
            if self.line_starts[middle] <= position:
                low = middle + 1
            else:
                high = middle
        line_index = max(0, low - 1)
        return line_index + 1, position - self.line_starts[line_index]

    def span(self, start: int, end: int) -> DocumentSourceSpan:
        if start < 0 or end < start or end > len(self.source):
            raise SourceSanitizerError(f"invalid source span {start}:{end}")
        line_start, column_start = self._line_column(start)
        line_end, column_end = self._line_column(end)
        return DocumentSourceSpan(
            char_start=start,
            char_end=end,
            byte_start=self.byte_prefix[start],
            byte_end=self.byte_prefix[end],
            line_start=line_start,
            column_start=column_start,
            line_end=line_end,
            column_end=column_end,
        )


def _node_end(node: Any) -> int:
    return int(node.pos) + int(node.len)


def _stable_edit_id(
    source_id: str,
    kind: str,
    name: str,
    start: int,
    end: int,
    raw_source: str,
) -> str:
    payload = "\x1f".join(
        (
            SOURCE_SANITIZER_VERSION,
            source_id,
            kind,
            name,
            str(start),
            str(end),
            hashlib.sha256(raw_source.encode("utf-8")).hexdigest(),
        )
    )
    return "sfv3_sanitize_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _is_definition_command(name: str) -> bool:
    base = name.rstrip("*")
    return name in _DEFINITION_COMMANDS or base in _DEFINITION_COMMANDS


def _is_style_or_class_source(source_id: str) -> bool:
    # ``source_id`` may carry a flattening suffix such as ``foo.sty#12:20``.
    path_part = source_id.split("#", 1)[0]
    return PurePosixPath(path_part).suffix.lower() in {".sty", ".cls"}


def _document_environment(nodes: Sequence[Any]) -> LatexEnvironmentNode | None:
    matches = [
        node
        for node in nodes
        if isinstance(node, LatexEnvironmentNode) and node.environmentname == "document"
    ]
    return matches[0] if len(matches) == 1 else None


def _iter_argument_nodes(node: Any) -> tuple[Any, ...]:
    nodeargd = getattr(node, "nodeargd", None)
    return tuple(arg for arg in (getattr(nodeargd, "argnlist", ()) or ()) if arg is not None)


def _balanced_delimited_end(
    source: str,
    opening: int,
    open_char: str = "{",
    close_char: str = "}",
) -> int | None:
    if opening >= len(source) or source[opening] != open_char:
        return None
    depth = 0
    cursor = opening
    while cursor < len(source):
        char = source[cursor]
        if char == "%":
            backslashes = 0
            probe = cursor - 1
            while probe >= 0 and source[probe] == "\\":
                backslashes += 1
                probe -= 1
            if backslashes % 2 == 0:
                newline = source.find("\n", cursor)
                cursor = len(source) if newline < 0 else newline + 1
                continue
        if char == "\\":
            # Skip the escaped character or complete alphabetic control word;
            # braces within command names cannot delimit the definition body.
            cursor += 1
            if cursor < len(source) and (source[cursor].isalpha() or source[cursor] == "@"):
                while cursor < len(source) and (
                    source[cursor].isalpha() or source[cursor] == "@"
                ):
                    cursor += 1
            else:
                cursor += 1
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return None


def _skip_space_and_comments(source: str, cursor: int) -> int:
    while cursor < len(source):
        if source[cursor].isspace():
            cursor += 1
            continue
        if source[cursor] == "%":
            newline = source.find("\n", cursor)
            cursor = len(source) if newline < 0 else newline + 1
            continue
        break
    return cursor


def _consume_control_sequence(source: str, cursor: int) -> int:
    if cursor >= len(source) or source[cursor] != "\\":
        return cursor
    cursor += 1
    if cursor < len(source) and (source[cursor].isalpha() or source[cursor] == "@"):
        while cursor < len(source) and (source[cursor].isalpha() or source[cursor] == "@"):
            cursor += 1
        return cursor
    return min(len(source), cursor + 1)


def _definition_extent(source: str, node: LatexMacroNode) -> int:
    """Protect definitions whose arguments walker may expose as siblings."""

    name = node.macroname.rstrip("*")
    cursor = _node_end(node)
    if name in {"def", "edef", "gdef", "xdef", "long"}:
        # Primitive definitions have parameter text between the control word
        # and their first balanced replacement group.
        while cursor < len(source):
            if source[cursor] == "%":
                newline = source.find("\n", cursor)
                cursor = len(source) if newline < 0 else newline + 1
                continue
            if source[cursor] == "{":
                return _balanced_delimited_end(source, cursor) or len(source)
            cursor += 1
        return _node_end(node)
    if name == "let":
        newline = source.find("\n", cursor)
        return len(source) if newline < 0 else newline

    # LaTeX/xparse declarations consist of an optional unbraced target
    # control sequence followed by bracket/brace arguments.  Consume every
    # adjacent argument so replacement bodies remain protected even when the
    # default pylatexenc context did not attach them to the macro node.
    last_end = cursor
    consumed_unbraced_target = False
    while cursor < len(source):
        cursor = _skip_space_and_comments(source, cursor)
        if cursor < len(source) and source[cursor] == "*":
            cursor += 1
            last_end = cursor
            continue
        if cursor < len(source) and source[cursor] == "\\" and not consumed_unbraced_target:
            cursor = _consume_control_sequence(source, cursor)
            consumed_unbraced_target = True
            last_end = cursor
            continue
        if cursor < len(source) and source[cursor] in "[{":
            open_char = source[cursor]
            close_char = "]" if open_char == "[" else "}"
            group_end = _balanced_delimited_end(source, cursor, open_char, close_char)
            if group_end is None:
                return len(source)
            cursor = group_end
            last_end = group_end
            continue
        break
    return max(_node_end(node), last_end)


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: str
    name: str
    start: int
    end: int


def _collect_candidates(
    nodes: Sequence[Any],
    *,
    source: str,
    drop_figures: bool,
    output: list[_Candidate],
) -> None:
    protected_until = -1
    for node in nodes:
        if int(node.pos) < protected_until:
            continue
        if isinstance(node, LatexCommentNode):
            continue
        if isinstance(node, LatexMacroNode):
            name = node.macroname
            if _is_definition_command(name):
                protected_until = max(
                    protected_until,
                    _definition_extent(source, node),
                )
                continue
            if name in _VERBATIM_MACROS:
                continue
            if name in _CITATION_MACRO_NAMES:
                output.append(_Candidate("citation", name, node.pos, _node_end(node)))
                continue
            if name in _CROSS_REFERENCE_COMMANDS:
                output.append(
                    _Candidate("cross_reference", name, node.pos, _node_end(node))
                )
                continue
            if name in _BIBLIOGRAPHY_COMMANDS:
                output.append(
                    _Candidate("bibliography_command", name, node.pos, _node_end(node))
                )
                continue
            _collect_candidates(
                _iter_argument_nodes(node),
                source=source,
                drop_figures=drop_figures,
                output=output,
            )
            continue
        if isinstance(node, LatexEnvironmentNode):
            name = node.environmentname
            if name in _VERBATIM_ENVIRONMENTS:
                continue
            if name in _BIBLIOGRAPHY_ENVIRONMENTS:
                output.append(
                    _Candidate("bibliography_environment", name, node.pos, _node_end(node))
                )
                continue
            if name in _FIGURE_ENVIRONMENTS and drop_figures:
                output.append(_Candidate("figure_environment", name, node.pos, _node_end(node)))
                continue
            _collect_candidates(
                tuple(node.nodelist or ()) + _iter_argument_nodes(node),
                source=source,
                drop_figures=drop_figures,
                output=output,
            )
            continue
        if isinstance(node, (LatexGroupNode, LatexMathNode)):
            _collect_candidates(
                tuple(getattr(node, "nodelist", ()) or ()),
                source=source,
                drop_figures=drop_figures,
                output=output,
            )


_PUNCTUATION_AFTER_CITATION = frozenset(",.;:!?)]}")


def _adjust_citation_removal(source: str, start: int, end: int) -> tuple[int, int, str | None]:
    """Avoid a source-created interword space immediately before punctuation."""

    right = end
    while right < len(source) and source[right] in " \t":
        right += 1
    if right >= len(source) or source[right] not in _PUNCTUATION_AFTER_CITATION:
        return start, end, None
    left = start
    while left > 0 and source[left - 1] in " \t~":
        left -= 1
    adjustment_parts: list[str] = []
    if left != start:
        adjustment_parts.append("preceding_horizontal_space")
    if right != end:
        adjustment_parts.append("following_horizontal_space")
    return left, right, "+".join(adjustment_parts) or None


def _expand_standalone_lines(source: str, start: int, end: int) -> tuple[int, int, str | None]:
    """Remove a complete standalone removable-construct line when safe."""

    line_start = source.rfind("\n", 0, start) + 1
    newline = source.find("\n", end)
    line_end = len(source) if newline < 0 else newline + 1
    if source[line_start:start].strip() or source[end:line_end].strip():
        return start, end, None
    return line_start, line_end, "standalone_source_lines"


def _merge_spans(spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _apply_removals(
    source: str,
    source_id: str,
    positions: _PositionIndex,
    removals: Sequence[tuple[int, int]],
) -> tuple[str, CanonicalSourceMap]:
    pieces: list[str] = []
    segments: list[FlattenedSourceSegment] = []
    input_cursor = 0
    output_cursor = 0
    for start, end in tuple(removals) + ((len(source), len(source)),):
        if start > input_cursor:
            piece = source[input_cursor:start]
            pieces.append(piece)
            original_span = positions.span(input_cursor, start)
            segments.append(
                FlattenedSourceSegment(
                    canonical_char_start=output_cursor,
                    canonical_char_end=output_cursor + len(piece),
                    source_path=source_id,
                    source_char_start=input_cursor,
                    source_byte_start=original_span.byte_start,
                    source_line_start=original_span.line_start,
                    source_column_start=original_span.column_start,
                )
            )
            output_cursor += len(piece)
        input_cursor = max(input_cursor, end)
    sanitized = "".join(pieces)
    return sanitized, CanonicalSourceMap(tuple(segments))


def _unchanged_result(
    source: str,
    source_id: str,
    *,
    status: str,
    drop_figures: bool,
    parse_mode: str = "full_document",
    message: str | None = None,
) -> SanitizedSource:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    audit = SanitizationAudit(
        version=SOURCE_SANITIZER_VERSION,
        status=status,
        source_id=source_id,
        original_sha256=digest,
        sanitized_sha256=digest,
        original_chars=len(source),
        sanitized_chars=len(source),
        original_bytes=len(source.encode("utf-8")),
        sanitized_bytes=len(source.encode("utf-8")),
        drop_figures=drop_figures,
        parse_mode=parse_mode,
        edit_counts=(),
        applied_removal_spans=(),
        message=message,
    )
    return SanitizedSource(
        source_id=source_id,
        original_sha256=digest,
        sanitized_source=source,
        source_map=CanonicalSourceMap.identity(source, source_id),
        edits=(),
        audit=audit,
    )


def sanitize_latex_source(
    source: str,
    *,
    source_id: str = "main.tex",
    drop_figures: bool = False,
) -> SanitizedSource:
    """Remove selected document-level LaTeX constructs with exact provenance.

    ``.sty`` and ``.cls`` sources are unconditionally returned unchanged.
    When a unique ``document`` environment exists, only its body is scanned;
    preamble definitions remain byte-for-byte untouched.  A body-only
    flattened fragment is also supported, with recognized macro-definition
    nodes protected from recursive scanning.
    """

    if not isinstance(source, str):
        raise SourceSanitizerError("source must be a string")
    if not isinstance(source_id, str) or not source_id:
        raise SourceSanitizerError("source_id must be a non-empty string")
    if not isinstance(drop_figures, bool):
        raise SourceSanitizerError("drop_figures must be a boolean")
    try:
        original_bytes = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SourceSanitizerError("source is not valid UTF-8 text") from exc
    if _is_style_or_class_source(source_id):
        return _unchanged_result(
            source,
            source_id,
            status="skipped_non_document_source",
            drop_figures=drop_figures,
            message="style/class source is protected from document sanitization",
        )

    parse_mode = "full_document"
    parse_message: str | None = None
    try:
        root_nodes, _, _ = LatexWalker(
            source,
            latex_context=_make_latex_context(),
            tolerant_parsing=False,
        ).get_latex_nodes(pos=0)
    except (LatexWalkerParseError, ValueError) as exc:
        parse_message = f"full canonical parse failed; body-only fallback used: {exc}"
        try:
            boundary = locate_document_body(source)
            body_nodes, parsed_start, parsed_length = LatexWalker(
                source,
                latex_context=_make_latex_context(),
                tolerant_parsing=False,
            ).get_latex_nodes(
                pos=boundary.body_start,
                stop_upon_end_environment="document",
            )
            body_nodes = tuple(body_nodes)
            if parsed_start != boundary.body_start:
                raise SourceSanitizerError(
                    "body parser changed the canonical source offset"
                )
            if parsed_start + parsed_length != boundary.end_command_end:
                raise SourceSanitizerError(
                    "body parser did not stop at recovered document end"
                )
            if any(
                int(node.pos) < boundary.body_start or _node_end(node) > boundary.body_end
                for node in body_nodes
            ):
                raise SourceSanitizerError(
                    "body parser emitted a node outside recovered bounds"
                )
        except (
            DocumentAstError,
            LatexWalkerParseError,
            SourceSanitizerError,
            ValueError,
        ) as fallback_exc:
            return _unchanged_result(
                source,
                source_id,
                status="rejected_parse_error",
                drop_figures=drop_figures,
                parse_mode="rejected",
                message=(
                    f"pylatexenc full parse failed ({exc}); "
                    f"body-only fallback failed ({fallback_exc}); source left unchanged"
                ),
            )
        parse_mode = "body_only_fallback"
        scan_nodes = body_nodes
    else:
        root_nodes = tuple(root_nodes)
        document_matches = [
            node
            for node in root_nodes
            if isinstance(node, LatexEnvironmentNode)
            and node.environmentname == "document"
        ]
        if len(document_matches) > 1:
            return _unchanged_result(
                source,
                source_id,
                status="rejected_multiple_document_environments",
                drop_figures=drop_figures,
                parse_mode="rejected",
                message="multiple document environments; source left unchanged",
            )
        document = _document_environment(root_nodes)
        if document is None:
            parse_mode = "fragment"
            scan_nodes = root_nodes
        else:
            scan_nodes = tuple(document.nodelist or ())
    candidates: list[_Candidate] = []
    _collect_candidates(
        scan_nodes,
        source=source,
        drop_figures=drop_figures,
        output=candidates,
    )
    if not candidates:
        return _unchanged_result(
            source,
            source_id,
            status="unchanged",
            drop_figures=drop_figures,
            parse_mode=parse_mode,
            message=parse_message,
        )

    positions = _PositionIndex(source)
    edits: list[SanitizationEdit] = []
    for candidate in sorted(candidates, key=lambda row: (row.start, row.end, row.kind)):
        removal_start, removal_end = candidate.start, candidate.end
        adjustment: str | None = None
        if candidate.kind in {"citation", "cross_reference"}:
            # A reference can be the only construct on a source line inside a
            # braced command argument (for example a subsection title).  Drop
            # that complete line so its neighbouring newlines do not become a
            # source-created blank paragraph (``\\par``) in the argument.
            removal_start, removal_end, adjustment = _expand_standalone_lines(
                source, removal_start, removal_end
            )
            if adjustment is None:
                removal_start, removal_end, adjustment = _adjust_citation_removal(
                    source, candidate.start, candidate.end
                )
        else:
            removal_start, removal_end, adjustment = _expand_standalone_lines(
                source, removal_start, removal_end
            )
        raw = source[candidate.start : candidate.end]
        removed = source[removal_start:removal_end]
        edits.append(
            SanitizationEdit(
                edit_id=_stable_edit_id(
                    source_id,
                    candidate.kind,
                    candidate.name,
                    candidate.start,
                    candidate.end,
                    raw,
                ),
                kind=candidate.kind,
                construct_name=candidate.name,
                source_span=positions.span(candidate.start, candidate.end),
                removal_span=positions.span(removal_start, removal_end),
                raw_source=raw,
                removed_source_sha256=hashlib.sha256(removed.encode("utf-8")).hexdigest(),
                whitespace_adjustment=adjustment,
            )
        )
    removals = _merge_spans([edit.removal_span.char_span for edit in edits])
    sanitized, source_map = _apply_removals(source, source_id, positions, removals)
    sanitized_bytes = sanitized.encode("utf-8")
    counts = Counter(edit.kind for edit in edits)
    audit = SanitizationAudit(
        version=SOURCE_SANITIZER_VERSION,
        status="sanitized",
        source_id=source_id,
        original_sha256=hashlib.sha256(original_bytes).hexdigest(),
        sanitized_sha256=hashlib.sha256(sanitized_bytes).hexdigest(),
        original_chars=len(source),
        sanitized_chars=len(sanitized),
        original_bytes=len(original_bytes),
        sanitized_bytes=len(sanitized_bytes),
        drop_figures=drop_figures,
        parse_mode=parse_mode,
        edit_counts=tuple(sorted(counts.items())),
        applied_removal_spans=removals,
        message=parse_message,
    )
    return SanitizedSource(
        source_id=source_id,
        original_sha256=audit.original_sha256,
        sanitized_source=sanitized,
        source_map=source_map,
        edits=tuple(edits),
        audit=audit,
    )


__all__ = [
    "SOURCE_SANITIZER_VERSION",
    "SanitizationAudit",
    "SanitizationEdit",
    "SanitizedSource",
    "SourceSanitizerError",
    "sanitize_latex_source",
]
