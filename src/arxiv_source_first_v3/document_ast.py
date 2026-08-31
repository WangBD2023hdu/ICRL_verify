"""Standalone, compiler-facing document AST for source-first v3.

This module owns the *block* layer of the v3 source representation.  It takes
one canonical, already-flattened LaTeX string and uses :mod:`pylatexenc` to
identify structural blocks.  Inline content is delegated to
``arxiv_source_first_v3.ast_ir``; PDF text is never an input.

The parser is intentionally conservative.  Unknown top-level environments,
malformed structures, and inline fragments that ``ast_ir`` cannot represent
produce explicit rejection records.  Comments remain in provenance spans
when they occur inside a visible block, but never become visible atoms or
ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import re
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pylatexenc.latexwalker import (
    LatexCharsNode,
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

from .ast_ir import (
    MathMacroDefinition,
    SourceDocumentIR,
    SourceIrError,
    build_display_math_ir,
    build_numbered_align_ir,
    parse_source_ir,
    resolve_display_math_reference_tag,
)
from .semantic_declarations import SemanticEnvironmentDefinition
from .source_environment_definitions import SourceListEnvironmentDefinition
from .table_ast import TableAstError, parse_strict_table


DOCUMENT_AST_VERSION = "source_first_v3_document_ast_v3"


class DocumentAstError(ValueError):
    """The canonical source or source map is invalid."""


@dataclass(frozen=True, slots=True)
class DocumentBodyBoundary:
    """Absolute canonical offsets of one top-level document environment."""

    begin_command_start: int
    begin_command_end: int
    body_start: int
    body_end: int
    end_command_start: int
    end_command_end: int

    def __post_init__(self) -> None:
        values = (
            self.begin_command_start,
            self.begin_command_end,
            self.body_start,
            self.body_end,
            self.end_command_start,
            self.end_command_end,
        )
        if min(values) < 0 or tuple(sorted(values)) != values:
            raise DocumentAstError("document boundary offsets are inconsistent")


@dataclass(frozen=True, slots=True)
class DocumentSourceSpan:
    """Half-open canonical-source span with exact UTF-8 and line positions.

    Lines are one-based and columns are zero-based.  The end position is the
    insertion point immediately after the span, so it may be on the next line
    when the final source character is a newline.
    """

    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    line_start: int
    column_start: int
    line_end: int
    column_end: int

    def __post_init__(self) -> None:
        values = (
            self.char_start,
            self.char_end,
            self.byte_start,
            self.byte_end,
            self.column_start,
            self.column_end,
        )
        if min(values) < 0 or self.line_start < 1 or self.line_end < 1:
            raise DocumentAstError("source span coordinates must be non-negative")
        if self.char_end < self.char_start or self.byte_end < self.byte_start:
            raise DocumentAstError("source span end precedes its start")

    @property
    def char_span(self) -> tuple[int, int]:
        return self.char_start, self.char_end

    @property
    def byte_span(self) -> tuple[int, int]:
        return self.byte_start, self.byte_end

    @property
    def line_span(self) -> tuple[int, int]:
        """Inclusive one-based line range touched by this span."""

        if self.char_start == self.char_end:
            return self.line_start, self.line_start
        if self.column_end == 0 and self.line_end > self.line_start:
            return self.line_start, self.line_end - 1
        return self.line_start, self.line_end


@dataclass(frozen=True, slots=True)
class FlattenedSourceSegment:
    """One length-preserving piece of a canonical flattened source.

    ``canonical_char_*`` addresses the flattened input.  ``source_*_start``
    addresses the first corresponding character in the original file.  The
    canonical slice and original slice are required to be character-for-
    character identical; generated separators can be represented by gaps
    between segments rather than by pretending they came from a source file.
    """

    canonical_char_start: int
    canonical_char_end: int
    source_path: str
    source_char_start: int = 0
    source_byte_start: int = 0
    source_line_start: int = 1
    source_column_start: int = 0

    def __post_init__(self) -> None:
        if self.canonical_char_start < 0 or self.canonical_char_end <= self.canonical_char_start:
            raise DocumentAstError("flattened source segment must be non-empty")
        if self.source_char_start < 0 or self.source_byte_start < 0:
            raise DocumentAstError("original source offsets must be non-negative")
        if self.source_line_start < 1 or self.source_column_start < 0:
            raise DocumentAstError("invalid original source line/column")
        if not self.source_path:
            raise DocumentAstError("source_path must be non-empty")


@dataclass(frozen=True, slots=True)
class OriginalSourceSpan:
    """The original-file provenance of part of a canonical block."""

    source_path: str
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    line_start: int
    column_start: int
    line_end: int
    column_end: int


@dataclass(frozen=True, slots=True)
class CanonicalSourceMap:
    """Length-preserving canonical-to-original source mapping."""

    segments: tuple[FlattenedSourceSegment, ...]

    def __post_init__(self) -> None:
        previous_end = 0
        for index, segment in enumerate(self.segments):
            if index and segment.canonical_char_start < previous_end:
                raise DocumentAstError("flattened source-map segments overlap")
            previous_end = segment.canonical_char_end

    @classmethod
    def identity(cls, source: str, source_id: str) -> "CanonicalSourceMap":
        if not source:
            return cls(())
        return cls(
            (
                FlattenedSourceSegment(
                    canonical_char_start=0,
                    canonical_char_end=len(source),
                    source_path=source_id,
                ),
            )
        )

    def validate_for_source(self, source: str) -> None:
        for segment in self.segments:
            if segment.canonical_char_end > len(source):
                raise DocumentAstError("source-map segment exceeds canonical source")

    def origins_for_span(
        self,
        source: str,
        char_start: int,
        char_end: int,
    ) -> tuple[OriginalSourceSpan, ...]:
        origins: list[OriginalSourceSpan] = []
        for segment in self.segments:
            start = max(char_start, segment.canonical_char_start)
            end = min(char_end, segment.canonical_char_end)
            if end <= start:
                continue
            segment_text = source[segment.canonical_char_start : segment.canonical_char_end]
            local_start = start - segment.canonical_char_start
            local_end = end - segment.canonical_char_start
            start_line, start_column = _advance_line_column(
                segment_text[:local_start],
                segment.source_line_start,
                segment.source_column_start,
            )
            end_line, end_column = _advance_line_column(
                segment_text[:local_end],
                segment.source_line_start,
                segment.source_column_start,
            )
            byte_before = len(segment_text[:local_start].encode("utf-8"))
            byte_piece = len(segment_text[local_start:local_end].encode("utf-8"))
            origins.append(
                OriginalSourceSpan(
                    source_path=segment.source_path,
                    char_start=segment.source_char_start + local_start,
                    char_end=segment.source_char_start + local_end,
                    byte_start=segment.source_byte_start + byte_before,
                    byte_end=segment.source_byte_start + byte_before + byte_piece,
                    line_start=start_line,
                    column_start=start_column,
                    line_end=end_line,
                    column_end=end_column,
                )
            )
        return tuple(origins)


@dataclass(frozen=True, slots=True)
class DocumentBlockNode:
    """One source-derived document block or structural container."""

    node_id: str
    kind: str
    span: DocumentSourceSpan
    content_span: DocumentSourceSpan | None
    parent_node_id: str | None
    child_node_ids: tuple[str, ...]
    raw_source: str
    origins: tuple[OriginalSourceSpan, ...]
    inline_ir: SourceDocumentIR | None = None
    command_name: str | None = None
    environment_name: str | None = None
    heading_level: int | None = None
    starred: bool = False
    optional_label: str | None = None
    optional_inline_ir: SourceDocumentIR | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def is_container(self) -> bool:
        return self.inline_ir is None and bool(self.child_node_ids)


@dataclass(frozen=True, slots=True)
class DocumentAstRejection:
    """An explicit fail-closed source rejection."""

    rejection_id: str
    code: str
    message: str
    span: DocumentSourceSpan
    raw_source: str
    parent_node_id: str | None = None
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class GeneratedLineJoin:
    """Nodes and original spans associated with one canonical source line."""

    generated_line: int
    canonical_span: DocumentSourceSpan
    node_ids: tuple[str, ...]
    origins: tuple[OriginalSourceSpan, ...]


@dataclass(frozen=True, slots=True)
class DocumentAst:
    """Immutable block AST plus source-map and line-join indexes."""

    version: str
    source_id: str
    source_sha256: str
    source: str
    body_span: DocumentSourceSpan
    nodes: tuple[DocumentBlockNode, ...]
    rejections: tuple[DocumentAstRejection, ...]
    source_map: CanonicalSourceMap
    line_index: tuple[GeneratedLineJoin, ...]
    parse_mode: str = "full_document"
    parse_diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise DocumentAstError("duplicate document AST node IDs")
        known = set(node_ids)
        for node in self.nodes:
            if node.parent_node_id is not None and node.parent_node_id not in known:
                raise DocumentAstError(f"unknown parent node: {node.parent_node_id}")
            if not set(node.child_node_ids).issubset(known):
                raise DocumentAstError(f"unknown child node on {node.node_id}")

    @property
    def accepted(self) -> bool:
        return not self.rejections

    @property
    def leaf_nodes(self) -> tuple[DocumentBlockNode, ...]:
        return tuple(node for node in self.nodes if not node.child_node_ids)

    @property
    def source_ordered_leaf_nodes(self) -> tuple[DocumentBlockNode, ...]:
        """Leaf blocks in canonical source order, independent of tree build order."""

        return tuple(
            sorted(
                self.leaf_nodes,
                key=lambda node: (
                    node.span.char_start,
                    node.span.char_end,
                    node.node_id,
                ),
            )
        )

    def get_node(self, node_id: str) -> DocumentBlockNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def nodes_for_line(self, generated_line: int) -> tuple[DocumentBlockNode, ...]:
        if generated_line < 1 or generated_line > len(self.line_index):
            raise IndexError(generated_line)
        ids = set(self.line_index[generated_line - 1].node_ids)
        return tuple(node for node in self.nodes if node.node_id in ids)

    def to_dict(self, *, include_source: bool = False) -> dict[str, Any]:
        """Return a deterministic JSON-compatible audit representation."""

        def span_dict(span: DocumentSourceSpan | None) -> dict[str, Any] | None:
            if span is None:
                return None
            return {
                "char_span": list(span.char_span),
                "byte_span": list(span.byte_span),
                "line_start": span.line_start,
                "column_start": span.column_start,
                "line_end": span.line_end,
                "column_end": span.column_end,
            }

        payload: dict[str, Any] = {
            "version": self.version,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "parse_mode": self.parse_mode,
            "parse_diagnostics": list(self.parse_diagnostics),
            "body_span": span_dict(self.body_span),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "span": span_dict(node.span),
                    "content_span": span_dict(node.content_span),
                    "parent_node_id": node.parent_node_id,
                    "child_node_ids": list(node.child_node_ids),
                    "command_name": node.command_name,
                    "environment_name": node.environment_name,
                    "heading_level": node.heading_level,
                    "starred": node.starred,
                    "optional_label": node.optional_label,
                    "optional_inline_ir": (
                        None
                        if node.optional_inline_ir is None
                        else {
                            "source_id": node.optional_inline_ir.source_id,
                            "atom_ids": [
                                atom.atom_id for atom in node.optional_inline_ir.atoms
                            ],
                            "opaque_atom_ids": [
                                atom.atom_id
                                for atom in node.optional_inline_ir.opaque_atoms
                            ],
                        }
                    ),
                    "metadata": dict(node.metadata),
                    "opaque_inline_atoms": (
                        []
                        if node.inline_ir is None
                        else [atom.atom_id for atom in node.inline_ir.opaque_atoms]
                    ),
                }
                for node in self.nodes
            ],
            "rejections": [
                {
                    "rejection_id": row.rejection_id,
                    "code": row.code,
                    "message": row.message,
                    "span": span_dict(row.span),
                    "parent_node_id": row.parent_node_id,
                    "node_id": row.node_id,
                }
                for row in self.rejections
            ],
            "line_index": [
                {
                    "generated_line": row.generated_line,
                    "canonical_span": span_dict(row.canonical_span),
                    "node_ids": list(row.node_ids),
                    "origins": [
                        {
                            "source_path": origin.source_path,
                            "char_span": [origin.char_start, origin.char_end],
                            "byte_span": [origin.byte_start, origin.byte_end],
                            "line_start": origin.line_start,
                            "column_start": origin.column_start,
                            "line_end": origin.line_end,
                            "column_end": origin.column_end,
                        }
                        for origin in row.origins
                    ],
                }
                for row in self.line_index
            ],
        }
        if include_source:
            payload["source"] = self.source
        return payload


_FRONTMATTER_COMMANDS = frozenset(
    {
        "title",
        "subtitle",
        "author",
        "affiliation",
        "affil",
        "institute",
        "address",
        "email",
        "date",
        "thanks",
        "keywords",
        "subject",
    }
)

_HEADING_LEVELS = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}

_LIST_ENVIRONMENTS = frozenset({"itemize", "enumerate"})
_TABLE_ENVIRONMENTS = frozenset({"table", "table*"})
_TABULAR_ENVIRONMENTS = frozenset(
    {"tabular", "tabular*", "tabularx", "longtable", "array"}
)
_STRICT_TABLE_BOX_WRAPPERS = frozenset({"adjustbox", "resizebox", "scalebox"})
_STRICT_TABLE_DECLARATIONS = frozenset(
    {
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
    }
)
_STRICT_TABLE_BOX_DIMENSION = re.compile(
    r"(?:!|[+\-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em)|"
    r"(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))?"
    r"\\(?:textwidth|linewidth|columnwidth))"
)
_STRICT_TABLE_SCALE = re.compile(r"[+\-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))")
_STRICT_TABLE_ADJUSTBOX_OPTIONS = re.compile(r"[A-Za-z0-9 .,:;=+\-\\]+")
_STRICT_TABLE_LENGTH_CONTROL = re.compile(
    r"\\(?:setlength|addtolength)\s*"
    r"\{\s*\\(?:tabcolsep|extrarowheight|arrayrulewidth)\s*\}\s*"
    r"\{\s*[+\-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em)\s*\}\s*"
)
_STRICT_TABLE_SPACING_CONTROL = re.compile(
    r"\\(?:vspace|hspace)\*?\s*"
    r"\{\s*[+\-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em)\s*\}\s*"
)
_STRICT_TABLE_RENEW_CONTROL = re.compile(
    r"\\renewcommand\*?\s*"
    r"\{\s*\\(?:arraystretch|tabcolsep)\s*\}\s*"
    r"\{\s*(?:[+\-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em)?)\s*\}\s*"
)
_DISPLAY_MATH_ENVIRONMENTS = frozenset(
    {
        "equation",
        "equation*",
        "displaymath",
        "align",
        "align*",
        "alignat",
        "alignat*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "flalign",
        "flalign*",
        "split",
    }
)
_SINGLE_DISPLAY_MATH_ENVIRONMENTS = frozenset(
    {"equation", "equation*", "displaymath"}
)
_UNNUMBERED_MULTILINE_MATH_ENVIRONMENTS = {
    "align*": "aligned",
    "gather*": "gathered",
    "flalign*": "aligned",
    "multline*": "aligned",
}
_TRANSPARENT_BLOCK_ENVIRONMENTS = frozenset(
    {
        "abstract",
        "quote",
        "quotation",
        "center",
        "flushleft",
        "flushright",
        "small",
    }
)
_EXTERNAL_VERBATIM_CALL_ID = re.compile(r"^extverb-call-[0-9a-f]{16,64}$")
_THEOREM_BLOCK_ENVIRONMENTS = frozenset(
    {
        "assumption",
        "axiom",
        "claim",
        "conjecture",
        "corollary",
        "definition",
        "example",
        "fact",
        "lemma",
        "notation",
        "proposition",
        "property",
        "remark",
        "theorem",
    }
)
_PROOF_BLOCK_ENVIRONMENTS = frozenset({"proof"})
_INVISIBLE_TOP_LEVEL_COMMANDS = frozenset(
    {
        "maketitle",
        "tableofcontents",
        "newpage",
        "clearpage",
        "pagebreak",
        "nopagebreak",
        "vfill",
        "smallskip",
        "medskip",
        "bigskip",
    }
)

_SOURCE_DEFINITION_COMMANDS = frozenset(
    {
        "AddToHook",
        "AtBeginDocument",
        "AtEndDocument",
        "DeclareDocumentCommand",
        "DeclareMathOperator",
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
_SOURCE_VERBATIM_ENVIRONMENTS = frozenset(
    {"BVerbatim", "LVerbatim", "SaveVerbatim", "Verbatim", "lstlisting", "minted", "verbatim", "verbatim*"}
)


def _make_latex_context(*, enable_strict_tables: bool = False):
    context = get_default_latex_context_db()
    macros: list[MacroSpec] = []
    for name in sorted(_FRONTMATTER_COMMANDS):
        # Optional short form followed by one required visible argument.
        macros.append(MacroSpec(name, MacroStandardArgsParser("[{")))
    for name in sorted(_HEADING_LEVELS):
        macros.append(MacroSpec(name, MacroStandardArgsParser("*[{")))
    macros.extend(
        [
            MacroSpec("caption", MacroStandardArgsParser("[{")),
            MacroSpec("item", MacroStandardArgsParser("[")),
            MacroSpec("footnote", MacroStandardArgsParser("[{")),
            MacroSpec("label", MacroStandardArgsParser("{")),
            MacroSpec("verbatiminput", MacroStandardArgsParser("{")),
        ]
    )
    if enable_strict_tables:
        # These specs are enabled only for the strict source-only table
        # adapter.  They attach wrapper/control arguments to the macro node so
        # ``process_table_float`` can audit them without guessing about
        # adjacent groups.
        macros.extend(
            [
                MacroSpec("resizebox", MacroStandardArgsParser("*{{{")),
                MacroSpec("scalebox", MacroStandardArgsParser("{[{")),
                MacroSpec("adjustbox", MacroStandardArgsParser("{{")),
                MacroSpec("setlength", MacroStandardArgsParser("{{")),
                MacroSpec("addtolength", MacroStandardArgsParser("{{")),
                MacroSpec("renewcommand", MacroStandardArgsParser("*{[[{")),
                MacroSpec("vspace", MacroStandardArgsParser("*{")),
                MacroSpec("hspace", MacroStandardArgsParser("*{")),
            ]
        )
    context.add_context_category(
        "source-first-v3-document-ast",
        macros=macros,
        prepend=True,
    )
    return context


def _advance_line_column(text: str, line: int, column: int) -> tuple[int, int]:
    newline_count = text.count("\n")
    if newline_count == 0:
        return line, column + len(text)
    return line + newline_count, len(text) - text.rfind("\n") - 1


class _PositionIndex:
    def __init__(self, source: str) -> None:
        self.source = source
        self.byte_prefix = [0]
        self.line_starts = [0]
        for index, char in enumerate(source):
            self.byte_prefix.append(self.byte_prefix[-1] + len(char.encode("utf-8")))
            if char == "\n":
                self.line_starts.append(index + 1)

    def line_column(self, position: int) -> tuple[int, int]:
        line_index = bisect_right(self.line_starts, position) - 1
        return line_index + 1, position - self.line_starts[line_index]

    def span(self, start: int, end: int) -> DocumentSourceSpan:
        if start < 0 or end < start or end > len(self.source):
            raise DocumentAstError(f"invalid canonical span: {start}:{end}")
        start_line, start_column = self.line_column(start)
        end_line, end_column = self.line_column(end)
        return DocumentSourceSpan(
            char_start=start,
            char_end=end,
            byte_start=self.byte_prefix[start],
            byte_end=self.byte_prefix[end],
            line_start=start_line,
            column_start=start_column,
            line_end=end_line,
            column_end=end_column,
        )


def _stable_id(
    prefix: str,
    source_id: str,
    kind: str,
    span: DocumentSourceSpan,
    raw_source: str,
    parent_node_id: str | None,
) -> str:
    payload = "\x1f".join(
        (
            DOCUMENT_AST_VERSION,
            source_id,
            kind,
            str(span.char_start),
            str(span.char_end),
            str(span.byte_start),
            str(span.byte_end),
            parent_node_id or "",
            hashlib.sha256(raw_source.encode("utf-8")).hexdigest(),
        )
    )
    return f"sfv3_doc_{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _node_end(node: Any) -> int:
    return int(node.pos) + int(node.len)


def _group_content_span(node: LatexGroupNode | None) -> tuple[int, int] | None:
    if node is None or not isinstance(node, LatexGroupNode):
        return None
    delimiters = getattr(node, "delimiters", None)
    if not delimiters or len(delimiters[0]) != 1 or len(delimiters[1]) != 1:
        return None
    return node.pos + 1, _node_end(node) - 1


def _macro_arguments(
    node: LatexMacroNode | LatexEnvironmentNode,
) -> tuple[Any | None, ...]:
    nodeargd = getattr(node, "nodeargd", None)
    return tuple(getattr(nodeargd, "argnlist", ()) or ())


def _required_group(node: LatexMacroNode) -> LatexGroupNode | None:
    groups = [
        arg
        for arg in _macro_arguments(node)
        if isinstance(arg, LatexGroupNode) and getattr(arg, "delimiters", None) == ("{", "}")
    ]
    return groups[-1] if groups else None


def _optional_group(
    node: LatexMacroNode | LatexEnvironmentNode,
) -> LatexGroupNode | None:
    groups = [
        arg
        for arg in _macro_arguments(node)
        if isinstance(arg, LatexGroupNode) and getattr(arg, "delimiters", None) == ("[", "]")
    ]
    return groups[0] if groups else None


def _macro_is_starred(node: LatexMacroNode) -> bool:
    return any(
        isinstance(arg, LatexCharsNode) and arg.chars == "*"
        for arg in _macro_arguments(node)
    )


def _is_unescaped_percent(source: str, index: int) -> bool:
    if source[index] != "%":
        return False
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and source[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 0


def _source_command_end(source: str, start: int, limit: int) -> int:
    cursor = start + 1
    if cursor >= limit:
        return cursor
    if source[cursor].isalpha() or source[cursor] == "@":
        cursor += 1
        while cursor < limit and (source[cursor].isalpha() or source[cursor] == "@"):
            cursor += 1
        return cursor
    return min(limit, cursor + 1)


def _skip_source_ignored(source: str, cursor: int, limit: int) -> int:
    while cursor < limit:
        if source[cursor].isspace():
            cursor += 1
            continue
        if source[cursor] == "%" and _is_unescaped_percent(source, cursor):
            newline = source.find("\n", cursor, limit)
            cursor = limit if newline < 0 else newline + 1
            continue
        break
    return cursor


def _source_balanced_end(
    source: str,
    opening: int,
    limit: int,
    open_char: str = "{",
    close_char: str = "}",
) -> int | None:
    if opening >= limit or source[opening] != open_char:
        return None
    depth = 0
    cursor = opening
    while cursor < limit:
        char = source[cursor]
        if char == "%" and _is_unescaped_percent(source, cursor):
            newline = source.find("\n", cursor, limit)
            cursor = limit if newline < 0 else newline + 1
            continue
        if char == "\\":
            cursor = _source_command_end(source, cursor, limit)
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return None


def _skip_inline_verb(source: str, command_end: int, limit: int) -> int:
    if command_end >= limit:
        return command_end
    delimiter = source[command_end]
    if delimiter.isspace() or delimiter.isalpha():
        return command_end
    closing = source.find(delimiter, command_end + 1, limit)
    return limit if closing < 0 else closing + 1


def _skip_verbatim_environment(
    source: str,
    environment: str,
    body_start: int,
    limit: int,
) -> int:
    closing = re.compile(
        r"\\end\s*\{" + re.escape(environment) + r"\}"
    ).search(source, body_start, limit)
    if closing is None:
        raise DocumentAstError(
            f"unterminated verbatim-like environment while locating document: {environment}"
        )
    return closing.end()


def locate_document_body(source: str) -> DocumentBodyBoundary:
    """Locate one top-level, uncommented ``document`` environment.

    The scan is brace-, comment-, inline-verbatim-, and verbatim-environment
    aware.  It deliberately does not parse preamble macro definitions, so a
    mismatched environment token inside a balanced replacement body cannot
    prevent recovery of the real document body.
    """

    if not isinstance(source, str):
        raise DocumentAstError("source must be a string")
    begins: list[tuple[int, int]] = []
    ends: list[tuple[int, int]] = []
    depth = 0
    cursor = 0
    limit = len(source)
    while cursor < limit:
        char = source[cursor]
        if char == "%" and _is_unescaped_percent(source, cursor):
            newline = source.find("\n", cursor, limit)
            cursor = limit if newline < 0 else newline + 1
            continue
        if char == "\\":
            command_end = _source_command_end(source, cursor, limit)
            name = source[cursor + 1 : command_end]
            if depth == 0 and name in {"verb", "verb*", "lstinline"}:
                cursor = _skip_inline_verb(source, command_end, limit)
                continue
            if depth == 0 and name in {"begin", "end"}:
                opening = _skip_source_ignored(source, command_end, limit)
                group_end = _source_balanced_end(source, opening, limit)
                if group_end is not None:
                    environment = source[opening + 1 : group_end - 1].strip()
                    if environment == "document":
                        target = begins if name == "begin" else ends
                        target.append((cursor, group_end))
                    if name == "begin" and environment in _SOURCE_VERBATIM_ENVIRONMENTS:
                        cursor = _skip_verbatim_environment(
                            source, environment, group_end, limit
                        )
                        continue
                    cursor = group_end
                    continue
            cursor = command_end
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        cursor += 1
    if len(begins) != 1 or len(ends) != 1:
        raise DocumentAstError(
            "canonical source must contain exactly one top-level uncommented "
            f"document environment; begin={len(begins)} end={len(ends)}"
        )
    begin_start, begin_end = begins[0]
    end_start, end_end = ends[0]
    if end_start < begin_end:
        raise DocumentAstError("document environment end precedes its begin")
    return DocumentBodyBoundary(
        begin_command_start=begin_start,
        begin_command_end=begin_end,
        body_start=begin_end,
        body_end=end_start,
        end_command_start=end_start,
        end_command_end=end_end,
    )


def _definition_extent_for_frontmatter(source: str, start: int, command_end: int, name: str, limit: int) -> int:
    base = name.rstrip("*")
    cursor = command_end
    if base in {"def", "edef", "gdef", "xdef", "long"}:
        while cursor < limit:
            if source[cursor] == "%" and _is_unescaped_percent(source, cursor):
                newline = source.find("\n", cursor, limit)
                cursor = limit if newline < 0 else newline + 1
                continue
            if source[cursor] == "{":
                return _source_balanced_end(source, cursor, limit) or limit
            cursor += 1
        return command_end
    if base == "let":
        newline = source.find("\n", cursor, limit)
        return limit if newline < 0 else newline
    last_end = command_end
    saw_argument = False
    while cursor < limit:
        cursor = _skip_source_ignored(source, cursor, limit)
        if cursor < limit and source[cursor] == "*":
            cursor += 1
            last_end = cursor
            continue
        if cursor < limit and source[cursor] == "\\" and not saw_argument:
            cursor = _source_command_end(source, cursor, limit)
            saw_argument = True
            last_end = cursor
            continue
        if cursor < limit and source[cursor] in "[{":
            opening = source[cursor]
            closing = "]" if opening == "[" else "}"
            group_end = _source_balanced_end(
                source, cursor, limit, opening, closing
            )
            if group_end is None:
                return limit
            cursor = group_end
            saw_argument = True
            last_end = group_end
            continue
        break
    return max(command_end, last_end)


@dataclass(frozen=True, slots=True)
class _FrontmatterCall:
    name: str
    start: int
    end: int
    content_start: int
    content_end: int
    optional_label: str | None


def _scan_safe_frontmatter(
    source: str,
    limit: int,
) -> tuple[tuple[_FrontmatterCall, ...], tuple[tuple[int, int, str], ...]]:
    calls: list[_FrontmatterCall] = []
    failures: list[tuple[int, int, str]] = []
    depth = 0
    cursor = 0
    while cursor < limit:
        char = source[cursor]
        if char == "%" and _is_unescaped_percent(source, cursor):
            newline = source.find("\n", cursor, limit)
            cursor = limit if newline < 0 else newline + 1
            continue
        if char == "\\":
            command_end = _source_command_end(source, cursor, limit)
            name = source[cursor + 1 : command_end]
            if depth == 0 and name.rstrip("*") in _SOURCE_DEFINITION_COMMANDS:
                cursor = _definition_extent_for_frontmatter(
                    source, cursor, command_end, name, limit
                )
                continue
            if depth == 0 and name in _FRONTMATTER_COMMANDS:
                argument_cursor = _skip_source_ignored(source, command_end, limit)
                optional_label: str | None = None
                if argument_cursor < limit and source[argument_cursor] == "[":
                    optional_end = _source_balanced_end(
                        source, argument_cursor, limit, "[", "]"
                    )
                    if optional_end is None:
                        failures.append(
                            (cursor, min(limit, command_end), f"unterminated optional argument on \\{name}")
                        )
                        cursor = command_end
                        continue
                    optional_label = source[argument_cursor + 1 : optional_end - 1]
                    argument_cursor = _skip_source_ignored(source, optional_end, limit)
                if argument_cursor >= limit or source[argument_cursor] != "{":
                    failures.append(
                        (cursor, command_end, f"missing required argument on \\{name}")
                    )
                    cursor = command_end
                    continue
                required_end = _source_balanced_end(
                    source, argument_cursor, limit
                )
                if required_end is None:
                    failures.append(
                        (cursor, command_end, f"unterminated required argument on \\{name}")
                    )
                    cursor = command_end
                    continue
                calls.append(
                    _FrontmatterCall(
                        name=name,
                        start=cursor,
                        end=required_end,
                        content_start=argument_cursor + 1,
                        content_end=required_end - 1,
                        optional_label=optional_label,
                    )
                )
                cursor = required_end
                continue
            cursor = command_end
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        cursor += 1
    return tuple(calls), tuple(failures)


def _paragraph_breaks(source: str, start: int, end: int) -> tuple[tuple[int, int], ...]:
    """Find top-level TeX blank lines without splitting groups/comments."""

    masked = list(source[start:end])
    brace_depth = 0
    bracket_depth = 0
    cursor = start
    while cursor < end:
        local = cursor - start
        char = source[cursor]
        if char == "%" and _is_unescaped_percent(source, cursor):
            newline = source.find("\n", cursor, end)
            comment_end = end if newline < 0 else newline + 1
            for index in range(cursor, comment_end):
                # TeX discards the comment newline too; hiding it prevents a
                # comment-only physical line from inventing a paragraph.
                masked[index - start] = " "
            cursor = comment_end
            continue
        escaped = cursor > start and source[cursor - 1] == "\\"
        if char == "{" and not escaped:
            brace_depth += 1
        elif char == "}" and not escaped:
            brace_depth = max(0, brace_depth - 1)
        elif char == "[" and not escaped:
            bracket_depth += 1
        elif char == "]" and not escaped:
            bracket_depth = max(0, bracket_depth - 1)
        if char == "\n" and (brace_depth or bracket_depth):
            masked[local] = " "
        cursor += 1
    masked_text = "".join(masked)
    return tuple(
        (start + match.start(), start + match.end())
        for match in re.finditer(r"\n[ \t\r\f\v]*\n(?:[ \t\r\f\v]*\n)*", masked_text)
    )


def _trim_source_region(source: str, start: int, end: int) -> tuple[int, int]:
    while start < end:
        if source[start].isspace():
            start += 1
            continue
        if source[start] == "%" and _is_unescaped_percent(source, start):
            newline = source.find("\n", start, end)
            start = end if newline < 0 else newline + 1
            continue
        break
    while end > start and source[end - 1].isspace():
        end -= 1
    return start, end


@dataclass(slots=True)
class _NodeState:
    node: DocumentBlockNode
    children: list[str] = field(default_factory=list)


class _DocumentBuilder:
    def __init__(
        self,
        source: str,
        source_id: str,
        source_map: CanonicalSourceMap,
        reference_values: Mapping[str, str],
        math_macros: Mapping[str, MathMacroDefinition],
        semantic_environments: Mapping[str, SemanticEnvironmentDefinition],
        list_environments: Mapping[str, SourceListEnvironmentDefinition],
        external_verbatim_calls: Mapping[tuple[int, int], str],
        enable_strict_tables: bool,
    ) -> None:
        self.source = source
        self.source_id = source_id
        self.source_map = source_map
        self.positions = _PositionIndex(source)
        self.reference_values = reference_values
        self.math_macros = dict(math_macros)
        self.semantic_environments = dict(semantic_environments)
        self.list_environments = dict(list_environments)
        self.external_verbatim_calls = dict(external_verbatim_calls)
        self.enable_strict_tables = enable_strict_tables
        self.consumed_external_verbatim_calls: set[tuple[int, int]] = set()
        # ``\label`` is invisible and its exact argument has no GT text.
        self.math_macros.setdefault("label", MathMacroDefinition("label", 1, ""))
        self.states: list[_NodeState] = []
        self.state_by_id: dict[str, _NodeState] = {}
        self.rejections: list[DocumentAstRejection] = []

    def add_node(
        self,
        kind: str,
        start: int,
        end: int,
        parent_node_id: str | None,
        *,
        content: tuple[int, int] | None = None,
        inline_ir: SourceDocumentIR | None = None,
        command_name: str | None = None,
        environment_name: str | None = None,
        heading_level: int | None = None,
        starred: bool = False,
        optional_label: str | None = None,
        optional_inline_ir: SourceDocumentIR | None = None,
        metadata: Sequence[tuple[str, str]] = (),
    ) -> DocumentBlockNode:
        span = self.positions.span(start, end)
        node_id = _stable_id(
            "node", self.source_id, kind, span, self.source[start:end], parent_node_id
        )
        node = DocumentBlockNode(
            node_id=node_id,
            kind=kind,
            span=span,
            content_span=(None if content is None else self.positions.span(*content)),
            parent_node_id=parent_node_id,
            child_node_ids=(),
            raw_source=self.source[start:end],
            origins=self.source_map.origins_for_span(self.source, start, end),
            inline_ir=inline_ir,
            command_name=command_name,
            environment_name=environment_name,
            heading_level=heading_level,
            starred=starred,
            optional_label=optional_label,
            optional_inline_ir=optional_inline_ir,
            metadata=tuple(metadata),
        )
        state = _NodeState(node)
        self.states.append(state)
        self.state_by_id[node_id] = state
        if parent_node_id is not None:
            self.state_by_id[parent_node_id].children.append(node_id)
        return node

    def reject(
        self,
        code: str,
        message: str,
        start: int,
        end: int,
        parent_node_id: str | None,
        *,
        node_id: str | None = None,
    ) -> None:
        span = self.positions.span(start, end)
        raw = self.source[start:end]
        self.rejections.append(
            DocumentAstRejection(
                rejection_id=_stable_id(
                    "reject", self.source_id, code, span, raw, parent_node_id
                ),
                code=code,
                message=message,
                span=span,
                raw_source=raw,
                parent_node_id=parent_node_id,
                node_id=node_id,
            )
        )

    def _semantic_body_style_stack(
        self,
        parent_node_id: str | None,
    ) -> tuple[str, ...]:
        current = parent_node_id
        while current is not None:
            parent = self.state_by_id[current].node
            if parent.kind in {"theorem_region", "proof_region"}:
                body_style = dict(parent.metadata).get("body_style", "plain")
                if body_style == "plain":
                    return ()
                if body_style == "em":
                    return ("body_em",)
                if body_style == "strong":
                    return ("strong",)
                if body_style == "strong_em":
                    return ("strong", "body_em")
                raise DocumentAstError(
                    f"unsupported semantic body style: {body_style}"
                )
            current = parent.parent_node_id
        return ()

    def inline_ir(
        self,
        start: int,
        end: int,
        node_hint: str,
        *,
        parent_node_id: str | None = None,
    ) -> SourceDocumentIR:
        return parse_source_ir(
            self.source[start:end],
            source_id=f"{self.source_id}#{node_hint}:{start}:{end}",
            source_char_base=start,
            source_byte_base=self.positions.byte_prefix[start],
            reference_values=self.reference_values,
            math_macros=self.math_macros,
            initial_style_stack=self._semantic_body_style_stack(parent_node_id),
        )

    def add_inline_block(
        self,
        kind: str,
        start: int,
        end: int,
        parent_node_id: str | None,
        *,
        content: tuple[int, int] | None = None,
        **attributes: Any,
    ) -> DocumentBlockNode | None:
        content_start, content_end = content or (start, end)
        content_start, content_end = _trim_source_region(
            self.source, content_start, content_end
        )
        if content_end <= content_start:
            self.reject(
                "empty_visible_block",
                f"{kind} has no visible source content",
                start,
                end,
                parent_node_id,
            )
            return None
        try:
            ir = self.inline_ir(
                content_start,
                content_end,
                kind,
                parent_node_id=parent_node_id,
            )
        except SourceIrError as exc:
            self.reject(
                "inline_parse_error",
                f"{kind} inline parse failed: {exc}",
                start,
                end,
                parent_node_id,
            )
            return None
        if not ir.atoms:
            # A block containing only comments, labels, or layout controls is
            # non-visible and therefore not a GT block.
            return None
        node = self.add_node(
            kind,
            start,
            end,
            parent_node_id,
            content=(content_start, content_end),
            inline_ir=ir,
            **attributes,
        )
        for opaque in ir.opaque_atoms:
            self.reject(
                "opaque_inline_latex",
                f"unsupported inline LaTeX in {kind}: {opaque.raw_source[:120]}",
                opaque.span.char_start,
                opaque.span.char_end,
                parent_node_id,
                node_id=node.node_id,
            )
        return node

    def emit_paragraph_region(
        self,
        start: int,
        end: int,
        parent_node_id: str | None,
    ) -> None:
        cursor = start
        for break_start, break_end in _paragraph_breaks(self.source, start, end):
            self._emit_one_paragraph(cursor, break_start, parent_node_id)
            cursor = break_end
        self._emit_one_paragraph(cursor, end, parent_node_id)

    def _emit_one_paragraph(
        self,
        start: int,
        end: int,
        parent_node_id: str | None,
    ) -> None:
        start, end = _trim_source_region(self.source, start, end)
        if end <= start:
            return
        self.add_inline_block("paragraph", start, end, parent_node_id)

    def process_sequence(
        self,
        nodes: Sequence[Any],
        scope_start: int,
        scope_end: int,
        parent_node_id: str | None,
    ) -> None:
        pending_start = scope_start
        for node in nodes:
            node_start = int(node.pos)
            node_end = _node_end(node)
            is_barrier = self._is_block_barrier(node)
            if not is_barrier:
                continue
            if node_start > pending_start:
                self.emit_paragraph_region(pending_start, node_start, parent_node_id)
            self.process_block(node, parent_node_id)
            pending_start = max(pending_start, node_end)
        if pending_start < scope_end:
            self.emit_paragraph_region(pending_start, scope_end, parent_node_id)

    def _is_block_barrier(self, node: Any) -> bool:
        if isinstance(node, LatexMathNode):
            return getattr(node, "displaytype", None) == "display"
        if isinstance(node, LatexEnvironmentNode):
            return True
        if not isinstance(node, LatexMacroNode):
            return False
        name = node.macroname
        return (
            name in _FRONTMATTER_COMMANDS
            or name in _HEADING_LEVELS
            or name in _INVISIBLE_TOP_LEVEL_COMMANDS
            or name == "verbatiminput"
            or name == "par"
        )

    def process_block(self, node: Any, parent_node_id: str | None) -> None:
        if isinstance(node, LatexMathNode):
            self.process_display_math(node, parent_node_id)
            return
        if isinstance(node, LatexMacroNode):
            if node.macroname in _INVISIBLE_TOP_LEVEL_COMMANDS or node.macroname == "par":
                return
            if node.macroname in _FRONTMATTER_COMMANDS:
                self.process_structured_command(node, parent_node_id, "frontmatter")
                return
            if node.macroname in _HEADING_LEVELS:
                self.process_structured_command(node, parent_node_id, "heading")
                return
            if node.macroname == "verbatiminput":
                self.process_external_verbatim(node, parent_node_id)
                return
        if isinstance(node, LatexEnvironmentNode):
            self.process_environment(node, parent_node_id)
            return
        self.reject(
            "unsupported_block_node",
            f"unsupported block node {type(node).__name__}",
            int(node.pos),
            _node_end(node),
            parent_node_id,
        )

    def process_external_verbatim(
        self,
        node: LatexMacroNode,
        parent_node_id: str | None,
    ) -> None:
        """Register one externally serialized literal source block.

        The caller has already proved path safety and clean-compile execution.
        This AST node contributes only source identity and document ordering;
        its literal line records and page slices remain outside inline IR and
        must be supplied by the compiler-native external-verbatim trace.
        """

        span = (int(node.pos), _node_end(node))
        call_id = self.external_verbatim_calls.get(span)
        if call_id is None:
            opaque = self.add_node(
                "external_verbatim",
                span[0],
                span[1],
                parent_node_id,
                command_name="verbatiminput",
                metadata=(("external_verbatim_status", "unresolved"),),
            )
            self.reject(
                "unsafe_external_verbatim_call",
                "external verbatim call lacks clean-executed literal source proof",
                span[0],
                span[1],
                parent_node_id,
                node_id=opaque.node_id,
            )
            return
        self.consumed_external_verbatim_calls.add(span)
        self.add_node(
            "external_verbatim",
            span[0],
            span[1],
            parent_node_id,
            command_name="verbatiminput",
            metadata=(
                ("external_verbatim_call_id", call_id),
                ("ground_truth_source", "external_verbatim_file"),
            ),
        )

    def process_structured_command(
        self,
        node: LatexMacroNode,
        parent_node_id: str | None,
        kind: str,
    ) -> None:
        group = _required_group(node)
        content = _group_content_span(group)
        if content is None:
            self.reject(
                "missing_required_block_argument",
                f"\\{node.macroname} has no unambiguous required argument",
                node.pos,
                _node_end(node),
                parent_node_id,
            )
            return
        optional = _optional_group(node)
        optional_text = None
        if optional is not None:
            optional_span = _group_content_span(optional)
            if optional_span is not None:
                optional_text = self.source[slice(*optional_span)]
        attributes: dict[str, Any] = {
            "command_name": node.macroname,
            "starred": _macro_is_starred(node),
            "optional_label": optional_text,
        }
        if kind == "heading":
            attributes["heading_level"] = _HEADING_LEVELS[node.macroname]
        self.add_inline_block(
            kind,
            node.pos,
            _node_end(node),
            parent_node_id,
            content=content,
            **attributes,
        )

    def process_display_math(self, node: LatexMathNode, parent_node_id: str | None) -> None:
        delimiters = getattr(node, "delimiters", None)
        content: tuple[int, int] | None = None
        if delimiters and len(delimiters) == 2:
            content = (node.pos + len(delimiters[0]), _node_end(node) - len(delimiters[1]))
        if content is None or content[1] < content[0]:
            self.reject(
                "malformed_display_math",
                "display math delimiters are not recoverable",
                node.pos,
                _node_end(node),
                parent_node_id,
            )
            return
        start, end = int(node.pos), _node_end(node)
        try:
            ir = build_display_math_ir(
                self.source[start:end],
                content_start=content[0] - start,
                content_end=content[1] - start,
                source_id=f"{self.source_id}#display_math:{start}:{end}",
                source_char_base=start,
                source_byte_base=self.positions.byte_prefix[start],
                math_macros=self.math_macros,
            )
        except SourceIrError as exc:
            self.reject(
                "display_math_parse_error",
                f"display math source serialization failed: {exc}",
                start,
                end,
                parent_node_id,
            )
            return
        self.add_node(
            "display_math",
            start,
            end,
            parent_node_id,
            content=content,
            inline_ir=ir,
            metadata=(("delimiter_open", delimiters[0]), ("delimiter_close", delimiters[1])),
        )

    def process_environment(
        self,
        node: LatexEnvironmentNode,
        parent_node_id: str | None,
    ) -> None:
        def reject_opaque_environment(code: str, message: str) -> None:
            opaque_content, point_anchor_safe = self._opaque_environment_body_span(
                node
            )
            opaque = self.add_node(
                "opaque_environment",
                node.pos,
                _node_end(node),
                parent_node_id,
                content=opaque_content,
                environment_name=name,
                metadata=(
                    (
                        "compiler_point_anchor_safe",
                        "true" if point_anchor_safe and name == "algorithm" else "false",
                    ),
                ),
            )
            self.reject(
                code,
                message,
                node.pos,
                _node_end(node),
                parent_node_id,
                node_id=opaque.node_id,
            )

        name = node.environmentname
        semantic = self.semantic_environments.get(name)
        if semantic is not None:
            self.process_semantic_text_environment(
                node,
                parent_node_id,
                region_kind=(
                    "proof_region" if semantic.kind == "proof" else "theorem_region"
                ),
                definition=semantic,
            )
        elif name in _LIST_ENVIRONMENTS:
            self.process_list(node, parent_node_id, list_kind=name)
        elif name in self.list_environments:
            self.process_list(
                node,
                parent_node_id,
                list_kind=self.list_environments[name].list_kind,
            )
        elif name in _TABLE_ENVIRONMENTS:
            self.process_table_float(node, parent_node_id)
        elif name in _TABULAR_ENVIRONMENTS:
            self.process_tabular(node, parent_node_id)
        elif (
            name in _SINGLE_DISPLAY_MATH_ENVIRONMENTS
            or name in _UNNUMBERED_MULTILINE_MATH_ENVIRONMENTS
        ):
            self.process_math_environment(node, parent_node_id)
        elif name == "align":
            self.process_numbered_align_environment(node, parent_node_id)
        elif name in _DISPLAY_MATH_ENVIRONMENTS:
            unsupported = self.add_node(
                "display_math",
                node.pos,
                _node_end(node),
                parent_node_id,
                content=self._environment_body_span(node),
                environment_name=name,
                starred=name.endswith("*"),
            )
            self.reject(
                "unsupported_multiline_display_math",
                f"display math environment requires row-level compiler semantics: {name}",
                node.pos,
                _node_end(node),
                parent_node_id,
                node_id=unsupported.node_id,
            )
        elif name in _THEOREM_BLOCK_ENVIRONMENTS:
            reject_opaque_environment(
                "unresolved_semantic_environment",
                f"theorem caption/style is not resolved from source declarations: {name}",
            )
        elif name in _PROOF_BLOCK_ENVIRONMENTS:
            reject_opaque_environment(
                "unresolved_semantic_environment",
                f"proof caption/style is not resolved from source declarations: {name}",
            )
        elif name in _TRANSPARENT_BLOCK_ENVIRONMENTS:
            self.process_transparent_environment(node, parent_node_id)
        else:
            reject_opaque_environment(
                "unsupported_opaque_environment",
                f"unsupported block environment: {name}",
            )

    def _environment_body_span(self, node: LatexEnvironmentNode) -> tuple[int, int]:
        nodelist = tuple(node.nodelist or ())
        if nodelist:
            return int(nodelist[0].pos), _node_end(nodelist[-1])
        begin_match = re.match(r"\\begin\s*\{[^{}]+\}", self.source[node.pos : _node_end(node)])
        end_match = re.search(r"\\end\s*\{[^{}]+\}\s*$", self.source[node.pos : _node_end(node)])
        body_start = node.pos + (0 if begin_match is None else begin_match.end())
        body_end = _node_end(node) - (0 if end_match is None else len(end_match.group(0)))
        return body_start, max(body_start, body_end)

    def _opaque_environment_body_span(
        self, node: LatexEnvironmentNode
    ) -> tuple[tuple[int, int], bool]:
        """Return an inside-environment span and point-anchor safety proof.

        Unknown environments are not parsed using guessed argument specs.  A
        compiler point anchor is safe only when the begin token is exact and
        every immediate optional argument is a balanced printable literal.
        An immediate mandatory group is deliberately treated as unknown.  The
        current point-anchor allowlist additionally restricts use to the known
        floating ``algorithm`` environment.
        """

        start, end = int(node.pos), _node_end(node)
        raw = self.source[start:end]
        begin_match = re.match(r"\\begin\s*\{[A-Za-z@*]+\}", raw)
        end_match = re.search(r"\\end\s*\{[A-Za-z@*]+\}\s*$", raw)
        if begin_match is None or end_match is None:
            return self._environment_body_span(node), False
        cursor = start + begin_match.end()
        while cursor < end and self.source[cursor].isspace():
            cursor += 1
        if cursor < end and self.source[cursor] == "%":
            # TeX skips comments while scanning an environment's optional
            # argument.  Inserting a marker before that comment could stop
            # the following ``[...]`` from being consumed by ``\begin``.
            return self._environment_body_span(node), False
        while cursor < end and self.source[cursor] == "[":
            closing = self.source.find("]", cursor + 1, end)
            if closing < 0:
                return self._environment_body_span(node), False
            literal = self.source[cursor + 1 : closing]
            if not re.fullmatch(r"[A-Za-z0-9, !+\-]*", literal):
                return self._environment_body_span(node), False
            cursor = closing + 1
            while cursor < end and self.source[cursor].isspace():
                cursor += 1
        if cursor < end and self.source[cursor] == "{":
            return self._environment_body_span(node), False
        body_end = start + end_match.start()
        return (cursor, max(cursor, body_end)), True

    def process_math_environment(
        self,
        node: LatexEnvironmentNode,
        parent_node_id: str | None,
    ) -> None:
        start, end = int(node.pos), _node_end(node)
        body_start, body_end = self._environment_body_span(node)
        try:
            resolved_tag = (
                resolve_display_math_reference_tag(
                    self.source[body_start:body_end],
                    self.reference_values,
                )
                if node.environmentname == "equation"
                else None
            )
            ir = build_display_math_ir(
                self.source[start:end],
                content_start=body_start - start,
                content_end=body_end - start,
                source_id=f"{self.source_id}#display_math:{start}:{end}",
                source_char_base=start,
                source_byte_base=self.positions.byte_prefix[start],
                math_macros=self.math_macros,
                markdown_environment=_UNNUMBERED_MULTILINE_MATH_ENVIRONMENTS.get(
                    node.environmentname
                ),
                resolved_tag=resolved_tag,
            )
        except SourceIrError as exc:
            self.reject(
                "display_math_parse_error",
                f"display math source serialization failed: {exc}",
                start,
                end,
                parent_node_id,
            )
            return
        self.add_node(
            "display_math",
            start,
            end,
            parent_node_id,
            content=(body_start, body_end),
            inline_ir=ir,
            environment_name=node.environmentname,
            starred=node.environmentname.endswith("*"),
        )

    def process_numbered_align_environment(
        self,
        node: LatexEnvironmentNode,
        parent_node_id: str | None,
    ) -> None:
        """Serialize only align rows whose numbering is compiler-provable."""

        start, end = int(node.pos), _node_end(node)
        body_start, body_end = self._environment_body_span(node)
        try:
            ir = build_numbered_align_ir(
                self.source[start:end],
                content_start=body_start - start,
                content_end=body_end - start,
                reference_values=self.reference_values,
                source_id=f"{self.source_id}#display_math:{start}:{end}",
                source_char_base=start,
                source_byte_base=self.positions.byte_prefix[start],
                math_macros=self.math_macros,
            )
        except SourceIrError as exc:
            rejected = self.add_node(
                "display_math",
                start,
                end,
                parent_node_id,
                content=(body_start, body_end),
                environment_name=node.environmentname,
                starred=False,
            )
            self.reject(
                "unsafe_numbered_align",
                f"numbered align row semantics are not compiler-provable: {exc}",
                start,
                end,
                parent_node_id,
                node_id=rejected.node_id,
            )
            return
        self.add_node(
            "display_math",
            start,
            end,
            parent_node_id,
            content=(body_start, body_end),
            inline_ir=ir,
            environment_name=node.environmentname,
            starred=False,
            metadata=(("numbering_source", "source_tag_or_compiler_aux"),),
        )

    def process_transparent_environment(
        self,
        node: LatexEnvironmentNode,
        parent_node_id: str | None,
    ) -> None:
        container = self.add_node(
            "region",
            node.pos,
            _node_end(node),
            parent_node_id,
            content=self._environment_body_span(node),
            environment_name=node.environmentname,
        )
        body_start, body_end = self._environment_body_span(node)
        self.process_sequence(tuple(node.nodelist or ()), body_start, body_end, container.node_id)

    def process_semantic_text_environment(
        self,
        node: LatexEnvironmentNode,
        parent_node_id: str | None,
        *,
        region_kind: str,
        definition: SemanticEnvironmentDefinition,
    ) -> None:
        """Parse theorem/proof bodies and retain their source environment name."""

        body_start, body_end = self._environment_body_span(node)
        environment = node.environmentname
        starred = not definition.numbered
        optional_group = _optional_group(node)
        optional_label: str | None = None
        optional_inline_ir: SourceDocumentIR | None = None
        optional_error: str | None = None
        optional_rejection_span = (node.pos, _node_end(node))
        if optional_group is not None:
            optional_rejection_span = (optional_group.pos, _node_end(optional_group))
            optional_span = _group_content_span(optional_group)
            if optional_span is None:
                # Keep a non-None sentinel so every serializer knows that an
                # optional title existed but was not safely represented.
                optional_label = ""
                optional_error = "optional title has malformed delimiters"
            else:
                optional_label = self.source[slice(*optional_span)]
                try:
                    optional_inline_ir = self.inline_ir(
                        *optional_span,
                        f"{region_kind}_optional_title",
                    )
                except SourceIrError as exc:
                    optional_error = f"optional title inline parse failed: {exc}"
                else:
                    if optional_inline_ir.opaque_atoms:
                        optional_error = (
                            "optional title contains unsupported inline LaTeX: "
                            + optional_inline_ir.opaque_atoms[0].raw_source[:120]
                        )
                    elif optional_inline_ir.footnotes:
                        optional_error = (
                            "optional title contains a footnote whose callout is not "
                            "compiler-numbered in the semantic header"
                        )
                    elif not any(
                        not atom.is_whitespace for atom in optional_inline_ir.atoms
                    ):
                        optional_error = "optional title has no visible source content"
        container = self.add_node(
            region_kind,
            node.pos,
            _node_end(node),
            parent_node_id,
            content=(body_start, body_end),
            environment_name=environment,
            starred=starred,
            optional_label=optional_label,
            optional_inline_ir=optional_inline_ir,
            metadata=definition.metadata(),
        )
        if optional_error is not None:
            self.reject(
                "unsupported_semantic_optional_title",
                optional_error,
                *optional_rejection_span,
                parent_node_id,
                node_id=container.node_id,
            )
        self.process_sequence(
            tuple(node.nodelist or ()), body_start, body_end, container.node_id
        )

    def process_list(
        self,
        node: LatexEnvironmentNode,
        parent_node_id: str | None,
        *,
        list_kind: str,
    ) -> None:
        if list_kind not in _LIST_ENVIRONMENTS:
            raise DocumentAstError(f"unsupported canonical list kind: {list_kind}")
        container = self.add_node(
            "list",
            node.pos,
            _node_end(node),
            parent_node_id,
            content=self._environment_body_span(node),
            environment_name=node.environmentname,
            metadata=(("list_kind", list_kind),),
        )
        body_nodes = tuple(node.nodelist or ())
        items = [
            (index, child)
            for index, child in enumerate(body_nodes)
            if isinstance(child, LatexMacroNode) and child.macroname == "item"
        ]
        if not items:
            self.reject(
                "list_without_items",
                f"{node.environmentname} contains no \\item",
                node.pos,
                _node_end(node),
                container.node_id,
                node_id=container.node_id,
            )
            return
        first_item = items[0][1]
        prefix_start, prefix_end = self._environment_body_span(node)[0], first_item.pos
        if self._region_has_visible_source(prefix_start, prefix_end):
            self.reject(
                "visible_content_before_first_item",
                "list has visible content before its first item",
                prefix_start,
                prefix_end,
                container.node_id,
                node_id=container.node_id,
            )
        env_body_end = self._environment_body_span(node)[1]
        for item_position, (child_index, item) in enumerate(items):
            next_start = (
                items[item_position + 1][1].pos
                if item_position + 1 < len(items)
                else env_body_end
            )
            item_nodes = [
                child
                for child in body_nodes[child_index + 1 :]
                if child.pos < next_start
            ]
            optional = _optional_group(item)
            optional_text = None
            if optional is not None:
                label_span = _group_content_span(optional)
                if label_span is not None:
                    optional_text = self.source[slice(*label_span)]
            content_start = _node_end(item)
            metadata = (
                ("list_kind", list_kind),
                ("item_index", str(item_position + 1)),
            )
            if any(self._is_block_barrier(child) for child in item_nodes):
                item_container = self.add_node(
                    "list_item_region",
                    item.pos,
                    next_start,
                    container.node_id,
                    content=(content_start, next_start),
                    command_name="item",
                    optional_label=optional_text,
                    metadata=metadata,
                )
                self.process_sequence(
                    tuple(item_nodes),
                    content_start,
                    next_start,
                    item_container.node_id,
                )
                if not self.state_by_id[item_container.node_id].children:
                    self.reject(
                        "empty_list_item_region",
                        "structured list item has no serializable child blocks",
                        item.pos,
                        next_start,
                        container.node_id,
                        node_id=item_container.node_id,
                    )
            else:
                self.add_inline_block(
                    "list_item",
                    item.pos,
                    next_start,
                    container.node_id,
                    content=(content_start, next_start),
                    command_name="item",
                    optional_label=optional_text,
                    metadata=metadata,
                )

    def _region_has_visible_source(self, start: int, end: int) -> bool:
        start, end = _trim_source_region(self.source, start, end)
        if end <= start:
            return False
        try:
            ir = self.inline_ir(start, end, "visibility")
        except SourceIrError:
            return True
        return bool(ir.atoms)

    def process_table_float(
        self,
        node: LatexEnvironmentNode,
        parent_node_id: str | None,
    ) -> None:
        container = self.add_node(
            "table_float",
            node.pos,
            _node_end(node),
            parent_node_id,
            content=self._environment_body_span(node),
            environment_name=node.environmentname,
        )
        captions: list[LatexMacroNode] = []
        tabulars: list[LatexEnvironmentNode] = []
        unsupported: list[Any] = []

        def group_text(group: LatexGroupNode) -> str | None:
            span = _group_content_span(group)
            return None if span is None else self.source[slice(*span)].strip()

        def wrapper_content(macro: LatexMacroNode) -> LatexGroupNode | None:
            groups = tuple(
                arg
                for arg in tuple(macro.nodeargd.argnlist or ())
                if isinstance(arg, LatexGroupNode)
            )
            if macro.macroname == "resizebox" and len(groups) == 3:
                width, height = group_text(groups[0]), group_text(groups[1])
                if (
                    width is not None
                    and height is not None
                    and _STRICT_TABLE_BOX_DIMENSION.fullmatch(width)
                    and _STRICT_TABLE_BOX_DIMENSION.fullmatch(height)
                ):
                    return groups[2]
            elif macro.macroname == "scalebox" and len(groups) in {2, 3}:
                scales = tuple(group_text(group) for group in groups[:-1])
                if scales and all(
                    scale is not None and _STRICT_TABLE_SCALE.fullmatch(scale)
                    for scale in scales
                ):
                    return groups[-1]
            elif macro.macroname == "adjustbox" and len(groups) == 2:
                options = group_text(groups[0])
                if (
                    options is not None
                    and _STRICT_TABLE_ADJUSTBOX_OPTIONS.fullmatch(options)
                ):
                    return groups[1]
            return None

        def safe_layout_control(macro: LatexMacroNode) -> bool:
            raw = self.source[macro.pos : _node_end(macro)]
            return bool(
                _STRICT_TABLE_LENGTH_CONTROL.fullmatch(raw)
                or _STRICT_TABLE_RENEW_CONTROL.fullmatch(raw)
                or _STRICT_TABLE_SPACING_CONTROL.fullmatch(raw)
            )

        def collect(children: Sequence[Any]) -> None:
            for child in children:
                if isinstance(child, LatexMacroNode) and child.macroname == "caption":
                    captions.append(child)
                elif (
                    isinstance(child, LatexEnvironmentNode)
                    and child.environmentname in _TABULAR_ENVIRONMENTS
                ):
                    tabulars.append(child)
                elif isinstance(child, LatexCommentNode):
                    continue
                elif isinstance(child, LatexCharsNode):
                    if child.chars.strip():
                        unsupported.append(child)
                elif isinstance(child, LatexMacroNode) and child.macroname in {
                    "label",
                    "centering",
                    *_STRICT_TABLE_DECLARATIONS,
                }:
                    continue
                elif (
                    self.enable_strict_tables
                    and isinstance(child, LatexMacroNode)
                    and child.macroname in _STRICT_TABLE_BOX_WRAPPERS
                ):
                    content = wrapper_content(child)
                    if content is None:
                        unsupported.append(child)
                    else:
                        collect(tuple(content.nodelist or ()))
                elif (
                    self.enable_strict_tables
                    and isinstance(child, LatexMacroNode)
                    and (
                        (
                            child.macroname
                            in {
                                "setlength",
                                "addtolength",
                                "renewcommand",
                                "vspace",
                                "hspace",
                            }
                            and safe_layout_control(child)
                        )
                        or child.macroname == "hfill"
                    )
                ):
                    continue
                elif self.enable_strict_tables and isinstance(child, LatexGroupNode):
                    collect(tuple(child.nodelist or ()))
                else:
                    unsupported.append(child)

        collect(tuple(node.nodelist or ()))
        for caption in captions:
            group = _required_group(caption)
            content = _group_content_span(group)
            if content is None:
                self.reject(
                    "malformed_table_caption",
                    "table caption has no required argument",
                    caption.pos,
                    _node_end(caption),
                    container.node_id,
                )
                continue
            optional = _optional_group(caption)
            optional_text = None
            if optional is not None:
                optional_span = _group_content_span(optional)
                if optional_span is not None:
                    optional_text = self.source[slice(*optional_span)]
            self.add_inline_block(
                "caption",
                caption.pos,
                _node_end(caption),
                container.node_id,
                content=content,
                command_name="caption",
                optional_label=optional_text,
            )
        for tabular in tabulars:
            self.process_tabular(tabular, container.node_id)
        if not tabulars:
            self.reject(
                "table_without_supported_tabular",
                "table float has no supported tabular environment",
                node.pos,
                _node_end(node),
                container.node_id,
                node_id=container.node_id,
            )
        for child in unsupported:
            self.reject(
                "unsupported_table_child",
                f"unsupported table child: {type(child).__name__}",
                child.pos,
                _node_end(child),
                container.node_id,
                node_id=container.node_id,
            )

    def process_tabular(
        self,
        node: LatexEnvironmentNode,
        parent_node_id: str | None,
    ) -> None:
        """Build a strict source-only table leaf or record an explicit rejection."""

        start = node.pos
        end = _node_end(node)
        if not self.enable_strict_tables:
            self.add_node(
                "table",
                start,
                end,
                parent_node_id,
                content=self._environment_body_span(node),
                environment_name=node.environmentname,
            )
            return
        try:
            table = parse_strict_table(
                self.source,
                start=start,
                end=end,
                source_id=self.source_id,
                reference_values=self.reference_values,
                math_macros=self.math_macros,
            )
        except TableAstError as exc:
            table_node = self.add_node(
                "table",
                start,
                end,
                parent_node_id,
                content=self._environment_body_span(node),
                environment_name=node.environmentname,
            )
            self.reject(
                "unsupported_table_serialization",
                str(exc),
                start,
                end,
                parent_node_id,
                node_id=table_node.node_id,
            )
            return
        self.add_node(
            "table",
            start,
            end,
            parent_node_id,
            content=self._environment_body_span(node),
            environment_name=node.environmentname,
            metadata=table.to_metadata(),
        )

    def freeze_nodes(self) -> tuple[DocumentBlockNode, ...]:
        return tuple(
            replace(state.node, child_node_ids=tuple(state.children)) for state in self.states
        )


def _find_document_environment(nodes: Sequence[Any]) -> LatexEnvironmentNode | None:
    matches = [
        node
        for node in nodes
        if isinstance(node, LatexEnvironmentNode) and node.environmentname == "document"
    ]
    return matches[0] if len(matches) == 1 else None


def _build_line_index(
    source: str,
    positions: _PositionIndex,
    nodes: Sequence[DocumentBlockNode],
    source_map: CanonicalSourceMap,
) -> tuple[GeneratedLineJoin, ...]:
    entries: list[GeneratedLineJoin] = []
    starts = positions.line_starts
    for line_offset, start in enumerate(starts):
        end = starts[line_offset + 1] if line_offset + 1 < len(starts) else len(source)
        # Empty final insertion line still has a useful zero-width join.
        ids = tuple(
            node.node_id
            for node in nodes
            if node.span.char_start < end and node.span.char_end > start
        )
        entries.append(
            GeneratedLineJoin(
                generated_line=line_offset + 1,
                canonical_span=positions.span(start, end),
                node_ids=ids,
                origins=source_map.origins_for_span(source, start, end),
            )
        )
    return tuple(entries)


def parse_document_ast(
    source: str,
    *,
    source_id: str = "<memory>",
    source_map: CanonicalSourceMap | None = None,
    reference_values: Mapping[str, str] | None = None,
    math_macros: Mapping[str, MathMacroDefinition] | None = None,
    semantic_environments: Mapping[str, SemanticEnvironmentDefinition] | None = None,
    list_environments: Mapping[str, SourceListEnvironmentDefinition] | None = None,
    external_verbatim_calls: Mapping[tuple[int, int], str] | None = None,
    enable_strict_tables: bool = False,
) -> DocumentAst:
    """Parse a canonical flattened LaTeX string into stable document blocks.

    Numbering is deliberately absent: heading, equation, table, footnote, and
    list numbers must later come from compiler-native metadata.  A parse can
    return safe nodes together with rejections; consumers must require
    ``document.accepted`` (or apply a documented rejection policy) before
    admitting page GT.  ``list_environments`` contains only source-defined
    aliases already proven by the caller to wrap a built-in ``itemize`` or
    ``enumerate`` environment; the node keeps its real environment name while
    list metadata uses the canonical built-in kind. ``external_verbatim_calls``
    maps exact canonical invocation spans to call IDs already proven from
    clean-executed local source files; only ordering/provenance enters this
    AST, while literal text and placement remain in the external trace.
    ``enable_strict_tables`` opts into the experimental fail-closed table
    serializer; the default preserves the frozen V3.35 table behavior.
    """

    if not isinstance(source, str):
        raise DocumentAstError("source must be a string")
    if not isinstance(source_id, str) or not source_id:
        raise DocumentAstError("source_id must be a non-empty string")
    try:
        source_bytes = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DocumentAstError("source is not valid UTF-8 text") from exc
    resolved_map = source_map or CanonicalSourceMap.identity(source, source_id)
    resolved_map.validate_for_source(source)
    resolved_external_calls: dict[tuple[int, int], str] = {}
    external_call_ids: set[str] = set()
    for span, call_id in (external_verbatim_calls or {}).items():
        if (
            not isinstance(span, tuple)
            or len(span) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
        ):
            raise DocumentAstError("external verbatim call spans must be integer pairs")
        start, end = span
        if start < 0 or end <= start or end > len(source):
            raise DocumentAstError("external verbatim call span is outside source")
        if not source.startswith(r"\verbatiminput", start):
            raise DocumentAstError(
                "external verbatim call span does not start at a literal invocation"
            )
        if not isinstance(call_id, str) or _EXTERNAL_VERBATIM_CALL_ID.fullmatch(call_id) is None:
            raise DocumentAstError("external verbatim call ID is unsafe")
        if call_id in external_call_ids:
            raise DocumentAstError("external verbatim call IDs must be unique")
        external_call_ids.add(call_id)
        resolved_external_calls[(start, end)] = call_id
    positions = _PositionIndex(source)
    builder = _DocumentBuilder(
        source,
        source_id,
        resolved_map,
        reference_values or {},
        math_macros or {},
        semantic_environments or {},
        list_environments or {},
        resolved_external_calls,
        enable_strict_tables,
    )
    parse_mode = "full_document"
    parse_diagnostics: list[str] = []
    try:
        walker = LatexWalker(
            source,
            latex_context=_make_latex_context(
                enable_strict_tables=enable_strict_tables
            ),
            tolerant_parsing=False,
        )
        root_nodes, _, _ = walker.get_latex_nodes(pos=0)
    except (LatexWalkerParseError, ValueError) as exc:
        full_parse_message = f"full canonical parse failed: {exc}"
        parse_diagnostics.append(full_parse_message)
        try:
            boundary = locate_document_body(source)
        except DocumentAstError as boundary_exc:
            builder.reject(
                "document_parse_error",
                f"{full_parse_message}; body recovery failed: {boundary_exc}",
                0,
                len(source),
                None,
            )
            empty_span = positions.span(0, len(source))
            return DocumentAst(
                version=DOCUMENT_AST_VERSION,
                source_id=source_id,
                source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                source=source,
                body_span=empty_span,
                nodes=(),
                rejections=tuple(builder.rejections),
                source_map=resolved_map,
                line_index=_build_line_index(source, positions, (), resolved_map),
                parse_mode="rejected",
                parse_diagnostics=tuple(parse_diagnostics),
            )

        parse_mode = "body_only_fallback"
        calls, frontmatter_failures = _scan_safe_frontmatter(
            source, boundary.begin_command_start
        )
        for start, end, message in frontmatter_failures:
            builder.reject(
                "unsafe_fallback_frontmatter",
                message,
                start,
                end,
                None,
            )
        for call in calls:
            builder.add_inline_block(
                "frontmatter",
                call.start,
                call.end,
                None,
                content=(call.content_start, call.content_end),
                command_name=call.name,
                optional_label=call.optional_label,
            )
        try:
            body_nodes, parsed_start, parsed_length = LatexWalker(
                source,
                latex_context=_make_latex_context(
                    enable_strict_tables=enable_strict_tables
                ),
                tolerant_parsing=False,
            ).get_latex_nodes(
                pos=boundary.body_start,
                stop_upon_end_environment="document",
            )
            body_nodes = tuple(body_nodes)
            if parsed_start != boundary.body_start:
                raise DocumentAstError("body parser changed the canonical start offset")
            if any(
                int(node.pos) < boundary.body_start or _node_end(node) > boundary.body_end
                for node in body_nodes
            ):
                raise DocumentAstError("body parser emitted a node outside document bounds")
            parsed_stop = parsed_start + parsed_length
            if parsed_stop != boundary.end_command_end:
                raise DocumentAstError(
                    "body parser did not stop at the recovered document end "
                    f"({parsed_stop} != {boundary.end_command_end})"
                )
        except (LatexWalkerParseError, ValueError, DocumentAstError) as body_exc:
            builder.reject(
                "body_only_parse_error",
                f"recovered document body could not be parsed: {body_exc}",
                boundary.body_start,
                boundary.body_end,
                None,
            )
            body_nodes = ()
        body_start, body_end = boundary.body_start, boundary.body_end
        if body_nodes:
            builder.process_sequence(body_nodes, body_start, body_end, None)
        root_nodes = None

    if root_nodes is not None:
        root_nodes = tuple(root_nodes)
        document_environment = _find_document_environment(root_nodes)
    else:
        document_environment = None
    if root_nodes is not None and document_environment is not None:
        # Frontmatter declarations often live in the preamble.  Parse only
        # explicitly supported declarations there; preamble definitions and
        # packages are not document content.
        for node in root_nodes:
            if node is document_environment:
                break
            if isinstance(node, LatexMacroNode) and node.macroname in _FRONTMATTER_COMMANDS:
                builder.process_structured_command(node, None, "frontmatter")
        body_nodes = tuple(document_environment.nodelist or ())
        if body_nodes:
            body_start = int(body_nodes[0].pos)
            body_end = _node_end(body_nodes[-1])
        else:
            body_start, body_end = builder._environment_body_span(document_environment)
        builder.process_sequence(body_nodes, body_start, body_end, None)
    elif root_nodes is not None:
        parse_mode = "fragment"
        document_matches = [
            node
            for node in root_nodes
            if isinstance(node, LatexEnvironmentNode) and node.environmentname == "document"
        ]
        if len(document_matches) > 1:
            builder.reject(
                "multiple_document_environments",
                "canonical source has multiple document environments",
                0,
                len(source),
                None,
            )
        body_start, body_end = 0, len(source)
        builder.process_sequence(root_nodes, body_start, body_end, None)

    for span, call_id in sorted(builder.external_verbatim_calls.items()):
        if span in builder.consumed_external_verbatim_calls:
            continue
        builder.reject(
            "unmatched_external_verbatim_call",
            f"proved external verbatim call was not consumed by the AST: {call_id}",
            span[0],
            span[1],
            None,
        )

    frozen_nodes = builder.freeze_nodes()
    return DocumentAst(
        version=DOCUMENT_AST_VERSION,
        source_id=source_id,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source=source,
        body_span=positions.span(body_start, body_end),
        nodes=frozen_nodes,
        rejections=tuple(builder.rejections),
        source_map=resolved_map,
        line_index=_build_line_index(source, positions, frozen_nodes, resolved_map),
        parse_mode=parse_mode,
        parse_diagnostics=tuple(parse_diagnostics),
    )


__all__ = [
    "DOCUMENT_AST_VERSION",
    "CanonicalSourceMap",
    "DocumentAst",
    "DocumentAstError",
    "DocumentAstRejection",
    "DocumentBlockNode",
    "DocumentBodyBoundary",
    "DocumentSourceSpan",
    "FlattenedSourceSegment",
    "GeneratedLineJoin",
    "OriginalSourceSpan",
    "locate_document_body",
    "parse_document_ast",
]
