"""Deterministic source AST/IR for compiler-native page ground truth.

The v3 representation is deliberately independent of the v2 parser.  It is
not a TeX interpreter and it never accepts PDF text as ground truth.  Instead,
it parses a small, explicitly supported inline-LaTeX language and records
every visible source atom with both Python-character and UTF-8-byte spans.

Unknown or malformed constructs are represented as explicit opaque atoms.
Comments are omitted because TeX does not typeset them.  Footnotes are strict
nodes: their source callout and body spans are recorded, but no marker number
or Markdown footnote syntax is invented.  A caller must inject the footnote
representation when reconstructing Markdown.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from typing import TypeAlias

IR_VERSION = "source_first_v3_ast_ir_v5"
OPAQUE_MARKER_PREFIX = "⟦opaque:"
OPAQUE_MARKER_SUFFIX = "⟧"

_STYLE_COMMANDS = {
    "textbf": "strong",
    "emph": "em",
    "textit": "em",
    "texttt": "code",
    "textsuperscript": "sup",
    # Small caps changes the visible glyph letters even though Markdown has
    # no corresponding delimiter.  Keep the compiler/source semantic in the
    # style stack so visible text can be serialized as printed rather than
    # silently preserving the mixed-case source spelling.
    "textsc": "smallcaps",
}

# Font-family/shape controls whose visible content is represented faithfully
# even though Markdown has no dedicated syntax for that exact TeX style.
_TRANSPARENT_TEXT_COMMANDS = frozenset(
    {
        "textmd",
        "textnormal",
        "textrm",
        "textup",
        "textsf",
        "mbox",
        "hbox",
    }
)
_SERIALIZABLE_STYLES = frozenset(
    {"strong", "em", "body_em", "code", "sup", "smallcaps"}
)
_FONT_STYLES = frozenset({"strong", "em", "body_em", "code", "smallcaps"})

# These are deliberately tiny allowlists.  Layout controls have no visible
# representation, while symbol commands have one class-independent Unicode
# rendering that the independent PDF verifier must still confirm.  Keep
# commands which consume an argument separate: their argument must first be
# proved to be a literal token sequence before the command can be omitted.
_INVISIBLE_LAYOUT_COMMANDS = frozenset(
    {
        "noindent",
        "xspace",
        # Font-size declarations change layout but do not themselves typeset
        # a glyph.  Shape/family declarations are intentionally not included:
        # their visible effect cannot be represented by this IR's style stack.
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
        "selectfont",
        # TeX grouping primitives are pure scope controls.  Braced groups
        # are already represented by ``group`` nodes below.
        "begingroup",
        "endgroup",
        "bgroup",
        "egroup",
    }
)
_INVISIBLE_LAYOUT_ARGUMENT_COMMANDS = frozenset({"vspace", "hspace"})
_TRANSPARENT_MACRO_ENVIRONMENTS = frozenset(
    {
        # LaTeX exposes font-size declarations as environments as well as
        # commands.  Inside a source-defined macro these wrappers contribute
        # no visible characters of their own, so they may be rewritten to
        # explicit grouping plus the already-supported declaration before the
        # expansion is parsed.  Keep this list identical to the proven
        # no-visible-text size controls above; arbitrary environments remain
        # opaque.
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
_VISIBLE_WHITESPACE_COMMANDS = frozenset({","})
_LEGACY_STYLE_DECLARATIONS = {
    "bf": "strong",
    "tt": "code",
    "em": "em",
    "it": "em",
}
_LITERAL_LAYOUT_ARGUMENT = re.compile(
    r"[A-Za-z0-9.!+\-*,/:;=() \t]+"
)
_LITERAL_DIMENSION = re.compile(
    r"[+\-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em)"
)
_LITERAL_URI = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*:[A-Za-z0-9._~:/?=+,@;!\-]+"
)
_LITERAL_COUNTER_NAME = re.compile(r"[A-Za-z@]+")
_LITERAL_COUNTER_VALUE = re.compile(r"[+\-]?\d+")
_VISIBLE_SYMBOL_COMMANDS = {
    "copyright": ("©", "©"),
    "textasciicircum": ("^", "^"),
    "textasciitilde": ("~", "~"),
    "texttildelow": ("~", "~"),
}

_LITERAL_ESCAPES = {
    "&": "&",
    "%": "%",
    "$": "$",
    "#": "#",
    "_": "_",
    "{": "{",
    "}": "}",
}

_MARKDOWN_SPECIAL = frozenset("\\`*_#$![]<>{}")


class SourceIrError(ValueError):
    """Base class for deterministic v3 source-IR failures."""


class FootnoteRendererRequired(SourceIrError):
    """Raised when selected footnote source has no caller representation."""


class InvalidAtomSelection(SourceIrError):
    """Raised for unknown or structurally incomplete selected atom IDs."""


@dataclasses.dataclass(frozen=True, slots=True)
class MathMacroDefinition:
    """One unambiguous source-defined macro admitted inside inline math.

    Definitions are collected from the LaTeX source tree, never inferred from
    PDF text.  Optional/default arguments are deliberately unsupported by the
    collector because reproducing TeX's argument scanner would otherwise be
    ambiguous.
    """

    name: str
    argument_count: int
    body: str


@dataclasses.dataclass(frozen=True, slots=True)
class SourceSpan:
    """Half-open absolute source span in characters and UTF-8 bytes."""

    char_start: int
    char_end: int
    byte_start: int
    byte_end: int

    def __post_init__(self) -> None:
        if min(self.char_start, self.char_end, self.byte_start, self.byte_end) < 0:
            raise SourceIrError("source span offsets must be non-negative")
        if self.char_end < self.char_start:
            raise SourceIrError("source span char_end precedes char_start")
        if self.byte_end < self.byte_start:
            raise SourceIrError("source span byte_end precedes byte_start")

    @property
    def char_span(self) -> tuple[int, int]:
        return self.char_start, self.char_end

    @property
    def byte_span(self) -> tuple[int, int]:
        return self.byte_start, self.byte_end


@dataclasses.dataclass(frozen=True, slots=True)
class SourceNode:
    """One deterministic structural or leaf node in source order."""

    node_id: str
    kind: str
    span: SourceSpan
    parent_node_id: str | None
    child_node_ids: tuple[str, ...]
    atom_ids: tuple[str, ...]
    content_span: SourceSpan | None = None
    style: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class FootnoteNode:
    """A strict source footnote invocation with no inferred marker number.

    ``callout_span`` covers the command prefix through the opening body brace;
    ``body_span`` covers only the source inside that brace pair.  An optional
    explicit TeX mark is exposed as ``explicit_mark_span`` but is never
    interpreted by this module.  ``command_name`` distinguishes ordinary
    ``\\footnote`` from template front-matter ``\\thanks``; both keep their
    marker semantics outside the source-only inline parser.
    """

    node_id: str
    kind: str
    span: SourceSpan
    parent_node_id: str | None
    child_node_ids: tuple[str, ...]
    atom_ids: tuple[str, ...]
    callout_span: SourceSpan
    body_span: SourceSpan
    callout_atom_id: str
    body_atom_ids: tuple[str, ...]
    explicit_mark_span: SourceSpan | None = None
    command_name: str = "footnote"


AstNode: TypeAlias = SourceNode | FootnoteNode


@dataclasses.dataclass(frozen=True, slots=True)
class SourceAtom:
    """One source-derived unit that a compiler trace may assign to a page."""

    atom_id: str
    ordinal: int
    node_id: str
    kind: str
    span: SourceSpan
    raw_source: str
    visible_text: str
    markdown_fragment: str
    style_stack: tuple[str, ...]
    footnote_path: tuple[str, ...]
    verifier_fragments: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self) -> None:
        for fragment in self.verifier_fragments:
            if (
                not isinstance(fragment, tuple)
                or len(fragment) != 2
                or not isinstance(fragment[0], str)
                or not isinstance(fragment[1], bool)
            ):
                raise SourceIrError(
                    "source atom verifier fragments are malformed"
                )

    @property
    def source_span(self) -> tuple[int, int]:
        return self.span.char_span

    @property
    def source_byte_span(self) -> tuple[int, int]:
        return self.span.byte_span

    @property
    def is_whitespace(self) -> bool:
        return self.kind == "whitespace"


@dataclasses.dataclass(frozen=True, slots=True)
class SourceDocumentIR:
    """Immutable AST/atom bundle for one exact source fragment."""

    version: str
    source_id: str
    source_sha256: str
    source_char_base: int
    source_byte_base: int
    root_node_id: str
    nodes: tuple[AstNode, ...]
    atoms: tuple[SourceAtom, ...]

    def __post_init__(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        atom_ids = [atom.atom_id for atom in self.atoms]
        if len(node_ids) != len(set(node_ids)):
            raise SourceIrError("duplicate node IDs in source IR")
        if len(atom_ids) != len(set(atom_ids)):
            raise SourceIrError("duplicate atom IDs in source IR")
        if self.root_node_id not in set(node_ids):
            raise SourceIrError("root node ID is absent from source IR")
        if [atom.ordinal for atom in self.atoms] != list(range(len(self.atoms))):
            raise SourceIrError("source atom ordinals are not contiguous")
        known_nodes = set(node_ids)
        known_atoms = set(atom_ids)
        atom_by_id = {atom.atom_id: atom for atom in self.atoms}
        for node in self.nodes:
            if node.parent_node_id is not None and node.parent_node_id not in known_nodes:
                raise SourceIrError(f"unknown parent node: {node.parent_node_id}")
            if not set(node.child_node_ids).issubset(known_nodes):
                raise SourceIrError(f"unknown child node on {node.node_id}")
            if not set(node.atom_ids).issubset(known_atoms):
                raise SourceIrError(f"unknown atom on {node.node_id}")
            if isinstance(node, FootnoteNode):
                if node.callout_atom_id not in known_atoms:
                    raise SourceIrError(f"unknown footnote callout on {node.node_id}")
                if not set(node.body_atom_ids).issubset(known_atoms):
                    raise SourceIrError(f"unknown footnote body atom on {node.node_id}")
                callout = atom_by_id[node.callout_atom_id]
                if callout.kind != "footnote_callout" or callout.node_id != node.node_id:
                    raise SourceIrError(f"invalid footnote callout on {node.node_id}")
                if callout.span != node.callout_span:
                    raise SourceIrError(f"footnote callout span mismatch on {node.node_id}")
                if node.callout_span.char_end != node.body_span.char_start:
                    raise SourceIrError(f"non-adjacent footnote source spans on {node.node_id}")
                if node.callout_span.byte_end != node.body_span.byte_start:
                    raise SourceIrError(f"non-adjacent footnote byte spans on {node.node_id}")
                for body_atom_id in node.body_atom_ids:
                    if node.node_id not in atom_by_id[body_atom_id].footnote_path:
                        raise SourceIrError(
                            f"footnote body path mismatch on {node.node_id}"
                        )
        for atom in self.atoms:
            if atom.node_id not in known_nodes:
                raise SourceIrError(f"atom {atom.atom_id} has unknown node")

    def get_node(self, node_id: str) -> AstNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def get_atom(self, atom_id: str) -> SourceAtom:
        for atom in self.atoms:
            if atom.atom_id == atom_id:
                return atom
        raise KeyError(atom_id)

    @property
    def footnotes(self) -> tuple[FootnoteNode, ...]:
        return tuple(node for node in self.nodes if isinstance(node, FootnoteNode))

    @property
    def opaque_atoms(self) -> tuple[SourceAtom, ...]:
        return tuple(atom for atom in self.atoms if atom.kind == "opaque")


@dataclasses.dataclass(frozen=True, slots=True)
class FootnoteRenderContext:
    """Strict source context passed to a caller-supplied footnote renderer."""

    document: SourceDocumentIR
    node: FootnoteNode
    callout_atom: SourceAtom
    selected_body_atom_ids: tuple[str, ...]
    body_markdown: str
    body_complete: bool


FootnoteRenderer: TypeAlias = Callable[[FootnoteRenderContext], str]


@dataclasses.dataclass(slots=True)
class _NodeState:
    node: AstNode
    children: list[str]
    atoms: list[str]


def _stable_id(
    prefix: str,
    source_id: str,
    kind: str,
    span: SourceSpan,
    raw_source: str,
    parent_node_id: str | None,
) -> str:
    payload = "\x1f".join(
        (
            IR_VERSION,
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
    return f"sfv3_{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _command_end(source: str, start: int, end: int) -> int:
    cursor = start + 1
    if cursor >= end:
        return cursor
    if source[cursor].isalpha() or source[cursor] == "@":
        cursor += 1
        while cursor < end and (source[cursor].isalpha() or source[cursor] == "@"):
            cursor += 1
        return cursor
    return cursor + 1


def _linebreak_end(source: str, start: int, end: int) -> int | None:
    r"""Return the complete span of one explicit LaTeX line-break command.

    LaTeX accepts both ``\\`` and the starred form ``\\*``.  Either form
    may carry an immediate optional vertical-space argument, for example
    ``\\[4pt]``.  Keeping the optional syntax in the same source atom is
    important: placing a compiler marker between ``\\`` and ``[4pt]`` would
    make the marker consume the optional argument and change the compiled
    layout.  An unterminated optional argument is returned as ``None`` so the
    caller can fail closed instead of pretending that the following source is
    ordinary prose.
    """

    command_end = start + 2
    if command_end > end or source[start:command_end] != r"\\":
        return None
    cursor = command_end
    if cursor < end and source[cursor] == "*":
        cursor += 1
    if cursor < end and source[cursor] == "[":
        optional_end = _find_balanced(source, cursor, end, "[", "]")
        if optional_end is None:
            return None
        cursor = optional_end
    return cursor


def _is_comment_start(source: str, index: int) -> bool:
    if source[index] != "%":
        return False
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and source[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 0


def _skip_comment(source: str, start: int, end: int) -> int:
    newline = source.find("\n", start, end)
    return end if newline < 0 else newline + 1


def _find_balanced(
    source: str,
    opening: int,
    end: int,
    open_char: str = "{",
    close_char: str = "}",
) -> int | None:
    """Return the exclusive end of one balanced group, respecting comments."""

    if opening >= end or source[opening] != open_char:
        return None
    depth = 0
    cursor = opening
    while cursor < end:
        if source[cursor] == "%" and _is_comment_start(source, cursor):
            cursor = _skip_comment(source, cursor, end)
            continue
        if source[cursor] == "\\":
            cursor = min(end, _command_end(source, cursor, end))
            continue
        if source[cursor] == open_char:
            depth += 1
        elif source[cursor] == close_char:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return None


def _strip_comments(value: str) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(value):
        if value[cursor] == "%" and _is_comment_start(value, cursor):
            cursor = _skip_comment(value, cursor, len(value))
            continue
        output.append(value[cursor])
        cursor += 1
    return "".join(output)


def _rewrite_transparent_macro_environments(value: str) -> str | None:
    r"""Rewrite balanced, literal font-size environments for macro parsing.

    This helper runs only on the source-defined replacement body of a macro.
    ``parse_source_ir`` already understands the corresponding size declaration
    and TeX grouping controls, but not block-environment syntax.  Rewriting
    ``\begin{small}...\end{small}`` to
    ``\begingroup\small...\endgroup`` therefore preserves the visible token
    sequence and scope without executing TeX.  Mismatched safe wrappers fail
    closed, and every environment outside the explicit allowlist is left
    untouched so the downstream parser will keep it opaque.
    """

    output: list[str] = []
    stack: list[str] = []
    cursor = 0
    pattern = re.compile(r"\\(begin|end)\s*\{([A-Za-z]+)\}")
    while cursor < len(value):
        if value[cursor] == "%" and _is_comment_start(value, cursor):
            comment_end = _skip_comment(value, cursor, len(value))
            output.append(value[cursor:comment_end])
            cursor = comment_end
            continue
        if value[cursor] != "\\":
            output.append(value[cursor])
            cursor += 1
            continue
        match = pattern.match(value, cursor)
        if match is None:
            command_end = _command_end(value, cursor, len(value))
            output.append(value[cursor:command_end])
            cursor = command_end
            continue
        direction, environment = match.groups()
        if environment not in _TRANSPARENT_MACRO_ENVIRONMENTS:
            output.append(match.group(0))
            cursor = match.end()
            continue
        if direction == "begin":
            stack.append(environment)
            output.append(r"\begingroup" + "\\" + environment)
        else:
            if not stack or stack[-1] != environment:
                return None
            stack.pop()
            output.append(r"\endgroup")
        cursor = match.end()
    if stack:
        return None
    return "".join(output)


def _normalize_math(body: str) -> str:
    sentinel = "\x00"
    protected = _strip_comments(body).replace("\\ ", sentinel)
    protected = protected.strip()
    return protected.replace(sentinel, "\\ ")


def _expand_math_macros(
    value: str,
    macros: Mapping[str, MathMacroDefinition],
    *,
    maximum_passes: int = 8,
) -> str:
    """Expand only complete, locally defined required-argument macros.

    This is a source serializer, not a TeX executor.  A macro is substituted
    only when its exact definition was admitted by the caller and every
    required argument is a complete balanced brace group.  Recursive or
    unstable definitions stop after a small deterministic pass bound and will
    subsequently fail the independent PDF verifier.
    """

    current = value
    command_pattern = re.compile(r"\\([A-Za-z@]+)")
    for _ in range(maximum_passes):
        pieces: list[str] = []
        cursor = 0
        changed = False
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
                group_end = _find_balanced(current, argument_cursor, len(current))
                if group_end is None:
                    valid = False
                    break
                arguments.append(current[argument_cursor + 1 : group_end - 1])
                argument_cursor = group_end
            if not valid:
                continue
            replacement = macro.body
            for argument_index, argument in enumerate(arguments, 1):
                replacement = replacement.replace(f"#{argument_index}", argument)
            pieces.append(current[cursor : match.start()])
            pieces.append(replacement)
            cursor = argument_cursor
            changed = True
        if not changed:
            break
        pieces.append(current[cursor:])
        updated = "".join(pieces)
        if updated == current:
            break
        current = updated
    # xspace controls following TeX spacing and has no mathematical glyph.
    return re.sub(r"\\xspace\b", "", current)


def _find_inline_math(source: str, start: int, end: int) -> tuple[int, str] | None:
    if source.startswith("\\(", start):
        cursor = start + 2
        while cursor < end:
            if source[cursor] == "%" and _is_comment_start(source, cursor):
                cursor = _skip_comment(source, cursor, end)
                continue
            if source.startswith("\\)", cursor):
                return cursor + 2, source[start + 2 : cursor]
            if source[cursor] == "\\":
                cursor = min(end, _command_end(source, cursor, end))
            else:
                cursor += 1
        return None
    if source[start] != "$" or source.startswith("$$", start):
        return None
    cursor = start + 1
    while cursor < end:
        if source[cursor] == "%" and _is_comment_start(source, cursor):
            cursor = _skip_comment(source, cursor, end)
            continue
        if source[cursor] == "\\":
            cursor = min(end, _command_end(source, cursor, end))
            continue
        if source[cursor] == "$":
            return cursor + 1, source[start + 1 : cursor]
        cursor += 1
    return None


def _decode_literal(raw: str, *, tex_text_ligatures: bool = True) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(raw):
        if raw[cursor] == "\\" and cursor + 1 < len(raw):
            escaped = raw[cursor + 1]
            if escaped in _LITERAL_ESCAPES:
                output.append(_LITERAL_ESCAPES[escaped])
                cursor += 2
                continue
        output.append(raw[cursor])
        cursor += 1
    value = "".join(output)
    if tex_text_ligatures:
        # Deterministic TeX text ligatures.  These are derived from the source
        # token stream and compiler rules; PDF text is not consulted.  Code
        # style deliberately opts out because typewriter fonts commonly keep
        # the literal ASCII characters.
        value = value.replace("---", "—").replace("--", "–")
        value = value.replace("``", "“").replace("''", "”")
        value = value.replace("`", "‘").replace("'", "’")
    return value


def _escape_markdown_literal(value: str, *, in_code: bool = False) -> str:
    if in_code:
        # ``texttt`` is emitted as an HTML code element, so only HTML syntax
        # characters need escaping and Markdown punctuation remains literal.
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    output: list[str] = []
    for char in value:
        if char in _MARKDOWN_SPECIAL:
            output.append("\\")
        output.append(char)
    return "".join(output)


def _is_literal_layout_argument(value: str) -> bool:
    """Return whether an omitted layout argument is a safe literal.

    ``vspace``/``hspace`` lengths and ``color`` names are consumed by TeX;
    they do not typeset their argument.  That is only safe to rely on when
    the complete balanced argument contains no control sequence, comment,
    nested group, or other TeX-special token.  A small printable whitelist is
    deliberately used instead of trying to evaluate dimensions or xcolor
    expressions.  Macro-bearing forms such as ``{\\baselineskip}`` therefore
    remain opaque and fail closed.
    """

    return bool(value and value.strip() and _LITERAL_LAYOUT_ARGUMENT.fullmatch(value))


class _Parser:
    def __init__(
        self,
        source: str,
        source_id: str,
        source_char_base: int,
        source_byte_base: int,
        reference_values: Mapping[str, str],
        math_macros: Mapping[str, MathMacroDefinition],
        macro_expansion_stack: tuple[str, ...] = (),
    ) -> None:
        self.source = source
        self.source_id = source_id
        self.char_base = source_char_base
        self.byte_base = source_byte_base
        self.reference_values = reference_values
        self.math_macros = math_macros
        self.macro_expansion_stack = macro_expansion_stack
        self.byte_prefix = [0]
        for char in source:
            self.byte_prefix.append(self.byte_prefix[-1] + len(char.encode("utf-8")))
        self.node_states: list[_NodeState] = []
        self.node_state_by_id: dict[str, _NodeState] = {}
        self.atoms: list[SourceAtom] = []

    def span(self, start: int, end: int) -> SourceSpan:
        return SourceSpan(
            char_start=self.char_base + start,
            char_end=self.char_base + end,
            byte_start=self.byte_base + self.byte_prefix[start],
            byte_end=self.byte_base + self.byte_prefix[end],
        )

    def add_node(
        self,
        kind: str,
        start: int,
        end: int,
        parent_node_id: str | None,
        *,
        content_span: SourceSpan | None = None,
        style: str | None = None,
    ) -> SourceNode:
        span = self.span(start, end)
        node = SourceNode(
            node_id=_stable_id(
                "node",
                self.source_id,
                kind,
                span,
                self.source[start:end],
                parent_node_id,
            ),
            kind=kind,
            span=span,
            parent_node_id=parent_node_id,
            child_node_ids=(),
            atom_ids=(),
            content_span=content_span,
            style=style,
        )
        state = _NodeState(node=node, children=[], atoms=[])
        self.node_states.append(state)
        self.node_state_by_id[node.node_id] = state
        if parent_node_id is not None:
            self.node_state_by_id[parent_node_id].children.append(node.node_id)
        return node

    def add_footnote_node(
        self,
        start: int,
        end: int,
        parent_node_id: str,
        callout_span: SourceSpan,
        body_span: SourceSpan,
        explicit_mark_span: SourceSpan | None,
        command_name: str,
    ) -> FootnoteNode:
        span = self.span(start, end)
        node = FootnoteNode(
            node_id=_stable_id(
                "node",
                self.source_id,
                command_name,
                span,
                self.source[start:end],
                parent_node_id,
            ),
            kind="footnote",
            span=span,
            parent_node_id=parent_node_id,
            child_node_ids=(),
            atom_ids=(),
            callout_span=callout_span,
            body_span=body_span,
            callout_atom_id="",
            body_atom_ids=(),
            explicit_mark_span=explicit_mark_span,
            command_name=command_name,
        )
        state = _NodeState(node=node, children=[], atoms=[])
        self.node_states.append(state)
        self.node_state_by_id[node.node_id] = state
        self.node_state_by_id[parent_node_id].children.append(node.node_id)
        return node

    def add_atom(
        self,
        node_id: str,
        kind: str,
        start: int,
        end: int,
        raw_source: str,
        visible_text: str,
        markdown_fragment: str,
        style_stack: tuple[str, ...],
        footnote_path: tuple[str, ...],
        verifier_fragments: tuple[tuple[str, bool], ...] = (),
    ) -> SourceAtom:
        span = self.span(start, end)
        atom = SourceAtom(
            atom_id=_stable_id(
                "atom",
                self.source_id,
                kind + ":" + "/".join(style_stack + footnote_path),
                span,
                raw_source,
                node_id,
            ),
            ordinal=len(self.atoms),
            node_id=node_id,
            kind=kind,
            span=span,
            raw_source=raw_source,
            visible_text=visible_text,
            markdown_fragment=markdown_fragment,
            style_stack=style_stack,
            footnote_path=footnote_path,
            verifier_fragments=verifier_fragments,
        )
        self.atoms.append(atom)
        self.node_state_by_id[node_id].atoms.append(atom.atom_id)
        return atom

    def emit_plain(
        self,
        start: int,
        end: int,
        parent_node_id: str,
        style_stack: tuple[str, ...],
        footnote_path: tuple[str, ...],
    ) -> None:
        raw = self.source[start:end]
        for match in re.finditer(r"\s+|\S+", raw, flags=re.DOTALL):
            local_start = start + match.start()
            local_end = start + match.end()
            piece = match.group(0)
            kind = "whitespace" if piece.isspace() else "text"
            visible = (
                " "
                if kind == "whitespace"
                else _decode_literal(
                    piece,
                    tex_text_ligatures="code" not in style_stack,
                )
            )
            if kind != "whitespace" and "smallcaps" in style_stack:
                visible = visible.upper()
            fragment = (
                " "
                if kind == "whitespace"
                else _escape_markdown_literal(visible, in_code="code" in style_stack)
            )
            node = self.add_node(kind, local_start, local_end, parent_node_id)
            self.add_atom(
                node.node_id,
                kind,
                local_start,
                local_end,
                piece,
                visible,
                fragment,
                style_stack,
                footnote_path,
            )

    def emit_whitespace(
        self,
        start: int,
        end: int,
        parent_node_id: str,
        style_stack: tuple[str, ...],
        footnote_path: tuple[str, ...],
    ) -> None:
        node = self.add_node("whitespace", start, end, parent_node_id)
        self.add_atom(
            node.node_id,
            "whitespace",
            start,
            end,
            self.source[start:end],
            " ",
            " ",
            style_stack,
            footnote_path,
        )

    def emit_opaque(
        self,
        start: int,
        end: int,
        parent_node_id: str,
        style_stack: tuple[str, ...],
        footnote_path: tuple[str, ...],
    ) -> None:
        raw = self.source[start:end]
        marker = (
            OPAQUE_MARKER_PREFIX
            + _escape_markdown_literal(raw, in_code="code" in style_stack)
            + OPAQUE_MARKER_SUFFIX
        )
        node = self.add_node("opaque", start, end, parent_node_id)
        self.add_atom(
            node.node_id,
            "opaque",
            start,
            end,
            raw,
            marker,
            marker,
            style_stack,
            footnote_path,
        )

    def skip_ignored(self, start: int, end: int) -> int:
        cursor = start
        while cursor < end:
            if self.source[cursor].isspace():
                cursor += 1
                continue
            if self.source[cursor] == "%" and _is_comment_start(self.source, cursor):
                cursor = _skip_comment(self.source, cursor, end)
                continue
            break
        return cursor

    def required_group_after(self, start: int, end: int) -> tuple[int, int] | None:
        opening = self.skip_ignored(start, end)
        if opening >= end or self.source[opening] != "{":
            return None
        group_end = _find_balanced(self.source, opening, end)
        if group_end is None:
            return opening, end + 1
        return opening, group_end

    def control_word_resume(self, command_end: int, end: int) -> int:
        """Return the first token after TeX's ignored control-word delimiter.

        The parser normally consumes inter-token whitespace together with a
        mandatory argument.  No-argument control words need the same lexical
        treatment explicitly: the blanks/comments that terminate ``\\small``
        or ``\\copyright`` are not character tokens and must never leak into
        Markdown.  Control symbols such as ``\\,`` do not call this helper.
        """

        return self.skip_ignored(command_end, end)

    @staticmethod
    def legacy_style_stack(
        style_stack: tuple[str, ...],
        style: str,
        *,
        toggle_emphasis: bool = False,
    ) -> tuple[str, ...]:
        """Apply one explicitly scoped legacy font declaration.

        LaTeX's legacy ``\\em`` declaration toggles emphasis when nested;
        ``\\bf`` selects bold, ``\\it`` selects italics, and ``\\tt`` selects
        a code-like typewriter family.  This helper is used only for a
        declaration that is the first non-ignored token of a balanced group,
        so the style can never escape its proven scope.
        """

        if toggle_emphasis and style == "em":
            for active_style in ("em", "body_em"):
                if active_style in style_stack:
                    index = len(style_stack) - 1 - style_stack[::-1].index(
                        active_style
                    )
                    return style_stack[:index] + style_stack[index + 1 :]
        if style == "em" and "body_em" in style_stack:
            return style_stack
        if style_stack and style_stack[-1] == style:
            return style_stack
        return style_stack + (style,)

    @staticmethod
    def transparent_style_stack(
        style_stack: tuple[str, ...],
        command: str,
    ) -> tuple[str, ...]:
        """Apply the representable reset semantics of one text command."""

        removed: frozenset[str]
        if command == "textnormal":
            removed = _FONT_STYLES
        elif command == "textmd":
            removed = frozenset({"strong"})
        elif command == "textup":
            removed = frozenset({"em", "body_em"})
        elif command in {"textrm", "textsf"}:
            removed = frozenset({"code"})
        else:
            removed = frozenset()
        return tuple(style for style in style_stack if style not in removed)

    def parse_braced_content(
        self,
        opening: int,
        group_end: int,
        parent_node_id: str,
        style_stack: tuple[str, ...],
        footnote_path: tuple[str, ...],
    ) -> None:
        """Parse one balanced group's body, including scoped legacy styles."""

        content_start = opening + 1
        content_end = group_end - 1
        declaration_start = self.skip_ignored(content_start, content_end)
        if declaration_start < content_end and self.source[declaration_start] == "\\":
            declaration_end = _command_end(
                self.source, declaration_start, content_end
            )
            declaration = self.source[declaration_start + 1 : declaration_end]
            style = _LEGACY_STYLE_DECLARATIONS.get(declaration)
            if style is not None and declaration not in self.math_macros:
                # Whitespace before the declaration is source content and is
                # parsed normally.  Whitespace after a control word is only a
                # TeX token delimiter and is therefore intentionally omitted.
                if declaration_start > content_start:
                    self.parse_sequence(
                        content_start,
                        declaration_start,
                        parent_node_id,
                        style_stack,
                        footnote_path,
                    )
                body_start = self.control_word_resume(
                    declaration_end, content_end
                )
                node = self.add_node(
                    "legacy_style",
                    declaration_start,
                    content_end,
                    parent_node_id,
                    content_span=self.span(body_start, content_end),
                    style=style,
                )
                self.parse_sequence(
                    body_start,
                    content_end,
                    node.node_id,
                    self.legacy_style_stack(
                        style_stack,
                        style,
                        toggle_emphasis=declaration == "em",
                    ),
                    footnote_path,
                )
                return
        self.parse_sequence(
            content_start,
            content_end,
            parent_node_id,
            style_stack,
            footnote_path,
        )

    def consume_unknown(self, command_end: int, end: int) -> int:
        opaque_end = command_end
        cursor = command_end
        while cursor < end:
            probe = self.skip_ignored(cursor, end)
            if probe >= end or self.source[probe] not in "[{":
                break
            opening = self.source[probe]
            close = "]" if opening == "[" else "}"
            group_end = _find_balanced(self.source, probe, end, opening, close)
            if group_end is None:
                return end
            opaque_end = group_end
            cursor = group_end
        return opaque_end

    def footnote_group_after(
        self,
        command_end: int,
        end: int,
    ) -> tuple[int, int, tuple[int, int] | None] | None:
        cursor = self.skip_ignored(command_end, end)
        explicit_mark: tuple[int, int] | None = None
        if cursor < end and self.source[cursor] == "[":
            optional_end = _find_balanced(self.source, cursor, end, "[", "]")
            if optional_end is None:
                return None
            explicit_mark = (cursor + 1, optional_end - 1)
            cursor = self.skip_ignored(optional_end, end)
        if cursor >= end or self.source[cursor] != "{":
            return None
        group_end = _find_balanced(self.source, cursor, end)
        if group_end is None:
            return None
        return cursor, group_end, explicit_mark

    def source_macro_after(
        self,
        command: str,
        command_end: int,
        end: int,
    ) -> tuple[int, str] | None:
        """Return invocation end and source-expanded body for one safe macro."""

        macro = self.math_macros.get(command)
        if macro is None or command in self.macro_expansion_stack:
            return None
        argument_cursor = command_end
        arguments: list[str] = []
        for _argument_index in range(macro.argument_count):
            opening = self.skip_ignored(argument_cursor, end)
            group_end = _find_balanced(self.source, opening, end)
            if group_end is None:
                return None
            arguments.append(self.source[opening + 1 : group_end - 1])
            argument_cursor = group_end
        replacement = macro.body
        for argument_index, argument in enumerate(arguments, 1):
            replacement = replacement.replace(f"#{argument_index}", argument)
        replacement = re.sub(r"\\xspace\b", "", replacement)
        return argument_cursor, replacement

    def render_source_macro(
        self,
        command: str,
        replacement: str,
    ) -> tuple[str, str, tuple[tuple[str, bool], ...]] | None:
        """Serialize one expansion from source definitions only.

        The complete expansion remains one atom tied to the original macro
        invocation span.  This preserves compiler localization while allowing
        a definition such as ``\\DAL -> \\textmd{DA-CR}`` to contribute its
        actual source-defined visible text.
        """

        replacement = _rewrite_transparent_macro_environments(replacement)
        if replacement is None:
            return None

        nested = parse_source_ir(
            replacement,
            source_id=f"{self.source_id}::macro:{command}",
            reference_values=self.reference_values,
            math_macros=self.math_macros,
            _macro_expansion_stack=self.macro_expansion_stack + (command,),
        )
        if nested.opaque_atoms or nested.footnotes:
            return None
        markdown = reconstruct_markdown(nested)
        visible = "".join(atom.visible_text for atom in nested.atoms)
        verifier_fragments: list[tuple[str, bool]] = []

        def append_verifier_fragment(text: str, verifiable: bool) -> None:
            if not text:
                return
            if verifier_fragments and verifier_fragments[-1][1] == verifiable:
                prior, _prior_verifiable = verifier_fragments[-1]
                verifier_fragments[-1] = (prior + text, verifiable)
            else:
                verifier_fragments.append((text, verifiable))

        for atom in nested.atoms:
            if atom.kind == "source_macro" and atom.verifier_fragments:
                for text, verifiable in atom.verifier_fragments:
                    append_verifier_fragment(text, verifiable)
            else:
                append_verifier_fragment(
                    atom.visible_text,
                    atom.kind in {"text", "reference", "whitespace"},
                )
        return visible, markdown, tuple(verifier_fragments)

    def parse_sequence(
        self,
        start: int,
        end: int,
        parent_node_id: str,
        style_stack: tuple[str, ...],
        footnote_path: tuple[str, ...],
    ) -> None:
        cursor = start
        text_start = start

        def flush(until: int) -> None:
            nonlocal text_start
            if until > text_start:
                self.emit_plain(
                    text_start,
                    until,
                    parent_node_id,
                    style_stack,
                    footnote_path,
                )
            text_start = until

        while cursor < end:
            char = self.source[cursor]

            if char == "%" and _is_comment_start(self.source, cursor):
                flush(cursor)
                cursor = _skip_comment(self.source, cursor, end)
                text_start = cursor
                continue

            if char == "~":
                flush(cursor)
                self.emit_whitespace(
                    cursor,
                    cursor + 1,
                    parent_node_id,
                    style_stack,
                    footnote_path,
                )
                cursor += 1
                text_start = cursor
                continue

            if self.source.startswith("$$", cursor):
                flush(cursor)
                close = self.source.find("$$", cursor + 2, end)
                opaque_end = end if close < 0 else close + 2
                self.emit_opaque(
                    cursor,
                    opaque_end,
                    parent_node_id,
                    style_stack,
                    footnote_path,
                )
                cursor = opaque_end
                text_start = cursor
                continue

            math = _find_inline_math(self.source, cursor, end)
            if math is not None:
                math_end, body = math
                flush(cursor)
                node = self.add_node(
                    "math",
                    cursor,
                    math_end,
                    parent_node_id,
                    content_span=self.span(
                        cursor + (2 if self.source.startswith("\\(", cursor) else 1),
                        math_end - (2 if self.source.startswith("\\(", cursor) else 1),
                    ),
                )
                normalized = _expand_math_macros(
                    _normalize_math(body),
                    self.math_macros,
                )
                self.add_atom(
                    node.node_id,
                    "math",
                    cursor,
                    math_end,
                    self.source[cursor:math_end],
                    normalized,
                    f"${normalized}$",
                    style_stack,
                    footnote_path,
                )
                cursor = math_end
                text_start = cursor
                continue

            if self.source.startswith("\\(", cursor):
                # An unterminated inline-math opener makes the remaining
                # fragment ambiguous.  Keep the complete tail opaque instead
                # of leaking formula tokens into ordinary prose atoms.
                flush(cursor)
                self.emit_opaque(
                    cursor,
                    end,
                    parent_node_id,
                    style_stack,
                    footnote_path,
                )
                return

            if char == "{":
                group_end = _find_balanced(self.source, cursor, end)
                flush(cursor)
                if group_end is None:
                    self.emit_opaque(
                        cursor,
                        end,
                        parent_node_id,
                        style_stack,
                        footnote_path,
                    )
                    return
                node = self.add_node(
                    "group",
                    cursor,
                    group_end,
                    parent_node_id,
                    content_span=self.span(cursor + 1, group_end - 1),
                )
                self.parse_braced_content(
                    cursor,
                    group_end,
                    node.node_id,
                    style_stack,
                    footnote_path,
                )
                cursor = group_end
                text_start = cursor
                continue

            if char == "}":
                flush(cursor)
                self.emit_opaque(
                    cursor,
                    cursor + 1,
                    parent_node_id,
                    style_stack,
                    footnote_path,
                )
                cursor += 1
                text_start = cursor
                continue

            if char == "\\":
                if self.source.startswith(r"\\", cursor):
                    linebreak_end = _linebreak_end(self.source, cursor, end)
                    flush(cursor)
                    if linebreak_end is None:
                        # A malformed optional spacing argument makes the
                        # remainder ambiguous.  Keep it opaque rather than
                        # leaking the argument body into Markdown GT.
                        self.emit_opaque(
                            cursor,
                            end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        return
                    node = self.add_node(
                        "linebreak",
                        cursor,
                        linebreak_end,
                        parent_node_id,
                    )
                    # ``native_trace_plan`` intentionally admits the stable
                    # leaf kinds ``text``, ``math`` and ``reference``.  Keep
                    # this visible control as a text atom while recording its
                    # structural node kind separately; this lets the normal
                    # compiler tracer wrap the complete command without
                    # widening its control-command allowlist.
                    self.add_atom(
                        node.node_id,
                        "text",
                        cursor,
                        linebreak_end,
                        self.source[cursor:linebreak_end],
                        "\n",
                        "<br>\n",
                        style_stack,
                        footnote_path,
                    )
                    cursor = linebreak_end
                    text_start = cursor
                    continue
                command_end = _command_end(self.source, cursor, end)
                command = self.source[cursor + 1 : command_end]

                if command in _LITERAL_ESCAPES:
                    cursor = command_end
                    continue

                if len(command) == 1 and command.isspace():
                    flush(cursor)
                    self.emit_whitespace(
                        cursor,
                        command_end,
                        parent_node_id,
                        style_stack,
                        footnote_path,
                    )
                    cursor = command_end
                    text_start = cursor
                    continue

                if command in _VISIBLE_WHITESPACE_COMMANDS:
                    flush(cursor)
                    self.emit_whitespace(
                        cursor,
                        command_end,
                        parent_node_id,
                        style_stack,
                        footnote_path,
                    )
                    cursor = command_end
                    text_start = cursor
                    continue

                if command == "ensuremath" and command not in self.math_macros:
                    # LaTeX's \ensuremath prints its mandatory argument in
                    # math mode regardless of the surrounding mode.  The
                    # complete balanced invocation is one compiler-traceable
                    # source atom; no PDF text or macro execution is needed.
                    group = self.required_group_after(command_end, end)
                    flush(cursor)
                    if group is None or group[1] > end:
                        opaque_end = self.consume_unknown(command_end, end)
                        if opaque_end <= command_end:
                            opaque_end = end
                        self.emit_opaque(
                            cursor,
                            opaque_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = opaque_end
                        text_start = cursor
                        continue
                    opening, group_end = group
                    body = self.source[opening + 1 : group_end - 1]
                    normalized = _expand_math_macros(
                        _normalize_math(body),
                        self.math_macros,
                    )
                    if not normalized:
                        self.emit_opaque(
                            cursor,
                            group_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = group_end
                        text_start = cursor
                        continue
                    node = self.add_node(
                        "math",
                        cursor,
                        group_end,
                        parent_node_id,
                        content_span=self.span(opening + 1, group_end - 1),
                    )
                    self.add_atom(
                        node.node_id,
                        "math",
                        cursor,
                        group_end,
                        self.source[cursor:group_end],
                        normalized,
                        f"${normalized}$",
                        style_stack,
                        footnote_path,
                    )
                    cursor = group_end
                    text_start = cursor
                    continue

                if (
                    command in _INVISIBLE_LAYOUT_COMMANDS
                    and command not in self.math_macros
                ):
                    flush(cursor)
                    self.add_node(
                        "layout_control",
                        cursor,
                        command_end,
                        parent_node_id,
                    )
                    cursor = self.control_word_resume(command_end, end)
                    text_start = cursor
                    continue

                if (
                    (
                        command in _INVISIBLE_LAYOUT_ARGUMENT_COMMANDS
                        or command == "color"
                    )
                    and command not in self.math_macros
                ):
                    # ``vspace`` and ``hspace`` have an immediate starred
                    # variant.  TeX ignores spaces/comments after a control
                    # word, so recognize that spelling without treating the
                    # star as visible source text.
                    argument_cursor = self.skip_ignored(command_end, end)
                    starred = (
                        command in _INVISIBLE_LAYOUT_ARGUMENT_COMMANDS
                        and argument_cursor < end
                        and self.source[argument_cursor] == "*"
                    )
                    if starred:
                        argument_cursor += 1
                    group = self.required_group_after(argument_cursor, end)
                    flush(cursor)
                    if group is None:
                        # A missing argument is ambiguous: the following
                        # source may be the token TeX would consume.  Keep the
                        # complete remainder opaque instead of leaking it as
                        # ordinary Markdown text.
                        opaque_end = self.consume_unknown(
                            argument_cursor if starred else command_end,
                            end,
                        )
                        if opaque_end <= command_end:
                            opaque_end = end
                        self.emit_opaque(
                            cursor,
                            opaque_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = opaque_end
                        text_start = cursor
                        continue
                    opening, group_end = group
                    literal = (
                        ""
                        if group_end > end
                        else self.source[opening + 1 : group_end - 1]
                    )
                    if group_end > end or not _is_literal_layout_argument(literal):
                        # Preserve the full balanced invocation in one
                        # rejection.  In particular, never parse a macro or
                        # nested group from a parameter that we declined to
                        # prove literal.
                        opaque_end = self.consume_unknown(
                            argument_cursor if starred else command_end,
                            end,
                        )
                        if opaque_end <= command_end:
                            opaque_end = end
                        self.emit_opaque(
                            cursor,
                            opaque_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = opaque_end
                        text_start = cursor
                        continue
                    self.add_node(
                        "layout_control",
                        cursor,
                        group_end,
                        parent_node_id,
                    )
                    cursor = group_end
                    text_start = cursor
                    continue

                if command == "setcounter" and command not in self.math_macros:
                    # ``\setcounter`` changes later compiler-resolved numbering
                    # but does not itself typeset a glyph.  Admit only the exact
                    # two-argument literal form.  Macros, comments between
                    # arguments, nested groups, arithmetic, optional arguments,
                    # and malformed invocations remain opaque and fail closed.
                    first_open = command_end
                    while first_open < end and self.source[first_open].isspace():
                        first_open += 1
                    first_end = (
                        _find_balanced(self.source, first_open, end)
                        if first_open < end and self.source[first_open] == "{"
                        else None
                    )
                    second_open = first_end if first_end is not None else end
                    while second_open < end and self.source[second_open].isspace():
                        second_open += 1
                    second_end = (
                        _find_balanced(self.source, second_open, end)
                        if second_open < end and self.source[second_open] == "{"
                        else None
                    )
                    counter_name = (
                        self.source[first_open + 1 : first_end - 1]
                        if first_end is not None
                        else ""
                    )
                    counter_value = (
                        self.source[second_open + 1 : second_end - 1]
                        if second_end is not None
                        else ""
                    )
                    if (
                        first_end is None
                        or second_end is None
                        or not _LITERAL_COUNTER_NAME.fullmatch(counter_name)
                        or not _LITERAL_COUNTER_VALUE.fullmatch(counter_value)
                    ):
                        flush(cursor)
                        opaque_end = self.consume_unknown(command_end, end)
                        if opaque_end <= command_end:
                            opaque_end = end
                        self.emit_opaque(
                            cursor,
                            opaque_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = opaque_end
                        text_start = cursor
                        continue
                    flush(cursor)
                    self.add_node(
                        "layout_control",
                        cursor,
                        second_end,
                        parent_node_id,
                    )
                    cursor = second_end
                    text_start = cursor
                    continue

                if (
                    command in _VISIBLE_SYMBOL_COMMANDS
                    and command not in self.math_macros
                ):
                    flush(cursor)
                    visible, markdown = _VISIBLE_SYMBOL_COMMANDS[command]
                    node = self.add_node(
                        "symbol",
                        cursor,
                        command_end,
                        parent_node_id,
                    )
                    self.add_atom(
                        node.node_id,
                        "symbol",
                        cursor,
                        command_end,
                        self.source[cursor:command_end],
                        visible,
                        markdown,
                        style_stack,
                        footnote_path,
                    )
                    cursor = self.control_word_resume(command_end, end)
                    text_start = cursor
                    continue

                if command == "raisebox" and command not in self.math_macros:
                    # Admit only the two mandatory-argument form.  Optional
                    # height/depth overrides or macro-valued dimensions are
                    # deliberately opaque because reproducing TeX's box
                    # metrics would otherwise be ambiguous.
                    lift_group = self.required_group_after(command_end, end)
                    lift_end = None if lift_group is None else lift_group[1]
                    content_probe = (
                        end
                        if lift_end is None or lift_end > end
                        else self.skip_ignored(lift_end, end)
                    )
                    content_group = (
                        None
                        if content_probe < end
                        and self.source[content_probe] == "["
                        else self.required_group_after(content_probe, end)
                    )
                    lift = (
                        ""
                        if lift_group is None or lift_group[1] > end
                        else self.source[lift_group[0] + 1 : lift_group[1] - 1]
                    )
                    if (
                        lift_group is None
                        or lift_group[1] > end
                        or content_group is None
                        or content_group[1] > end
                        or _LITERAL_DIMENSION.fullmatch(lift.strip()) is None
                    ):
                        flush(cursor)
                        opaque_end = self.consume_unknown(command_end, end)
                        if opaque_end <= command_end:
                            opaque_end = end
                        self.emit_opaque(
                            cursor,
                            opaque_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = opaque_end
                        text_start = cursor
                        continue
                    opening, group_end = content_group
                    flush(cursor)
                    node = self.add_node(
                        "transparent_text_style",
                        cursor,
                        group_end,
                        parent_node_id,
                        content_span=self.span(opening + 1, group_end - 1),
                    )
                    self.parse_braced_content(
                        opening,
                        group_end,
                        node.node_id,
                        style_stack,
                        footnote_path,
                    )
                    cursor = group_end
                    text_start = cursor
                    continue

                if command == "href" and command not in self.math_macros:
                    # Hyperref prints only its second mandatory argument.  A
                    # literal URI proves that the first argument contributes
                    # no source-derived visible text; Markdown intentionally
                    # keeps only the visible body so page slices stay balanced.
                    uri_group = self.required_group_after(command_end, end)
                    uri_end = None if uri_group is None else uri_group[1]
                    visible_probe = (
                        end
                        if uri_end is None or uri_end > end
                        else self.skip_ignored(uri_end, end)
                    )
                    visible_group = self.required_group_after(visible_probe, end)
                    uri = (
                        ""
                        if uri_group is None or uri_group[1] > end
                        else self.source[uri_group[0] + 1 : uri_group[1] - 1]
                    )
                    if (
                        uri_group is None
                        or uri_group[1] > end
                        or visible_group is None
                        or visible_group[1] > end
                        or _LITERAL_URI.fullmatch(uri) is None
                    ):
                        flush(cursor)
                        opaque_end = self.consume_unknown(command_end, end)
                        if opaque_end <= command_end:
                            opaque_end = end
                        self.emit_opaque(
                            cursor,
                            opaque_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = opaque_end
                        text_start = cursor
                        continue
                    opening, group_end = visible_group
                    flush(cursor)
                    node = self.add_node(
                        "hyperlink_visible_text",
                        cursor,
                        group_end,
                        parent_node_id,
                        content_span=self.span(opening + 1, group_end - 1),
                    )
                    self.parse_braced_content(
                        opening,
                        group_end,
                        node.node_id,
                        style_stack,
                        footnote_path,
                    )
                    cursor = group_end
                    text_start = cursor
                    continue

                if command in _STYLE_COMMANDS:
                    group = self.required_group_after(command_end, end)
                    flush(cursor)
                    if group is None:
                        self.emit_opaque(
                            cursor,
                            command_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = command_end
                        text_start = cursor
                        continue
                    opening, group_end = group
                    if group_end > end:
                        self.emit_opaque(
                            cursor,
                            end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        return
                    style = _STYLE_COMMANDS[command]
                    if command == "emph" and "body_em" in style_stack:
                        nested_style_stack = self.legacy_style_stack(
                            style_stack,
                            style,
                            toggle_emphasis=True,
                        )
                    elif style == "em" and "body_em" in style_stack:
                        nested_style_stack = style_stack
                    else:
                        nested_style_stack = (
                            style_stack
                            if style_stack and style_stack[-1] == style
                            else style_stack + (style,)
                        )
                    node = self.add_node(
                        "style",
                        cursor,
                        group_end,
                        parent_node_id,
                        content_span=self.span(opening + 1, group_end - 1),
                        style=style,
                    )
                    self.parse_sequence(
                        opening + 1,
                        group_end - 1,
                        node.node_id,
                        # Repeating one Markdown delimiter would turn nested
                        # italics into accidental bold (``**``).  Adjacent
                        # equivalent text styles are idempotent in this
                        # conservative IR, so retain one active delimiter.
                        nested_style_stack,
                        footnote_path,
                    )
                    cursor = group_end
                    text_start = cursor
                    continue

                if command in _TRANSPARENT_TEXT_COMMANDS:
                    group = self.required_group_after(command_end, end)
                    flush(cursor)
                    if group is None:
                        self.emit_opaque(
                            cursor,
                            command_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = command_end
                        text_start = cursor
                        continue
                    opening, group_end = group
                    if group_end > end:
                        self.emit_opaque(
                            cursor,
                            end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        return
                    node = self.add_node(
                        "transparent_text_style",
                        cursor,
                        group_end,
                        parent_node_id,
                        content_span=self.span(opening + 1, group_end - 1),
                    )
                    self.parse_sequence(
                        opening + 1,
                        group_end - 1,
                        node.node_id,
                        self.transparent_style_stack(style_stack, command),
                        footnote_path,
                    )
                    cursor = group_end
                    text_start = cursor
                    continue

                if command in {"ref", "eqref"}:
                    group = self.required_group_after(command_end, end)
                    flush(cursor)
                    if group is None:
                        self.emit_opaque(
                            cursor,
                            command_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = command_end
                        text_start = cursor
                        continue
                    opening, group_end = group
                    raw_label = self.source[opening + 1 : group_end - 1]
                    label = raw_label.strip()
                    # Compiler-derived references are admitted only for a
                    # literal label.  Macro-generated/whitespace-commented
                    # labels remain opaque instead of being guessed.
                    if (
                        not label
                        or label != raw_label
                        or re.fullmatch(r"[A-Za-z0-9_.:/-]+", label) is None
                        or label not in self.reference_values
                    ):
                        self.emit_opaque(
                            cursor,
                            group_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = group_end
                        text_start = cursor
                        continue
                    number = self.reference_values[label]
                    visible = f"({number})" if command == "eqref" else number
                    node = self.add_node(
                        "reference",
                        cursor,
                        group_end,
                        parent_node_id,
                        content_span=self.span(opening + 1, group_end - 1),
                    )
                    self.add_atom(
                        node.node_id,
                        "reference",
                        cursor,
                        group_end,
                        self.source[cursor:group_end],
                        visible,
                        _escape_markdown_literal(visible),
                        style_stack,
                        footnote_path,
                    )
                    cursor = group_end
                    text_start = cursor
                    continue

                if command in {"footnote", "thanks"}:
                    group = self.footnote_group_after(command_end, end)
                    flush(cursor)
                    if group is None:
                        opaque_end = self.consume_unknown(command_end, end)
                        self.emit_opaque(
                            cursor,
                            opaque_end,
                            parent_node_id,
                            style_stack,
                            footnote_path,
                        )
                        cursor = max(opaque_end, cursor + 1)
                        text_start = cursor
                        continue
                    opening, group_end, explicit_mark = group
                    callout_span = self.span(cursor, opening + 1)
                    body_span = self.span(opening + 1, group_end - 1)
                    mark_span = (
                        None
                        if explicit_mark is None
                        else self.span(explicit_mark[0], explicit_mark[1])
                    )
                    node = self.add_footnote_node(
                        cursor,
                        group_end,
                        parent_node_id,
                        callout_span,
                        body_span,
                        mark_span,
                        command,
                    )
                    callout_atom = self.add_atom(
                        node.node_id,
                        "footnote_callout",
                        cursor,
                        opening + 1,
                        self.source[cursor : opening + 1],
                        "",
                        "",
                        style_stack,
                        footnote_path,
                    )
                    body_atom_start = len(self.atoms)
                    self.parse_sequence(
                        opening + 1,
                        group_end - 1,
                        node.node_id,
                        # ``\\thanks`` stores its body for the later
                        # ``\\maketitle`` footnote pass, outside the author
                        # name's local bold/italic group.  Its callout keeps
                        # the surrounding style, but its body starts from the
                        # template footnote font rather than inheriting the
                        # author's text style.
                        () if command == "thanks" else style_stack,
                        footnote_path + (node.node_id,),
                    )
                    body_atom_ids = tuple(
                        atom.atom_id for atom in self.atoms[body_atom_start:]
                    )
                    state = self.node_state_by_id[node.node_id]
                    state.node = dataclasses.replace(
                        node,
                        callout_atom_id=callout_atom.atom_id,
                        body_atom_ids=body_atom_ids,
                    )
                    cursor = group_end
                    text_start = cursor
                    continue

                source_macro = self.source_macro_after(command, command_end, end)
                if source_macro is not None:
                    invocation_end, replacement = source_macro
                    rendered = self.render_source_macro(command, replacement)
                    if rendered is not None:
                        flush(cursor)
                        visible, markdown, verifier_fragments = rendered
                        node = self.add_node(
                            "source_macro",
                            cursor,
                            invocation_end,
                            parent_node_id,
                        )
                        if visible or markdown:
                            self.add_atom(
                                node.node_id,
                                "source_macro",
                                cursor,
                                invocation_end,
                                self.source[cursor:invocation_end],
                                visible,
                                markdown,
                                style_stack,
                                footnote_path,
                                verifier_fragments,
                            )
                        cursor = invocation_end
                        text_start = cursor
                        continue

                flush(cursor)
                opaque_end = self.consume_unknown(command_end, end)
                self.emit_opaque(
                    cursor,
                    opaque_end,
                    parent_node_id,
                    style_stack,
                    footnote_path,
                )
                cursor = max(opaque_end, cursor + 1)
                text_start = cursor
                continue

            if char == "$":
                # An unmatched single dollar makes the complete remaining
                # fragment ambiguous; do not leak its body into plain text.
                flush(cursor)
                self.emit_opaque(
                    cursor,
                    end,
                    parent_node_id,
                    style_stack,
                    footnote_path,
                )
                return

            cursor += 1

        flush(end)

    def freeze_nodes(self) -> tuple[AstNode, ...]:
        frozen: list[AstNode] = []
        for state in self.node_states:
            frozen.append(
                dataclasses.replace(
                    state.node,
                    child_node_ids=tuple(state.children),
                    atom_ids=tuple(state.atoms),
                )
            )
        return tuple(frozen)


def parse_source_ir(
    source: str,
    *,
    source_id: str = "<memory>",
    source_char_base: int = 0,
    source_byte_base: int = 0,
    reference_values: Mapping[str, str] | None = None,
    math_macros: Mapping[str, MathMacroDefinition] | None = None,
    initial_style_stack: tuple[str, ...] = (),
    _macro_expansion_stack: tuple[str, ...] = (),
) -> SourceDocumentIR:
    """Parse supported inline LaTeX into an immutable source AST/IR.

    IDs are deterministic SHA-256-derived identifiers over the source ID,
    construct kind, absolute dual span, parent ID, and exact source bytes.
    They never depend on process state or Python's randomized hash function.
    """

    if not isinstance(source, str):
        raise SourceIrError("source must be a string")
    if not isinstance(source_id, str) or not source_id:
        raise SourceIrError("source_id must be a non-empty string")
    if (
        not isinstance(initial_style_stack, tuple)
        or any(style not in _SERIALIZABLE_STYLES for style in initial_style_stack)
        or len(set(initial_style_stack)) != len(initial_style_stack)
    ):
        raise SourceIrError("initial_style_stack contains unsupported styles")
    for name, value in (
        ("source_char_base", source_char_base),
        ("source_byte_base", source_byte_base),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SourceIrError(f"{name} must be a non-negative integer")
    try:
        source_bytes = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SourceIrError("source is not encodable as UTF-8") from exc

    normalized_references: dict[str, str] = {}
    for raw_label, raw_value in (reference_values or {}).items():
        label = str(raw_label)
        value = str(raw_value)
        if re.fullmatch(r"[A-Za-z0-9_.:/-]+", label) is None:
            raise SourceIrError(f"invalid compiler reference label: {label!r}")
        if re.fullmatch(r"[A-Za-z0-9]+(?:[.\-:][A-Za-z0-9]+)*", value) is None:
            raise SourceIrError(
                f"invalid compiler reference value for {label!r}: {value!r}"
            )
        normalized_references[label] = value

    normalized_math_macros: dict[str, MathMacroDefinition] = {}
    for raw_name, raw_macro in (math_macros or {}).items():
        name = str(raw_name)
        if re.fullmatch(r"[A-Za-z@]+", name) is None:
            raise SourceIrError(f"invalid math macro name: {name!r}")
        if not isinstance(raw_macro, MathMacroDefinition):
            raise SourceIrError(f"invalid math macro definition for {name!r}")
        if raw_macro.name != name:
            raise SourceIrError(f"math macro key/name mismatch for {name!r}")
        if (
            not isinstance(raw_macro.argument_count, int)
            or isinstance(raw_macro.argument_count, bool)
            or not 0 <= raw_macro.argument_count <= 4
        ):
            raise SourceIrError(f"invalid math macro arity for {name!r}")
        if not isinstance(raw_macro.body, str):
            raise SourceIrError(f"invalid math macro body for {name!r}")
        normalized_math_macros[name] = raw_macro

    parser = _Parser(
        source,
        source_id,
        source_char_base,
        source_byte_base,
        normalized_references,
        normalized_math_macros,
        _macro_expansion_stack,
    )
    root = parser.add_node("document", 0, len(source), None, content_span=parser.span(0, len(source)))
    parser.parse_sequence(
        0,
        len(source),
        root.node_id,
        initial_style_stack,
        (),
    )
    return SourceDocumentIR(
        version=IR_VERSION,
        source_id=source_id,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_char_base=source_char_base,
        source_byte_base=source_byte_base,
        root_node_id=root.node_id,
        nodes=parser.freeze_nodes(),
        atoms=tuple(parser.atoms),
    )


def build_display_math_ir(
    source: str,
    *,
    content_start: int,
    content_end: int,
    source_id: str = "<memory>",
    source_char_base: int = 0,
    source_byte_base: int = 0,
    math_macros: Mapping[str, MathMacroDefinition] | None = None,
    markdown_environment: str | None = None,
    resolved_tag: str | None = None,
) -> SourceDocumentIR:
    """Build one exact source-derived IR atom for a display-math block.

    ``source`` is the complete, unchanged TeX construct (delimiters or
    environment included), while ``content_start``/``content_end`` delimit
    the mathematical body inside that local string.  The complete construct
    remains one compiler-trace atom, so marker insertion can occur only
    outside a balanced ``$$...$$``, ``\\[...\\]``, or equation environment.

    The Markdown formula is serialized from the source body alone.  Compiler
    positions may place this immutable atom on a page, but PDF text is never
    used to construct or repair the formula.
    """

    if not isinstance(source, str):
        raise SourceIrError("source must be a string")
    if not isinstance(source_id, str) or not source_id:
        raise SourceIrError("source_id must be a non-empty string")
    if (
        isinstance(content_start, bool)
        or isinstance(content_end, bool)
        or not isinstance(content_start, int)
        or not isinstance(content_end, int)
        or content_start < 0
        or content_end < content_start
        or content_end > len(source)
    ):
        raise SourceIrError("display-math content span is invalid")
    for name, value in (
        ("source_char_base", source_char_base),
        ("source_byte_base", source_byte_base),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SourceIrError(f"{name} must be a non-negative integer")

    macros: dict[str, MathMacroDefinition] = {}
    for raw_name, raw_macro in (math_macros or {}).items():
        name = str(raw_name)
        if re.fullmatch(r"[A-Za-z@]+", name) is None:
            raise SourceIrError(f"invalid math macro name: {name!r}")
        if not isinstance(raw_macro, MathMacroDefinition) or raw_macro.name != name:
            raise SourceIrError(f"invalid math macro definition for {name!r}")
        macros[name] = raw_macro

    source_bytes = source.encode("utf-8")
    byte_prefix = [0]
    for char in source:
        byte_prefix.append(byte_prefix[-1] + len(char.encode("utf-8")))

    def span(start: int, end: int) -> SourceSpan:
        return SourceSpan(
            char_start=source_char_base + start,
            char_end=source_char_base + end,
            byte_start=source_byte_base + byte_prefix[start],
            byte_end=source_byte_base + byte_prefix[end],
        )

    full_span = span(0, len(source))
    body_span = span(content_start, content_end)
    normalized = _expand_math_macros(
        _normalize_math(source[content_start:content_end]),
        macros,
    )
    if not normalized:
        raise SourceIrError("display-math body has no source-visible content")

    if resolved_tag is not None:
        resolved_tag = _validate_resolved_math_tag(resolved_tag)
        if not _has_literal_math_tag(normalized):
            normalized = f"{normalized.rstrip()}\\tag{{{resolved_tag}}}"

    root_id = _stable_id("node", source_id, "document", full_span, source, None)
    math_id = _stable_id("node", source_id, "math", full_span, source, root_id)
    atom_id = _stable_id("atom", source_id, "math", full_span, source, math_id)
    root = SourceNode(
        node_id=root_id,
        kind="document",
        span=full_span,
        parent_node_id=None,
        child_node_ids=(math_id,),
        atom_ids=(),
        content_span=full_span,
    )
    math = SourceNode(
        node_id=math_id,
        kind="math",
        span=full_span,
        parent_node_id=root_id,
        child_node_ids=(),
        atom_ids=(atom_id,),
        content_span=body_span,
    )
    if markdown_environment is not None:
        if re.fullmatch(r"[A-Za-z]+", markdown_environment) is None:
            raise SourceIrError("unsafe Markdown math environment name")
        normalized_markdown = (
            f"\\begin{{{markdown_environment}}}\n"
            f"{normalized}\n"
            f"\\end{{{markdown_environment}}}"
        )
    else:
        normalized_markdown = normalized
    atom = SourceAtom(
        atom_id=atom_id,
        ordinal=0,
        node_id=math_id,
        kind="math",
        span=full_span,
        raw_source=source,
        visible_text=normalized,
        markdown_fragment=f"$$\n{normalized_markdown}\n$$",
        style_stack=(),
        footnote_path=(),
    )
    return SourceDocumentIR(
        version=IR_VERSION,
        source_id=source_id,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_char_base=source_char_base,
        source_byte_base=source_byte_base,
        root_node_id=root_id,
        nodes=(root, math),
        atoms=(atom,),
    )


_ALIGN_CONTROL_WORD = re.compile(r"\\([A-Za-z@]+)")
_ALIGN_LITERAL_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._/@-]*")


def _validate_resolved_math_tag(value: str) -> str:
    """Validate one compiler-resolved equation number before TeX embedding."""

    if not isinstance(value, str):
        raise SourceIrError("resolved math tag must be a string")
    stripped = value.strip()
    if (
        not stripped
        or len(stripped) > 128
        or any(char in stripped for char in "\\{}%\r\n")
    ):
        raise SourceIrError("resolved math tag is not a safe literal value")
    return stripped


def _has_literal_math_tag(value: str) -> bool:
    """Return whether comment-free math contains a literal ``\\tag{...}``."""

    for match in re.finditer(r"\\tag\*?\s*\{", value):
        group_start = value.find("{", match.start(), match.end())
        group_end = _find_balanced(value, group_start, len(value))
        if group_end is not None:
            return True
    return False


def resolve_display_math_reference_tag(
    body: str,
    reference_values: Mapping[str, str],
) -> str | None:
    """Resolve one equation label exclusively from source plus compiler AUX.

    An explicit source ``\\tag`` remains authoritative and therefore returns
    ``None`` here.  A missing AUX value also returns ``None`` so a caller may
    still require its independent compiler-number trace.  Ambiguous or
    non-literal labels fail closed.
    """

    normalized = _normalize_math(body)
    labels: list[str] = []
    cursor = 0
    while cursor < len(normalized):
        match = re.search(r"\\label\b", normalized[cursor:])
        if match is None:
            break
        command_start = cursor + match.start()
        command_end = cursor + match.end()
        group_end, key = _align_command_group(
            normalized,
            command_end,
            command_name="label",
        )
        if _ALIGN_LITERAL_KEY.fullmatch(key) is None:
            raise SourceIrError("equation label key is not literal")
        labels.append(key)
        cursor = group_end
    if len(labels) > 1:
        raise SourceIrError("equation has multiple literal labels")
    if _has_literal_math_tag(normalized):
        return None
    if not labels or labels[0] not in reference_values:
        return None
    return _validate_resolved_math_tag(str(reference_values[labels[0]]))


def _align_command_group(
    row: str,
    command_end: int,
    *,
    command_name: str,
) -> tuple[int, str]:
    cursor = command_end
    while cursor < len(row) and row[cursor].isspace():
        cursor += 1
    if cursor >= len(row) or row[cursor] != "{":
        raise SourceIrError(
            f"numbered align has non-literal \\{command_name} command"
        )
    group_end = _find_balanced(row, cursor, len(row))
    if group_end is None:
        raise SourceIrError(
            f"numbered align has unbalanced \\{command_name} argument"
        )
    return group_end, row[cursor + 1 : group_end - 1]


def _split_numbered_align_rows(body: str) -> tuple[tuple[str, str], ...]:
    """Split an align body only at proven top-level TeX row separators."""

    if re.search(r"\\(?:begin|end)\s*\{", body):
        raise SourceIrError("numbered align contains a nested environment")
    if re.search(r"\\(?:intertext|shortintertext)\b", body):
        raise SourceIrError("numbered align contains intertext")

    rows: list[tuple[str, str]] = []
    row_start = 0
    cursor = 0
    brace_depth = 0
    while cursor < len(body):
        char = body[cursor]
        if char == "%" and _is_comment_start(body, cursor):
            cursor = _skip_comment(body, cursor, len(body))
            continue
        if char == "{" and (cursor == 0 or body[cursor - 1] != "\\"):
            brace_depth += 1
            cursor += 1
            continue
        if char == "}" and (cursor == 0 or body[cursor - 1] != "\\"):
            brace_depth -= 1
            if brace_depth < 0:
                raise SourceIrError("numbered align has an unmatched closing brace")
            cursor += 1
            continue
        if body.startswith("\\\\", cursor) and brace_depth == 0:
            separator_end = cursor + 2
            if separator_end < len(body) and body[separator_end] == "*":
                separator_end += 1
            spacing_cursor = separator_end
            while spacing_cursor < len(body) and body[spacing_cursor].isspace():
                spacing_cursor += 1
            if spacing_cursor < len(body) and body[spacing_cursor] == "[":
                bracket_end = _find_balanced(
                    body,
                    spacing_cursor,
                    len(body),
                    open_char="[",
                    close_char="]",
                )
                if bracket_end is None:
                    raise SourceIrError(
                        "numbered align has an unbalanced row-spacing argument"
                    )
                separator_end = bracket_end
            rows.append((body[row_start:cursor], body[cursor:separator_end]))
            row_start = separator_end
            cursor = separator_end
            continue
        if body.startswith("\\\\", cursor):
            raise SourceIrError("numbered align contains a nested row separator")
        if char == "\\":
            cursor = _command_end(body, cursor, len(body))
        else:
            cursor += 1
    if brace_depth:
        raise SourceIrError("numbered align has an unclosed brace group")
    rows.append((body[row_start:], ""))
    if any(not _normalize_math(row) for row, _separator in rows):
        raise SourceIrError("numbered align contains an empty row")
    return tuple(rows)


def _serialize_numbered_align_row(
    raw_row: str,
    reference_values: Mapping[str, str],
) -> tuple[str, str | None]:
    """Remove invisible controls and prove one align row's number source."""

    row = _normalize_math(raw_row)
    removals: list[tuple[int, int]] = []
    labels: list[str] = []
    tags: list[str] = []
    suppression_count = 0
    cursor = 0
    brace_depth = 0
    while cursor < len(row):
        if row[cursor] == "{" and (cursor == 0 or row[cursor - 1] != "\\"):
            brace_depth += 1
            cursor += 1
            continue
        if row[cursor] == "}" and (cursor == 0 or row[cursor - 1] != "\\"):
            brace_depth -= 1
            if brace_depth < 0:
                raise SourceIrError("numbered align row has unmatched braces")
            cursor += 1
            continue
        if row[cursor] != "\\":
            cursor += 1
            continue
        match = _ALIGN_CONTROL_WORD.match(row, cursor)
        if match is None:
            cursor = min(len(row), cursor + 2)
            continue
        name = match.group(1)
        command_end = match.end()
        if brace_depth != 0 and name in {"label", "tag", "notag", "nonumber"}:
            raise SourceIrError(
                f"numbered align contains nested numbering control: \\{name}"
            )
        if name not in {"label", "tag", "notag", "nonumber"}:
            cursor = command_end
            continue
        if name == "label":
            group_end, key = _align_command_group(
                row, command_end, command_name=name
            )
            if _ALIGN_LITERAL_KEY.fullmatch(key) is None:
                raise SourceIrError("numbered align label key is not literal")
            labels.append(key)
            removals.append((cursor, group_end))
            cursor = group_end
            continue
        if name == "tag":
            star_end = command_end + (1 if row.startswith("*", command_end) else 0)
            group_end, literal = _align_command_group(
                row, star_end, command_name=name
            )
            if (
                not literal.strip()
                or len(literal) > 128
                or any(char in literal for char in "%\r\n")
                or "\\" in literal
                or "{" in literal
                or "}" in literal
            ):
                raise SourceIrError("numbered align tag is not literal")
            tags.append(row[cursor:group_end])
            cursor = group_end
            continue
        suppression_count += 1
        removals.append((cursor, command_end))
        cursor = command_end

    if brace_depth:
        raise SourceIrError("numbered align row has unclosed braces")
    if len(labels) > 1 or len(tags) > 1 or suppression_count > 1:
        raise SourceIrError("numbered align row has ambiguous numbering controls")
    if suppression_count and (labels or tags):
        raise SourceIrError("unnumbered align row also contains a label or tag")

    pieces: list[str] = []
    previous = 0
    for start, end in sorted(removals):
        pieces.append(row[previous:start])
        previous = end
    pieces.append(row[previous:])
    visible_row = "".join(pieces).strip()
    if not visible_row:
        raise SourceIrError("numbered align row has no mathematical content")
    if suppression_count or tags:
        return visible_row, (labels[0] if labels else None)
    if len(labels) != 1:
        raise SourceIrError(
            "numbered align row lacks a unique literal label or explicit tag"
        )
    key = labels[0]
    if key not in reference_values:
        raise SourceIrError(
            f"numbered align label has no compiler AUX value: {key}"
        )
    tag = _validate_resolved_math_tag(str(reference_values[key]))
    return f"{visible_row.rstrip()}\\tag{{{tag}}}", key


def build_numbered_align_ir(
    source: str,
    *,
    content_start: int,
    content_end: int,
    reference_values: Mapping[str, str],
    source_id: str = "<memory>",
    source_char_base: int = 0,
    source_byte_base: int = 0,
    math_macros: Mapping[str, MathMacroDefinition] | None = None,
) -> SourceDocumentIR:
    """Build a complete source atom for the provable subset of ``align``."""

    base = build_display_math_ir(
        source,
        content_start=content_start,
        content_end=content_end,
        source_id=source_id,
        source_char_base=source_char_base,
        source_byte_base=source_byte_base,
        math_macros=math_macros,
    )
    body = source[content_start:content_end]
    rows = _split_numbered_align_rows(body)
    serialized_rows = [
        (*_serialize_numbered_align_row(row, reference_values), separator)
        for row, separator in rows
    ]
    labels = [label for _row, label, _separator in serialized_rows if label]
    if len(labels) != len(set(labels)):
        raise SourceIrError("numbered align reuses one literal label across rows")
    serialized = "".join(
        row + separator for row, _label, separator in serialized_rows
    )
    serialized = _expand_math_macros(serialized, dict(math_macros or {}))
    normalized_markdown = (
        "\\begin{aligned}\n"
        f"{serialized}\n"
        "\\end{aligned}"
    )
    atom = dataclasses.replace(
        base.atoms[0],
        visible_text=serialized,
        markdown_fragment=f"$$\n{normalized_markdown}\n$$",
    )
    return dataclasses.replace(base, atoms=(atom,))


def build_source_ir(
    source: str,
    *,
    source_id: str = "<memory>",
    source_char_base: int = 0,
    source_byte_base: int = 0,
) -> SourceDocumentIR:
    """Alias emphasizing that parsing builds compiler-localization IR."""

    return parse_source_ir(
        source,
        source_id=source_id,
        source_char_base=source_char_base,
        source_byte_base=source_byte_base,
    )


def _style_delimiters(style: str) -> tuple[str, str]:
    return {
        "strong": ("**", "**"),
        "em": ("*", "*"),
        # A semantic theorem/proof body is compiler/source-contract-derived
        # rather than delimited by a source command.  HTML keeps independent
        # page slices balanced when an explicit ``\emph`` toggles upright in
        # the middle of inherited italic text.
        "body_em": ("<em>", "</em>"),
        "code": ("<code>", "</code>"),
        "sup": ("<sup>", "</sup>"),
        # The visible letters were already transformed while building the
        # immutable atoms.  Empty delimiters preserve that transformation in
        # Markdown without inventing a non-standard HTML wrapper.
        "smallcaps": ("", ""),
    }[style]


def reconstruct_markdown(
    document: SourceDocumentIR,
    selected_atom_ids: Iterable[str] | None = None,
    *,
    footnote_renderer: FootnoteRenderer | None = None,
) -> str:
    """Render selected source atoms as independently balanced Markdown.

    Atom input order is ignored; source order always wins.  A selection gap
    closes and reopens active formatting.  Footnote body atoms are rendered
    only through the caller callback, so this function can never guess a
    marker number or silently choose a footnote syntax.
    """

    known = {atom.atom_id for atom in document.atoms}
    if selected_atom_ids is None:
        wanted = set(known)
    else:
        wanted = {str(atom_id) for atom_id in selected_atom_ids}
        unknown = wanted - known
        if unknown:
            raise InvalidAtomSelection(
                "unknown selected atom IDs: " + ", ".join(sorted(unknown))
            )

    atom_by_id = {atom.atom_id: atom for atom in document.atoms}
    footnote_by_id = {node.node_id: node for node in document.footnotes}

    def render_scope(scope: tuple[str, ...]) -> str:
        scoped_atoms = [atom for atom in document.atoms if atom.footnote_path == scope]
        scoped_positions = {atom.atom_id: index for index, atom in enumerate(scoped_atoms)}

        # Selecting any body atom without the corresponding source callout is
        # structurally ambiguous.  Reject it instead of inventing placement.
        for node in document.footnotes:
            callout = atom_by_id[node.callout_atom_id]
            if callout.footnote_path != scope:
                continue
            selected_body = wanted.intersection(node.body_atom_ids)
            if selected_body and node.callout_atom_id not in wanted:
                raise InvalidAtomSelection(
                    f"footnote body selected without callout: {node.node_id}"
                )

        selected = [atom for atom in scoped_atoms if atom.atom_id in wanted]
        if not selected:
            return ""

        output: list[str] = []
        active: tuple[str, ...] = ()
        previous_position: int | None = None

        def close_all() -> None:
            nonlocal active
            for style in reversed(active):
                output.append(_style_delimiters(style)[1])
            active = ()

        for atom in selected:
            position = scoped_positions[atom.atom_id]
            if previous_position is not None and position != previous_position + 1:
                close_all()

            target = atom.style_stack
            common = 0
            while (
                common < len(active)
                and common < len(target)
                and active[common] == target[common]
            ):
                common += 1
            for style in reversed(active[common:]):
                output.append(_style_delimiters(style)[1])
            for style in target[common:]:
                output.append(_style_delimiters(style)[0])

            if atom.kind == "footnote_callout":
                if footnote_renderer is None:
                    raise FootnoteRendererRequired(
                        f"footnote renderer required for {atom.node_id}"
                    )
                node = footnote_by_id[atom.node_id]
                selected_body_ids = tuple(
                    body_id for body_id in node.body_atom_ids if body_id in wanted
                )
                body_markdown = render_scope(scope + (node.node_id,))
                rendered = footnote_renderer(
                    FootnoteRenderContext(
                        document=document,
                        node=node,
                        callout_atom=atom,
                        selected_body_atom_ids=selected_body_ids,
                        body_markdown=body_markdown,
                        body_complete=len(selected_body_ids) == len(node.body_atom_ids),
                    )
                )
                if not isinstance(rendered, str):
                    raise SourceIrError("footnote renderer must return a string")
                output.append(rendered)
            else:
                output.append(atom.markdown_fragment)

            active = target
            previous_position = position

        close_all()
        return "".join(output)

    return render_scope(())


def atoms_to_markdown(
    document: SourceDocumentIR,
    *,
    footnote_renderer: FootnoteRenderer | None = None,
) -> str:
    """Render every atom; footnotes still require an injected renderer."""

    return reconstruct_markdown(document, footnote_renderer=footnote_renderer)


__all__ = [
    "IR_VERSION",
    "OPAQUE_MARKER_PREFIX",
    "OPAQUE_MARKER_SUFFIX",
    "AstNode",
    "FootnoteNode",
    "FootnoteRenderContext",
    "FootnoteRenderer",
    "FootnoteRendererRequired",
    "InvalidAtomSelection",
    "MathMacroDefinition",
    "build_display_math_ir",
    "build_numbered_align_ir",
    "resolve_display_math_reference_tag",
    "SourceAtom",
    "SourceDocumentIR",
    "SourceIrError",
    "SourceNode",
    "SourceSpan",
    "atoms_to_markdown",
    "build_source_ir",
    "parse_source_ir",
    "reconstruct_markdown",
]
