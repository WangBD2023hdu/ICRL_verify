#!/usr/bin/env python3
"""Deterministic inline LaTeX-to-Markdown alignment helpers.

The module parses a LaTeX prose fragment into a small tree, builds a regular
expression that can align that tree with text extracted from the compiled PDF,
and renders the match as Markdown.  It deliberately does *not* try to predict
what TeX produced for mathematics, references, citations, footnote marks, or
unknown macros:

* inline mathematics is replaced with the original LaTeX between ``$``;
* supported style commands become Markdown emphasis/strong/code;
* opaque commands keep exactly the substring observed in the PDF;
* footnote bodies and labels never leak into the prose output.

Parsing of braced and optional arguments uses a balanced scanner.  Regex
wildcards are named and length-bounded so malformed input cannot create an
unbounded ``.*`` alignment expression.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import re
import signal
import threading
import unicodedata
from typing import Iterator, Mapping, Match, Pattern


DEFAULT_MAX_WILDCARD = 96
MAX_CONFIGURED_WILDCARD = 4096
INLINE_REGEX_TIMEOUT_SECONDS = 0.25

STYLE_COMMANDS = {
    "textbf": "strong",
    "emph": "em",
    "textit": "em",
    "texttt": "code",
}
CITE_COMMANDS = {
    "cite",
    "citep",
    "citet",
    "citealp",
    "citealt",
    "citeauthor",
    "citeauthor*",
    "citeyear",
    "citeyearpar",
}
REF_COMMANDS = {"ref", "pageref", "eqref", "autoref", "cref", "Cref"}
REQUIRED_OPAQUE_ARGUMENTS = CITE_COMMANDS | REF_COMMANDS | {"footnote", "label"}

ESCAPED_TEXT = {
    "&": "&",
    "%": "%",
    "$": "$",
    "#": "#",
    "_": "_",
    "{": "{",
    "}": "}",
    " ": " ",
    "\\": " ",  # TeX line break in prose.
}


class InlineParseError(ValueError):
    """Raised when an inline LaTeX fragment is structurally incomplete."""


class InlineRegexTimeout(TimeoutError):
    """Raised internally when a pathological alignment regex exceeds its budget."""


def _bounded_regex_call(function: object, value: str) -> Match[str] | None:
    """Run one stdlib ``re`` call with a fail-closed wall-clock budget.

    CPython's regex engine can backtrack catastrophically on a long PDF text
    window even though every individual wildcard is length-bounded.  The page
    builder is single-threaded, so a short POSIX interval timer is sufficient
    to turn that candidate into a normal non-match.  Other threads retain the
    previous behavior because Python only permits signal handlers in the main
    thread.
    """

    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        return function(value)  # type: ignore[operator]

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise InlineRegexTimeout("inline alignment regex timed out")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, INLINE_REGEX_TIMEOUT_SECONDS)
    try:
        return function(value)  # type: ignore[operator]
    except InlineRegexTimeout:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _bounded_regex_finditer(pattern: Pattern[str], value: str) -> tuple[Match[str], ...]:
    """Materialize ``finditer`` under the same fail-closed regex budget.

    Returning the generator from :func:`_bounded_regex_call` would not help:
    CPython performs the expensive search only while that generator is
    consumed.  Materializing inside the timer ensures a pathological page is
    rejected in bounded time instead of pinning one dataset worker for hours.
    """

    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        return tuple(pattern.finditer(value))

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise InlineRegexTimeout("inline alignment regex iteration timed out")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, INLINE_REGEX_TIMEOUT_SECONDS)
    try:
        return tuple(pattern.finditer(value))
    except InlineRegexTimeout:
        return ()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclasses.dataclass(frozen=True)
class InlineNode:
    """One node in an :class:`InlinePlan` tree.

    ``kind`` is one of ``root``, ``text``, ``math``, ``strong``, ``em``,
    ``code``, or ``opaque``.  Text values are the expected visible text; math
    values are the un-delimited source formula.  Opaque values are intentionally
    empty because the compiled PDF is authoritative for those nodes.
    """

    node_id: int
    kind: str
    raw: str
    value: str = ""
    children: tuple["InlineNode", ...] = ()
    command: str | None = None
    opaque_role: str | None = None


@dataclasses.dataclass(frozen=True)
class InlinePlan:
    """Parsed inline tree plus cheap feature and anchor statistics."""

    raw: str
    root: InlineNode
    feature_counts: Mapping[str, int]
    anchors: tuple[str, ...]

    @property
    def anchor_count(self) -> int:
        return len(self.anchors)

    @property
    def anchor_characters(self) -> int:
        return sum(len(anchor) for anchor in self.anchors)

    @property
    def wildcard_count(self) -> int:
        return int(self.feature_counts.get("math", 0)) + int(
            self.feature_counts.get("opaque", 0)
        )


@dataclasses.dataclass(frozen=True)
class InlineRegex:
    """Named-group alignment regex generated from an :class:`InlinePlan`."""

    pattern: str
    compiled: Pattern[str] = dataclasses.field(repr=False, compare=False)
    group_names: Mapping[int, str]
    max_wildcard: int

    def fullmatch(self, value: str) -> Match[str] | None:
        return _bounded_regex_call(self.compiled.fullmatch, value)

    def search(self, value: str) -> Match[str] | None:
        return _bounded_regex_call(self.compiled.search, value)

    def finditer(self, value: str) -> tuple[Match[str], ...]:
        return _bounded_regex_finditer(self.compiled, value)


@dataclasses.dataclass(frozen=True)
class InlineRenderResult:
    """Rendered Markdown and provenance for one successful PDF-text match."""

    markdown: str
    matched_text: str
    span: tuple[int, int]
    regex: InlineRegex = dataclasses.field(repr=False, compare=False)


@dataclasses.dataclass(frozen=True)
class FootnoteSource:
    """Safely extracted source fields for one opaque ``\\footnote`` node."""

    raw: str
    body_raw: str
    optional_arguments: tuple[str, ...] = ()


def iter_inline_nodes(node: InlineNode) -> Iterator[InlineNode]:
    """Yield ``node`` and all descendants in source order."""

    yield node
    for child in node.children:
        yield from iter_inline_nodes(child)


class _Parser:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.length = len(raw)
        self.position = 0
        self.next_node_id = 0
        self.features: collections.Counter[str] = collections.Counter()

    def _node(
        self,
        kind: str,
        raw: str,
        *,
        value: str = "",
        children: tuple[InlineNode, ...] = (),
        command: str | None = None,
        opaque_role: str | None = None,
    ) -> InlineNode:
        node = InlineNode(
            node_id=self.next_node_id,
            kind=kind,
            raw=raw,
            value=value,
            children=children,
            command=command,
            opaque_role=opaque_role,
        )
        self.next_node_id += 1
        if kind != "root":
            self.features[kind] += 1
        if kind == "opaque" and opaque_role:
            self.features[opaque_role] += 1
        return node

    def parse(self) -> InlinePlan:
        children = self._parse_sequence(stop=None)
        children = _clean_children(children, trim_edges=True)
        root = self._node("root", self.raw, children=children)
        anchors = tuple(
            node.value
            for node in iter_inline_nodes(root)
            if node.kind == "text" and node.value and node.value.strip()
        )
        return InlinePlan(
            raw=self.raw,
            root=root,
            feature_counts=dict(sorted(self.features.items())),
            anchors=anchors,
        )

    def _parse_sequence(self, stop: str | None) -> tuple[InlineNode, ...]:
        nodes: list[InlineNode] = []
        text_start = self.position

        def flush_text(end: int) -> None:
            nonlocal text_start
            if end <= text_start:
                return
            raw_text = self.raw[text_start:end]
            value = _tex_text_to_visible(raw_text)
            if value:
                nodes.append(self._node("text", raw_text, value=value))

        while self.position < self.length:
            character = self.raw[self.position]
            if stop is not None and character == stop:
                flush_text(self.position)
                self.position += 1
                return tuple(nodes)
            if character == "}" and stop is None:
                flush_text(self.position)
                raise InlineParseError(f"unexpected closing brace at offset {self.position}")
            if character not in "{\\$%":
                self.position += 1
                continue

            flush_text(self.position)
            if character == "{":
                group_start = self.position
                legacy_em_start = _legacy_em_content_start(
                    self.raw, group_start + 1
                )
                if legacy_em_start is not None:
                    self.position = legacy_em_start
                    group_children = _clean_children(
                        self._parse_sequence(stop="}"), trim_edges=False
                    )
                    nodes.append(
                        self._node(
                            "em",
                            self.raw[group_start : self.position],
                            children=group_children,
                            command="em",
                        )
                    )
                else:
                    self.position += 1
                    group_children = self._parse_sequence(stop="}")
                    nodes.extend(group_children)
                text_start = self.position
                if self.position <= group_start + 1:
                    raise AssertionError("balanced scanner failed to advance")
                continue
            if character == "%":
                self._skip_comment()
                text_start = self.position
                continue
            if character == "$":
                nodes.append(self._parse_dollar_math())
                text_start = self.position
                continue
            nodes.append(self._parse_command())
            text_start = self.position

        flush_text(self.position)
        if stop is not None:
            raise InlineParseError(f"unclosed group; expected {stop!r}")
        return tuple(nodes)

    def _skip_comment(self) -> None:
        newline = self.raw.find("\n", self.position + 1)
        if newline < 0:
            self.position = self.length
            return
        # A TeX comment consumes the newline too.  Indentation on the next line
        # remains ordinary whitespace and is normalized with surrounding text.
        self.position = newline + 1

    def _parse_dollar_math(self) -> InlineNode:
        start = self.position
        if self.raw.startswith("$$", start):
            raise InlineParseError(
                f"display math is not valid in an inline plan at offset {start}"
            )
        cursor = start + 1
        while cursor < self.length:
            if self.raw[cursor] == "$" and not _is_escaped(self.raw, cursor):
                body = self.raw[start + 1 : cursor]
                self.position = cursor + 1
                return self._node(
                    "math",
                    self.raw[start : self.position],
                    value=_clean_math_body(body),
                )
            cursor += 1
        raise InlineParseError(f"unclosed inline math starting at offset {start}")

    def _parse_parenthesized_math(self) -> InlineNode:
        start = self.position
        cursor = start + 2
        while cursor < self.length - 1:
            if self.raw.startswith(r"\)", cursor) and not _is_escaped(
                self.raw, cursor
            ):
                body = self.raw[start + 2 : cursor]
                self.position = cursor + 2
                return self._node(
                    "math",
                    self.raw[start : self.position],
                    value=_clean_math_body(body),
                )
            cursor += 1
        raise InlineParseError(f"unclosed \\( math starting at offset {start}")

    def _parse_verb(self, start: int, command_end: int) -> InlineNode:
        cursor = command_end
        if cursor >= self.length:
            raise InlineParseError(f"missing delimiter after \\verb at offset {start}")
        delimiter = self.raw[cursor]
        if delimiter.isspace() or delimiter.isalnum() or delimiter in "{}\\":
            raise InlineParseError(f"invalid \\verb delimiter at offset {cursor}")
        end = self.raw.find(delimiter, cursor + 1)
        if end < 0:
            raise InlineParseError(f"unclosed \\verb starting at offset {start}")
        body = self.raw[cursor + 1 : end]
        self.position = end + 1
        child = self._node("text", body, value=body)
        return self._node(
            "code",
            self.raw[start : self.position],
            children=(child,),
            command="verb",
        )

    def _parse_command(self) -> InlineNode:
        start = self.position
        if start + 1 >= self.length:
            raise InlineParseError(f"dangling backslash at offset {start}")

        if self.raw.startswith(r"\(", start):
            return self._parse_parenthesized_math()

        next_character = self.raw[start + 1]
        if not (next_character.isalpha() or next_character == "@"):
            self.position = start + 2
            if next_character in ESCAPED_TEXT:
                self.features["escape"] += 1
                return self._node(
                    "text",
                    self.raw[start : self.position],
                    value=ESCAPED_TEXT[next_character],
                )
            # Unknown control symbols are opaque for the same reason as unknown
            # control words: only the compiled PDF tells us what was visible.
            return self._node(
                "opaque",
                self.raw[start : self.position],
                command=next_character,
                opaque_role="unknown_command",
            )

        cursor = start + 1
        while cursor < self.length and (
            self.raw[cursor].isalpha() or self.raw[cursor] == "@"
        ):
            cursor += 1
        command = self.raw[start + 1 : cursor]
        if cursor < self.length and self.raw[cursor] == "*":
            command += "*"
            cursor += 1

        if command.rstrip("*") == "verb":
            return self._parse_verb(start, cursor)

        argument_start = _skip_space(self.raw, cursor)
        if command in STYLE_COMMANDS:
            if argument_start >= self.length or self.raw[argument_start] != "{":
                raise InlineParseError(
                    f"\\{command} requires a braced argument at offset {start}"
                )
            self.position = argument_start + 1
            children = _clean_children(
                self._parse_sequence(stop="}"), trim_edges=False
            )
            return self._node(
                STYLE_COMMANDS[command],
                self.raw[start : self.position],
                children=children,
                command=command,
            )

        role = _opaque_role(command)
        end, _argument_count, braced_argument_count = _consume_opaque_arguments(
            self.raw, argument_start
        )
        requires_braced_argument = (
            command in REQUIRED_OPAQUE_ARGUMENTS or command.startswith("cite")
        )
        if requires_braced_argument and braced_argument_count == 0:
            raise InlineParseError(
                f"\\{command} requires a braced argument at offset {start}"
            )
        self.position = end
        return self._node(
            "opaque",
            self.raw[start:end],
            command=command,
            opaque_role=role,
        )


def _is_escaped(value: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and value[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def _clean_math_body(value: str) -> str:
    """Remove layout-only edge whitespace without breaking TeX ``\\ ``.

    A trailing space immediately preceded by an odd number of backslashes is a
    TeX control-space token.  Plain ``str.strip()`` would remove that space and
    leave a dangling backslash which then escapes the Markdown closing dollar.
    Leading whitespace cannot be the whitespace half of ``\\ ``; at the right
    edge we trim only until the first control-space is reached.
    """

    start = 0
    while start < len(value) and value[start].isspace():
        start += 1
    end = len(value)
    while end > start and value[end - 1].isspace():
        if _is_escaped(value, end - 1):
            break
        end -= 1
    return value[start:end]


def _skip_space(value: str, position: int) -> int:
    while position < len(value) and value[position].isspace():
        position += 1
    return position


def _legacy_em_content_start(value: str, position: int) -> int | None:
    """Recognize only the balanced-group declaration form ``{\\em text}``."""

    if not value.startswith(r"\em", position):
        return None
    command_end = position + len(r"\em")
    if command_end >= len(value) or not value[command_end].isspace():
        return None
    return _skip_space(value, command_end)


def _scan_balanced(value: str, start: int, opening: str, closing: str) -> int:
    """Return the first position after a balanced group starting at ``start``."""

    if start >= len(value) or value[start] != opening:
        raise InlineParseError(f"expected {opening!r} at offset {start}")
    depth = 0
    cursor = start
    while cursor < len(value):
        character = value[cursor]
        if character == "%" and not _is_escaped(value, cursor):
            newline = value.find("\n", cursor + 1)
            if newline < 0:
                break
            cursor = newline + 1
            continue
        if character == "\\":
            if value.startswith(r"\verb", cursor):
                command_end = cursor + len(r"\verb")
                if command_end < len(value) and value[command_end] == "*":
                    command_end += 1
                if command_end < len(value) and not (
                    value[command_end].isalpha() or value[command_end] == "@"
                ):
                    delimiter = value[command_end]
                    if delimiter.isspace() or delimiter in "{}\\":
                        raise InlineParseError(
                            f"unsafe \\verb delimiter at offset {command_end}"
                        )
                    verb_end = value.find(delimiter, command_end + 1)
                    if verb_end < 0:
                        raise InlineParseError(
                            f"unclosed \\verb at offset {cursor}"
                        )
                    cursor = verb_end + 1
                    continue
            cursor += 2
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    raise InlineParseError(f"unclosed {opening!r} group starting at offset {start}")


def extract_footnote_source(node: InlineNode) -> FootnoteSource | None:
    """Extract the unique braced body of an opaque footnote node.

    Optional ``[]`` arguments are retained structurally.  Multiple top-level
    braced arguments, trailing tokens, malformed comments/verbatim, and any
    unbalanced input fail closed with ``None``.
    """

    if not isinstance(node, InlineNode):
        raise TypeError("node must be an InlineNode")
    if (
        node.kind != "opaque"
        or node.opaque_role != "footnote"
        or node.command != "footnote"
    ):
        return None

    raw = node.raw
    command = r"\footnote"
    if not raw.startswith(command):
        return None
    cursor = len(command)
    if cursor < len(raw) and (raw[cursor].isalpha() or raw[cursor] in "@*"):
        return None
    cursor = _skip_space(raw, cursor)
    optional_arguments: list[str] = []
    try:
        while cursor < len(raw) and raw[cursor] == "[":
            end = _scan_balanced(raw, cursor, "[", "]")
            optional_arguments.append(raw[cursor + 1 : end - 1])
            cursor = _skip_space(raw, end)
        if cursor >= len(raw) or raw[cursor] != "{":
            return None
        body_end = _scan_balanced(raw, cursor, "{", "}")
    except InlineParseError:
        return None
    body_raw = raw[cursor + 1 : body_end - 1]
    if _skip_space(raw, body_end) != len(raw):
        return None
    return FootnoteSource(
        raw=raw,
        body_raw=body_raw,
        optional_arguments=tuple(optional_arguments),
    )


def extract_footnote_body(node: InlineNode) -> str | None:
    """Return a validated footnote body, or ``None`` when extraction is unsafe."""

    source = extract_footnote_source(node)
    return None if source is None else source.body_raw


def _consume_opaque_arguments(value: str, position: int) -> tuple[int, int, int]:
    """Consume adjacent balanced ``[]``/``{}`` arguments of an opaque macro."""

    cursor = position
    count = 0
    braced_count = 0
    while cursor < len(value) and value[cursor] in "[{":
        if value[cursor] == "[":
            cursor = _scan_balanced(value, cursor, "[", "]")
        else:
            cursor = _scan_balanced(value, cursor, "{", "}")
            braced_count += 1
        count += 1
        next_cursor = _skip_space(value, cursor)
        if next_cursor >= len(value) or value[next_cursor] not in "[{":
            break
        cursor = next_cursor
    # If there was no argument, TeX still gobbles delimiter whitespace after a
    # control word.  It belongs to the opaque node, not the following anchor.
    return (cursor if count else position), count, braced_count


def _opaque_role(command: str) -> str:
    if command in CITE_COMMANDS or command.startswith("cite"):
        return "citation"
    if command in REF_COMMANDS:
        return "reference"
    if command == "footnote":
        return "footnote"
    if command == "label":
        return "label"
    return "unknown_command"


def _tex_text_to_visible(value: str) -> str:
    """Normalize plain (already command-free) TeX text to a PDF-text anchor."""

    value = unicodedata.normalize("NFKC", value)
    value = value.replace("~", " ")
    value = value.replace("---", "—").replace("--", "–")
    value = value.replace("``", "“").replace("''", "”")
    return re.sub(r"\s+", " ", value)


def _clean_children(
    children: tuple[InlineNode, ...], *, trim_edges: bool
) -> tuple[InlineNode, ...]:
    """Coalesce adjacent text leaves and normalize boundary whitespace."""

    cleaned: list[InlineNode] = []
    for node in children:
        if node.kind == "text" and not node.value:
            continue
        if cleaned and node.kind == "text" and cleaned[-1].kind == "text":
            previous = cleaned.pop()
            cleaned.append(
                dataclasses.replace(
                    previous,
                    raw=previous.raw + node.raw,
                    value=previous.value + node.value,
                )
            )
        else:
            cleaned.append(node)
    if trim_edges and cleaned:
        if cleaned[0].kind == "text":
            cleaned[0] = dataclasses.replace(
                cleaned[0], value=cleaned[0].value.lstrip()
            )
        if cleaned and cleaned[-1].kind == "text":
            cleaned[-1] = dataclasses.replace(
                cleaned[-1], value=cleaned[-1].value.rstrip()
            )
        cleaned = [
            node for node in cleaned if node.kind != "text" or bool(node.value)
        ]
    return tuple(cleaned)


def parse_inline_plan(raw: str) -> InlinePlan:
    """Parse one LaTeX prose fragment without invoking TeX or a language model.

    Raises :class:`InlineParseError` for unbalanced groups/math and malformed
    supported commands.  Unknown commands are accepted as opaque nodes.
    """

    if not isinstance(raw, str):
        raise TypeError("raw must be a string")
    return _Parser(raw).parse()


def focus_inline_plan(plan: InlinePlan, *, context_characters: int = 120) -> InlinePlan:
    """Keep target features plus bounded literal context for PDF alignment.

    A source sentence may cross a PDF page or column boundary after its last
    inline feature.  Matching the entire sentence would then fail even though
    the target formula/style is fully visible.  This helper retains every
    top-level subtree containing math/strong/em/code, everything between those
    subtrees, and one bounded literal anchor on either side.  PDF text outside
    that focused span remains untouched by the caller.
    """

    if not isinstance(plan, InlinePlan):
        raise TypeError("plan must be an InlinePlan")
    if not isinstance(context_characters, int) or isinstance(context_characters, bool):
        raise TypeError("context_characters must be an integer")
    if context_characters < 1 or context_characters > 4096:
        raise ValueError("context_characters must be between 1 and 4096")

    children = list(plan.root.children)

    def has_target(node: InlineNode) -> bool:
        return node.kind in {"math", "strong", "em", "code"} or any(
            has_target(child) for child in node.children
        )

    target_indices = [index for index, node in enumerate(children) if has_target(node)]
    if not target_indices:
        return plan
    first = target_indices[0]
    last = target_indices[-1]
    start = first - 1 if first > 0 and children[first - 1].kind == "text" else first
    end = last + 2 if last + 1 < len(children) and children[last + 1].kind == "text" else last + 1
    selected = children[start:end]

    if selected and selected[0].kind == "text" and len(selected[0].value) > context_characters:
        value = selected[0].value[-context_characters:]
        whitespace = re.search(r"\s", value)
        if whitespace:
            value = value[whitespace.end() :]
        selected[0] = dataclasses.replace(selected[0], raw=value, value=value)
    if selected and selected[-1].kind == "text" and len(selected[-1].value) > context_characters:
        value = selected[-1].value[:context_characters]
        boundary = value.rfind(" ")
        if boundary > 0:
            value = value[:boundary]
        selected[-1] = dataclasses.replace(selected[-1], raw=value, value=value)

    root = dataclasses.replace(plan.root, raw="".join(node.raw for node in selected), children=tuple(selected))
    features: collections.Counter[str] = collections.Counter()
    for node in iter_inline_nodes(root):
        if node.kind == "root":
            continue
        features[node.kind] += 1
        if node.kind == "opaque" and node.opaque_role:
            features[node.opaque_role] += 1
    anchors = tuple(
        node.value
        for node in iter_inline_nodes(root)
        if node.kind == "text" and node.value and node.value.strip()
    )
    return InlinePlan(
        raw=root.raw,
        root=root,
        feature_counts=dict(sorted(features.items())),
        anchors=anchors,
    )


def _alphabetic_run_pattern(value: str) -> str:
    """Match one source word with at most one discretionary PDF hyphen.

    The previous implementation placed an independent optional branch between
    every pair of letters.  Long prose anchors then had exponentially many
    backtracking paths and one formula-dense page could spend minutes inside a
    single failed ``re.search``.  A rendered word can contain at most one
    line-break hyphen, so an explicit exact-or-one-break alternation is both
    faithful and bounded.
    """

    exact = re.escape(value)
    if len(value) < 2:
        return exact
    hyphen = r"[-‐‑‒–—]\s*"
    alternatives = [exact]
    alternatives.extend(
        re.escape(value[:index]) + hyphen + re.escape(value[index:])
        for index in range(1, len(value))
    )
    return "(?:" + "|".join(alternatives) + ")"


def _literal_pattern(value: str) -> str:
    pieces: list[str] = []
    cursor = 0
    while cursor < len(value):
        character = value[cursor]
        if character.isalpha():
            end = cursor + 1
            while end < len(value) and value[end].isalpha():
                end += 1
            pieces.append(_alphabetic_run_pattern(value[cursor:end]))
            cursor = end
            continue
        if character.isspace():
            while cursor < len(value) and value[cursor].isspace():
                cursor += 1
            pieces.append(r"\s+")
            continue
        if character == "—":
            pieces.append(r"(?:—|---)")
        elif character == "–":
            pieces.append(r"(?:–|--)")
        elif character == "“":
            pieces.append(r"(?:“|``|\")")
        elif character == "”":
            pieces.append(r"(?:”|''|\")")
        elif character == "'":
            pieces.append(r"(?:'|’|ʼ)")
        elif character == "-":
            # PDF font encodings sometimes omit a short hyphen adjacent to an
            # inline formula (for example ``$k$-bonacci`` -> ``kbonacci``).
            # The surrounding literal anchors still make the match specific.
            pieces.append(r"(?:[-‐‑‒–—])?")
        else:
            pieces.append(re.escape(character))
        cursor += 1
    return "".join(pieces)


def _repair_literal_hyphens(source: str, captured: str) -> str:
    """Repair two source-verifiable PDF text-layer hyphen artifacts.

    This mirrors :func:`_literal_pattern` against an already successful text
    capture.  It may (1) insert an ASCII ``-`` explicitly present in the source
    at an optional-hyphen transition, or (2) collapse ``alpha-hyphen-whitespace-
    alpha`` back to one continuous alphabetic source word.  A PDF hyphen with
    no following whitespace remains visible.  If the complete source/capture
    alignment cannot be reconstructed, the original capture is unchanged.

    Keeping this repair on ``text`` nodes is important: opaque macro captures
    (including footnote marks) remain entirely PDF-authoritative.
    """

    hyphens = "-‐‑‒–—"

    @functools.lru_cache(maxsize=None)
    def align(source_index: int, captured_index: int) -> str | None:
        if source_index == len(source):
            return "" if captured_index == len(captured) else None

        character = source[source_index]
        candidates: list[tuple[int, int, str]] = []

        if character.isalpha():
            source_end = source_index + 1
            while source_end < len(source) and source[source_end].isalpha():
                source_end += 1
            word = source[source_index:source_end]
            if captured.startswith(word, captured_index):
                candidates.append(
                    (source_end, captured_index + len(word), word)
                )
            for split in range(1, len(word)):
                prefix = word[:split]
                suffix = word[split:]
                if not captured.startswith(prefix, captured_index):
                    continue
                hyphen_index = captured_index + len(prefix)
                if (
                    hyphen_index >= len(captured)
                    or captured[hyphen_index] not in hyphens
                ):
                    continue
                suffix_index = hyphen_index + 1
                while suffix_index < len(captured) and captured[suffix_index].isspace():
                    suffix_index += 1
                if captured.startswith(suffix, suffix_index):
                    captured_end = suffix_index + len(suffix)
                    visible = captured[captured_index:captured_end]
                    if suffix_index > hyphen_index + 1:
                        visible = word
                    candidates.append(
                        (
                            source_end,
                            captured_end,
                            visible,
                        )
                    )
        elif character.isspace():
            source_end = source_index + 1
            while source_end < len(source) and source[source_end].isspace():
                source_end += 1
            captured_end = captured_index
            while captured_end < len(captured) and captured[captured_end].isspace():
                captured_end += 1
            if captured_end > captured_index:
                candidates.append(
                    (
                        source_end,
                        captured_end,
                        captured[captured_index:captured_end],
                    )
                )
        elif character == "-":
            # Prefer and preserve a PDF-visible hyphen; insert the source ASCII
            # hyphen only when the optional regex branch matched zero width.
            if captured_index < len(captured) and captured[captured_index] in hyphens:
                candidates.append(
                    (source_index + 1, captured_index + 1, captured[captured_index])
                )
            candidates.append((source_index + 1, captured_index, "-"))
        else:
            alternatives: tuple[str, ...]
            if character == "—":
                alternatives = ("—", "---")
            elif character == "–":
                alternatives = ("–", "--")
            elif character == "“":
                alternatives = ("“", "``", '"')
            elif character == "”":
                alternatives = ("”", "''", '"')
            elif character == "'":
                alternatives = ("'", "’", "ʼ")
            else:
                alternatives = (character,)
            for alternative in alternatives:
                if captured.startswith(alternative, captured_index):
                    candidates.append(
                        (
                            source_index + 1,
                            captured_index + len(alternative),
                            alternative,
                        )
                    )

        for next_source, next_captured, visible in candidates:
            remainder = align(next_source, next_captured)
            if remainder is not None:
                return visible + remainder
        return None

    repaired = align(0, 0)
    return captured if repaired is None else repaired


def _wildcard_limit(node: InlineNode, configured: int) -> int:
    if node.kind == "math":
        return configured
    if node.opaque_role == "label":
        return 0
    if node.opaque_role == "footnote":
        return min(configured, 16)
    return configured


def build_inline_regex(
    plan: InlinePlan, *, max_wildcard: int = DEFAULT_MAX_WILDCARD
) -> InlineRegex:
    """Build a named-group regex for locating ``plan`` in compiled PDF text.

    All variable spans use ``[\\s\\S]{0,N}?`` with a finite ``N``.  The
    pattern is intentionally unanchored so callers can either ``search`` a PDF
    line/paragraph or ``fullmatch`` a candidate substring.
    """

    if not isinstance(plan, InlinePlan):
        raise TypeError("plan must be an InlinePlan")
    if not isinstance(max_wildcard, int) or isinstance(max_wildcard, bool):
        raise TypeError("max_wildcard must be an integer")
    if max_wildcard < 0 or max_wildcard > MAX_CONFIGURED_WILDCARD:
        raise ValueError(
            f"max_wildcard must be between 0 and {MAX_CONFIGURED_WILDCARD}"
        )

    names: dict[int, str] = {}

    def emit(node: InlineNode) -> str:
        if node.kind == "root":
            return "".join(emit(child) for child in node.children)
        name = f"inline_{node.node_id:04d}"
        names[node.node_id] = name
        if node.kind == "text":
            body = _literal_pattern(node.value)
        elif node.kind in {"math", "opaque"}:
            limit = _wildcard_limit(node, max_wildcard)
            body = rf"[\s\S]{{0,{limit}}}?"
        elif node.kind in {"strong", "em", "code"}:
            body = "".join(emit(child) for child in node.children)
        else:  # Defensive: plans may only be built by this module.
            raise ValueError(f"unsupported inline node kind: {node.kind!r}")
        return f"(?P<{name}>{body})"

    pattern = emit(plan.root)
    return InlineRegex(
        pattern=pattern,
        compiled=re.compile(pattern),
        group_names=names,
        max_wildcard=max_wildcard,
    )


def _code_fence(value: str) -> str:
    if not value:
        return ""
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * (longest + 1)
    if value.startswith((" ", "`")) or value.endswith((" ", "`")):
        return f"{fence} {value} {fence}"
    return f"{fence}{value}{fence}"


def _markdown_style(value: str, delimiter: str) -> str:
    """Wrap content while keeping boundary whitespace outside the markers."""

    if not value:
        return ""
    leading_size = len(value) - len(value.lstrip())
    trailing_size = len(value) - len(value.rstrip())
    leading = value[:leading_size]
    trailing = value[len(value) - trailing_size :] if trailing_size else ""
    end = len(value) - trailing_size if trailing_size else len(value)
    core = value[leading_size:end]
    if not core:
        return value
    return f"{leading}{delimiter}{core}{delimiter}{trailing}"


def render_inline_match(
    plan: InlinePlan, match: Match[str], regex: InlineRegex
) -> str:
    """Recursively render a successful regex match as Markdown."""

    if not isinstance(plan, InlinePlan):
        raise TypeError("plan must be an InlinePlan")

    def captured(node: InlineNode) -> str:
        name = regex.group_names.get(node.node_id)
        if not name:
            return ""
        value = match.groupdict().get(name)
        return value or ""

    def render(node: InlineNode) -> str:
        if node.kind == "root":
            return "".join(render(child) for child in node.children)
        if node.kind == "text":
            return _repair_literal_hyphens(node.value, captured(node))
        if node.kind == "opaque":
            visible = captured(node)
            if node.opaque_role == "footnote" and re.fullmatch(r"[0-9]+", visible):
                return f"<sup>{visible}</sup>"
            return visible
        if node.kind == "math":
            return f"${node.value}$"
        if node.kind == "strong":
            return _markdown_style(
                "".join(render(child) for child in node.children), "**"
            )
        if node.kind == "em":
            return _markdown_style(
                "".join(render(child) for child in node.children), "*"
            )
        if node.kind == "code":
            # A code span cannot safely contain nested Markdown delimiters.  Its
            # own named capture is exactly the visible PDF substring.
            return _code_fence(captured(node))
        raise ValueError(f"unsupported inline node kind: {node.kind!r}")

    return render(plan.root)


def render_inline_source(plan: InlinePlan) -> str:
    """Render a focused plan directly when it contains no opaque commands.

    This is intentionally unavailable for citations, references, footnotes,
    labels, or unknown macros because only the compiled PDF is authoritative
    for their visible expansion.
    """

    if not isinstance(plan, InlinePlan):
        raise TypeError("plan must be an InlinePlan")
    if int(plan.feature_counts.get("opaque", 0)):
        raise ValueError("cannot source-render a plan containing opaque commands")

    def render(node: InlineNode) -> str:
        if node.kind == "root":
            return "".join(render(child) for child in node.children)
        if node.kind == "text":
            return node.value
        if node.kind == "math":
            return f"${node.value}$"
        if node.kind == "strong":
            return _markdown_style("".join(render(child) for child in node.children), "**")
        if node.kind == "em":
            return _markdown_style("".join(render(child) for child in node.children), "*")
        if node.kind == "code":
            return _code_fence("".join(render(child) for child in node.children))
        raise ValueError(f"cannot source-render node kind {node.kind!r}")

    return render(plan.root)


def build_text_anchor_regex(value: str) -> Pattern[str]:
    """Compile the same PDF-tolerant literal matcher used by inline plans."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if not value:
        raise ValueError("value must not be empty")
    return re.compile(_literal_pattern(value))


def apply_inline_plan(
    plan: InlinePlan,
    pdf_text: str,
    *,
    max_wildcard: int = DEFAULT_MAX_WILDCARD,
    fullmatch: bool = True,
) -> InlineRenderResult | None:
    """Align ``plan`` with PDF text and return deterministic Markdown.

    ``fullmatch=True`` is safest when the caller has already selected a PDF
    paragraph.  Set it to false to search for the source fragment inside a
    larger text-layer string.  A failed alignment returns ``None``.
    """

    if not isinstance(pdf_text, str):
        raise TypeError("pdf_text must be a string")
    regex = build_inline_regex(plan, max_wildcard=max_wildcard)
    match = regex.fullmatch(pdf_text) if fullmatch else regex.search(pdf_text)
    if match is None:
        return None
    return InlineRenderResult(
        markdown=render_inline_match(plan, match, regex),
        matched_text=match.group(0),
        span=(match.start(), match.end()),
        regex=regex,
    )


def render_footnote_body(
    body_raw: str,
    pdf_visible_body: str,
    *,
    max_wildcard: int = DEFAULT_MAX_WILDCARD,
) -> str | None:
    """Align a source footnote body with its number-free PDF text.

    The normal inline parser, bounded regex, and recursive renderer are reused,
    so source math and supported styles are restored while citation/reference
    expansions remain PDF-authoritative.  Structural or alignment failures
    return ``None`` rather than falling back to source-only rendering.
    """

    if not isinstance(body_raw, str):
        raise TypeError("body_raw must be a string")
    if not isinstance(pdf_visible_body, str):
        raise TypeError("pdf_visible_body must be a string")
    try:
        plan = parse_inline_plan(body_raw)
        result = apply_inline_plan(
            plan,
            pdf_visible_body,
            max_wildcard=max_wildcard,
            fullmatch=True,
        )
    except (InlineParseError, re.error, ValueError):
        return None
    return None if result is None else result.markdown


def summarize_inline_plan(plan: InlinePlan) -> dict[str, object]:
    """Return JSON-serializable feature, wildcard, and anchor statistics."""

    if not isinstance(plan, InlinePlan):
        raise TypeError("plan must be an InlinePlan")
    return {
        "features": dict(plan.feature_counts),
        "anchors": {
            "count": plan.anchor_count,
            "characters": plan.anchor_characters,
            "longest": max((len(value) for value in plan.anchors), default=0),
            "values": list(plan.anchors),
        },
        "wildcards": plan.wildcard_count,
        "nodes": sum(1 for _ in iter_inline_nodes(plan.root)) - 1,
    }


__all__ = [
    "DEFAULT_MAX_WILDCARD",
    "FootnoteSource",
    "InlineNode",
    "InlineParseError",
    "InlinePlan",
    "InlineRegex",
    "InlineRenderResult",
    "apply_inline_plan",
    "build_inline_regex",
    "build_text_anchor_regex",
    "extract_footnote_body",
    "extract_footnote_source",
    "focus_inline_plan",
    "iter_inline_nodes",
    "parse_inline_plan",
    "render_footnote_body",
    "render_inline_match",
    "render_inline_source",
    "summarize_inline_plan",
]
