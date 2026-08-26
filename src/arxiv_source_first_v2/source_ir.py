"""Source-first inline representation for ordinary-prose LaTeX.

This module is intentionally small and conservative.  It is not a TeX
interpreter and it never uses text extracted from a PDF to manufacture a
Markdown string.  Instead, it scans a source fragment into visible atoms.
Each atom keeps the source span that produced it, a source-order ordinal, its
semantic kind, the active style stack, and a Markdown fragment derived from
the source itself.

The representation is useful to a page mapper: a renderer/locator can attach
PDF page membership to atom ordinals, and :func:`reconstruct_page_markdown`
can then render any page from those atoms.  A style that starts on an earlier
page is opened at the beginning of the selected page and closed at its end,
so every page is independently balanced Markdown.

Unsupported TeX is represented by a visible ``opaque`` atom containing an
explicit source marker.  In particular, this module never substitutes text
observed in a PDF for a citation, reference, macro, or malformed construct.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Iterator, Sequence


OPAQUE_MARKER_PREFIX = "⟦opaque:"
OPAQUE_MARKER_SUFFIX = "⟧"

_STYLE_COMMANDS = {
    "textbf": "strong",
    "emph": "em",
    "textit": "em",
    "texttt": "code",
}

_ESCAPED_CHARS = {
    "&": "&",
    "%": "%",
    "$": "$",
    "#": "#",
    "_": "_",
    "{": "{",
    "}": "}",
    "~": "~",
    "^": "^",
    "\\": "\\",
}

# Match the conservative text escaping used by the existing inline renderer.
# In particular, do not escape ordinary punctuation such as ``.`` or ``-``:
# doing so makes source-derived prose noisy without protecting a Markdown
# construct in the middle of a paragraph.
_MARKDOWN_SPECIAL = set("\\`*_#$![]<>{}")


@dataclasses.dataclass(frozen=True)
class SourceAtom:
    """One visible source-derived unit.

    ``source_start`` and ``source_end`` are absolute offsets into the source
    file (the caller supplies ``source_base_offset`` to the parser).  The end
    offset is exclusive.  ``markdown_fragment`` is the atom content without
    style delimiters; ``style_stack`` is rendered by
    :func:`reconstruct_page_markdown`.

    Whitespace is represented as atoms as well.  This makes page membership
    and source order lossless at token boundaries while still allowing a
    caller to ignore whitespace atoms for geometry matching.
    """

    ordinal: int
    source_start: int
    source_end: int
    kind: str
    style_stack: tuple[str, ...]
    markdown_fragment: str
    raw_source: str
    visible_text: str

    @property
    def source_span(self) -> tuple[int, int]:
        return self.source_start, self.source_end

    @property
    def semantic(self) -> str:
        """Compatibility alias for callers that call ``kind`` semantic."""

        return self.kind

    @property
    def markdown(self) -> str:
        """The source-derived content fragment (style-free by design)."""

        return self.markdown_fragment

    @property
    def is_whitespace(self) -> bool:
        return self.kind == "whitespace"


@dataclasses.dataclass(frozen=True)
class _Token:
    kind: str
    start: int
    end: int
    raw: str
    value: str = ""
    children: tuple["_Token", ...] = ()
    style: str | None = None


class SourceParseError(ValueError):
    """Raised only for invalid parser arguments, not unsupported TeX.

    Unsupported source constructs are deliberately converted to opaque atoms
    so that a single macro does not make an otherwise useful source fragment
    disappear.
    """


def _opaque_marker(raw: str) -> str:
    return f"{OPAQUE_MARKER_PREFIX}{raw}{OPAQUE_MARKER_SUFFIX}"


def _escape_markdown_literal(value: str) -> str:
    """Escape Markdown punctuation while retaining source-visible text.

    TeX escapes are decoded before this function.  Escaping only the small
    set of punctuation that can change Markdown structure keeps normal prose
    readable and prevents source punctuation from becoming formatting.
    """

    out: list[str] = []
    for char in value:
        if char in _MARKDOWN_SPECIAL:
            out.append("\\")
        out.append(char)
    return "".join(out)


def _decode_plain_text(raw: str) -> str:
    """Decode TeX's literal escapes to the source-visible character stream."""

    result: list[str] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == "\\" and index + 1 < len(raw):
            escaped = raw[index + 1]
            if escaped in _ESCAPED_CHARS:
                result.append(_ESCAPED_CHARS[escaped])
                index += 2
                continue
            # A backslash followed by whitespace is a TeX control space.  It
            # is visible as one ordinary space, not as a source macro.
            if escaped.isspace():
                result.append(" ")
                index += 2
                continue
        # TeX collapses prose whitespace.  Preserve a single explicit space
        # here; page-level source reconstruction should not inherit source
        # line wrapping.
        if char.isspace() or char == "~":
            result.append(" ")
        else:
            result.append(char)
        index += 1
    return "".join(result)


def _decode_text(raw: str) -> str:
    """Convert a plain source run to a source-derived Markdown fragment."""

    return _escape_markdown_literal(_decode_plain_text(raw))


def _normalize_math(body: str) -> str:
    """Normalize only layout whitespace around an inline formula.

    Newlines/indentation around a formula are source layout, whereas ``\\ ``
    is a TeX control-space and is meaningful inside math.  Protect that pair
    while trimming the outer layout.
    """

    sentinel = "\x00"
    protected = body.replace("\\ ", sentinel)
    protected = protected.strip()
    return protected.replace(sentinel, "\\ ")


def _find_balanced_group(source: str, opening: int) -> int | None:
    """Return the exclusive end of a braced group, or ``None``."""

    if opening >= len(source) or source[opening] != "{":
        return None
    depth = 0
    index = opening
    while index < len(source):
        char = source[index]
        if char == "\\":
            # Escaped braces do not affect group balance.  A trailing slash is
            # handled as malformed by returning None.
            if index + 1 >= len(source):
                return None
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _find_math_end(source: str, start: int) -> tuple[int, str] | None:
    """Find a ``$...$`` or ``\\(...\\)`` inline formula."""

    if source.startswith("\\(", start):
        cursor = start + 2
        while cursor < len(source):
            if source.startswith("\\)", cursor):
                return cursor + 2, source[start + 2 : cursor]
            if source[cursor] == "\\" and cursor + 1 < len(source):
                cursor += 2
            else:
                cursor += 1
        return None
    if source[start] != "$" or source.startswith("$$", start):
        return None
    cursor = start + 1
    while cursor < len(source):
        if source[cursor] == "\\" and cursor + 1 < len(source):
            cursor += 2
            continue
        if source[cursor] == "$":
            return cursor + 1, source[start + 1 : cursor]
        cursor += 1
    return None


def _command_end(source: str, start: int) -> int:
    """Return the end of a TeX control sequence beginning at ``start``."""

    cursor = start + 1
    if cursor >= len(source):
        return cursor
    if source[cursor].isalpha() or source[cursor] == "@":
        cursor += 1
        while cursor < len(source) and (source[cursor].isalpha() or source[cursor] == "@"):
            cursor += 1
        # A space after a word command is consumed by TeX.  Keep it in the
        # opaque token only for source accounting; it produces no visible atom
        # when this is a supported style command.
        return cursor
    return cursor + 1


def _parse_sequence(
    source: str,
    start: int,
    end: int,
    style_stack: tuple[str, ...] = (),
) -> tuple[_Token, ...]:
    """Parse one balanced sequence into internal source tokens."""

    tokens: list[_Token] = []
    cursor = start
    text_start = cursor

    def flush_text(until: int) -> None:
        nonlocal text_start
        if until > text_start:
            raw = source[text_start:until]
            # Comments are TeX-invisible.  Keep visible text before/after a
            # comment as distinct tokens so source offsets remain truthful.
            parts: list[tuple[int, int]] = []
            part_start = text_start
            part_cursor = text_start
            while part_cursor < until:
                if source[part_cursor] == "%" and (part_cursor == 0 or source[part_cursor - 1] != "\\"):
                    if part_cursor > part_start:
                        parts.append((part_start, part_cursor))
                    newline = source.find("\n", part_cursor, until)
                    if newline < 0:
                        part_start = until
                        break
                    part_cursor = newline + 1
                    part_start = part_cursor
                    continue
                part_cursor += 1
            if part_start < until:
                parts.append((part_start, until))
            for part_begin, part_end in parts:
                value = source[part_begin:part_end]
                # Keep source whitespace and non-whitespace in separate atoms
                # later; at this level a plain run is enough.
                tokens.append(_Token("text", part_begin, part_end, value, value))
        text_start = until

    while cursor < end:
        char = source[cursor]
        # A raw braced group is transparent unless it is consumed as the body
        # of a supported style command below.
        if char == "{" and cursor >= start:
            group_end = _find_balanced_group(source, cursor)
            if group_end is None or group_end > end:
                flush_text(cursor)
                tokens.append(_Token("opaque", cursor, end, source[cursor:end]))
                return tuple(tokens)
            flush_text(cursor)
            inner = _parse_sequence(source, cursor + 1, group_end - 1, style_stack)
            tokens.extend(inner)
            cursor = group_end
            text_start = cursor
            continue

        math = _find_math_end(source, cursor)
        if math is not None:
            math_end, body = math
            if math_end <= end:
                flush_text(cursor)
                tokens.append(_Token("math", cursor, math_end, source[cursor:math_end], _normalize_math(body)))
                cursor = math_end
                text_start = cursor
                continue

        if char == "\\":
            command_end = _command_end(source, cursor)
            command = source[cursor + 1 : command_end]
            command_name = command
            if command_name in _STYLE_COMMANDS:
                # TeX permits whitespace between a command and its argument,
                # but a supported style call must have a braced body.  The
                # whitespace belongs to the command's source span.
                argument_start = command_end
                while argument_start < end and source[argument_start] in " \t":
                    argument_start += 1
                group_end = _find_balanced_group(source, argument_start)
                if group_end is not None and group_end <= end:
                    flush_text(cursor)
                    inner = _parse_sequence(
                        source,
                        argument_start + 1,
                        group_end - 1,
                        style_stack + (_STYLE_COMMANDS[command_name],),
                    )
                    tokens.extend(inner)
                    cursor = group_end
                    text_start = cursor
                    continue
                # A malformed supported style command is opaque as a local
                # marker; no PDF text is copied into the output.
                flush_text(cursor)
                opaque_end = min(end, argument_start if argument_start > command_end else command_end)
                opaque_finish = max(opaque_end, cursor + 1)
                tokens.append(_Token("opaque", cursor, opaque_finish, source[cursor:opaque_finish]))
                cursor = max(opaque_end, cursor + 1)
                text_start = cursor
                continue

            # Escaped punctuation is plain source text and can be combined
            # with surrounding text only at the flattening stage.
            if command in _ESCAPED_CHARS:
                cursor = command_end
                continue

            # Unknown command: consume one balanced braced argument, if any,
            # into one opaque token.  Optional arguments are included too;
            # they are not interpreted because their PDF expansion is not
            # source-deterministic here.
            flush_text(cursor)
            opaque_end = command_end
            probe = opaque_end
            while probe < end and source[probe] in " \t":
                probe += 1
            if probe < end and source[probe] == "[":
                optional_end = source.find("]", probe + 1, end)
                if optional_end < 0:
                    opaque_end = end
                else:
                    opaque_end = optional_end + 1
                    probe = opaque_end
            if probe < end and source[probe] == "{":
                group_end = _find_balanced_group(source, probe)
                opaque_end = end if group_end is None or group_end > end else group_end
            tokens.append(_Token("opaque", cursor, opaque_end, source[cursor:opaque_end]))
            cursor = max(opaque_end, cursor + 1)
            text_start = cursor
            continue

        # ``$$`` and display-math delimiters are not ordinary inline math.
        # Mark the delimiter/local construct opaque instead of pretending it
        # is ordinary text.
        if char == "$" and source.startswith("$$", cursor):
            flush_text(cursor)
            close = source.find("$$", cursor + 2, end)
            opaque_end = end if close < 0 else close + 2
            tokens.append(_Token("opaque", cursor, opaque_end, source[cursor:opaque_end]))
            cursor = max(opaque_end, cursor + 1)
            text_start = cursor
            continue

        cursor += 1

    flush_text(end)
    return tuple(tokens)


def _split_text_token(token: _Token) -> Iterator[_Token]:
    """Split plain text into whitespace/non-whitespace source atoms."""

    raw = token.raw
    cursor = 0
    for match in re.finditer(r"\s+|\S+", raw, flags=re.DOTALL):
        begin, finish = match.span()
        piece = raw[begin:finish]
        yield _Token(
            "whitespace" if piece.isspace() else "text",
            token.start + begin,
            token.start + finish,
            piece,
            piece,
            style=token.style,
        )
        cursor = finish
    if cursor < len(raw):
        # Defensive only; ``finditer`` covers the complete string.
        yield _Token("text", token.start + cursor, token.end, raw[cursor:], raw[cursor:], style=token.style)


def _flatten_tokens(tokens: Sequence[_Token], inherited_style: tuple[str, ...] = ()) -> Iterator[tuple[_Token, tuple[str, ...]]]:
    for token in tokens:
        if token.kind == "text":
            # Style is attached to child tokens by the parser through the
            # nested parse call; inherited_style remains available for future
            # internal callers.
            stack = inherited_style + ((token.style,) if token.style else ())
            yield from ((piece, stack) for piece in _split_text_token(token))
        else:
            stack = inherited_style + ((token.style,) if token.style else ())
            yield token, stack


def _tokenize_source(source: str) -> Iterator[tuple[_Token, tuple[str, ...]]]:
    """Yield flattened tokens and style stacks in source order."""

    # ``_parse_sequence`` stores the active style on child tokens indirectly
    # by parsing nested content.  To retain that stack, parse recursively with
    # a dedicated flatten pass instead of relying on the currently-unused
    # ``style_stack`` argument.  The public scanner below handles this by
    # attaching style to every leaf as it descends.
    yield from _tokenize_sequence(source, 0, len(source), ())


def _tokenize_sequence(
    source: str,
    start: int,
    end: int,
    style_stack: tuple[str, ...],
) -> Iterator[tuple[_Token, tuple[str, ...]]]:
    """Recursive scanner that preserves styles and source spans."""

    cursor = start
    text_start = cursor

    def emit_text(until: int) -> Iterator[tuple[_Token, tuple[str, ...]]]:
        nonlocal text_start
        if until <= text_start:
            text_start = until
            return
        raw = source[text_start:until]
        part_start = text_start
        index = text_start
        while index < until:
            if source[index] == "%" and (index == 0 or source[index - 1] != "\\"):
                if index > part_start:
                    for piece in _split_text_token(_Token("text", part_start, index, source[part_start:index], source[part_start:index])):
                        yield piece, style_stack
                newline = source.find("\n", index, until)
                if newline < 0:
                    part_start = until
                    break
                part_start = newline + 1
                index = part_start
                continue
            index += 1
        if part_start < until:
            for piece in _split_text_token(_Token("text", part_start, until, source[part_start:until], source[part_start:until])):
                yield piece, style_stack
        text_start = until

    while cursor < end:
        if source[cursor] == "{" or source[cursor] == "}":
            if source[cursor] == "{":
                group_end = _find_balanced_group(source, cursor)
                if group_end is not None and group_end <= end:
                    yield from emit_text(cursor)
                    yield from _tokenize_sequence(source, cursor + 1, group_end - 1, style_stack)
                    cursor = group_end
                    text_start = cursor
                    continue
            # An unmatched brace is explicit opaque source.
            yield from emit_text(cursor)
            yield _Token("opaque", cursor, cursor + 1, source[cursor : cursor + 1]), style_stack
            cursor += 1
            text_start = cursor
            continue

        math = _find_math_end(source, cursor)
        if math is not None:
            math_end, body = math
            if math_end <= end:
                yield from emit_text(cursor)
                yield _Token("math", cursor, math_end, source[cursor:math_end], _normalize_math(body)), style_stack
                cursor = math_end
                text_start = cursor
                continue

        if source[cursor] == "\\":
            command_end = _command_end(source, cursor)
            command = source[cursor + 1 : command_end]
            if command in _STYLE_COMMANDS:
                argument_start = command_end
                while argument_start < end and source[argument_start] in " \t":
                    argument_start += 1
                group_end = _find_balanced_group(source, argument_start)
                if group_end is not None and group_end <= end:
                    yield from emit_text(cursor)
                    yield from _tokenize_sequence(
                        source,
                        argument_start + 1,
                        group_end - 1,
                        style_stack + (_STYLE_COMMANDS[command],),
                    )
                    cursor = group_end
                    text_start = cursor
                    continue
                yield from emit_text(cursor)
                opaque_end = max(command_end, cursor + 1)
                yield _Token("opaque", cursor, opaque_end, source[cursor:opaque_end]), style_stack
                cursor = opaque_end
                text_start = cursor
                continue
            if command == "verb":
                # ``\verb`` uses its next character as a delimiter and does
                # not take a braced argument.  Keep the complete local
                # construct opaque, including malformed/unclosed forms.
                yield from emit_text(cursor)
                delimiter_start = command_end
                if delimiter_start >= end:
                    opaque_end = end
                else:
                    delimiter = source[delimiter_start]
                    close = source.find(delimiter, delimiter_start + 1, end)
                    opaque_end = end if close < 0 else close + 1
                yield _Token("opaque", cursor, opaque_end, source[cursor:opaque_end]), style_stack
                cursor = max(opaque_end, cursor + 1)
                text_start = cursor
                continue
            if command in _ESCAPED_CHARS or (len(command) == 1 and command.isspace()):
                # Leave escaped punctuation in the surrounding plain token so
                # its source span and output remain contiguous.
                cursor = command_end
                continue
            yield from emit_text(cursor)
            opaque_end = command_end
            probe = opaque_end
            while probe < end and source[probe] in " \t":
                probe += 1
            if probe < end and source[probe] == "[":
                close = source.find("]", probe + 1, end)
                opaque_end = end if close < 0 else close + 1
                probe = opaque_end
            if probe < end and source[probe] == "{":
                group_end = _find_balanced_group(source, probe)
                opaque_end = end if group_end is None or group_end > end else group_end
            yield _Token("opaque", cursor, opaque_end, source[cursor:opaque_end]), style_stack
            cursor = max(opaque_end, cursor + 1)
            text_start = cursor
            continue

        if source[cursor] == "$" and source.startswith("$$", cursor):
            yield from emit_text(cursor)
            close = source.find("$$", cursor + 2, end)
            opaque_end = end if close < 0 else close + 2
            yield _Token("opaque", cursor, opaque_end, source[cursor:opaque_end]), style_stack
            cursor = max(opaque_end, cursor + 1)
            text_start = cursor
            continue
        if source[cursor] == "$":
            # An unmatched single dollar is not silently copied as prose: it
            # is an explicit local opaque marker, just like any other
            # malformed inline construct.
            yield from emit_text(cursor)
            yield _Token("opaque", cursor, cursor + 1, source[cursor : cursor + 1]), style_stack
            cursor += 1
            text_start = cursor
            continue
        cursor += 1
    yield from emit_text(end)


def build_source_atoms(source: str, source_base_offset: int = 0) -> tuple[SourceAtom, ...]:
    """Build source-derived visible atoms for one ordinary-prose fragment.

    The parser is deliberately fail-closed at the construct level.  A
    malformed or unknown macro is emitted as an opaque atom with a marker;
    neighboring ordinary text remains usable and retains exact offsets.
    """

    if not isinstance(source, str):
        raise SourceParseError("source must be a string")
    if not isinstance(source_base_offset, int) or isinstance(source_base_offset, bool):
        raise SourceParseError("source_base_offset must be an integer")
    atoms: list[SourceAtom] = []
    for ordinal, (token, style_stack) in enumerate(_tokenize_source(source)):
        if token.kind == "text":
            fragment = _decode_text(token.raw)
            visible = _decode_plain_text(token.raw)
        elif token.kind == "whitespace":
            # ``_decode_text`` intentionally turns all source line wrapping
            # into one space.  Keep the fragment readable but source-derived.
            fragment = _decode_text(token.raw)
            visible = _decode_plain_text(token.raw)
        elif token.kind == "math":
            fragment = f"${token.value}$"
            visible = token.value
        else:
            fragment = _opaque_marker(token.raw)
            visible = fragment
        atoms.append(
            SourceAtom(
                ordinal=ordinal,
                source_start=source_base_offset + token.start,
                source_end=source_base_offset + token.end,
                kind=token.kind,
                style_stack=tuple(style_stack),
                markdown_fragment=fragment,
                raw_source=token.raw,
                visible_text=visible,
            )
        )
    return tuple(atoms)


def parse_source_atoms(source: str, source_base_offset: int = 0) -> tuple[SourceAtom, ...]:
    """Alias emphasizing that the operation is a parser, not PDF extraction."""

    return build_source_atoms(source, source_base_offset)


def _delimiter(style: str) -> str:
    return {"strong": "**", "em": "*", "code": "`"}.get(style, "")


def _render_atom_content(atom: SourceAtom) -> str:
    return atom.markdown_fragment


def reconstruct_page_markdown(
    atoms: Sequence[SourceAtom],
    page_atom_ordinals: Iterable[int] | None = None,
) -> str:
    """Reconstruct balanced source Markdown for one page's atom ordinals.

    ``page_atom_ordinals`` may be any iterable of atom ordinals.  Atoms are
    always emitted in source order, making a geometry mapper's unordered hit
    list safe.  Gaps are allowed: styles are closed/reopened across a gap,
    which keeps each selected fragment independently balanced.
    """

    if page_atom_ordinals is None:
        selected = list(atoms)
    else:
        wanted = {int(value) for value in page_atom_ordinals}
        selected = [atom for atom in atoms if atom.ordinal in wanted]
    if not selected:
        return ""

    output: list[str] = []
    active: tuple[str, ...] = ()
    for atom in selected:
        target = tuple(atom.style_stack)
        common = 0
        while common < len(active) and common < len(target) and active[common] == target[common]:
            common += 1
        for style in reversed(active[common:]):
            delimiter = _delimiter(style)
            if delimiter:
                output.append(delimiter)
        for style in target[common:]:
            delimiter = _delimiter(style)
            if delimiter:
                output.append(delimiter)
        output.append(_render_atom_content(atom))
        active = target
    for style in reversed(active):
        delimiter = _delimiter(style)
        if delimiter:
            output.append(delimiter)
    return "".join(output)


def split_atoms_by_page(
    atoms: Sequence[SourceAtom],
    page_atom_ordinals: Iterable[Iterable[int]],
) -> tuple[str, ...]:
    """Reconstruct several pages in order from their atom ordinal sets."""

    return tuple(reconstruct_page_markdown(atoms, ordinals) for ordinals in page_atom_ordinals)


def atoms_to_markdown(atoms: Sequence[SourceAtom]) -> str:
    """Render all atoms as one balanced source-derived Markdown fragment."""

    return reconstruct_page_markdown(atoms)


__all__ = [
    "OPAQUE_MARKER_PREFIX",
    "OPAQUE_MARKER_SUFFIX",
    "SourceAtom",
    "SourceParseError",
    "atoms_to_markdown",
    "build_source_atoms",
    "parse_source_atoms",
    "reconstruct_page_markdown",
    "split_atoms_by_page",
]
