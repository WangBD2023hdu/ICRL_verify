"""Fail-closed, source-only LaTeX list serialization.

This module is intentionally independent from the page GT scripts.  It turns
the list-related fields carried by ``SourceParagraph``-like objects into a
small intermediate representation and then into Markdown.  No PDF text is
read here.  A list item may be represented by more than one source paragraph:
paragraphs with the same ``(list_id, environment, depth, ordinal)`` are
continuations of the same item and never receive another bullet/number.

The parser in :func:`parse_latex_list` is deliberately narrow.  It recognizes
literal ``itemize``, ``enumerate`` and ``description`` environments and literal
``\\item`` commands only.  Dynamic environments, unknown commands in list
control positions, malformed groups, and unbalanced nesting raise
``ListIRSafetyError`` instead of guessing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import collections
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


LIST_IR_SCHEMA_VERSION = 1
LIST_IR_CONTRACT = "arxiv_source_first_v2_source_list_markdown"
SUPPORTED_LIST_ENVIRONMENTS = frozenset({"itemize", "enumerate", "description"})


class ListIRError(ValueError):
    """Base class for list IR input errors."""


class ListIRSafetyError(ListIRError):
    """Raised when list semantics cannot be proven from source."""


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ListIRSafetyError(f"{name} must be a positive integer")
    return value


def _source_lines(value: Any, *, start: Any = None, end: Any = None) -> tuple[int, ...]:
    if value is None:
        if start is None or end is None:
            raise ListIRSafetyError("source_lines or start_line/end_line is required")
        start_i = _as_positive_int(start, "start_line")
        end_i = _as_positive_int(end, "end_line")
        if end_i < start_i:
            raise ListIRSafetyError("end_line must not precede start_line")
        return tuple(range(start_i, end_i + 1))
    if isinstance(value, (str, bytes)):
        raise ListIRSafetyError("source_lines must be an integer sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ListIRSafetyError("source_lines must be an integer sequence") from exc
    if not values:
        raise ListIRSafetyError("source_lines must not be empty")
    result = tuple(_as_positive_int(item, "source line") for item in values)
    if tuple(sorted(set(result))) != result:
        raise ListIRSafetyError("source_lines must be sorted and unique")
    return result


def _value(obj: object, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _relative_source(value: object) -> str:
    if value is None:
        raise ListIRSafetyError("source_file is required")
    text = str(value)
    if not text or "\x00" in text:
        raise ListIRSafetyError("source_file must be a non-empty path")
    return text


def _strip_tex_comment(line: str) -> str:
    escaped = False
    for index, char in enumerate(line):
        if char == "%" and not escaped:
            return line[:index]
        if char == "\\":
            escaped = not escaped
        else:
            escaped = False
    return line


def _balanced_group(source: str, opening: int, left: str, right: str) -> int | None:
    if opening >= len(source) or source[opening] != left:
        return None
    depth = 0
    index = opening
    while index < len(source):
        char = source[index]
        if char == "\\":
            if index + 1 >= len(source):
                return None
            index += 2
            continue
        if char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _static_label(value: str) -> str:
    """Validate a description label without interpreting dynamic TeX."""

    if not isinstance(value, str):
        raise ListIRSafetyError("description label must be a string")
    if not value.strip():
        raise ListIRSafetyError("description label must not be empty")
    # Literal escaped punctuation is safe; control words, groups, math and
    # list controls are not statically decidable by this module.
    index = 0
    while index < len(value):
        if value[index] == "\\":
            if index + 1 >= len(value) or value[index + 1].isalpha():
                raise ListIRSafetyError("dynamic TeX in description label")
            index += 2
            continue
        if value[index] in "{}$":
            raise ListIRSafetyError("dynamic TeX in description label")
        index += 1
    return value.strip()


def _parse_item_prefix(raw: str) -> tuple[str, str | None, str]:
    match = re.match(r"^\s*\\item", raw)
    if match is None:
        return raw, None, raw
    index = match.end()
    while index < len(raw) and raw[index].isspace():
        index += 1
    label: str | None = None
    if index < len(raw) and raw[index] == "[":
        end = _balanced_group(raw, index, "[", "]")
        if end is None:
            raise ListIRSafetyError("unbalanced optional item label")
        label = _static_label(raw[index + 1 : end - 1])
        index = end
    return raw, label, raw[index:]


@dataclasses.dataclass(frozen=True, slots=True)
class ListSourceParagraph:
    """A source paragraph carrying one list item's identity.

    The fields intentionally mirror the list fields on the existing
    ``scripts.build_arxiv_page_markdown_gt.SourceParagraph`` class.  The
    optional ``list_id`` distinguishes two separate lists which both start at
    ordinal one; source parsers in this module provide it automatically.
    """

    paragraph_id: str
    kind: str
    source_file: str | Path
    source_lines: tuple[int, ...]
    raw_latex: str
    list_environment: str
    item_depth: int
    item_ordinal: int
    list_id: str | None = None
    description_label: str | None = None

    def __post_init__(self) -> None:
        if not self.paragraph_id:
            raise ListIRSafetyError("paragraph_id must not be empty")
        if not isinstance(self.raw_latex, str) or not self.raw_latex.strip():
            raise ListIRSafetyError("raw_latex must be non-empty")
        if self.list_environment not in SUPPORTED_LIST_ENVIRONMENTS:
            raise ListIRSafetyError(f"unsupported list environment: {self.list_environment}")
        _source_lines(self.source_lines)
        _as_positive_int(self.item_depth, "item_depth")
        _as_positive_int(self.item_ordinal, "item_ordinal")
        _relative_source(self.source_file)
        if self.description_label is not None:
            _static_label(self.description_label)
        if self.list_id is not None and (not self.list_id or "\x00" in self.list_id):
            raise ListIRSafetyError("list_id must be a non-empty string when provided")

    @property
    def item_key(self) -> str:
        list_id = self.list_id or f"{self.source_file}:{self.item_depth}:{self.list_environment}"
        raw = f"{list_id}|{self.list_environment}|{self.item_depth}|{self.item_ordinal}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    @property
    def is_first_fragment(self) -> bool:
        return bool(re.match(r"^\s*\\item(?:\s|\[|$)", self.raw_latex))

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": LIST_IR_SCHEMA_VERSION,
            "contract": LIST_IR_CONTRACT,
            "paragraph_id": self.paragraph_id,
            "kind": self.kind,
            "source_file": str(self.source_file),
            "source_lines": list(self.source_lines),
            "raw_latex": self.raw_latex,
            "list_environment": self.list_environment,
            "item_depth": self.item_depth,
            "item_ordinal": self.item_ordinal,
            "list_id": self.list_id,
            "item_key": self.item_key,
            "description_label": self.description_label,
            "generation_source": "latex_source",
            "pdf_text_used": False,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SerializedListItem:
    """One Markdown list item, including source provenance."""

    item_key: str
    list_id: str
    environment: str
    depth: int
    ordinal: int
    marker: str
    description_label: str | None
    markdown: str
    paragraph_ids: tuple[str, ...]
    source_lines: tuple[int, ...]
    provenance: tuple[Mapping[str, Any], ...]

    @property
    def continuation_count(self) -> int:
        return max(0, len(self.paragraph_ids) - 1)

    def as_json(self) -> dict[str, Any]:
        return {
            "item_key": self.item_key,
            "list_id": self.list_id,
            "environment": self.environment,
            "depth": self.depth,
            "ordinal": self.ordinal,
            "marker": self.marker,
            "description_label": self.description_label,
            "markdown": self.markdown,
            "paragraph_ids": list(self.paragraph_ids),
            "source_lines": list(self.source_lines),
            "continuation_count": self.continuation_count,
            "provenance": [dict(item) for item in self.provenance],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ListSerializationResult:
    """Source-only Markdown plus the per-item IR used to produce it."""

    markdown: str
    items: tuple[SerializedListItem, ...]
    source_paragraphs: tuple[ListSourceParagraph, ...]
    provenance: Mapping[str, Any]

    @property
    def accepted(self) -> bool:
        return bool(self.items) and bool(self.markdown)

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": LIST_IR_SCHEMA_VERSION,
            "contract": LIST_IR_CONTRACT,
            "markdown": self.markdown,
            "items": [item.as_json() for item in self.items],
            "source_paragraphs": [item.as_json() for item in self.source_paragraphs],
            "provenance": dict(self.provenance),
        }


def coerce_source_paragraph(value: object) -> ListSourceParagraph:
    """Convert a mapping or SourceParagraph-like object without PDF fields."""

    if isinstance(value, ListSourceParagraph):
        return value
    raw = _value(value, "raw_latex", "raw_source")
    if not isinstance(raw, str):
        raise ListIRSafetyError("list paragraph raw_latex is required")
    environment = _value(value, "list_environment", "environment")
    kind = str(_value(value, "kind", default="list_item"))
    if environment is None:
        match = re.match(r"^\s*([A-Za-z]+)_item\b", kind)
        environment = match.group(1) if match else None
    if environment not in SUPPORTED_LIST_ENVIRONMENTS:
        raise ListIRSafetyError(f"unsupported or missing list environment: {environment}")
    lines = _source_lines(
        _value(value, "source_lines", "source_line_numbers"),
        start=_value(value, "start_line"),
        end=_value(value, "end_line"),
    )
    paragraph_id = _value(value, "paragraph_id", "source_paragraph_id")
    if not isinstance(paragraph_id, str) or not paragraph_id:
        raise ListIRSafetyError("list paragraph_id is required")
    source_file = _value(value, "source_file")
    depth = _value(value, "item_depth", "depth")
    ordinal = _value(value, "item_ordinal", "ordinal")
    if depth is None or ordinal is None:
        raise ListIRSafetyError("item_depth and item_ordinal are required")
    explicit_label = _value(value, "description_label", "item_label", default=None)
    _, parsed_label, _ = _parse_item_prefix(raw)
    if explicit_label is not None and parsed_label is not None and explicit_label != parsed_label:
        raise ListIRSafetyError("description label disagrees with raw_latex")
    label = explicit_label if explicit_label is not None else parsed_label
    if label is not None and environment != "description":
        raise ListIRSafetyError("optional item labels are supported only for description")
    return ListSourceParagraph(
        paragraph_id=paragraph_id,
        kind=kind,
        source_file=source_file,
        source_lines=lines,
        raw_latex=raw,
        list_environment=environment,
        item_depth=depth,
        item_ordinal=ordinal,
        list_id=_value(value, "list_id", "list_key", default=None),
        description_label=label,
    )


@dataclasses.dataclass
class _ListContext:
    environment: str
    depth: int
    list_id: str
    next_ordinal: int = 0
    active_ordinal: int | None = None


@dataclasses.dataclass
class _ListInstanceState:
    """State for one inferred list lane in execution order."""

    signature: tuple[str, str, int]
    list_id: str
    last_ordinal: int | None = None


def assign_list_instance_ids(
    paragraphs: Sequence[object],
) -> tuple[ListSourceParagraph, ...]:
    """Attach deterministic instance IDs to paragraphs lacking ``list_id``.

    ``SourceParagraph`` predates this module and has no list-instance field.
    This adapter uses execution order, source file, environment, depth, and
    the fixed item ordinal to recover one when it is absent.  A same-ordinal
    non-``\\item`` fragment is a continuation and keeps its current instance;
    a new ``\\item`` at an ordinal which is not increasing starts a new
    instance.  Returning to an outer depth after a nested list keeps the outer
    instance alive.  A continuation for which no active instance can be proved
    is rejected rather than guessed.
    """

    if not paragraphs:
        raise ListIRSafetyError("at least one list paragraph is required")
    records = tuple(coerce_source_paragraph(value) for value in paragraphs)
    active: dict[int, _ListInstanceState] = {}
    assigned: list[ListSourceParagraph] = []
    previous_depth: int | None = None
    instance_counter = 0

    for record in records:
        depth = record.item_depth
        if previous_depth is not None and depth > previous_depth + 1:
            raise ListIRSafetyError("list nesting depth skips a level")
        # A transition to a shallower level closes all child list lanes.  The
        # state at the returned depth remains available for the parent item.
        for closed_depth in tuple(active):
            if closed_depth > depth:
                del active[closed_depth]
        signature = (str(record.source_file), record.list_environment, depth)
        state = active.get(depth)
        first = record.is_first_fragment

        if record.list_id is not None:
            # Explicit source parser IDs are authoritative, but still require
            # a literal item to introduce a new explicit instance.
            list_id = record.list_id
            if state is None or state.list_id != list_id:
                if not first:
                    raise ListIRSafetyError(
                        "continuation has no provable explicit list instance"
                    )
                state = _ListInstanceState(signature, list_id)
                active[depth] = state
            elif state.signature != signature:
                raise ListIRSafetyError("explicit list_id changes source signature")
        else:
            starts_new = state is None or state.signature != signature
            if starts_new:
                if not first:
                    raise ListIRSafetyError(
                        "continuation follows a source/list boundary without list_id"
                    )
                instance_counter += 1
                list_id = f"auto-list-{instance_counter:06d}"
                state = _ListInstanceState(signature, list_id)
                active[depth] = state
            else:
                list_id = state.list_id
                previous_ordinal = state.last_ordinal
                if previous_ordinal is None:
                    if not first:
                        raise ListIRSafetyError("continuation precedes a first list item")
                elif record.item_ordinal == previous_ordinal:
                    if first:
                        # A literal new item at the same ordinal is the
                        # observable reset separating two list instances.
                        instance_counter += 1
                        list_id = f"auto-list-{instance_counter:06d}"
                        state = _ListInstanceState(signature, list_id)
                        active[depth] = state
                    # Otherwise this is a continuation of the same item.
                elif record.item_ordinal > previous_ordinal:
                    if not first:
                        raise ListIRSafetyError(
                            "continuation changes item ordinal without \\item"
                        )
                else:  # ordinal decreased/reset
                    if not first:
                        raise ListIRSafetyError(
                            "ordinal reset on a continuation is ambiguous"
                        )
                    instance_counter += 1
                    list_id = f"auto-list-{instance_counter:06d}"
                    state = _ListInstanceState(signature, list_id)
                    active[depth] = state

        if state is None:  # defensive; every branch above establishes state
            raise ListIRSafetyError("could not establish list instance")
        state.last_ordinal = record.item_ordinal
        assigned.append(dataclasses.replace(record, list_id=list_id))
        previous_depth = depth
    return tuple(assigned)


def parse_latex_list(
    source: str,
    *,
    source_file: str | Path = "<source>",
    start_line: int = 1,
    list_id_prefix: str = "list",
) -> tuple[ListSourceParagraph, ...]:
    """Parse one or more literal supported LaTeX lists into source records.

    Only static list control syntax is accepted.  Ordinary inline TeX is kept
    in ``raw_latex`` for the caller's renderer; it is not interpreted here.
    """

    if not isinstance(source, str) or not source.strip():
        raise ListIRSafetyError("source must be non-empty text")
    _as_positive_int(start_line, "start_line")
    if not isinstance(list_id_prefix, str) or not list_id_prefix:
        raise ListIRSafetyError("list_id_prefix must be non-empty")
    source_file_text = _relative_source(source_file)
    contexts: list[_ListContext] = []
    records: list[ListSourceParagraph] = []
    buffer: list[tuple[int, str]] = []
    active_key: tuple[str, int, int, str] | None = None
    block_counter = 0

    def active_item() -> tuple[str, int, int, str] | None:
        for context in reversed(contexts):
            if context.active_ordinal is not None:
                return (context.environment, context.depth, context.active_ordinal, context.list_id)
        return None

    def flush() -> None:
        nonlocal buffer, active_key
        if not buffer:
            return
        key = active_key or active_item()
        if key is None:
            raise ListIRSafetyError("visible text outside a list item")
        raw_body = "\n".join(value for _, value in buffer).strip()
        lines = tuple(dict.fromkeys(line for line, _ in buffer))
        buffer = []
        active_key = None
        if not raw_body:
            return
        environment, depth, ordinal, list_id = key
        # First fragments receive the literal item command; continuations do
        # not.  This lets the serializer prove that it must not add another
        # marker to a continuation.
        has_first = any(re.match(r"^\s*\\item(?:\s|\[|$)", value) for _, value in buffer)
        # ``buffer`` has just been cleared, so inspect the saved raw text.
        has_first = bool(re.match(r"^\s*\\item(?:\s|\[|$)", raw_body))
        paragraph_id = f"{list_id_prefix}-p{len(records) + 1:05d}"
        label = _parse_item_prefix(raw_body)[1] if has_first else None
        if label is not None and environment != "description":
            raise ListIRSafetyError("optional item label outside description")
        records.append(
            ListSourceParagraph(
                paragraph_id=paragraph_id,
                kind=f"{environment}_item",
                source_file=source_file_text,
                source_lines=lines,
                raw_latex=raw_body,
                list_environment=environment,
                item_depth=depth,
                item_ordinal=ordinal,
                list_id=list_id,
                description_label=label,
            )
        )

    def append(line_number: int, value: str) -> None:
        nonlocal active_key
        if not value.strip():
            flush()
            return
        key = active_item()
        if key is None:
            raise ListIRSafetyError("visible text outside a list item")
        if active_key is None:
            active_key = key
        elif active_key != key:
            flush()
            active_key = key
        buffer.append((line_number, value))

    def consume(line_number: int, text: str) -> None:
        nonlocal block_counter
        cursor = 0
        command_pattern = re.compile(r"\\(begin|end)\s*\{([^{}]*)\}|\\item\b")
        for match in command_pattern.finditer(text):
            before = text[cursor : match.start()]
            if before.strip():
                append(line_number, before)
            token = match.group(0)
            if match.group(1) is not None:
                action = match.group(1)
                environment = match.group(2)
                if environment not in SUPPORTED_LIST_ENVIRONMENTS:
                    raise ListIRSafetyError(f"unsupported or dynamic environment: {environment!r}")
                if action == "begin":
                    flush()
                    block_counter += 1
                    depth = len(contexts) + 1
                    contexts.append(
                        _ListContext(
                            environment=environment,
                            depth=depth,
                            list_id=f"{list_id_prefix}-{block_counter}",
                        )
                    )
                else:
                    if not contexts or contexts[-1].environment != environment:
                        raise ListIRSafetyError("unbalanced or mismatched list environment")
                    flush()
                    contexts.pop()
                cursor = match.end()
                continue

            if not contexts:
                raise ListIRSafetyError("item command outside a supported list")
            flush()
            context = contexts[-1]
            context.next_ordinal += 1
            context.active_ordinal = context.next_ordinal
            # Keep ``\\item`` and a static optional label in this fragment.
            item_end = match.end()
            if item_end < len(text) and text[item_end] == "[":
                label_end = _balanced_group(text, item_end, "[", "]")
                if label_end is None:
                    raise ListIRSafetyError("unbalanced optional item label")
                label = _static_label(text[item_end + 1 : label_end - 1])
                if context.environment != "description":
                    raise ListIRSafetyError("optional item label outside description")
                item_end = label_end
            append(line_number, text[match.start() : item_end])
            cursor = item_end
        tail = text[cursor:]
        if tail.strip():
            append(line_number, tail)

    for offset, raw_line in enumerate(source.splitlines(), start=start_line):
        line = _strip_tex_comment(raw_line)
        if re.search(
            r"\\(?:csname|endcsname|expandafter|if[a-zA-Z]*|else|fi|"
            r"def|gdef|edef|xdef|newenvironment|renewenvironment|let)\b",
            line,
        ):
            raise ListIRSafetyError("dynamic TeX list control construct")
        if line.strip():
            consume(offset, line)
        else:
            flush()
    flush()
    if contexts:
        raise ListIRSafetyError("unbalanced list environment at end of source")
    if not records:
        raise ListIRSafetyError("source contains no supported list items")
    return tuple(records)


def _default_render_inline(source: str) -> str:
    # Import lazily so this module remains usable with a caller-provided
    # serializer even when the larger source IR is not imported at startup.
    from arxiv_source_first_v2.source_ir import atoms_to_markdown, build_source_atoms

    return atoms_to_markdown(build_source_atoms(source))


def _clean_fragment(raw: str, *, first: bool) -> tuple[str, str | None]:
    if first:
        _, label, body = _parse_item_prefix(raw)
        if not body.strip():
            raise ListIRSafetyError("list item has no visible source body")
        return body.strip(), label
    if re.search(r"\\(?:item|begin|end)\b", raw):
        raise ListIRSafetyError("continuation contains a list control command")
    if not raw.strip():
        raise ListIRSafetyError("empty list continuation")
    return raw.strip(), None


def _indent_fragment(value: str, prefix: str) -> str:
    lines = value.splitlines() or [value]
    return "\n".join(prefix + line if line else prefix.rstrip() for line in lines)


def serialize_source_list(
    paragraphs: Sequence[object],
    *,
    render_inline: Callable[[str], str] | None = None,
) -> ListSerializationResult:
    """Serialize source list paragraphs into structured Markdown.

    Paragraphs must be in source execution order.  The first paragraph for an
    item must contain ``\\item``; later paragraphs with the same item key are
    continuations and are indented without a marker.  Item ordinals are never
    renumbered.
    """

    if not paragraphs:
        raise ListIRSafetyError("at least one list paragraph is required")
    records = assign_list_instance_ids(paragraphs)
    renderer = render_inline or _default_render_inline
    # Keep both a complete per-item group and the execution-order event list.
    # A parent item can reappear as a continuation after a nested child; the
    # group owns its final Markdown while events retain the child/parent order.
    grouped: dict[str, list[Any]] = {}
    group_order: list[str] = []
    events: list[tuple[str, ListSourceParagraph]] = []
    active_by_depth: dict[int, str] = {}
    previous_depth: int | None = None
    for record in records:
        key = record.item_key
        if previous_depth is not None and record.item_depth > previous_depth + 1:
            raise ListIRSafetyError("list nesting depth skips a level")
        for depth in tuple(active_by_depth):
            if depth > record.item_depth:
                del active_by_depth[depth]
        if record.is_first_fragment:
            if key in grouped:
                raise ListIRSafetyError("duplicate first fragment for one list item")
            grouped[key] = [record, [record]]
            group_order.append(key)
            active_by_depth[record.item_depth] = key
            for depth in tuple(active_by_depth):
                if depth > record.item_depth:
                    del active_by_depth[depth]
        else:
            if active_by_depth.get(record.item_depth) != key:
                raise ListIRSafetyError(
                    "continuation for an item is not contiguous in its list depth"
                )
            if key not in grouped:
                raise ListIRSafetyError("continuation has no first list fragment")
            grouped[key][1].append(record)
        events.append((key, record))
        previous_depth = record.item_depth
    serialized: list[SerializedListItem] = []
    rendered_lines_by_key: dict[str, list[str]] = {}
    for key in group_order:
        first_record, fragments = grouped[key]
        if not first_record.is_first_fragment:
            raise ListIRSafetyError("first fragment of an item must contain \\item")
        if first_record.list_environment != "description" and first_record.description_label is not None:
            raise ListIRSafetyError("description label on non-description item")
        if first_record.list_environment == "itemize":
            marker = "-"
        elif first_record.list_environment == "enumerate":
            marker = f"{first_record.item_ordinal}."
        else:
            marker = "-"
        label = first_record.description_label
        rendered_fragments: list[str] = []
        for index, fragment in enumerate(fragments):
            body, parsed_label = _clean_fragment(fragment.raw_latex, first=index == 0)
            if index == 0 and parsed_label is not None:
                if label is not None and label != parsed_label:
                    raise ListIRSafetyError("description label disagrees with source")
                label = parsed_label
            rendered = renderer(body)
            if not isinstance(rendered, str) or not rendered.strip():
                raise ListIRSafetyError("inline renderer returned empty/non-string output")
            rendered_fragments.append(rendered.strip())
        if label is not None:
            label = _static_label(label)
            label_rendered = renderer(label).strip()
            if not label_rendered:
                raise ListIRSafetyError("description label rendered empty")
            # Description labels are source text, not a synthesized sentence
            # prefix.  Preserve a source colon only when it was present in
            # ``label`` itself.
            rendered_fragments[0] = f"**{label_rendered}** {rendered_fragments[0]}"
        base_indent = "  " * (first_record.item_depth - 1)
        marker_prefix = f"{marker} "
        continuation_indent = base_indent + " " * len(marker_prefix)
        lines = [base_indent + marker_prefix + rendered_fragments[0]]
        for rendered in rendered_fragments[1:]:
            lines.append(_indent_fragment(rendered, continuation_indent))
        item_markdown = "\n".join(lines)
        rendered_lines_by_key[key] = lines
        provenance = tuple(
            {
                **fragment.as_json(),
                "item_key": first_record.item_key,
                "continuation": index > 0,
            }
            for index, fragment in enumerate(fragments)
        )
        serialized.append(
            SerializedListItem(
                item_key=first_record.item_key,
                list_id=str(first_record.list_id or f"{first_record.source_file}:{first_record.item_depth}:{first_record.list_environment}"),
                environment=first_record.list_environment,
                depth=first_record.item_depth,
                ordinal=first_record.item_ordinal,
                marker=marker,
                description_label=label,
                markdown=item_markdown,
                paragraph_ids=tuple(fragment.paragraph_id for fragment in fragments),
                source_lines=tuple(line for fragment in fragments for line in fragment.source_lines),
                provenance=provenance,
            )
        )
    if not serialized:
        raise ListIRSafetyError("no serializable list items")
    output_parts: list[str] = []
    event_fragment_index: dict[str, int] = collections.defaultdict(int)
    for key, record in events:
        fragment_index = event_fragment_index[key]
        event_fragment_index[key] += 1
        lines = rendered_lines_by_key[key]
        if fragment_index >= len(lines):
            raise ListIRSafetyError("list item event/render count mismatch")
        if output_parts:
            output_parts.append("\n")
        output_parts.append(lines[fragment_index])
    provenance = {
        "schema_version": LIST_IR_SCHEMA_VERSION,
        "contract": LIST_IR_CONTRACT,
        "generation_source": "latex_source",
        "pdf_text_used": False,
        "item_count": len(serialized),
        "continuation_count": sum(item.continuation_count for item in serialized),
    }
    return ListSerializationResult(
        markdown="".join(output_parts),
        items=tuple(serialized),
        source_paragraphs=records,
        provenance=provenance,
    )


__all__ = [
    "LIST_IR_CONTRACT",
    "LIST_IR_SCHEMA_VERSION",
    "SUPPORTED_LIST_ENVIRONMENTS",
    "ListIRError",
    "ListIRSafetyError",
    "ListSerializationResult",
    "ListSourceParagraph",
    "SerializedListItem",
    "assign_list_instance_ids",
    "coerce_source_paragraph",
    "parse_latex_list",
    "serialize_source_list",
]
