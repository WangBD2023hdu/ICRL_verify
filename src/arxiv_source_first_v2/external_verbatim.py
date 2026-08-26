"""Source-derived IR for literal external verbatim inputs.

The page GT pipeline must not use PDF-extracted text to recreate a code block.
This module instead scans already-selected execution-source occurrences for
literal ``\\verbatiminput{...}``/``\\VerbatimInput{...}`` calls, resolves the
target below a declared source root, and emits one record per original
external-source line.  A later SyncTeX or shipout tracer can therefore attach
page membership to individual lines without losing source provenance.

This is deliberately not a TeX interpreter.  Dynamic paths, path traversal,
unknown options, malformed calls, non-UTF-8 files, and targets outside the
source root fail closed.  TeX comments in the execution source are ignored;
percent signs and other comments in the external verbatim file remain literal
visible text.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


EXTERNAL_VERBATIM_SCHEMA_VERSION = 1
EXTERNAL_VERBATIM_CONTRACT = "arxiv_source_first_v2_external_verbatim_lines"
EXTERNAL_VERBATIM_KIND = "external_verbatim_line"

_COMMAND_NAMES = frozenset({"verbatiminput", "VerbatimInput"})
_INTEGER_OPTIONS = frozenset({"firstline", "lastline", "gobble"})
_SUPPORTED_OPTIONS = _INTEGER_OPTIONS | {"encoding"}
_UNSAFE_LITERAL_CHARACTERS = frozenset("\\{}[]#$%~^\x00")
_OPTION_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_BACKTICK_RUN = re.compile(r"`+")


class ExternalVerbatimError(ValueError):
    """Base class for external-verbatim IR failures."""


class ExternalVerbatimSafetyError(ExternalVerbatimError):
    """Raised when a call cannot be proven safe and source-derived."""

    def __init__(self, rejections: Sequence["ExternalVerbatimRejection"]):
        self.rejections = tuple(rejections)
        detail = "; ".join(
            f"{item.code} at {item.execution_source}:"
            f"{item.execution_source_line}: {item.message}"
            for item in self.rejections
        )
        super().__init__(detail or "external verbatim input failed closed")


@dataclasses.dataclass(frozen=True)
class ExternalVerbatimRejection:
    """A source-addressed call that was deliberately not expanded."""

    code: str
    message: str
    source_root: Path
    execution_source: Path
    execution_index: int
    execution_source_line: int
    execution_source_column: int
    execution_source_byte_start: int
    execution_source_byte_end: int
    raw_invocation: str

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": EXTERNAL_VERBATIM_SCHEMA_VERSION,
            "contract": EXTERNAL_VERBATIM_CONTRACT,
            "status": "rejected",
            "code": self.code,
            "message": self.message,
            "generation_source": "latex_source",
            "pdf_text_used": False,
            "execution": {
                "source_file": _relative_posix(self.execution_source, self.source_root),
                "execution_index": self.execution_index,
                "source_line": self.execution_source_line,
                "source_column_bytes": self.execution_source_column,
                "source_byte_start": self.execution_source_byte_start,
                "source_byte_end": self.execution_source_byte_end,
            },
            "raw_invocation": self.raw_invocation,
        }


@dataclasses.dataclass(frozen=True)
class ExternalVerbatimLineRecord:
    """One visible line read directly from an external source file.

    ``raw_text`` is the original decoded line before a static ``gobble``
    option.  ``visible_text`` and ``markdown_fragment`` are the source-derived
    line after that option.  Line endings are excluded; empty physical lines
    are represented by records whose text is ``""``.
    """

    record_id: str
    call_id: str
    source_root: Path
    command_name: str
    execution_index: int
    call_ordinal: int
    execution_source: Path
    execution_source_line: int
    execution_source_column: int
    execution_source_byte_start: int
    execution_source_byte_end: int
    raw_invocation: str
    requested_path: str
    options: tuple[tuple[str, str], ...]
    external_source: Path
    external_source_line: int
    external_source_byte_start: int
    external_source_byte_end: int
    raw_text: str
    visible_text: str
    raw_line_sha256: str

    @property
    def kind(self) -> str:
        return EXTERNAL_VERBATIM_KIND

    @property
    def markdown_fragment(self) -> str:
        """Text to place inside the enclosing fenced code block."""

        return self.visible_text

    @property
    def markdown(self) -> str:
        return self.markdown_fragment

    @property
    def provenance(self) -> dict[str, Any]:
        """Strict source-only provenance suitable for a JSONL ledger."""

        return {
            "schema_version": EXTERNAL_VERBATIM_SCHEMA_VERSION,
            "contract": EXTERNAL_VERBATIM_CONTRACT,
            "kind": EXTERNAL_VERBATIM_KIND,
            "record_id": self.record_id,
            "call_id": self.call_id,
            "generation_source": "latex_source",
            "pdf_text_used": False,
            "execution": {
                "source_file": _relative_posix(self.execution_source, self.source_root),
                "execution_index": self.execution_index,
                "call_ordinal": self.call_ordinal,
                "source_line": self.execution_source_line,
                "source_column_bytes": self.execution_source_column,
                "source_byte_start": self.execution_source_byte_start,
                "source_byte_end": self.execution_source_byte_end,
                "command": self.command_name,
                "raw_invocation_sha256": _sha256_text(self.raw_invocation),
            },
            "external": {
                "source_file": _relative_posix(self.external_source, self.source_root),
                "requested_path": self.requested_path,
                "source_line": self.external_source_line,
                "source_byte_start": self.external_source_byte_start,
                "source_byte_end": self.external_source_byte_end,
                "raw_line_sha256": self.raw_line_sha256,
            },
            "options": dict(self.options),
        }

    def as_json(self) -> dict[str, Any]:
        return {
            **self.provenance,
            "raw_text": self.raw_text,
            "visible_text": self.visible_text,
            "markdown_fragment": self.markdown_fragment,
        }


@dataclasses.dataclass(frozen=True)
class ExternalVerbatimBlock:
    """One literal invocation and all selected external line records."""

    call_id: str
    source_root: Path
    command_name: str
    execution_index: int
    call_ordinal: int
    execution_source: Path
    execution_source_line: int
    execution_source_column: int
    execution_source_byte_start: int
    execution_source_byte_end: int
    raw_invocation: str
    requested_path: str
    options: tuple[tuple[str, str], ...]
    external_source: Path
    records: tuple[ExternalVerbatimLineRecord, ...]

    @property
    def visible_text(self) -> str:
        return "\n".join(record.visible_text for record in self.records)

    @property
    def fenced_markdown(self) -> str:
        return render_fenced_code(self.records)

    @property
    def markdown(self) -> str:
        return self.fenced_markdown

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": EXTERNAL_VERBATIM_SCHEMA_VERSION,
            "contract": EXTERNAL_VERBATIM_CONTRACT,
            "kind": "external_verbatim_block",
            "call_id": self.call_id,
            "generation_source": "latex_source",
            "pdf_text_used": False,
            "execution": {
                "source_file": _relative_posix(self.execution_source, self.source_root),
                "execution_index": self.execution_index,
                "call_ordinal": self.call_ordinal,
                "source_line": self.execution_source_line,
                "source_column_bytes": self.execution_source_column,
                "source_byte_start": self.execution_source_byte_start,
                "source_byte_end": self.execution_source_byte_end,
                "command": self.command_name,
                "raw_invocation_sha256": _sha256_text(self.raw_invocation),
            },
            "external": {
                "source_file": _relative_posix(self.external_source, self.source_root),
                "requested_path": self.requested_path,
                "selected_source_lines": [
                    record.external_source_line for record in self.records
                ],
            },
            "options": dict(self.options),
            "line_record_ids": [record.record_id for record in self.records],
        }

    def as_json(self) -> dict[str, Any]:
        return {
            **self.provenance,
            "visible_text": self.visible_text,
            "fenced_markdown": self.fenced_markdown,
            "records": [record.as_json() for record in self.records],
        }


@dataclasses.dataclass(frozen=True)
class ExternalVerbatimIR:
    """All literal external-verbatim calls in supplied execution order."""

    source_root: Path
    blocks: tuple[ExternalVerbatimBlock, ...]
    rejections: tuple[ExternalVerbatimRejection, ...] = ()

    @property
    def records(self) -> tuple[ExternalVerbatimLineRecord, ...]:
        return tuple(record for block in self.blocks for record in block.records)

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": EXTERNAL_VERBATIM_SCHEMA_VERSION,
            "contract": EXTERNAL_VERBATIM_CONTRACT,
            "status": "passed" if not self.rejections else "partial",
            "generation_source": "latex_source",
            "pdf_text_used": False,
            "source_root": str(self.source_root),
            "blocks": [block.as_json() for block in self.blocks],
            "rejections": [item.as_json() for item in self.rejections],
            "summary": {
                "blocks": len(self.blocks),
                "line_records": len(self.records),
                "rejections": len(self.rejections),
            },
        }


@dataclasses.dataclass(frozen=True)
class _Invocation:
    command_name: str
    start: int
    end: int
    line: int
    column: int
    raw: bytes
    option_bytes: bytes | None
    path_bytes: bytes | None
    parse_error: tuple[str, str] | None = None


@dataclasses.dataclass(frozen=True)
class _ExternalLine:
    number: int
    byte_start: int
    byte_end: int
    raw_bytes: bytes
    text: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative_posix(path: Path, source_root: Path) -> str:
    return path.relative_to(source_root).as_posix()


def _is_command_letter(value: int) -> bool:
    return 65 <= value <= 90 or 97 <= value <= 122 or value == 64


def _unescaped_percent(data: bytes, index: int) -> bool:
    slashes = 0
    cursor = index - 1
    while cursor >= 0 and data[cursor] == 0x5C:
        slashes += 1
        cursor -= 1
    return slashes % 2 == 0


def _skip_ascii_space(data: bytes, cursor: int) -> int:
    while cursor < len(data) and data[cursor] in b" \t\r\n":
        cursor += 1
    return cursor


def _source_line_column(data: bytes, offset: int) -> tuple[int, int]:
    line = data.count(b"\n", 0, offset) + 1
    previous = data.rfind(b"\n", 0, offset)
    return line, offset if previous < 0 else offset - previous - 1


def _closing_square_bracket(data: bytes, opening: int) -> int | None:
    """Find a simple option-list end while retaining unsafe bytes to reject."""

    cursor = opening + 1
    brace_depth = 0
    while cursor < len(data):
        value = data[cursor]
        if value == 0x7B:  # ``{``
            brace_depth += 1
        elif value == 0x7D and brace_depth:
            brace_depth -= 1
        elif value == 0x5D and brace_depth == 0:  # ``]``
            return cursor
        cursor += 1
    return None


def _closing_group(data: bytes, opening: int) -> int | None:
    if opening >= len(data) or data[opening] != 0x7B:
        return None
    depth = 0
    cursor = opening
    while cursor < len(data):
        value = data[cursor]
        if value == 0x5C and cursor + 1 < len(data):
            # Keep escaped/dynamic content in the raw group, but do not let an
            # escaped brace change syntactic balance.  Literal validation will
            # reject the backslash before target resolution.
            cursor += 2
            continue
        if value == 0x7B:
            depth += 1
        elif value == 0x7D:
            depth -= 1
            if depth == 0:
                return cursor
        cursor += 1
    return None


def _scan_invocations(data: bytes) -> tuple[_Invocation, ...]:
    invocations: list[_Invocation] = []
    cursor = 0
    while cursor < len(data):
        value = data[cursor]
        if value == 0x25 and _unescaped_percent(data, cursor):
            newline = data.find(b"\n", cursor + 1)
            cursor = len(data) if newline < 0 else newline + 1
            continue
        if value != 0x5C:
            cursor += 1
            continue

        start = cursor
        name_start = cursor + 1
        name_end = name_start
        while name_end < len(data) and _is_command_letter(data[name_end]):
            name_end += 1
        if name_end == name_start:
            cursor += 2
            continue
        try:
            name = data[name_start:name_end].decode("ascii")
        except UnicodeDecodeError:
            cursor = name_end
            continue
        if name not in _COMMAND_NAMES:
            cursor = name_end
            continue

        line, column = _source_line_column(data, start)
        parse_cursor = _skip_ascii_space(data, name_end)
        option_bytes: bytes | None = None
        if parse_cursor < len(data) and data[parse_cursor] == 0x5B:
            option_end = _closing_square_bracket(data, parse_cursor)
            if option_end is None:
                invocations.append(
                    _Invocation(
                        name,
                        start,
                        len(data),
                        line,
                        column,
                        data[start:],
                        None,
                        None,
                        ("malformed_invocation", "unterminated option list"),
                    )
                )
                break
            option_bytes = data[parse_cursor + 1 : option_end]
            parse_cursor = _skip_ascii_space(data, option_end + 1)

        if parse_cursor >= len(data) or data[parse_cursor] != 0x7B:
            end = min(len(data), max(parse_cursor + 1, name_end))
            invocations.append(
                _Invocation(
                    name,
                    start,
                    end,
                    line,
                    column,
                    data[start:end],
                    option_bytes,
                    None,
                    ("dynamic_path", "argument is not one literal braced path"),
                )
            )
            cursor = end
            continue

        group_end = _closing_group(data, parse_cursor)
        if group_end is None:
            invocations.append(
                _Invocation(
                    name,
                    start,
                    len(data),
                    line,
                    column,
                    data[start:],
                    option_bytes,
                    None,
                    ("malformed_invocation", "unterminated path group"),
                )
            )
            break
        end = group_end + 1
        invocations.append(
            _Invocation(
                name,
                start,
                end,
                line,
                column,
                data[start:end],
                option_bytes,
                data[parse_cursor + 1 : group_end],
            )
        )
        cursor = end
    return tuple(invocations)


def _parse_options(raw: bytes | None) -> tuple[tuple[str, str], ...]:
    if raw is None or not raw.strip():
        return ()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ExternalVerbatimError("option list is not static ASCII") from error
    if any(char in _UNSAFE_LITERAL_CHARACTERS or ord(char) < 0x20 for char in text):
        raise ExternalVerbatimError("option list contains dynamic or unsafe syntax")

    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item or "=" not in item:
            raise ExternalVerbatimError("every supported option must be key=value")
        raw_key, raw_value = item.split("=", 1)
        key = raw_key.strip().lower()
        value = raw_value.strip()
        if not _OPTION_KEY.fullmatch(key) or key not in _SUPPORTED_OPTIONS:
            raise ExternalVerbatimError(f"unsupported option: {raw_key.strip()!r}")
        if key in seen:
            raise ExternalVerbatimError(f"duplicate option: {key}")
        seen.add(key)
        if key in _INTEGER_OPTIONS:
            if not value.isascii() or not value.isdigit():
                raise ExternalVerbatimError(f"{key} must be a static integer")
            number = int(value)
            minimum = 0 if key == "gobble" else 1
            if number < minimum:
                raise ExternalVerbatimError(f"{key} must be >= {minimum}")
        elif key == "encoding" and value.lower() not in {"utf8", "utf-8"}:
            raise ExternalVerbatimError("only UTF-8 external verbatim input is supported")
        parsed.append((key, value))

    values = dict(parsed)
    if "firstline" in values and "lastline" in values:
        if int(values["lastline"]) < int(values["firstline"]):
            raise ExternalVerbatimError("lastline precedes firstline")
    return tuple(parsed)


def _literal_path(raw: bytes) -> str:
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExternalVerbatimError("path is not valid UTF-8") from error
    if not value or value != value.strip():
        raise ExternalVerbatimError("path must be a non-empty literal without edge whitespace")
    if any(char in _UNSAFE_LITERAL_CHARACTERS or ord(char) < 0x20 for char in value):
        raise ExternalVerbatimError("path contains dynamic or unsafe TeX syntax")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or value.startswith("~"):
        raise ExternalVerbatimError("path traversal and absolute paths are forbidden")
    if not pure.parts or any(part in {"", "."} for part in pure.parts):
        raise ExternalVerbatimError("path is not a normalized relative source path")
    return value


def _resolve_under_root(root: Path, relative: str) -> Path:
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ExternalVerbatimError("resolved path escapes source root") from error
    if not candidate.is_file():
        raise ExternalVerbatimError("external verbatim target is not a regular file")
    return candidate


def _external_lines(data: bytes) -> tuple[_ExternalLine, ...]:
    if not data:
        return ()
    raw_parts = data.split(b"\n")
    if raw_parts and raw_parts[-1] == b"":
        raw_parts.pop()
    lines: list[_ExternalLine] = []
    byte_start = 0
    for number, raw_with_cr in enumerate(raw_parts, start=1):
        raw = raw_with_cr[:-1] if raw_with_cr.endswith(b"\r") else raw_with_cr
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ExternalVerbatimError(
                f"external source line {number} is not valid UTF-8"
            ) from error
        lines.append(
            _ExternalLine(
                number=number,
                byte_start=byte_start,
                byte_end=byte_start + len(raw),
                raw_bytes=raw,
                text=text,
            )
        )
        byte_start += len(raw_with_cr) + 1
    return tuple(lines)


def _selected_lines(
    lines: Sequence[_ExternalLine], options: Mapping[str, str]
) -> tuple[_ExternalLine, ...]:
    first = int(options.get("firstline", "1"))
    last = int(options.get("lastline", str(len(lines))))
    return tuple(line for line in lines if first <= line.number <= last)


def _rejection(
    *,
    code: str,
    message: str,
    root: Path,
    execution_source: Path,
    execution_index: int,
    invocation: _Invocation,
) -> ExternalVerbatimRejection:
    return ExternalVerbatimRejection(
        code=code,
        message=message,
        source_root=root,
        execution_source=execution_source,
        execution_index=execution_index,
        execution_source_line=invocation.line,
        execution_source_column=invocation.column,
        execution_source_byte_start=invocation.start,
        execution_source_byte_end=invocation.end,
        raw_invocation=invocation.raw.decode("utf-8", errors="replace"),
    )


def render_fenced_code(
    records: Iterable[ExternalVerbatimLineRecord], *, language: str = "text"
) -> str:
    """Render line records as one balanced Markdown fenced code block.

    The fence grows beyond any literal backtick run in the source.  The
    language is metadata, not recovered visible content; it is restricted to
    a conservative Markdown info-string token.
    """

    if not re.fullmatch(r"[A-Za-z0-9_+.-]*", language):
        raise ExternalVerbatimError("unsafe Markdown fence language")
    materialized = tuple(records)
    body = "\n".join(record.markdown_fragment for record in materialized)
    longest = max(
        (len(match.group(0)) for match in _BACKTICK_RUN.finditer(body)), default=0
    )
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{body}\n{fence}"


def build_external_verbatim_ir(
    source_root: str | os.PathLike[str] | Path,
    execution_sources: Sequence[str | os.PathLike[str] | Path],
    *,
    strict: bool = True,
) -> ExternalVerbatimIR:
    """Build line-level IR in the supplied source-execution order.

    ``execution_sources`` is an ordered sequence of already-selected TeX
    source occurrences; duplicate paths intentionally represent repeated
    executions and produce duplicate blocks with different execution indices.
    With ``strict=True`` (the default), any unsafe/malformed invocation raises
    :class:`ExternalVerbatimSafetyError` and no IR is returned.  Non-strict
    mode retains explicit rejection records but still emits no content for an
    unsafe invocation.
    """

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ExternalVerbatimError(f"source root is not a directory: {root}")

    blocks: list[ExternalVerbatimBlock] = []
    rejections: list[ExternalVerbatimRejection] = []
    call_ordinal = 0
    for execution_index, raw_execution_source in enumerate(execution_sources):
        supplied = Path(raw_execution_source).expanduser()
        execution_source = (
            supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
        )
        try:
            execution_source.relative_to(root)
        except ValueError as error:
            raise ExternalVerbatimError(
                f"execution source escapes source root: {execution_source}"
            ) from error
        if not execution_source.is_file():
            raise ExternalVerbatimError(
                f"execution source is not a regular file: {execution_source}"
            )
        source_data = execution_source.read_bytes()

        for invocation in _scan_invocations(source_data):
            current_ordinal = call_ordinal
            call_ordinal += 1
            if invocation.parse_error is not None:
                code, message = invocation.parse_error
                rejections.append(
                    _rejection(
                        code=code,
                        message=message,
                        root=root,
                        execution_source=execution_source,
                        execution_index=execution_index,
                        invocation=invocation,
                    )
                )
                continue

            try:
                options = _parse_options(invocation.option_bytes)
            except ExternalVerbatimError as error:
                rejections.append(
                    _rejection(
                        code="unsafe_options",
                        message=str(error),
                        root=root,
                        execution_source=execution_source,
                        execution_index=execution_index,
                        invocation=invocation,
                    )
                )
                continue
            try:
                if invocation.path_bytes is None:
                    raise ExternalVerbatimError("path is not a static literal")
                requested_path = _literal_path(invocation.path_bytes)
                external_source = _resolve_under_root(root, requested_path)
            except ExternalVerbatimError as error:
                rejections.append(
                    _rejection(
                        code="unsafe_path",
                        message=str(error) or "path is not a static literal",
                        root=root,
                        execution_source=execution_source,
                        execution_index=execution_index,
                        invocation=invocation,
                    )
                )
                continue

            try:
                all_lines = _external_lines(external_source.read_bytes())
            except (OSError, ExternalVerbatimError) as error:
                rejections.append(
                    _rejection(
                        code="external_read_failed",
                        message=str(error),
                        root=root,
                        execution_source=execution_source,
                        execution_index=execution_index,
                        invocation=invocation,
                    )
                )
                continue

            option_values = dict(options)
            selected = _selected_lines(all_lines, option_values)
            raw_invocation = invocation.raw.decode("utf-8", errors="replace")
            execution_relative = _relative_posix(execution_source, root)
            external_relative = _relative_posix(external_source, root)
            call_payload = "\x1f".join(
                (
                    execution_relative,
                    str(execution_index),
                    str(invocation.start),
                    raw_invocation,
                    external_relative,
                )
            )
            call_id = "extverb-" + hashlib.sha256(
                call_payload.encode("utf-8")
            ).hexdigest()[:20]
            gobble = int(option_values.get("gobble", "0"))
            records: list[ExternalVerbatimLineRecord] = []
            for line in selected:
                visible = line.text[gobble:]
                record_payload = "\x1f".join(
                    (call_id, str(line.number), line.raw_bytes.hex())
                )
                record_id = "extverbline-" + hashlib.sha256(
                    record_payload.encode("utf-8")
                ).hexdigest()[:20]
                records.append(
                    ExternalVerbatimLineRecord(
                        record_id=record_id,
                        call_id=call_id,
                        source_root=root,
                        command_name=invocation.command_name,
                        execution_index=execution_index,
                        call_ordinal=current_ordinal,
                        execution_source=execution_source,
                        execution_source_line=invocation.line,
                        execution_source_column=invocation.column,
                        execution_source_byte_start=invocation.start,
                        execution_source_byte_end=invocation.end,
                        raw_invocation=raw_invocation,
                        requested_path=requested_path,
                        options=options,
                        external_source=external_source,
                        external_source_line=line.number,
                        external_source_byte_start=line.byte_start,
                        external_source_byte_end=line.byte_end,
                        raw_text=line.text,
                        visible_text=visible,
                        raw_line_sha256=hashlib.sha256(line.raw_bytes).hexdigest(),
                    )
                )
            blocks.append(
                ExternalVerbatimBlock(
                    call_id=call_id,
                    source_root=root,
                    command_name=invocation.command_name,
                    execution_index=execution_index,
                    call_ordinal=current_ordinal,
                    execution_source=execution_source,
                    execution_source_line=invocation.line,
                    execution_source_column=invocation.column,
                    execution_source_byte_start=invocation.start,
                    execution_source_byte_end=invocation.end,
                    raw_invocation=raw_invocation,
                    requested_path=requested_path,
                    options=options,
                    external_source=external_source,
                    records=tuple(records),
                )
            )

    if strict and rejections:
        raise ExternalVerbatimSafetyError(rejections)
    return ExternalVerbatimIR(root, tuple(blocks), tuple(rejections))


def extract_external_verbatim_records(
    source_root: str | os.PathLike[str] | Path,
    execution_sources: Sequence[str | os.PathLike[str] | Path],
    *,
    strict: bool = True,
) -> tuple[ExternalVerbatimLineRecord, ...]:
    """Convenience API returning only the page-mappable line records."""

    return build_external_verbatim_ir(
        source_root, execution_sources, strict=strict
    ).records


__all__ = [
    "EXTERNAL_VERBATIM_CONTRACT",
    "EXTERNAL_VERBATIM_KIND",
    "EXTERNAL_VERBATIM_SCHEMA_VERSION",
    "ExternalVerbatimBlock",
    "ExternalVerbatimError",
    "ExternalVerbatimIR",
    "ExternalVerbatimLineRecord",
    "ExternalVerbatimRejection",
    "ExternalVerbatimSafetyError",
    "build_external_verbatim_ir",
    "extract_external_verbatim_records",
    "render_fenced_code",
]
