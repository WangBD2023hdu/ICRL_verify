"""Execution-order IR for source-first LaTeX page reconstruction.

The physical order of files on disk is not the order in which TeX reads
their contents.  A main file can interleave its own material with nested
``\\input`` and ``\\include`` files, and the same file can be executed more
than once.  This module builds a small, conservative execution-order model
for those cases.

Only literal include commands are interpreted.  Comments are ignored and an
``.fls``-derived set of executed sources is used as an allow-list, so a
literal include which was not observed by the recorder is not expanded.  The
model deliberately preserves repeated executions.  A position in a file
executed twice therefore resolves to ``ambiguous``; ordering APIs fail closed
instead of silently choosing one occurrence.

Offsets in this module are zero-based offsets in the original source *bytes*.
Line numbers are one-based and columns are zero-based byte columns.  No PDF
text is used anywhere in this representation.
"""

from __future__ import annotations

import dataclasses
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


SOURCE_UNIT_KINDS = frozenset({"paragraph", "heading", "table", "frontmatter"})


class ExecutionIRError(ValueError):
    """Base class for conservative execution-IR failures."""


class MainSourceNotExecutedError(ExecutionIRError):
    """Raised when the main source is absent from the FLS allow-list."""


class ExecutedSourceMissingError(ExecutionIRError):
    """Raised when an allowed literal include does not exist on disk."""


class IncludeCycleError(ExecutionIRError):
    """Raised when recursive expansion reaches a file already on its stack."""


class InvalidSourcePositionError(ExecutionIRError):
    """Raised for an invalid byte offset, line, or byte column."""


class UnresolvedExecutionError(ExecutionIRError):
    """Raised when a source unit was not executed according to the IR."""


class AmbiguousExecutionError(ExecutionIRError):
    """Raised when a source position belongs to multiple executions."""


@dataclasses.dataclass(frozen=True, order=True)
class ExecutionOrdinal:
    """A sortable position in the flattened TeX execution stream.

    ``segment`` orders interleaved source ranges and ``byte_offset`` orders
    positions inside one range.  Occurrence metadata does not participate in
    comparisons; it is retained for auditing and ambiguity diagnostics.
    """

    segment: int
    byte_offset: int
    occurrence_id: int = dataclasses.field(compare=False)
    source_file: Path = dataclasses.field(compare=False)

    @property
    def key(self) -> tuple[int, int]:
        return self.segment, self.byte_offset


@dataclasses.dataclass(frozen=True)
class ExecutionSegment:
    """One contiguous range from one source-file execution occurrence."""

    ordinal: int
    occurrence_id: int
    source_file: Path
    source_start: int
    source_end: int
    terminal_point: bool = False

    def contains(self, byte_offset: int, *, file_size: int) -> bool:
        if self.terminal_point:
            return byte_offset == file_size
        return self.source_start <= byte_offset < self.source_end


@dataclasses.dataclass(frozen=True)
class FileExecution:
    """One execution occurrence, including the include call which opened it."""

    occurrence_id: int
    source_file: Path
    parent_occurrence_id: int | None
    include_source_file: Path | None
    include_start: int | None
    include_end: int | None
    depth: int


@dataclasses.dataclass(frozen=True)
class ExecutionDiagnostic:
    """A non-fatal, source-addressed execution-model diagnostic."""

    code: str
    source_file: Path
    byte_offset: int | None
    message: str
    target_file: Path | None = None


@dataclasses.dataclass(frozen=True)
class ExecutionResolution:
    """Resolution of one source position against all execution occurrences."""

    status: str
    source_file: Path
    byte_offset: int
    ordinals: tuple[ExecutionOrdinal, ...]
    message: str = ""

    @property
    def is_unique(self) -> bool:
        return self.status == "resolved" and len(self.ordinals) == 1

    @property
    def ordinal(self) -> ExecutionOrdinal | None:
        """Return the unique ordinal, never an arbitrary ambiguous match."""

        if not self.is_unique:
            return None
        return self.ordinals[0]

    def require_unique(self) -> ExecutionOrdinal:
        if self.status == "ambiguous":
            occurrences = [item.occurrence_id for item in self.ordinals]
            raise AmbiguousExecutionError(
                f"source position is executed multiple times: "
                f"{self.source_file}:{self.byte_offset}; occurrences={occurrences}"
            )
        if not self.is_unique:
            raise UnresolvedExecutionError(
                self.message
                or f"source position was not executed: {self.source_file}:{self.byte_offset}"
            )
        return self.ordinals[0]


@dataclasses.dataclass(frozen=True)
class SourceUnitRef:
    """A paragraph/heading/table/frontmatter position to order.

    Exactly one of ``byte_offset`` and ``line`` must be supplied.  ``payload``
    is carried through untouched so callers can associate their own unit IR.
    Kinds beyond :data:`SOURCE_UNIT_KINDS` are allowed for forward-compatible
    experiments; the four declared kinds receive no ordering preference and
    are all sorted solely by their execution position.
    """

    kind: str
    source_file: str | os.PathLike[str]
    byte_offset: int | None = None
    line: int | None = None
    column: int = 0
    payload: Any = dataclasses.field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.kind:
            raise InvalidSourcePositionError("source unit kind must be non-empty")
        if (self.byte_offset is None) == (self.line is None):
            raise InvalidSourcePositionError(
                "exactly one of byte_offset and line must be supplied"
            )


@dataclasses.dataclass(frozen=True)
class OrderedSourceUnit:
    """A source unit paired with its unique execution ordinal."""

    unit: SourceUnitRef
    ordinal: ExecutionOrdinal
    input_index: int


@dataclasses.dataclass(frozen=True)
class _IncludeCommand:
    kind: str
    start: int
    end: int
    target: str


@dataclasses.dataclass(frozen=True)
class _SourceRecord:
    path: Path
    data: bytes
    line_starts: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.data)

    def offset_for_line(self, line: int, column: int) -> int:
        if line < 1 or line > len(self.line_starts):
            raise InvalidSourcePositionError(
                f"line outside source: {self.path}: line={line}; "
                f"line_count={len(self.line_starts)}"
            )
        if column < 0:
            raise InvalidSourcePositionError(
                f"column must be non-negative: {self.path}:{line}:{column}"
            )
        start = self.line_starts[line - 1]
        newline = self.data.find(b"\n", start)
        raw_end = self.size if newline < 0 else newline
        # Treat CRLF as one line ending and do not expose the CR as content.
        content_end = raw_end
        if content_end > start and self.data[content_end - 1 : content_end] == b"\r":
            content_end -= 1
        if start + column > content_end:
            raise InvalidSourcePositionError(
                f"column outside source line: {self.path}:{line}:{column}; "
                f"line_byte_length={content_end - start}"
            )
        return start + column


def _line_starts(data: bytes) -> tuple[int, ...]:
    starts = [0]
    starts.extend(index + 1 for index, value in enumerate(data) if value == 0x0A)
    # A final newline does not introduce an addressable extra source line.
    if len(starts) > 1 and starts[-1] == len(data):
        starts.pop()
    return tuple(starts) or (0,)


def _is_command_letter(value: int) -> bool:
    return 65 <= value <= 90 or 97 <= value <= 122 or value == 64


def _unescaped_percent(data: bytes, index: int) -> bool:
    slashes = 0
    cursor = index - 1
    while cursor >= 0 and data[cursor] == 0x5C:
        slashes += 1
        cursor -= 1
    return slashes % 2 == 0


def _literal_include_commands(data: bytes) -> tuple[_IncludeCommand, ...]:
    """Scan literal include commands while retaining exact byte spans."""

    commands: list[_IncludeCommand] = []
    cursor = 0
    size = len(data)
    while cursor < size:
        value = data[cursor]
        if value == 0x25 and _unescaped_percent(data, cursor):  # ``%``
            newline = data.find(b"\n", cursor + 1)
            cursor = size if newline < 0 else newline + 1
            continue
        if value != 0x5C:  # ``\\``
            cursor += 1
            continue

        start = cursor
        cursor += 1
        name_start = cursor
        while cursor < size and _is_command_letter(data[cursor]):
            cursor += 1
        name = data[name_start:cursor]
        if name not in {b"input", b"include"}:
            # A control symbol consumes one non-letter after the slash.  The
            # normal loop still advances safely for both forms.
            if cursor == name_start and cursor < size:
                cursor += 1
            continue

        argument = cursor
        while argument < size and data[argument] in b" \t\r\n":
            argument += 1
        if argument >= size:
            continue

        if data[argument] == 0x7B:  # ``{``
            depth = 1
            end = argument + 1
            while end < size and depth:
                if data[end] == 0x25 and _unescaped_percent(data, end):
                    # A comment inside a filename makes it non-literal for
                    # this conservative parser.
                    break
                if data[end] == 0x7B:
                    depth += 1
                elif data[end] == 0x7D:
                    depth -= 1
                end += 1
            if depth:
                continue
            raw_target = data[argument + 1 : end - 1]
            command_end = end
        else:
            end = argument
            while end < size and data[end] not in b" \t\r\n%":
                end += 1
            raw_target = data[argument:end]
            command_end = end

        try:
            target = raw_target.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if not target or any(char in target for char in "\\#$~{}"):
            continue
        commands.append(
            _IncludeCommand(
                kind=name.decode("ascii"),
                start=start,
                end=command_end,
                target=target,
            )
        )
        cursor = command_end
    return tuple(commands)


def _normalize_path(path: str | os.PathLike[str], *, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve(strict=False)


def parse_fls_executed_sources(
    fls_path: str | os.PathLike[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Parse recorder ``INPUT`` entries as an ordered, de-duplicated allow-list.

    Relative paths use the explicit ``cwd`` when supplied, otherwise the first
    recorder ``PWD`` entry, otherwise the ``.fls`` parent directory.  Recorder
    duplicates are intentionally de-duplicated: an FLS file is an allow-list,
    not a reliable trace of include occurrence counts.
    """

    recorder = Path(fls_path).expanduser().resolve()
    lines = recorder.read_text(encoding="utf-8", errors="replace").splitlines()
    if cwd is not None:
        base_dir = Path(cwd).expanduser().resolve()
    else:
        pwd_entry = next((line[4:] for line in lines if line.startswith("PWD ")), None)
        base_dir = (
            Path(pwd_entry).expanduser().resolve()
            if pwd_entry
            else recorder.parent
        )

    result: list[Path] = []
    seen: set[Path] = set()
    for line in lines:
        if not line.startswith("INPUT "):
            continue
        value = line[6:]
        if not value:
            continue
        path = _normalize_path(value, base_dir=base_dir)
        if path not in seen:
            seen.add(path)
            result.append(path)
    return tuple(result)


class ExecutionIR:
    """Immutable public view of a recursively flattened source execution."""

    def __init__(
        self,
        *,
        main_tex: Path,
        allowlist: frozenset[Path],
        occurrences: Sequence[FileExecution],
        segments: Sequence[ExecutionSegment],
        diagnostics: Sequence[ExecutionDiagnostic],
        records: dict[Path, _SourceRecord],
    ) -> None:
        self.main_tex = main_tex
        self.base_dir = main_tex.parent
        self.allowlist = allowlist
        self.occurrences = tuple(occurrences)
        self.segments = tuple(segments)
        self.diagnostics = tuple(diagnostics)
        self._records = dict(records)
        by_file: dict[Path, list[ExecutionSegment]] = {}
        for segment in self.segments:
            by_file.setdefault(segment.source_file, []).append(segment)
        self._segments_by_file = {
            path: tuple(items) for path, items in by_file.items()
        }

    @property
    def executed_sources(self) -> tuple[Path, ...]:
        """Sources in first-execution order, with duplicates removed."""

        return tuple(dict.fromkeys(item.source_file for item in self.occurrences))

    def _path(self, source_file: str | os.PathLike[str]) -> Path:
        return _normalize_path(source_file, base_dir=self.base_dir)

    def resolve(
        self,
        source_file: str | os.PathLike[str],
        *,
        byte_offset: int | None = None,
        line: int | None = None,
        column: int = 0,
    ) -> ExecutionResolution:
        """Resolve a source byte position without hiding repeated execution."""

        if (byte_offset is None) == (line is None):
            raise InvalidSourcePositionError(
                "exactly one of byte_offset and line must be supplied"
            )
        path = self._path(source_file)
        record = self._records.get(path)
        if record is None:
            return ExecutionResolution(
                status="not_executed",
                source_file=path,
                byte_offset=-1 if byte_offset is None else byte_offset,
                ordinals=(),
                message=f"source is not in the executed source graph: {path}",
            )
        if line is not None:
            offset = record.offset_for_line(line, column)
        else:
            assert byte_offset is not None
            offset = byte_offset
            if offset < 0 or offset > record.size:
                raise InvalidSourcePositionError(
                    f"byte offset outside source: {path}:{offset}; size={record.size}"
                )

        ordinals = tuple(
            ExecutionOrdinal(
                segment=segment.ordinal,
                byte_offset=offset,
                occurrence_id=segment.occurrence_id,
                source_file=path,
            )
            for segment in self._segments_by_file.get(path, ())
            if segment.contains(offset, file_size=record.size)
        )
        if not ordinals:
            return ExecutionResolution(
                status="not_executed",
                source_file=path,
                byte_offset=offset,
                ordinals=(),
                message=f"source position is not in an executed range: {path}:{offset}",
            )
        if len(ordinals) > 1:
            return ExecutionResolution(
                status="ambiguous",
                source_file=path,
                byte_offset=offset,
                ordinals=ordinals,
                message=f"source position has {len(ordinals)} execution occurrences",
            )
        return ExecutionResolution(
            status="resolved",
            source_file=path,
            byte_offset=offset,
            ordinals=ordinals,
        )

    def order_source_units(
        self, units: Iterable[SourceUnitRef]
    ) -> tuple[OrderedSourceUnit, ...]:
        """Order heterogeneous source units, failing closed on uncertainty."""

        resolved: list[OrderedSourceUnit] = []
        for input_index, unit in enumerate(units):
            resolution = self.resolve(
                unit.source_file,
                byte_offset=unit.byte_offset,
                line=unit.line,
                column=unit.column,
            )
            resolved.append(
                OrderedSourceUnit(
                    unit=unit,
                    ordinal=resolution.require_unique(),
                    input_index=input_index,
                )
            )
        return tuple(
            sorted(
                resolved,
                key=lambda item: (item.ordinal, item.input_index),
            )
        )


class _ExecutionBuilder:
    def __init__(self, main_tex: Path, allowlist: frozenset[Path]) -> None:
        self.main_tex = main_tex
        self.allowlist = allowlist
        self.occurrences: list[FileExecution] = []
        self.segments: list[ExecutionSegment] = []
        self.diagnostics: list[ExecutionDiagnostic] = []
        self.records: dict[Path, _SourceRecord] = {}

    def _record(self, path: Path) -> _SourceRecord:
        record = self.records.get(path)
        if record is not None:
            return record
        if not path.is_file():
            raise ExecutedSourceMissingError(f"executed source is missing: {path}")
        data = path.read_bytes()
        record = _SourceRecord(path=path, data=data, line_starts=_line_starts(data))
        self.records[path] = record
        return record

    def _target_candidates(self, current: Path, target: str) -> tuple[Path, ...]:
        literal = _normalize_path(target, base_dir=current.parent)
        if Path(target).suffix:
            return (literal,)
        with_tex = literal.with_name(literal.name + ".tex")
        return (with_tex, literal)

    def _allowed_target(self, current: Path, command: _IncludeCommand) -> Path | None:
        candidates = self._target_candidates(current, command.target)
        allowed = tuple(path for path in candidates if path in self.allowlist)
        if not allowed:
            self.diagnostics.append(
                ExecutionDiagnostic(
                    code="include_not_in_fls_allowlist",
                    source_file=current,
                    byte_offset=command.start,
                    target_file=candidates[0],
                    message=(
                        f"literal \\{command.kind} target was not executed "
                        f"according to the FLS allow-list: {command.target}"
                    ),
                )
            )
            return None
        if len(allowed) > 1:
            raise ExecutionIRError(
                f"literal include resolves to multiple allowed paths: "
                f"{current}:{command.start}: {allowed}"
            )
        return allowed[0]

    def _segment(
        self,
        occurrence_id: int,
        path: Path,
        start: int,
        end: int,
        *,
        terminal_point: bool = False,
    ) -> None:
        if not terminal_point and start >= end:
            return
        self.segments.append(
            ExecutionSegment(
                ordinal=len(self.segments),
                occurrence_id=occurrence_id,
                source_file=path,
                source_start=start,
                source_end=end,
                terminal_point=terminal_point,
            )
        )

    def execute(
        self,
        path: Path,
        *,
        parent_occurrence_id: int | None,
        include_source_file: Path | None,
        include_start: int | None,
        include_end: int | None,
        stack: tuple[Path, ...],
    ) -> None:
        if path in stack:
            cycle = stack[stack.index(path) :] + (path,)
            raise IncludeCycleError(
                "recursive include cycle: " + " -> ".join(str(item) for item in cycle)
            )
        record = self._record(path)
        occurrence_id = len(self.occurrences)
        self.occurrences.append(
            FileExecution(
                occurrence_id=occurrence_id,
                source_file=path,
                parent_occurrence_id=parent_occurrence_id,
                include_source_file=include_source_file,
                include_start=include_start,
                include_end=include_end,
                depth=len(stack),
            )
        )

        cursor = 0
        next_stack = stack + (path,)
        for command in _literal_include_commands(record.data):
            # The command token itself precedes execution of its target.
            self._segment(occurrence_id, path, cursor, command.end)
            target = self._allowed_target(path, command)
            if target is not None:
                if not target.is_file():
                    raise ExecutedSourceMissingError(
                        f"allowed include target is missing: {path}:{command.start}: {target}"
                    )
                self.execute(
                    target,
                    parent_occurrence_id=occurrence_id,
                    include_source_file=path,
                    include_start=command.start,
                    include_end=command.end,
                    stack=next_stack,
                )
            cursor = command.end
        self._segment(occurrence_id, path, cursor, record.size)
        # The terminal point is separate because a file ending in an include
        # resumes only after that included file has completed.
        self._segment(
            occurrence_id,
            path,
            record.size,
            record.size,
            terminal_point=True,
        )

    def build(self) -> ExecutionIR:
        self.execute(
            self.main_tex,
            parent_occurrence_id=None,
            include_source_file=None,
            include_start=None,
            include_end=None,
            stack=(),
        )
        counts = Counter(item.source_file for item in self.occurrences)
        for path, count in sorted(counts.items(), key=lambda item: str(item[0])):
            if count > 1:
                self.diagnostics.append(
                    ExecutionDiagnostic(
                        code="repeated_source_execution",
                        source_file=path,
                        byte_offset=None,
                        target_file=path,
                        message=(
                            f"source was executed {count} times; positions in this "
                            "file require occurrence-aware handling"
                        ),
                    )
                )
        return ExecutionIR(
            main_tex=self.main_tex,
            allowlist=self.allowlist,
            occurrences=self.occurrences,
            segments=self.segments,
            diagnostics=self.diagnostics,
            records=self.records,
        )


def build_execution_ir(
    main_tex: str | os.PathLike[str],
    *,
    fls_sources: Iterable[str | os.PathLike[str]],
    fls_base_dir: str | os.PathLike[str] | None = None,
) -> ExecutionIR:
    """Build a recursively expanded execution IR under an FLS allow-list."""

    main = Path(main_tex).expanduser().resolve()
    base_dir = (
        Path(fls_base_dir).expanduser().resolve()
        if fls_base_dir is not None
        else main.parent
    )
    allowlist = frozenset(
        _normalize_path(path, base_dir=base_dir) for path in fls_sources
    )
    if main not in allowlist:
        raise MainSourceNotExecutedError(
            f"main source is absent from the FLS allow-list: {main}"
        )
    return _ExecutionBuilder(main, allowlist).build()


def build_execution_ir_from_fls(
    main_tex: str | os.PathLike[str],
    fls_path: str | os.PathLike[str],
    *,
    fls_cwd: str | os.PathLike[str] | None = None,
) -> ExecutionIR:
    """Convenience wrapper around :func:`parse_fls_executed_sources`."""

    sources = parse_fls_executed_sources(fls_path, cwd=fls_cwd)
    return build_execution_ir(main_tex, fls_sources=sources)


def order_source_units(
    execution_ir: ExecutionIR,
    units: Iterable[SourceUnitRef],
) -> tuple[OrderedSourceUnit, ...]:
    """Functional alias for :meth:`ExecutionIR.order_source_units`."""

    return execution_ir.order_source_units(units)
