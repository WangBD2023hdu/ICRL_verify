"""Deterministic, source-mapped LaTeX project flattening for true v3.

Only ``\\input{...}`` and ``\\include{...}`` source-file edges are expanded.
The result is a canonical TeX stream whose visible content is still entirely
derived from the original project.  Every copied character has exact
file/character/byte/line provenance; generated separators deliberately occupy
source-map gaps instead of pretending to be original text.

This module does not compile TeX, inspect PDFs, or import a v1/v2 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any

from .document_ast import CanonicalSourceMap, FlattenedSourceSegment


PROJECT_FLATTENER_VERSION = "source_first_v3_project_flattener_v1"
_INPUT = re.compile(r"\\(input|include)\s*\{")
_ENDINPUT = re.compile(r"\\endinput\b")
_VERBATIM_BEGIN = re.compile(
    r"\\begin\s*\{(verbatim\*?|Verbatim|lstlisting|minted|comment)\}"
)


class SourceProjectError(ValueError):
    """The project cannot be flattened without changing unknown semantics."""


@dataclass(frozen=True, slots=True)
class FlattenedProject:
    version: str
    source_root: str
    main_tex: str
    source: str
    source_sha256: str
    source_map: CanonicalSourceMap
    files: tuple[str, ...]
    include_edges: tuple[tuple[str, str, str], ...]
    generated_characters: int

    def to_dict(self, *, include_source: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "source_root": self.source_root,
            "main_tex": self.main_tex,
            "source_sha256": self.source_sha256,
            "files": list(self.files),
            "include_edges": [list(row) for row in self.include_edges],
            "generated_characters": self.generated_characters,
            "source_characters": len(self.source),
            "pdf_text_used": False,
        }
        if include_source:
            payload["source"] = self.source
        return payload


@dataclass(slots=True)
class _Piece:
    text: str
    source_path: str | None
    source_char_start: int = 0
    source_byte_start: int = 0
    source_line_start: int = 1
    source_column_start: int = 0


def _line_column(source: str, position: int) -> tuple[int, int]:
    line = source.count("\n", 0, position) + 1
    previous = source.rfind("\n", 0, position)
    return line, position if previous < 0 else position - previous - 1


def _is_escaped(source: str, index: int) -> bool:
    slashes = 0
    cursor = index - 1
    while cursor >= 0 and source[cursor] == "\\":
        slashes += 1
        cursor -= 1
    return slashes % 2 == 1


def _mask_nonexecuted_regions(source: str) -> str:
    """Return an equal-length mask for comments and verbatim-like bodies."""

    masked = list(source)
    cursor = 0
    while cursor < len(source):
        if source[cursor] == "%" and not _is_escaped(source, cursor):
            end = source.find("\n", cursor)
            end = len(source) if end < 0 else end
            for index in range(cursor, end):
                masked[index] = " "
            cursor = end
            continue
        verbatim = _VERBATIM_BEGIN.match(source, cursor)
        if verbatim is not None:
            environment = verbatim.group(1)
            closing = re.compile(r"\\end\s*\{" + re.escape(environment) + r"\}")
            match = closing.search(source, verbatim.end())
            if match is None:
                raise SourceProjectError(
                    f"unterminated verbatim-like environment: {environment}"
                )
            for index in range(cursor, match.end()):
                if masked[index] != "\n":
                    masked[index] = " "
            cursor = match.end()
            continue
        cursor += 1
    return "".join(masked)


def _balanced_group_end(source: str, opening: int) -> int:
    if opening >= len(source) or source[opening] != "{":
        raise SourceProjectError("expected an opening input-file brace")
    depth = 0
    cursor = opening
    while cursor < len(source):
        char = source[cursor]
        if char == "%" and not _is_escaped(source, cursor):
            newline = source.find("\n", cursor)
            cursor = len(source) if newline < 0 else newline + 1
            continue
        if char == "{" and not _is_escaped(source, cursor):
            depth += 1
        elif char == "}" and not _is_escaped(source, cursor):
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    raise SourceProjectError("unterminated input-file argument")


def _normalised_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise SourceProjectError(f"source include escapes project root: {path}") from exc
    posix = relative.as_posix()
    pure = PurePosixPath(posix)
    if not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise SourceProjectError(f"invalid project-relative path: {posix!r}")
    return posix


def _resolve_include(root: Path, current: Path, argument: str) -> Path:
    value = argument.strip()
    if not value or any(token in value for token in ("\\", "{", "}", "%", "#")):
        raise SourceProjectError(f"dynamic or unsafe input path: {argument!r}")
    candidate = Path(value)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".tex")
    # TeX normally resolves project-relative inputs from the compiler cwd.
    # A current-file-relative fallback covers common generated archives while
    # remaining deterministic; ambiguity is rejected.
    choices = [root / candidate]
    local = current.parent / candidate
    if local != choices[0]:
        choices.append(local)
    existing = [path.resolve() for path in choices if path.is_file()]
    unique = sorted(set(existing))
    if len(unique) != 1:
        raise SourceProjectError(
            f"input path must resolve uniquely: {argument!r}; matches={len(unique)}"
        )
    _normalised_relative(unique[0], root)
    return unique[0]


def flatten_source_project(
    source_root: str | Path,
    main_tex: str | Path,
    *,
    max_depth: int = 64,
    max_source_bytes: int = 64 * 1024 * 1024,
) -> FlattenedProject:
    """Flatten one LaTeX project with exact provenance and cycle detection."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise SourceProjectError(f"source_root is not a directory: {root}")
    requested = Path(main_tex)
    main = (requested if requested.is_absolute() else root / requested).resolve()
    main_relative = _normalised_relative(main, root)
    if not main.is_file():
        raise SourceProjectError(f"main TeX file does not exist: {main}")
    if isinstance(max_depth, bool) or max_depth < 1:
        raise SourceProjectError("max_depth must be positive")
    if isinstance(max_source_bytes, bool) or max_source_bytes < 1:
        raise SourceProjectError("max_source_bytes must be positive")

    pieces: list[_Piece] = []
    files: set[str] = set()
    edges: list[tuple[str, str, str]] = []
    bytes_seen = 0

    def append_original(text: str, path: str, source: str, start: int, end: int) -> None:
        if end <= start:
            return
        line, column = _line_column(source, start)
        pieces.append(
            _Piece(
                text=text,
                source_path=path,
                source_char_start=start,
                source_byte_start=len(source[:start].encode("utf-8")),
                source_line_start=line,
                source_column_start=column,
            )
        )

    def append_generated(text: str) -> None:
        if text:
            pieces.append(_Piece(text=text, source_path=None))

    def visit(path: Path, stack: tuple[Path, ...]) -> None:
        nonlocal bytes_seen
        if len(stack) >= max_depth:
            raise SourceProjectError(f"input nesting exceeds max_depth={max_depth}")
        resolved = path.resolve()
        if resolved in stack:
            cycle = " -> ".join(_normalised_relative(row, root) for row in (*stack, resolved))
            raise SourceProjectError(f"cyclic source input: {cycle}")
        relative = _normalised_relative(resolved, root)
        try:
            raw = resolved.read_bytes()
            source = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceProjectError(f"source file is not UTF-8: {relative}") from exc
        bytes_seen += len(raw)
        if bytes_seen > max_source_bytes:
            raise SourceProjectError(
                f"flattened source exceeds max_source_bytes={max_source_bytes}"
            )
        files.add(relative)
        masked = _mask_nonexecuted_regions(source)
        endinput = _ENDINPUT.search(masked)
        logical_end = endinput.start() if endinput is not None else len(source)
        cursor = 0
        for match in _INPUT.finditer(masked, 0, logical_end):
            if match.start() < cursor:
                continue
            opening = match.end() - 1
            closing = _balanced_group_end(source, opening)
            if closing > logical_end:
                raise SourceProjectError("input argument crosses endinput")
            append_original(source[cursor : match.start()], relative, source, cursor, match.start())
            argument = source[opening + 1 : closing - 1]
            child = _resolve_include(root, resolved, argument)
            child_relative = _normalised_relative(child, root)
            kind = match.group(1)
            edges.append((relative, child_relative, kind))
            # Generated newlines prevent tokens on either side of the removed
            # command from merging.  Include retains TeX's documented page
            # boundary semantics with explicit clearpage tokens.
            append_generated("\n" + (r"\clearpage" + "\n" if kind == "include" else ""))
            visit(child, (*stack, resolved))
            append_generated(("\n" + r"\clearpage" if kind == "include" else "") + "\n")
            cursor = closing
        append_original(source[cursor:logical_end], relative, source, cursor, logical_end)

    visit(main, ())
    flattened_parts: list[str] = []
    segments: list[FlattenedSourceSegment] = []
    generated = 0
    offset = 0
    for piece in pieces:
        flattened_parts.append(piece.text)
        end = offset + len(piece.text)
        if piece.source_path is None:
            generated += len(piece.text)
        elif piece.text:
            segments.append(
                FlattenedSourceSegment(
                    canonical_char_start=offset,
                    canonical_char_end=end,
                    source_path=piece.source_path,
                    source_char_start=piece.source_char_start,
                    source_byte_start=piece.source_byte_start,
                    source_line_start=piece.source_line_start,
                    source_column_start=piece.source_column_start,
                )
            )
        offset = end
    flattened = "".join(flattened_parts)
    return FlattenedProject(
        version=PROJECT_FLATTENER_VERSION,
        source_root=str(root),
        main_tex=main_relative,
        source=flattened,
        source_sha256=hashlib.sha256(flattened.encode("utf-8")).hexdigest(),
        source_map=CanonicalSourceMap(tuple(segments)),
        files=tuple(sorted(files)),
        include_edges=tuple(edges),
        generated_characters=generated,
    )


__all__ = [
    "FlattenedProject",
    "PROJECT_FLATTENER_VERSION",
    "SourceProjectError",
    "flatten_source_project",
]
