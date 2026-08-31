"""Fail-closed source macro discovery for the compiler-native V3 pipeline.

Only local files that the clean TeX compilation actually opened are eligible.
Definitions are source provenance, not a TeX execution substitute: optional
arguments, conflicting definitions, malformed parameter references, and
oversized bodies are rejected.  The AST serializer independently parses every
expansion and keeps opaque or side-effecting macro bodies rejected.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .ast_ir import MathMacroDefinition


class SourceMacroDefinitionError(ValueError):
    """The compiler/source evidence for macro discovery is malformed."""


_DEFINITION = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?\s*"
    r"(?:\{\s*\\(?P<braced>[A-Za-z]+)\s*\}|\\(?P<plain>[A-Za-z]+))"
)
_OPERATOR = re.compile(
    r"\\DeclareMathOperator\*?\s*\{\s*\\(?P<name>[A-Za-z]+)\s*\}\s*"
)
_SAFE_SUFFIXES = frozenset({".tex", ".ltx", ".sty", ".cls"})
_MAX_ARGUMENTS = 4
_MAX_BODY_CHARACTERS = 4096


@dataclass(frozen=True, slots=True)
class SourceMacroRejection:
    source_path: str
    command_name: str | None
    reason: str
    char_offset: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "command_name": self.command_name,
            "reason": self.reason,
            "char_offset": self.char_offset,
        }


@dataclass(frozen=True, slots=True)
class SourceMacroRegistry:
    source_files: tuple[str, ...]
    definitions: tuple[MathMacroDefinition, ...]
    rejections: tuple[SourceMacroRejection, ...]

    @property
    def by_name(self) -> dict[str, MathMacroDefinition]:
        return {row.name: row for row in self.definitions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "clean_compile_fls_local_sources_only",
            "source_files": list(self.source_files),
            "definitions": {
                row.name: {
                    "argument_count": row.argument_count,
                    "body": row.body,
                }
                for row in self.definitions
            },
            "rejections": [row.to_dict() for row in self.rejections],
            "pdf_text_used": False,
        }


def _is_unescaped_percent(value: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and value[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 0


def _strip_comments_preserving_lines(value: str) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(value):
        if value[cursor] == "%" and _is_unescaped_percent(value, cursor):
            newline = value.find("\n", cursor)
            if newline < 0:
                output.extend(" " for _ in value[cursor:])
                break
            output.extend(" " for _ in value[cursor:newline])
            output.append("\n")
            cursor = newline + 1
            continue
        output.append(value[cursor])
        cursor += 1
    return "".join(output)


def _skip_space(value: str, cursor: int) -> int:
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    return cursor


def _balanced_group(
    value: str,
    opening: int,
    *,
    open_char: str = "{",
    close_char: str = "}",
) -> tuple[str, int] | None:
    if opening >= len(value) or value[opening] != open_char:
        return None
    depth = 0
    cursor = opening
    while cursor < len(value):
        char = value[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return value[opening + 1 : cursor], cursor + 1
        cursor += 1
    return None


def _parameter_rejection(body: str, argument_count: int) -> str | None:
    cursor = 0
    while cursor < len(body):
        if body[cursor] != "#":
            cursor += 1
            continue
        if cursor + 1 >= len(body) or not body[cursor + 1].isdigit():
            return "unsupported_parameter_token"
        parameter = int(body[cursor + 1])
        if parameter < 1 or parameter > argument_count:
            return "parameter_outside_declared_arity"
        cursor += 2
    return None


def _definitions_from_source(
    source_path: str,
    source: str,
) -> tuple[list[MathMacroDefinition], list[SourceMacroRejection]]:
    text = _strip_comments_preserving_lines(source)
    definitions: list[MathMacroDefinition] = []
    rejections: list[SourceMacroRejection] = []
    for match in _DEFINITION.finditer(text):
        name = match.group("braced") or match.group("plain")
        cursor = _skip_space(text, match.end())
        argument_count = 0
        if cursor < len(text) and text[cursor] == "[":
            arity = _balanced_group(
                text, cursor, open_char="[", close_char="]"
            )
            if arity is None or not arity[0].strip().isdigit():
                rejections.append(
                    SourceMacroRejection(
                        source_path, name, "invalid_argument_count", match.start()
                    )
                )
                continue
            argument_count = int(arity[0].strip())
            cursor = _skip_space(text, arity[1])
            if cursor < len(text) and text[cursor] == "[":
                rejections.append(
                    SourceMacroRejection(
                        source_path,
                        name,
                        "optional_default_argument_not_supported",
                        match.start(),
                    )
                )
                continue
        if argument_count > _MAX_ARGUMENTS:
            rejections.append(
                SourceMacroRejection(
                    source_path, name, "too_many_arguments", match.start()
                )
            )
            continue
        body_group = _balanced_group(text, cursor)
        if body_group is None:
            rejections.append(
                SourceMacroRejection(
                    source_path, name, "missing_or_unbalanced_body", match.start()
                )
            )
            continue
        body = body_group[0]
        if len(body) > _MAX_BODY_CHARACTERS:
            rejections.append(
                SourceMacroRejection(
                    source_path, name, "body_too_large", match.start()
                )
            )
            continue
        parameter_error = _parameter_rejection(body, argument_count)
        if parameter_error is not None:
            rejections.append(
                SourceMacroRejection(
                    source_path, name, parameter_error, match.start()
                )
            )
            continue
        definitions.append(MathMacroDefinition(name, argument_count, body))

    for match in _OPERATOR.finditer(text):
        body_group = _balanced_group(text, _skip_space(text, match.end()))
        if body_group is None or len(body_group[0]) > _MAX_BODY_CHARACTERS:
            rejections.append(
                SourceMacroRejection(
                    source_path,
                    match.group("name"),
                    "missing_or_unbalanced_operator_body",
                    match.start(),
                )
            )
            continue
        definitions.append(
            MathMacroDefinition(
                match.group("name"),
                0,
                r"\operatorname{" + body_group[0] + "}",
            )
        )
    return definitions, rejections


def collect_source_macro_definitions(
    sources: Mapping[str, str],
) -> SourceMacroRegistry:
    """Collect only definitions that are identical across executed sources."""

    candidates: dict[str, set[tuple[int, str]]] = defaultdict(set)
    rejections: list[SourceMacroRejection] = []
    for source_path, source in sorted(sources.items()):
        rows, local_rejections = _definitions_from_source(source_path, source)
        rejections.extend(local_rejections)
        for row in rows:
            candidates[row.name].add((row.argument_count, row.body))

    definitions: list[MathMacroDefinition] = []
    for name, values in sorted(candidates.items()):
        if len(values) != 1:
            rejections.append(
                SourceMacroRejection(
                    "<executed-source-set>",
                    name,
                    "conflicting_active_definitions",
                )
            )
            continue
        argument_count, body = next(iter(values))
        definitions.append(MathMacroDefinition(name, argument_count, body))
    return SourceMacroRegistry(
        source_files=tuple(sorted(sources)),
        definitions=tuple(definitions),
        rejections=tuple(rejections),
    )


def load_executed_local_sources(
    source_root: Path,
    fls_paths: Sequence[Path],
) -> dict[str, str]:
    """Read local LaTeX inputs named by clean-compile recorder files."""

    root = source_root.resolve()
    discovered: dict[str, Path] = {}
    for fls_path in fls_paths:
        if not fls_path.is_file():
            continue
        for line in fls_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("INPUT "):
                continue
            raw_path = line[6:].strip()
            if not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            if resolved.suffix.casefold() not in _SAFE_SUFFIXES or not resolved.is_file():
                continue
            discovered[relative.as_posix()] = resolved
    return {
        relative: path.read_text(encoding="utf-8", errors="replace")
        for relative, path in sorted(discovered.items())
    }


__all__ = [
    "SourceMacroDefinitionError",
    "SourceMacroRegistry",
    "SourceMacroRejection",
    "collect_source_macro_definitions",
    "load_executed_local_sources",
]
