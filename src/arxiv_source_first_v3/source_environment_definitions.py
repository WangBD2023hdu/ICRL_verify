"""Fail-closed source-defined aliases for ordinary LaTeX lists.

The document AST knows how to serialize the built-in ``itemize`` and
``enumerate`` environments.  A paper may wrap one of those environments in a
small local alias (for example, a compact list environment).  This module
recognizes only wrappers whose source definition is sufficient to prove that
the alias has exactly the same list semantics:

* zero arguments, with no optional/default argument syntax;
* one matching ``itemize`` or ``enumerate`` begin/end pair;
* comments/whitespace, literal ``\\vspace*`` dimensions, and literal
  ``\\setlength`` controls for the four list spacing lengths around it.

Everything else remains unresolved.  In particular, visible text, ``\\item``
or other item injection, labels/counters, dynamic dimensions, unknown control
sequences, recursion, and conflicting definitions are rejected.  The caller
is responsible for supplying only source files opened by the clean compiler
run; this module never scans the TeX installation or guesses inactive source
definitions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Mapping


_STANDARD_DECLARATION = re.compile(
    r"\\(?P<command>newenvironment|renewenvironment|provideenvironment)"
    r"(?P<star>\*)?"
)
_XPARSE_DECLARATION = re.compile(
    r"\\(?P<command>NewDocumentEnvironment|RenewDocumentEnvironment|"
    r"ProvideDocumentEnvironment|DeclareDocumentEnvironment)"
    r"(?P<star>\*)?"
)
_SAFE_ENVIRONMENT = re.compile(r"^[A-Za-z][A-Za-z0-9@*_-]*$")
_LITERAL_DIMENSION = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em|mu)$",
    re.IGNORECASE,
)
_LIST_SPACING_LENGTHS = frozenset(
    {"partopsep", "itemsep", "parskip", "parsep"}
)
_LIST_KINDS = frozenset({"itemize", "enumerate"})
_SAFE_SETUP_COMMANDS = frozenset({"vspace", "setlength"})


@dataclass(frozen=True, slots=True)
class SourceListEnvironmentRejection:
    """One source-defined environment that was not admitted."""

    source_path: str
    environment_name: str | None
    command_name: str | None
    reason: str
    char_offset: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "environment_name": self.environment_name,
            "command_name": self.command_name,
            "reason": self.reason,
            "char_offset": self.char_offset,
        }


@dataclass(frozen=True, slots=True)
class SourceListEnvironmentDefinition:
    """A compiler-executed source alias with proven built-in list semantics."""

    environment_name: str
    list_kind: str
    declaration_command: str
    begin_body: str
    end_body: str
    source_path: str

    def __post_init__(self) -> None:
        if (
            not _SAFE_ENVIRONMENT.fullmatch(self.environment_name)
            or self.environment_name in _LIST_KINDS
            or self.list_kind not in _LIST_KINDS
            or not self.declaration_command
            or not self.source_path
        ):
            raise ValueError("invalid source list-environment definition")

    def to_dict(self) -> dict[str, str]:
        return {
            "environment_name": self.environment_name,
            "list_kind": self.list_kind,
            "declaration_command": self.declaration_command,
            "begin_body": self.begin_body,
            "end_body": self.end_body,
            "source_path": self.source_path,
        }


@dataclass(frozen=True, slots=True)
class SourceListEnvironmentRegistry:
    """Resolved aliases and explicit rejection audit from executed sources."""

    source_files: tuple[str, ...]
    definitions: tuple[SourceListEnvironmentDefinition, ...]
    rejections: tuple[SourceListEnvironmentRejection, ...]

    @property
    def by_name(self) -> dict[str, SourceListEnvironmentDefinition]:
        return {row.environment_name: row for row in self.definitions}

    def to_dict(self) -> dict[str, object]:
        return {
            "source": "clean_compile_fls_local_sources_only",
            "source_files": list(self.source_files),
            "definitions": {
                row.environment_name: row.to_dict() for row in self.definitions
            },
            "rejections": [row.to_dict() for row in self.rejections],
            "pdf_text_used": False,
        }


def _is_comment(value: str, position: int) -> bool:
    if value[position] != "%":
        return False
    backslashes = 0
    cursor = position - 1
    while cursor >= 0 and value[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 0


def _position_is_commented(value: str, position: int) -> bool:
    line_start = value.rfind("\n", 0, position) + 1
    cursor = line_start
    while cursor < position:
        if _is_comment(value, cursor):
            return True
        cursor += 1
    return False


def _strip_comments(value: str) -> str:
    output = list(value)
    cursor = 0
    while cursor < len(value):
        if _is_comment(value, cursor):
            newline = value.find("\n", cursor)
            end = len(value) if newline < 0 else newline
            for index in range(cursor, end):
                output[index] = " "
            cursor = end
            continue
        cursor += 1
    return "".join(output)


def _skip(value: str, position: int) -> int:
    while position < len(value):
        if value[position].isspace():
            position += 1
            continue
        if _is_comment(value, position):
            newline = value.find("\n", position + 1)
            position = len(value) if newline < 0 else newline + 1
            continue
        break
    return position


def _group(
    value: str,
    position: int,
    opening: str,
    closing: str,
) -> tuple[str, int] | None:
    position = _skip(value, position)
    if position >= len(value) or value[position] != opening:
        return None
    depth = 0
    content_start = position + 1
    cursor = position
    while cursor < len(value):
        if _is_comment(value, cursor):
            newline = value.find("\n", cursor + 1)
            cursor = len(value) if newline < 0 else newline + 1
            continue
        if value[cursor] == "\\":
            # Escaped braces are literal characters.  A control sequence's
            # arguments remain visible to this balanced-group scanner.
            cursor += 2
            continue
        if value[cursor] == opening:
            depth += 1
        elif value[cursor] == closing:
            depth -= 1
            if depth == 0:
                return value[content_start:cursor], cursor + 1
        cursor += 1
    return None


def _control_sequence(value: str, position: int) -> tuple[str, int] | None:
    if position >= len(value) or value[position] != "\\":
        return None
    cursor = position + 1
    if cursor >= len(value):
        return None
    if value[cursor].isalpha() or value[cursor] == "@":
        start = cursor
        cursor += 1
        while cursor < len(value) and (
            value[cursor].isalpha() or value[cursor] == "@"
        ):
            cursor += 1
        return value[start:cursor], cursor
    return value[cursor], cursor + 1


def _literal_dimension(value: str) -> bool:
    return bool(_LITERAL_DIMENSION.fullmatch(_strip_comments(value).strip()))


def _wrapper_body(
    body: str,
    expected_endpoint: str,
) -> tuple[str | None, str | None]:
    """Validate one replacement body and return its one list endpoint.

    ``expected_endpoint`` is either ``"begin"`` or ``"end"``.  Setup controls
    are intentionally parsed rather than regex-matched, so braces, extra
    commands, and visible text cannot hide inside an accepted wrapper.
    """

    text = _strip_comments(body)
    cursor = 0
    begin_kind: str | None = None
    end_kind: str | None = None
    while True:
        cursor = _skip(text, cursor)
        if cursor >= len(text):
            break
        command = _control_sequence(text, cursor)
        if command is None:
            return None, "visible_literal_or_malformed_control"
        name, after_name = command
        cursor = after_name
        if name in {"begin", "end"}:
            group = _group(text, cursor, "{", "}")
            if group is None:
                return None, "malformed_environment_group"
            environment, cursor = group
            environment = environment.strip()
            if environment not in _LIST_KINDS:
                return None, "wrapper_target_is_not_builtin_list"
            if name == "begin":
                if begin_kind is not None:
                    return None, "multiple_list_begin_tokens"
                begin_kind = environment
            else:
                if end_kind is not None:
                    return None, "multiple_list_end_tokens"
                end_kind = environment
            continue
        if name == "vspace":
            if cursor >= len(text) or text[cursor] != "*":
                return None, "only_starred_vspace_is_supported"
            cursor += 1
            dimension = _group(text, cursor, "{", "}")
            if dimension is None or not _literal_dimension(dimension[0]):
                return None, "vspace_dimension_is_not_literal"
            cursor = dimension[1]
            continue
        if name == "setlength":
            length_group = _group(text, cursor, "{", "}")
            if length_group is None:
                return None, "malformed_setlength_name"
            length_name = _strip_comments(length_group[0]).strip()
            if length_name.startswith("\\"):
                length_name = length_name[1:]
            if length_name not in _LIST_SPACING_LENGTHS:
                return None, "setlength_target_is_not_list_spacing"
            dimension_group = _group(text, length_group[1], "{", "}")
            if dimension_group is None or not _literal_dimension(
                dimension_group[0]
            ):
                return None, "setlength_dimension_is_not_literal"
            cursor = dimension_group[1]
            continue
        return None, "unknown_or_visible_command"

    if begin_kind is not None and end_kind is not None:
        return None, "wrapper_contains_both_begin_and_end"
    if expected_endpoint == "begin":
        if begin_kind is None:
            return None, "wrapper_has_no_begin_endpoint"
        return begin_kind, None
    if expected_endpoint == "end":
        if end_kind is None:
            return None, "wrapper_has_no_end_endpoint"
        return end_kind, None
    return None, "invalid_wrapper_endpoint_expectation"


def _declaration_rows(
    source_path: str,
    source: str,
) -> tuple[
    list[SourceListEnvironmentDefinition],
    list[SourceListEnvironmentRejection],
]:
    definitions: list[SourceListEnvironmentDefinition] = []
    rejections: list[SourceListEnvironmentRejection] = []
    declaration_matches = sorted(
        (
            match
            for pattern in (_STANDARD_DECLARATION, _XPARSE_DECLARATION)
            for match in pattern.finditer(source)
            if not _position_is_commented(source, match.start())
        ),
        key=lambda match: match.start(),
    )
    seen_offsets: set[int] = set()
    for match in declaration_matches:
        if match.start() in seen_offsets:
            continue
        seen_offsets.add(match.start())
        command = match.group("command")
        cursor = match.end()
        name_group = _group(source, cursor, "{", "}")
        if name_group is None:
            rejections.append(
                SourceListEnvironmentRejection(
                    source_path, None, command, "missing_or_unbalanced_name", match.start()
                )
            )
            continue
        environment = _strip_comments(name_group[0]).strip()
        cursor = name_group[1]
        if _SAFE_ENVIRONMENT.fullmatch(environment) is None:
            rejections.append(
                SourceListEnvironmentRejection(
                    source_path, environment, command, "invalid_environment_name", match.start()
                )
            )
            continue
        if environment in _LIST_KINDS:
            rejections.append(
                SourceListEnvironmentRejection(
                    source_path, environment, command, "cannot_override_builtin_list", match.start()
                )
            )
            continue

        xparse = command.endswith("DocumentEnvironment")
        if xparse:
            argument_spec = _group(source, cursor, "{", "}")
            if argument_spec is None:
                rejections.append(
                    SourceListEnvironmentRejection(
                        source_path, environment, command, "missing_argument_spec", match.start()
                    )
                )
                continue
            if _strip_comments(argument_spec[0]).strip():
                rejections.append(
                    SourceListEnvironmentRejection(
                        source_path,
                        environment,
                        command,
                        "nonzero_or_optional_arguments_not_supported",
                        match.start(),
                    )
                )
                continue
            cursor = argument_spec[1]
        else:
            optional = _skip(source, cursor)
            if optional < len(source) and source[optional] == "[":
                # ``[0]`` is an explicit zero-arity declaration, not an
                # optional/default argument.  It is equivalent to omitting
                # the arity for the purpose of this proof.  Every other
                # bracketed form remains rejected, including a second
                # bracketed default after ``[0]``.
                arity_group = _group(source, optional, "[", "]")
                if arity_group is not None and _strip_comments(
                    arity_group[0]
                ).strip() == "0":
                    cursor = _skip(source, arity_group[1])
                    if cursor >= len(source) or source[cursor] != "[":
                        pass
                    else:
                        rejections.append(
                            SourceListEnvironmentRejection(
                                source_path,
                                environment,
                                command,
                                "optional_or_default_arguments_not_supported",
                                match.start(),
                            )
                        )
                        continue
                else:
                    rejections.append(
                        SourceListEnvironmentRejection(
                            source_path,
                            environment,
                            command,
                            "optional_or_default_arguments_not_supported",
                            match.start(),
                        )
                    )
                    continue

        begin_group = _group(source, cursor, "{", "}")
        if begin_group is None:
            rejections.append(
                SourceListEnvironmentRejection(
                    source_path, environment, command, "missing_or_unbalanced_begin_body", match.start()
                )
            )
            continue
        end_group = _group(source, begin_group[1], "{", "}")
        if end_group is None:
            rejections.append(
                SourceListEnvironmentRejection(
                    source_path, environment, command, "missing_or_unbalanced_end_body", match.start()
                )
            )
            continue
        begin_body, begin_reason = _wrapper_body(begin_group[0], "begin")
        end_body, end_reason = _wrapper_body(end_group[0], "end")
        if begin_reason is not None:
            rejections.append(
                SourceListEnvironmentRejection(
                    source_path,
                    environment,
                    command,
                    f"begin_body_{begin_reason}",
                    match.start(),
                )
            )
            continue
        if end_reason is not None:
            rejections.append(
                SourceListEnvironmentRejection(
                    source_path,
                    environment,
                    command,
                    f"end_body_{end_reason}",
                    match.start(),
                )
            )
            continue
        assert begin_body is not None and end_body is not None
        if begin_body != end_body:
            rejections.append(
                SourceListEnvironmentRejection(
                    source_path,
                    environment,
                    command,
                    "wrapper_begin_end_list_mismatch",
                    match.start(),
                )
            )
            continue
        definitions.append(
            SourceListEnvironmentDefinition(
                environment_name=environment,
                list_kind=begin_body,
                declaration_command=command,
                begin_body=_strip_comments(begin_group[0]).strip(),
                end_body=_strip_comments(end_group[0]).strip(),
                source_path=source_path,
            )
        )
    return definitions, rejections


def collect_source_list_environment_definitions(
    sources: Mapping[str, str],
) -> SourceListEnvironmentRegistry:
    """Collect unique safe list aliases from already-selected source files.

    ``sources`` is intentionally a mapping supplied by the caller after the
    clean compilation's recorder-file filtering.  Passing an inactive source
    here is therefore a caller-visible provenance error, not something this
    parser silently discovers from the filesystem.
    """

    candidates: dict[
        str,
        set[tuple[str, str, str, str]],
    ] = defaultdict(set)
    # A source file can contain more than one declaration for an environment.
    # Keeping a valid earlier declaration when a later declaration is
    # malformed would silently guess TeX's execution semantics.  Mark every
    # named rejected declaration as tainted and keep that alias unresolved.
    invalid_names: set[str] = set()
    rejections: list[SourceListEnvironmentRejection] = []
    for source_path, source in sorted(sources.items()):
        rows, local_rejections = _declaration_rows(source_path, source)
        rejections.extend(local_rejections)
        invalid_names.update(
            row.environment_name
            for row in local_rejections
            if row.environment_name is not None
        )
        for row in rows:
            candidates[row.environment_name].add(
                (
                    row.list_kind,
                    row.declaration_command,
                    row.begin_body,
                    row.end_body,
                )
            )

    definitions: list[SourceListEnvironmentDefinition] = []
    for environment, values in sorted(candidates.items()):
        if environment in invalid_names:
            continue
        if len(values) != 1:
            rejections.append(
                SourceListEnvironmentRejection(
                    "<executed-source-set>",
                    environment,
                    None,
                    "conflicting_active_definitions",
                )
            )
            continue
        list_kind, command, begin_body, end_body = next(iter(values))
        source_path = next(
            source_path
            for source_path, source in sorted(sources.items())
            for row in _declaration_rows(source_path, source)[0]
            if (
                row.environment_name == environment
                and row.list_kind == list_kind
                and row.declaration_command == command
                and row.begin_body == begin_body
                and row.end_body == end_body
            )
        )
        definitions.append(
            SourceListEnvironmentDefinition(
                environment_name=environment,
                list_kind=list_kind,
                declaration_command=command,
                begin_body=begin_body,
                end_body=end_body,
                source_path=source_path,
            )
        )
    return SourceListEnvironmentRegistry(
        source_files=tuple(sorted(sources)),
        definitions=tuple(definitions),
        rejections=tuple(rejections),
    )


# Concise aliases for callers that use the broader source-environment name.
collect_source_environment_definitions = collect_source_list_environment_definitions


__all__ = [
    "SourceListEnvironmentDefinition",
    "SourceListEnvironmentRejection",
    "SourceListEnvironmentRegistry",
    "collect_source_environment_definitions",
    "collect_source_list_environment_definitions",
]
