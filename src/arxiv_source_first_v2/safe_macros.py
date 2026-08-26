"""Fail-closed expansion of simple, source-defined visible LaTeX macros.

This module is deliberately not a TeX interpreter.  It accepts only project
macros whose definitions are statically provable to be transparent visible
text wrappers:

* zero or one mandatory argument;
* a one-argument macro uses ``#1`` exactly once (no deletion or duplication);
* every control sequence in the body is either another accepted project macro
  or a small, explicit formatting/math allow-list; and
* the dependency graph is finite and acyclic.

Definitions containing conditionals, assignments, counters, I/O, layout
operations, labels/references/keys, dynamic control-sequence construction, or
unknown commands are rejected.  The caller supplies the *executed project
source files*; no class/package tree is searched implicitly.

Expansion returns source-only text plus auditable provenance.  PDF text is not
an input to this module and can never become expansion content.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
import re
from collections.abc import Iterable, Mapping, Sequence


DEFAULT_MAX_DEPENDENCY_DEPTH = 16
DEFAULT_MAX_EXPANSION_DEPTH = 32
DEFAULT_MAX_EXPANSIONS = 10_000
DEFAULT_MAX_OUTPUT_CHARACTERS = 1_000_000
DEFAULT_MAX_DEFINITIONS = 10_000
DEFAULT_MAX_BODY_CHARACTERS = 65_536


class SafeMacroError(ValueError):
    """Base class for fail-closed collection and expansion errors."""


class MacroExpansionError(SafeMacroError):
    """Raised when a fragment cannot be expanded without guessing."""


@dataclasses.dataclass(frozen=True)
class SafeMacroDefinition:
    """One accepted source definition.

    All spans are half-open character offsets in ``source_file``.  ``body_span``
    excludes its outer braces, while ``declaration_span`` includes the complete
    declaration.
    """

    name: str
    arity: int
    body: str
    source_file: Path
    source_line: int
    declaration_kind: str
    declaration_span: tuple[int, int]
    body_span: tuple[int, int]
    dependencies: tuple[str, ...]
    uses_xspace: bool
    body_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arity": self.arity,
            "body": self.body,
            "source_file": str(self.source_file),
            "source_line": self.source_line,
            "declaration_kind": self.declaration_kind,
            "declaration_span": list(self.declaration_span),
            "body_span": list(self.body_span),
            "dependencies": list(self.dependencies),
            "uses_xspace": self.uses_xspace,
            "body_sha256": self.body_sha256,
        }


@dataclasses.dataclass(frozen=True)
class MacroRejection:
    """A definition that was not admitted to the safe registry."""

    name: str
    source_file: Path
    source_line: int
    declaration_span: tuple[int, int]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_file": str(self.source_file),
            "source_line": self.source_line,
            "declaration_span": list(self.declaration_span),
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class SafeMacroRegistry:
    """Immutable result of scanning an explicit executed-source allow-list."""

    definitions: tuple[SafeMacroDefinition, ...]
    rejections: tuple[MacroRejection, ...]
    source_files: tuple[Path, ...]
    definitions_seen: int
    allowed_commands: tuple[tuple[str, int], ...]

    @property
    def by_name(self) -> dict[str, SafeMacroDefinition]:
        return {definition.name: definition for definition in self.definitions}

    @property
    def accepted_names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.definitions)

    @property
    def rejected_names(self) -> tuple[str, ...]:
        return tuple(sorted({rejection.name for rejection in self.rejections}))

    def as_report(self) -> dict[str, object]:
        return {
            "policy": "source_only_visible_macro_dag_v1",
            "source_files": [str(path) for path in self.source_files],
            "definitions_seen": self.definitions_seen,
            "definitions_accepted": len(self.definitions),
            "definitions_rejected": len(self.rejections),
            "accepted_names": list(self.accepted_names),
            "rejections": [rejection.as_dict() for rejection in self.rejections],
            "allowed_commands": [
                {"name": name, "arity": arity}
                for name, arity in self.allowed_commands
            ],
        }


@dataclasses.dataclass(frozen=True)
class ExpansionProvenance:
    """One source macro invocation contributing to expanded output.

    ``invocation_span`` is exact for invocations in the caller's fragment.  A
    nested invocation originating in a substituted definition body has no
    direct span in that fragment and therefore records ``None`` rather than a
    fabricated offset.  Its definition span and complete expansion stack are
    still exact.
    """

    macro_name: str
    invocation_source_file: Path | None
    invocation_span: tuple[int, int] | None
    argument_spans: tuple[tuple[int, int], ...]
    definition_source_file: Path
    definition_span: tuple[int, int]
    definition_body_span: tuple[int, int]
    definition_body_sha256: str
    expansion_stack: tuple[str, ...]
    depth: int
    output_span: tuple[int, int]
    argument_sha256: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "macro_name": self.macro_name,
            "invocation_source_file": (
                str(self.invocation_source_file)
                if self.invocation_source_file is not None
                else None
            ),
            "invocation_span": (
                list(self.invocation_span)
                if self.invocation_span is not None
                else None
            ),
            "argument_spans": [list(span) for span in self.argument_spans],
            "definition_source_file": str(self.definition_source_file),
            "definition_span": list(self.definition_span),
            "definition_body_span": list(self.definition_body_span),
            "definition_body_sha256": self.definition_body_sha256,
            "expansion_stack": list(self.expansion_stack),
            "depth": self.depth,
            "output_span": list(self.output_span),
            "argument_sha256": list(self.argument_sha256),
        }


@dataclasses.dataclass(frozen=True)
class ExpansionResult:
    """Expanded source text and strict source-definition provenance."""

    text: str
    provenance: tuple[ExpansionProvenance, ...]
    macros_used: tuple[str, ...]
    expansion_count: int
    maximum_depth: int

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "macros_used": list(self.macros_used),
            "expansion_count": self.expansion_count,
            "maximum_depth": self.maximum_depth,
            "provenance": [item.as_dict() for item in self.provenance],
        }


@dataclasses.dataclass(frozen=True)
class _ParsedDefinition:
    name: str
    arity: int
    body: str
    source_file: Path
    source_line: int
    declaration_kind: str
    declaration_span: tuple[int, int]
    body_span: tuple[int, int]


@dataclasses.dataclass
class _Budget:
    max_depth: int
    max_expansions: int
    max_output_characters: int
    expansions: int = 0
    maximum_depth: int = 0


_DECLARATION_RE = re.compile(
    r"\\(?P<kind>newcommand|renewcommand|providecommand|DeclareRobustCommand)"
    r"\*?\s*(?:\{\s*\\(?P<braced>[A-Za-z@]+)\s*\}|"
    r"\\(?P<direct>[A-Za-z@]+))"
)
_UNSUPPORTED_DECLARATION_RE = re.compile(
    r"\\(?P<kind>def|gdef|edef|xdef|let|futurelet)\s*\\(?P<name>[A-Za-z@]+)"
)
_CONTROL_WORD_RE = re.compile(r"\\([A-Za-z@]+)")


# Only TeX control symbols with an unambiguous visible-character meaning are
# admitted.  In particular, ``\\\\`` (line break), ``\\ `` (forced space),
# accent commands, and math-spacing commands are not silently treated as
# literal source.  A downstream caller can keep using the word-form wrappers
# above, but cannot widen this set: control symbols are deliberately a closed
# policy surface.
_ALLOWED_VISIBLE_CONTROL_SYMBOLS = frozenset("%$#&_{}")


# These commands only wrap visible content or preserve source math.  Their
# output is intentionally left as LaTeX for the downstream source parser.
_DEFAULT_ALLOWED_COMMAND_ARITIES: dict[str, int] = {
    "emph": 1,
    "ensuremath": 1,
    "mathbf": 1,
    "mathbb": 1,
    "mathcal": 1,
    "mathit": 1,
    "mathrm": 1,
    "mathsf": 1,
    "mathtt": 1,
    "operatorname": 1,
    "textbf": 1,
    "textit": 1,
    "textmd": 1,
    "textnormal": 1,
    "textrm": 1,
    "textsc": 1,
    "textsf": 1,
    "textsl": 1,
    "texttt": 1,
    "textup": 1,
    "dfrac": 2,
    "frac": 2,
    "sfrac": 2,
    "tfrac": 2,
    # xspace is removed only after a conservative follower check.
    "xspace": 0,
}


_FORBIDDEN_COMMANDS_BY_CLASS: dict[str, frozenset[str]] = {
    "condition": frozenset(
        {
            "if",
            "ifcase",
            "ifcat",
            "ifcsname",
            "ifdefined",
            "ifdim",
            "ifeof",
            "iffalse",
            "ifhbox",
            "ifhmode",
            "ifinner",
            "ifmmode",
            "ifnum",
            "ifodd",
            "iftrue",
            "ifvbox",
            "ifvmode",
            "ifvoid",
            "ifx",
            "unless",
            "else",
            "fi",
            "or",
        }
    ),
    "counter": frozenset(
        {
            "addtocounter",
            "alph",
            "Alph",
            "arabic",
            "fnsymbol",
            "newcounter",
            "number",
            "refstepcounter",
            "roman",
            "Roman",
            "setcounter",
            "stepcounter",
            "the",
            "value",
        }
    ),
    "assignment": frozenset(
        {
            "advance",
            "chardef",
            "count",
            "countdef",
            "def",
            "dimen",
            "dimendef",
            "divide",
            "edef",
            "futurelet",
            "gdef",
            "global",
            "let",
            "multiply",
            "muskip",
            "muskipdef",
            "newcommand",
            "providecommand",
            "renewcommand",
            "skip",
            "skipdef",
            "toks",
            "toksdef",
            "xdef",
        }
    ),
    "io": frozenset(
        {
            "closein",
            "closeout",
            "endinput",
            "immediate",
            "include",
            "input",
            "openin",
            "openout",
            "read",
            "readline",
            "special",
            "usepackage",
            "write",
        }
    ),
    "layout": frozenset(
        {
            "begin",
            "break",
            "clearpage",
            "end",
            "enlargethispage",
            "footnote",
            "hfill",
            "hfil",
            "hskip",
            "hspace",
            "includegraphics",
            "indent",
            "kern",
            "linebreak",
            "marginpar",
            "mbox",
            "newpage",
            "noindent",
            "nopagebreak",
            "pagebreak",
            "par",
            "parbox",
            "raisebox",
            "rule",
            "small",
            "tiny",
            "vfill",
            "vfil",
            "vskip",
            "vspace",
        }
    ),
    "label_or_key": frozenset(
        {
            "autoref",
            "cite",
            "citep",
            "citet",
            "Cref",
            "cref",
            "eqref",
            "href",
            "label",
            "pageref",
            "ref",
            "url",
        }
    ),
    "control_sequence_name": frozenset(
        {
            "csname",
            "endcsname",
            "expandafter",
            "meaning",
            "noexpand",
            "string",
        }
    ),
    "color_or_literal": frozenset(
        {
            "color",
            "colorbox",
            "definecolor",
            "fcolorbox",
            "pagecolor",
            "pdfcolorstack",
            "pdfliteral",
            "textcolor",
        }
    ),
}

_FORBIDDEN_PREFIXES = (
    "pdfextension",
    "pdfobj",
    "pdfrefobj",
    "pdfshellescape",
    "pdfxform",
    "shellescape",
)
_XSPACE_NO_SPACE_BEFORE = frozenset(".,;:!?/)-]}$'\"")


def _line_number(source: str, offset: int) -> int:
    return source.count("\n", 0, max(0, offset)) + 1


def _is_escaped(source: str, offset: int) -> bool:
    slashes = 0
    cursor = offset - 1
    while cursor >= 0 and source[cursor] == "\\":
        slashes += 1
        cursor -= 1
    return bool(slashes % 2)


def _comment_mask(source: str) -> str:
    """Blank comments while retaining every source offset."""

    output = list(source)
    cursor = 0
    while cursor < len(source):
        if source[cursor] != "%" or _is_escaped(source, cursor):
            cursor += 1
            continue
        end = source.find("\n", cursor)
        end = len(source) if end < 0 else end
        for index in range(cursor, end):
            output[index] = " "
        cursor = end
    return "".join(output)


def _contains_active_comment(source: str) -> bool:
    return any(
        character == "%" and not _is_escaped(source, index)
        for index, character in enumerate(source)
    )


def _balanced_end(source: str, opening: int) -> int | None:
    if opening >= len(source) or source[opening] != "{":
        return None
    depth = 0
    cursor = opening
    while cursor < len(source):
        if source[cursor] == "\\" and cursor + 1 < len(source):
            cursor += 2
            continue
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return None


def _skip_space(source: str, cursor: int) -> int:
    while cursor < len(source) and source[cursor].isspace():
        cursor += 1
    return cursor


def _control_word_end(source: str, start: int) -> int:
    cursor = start + 1
    while cursor < len(source) and (
        source[cursor].isalpha() or source[cursor] == "@"
    ):
        cursor += 1
    return cursor


def _forbidden_reason(name: str) -> str | None:
    for category, commands in _FORBIDDEN_COMMANDS_BY_CLASS.items():
        if name in commands:
            return f"forbidden_command:{category}:{name}"
    if any(name.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
        return f"forbidden_command:io_or_pdf_primitive:{name}"
    return None


def _argument_spans(
    source: str,
    command_end: int,
    arity: int,
) -> tuple[tuple[tuple[int, int], ...], str | None]:
    cursor = command_end
    spans: list[tuple[int, int]] = []
    for _ in range(arity):
        cursor = _skip_space(source, cursor)
        if cursor >= len(source) or source[cursor] != "{":
            return (), "mandatory_braced_argument_missing"
        end = _balanced_end(source, cursor)
        if end is None:
            return (), "mandatory_braced_argument_unbalanced"
        spans.append((cursor + 1, end - 1))
        cursor = end
    return tuple(spans), None


def _parameter_count(body: str) -> tuple[int, str | None]:
    count = 0
    cursor = 0
    while cursor < len(body):
        if body[cursor] == "\\" and cursor + 1 < len(body):
            cursor += 2
            continue
        if body[cursor] != "#":
            cursor += 1
            continue
        if body.startswith("#1", cursor):
            count += 1
            cursor += 2
            continue
        return count, "unsupported_parameter_token"
    return count, None


def _parse_supported_declarations(
    source_file: Path,
    source: str,
) -> tuple[list[_ParsedDefinition], list[MacroRejection], list[tuple[int, int]]]:
    active = _comment_mask(source)
    parsed: list[_ParsedDefinition] = []
    rejected: list[MacroRejection] = []
    declaration_ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        match = _DECLARATION_RE.search(active, cursor)
        if match is None:
            break
        name = str(match.group("braced") or match.group("direct"))
        position = _skip_space(active, match.end())
        arity = 0
        reason: str | None = None
        if position < len(active) and active[position] == "[":
            closing = active.find("]", position + 1)
            if closing < 0:
                reason = "malformed_arity"
                closing = position
            else:
                value = active[position + 1 : closing].strip()
                if not value.isdigit():
                    reason = "malformed_arity"
                else:
                    arity = int(value)
                position = _skip_space(active, closing + 1)
                if position < len(active) and active[position] == "[":
                    reason = "optional_argument_default_is_unsupported"
        if position >= len(active) or active[position] != "{":
            end = max(match.end(), position)
            rejected.append(
                MacroRejection(
                    name=name,
                    source_file=source_file,
                    source_line=_line_number(source, match.start()),
                    declaration_span=(match.start(), end),
                    reason=reason or "definition_body_missing",
                )
            )
            cursor = max(match.end(), position + 1)
            continue
        body_end = _balanced_end(active, position)
        if body_end is None:
            rejected.append(
                MacroRejection(
                    name=name,
                    source_file=source_file,
                    source_line=_line_number(source, match.start()),
                    declaration_span=(match.start(), len(source)),
                    reason=reason or "definition_body_unbalanced",
                )
            )
            break
        declaration_span = (match.start(), body_end)
        declaration_ranges.append(declaration_span)
        if reason is not None or arity not in {0, 1}:
            rejected.append(
                MacroRejection(
                    name=name,
                    source_file=source_file,
                    source_line=_line_number(source, match.start()),
                    declaration_span=declaration_span,
                    reason=reason or f"unsupported_arity:{arity}",
                )
            )
        else:
            parsed.append(
                _ParsedDefinition(
                    name=name,
                    arity=arity,
                    body=source[position + 1 : body_end - 1],
                    source_file=source_file,
                    source_line=_line_number(source, match.start()),
                    declaration_kind=str(match.group("kind")),
                    declaration_span=declaration_span,
                    body_span=(position + 1, body_end - 1),
                )
            )
        cursor = body_end
    return parsed, rejected, declaration_ranges


def _inside_ranges(offset: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start <= offset < end for start, end in ranges)


def _definition_control_sequences(
    body: str,
) -> list[tuple[str, int, int, bool]]:
    """Return every control word/symbol while retaining exact body offsets."""

    active = _comment_mask(body)
    output: list[tuple[str, int, int, bool]] = []
    cursor = 0
    while cursor < len(active):
        if active[cursor] != "\\":
            cursor += 1
            continue
        if cursor + 1 >= len(active):
            output.append(("", cursor, cursor + 1, False))
            cursor += 1
            continue
        if active[cursor + 1].isalpha() or active[cursor + 1] == "@":
            end = _control_word_end(active, cursor)
            output.append((active[cursor + 1 : end], cursor, end, True))
            cursor = end
            continue
        output.append((active[cursor + 1], cursor, cursor + 2, False))
        cursor += 2
    return output


def _has_literal_visibility(body: str, arity: int) -> bool:
    value = body.replace("#1", "ARGUMENT" if arity == 1 else "")
    value = _CONTROL_WORD_RE.sub("", value)
    value = re.sub(r"[{}$\s]", "", value)
    return bool(value)


def _definition_rejection_reason(
    definition: _ParsedDefinition,
    candidate_arities: Mapping[str, int],
    allowed_commands: Mapping[str, int],
    *,
    max_body_characters: int,
) -> tuple[str | None, tuple[str, ...], bool, bool]:
    body = definition.body
    if len(body) > max_body_characters:
        return "definition_body_too_large", (), False, False
    if _contains_active_comment(body):
        return "comment_in_definition_body", (), False, False
    parameters, parameter_error = _parameter_count(body)
    if parameter_error is not None:
        return parameter_error, (), False, False
    if parameters != definition.arity:
        return (
            f"argument_occurrences={parameters}:expected={definition.arity}",
            (),
            False,
            False,
        )
    dependencies: set[str] = set()
    uses_xspace = False
    for name, start, end, is_word in _definition_control_sequences(body):
        if not is_word:
            if name not in _ALLOWED_VISIBLE_CONTROL_SYMBOLS:
                rendered = "trailing_backslash" if not name else repr(name)
                return (
                    f"unknown_or_nonvisible_control_symbol:{rendered}",
                    (),
                    False,
                    False,
                )
            continue
        forbidden = _forbidden_reason(name)
        if forbidden is not None:
            return forbidden, (), False, False
        if name in allowed_commands:
            arity = int(allowed_commands[name])
            spans, argument_error = _argument_spans(body, end, arity)
            if argument_error is not None:
                return f"{argument_error}:{name}", (), False, False
            del spans
            if name == "xspace":
                tail = body[end:]
                # xspace may sit just inside one or more grouping braces, but
                # static content after it would require emulating xspace.
                if tail.strip().strip("}").strip():
                    return "xspace_is_not_terminal", (), False, False
                uses_xspace = True
            continue
        if name in candidate_arities:
            arity = int(candidate_arities[name])
            _, argument_error = _argument_spans(body, end, arity)
            if argument_error is not None:
                return f"{argument_error}:{name}", (), False, False
            dependencies.add(name)
            continue
        return f"unknown_command:{name}", (), False, False
    return (
        None,
        tuple(sorted(dependencies)),
        uses_xspace,
        _has_literal_visibility(body, definition.arity),
    )


def collect_safe_macros(
    executed_source_files: Iterable[str | Path],
    *,
    additional_allowed_commands: Mapping[str, int] | None = None,
    max_dependency_depth: int = DEFAULT_MAX_DEPENDENCY_DEPTH,
    max_definitions: int = DEFAULT_MAX_DEFINITIONS,
    max_body_characters: int = DEFAULT_MAX_BODY_CHARACTERS,
) -> SafeMacroRegistry:
    r"""Collect a safe macro DAG from the caller's executed project sources.

    Multiple declarations of the same name are always ambiguous, even when
    their bodies happen to be identical.  Unsupported ``\def``/``\let``-style
    declarations also poison that name.  This deliberately avoids simulating
    TeX scoping or execution-time redefinition.
    """

    if max_dependency_depth < 1 or max_definitions < 1 or max_body_characters < 1:
        raise ValueError("macro collection limits must be positive")
    allowed_commands = dict(_DEFAULT_ALLOWED_COMMAND_ARITIES)
    for name, arity in (additional_allowed_commands or {}).items():
        if not re.fullmatch(r"[A-Za-z@]+", str(name)):
            raise ValueError(f"invalid allowed command name: {name!r}")
        if int(arity) < 0 or int(arity) > 4:
            raise ValueError(f"invalid allowed command arity: {name}={arity}")
        forbidden = _forbidden_reason(str(name))
        if forbidden is not None:
            raise ValueError(f"cannot allow forbidden command {name}: {forbidden}")
        allowed_commands[str(name)] = int(arity)

    files: list[Path] = []
    seen_files: set[Path] = set()
    for value in executed_source_files:
        path = Path(value).expanduser().resolve()
        if path in seen_files:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        seen_files.add(path)
        files.append(path)

    parsed: list[_ParsedDefinition] = []
    rejections: list[MacroRejection] = []
    unsupported_names: set[str] = set()
    definitions_seen = 0
    for path in files:
        source = path.read_text(encoding="utf-8", errors="replace")
        active = _comment_mask(source)
        file_parsed, file_rejected, declaration_ranges = (
            _parse_supported_declarations(path, source)
        )
        parsed.extend(file_parsed)
        rejections.extend(file_rejected)
        definitions_seen += len(file_parsed) + len(file_rejected)
        for match in _UNSUPPORTED_DECLARATION_RE.finditer(active):
            if _inside_ranges(match.start(), declaration_ranges):
                continue
            name = str(match.group("name"))
            unsupported_names.add(name)
            definitions_seen += 1
            rejections.append(
                MacroRejection(
                    name=name,
                    source_file=path,
                    source_line=_line_number(source, match.start()),
                    declaration_span=(match.start(), match.end()),
                    reason=f"unsupported_declaration:{match.group('kind')}",
                )
            )
    if definitions_seen > max_definitions:
        raise SafeMacroError(
            f"definition_limit_exceeded:{definitions_seen}>{max_definitions}"
        )

    grouped: dict[str, list[_ParsedDefinition]] = {}
    for definition in parsed:
        grouped.setdefault(definition.name, []).append(definition)
    reasons: dict[str, str] = {}
    unique: dict[str, _ParsedDefinition] = {}
    for name, values in grouped.items():
        if len(values) != 1 or name in unsupported_names:
            reasons[name] = "ambiguous_or_unsupported_redefinition"
            for value in values:
                rejections.append(
                    MacroRejection(
                        name=name,
                        source_file=value.source_file,
                        source_line=value.source_line,
                        declaration_span=value.declaration_span,
                        reason=reasons[name],
                    )
                )
        else:
            unique[name] = values[0]
    # The fixed allow-list describes the standard command semantics only.  A
    # project declaration with the same name makes those semantics ambiguous;
    # never let a parent macro bypass the declaration by resolving the name as
    # a built-in wrapper.  This also catches unsupported ``\\def`` overrides.
    shadowed_allowed = (set(grouped) | unsupported_names) & set(allowed_commands)
    effective_allowed_commands = {
        name: arity
        for name, arity in allowed_commands.items()
        if name not in shadowed_allowed
    }
    for name in shadowed_allowed & set(unique):
        reasons[name] = "safe_passthrough_command_redefined"
    for name in set(unique):
        forbidden = _forbidden_reason(name)
        if forbidden is not None:
            reasons[name] = f"forbidden_definition_name:{forbidden}"
    candidate_arities = {name: value.arity for name, value in unique.items()}
    dependency_map: dict[str, tuple[str, ...]] = {}
    xspace_map: dict[str, bool] = {}
    literal_visibility: dict[str, bool] = {}
    for name, definition in unique.items():
        if name in reasons:
            continue
        reason, dependencies, uses_xspace, visible = _definition_rejection_reason(
            definition,
            candidate_arities,
            effective_allowed_commands,
            max_body_characters=max_body_characters,
        )
        if reason is not None:
            reasons[name] = reason
        else:
            dependency_map[name] = dependencies
            xspace_map[name] = uses_xspace
            literal_visibility[name] = visible

    states: dict[str, str] = {}
    cycle_names: set[str] = set()

    def visit(name: str, path: tuple[str, ...]) -> bool:
        if name in reasons:
            return False
        state = states.get(name)
        if state == "passed":
            return True
        if state == "failed":
            return False
        if state == "active":
            if name in path:
                cycle_names.update(path[path.index(name) :])
            else:
                cycle_names.add(name)
            return False
        if len(path) >= max_dependency_depth:
            reasons[name] = f"dependency_depth_exceeded:>{max_dependency_depth}"
            states[name] = "failed"
            return False
        states[name] = "active"
        dependencies = dependency_map.get(name, ())
        for dependency in dependencies:
            if dependency not in unique or not visit(dependency, (*path, name)):
                # Only nodes on the active recursion slice are members of the
                # cycle.  An upstream dependent is rejected, but is not itself
                # mislabeled as cyclic.
                if name not in cycle_names and name not in reasons:
                    reasons[name] = f"dependency_rejected:{dependency}"
                states[name] = "failed"
                return False
        if not literal_visibility.get(name, False) and not dependencies:
            reasons[name] = "no_visible_output"
            states[name] = "failed"
            return False
        states[name] = "passed"
        return True

    for name in sorted(unique):
        visit(name, ())
    for name in cycle_names:
        reasons[name] = "dependency_cycle"
        states[name] = "failed"

    accepted: list[SafeMacroDefinition] = []
    already_rejected = {
        (item.name, item.source_file, item.declaration_span, item.reason)
        for item in rejections
    }
    for name, definition in sorted(
        unique.items(), key=lambda item: (str(item[1].source_file), item[1].declaration_span)
    ):
        reason = reasons.get(name)
        if reason is not None:
            key = (name, definition.source_file, definition.declaration_span, reason)
            if key not in already_rejected:
                rejections.append(
                    MacroRejection(
                        name=name,
                        source_file=definition.source_file,
                        source_line=definition.source_line,
                        declaration_span=definition.declaration_span,
                        reason=reason,
                    )
                )
            continue
        accepted.append(
            SafeMacroDefinition(
                name=name,
                arity=definition.arity,
                body=definition.body,
                source_file=definition.source_file,
                source_line=definition.source_line,
                declaration_kind=definition.declaration_kind,
                declaration_span=definition.declaration_span,
                body_span=definition.body_span,
                dependencies=dependency_map[name],
                uses_xspace=xspace_map[name],
                body_sha256=hashlib.sha256(
                    definition.body.encode("utf-8")
                ).hexdigest(),
            )
        )
    rejections.sort(
        key=lambda item: (
            str(item.source_file),
            item.declaration_span,
            item.name,
            item.reason,
        )
    )
    return SafeMacroRegistry(
        definitions=tuple(accepted),
        rejections=tuple(rejections),
        source_files=tuple(files),
        definitions_seen=definitions_seen,
        allowed_commands=tuple(sorted(effective_allowed_commands.items())),
    )


def _shift_provenance(
    item: ExpansionProvenance,
    offset: int,
) -> ExpansionProvenance:
    return dataclasses.replace(
        item,
        output_span=(item.output_span[0] + offset, item.output_span[1] + offset),
    )


def _replace_sentinel_in_provenance(
    items: Sequence[ExpansionProvenance],
    sentinel_span: tuple[int, int],
    replacement_length: int,
) -> tuple[ExpansionProvenance, ...]:
    left, right = sentinel_span
    delta = replacement_length - (right - left)
    output: list[ExpansionProvenance] = []
    for item in items:
        start, end = item.output_span
        if start >= right:
            start += delta
            end += delta
        elif start <= left and end >= right:
            end += delta
        elif end > left and start < right:
            raise MacroExpansionError("provenance partially overlaps argument sentinel")
        output.append(dataclasses.replace(item, output_span=(start, end)))
    return tuple(output)


def _xspace_suffix(
    source: str,
    invocation_end: int,
    *,
    consumed_control_space: bool,
) -> str:
    """Return the only conservatively provable visible xspace suffix."""

    if invocation_end >= len(source):
        return ""
    follower = source[invocation_end]
    if follower.isspace():
        # An argument-ending brace does not consume following whitespace.
        return ""
    if follower in _XSPACE_NO_SPACE_BEFORE:
        return ""
    if follower.isalnum():
        return " "
    if consumed_control_space and follower in "([{":
        raise MacroExpansionError(
            f"xspace follower is ambiguous: {follower!r}"
        )
    if follower in "([{\\":
        raise MacroExpansionError(
            f"xspace follower is ambiguous: {follower!r}"
        )
    return " "


def _strip_xspace(body: str) -> str:
    return re.sub(r"\\xspace\b", "", body)


def _expand_text(
    source: str,
    *,
    definitions: Mapping[str, SafeMacroDefinition],
    rejected_names: frozenset[str],
    allowed_commands: Mapping[str, int],
    source_file: Path | None,
    source_base_offset: int,
    precise_source_offsets: bool,
    depth: int,
    stack: tuple[str, ...],
    budget: _Budget,
) -> tuple[str, tuple[ExpansionProvenance, ...]]:
    if depth > budget.max_depth:
        raise MacroExpansionError(
            f"expansion_depth_exceeded:{depth}>{budget.max_depth}"
        )
    budget.maximum_depth = max(budget.maximum_depth, depth)
    if _contains_active_comment(source):
        raise MacroExpansionError("active comments are unsupported during expansion")
    output: list[str] = []
    provenance: list[ExpansionProvenance] = []
    output_length = 0

    def append(value: str) -> None:
        nonlocal output_length
        output_length += len(value)
        if output_length > budget.max_output_characters:
            raise MacroExpansionError(
                "expanded_output_limit_exceeded:>"
                f"{budget.max_output_characters}"
            )
        output.append(value)

    cursor = 0
    while cursor < len(source):
        if source[cursor] != "\\":
            append(source[cursor])
            cursor += 1
            continue
        if cursor + 1 >= len(source):
            raise MacroExpansionError("trailing backslash")
        if not (source[cursor + 1].isalpha() or source[cursor + 1] == "@"):
            symbol = source[cursor + 1]
            if symbol not in _ALLOWED_VISIBLE_CONTROL_SYMBOLS:
                raise MacroExpansionError(
                    f"unknown_or_nonvisible_control_symbol:{symbol!r}"
                )
            append(source[cursor : cursor + 2])
            cursor += 2
            continue
        command_end = _control_word_end(source, cursor)
        name = source[cursor + 1 : command_end]
        forbidden = _forbidden_reason(name)
        if forbidden is not None:
            raise MacroExpansionError(forbidden)
        definition = definitions.get(name)
        if definition is None:
            if name in rejected_names:
                raise MacroExpansionError(f"macro_definition_rejected:{name}")
            if name not in allowed_commands or name == "xspace":
                raise MacroExpansionError(f"unknown_command:{name}")
            # Validate mandatory arguments without consuming them.  They are
            # scanned next so project macros nested inside remain expandable.
            _, error = _argument_spans(source, command_end, allowed_commands[name])
            if error is not None:
                raise MacroExpansionError(f"{error}:{name}")
            append(source[cursor:command_end])
            cursor = command_end
            continue

        budget.expansions += 1
        if budget.expansions > budget.max_expansions:
            raise MacroExpansionError(
                f"expansion_count_exceeded:>{budget.max_expansions}"
            )
        consumed_control_space = False
        argument_values: list[str] = []
        argument_spans_local: list[tuple[int, int]] = []
        invocation_end = command_end
        if definition.arity == 0:
            # TeX discards whitespace used to terminate a control word.
            invocation_end = _skip_space(source, command_end)
            consumed_control_space = invocation_end > command_end
        else:
            if command_end < len(source) and source[command_end] == "[":
                raise MacroExpansionError(f"optional_invocation_unsupported:{name}")
            spans, error = _argument_spans(source, command_end, definition.arity)
            if error is not None:
                raise MacroExpansionError(f"{error}:{name}")
            argument_spans_local.extend(spans)
            invocation_end = spans[-1][1] + 1
            for start, end in spans:
                argument_values.append(source[start:end])

        expanded_arguments: list[tuple[str, tuple[ExpansionProvenance, ...]]] = []
        for argument_index, argument in enumerate(argument_values):
            span = argument_spans_local[argument_index]
            expanded_arguments.append(
                _expand_text(
                    argument,
                    definitions=definitions,
                    rejected_names=rejected_names,
                    allowed_commands=allowed_commands,
                    source_file=source_file,
                    source_base_offset=source_base_offset + span[0],
                    precise_source_offsets=precise_source_offsets,
                    depth=depth + 1,
                    stack=(*stack, name),
                    budget=budget,
                )
            )

        body = _strip_xspace(definition.body)
        sentinel = ""
        if definition.arity == 1:
            nonce = hashlib.sha256(
                (
                    definition.body_sha256
                    + str(budget.expansions)
                    + str(depth)
                ).encode("utf-8")
            ).hexdigest()[:24]
            sentinel = f"SFVTWOMACROARGUMENT{nonce}END"
            if sentinel in body or body.count("#1") != 1:
                raise MacroExpansionError(f"argument_sentinel_collision:{name}")
            body = body.replace("#1", sentinel)
        body_text, body_events = _expand_text(
            body,
            definitions=definitions,
            rejected_names=rejected_names,
            allowed_commands=allowed_commands,
            source_file=definition.source_file,
            source_base_offset=definition.body_span[0],
            precise_source_offsets=False,
            depth=depth + 1,
            stack=(*stack, name),
            budget=budget,
        )
        nested_events = body_events
        if definition.arity == 1:
            sentinel_start = body_text.find(sentinel)
            if sentinel_start < 0 or body_text.find(sentinel, sentinel_start + 1) >= 0:
                raise MacroExpansionError(f"argument_flow_is_not_unique:{name}")
            sentinel_end = sentinel_start + len(sentinel)
            argument_text, argument_events = expanded_arguments[0]
            nested_events = _replace_sentinel_in_provenance(
                body_events,
                (sentinel_start, sentinel_end),
                len(argument_text),
            )
            nested_events = (
                *nested_events,
                *(
                    _shift_provenance(event, sentinel_start)
                    for event in argument_events
                ),
            )
            body_text = (
                body_text[:sentinel_start]
                + argument_text
                + body_text[sentinel_end:]
            )
        if definition.uses_xspace:
            body_text += _xspace_suffix(
                source,
                invocation_end,
                consumed_control_space=consumed_control_space,
            )

        event_output_start = output_length
        for event in nested_events:
            provenance.append(_shift_provenance(event, event_output_start))
        append(body_text)
        invocation_span = (
            (
                source_base_offset + cursor,
                source_base_offset + invocation_end,
            )
            if precise_source_offsets
            else None
        )
        argument_spans = (
            tuple(
                (
                    source_base_offset + start,
                    source_base_offset + end,
                )
                for start, end in argument_spans_local
            )
            if precise_source_offsets
            else ()
        )
        provenance.append(
            ExpansionProvenance(
                macro_name=name,
                invocation_source_file=source_file if precise_source_offsets else None,
                invocation_span=invocation_span,
                argument_spans=argument_spans,
                definition_source_file=definition.source_file,
                definition_span=definition.declaration_span,
                definition_body_span=definition.body_span,
                definition_body_sha256=definition.body_sha256,
                expansion_stack=(*stack, name),
                depth=depth,
                output_span=(event_output_start, event_output_start + len(body_text)),
                argument_sha256=tuple(
                    hashlib.sha256(value.encode("utf-8")).hexdigest()
                    for value in argument_values
                ),
            )
        )
        cursor = invocation_end
    return "".join(output), tuple(provenance)


def expand_safe_macros(
    source: str,
    registry: SafeMacroRegistry,
    *,
    source_file: str | Path | None = None,
    source_base_offset: int = 0,
    additional_passthrough_commands: Mapping[str, int] | None = None,
    max_depth: int = DEFAULT_MAX_EXPANSION_DEPTH,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    max_output_characters: int = DEFAULT_MAX_OUTPUT_CHARACTERS,
) -> ExpansionResult:
    """Expand every admitted project macro in ``source`` or fail closed.

    Unknown commands are rejected unless they belong to the registry's fixed
    safe command set or are explicitly vouched for by
    ``additional_passthrough_commands``.  Forbidden command classes can never
    be overridden.  The returned string is still LaTeX source; this function
    only substitutes proven-safe project aliases/wrappers.
    """

    if source_base_offset < 0:
        raise ValueError("source_base_offset must be non-negative")
    if max_depth < 1 or max_expansions < 1 or max_output_characters < 1:
        raise ValueError("macro expansion limits must be positive")
    definitions = registry.by_name
    allowed_commands = dict(registry.allowed_commands)
    for name, arity in (additional_passthrough_commands or {}).items():
        name = str(name)
        forbidden = _forbidden_reason(name)
        if forbidden is not None:
            raise ValueError(f"cannot pass through forbidden command {name}: {forbidden}")
        if not re.fullmatch(r"[A-Za-z@]+", name) or int(arity) < 0 or int(arity) > 4:
            raise ValueError(f"invalid passthrough command: {name}={arity}")
        allowed_commands[name] = int(arity)
    resolved_source_file = (
        Path(source_file).expanduser().resolve() if source_file is not None else None
    )
    budget = _Budget(
        max_depth=max_depth,
        max_expansions=max_expansions,
        max_output_characters=max_output_characters,
    )
    text, provenance = _expand_text(
        source,
        definitions=definitions,
        rejected_names=frozenset(registry.rejected_names),
        allowed_commands=allowed_commands,
        source_file=resolved_source_file,
        source_base_offset=source_base_offset,
        precise_source_offsets=True,
        depth=0,
        stack=(),
        budget=budget,
    )
    ordered = tuple(
        sorted(
            provenance,
            key=lambda item: (
                item.output_span[0],
                -item.output_span[1],
                item.depth,
                item.macro_name,
            ),
        )
    )
    return ExpansionResult(
        text=text,
        provenance=ordered,
        macros_used=tuple(sorted({item.macro_name for item in ordered})),
        expansion_count=budget.expansions,
        maximum_depth=budget.maximum_depth,
    )


__all__ = [
    "DEFAULT_MAX_BODY_CHARACTERS",
    "DEFAULT_MAX_DEFINITIONS",
    "DEFAULT_MAX_DEPENDENCY_DEPTH",
    "DEFAULT_MAX_EXPANSION_DEPTH",
    "DEFAULT_MAX_EXPANSIONS",
    "DEFAULT_MAX_OUTPUT_CHARACTERS",
    "ExpansionProvenance",
    "ExpansionResult",
    "MacroExpansionError",
    "MacroRejection",
    "SafeMacroDefinition",
    "SafeMacroError",
    "SafeMacroRegistry",
    "collect_safe_macros",
    "expand_safe_macros",
]
