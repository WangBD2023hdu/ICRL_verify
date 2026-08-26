"""Strict source-first IR for theorem headings and equation numbers.

This module is intentionally a small lexical parser, not a TeX interpreter.
It admits only structure that can be proved from literal project source and a
caller-supplied AUX ``label -> number`` mapping.  PDF text is never an input.

The public entry points return explicit rejection records instead of guessing:

* :func:`collect_theorem_definitions_from_sources` admits static
  ``\\newtheorem``/LLNCS theorem declarations and poisons genuinely
  conflicting names;
* :func:`build_theorem_ir_from_sources` finds balanced theorem-like blocks and
  emits a finite set of source-derived heading candidates; and
* :func:`resolve_display_equation_tail` emits finite equation-number tails
  while preserving the caller's formula Markdown byte-for-byte.

All spans are half-open character offsets in the original, unmodified source.
Comment masking preserves source length, so provenance remains exact even when
commands span lines or comments occur between their arguments.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


STRUCTURAL_IR_SCHEMA_VERSION = 2
STRUCTURAL_IR_CONTRACT = "arxiv_source_first_v2_structural_ir"

SUPPORTED_THEOREM_ENVIRONMENTS = frozenset(
    {
        "case",
        "theorem",
        "lemma",
        "proposition",
        "definition",
        "corollary",
        "conjecture",
        "assumption",
        "remark",
        "example",
        "exercise",
        "claim",
        "note",
        "problem",
        "property",
        "question",
        "solution",
    }
)
SUPPORTED_THEOREM_DISPLAY_NAMES = frozenset(
    name.casefold() for name in SUPPORTED_THEOREM_ENVIRONMENTS
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9@:_-]*$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/\-]*$")
_AUX_NUMBER_RE = re.compile(r"^[A-Za-z0-9]+(?:[.\-:][A-Za-z0-9]+)*$")
_DECLARATION_COMMAND_RE = re.compile(
    r"\\(?P<command>newtheorem|renewtheorem|spnewtheorem|spn@wtheorem)"
    r"(?![A-Za-z@])"
)
_ENVIRONMENT_COMMAND_RE = re.compile(r"\\(?P<kind>begin|end)(?![A-Za-z@])")
_LABEL_COMMAND_RE = re.compile(r"\\label(?![A-Za-z@])")
_TAG_COMMAND_RE = re.compile(r"\\tag(?P<star>\*)?(?![A-Za-z@])")


class StructuralIRError(ValueError):
    """Base class for invalid API input."""


class StructuralIRSafetyError(StructuralIRError):
    """Raised when a caller explicitly requests exception-style rejection."""

    def __init__(self, rejections: Sequence["StructuralRejection"]):
        self.rejections = tuple(rejections)
        detail = "; ".join(
            f"{item.code} at {item.source_file}:{item.source_lines[0]}"
            for item in self.rejections
        )
        super().__init__(detail or "structural IR failed closed")


@dataclasses.dataclass(frozen=True, slots=True)
class SourceSpan:
    """Exact location in one original source string."""

    source_file: Path
    char_span: tuple[int, int]
    source_lines: tuple[int, int]
    sha256: str

    def as_json(self) -> dict[str, Any]:
        return {
            "source_file": str(self.source_file),
            "char_span": list(self.char_span),
            "source_lines": list(self.source_lines),
            "sha256": self.sha256,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class StructuralRejection:
    """A source-addressed construct which was deliberately not admitted."""

    stage: str
    code: str
    message: str
    source_file: Path
    char_span: tuple[int, int]
    source_lines: tuple[int, int]
    environment: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_IR_SCHEMA_VERSION,
            "contract": STRUCTURAL_IR_CONTRACT,
            "status": "rejected",
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "environment": self.environment,
            "generation_source": "latex_source_and_compiler_aux",
            "pdf_text_used": False,
            "source_file": str(self.source_file),
            "char_span": list(self.char_span),
            "source_lines": list(self.source_lines),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class StaticVisibleText:
    """A title serialized without executing unknown TeX."""

    source: str
    markdown: str
    plain_text: str


@dataclasses.dataclass(frozen=True, slots=True)
class TheoremDefinition:
    environment: str
    visible_name_source: str
    visible_name_markdown: str
    visible_name_plain: str
    shared_counter: str | None
    within_counter: str | None
    source_file: Path
    declaration_span: tuple[int, int]
    environment_span: tuple[int, int]
    visible_name_span: tuple[int, int]
    source_lines: tuple[int, int]
    declaration_sha256: str
    declaration_command: str = "newtheorem"
    numbered: bool = True
    counter_semantics: str = "literal_source"
    equivalent_declaration_sites: tuple[SourceSpan, ...] = ()

    @property
    def numbering_policy(self) -> str:
        if not self.numbered:
            return "unnumbered"
        if self.counter_semantics == "compiler_aux_only":
            return "compiler_aux_only"
        if self.shared_counter is not None:
            return "shared_counter"
        if self.within_counter is not None:
            return "within_counter"
        return "own_counter"

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_IR_SCHEMA_VERSION,
            "contract": STRUCTURAL_IR_CONTRACT,
            "kind": "theorem_definition",
            "generation_source": "latex_source",
            "pdf_text_used": False,
            "environment": self.environment,
            "visible_name_source": self.visible_name_source,
            "visible_name_markdown": self.visible_name_markdown,
            "visible_name_plain": self.visible_name_plain,
            "shared_counter": self.shared_counter,
            "within_counter": self.within_counter,
            "numbering_policy": self.numbering_policy,
            "declaration_command": self.declaration_command,
            "numbered": self.numbered,
            "counter_semantics": self.counter_semantics,
            "equivalent_declaration_sites": [
                item.as_json() for item in self.equivalent_declaration_sites
            ],
            "source_file": str(self.source_file),
            "declaration_span": list(self.declaration_span),
            "environment_span": list(self.environment_span),
            "visible_name_span": list(self.visible_name_span),
            "source_lines": list(self.source_lines),
            "declaration_sha256": self.declaration_sha256,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class TheoremDefinitionRegistry:
    definitions: tuple[TheoremDefinition, ...]
    rejections: tuple[StructuralRejection, ...]
    source_files: tuple[Path, ...]

    @property
    def by_environment(self) -> dict[str, TheoremDefinition]:
        return {item.environment: item for item in self.definitions}

    def as_report(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_IR_SCHEMA_VERSION,
            "contract": STRUCTURAL_IR_CONTRACT,
            "policy": "static_newtheorem_llncs_source_registry_v2",
            "generation_source": "latex_source",
            "pdf_text_used": False,
            "source_files": [str(path) for path in self.source_files],
            "definitions_accepted": len(self.definitions),
            "definitions_rejected": len(self.rejections),
            "definitions": [item.as_json() for item in self.definitions],
            "rejections": [item.as_json() for item in self.rejections],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class TheoremHeadingCandidate:
    candidate_id: str
    policy: str
    visible_text: str
    markdown: str
    source_file: Path
    block_span: tuple[int, int]
    definition_span: tuple[int, int]
    source_lines: tuple[int, int]
    environment: str
    label: str
    aux_number: str
    optional_title_source: str | None

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_IR_SCHEMA_VERSION,
            "contract": STRUCTURAL_IR_CONTRACT,
            "kind": "theorem_heading_candidate",
            "candidate_id": self.candidate_id,
            "policy": self.policy,
            "visible_text": self.visible_text,
            "markdown": self.markdown,
            "generation_sources": ["latex_source", "compiler_aux"],
            "pdf_text_used": False,
            "source_file": str(self.source_file),
            "block_span": list(self.block_span),
            "definition_span": list(self.definition_span),
            "source_lines": list(self.source_lines),
            "environment": self.environment,
            "label": self.label,
            "aux_number": self.aux_number,
            "optional_title_source": self.optional_title_source,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class TheoremBlock:
    block_id: str
    environment: str
    source_file: Path
    block_span: tuple[int, int]
    begin_span: tuple[int, int]
    body_span: tuple[int, int]
    end_span: tuple[int, int]
    source_lines: tuple[int, int]
    raw_latex: str
    optional_title_source: str | None
    optional_title_markdown: str | None
    optional_title_span: tuple[int, int] | None
    label: str
    label_span: tuple[int, int]
    aux_number: str
    definition: TheoremDefinition
    heading_candidates: tuple[TheoremHeadingCandidate, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_IR_SCHEMA_VERSION,
            "contract": STRUCTURAL_IR_CONTRACT,
            "kind": "theorem_block",
            "block_id": self.block_id,
            "generation_sources": ["latex_source", "compiler_aux"],
            "pdf_text_used": False,
            "environment": self.environment,
            "source_file": str(self.source_file),
            "block_span": list(self.block_span),
            "begin_span": list(self.begin_span),
            "body_span": list(self.body_span),
            "end_span": list(self.end_span),
            "source_lines": list(self.source_lines),
            "raw_latex_sha256": _sha256(self.raw_latex),
            "optional_title_source": self.optional_title_source,
            "optional_title_markdown": self.optional_title_markdown,
            "optional_title_span": (
                list(self.optional_title_span)
                if self.optional_title_span is not None
                else None
            ),
            "label": self.label,
            "label_span": list(self.label_span),
            "aux_number": self.aux_number,
            "definition": self.definition.as_json(),
            "heading_candidates": [item.as_json() for item in self.heading_candidates],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class TheoremStructuralIR:
    registry: TheoremDefinitionRegistry
    blocks: tuple[TheoremBlock, ...]
    rejections: tuple[StructuralRejection, ...]

    def as_report(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_IR_SCHEMA_VERSION,
            "contract": STRUCTURAL_IR_CONTRACT,
            "policy": "balanced_theorem_aux_heading_candidates_v1",
            "generation_sources": ["latex_source", "compiler_aux"],
            "pdf_text_used": False,
            "blocks_accepted": len(self.blocks),
            "blocks_rejected": len(self.rejections),
            "registry": self.registry.as_report(),
            "blocks": [item.as_json() for item in self.blocks],
            "rejections": [item.as_json() for item in self.rejections],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class EquationTailCandidate:
    candidate_id: str
    policy: str
    number_source: str
    tail_text: str
    formula_markdown: str
    markdown: str
    source_file: Path
    block_span: tuple[int, int]
    number_source_span: tuple[int, int]
    source_lines: tuple[int, int]
    label: str | None
    number: str

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_IR_SCHEMA_VERSION,
            "contract": STRUCTURAL_IR_CONTRACT,
            "kind": "equation_tail_candidate",
            "candidate_id": self.candidate_id,
            "policy": self.policy,
            "number_source": self.number_source,
            "tail_text": self.tail_text,
            "formula_markdown": self.formula_markdown,
            "markdown": self.markdown,
            "generation_sources": (
                ["latex_source"]
                if self.number_source == "explicit_tag"
                else ["latex_source", "compiler_aux"]
            ),
            "pdf_text_used": False,
            "source_file": str(self.source_file),
            "block_span": list(self.block_span),
            "number_source_span": list(self.number_source_span),
            "source_lines": list(self.source_lines),
            "label": self.label,
            "number": self.number,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class EquationTailResolution:
    status: str
    formula_markdown: str
    source_file: Path
    block_span: tuple[int, int]
    number_source_span: tuple[int, int] | None
    source_lines: tuple[int, int]
    label: str | None
    number: str | None
    number_source: str | None
    candidates: tuple[EquationTailCandidate, ...]
    rejections: tuple[StructuralRejection, ...]

    def as_report(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_IR_SCHEMA_VERSION,
            "contract": STRUCTURAL_IR_CONTRACT,
            "kind": "equation_tail_resolution",
            "status": self.status,
            "generation_source": "latex_source_and_compiler_aux",
            "pdf_text_used": False,
            "formula_markdown": self.formula_markdown,
            "source_file": str(self.source_file),
            "block_span": list(self.block_span),
            "number_source_span": (
                list(self.number_source_span)
                if self.number_source_span is not None
                else None
            ),
            "source_lines": list(self.source_lines),
            "label": self.label,
            "number": self.number,
            "number_source": self.number_source,
            "candidates": [item.as_json() for item in self.candidates],
            "rejections": [item.as_json() for item in self.rejections],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class _ParsedDeclaration:
    definition: TheoremDefinition | None
    rejection: StructuralRejection | None
    environment: str | None
    end: int


@dataclasses.dataclass(frozen=True, slots=True)
class _EnvironmentToken:
    kind: str
    environment: str
    command_span: tuple[int, int]


@dataclasses.dataclass(slots=True)
class _OpenEnvironment:
    token: _EnvironmentToken
    invalid: bool = False


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_escaped(source: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and source[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def mask_tex_comments(source: str) -> str:
    """Replace TeX comments with spaces while preserving every offset."""

    characters = list(source)
    index = 0
    while index < len(source):
        if source[index] == "%" and not _is_escaped(source, index):
            while index < len(source) and source[index] not in "\r\n":
                characters[index] = " "
                index += 1
            continue
        index += 1
    return "".join(characters)


def _skip_space(source: str, index: int) -> int:
    while index < len(source) and source[index].isspace():
        index += 1
    return index


def _balanced_end(source: str, opening: int, left: str, right: str) -> int | None:
    if opening >= len(source) or source[opening] != left:
        return None
    depth = 0
    index = opening
    while index < len(source):
        character = source[index]
        if character == "\\":
            index += 2
            continue
        if character == left:
            depth += 1
        elif character == right:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _line_range(source: str, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    start_line = source.count("\n", 0, max(0, start)) + 1
    inclusive_end = max(start, end - 1)
    end_line = source.count("\n", 0, inclusive_end) + 1
    return start_line, end_line


def _span_rejection(
    *,
    stage: str,
    code: str,
    message: str,
    source_file: Path,
    source: str,
    span: tuple[int, int],
    environment: str | None = None,
) -> StructuralRejection:
    bounded = (max(0, span[0]), min(len(source), max(span[0], span[1])))
    return StructuralRejection(
        stage=stage,
        code=code,
        message=message,
        source_file=source_file,
        char_span=bounded,
        source_lines=_line_range(source, bounded),
        environment=environment,
    )


_VISIBLE_CONTROL_SYMBOLS = {
    "%": ("%", "%"),
    "$": ("\\$", "$"),
    "#": ("\\#", "#"),
    "&": ("&", "&"),
    "_": ("\\_", "_"),
    "{": ("{", "{"),
    "}": ("}", "}"),
}
_WRAPPER_COMMANDS = {
    "textbf": "bold",
    "textit": "italic",
    "emph": "italic",
    "texttt": "code",
    "textrm": "plain",
    "textsf": "plain",
    "textsc": "plain",
    "mbox": "plain",
}
_ZERO_ARGUMENT_VISIBLE_COMMANDS = {"LaTeX": "LaTeX", "TeX": "TeX"}
_SAFE_MATH_COMMANDS = frozenset(
    {
        "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon",
        "theta", "lambda", "mu", "nu", "pi", "rho", "sigma", "tau",
        "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta",
        "Theta", "Lambda", "Pi", "Sigma", "Phi", "Psi", "Omega",
        "mathbb", "mathbf", "mathrm", "mathcal", "mathsf", "mathtt",
        "operatorname", "text", "frac", "sqrt", "sum", "prod", "int",
        "lim", "log", "exp", "min", "max", "argmin", "argmax", "in",
        "notin", "subset", "subseteq", "supset", "supseteq", "le", "leq",
        "ge", "geq", "neq", "approx", "sim", "equiv", "times", "cdot",
        "pm", "mp", "to", "rightarrow", "leftarrow", "leftrightarrow",
        "infty", "partial", "nabla", "ell", "star", "ast", "prime",
        "left", "right", "bigl", "bigr", "Bigl", "Bigr", "colon",
    }
)


class _StaticVisibleParser:
    def __init__(self, source: str):
        self.source = source
        self.index = 0

    def parse(self, stop: str | None = None) -> tuple[str, str]:
        markdown: list[str] = []
        plain: list[str] = []
        while self.index < len(self.source):
            character = self.source[self.index]
            if stop is not None and character == stop:
                self.index += 1
                return "".join(markdown), "".join(plain)
            if character == "}":
                raise StructuralIRError("unbalanced_group_in_visible_text")
            if character == "{":
                self.index += 1
                child_markdown, child_plain = self.parse("}")
                markdown.append(child_markdown)
                plain.append(child_plain)
                continue
            if character == "\\":
                self._parse_command(markdown, plain)
                continue
            if character == "$":
                end = self._math_end(self.index)
                content = self.source[self.index + 1 : end - 1]
                self._validate_math(content)
                rendered = "$" + content.strip() + "$"
                markdown.append(rendered)
                plain.append(rendered)
                self.index = end
                continue
            if character == "~":
                markdown.append(" ")
                plain.append(" ")
            elif character in "*_[]#`":
                markdown.append("\\" + character)
                plain.append(character)
            else:
                markdown.append(character)
                plain.append(character)
            self.index += 1
        if stop is not None:
            raise StructuralIRError("unbalanced_group_in_visible_text")
        return "".join(markdown), "".join(plain)

    def _parse_command(self, markdown: list[str], plain: list[str]) -> None:
        start = self.index
        self.index += 1
        if self.index >= len(self.source):
            raise StructuralIRError("trailing_backslash_in_visible_text")
        if self.source[self.index] == "\\":
            self.index += 1
            markdown.append(" ")
            plain.append(" ")
            return
        if not self.source[self.index].isalpha() and self.source[self.index] != "@":
            symbol = self.source[self.index]
            self.index += 1
            rendered = _VISIBLE_CONTROL_SYMBOLS.get(symbol)
            if rendered is None:
                raise StructuralIRError(f"unknown_control_symbol:{symbol!r}")
            markdown.append(rendered[0])
            plain.append(rendered[1])
            return
        command_start = self.index
        while self.index < len(self.source) and (
            self.source[self.index].isalpha() or self.source[self.index] == "@"
        ):
            self.index += 1
        command = self.source[command_start : self.index]
        if command == "protect":
            return
        if command in _ZERO_ARGUMENT_VISIBLE_COMMANDS:
            value = _ZERO_ARGUMENT_VISIBLE_COMMANDS[command]
            markdown.append(value)
            plain.append(value)
            return
        wrapper = _WRAPPER_COMMANDS.get(command)
        if wrapper is None:
            raise StructuralIRError(f"unknown_macro_in_visible_text:{command}")
        self.index = _skip_space(self.source, self.index)
        if self.index >= len(self.source) or self.source[self.index] != "{":
            raise StructuralIRError(f"missing_argument_for_visible_macro:{command}")
        self.index += 1
        child_markdown, child_plain = self.parse("}")
        if not child_plain.strip():
            raise StructuralIRError(f"empty_argument_for_visible_macro:{command}")
        if wrapper == "bold":
            markdown.append(f"**{child_markdown}**")
        elif wrapper == "italic":
            markdown.append(f"*{child_markdown}*")
        elif wrapper == "code":
            if "`" in child_markdown:
                raise StructuralIRError("backtick_in_texttt_visible_text")
            markdown.append(f"`{child_plain}`")
        else:
            markdown.append(child_markdown)
        plain.append(child_plain)
        if self.index == start:
            raise AssertionError("visible parser did not advance")

    def _math_end(self, opening: int) -> int:
        index = opening + 1
        while index < len(self.source):
            if self.source[index] == "$" and not _is_escaped(self.source, index):
                return index + 1
            index += 1
        raise StructuralIRError("unbalanced_inline_math_in_visible_text")

    @staticmethod
    def _validate_math(content: str) -> None:
        for match in re.finditer(r"\\([A-Za-z@]+)", content):
            if match.group(1) not in _SAFE_MATH_COMMANDS:
                raise StructuralIRError(
                    f"unknown_macro_in_visible_text:{match.group(1)}"
                )
        if "\\csname" in content or "\\input" in content:
            raise StructuralIRError("dynamic_math_in_visible_text")


def render_static_visible_text(source: str) -> StaticVisibleText:
    """Serialize a narrowly supported literal title, rejecting unknown macros."""

    if not isinstance(source, str):
        raise StructuralIRError("visible text must be a string")
    masked = mask_tex_comments(source)
    parser = _StaticVisibleParser(masked)
    markdown, plain = parser.parse()
    markdown = re.sub(r"\s+", " ", markdown).strip()
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain:
        raise StructuralIRError("visible text must not be empty")
    return StaticVisibleText(source=source, markdown=markdown, plain_text=plain)


def _parse_group(
    masked: str, index: int, left: str, right: str
) -> tuple[str, tuple[int, int], int] | None:
    index = _skip_space(masked, index)
    end = _balanced_end(masked, index, left, right)
    if end is None:
        return None
    return masked[index + 1 : end - 1], (index + 1, end - 1), end


_DYNAMIC_STYLE_COMMAND_RE = re.compile(
    r"\\(?:csname|endcsname|input|include|usepackage|def|gdef|edef|xdef|let|"
    r"futurelet|expandafter|noexpand|if[A-Za-z@]*|else|fi)(?![A-Za-z@])"
)
_MACRO_DEFINITION_TARGET_RE = re.compile(
    r"\\(?:def|gdef|edef|xdef|let)\s*$"
)
_MACRO_DEFINITION_COMMAND_RE = re.compile(
    r"\\(?:def|gdef|edef|xdef)(?![A-Za-z@])"
)


def _style_argument_is_static(value: str) -> bool:
    """Admit a balanced literal font declaration without interpreting it.

    LLNCS normally supplies values such as ``\\bfseries`` and ``\\itshape``.
    These arguments do not contribute text, but requiring a literal group and
    rejecting parameter tokens/dynamic expansion prevents a malformed macro
    definition from masquerading as a direct theorem declaration.
    """

    if not value.strip() or _DYNAMIC_STYLE_COMMAND_RE.search(value):
        return False
    return not any(
        character == "#" and not _is_escaped(value, index)
        for index, character in enumerate(value)
    )


def _parse_static_style_groups(
    *,
    source: str,
    masked: str,
    source_file: Path,
    start: int,
    cursor: int,
    environment: str,
    count: int = 2,
) -> tuple[int | None, StructuralRejection | None]:
    for ordinal in range(1, count + 1):
        group = _parse_group(masked, cursor, "{", "}")
        if group is None:
            return None, _span_rejection(
                stage="theorem_definition",
                code="malformed_style_argument",
                message=(
                    f"LLNCS theorem declaration requires balanced literal "
                    f"style argument {ordinal}/{count}"
                ),
                source_file=source_file,
                source=source,
                span=(start, max(cursor, start + 1)),
                environment=environment,
            )
        style_masked, _, cursor = group
        if not _style_argument_is_static(style_masked):
            return None, _span_rejection(
                stage="theorem_definition",
                code="dynamic_style_argument",
                message=(
                    f"LLNCS theorem style argument {ordinal}/{count} is not "
                    "a static literal group"
                ),
                source_file=source_file,
                source=source,
                span=(start, cursor),
                environment=environment,
            )
    return cursor, None


def _definition_site(
    source_file: Path,
    source: str,
    declaration_span: tuple[int, int],
) -> SourceSpan:
    fragment = source[slice(*declaration_span)]
    return SourceSpan(
        source_file=source_file,
        char_span=declaration_span,
        source_lines=_line_range(source, declaration_span),
        sha256=_sha256(fragment),
    )


def _macro_definition_body_spans(masked: str) -> tuple[tuple[int, int], ...]:
    """Find ordinary TeX ``\\def`` replacement bodies, preserving offsets.

    This deliberately recognizes only balanced literal replacement groups.
    A theorem declaration token inside such a body is dynamic source, not a
    declaration proved to execute, and is rejected by the registry collector.
    """

    spans: list[tuple[int, int]] = []
    for match in _MACRO_DEFINITION_COMMAND_RE.finditer(masked):
        cursor = _skip_space(masked, match.end())
        if cursor >= len(masked) or masked[cursor] != "\\":
            continue
        cursor += 1
        if cursor >= len(masked):
            continue
        if masked[cursor].isalpha() or masked[cursor] == "@":
            while cursor < len(masked) and (
                masked[cursor].isalpha() or masked[cursor] == "@"
            ):
                cursor += 1
        else:
            cursor += 1
        while cursor < len(masked):
            if masked[cursor] == "{" and not _is_escaped(masked, cursor):
                end = _balanced_end(masked, cursor, "{", "}")
                if end is not None:
                    spans.append((cursor, end))
                break
            if masked[cursor] in "\r\n" and "#" not in masked[match.end() : cursor]:
                # A malformed zero-argument definition should not consume the
                # next unrelated source line looking for an opening group.
                break
            cursor += 1
    return tuple(spans)


def _containing_span(
    index: int, spans: Sequence[tuple[int, int]]
) -> tuple[int, int] | None:
    for start, end in spans:
        if start < index < end:
            return start, end
    return None


def _parse_declaration(
    source: str,
    masked: str,
    source_file: Path,
    match: re.Match[str],
) -> _ParsedDeclaration:
    start = match.start()
    command = match.group("command")
    cursor = match.end()
    starred = cursor < len(masked) and masked[cursor] == "*"
    if starred:
        cursor += 1
    env_group = _parse_group(masked, cursor, "{", "}")
    if env_group is None:
        span = (start, min(len(source), source.find("\n", start) if "\n" in source[start:] else len(source)))
        rejection = _span_rejection(
            stage="theorem_definition",
            code="malformed_environment_argument",
            message="newtheorem environment must be one balanced literal group",
            source_file=source_file,
            source=source,
            span=span,
        )
        return _ParsedDeclaration(None, rejection, None, max(match.end(), span[1]))
    env_raw, env_span, cursor = env_group
    environment = env_raw.strip()
    if not _IDENTIFIER_RE.fullmatch(environment):
        rejection = _span_rejection(
            stage="theorem_definition",
            code="dynamic_environment_name",
            message="theorem environment name is not a static identifier",
            source_file=source_file,
            source=source,
            span=(start, cursor),
        )
        return _ParsedDeclaration(None, rejection, None, cursor)

    if command == "spn@wtheorem" and starred:
        rejection = _span_rejection(
            stage="theorem_definition",
            code="unsupported_starred_declaration",
            message="spn@wtheorem does not have a static starred form",
            source_file=source_file,
            source=source,
            span=(start, cursor),
            environment=environment,
        )
        return _ParsedDeclaration(None, rejection, environment, cursor)

    shared_counter: str | None = None
    if command != "spn@wtheorem":
        cursor = _skip_space(masked, cursor)
        if cursor < len(masked) and masked[cursor] == "[":
            shared_group = _parse_group(masked, cursor, "[", "]")
            if shared_group is None:
                rejection = _span_rejection(
                    stage="theorem_definition",
                    code="unbalanced_shared_counter",
                    message="shared theorem counter is unbalanced",
                    source_file=source_file,
                    source=source,
                    span=(start, cursor + 1),
                    environment=environment,
                )
                return _ParsedDeclaration(None, rejection, environment, cursor + 1)
            shared_raw, _, cursor = shared_group
            shared_counter = shared_raw.strip()

    visible_group = _parse_group(masked, cursor, "{", "}")
    if visible_group is None:
        rejection = _span_rejection(
            stage="theorem_definition",
            code="malformed_visible_name",
            message="newtheorem visible name must be one balanced literal group",
            source_file=source_file,
            source=source,
            span=(start, cursor),
            environment=environment,
        )
        return _ParsedDeclaration(None, rejection, environment, cursor)
    visible_masked, visible_span, cursor = visible_group
    within_counter: str | None = None
    if command != "spn@wtheorem":
        cursor = _skip_space(masked, cursor)
        if cursor < len(masked) and masked[cursor] == "[":
            within_group = _parse_group(masked, cursor, "[", "]")
            if within_group is None:
                rejection = _span_rejection(
                    stage="theorem_definition",
                    code="unbalanced_within_counter",
                    message="within counter is unbalanced",
                    source_file=source_file,
                    source=source,
                    span=(start, cursor + 1),
                    environment=environment,
                )
                return _ParsedDeclaration(None, rejection, environment, cursor + 1)
            within_raw, _, cursor = within_group
            within_counter = within_raw.strip()

    if command in {"spnewtheorem", "spn@wtheorem"}:
        style_end, style_rejection = _parse_static_style_groups(
            source=source,
            masked=masked,
            source_file=source_file,
            start=start,
            cursor=cursor,
            environment=environment,
        )
        if style_rejection is not None:
            return _ParsedDeclaration(
                None,
                style_rejection,
                environment,
                max(cursor, style_rejection.char_span[1]),
            )
        assert style_end is not None
        cursor = style_end

    declaration_span = (start, cursor)
    if command == "renewtheorem":
        rejection = _span_rejection(
            stage="theorem_definition",
            code="unsupported_redefinition",
            message=f"{command} is not an admissible static declaration",
            source_file=source_file,
            source=source,
            span=declaration_span,
            environment=environment,
        )
        return _ParsedDeclaration(None, rejection, environment, cursor)
    if starred and command == "newtheorem":
        rejection = _span_rejection(
            stage="theorem_definition",
            code="unsupported_unnumbered_declaration",
            message="starred newtheorem has no compiler number",
            source_file=source_file,
            source=source,
            span=declaration_span,
            environment=environment,
        )
        return _ParsedDeclaration(None, rejection, environment, cursor)
    if starred and (shared_counter is not None or within_counter is not None):
        rejection = _span_rejection(
            stage="theorem_definition",
            code="counter_on_unnumbered_declaration",
            message="starred spnewtheorem cannot carry a counter option",
            source_file=source_file,
            source=source,
            span=declaration_span,
            environment=environment,
        )
        return _ParsedDeclaration(None, rejection, environment, cursor)
    if shared_counter is not None and within_counter is not None:
        rejection = _span_rejection(
            stage="theorem_definition",
            code="ambiguous_counter_declaration",
            message="shared and within counters cannot both be present",
            source_file=source_file,
            source=source,
            span=declaration_span,
            environment=environment,
        )
        return _ParsedDeclaration(None, rejection, environment, cursor)
    for counter_kind, counter in (
        ("shared", shared_counter),
        ("within", within_counter),
    ):
        if counter is not None and not _IDENTIFIER_RE.fullmatch(counter):
            rejection = _span_rejection(
                stage="theorem_definition",
                code=f"dynamic_{counter_kind}_counter",
                message=f"{counter_kind} counter must be a static identifier",
                source_file=source_file,
                source=source,
                span=declaration_span,
                environment=environment,
            )
            return _ParsedDeclaration(None, rejection, environment, cursor)
    visible_source = source[slice(*visible_span)]
    try:
        visible = render_static_visible_text(visible_masked)
    except StructuralIRError as exc:
        rejection = _span_rejection(
            stage="theorem_definition",
            code="unsafe_visible_name",
            message=str(exc),
            source_file=source_file,
            source=source,
            span=declaration_span,
            environment=environment,
        )
        return _ParsedDeclaration(None, rejection, environment, cursor)

    supported = (
        environment in SUPPORTED_THEOREM_ENVIRONMENTS
        or visible.plain_text.casefold() in SUPPORTED_THEOREM_DISPLAY_NAMES
    )
    if not supported:
        rejection = _span_rejection(
            stage="theorem_definition",
            code="unsupported_theorem_kind",
            message="environment and visible name are outside the theorem allow-list",
            source_file=source_file,
            source=source,
            span=declaration_span,
            environment=environment,
        )
        return _ParsedDeclaration(None, rejection, environment, cursor)

    definition = TheoremDefinition(
        environment=environment,
        visible_name_source=visible_source,
        visible_name_markdown=visible.markdown,
        visible_name_plain=visible.plain_text,
        shared_counter=shared_counter,
        within_counter=within_counter,
        source_file=source_file,
        declaration_span=declaration_span,
        environment_span=env_span,
        visible_name_span=visible_span,
        source_lines=_line_range(source, declaration_span),
        declaration_sha256=_sha256(source[slice(*declaration_span)]),
        declaration_command=command + ("*" if starred else ""),
        numbered=not starred,
        counter_semantics=(
            "compiler_aux_only"
            if command == "spn@wtheorem"
            else ("unnumbered" if starred else "literal_source")
        ),
        equivalent_declaration_sites=(
            _definition_site(source_file, source, declaration_span),
        ),
    )
    return _ParsedDeclaration(definition, None, environment, cursor)


def _is_macro_definition_target(masked: str, command_start: int) -> bool:
    """Return true when the matched command is the target of ``\\def``/``\\let``.

    The LLNCS class necessarily defines ``\\spnewtheorem`` and
    ``\\spn@wtheorem`` before invoking them.  Those target tokens are not
    theorem declarations and must not generate a misleading parser rejection.
    """

    prefix = masked[max(0, command_start - 80) : command_start]
    return _MACRO_DEFINITION_TARGET_RE.search(prefix) is not None


def _compatible_static_declarations(
    definitions: Sequence[TheoremDefinition],
) -> bool:
    """Whether duplicate declarations prove exactly the same visible heading.

    Counter layouts may differ across mutually exclusive LLNCS class branches.
    This is safe only because block admission later requires one literal label
    with one compiler AUX number; the counter choice is never inferred here.
    """

    if not definitions:
        return False
    first = definitions[0]
    return all(
        item.environment == first.environment
        and item.visible_name_markdown == first.visible_name_markdown
        and item.visible_name_plain == first.visible_name_plain
        and item.numbered == first.numbered
        for item in definitions[1:]
    )


def _merge_compatible_declarations(
    definitions: Sequence[TheoremDefinition],
) -> TheoremDefinition:
    first = definitions[0]
    counter_signatures = {
        (
            item.shared_counter,
            item.within_counter,
            item.counter_semantics,
        )
        for item in definitions
    }
    command_names = {item.declaration_command for item in definitions}
    if len(counter_signatures) == 1:
        shared_counter = first.shared_counter
        within_counter = first.within_counter
        counter_semantics = first.counter_semantics
    else:
        shared_counter = None
        within_counter = None
        counter_semantics = "compiler_aux_only"
    sites = tuple(
        site
        for definition in definitions
        for site in definition.equivalent_declaration_sites
    )
    return dataclasses.replace(
        first,
        shared_counter=shared_counter,
        within_counter=within_counter,
        counter_semantics=counter_semantics,
        declaration_command=(
            first.declaration_command
            if len(command_names) == 1
            else "compatible_static_declarations"
        ),
        equivalent_declaration_sites=sites,
    )


def _normalize_sources(
    sources: Mapping[str | Path, str],
) -> tuple[tuple[Path, str], ...]:
    if not isinstance(sources, Mapping):
        raise StructuralIRError("sources must be a path-to-text mapping")
    output: list[tuple[Path, str]] = []
    for raw_path, source in sources.items():
        path = Path(raw_path)
        if not isinstance(source, str):
            raise StructuralIRError(f"source text must be str: {path}")
        output.append((path, source))
    return tuple(output)


def read_source_files(source_files: Sequence[str | Path]) -> dict[Path, str]:
    """Read an explicit source-file sequence as strict UTF-8."""

    output: dict[Path, str] = {}
    for value in source_files:
        path = Path(value)
        if path in output:
            raise StructuralIRError(f"duplicate source file: {path}")
        try:
            output[path] = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise StructuralIRError(f"cannot read UTF-8 source {path}: {exc}") from exc
    return output


def collect_theorem_definitions_from_sources(
    sources: Mapping[str | Path, str],
) -> TheoremDefinitionRegistry:
    """Collect literal standard and LLNCS theorem declarations in source order.

    Equal visible declarations in mutually exclusive source branches are
    merged and downgraded to compiler-AUX-only counter semantics.  A changed
    caption or numbered/unnumbered disagreement is a real conflict and poisons
    that environment.
    """

    normalized = _normalize_sources(sources)
    accepted: list[TheoremDefinition] = []
    rejections: list[StructuralRejection] = []
    poisoned: set[str] = set()
    occurrences: dict[str, list[TheoremDefinition]] = {}

    for source_file, source in normalized:
        masked = mask_tex_comments(source)
        macro_definition_bodies = _macro_definition_body_spans(masked)
        for match in _DECLARATION_COMMAND_RE.finditer(masked):
            if _is_macro_definition_target(masked, match.start()):
                continue
            dynamic_body = _containing_span(
                match.start(), macro_definition_bodies
            )
            if dynamic_body is not None:
                cursor = match.end()
                if cursor < len(masked) and masked[cursor] == "*":
                    cursor += 1
                env_group = _parse_group(masked, cursor, "{", "}")
                environment: str | None = None
                if env_group is not None:
                    candidate = env_group[0].strip()
                    if _IDENTIFIER_RE.fullmatch(candidate):
                        environment = candidate
                        poisoned.add(environment)
                        cursor = env_group[2]
                rejections.append(
                    _span_rejection(
                        stage="theorem_definition",
                        code="dynamic_declaration_context",
                        message=(
                            "theorem declaration occurs inside a TeX macro "
                            "replacement body and is not proved to execute"
                        ),
                        source_file=source_file,
                        source=source,
                        span=(match.start(), max(match.end(), cursor)),
                        environment=environment,
                    )
                )
                continue
            parsed = _parse_declaration(source, masked, source_file, match)
            if parsed.rejection is not None:
                rejections.append(parsed.rejection)
                if parsed.environment is not None:
                    poisoned.add(parsed.environment)
                continue
            assert parsed.definition is not None
            occurrences.setdefault(parsed.definition.environment, []).append(
                parsed.definition
            )

    for environment, definitions in occurrences.items():
        if environment in poisoned:
            continue
        if len(definitions) > 1:
            if _compatible_static_declarations(definitions):
                accepted.append(_merge_compatible_declarations(definitions))
            else:
                for definition in definitions:
                    source = dict(normalized)[definition.source_file]
                    rejections.append(
                        _span_rejection(
                            stage="theorem_definition",
                            code="conflicting_or_redefined_environment",
                            message=(
                                f"environment {environment!r} has "
                                f"{len(definitions)} declarations"
                            ),
                            source_file=definition.source_file,
                            source=source,
                            span=definition.declaration_span,
                            environment=environment,
                        )
                    )
            continue
        accepted.append(definitions[0])

    accepted.sort(
        key=lambda item: (
            [path for path, _ in normalized].index(item.source_file),
            item.declaration_span[0],
        )
    )
    return TheoremDefinitionRegistry(
        definitions=tuple(accepted),
        rejections=tuple(rejections),
        source_files=tuple(path for path, _ in normalized),
    )


def collect_theorem_definitions(
    source_files: Sequence[str | Path],
) -> TheoremDefinitionRegistry:
    return collect_theorem_definitions_from_sources(read_source_files(source_files))


def _parse_environment_token(
    masked: str, match: re.Match[str]
) -> _EnvironmentToken | None:
    group = _parse_group(masked, match.end(), "{", "}")
    if group is None:
        return None
    raw_environment, _, end = group
    environment = raw_environment.strip()
    if not _IDENTIFIER_RE.fullmatch(environment):
        return None
    return _EnvironmentToken(match.group("kind"), environment, (match.start(), end))


def _scan_labels(
    source: str,
    masked: str,
    source_file: Path,
    base_offset: int,
    environment: str | None,
) -> tuple[list[tuple[str, tuple[int, int]]], list[StructuralRejection]]:
    labels: list[tuple[str, tuple[int, int]]] = []
    rejections: list[StructuralRejection] = []
    for match in _LABEL_COMMAND_RE.finditer(masked):
        group = _parse_group(masked, match.end(), "{", "}")
        if group is None:
            local_span = (match.start(), match.end())
            rejections.append(
                _span_rejection(
                    stage="aux_number_resolution",
                    code="malformed_label",
                    message="label must have one balanced literal argument",
                    source_file=source_file,
                    source=source,
                    span=local_span,
                    environment=environment,
                )
            )
            continue
        raw_label, content_span, end = group
        label = raw_label.strip()
        if not _LABEL_RE.fullmatch(label):
            rejections.append(
                _span_rejection(
                    stage="aux_number_resolution",
                    code="dynamic_label",
                    message="label key is not a static literal",
                    source_file=source_file,
                    source=source,
                    span=(match.start(), end),
                    environment=environment,
                )
            )
            continue
        labels.append(
            (
                label,
                (base_offset + content_span[0], base_offset + content_span[1]),
            )
        )
    return labels, rejections


def _aux_numbers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if hasattr(value, "number"):
        number = str(getattr(value, "number")).strip()
        return (number,) if number else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        output: list[str] = []
        for item in value:
            if hasattr(item, "number"):
                item = getattr(item, "number")
            number = str(item).strip()
            if number:
                output.append(number)
        return tuple(output)
    number = str(value).strip()
    return (number,) if number else ()


def _heading_candidates(
    *,
    definition: TheoremDefinition,
    optional_title: StaticVisibleText | None,
    optional_title_source: str | None,
    label: str,
    number: str,
    source_file: Path,
    block_span: tuple[int, int],
    source_lines: tuple[int, int],
) -> tuple[TheoremHeadingCandidate, ...]:
    prefix = f"{definition.visible_name_markdown} {number}"
    variants: list[tuple[str, str]] = []
    if optional_title is None:
        variants.extend(
            [
                ("name_number_bare", prefix),
                ("name_number_period", prefix + "."),
                ("name_number_colon", prefix + ":"),
            ]
        )
    else:
        title = optional_title.markdown
        variants.extend(
            [
                ("name_number_parenthesized_title", f"{prefix} ({title})"),
                ("name_number_parenthesized_title_period", f"{prefix} ({title})."),
                ("name_number_period_title", f"{prefix}. {title}"),
                ("name_number_colon_title", f"{prefix}: {title}"),
                ("name_number_period_parenthesized_title", f"{prefix}. ({title})"),
            ]
        )
    candidates: list[TheoremHeadingCandidate] = []
    seen: set[str] = set()
    for variant, visible_text in variants:
        if visible_text in seen:
            continue
        seen.add(visible_text)
        policy = f"source_aux_theorem_heading_{variant}_v1"
        candidate_id = hashlib.sha256(
            (
                f"{source_file}|{block_span[0]}|{block_span[1]}|"
                f"{label}|{number}|{policy}|{visible_text}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        candidates.append(
            TheoremHeadingCandidate(
                candidate_id=candidate_id,
                policy=policy,
                visible_text=visible_text,
                markdown=visible_text,
                source_file=source_file,
                block_span=block_span,
                definition_span=definition.declaration_span,
                source_lines=source_lines,
                environment=definition.environment,
                label=label,
                aux_number=number,
                optional_title_source=optional_title_source,
            )
        )
    return tuple(candidates)


def _build_one_theorem_block(
    *,
    source: str,
    masked: str,
    source_file: Path,
    opening: _EnvironmentToken,
    closing: _EnvironmentToken,
    definition: TheoremDefinition,
    aux_label_numbers: Mapping[str, Any],
    block_ordinal: int,
) -> tuple[TheoremBlock | None, tuple[StructuralRejection, ...]]:
    block_span = (opening.command_span[0], closing.command_span[1])
    source_lines = _line_range(source, block_span)
    if not definition.numbered:
        return None, (
            _span_rejection(
                stage="aux_number_resolution",
                code="unnumbered_theorem_has_no_aux_number",
                message=(
                    "starred theorem declaration is admitted as source "
                    "metadata but cannot produce a numbered heading candidate"
                ),
                source_file=source_file,
                source=source,
                span=block_span,
                environment=opening.environment,
            ),
        )
    cursor = _skip_space(masked, opening.command_span[1])
    optional_title_source: str | None = None
    optional_title: StaticVisibleText | None = None
    optional_title_span: tuple[int, int] | None = None
    if cursor < closing.command_span[0] and masked[cursor] == "[":
        group = _parse_group(masked, cursor, "[", "]")
        if group is None or group[2] > closing.command_span[0]:
            return None, (
                _span_rejection(
                    stage="theorem_block",
                    code="unbalanced_optional_title",
                    message="theorem optional title is not balanced",
                    source_file=source_file,
                    source=source,
                    span=block_span,
                    environment=opening.environment,
                ),
            )
        title_masked, title_span, cursor = group
        optional_title_source = source[slice(*title_span)]
        optional_title_span = title_span
        try:
            optional_title = render_static_visible_text(title_masked)
        except StructuralIRError as exc:
            return None, (
                _span_rejection(
                    stage="theorem_block",
                    code="unsafe_optional_title",
                    message=str(exc),
                    source_file=source_file,
                    source=source,
                    span=title_span,
                    environment=opening.environment,
                ),
            )

    body_span = (cursor, closing.command_span[0])
    local_source = source[slice(*block_span)]
    local_masked = masked[slice(*block_span)]
    labels, label_rejections = _scan_labels(
        local_source,
        local_masked,
        source_file,
        block_span[0],
        opening.environment,
    )
    if label_rejections:
        # _scan_labels was given a local string, so fix its line/span metadata
        # to the original source coordinates before exposing it.
        adjusted: list[StructuralRejection] = []
        for rejection in label_rejections:
            absolute_span = (
                block_span[0] + rejection.char_span[0],
                block_span[0] + rejection.char_span[1],
            )
            adjusted.append(
                _span_rejection(
                    stage=rejection.stage,
                    code=rejection.code,
                    message=rejection.message,
                    source_file=source_file,
                    source=source,
                    span=absolute_span,
                    environment=opening.environment,
                )
            )
        return None, tuple(adjusted)
    if len(labels) != 1:
        code = "missing_unique_label" if not labels else "multiple_labels_in_block"
        return None, (
            _span_rejection(
                stage="aux_number_resolution",
                code=code,
                message=f"expected exactly one literal label, found {len(labels)}",
                source_file=source_file,
                source=source,
                span=block_span,
                environment=opening.environment,
            ),
        )
    label, label_span = labels[0]
    numbers = _aux_numbers(aux_label_numbers.get(label))
    if len(numbers) != 1:
        code = "missing_aux_number" if not numbers else "ambiguous_aux_number"
        return None, (
            _span_rejection(
                stage="aux_number_resolution",
                code=code,
                message=f"label {label!r} resolved to {len(numbers)} AUX numbers",
                source_file=source_file,
                source=source,
                span=label_span,
                environment=opening.environment,
            ),
        )
    number = numbers[0]
    if not _AUX_NUMBER_RE.fullmatch(number):
        return None, (
            _span_rejection(
                stage="aux_number_resolution",
                code="unsafe_aux_number",
                message=f"AUX number is not a static visible token: {number!r}",
                source_file=source_file,
                source=source,
                span=label_span,
                environment=opening.environment,
            ),
        )
    candidates = _heading_candidates(
        definition=definition,
        optional_title=optional_title,
        optional_title_source=optional_title_source,
        label=label,
        number=number,
        source_file=source_file,
        block_span=block_span,
        source_lines=source_lines,
    )
    block_id = hashlib.sha256(
        f"{source_file}|{block_span[0]}|{block_span[1]}".encode("utf-8")
    ).hexdigest()[:24]
    return (
        TheoremBlock(
            block_id=f"theorem-{block_ordinal:06d}-{block_id}",
            environment=opening.environment,
            source_file=source_file,
            block_span=block_span,
            begin_span=opening.command_span,
            body_span=body_span,
            end_span=closing.command_span,
            source_lines=source_lines,
            raw_latex=source[slice(*block_span)],
            optional_title_source=optional_title_source,
            optional_title_markdown=(
                optional_title.markdown if optional_title is not None else None
            ),
            optional_title_span=optional_title_span,
            label=label,
            label_span=label_span,
            aux_number=number,
            definition=definition,
            heading_candidates=candidates,
        ),
        (),
    )


def build_theorem_ir_from_sources(
    sources: Mapping[str | Path, str],
    aux_label_numbers: Mapping[str, Any],
    *,
    registry: TheoremDefinitionRegistry | None = None,
) -> TheoremStructuralIR:
    """Parse balanced theorem blocks and construct finite heading candidates."""

    normalized = _normalize_sources(sources)
    source_mapping = dict(normalized)
    if registry is None:
        registry = collect_theorem_definitions_from_sources(source_mapping)
    definitions = registry.by_environment
    theorem_environments = set(SUPPORTED_THEOREM_ENVIRONMENTS) | set(definitions)
    blocks: list[TheoremBlock] = []
    rejections: list[StructuralRejection] = []

    for source_file, source in normalized:
        masked = mask_tex_comments(source)
        stack: list[_OpenEnvironment] = []
        for match in _ENVIRONMENT_COMMAND_RE.finditer(masked):
            token = _parse_environment_token(masked, match)
            if token is None or token.environment not in theorem_environments:
                continue
            if token.kind == "begin":
                if stack:
                    for frame in stack:
                        frame.invalid = True
                    rejections.append(
                        _span_rejection(
                            stage="theorem_block",
                            code="nested_theorem_environment",
                            message=(
                                f"nested {token.environment!r} inside "
                                f"{stack[-1].token.environment!r}"
                            ),
                            source_file=source_file,
                            source=source,
                            span=(stack[-1].token.command_span[0], token.command_span[1]),
                            environment=stack[-1].token.environment,
                        )
                    )
                    stack.append(_OpenEnvironment(token=token, invalid=True))
                else:
                    stack.append(_OpenEnvironment(token=token))
                continue

            if not stack:
                rejections.append(
                    _span_rejection(
                        stage="theorem_block",
                        code="unexpected_theorem_end",
                        message=f"end for {token.environment!r} has no matching begin",
                        source_file=source_file,
                        source=source,
                        span=token.command_span,
                        environment=token.environment,
                    )
                )
                continue
            frame = stack[-1]
            if frame.token.environment != token.environment:
                frame.invalid = True
                rejections.append(
                    _span_rejection(
                        stage="theorem_block",
                        code="mismatched_theorem_end",
                        message=(
                            f"expected end for {frame.token.environment!r}, got "
                            f"{token.environment!r}"
                        ),
                        source_file=source_file,
                        source=source,
                        span=(frame.token.command_span[0], token.command_span[1]),
                        environment=frame.token.environment,
                    )
                )
                # Clear the complete uncertain nesting context.  Continuing
                # with any of these opens would fabricate a source span.
                stack.clear()
                continue
            stack.pop()
            if frame.invalid:
                continue
            definition = definitions.get(token.environment)
            if definition is None:
                rejections.append(
                    _span_rejection(
                        stage="theorem_block",
                        code="missing_static_theorem_definition",
                        message=(
                            f"no unique admitted newtheorem declaration for "
                            f"{token.environment!r}"
                        ),
                        source_file=source_file,
                        source=source,
                        span=(frame.token.command_span[0], token.command_span[1]),
                        environment=token.environment,
                    )
                )
                continue
            block, block_rejections = _build_one_theorem_block(
                source=source,
                masked=masked,
                source_file=source_file,
                opening=frame.token,
                closing=token,
                definition=definition,
                aux_label_numbers=aux_label_numbers,
                block_ordinal=len(blocks) + 1,
            )
            rejections.extend(block_rejections)
            if block is not None:
                blocks.append(block)

        for frame in stack:
            rejections.append(
                _span_rejection(
                    stage="theorem_block",
                    code="unbalanced_theorem_environment",
                    message=f"begin for {frame.token.environment!r} has no matching end",
                    source_file=source_file,
                    source=source,
                    span=(frame.token.command_span[0], len(source)),
                    environment=frame.token.environment,
                )
            )

    blocks.sort(
        key=lambda item: (
            [path for path, _ in normalized].index(item.source_file),
            item.block_span[0],
        )
    )
    return TheoremStructuralIR(
        registry=registry,
        blocks=tuple(blocks),
        rejections=tuple(rejections),
    )


def build_theorem_ir(
    source_files: Sequence[str | Path],
    aux_label_numbers: Mapping[str, Any],
) -> TheoremStructuralIR:
    sources = read_source_files(source_files)
    return build_theorem_ir_from_sources(sources, aux_label_numbers)


def _equation_rejected(
    *,
    formula_markdown: str,
    source_file: Path,
    block_span: tuple[int, int],
    source_lines: tuple[int, int],
    rejections: Sequence[StructuralRejection],
) -> EquationTailResolution:
    return EquationTailResolution(
        status="rejected",
        formula_markdown=formula_markdown,
        source_file=source_file,
        block_span=block_span,
        number_source_span=None,
        source_lines=source_lines,
        label=None,
        number=None,
        number_source=None,
        candidates=(),
        rejections=tuple(rejections),
    )


def resolve_display_equation_tail(
    raw_latex: str,
    formula_markdown: str,
    *,
    source_file: str | Path,
    aux_label_numbers: Mapping[str, Any],
    source_offset: int = 0,
    start_line: int = 1,
) -> EquationTailResolution:
    """Resolve finite equation tails from one source-derived display block.

    ``formula_markdown`` is stored unchanged and is only concatenated with a
    newline plus the candidate tail.  An explicit literal ``\\tag{...}`` wins;
    otherwise exactly one literal label and one AUX number are required.
    """

    if not isinstance(raw_latex, str) or not isinstance(formula_markdown, str):
        raise StructuralIRError("raw_latex and formula_markdown must be strings")
    if source_offset < 0 or start_line < 1:
        raise StructuralIRError("source_offset must be non-negative and start_line positive")
    path = Path(source_file)
    masked = mask_tex_comments(raw_latex)
    block_span = (source_offset, source_offset + len(raw_latex))
    source_lines = (start_line, start_line + raw_latex.count("\n"))
    tags: list[tuple[StaticVisibleText, tuple[int, int]]] = []
    tag_rejections: list[StructuralRejection] = []
    for match in _TAG_COMMAND_RE.finditer(masked):
        if match.group("star"):
            tag_rejections.append(
                _span_rejection(
                    stage="equation_number_resolution",
                    code="unsupported_starred_tag",
                    message="only literal non-starred tag is admitted",
                    source_file=path,
                    source=raw_latex,
                    span=(match.start(), match.end()),
                )
            )
            continue
        group = _parse_group(masked, match.end(), "{", "}")
        if group is None:
            tag_rejections.append(
                _span_rejection(
                    stage="equation_number_resolution",
                    code="malformed_equation_tag",
                    message="tag must have one balanced literal argument",
                    source_file=path,
                    source=raw_latex,
                    span=(match.start(), match.end()),
                )
            )
            continue
        tag_masked, tag_span, _ = group
        try:
            tag = render_static_visible_text(tag_masked)
        except StructuralIRError as exc:
            tag_rejections.append(
                _span_rejection(
                    stage="equation_number_resolution",
                    code="unsafe_equation_tag",
                    message=str(exc),
                    source_file=path,
                    source=raw_latex,
                    span=(match.start(), tag_span[1] + 1),
                )
            )
            continue
        tags.append(
            (
                tag,
                (source_offset + tag_span[0], source_offset + tag_span[1]),
            )
        )
    if tag_rejections:
        adjusted = tuple(
            dataclasses.replace(
                item,
                char_span=(
                    source_offset + item.char_span[0],
                    source_offset + item.char_span[1],
                ),
                source_lines=(
                    start_line + item.source_lines[0] - 1,
                    start_line + item.source_lines[1] - 1,
                ),
            )
            for item in tag_rejections
        )
        return _equation_rejected(
            formula_markdown=formula_markdown,
            source_file=path,
            block_span=block_span,
            source_lines=source_lines,
            rejections=adjusted,
        )
    if len(tags) > 1:
        rejection = StructuralRejection(
            stage="equation_number_resolution",
            code="multiple_equation_tags",
            message=f"expected at most one literal tag, found {len(tags)}",
            source_file=path,
            char_span=block_span,
            source_lines=source_lines,
        )
        return _equation_rejected(
            formula_markdown=formula_markdown,
            source_file=path,
            block_span=block_span,
            source_lines=source_lines,
            rejections=(rejection,),
        )

    label: str | None = None
    if tags:
        # The static serializer may intentionally add Markdown escaping or
        # formatting.  Keep that source-derived rendering for the candidate;
        # unlike an AUX counter, an explicit tag need not be alphanumeric
        # (``\\tag{*}`` is common and deterministic).
        number = tags[0][0].markdown
        number_source_span = tags[0][1]
        number_source = "explicit_tag"
    else:
        labels, local_rejections = _scan_labels(
            raw_latex, masked, path, source_offset, None
        )
        if local_rejections:
            adjusted = tuple(
                dataclasses.replace(
                    item,
                    char_span=(
                        source_offset + item.char_span[0],
                        source_offset + item.char_span[1],
                    ),
                    source_lines=(
                        start_line + item.source_lines[0] - 1,
                        start_line + item.source_lines[1] - 1,
                    ),
                )
                for item in local_rejections
            )
            return _equation_rejected(
                formula_markdown=formula_markdown,
                source_file=path,
                block_span=block_span,
                source_lines=source_lines,
                rejections=adjusted,
            )
        if len(labels) != 1:
            code = "missing_unique_label" if not labels else "multiple_labels_in_block"
            rejection = StructuralRejection(
                stage="equation_number_resolution",
                code=code,
                message=f"expected exactly one literal equation label, found {len(labels)}",
                source_file=path,
                char_span=block_span,
                source_lines=source_lines,
            )
            return _equation_rejected(
                formula_markdown=formula_markdown,
                source_file=path,
                block_span=block_span,
                source_lines=source_lines,
                rejections=(rejection,),
            )
        label = labels[0][0]
        number_source_span = labels[0][1]
        numbers = _aux_numbers(aux_label_numbers.get(label))
        if len(numbers) != 1:
            code = "missing_aux_number" if not numbers else "ambiguous_aux_number"
            rejection = StructuralRejection(
                stage="equation_number_resolution",
                code=code,
                message=f"label {label!r} resolved to {len(numbers)} AUX numbers",
                source_file=path,
                char_span=labels[0][1],
                source_lines=source_lines,
            )
            return _equation_rejected(
                formula_markdown=formula_markdown,
                source_file=path,
                block_span=block_span,
                source_lines=source_lines,
                rejections=(rejection,),
            )
        number = numbers[0]
        number_source = "compiler_aux"

    if (
        number_source == "compiler_aux"
        and not _AUX_NUMBER_RE.fullmatch(number)
    ) or (
        number_source == "explicit_tag"
        and (not number.strip() or len(number) > 128 or "\n" in number or "\r" in number)
    ):
        rejection = StructuralRejection(
            stage="equation_number_resolution",
            code="unsafe_equation_number",
            message=f"equation number is not a static visible token: {number!r}",
            source_file=path,
            char_span=block_span,
            source_lines=source_lines,
        )
        return _equation_rejected(
            formula_markdown=formula_markdown,
            source_file=path,
            block_span=block_span,
            source_lines=source_lines,
            rejections=(rejection,),
        )

    candidates: list[EquationTailCandidate] = []
    for variant, tail in (("parenthesized", f"({number})"), ("bare", number)):
        policy = f"source_{number_source}_equation_tail_{variant}_v1"
        candidate_id = hashlib.sha256(
            (
                f"{path}|{block_span[0]}|{block_span[1]}|{label}|"
                f"{number}|{policy}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        candidates.append(
            EquationTailCandidate(
                candidate_id=candidate_id,
                policy=policy,
                number_source=number_source,
                tail_text=tail,
                formula_markdown=formula_markdown,
                markdown=formula_markdown + "\n" + tail,
                source_file=path,
                block_span=block_span,
                number_source_span=number_source_span,
                source_lines=source_lines,
                label=label,
                number=number,
            )
        )
    return EquationTailResolution(
        status="accepted",
        formula_markdown=formula_markdown,
        source_file=path,
        block_span=block_span,
        number_source_span=number_source_span,
        source_lines=source_lines,
        label=label,
        number=number,
        number_source=number_source,
        candidates=tuple(candidates),
        rejections=(),
    )


__all__ = [
    "EquationTailCandidate",
    "EquationTailResolution",
    "SourceSpan",
    "StaticVisibleText",
    "StructuralIRError",
    "StructuralIRSafetyError",
    "StructuralRejection",
    "SUPPORTED_THEOREM_ENVIRONMENTS",
    "TheoremBlock",
    "TheoremDefinition",
    "TheoremDefinitionRegistry",
    "TheoremHeadingCandidate",
    "TheoremStructuralIR",
    "build_theorem_ir",
    "build_theorem_ir_from_sources",
    "collect_theorem_definitions",
    "collect_theorem_definitions_from_sources",
    "mask_tex_comments",
    "read_source_files",
    "render_static_visible_text",
    "resolve_display_equation_tail",
]
