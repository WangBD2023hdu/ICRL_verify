"""Recover theorem/proof captions from explicit LaTeX source declarations.

Environment names are implementation identifiers and are never treated as
visible titles.  This scanner accepts only balanced, literal declarations in
the submitted ``.tex/.sty/.cls`` source tree.  Conflicting or macro-generated
captions remain unresolved and therefore fail closed in ``DocumentAst``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Mapping


_DECLARATION = re.compile(r"\\(spnewtheorem|newtheorem)(\*)?")
_THEOREM_STYLE = re.compile(r"\\theoremstyle")
_AMSTHM_PACKAGE = re.compile(
    r"\\(?:usepackage|RequirePackage)(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}"
)
_SAFE_ENVIRONMENT = re.compile(r"^[A-Za-z][A-Za-z0-9@*_-]*$")
_SAFE_CAPTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .()'’\-:]*$")
COMPILER_CONTRACT_HEADER_STYLE = "compiler_contract_required"
COMPILER_CONTRACT_PUNCTUATION = "compiler_contract_required"


@dataclass(frozen=True, slots=True)
class SemanticEnvironmentDefinition:
    environment_name: str
    display_name: str
    kind: str
    numbered: bool
    header_style: str
    punctuation: str
    declaration_command: str
    source_path: str
    body_style: str = "plain"
    counter_name: str | None = None
    compiler_contract_required: bool = False

    def metadata(self) -> tuple[tuple[str, str], ...]:
        metadata = [
            ("display_name", self.display_name),
            ("header_style", self.header_style),
            ("body_style", self.body_style),
            ("punctuation", self.punctuation),
            ("declaration_command", self.declaration_command),
            ("declaration_source", self.source_path),
            (
                "compiler_contract_required",
                "true" if self.compiler_contract_required else "false",
            ),
        ]
        if self.counter_name is not None:
            metadata.append(("counter_name", self.counter_name))
        return tuple(metadata)


def _explicit_environment_overrides(
    sources: Mapping[str, str],
) -> frozenset[str]:
    """Return literal environment names explicitly redefined in source.

    The scanner is intentionally conservative.  A standard ``\\newtheorem``
    declaration is useful only while its generated begin/end macros remain
    intact.  Any later literal environment/command definition for that name
    vetoes the compiler-contract path instead of asking the AST to guess which
    definition wins after TeX expansion.
    """

    names: set[str] = set()
    environment_pattern = re.compile(r"\\(?:new|renew)environment\*?")
    command_pattern = re.compile(
        r"\\(?:newcommand|renewcommand|providecommand)\*?"
    )
    primitive_pattern = re.compile(r"\\(?:def|gdef|edef|xdef)\s*\\([A-Za-z@]+)")
    let_pattern = re.compile(r"\\let\s*\\([A-Za-z@]+)")
    for source in sources.values():
        for match in environment_pattern.finditer(source):
            if _position_is_commented(source, match.start()):
                continue
            group = _group(source, match.end(), "{", "}")
            if group is not None and _SAFE_ENVIRONMENT.fullmatch(group[0].strip()):
                names.add(group[0].strip())
        for match in command_pattern.finditer(source):
            if _position_is_commented(source, match.start()):
                continue
            group = _group(source, match.end(), "{", "}")
            if group is None:
                continue
            command = group[0].strip()
            if command.startswith("\\") and _SAFE_ENVIRONMENT.fullmatch(command[1:]):
                names.add(command[1:])
        for pattern in (primitive_pattern, let_pattern):
            for match in pattern.finditer(source):
                if not _position_is_commented(source, match.start()):
                    names.add(match.group(1))
    return frozenset(names)


def _is_comment(value: str, position: int) -> bool:
    if value[position] != "%":
        return False
    slashes = 0
    cursor = position - 1
    while cursor >= 0 and value[cursor] == "\\":
        slashes += 1
        cursor -= 1
    return slashes % 2 == 0


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


def _position_is_commented(value: str, position: int) -> bool:
    line_start = value.rfind("\n", 0, position) + 1
    cursor = line_start
    while cursor < position:
        if value[cursor] == "%" and _is_comment(value, cursor):
            return True
        cursor += 1
    return False


def _group(value: str, position: int, opening: str, closing: str) -> tuple[str, int] | None:
    position = _skip(value, position)
    if position >= len(value) or value[position] != opening:
        return None
    depth = 0
    content_start = position + 1
    cursor = position
    while cursor < len(value):
        if value[cursor] == "%" and _is_comment(value, cursor):
            newline = value.find("\n", cursor + 1)
            cursor = len(value) if newline < 0 else newline + 1
            continue
        if value[cursor] == "\\":
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


def _header_style(raw: str) -> str | None:
    normalized = re.sub(r"\s+", "", raw)
    if normalized in {r"\bf", r"\bfseries"}:
        return "strong"
    if normalized in {r"\it", r"\itshape", r"\sl", r"\slshape"}:
        return "em"
    if normalized in {r"\rm", r"\rmfamily", r"\normalfont"}:
        return "plain"
    return None


_AMSTHM_BUILTIN_STYLES = {
    "plain": ("strong", "em"),
    "definition": ("strong", "plain"),
    "remark": ("em", "plain"),
}


def _active_builtin_theorem_style(
    source: str,
    position: int,
) -> tuple[str, str] | None:
    """Resolve the last explicit standard ``\theoremstyle`` before a declaration."""

    resolved: str | None = None
    for match in _THEOREM_STYLE.finditer(source, 0, position):
        if _position_is_commented(source, match.start()):
            continue
        group = _group(source, match.end(), "{", "}")
        if group is None:
            continue
        name = group[0].strip()
        resolved = _AMSTHM_BUILTIN_STYLES.get(name)
    return resolved


def _uses_amsthm(source: str) -> bool:
    for match in _AMSTHM_PACKAGE.finditer(source):
        if _position_is_commented(source, match.start()):
            continue
        packages = {part.strip() for part in match.group(1).split(",")}
        if "amsthm" in packages:
            return True
    return False


def extract_semantic_environment_definitions(
    sources: Mapping[str, str],
) -> tuple[dict[str, SemanticEnvironmentDefinition], tuple[dict[str, str], ...]]:
    """Return unique literal theorem declarations and rejection audit rows."""

    candidates: dict[str, list[SemanticEnvironmentDefinition]] = {}
    rejections: list[dict[str, str]] = []
    explicit_overrides = _explicit_environment_overrides(sources)
    amsthm_sources = tuple(
        source_path
        for source_path, source in sorted(sources.items())
        if _uses_amsthm(source)
    )
    if amsthm_sources:
        if "proof" in explicit_overrides:
            rejections.append(
                {
                    "source_path": ",".join(amsthm_sources),
                    "environment_name": "proof",
                    "reason": "semantic_environment_explicitly_redefined",
                }
            )
        else:
            # ``amsthm`` installs a visible Proof header and a graphical QED
            # end decoration.  Source alone does not prove either runtime
            # macro chain: admit the region only as a provisional semantic
            # environment.  The expansion-only compiler proof contract must
            # later match the complete audited amsthm ABI before any page
            # slice can serialize it.  The QED open box is then recorded as
            # an intentionally ignored graphical element, never invented as
            # a Markdown character.
            definition = SemanticEnvironmentDefinition(
                environment_name="proof",
                display_name="Proof",
                kind="proof",
                numbered=False,
                header_style=COMPILER_CONTRACT_HEADER_STYLE,
                body_style="plain",
                punctuation=COMPILER_CONTRACT_PUNCTUATION,
                declaration_command="usepackage{amsthm}",
                source_path=amsthm_sources[0],
                counter_name=None,
                compiler_contract_required=True,
            )
            candidates.setdefault("proof", []).append(definition)
    for source_path, source in sorted(sources.items()):
        for match in _DECLARATION.finditer(source):
            if _position_is_commented(source, match.start()):
                continue
            command = match.group(1)
            starred = bool(match.group(2))
            cursor = match.end()
            environment_group = _group(source, cursor, "{", "}")
            if environment_group is None:
                continue
            environment, cursor = environment_group
            optional_shared = _group(source, cursor, "[", "]")
            shared_counter: str | None = None
            if optional_shared is not None:
                shared_counter, cursor = optional_shared
                shared_counter = shared_counter.strip()
            caption_group = _group(source, cursor, "{", "}")
            if caption_group is None:
                continue
            caption, cursor = caption_group
            if _SAFE_ENVIRONMENT.fullmatch(environment.strip()) is None:
                continue
            environment = environment.strip()
            if (
                shared_counter is not None
                and _SAFE_ENVIRONMENT.fullmatch(shared_counter) is None
            ):
                rejections.append(
                    {
                        "source_path": source_path,
                        "environment_name": environment,
                        "reason": "nonliteral_semantic_counter",
                    }
                )
                continue
            if environment in explicit_overrides:
                rejections.append(
                    {
                        "source_path": source_path,
                        "environment_name": environment,
                        "reason": "semantic_environment_explicitly_redefined",
                    }
                )
                continue
            display_name = caption.strip()
            if _SAFE_CAPTION.fullmatch(display_name) is None:
                rejections.append(
                    {
                        "source_path": source_path,
                        "environment_name": environment,
                        "reason": "nonliteral_semantic_caption",
                    }
                )
                continue
            style: str | None = None
            body_style: str | None = None
            punctuation = "."
            compiler_contract_required = False
            if command == "spnewtheorem":
                optional_within = _group(source, cursor, "[", "]")
                if optional_within is not None:
                    _within, cursor = optional_within
                header_group = _group(source, cursor, "{", "}")
                if header_group is not None:
                    header, cursor = header_group
                    style = _header_style(header)
                    body_group = _group(source, cursor, "{", "}")
                    if body_group is not None:
                        body, cursor = body_group
                        body_style = _header_style(body)
            elif command == "newtheorem":
                builtin_styles = _active_builtin_theorem_style(
                    source, match.start()
                )
                if builtin_styles is not None:
                    style, body_style = builtin_styles
                if style is None and not starred:
                    # A literal kernel-style ``\newtheorem`` declaration has
                    # source-derived caption/counter semantics, but its
                    # header style and punctuation remain class-dependent.
                    # Admit its body into the AST only as a pending semantic
                    # environment.  The compiler-number shadow must later
                    # prove the exact standard begin/end and theorem-header
                    # macro contract before any page can serialize it.
                    style = COMPILER_CONTRACT_HEADER_STYLE
                    punctuation = COMPILER_CONTRACT_PUNCTUATION
                    compiler_contract_required = True
                    body_style = "em"
            if style is None or body_style is None:
                rejections.append(
                    {
                        "source_path": source_path,
                        "environment_name": environment,
                        "reason": "semantic_header_style_not_source_resolved",
                    }
                )
                continue
            definition = SemanticEnvironmentDefinition(
                environment_name=environment,
                display_name=display_name,
                kind=("proof" if environment == "proof" else "theorem"),
                numbered=not starred,
                header_style=style,
                body_style=body_style,
                punctuation=punctuation,
                declaration_command=command,
                source_path=source_path,
                counter_name=(None if starred else shared_counter or environment),
                compiler_contract_required=compiler_contract_required,
            )
            candidates.setdefault(definition.environment_name, []).append(definition)

    resolved: dict[str, SemanticEnvironmentDefinition] = {}
    for environment, rows in sorted(candidates.items()):
        semantic_keys = {
            (
                row.display_name,
                row.kind,
                row.numbered,
                row.header_style,
                row.body_style,
                row.punctuation,
                row.counter_name,
                row.compiler_contract_required,
            )
            for row in rows
        }
        if len(semantic_keys) == 1:
            resolved[environment] = rows[0]
        else:
            rejections.append(
                {
                    "source_path": ",".join(sorted({row.source_path for row in rows})),
                    "environment_name": environment,
                    "reason": "conflicting_semantic_declarations",
                }
            )
    return resolved, tuple(rejections)


__all__ = [
    "COMPILER_CONTRACT_HEADER_STYLE",
    "COMPILER_CONTRACT_PUNCTUATION",
    "SemanticEnvironmentDefinition",
    "extract_semantic_environment_definitions",
]
