"""Strict PDF-side projections for the experimental source-first verifier.

The source Markdown is already frozen when this module is called.  A
projection may remove a layout-only token from the *observed PDF verifier
stream*, but it must never copy PDF text into, repair, or otherwise alter the
ground truth.

The PDF-side projection implemented here is deliberately narrow: remove one
isolated, horizontally centred, pure-numeric folio from the bottom page
margin.  Ambiguous geometry, multiple folio-like tokens, non-centred tokens,
body-region numbers, and verifier-order disagreement all fail closed.

The module also provides a source-only projection for the verifier's math
stream.  It removes a small allow-list of font/style wrappers and paired
layout environments, preserves control-word boundaries across removed
wrappers, and maps a tiny allow-list of unambiguous operators *inside
already-delimited math only*.  This is not a TeX parser and deliberately does
not guess: malformed delimiters or groups, unsupported environments, and
ambiguous layout arguments reject the entire projection and return the frozen
Markdown unchanged.

The module does not import the stable verifier.  Callers can pass its character
stream function as ``stream_projector``; this keeps discretionary line-end
hyphen handling and source-side math-brace projection independently
composable.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

PROJECTION_VERSION = "strict_bottom_centered_folio_v1"
MATH_VISIBLE_FLOW_PROJECTION_VERSION = "strict_source_math_visible_flow_v2"


MATH_VISIBLE_STYLE_WRAPPERS = frozenset(
    {
        "textmd",
        "textnormal",
        "textrm",
        "textsf",
        "texttt",
        "textup",
        "textit",
        "textsl",
        "textbf",
        "mathrm",
        "mathbf",
        "mathit",
        "mathsf",
        "mathtt",
        "mathcal",
        "mathbb",
        "mathds",
        "mathfrak",
        "boldsymbol",
    }
)


# These control words have one context-independent visible glyph in ordinary
# TeX math.  Keep this allow-list intentionally small: an unknown or
# context-sensitive command remains in the projected source so the downstream
# exact verifier fails closed instead of silently guessing its appearance.
MATH_VISIBLE_OPERATOR_COMMANDS = {
    "land": "∧",
    "lor": "∨",
    "neg": "¬",
}

_TEXT_MODE_STYLE_WRAPPERS = frozenset(
    {
        "textmd",
        "textnormal",
        "textrm",
        "textsf",
        "texttt",
        "textup",
        "textit",
        "textsl",
        "textbf",
    }
)


MATH_LAYOUT_ENVIRONMENTS = frozenset(
    {
        "aligned",
        "alignedat",
        "gathered",
        "split",
        "cases",
    }
)


@dataclass(frozen=True, slots=True)
class MathVisibleFlowProjectionResult:
    """Source-derived Markdown used only by the exact verifier.

    ``source_markdown`` is the frozen GT and is always returned byte-for-byte
    unchanged.  ``projected_markdown`` is a deterministic source-only view;
    on any safety rejection it is rolled back to ``source_markdown``.  PDF
    text is neither accepted by nor available to this API.
    """

    source_markdown: str
    projected_markdown: str
    provenance: Mapping[str, Any]

    @property
    def projection_applied(self) -> bool:
        return bool(self.provenance.get("projection_applied", False))

    @property
    def status(self) -> str:
        return str(self.provenance.get("status", "unknown"))


class _MathProjectionRejected(ValueError):
    """Internal all-or-nothing rejection carrying a source character offset."""

    def __init__(self, reason: str, offset: int) -> None:
        super().__init__(f"{reason} at source offset {offset}")
        self.reason = reason
        self.offset = offset


@dataclass(slots=True)
class _MathProjectionCounters:
    math_regions_seen: int = 0
    math_regions_projected: int = 0
    style_wrappers_removed: int = 0
    layout_environments_removed: int = 0
    layout_alignment_tabs_removed: int = 0
    layout_row_breaks_removed: int = 0
    fenced_code_blocks_seen: int = 0
    fenced_code_dollars_guarded: int = 0
    text_mode_hyphens_guarded: int = 0
    text_mode_underscores_guarded: int = 0
    control_word_boundaries_inserted: int = 0
    operator_commands_projected: int = 0
    style_wrapper_counts: dict[str, int] = field(default_factory=dict)
    layout_environment_counts: dict[str, int] = field(default_factory=dict)
    math_delimiter_counts: dict[str, int] = field(default_factory=dict)
    operator_command_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "math_regions_seen": self.math_regions_seen,
            "math_regions_projected": self.math_regions_projected,
            "style_wrappers_removed": self.style_wrappers_removed,
            "style_wrapper_counts": dict(sorted(self.style_wrapper_counts.items())),
            "layout_environments_removed": self.layout_environments_removed,
            "layout_environment_counts": dict(
                sorted(self.layout_environment_counts.items())
            ),
            "layout_alignment_tabs_removed": self.layout_alignment_tabs_removed,
            "layout_row_breaks_removed": self.layout_row_breaks_removed,
            "fenced_code_blocks_seen": self.fenced_code_blocks_seen,
            "fenced_code_dollars_guarded": self.fenced_code_dollars_guarded,
            "text_mode_hyphens_guarded": self.text_mode_hyphens_guarded,
            "text_mode_underscores_guarded": self.text_mode_underscores_guarded,
            "control_word_boundaries_inserted": (
                self.control_word_boundaries_inserted
            ),
            "operator_commands_projected": self.operator_commands_projected,
            "operator_command_counts": dict(
                sorted(self.operator_command_counts.items())
            ),
            "math_delimiter_counts": dict(sorted(self.math_delimiter_counts.items())),
        }


_FULLWIDTH_DOLLAR = "\N{FULLWIDTH DOLLAR SIGN}"
_FULLWIDTH_HYPHEN_MINUS = "\N{FULLWIDTH HYPHEN-MINUS}"
_FULLWIDTH_LOW_LINE = "\N{FULLWIDTH LOW LINE}"
_FENCE_OPENING = re.compile(
    r"(?m)^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^\r\n]*)"
    r"(?P<eol>\r?\n|$)"
)
_LAYOUT_ROW_SPACING = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))\s*"
    r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em)"
)


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _is_escaped(value: str, offset: int) -> bool:
    backslashes = 0
    cursor = offset - 1
    while cursor >= 0 and value[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _starts_with_control_word_letter(value: str) -> bool:
    """Return whether ``value`` can extend a preceding TeX control word."""

    return bool(
        value
        and value[0].isascii()
        and (value[0].isalpha() or value[0] == "@")
    )


def _ends_with_control_word(value: str) -> bool:
    """Recognize an unterminated TeX control word at the end of ``value``.

    Style-wrapper projection removes source braces.  Without this check,
    ``\\foo\\textsf{bar}`` would become ``\\foobar`` and silently change the
    command token seen by the downstream strict verifier.
    """

    cursor = len(value)
    while cursor > 0 and value[cursor - 1].isascii() and (
        value[cursor - 1].isalpha() or value[cursor - 1] == "@"
    ):
        cursor -= 1
    return (
        cursor < len(value)
        and cursor > 0
        and value[cursor - 1] == "\\"
        and not _is_escaped(value, cursor - 1)
    )


def _projection_provenance(
    source_markdown: str,
    projected_markdown: str,
    counters: _MathProjectionCounters,
    *,
    status: str,
    reason: str,
    projection_applied: bool,
    rejection: _MathProjectionRejected | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "version": MATH_VISIBLE_FLOW_PROJECTION_VERSION,
        "status": status,
        "reason": reason,
        "projection_applied": projection_applied,
        "ground_truth_changed": False,
        "pdf_text_used_for_ground_truth": False,
        "source_only_projection": True,
        "all_or_nothing": True,
        "edits_rolled_back": status == "rejected",
        "source_markdown_sha256": hashlib.sha256(
            source_markdown.encode("utf-8")
        ).hexdigest(),
        "projected_markdown_sha256": hashlib.sha256(
            projected_markdown.encode("utf-8")
        ).hexdigest(),
        "nfkc_placeholders": {
            "fenced_code_dollar": {
                "codepoint": f"U+{ord(_FULLWIDTH_DOLLAR):04X}",
                "nfkc_value": unicodedata.normalize("NFKC", _FULLWIDTH_DOLLAR),
            },
            "text_mode_hyphen": {
                "codepoint": f"U+{ord(_FULLWIDTH_HYPHEN_MINUS):04X}",
                "nfkc_value": unicodedata.normalize(
                    "NFKC", _FULLWIDTH_HYPHEN_MINUS
                ),
            },
            "text_mode_underscore": {
                "codepoint": f"U+{ord(_FULLWIDTH_LOW_LINE):04X}",
                "nfkc_value": unicodedata.normalize("NFKC", _FULLWIDTH_LOW_LINE),
            },
        },
        **counters.as_dict(),
    }
    if rejection is not None:
        provenance["rejection"] = {
            "reason": rejection.reason,
            "source_character_offset": rejection.offset,
        }
    return provenance


def _result_for_source_projection(
    source_markdown: str,
    candidate: str,
    counters: _MathProjectionCounters,
) -> MathVisibleFlowProjectionResult:
    applied = candidate != source_markdown
    return MathVisibleFlowProjectionResult(
        source_markdown=source_markdown,
        projected_markdown=candidate,
        provenance=_projection_provenance(
            source_markdown,
            candidate,
            counters,
            status="projected" if applied else "unchanged",
            reason=(
                "strict_source_visible_flow_projected"
                if applied
                else "no_supported_projection_needed"
            ),
            projection_applied=applied,
        ),
    )


def _rejected_source_projection(
    source_markdown: str,
    counters: _MathProjectionCounters,
    rejection: _MathProjectionRejected,
) -> MathVisibleFlowProjectionResult:
    # Roll back every prior edit.  A caller can therefore safely ignore the
    # status without ever receiving a partially projected verifier input.
    return MathVisibleFlowProjectionResult(
        source_markdown=source_markdown,
        projected_markdown=source_markdown,
        provenance=_projection_provenance(
            source_markdown,
            source_markdown,
            counters,
            status="rejected",
            reason=rejection.reason,
            projection_applied=False,
            rejection=rejection,
        ),
    )


def _guard_fenced_code_dollars(
    source_markdown: str,
    counters: _MathProjectionCounters,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Guard dollars in strictly paired fenced-code bodies.

    U+FF04 is one code point just like ``$`` and NFKC-normalizes back to the
    ASCII dollar.  Replacing it therefore preserves all source offsets and the
    verifier's final exact character stream while preventing a Markdown math
    recognizer from consuming a code fragment such as ``$i`` first.
    """

    edits: list[tuple[int, int, str]] = []
    opaque_ranges: list[tuple[int, int]] = []
    cursor = 0
    while opening := _FENCE_OPENING.search(source_markdown, cursor):
        fence = opening.group("fence")
        info = opening.group("info")
        if fence[0] == "`" and "`" in info:
            raise _MathProjectionRejected(
                "backtick_in_backtick_fence_info_string", opening.start("info")
            )
        # An opening fence without a line ending has no possible body/closing
        # fence.  Reject rather than treating it as ordinary punctuation.
        if not opening.group("eol"):
            raise _MathProjectionRejected(
                "unclosed_fenced_code_block", opening.start("fence")
            )
        marker = re.escape(fence[0])
        closing_pattern = re.compile(
            rf"(?m)^ {{0,3}}{marker}{{{len(fence)},}}[ \t]*(?:\r?\n|$)"
        )
        closing = closing_pattern.search(source_markdown, opening.end())
        if closing is None:
            raise _MathProjectionRejected(
                "unclosed_fenced_code_block", opening.start("fence")
            )
        body_start = opening.end()
        body_end = closing.start()
        body = source_markdown[body_start:body_end]
        guarded_count = body.count("$")
        if guarded_count:
            edits.append((body_start, body_end, body.replace("$", _FULLWIDTH_DOLLAR)))
        counters.fenced_code_blocks_seen += 1
        counters.fenced_code_dollars_guarded += guarded_count
        opaque_ranges.append((opening.start(), closing.end()))
        cursor = closing.end()

    projected = source_markdown
    for start, end, replacement in reversed(edits):
        projected = projected[:start] + replacement + projected[end:]
    return projected, tuple(opaque_ranges)


class _MathBodyProjector:
    """Small balanced parser for one already delimited math body."""

    def __init__(
        self,
        body: str,
        *,
        source_offset: int,
        counters: _MathProjectionCounters,
    ) -> None:
        self.body = body
        self.source_offset = source_offset
        self.counters = counters
        self.environments: list[tuple[str, int]] = []

    def _global(self, local_offset: int) -> int:
        return self.source_offset + local_offset

    def _skip_whitespace(self, offset: int) -> int:
        while offset < len(self.body) and self.body[offset].isspace():
            offset += 1
        return offset

    def _plain_group(
        self,
        offset: int,
        *,
        missing_reason: str,
        malformed_reason: str,
    ) -> tuple[str, int, int]:
        opening = self._skip_whitespace(offset)
        if opening >= len(self.body) or self.body[opening] != "{":
            raise _MathProjectionRejected(missing_reason, self._global(offset))
        cursor = opening + 1
        while cursor < len(self.body):
            character = self.body[cursor]
            if character == "}" and not _is_escaped(self.body, cursor):
                return self.body[opening + 1 : cursor], cursor + 1, opening
            if character == "{" and not _is_escaped(self.body, cursor):
                raise _MathProjectionRejected(
                    malformed_reason, self._global(cursor)
                )
            cursor += 1
        raise _MathProjectionRejected(
            "unbalanced_math_brace", self._global(opening)
        )

    def _optional_bracket(self, offset: int) -> tuple[str | None, int, int]:
        opening = self._skip_whitespace(offset)
        if opening >= len(self.body) or self.body[opening] != "[":
            return None, offset, opening
        closing = self.body.find("]", opening + 1)
        if closing < 0:
            raise _MathProjectionRejected(
                "unclosed_layout_optional_argument", self._global(opening)
            )
        content = self.body[opening + 1 : closing]
        if "[" in content:
            raise _MathProjectionRejected(
                "nested_layout_optional_argument", self._global(opening)
            )
        return content, closing + 1, opening

    def _begin_environment(self, offset: int, command_offset: int) -> int:
        name, cursor, name_opening = self._plain_group(
            offset,
            missing_reason="layout_environment_missing_name",
            malformed_reason="malformed_layout_environment_name",
        )
        if name not in MATH_LAYOUT_ENVIRONMENTS:
            raise _MathProjectionRejected(
                f"unsupported_math_layout_environment:{name}",
                self._global(name_opening),
            )

        if name in {"aligned", "alignedat", "gathered"}:
            position, after_position, position_opening = self._optional_bracket(cursor)
            if position is not None:
                if position.strip() not in {"t", "b", "c"}:
                    raise _MathProjectionRejected(
                        "unsupported_layout_position_argument",
                        self._global(position_opening),
                    )
                cursor = after_position
        if name == "alignedat":
            columns, cursor, columns_opening = self._plain_group(
                cursor,
                missing_reason="alignedat_missing_column_count",
                malformed_reason="malformed_alignedat_column_count",
            )
            if re.fullmatch(r"[1-9][0-9]*", columns.strip()) is None:
                raise _MathProjectionRejected(
                    "invalid_alignedat_column_count", self._global(columns_opening)
                )

        self.environments.append((name, self._global(command_offset)))
        self.counters.layout_environments_removed += 1
        _increment(self.counters.layout_environment_counts, name)
        return cursor

    def _end_environment(self, offset: int) -> int:
        name, cursor, name_opening = self._plain_group(
            offset,
            missing_reason="layout_environment_end_missing_name",
            malformed_reason="malformed_layout_environment_end_name",
        )
        if name not in MATH_LAYOUT_ENVIRONMENTS:
            raise _MathProjectionRejected(
                f"unsupported_math_layout_environment:{name}",
                self._global(name_opening),
            )
        if not self.environments:
            raise _MathProjectionRejected(
                f"layout_environment_end_without_begin:{name}",
                self._global(name_opening),
            )
        expected, _ = self.environments[-1]
        if expected != name:
            raise _MathProjectionRejected(
                f"mismatched_layout_environment:{expected}!={name}",
                self._global(name_opening),
            )
        self.environments.pop()
        return cursor

    def _layout_row_break(self, offset: int) -> int:
        cursor = offset + 2
        if cursor < len(self.body) and self.body[cursor] == "*":
            cursor += 1
        spacing, after_spacing, spacing_opening = self._optional_bracket(cursor)
        if spacing is not None:
            if _LAYOUT_ROW_SPACING.fullmatch(spacing.strip()) is None:
                raise _MathProjectionRejected(
                    "unsupported_layout_row_spacing",
                    self._global(spacing_opening),
                )
            cursor = after_spacing
        self.counters.layout_row_breaks_removed += 1
        return cursor

    def _segment(
        self,
        offset: int,
        *,
        closing_brace_required: bool,
        opening_brace_offset: int | None,
        text_mode: bool,
    ) -> tuple[str, int]:
        output: list[str] = []
        cursor = offset
        environment_depth = len(self.environments)
        while cursor < len(self.body):
            character = self.body[cursor]
            if character == "}":
                if not closing_brace_required:
                    raise _MathProjectionRejected(
                        "unmatched_math_closing_brace", self._global(cursor)
                    )
                if len(self.environments) != environment_depth:
                    raise _MathProjectionRejected(
                        "layout_environment_crosses_brace_group",
                        self._global(cursor),
                    )
                return "".join(output), cursor + 1

            if character == "{":
                nested, cursor = self._segment(
                    cursor + 1,
                    closing_brace_required=True,
                    opening_brace_offset=cursor,
                    text_mode=text_mode,
                )
                output.extend(("{", nested, "}"))
                continue

            if character == "&" and self.environments:
                if text_mode:
                    raise _MathProjectionRejected(
                        "ambiguous_alignment_tab_inside_text_style_wrapper",
                        self._global(cursor),
                    )
                self.counters.layout_alignment_tabs_removed += 1
                cursor += 1
                continue

            if character != "\\":
                if text_mode and character == "-":
                    output.append(_FULLWIDTH_HYPHEN_MINUS)
                    self.counters.text_mode_hyphens_guarded += 1
                else:
                    output.append(character)
                cursor += 1
                continue

            if cursor + 1 >= len(self.body):
                raise _MathProjectionRejected(
                    "dangling_backslash_in_math", self._global(cursor)
                )
            following = self.body[cursor + 1]
            if following == "\\":
                if self.environments:
                    if text_mode:
                        raise _MathProjectionRejected(
                            "ambiguous_row_break_inside_text_style_wrapper",
                            self._global(cursor),
                        )
                    cursor = self._layout_row_break(cursor)
                    continue
                output.append("\\\\")
                cursor += 2
                continue
            if not (following.isascii() and (following.isalpha() or following == "@")):
                # A TeX control symbol consumes exactly one following byte;
                # escaped braces therefore never affect structural balance.
                if text_mode and following == "_":
                    output.append(_FULLWIDTH_LOW_LINE)
                    self.counters.text_mode_underscores_guarded += 1
                elif text_mode and following == "-":
                    raise _MathProjectionRejected(
                        "discretionary_hyphen_inside_text_style_wrapper",
                        self._global(cursor),
                    )
                else:
                    output.append(self.body[cursor : cursor + 2])
                cursor += 2
                continue

            end = cursor + 2
            while end < len(self.body) and self.body[end].isascii() and (
                self.body[end].isalpha() or self.body[end] == "@"
            ):
                end += 1
            command = self.body[cursor + 1 : end]
            operator_glyph = MATH_VISIBLE_OPERATOR_COMMANDS.get(command)
            if operator_glyph is not None:
                output.append(operator_glyph)
                self.counters.operator_commands_projected += 1
                _increment(self.counters.operator_command_counts, command)
                cursor = end
                continue
            if command in MATH_VISIBLE_STYLE_WRAPPERS:
                argument_opening = self._skip_whitespace(end)
                if (
                    argument_opening >= len(self.body)
                    or self.body[argument_opening] != "{"
                ):
                    raise _MathProjectionRejected(
                        f"style_wrapper_missing_braced_argument:{command}",
                        self._global(cursor),
                    )
                argument, cursor = self._segment(
                    argument_opening + 1,
                    closing_brace_required=True,
                    opening_brace_offset=argument_opening,
                    text_mode=text_mode or command in _TEXT_MODE_STYLE_WRAPPERS,
                )
                if _ends_with_control_word("".join(output)) and (
                    _starts_with_control_word_letter(argument)
                ):
                    # Empty braces delimit a TeX control word but have no
                    # visible character.  This preserves the token boundary
                    # that the removed wrapper command supplied.
                    output.append("{}")
                    self.counters.control_word_boundaries_inserted += 1
                output.append(argument)
                if (
                    _ends_with_control_word("".join(output))
                    and cursor < len(self.body)
                    and _starts_with_control_word_letter(self.body[cursor])
                ):
                    # The wrapper's removed closing brace may also have been
                    # the only boundary before a following literal letter.
                    output.append("{}")
                    self.counters.control_word_boundaries_inserted += 1
                self.counters.style_wrappers_removed += 1
                _increment(self.counters.style_wrapper_counts, command)
                continue
            if command == "begin":
                if text_mode:
                    raise _MathProjectionRejected(
                        "layout_environment_inside_text_style_wrapper",
                        self._global(cursor),
                    )
                cursor = self._begin_environment(end, cursor)
                continue
            if command == "end":
                if text_mode:
                    raise _MathProjectionRejected(
                        "layout_environment_inside_text_style_wrapper",
                        self._global(cursor),
                    )
                cursor = self._end_environment(end)
                continue

            output.append(self.body[cursor:end])
            cursor = end

        if closing_brace_required:
            assert opening_brace_offset is not None
            raise _MathProjectionRejected(
                "unbalanced_math_brace", self._global(opening_brace_offset)
            )
        if self.environments:
            name, opening = self.environments[-1]
            raise _MathProjectionRejected(
                f"unclosed_math_layout_environment:{name}", opening
            )
        return "".join(output), cursor

    def project(self) -> str:
        projected, cursor = self._segment(
            0,
            closing_brace_required=False,
            opening_brace_offset=None,
            text_mode=False,
        )
        assert cursor == len(self.body)
        return projected


def _inline_code_end(value: str, offset: int) -> int | None:
    fence_end = offset
    while fence_end < len(value) and value[fence_end] == "`":
        fence_end += 1
    fence = value[offset:fence_end]
    closing = value.find(fence, fence_end)
    if closing < 0:
        return None
    return closing + len(fence)


def _find_math_close(
    value: str,
    body_start: int,
    *,
    opener: str,
    closer: str,
    opaque_start_to_end: Mapping[int, int],
) -> int:
    cursor = body_start
    while cursor < len(value):
        if cursor in opaque_start_to_end:
            raise _MathProjectionRejected(
                "math_region_crosses_fenced_code_block", cursor
            )
        if closer in {"$", "$$"}:
            if value.startswith("$$", cursor) and not _is_escaped(value, cursor):
                if closer == "$$":
                    return cursor
                raise _MathProjectionRejected(
                    "mismatched_math_delimiter", cursor
                )
            if (
                closer == "$"
                and value[cursor] == "$"
                and not _is_escaped(value, cursor)
            ):
                return cursor
        elif value.startswith(closer, cursor) and not _is_escaped(value, cursor):
            return cursor

        # TeX math delimiters cannot be nested or mixed.  Spot a foreign
        # opener/closer early instead of letting a later expected closer
        # produce a plausible but false span.
        if not _is_escaped(value, cursor):
            token: str | None = None
            for candidate in ("$$", "$", r"\(", r"\)", r"\[", r"\]"):
                if value.startswith(candidate, cursor):
                    token = candidate
                    break
            if token is not None and token != closer:
                reason = (
                    "nested_math_delimiter"
                    if token in {"$", "$$", r"\(", r"\["}
                    else "mismatched_math_delimiter"
                )
                raise _MathProjectionRejected(reason, cursor)
        cursor += 1
    raise _MathProjectionRejected("unclosed_math_delimiter", body_start - len(opener))


def _project_math_regions(
    guarded_markdown: str,
    opaque_ranges: Sequence[tuple[int, int]],
    counters: _MathProjectionCounters,
) -> str:
    opaque_start_to_end = {start: end for start, end in opaque_ranges}
    output: list[str] = []
    cursor = 0
    while cursor < len(guarded_markdown):
        opaque_end = opaque_start_to_end.get(cursor)
        if opaque_end is not None:
            output.append(guarded_markdown[cursor:opaque_end])
            cursor = opaque_end
            continue

        if guarded_markdown[cursor] == "`":
            inline_end = _inline_code_end(guarded_markdown, cursor)
            if inline_end is not None:
                output.append(guarded_markdown[cursor:inline_end])
                cursor = inline_end
                continue

        opener: str | None = None
        closer: str | None = None
        if guarded_markdown.startswith("$$", cursor) and not _is_escaped(
            guarded_markdown, cursor
        ):
            opener = closer = "$$"
        elif (
            guarded_markdown[cursor] == "$"
            and not _is_escaped(guarded_markdown, cursor)
        ):
            opener = closer = "$"
        elif guarded_markdown.startswith(r"\(", cursor) and not _is_escaped(
            guarded_markdown, cursor
        ):
            opener, closer = r"\(", r"\)"
        elif guarded_markdown.startswith(r"\[", cursor) and not _is_escaped(
            guarded_markdown, cursor
        ):
            opener, closer = r"\[", r"\]"
        elif (
            guarded_markdown.startswith(r"\)", cursor)
            or guarded_markdown.startswith(r"\]", cursor)
        ) and not _is_escaped(guarded_markdown, cursor):
            raise _MathProjectionRejected(
                "math_closing_delimiter_without_opener", cursor
            )

        if opener is None or closer is None:
            output.append(guarded_markdown[cursor])
            cursor += 1
            continue

        body_start = cursor + len(opener)
        body_end = _find_math_close(
            guarded_markdown,
            body_start,
            opener=opener,
            closer=closer,
            opaque_start_to_end=opaque_start_to_end,
        )
        body = guarded_markdown[body_start:body_end]
        counters.math_regions_seen += 1
        _increment(counters.math_delimiter_counts, opener)
        projected_body = _MathBodyProjector(
            body,
            source_offset=body_start,
            counters=counters,
        ).project()
        if projected_body != body:
            counters.math_regions_projected += 1
        output.extend((opener, projected_body, closer))
        cursor = body_end + len(closer)
    return "".join(output)


def project_fenced_code_dollar_guards(
    frozen_source_markdown: str,
) -> MathVisibleFlowProjectionResult:
    """Protect fenced-code ``$`` from premature Markdown math parsing.

    Only code-body dollars are changed, to U+FF04.  NFKC restores them to
    ASCII dollars in the final exact stream.  Fences must pair strictly; any
    incomplete block returns the original frozen Markdown with ``rejected``
    provenance.
    """

    counters = _MathProjectionCounters()
    try:
        guarded, _ = _guard_fenced_code_dollars(frozen_source_markdown, counters)
    except _MathProjectionRejected as rejection:
        return _rejected_source_projection(
            frozen_source_markdown, counters, rejection
        )
    return _result_for_source_projection(frozen_source_markdown, guarded, counters)


def project_math_visible_flow(
    frozen_source_markdown: str,
) -> MathVisibleFlowProjectionResult:
    r"""Build a strict source-only verifier view of frozen Markdown.

    Supported style commands are transparent one-braced-argument wrappers,
    but only inside ``$...$``, ``$$...$$``, ``\(...\)``, or ``\[...\]``.
    Removed wrappers retain zero-visible ``{}`` delimiters where needed to
    keep adjacent TeX control-word tokens distinct.  A small static allow-list
    of context-independent operator commands is mapped to printed Unicode.
    Supported layout environments are removed only when correctly nested;
    their alignment tabs and row breaks are layout-only as well.  The source
    Markdown field is never edited, and a malformed construct rolls back the
    complete projected view.
    """

    counters = _MathProjectionCounters()
    try:
        guarded, opaque_ranges = _guard_fenced_code_dollars(
            frozen_source_markdown, counters
        )
        projected = _project_math_regions(guarded, opaque_ranges, counters)
    except _MathProjectionRejected as rejection:
        return _rejected_source_projection(
            frozen_source_markdown, counters, rejection
        )
    return _result_for_source_projection(
        frozen_source_markdown, projected, counters
    )


# The longer name makes the combined source-only contract explicit to builder
# callers; the math-specific name remains convenient for focused tests.
project_source_verifier_visible_flow = project_math_visible_flow


@dataclass(frozen=True, slots=True)
class FolioProjectionPolicy:
    """Conservative geometry thresholds for a page folio.

    ``bottom_margin_start_ratio`` uses pdfplumber's top-origin coordinates.
    Roman numerals are disabled by default and, when enabled, must be a
    canonical numeral no larger than ``max_roman_value``.
    """

    # Several journal classes place the folio at the bottom of the text block,
    # well above the physical paper edge (Springer is around 0.76 on A4).
    bottom_margin_start_ratio: float = 0.74
    center_tolerance_ratio: float = 0.01
    center_tolerance_points: float = 2.0
    minimum_body_gap_points: float = 6.0
    same_line_tolerance_points: float = 3.0
    allow_roman: bool = False
    max_roman_value: int = 128

    def __post_init__(self) -> None:
        if not 0.5 <= self.bottom_margin_start_ratio < 1.0:
            raise ValueError("bottom_margin_start_ratio must be in [0.5, 1.0)")
        if self.center_tolerance_ratio < 0:
            raise ValueError("center_tolerance_ratio must be non-negative")
        if self.center_tolerance_points < 0:
            raise ValueError("center_tolerance_points must be non-negative")
        if self.minimum_body_gap_points < 0:
            raise ValueError("minimum_body_gap_points must be non-negative")
        if self.same_line_tolerance_points < 0:
            raise ValueError("same_line_tolerance_points must be non-negative")
        if self.max_roman_value < 1:
            raise ValueError("max_roman_value must be positive")


@dataclass(frozen=True, slots=True)
class TextBox:
    """One PDF text word with top-origin page coordinates."""

    text: str
    x0: float
    top: float
    x1: float
    bottom: float

    def __post_init__(self) -> None:
        coordinates = (self.x0, self.top, self.x1, self.bottom)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("text-box coordinates must be finite")
        if self.x1 <= self.x0 or self.bottom <= self.top:
            raise ValueError("text box must have positive width and height")
        if not self.text or not self.text.strip():
            raise ValueError("text box must contain visible text")

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x0, self.top, self.x1, self.bottom)


@dataclass(frozen=True, slots=True)
class VerifierProjectionResult:
    """PDF verifier text after a fail-closed layout-only projection.

    ``source_markdown`` is returned byte-for-byte unchanged as an explicit
    audit invariant.  ``verifier_text`` and ``verifier_stream`` are PDF-side
    observations and must not be used as ground truth.
    """

    source_markdown: str
    verifier_text: str
    verifier_stream: str
    provenance: Mapping[str, Any]

    @property
    def projection_applied(self) -> bool:
        return bool(self.provenance.get("projection_applied", False))


def default_verifier_stream(value: str) -> str:
    """Whitespace-free PDF stream without changing punctuation or sentinels.

    This mirrors the PDF half of the existing exact character stream closely
    enough for standalone use.  A builder should pass its own canonical stream
    function when it has additional established behaviour.
    """

    value = unicodedata.normalize("NFKC", value).replace("\u00ad", "")
    return re.sub(r"\s+", "", value)


def _number_kind(text: str, policy: FolioProjectionPolicy) -> str | None:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if re.fullmatch(r"[0-9]+", normalized):
        return "arabic"
    if not policy.allow_roman or not re.fullmatch(r"[ivxlcdmIVXLCDM]+", normalized):
        return None
    value = _roman_value(normalized)
    if value is None or value > policy.max_roman_value:
        return None
    canonical = _int_to_roman(value)
    if normalized not in {canonical, canonical.lower()}:
        return None
    return "roman"


def _roman_value(value: str) -> int | None:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(value.upper()):
        current = values.get(character)
        if current is None:
            return None
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if total > 0 else None


def _int_to_roman(value: int) -> str:
    pieces: list[str] = []
    remainder = value
    for number, symbol in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        count, remainder = divmod(remainder, number)
        pieces.append(symbol * count)
    return "".join(pieces)


def _coerce_text_box(item: Mapping[str, Any] | TextBox) -> TextBox | None:
    if isinstance(item, TextBox):
        return item
    text = unicodedata.normalize("NFKC", str(item.get("text", ""))).strip()
    if not text:
        return None
    try:
        return TextBox(
            text=text,
            x0=float(item["x0"]),
            top=float(item["top"]),
            x1=float(item["x1"]),
            bottom=float(item["bottom"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid PDF text box: {item!r}") from exc


def _chars_to_words(
    characters: Sequence[Mapping[str, Any]],
    *,
    same_line_tolerance: float,
) -> tuple[TextBox, ...]:
    """Build conservative word boxes for minimal synthetic/page char inputs."""

    glyphs = [box for item in characters if (box := _coerce_text_box(item)) is not None]
    if not glyphs:
        return ()
    lines: list[list[TextBox]] = []
    for glyph in sorted(glyphs, key=lambda item: (item.top, item.x0)):
        target: list[TextBox] | None = None
        for line in reversed(lines[-8:]):
            line_top = statistics.median(item.top for item in line)
            if abs(glyph.top - line_top) <= same_line_tolerance:
                target = line
                break
            if line_top < glyph.top - same_line_tolerance:
                break
        if target is None:
            target = []
            lines.append(target)
        target.append(glyph)

    words: list[TextBox] = []
    for line in lines:
        current: list[TextBox] = []

        def flush() -> None:
            if not current:
                return
            text = "".join(item.text for item in current).strip()
            if text:
                words.append(
                    TextBox(
                        text=text,
                        x0=min(item.x0 for item in current),
                        top=min(item.top for item in current),
                        x1=max(item.x1 for item in current),
                        bottom=max(item.bottom for item in current),
                    )
                )
            current.clear()

        previous: TextBox | None = None
        widths = [item.x1 - item.x0 for item in line if not item.text.isspace()]
        typical_width = statistics.median(widths) if widths else 2.0
        for glyph in sorted(line, key=lambda item: item.x0):
            gap = glyph.x0 - previous.x1 if previous is not None else 0.0
            if glyph.text.isspace() or (
                previous is not None and gap > max(2.0, typical_width * 0.8)
            ):
                flush()
                previous = glyph
                continue
            current.append(glyph)
            previous = glyph
        flush()
    return tuple(words)


def _page_words(page: Any, policy: FolioProjectionPolicy) -> tuple[TextBox, ...]:
    extractor = getattr(page, "extract_words", None)
    if callable(extractor):
        extracted = extractor(
            x_tolerance=1.0,
            y_tolerance=3.0,
            keep_blank_chars=False,
            use_text_flow=False,
        )
        return tuple(
            box for item in extracted if (box := _coerce_text_box(item)) is not None
        )
    characters = getattr(page, "chars", None)
    if characters is None:
        raise ValueError("page must expose extract_words() or chars")
    return _chars_to_words(
        characters,
        same_line_tolerance=policy.same_line_tolerance_points,
    )


def _same_visual_line(left: TextBox, right: TextBox, tolerance: float) -> bool:
    overlap = max(0.0, min(left.bottom, right.bottom) - max(left.top, right.top))
    if overlap >= min(left.height, right.height) * 0.45:
        return True
    return abs(left.center_y - right.center_y) <= tolerance


def _provenance(
    *,
    source_markdown: str,
    status: str,
    reason: str,
    page_width: float,
    page_height: float,
    projection_applied: bool,
    bottom_folio_like_count: int,
    candidate: TextBox | None = None,
    number_kind: str | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": PROJECTION_VERSION,
        "status": status,
        "reason": reason,
        "projection_applied": projection_applied,
        "ground_truth_changed": False,
        "pdf_text_used_for_ground_truth": False,
        "source_markdown_sha256": hashlib.sha256(source_markdown.encode("utf-8")).hexdigest(),
        "page_width": page_width,
        "page_height": page_height,
        "bottom_folio_like_count": bottom_folio_like_count,
    }
    if candidate is not None:
        result["removed_folio" if projection_applied else "folio_candidate"] = {
            "text": candidate.text,
            "kind": number_kind,
            "bbox": list(candidate.bbox),
        }
    if geometry is not None:
        result["geometry"] = dict(geometry)
    return result


def _horizontal_center_geometry(
    candidate: TextBox,
    other_words: Sequence[TextBox],
    *,
    page_width: float,
    tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    """Check physical-page and conservative text-block centring."""

    page_center = page_width / 2.0
    references: list[tuple[str, float]] = [("physical_page", page_center)]
    body_words = [word for word in other_words if word.bottom <= candidate.top]
    if len(body_words) >= 4:
        body_left = min(word.x0 for word in body_words)
        body_right = max(word.x1 for word in body_words)
        if body_right - body_left >= page_width * 0.25:
            references.append(("body_text_envelope", (body_left + body_right) / 2.0))
    ranked = sorted(
        (
            (abs(candidate.center_x - reference_x), name, reference_x)
            for name, reference_x in references
        ),
        key=lambda item: (item[0], item[1]),
    )
    error, name, reference_x = ranked[0]
    return error <= tolerance, {
        "horizontal_center_reference": name,
        "horizontal_center_reference_x": reference_x,
        "candidate_center_x": candidate.center_x,
        "horizontal_center_error_points": error,
        "horizontal_center_tolerance_points": tolerance,
    }


def project_bottom_margin_folio(
    frozen_source_markdown: str,
    pdf_verifier_text: str,
    *,
    page: Any | None = None,
    layout_words: Sequence[Mapping[str, Any] | TextBox] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
    policy: FolioProjectionPolicy | None = None,
    stream_projector: Callable[[str], str] | None = None,
) -> VerifierProjectionResult:
    """Remove one strictly identified bottom-centred folio from PDF verifier text.

    Supply either a pdfplumber-like ``page`` or ``layout_words`` plus explicit
    page dimensions.  The caller's source Markdown is treated as frozen and is
    returned unchanged.  A rejected or ambiguous projection returns the
    original PDF verifier text/stream with a provenance reason.
    """

    selected_policy = policy or FolioProjectionPolicy()
    projector = stream_projector or default_verifier_stream

    if page is not None:
        inferred_width = float(getattr(page, "width"))
        inferred_height = float(getattr(page, "height"))
        width = inferred_width if page_width is None else float(page_width)
        height = inferred_height if page_height is None else float(page_height)
        words = _page_words(page, selected_policy) if layout_words is None else tuple(
            box
            for item in layout_words
            if (box := _coerce_text_box(item)) is not None
        )
    else:
        if layout_words is None or page_width is None or page_height is None:
            raise ValueError(
                "provide page, or layout_words together with page_width/page_height"
            )
        width = float(page_width)
        height = float(page_height)
        words = tuple(
            box
            for item in layout_words
            if (box := _coerce_text_box(item)) is not None
        )
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("page dimensions must be finite and positive")

    def unchanged(
        reason: str,
        *,
        count: int,
        candidate: TextBox | None = None,
    ) -> VerifierProjectionResult:
        return VerifierProjectionResult(
            source_markdown=frozen_source_markdown,
            verifier_text=pdf_verifier_text,
            verifier_stream=projector(pdf_verifier_text),
            provenance=_provenance(
                source_markdown=frozen_source_markdown,
                status="unchanged",
                reason=reason,
                page_width=width,
                page_height=height,
                projection_applied=False,
                bottom_folio_like_count=count,
                candidate=candidate,
                number_kind=(
                    _number_kind(candidate.text, selected_policy)
                    if candidate is not None
                    else None
                ),
            ),
        )

    margin_top = height * selected_policy.bottom_margin_start_ratio
    folio_like = [
        word
        for word in words
        if word.top >= margin_top and _number_kind(word.text, selected_policy) is not None
    ]
    if not folio_like:
        centred_body_numbers = [
            word
            for word in words
            if word.top < margin_top
            and _number_kind(word.text, selected_policy) is not None
            and abs(word.center_x - width / 2.0)
            <= max(
                selected_policy.center_tolerance_points,
                width * selected_policy.center_tolerance_ratio,
            )
        ]
        reason = (
            "folio_like_text_in_body_region"
            if centred_body_numbers
            else "no_bottom_folio_candidate"
        )
        return unchanged(reason, count=0)
    if len(folio_like) != 1:
        return unchanged("multiple_bottom_folio_candidates", count=len(folio_like))

    candidate = folio_like[0]
    center_tolerance = max(
        selected_policy.center_tolerance_points,
        width * selected_policy.center_tolerance_ratio,
    )
    other_words = [word for word in words if word is not candidate]
    centered, center_geometry = _horizontal_center_geometry(
        candidate,
        other_words,
        page_width=width,
        tolerance=center_tolerance,
    )
    if not centered:
        result = unchanged(
            "bottom_folio_not_horizontally_centered", count=1, candidate=candidate
        )
        provenance = dict(result.provenance)
        provenance["geometry"] = center_geometry
        return VerifierProjectionResult(
            source_markdown=result.source_markdown,
            verifier_text=result.verifier_text,
            verifier_stream=result.verifier_stream,
            provenance=provenance,
        )
    if candidate.x0 < 0 or candidate.x1 > width or candidate.top < 0 or candidate.bottom > height:
        return unchanged("bottom_folio_bbox_outside_page", count=1, candidate=candidate)

    if any(
        _same_visual_line(candidate, other, selected_policy.same_line_tolerance_points)
        for other in other_words
    ):
        return unchanged("bottom_folio_not_isolated_on_line", count=1, candidate=candidate)
    if any(other.top >= margin_top for other in other_words):
        return unchanged("bottom_margin_contains_other_text", count=1, candidate=candidate)
    if any(other.center_y > candidate.center_y for other in other_words):
        return unchanged("text_below_bottom_folio", count=1, candidate=candidate)

    above = [other.bottom for other in other_words if other.bottom <= candidate.top]
    if above and candidate.top - max(above) < selected_policy.minimum_body_gap_points:
        return unchanged("bottom_folio_too_close_to_body", count=1, candidate=candidate)

    terminal = re.search(r"(?<!\S)" + re.escape(candidate.text) + r"\s*$", pdf_verifier_text)
    if terminal is None:
        return unchanged("bottom_folio_not_terminal_in_verifier_text", count=1, candidate=candidate)
    projected_text = pdf_verifier_text[: terminal.start()].rstrip()
    number_kind = _number_kind(candidate.text, selected_policy)
    return VerifierProjectionResult(
        source_markdown=frozen_source_markdown,
        verifier_text=projected_text,
        verifier_stream=projector(projected_text),
        provenance=_provenance(
            source_markdown=frozen_source_markdown,
            status="projected",
            reason="unique_bottom_centered_folio_removed",
            page_width=width,
            page_height=height,
            projection_applied=True,
            bottom_folio_like_count=1,
            candidate=candidate,
            number_kind=number_kind,
            geometry=center_geometry,
        ),
    )


__all__ = [
    "MATH_LAYOUT_ENVIRONMENTS",
    "MATH_VISIBLE_FLOW_PROJECTION_VERSION",
    "MATH_VISIBLE_OPERATOR_COMMANDS",
    "MATH_VISIBLE_STYLE_WRAPPERS",
    "PROJECTION_VERSION",
    "FolioProjectionPolicy",
    "MathVisibleFlowProjectionResult",
    "TextBox",
    "VerifierProjectionResult",
    "default_verifier_stream",
    "project_bottom_margin_folio",
    "project_fenced_code_dollar_guards",
    "project_math_visible_flow",
    "project_source_verifier_visible_flow",
]
