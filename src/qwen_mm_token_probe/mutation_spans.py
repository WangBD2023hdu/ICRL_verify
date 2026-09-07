"""Resolve edited-word spans in Markdown ground truth text."""

from __future__ import annotations

import re
from collections.abc import Sequence
from numbers import Integral
from typing import Any

_WORD_CHARACTER_RE = re.compile(r"\w")


def resolve_mutation_spans(
    ground_truth: str,
    changes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return copied changes with safe Markdown spans attached.

    A supplied span is trusted only when it is an in-bounds, non-empty slice
    whose text is exactly the change's edited word (``ocr_ans``).  Changes
    without such a span are resolved from a unique, case-sensitive whole-word
    occurrence in ``ground_truth``.  Repeated or missing words stay unresolved
    rather than being assigned an arbitrary occurrence.
    """

    resolved_changes: list[dict[str, Any]] = []
    for change in changes:
        resolved = dict(change)
        edited_word = change.get("ocr_ans")
        matches = _whole_word_matches(ground_truth, edited_word)
        supplied_span = _valid_span_for_text(
            ground_truth,
            change.get("markdown_span"),
            edited_word,
        )

        if supplied_span is not None:
            resolved["markdown_span"] = list(supplied_span)
            resolved["span_resolution"] = "provided"
        elif len(matches) == 1:
            resolved["markdown_span"] = list(matches[0])
            resolved["span_resolution"] = "unique_gt_match"
        else:
            resolved["markdown_span"] = []
            resolved["span_resolution"] = (
                "ambiguous_gt_match" if len(matches) > 1 else "not_found"
            )
        resolved["span_match_count"] = len(matches)
        resolved_changes.append(resolved)
    return resolved_changes


def _whole_word_matches(
    ground_truth: str,
    edited_word: Any,
) -> list[tuple[int, int]]:
    r"""Find literal whole-word occurrences using only needed ``\w`` edges."""

    if not isinstance(edited_word, str) or not edited_word:
        return []

    prefix = r"(?<!\w)" if _is_word_character(edited_word[0]) else ""
    suffix = r"(?!\w)" if _is_word_character(edited_word[-1]) else ""
    pattern = re.compile(f"{prefix}{re.escape(edited_word)}{suffix}")
    return [(match.start(), match.end()) for match in pattern.finditer(ground_truth)]


def _is_word_character(character: str) -> bool:
    return _WORD_CHARACTER_RE.fullmatch(character) is not None


def _valid_span_for_text(
    ground_truth: str,
    span: Any,
    edited_word: Any,
) -> tuple[int, int] | None:
    if not isinstance(edited_word, str) or not edited_word:
        return None
    if isinstance(span, (str, bytes)) or not isinstance(span, Sequence):
        return None
    if len(span) != 2:
        return None
    start, end = span
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, Integral)
        or not isinstance(end, Integral)
    ):
        return None
    start_int, end_int = int(start), int(end)
    if not 0 <= start_int < end_int <= len(ground_truth):
        return None
    if ground_truth[start_int:end_int] != edited_word:
        return None
    return start_int, end_int


__all__ = ["resolve_mutation_spans"]
