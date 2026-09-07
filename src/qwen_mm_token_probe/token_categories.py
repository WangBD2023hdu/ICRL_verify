"""Three disjoint roles for scored document-response tokens, not correctness labels."""

from __future__ import annotations

import re
from collections.abc import Sequence
from statistics import fmean
from typing import Any

TOKEN_CATEGORIES = ("formatting", "body", "mutation")
CATEGORY_LABELS = {"formatting": "格式", "body": "正文", "mutation": "变异"}
CATEGORY_RULES = {
    "unit": "actual_response_token_id",
    "priority": ["mutation_word_overlap", "pure_formatting", "body"],
    "mutation_scope": "whole_aligned_mutation_word_including_wrong_readbacks",
    "mixed_token_policy": "mutation_if_overlapping_else_body",
    "mean_weighting": "one_vote_per_token_not_per_character_or_per_sample",
    "probability_gate": "none_all_response_tokens",
    "teacher_lower_rule": "p_teacher < p_original",
    "correctness_filter": "none_correct_and_incorrect_tokens_included",
}

_MATH_RE = re.compile(
    r"(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$"
    r"|(?<![\\$])\$(?!\$)[^\n]*?(?<!\\)\$"
    r"|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]"
)
_FENCE_RE = re.compile(r"(?m)^[ \t]{0,3}(?:`{3,}|~{3,})[^\n]*$")
_HTML_RE = re.compile(
    r"<!--[\s\S]*?-->|</?[A-Za-z][\w:-]*(?:\s+(?:[^<>\"']|\"[^\"]*\"|'[^']*')*)?\s*/?>"
)
_BLOCK_MARKER_RE = re.compile(
    r"(?m)^[ \t]{0,3}(?:#{1,6}(?!#)|(?:>[ \t]*)+|(?:[-+*]|\d+[.)])(?=[ \t]))"
)
_RULE_RE = re.compile(r"(?m)^[ \t]*(?:[-*_][ \t]*){3,}$|^[ \t]*={3,}[ \t]*$")
_EMPHASIS_RE = re.compile(
    r"(?<![\\\w])(?P<mark>\*{1,3}|_{1,3}|~~)(?=\S)(?P<body>.+?)(?<=\S)(?P=mark)(?!\w)"
)
_INLINE_CODE_RE = re.compile(r"(?<!`)(`{1,2})(?!`)([^\n]*?)(?<!`)\1(?!`)")
_LINK_RE = re.compile(r"!?\[([^\]\n]*)\]\(([^)\n]*)\)")


def _format_mask(text: str) -> list[bool]:
    """Recognize common document Markdown/HTML in context across BPE boundaries.

    Math and inline-code contents are content, not Markdown punctuation. This
    intentionally does not alter the independent GT correctness normalizer.
    """
    mask = [character.isspace() for character in text]
    protected = [False] * len(text)

    def mark(start: int, end: int) -> None:
        for index in range(start, end):
            if not protected[index]:
                mask[index] = True

    for match in _FENCE_RE.finditer(text):
        mark(*match.span())
    for match in _MATH_RE.finditer(text):
        for index in range(*match.span()):
            if not mask[index]:
                protected[index] = True
    for match in _INLINE_CODE_RE.finditer(text):
        if any(protected[match.start() : match.end()]):
            continue
        mark(match.start(), match.start(2))
        mark(match.end(2), match.end())
        for index in range(*match.span(2)):
            protected[index] = True
    for pattern in (_HTML_RE, _BLOCK_MARKER_RE, _RULE_RE):
        for match in pattern.finditer(text):
            mark(*match.span())
    for match in _EMPHASIS_RE.finditer(text):
        mark(match.start(), match.start("body"))
        mark(match.end("body"), match.end())
    for match in _LINK_RE.finditer(text):
        mark(match.start(), match.start(1))
        mark(match.end(1), match.start(2))
        mark(match.end(2), match.end())
    # Pipe-table separators, including the alignment rule (--- / :---:).
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.count("|") >= 2:
            rule = re.fullmatch(r"[\s|:\-]+", line) is not None
            for index, character in enumerate(line, offset):
                if rule or character == "|":
                    mark(index, index + 1)
        offset += len(line)
    return mask


def annotate_token_categories(
    rows: Sequence[dict[str, Any]],
    *,
    response_text: str,
) -> None:
    """Annotate existing score rows in place; never split or rescore a token."""
    pieces = [str(row.get("raw_token", row.get("token", ""))) for row in rows]
    joined = "".join(pieces)
    # BPE pieces are the source of token boundaries. Record decode discrepancies
    # explicitly instead of indexing a different surface with these offsets.
    mask = _format_mask(joined)
    offset = 0
    for row, piece in zip(rows, pieces):
        formatting_count = sum(mask[offset : offset + len(piece)])
        content_count = len(piece) - formatting_count
        mutation_ids = str(row.get("mutation_ids", "")).strip(", \t\n")
        category = (
            "mutation"
            if mutation_ids
            else "formatting"
            if not content_count
            else "body"
        )
        row.update(
            token_category=category,
            format_char_count=formatting_count,
            body_char_count=content_count,
            mixed_format_content=bool(formatting_count and content_count),
            category_surface_matches_response=joined == response_text,
        )
        offset += len(piece)


def summarize_token_categories(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Micro-average actual token probabilities; empty groups have null means."""
    summaries = []
    for category in TOKEN_CATEGORIES:
        group = [row for row in rows if row["token_category"] == category]
        original = [float(row["p_original"]) for row in group]
        teacher = [float(row["p_teacher"]) for row in group]
        lower = sum(pt < po for po, pt in zip(original, teacher))
        summaries.append(
            {
                "token_category": category,
                "token_count": len(group),
                "token_fraction": len(group) / len(rows) if rows else None,
                "mean_p_original": fmean(original) if group else None,
                "mean_p_teacher": fmean(teacher) if group else None,
                "mean_delta_p_teacher_minus_original": (
                    fmean(pt - po for po, pt in zip(original, teacher))
                    if group
                    else None
                ),
                "mean_delta_logp_teacher_minus_original": (
                    fmean(
                        float(row["delta_logp_teacher_minus_original"]) for row in group
                    )
                    if group
                    else None
                ),
                "teacher_lower_probability_count": lower,
                "teacher_lower_probability_rate": lower / len(group) if group else None,
                "mixed_format_content_token_count": sum(
                    bool(row["mixed_format_content"]) for row in group
                ),
            }
        )
    return summaries
