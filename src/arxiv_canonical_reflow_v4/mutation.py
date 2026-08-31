"""Deterministic, source-derived confusable mutations for canonical V4 pages."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .core import CanonicalBlock, CanonicalPage

MUTATION_POLICY_VERSION = "canonical_reflow_confusable_v1_aligned"

# Frozen V1-compatible lower-case, one-codepoint substitutions.  Digits,
# Unicode homoglyphs, insertions, and deletions are intentionally excluded.
CONFUSABLES: dict[str, tuple[str, ...]] = {
    "a": ("o",),
    "c": ("e", "o"),
    "e": ("c",),
    "g": ("q",),
    "h": ("n",),
    "i": ("l",),
    "l": ("i",),
    "n": ("h",),
    "o": ("a", "c"),
    "q": ("g",),
    "s": ("z",),
    "u": ("v",),
    "v": ("u",),
    "z": ("s",),
}

WORD_RE = re.compile(r"[A-Za-z]{4,}")
_HIDDEN_MARKDOWN_RE = re.compile(
    r"```.*?```"
    r"|~~~.*?~~~"
    r"|<[^>]*>"
    r"|`[^`]*`"
    r"|\$\$.*?\$\$"
    r"|\$[^$\n]*\$"
    r"|https?://[^\s<>)]+"
    r"|&[A-Za-z]+;"
    r"|\]\([^)]*\)",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class RenderedWord:
    text: str
    column: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True, slots=True)
class PageMutation:
    original_word: str
    mutated_word: str
    from_char: str
    to_char: str
    char_index_in_word: int
    block_id: str
    node_id: str
    block_index: int
    markdown_start: int
    markdown_end: int
    block_markdown_start: int
    block_markdown_end: int
    block_latex_start: int
    block_latex_end: int
    block_verifier_start: int
    block_verifier_end: int
    rendered_word_index: int
    clean_bbox_points: tuple[float, float, float, float]
    source_files: tuple[str, ...]
    source_char_span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MutationValidation:
    passed: bool
    reason: str
    max_vertical_shift_points: float | None


def _stable_seed(seed: int, label: str) -> int:
    payload = f"{seed}:{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _visible_word_spans(markdown: str) -> tuple[tuple[str, int, int], ...]:
    hidden = [False] * len(markdown)
    for match in _HIDDEN_MARKDOWN_RE.finditer(markdown):
        hidden[match.start() : match.end()] = [True] * (match.end() - match.start())
    return tuple(
        (match.group(0), match.start(), match.end())
        for match in WORD_RE.finditer(markdown)
        if not any(hidden[match.start() : match.end()])
    )


def _literal_spans(value: str, word: str, *, latex: bool) -> list[tuple[int, int]]:
    prefix = r"(?<![A-Za-z\\])" if latex else r"(?<![A-Za-z])"
    pattern = re.compile(prefix + re.escape(word) + r"(?![A-Za-z])")
    return [(match.start(), match.end()) for match in pattern.finditer(value)]


def _block_offsets(page: CanonicalPage) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    markdown = page.markdown
    for block in page.blocks:
        position = markdown.find(block.markdown, cursor)
        if position < 0:
            raise ValueError(f"block markdown not found in page: {block.block_id}")
        offsets.append(position)
        cursor = position + len(block.markdown)
    return tuple(offsets)


def choose_page_mutations(
    page: CanonicalPage,
    rendered_words: Sequence[RenderedWord],
    *,
    seed: int,
    minimum: int = 3,
    maximum: int = 4,
    maximum_probability: float = 0.6,
) -> tuple[PageMutation, ...]:
    """Choose V1-compatible page-local mutations from paired source channels."""

    if minimum < 1 or maximum < minimum:
        raise ValueError("mutation bounds must satisfy 1 <= minimum <= maximum")
    if not 0.0 <= maximum_probability <= 1.0:
        raise ValueError("maximum_probability must be in [0, 1]")

    block_offsets = _block_offsets(page)
    visible_by_block = [_visible_word_spans(block.markdown) for block in page.blocks]
    visible_counts = Counter(word for spans in visible_by_block for word, _, _ in spans)
    rendered_indices: dict[str, list[int]] = {}
    for index, word in enumerate(rendered_words):
        rendered_indices.setdefault(word.text, []).append(index)
    page_vocabulary = set(rendered_indices)

    candidates: list[PageMutation] = []
    for block_index, (block, spans) in enumerate(zip(page.blocks, visible_by_block)):
        # Keep structural and high-risk material byte-for-byte stable.  Dense
        # prose provides enough candidates in normal pages, while excluding
        # headings/frontmatter/captions/tables/formulas avoids changing author
        # superscripts, table structure, numbering, or mathematical syntax.
        if block.kind != "paragraph":
            continue
        for word, markdown_start, markdown_end in spans:
            if visible_counts[word] != 1 or not any(char in CONFUSABLES for char in word):
                continue
            word_indices = rendered_indices.get(word, [])
            if len(word_indices) != 1:
                continue
            rendered_index = word_indices[0]
            rendered = rendered_words[rendered_index]
            same_line_follower = any(
                other.column == rendered.column
                and abs(other.y_min - rendered.y_min) <= 1.0
                and other.x_min > rendered.x_max
                for other in rendered_words
            )
            if not same_line_follower:
                continue
            latex_spans = _literal_spans(block.latex, word, latex=True)
            verifier_spans = _literal_spans(block.verifier_text, word, latex=False)
            if len(latex_spans) != 1 or len(verifier_spans) != 1:
                continue
            candidates.append(
                PageMutation(
                    original_word=word,
                    mutated_word="",
                    from_char="",
                    to_char="",
                    char_index_in_word=-1,
                    block_id=block.block_id,
                    node_id=block.node_id,
                    block_index=block_index,
                    markdown_start=block_offsets[block_index] + markdown_start,
                    markdown_end=block_offsets[block_index] + markdown_end,
                    block_markdown_start=markdown_start,
                    block_markdown_end=markdown_end,
                    block_latex_start=latex_spans[0][0],
                    block_latex_end=latex_spans[0][1],
                    block_verifier_start=verifier_spans[0][0],
                    block_verifier_end=verifier_spans[0][1],
                    rendered_word_index=rendered_index,
                    clean_bbox_points=(
                        rendered.x_min,
                        rendered.y_min,
                        rendered.x_max,
                        rendered.y_max,
                    ),
                    source_files=block.source_files,
                    source_char_span=block.source_char_span,
                )
            )

    rng = random.Random(_stable_seed(seed, page.page_id))
    rng.shuffle(candidates)
    requested = maximum if rng.random() < maximum_probability else minimum
    selected: list[PageMutation] = []
    selected_words: set[str] = set()
    selected_mutated_words: set[str] = set()
    selected_lines: set[tuple[int, int]] = set()
    for candidate in candidates:
        if candidate.original_word in selected_words:
            continue
        rendered = rendered_words[candidate.rendered_word_index]
        line_key = (rendered.column, round(rendered.y_min))
        if line_key in selected_lines:
            continue
        positions = [
            index
            for index, character in enumerate(candidate.original_word)
            if character in CONFUSABLES
        ]
        rng.shuffle(positions)
        completed: PageMutation | None = None
        for character_index in positions:
            from_char = candidate.original_word[character_index]
            targets = list(CONFUSABLES[from_char])
            rng.shuffle(targets)
            for to_char in targets:
                mutated_word = (
                    candidate.original_word[:character_index]
                    + to_char
                    + candidate.original_word[character_index + 1 :]
                )
                if (
                    mutated_word in page_vocabulary
                    or mutated_word in selected_mutated_words
                    or len(mutated_word) != len(candidate.original_word)
                ):
                    continue
                completed = replace(
                    candidate,
                    mutated_word=mutated_word,
                    from_char=from_char,
                    to_char=to_char,
                    char_index_in_word=character_index,
                )
                break
            if completed is not None:
                break
        if completed is None:
            continue
        selected.append(completed)
        selected_words.add(completed.original_word)
        selected_mutated_words.add(completed.mutated_word)
        selected_lines.add(line_key)
        if len(selected) >= requested:
            break
    if len(selected) < minimum:
        return ()
    return tuple(selected[:requested])


def _replace_spans(
    value: str,
    replacements: Sequence[tuple[int, int, str, str]],
) -> str:
    edited = value
    for start, end, expected, replacement in sorted(replacements, reverse=True):
        actual = edited[start:end]
        if actual != expected:
            raise ValueError(
                f"mutation span mismatch: expected={expected!r} actual={actual!r}"
            )
        edited = edited[:start] + replacement + edited[end:]
    return edited


def apply_page_mutations(
    page: CanonicalPage,
    mutations: Sequence[PageMutation],
    *,
    page_id: str,
) -> CanonicalPage:
    """Apply identical substitutions to Markdown, LaTeX, and verifier text."""

    by_block: dict[int, list[PageMutation]] = {}
    for mutation in mutations:
        by_block.setdefault(mutation.block_index, []).append(mutation)
    blocks: list[CanonicalBlock] = []
    for index, block in enumerate(page.blocks):
        block_mutations = by_block.get(index, [])
        if not block_mutations:
            blocks.append(block)
            continue
        if any(
            item.block_id != block.block_id or item.node_id != block.node_id
            for item in block_mutations
        ):
            raise ValueError(f"mutation block identity mismatch: {block.block_id}")
        markdown = _replace_spans(
            block.markdown,
            [
                (
                    item.block_markdown_start,
                    item.block_markdown_end,
                    item.original_word,
                    item.mutated_word,
                )
                for item in block_mutations
            ],
        )
        latex = _replace_spans(
            block.latex,
            [
                (
                    item.block_latex_start,
                    item.block_latex_end,
                    item.original_word,
                    item.mutated_word,
                )
                for item in block_mutations
            ],
        )
        verifier_text = _replace_spans(
            block.verifier_text,
            [
                (
                    item.block_verifier_start,
                    item.block_verifier_end,
                    item.original_word,
                    item.mutated_word,
                )
                for item in block_mutations
            ],
        )
        blocks.append(
            replace(
                block,
                markdown=markdown,
                latex=latex,
                verifier_text=verifier_text,
            )
        )
    return CanonicalPage(
        page_id=page_id,
        paper_id=page.paper_id,
        ordinal=page.ordinal,
        layout=page.layout,
        blocks=tuple(blocks),
    )


def validate_mutated_word_geometry(
    clean_words: Sequence[RenderedWord],
    edited_words: Sequence[RenderedWord],
    mutations: Sequence[PageMutation],
    *,
    max_vertical_shift_points: float = 1.25,
) -> MutationValidation:
    indexes = [mutation.rendered_word_index for mutation in mutations]
    if any(index < 0 for index in indexes):
        return MutationValidation(False, "negative_rendered_word_index", None)
    if len(set(indexes)) != len(indexes):
        return MutationValidation(False, "duplicate_rendered_word_index", None)
    for word in (*clean_words, *edited_words):
        coordinates = (word.x_min, word.y_min, word.x_max, word.y_max)
        if not all(math.isfinite(value) for value in coordinates):
            return MutationValidation(False, "non_finite_word_geometry", None)
        if word.x_min > word.x_max or word.y_min > word.y_max:
            return MutationValidation(False, "invalid_word_bbox", None)
    if len(clean_words) != len(edited_words):
        return MutationValidation(
            False,
            f"word_count_changed:{len(clean_words)}->{len(edited_words)}",
            None,
        )
    expected = [word.text for word in clean_words]
    for mutation in mutations:
        index = mutation.rendered_word_index
        if (
            len(mutation.original_word) != len(mutation.mutated_word)
            or sum(
                left != right
                for left, right in zip(
                    mutation.original_word,
                    mutation.mutated_word,
                )
            )
            != 1
            or mutation.from_char not in CONFUSABLES
            or mutation.to_char not in CONFUSABLES[mutation.from_char]
        ):
            return MutationValidation(False, "invalid_confusable_substitution", None)
        if index >= len(expected) or expected[index] != mutation.original_word:
            return MutationValidation(
                False,
                f"clean_word_index_mismatch:{mutation.original_word}",
                None,
            )
        expected[index] = mutation.mutated_word
    actual = [word.text for word in edited_words]
    if actual != expected:
        mismatch = next(
            (
                index
                for index, (expected_word, actual_word) in enumerate(zip(expected, actual))
                if expected_word != actual_word
            ),
            -1,
        )
        return MutationValidation(
            False,
            f"edited_word_sequence_mismatch:index={mismatch}",
            None,
        )
    if any(clean.column != edited.column for clean, edited in zip(clean_words, edited_words)):
        return MutationValidation(False, "column_assignment_changed", None)
    maximum = max(
        (
            abs(clean.y_min - edited.y_min)
            for clean, edited in zip(clean_words, edited_words)
        ),
        default=0.0,
    )
    if maximum > max_vertical_shift_points:
        return MutationValidation(
            False,
            f"line_reflow:max_vertical_shift={maximum:.3f}",
            maximum,
        )
    return MutationValidation(True, "passed", maximum)


def markdown_diff_count(clean: str, edited: str) -> int:
    if len(clean) != len(edited):
        return -1
    return sum(left != right for left, right in zip(clean, edited))


__all__ = [
    "CONFUSABLES",
    "MUTATION_POLICY_VERSION",
    "MutationValidation",
    "PageMutation",
    "RenderedWord",
    "apply_page_mutations",
    "choose_page_mutations",
    "markdown_diff_count",
    "validate_mutated_word_geometry",
]
