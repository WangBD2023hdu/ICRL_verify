"""Pair-first sampling from the user's 1012 observed letter substitutions."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace

from .core import CanonicalPage
from .mutation import (
    PageMutation,
    _block_offsets,
    _literal_spans,
    _stable_seed,
    _visible_word_spans,
)

POLICY_NAME = "empirical_1012"
POLICY_VERSION = "canonical_reflow_empirical_1012_pair_first_v1"

# Integer counts, not rounded percentages. The requested 0 -> e is excluded.
PAIR_COUNTS: dict[str, dict[str, int]] = {
    "e": {"c": 214, "o": 55, "a": 50, "f": 1},
    "o": {"c": 128, "e": 52, "a": 35},
    "c": {"e": 53, "o": 29, "a": 9},
    "a": {"e": 31, "c": 28, "o": 20, "n": 1},
    "n": {"m": 56, "h": 7, "r": 7, "v": 1},
    "m": {"n": 63},
    "u": {"v": 38, "a": 3, "o": 3, "y": 1},
    "v": {"u": 18, "w": 17},
    "r": {"n": 11, "t": 10, "i": 3, "c": 1},
    "w": {"v": 19, "u": 1},
    "i": {"l": 17, "t": 1},
    "t": {"i": 4, "r": 4},
    "l": {"i": 7},
    "p": {"q": 4},
    "b": {"d": 1, "q": 1, "r": 1},
    "d": {"b": 2, "o": 1},
    "h": {"n": 2},
    "f": {"l": 1},
    "s": {"c": 1},
}
PAIR_WEIGHTS = {
    (source, target): count
    for source, targets in PAIR_COUNTS.items()
    for target, count in targets.items()
}
TOTAL_WEIGHT = sum(PAIR_WEIGHTS.values())
POLICY_FINGERPRINT = hashlib.sha256(
    json.dumps(PAIR_COUNTS, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:12]


def choose_weighted_source_page_mutations(
    page: CanonicalPage,
    *,
    seed: int,
    minimum: int = 3,
    maximum: int = 4,
    maximum_probability: float = 0.6,
) -> tuple[PageMutation, ...]:
    """Draw a pair by its weight, then a matching source word and position.

    Pair weights are not multiplied by the number of occurrences in the page.
    A pair with no valid remaining word is removed and the other weights are
    renormalized. Compilation and page acceptance may further change the
    exported distribution; these are sampling targets, not corpus quotas.
    """
    if minimum < 1 or maximum < minimum:
        raise ValueError("mutation bounds must satisfy 1 <= minimum <= maximum")
    if not 0.0 <= maximum_probability <= 1.0:
        raise ValueError("maximum_probability must be in [0, 1]")

    offsets = _block_offsets(page)
    visible = [_visible_word_spans(block.markdown) for block in page.blocks]
    counts = Counter(word for spans in visible for word, _, _ in spans)
    candidates: list[PageMutation] = []
    by_char: dict[str, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for block_index, (block, spans) in enumerate(zip(page.blocks, visible)):
        if block.kind != "paragraph":
            continue
        for word, start, end in spans:
            if counts[word] != 1:
                continue
            positions: dict[str, list[int]] = defaultdict(list)
            for position, char in enumerate(word):
                if char in PAIR_COUNTS:
                    positions[char].append(position)
            if not positions:
                continue
            latex = _literal_spans(block.latex, word, latex=True)
            verifier = _literal_spans(block.verifier_text, word, latex=False)
            if len(latex) != 1 or len(verifier) != 1:
                continue
            index = len(candidates)
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
                    markdown_start=offsets[block_index] + start,
                    markdown_end=offsets[block_index] + end,
                    block_markdown_start=start,
                    block_markdown_end=end,
                    block_latex_start=latex[0][0],
                    block_latex_end=latex[0][1],
                    block_verifier_start=verifier[0][0],
                    block_verifier_end=verifier[0][1],
                    rendered_word_index=-1,
                    clean_bbox_points=(0.0, 0.0, 0.0, 0.0),
                    source_files=block.source_files,
                    source_char_span=block.source_char_span,
                )
            )
            for char, indexes in positions.items():
                by_char[char].append((index, tuple(indexes)))

    rng = random.Random(_stable_seed(seed, f"{POLICY_VERSION}:{page.page_id}"))
    requested = maximum if rng.random() < maximum_probability else minimum
    selected: list[PageMutation] = []
    selected_indexes: set[int] = set()
    selected_words: set[str] = set()
    available = [pair for pair in PAIR_WEIGHTS if pair[0] in by_char]
    while available and len(selected) < requested:
        pair = rng.choices(available, weights=[PAIR_WEIGHTS[p] for p in available])[0]
        source, target = pair
        options = [item for item in by_char[source] if item[0] not in selected_indexes]
        completed: PageMutation | None = None
        # Sampling without replacement here handles word collisions without
        # increasing the probability of pairs having many candidate words.
        while options and completed is None:
            option = rng.randrange(len(options))
            index, positions_tuple = options[option]
            options[option] = options[-1]
            options.pop()
            candidate = candidates[index]
            positions_list = list(positions_tuple)
            rng.shuffle(positions_list)
            for position in positions_list:
                word = candidate.original_word
                mutated = word[:position] + target + word[position + 1 :]
                if mutated in counts or mutated in selected_words:
                    continue
                completed = replace(
                    candidate,
                    mutated_word=mutated,
                    from_char=source,
                    to_char=target,
                    char_index_in_word=position,
                )
                selected_indexes.add(index)
                selected_words.add(mutated)
                break
        if completed is None:
            available.remove(pair)
        else:
            selected.append(completed)
    return tuple(selected) if len(selected) >= minimum else ()


def count_changes(changes: list[dict], counts: Counter[str]) -> None:
    """Count the saved VERL word substitutions without needing PDF artifacts."""
    for change in changes:
        before, after = change["origin_ans"], change["ocr_ans"]
        if len(before) != len(after):
            continue
        differences = [(a, b) for a, b in zip(before, after) if a != b]
        if len(differences) == 1:
            a, b = differences[0]
            counts[f"{a}->{b}"] += 1


def distribution_report(counts: Counter[str]) -> dict:
    total = sum(counts.values())
    return {
        "mutation_policy": POLICY_NAME,
        "mutation_policy_version": POLICY_VERSION,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "total_weight": TOTAL_WEIGHT,
        "saved_mutations": total,
        "sampling": "pair_weight_then_available_word_then_position",
        "pair_distribution": {
            f"{a}->{b}": {
                "weight": weight,
                "target_ratio": weight / TOTAL_WEIGHT,
                "saved_count": counts[f"{a}->{b}"],
                "saved_ratio": counts[f"{a}->{b}"] / total if total else 0.0,
            }
            for (a, b), weight in PAIR_WEIGHTS.items()
        },
        "other_pairs": {
            key: value
            for key, value in counts.items()
            if key not in {f"{a}->{b}" for a, b in PAIR_WEIGHTS}
        },
    }
