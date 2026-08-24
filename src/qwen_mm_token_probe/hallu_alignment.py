from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence


_HTML_TAG_RE = re.compile(r"<[^>]*>")
_MARKDOWN_FENCE_RE = re.compile(r"```(?:markdown|md|html)?", re.IGNORECASE)
_MARKDOWN_RULE_RE = re.compile(
    r"(?m)^[ \t]*(?:\|?[ \t]*:?-{3,}:?[ \t]*\|?)+[ \t]*$"
)


@dataclass(frozen=True)
class NormalizedText:
    text: str
    source_indices: tuple[int, ...]

    def positions_for_source_span(self, start: int, end: int) -> list[int]:
        return [
            index
            for index, source_index in enumerate(self.source_indices)
            if start <= source_index < end
        ]


@dataclass(frozen=True)
class TextAlignment:
    output_labels: tuple[str, ...]
    output_to_gt: tuple[int | None, ...]
    gt_to_output: tuple[tuple[int, ...], ...]
    substitutions: int
    insertions: int
    deletions: int

    @property
    def edit_distance(self) -> int:
        return self.substitutions + self.insertions + self.deletions


@dataclass(frozen=True)
class MutationObservation:
    mutation_id: str
    category: str
    similarity: str
    zone: str
    expected: str
    opposite_variant: str
    predicted: str
    relation: str
    gt_positions: tuple[int, ...]
    output_positions: tuple[int, ...]


def normalize_ocr_text(text: str) -> NormalizedText:
    """Remove Markdown/HTML layout syntax while preserving source character offsets."""

    skipped = [False] * len(text)
    for match in _HTML_TAG_RE.finditer(text):
        for index in range(match.start(), match.end()):
            skipped[index] = True
    for match in _MARKDOWN_FENCE_RE.finditer(text):
        for index in range(match.start(), match.end()):
            skipped[index] = True
    for match in _MARKDOWN_RULE_RE.finditer(text):
        for index in range(match.start(), match.end()):
            skipped[index] = True

    characters: list[str] = []
    source_indices: list[int] = []
    for index, character in enumerate(text):
        if skipped[index] or character.isspace():
            continue
        if character in {"#", "*", "`", "|", ">"}:
            continue
        normalized = unicodedata.normalize("NFKC", character)
        for normalized_character in normalized:
            if normalized_character.isspace():
                continue
            characters.append(normalized_character)
            source_indices.append(index)
    return NormalizedText(text="".join(characters), source_indices=tuple(source_indices))


def align_normalized_text(reference: str, output: str) -> TextAlignment:
    matcher = difflib.SequenceMatcher(a=reference, b=output, autojunk=False)
    output_labels = ["insertion"] * len(output)
    output_to_gt: list[int | None] = [None] * len(output)
    gt_to_output: list[list[int]] = [[] for _ in reference]
    substitutions = 0
    insertions = 0
    deletions = 0

    for operation, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        reference_length = a_end - a_start
        output_length = b_end - b_start
        if operation == "equal":
            for offset in range(reference_length):
                gt_index = a_start + offset
                output_index = b_start + offset
                output_labels[output_index] = "correct"
                output_to_gt[output_index] = gt_index
                gt_to_output[gt_index].append(output_index)
        elif operation == "replace":
            substitutions += min(reference_length, output_length)
            deletions += max(0, reference_length - output_length)
            insertions += max(0, output_length - reference_length)
            for offset in range(output_length):
                output_index = b_start + offset
                if reference_length:
                    relative = min(reference_length - 1, int(offset * reference_length / max(1, output_length)))
                    gt_index = a_start + relative
                    output_to_gt[output_index] = gt_index
                    gt_to_output[gt_index].append(output_index)
                output_labels[output_index] = "substitution" if offset < reference_length else "insertion"
        elif operation == "insert":
            insertions += output_length
            for output_index in range(b_start, b_end):
                output_labels[output_index] = "insertion"
        elif operation == "delete":
            deletions += reference_length
        else:
            raise RuntimeError(f"unexpected alignment operation: {operation}")

    return TextAlignment(
        output_labels=tuple(output_labels),
        output_to_gt=tuple(output_to_gt),
        gt_to_output=tuple(tuple(value) for value in gt_to_output),
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
    )


def recover_relocated_matches(
    reference: str,
    output: str,
    alignment: TextAlignment,
) -> TextAlignment:
    """Recover exact phrases moved by Markdown/layout reformatting.

    The primary alignment remains monotonic so genuine substitutions stay visible.
    This second pass only consumes unmatched reference/output spans one-to-one and
    requires a multi-character exact phrase, which prevents a changed single glyph
    from being explained away merely because it appears elsewhere on the page.
    """

    labels = list(alignment.output_labels)
    output_to_gt = list(alignment.output_to_gt)
    correct_gt = {
        gt_index
        for output_index, gt_index in enumerate(output_to_gt)
        if labels[output_index] == "correct" and gt_index is not None
    }
    output_ranges = _contiguous_ranges(
        [index for index, label in enumerate(labels) if label != "correct"]
    )
    gt_ranges = _contiguous_ranges(
        [index for index in range(len(reference)) if index not in correct_gt]
    )

    while output_ranges and gt_ranges:
        best: tuple[int, int, int, int, int] | None = None
        for output_range_index, (output_start, output_end) in enumerate(output_ranges):
            output_piece = output[output_start:output_end]
            for gt_range_index, (gt_start, gt_end) in enumerate(gt_ranges):
                match = difflib.SequenceMatcher(
                    a=reference[gt_start:gt_end],
                    b=output_piece,
                    autojunk=False,
                ).find_longest_match()
                matched_text = output_piece[match.b : match.b + match.size]
                if not _eligible_relocation(matched_text):
                    continue
                candidate = (
                    match.size,
                    output_range_index,
                    gt_range_index,
                    output_start + match.b,
                    gt_start + match.a,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            break

        size, output_range_index, gt_range_index, output_start, gt_start = best
        for offset in range(size):
            output_index = output_start + offset
            labels[output_index] = "relocated_correct"
            output_to_gt[output_index] = gt_start + offset
        output_ranges = _subtract_range(
            output_ranges,
            range_index=output_range_index,
            consumed_start=output_start,
            consumed_end=output_start + size,
        )
        gt_ranges = _subtract_range(
            gt_ranges,
            range_index=gt_range_index,
            consumed_start=gt_start,
            consumed_end=gt_start + size,
        )

    exact_gt = {
        gt_index
        for output_index, gt_index in enumerate(output_to_gt)
        if labels[output_index] in {"correct", "relocated_correct"}
        and gt_index is not None
    }
    gt_to_output: list[list[int]] = [[] for _ in reference]
    mapped_gt = set(exact_gt)
    for output_index, (label, gt_index) in enumerate(zip(labels, output_to_gt)):
        if gt_index is None:
            continue
        if label in {"correct", "relocated_correct"}:
            gt_to_output[gt_index].append(output_index)
        elif label == "substitution" and gt_index not in mapped_gt:
            gt_to_output[gt_index].append(output_index)
            mapped_gt.add(gt_index)

    substitutions = labels.count("substitution")
    insertions = labels.count("insertion")
    deletions = sum(not positions for positions in gt_to_output)
    return TextAlignment(
        output_labels=tuple(labels),
        output_to_gt=tuple(output_to_gt),
        gt_to_output=tuple(tuple(value) for value in gt_to_output),
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
    )


def _contiguous_ranges(indices: Sequence[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((start, previous + 1))
            start = index
        previous = index
    ranges.append((start, previous + 1))
    return ranges


def _subtract_range(
    ranges: Sequence[tuple[int, int]],
    *,
    range_index: int,
    consumed_start: int,
    consumed_end: int,
) -> list[tuple[int, int]]:
    start, end = ranges[range_index]
    replacement = []
    if start < consumed_start:
        replacement.append((start, consumed_start))
    if consumed_end < end:
        replacement.append((consumed_end, end))
    return [*ranges[:range_index], *replacement, *ranges[range_index + 1 :]]


def _eligible_relocation(text: str) -> bool:
    if len(text) >= 4:
        return True
    return len(text) >= 2 and any("\u3400" <= character <= "\u9fff" for character in text)


def observe_mutations(
    *,
    record: dict[str, Any],
    variant: str,
    normalized_ground_truth: NormalizedText,
    normalized_output: NormalizedText,
    alignment: TextAlignment,
) -> list[MutationObservation]:
    if variant not in {"clean", "edited"}:
        raise ValueError(f"unsupported pair variant: {variant}")
    span_key = "clean_markdown_span" if variant == "clean" else "edited_markdown_span"
    observations: list[MutationObservation] = []
    for mutation in record.get("mutations", []):
        span = mutation.get(span_key)
        if not isinstance(span, list) or len(span) != 2:
            continue
        gt_positions = normalized_ground_truth.positions_for_source_span(int(span[0]), int(span[1]))
        output_positions = sorted(
            {
                output_position
                for gt_position in gt_positions
                if gt_position < len(alignment.gt_to_output)
                for output_position in alignment.gt_to_output[gt_position]
            }
        )
        predicted = "".join(
            normalized_output.text[position]
            for position in output_positions
            if position < len(normalized_output.text)
        )
        expected = str(mutation["original"] if variant == "clean" else mutation["replacement"])
        opposite = str(mutation["replacement"] if variant == "clean" else mutation["original"])
        expected_normalized = unicodedata.normalize("NFKC", expected)
        opposite_normalized = unicodedata.normalize("NFKC", opposite)
        if not output_positions:
            relation = "deleted"
        elif predicted == expected_normalized:
            relation = "expected"
        elif predicted == opposite_normalized:
            relation = "opposite_variant"
        else:
            relation = "other"
        observations.append(
            MutationObservation(
                mutation_id=str(mutation.get("mutation_id", "")),
                category=str(mutation.get("category", "unknown")),
                similarity=str(mutation.get("similarity", "unknown")),
                zone=str(mutation.get("zone", "unknown")),
                expected=expected_normalized,
                opposite_variant=opposite_normalized,
                predicted=predicted,
                relation=relation,
                gt_positions=tuple(gt_positions),
                output_positions=tuple(output_positions),
            )
        )
    return observations


def token_alignment_rows(
    *,
    tokens: Sequence[dict[str, Any]],
    normalized_output: NormalizedText,
    alignment: TextAlignment,
    mutation_observations: Sequence[MutationObservation],
) -> list[dict[str, object]]:
    raw_spans = _raw_token_spans(tokens)
    mutation_by_output_position: dict[int, MutationObservation] = {}
    for observation in mutation_observations:
        for position in observation.output_positions:
            mutation_by_output_position[position] = observation

    rows: list[dict[str, object]] = []
    previous_lexical_character = ""
    for token, (source_start, source_end) in zip(tokens, raw_spans):
        normalized_positions = [
            position
            for position, source_index in enumerate(normalized_output.source_indices)
            if source_start <= source_index < source_end
        ]
        char_labels = [alignment.output_labels[position] for position in normalized_positions]
        mutation_matches = {
            mutation_by_output_position[position]
            for position in normalized_positions
            if position in mutation_by_output_position
        }
        raw_token = str(token.get("raw_token", token.get("token", "")))
        normalized_piece = "".join(normalized_output.text[position] for position in normalized_positions)
        lexical_role = classify_lexical_role(
            raw_token=raw_token,
            normalized_piece=normalized_piece,
            previous_lexical_character=previous_lexical_character,
        )
        if normalized_piece:
            previous_lexical_character = normalized_piece[-1]

        mutation_relation = "unrelated"
        mutation_ids = ""
        mutation_categories = ""
        mutation_similarities = ""
        mutation_zones = ""
        if mutation_matches:
            relations = {item.relation for item in mutation_matches}
            mutation_ids = ",".join(sorted(item.mutation_id for item in mutation_matches))
            mutation_categories = ",".join(
                sorted({item.category for item in mutation_matches})
            )
            mutation_similarities = ",".join(
                sorted({item.similarity for item in mutation_matches})
            )
            mutation_zones = ",".join(sorted({item.zone for item in mutation_matches}))
            if "opposite_variant" in relations:
                mutation_relation = "opposite_variant"
            elif "other" in relations:
                mutation_relation = "other"
            elif "expected" in relations:
                mutation_relation = "expected"
            else:
                mutation_relation = sorted(relations)[0]

        if not normalized_positions:
            token_label = "formatting"
        elif mutation_relation == "opposite_variant":
            token_label = "mutation_opposite_variant"
        elif mutation_relation == "other" and any(label != "correct" for label in char_labels):
            token_label = "mutation_other"
        elif "insertion" in char_labels:
            token_label = "hallucinated_insertion"
        elif "substitution" in char_labels:
            token_label = "hallucinated_substitution"
        else:
            token_label = "correct"

        rows.append(
            {
                "index": int(token.get("index", len(rows))),
                "token_id": int(token.get("token_id", -1)),
                "token": str(token.get("token", raw_token)),
                "raw_token": raw_token,
                "normalized_piece": normalized_piece,
                "normalized_char_count": len(normalized_positions),
                "correct_char_count": sum(
                    label in {"correct", "relocated_correct"} for label in char_labels
                ),
                "relocated_correct_char_count": char_labels.count("relocated_correct"),
                "substitution_char_count": char_labels.count("substitution"),
                "insertion_char_count": char_labels.count("insertion"),
                "token_label": token_label,
                "is_hallucination": token_label not in {"correct", "formatting"},
                "mutation_relation": mutation_relation,
                "mutation_ids": mutation_ids,
                "mutation_categories": mutation_categories,
                "mutation_similarities": mutation_similarities,
                "mutation_zones": mutation_zones,
                "script": classify_script(normalized_piece),
                "lexical_role": lexical_role,
                "is_multi_character": len(normalized_piece) > 1,
                "raw_source_start": source_start,
                "raw_source_end": source_end,
            }
        )
    return rows


def classify_script(text: str) -> str:
    if not text:
        return "formatting"
    classes: set[str] = set()
    for character in text:
        if "\u3400" <= character <= "\u9fff":
            classes.add("cjk")
        elif character.isalpha():
            classes.add("latin")
        elif character.isdigit():
            classes.add("digit")
        elif unicodedata.category(character).startswith(("P", "S")):
            classes.add("punctuation")
        else:
            classes.add("other")
    return next(iter(classes)) if len(classes) == 1 else "mixed"


def classify_lexical_role(
    *,
    raw_token: str,
    normalized_piece: str,
    previous_lexical_character: str,
) -> str:
    script = classify_script(normalized_piece)
    if script == "formatting":
        return "formatting"
    begins_with_space = bool(raw_token) and raw_token[0].isspace()
    numeric_punctuation = {".", ",", "+", "-", "−", "%", "‰", "$", "¥", "€", "£"}
    numeric_piece = bool(normalized_piece) and any(
        character.isdigit() for character in normalized_piece
    ) and all(
        character.isdigit() or character in numeric_punctuation
        for character in normalized_piece
    )
    previous_is_numeric = bool(previous_lexical_character) and (
        previous_lexical_character.isdigit()
        or previous_lexical_character in numeric_punctuation
    )
    if numeric_piece:
        return (
            "number_continuation"
            if previous_is_numeric and not begins_with_space
            else "number_initial"
        )
    if script == "punctuation":
        if previous_is_numeric and all(
            character in numeric_punctuation for character in normalized_piece
        ):
            return "number_continuation"
        return "punctuation"
    if script == "cjk":
        return "multi_cjk" if len(normalized_piece) > 1 else "single_cjk"
    previous_is_word = bool(previous_lexical_character) and previous_lexical_character.isalnum()
    if script in {"latin", "digit", "mixed"}:
        return "continuation" if previous_is_word and not begins_with_space else "word_initial"
    return "other"


def _raw_token_spans(tokens: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for token in tokens:
        raw_token = str(token.get("raw_token", token.get("token", "")))
        spans.append((offset, offset + len(raw_token)))
        offset += len(raw_token)
    return spans
