from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OperationKind = Literal["match", "substitute", "delete", "insert"]


@dataclass(frozen=True)
class AlignmentOp:
    kind: OperationKind
    reference: str
    prediction: str


@dataclass(frozen=True)
class AlignmentCounts:
    matches: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0


@dataclass(frozen=True)
class Alignment:
    reference: str
    prediction: str
    operations: list[AlignmentOp]
    counts: AlignmentCounts


def align_text(reference: str, prediction: str) -> Alignment:
    """Return a character-level Levenshtein alignment."""
    rows = len(reference) + 1
    cols = len(prediction) + 1
    costs = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        costs[i][0] = i
    for j in range(1, cols):
        costs[0][j] = j

    for i in range(1, rows):
        ref_char = reference[i - 1]
        for j in range(1, cols):
            pred_char = prediction[j - 1]
            substitute_cost = 0 if ref_char == pred_char else 1
            costs[i][j] = min(
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
                costs[i - 1][j - 1] + substitute_cost,
            )

    operations: list[AlignmentOp] = []
    i = len(reference)
    j = len(prediction)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            ref_char = reference[i - 1]
            pred_char = prediction[j - 1]
            diagonal_cost = 0 if ref_char == pred_char else 1
            if costs[i][j] == costs[i - 1][j - 1] + diagonal_cost:
                kind: OperationKind = "match" if diagonal_cost == 0 else "substitute"
                operations.append(AlignmentOp(kind, ref_char, pred_char))
                i -= 1
                j -= 1
                continue
        if j > 0 and costs[i][j] == costs[i][j - 1] + 1:
            operations.append(AlignmentOp("insert", "", prediction[j - 1]))
            j -= 1
            continue
        if i > 0:
            operations.append(AlignmentOp("delete", reference[i - 1], ""))
            i -= 1

    operations.reverse()
    return Alignment(
        reference=reference,
        prediction=prediction,
        operations=operations,
        counts=_count_operations(operations),
    )


def _count_operations(operations: list[AlignmentOp]) -> AlignmentCounts:
    return AlignmentCounts(
        matches=sum(1 for op in operations if op.kind == "match"),
        substitutions=sum(1 for op in operations if op.kind == "substitute"),
        deletions=sum(1 for op in operations if op.kind == "delete"),
        insertions=sum(1 for op in operations if op.kind == "insert"),
    )

