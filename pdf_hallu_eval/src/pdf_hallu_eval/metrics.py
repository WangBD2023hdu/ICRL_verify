from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from .align import Alignment


@dataclass(frozen=True)
class PageMetrics:
    reference_chars: int
    prediction_chars: int
    matches: int
    substitutions: int
    deletions: int
    insertions: int
    cer: float
    hallucination_rate: float
    pure_insertion_rate: float
    omission_rate: float
    coverage: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_page_metrics(alignment: Alignment) -> PageMetrics:
    counts = alignment.counts
    reference_chars = len(alignment.reference)
    prediction_chars = len(alignment.prediction)
    cer = _safe_rate(counts.substitutions + counts.deletions + counts.insertions, reference_chars)
    hallucination_rate = _safe_rate(counts.substitutions + counts.insertions, prediction_chars)
    pure_insertion_rate = _safe_rate(counts.insertions, prediction_chars)
    omission_rate = _safe_rate(counts.substitutions + counts.deletions, reference_chars)
    coverage = _safe_rate(counts.matches, reference_chars)
    return PageMetrics(
        reference_chars=reference_chars,
        prediction_chars=prediction_chars,
        matches=counts.matches,
        substitutions=counts.substitutions,
        deletions=counts.deletions,
        insertions=counts.insertions,
        cer=cer,
        hallucination_rate=hallucination_rate,
        pure_insertion_rate=pure_insertion_rate,
        omission_rate=omission_rate,
        coverage=coverage,
    )


def summarize_metrics(metrics: list[PageMetrics]) -> dict[str, float | int]:
    if not metrics:
        return {
            "pages": 0,
            "reference_chars": 0,
            "prediction_chars": 0,
            "cer": 0.0,
            "hallucination_rate": 0.0,
            "pure_insertion_rate": 0.0,
            "omission_rate": 0.0,
            "coverage": 0.0,
        }

    total_reference = sum(item.reference_chars for item in metrics)
    total_prediction = sum(item.prediction_chars for item in metrics)
    total_matches = sum(item.matches for item in metrics)
    total_substitutions = sum(item.substitutions for item in metrics)
    total_deletions = sum(item.deletions for item in metrics)
    total_insertions = sum(item.insertions for item in metrics)
    return {
        "pages": len(metrics),
        "reference_chars": total_reference,
        "prediction_chars": total_prediction,
        "cer": _safe_rate(total_substitutions + total_deletions + total_insertions, total_reference),
        "hallucination_rate": _safe_rate(total_substitutions + total_insertions, total_prediction),
        "pure_insertion_rate": _safe_rate(total_insertions, total_prediction),
        "omission_rate": _safe_rate(total_substitutions + total_deletions, total_reference),
        "coverage": _safe_rate(total_matches, total_reference),
        "page_mean_hallucination_rate": mean(item.hallucination_rate for item in metrics),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator

