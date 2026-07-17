from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Protocol

from .score_types import TokenScore
from .token_grouping import WordScore, group_token_scores


class GeneratedTokenStatsLike(Protocol):
    probabilities: list[float]
    log_probabilities: list[float]
    top_token_ids: list[int]
    top_probabilities: list[float]
    top_log_probabilities: list[float]


def baseline_token_records(
    token_ids: list[int],
    stats: GeneratedTokenStatsLike,
    tokenizer: Any,
) -> list[dict[str, object]]:
    from .hf_qwen import decode_token_piece, display_token

    _validate_stats_length(token_ids, stats)
    return [
        {
            "index": index,
            "token_id": token_id,
            "token": display_token(tokenizer, token_id),
            "raw_token": decode_token_piece(tokenizer, token_id),
            "p_original": float(stats.probabilities[index]),
            "logp_original": float(stats.log_probabilities[index]),
            "top_token_id_original": int(stats.top_token_ids[index]),
            "top_token_original": display_token(tokenizer, stats.top_token_ids[index]),
            "top_raw_token_original": decode_token_piece(
                tokenizer,
                stats.top_token_ids[index],
            ),
            "top_p_original": float(stats.top_probabilities[index]),
            "top_logp_original": float(stats.top_log_probabilities[index]),
        }
        for index, token_id in enumerate(token_ids)
    ]


def blurred_token_records(
    token_ids: list[int],
    stats: GeneratedTokenStatsLike,
    tokenizer: Any,
) -> list[dict[str, object]]:
    from .hf_qwen import decode_token_piece, display_token

    _validate_stats_length(token_ids, stats)
    return [
        {
            "index": index,
            "p_blurred": float(stats.probabilities[index]),
            "logp_blurred": float(stats.log_probabilities[index]),
            "top_token_id_blurred": int(stats.top_token_ids[index]),
            "top_token_blurred": display_token(tokenizer, stats.top_token_ids[index]),
            "top_raw_token_blurred": decode_token_piece(
                tokenizer,
                stats.top_token_ids[index],
            ),
            "top_p_blurred": float(stats.top_probabilities[index]),
            "top_logp_blurred": float(stats.top_log_probabilities[index]),
        }
        for index in range(len(token_ids))
    ]


def comparison_scores_from_records(
    baseline: list[dict[str, object]],
    blurred: list[dict[str, object]],
) -> list[TokenScore]:
    if len(baseline) != len(blurred):
        raise ValueError(
            f"token record length mismatch: baseline={len(baseline)}, blurred={len(blurred)}"
        )

    scores = []
    for baseline_row, blurred_row in zip(baseline, blurred):
        if int(baseline_row["index"]) != int(blurred_row["index"]):
            raise ValueError("token record indices are not aligned")
        scores.append(
            TokenScore(
                index=int(baseline_row["index"]),
                token_id=int(baseline_row["token_id"]),
                token=str(baseline_row["token"]),
                raw_token=str(baseline_row["raw_token"]),
                p_original=float(baseline_row["p_original"]),
                p_masked=float(blurred_row["p_blurred"]),
                logp_original=float(baseline_row["logp_original"]),
                logp_masked=float(blurred_row["logp_blurred"]),
                top_token_id_original=int(baseline_row["top_token_id_original"]),
                top_token_original=str(baseline_row["top_token_original"]),
                top_raw_token_original=str(baseline_row["top_raw_token_original"]),
                top_p_original=float(baseline_row["top_p_original"]),
                top_logp_original=float(baseline_row["top_logp_original"]),
                top_token_id_masked=int(blurred_row["top_token_id_blurred"]),
                top_token_masked=str(blurred_row["top_token_blurred"]),
                top_raw_token_masked=str(blurred_row["top_raw_token_blurred"]),
                top_p_masked=float(blurred_row["top_p_blurred"]),
                top_logp_masked=float(blurred_row["top_logp_blurred"]),
            )
        )
    return scores


def summarize_scores(
    scores: list[TokenScore],
    *,
    word_scores: list[WordScore] | None = None,
) -> dict[str, float | int]:
    if not scores:
        raise ValueError("cannot summarize an empty token sequence")

    if word_scores is None:
        word_scores = group_token_scores(scores)
    lexical_units = [score for score in word_scores if score.unit_type == "word"]

    p_original = [score.p_original for score in scores]
    p_blurred = [score.p_masked for score in scores]
    logp_original = [score.logp_original for score in scores]
    logp_blurred = [score.logp_masked for score in scores]
    delta_p = [score.delta_p for score in scores]
    delta_logp = [score.delta_logp for score in scores]
    abs_delta_logp = [abs(value) for value in delta_logp]
    blur_gain_logp = [max(0.0, -value) for value in delta_logp]
    positive_blur_gains = [-value for value in delta_logp if value < 0.0]
    blur_gain_p = [max(0.0, score.p_masked - score.p_original) for score in scores]
    positive_blur_gain_p = [
        score.p_masked - score.p_original
        for score in scores
        if score.logp_masked > score.logp_original
    ]
    mean_logp_original = statistics.fmean(logp_original)
    mean_logp_blurred = statistics.fmean(logp_blurred)

    summary: dict[str, float | int] = {
        "num_tokens": len(scores),
        "mean_p_original": statistics.fmean(p_original),
        "mean_p_blurred": statistics.fmean(p_blurred),
        "mean_logp_original": mean_logp_original,
        "mean_logp_blurred": mean_logp_blurred,
        "perplexity_original": _perplexity(mean_logp_original),
        "perplexity_blurred": _perplexity(mean_logp_blurred),
        "mean_delta_p": statistics.fmean(delta_p),
        "mean_abs_delta_p": statistics.fmean(abs(value) for value in delta_p),
        "mean_delta_logp": statistics.fmean(delta_logp),
        "median_delta_logp": statistics.median(delta_logp),
        "p90_abs_delta_logp": _percentile(abs_delta_logp, 0.90),
        "probability_drop_rate": _mean_bool(
            score.logp_masked < score.logp_original for score in scores
        ),
        "probability_gain_rate": _mean_bool(
            score.logp_masked > score.logp_original for score in scores
        ),
        "num_probability_gain_tokens": sum(
            1 for score in scores if score.logp_masked > score.logp_original
        ),
        "mean_blur_gain_p_all_tokens": statistics.fmean(blur_gain_p),
        "mean_blur_gain_p_gained_tokens": _mean_or_zero(positive_blur_gain_p),
        "p90_blur_gain_p": _percentile(positive_blur_gain_p, 0.90),
        "mean_blur_gain_logp_all_tokens": statistics.fmean(blur_gain_logp),
        "mean_blur_gain_logp_gained_tokens": _mean_or_zero(positive_blur_gains),
        "p90_blur_gain_logp": _percentile(positive_blur_gains, 0.90),
        "strong_drop_rate": _mean_bool(score.delta_logp >= 1.0 for score in scores),
        "strong_gain_rate": _mean_bool(score.delta_logp <= -1.0 for score in scores),
        "top1_changed_rate": _mean_bool(score.top_token_changed for score in scores),
        "target_top_rate_original": _mean_bool(
            score.target_is_top_original for score in scores
        ),
        "target_top_rate_blurred": _mean_bool(score.target_is_top_masked for score in scores),
        "num_word_units": len(lexical_units),
    }

    first_token_delta = [score.first_token_delta_logp for score in lexical_units]
    summary.update(
        {
            "mean_word_first_token_delta_logp": _mean_or_zero(first_token_delta),
            "median_word_first_token_delta_logp": _median_or_zero(first_token_delta),
            "word_first_token_drop_rate": _mean_bool(value > 0 for value in first_token_delta),
        }
    )
    return summary


def _validate_stats_length(token_ids: list[int], stats: GeneratedTokenStatsLike) -> None:
    lengths = {
        len(token_ids),
        len(stats.probabilities),
        len(stats.log_probabilities),
        len(stats.top_token_ids),
        len(stats.top_probabilities),
        len(stats.top_log_probabilities),
    }
    if len(lengths) != 1:
        raise ValueError(f"generated-token statistics are not aligned: lengths={sorted(lengths)}")


def _mean_bool(values: Iterable[bool]) -> float:
    numbers = [1.0 if value else 0.0 for value in values]
    return statistics.fmean(numbers) if numbers else 0.0


def _mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median_or_zero(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _perplexity(mean_logp: float) -> float:
    return math.exp(min(700.0, max(-700.0, -mean_logp)))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)
