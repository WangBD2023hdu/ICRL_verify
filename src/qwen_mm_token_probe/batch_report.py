from __future__ import annotations

import csv
import html
import json
import math
import os
import random
import statistics
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .batch_stats import comparison_scores_from_records, summarize_scores
from .score_types import TokenScore
from .token_grouping import WordScore, group_token_scores


TOKEN_FIELDS = [
    "sample_key",
    "record_id",
    "line_number",
    "image_index",
    "image_path",
    "blur_radius",
    "index",
    "token_id",
    "token",
    "raw_token",
    "p_original",
    "p_blurred",
    "delta_p_original_minus_blurred",
    "blur_minus_original_p",
    "logp_original",
    "logp_blurred",
    "delta_logp_original_minus_blurred",
    "blur_minus_original_logp",
    "probability_increased_after_blur",
    "top_token_id_original",
    "top_token_original",
    "top_p_original",
    "top_token_id_blurred",
    "top_token_blurred",
    "top_p_blurred",
    "top_token_changed",
    "target_is_top_original",
    "target_is_top_blurred",
]

WORD_FIELDS = [
    "sample_key",
    "record_id",
    "line_number",
    "image_index",
    "image_path",
    "blur_radius",
    "index",
    "unit_type",
    "text",
    "token_start",
    "token_end",
    "token_count",
    "first_token_p_original",
    "first_token_p_blurred",
    "first_token_delta_logp_original_minus_blurred",
    "blur_minus_original_first_token_p",
    "blur_minus_original_first_token_logp",
    "first_token_probability_increased_after_blur",
    "sum_logp_original",
    "sum_logp_blurred",
    "delta_sum_logp_original_minus_blurred",
    "mean_logp_original",
    "mean_logp_blurred",
    "delta_mean_logp_original_minus_blurred",
]

SUMMARY_METRICS = [
    "num_tokens",
    "mean_p_original",
    "mean_p_blurred",
    "mean_logp_original",
    "mean_logp_blurred",
    "perplexity_original",
    "perplexity_blurred",
    "mean_delta_p",
    "mean_abs_delta_p",
    "mean_delta_logp",
    "median_delta_logp",
    "p90_abs_delta_logp",
    "probability_drop_rate",
    "probability_gain_rate",
    "num_probability_gain_tokens",
    "mean_blur_gain_p_all_tokens",
    "mean_blur_gain_p_gained_tokens",
    "p90_blur_gain_p",
    "mean_blur_gain_logp_all_tokens",
    "mean_blur_gain_logp_gained_tokens",
    "p90_blur_gain_logp",
    "strong_drop_rate",
    "strong_gain_rate",
    "top1_changed_rate",
    "target_top_rate_original",
    "target_top_rate_blurred",
    "num_word_units",
    "mean_word_first_token_delta_logp",
    "median_word_first_token_delta_logp",
    "word_first_token_drop_rate",
]

SAMPLE_SUMMARY_FIELDS = [
    "sample_key",
    "record_id",
    "line_number",
    "image_index",
    "image_path",
    "blur_radius",
    *SUMMARY_METRICS,
]

AGGREGATE_SUMMARY_FIELDS = [
    "blur_radius",
    "num_samples",
    "quantile_sample_size",
    "quantiles_approximate",
    *SUMMARY_METRICS,
]


def rebuild_batch_reports(
    output_dir: str | Path,
    *,
    group_tokens: str = "word",
    report_max_tokens: int = 4096,
    top_affected_tokens: int = 100,
    allowed_sample_keys: set[str] | None = None,
) -> dict[str, object]:
    output_root = Path(output_dir).expanduser().resolve()
    result_paths = sorted((output_root / "samples").glob("*/result.json"))
    if allowed_sample_keys is not None:
        result_paths = [
            result_path
            for result_path in result_paths
            if result_path.parent.name in allowed_sample_keys
        ]
    config = _read_json_if_exists(output_root / "config.json") or {}
    sample_summary_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    accumulators: dict[float, _AggregateAccumulator] = {}

    token_tmp = output_root / "token_probabilities.csv.tmp"
    word_tmp = output_root / "word_probabilities.csv.tmp"
    token_tmp.parent.mkdir(parents=True, exist_ok=True)

    with token_tmp.open("w", encoding="utf-8", newline="") as token_handle:
        token_writer = csv.DictWriter(token_handle, fieldnames=TOKEN_FIELDS)
        token_writer.writeheader()
        word_handle = None
        word_writer = None
        if group_tokens == "word":
            word_handle = word_tmp.open("w", encoding="utf-8", newline="")
            word_writer = csv.DictWriter(word_handle, fieldnames=WORD_FIELDS)
            word_writer.writeheader()

        try:
            for result_path in result_paths:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                sample = result["sample"]
                baseline = result["original"]["tokens"]
                sample_dir = result_path.parent
                sample_token_path = sample_dir / "token_probabilities.csv"
                sample_token_tmp = sample_token_path.with_suffix(".csv.tmp")
                sample_word_path = sample_dir / "word_probabilities.csv"
                sample_word_tmp = sample_word_path.with_suffix(".csv.tmp")

                with sample_token_tmp.open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as sample_token_handle:
                    sample_token_writer = csv.DictWriter(
                        sample_token_handle,
                        fieldnames=TOKEN_FIELDS,
                    )
                    sample_token_writer.writeheader()
                    sample_word_handle = None
                    sample_word_writer = None
                    if group_tokens == "word":
                        sample_word_handle = sample_word_tmp.open(
                            "w",
                            encoding="utf-8",
                            newline="",
                        )
                        sample_word_writer = csv.DictWriter(
                            sample_word_handle,
                            fieldnames=WORD_FIELDS,
                        )
                        sample_word_writer.writeheader()

                    try:
                        for level in result["blur_levels"]:
                            radius = float(level["blur_radius"])
                            scores = comparison_scores_from_records(
                                baseline,
                                level["tokens"],
                            )
                            word_scores = group_token_scores(scores)
                            summary = summarize_scores(scores, word_scores=word_scores)
                            sample_summary_rows.append(
                                _summary_row(sample, radius, summary)
                            )

                            accumulator = accumulators.setdefault(
                                radius,
                                _AggregateAccumulator(radius=radius),
                            )
                            accumulator.add_sample(scores, word_scores)

                            for score in scores:
                                row = _token_row(sample, radius, score)
                                token_writer.writerow(row)
                                sample_token_writer.writerow(row)
                            if word_writer is not None and sample_word_writer is not None:
                                for score in word_scores:
                                    row = _word_row(sample, radius, score)
                                    word_writer.writerow(row)
                                    sample_word_writer.writerow(row)
                    finally:
                        if sample_word_handle is not None:
                            sample_word_handle.close()

                sample_token_tmp.replace(sample_token_path)
                if group_tokens == "word":
                    sample_word_tmp.replace(sample_word_path)
                _write_sample_report(
                    sample_dir / "report.html",
                    result=result,
                    group_tokens=group_tokens,
                    report_max_tokens=report_max_tokens,
                    top_affected_tokens=top_affected_tokens,
                )
                manifest_rows.append(
                    {
                        "sample_key": sample["sample_key"],
                        "record_id": sample["record_id"],
                        "image_path": sample["image_path"],
                        "num_tokens": len(baseline),
                        "num_blur_levels": len(result["blur_levels"]),
                        "result_path": str(result_path),
                        "report_path": str(sample_dir / "report.html"),
                    }
                )
        finally:
            if word_handle is not None:
                word_handle.close()

    token_tmp.replace(output_root / "token_probabilities.csv")
    if group_tokens == "word":
        word_tmp.replace(output_root / "word_probabilities.csv")

    aggregate_rows = [
        accumulators[radius].finalize()
        for radius in sorted(accumulators)
    ]
    _write_csv_atomic(
        output_root / "sample_summary.csv",
        SAMPLE_SUMMARY_FIELDS,
        sample_summary_rows,
    )
    _write_csv_atomic(
        output_root / "aggregate_summary.csv",
        AGGREGATE_SUMMARY_FIELDS,
        aggregate_rows,
    )
    _write_json_atomic(
        output_root / "aggregate_summary.json",
        {
            "completed_samples": len(manifest_rows),
            "aggregate": aggregate_rows,
        },
    )
    _write_jsonl_atomic(output_root / "manifest.jsonl", manifest_rows)
    plot_written = _write_aggregate_plot(output_root / "aggregate_summary.png", aggregate_rows)
    unresolved_failures = _unresolved_failure_count(
        output_root,
        manifest_rows,
        allowed_sample_keys=allowed_sample_keys,
    )
    _write_index_report(
        output_root / "report.html",
        config=config,
        aggregate_rows=aggregate_rows,
        sample_rows=sample_summary_rows,
        completed_samples=len(manifest_rows),
        unresolved_failures=unresolved_failures,
        plot_written=plot_written,
        group_tokens=group_tokens,
    )
    return {
        "completed_samples": len(manifest_rows),
        "unresolved_failures": unresolved_failures,
        "num_blur_levels": len(aggregate_rows),
        "report_path": str(output_root / "report.html"),
    }


@dataclass
class _Reservoir:
    limit: int = 100_000
    seed: int = 0
    values: array = field(default_factory=lambda: array("d"))
    seen: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.limit:
            self.values.append(float(value))
            return
        index = self._rng.randrange(self.seen)
        if index < self.limit:
            self.values[index] = float(value)

    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    def percentile(self, quantile: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        fraction = position - lower
        return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


@dataclass
class _AggregateAccumulator:
    radius: float
    num_samples: int = 0
    num_tokens: int = 0
    sum_p_original: float = 0.0
    sum_p_blurred: float = 0.0
    sum_logp_original: float = 0.0
    sum_logp_blurred: float = 0.0
    sum_delta_p: float = 0.0
    sum_abs_delta_p: float = 0.0
    sum_delta_logp: float = 0.0
    probability_drop_count: int = 0
    probability_gain_count: int = 0
    strong_drop_count: int = 0
    strong_gain_count: int = 0
    top1_changed_count: int = 0
    target_top_original_count: int = 0
    target_top_blurred_count: int = 0
    sum_blur_gain_all: float = 0.0
    sum_blur_gain_gained: float = 0.0
    sum_blur_gain_p_all: float = 0.0
    sum_blur_gain_p_gained: float = 0.0
    num_word_units: int = 0
    sum_word_first_delta: float = 0.0
    word_first_drop_count: int = 0
    delta_logp_reservoir: _Reservoir = field(init=False)
    abs_delta_logp_reservoir: _Reservoir = field(init=False)
    gain_logp_reservoir: _Reservoir = field(init=False)
    gain_p_reservoir: _Reservoir = field(init=False)
    word_delta_reservoir: _Reservoir = field(init=False)

    def __post_init__(self) -> None:
        seed = int(round(self.radius * 1000)) + 17
        self.delta_logp_reservoir = _Reservoir(seed=seed)
        self.abs_delta_logp_reservoir = _Reservoir(seed=seed + 1)
        self.gain_logp_reservoir = _Reservoir(seed=seed + 2)
        self.gain_p_reservoir = _Reservoir(seed=seed + 3)
        self.word_delta_reservoir = _Reservoir(seed=seed + 4)

    def add_sample(self, scores: list[TokenScore], word_scores: list[WordScore]) -> None:
        self.num_samples += 1
        for score in scores:
            self.num_tokens += 1
            self.sum_p_original += score.p_original
            self.sum_p_blurred += score.p_masked
            self.sum_logp_original += score.logp_original
            self.sum_logp_blurred += score.logp_masked
            self.sum_delta_p += score.delta_p
            self.sum_abs_delta_p += abs(score.delta_p)
            self.sum_delta_logp += score.delta_logp
            self.delta_logp_reservoir.add(score.delta_logp)
            self.abs_delta_logp_reservoir.add(abs(score.delta_logp))
            if score.logp_masked < score.logp_original:
                self.probability_drop_count += 1
            if score.logp_masked > score.logp_original:
                gain = -score.delta_logp
                gain_p = -score.delta_p
                self.probability_gain_count += 1
                self.sum_blur_gain_gained += gain
                self.sum_blur_gain_p_gained += gain_p
                self.gain_logp_reservoir.add(gain)
                self.gain_p_reservoir.add(gain_p)
            self.sum_blur_gain_all += max(0.0, -score.delta_logp)
            self.sum_blur_gain_p_all += max(0.0, -score.delta_p)
            if score.delta_logp >= 1.0:
                self.strong_drop_count += 1
            if score.delta_logp <= -1.0:
                self.strong_gain_count += 1
            if score.top_token_changed:
                self.top1_changed_count += 1
            if score.target_is_top_original:
                self.target_top_original_count += 1
            if score.target_is_top_masked:
                self.target_top_blurred_count += 1

        for word_score in word_scores:
            if word_score.unit_type != "word":
                continue
            self.num_word_units += 1
            delta = word_score.first_token_delta_logp
            self.sum_word_first_delta += delta
            self.word_delta_reservoir.add(delta)
            if delta > 0:
                self.word_first_drop_count += 1

    def finalize(self) -> dict[str, float | int]:
        count = max(1, self.num_tokens)
        word_count = max(1, self.num_word_units)
        mean_logp_original = self.sum_logp_original / count
        mean_logp_blurred = self.sum_logp_blurred / count
        return {
            "blur_radius": self.radius,
            "num_samples": self.num_samples,
            "quantile_sample_size": len(self.delta_logp_reservoir.values),
            "quantiles_approximate": (
                self.num_tokens > len(self.delta_logp_reservoir.values)
            ),
            "num_tokens": self.num_tokens,
            "mean_p_original": self.sum_p_original / count,
            "mean_p_blurred": self.sum_p_blurred / count,
            "mean_logp_original": mean_logp_original,
            "mean_logp_blurred": mean_logp_blurred,
            "perplexity_original": _perplexity(mean_logp_original),
            "perplexity_blurred": _perplexity(mean_logp_blurred),
            "mean_delta_p": self.sum_delta_p / count,
            "mean_abs_delta_p": self.sum_abs_delta_p / count,
            "mean_delta_logp": self.sum_delta_logp / count,
            "median_delta_logp": self.delta_logp_reservoir.median(),
            "p90_abs_delta_logp": self.abs_delta_logp_reservoir.percentile(0.90),
            "probability_drop_rate": self.probability_drop_count / count,
            "probability_gain_rate": self.probability_gain_count / count,
            "num_probability_gain_tokens": self.probability_gain_count,
            "mean_blur_gain_p_all_tokens": self.sum_blur_gain_p_all / count,
            "mean_blur_gain_p_gained_tokens": (
                self.sum_blur_gain_p_gained / max(1, self.probability_gain_count)
            ),
            "p90_blur_gain_p": self.gain_p_reservoir.percentile(0.90),
            "mean_blur_gain_logp_all_tokens": self.sum_blur_gain_all / count,
            "mean_blur_gain_logp_gained_tokens": (
                self.sum_blur_gain_gained / max(1, self.probability_gain_count)
            ),
            "p90_blur_gain_logp": self.gain_logp_reservoir.percentile(0.90),
            "strong_drop_rate": self.strong_drop_count / count,
            "strong_gain_rate": self.strong_gain_count / count,
            "top1_changed_rate": self.top1_changed_count / count,
            "target_top_rate_original": self.target_top_original_count / count,
            "target_top_rate_blurred": self.target_top_blurred_count / count,
            "num_word_units": self.num_word_units,
            "mean_word_first_token_delta_logp": self.sum_word_first_delta / word_count,
            "median_word_first_token_delta_logp": self.word_delta_reservoir.median(),
            "word_first_token_drop_rate": self.word_first_drop_count / word_count,
        }


def _summary_row(
    sample: dict[str, object],
    radius: float,
    summary: dict[str, float | int],
) -> dict[str, object]:
    row = {
        "sample_key": sample["sample_key"],
        "record_id": sample["record_id"],
        "line_number": sample["line_number"],
        "image_index": sample["image_index"],
        "image_path": sample["image_path"],
        "blur_radius": radius,
    }
    row.update(summary)
    return row


def _token_row(
    sample: dict[str, object],
    radius: float,
    score: TokenScore,
) -> dict[str, object]:
    return {
        "sample_key": sample["sample_key"],
        "record_id": sample["record_id"],
        "line_number": sample["line_number"],
        "image_index": sample["image_index"],
        "image_path": sample["image_path"],
        "blur_radius": radius,
        "index": score.index,
        "token_id": score.token_id,
        "token": score.token,
        "raw_token": score.raw_token,
        "p_original": score.p_original,
        "p_blurred": score.p_masked,
        "delta_p_original_minus_blurred": score.delta_p,
        "blur_minus_original_p": -score.delta_p,
        "logp_original": score.logp_original,
        "logp_blurred": score.logp_masked,
        "delta_logp_original_minus_blurred": score.delta_logp,
        "blur_minus_original_logp": -score.delta_logp,
        "probability_increased_after_blur": score.logp_masked > score.logp_original,
        "top_token_id_original": score.top_token_id_original,
        "top_token_original": score.top_token_original,
        "top_p_original": score.top_p_original,
        "top_token_id_blurred": score.top_token_id_masked,
        "top_token_blurred": score.top_token_masked,
        "top_p_blurred": score.top_p_masked,
        "top_token_changed": score.top_token_changed,
        "target_is_top_original": score.target_is_top_original,
        "target_is_top_blurred": score.target_is_top_masked,
    }


def _word_row(
    sample: dict[str, object],
    radius: float,
    score: WordScore,
) -> dict[str, object]:
    return {
        "sample_key": sample["sample_key"],
        "record_id": sample["record_id"],
        "line_number": sample["line_number"],
        "image_index": sample["image_index"],
        "image_path": sample["image_path"],
        "blur_radius": radius,
        "index": score.index,
        "unit_type": score.unit_type,
        "text": score.text,
        "token_start": score.token_start,
        "token_end": score.token_end,
        "token_count": score.token_count,
        "first_token_p_original": score.first_token_p_original,
        "first_token_p_blurred": score.first_token_p_masked,
        "first_token_delta_logp_original_minus_blurred": score.first_token_delta_logp,
        "blur_minus_original_first_token_p": -score.first_token_delta_p,
        "blur_minus_original_first_token_logp": -score.first_token_delta_logp,
        "first_token_probability_increased_after_blur": (
            score.first_token_logp_masked > score.first_token_logp_original
        ),
        "sum_logp_original": score.sum_logp_original,
        "sum_logp_blurred": score.sum_logp_masked,
        "delta_sum_logp_original_minus_blurred": score.delta_sum_logp,
        "mean_logp_original": score.mean_logp_original,
        "mean_logp_blurred": score.mean_logp_masked,
        "delta_mean_logp_original_minus_blurred": score.delta_mean_logp,
    }


def _write_aggregate_plot(path: Path, rows: list[dict[str, object]]) -> bool:
    if not rows:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radii = [0.0] + [float(row["blur_radius"]) for row in rows]
    first = rows[0]
    original_p = [float(first["mean_p_original"])] + [
        float(row["mean_p_original"]) for row in rows
    ]
    blurred_p = [float(first["mean_p_original"])] + [
        float(row["mean_p_blurred"]) for row in rows
    ]
    mean_delta = [0.0] + [float(row["mean_delta_logp"]) for row in rows]
    gain_rate = [0.0] + [100.0 * float(row["probability_gain_rate"]) for row in rows]
    drop_rate = [0.0] + [100.0 * float(row["probability_drop_rate"]) for row in rows]
    top_changed = [0.0] + [100.0 * float(row["top1_changed_rate"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
    axes[0, 0].plot(radii, original_p, marker="o", label="original condition")
    axes[0, 0].plot(radii, blurred_p, marker="s", label="blurred condition")
    axes[0, 0].set_title("Mean selected-token probability")
    axes[0, 0].set_ylabel("probability")
    axes[0, 0].legend()

    axes[0, 1].plot(radii, mean_delta, marker="o", color="#c2410c")
    axes[0, 1].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[0, 1].set_title("Mean delta logp (original - blurred)")
    axes[0, 1].set_ylabel("log probability")

    axes[1, 0].plot(radii, gain_rate, marker="o", label="gain after blur")
    axes[1, 0].plot(radii, drop_rate, marker="s", label="drop after blur")
    axes[1, 0].set_title("Direction of token probability change")
    axes[1, 0].set_ylabel("tokens (%)")
    axes[1, 0].legend()

    axes[1, 1].plot(radii, top_changed, marker="o", color="#7c3aed")
    axes[1, 1].set_title("Top-1 token changed")
    axes[1, 1].set_ylabel("positions (%)")

    for axis in axes.flat:
        axis.set_xlabel("Gaussian blur radius")
        axis.grid(alpha=0.22, linewidth=0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _write_index_report(
    path: Path,
    *,
    config: dict[str, object],
    aggregate_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    completed_samples: int,
    unresolved_failures: int,
    plot_written: bool,
    group_tokens: str,
) -> None:
    total_tokens = sum(int(row["num_tokens"]) for row in aggregate_rows[:1])
    aggregate_table = "".join(_aggregate_table_row(row) for row in aggregate_rows)
    sample_table = "".join(_sample_table_row(row) for row in sample_rows)
    plot = (
        '<img class="plot" src="aggregate_summary.png" alt="Aggregate blur sensitivity charts">'
        if plot_written
        else '<p class="empty">No completed samples yet.</p>'
    )
    word_link = (
        '<a href="word_probabilities.csv">Word-first-token CSV</a>'
        if group_tokens == "word"
        else ""
    )
    model_id = html.escape(str(config.get("model_id", "")))
    source = html.escape(str(config.get("input_jsonl", "")))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Blur Sensitivity Report</title>
<style>{_report_css()}</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>OCR Blur Sensitivity</h1>
      <p class="muted">{model_id}</p>
    </div>
    <nav><a href="aggregate_summary.csv">Aggregate CSV</a><a href="sample_summary.csv">Sample CSV</a><a href="token_probabilities.csv">Token CSV</a>{word_link}</nav>
  </header>
  <section class="stats" aria-label="Run summary">
    <div class="stat"><span>Completed images</span><strong>{completed_samples}</strong></div>
    <div class="stat"><span>Failed images</span><strong>{unresolved_failures}</strong></div>
    <div class="stat"><span>Baseline tokens</span><strong>{total_tokens}</strong></div>
    <div class="stat"><span>Blur levels</span><strong>{len(aggregate_rows)}</strong></div>
  </section>
  <section>
    <h2>Aggregate response sensitivity</h2>
    {plot}
    <div class="table-wrap"><table>
      <thead><tr><th>Radius</th><th>Images</th><th>Tokens</th><th>p original</th><th>p blurred</th><th>Mean delta logp</th><th>Gain after blur</th><th>Mean gain delta p</th><th>Top-1 changed</th></tr></thead>
      <tbody>{aggregate_table}</tbody>
    </table></div>
  </section>
  <section>
    <h2>Samples</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Sample</th><th>Radius</th><th>Tokens</th><th>Mean delta logp</th><th>Gain after blur</th><th>Mean gain delta p</th><th>Drop after blur</th><th>Top-1 changed</th></tr></thead>
      <tbody>{sample_table}</tbody>
    </table></div>
  </section>
  <footer><code>delta_logp = logp_original - logp_blurred</code>. Negative values and positive <code>blur_minus_original_logp</code> identify tokens whose probability increased after blur.<br><span class="muted">Source: {source}</span></footer>
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _write_sample_report(
    path: Path,
    *,
    result: dict[str, object],
    group_tokens: str,
    report_max_tokens: int,
    top_affected_tokens: int,
) -> None:
    sample = result["sample"]
    baseline = result["original"]["tokens"]
    levels_payload = []
    image_cards = [
        _image_card(
            label="Original",
            image_path=Path(str(result["original"]["image_path"])),
            response_path=Path(str(result["original"]["response_path"])),
            report_dir=path.parent,
        )
    ]

    for level in result["blur_levels"]:
        radius = float(level["blur_radius"])
        scores = comparison_scores_from_records(baseline, level["tokens"])
        summary = summarize_scores(scores, word_scores=group_token_scores(scores))
        display_scores = scores if report_max_tokens == 0 else scores[:report_max_tokens]
        drop_scores = sorted(
            (score for score in scores if score.delta_logp > 0.0),
            key=lambda score: score.delta_p,
            reverse=True,
        )[:top_affected_tokens]
        gain_scores = sorted(
            (score for score in scores if score.delta_logp < 0.0),
            key=lambda score: score.delta_p,
        )[:top_affected_tokens]
        levels_payload.append(
            {
                "radius": radius,
                "summary": summary,
                "displayedTokenCount": len(display_scores),
                "totalTokenCount": len(scores),
                "scale": max(0.1, float(summary["p90_abs_delta_logp"])),
                "tokens": [_browser_token(score) for score in display_scores],
                "gains": [_browser_token(score) for score in gain_scores],
                "drops": [_browser_token(score) for score in drop_scores],
            }
        )
        generated_response = level.get("generated_response")
        response_path = None
        if isinstance(generated_response, dict) and generated_response.get("path"):
            response_path = Path(str(generated_response["path"]))
        image_cards.append(
            _image_card(
                label=f"Blur r={radius:g}",
                image_path=Path(str(level["image_path"])),
                response_path=response_path,
                report_dir=path.parent,
            )
        )

    browser_json = json.dumps(levels_payload, ensure_ascii=False, separators=(",", ":"))
    browser_json = browser_json.replace("<", "\\u003c")
    generated_text = html.escape(str(result["original"]["generated_text"]))
    sample_key = html.escape(str(sample["sample_key"]))
    image_source = html.escape(str(sample["image_path"]))
    cards = "".join(image_cards)
    word_link = (
        '<a href="word_probabilities.csv">Word CSV</a>'
        if group_tokens == "word"
        else ""
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{sample_key} - OCR Blur Sensitivity</title>
<style>{_report_css()}</style>
</head>
<body>
<main>
  <header>
    <div><h1>{sample_key}</h1><p class="muted path">{image_source}</p></div>
    <nav><a href="../../report.html">Batch report</a><a href="token_probabilities.csv">Token CSV</a>{word_link}</nav>
  </header>
  <section><h2>Image conditions</h2><div class="image-grid">{cards}</div></section>
  <section>
    <div class="controls"><label for="blur-level">Blur radius</label><select id="blur-level"></select></div>
    <div class="stats" aria-label="Selected blur summary">
      <div class="stat"><span>Mean p blurred</span><strong id="mean-p">-</strong></div>
      <div class="stat"><span>Tokens gaining p</span><strong id="gain-rate">-</strong></div>
      <div class="stat"><span>Mean probability gain</span><strong id="mean-gain">-</strong></div>
      <div class="stat"><span>Top-1 changed</span><strong id="top-changed">-</strong></div>
    </div>
  </section>
  <section>
    <h2>Token sensitivity</h2>
    <div class="legend"><span><i class="gain-key"></i>Probability increased after blur</span><span><i class="drop-key"></i>Probability decreased after blur</span></div>
    <div id="token-ribbon" class="token-ribbon" aria-live="polite"></div>
    <p id="token-limit" class="muted"></p>
  </section>
  <section class="comparison-grid">
    <div><h2>Largest gains after blur</h2><div class="table-wrap"><table><thead><tr><th>Index</th><th>Token</th><th>p original</th><th>p blurred</th><th>Blur - original p</th><th>Blur - original logp</th><th>Top-1 changed</th></tr></thead><tbody id="gain-rows"></tbody></table></div></div>
    <div><h2>Largest drops after blur</h2><div class="table-wrap"><table><thead><tr><th>Index</th><th>Token</th><th>p original</th><th>p blurred</th><th>Original - blur p</th><th>Original - blur logp</th><th>Top-1 changed</th></tr></thead><tbody id="drop-rows"></tbody></table></div></div>
  </section>
  <section><h2>Original OCR response</h2><pre>{generated_text}</pre></section>
  <footer><code>blur_minus_original_logp &gt; 0</code> means the fixed OCR token became more likely after image blur.</footer>
</main>
<script>
const levels = {browser_json};
const select = document.getElementById("blur-level");
const ribbon = document.getElementById("token-ribbon");
const limitNote = document.getElementById("token-limit");
const gainRows = document.getElementById("gain-rows");
const dropRows = document.getElementById("drop-rows");

for (const level of levels) {{
  const option = document.createElement("option");
  option.value = String(level.radius);
  option.textContent = `r=${{level.radius}}`;
  select.appendChild(option);
}}

function formatProbability(value) {{
  if (value >= 0.001) return value.toFixed(6);
  return value.toExponential(3);
}}

function renderTable(target, rows, gainMode) {{
  target.replaceChildren();
  for (const token of rows) {{
    const tr = document.createElement("tr");
    const values = [
      token.index,
      token.token,
      formatProbability(token.pOriginal),
      formatProbability(token.pBlurred),
      (gainMode ? token.blurGainP : -token.blurGainP).toFixed(6),
      (gainMode ? token.blurGainLogp : token.deltaLogp).toFixed(6),
      token.topChanged ? "yes" : "no",
    ];
    for (const value of values) {{
      const td = document.createElement("td");
      td.textContent = String(value);
      tr.appendChild(td);
    }}
    target.appendChild(tr);
  }}
  if (rows.length === 0) {{
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.className = "muted";
    td.textContent = "No tokens in this direction.";
    tr.appendChild(td);
    target.appendChild(tr);
  }}
}}

function renderLevel() {{
  const level = levels.find(item => String(item.radius) === select.value) || levels[0];
  if (!level) return;
  document.getElementById("mean-p").textContent = formatProbability(level.summary.mean_p_blurred);
  document.getElementById("gain-rate").textContent = `${{(100 * level.summary.probability_gain_rate).toFixed(2)}}%`;
  document.getElementById("mean-gain").textContent = formatProbability(level.summary.mean_blur_gain_p_gained_tokens);
  document.getElementById("top-changed").textContent = `${{(100 * level.summary.top1_changed_rate).toFixed(2)}}%`;
  ribbon.replaceChildren();
  for (const token of level.tokens) {{
    const span = document.createElement("span");
    const direction = token.blurGainLogp > 0 ? "gain" : (token.blurGainLogp < 0 ? "drop" : "neutral");
    span.className = `token-mark ${{direction}}`;
    const strength = Math.min(1, Math.abs(token.deltaLogp) / level.scale);
    span.style.setProperty("--strength", `${{(8 + 72 * strength).toFixed(2)}}%`);
    span.textContent = token.rawToken;
    span.title = `#${{token.index}} p(original)=${{formatProbability(token.pOriginal)}} p(blurred)=${{formatProbability(token.pBlurred)}} blur-original logp=${{token.blurGainLogp.toFixed(6)}}`;
    ribbon.appendChild(span);
  }}
  limitNote.textContent = level.displayedTokenCount < level.totalTokenCount
    ? `Showing the first ${{level.displayedTokenCount}} of ${{level.totalTokenCount}} tokens; CSV contains all tokens.`
    : `${{level.totalTokenCount}} tokens`;
  renderTable(gainRows, level.gains, true);
  renderTable(dropRows, level.drops, false);
}}

select.addEventListener("change", renderLevel);
renderLevel();
</script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _browser_token(score: TokenScore) -> dict[str, object]:
    return {
        "index": score.index,
        "token": score.token,
        "rawToken": score.raw_token,
        "pOriginal": score.p_original,
        "pBlurred": score.p_masked,
        "blurGainP": -score.delta_p,
        "deltaLogp": score.delta_logp,
        "blurGainLogp": -score.delta_logp,
        "topChanged": score.top_token_changed,
    }


def _image_card(
    *,
    label: str,
    image_path: Path,
    response_path: Path | None,
    report_dir: Path,
) -> str:
    image_rel = os.path.relpath(image_path.resolve(), report_dir.resolve())
    response_link = ""
    if response_path is not None:
        response_rel = os.path.relpath(response_path.resolve(), report_dir.resolve())
        response_link = f'<a href="{html.escape(response_rel, quote=True)}">Response</a>'
    return f"""<figure class="image-card">
  <img src="{html.escape(image_rel, quote=True)}" alt="{html.escape(label)} image">
  <figcaption><span>{html.escape(label)}</span>{response_link}</figcaption>
</figure>"""


def _aggregate_table_row(row: dict[str, object]) -> str:
    return f"""<tr>
<td>{float(row['blur_radius']):g}</td><td>{int(row['num_samples'])}</td><td>{int(row['num_tokens'])}</td>
<td>{float(row['mean_p_original']):.6f}</td><td>{float(row['mean_p_blurred']):.6f}</td>
<td>{float(row['mean_delta_logp']):+.6f}</td><td>{100 * float(row['probability_gain_rate']):.2f}%</td>
<td>{float(row['mean_blur_gain_p_gained_tokens']):.6f}</td><td>{100 * float(row['top1_changed_rate']):.2f}%</td>
</tr>"""


def _sample_table_row(row: dict[str, object]) -> str:
    sample_key = str(row["sample_key"])
    report_href = f"samples/{sample_key}/report.html"
    return f"""<tr>
<td><a href="{html.escape(report_href, quote=True)}">{html.escape(sample_key)}</a></td>
<td>{float(row['blur_radius']):g}</td><td>{int(row['num_tokens'])}</td>
<td>{float(row['mean_delta_logp']):+.6f}</td><td>{100 * float(row['probability_gain_rate']):.2f}%</td>
<td>{float(row['mean_blur_gain_p_gained_tokens']):.6f}</td><td>{100 * float(row['probability_drop_rate']):.2f}%</td>
<td>{100 * float(row['top1_changed_rate']):.2f}%</td>
</tr>"""


def _report_css() -> str:
    return """
:root { color-scheme: light dark; --bg:#f7f8fa; --surface:#ffffff; --fg:#18202a; --muted:#66717f; --border:#d8dde4; --gain:#167d72; --drop:#c24132; --link:#245da8; }
@media (prefers-color-scheme: dark) { :root { --bg:#111418; --surface:#191e24; --fg:#e7eaf0; --muted:#a8b0bc; --border:#343c47; --gain:#4fc7b7; --drop:#f08473; --link:#8cbcff; } }
* { box-sizing:border-box; letter-spacing:0; }
body { margin:0; background:var(--bg); color:var(--fg); font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
main { width:min(1480px,100%); margin:0 auto; padding:20px; }
header { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; padding-bottom:16px; border-bottom:1px solid var(--border); }
h1 { margin:0; font-size:1.6rem; font-weight:500; overflow-wrap:anywhere; }
h2 { margin:0 0 12px; font-size:1.05rem; font-weight:500; }
p { margin:5px 0; }
.muted { color:var(--muted); }
.path { overflow-wrap:anywhere; }
nav { display:flex; flex-wrap:wrap; gap:12px; }
a { color:var(--link); text-underline-offset:3px; }
section { margin-top:24px; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; }
.stat { min-height:78px; padding:12px; border:1px solid var(--border); border-radius:6px; background:var(--surface); }
.stat span { display:block; color:var(--muted); font-size:.82rem; }
.stat strong { display:block; margin-top:8px; font-size:1.22rem; font-weight:500; }
.plot { display:block; width:min(100%,1200px); height:auto; background:#fff; border-radius:6px; }
.table-wrap { width:100%; overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
th,td { padding:8px 9px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }
th { color:var(--muted); font-size:.78rem; font-weight:500; }
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
.image-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
.image-card { margin:0; padding:10px; border:1px solid var(--border); border-radius:6px; background:var(--surface); }
.image-card img { display:block; width:100%; max-height:420px; object-fit:contain; background:#fff; }
.image-card figcaption { display:flex; justify-content:space-between; gap:10px; margin-top:8px; }
.controls { display:flex; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:12px; }
select { min-width:120px; padding:7px 9px; border:1px solid var(--border); border-radius:4px; background:var(--surface); color:var(--fg); }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin-bottom:9px; color:var(--muted); font-size:.84rem; }
.legend span { display:flex; align-items:center; gap:6px; }
.legend i { width:13px; height:13px; border-radius:2px; }
.gain-key { background:color-mix(in srgb,var(--gain) 55%,transparent); }
.drop-key { background:color-mix(in srgb,var(--drop) 55%,transparent); }
.token-ribbon { padding:12px; border:1px solid var(--border); background:var(--surface); border-radius:6px; line-height:2.05; overflow-wrap:anywhere; }
.token-mark { padding:2px 1px; border-radius:2px; white-space:pre-wrap; }
.token-mark.gain { background:color-mix(in srgb,var(--gain) var(--strength),transparent); }
.token-mark.drop { background:color-mix(in srgb,var(--drop) var(--strength),transparent); }
.token-mark.neutral { background:transparent; }
.comparison-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; }
pre { margin:0; padding:14px; border:1px solid var(--border); border-radius:6px; background:var(--surface); white-space:pre-wrap; overflow-wrap:anywhere; font:inherit; }
footer { margin-top:28px; padding-top:14px; border-top:1px solid var(--border); color:var(--muted); font-size:.84rem; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.empty { padding:30px 0; color:var(--muted); }
@media (max-width:760px) { main { padding:14px; } header { display:block; } nav { margin-top:12px; } .comparison-grid { grid-template-columns:1fr; } .image-grid { grid-template-columns:1fr; } }
"""


def _unresolved_failure_count(
    output_root: Path,
    manifest_rows: list[dict[str, object]],
    *,
    allowed_sample_keys: set[str] | None,
) -> int:
    failure_path = output_root / "failures.jsonl"
    if not failure_path.exists():
        return 0
    completed = {str(row["sample_key"]) for row in manifest_rows}
    failed = set()
    for line in failure_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            sample_key = str(record["sample"]["sample_key"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if allowed_sample_keys is not None and sample_key not in allowed_sample_keys:
            continue
        if sample_key not in completed:
            failed.add(sample_key)
    return len(failed)


def _write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _perplexity(mean_logp: float) -> float:
    return math.exp(min(700.0, max(-700.0, -mean_logp)))
