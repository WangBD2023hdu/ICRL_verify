from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .hallu_alignment import (
    align_normalized_text,
    normalize_ocr_text,
    observe_mutations,
    recover_relocated_matches,
    token_alignment_rows,
)
from .progress import ProgressTracker
from .prompts import DEFAULT_PDF_OCR_PROMPT
from .vllm_api import (
    RequestLimiter,
    ThreadLocalClients,
    canonical_prompt_scores,
    generate_response,
    safe_error,
    utc_now,
)


SCHEMA_VERSION = 1
DEFAULT_MODEL = "qwen-4b"
DEFAULT_PROMPT = DEFAULT_PDF_OCR_PROMPT
DEFAULT_PRIVILEGED_INSTRUCTION = "请转写上述文本"


@dataclass(frozen=True)
class PrivilegedProbeSample:
    ordinal: int
    pair_id: str
    image_path: Path
    ground_truth_path: Path
    changes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PrivilegedProbeSummary:
    output_dir: Path
    total_items: int
    completed_items: int
    skipped_items: int
    failed_items: int
    interrupted: bool


def load_release_samples(
    dataset_root: str | Path,
    *,
    limit: int | None = None,
) -> list[PrivilegedProbeSample]:
    root = Path(dataset_root).expanduser().resolve()
    pairs_path = root / "pairs.jsonl"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"pairs.jsonl not found under dataset root: {root}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    samples: list[PrivilegedProbeSample] = []
    with pairs_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            pair_id = str(record.get("pair_id", "")).strip()
            if not pair_id:
                raise ValueError(f"missing pair_id at {pairs_path}:{line_number}")
            image_path = _resolve_release_path(root, str(record.get("edited_image", "")))
            gt_path = _resolve_release_path(root, str(record.get("edited_markdown", "")))
            if not image_path.is_file():
                raise FileNotFoundError(f"edited image is missing for {pair_id}: {image_path}")
            if not gt_path.is_file():
                raise FileNotFoundError(f"edited Markdown GT is missing for {pair_id}: {gt_path}")
            changes = tuple(dict(item) for item in record.get("changes", []))
            samples.append(
                PrivilegedProbeSample(
                    ordinal=len(samples) + 1,
                    pair_id=pair_id,
                    image_path=image_path,
                    ground_truth_path=gt_path,
                    changes=changes,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise RuntimeError(f"dataset contains no samples: {pairs_path}")
    return samples


def run_api_privileged_probe(
    *,
    base_url: str,
    model: str,
    dataset_root: str | Path,
    output_dir: str | Path,
    api_key: str | None = None,
    api_key_env: str = "INF_API_KEY",
    prompt: str = DEFAULT_PROMPT,
    privileged_instruction: str = DEFAULT_PRIVILEGED_INSTRUCTION,
    max_tokens: int = 4096,
    top_logprobs: int = 5,
    seed: int = 7,
    verify_tls: bool = True,
    timeout_seconds: float = 900.0,
    max_retries: int = 5,
    retry_base_seconds: float = 3.0,
    request_interval_seconds: float = 0.0,
    resume: bool = True,
    fail_fast: bool = False,
    limit: int | None = None,
    heartbeat_seconds: float = 30.0,
) -> PrivilegedProbeSummary:
    normalized_base_url = base_url.rstrip("/")
    resolved_key = api_key or os.environ.get(api_key_env)
    if not normalized_base_url:
        raise ValueError("base_url must not be empty")
    if not model.strip():
        raise ValueError("model must not be empty")
    if not resolved_key:
        raise RuntimeError(f"API key is missing; set environment variable {api_key_env}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not privileged_instruction.strip():
        raise ValueError("privileged_instruction must not be empty")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 1 <= top_logprobs <= 20:
        raise ValueError("top_logprobs must be between 1 and 20")
    if timeout_seconds <= 0 or retry_base_seconds <= 0:
        raise ValueError("timeout and retry delay must be positive")
    if max_retries < 0 or request_interval_seconds < 0:
        raise ValueError("retry count and request interval must be non-negative")

    dataset_path = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    samples_root = output_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    samples = load_release_samples(dataset_path, limit=limit)
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "backend": "vllm-openai-chat-completions",
        "base_url": normalized_base_url,
        "model": model,
        "dataset_root": str(dataset_path),
        "output_dir": str(output_root),
        "prompt": prompt,
        "privileged_prompt_template": "{ground_truth}\\n\\n{instruction}",
        "privileged_instruction": privileged_instruction,
        "generation_count_per_sample": 1,
        "fixed_response_scoring_conditions": ["original_image", "privileged_text"],
        "strict_response_id_equality": True,
        "max_tokens": max_tokens,
        "top_logprobs": top_logprobs,
        "seed": seed,
        "verify_tls": verify_tls,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "retry_base_seconds": retry_base_seconds,
        "request_interval_seconds": request_interval_seconds,
        "resume": resume,
        "limit": limit,
    }
    _write_json_atomic(output_root / "config.json", config)

    tracker = ProgressTracker(
        task="qwen-mm-api-privileged-probe",
        total_items=len(samples),
        total_bytes=sum(sample.image_path.stat().st_size for sample in samples),
        shard="single-process/vllm-api",
        heartbeat_seconds=heartbeat_seconds,
    )
    tracker.start()
    clients = ThreadLocalClients(
        base_url=normalized_base_url,
        api_key=resolved_key,
        verify_tls=verify_tls,
        timeout_seconds=timeout_seconds,
    )
    limiter = RequestLimiter(request_interval_seconds)
    completed_now = 0
    skipped = 0
    failed = 0
    interrupted = False
    fatal_error: BaseException | None = None

    try:
        client = clients.get()
        for sample in samples:
            sample_dir = samples_root / f"{sample.ordinal:03d}_{sample.pair_id}"
            result_path = sample_dir / "result.json"
            fingerprint = _sample_fingerprint(sample=sample, config=config)
            if resume and _result_matches(result_path, fingerprint):
                skipped += 1
                tracker.complete_unit(
                    status="skipped",
                    records=1,
                    bytes_count=sample.image_path.stat().st_size,
                    index=sample.ordinal,
                    name=sample.pair_id,
                )
                continue
            sample_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = _run_sample(
                    sample=sample,
                    sample_dir=sample_dir,
                    fingerprint=fingerprint,
                    client=client,
                    model=model,
                    prompt=prompt,
                    privileged_instruction=privileged_instruction,
                    max_tokens=max_tokens,
                    top_logprobs=top_logprobs,
                    seed=seed + sample.ordinal - 1,
                    max_retries=max_retries,
                    retry_base_seconds=retry_base_seconds,
                    request_limiter=limiter,
                    tracker=tracker,
                )
                _write_json_atomic(result_path, result)
                _write_sample_outputs(sample_dir, result)
                completed_now += 1
                tracker.complete_unit(
                    status="accepted",
                    records=len(result["tokens"]),
                    bytes_count=sample.image_path.stat().st_size,
                    index=sample.ordinal,
                    name=sample.pair_id,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve the rest of the batch
                failed += 1
                _append_jsonl(
                    output_root / "failures.jsonl",
                    {
                        "timestamp": utc_now(),
                        "pair_id": sample.pair_id,
                        "exception_type": type(exc).__name__,
                        "message": safe_error(exc),
                    },
                )
                tracker.complete_unit(
                    status="error",
                    records=0,
                    bytes_count=sample.image_path.stat().st_size,
                    index=sample.ordinal,
                    name=f"{sample.pair_id}: {type(exc).__name__}: {safe_error(exc)}",
                )
                if fail_fast:
                    fatal_error = exc
                    break
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:  # noqa: BLE001 - finalize logs before surfacing setup failures
        fatal_error = exc
        tracker.note_error(phase="api-setup-error", name=f"{type(exc).__name__}: {safe_error(exc)}")
    finally:
        clients.close()

    tracker.set_current(index=len(samples), name="aggregate-report", phase="building-report")
    completed_total = 0
    report_error: Exception | None = None
    try:
        report_summary = rebuild_api_privileged_report(output_root)
        completed_total = int(report_summary["completed_samples"])
    except Exception as exc:  # noqa: BLE001 - finish progress before surfacing report errors
        report_error = exc
    tracker.finish(interrupted=interrupted)

    summary = PrivilegedProbeSummary(
        output_dir=output_root,
        total_items=len(samples),
        completed_items=completed_total,
        skipped_items=skipped,
        failed_items=failed,
        interrupted=interrupted,
    )
    _write_json_atomic(output_root / "run_summary.json", asdict(summary) | {"output_dir": str(output_root)})
    if fatal_error is not None:
        raise fatal_error
    if interrupted:
        raise KeyboardInterrupt
    if report_error is not None:
        raise report_error
    return summary


def _run_sample(
    *,
    sample: PrivilegedProbeSample,
    sample_dir: Path,
    fingerprint: str,
    client: Any,
    model: str,
    prompt: str,
    privileged_instruction: str,
    max_tokens: int,
    top_logprobs: int,
    seed: int,
    max_retries: int,
    retry_base_seconds: float,
    request_limiter: RequestLimiter,
    tracker: ProgressTracker,
) -> dict[str, Any]:
    image_copy = sample_dir / f"input{sample.image_path.suffix.lower()}"
    if sample.image_path.resolve() != image_copy.resolve():
        shutil.copy2(sample.image_path, image_copy)
    ground_truth = sample.ground_truth_path.read_text(encoding="utf-8")
    privileged_prompt = f"{ground_truth.rstrip()}\n\n{privileged_instruction.strip()}"
    _write_text_atomic(sample_dir / "ground_truth.md", ground_truth)
    _write_text_atomic(sample_dir / "privileged_prompt.txt", privileged_prompt)

    tracker.set_current(index=sample.ordinal, name=sample.pair_id, phase="generating-response-once")
    generated = generate_response(
        client=client,
        model=model,
        image_path=image_copy,
        prompt=prompt,
        max_tokens=max_tokens,
        top_logprobs=top_logprobs,
        seed=seed,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        request_limiter=request_limiter,
        request_label=f"{sample.pair_id}/generate-once",
    )
    response_text = str(generated["text"])
    response_ids = [int(row["token_id"]) for row in generated["tokens"]]
    if not response_ids:
        raise RuntimeError("model generated no scoreable response token IDs")
    _write_text_atomic(sample_dir / "response.md", response_text)
    _write_json_atomic(sample_dir / "response_ids.json", response_ids)

    tracker.set_current(index=sample.ordinal, name=sample.pair_id, phase="scoring-original-fixed-response")
    original_scores = canonical_prompt_scores(
        client=client,
        model=model,
        image_path=image_copy,
        prompt=prompt,
        response_text=response_text,
        prompt_logprobs=top_logprobs,
        seed=seed,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        request_limiter=request_limiter,
        request_label=f"{sample.pair_id}/original-fixed-response",
        expected_token_ids=response_ids,
    )

    tracker.set_current(index=sample.ordinal, name=sample.pair_id, phase="scoring-privileged-fixed-response")
    teacher_scores = canonical_prompt_scores(
        client=client,
        model=model,
        image_path=None,
        prompt=privileged_prompt,
        response_text=response_text,
        prompt_logprobs=top_logprobs,
        seed=seed,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        request_limiter=request_limiter,
        request_label=f"{sample.pair_id}/privileged-fixed-response",
        expected_token_ids=response_ids,
    )
    rows = _combine_scores(response_ids, original_scores, teacher_scores)
    mutation_observations = _attach_gt_and_mutation_alignment(
        rows=rows,
        response_text=response_text,
        ground_truth=ground_truth,
        changes=sample.changes,
    )
    mutation_rows = _build_mutation_rows(rows, mutation_observations, sample.changes)
    summary = _summarize_rows(rows, mutation_rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "completed_at": utc_now(),
        "fingerprint": fingerprint,
        "pair_id": sample.pair_id,
        "sample": {
            "ordinal": sample.ordinal,
            "source_image": str(sample.image_path),
            "source_ground_truth": str(sample.ground_truth_path),
            "image_copy": str(image_copy),
            "changes": list(sample.changes),
        },
        "protocol": {
            "generation_count": 1,
            "api_call_count": 3,
            "response_ids_reused_for_all_scoring": True,
            "strict_response_id_equality": True,
            "scoring_conditions": ["original_image", "privileged_text"],
            "original_prompt_sha256": _sha256_text(prompt),
            "privileged_prompt_sha256": _sha256_text(privileged_prompt),
            "ground_truth_sha256": _sha256_text(ground_truth),
            "privileged_instruction": privileged_instruction,
            "top_logprobs": top_logprobs,
        },
        "response": {
            "text": response_text,
            "token_ids": response_ids,
            "token_count": len(response_ids),
            "finish_reason": generated["finish_reason"],
        },
        "ground_truth": ground_truth,
        "summary": summary,
        "mutation_observations": mutation_rows,
        "tokens": rows,
    }


def _combine_scores(
    response_ids: Sequence[int],
    original_scores: Sequence[dict[str, object]],
    teacher_scores: Sequence[dict[str, object]],
) -> list[dict[str, Any]]:
    expected_length = len(response_ids)
    if len(original_scores) != expected_length or len(teacher_scores) != expected_length:
        raise RuntimeError(
            "fixed-response score length mismatch: "
            f"ids={expected_length}, original={len(original_scores)}, teacher={len(teacher_scores)}"
        )
    rows: list[dict[str, Any]] = []
    for index, (token_id, original, teacher) in enumerate(
        zip(response_ids, original_scores, teacher_scores)
    ):
        if int(original["token_id"]) != token_id or int(teacher["token_id"]) != token_id:
            raise RuntimeError(f"fixed-response token ID mismatch at index {index}")
        raw_token = str(original["raw_token"])
        teacher_raw = str(teacher["raw_token"])
        if teacher_raw != raw_token:
            raise RuntimeError(
                f"fixed-response decoded token mismatch at index {index}: "
                f"original={raw_token!r}, teacher={teacher_raw!r}"
            )
        logp_original = float(original["log_probability"])
        logp_teacher = float(teacher["log_probability"])
        top_original_id = int(original["top_token_id"])
        top_teacher_id = int(teacher["top_token_id"])
        top_teacher_raw = str(teacher["top_raw_token"])
        if top_teacher_id == token_id:
            teacher_preference = "response_token_top1"
        elif top_teacher_raw == raw_token:
            teacher_preference = "same_surface_different_token_id"
        else:
            teacher_preference = "different_surface_top1"
        original_keeps_response = top_original_id == token_id
        teacher_keeps_response = top_teacher_id == token_id
        if original_keeps_response and teacher_keeps_response:
            top1_transition = "response_top1_both"
        elif not original_keeps_response and teacher_keeps_response:
            top1_transition = "teacher_recovers_response"
        elif original_keeps_response and not teacher_keeps_response:
            top1_transition = "teacher_rejects_response"
        else:
            top1_transition = "response_not_top1_both"
        rows.append(
            {
                "index": index,
                "token_id": token_id,
                "token": str(original["token"]),
                "raw_token": raw_token,
                "p_original": float(original["probability"]),
                "logp_original": logp_original,
                "rank_original": int(original["target_rank"]),
                "top_token_id_original": top_original_id,
                "top_token_original": str(original["top_token"]),
                "top_raw_token_original": str(original["top_raw_token"]),
                "top_p_original": float(original["top_probability"]),
                "top_logp_original": float(original["top_log_probability"]),
                "p_teacher": float(teacher["probability"]),
                "logp_teacher": logp_teacher,
                "rank_teacher": int(teacher["target_rank"]),
                "top_token_id_teacher": top_teacher_id,
                "top_token_teacher": str(teacher["top_token"]),
                "top_raw_token_teacher": top_teacher_raw,
                "top_p_teacher": float(teacher["top_probability"]),
                "top_logp_teacher": float(teacher["top_log_probability"]),
                "delta_p_teacher_minus_original": (
                    float(teacher["probability"]) - float(original["probability"])
                ),
                "delta_logp_teacher_minus_original": logp_teacher - logp_original,
                "probability_increased_with_privileged_info": logp_teacher > logp_original,
                "top1_changed": top_original_id != top_teacher_id,
                "target_is_top_original": top_original_id == token_id,
                "target_is_top_teacher": top_teacher_id == token_id,
                "teacher_top_same_surface_different_id": (
                    top_teacher_id != token_id and top_teacher_raw == raw_token
                ),
                "teacher_preference": teacher_preference,
                "top1_transition": top1_transition,
            }
        )
    return rows


def _attach_gt_and_mutation_alignment(
    *,
    rows: list[dict[str, Any]],
    response_text: str,
    ground_truth: str,
    changes: Sequence[dict[str, Any]],
) -> list[Any]:
    normalized_gt = normalize_ocr_text(ground_truth)
    normalized_response = normalize_ocr_text(response_text)
    alignment = recover_relocated_matches(
        normalized_gt.text,
        normalized_response.text,
        align_normalized_text(normalized_gt.text, normalized_response.text),
    )
    mutation_record = {
        "mutations": [
            {
                "mutation_id": f"m{index + 1:03d}",
                "category": "confusable_character",
                "similarity": "synthetic_confusable",
                "zone": "document",
                "original": str(change.get("origin_ans", "")),
                "replacement": str(change.get("ocr_ans", "")),
                "edited_markdown_span": list(change.get("markdown_span", [])),
            }
            for index, change in enumerate(changes)
        ]
    }
    observations = observe_mutations(
        record=mutation_record,
        variant="edited",
        normalized_ground_truth=normalized_gt,
        normalized_output=normalized_response,
        alignment=alignment,
    )
    labels = token_alignment_rows(
        tokens=rows,
        normalized_output=normalized_response,
        alignment=alignment,
        mutation_observations=observations,
    )
    for row, label in zip(rows, labels):
        for key, value in label.items():
            if key not in {"index", "token_id", "token", "raw_token"}:
                row[key] = value
    return observations


def _build_mutation_rows(
    rows: Sequence[dict[str, Any]],
    observations: Sequence[Any],
    changes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    mutation_rows: list[dict[str, Any]] = []
    for index, (observation, change) in enumerate(zip(observations, changes), start=1):
        mutation_id = f"m{index:03d}"
        token_indices = [
            int(row["index"])
            for row in rows
            if mutation_id in str(row.get("mutation_ids", "")).split(",")
        ]
        token_rows = [rows[token_index] for token_index in token_indices]
        decision = token_rows[0] if token_rows else None
        mutation_rows.append(
            {
                "mutation_id": mutation_id,
                "ocr_ans": str(change.get("ocr_ans", "")),
                "origin_ans": str(change.get("origin_ans", "")),
                "bbox": list(change.get("bbox", [])),
                "markdown_span": list(change.get("markdown_span", [])),
                "predicted": observation.predicted,
                "relation": observation.relation,
                "response_token_indices": token_indices,
                "decision_token_index": int(decision["index"]) if decision else None,
                "decision_token": str(decision["token"]) if decision else "",
                "decision_p_original": float(decision["p_original"]) if decision else None,
                "decision_p_teacher": float(decision["p_teacher"]) if decision else None,
                "decision_delta_logp": (
                    float(decision["delta_logp_teacher_minus_original"])
                    if decision
                    else None
                ),
                "sum_logp_original": sum(float(row["logp_original"]) for row in token_rows),
                "sum_logp_teacher": sum(float(row["logp_teacher"]) for row in token_rows),
                "delta_sum_logp": sum(
                    float(row["delta_logp_teacher_minus_original"]) for row in token_rows
                ),
            }
        )
    return mutation_rows


def _summarize_rows(
    rows: Sequence[dict[str, Any]],
    mutation_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty response")
    deltas = [float(row["delta_logp_teacher_minus_original"]) for row in rows]
    label_counts = _count_values(str(row.get("token_label", "unknown")) for row in rows)
    mutation_counts = _count_values(str(row["relation"]) for row in mutation_rows)
    return {
        "token_count": len(rows),
        "mean_p_original": statistics.fmean(float(row["p_original"]) for row in rows),
        "mean_p_teacher": statistics.fmean(float(row["p_teacher"]) for row in rows),
        "mean_delta_logp_teacher_minus_original": statistics.fmean(deltas),
        "median_delta_logp_teacher_minus_original": statistics.median(deltas),
        "privileged_probability_gain_rate": _mean_bool(
            bool(row["probability_increased_with_privileged_info"]) for row in rows
        ),
        "top1_changed_rate": _mean_bool(bool(row["top1_changed"]) for row in rows),
        "target_top_rate_original": _mean_bool(bool(row["target_is_top_original"]) for row in rows),
        "target_top_rate_teacher": _mean_bool(bool(row["target_is_top_teacher"]) for row in rows),
        "same_surface_different_id_count": sum(
            bool(row["teacher_top_same_surface_different_id"]) for row in rows
        ),
        "teacher_preference_counts": _count_values(
            str(row["teacher_preference"]) for row in rows
        ),
        "top1_transition_counts": _count_values(str(row["top1_transition"]) for row in rows),
        "hallucination_token_count": sum(bool(row.get("is_hallucination")) for row in rows),
        "token_label_counts": label_counts,
        "mutation_relation_counts": mutation_counts,
    }


def rebuild_api_privileged_report(output_dir: str | Path) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    result_paths = sorted((output_root / "samples").glob("*/result.json"))
    if not result_paths:
        raise RuntimeError(f"no completed result.json files under {output_root / 'samples'}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    all_tokens: list[dict[str, Any]] = []
    all_mutations: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for result_path, result in zip(result_paths, results):
        relative_report = result_path.parent.relative_to(output_root) / "report.html"
        sample_rows.append(
            {
                "pair_id": result["pair_id"],
                "report": str(relative_report),
                "finish_reason": result["response"].get("finish_reason"),
                **result["summary"],
            }
        )
        for row in result["tokens"]:
            all_tokens.append({"pair_id": result["pair_id"], **row})
        for row in result["mutation_observations"]:
            all_mutations.append({"pair_id": result["pair_id"], **row})

    _write_csv(output_root / "token_probabilities.csv", all_tokens)
    _write_csv(output_root / "mutation_probabilities.csv", all_mutations)
    _write_csv(output_root / "sample_summary.csv", sample_rows)
    global_summary = _summarize_global(results, all_tokens, all_mutations)
    _write_json_atomic(output_root / "summary.json", global_summary)
    _write_text_atomic(
        output_root / "report.html",
        _render_aggregate_html(global_summary, sample_rows, all_tokens, all_mutations),
    )
    return global_summary


def _summarize_global(
    results: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    mutations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    deltas = [float(row["delta_logp_teacher_minus_original"]) for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "completed_samples": len(results),
        "total_tokens": len(rows),
        "total_mutations": len(mutations),
        "mean_p_original": _mean(float(row["p_original"]) for row in rows),
        "mean_p_teacher": _mean(float(row["p_teacher"]) for row in rows),
        "mean_delta_logp_teacher_minus_original": _mean(deltas),
        "median_delta_logp_teacher_minus_original": statistics.median(deltas) if deltas else 0.0,
        "privileged_probability_gain_rate": _mean_bool(
            bool(row["probability_increased_with_privileged_info"]) for row in rows
        ),
        "top1_changed_rate": _mean_bool(bool(row["top1_changed"]) for row in rows),
        "target_top_rate_original": _mean_bool(bool(row["target_is_top_original"]) for row in rows),
        "target_top_rate_teacher": _mean_bool(bool(row["target_is_top_teacher"]) for row in rows),
        "same_surface_different_id_count": sum(
            bool(row["teacher_top_same_surface_different_id"]) for row in rows
        ),
        "teacher_preference_counts": _count_values(
            str(row["teacher_preference"]) for row in rows
        ),
        "top1_transition_counts": _count_values(str(row["top1_transition"]) for row in rows),
        "finish_reason_counts": _count_values(
            str(result["response"].get("finish_reason", "unknown")) for result in results
        ),
        "token_label_counts": _count_values(str(row.get("token_label", "unknown")) for row in rows),
        "mutation_relation_counts": _count_values(str(row["relation"]) for row in mutations),
    }


def _write_sample_outputs(sample_dir: Path, result: dict[str, Any]) -> None:
    _write_csv(sample_dir / "token_probabilities.csv", result["tokens"])
    _write_csv(sample_dir / "mutation_probabilities.csv", result["mutation_observations"])
    _write_text_atomic(sample_dir / "report.html", _render_sample_html(result))


def _render_sample_html(result: dict[str, Any]) -> str:
    rows = result["tokens"]
    chart_data = [
        {
            "i": row["index"],
            "token": row["token"],
            "po": row["p_original"],
            "pt": row["p_teacher"],
            "d": row["delta_logp_teacher_minus_original"],
        }
        for row in rows
    ]
    token_rows = "".join(_token_table_row(row) for row in rows)
    original_strip = "".join(_probability_span(row, "p_original") for row in rows)
    teacher_strip = "".join(_probability_span(row, "p_teacher") for row in rows)
    delta_strip = "".join(_delta_span(row) for row in rows)
    mutation_rows = "".join(_mutation_table_row(row) for row in result["mutation_observations"])
    image_name = Path(result["sample"]["image_copy"]).name
    summary = result["summary"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(result['pair_id'])} privileged probe</title>{_report_css()}</head>
<body><main>
<nav><a href="../../report.html">汇总报告</a><a href="token_probabilities.csv">Token CSV</a><a href="mutation_probabilities.csv">变异 CSV</a></nav>
<h1>{html.escape(result['pair_id'])}</h1>
<div class="stats">{_stat('Tokens', summary['token_count'])}{_stat('Mean p original', summary['mean_p_original'])}{_stat('Mean p teacher', summary['mean_p_teacher'])}{_stat('Mean Δlogp', summary['mean_delta_logp_teacher_minus_original'], signed=True)}{_stat('Top-1 changed', summary['top1_changed_rate'], percent=True)}</div>
<section class="source-grid"><figure><img src="{html.escape(image_name)}" alt="document page"><figcaption>输入图片</figcaption></figure><div><h2>模型 Response</h2><pre>{html.escape(result['response']['text'])}</pre><details><summary>Ground Truth</summary><pre>{html.escape(result['ground_truth'])}</pre></details></div></section>
<section><h2>逐 Token 概率</h2><canvas id="prob-chart" width="1600" height="360"></canvas><h3>原图条件</h3><div class="token-strip">{original_strip}</div><h3>特权信息条件</h3><div class="token-strip">{teacher_strip}</div><h3>Δlogp = teacher - original</h3><div class="token-strip">{delta_strip}</div></section>
<section><h2>人工变异位置</h2><div class="table-scroll"><table><thead><tr><th>ID</th><th>图片文字</th><th>原词</th><th>模型读回</th><th>关系</th><th>Decision token</th><th>p original</th><th>p teacher</th><th>Δlogp</th></tr></thead><tbody>{mutation_rows}</tbody></table></div></section>
<section><h2>Token 明细</h2><div class="table-scroll"><table><thead><tr><th>#</th><th>ID</th><th>Token</th><th>GT 标签</th><th>p original</th><th>p teacher</th><th>Δp</th><th>Δlogp</th><th>Original top-1</th><th>Original top-1 ID</th><th>Teacher top-1</th><th>Teacher top-1 ID</th><th>Teacher top-1 p</th><th>Top-1 changed</th><th>解释</th><th>Top-1 transition</th></tr></thead><tbody>{token_rows}</tbody></table></div></section>
</main><script>const DATA={json.dumps(chart_data, ensure_ascii=False)};{_chart_javascript()}</script></body></html>"""


def _render_aggregate_html(
    summary: dict[str, Any],
    sample_rows: Sequence[dict[str, Any]],
    tokens: Sequence[dict[str, Any]],
    mutations: Sequence[dict[str, Any]],
) -> str:
    sample_table = "".join(
        "<tr>"
        f"<td><a href='{html.escape(str(row['report']))}'>{html.escape(str(row['pair_id']))}</a></td>"
        f"<td>{int(row['token_count'])}</td>"
        f"<td>{float(row['mean_p_original']):.6f}</td>"
        f"<td>{float(row['mean_p_teacher']):.6f}</td>"
        f"<td class='{_delta_class(float(row['mean_delta_logp_teacher_minus_original']))}'>{float(row['mean_delta_logp_teacher_minus_original']):+.4f}</td>"
        f"<td>{100 * float(row['top1_changed_rate']):.2f}%</td>"
        f"<td>{html.escape(str(row.get('finish_reason', '')))}</td>"
        "</tr>"
        for row in sample_rows
    )
    mutation_table = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['pair_id']))}</td>"
        f"<td>{html.escape(str(row['ocr_ans']))}</td>"
        f"<td>{html.escape(str(row['origin_ans']))}</td>"
        f"<td>{html.escape(str(row['predicted']))}</td>"
        f"<td>{html.escape(str(row['relation']))}</td>"
        f"<td>{_optional_float(row.get('decision_p_original'))}</td>"
        f"<td>{_optional_float(row.get('decision_p_teacher'))}</td>"
        f"<td class='{_delta_class(row.get('decision_delta_logp'))}'>{_optional_float(row.get('decision_delta_logp'), signed=True)}</td>"
        "</tr>"
        for row in mutations
    )
    strongest = sorted(
        tokens,
        key=lambda row: abs(float(row["delta_logp_teacher_minus_original"])),
        reverse=True,
    )[:100]
    strongest_table = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['pair_id']))}</td><td>{int(row['index'])}</td>"
        f"<td>{html.escape(str(row['token']))}</td><td>{html.escape(str(row.get('token_label', '')))}</td>"
        f"<td>{float(row['p_original']):.6f}</td><td>{float(row['p_teacher']):.6f}</td>"
        f"<td class='{_delta_class(float(row['delta_logp_teacher_minus_original']))}'>{float(row['delta_logp_teacher_minus_original']):+.4f}</td>"
        f"<td>{html.escape(str(row['top_token_teacher']))}</td>"
        f"<td>{int(row['top_token_id_teacher'])}</td><td>{float(row['top_p_teacher']):.6f}</td>"
        f"<td>{html.escape(str(row['teacher_preference']))}</td>"
        f"<td>{html.escape(str(row['top1_transition']))}</td></tr>"
        for row in strongest
    )
    alternative_rows = [
        row for row in tokens if str(row["teacher_preference"]) != "response_token_top1"
    ]
    alternative_rows.sort(
        key=lambda row: (
            str(row["teacher_preference"]) != "different_surface_top1",
            float(row["p_teacher"]),
        )
    )
    alternatives_table = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['pair_id']))}</td><td>{int(row['index'])}</td>"
        f"<td>{html.escape(str(row['token']))}</td><td>{int(row['token_id'])}</td>"
        f"<td>{float(row['p_original']):.6f}</td><td>{float(row['p_teacher']):.6f}</td>"
        f"<td>{html.escape(str(row['top_token_teacher']))}</td>"
        f"<td>{int(row['top_token_id_teacher'])}</td><td>{float(row['top_p_teacher']):.6f}</td>"
        f"<td>{html.escape(str(row['teacher_preference']))}</td>"
        f"<td>{html.escape(str(row['top1_transition']))}</td></tr>"
        for row in alternative_rows[:500]
    )
    scatter = [
        {"x": float(row["p_original"]), "y": float(row["p_teacher"]), "label": str(row.get("token_label", ""))}
        for row in _even_sample(tokens, 12000)
    ]
    deltas = [float(row["delta_logp_teacher_minus_original"]) for row in tokens]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Privileged Response Token Probe</title>{_report_css()}</head><body><main>
<nav><a href="summary.json">Summary JSON</a><a href="sample_summary.csv">Sample CSV</a><a href="token_probabilities.csv">Token CSV</a><a href="mutation_probabilities.csv">Mutation CSV</a></nav>
<h1>特权信息固定 Response 概率分析</h1>
<div class="stats">{_stat('Samples', summary['completed_samples'])}{_stat('Tokens', summary['total_tokens'])}{_stat('Mutations', summary['total_mutations'])}{_stat('Mean p original', summary['mean_p_original'])}{_stat('Mean p teacher', summary['mean_p_teacher'])}{_stat('Mean Δlogp', summary['mean_delta_logp_teacher_minus_original'], signed=True)}{_stat('Top-1 changed', summary['top1_changed_rate'], percent=True)}{_stat('Same text / different ID', summary['teacher_preference_counts'].get('same_surface_different_token_id', 0))}{_stat('Teacher different text', summary['teacher_preference_counts'].get('different_surface_top1', 0))}{_stat('Teacher recovers response', summary['top1_transition_counts'].get('teacher_recovers_response', 0))}{_stat('Teacher rejects response', summary['top1_transition_counts'].get('teacher_rejects_response', 0))}</div>
<section class="chart-grid"><div><h2>p original vs p teacher</h2><canvas id="scatter" width="720" height="520"></canvas></div><div><h2>Δlogp 分布</h2><canvas id="histogram" width="720" height="520"></canvas></div></section>
<section><h2>逐样本</h2><div class="table-scroll"><table><thead><tr><th>Sample</th><th>Tokens</th><th>Mean p original</th><th>Mean p teacher</th><th>Mean Δlogp</th><th>Top-1 changed</th><th>Finish</th></tr></thead><tbody>{sample_table}</tbody></table></div></section>
<section><h2>{summary['total_mutations']} 个定向变异位置</h2><div class="table-scroll"><table><thead><tr><th>Sample</th><th>图片文字</th><th>原词</th><th>模型读回</th><th>关系</th><th>p original</th><th>p teacher</th><th>Δlogp</th></tr></thead><tbody>{mutation_table}</tbody></table></div></section>
<section><h2>教师 Top-1 不再是 Response token</h2><div class="table-scroll"><table><thead><tr><th>Sample</th><th>#</th><th>Response token</th><th>Response ID</th><th>p original</th><th>p teacher</th><th>Teacher top-1</th><th>Top-1 ID</th><th>Top-1 p</th><th>解释</th><th>Top-1 transition</th></tr></thead><tbody>{alternatives_table}</tbody></table></div></section>
<section><h2>|Δlogp| 最大的 Token</h2><div class="table-scroll"><table><thead><tr><th>Sample</th><th>#</th><th>Token</th><th>GT 标签</th><th>p original</th><th>p teacher</th><th>Δlogp</th><th>Teacher top-1</th><th>Top-1 ID</th><th>Top-1 p</th><th>解释</th><th>Top-1 transition</th></tr></thead><tbody>{strongest_table}</tbody></table></div></section>
</main><script>const SCATTER={json.dumps(scatter, ensure_ascii=False)};const DELTAS={json.dumps(deltas)};{_aggregate_chart_javascript()}</script></body></html>"""


def _token_table_row(row: dict[str, Any]) -> str:
    delta = float(row["delta_logp_teacher_minus_original"])
    return (
        "<tr>"
        f"<td>{int(row['index'])}</td><td>{int(row['token_id'])}</td>"
        f"<td class='token'>{html.escape(str(row['token']))}</td>"
        f"<td>{html.escape(str(row.get('token_label', '')))}</td>"
        f"<td>{float(row['p_original']):.6f}</td><td>{float(row['p_teacher']):.6f}</td>"
        f"<td>{float(row['delta_p_teacher_minus_original']):+.6f}</td>"
        f"<td class='{_delta_class(delta)}'>{delta:+.4f}</td>"
        f"<td class='token'>{html.escape(str(row['top_token_original']))}</td>"
        f"<td>{int(row['top_token_id_original'])}</td>"
        f"<td class='token'>{html.escape(str(row['top_token_teacher']))}</td>"
        f"<td>{int(row['top_token_id_teacher'])}</td>"
        f"<td>{float(row['top_p_teacher']):.6f}</td>"
        f"<td>{'yes' if row['top1_changed'] else 'no'}</td>"
        f"<td>{html.escape(str(row['teacher_preference']))}</td>"
        f"<td>{html.escape(str(row['top1_transition']))}</td></tr>"
    )


def _mutation_table_row(row: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(str(row['mutation_id']))}</td>"
        f"<td>{html.escape(str(row['ocr_ans']))}</td>"
        f"<td>{html.escape(str(row['origin_ans']))}</td>"
        f"<td>{html.escape(str(row['predicted']))}</td>"
        f"<td>{html.escape(str(row['relation']))}</td>"
        f"<td class='token'>{html.escape(str(row['decision_token']))}</td>"
        f"<td>{_optional_float(row.get('decision_p_original'))}</td>"
        f"<td>{_optional_float(row.get('decision_p_teacher'))}</td>"
        f"<td class='{_delta_class(row.get('decision_delta_logp'))}'>{_optional_float(row.get('decision_delta_logp'), signed=True)}</td>"
        "</tr>"
    )


def _probability_span(row: dict[str, Any], key: str) -> str:
    value = min(1.0, max(0.0, float(row[key])))
    hue = 5.0 + 120.0 * value
    title = (
        f"#{row['index']} {row['token']} | {key}={value:.6f} | "
        f"delta_logp={float(row['delta_logp_teacher_minus_original']):+.4f}"
    )
    return f"<span class='heat-token' style='background:hsl({hue:.1f} 68% 78%)' title='{html.escape(title)}'>{html.escape(str(row['token']))}</span>"


def _delta_span(row: dict[str, Any]) -> str:
    value = float(row["delta_logp_teacher_minus_original"])
    magnitude = min(1.0, abs(value) / 3.0)
    hue = 145 if value >= 0 else 2
    lightness = 94 - 35 * magnitude
    title = f"#{row['index']} {row['token']} | delta_logp={value:+.6f}"
    return f"<span class='heat-token' style='background:hsl({hue} 62% {lightness:.1f}%)' title='{html.escape(title)}'>{html.escape(str(row['token']))}</span>"


def _report_css() -> str:
    return """<style>
:root{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:#1d2528;background:#f5f6f4;line-height:1.45}*{box-sizing:border-box}body{margin:0}main{max-width:1540px;margin:auto;padding:24px}nav{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px}a{color:#075b70}h1{font-size:28px;margin:0 0 18px;letter-spacing:0}h2{font-size:19px;margin:28px 0 12px;letter-spacing:0}h3{font-size:14px;margin:16px 0 6px;letter-spacing:0;color:#4c5b60}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}.stat{background:#fff;border:1px solid #d9dedc;border-radius:6px;padding:12px}.stat span{display:block;color:#617075;font-size:12px}.stat strong{font-size:21px}.source-grid,.chart-grid{display:grid;grid-template-columns:minmax(300px,0.8fr) minmax(420px,1.2fr);gap:24px;align-items:start}figure{margin:0}figure img{width:100%;max-height:900px;object-fit:contain;background:#fff;border:1px solid #d9dedc}figcaption{font-size:12px;color:#617075}pre{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid #d9dedc;padding:14px;max-height:520px;overflow:auto}canvas{width:100%;height:auto;background:#fff;border:1px solid #d9dedc}.token-strip{display:flex;flex-wrap:wrap;gap:2px;background:#fff;border:1px solid #d9dedc;padding:8px}.heat-token{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;padding:2px 3px;border-radius:3px;white-space:pre}.table-scroll{overflow:auto;background:#fff;border:1px solid #d9dedc}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px 9px;border-bottom:1px solid #e7eae8;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#eaf0ed;z-index:1}td:first-child,th:first-child{text-align:left}.token{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;text-align:left}.gain{color:#08764a;font-weight:650}.drop{color:#b02b2b;font-weight:650}details{margin-top:12px}summary{cursor:pointer;color:#075b70}@media(max-width:850px){main{padding:14px}.source-grid,.chart-grid{grid-template-columns:1fr}}
</style>"""


def _chart_javascript() -> str:
    return """
function setup(canvas){const dpr=window.devicePixelRatio||1;const w=canvas.clientWidth||800;canvas.width=w*dpr;canvas.height=360*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);return {c,w,h:360}}
function draw(){const {c,w,h}=setup(document.getElementById('prob-chart'));c.clearRect(0,0,w,h);const pad={l:48,r:16,t:18,b:34};const pw=w-pad.l-pad.r,ph=210;const x=i=>pad.l+(DATA.length<=1?0:i/(DATA.length-1))*pw;const y=p=>pad.t+(1-p)*ph;c.strokeStyle='#cad1ce';for(let q=0;q<=4;q++){const yy=y(q/4);c.beginPath();c.moveTo(pad.l,yy);c.lineTo(w-pad.r,yy);c.stroke();c.fillStyle='#59686d';c.fillText((q/4).toFixed(2),6,yy+4)}function line(key,color){c.strokeStyle=color;c.lineWidth=1.5;c.beginPath();DATA.forEach((d,i)=>{const xx=x(i),yy=y(d[key]);i?c.lineTo(xx,yy):c.moveTo(xx,yy)});c.stroke()}line('po','#166b8f');line('pt','#a33a55');c.fillStyle='#166b8f';c.fillText('original',pad.l,12);c.fillStyle='#a33a55';c.fillText('teacher',pad.l+62,12);const base=330,scale=Math.max(1,...DATA.map(d=>Math.abs(d.d)));DATA.forEach((d,i)=>{const xx=x(i),bh=Math.min(74,Math.abs(d.d)/scale*74);c.fillStyle=d.d>=0?'#238a63':'#c94b4b';c.fillRect(xx, d.d>=0?base-bh:base, Math.max(1,pw/Math.max(DATA.length,1)), bh)});c.strokeStyle='#59686d';c.beginPath();c.moveTo(pad.l,base);c.lineTo(w-pad.r,base);c.stroke();c.fillStyle='#59686d';c.fillText('Δlogp',6,base+4)}window.addEventListener('resize',draw);draw();
"""


def _aggregate_chart_javascript() -> str:
    return """
function ctx(id){const el=document.getElementById(id),dpr=window.devicePixelRatio||1,w=el.clientWidth||650;el.width=w*dpr;el.height=480*dpr;const c=el.getContext('2d');c.scale(dpr,dpr);return {c,w,h:480}}
function axes(c,w,h){c.strokeStyle='#bec8c4';c.beginPath();c.moveTo(45,15);c.lineTo(45,h-38);c.lineTo(w-15,h-38);c.stroke()}
function scatter(){const {c,w,h}=ctx('scatter');axes(c,w,h);const sx=x=>45+x*(w-60),sy=y=>15+(1-y)*(h-53);c.strokeStyle='#9aa7a2';c.setLineDash([4,4]);c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(1),sy(1));c.stroke();c.setLineDash([]);SCATTER.forEach(p=>{c.fillStyle=p.label==='correct'?'rgba(30,115,87,.24)':'rgba(190,54,54,.48)';c.fillRect(sx(p.x),sy(p.y),2,2)});c.fillStyle='#45545a';c.fillText('p original',w/2,h-12);c.save();c.translate(13,h/2);c.rotate(-Math.PI/2);c.fillText('p teacher',0,0);c.restore()}
function histogram(){const {c,w,h}=ctx('histogram');axes(c,w,h);if(!DELTAS.length)return;const ordered=[...DELTAS].sort((a,b)=>a-b),lo=ordered[Math.floor(.01*(ordered.length-1))],hi=ordered[Math.floor(.99*(ordered.length-1))],min=Math.min(-.01,lo),max=Math.max(.01,hi),bins=50,count=Array(bins).fill(0);DELTAS.forEach(v=>{const cl=Math.max(min,Math.min(max,v));count[Math.min(bins-1,Math.floor((cl-min)/(max-min)*bins))]++});const peak=Math.max(...count,1),bw=(w-60)/bins;count.forEach((n,i)=>{const bh=n/peak*(h-70);const center=min+(i+.5)/bins*(max-min);c.fillStyle=center>=0?'#3b9873':'#cf6262';c.fillRect(45+i*bw,h-38-bh,Math.max(1,bw-1),bh)});const zx=45+(0-min)/(max-min)*(w-60);c.strokeStyle='#34454b';c.beginPath();c.moveTo(zx,15);c.lineTo(zx,h-38);c.stroke();c.fillStyle='#45545a';c.fillText(min.toFixed(2),45,h-18);c.fillText(max.toFixed(2),w-48,h-18);c.fillText('Δlogp teacher - original',w/2-70,h-5)}function draw(){scatter();histogram()}window.addEventListener('resize',draw);draw();
"""


def _stat(label: str, value: Any, *, percent: bool = False, signed: bool = False) -> str:
    if isinstance(value, float):
        rendered = f"{100 * value:.2f}%" if percent else (f"{value:+.4f}" if signed else f"{value:.6f}")
    else:
        rendered = str(value)
    return f"<div class='stat'><span>{html.escape(label)}</span><strong>{html.escape(rendered)}</strong></div>"


def _delta_class(value: Any) -> str:
    if value is None:
        return ""
    return "gain" if float(value) >= 0 else "drop"


def _optional_float(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}" if signed else f"{float(value):.6f}"


def _even_sample(values: Sequence[Any], maximum: int) -> list[Any]:
    if len(values) <= maximum:
        return list(values)
    return [values[min(len(values) - 1, int(index * len(values) / maximum))] for index in range(maximum)]


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def _mean_bool(values: Iterable[bool]) -> float:
    materialized = [1.0 if value else 0.0 for value in values]
    return statistics.fmean(materialized) if materialized else 0.0


def _resolve_release_path(root: Path, value: str) -> Path:
    if not value:
        raise ValueError("release path must not be empty")
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"release path escapes dataset root: {value}")
    return resolved


def _sample_fingerprint(*, sample: PrivilegedProbeSample, config: dict[str, Any]) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": sample.pair_id,
        "image_path": str(sample.image_path),
        "image_size": sample.image_path.stat().st_size,
        "image_mtime_ns": sample.image_path.stat().st_mtime_ns,
        "ground_truth_sha256": _sha256_text(sample.ground_truth_path.read_text(encoding="utf-8")),
        "changes": list(sample.changes),
        "model": config["model"],
        "prompt": config["prompt"],
        "privileged_instruction": config["privileged_instruction"],
        "max_tokens": config["max_tokens"],
        "top_logprobs": config["top_logprobs"],
        "seed": int(config["seed"]) + sample.ordinal - 1,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _result_matches(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fingerprint") == fingerprint
    except (OSError, json.JSONDecodeError):
        return False


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-mm-api-privileged-probe",
        description=(
            "Generate OCR once, then score the identical response token IDs under "
            "the original image prompt and a standalone GT privileged prompt."
        ),
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="INF_API_KEY")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file")
    parser.add_argument("--privileged-instruction", default=DEFAULT_PRIVILEGED_INSTRUCTION)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=3.0)
    parser.add_argument("--request-interval-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-verify-tls", action="store_true")
    parser.add_argument(
        "--rebuild-report-only",
        action="store_true",
        help="Rebuild CSV and HTML files from existing sample result.json files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.rebuild_report_only:
        summary = rebuild_api_privileged_report(args.output_dir)
        print(
            f"Rebuilt report: {Path(args.output_dir).expanduser().resolve() / 'report.html'} "
            f"samples={summary['completed_samples']} tokens={summary['total_tokens']}",
            flush=True,
        )
        return 0
    if not args.base_url:
        parser.error("--base-url is required unless --rebuild-report-only is used")
    if not args.dataset_root:
        parser.error("--dataset-root is required unless --rebuild-report-only is used")
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    summary = run_api_privileged_probe(
        base_url=args.base_url,
        model=args.model,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        prompt=prompt,
        privileged_instruction=args.privileged_instruction,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
        seed=args.seed,
        verify_tls=not args.no_verify_tls,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        request_interval_seconds=args.request_interval_seconds,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
        limit=args.limit,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    print(
        f"Privileged probe complete: output={summary.output_dir} "
        f"completed={summary.completed_items}/{summary.total_items} "
        f"skipped={summary.skipped_items} failed={summary.failed_items}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
