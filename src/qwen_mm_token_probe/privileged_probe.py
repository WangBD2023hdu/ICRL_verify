from __future__ import annotations

import argparse
import csv
import hashlib
import html
import inspect
import json
import math
import shutil
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hallu_alignment import (
    align_normalized_text,
    normalize_ocr_text,
    observe_mutations,
    recover_relocated_matches,
    token_alignment_rows,
)
from .hf_qwen import (
    ModelBundle,
    append_generated_tokens,
    decode_generated_tokens,
    decode_token_piece,
    generate_from_prompt,
    load_model_bundle,
    move_inputs_to_device,
    prepare_prompt_inputs,
)
from .progress import ProgressTracker
from .prompts import DEFAULT_PDF_OCR_PROMPT

SCHEMA_VERSION = 2
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
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
            image_path = _resolve_release_path(
                root, str(record.get("edited_image", ""))
            )
            gt_path = _resolve_release_path(
                root, str(record.get("edited_markdown", ""))
            )
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"edited image is missing for {pair_id}: {image_path}"
                )
            if not gt_path.is_file():
                raise FileNotFoundError(
                    f"edited Markdown GT is missing for {pair_id}: {gt_path}"
                )
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


def run_privileged_probe(
    *,
    model_id: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    prompt: str = DEFAULT_PROMPT,
    privileged_instruction: str = DEFAULT_PRIVILEGED_INSTRUCTION,
    max_new_tokens: int = 4096,
    top_k: int = 5,
    forward_chunk_size: int = 16,
    device_map: str | None = "auto",
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    min_pixels: int = 2048,
    max_pixels: int = 16777216,
    image_patch_size: int = 16,
    seed: int = 7,
    resume: bool = True,
    fail_fast: bool = False,
    limit: int | None = None,
    heartbeat_seconds: float = 30.0,
) -> PrivilegedProbeSummary:
    """Generate once and score the exact generated IDs in two offline contexts."""

    if not str(model_id).strip():
        raise ValueError("model_id must not be empty")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not privileged_instruction.strip():
        raise ValueError("privileged_instruction must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not 2 <= top_k <= 50:
        raise ValueError("top_k must be between 2 and 50 so Top-1/Top-2 are available")
    if forward_chunk_size <= 0:
        raise ValueError("forward_chunk_size must be positive")
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("pixel limits must be positive and max_pixels >= min_pixels")
    if image_patch_size <= 0:
        raise ValueError("image_patch_size must be positive")

    dataset_path = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    samples_root = output_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    samples = load_release_samples(dataset_path, limit=limit)
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "backend": "huggingface-transformers-offline",
        "model_id": str(model_id),
        "dataset_root": str(dataset_path),
        "output_dir": str(output_root),
        "prompt": prompt,
        "privileged_prompt_template": "{ground_truth}\\n\\n{instruction}",
        "privileged_instruction": privileged_instruction,
        "generation_count_per_sample": 1,
        "teacher_forced_forward_count_per_sample": 2,
        "fixed_response_scoring_conditions": ["original_image", "privileged_text"],
        "response_ids_directly_concatenated": True,
        "response_text_retokenized": False,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "forward_chunk_size": forward_chunk_size,
        "device_map": device_map,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "image_patch_size": image_patch_size,
        "enable_thinking": False,
        "seed": seed,
        "resume": resume,
        "limit": limit,
    }
    _write_json_atomic(output_root / "config.json", config)

    tracker = ProgressTracker(
        task="qwen-mm-privileged-probe",
        total_items=len(samples),
        total_bytes=sum(sample.image_path.stat().st_size for sample in samples),
        shard="single-process/huggingface-transformers",
        heartbeat_seconds=heartbeat_seconds,
    )
    tracker.start()
    tracker.set_current(index=0, name=str(model_id), phase="loading-local-model")
    try:
        model_bundle = load_model_bundle(
            str(model_id),
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        tracker.note_error(
            phase="model-load-error",
            name=f"{type(exc).__name__}: {_safe_error(exc)}",
        )
        tracker.finish(interrupted=False)
        raise

    completed_now = 0
    skipped = 0
    failed = 0
    interrupted = False
    fatal_error: BaseException | None = None
    try:
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
                    model_bundle=model_bundle,
                    prompt=prompt,
                    privileged_instruction=privileged_instruction,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    forward_chunk_size=forward_chunk_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    image_patch_size=image_patch_size,
                    seed=seed + sample.ordinal - 1,
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
                        "timestamp": _utc_now(),
                        "pair_id": sample.pair_id,
                        "exception_type": type(exc).__name__,
                        "message": _safe_error(exc),
                    },
                )
                tracker.complete_unit(
                    status="error",
                    records=0,
                    bytes_count=sample.image_path.stat().st_size,
                    index=sample.ordinal,
                    name=f"{sample.pair_id}: {type(exc).__name__}: {_safe_error(exc)}",
                )
                if fail_fast:
                    fatal_error = exc
                    break
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:  # noqa: BLE001 - finalize progress before surfacing
        fatal_error = exc
        tracker.note_error(
            phase="offline-probe-error",
            name=f"{type(exc).__name__}: {_safe_error(exc)}",
        )

    tracker.set_current(
        index=len(samples), name="aggregate-report", phase="building-report"
    )
    completed_total = 0
    report_error: Exception | None = None
    try:
        report_summary = rebuild_privileged_report(output_root)
        completed_total = int(report_summary["completed_samples"])
    except Exception as exc:  # noqa: BLE001 - report errors must not hide run status
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
    _write_json_atomic(
        output_root / "run_summary.json",
        asdict(summary)
        | {
            "output_dir": str(output_root),
            "completed_now": completed_now,
        },
    )
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
    model_bundle: ModelBundle,
    prompt: str,
    privileged_instruction: str,
    max_new_tokens: int,
    top_k: int,
    forward_chunk_size: int,
    min_pixels: int,
    max_pixels: int,
    image_patch_size: int,
    seed: int,
    tracker: ProgressTracker,
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_copy = sample_dir / f"input{sample.image_path.suffix.lower()}"
    partial_path = sample_dir / "partial.json"
    if sample.image_path.resolve() != image_copy.resolve():
        shutil.copy2(sample.image_path, image_copy)
    ground_truth = sample.ground_truth_path.read_text(encoding="utf-8")
    privileged_prompt = f"{ground_truth}\n\n{privileged_instruction}"
    _write_text_atomic(sample_dir / "ground_truth.md", ground_truth)
    _write_text_atomic(sample_dir / "privileged_prompt.txt", privileged_prompt)

    tracker.set_current(
        index=sample.ordinal,
        name=sample.pair_id,
        phase="preparing-original-multimodal-prompt",
    )
    original_inputs = prepare_prompt_inputs(
        processor=model_bundle.processor,
        image_path=image_copy,
        prompt=prompt,
        device=model_bundle.device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_patch_size=image_patch_size,
        enable_thinking=False,
    )
    tracker.set_current(
        index=sample.ordinal,
        name=sample.pair_id,
        phase="preparing-privileged-text-prompt",
    )
    teacher_inputs = _prepare_text_prompt_inputs(
        model_bundle=model_bundle,
        prompt=privileged_prompt,
    )

    partial = _load_partial(partial_path, fingerprint)
    generation_performed = False
    if partial is None:
        tracker.set_current(
            index=sample.ordinal,
            name=sample.pair_id,
            phase="generating-response-once",
        )
        _seed_everything(seed)
        response_ids, _ = generate_from_prompt(
            model=model_bundle.model,
            tokenizer=model_bundle.tokenizer,
            prompt_inputs=original_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        if not response_ids:
            raise RuntimeError("model generated no scoreable response token IDs")
        response_text = decode_generated_tokens(model_bundle.tokenizer, response_ids)
        finish_reason = "length" if len(response_ids) >= max_new_tokens else "stop"
        generation_performed = True
        partial = {
            "schema_version": SCHEMA_VERSION,
            "partial": True,
            "fingerprint": fingerprint,
            "pair_id": sample.pair_id,
            "generation_count": 1,
            "response_ids": response_ids,
            "response_text": response_text,
            "finish_reason": finish_reason,
            "forwards": {},
        }
        _write_json_atomic(partial_path, partial)
    else:
        response_ids = [int(value) for value in partial["response_ids"]]
        response_text = str(partial["response_text"])
        finish_reason = str(partial.get("finish_reason", "unknown"))

    _write_text_atomic(sample_dir / "response.md", response_text)
    _write_json_atomic(sample_dir / "response_ids.json", response_ids)

    forwards = dict(partial.get("forwards", {}))
    scoring_contexts = (
        ("original", original_inputs, "scoring-original-exact-response-ids"),
        ("teacher", teacher_inputs, "scoring-privileged-exact-response-ids"),
    )
    for context_name, context_inputs, phase in scoring_contexts:
        if context_name in forwards:
            continue
        tracker.set_current(index=sample.ordinal, name=sample.pair_id, phase=phase)
        forwards[context_name] = _score_fixed_response_ids(
            model_bundle=model_bundle,
            prompt_inputs=context_inputs,
            response_ids=response_ids,
            top_k=top_k,
            chunk_size=forward_chunk_size,
        )
        partial["forwards"] = forwards
        _write_json_atomic(partial_path, partial)
        _empty_device_cache(model_bundle.device)

    original_scores = list(forwards["original"])
    teacher_scores = list(forwards["teacher"])
    rows = _combine_scores(response_ids, original_scores, teacher_scores)
    mutation_observations = _attach_gt_and_mutation_alignment(
        rows=rows,
        response_text=response_text,
        ground_truth=ground_truth,
        changes=sample.changes,
    )
    mutation_rows = _build_mutation_rows(rows, mutation_observations, sample.changes)
    summary = _summarize_rows(rows, mutation_rows)
    reconstructed = "".join(str(row["raw_token"]) for row in rows)

    result = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": _utc_now(),
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
            "backend": "huggingface-transformers-offline",
            "generation_count": 1,
            "generation_performed_this_run": generation_performed,
            "teacher_forced_forward_count": 2,
            "response_ids_reused_for_all_forwards": True,
            "response_ids_directly_concatenated": True,
            "response_text_retokenized": False,
            "scoring_conditions": ["original_image", "privileged_text"],
            "original_prompt_token_count": int(original_inputs["input_ids"].shape[-1]),
            "privileged_prompt_token_count": int(teacher_inputs["input_ids"].shape[-1]),
            "response_ids_sha256": _sha256_token_ids(response_ids),
            "original_prompt_sha256": _sha256_text(prompt),
            "privileged_prompt_sha256": _sha256_text(privileged_prompt),
            "ground_truth_sha256": _sha256_text(ground_truth),
            "privileged_instruction": privileged_instruction,
            "top_k": top_k,
            "forward_chunk_size": forward_chunk_size,
        },
        "response": {
            "text": response_text,
            "token_ids": response_ids,
            "token_count": len(response_ids),
            "finish_reason": finish_reason,
            "piece_reconstruction_matches_response": reconstructed == response_text,
        },
        "ground_truth": ground_truth,
        "summary": summary,
        "mutation_observations": mutation_rows,
        "tokens": rows,
    }
    partial_path.unlink(missing_ok=True)
    return result


def _prepare_text_prompt_inputs(
    *,
    model_bundle: ModelBundle,
    prompt: str,
) -> dict[str, Any]:
    """Build a text-only chat prefix ending at the assistant generation marker."""

    messages = [{"role": "user", "content": prompt}]
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    try:
        encoded = model_bundle.tokenizer.apply_chat_template(
            messages, **template_kwargs
        )
    except TypeError:
        template_kwargs.pop("enable_thinking")
        try:
            encoded = model_bundle.tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        except TypeError:
            rendered = model_bundle.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            encoded = model_bundle.tokenizer(
                rendered,
                add_special_tokens=False,
                return_tensors="pt",
            )
    if not hasattr(encoded, "items"):
        encoded = {
            "input_ids": encoded,
            "attention_mask": _ones_like(encoded),
        }
    return move_inputs_to_device(dict(encoded), model_bundle.device)


def _score_fixed_response_ids(
    *,
    model_bundle: ModelBundle,
    prompt_inputs: dict[str, Any],
    response_ids: Sequence[int],
    top_k: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    """Append response IDs directly and score them without decoding/re-tokenizing."""

    import torch

    if not response_ids:
        raise ValueError("response_ids must not be empty")
    prompt_length = int(prompt_inputs["input_ids"].shape[-1])
    scoring_inputs = append_generated_tokens(prompt_inputs, list(response_ids))
    scoring_inputs.pop("position_ids", None)
    forward_kwargs = dict(scoring_inputs)
    forward_kwargs["use_cache"] = False
    try:
        supports_keep = (
            "logits_to_keep" in inspect.signature(model_bundle.model.forward).parameters
        )
    except (TypeError, ValueError):
        supports_keep = False
    if supports_keep:
        forward_kwargs["logits_to_keep"] = len(response_ids) + 1

    with torch.inference_mode():
        outputs = model_bundle.model(**forward_kwargs)
    logits = outputs.logits[0]
    if supports_keep and logits.shape[0] <= len(response_ids) + 1:
        required = len(response_ids) + 1
        if logits.shape[0] < required:
            raise RuntimeError(
                "model returned too few logits for exact-ID scoring: "
                f"got={logits.shape[0]} required={required}"
            )
        target_logits = logits[-required:-1]
    else:
        target_logits = logits[
            prompt_length - 1 : prompt_length - 1 + len(response_ids)
        ]
    if target_logits.shape[0] != len(response_ids):
        raise RuntimeError(
            "fixed-response logits are not aligned with response IDs: "
            f"logits={target_logits.shape[0]} ids={len(response_ids)}"
        )

    rows: list[dict[str, Any]] = []
    for start in range(0, len(response_ids), chunk_size):
        end = min(len(response_ids), start + chunk_size)
        chunk = target_logits[start:end].float()
        target_tensor = torch.tensor(
            [int(value) for value in response_ids[start:end]],
            dtype=torch.long,
            device=chunk.device,
        )
        log_normalizer = torch.logsumexp(chunk, dim=-1)
        selected_logits = chunk.gather(1, target_tensor[:, None]).squeeze(1)
        selected_logp = selected_logits - log_normalizer
        target_ranks = 1 + (chunk > selected_logits[:, None]).sum(dim=-1)
        top_values, top_ids = torch.topk(
            chunk,
            k=min(top_k, int(chunk.shape[-1])),
            dim=-1,
        )
        top_logp = top_values - log_normalizer[:, None]
        probabilities = torch.softmax(chunk, dim=-1)
        entropies = -(probabilities * torch.log_softmax(chunk, dim=-1)).sum(dim=-1)

        selected_logp_cpu = selected_logp.detach().cpu().tolist()
        ranks_cpu = target_ranks.detach().cpu().tolist()
        entropy_cpu = entropies.detach().cpu().tolist()
        top_ids_cpu = top_ids.detach().cpu().tolist()
        top_logp_cpu = top_logp.detach().cpu().tolist()
        for offset, token_id in enumerate(response_ids[start:end]):
            raw_token = decode_token_piece(model_bundle.tokenizer, int(token_id))
            candidates: list[dict[str, Any]] = []
            for rank, (candidate_id, candidate_logp) in enumerate(
                zip(top_ids_cpu[offset], top_logp_cpu[offset]),
                start=1,
            ):
                candidate_raw = decode_token_piece(
                    model_bundle.tokenizer,
                    int(candidate_id),
                )
                candidates.append(
                    {
                        "rank": rank,
                        "token_id": int(candidate_id),
                        "token": _display_token(candidate_raw),
                        "raw_token": candidate_raw,
                        "probability": math.exp(float(candidate_logp)),
                        "log_probability": float(candidate_logp),
                    }
                )
            target_logp = float(selected_logp_cpu[offset])
            top = candidates[0]
            rows.append(
                {
                    "token_id": int(token_id),
                    "token": _display_token(raw_token),
                    "raw_token": raw_token,
                    "probability": math.exp(target_logp),
                    "log_probability": target_logp,
                    "target_rank": int(ranks_cpu[offset]),
                    "entropy": float(entropy_cpu[offset]),
                    "top_token_id": int(top["token_id"]),
                    "top_token": str(top["token"]),
                    "top_raw_token": str(top["raw_token"]),
                    "top_probability": float(top["probability"]),
                    "top_log_probability": float(top["log_probability"]),
                    "top_candidates": candidates,
                }
            )
        del chunk, probabilities
    del outputs, logits, target_logits
    return rows


def _load_partial(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if value.get("fingerprint") != fingerprint:
        return None
    if not value.get("response_ids"):
        return None
    return value


def _seed_everything(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _empty_device_cache(device: Any) -> None:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif str(device).startswith("mps") and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _ones_like(value: Any) -> Any:
    import torch

    return torch.ones_like(value)


def _display_token(raw_token: str) -> str:
    return raw_token.replace("\n", "\\n").replace("\t", "\\t")


def _combine_scores(
    response_ids: Sequence[int],
    original_scores: Sequence[dict[str, object]],
    teacher_scores: Sequence[dict[str, object]],
) -> list[dict[str, Any]]:
    expected_length = len(response_ids)
    if (
        len(original_scores) != expected_length
        or len(teacher_scores) != expected_length
    ):
        raise RuntimeError(
            "fixed-response score length mismatch: "
            f"ids={expected_length}, original={len(original_scores)}, teacher={len(teacher_scores)}"
        )
    rows: list[dict[str, Any]] = []
    for index, (token_id, original, teacher) in enumerate(
        zip(response_ids, original_scores, teacher_scores)
    ):
        if (
            int(original["token_id"]) != token_id
            or int(teacher["token_id"]) != token_id
        ):
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
                "entropy_original": float(original.get("entropy", 0.0)),
                "top_token_id_original": top_original_id,
                "top_token_original": str(original["top_token"]),
                "top_raw_token_original": str(original["top_raw_token"]),
                "top_p_original": float(original["top_probability"]),
                "top_logp_original": float(original["top_log_probability"]),
                "p_teacher": float(teacher["probability"]),
                "logp_teacher": logp_teacher,
                "rank_teacher": int(teacher["target_rank"]),
                "entropy_teacher": float(teacher.get("entropy", 0.0)),
                "top_token_id_teacher": top_teacher_id,
                "top_token_teacher": str(teacher["top_token"]),
                "top_raw_token_teacher": top_teacher_raw,
                "top_p_teacher": float(teacher["top_probability"]),
                "top_logp_teacher": float(teacher["top_log_probability"]),
                "top_candidates_original": list(original.get("top_candidates", [])),
                "top_candidates_teacher": list(teacher.get("top_candidates", [])),
                "delta_p_teacher_minus_original": (
                    float(teacher["probability"]) - float(original["probability"])
                ),
                "delta_logp_teacher_minus_original": logp_teacher - logp_original,
                "probability_increased_with_privileged_info": logp_teacher
                > logp_original,
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
        response_tokens = [
            {
                "index": int(row["index"]),
                "token_id": int(row["token_id"]),
                "token": str(row["token"]),
                "raw_token": str(row["raw_token"]),
                "p_original": float(row["p_original"]),
                "p_teacher": float(row["p_teacher"]),
                "rank_original": int(row["rank_original"]),
                "rank_teacher": int(row["rank_teacher"]),
                "delta_p_teacher_minus_original": float(
                    row["delta_p_teacher_minus_original"]
                ),
                "delta_logp_teacher_minus_original": float(
                    row["delta_logp_teacher_minus_original"]
                ),
                "top_candidates_original": list(row.get("top_candidates_original", [])),
                "top_candidates_teacher": list(row.get("top_candidates_teacher", [])),
                "top_token_id_original": int(row["top_token_id_original"]),
                "top_token_original": str(row["top_token_original"]),
                "top_p_original": float(row["top_p_original"]),
                "top_token_id_teacher": int(row["top_token_id_teacher"]),
                "top_token_teacher": str(row["top_token_teacher"]),
                "top_p_teacher": float(row["top_p_teacher"]),
                "top1_changed": bool(row["top1_changed"]),
            }
            for row in token_rows
        ]
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
                "response_tokens": response_tokens,
                "decision_token_index": int(decision["index"]) if decision else None,
                "decision_token": str(decision["token"]) if decision else "",
                "decision_p_original": float(decision["p_original"])
                if decision
                else None,
                "decision_p_teacher": float(decision["p_teacher"])
                if decision
                else None,
                "decision_delta_logp": (
                    float(decision["delta_logp_teacher_minus_original"])
                    if decision
                    else None
                ),
                "sum_logp_original": sum(
                    float(row["logp_original"]) for row in token_rows
                ),
                "sum_logp_teacher": sum(
                    float(row["logp_teacher"]) for row in token_rows
                ),
                "delta_sum_logp": sum(
                    float(row["delta_logp_teacher_minus_original"])
                    for row in token_rows
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
        "target_top_rate_original": _mean_bool(
            bool(row["target_is_top_original"]) for row in rows
        ),
        "target_top_rate_teacher": _mean_bool(
            bool(row["target_is_top_teacher"]) for row in rows
        ),
        "same_surface_different_id_count": sum(
            bool(row["teacher_top_same_surface_different_id"]) for row in rows
        ),
        "teacher_preference_counts": _count_values(
            str(row["teacher_preference"]) for row in rows
        ),
        "top1_transition_counts": _count_values(
            str(row["top1_transition"]) for row in rows
        ),
        "hallucination_token_count": sum(
            bool(row.get("is_hallucination")) for row in rows
        ),
        "token_label_counts": label_counts,
        "mutation_relation_counts": mutation_counts,
    }


def rebuild_privileged_report(output_dir: str | Path) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    result_paths = sorted((output_root / "samples").glob("*/result.json"))
    if not result_paths:
        raise RuntimeError(
            f"no completed result.json files under {output_root / 'samples'}"
        )
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    all_tokens: list[dict[str, Any]] = []
    all_mutations: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for result_path, result in zip(result_paths, results):
        _write_sample_outputs(result_path.parent, result)
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
        _render_aggregate_html(sample_rows),
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
        "created_at": _utc_now(),
        "completed_samples": len(results),
        "total_tokens": len(rows),
        "total_mutations": len(mutations),
        "mean_p_original": _mean(float(row["p_original"]) for row in rows),
        "mean_p_teacher": _mean(float(row["p_teacher"]) for row in rows),
        "mean_delta_logp_teacher_minus_original": _mean(deltas),
        "median_delta_logp_teacher_minus_original": statistics.median(deltas)
        if deltas
        else 0.0,
        "privileged_probability_gain_rate": _mean_bool(
            bool(row["probability_increased_with_privileged_info"]) for row in rows
        ),
        "top1_changed_rate": _mean_bool(bool(row["top1_changed"]) for row in rows),
        "target_top_rate_original": _mean_bool(
            bool(row["target_is_top_original"]) for row in rows
        ),
        "target_top_rate_teacher": _mean_bool(
            bool(row["target_is_top_teacher"]) for row in rows
        ),
        "same_surface_different_id_count": sum(
            bool(row["teacher_top_same_surface_different_id"]) for row in rows
        ),
        "teacher_preference_counts": _count_values(
            str(row["teacher_preference"]) for row in rows
        ),
        "top1_transition_counts": _count_values(
            str(row["top1_transition"]) for row in rows
        ),
        "finish_reason_counts": _count_values(
            str(result["response"].get("finish_reason", "unknown"))
            for result in results
        ),
        "token_label_counts": _count_values(
            str(row.get("token_label", "unknown")) for row in rows
        ),
        "mutation_relation_counts": _count_values(
            str(row["relation"]) for row in mutations
        ),
    }


def _write_sample_outputs(sample_dir: Path, result: dict[str, Any]) -> None:
    _write_csv(sample_dir / "token_probabilities.csv", result["tokens"])
    _write_csv(
        sample_dir / "mutation_probabilities.csv", result["mutation_observations"]
    )
    _write_text_atomic(sample_dir / "report.html", _render_sample_html(result))


def _response_rows_in_generation_order(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = list(result["tokens"])
    response_ids = [int(token_id) for token_id in result["response"]["token_ids"]]
    if len(rows) != len(response_ids):
        raise RuntimeError(
            "report token count does not match generated response IDs: "
            f"rows={len(rows)} ids={len(response_ids)}"
        )
    for position, (row, response_id) in enumerate(zip(rows, response_ids)):
        row_index = int(row["index"])
        row_token_id = int(row["token_id"])
        if row_index != position or row_token_id != response_id:
            raise RuntimeError(
                "report tokens are not in generated response order at position "
                f"{position}: index={row_index} token_id={row_token_id} "
                f"response_id={response_id}"
            )
    return rows


def _render_sample_html(result: dict[str, Any]) -> str:
    rows = _response_rows_in_generation_order(result)
    token_rows = "".join(_token_table_row(row) for row in rows)
    image_name = Path(result["sample"]["image_copy"]).name
    ground_truth = html.escape(str(result["ground_truth"]))
    response_text = html.escape(str(result["response"]["text"]))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(result["pair_id"])} privileged probe</title>{_report_css()}</head>
<body><main>
<nav><a href="../../report.html" target="_top">全部样本</a><a href="token_probabilities.csv">完整 Token CSV</a></nav>
<h1>{html.escape(result["pair_id"])}</h1>
<section><h2>原始文本</h2>
<div class="source-review">
<figure><img src="{html.escape(image_name)}" alt="document page"><figcaption>输入图片</figcaption></figure>
<div class="transcript-grid">
<div class="transcript-panel"><div class="panel-heading"><h3>Ground Truth</h3><span>{len(str(result["ground_truth"]))} chars</span></div><pre class="transcript">{ground_truth}</pre></div>
<div class="transcript-panel"><div class="panel-heading"><h3>模型 Response</h3><span>{len(rows)} tokens</span></div><pre class="transcript">{response_text}</pre></div>
</div></div>
</section>
<section id="token-details"><div class="section-heading"><h2>全部 Response Token（严格按生成顺序）</h2><output>{len(rows)} tokens</output></div>
<div class="table-scroll token-table-scroll"><table class="token-detail-table">
<thead><tr><th rowspan="2">生成索引</th><th rowspan="2">Response token</th><th colspan="4" class="condition original-condition">原图条件（image + prompt）</th><th colspan="4" class="condition teacher-condition">GT Teacher-Forcing 条件</th><th colspan="2" class="condition delta-condition">概率变化（Teacher - Original）</th></tr>
<tr><th>p(response token)</th><th>response rank</th><th>Top-1</th><th>Top-2</th><th>p(same response token)</th><th>response rank</th><th>Top-1</th><th>Top-2</th><th>Δp</th><th>Δlogp</th></tr></thead>
<tbody>{token_rows}</tbody></table></div></section>
</main></body></html>"""


def _render_aggregate_html(sample_rows: Sequence[dict[str, Any]]) -> str:
    first_report = str(sample_rows[0]["report"]) if sample_rows else ""
    sample_links = "".join(
        f"<a class='sample-link{' active' if index == 0 else ''}' "
        f"href='{html.escape(str(row['report']), quote=True)}' "
        f"target='sample-report' data-sample-link>"
        f"<strong>{html.escape(str(row['pair_id']))}</strong>"
        f"<span>{int(row['token_count'])} tokens</span></a>"
        for index, row in enumerate(sample_rows)
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Privileged Response Token Probe</title>{_report_css()}</head><body><main>
<nav><a href="token_probabilities.csv">完整 Token CSV</a></nav>
<h1>Response Token 概率对照</h1>
<div class="sample-browser">
<aside class="sample-list">{sample_links}</aside>
<iframe class="sample-frame" name="sample-report" src="{html.escape(first_report, quote=True)}" title="样本逐 Token 概率对照"></iframe>
</div>
</main><script>
document.querySelectorAll('[data-sample-link]').forEach(link => {{
  link.addEventListener('click', () => {{
    document.querySelectorAll('[data-sample-link]').forEach(item => item.classList.remove('active'));
    link.classList.add('active');
  }});
}});
</script></body></html>"""


def _candidate_at(
    row: dict[str, Any],
    condition: str,
    rank: int,
) -> dict[str, Any] | None:
    candidates = row.get(f"top_candidates_{condition}", [])
    if isinstance(candidates, list) and len(candidates) >= rank:
        candidate = dict(candidates[rank - 1])
        candidate.setdefault("rank", rank)
        return candidate
    if rank != 1:
        return None
    suffix = "original" if condition == "original" else "teacher"
    token_id = row.get(f"top_token_id_{suffix}")
    if token_id is None:
        return None
    return {
        "rank": 1,
        "token_id": int(token_id),
        "token": str(row.get(f"top_token_{suffix}", "")),
        "raw_token": str(
            row.get(
                f"top_raw_token_{suffix}",
                row.get(f"top_token_{suffix}", ""),
            )
        ),
        "probability": float(row.get(f"top_p_{suffix}", 0.0)),
    }


def _candidate_block(
    row: dict[str, Any],
    condition: str,
    rank: int,
) -> str:
    candidate = _candidate_at(row, condition, rank)
    rank_label = f"Top-{rank}"
    if candidate is None:
        return (
            "<div class='candidate missing'>"
            f"<span class='candidate-rank'>{rank_label}</span><span>-</span></div>"
        )
    token_id = int(candidate["token_id"])
    raw_token = str(candidate.get("raw_token", candidate.get("token", "")))
    decoded = _display_token(raw_token) or "<empty>"
    target_id = int(row.get("token_id", -1))
    target_raw = str(row.get("raw_token", row.get("token", "")))
    if token_id == target_id:
        relation = "target"
    elif raw_token == target_raw:
        relation = "same-surface"
    else:
        relation = "alternative"
    probability = float(candidate.get("probability", 0.0))
    return (
        f"<div class='candidate {relation}'>"
        f"<span class='candidate-rank'>{rank_label}</span>"
        f"<code>{html.escape(decoded)}</code>"
        f"<span class='candidate-id'>ID {token_id}</span>"
        f"<strong>p {probability:.6f}</strong></div>"
    )


def _probability_cell(value: Any, condition: str) -> str:
    probability = min(1.0, max(0.0, float(value)))
    return (
        f"<div class='probability-cell {condition}'>"
        f"<strong>{probability:.6f}</strong>"
        f"<span class='probability-track'><i style='width:{100 * probability:.4f}%'></i></span>"
        "</div>"
    )


def _token_table_row(row: dict[str, Any]) -> str:
    delta = float(row["delta_logp_teacher_minus_original"])
    delta_p = float(row["delta_p_teacher_minus_original"])
    return (
        f"<tr id='token-{int(row['index'])}' class='token-row'>"
        f"<td>{int(row['index'])}</td>"
        "<td><div class='response-token'>"
        f"<code>{html.escape(str(row['token'])) or '&lt;empty&gt;'}</code>"
        f"<span>ID {int(row['token_id'])}</span></div></td>"
        f"<td>{_probability_cell(row['p_original'], 'original')}</td>"
        f"<td>{int(row['rank_original'])}</td>"
        f"<td>{_candidate_block(row, 'original', 1)}</td>"
        f"<td>{_candidate_block(row, 'original', 2)}</td>"
        f"<td>{_probability_cell(row['p_teacher'], 'teacher')}</td>"
        f"<td>{int(row['rank_teacher'])}</td>"
        f"<td>{_candidate_block(row, 'teacher', 1)}</td>"
        f"<td>{_candidate_block(row, 'teacher', 2)}</td>"
        f"<td class='{_delta_class(delta_p)}'>{delta_p:+.6f}</td>"
        f"<td class='{_delta_class(delta)}'>{delta:+.4f}</td></tr>"
    )


def _mutation_focus_card(
    mutation: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> str:
    mutation_id = str(mutation.get("mutation_id", ""))
    indices = {
        int(index)
        for index in mutation.get("response_token_indices", [])
        if isinstance(index, int) or str(index).isdigit()
    }
    token_rows = [
        row
        for row in rows
        if int(row.get("index", -1)) in indices
        or mutation_id in str(row.get("mutation_ids", "")).split(",")
    ]
    detail_rows = "".join(_mutation_token_detail_row(row) for row in token_rows)
    if not detail_rows:
        detail_rows = (
            "<tr><td colspan='8' class='empty-state'>"
            "未对齐到模型 Response token</td></tr>"
        )
    relation = str(mutation.get("relation", "unknown"))
    relation_class = "gain" if relation == "expected" else "drop"
    bbox = mutation.get("bbox", [])
    bbox_text = json.dumps(bbox, ensure_ascii=False) if bbox else "-"
    return f"""<article class="mutation-focus">
<div class="mutation-heading"><div><span class="mutation-id">{html.escape(mutation_id)}</span><strong>变异词对比</strong></div><span class="{relation_class}">{html.escape(relation)}</span></div>
<div class="mutation-terms">
<div><span>图片中的变异词</span><code>{html.escape(str(mutation.get("ocr_ans", "")))}</code></div>
<div><span>原词</span><code>{html.escape(str(mutation.get("origin_ans", "")))}</code></div>
<div><span>模型读回</span><code>{html.escape(str(mutation.get("predicted", "")))}</code></div>
<div><span>BBox</span><code>{html.escape(bbox_text)}</code></div>
</div>
<div class="table-scroll mutation-token-table"><table><thead><tr><th>Response token</th><th>p original</th><th>p teacher</th><th>Δlogp</th><th>Original Top-1</th><th>Original Top-2</th><th>Teacher Top-1</th><th>Teacher Top-2</th></tr></thead><tbody>{detail_rows}</tbody></table></div>
</article>"""


def _mutation_token_detail_row(row: dict[str, Any]) -> str:
    delta = float(row["delta_logp_teacher_minus_original"])
    return (
        "<tr>"
        "<td><div class='response-token'>"
        f"<code>{html.escape(str(row['token'])) or '&lt;empty&gt;'}</code>"
        f"<span>#{int(row['index'])} · ID {int(row['token_id'])}</span></div></td>"
        f"<td>{_probability_cell(row['p_original'], 'original')}</td>"
        f"<td>{_probability_cell(row['p_teacher'], 'teacher')}</td>"
        f"<td class='{_delta_class(delta)}'>{delta:+.4f}</td>"
        f"<td>{_candidate_block(row, 'original', 1)}</td>"
        f"<td>{_candidate_block(row, 'original', 2)}</td>"
        f"<td>{_candidate_block(row, 'teacher', 1)}</td>"
        f"<td>{_candidate_block(row, 'teacher', 2)}</td>"
        "</tr>"
    )


def _highlight_ground_truth(
    ground_truth: str,
    changes: Sequence[dict[str, Any]],
) -> str:
    spans: list[tuple[int, int, str]] = []
    for index, change in enumerate(changes, start=1):
        span = change.get("markdown_span", [])
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start or end > len(ground_truth):
            continue
        label = (
            f"m{index:03d}: {change.get('origin_ans', '')} -> "
            f"{change.get('ocr_ans', '')}"
        )
        spans.append((start, end, label))
    spans.sort()
    parts: list[str] = []
    cursor = 0
    for start, end, label in spans:
        if start < cursor:
            continue
        parts.append(html.escape(ground_truth[cursor:start]))
        parts.append(
            f"<mark class='mutation-mark' title='{html.escape(label, quote=True)}'>"
            f"{html.escape(ground_truth[start:end])}</mark>"
        )
        cursor = end
    parts.append(html.escape(ground_truth[cursor:]))
    return "".join(parts)


def _highlight_response(
    response_text: str,
    rows: Sequence[dict[str, Any]],
) -> str:
    reconstructed = "".join(str(row.get("raw_token", "")) for row in rows)
    if reconstructed != response_text:
        return html.escape(response_text)
    parts: list[str] = []
    for row in rows:
        raw_token = str(row.get("raw_token", ""))
        escaped = html.escape(raw_token)
        mutation_ids = str(row.get("mutation_ids", "")).strip()
        is_mismatch = bool(row.get("is_hallucination"))
        if mutation_ids:
            title = f"#{row.get('index')} · ID {row.get('token_id')} · {mutation_ids}"
            parts.append(
                f"<mark class='mutation-mark' title='{html.escape(title, quote=True)}'>"
                f"{escaped}</mark>"
            )
        elif is_mismatch:
            title = f"#{row.get('index')} · ID {row.get('token_id')} · GT 不一致"
            parts.append(
                f"<mark class='mismatch-mark' title='{html.escape(title, quote=True)}'>"
                f"{escaped}</mark>"
            )
        else:
            parts.append(escaped)
    return "".join(parts)


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
    condition = "original" if key == "p_original" else "teacher"
    top1 = _candidate_at(row, condition, 1)
    top2 = _candidate_at(row, condition, 2)
    title = (
        f"#{row['index']} {row['token']} | {key}={value:.6f} | "
        f"delta_logp={float(row['delta_logp_teacher_minus_original']):+.4f} | "
        f"{_candidate_title(top1, 1)} | {_candidate_title(top2, 2)}"
    )
    return f"<button type='button' class='heat-token' data-token-index='{int(row['index'])}' style='background:hsl({hue:.1f} 68% 78%)' title='{html.escape(title, quote=True)}'>{html.escape(str(row['token']))}</button>"


def _delta_span(row: dict[str, Any]) -> str:
    value = float(row["delta_logp_teacher_minus_original"])
    magnitude = min(1.0, abs(value) / 3.0)
    hue = 145 if value >= 0 else 2
    lightness = 94 - 35 * magnitude
    title = (
        f"#{row['index']} {row['token']} | delta_logp={value:+.6f} | "
        f"original {_candidate_title(_candidate_at(row, 'original', 1), 1)} | "
        f"teacher {_candidate_title(_candidate_at(row, 'teacher', 1), 1)}"
    )
    return f"<button type='button' class='heat-token' data-token-index='{int(row['index'])}' style='background:hsl({hue} 62% {lightness:.1f}%)' title='{html.escape(title, quote=True)}'>{html.escape(str(row['token']))}</button>"


def _candidate_title(candidate: dict[str, Any] | None, rank: int) -> str:
    if candidate is None:
        return f"Top-{rank}=unavailable"
    raw_token = str(candidate.get("raw_token", candidate.get("token", "")))
    decoded = _display_token(raw_token) or "<empty>"
    return (
        f"Top-{rank}=ID {int(candidate['token_id'])} {decoded!r} "
        f"p={float(candidate.get('probability', 0.0)):.6f}"
    )


def _report_css() -> str:
    return """<style>
:root {
  font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
  color: #1d292d;
  background: #f3f5f5;
  line-height: 1.45;
  font-synthesis: none;
}
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1880px; margin: auto; padding: 24px; }
nav { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
a { color: #075d73; }
h1 { margin: 0 0 18px; font-size: 28px; letter-spacing: 0; }
h2 { margin: 30px 0 12px; font-size: 20px; letter-spacing: 0; }
h3 { margin: 0; font-size: 14px; letter-spacing: 0; color: #43565d; }
section { width: 100%; }
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
}
.stat {
  min-width: 0;
  padding: 12px;
  background: #fff;
  border: 1px solid #d5dcda;
  border-radius: 6px;
}
.stat span { display: block; color: #65767b; font-size: 12px; }
.stat strong { display: block; overflow-wrap: anywhere; font-size: 20px; }
.source-review {
  display: grid;
  grid-template-columns: minmax(300px, .72fr) minmax(680px, 1.28fr);
  gap: 20px;
  align-items: start;
}
figure { margin: 0; }
figure img {
  display: block;
  width: 100%;
  max-height: 980px;
  object-fit: contain;
  background: #fff;
  border: 1px solid #d5dcda;
}
figcaption { margin-top: 5px; color: #65767b; font-size: 12px; }
.transcript-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.transcript-panel {
  min-width: 0;
  background: #fff;
  border: 1px solid #d5dcda;
  border-radius: 6px;
}
.panel-heading, .section-heading, .mutation-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.panel-heading {
  min-height: 44px;
  padding: 10px 12px;
  border-bottom: 1px solid #dfe5e3;
}
.panel-heading span, .section-heading output { color: #65767b; font-size: 12px; }
.sample-browser {
  display: grid;
  grid-template-columns: minmax(210px, 280px) minmax(0, 1fr);
  gap: 12px;
  height: calc(100vh - 120px);
  min-height: 680px;
}
.sample-list {
  overflow: auto;
  padding: 6px;
  background: #fff;
  border: 1px solid #d5dcda;
}
.sample-link {
  display: grid;
  gap: 2px;
  margin-bottom: 4px;
  padding: 9px 10px;
  border-left: 3px solid transparent;
  color: #26383e;
  text-decoration: none;
}
.sample-link:hover { background: #f1f6f5; }
.sample-link.active { border-left-color: #08748d; background: #e5f1f2; }
.sample-link strong { overflow-wrap: anywhere; font-size: 12px; }
.sample-link span { color: #708086; font-size: 11px; }
.sample-frame {
  width: 100%;
  height: 100%;
  border: 1px solid #d5dcda;
  background: #fff;
}
pre.transcript {
  min-height: 520px;
  max-height: 840px;
  margin: 0;
  padding: 14px;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: 13px/1.58 ui-monospace, SFMono-Regular, Menlo, monospace;
}
mark.mutation-mark {
  padding: 1px 2px;
  color: inherit;
  background: #ffe083;
  box-shadow: inset 0 -2px 0 #c48b00;
}
mark.mismatch-mark {
  padding: 1px 2px;
  color: inherit;
  background: #ffd3d0;
  box-shadow: inset 0 -2px 0 #ba3e39;
}
.legend { display: flex; gap: 18px; margin-top: 9px; color: #596b71; font-size: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend-swatch { width: 12px; height: 12px; border: 1px solid #aab6b2; }
.legend-swatch.mutation { background: #ffe083; }
.legend-swatch.mismatch { background: #ffd3d0; }
.mutation-list { display: grid; gap: 12px; }
.mutation-focus {
  overflow: hidden;
  background: #fff;
  border: 1px solid #cfd8d5;
  border-left: 4px solid #d09a12;
  border-radius: 6px;
}
.mutation-heading { min-height: 46px; padding: 10px 12px; border-bottom: 1px solid #dfe5e3; }
.mutation-heading > div { display: flex; align-items: center; gap: 9px; }
.mutation-id {
  padding: 2px 6px;
  background: #e8eeee;
  border-radius: 4px;
  color: #405258;
  font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
}
.mutation-terms {
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 1px;
  background: #dfe5e3;
  border-bottom: 1px solid #dfe5e3;
}
.mutation-terms > div { min-width: 0; padding: 10px 12px; background: #f9fbfa; }
.mutation-terms span { display: block; margin-bottom: 4px; color: #65767b; font-size: 11px; }
.mutation-terms code { display: block; overflow-wrap: anywhere; font-size: 13px; }
.mutation-token-table { border: 0; }
.chart-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(360px, 1fr));
  gap: 20px;
}
canvas { width: 100%; height: auto; background: #fff; border: 1px solid #d5dcda; }
.strip-grid { display: grid; gap: 10px; margin-top: 12px; }
.strip-grid h3 { margin-bottom: 6px; }
.token-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  min-height: 38px;
  padding: 8px;
  background: #fff;
  border: 1px solid #d5dcda;
}
.heat-token {
  min-width: 18px;
  min-height: 24px;
  padding: 2px 4px;
  border: 1px solid rgba(37, 56, 61, .18);
  border-radius: 3px;
  color: #172327;
  font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: 0;
  white-space: pre;
  cursor: pointer;
}
.heat-token:focus-visible { outline: 2px solid #08748d; outline-offset: 1px; }
.token-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 10px;
}
.segmented {
  display: inline-flex;
  flex-wrap: wrap;
  overflow: hidden;
  border: 1px solid #bcc8c4;
  border-radius: 6px;
}
.filter-button {
  min-height: 34px;
  padding: 6px 11px;
  border: 0;
  border-right: 1px solid #ccd5d2;
  background: #fff;
  color: #30444a;
  cursor: pointer;
}
.filter-button:last-child { border-right: 0; }
.filter-button.active { background: #176b78; color: #fff; }
.token-search { display: flex; align-items: center; gap: 7px; color: #596b71; font-size: 12px; }
.token-search input {
  width: min(300px, 34vw);
  min-height: 34px;
  padding: 6px 9px;
  border: 1px solid #bcc8c4;
  border-radius: 5px;
  background: #fff;
  font: inherit;
}
.table-scroll { overflow: auto; background: #fff; border: 1px solid #d5dcda; }
.token-table-scroll { max-height: 860px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td {
  padding: 7px 9px;
  border-right: 1px solid #edf0ef;
  border-bottom: 1px solid #e2e7e5;
  text-align: right;
  vertical-align: top;
  white-space: nowrap;
}
th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: #e8eeee;
  color: #33464c;
}
.token-detail-table thead tr:first-child th { top: 0; }
.token-detail-table thead tr:nth-child(2) th { top: 33px; }
th.condition { text-align: center; }
th.original-condition { background: #dceff1; }
th.teacher-condition { background: #f5e5e8; }
td:first-child, th:first-child { text-align: left; }
.token-row.mutation-token-row { background: #fff9e8; }
.token-row.mismatch-token-row td:first-child { box-shadow: inset 4px 0 0 #c64d47; }
.token-row.top1-changed-row { outline: 1px solid rgba(164, 55, 76, .22); outline-offset: -1px; }
.response-token {
  display: grid;
  min-width: 100px;
  gap: 2px;
  text-align: left;
}
.response-token code {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #152a31;
  font-size: 13px;
}
.response-token span { color: #708086; font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; }
.label-cell { max-width: 180px; overflow: hidden; text-overflow: ellipsis; text-align: left; }
.probability-cell { display: grid; min-width: 92px; gap: 4px; }
.probability-cell strong { font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; }
.probability-track { width: 100%; height: 4px; overflow: hidden; background: #e6ebe9; }
.probability-track i { display: block; height: 100%; }
.probability-cell.original .probability-track i { background: #168299; }
.probability-cell.teacher .probability-track i { background: #b34b63; }
.candidate {
  display: grid;
  grid-template-columns: auto minmax(48px, 1fr);
  min-width: 174px;
  gap: 2px 7px;
  padding: 5px 7px;
  border: 1px solid #d6ddda;
  border-radius: 5px;
  background: #fff;
  text-align: left;
}
.candidate.target { border-color: #4d9a78; background: #f0faf5; }
.candidate.same-surface { border-color: #4f8fa1; background: #eff8fa; }
.candidate.alternative { border-color: #d39a54; background: #fff8ed; }
.candidate.missing { color: #8a989c; background: #f7f8f8; }
.candidate-rank { color: #68797e; font-size: 10px; }
.candidate code {
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 12px;
}
.candidate-id { color: #68797e; font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; }
.candidate strong { font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; }
.gain { color: #08764a; font-weight: 650; }
.drop { color: #b02b2b; font-weight: 650; }
.empty-state { margin: 0; padding: 14px; color: #718086; text-align: left; }
@media (max-width: 1180px) {
  .source-review { grid-template-columns: 1fr; }
  figure img { max-height: 720px; }
}
@media (max-width: 820px) {
  main { padding: 14px; }
  .transcript-grid, .chart-grid { grid-template-columns: 1fr; }
  .sample-browser { grid-template-columns: 1fr; height: auto; }
  .sample-list { max-height: 220px; }
  .sample-frame { min-height: 900px; }
  .mutation-terms { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .token-toolbar { align-items: stretch; flex-direction: column; }
  .token-search input { width: 100%; }
  pre.transcript { min-height: 300px; }
}
@media (max-width: 520px) {
  .mutation-terms { grid-template-columns: 1fr; }
}
</style>"""


def _chart_javascript() -> str:
    return """
function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 800;
  canvas.width = w * dpr;
  canvas.height = 360 * dpr;
  const c = canvas.getContext('2d');
  c.scale(dpr, dpr);
  return {c, w, h: 360};
}
function draw() {
  const canvas = document.getElementById('prob-chart');
  if (!canvas) return;
  const {c, w} = setup(canvas);
  c.clearRect(0, 0, w, 360);
  const pad = {l: 48, r: 16, t: 18, b: 34};
  const pw = w - pad.l - pad.r;
  const ph = 210;
  const x = i => pad.l + (DATA.length <= 1 ? 0 : i / (DATA.length - 1)) * pw;
  const y = p => pad.t + (1 - p) * ph;
  c.strokeStyle = '#cad1ce';
  for (let q = 0; q <= 4; q++) {
    const yy = y(q / 4);
    c.beginPath(); c.moveTo(pad.l, yy); c.lineTo(w - pad.r, yy); c.stroke();
    c.fillStyle = '#59686d'; c.fillText((q / 4).toFixed(2), 6, yy + 4);
  }
  function line(key, color) {
    c.strokeStyle = color; c.lineWidth = 1.5; c.beginPath();
    DATA.forEach((d, i) => {
      const xx = x(i), yy = y(d[key]);
      i ? c.lineTo(xx, yy) : c.moveTo(xx, yy);
    });
    c.stroke();
  }
  line('po', '#168299');
  line('pt', '#b34b63');
  c.fillStyle = '#168299'; c.fillText('original', pad.l, 12);
  c.fillStyle = '#b34b63'; c.fillText('GT teacher', pad.l + 64, 12);
  const base = 330;
  const scale = Math.max(1, DATA.reduce((m, d) => Math.max(m, Math.abs(d.d)), 0));
  DATA.forEach((d, i) => {
    const xx = x(i), bh = Math.min(74, Math.abs(d.d) / scale * 74);
    c.fillStyle = d.d >= 0 ? '#238a63' : '#c94b4b';
    c.fillRect(xx, d.d >= 0 ? base - bh : base, Math.max(1, pw / Math.max(DATA.length, 1)), bh);
  });
  c.strokeStyle = '#59686d';
  c.beginPath(); c.moveTo(pad.l, base); c.lineTo(w - pad.r, base); c.stroke();
  c.fillStyle = '#59686d'; c.fillText('Δlogp', 6, base + 4);
}
let activeTokenFilter = 'all';
function applyTokenFilter() {
  const query = (document.getElementById('token-search')?.value || '').trim().toLowerCase();
  const rows = [...document.querySelectorAll('.token-row')];
  let visible = 0;
  rows.forEach(row => {
    const modeMatch =
      activeTokenFilter === 'all' ||
      (activeTokenFilter === 'mutation' && row.dataset.mutation === '1') ||
      (activeTokenFilter === 'changed' && row.dataset.top1Changed === '1') ||
      (activeTokenFilter === 'gain' && row.dataset.delta === 'gain') ||
      (activeTokenFilter === 'drop' && row.dataset.delta === 'drop') ||
      (activeTokenFilter === 'mismatch' && row.dataset.mismatch === '1');
    const queryMatch = !query || (row.dataset.search || '').includes(query);
    const show = modeMatch && queryMatch;
    row.hidden = !show;
    if (show) visible++;
  });
  const counter = document.getElementById('visible-token-count');
  if (counter) counter.textContent = visible + ' / ' + rows.length;
}
function activateFilter(mode) {
  activeTokenFilter = mode;
  document.querySelectorAll('[data-token-filter]').forEach(button => {
    const active = button.dataset.tokenFilter === mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
  applyTokenFilter();
}
document.querySelectorAll('[data-token-filter]').forEach(button => {
  button.addEventListener('click', () => activateFilter(button.dataset.tokenFilter || 'all'));
});
document.getElementById('token-search')?.addEventListener('input', applyTokenFilter);
document.querySelectorAll('[data-token-index]').forEach(button => {
  button.addEventListener('click', () => {
    activateFilter('all');
    const search = document.getElementById('token-search');
    if (search) search.value = '';
    applyTokenFilter();
    const row = document.getElementById('token-' + button.dataset.tokenIndex);
    if (!row) return;
    row.scrollIntoView({behavior: 'smooth', block: 'center'});
    row.animate(
      [{backgroundColor: '#ffe083'}, {backgroundColor: ''}],
      {duration: 1200, easing: 'ease-out'}
    );
  });
});
window.addEventListener('resize', draw);
draw();
applyTokenFilter();
"""


def _aggregate_chart_javascript() -> str:
    return """
function ctx(id){const el=document.getElementById(id),dpr=window.devicePixelRatio||1,w=el.clientWidth||650;el.width=w*dpr;el.height=480*dpr;const c=el.getContext('2d');c.scale(dpr,dpr);return {c,w,h:480}}
function axes(c,w,h){c.strokeStyle='#bec8c4';c.beginPath();c.moveTo(45,15);c.lineTo(45,h-38);c.lineTo(w-15,h-38);c.stroke()}
function scatter(){const {c,w,h}=ctx('scatter');axes(c,w,h);const sx=x=>45+x*(w-60),sy=y=>15+(1-y)*(h-53);c.strokeStyle='#9aa7a2';c.setLineDash([4,4]);c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(1),sy(1));c.stroke();c.setLineDash([]);SCATTER.forEach(p=>{c.fillStyle=p.label==='correct'?'rgba(30,115,87,.24)':'rgba(190,54,54,.48)';c.fillRect(sx(p.x),sy(p.y),2,2)});c.fillStyle='#45545a';c.fillText('p original',w/2,h-12);c.save();c.translate(13,h/2);c.rotate(-Math.PI/2);c.fillText('p teacher',0,0);c.restore()}
function histogram(){const {c,w,h}=ctx('histogram');axes(c,w,h);if(!DELTAS.length)return;const ordered=[...DELTAS].sort((a,b)=>a-b),lo=ordered[Math.floor(.01*(ordered.length-1))],hi=ordered[Math.floor(.99*(ordered.length-1))],min=Math.min(-.01,lo),max=Math.max(.01,hi),bins=50,count=Array(bins).fill(0);DELTAS.forEach(v=>{const cl=Math.max(min,Math.min(max,v));count[Math.min(bins-1,Math.floor((cl-min)/(max-min)*bins))]++});const peak=Math.max(...count,1),bw=(w-60)/bins;count.forEach((n,i)=>{const bh=n/peak*(h-70);const center=min+(i+.5)/bins*(max-min);c.fillStyle=center>=0?'#3b9873':'#cf6262';c.fillRect(45+i*bw,h-38-bh,Math.max(1,bw-1),bh)});const zx=45+(0-min)/(max-min)*(w-60);c.strokeStyle='#34454b';c.beginPath();c.moveTo(zx,15);c.lineTo(zx,h-38);c.stroke();c.fillStyle='#45545a';c.fillText(min.toFixed(2),45,h-18);c.fillText(max.toFixed(2),w-48,h-18);c.fillText('Δlogp teacher - original',w/2-70,h-5)}function draw(){scatter();histogram()}window.addEventListener('resize',draw);draw();
"""


def _stat(
    label: str, value: Any, *, percent: bool = False, signed: bool = False
) -> str:
    if isinstance(value, float):
        rendered = (
            f"{100 * value:.2f}%"
            if percent
            else (f"{value:+.4f}" if signed else f"{value:.6f}")
        )
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
    return [
        values[min(len(values) - 1, int(index * len(values) / maximum))]
        for index in range(maximum)
    ]


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


def _sample_fingerprint(
    *, sample: PrivilegedProbeSample, config: dict[str, Any]
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": sample.pair_id,
        "image_path": str(sample.image_path),
        "image_size": sample.image_path.stat().st_size,
        "image_mtime_ns": sample.image_path.stat().st_mtime_ns,
        "ground_truth_sha256": _sha256_text(
            sample.ground_truth_path.read_text(encoding="utf-8")
        ),
        "changes": list(sample.changes),
        "model_id": config["model_id"],
        "prompt": config["prompt"],
        "privileged_instruction": config["privileged_instruction"],
        "max_new_tokens": config["max_new_tokens"],
        "top_k": config["top_k"],
        "forward_chunk_size": config["forward_chunk_size"],
        "device_map": config["device_map"],
        "dtype": config["dtype"],
        "trust_remote_code": config["trust_remote_code"],
        "min_pixels": config["min_pixels"],
        "max_pixels": config["max_pixels"],
        "image_patch_size": config["image_patch_size"],
        "seed": int(config["seed"]) + sample.ordinal - 1,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _result_matches(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        return (
            json.loads(path.read_text(encoding="utf-8")).get("fingerprint")
            == fingerprint
        )
    except (OSError, json.JSONDecodeError):
        return False


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_token_ids(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids)
    return _sha256_text(payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_error(exc: BaseException, *, limit: int = 2000) -> str:
    message = str(exc).replace("\x00", "")
    return message if len(message) <= limit else message[:limit] + "..."


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    )


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
        prog="qwen-mm-privileged-probe",
        description=(
            "Offline Transformers probe: generate OCR once, then directly append "
            "the identical response ID tensor under original-image and GT prompts."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file")
    parser.add_argument(
        "--privileged-instruction", default=DEFAULT_PRIVILEGED_INSTRUCTION
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--forward-chunk-size", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=16777216)
    parser.add_argument("--image-patch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
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
        summary = rebuild_privileged_report(args.output_dir)
        print(
            f"Rebuilt report: {Path(args.output_dir).expanduser().resolve() / 'report.html'} "
            f"samples={summary['completed_samples']} tokens={summary['total_tokens']}",
            flush=True,
        )
        return 0
    if not args.dataset_root:
        parser.error("--dataset-root is required unless --rebuild-report-only is used")
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    device_map = None if args.device_map.lower() == "none" else args.device_map
    summary = run_privileged_probe(
        model_id=args.model_id,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        prompt=prompt,
        privileged_instruction=args.privileged_instruction,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        forward_chunk_size=args.forward_chunk_size,
        device_map=device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        image_patch_size=args.image_patch_size,
        seed=args.seed,
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
