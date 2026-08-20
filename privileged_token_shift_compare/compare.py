#!/usr/bin/env python3
"""Compare fixed generated-token probabilities with and without privileged text.

This tool is intentionally independent from verl.  It accepts verl-style JSONL
records, reproduces the relevant message construction locally, generates once
from the original multimodal prompt, and scores the exact same response token
IDs under:

1. the original student image + prompt; and
2. a text-only standalone privileged rewrite prompt.

The output is a resumable static report. Each sample owns a separate page and
loads only its own JavaScript data payload, so reports also work from file://.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = 1
DEFAULT_PRIVILEGED_KEY = "reward_model.ground_truth"
DEFAULT_PRIVILEGED_TEMPLATE = "{privileged_text} 请复写上面的内容。"


@dataclass(frozen=True)
class Thresholds:
    student_high: float = 0.40
    teacher_low: float = 0.01
    probability_ratio: float = 10.0
    teacher_uncertain_entropy: float = 3.0
    max_equivalent_span: int = 4


@dataclass(frozen=True)
class RecordLocator:
    line_number: int
    byte_offset: int
    byte_length: int
    sample_id: str
    slug: str
    source_sha256: str


@dataclass
class TokenStats:
    target_probabilities: list[float]
    target_logprobs: list[float]
    target_ranks: list[int]
    entropies: list[float]
    top_candidates: list[list[dict[str, Any]]]


@dataclass
class ModelBundle:
    model: Any
    processor: Any
    tokenizer: Any
    device: Any
    dtype_name: str


def log(message: str) -> None:
    print(message, flush=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_javascript_assignment(path: Path, variable: str, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    atomic_write_text(path, f"window.{variable} = {payload};\n")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


class Heartbeat:
    """Print progress while one model operation runs for longer than expected."""

    def __init__(self, label: str, interval: float) -> None:
        self.label = label
        self.interval = interval
        self.started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Heartbeat":
        self.started = time.monotonic()
        log(f"START phase={self.label}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = time.monotonic() - self.started
            log(f"PROGRESS phase={self.label} elapsed={format_duration(elapsed)}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        elapsed = time.monotonic() - self.started
        status = "error" if exc_type is not None else "completed"
        log(f"END phase={self.label} status={status} elapsed={format_duration(elapsed)}")


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def get_nested_field(record: dict[str, Any], key: str) -> Any:
    """Read either a literal key or a dot-separated nested key."""
    if key in record:
        return record[key]
    value: Any = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (list, tuple)):
        value = "\n".join(str(item) for item in value)
    text = str(value).strip()
    return text or None


def resolve_local_path(value: str | os.PathLike[str], base_dir: Path) -> str:
    text = os.fspath(value)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def normalize_images(value: Any, base_dir: Path) -> list[Any]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    images: list[Any] = []
    for item in values:
        if isinstance(item, (str, os.PathLike)):
            images.append(resolve_local_path(item, base_dir))
        elif isinstance(item, dict):
            payload = copy.deepcopy(item)
            for key in ("image", "path", "url"):
                if key in payload and isinstance(payload[key], (str, os.PathLike)):
                    payload[key] = resolve_local_path(payload[key], base_dir)
            images.append(payload)
        else:
            images.append(item)
    return images


def image_content_item(image: Any, max_pixels: int) -> dict[str, Any]:
    if isinstance(image, dict):
        payload = {"type": "image", **copy.deepcopy(image)}
    elif isinstance(image, (str, os.PathLike)):
        payload = {"type": "image", "image": os.fspath(image)}
    else:
        payload = {"type": "image", "image": image}
    payload.setdefault("max_pixels", max_pixels)
    return payload


def count_image_references(messages: Sequence[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            count += content.count("<image>")
        elif isinstance(content, list):
            count += sum(
                1 for item in content if isinstance(item, dict) and item.get("type") == "image"
            )
    return count


def build_student_messages(
    record: dict[str, Any],
    *,
    prompt_key: str,
    image_key: str,
    dataset_dir: Path,
    max_pixels: int,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Reproduce verl's image placeholder handling without importing verl."""
    prompt = get_nested_field(record, prompt_key)
    if not isinstance(prompt, list):
        raise TypeError(f"{prompt_key!r} must contain a list of chat messages")
    messages: list[dict[str, Any]] = copy.deepcopy(prompt)
    images = normalize_images(get_nested_field(record, image_key), dataset_dir)

    missing = len(images) - count_image_references(messages)
    if missing < 0:
        raise ValueError(f"prompt has more image references than {image_key}: missing={missing}")
    if missing:
        target = next((item for item in messages if item.get("role") == "user"), None)
        if target is None:
            if not messages:
                messages.append({"role": "user", "content": "<image>" * missing})
                target = messages[0]
            else:
                target = messages[0]
        content = target.get("content")
        if isinstance(content, str):
            target["content"] = "<image>" * missing + content
        elif isinstance(content, list):
            target["content"] = [
                *[image_content_item(image, max_pixels) for image in images[:missing]],
                *content,
            ]
        else:
            target["content"] = "<image>" * missing + str(content or "")

    image_offset = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            normalized: list[Any] = []
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    normalized.append(item)
                    continue
                payload = copy.deepcopy(item)
                image_value = payload.get("image", payload.get("path", payload.get("url")))
                if image_value is None:
                    if image_offset >= len(images):
                        raise ValueError("structured image item has no payload and no matching images entry")
                    payload = image_content_item(images[image_offset], max_pixels)
                elif isinstance(image_value, (str, os.PathLike)):
                    resolved = resolve_local_path(image_value, dataset_dir)
                    if "image" in payload:
                        payload["image"] = resolved
                    elif "path" in payload:
                        payload["path"] = resolved
                    else:
                        payload["url"] = resolved
                    payload.setdefault("max_pixels", max_pixels)
                normalized.append(payload)
                image_offset += 1
            message["content"] = normalized
            continue
        if not isinstance(content, str):
            continue
        content_items: list[dict[str, Any]] = []
        for segment in filter(None, re.split(r"(<image>)", content)):
            if segment == "<image>":
                if image_offset >= len(images):
                    raise ValueError(
                        f"image placeholder {image_offset} has no matching value in {image_key!r}"
                    )
                content_items.append(image_content_item(images[image_offset], max_pixels))
                image_offset += 1
            else:
                content_items.append({"type": "text", "text": segment})
        message["content"] = content_items

    if image_offset != len(images):
        raise ValueError(f"consumed {image_offset} images but {image_key!r} contains {len(images)}")
    return messages, images


def build_teacher_messages(privileged_text: str, template: str) -> list[dict[str, Any]]:
    try:
        teacher_prompt = template.format(privileged_text=privileged_text)
    except (KeyError, ValueError) as exc:
        raise ValueError("privileged template must contain a valid {privileged_text} placeholder") from exc
    return [{"role": "user", "content": teacher_prompt}]


def sample_identifier(record: dict[str, Any], line_number: int) -> str:
    extra_info = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
    for value in (
        record.get("id"),
        record.get("pair_id"),
        record.get("doc_id"),
        record.get("_row_index"),
        extra_info.get("index"),
    ):
        if value is not None and str(value).strip():
            return str(value)
    return f"line_{line_number:08d}"


def safe_slug(sample_id: str, line_number: int) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._-")[:80] or "sample"
    digest = hashlib.sha1(f"{line_number}:{sample_id}".encode("utf-8")).hexdigest()[:8]
    return f"{line_number:08d}_{readable}_{digest}"


def scan_records(
    dataset: Path,
    *,
    privileged_key: str,
    requested_ids: set[str],
    selection: str,
    sample_count: int,
    offset: int,
    seed: int,
    progress_interval: float,
) -> tuple[list[RecordLocator], dict[str, int]]:
    """Select records while retaining only byte locators, not large JSON rows."""
    total_bytes = dataset.stat().st_size
    started = time.monotonic()
    last_progress = started
    eligible = 0
    missing_privileged = 0
    malformed = 0
    selected: list[RecordLocator] = []
    rng = random.Random(seed)

    log(
        "SCAN start "
        f"dataset={dataset} bytes={total_bytes} selection={selection} sample_count={sample_count}"
    )
    with dataset.open("rb") as handle:
        line_number = 0
        while True:
            byte_offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise TypeError("row is not an object")
            except Exception:
                malformed += 1
                continue
            sample_id = sample_identifier(record, line_number)
            privileged = normalize_optional_text(get_nested_field(record, privileged_key))
            if privileged is None:
                missing_privileged += 1
                continue
            if requested_ids and sample_id not in requested_ids:
                continue
            if eligible < offset:
                eligible += 1
                continue

            locator = RecordLocator(
                line_number=line_number,
                byte_offset=byte_offset,
                byte_length=len(raw),
                sample_id=sample_id,
                slug=safe_slug(sample_id, line_number),
                source_sha256=hashlib.sha256(raw).hexdigest(),
            )
            eligible += 1
            if sample_count == 0:
                selected.append(locator)
            elif selection == "first":
                if len(selected) < sample_count:
                    selected.append(locator)
                elif not requested_ids:
                    break
            else:
                seen_after_offset = eligible - offset
                if len(selected) < sample_count:
                    selected.append(locator)
                else:
                    replacement = rng.randrange(seen_after_offset)
                    if replacement < sample_count:
                        selected[replacement] = locator

            now = time.monotonic()
            if now - last_progress >= progress_interval:
                elapsed = now - started
                completed_bytes = handle.tell()
                percentage = 100.0 * completed_bytes / max(total_bytes, 1)
                throughput = line_number / max(elapsed, 1e-9)
                byte_rate = completed_bytes / max(elapsed, 1e-9)
                eta = (total_bytes - completed_bytes) / byte_rate if byte_rate > 0 else None
                log(
                    "SCAN progress "
                    f"rows={line_number} bytes={completed_bytes}/{total_bytes} "
                    f"percent={percentage:.2f}% rows_per_s={throughput:.2f} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                    f"selected={len(selected)} missing_privileged={missing_privileged} malformed={malformed}"
                )
                last_progress = now

    selected.sort(key=lambda item: item.line_number)
    missing_requested = requested_ids - {item.sample_id for item in selected}
    if missing_requested:
        raise RuntimeError(f"requested sample IDs not found: {sorted(missing_requested)[:10]}")
    stats = {
        "scanned_rows": line_number,
        "eligible_rows": eligible,
        "selected_rows": len(selected),
        "missing_privileged_rows": missing_privileged,
        "malformed_rows": malformed,
    }
    log(f"SCAN completed {json.dumps(stats, ensure_ascii=False)}")
    return selected, stats


def load_record(dataset: Path, locator: RecordLocator) -> dict[str, Any]:
    with dataset.open("rb") as handle:
        handle.seek(locator.byte_offset)
        raw = handle.read(locator.byte_length)
    if hashlib.sha256(raw).hexdigest() != locator.source_sha256:
        raise RuntimeError(f"dataset changed after selection at line {locator.line_number}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSONL row must be an object")
    return value


def load_model(
    model_path: str,
    *,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
    heartbeat_seconds: float,
) -> ModelBundle:
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("install torch, transformers>=5.3, Pillow and qwen-vl-utils") from exc

    resolved_device = resolve_device(torch, device)
    resolved_dtype = resolve_dtype(torch, dtype, resolved_device)
    with Heartbeat(f"load-model:{model_path}", heartbeat_seconds):
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model_kwargs: dict[str, Any] = {
            "dtype": resolved_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if resolved_device.type != "cpu":
            model_kwargs["device_map"] = {"": str(resolved_device)}
        model = AutoModelForMultimodalLM.from_pretrained(model_path, **model_kwargs)
        if resolved_device.type == "cpu":
            model.to(resolved_device)
        model.eval()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor does not expose a tokenizer")
    return ModelBundle(model, processor, tokenizer, resolved_device, str(resolved_dtype))


def resolve_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(torch: Any, requested: str, device: Any) -> Any:
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[requested]


def move_inputs(inputs: Any, device: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        result[key] = value.to(device) if hasattr(value, "to") else value
    return result


def prepare_prompt_inputs(
    bundle: ModelBundle,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": enable_thinking,
    }
    try:
        inputs = bundle.processor.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        inputs = bundle.processor.apply_chat_template(messages, **kwargs)
    return move_inputs(inputs, bundle.device)


def generate_response(
    bundle: ModelBundle,
    prompt_inputs: dict[str, Any],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
    seed: int,
) -> tuple[list[int], str, list[float]]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prompt_length = int(prompt_inputs["input_ids"].shape[-1])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if not greedy:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        output = bundle.model.generate(**prompt_inputs, **generation_kwargs)
    response_ids = [int(value) for value in output.sequences[0, prompt_length:].detach().cpu().tolist()]
    score_tensors = list(output.scores or [])

    eos_values = bundle.tokenizer.eos_token_id
    eos_ids = {int(value) for value in (eos_values if isinstance(eos_values, list) else [eos_values]) if value is not None}
    stop = next((index + 1 for index, token_id in enumerate(response_ids) if token_id in eos_ids), len(response_ids))
    response_ids = response_ids[:stop]
    score_tensors = score_tensors[:stop]

    rollout_logprobs: list[float] = []
    for token_id, scores in zip(response_ids, score_tensors):
        row = scores[0].float()
        rollout_logprobs.append(float(torch.log_softmax(row, dim=-1)[token_id].detach().cpu()))
    if len(rollout_logprobs) != len(response_ids):
        raise RuntimeError(
            f"generation returned {len(response_ids)} IDs but {len(rollout_logprobs)} score rows"
        )
    text = bundle.tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return response_ids, text, rollout_logprobs


def append_response_ids(prompt_inputs: dict[str, Any], response_ids: Sequence[int]) -> dict[str, Any]:
    import torch

    input_ids = prompt_inputs["input_ids"]
    response = torch.tensor([list(map(int, response_ids))], dtype=input_ids.dtype, device=input_ids.device)
    scoring = {
        key: value.clone() if hasattr(value, "clone") else value for key, value in prompt_inputs.items()
    }
    scoring["input_ids"] = torch.cat([input_ids, response], dim=-1)
    if "attention_mask" in scoring:
        attention = scoring["attention_mask"]
        extension = torch.ones(
            (attention.shape[0], len(response_ids)), dtype=attention.dtype, device=attention.device
        )
        scoring["attention_mask"] = torch.cat([attention, extension], dim=-1)
    scoring.pop("position_ids", None)
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key not in scoring:
            continue
        values = scoring[key]
        extension = torch.zeros(
            (values.shape[0], len(response_ids)), dtype=values.dtype, device=values.device
        )
        scoring[key] = torch.cat([values, extension], dim=-1)
    return scoring


def score_fixed_response(
    bundle: ModelBundle,
    prompt_inputs: dict[str, Any],
    response_ids: Sequence[int],
    *,
    top_k: int,
    probability_chunk_size: int,
) -> TokenStats:
    import torch

    if not response_ids:
        raise ValueError("response_ids must not be empty")
    prompt_length = int(prompt_inputs["input_ids"].shape[-1])
    scoring_inputs = append_response_ids(prompt_inputs, response_ids)
    forward_kwargs = dict(scoring_inputs)
    forward_kwargs["use_cache"] = False
    supports_keep = "logits_to_keep" in inspect.signature(bundle.model.forward).parameters
    if supports_keep:
        forward_kwargs["logits_to_keep"] = len(response_ids) + 1
    with torch.inference_mode():
        outputs = bundle.model(**forward_kwargs)
    logits = outputs.logits[0]
    if supports_keep and logits.shape[0] <= len(response_ids) + 1:
        if logits.shape[0] < len(response_ids) + 1:
            raise RuntimeError("model returned too few logits for response scoring")
        target_logits = logits[-(len(response_ids) + 1) : -1]
    else:
        target_logits = logits[prompt_length - 1 : prompt_length - 1 + len(response_ids)]
    if target_logits.shape[0] != len(response_ids):
        raise RuntimeError(
            f"response/logit alignment failed: response={len(response_ids)} logits={target_logits.shape[0]}"
        )

    target_probabilities: list[float] = []
    target_logprobs: list[float] = []
    target_ranks: list[int] = []
    entropies: list[float] = []
    top_candidates: list[list[dict[str, Any]]] = []
    for start in range(0, len(response_ids), probability_chunk_size):
        end = min(len(response_ids), start + probability_chunk_size)
        chunk = target_logits[start:end].float()
        log_normalizer = torch.logsumexp(chunk, dim=-1)
        targets = torch.tensor(response_ids[start:end], dtype=torch.long, device=chunk.device)
        selected_logits = chunk.gather(1, targets[:, None]).squeeze(1)
        selected_logp = selected_logits - log_normalizer
        ranks = 1 + (chunk > selected_logits[:, None]).sum(dim=-1)
        top_values, top_ids = torch.topk(chunk, k=min(top_k, chunk.shape[-1]), dim=-1)
        top_logp = top_values - log_normalizer[:, None]
        log_probs = torch.log_softmax(chunk, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

        target_logprobs.extend(float(value) for value in selected_logp.detach().cpu())
        target_probabilities.extend(float(value) for value in selected_logp.exp().detach().cpu())
        target_ranks.extend(int(value) for value in ranks.detach().cpu())
        entropies.extend(float(value) for value in entropy.detach().cpu())
        for ids, values in zip(top_ids.detach().cpu(), top_logp.detach().cpu()):
            candidates = []
            for rank, (token_id, logp) in enumerate(zip(ids.tolist(), values.tolist()), start=1):
                raw = decode_token_piece(bundle.tokenizer, int(token_id))
                candidates.append(
                    {
                        "rank": rank,
                        "token_id": int(token_id),
                        "token": display_token(raw),
                        "raw_token": raw,
                        "probability": math.exp(float(logp)),
                        "logprob": float(logp),
                    }
                )
            top_candidates.append(candidates)
        del chunk, log_probs
    del outputs, logits, target_logits
    return TokenStats(
        target_probabilities,
        target_logprobs,
        target_ranks,
        entropies,
        top_candidates,
    )


def decode_token_piece(tokenizer: Any, token_id: int) -> str:
    try:
        value = tokenizer.decode(
            [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        value = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if value == "":
        value = str(tokenizer.convert_ids_to_tokens(int(token_id)))
    return value


def display_token(value: str) -> str:
    return value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def find_text_equivalent_candidate(
    tokenizer: Any,
    response_ids: Sequence[int],
    index: int,
    candidates: Sequence[dict[str, Any]],
    max_span: int,
) -> dict[str, Any] | None:
    target_id = int(response_ids[index])
    max_length = min(max_span, len(response_ids) - index)
    for span_length in range(1, max_length + 1):
        span_ids = [int(value) for value in response_ids[index : index + span_length]]
        span_text = tokenizer.decode(
            span_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for candidate in candidates:
            candidate_id = int(candidate["token_id"])
            candidate_text = tokenizer.decode(
                [candidate_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if candidate_text != span_text:
                continue
            if span_length == 1 and candidate_id == target_id:
                continue
            return {
                "candidate_token_id": candidate_id,
                "candidate_token": candidate.get("token"),
                "candidate_probability": float(candidate["probability"]),
                "candidate_rank": int(candidate["rank"]),
                "target_span_length": span_length,
                "target_span_text": span_text,
                "equivalence_type": "same_text_different_id" if span_length == 1 else "teacher_merge",
            }
    return None


def build_token_rows(
    bundle: ModelBundle,
    response_ids: Sequence[int],
    rollout_logprobs: Sequence[float],
    student: TokenStats,
    teacher: TokenStats,
    thresholds: Thresholds,
) -> list[dict[str, Any]]:
    lengths = {
        len(response_ids),
        len(rollout_logprobs),
        len(student.target_logprobs),
        len(teacher.target_logprobs),
    }
    if len(lengths) != 1:
        raise RuntimeError(f"token statistic lengths do not agree: {sorted(lengths)}")
    eos_values = bundle.tokenizer.eos_token_id
    eos_ids = {int(value) for value in (eos_values if isinstance(eos_values, list) else [eos_values]) if value is not None}
    rows: list[dict[str, Any]] = []
    consumed_until = 0
    for index, token_id in enumerate(response_ids):
        student_p = student.target_probabilities[index]
        teacher_p = teacher.target_probabilities[index]
        student_logp = student.target_logprobs[index]
        teacher_logp = teacher.target_logprobs[index]
        gap = teacher_logp - student_logp
        teacher_rejects = student_p > thresholds.student_high and teacher_p < thresholds.teacher_low
        teacher_promotes = teacher_p / max(student_p, 1e-300) >= thresholds.probability_ratio
        equivalent = find_text_equivalent_candidate(
            bundle.tokenizer,
            response_ids,
            index,
            teacher.top_candidates[index],
            thresholds.max_equivalent_span,
        )
        covered = index < consumed_until
        if covered:
            classification = "TOKENIZATION_SPAN_CONTINUATION"
        elif equivalent is not None and equivalent["candidate_rank"] == 1:
            consumed_until = max(consumed_until, index + int(equivalent["target_span_length"]))
            classification = "TOKENIZATION_EQUIVALENT"
        elif int(token_id) in eos_ids and abs(gap) >= math.log(thresholds.probability_ratio):
            classification = "EOS_SHIFT"
        elif teacher_rejects and teacher.entropies[index] >= thresholds.teacher_uncertain_entropy:
            classification = "TEACHER_UNCERTAIN"
        elif teacher_rejects:
            classification = "TEACHER_REJECTS_STUDENT_TOKEN"
        elif teacher_promotes:
            classification = "TEACHER_PROMOTES_STUDENT_TOKEN"
        else:
            classification = "NO_STRONG_DISAGREEMENT"
        raw = decode_token_piece(bundle.tokenizer, int(token_id))
        context_start = max(0, index - 16)
        context_end = min(len(response_ids), index + 17)
        context_text = bundle.tokenizer.decode(
            list(response_ids[context_start:context_end]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        rows.append(
            {
                "index": index,
                "token_id": int(token_id),
                "token": display_token(raw),
                "raw_token": raw,
                "token_piece_repr": repr(bundle.tokenizer.convert_ids_to_tokens(int(token_id))),
                "single_decode_repr": repr(raw),
                "context_start": context_start,
                "context_end": context_end,
                "context_text": context_text,
                "rollout_logprob_t_sampling": float(rollout_logprobs[index]),
                "rollout_probability_t_sampling": math.exp(float(rollout_logprobs[index])),
                "student_probability_t1": student_p,
                "student_logprob_t1": student_logp,
                "student_rank_t1": student.target_ranks[index],
                "student_entropy_t1": student.entropies[index],
                "student_top_candidates": student.top_candidates[index],
                "teacher_probability_t1": teacher_p,
                "teacher_logprob_t1": teacher_logp,
                "teacher_rank_t1": teacher.target_ranks[index],
                "teacher_entropy_t1": teacher.entropies[index],
                "teacher_top_candidates": teacher.top_candidates[index],
                "delta_logp_teacher_minus_student": gap,
                "teacher_to_student_probability_ratio": teacher_p / max(student_p, 1e-300),
                "student_to_teacher_probability_ratio": student_p / max(teacher_p, 1e-300),
                "teacher_confidence_filtered": teacher_rejects,
                "is_eos": int(token_id) in eos_ids,
                "covered_by_previous_equivalent_span": covered,
                "text_equivalent_candidate": equivalent,
                "comparison_class": classification,
            }
        )
    return rows


def summarize_tokens(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"token_count": 0}
    gaps = [float(row["delta_logp_teacher_minus_student"]) for row in rows]
    abs_gaps = sorted(abs(value) for value in gaps)
    classes: dict[str, int] = {}
    for row in rows:
        label = str(row["comparison_class"])
        classes[label] = classes.get(label, 0) + 1
    return {
        "token_count": len(rows),
        "mean_student_probability": statistics.fmean(float(row["student_probability_t1"]) for row in rows),
        "mean_teacher_probability": statistics.fmean(float(row["teacher_probability_t1"]) for row in rows),
        "mean_teacher_minus_student_logp": statistics.fmean(gaps),
        "mean_absolute_logprob_gap": statistics.fmean(abs(value) for value in gaps),
        "p90_absolute_logprob_gap": percentile(abs_gaps, 0.90),
        "p99_absolute_logprob_gap": percentile(abs_gaps, 0.99),
        "max_absolute_logprob_gap": max(abs_gaps),
        "teacher_confidence_filtered_count": sum(bool(row["teacher_confidence_filtered"]) for row in rows),
        "teacher_confidence_filtered_ratio": sum(bool(row["teacher_confidence_filtered"]) for row in rows) / len(rows),
        "eos_count": sum(bool(row["is_eos"]) for row in rows),
        "class_counts": classes,
    }


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def tensor_ids(inputs: dict[str, Any]) -> list[int]:
    return [int(value) for value in inputs["input_ids"][0].detach().cpu().tolist()]


def process_sample(
    *,
    locator: RecordLocator,
    record: dict[str, Any],
    dataset: Path,
    bundle: ModelBundle,
    output_dir: Path,
    args: argparse.Namespace,
    thresholds: Thresholds,
) -> tuple[dict[str, Any], bool]:
    sample_dir = output_dir / "samples" / locator.slug
    result_path = sample_dir / "data.json"
    fingerprint_data = {
        "schema": SCHEMA_VERSION,
        "source": locator.source_sha256,
        "model": args.model,
        "prompt_key": args.prompt_key,
        "image_key": args.image_key,
        "privileged_key": args.privileged_key,
        "privileged_template": args.privileged_template,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "greedy": args.greedy,
        "top_k": args.top_k,
        "seed": args.seed + locator.line_number,
        "thresholds": asdict(thresholds),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if args.resume and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            write_sample_artifacts(sample_dir, existing)
            return existing, True

    privileged_text = normalize_optional_text(get_nested_field(record, args.privileged_key))
    if privileged_text is None:
        raise ValueError(f"missing privileged text at {args.privileged_key!r}")
    student_messages, images = build_student_messages(
        record,
        prompt_key=args.prompt_key,
        image_key=args.image_key,
        dataset_dir=dataset.parent,
        max_pixels=args.max_pixels,
    )
    teacher_messages = build_teacher_messages(privileged_text, args.privileged_template)

    with Heartbeat(f"sample:{locator.sample_id}:prepare-student", args.heartbeat_seconds):
        student_inputs = prepare_prompt_inputs(
            bundle, student_messages, enable_thinking=args.enable_thinking
        )
    with Heartbeat(f"sample:{locator.sample_id}:prepare-teacher", args.heartbeat_seconds):
        teacher_inputs = prepare_prompt_inputs(
            bundle, teacher_messages, enable_thinking=args.enable_thinking
        )
    student_prompt_ids = tensor_ids(student_inputs)
    teacher_prompt_ids = tensor_ids(teacher_inputs)

    with Heartbeat(f"sample:{locator.sample_id}:generate", args.heartbeat_seconds):
        response_ids, response_text, rollout_logprobs = generate_response(
            bundle,
            student_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
            seed=args.seed + locator.line_number,
        )
    if not response_ids:
        raise RuntimeError("model generated no response tokens")

    with Heartbeat(f"sample:{locator.sample_id}:score-student", args.heartbeat_seconds):
        student_stats = score_fixed_response(
            bundle,
            student_inputs,
            response_ids,
            top_k=args.top_k,
            probability_chunk_size=args.probability_chunk_size,
        )
    empty_device_cache(bundle.device)
    with Heartbeat(f"sample:{locator.sample_id}:score-teacher", args.heartbeat_seconds):
        teacher_stats = score_fixed_response(
            bundle,
            teacher_inputs,
            response_ids,
            top_k=args.top_k,
            probability_chunk_size=args.probability_chunk_size,
        )
    empty_device_cache(bundle.device)

    rows = build_token_rows(
        bundle,
        response_ids,
        rollout_logprobs,
        student_stats,
        teacher_stats,
        thresholds,
    )
    relative_images = copy_report_images(images, sample_dir)
    result = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "sample_id": locator.sample_id,
        "slug": locator.slug,
        "line_number": locator.line_number,
        "source_sha256": locator.source_sha256,
        "source_dataset": str(dataset),
        "model": args.model,
        "protocol": {
            "student_generated_once": True,
            "same_response_ids_scored_in_both_contexts": True,
            "teacher_prompt_mode": "standalone",
            "teacher_uses_images": False,
            "privileged_key": args.privileged_key,
            "privileged_template": args.privileged_template,
            "sampling_temperature": None if args.greedy else args.temperature,
            "sampling_top_p": None if args.greedy else args.top_p,
            "student_teacher_forcing_temperature": 1.0,
            "teacher_teacher_forcing_temperature": 1.0,
            "max_pixels": args.max_pixels,
            "student_prompt_tokens": len(student_prompt_ids),
            "teacher_prompt_tokens": len(teacher_prompt_ids),
        },
        "images": relative_images,
        "source_images": [image_source_string(value) for value in images],
        "student_messages": json_safe(student_messages),
        "teacher_messages": teacher_messages,
        "privileged_text": privileged_text,
        "response": {
            "text": response_text,
            "token_ids": response_ids,
            "token_count": len(response_ids),
        },
        "summary": summarize_tokens(rows),
        "thresholds": asdict(thresholds),
        "tokens": rows,
    }
    write_sample_artifacts(sample_dir, result)
    return result, False


def image_source_string(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("image", value.get("path", value.get("url", value))))
    return str(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if hasattr(value, "size") and value.__class__.__module__.startswith("PIL"):
        return f"<PIL.Image size={value.size}>"
    return value


def copy_report_images(images: Sequence[Any], sample_dir: Path) -> list[str]:
    from PIL import Image

    outputs: list[str] = []
    for index, image in enumerate(images):
        source = image_source_string(image)
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", source):
            outputs.append(source)
            continue
        source_path = Path(source)
        if not source_path.is_file():
            outputs.append(source)
            continue
        target = sample_dir / f"input_{index:02d}.webp"
        with Image.open(source_path) as loaded:
            converted = loaded.convert("RGB")
            converted.thumbnail((1800, 1800))
            target.parent.mkdir(parents=True, exist_ok=True)
            converted.save(target, format="WEBP", quality=88, method=4)
        outputs.append(target.name)
    return outputs


def empty_device_cache(device: Any) -> None:
    try:
        import torch

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def write_sample_page(sample_dir: Path) -> None:
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Token 差异详情</title>
  <link rel="stylesheet" href="../../assets/report.css">
</head>
<body data-page="sample">
  <div id="app"><div class="loading">正在加载样本数据...</div></div>
  <script src="../../manifest.js" defer></script>
  <script src="data.js" defer></script>
  <script src="../../assets/sample.js" defer></script>
</body>
</html>
"""
    atomic_write_text(sample_dir / "index.html", html)


def write_sample_artifacts(sample_dir: Path, result: dict[str, Any]) -> None:
    write_sample_page(sample_dir)
    atomic_write_json(sample_dir / "data.json", result)
    atomic_write_javascript_assignment(
        sample_dir / "data.js",
        "__PRIVILEGED_SAMPLE__",
        result,
    )


def write_report_shell(output_dir: Path) -> None:
    source_assets = Path(__file__).resolve().parent / "assets"
    target_assets = output_dir / "assets"
    target_assets.mkdir(parents=True, exist_ok=True)
    for source in source_assets.iterdir():
        if source.is_file():
            shutil.copy2(source, target_assets / source.name)
    index_html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>特权条件 Token 概率对比</title>
  <link rel="stylesheet" href="assets/report.css">
</head>
<body data-page="index">
  <div id="app"><div class="loading">正在加载批次清单...</div></div>
  <script src="manifest.js" defer></script>
  <script src="assets/index.js" defer></script>
</body>
</html>
"""
    atomic_write_text(output_dir / "index.html", index_html)


def rebuild_manifest(output_dir: Path, scan_stats: dict[str, int] | None = None) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for result_path in sorted((output_dir / "samples").glob("*/data.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        samples.append(
            {
                "sample_id": result["sample_id"],
                "slug": result["slug"],
                "line_number": result["line_number"],
                "url": f"samples/{result['slug']}/index.html",
                "thumbnail": (
                    f"samples/{result['slug']}/{result['images'][0]}" if result.get("images") else None
                ),
                "response_preview": str(result["response"]["text"])[:240],
                "summary": result["summary"],
            }
        )
    total_tokens = sum(int(item["summary"].get("token_count", 0)) for item in samples)
    filtered_tokens = sum(
        int(item["summary"].get("teacher_confidence_filtered_count", 0)) for item in samples
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(samples),
        "total_tokens": total_tokens,
        "teacher_confidence_filtered_tokens": filtered_tokens,
        "teacher_confidence_filtered_ratio": filtered_tokens / total_tokens if total_tokens else 0.0,
        "scan": scan_stats or {},
        "samples": samples,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_javascript_assignment(
        output_dir / "manifest.js",
        "__PRIVILEGED_MANIFEST__",
        manifest,
    )
    return manifest


def run(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_report_shell(output_dir)
    thresholds = Thresholds(
        student_high=args.student_high,
        teacher_low=args.teacher_low,
        probability_ratio=args.probability_ratio,
        teacher_uncertain_entropy=args.teacher_uncertain_entropy,
        max_equivalent_span=args.max_equivalent_span,
    )
    selected, scan_stats = scan_records(
        dataset,
        privileged_key=args.privileged_key,
        requested_ids=set(args.sample_ids or []),
        selection=args.selection,
        sample_count=args.sample_count,
        offset=args.offset,
        seed=args.seed,
        progress_interval=args.heartbeat_seconds,
    )
    if not selected:
        raise RuntimeError("no eligible records selected")
    config = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset),
        "output": str(output_dir),
        "model": args.model,
        "selected": [asdict(item) for item in selected],
        "arguments": {key: value for key, value in vars(args).items() if key != "function"},
        "thresholds": asdict(thresholds),
    }
    atomic_write_json(output_dir / "config.json", config)
    bundle = load_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    started = time.monotonic()
    accepted = skipped = errors = 0
    for ordinal, locator in enumerate(selected, start=1):
        sample_started = time.monotonic()
        try:
            record = load_record(dataset, locator)
            _, reused = process_sample(
                locator=locator,
                record=record,
                dataset=dataset,
                bundle=bundle,
                output_dir=output_dir,
                args=args,
                thresholds=thresholds,
            )
            if reused:
                skipped += 1
                status = "reused"
            else:
                accepted += 1
                status = "accepted"
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            errors += 1
            status = "error"
            append_jsonl(
                output_dir / "failures.jsonl",
                {
                    "sample_id": locator.sample_id,
                    "line_number": locator.line_number,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            log(f"ERROR sample={locator.sample_id} type={type(exc).__name__} message={exc}")
        rebuild_manifest(output_dir, scan_stats)
        completed = ordinal
        elapsed = time.monotonic() - started
        rate = completed / max(elapsed, 1e-9)
        eta = (len(selected) - completed) / rate if rate > 0 else None
        log(
            "SAMPLE complete "
            f"current={completed}/{len(selected)} sample={locator.sample_id} status={status} "
            f"sample_elapsed={format_duration(time.monotonic() - sample_started)} "
            f"global_elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
            f"samples_per_hour={rate * 3600:.2f} accepted={accepted} skipped={skipped} errors={errors}"
        )
    manifest = rebuild_manifest(output_dir, scan_stats)
    log(
        "RUN completed "
        f"samples={manifest['sample_count']} tokens={manifest['total_tokens']} "
        f"accepted={accepted} skipped={skipped} errors={errors} elapsed={format_duration(time.monotonic()-started)}"
    )
    log(f"REPORT html={output_dir / 'index.html'}")
    return 0 if errors == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Generate and score a batch of samples")
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--prompt-key", default="prompt")
    run_parser.add_argument("--image-key", default="images")
    run_parser.add_argument("--privileged-key", default=DEFAULT_PRIVILEGED_KEY)
    run_parser.add_argument("--privileged-template", default=DEFAULT_PRIVILEGED_TEMPLATE)
    run_parser.add_argument("--sample-id", action="append", dest="sample_ids")
    run_parser.add_argument("--selection", choices=("first", "random"), default="random")
    run_parser.add_argument("--sample-count", type=int, default=8, help="0 means all eligible rows")
    run_parser.add_argument("--offset", type=int, default=0)
    run_parser.add_argument("--max-new-tokens", type=int, default=1024)
    run_parser.add_argument("--temperature", type=float, default=0.6)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--greedy", action="store_true")
    run_parser.add_argument("--top-k", type=int, default=10)
    run_parser.add_argument("--probability-chunk-size", type=int, default=16)
    run_parser.add_argument("--max-pixels", type=int, default=4194304)
    run_parser.add_argument("--enable-thinking", action="store_true")
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    run_parser.add_argument("--attn-implementation")
    run_parser.add_argument("--trust-remote-code", action="store_true")
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--student-high", type=float, default=0.40)
    run_parser.add_argument("--teacher-low", type=float, default=0.01)
    run_parser.add_argument("--probability-ratio", type=float, default=10.0)
    run_parser.add_argument("--teacher-uncertain-entropy", type=float, default=3.0)
    run_parser.add_argument("--max-equivalent-span", type=int, default=4)
    run_parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    run_parser.add_argument("--no-resume", action="store_false", dest="resume")
    run_parser.set_defaults(resume=True, function=run)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command != "run":
        return
    positive_fields = (
        "max_new_tokens",
        "top_k",
        "probability_chunk_size",
        "max_pixels",
        "probability_ratio",
        "heartbeat_seconds",
        "max_equivalent_span",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.sample_count < 0 or args.offset < 0:
        raise ValueError("--sample-count and --offset must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if not args.greedy and args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 <= args.student_high <= 1 or not 0 <= args.teacher_low <= 1:
        raise ValueError("probability thresholds must be in [0, 1]")
    if "{privileged_text}" not in args.privileged_template:
        raise ValueError("--privileged-template must contain {privileged_text}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
