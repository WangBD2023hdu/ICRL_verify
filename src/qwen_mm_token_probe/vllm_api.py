from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


class RequestLimiter:
    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds < 0:
            raise ValueError("request_interval_seconds must be non-negative")
        self.interval_seconds = interval_seconds
        self.lock = threading.Lock()
        self.next_request_at = 0.0

    def wait(self) -> None:
        if self.interval_seconds == 0:
            return
        with self.lock:
            now = time.monotonic()
            request_at = max(now, self.next_request_at)
            self.next_request_at = request_at + self.interval_seconds
        delay = request_at - now
        if delay > 0:
            time.sleep(delay)


class ThreadLocalClients:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        verify_tls: bool,
        timeout_seconds: float,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.verify_tls = verify_tls
        self.timeout_seconds = timeout_seconds
        self.local = threading.local()
        self.clients: list[Any] = []
        self.lock = threading.Lock()

    def get(self) -> Any:
        client = getattr(self.local, "client", None)
        if client is not None:
            return client
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI API support requires the 'api' extra: "
                "pip install -e '.[api]'"
            ) from exc
        http_client = httpx.Client(
            verify=self.verify_tls,
            timeout=httpx.Timeout(self.timeout_seconds),
        )
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=http_client,
        )
        self.local.client = client
        with self.lock:
            self.clients.append(client)
        return client

    def close(self) -> None:
        with self.lock:
            clients = list(self.clients)
            self.clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - cleanup must preserve experiment results
                pass


def generate_response(
    *,
    client: Any,
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    top_logprobs: int,
    seed: int,
    max_retries: int,
    retry_base_seconds: float,
    request_limiter: RequestLimiter,
    request_label: str,
) -> dict[str, object]:
    response = _request_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=_image_messages(image_path=image_path, prompt=prompt),
            temperature=0,
            top_p=1,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=True,
            top_logprobs=top_logprobs,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "return_token_ids": True,
            },
        ),
        label=request_label,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        request_limiter=request_limiter,
    )
    if not response.choices:
        raise RuntimeError(f"{request_label} returned no choices")
    choice = response.choices[0]
    text = choice.message.content or ""
    logprob_items = list(choice.logprobs.content or []) if choice.logprobs else []
    token_ids = list((choice.model_extra or {}).get("token_ids") or [])
    scoreable_count = _scoreable_prefix_length(text, logprob_items)
    if scoreable_count <= 0:
        raise RuntimeError(
            f"{request_label} generated no scoreable content "
            f"(finish_reason={choice.finish_reason!r})"
        )
    if len(token_ids) < scoreable_count:
        raise RuntimeError(
            f"{request_label} did not return enough token IDs: "
            f"ids={len(token_ids)}, scoreable={scoreable_count}; "
            "the vLLM server must support return_token_ids"
        )

    records: list[dict[str, object]] = []
    for index, (token_id, item) in enumerate(
        zip(token_ids[:scoreable_count], logprob_items[:scoreable_count])
    ):
        raw_token = _decoded_piece(item)
        top_candidates = list(item.top_logprobs or [])
        top = max(top_candidates, key=lambda candidate: float(candidate.logprob), default=item)
        top_raw = _decoded_piece(top)
        top_id = (
            int(token_id)
            if _candidate_bytes(top) == _candidate_bytes(item)
            else _stable_candidate_id(top_raw)
        )
        logp = float(item.logprob)
        top_logp = float(top.logprob)
        records.append(
            {
                "index": index,
                "token_id": int(token_id),
                "token": _display_token(raw_token),
                "raw_token": raw_token,
                "p_original": _probability(logp),
                "logp_original": logp,
                "top_token_id_original": top_id,
                "top_token_original": _display_token(top_raw),
                "top_raw_token_original": top_raw,
                "top_p_original": _probability(top_logp),
                "top_logp_original": top_logp,
                "target_rank_original": 1 if top_id == int(token_id) else None,
            }
        )
    return {
        "text": text,
        "finish_reason": choice.finish_reason,
        "tokens": records,
    }


def canonical_prompt_scores(
    *,
    client: Any,
    model: str,
    image_path: Path | None,
    prompt: str,
    response_text: str,
    prompt_logprobs: int,
    seed: int,
    max_retries: int,
    retry_base_seconds: float,
    request_limiter: RequestLimiter,
    request_label: str,
    expected_token_ids: Sequence[int] | None,
) -> list[dict[str, object]]:
    messages = (
        _image_messages(image_path=image_path, prompt=prompt)
        if image_path is not None
        else [{"role": "user", "content": prompt}]
    )
    messages.append({"role": "assistant", "content": response_text})
    response = _request_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            top_p=1,
            max_tokens=1,
            seed=seed,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "prompt_logprobs": prompt_logprobs,
                "continue_final_message": True,
                "add_generation_prompt": False,
                "return_token_ids": True,
            },
        ),
        label=request_label,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        request_limiter=request_limiter,
    )
    extras = response.model_extra or {}
    prompt_ids = list(extras.get("prompt_token_ids") or [])
    all_logprobs = list(extras.get("prompt_logprobs") or [])
    if len(prompt_ids) != len(all_logprobs):
        raise RuntimeError(
            f"{request_label} returned unaligned prompt arrays: "
            f"ids={len(prompt_ids)}, logprobs={len(all_logprobs)}"
        )
    start = _find_response_suffix_start(
        prompt_ids=prompt_ids,
        prompt_logprobs=all_logprobs,
        response_text=response_text,
        request_label=request_label,
    )
    scored_ids = [int(value) for value in prompt_ids[start:]]
    entries = all_logprobs[start:]
    expected_ids = (
        [int(value) for value in expected_token_ids]
        if expected_token_ids is not None
        else None
    )
    if expected_ids is not None and scored_ids != expected_ids:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(zip(scored_ids, expected_ids))
                if actual != expected
            ),
            min(len(scored_ids), len(expected_ids)),
        )
        raise RuntimeError(
            f"{request_label} response-token alignment failed at index {mismatch}: "
            f"scored={scored_ids[mismatch:mismatch + 3]}, "
            f"expected={expected_ids[mismatch:mismatch + 3]}"
        )

    records: list[dict[str, object]] = []
    for index, (target_id, entry) in enumerate(zip(scored_ids, entries)):
        if not isinstance(entry, dict):
            raise RuntimeError(f"{request_label} has no logprobs at response index {index}")
        target_data = entry.get(str(target_id), entry.get(target_id))
        if not isinstance(target_data, dict):
            raise RuntimeError(
                f"{request_label} omitted target token {target_id} at response index {index}"
            )
        candidates = [
            (int(candidate_id), candidate)
            for candidate_id, candidate in entry.items()
            if isinstance(candidate, dict) and "logprob" in candidate
        ]
        if not candidates:
            raise RuntimeError(f"{request_label} has an empty candidate set at index {index}")
        top_id, top_data = min(
            candidates,
            key=lambda pair: (
                int(pair[1].get("rank", 10**9)),
                -float(pair[1]["logprob"]),
            ),
        )
        target_logp = float(target_data["logprob"])
        top_logp = float(top_data["logprob"])
        top_raw = str(top_data.get("decoded_token", top_id))
        records.append(
            {
                "token_id": target_id,
                "token": _display_token(str(target_data.get("decoded_token", target_id))),
                "raw_token": str(target_data.get("decoded_token", target_id)),
                "probability": _probability(target_logp),
                "log_probability": target_logp,
                "top_token_id": top_id,
                "top_token": _display_token(top_raw),
                "top_raw_token": top_raw,
                "top_probability": _probability(top_logp),
                "top_log_probability": top_logp,
                "target_rank": int(target_data.get("rank", -1)),
            }
        )
    return records


def safe_error(exc: BaseException) -> str:
    value = str(exc)
    value = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image-data>", value)
    return value[:1000]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_response_suffix_start(
    *,
    prompt_ids: Sequence[int],
    prompt_logprobs: Sequence[object],
    response_text: str,
    request_label: str,
) -> int:
    suffix = ""
    for index in range(len(prompt_ids) - 1, -1, -1):
        entry = prompt_logprobs[index]
        token_id = int(prompt_ids[index])
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"{request_label} has no target logprob while locating response suffix "
                f"at prompt index {index}"
            )
        target_data = entry.get(str(token_id), entry.get(token_id))
        if not isinstance(target_data, dict):
            raise RuntimeError(
                f"{request_label} omitted prompt token {token_id} at index {index}"
            )
        suffix = str(target_data.get("decoded_token", "")) + suffix
        if suffix == response_text:
            return index
        if not response_text.endswith(suffix):
            break
    raise RuntimeError(
        f"{request_label} could not locate the fixed response at the prompt tail: "
        f"response_characters={len(response_text)}, matched_suffix_characters={len(suffix)}"
    )


def _scoreable_prefix_length(text: str, logprob_items: Sequence[Any]) -> int:
    target = text.encode("utf-8")
    consumed = b""
    if not target:
        return 0
    for index, item in enumerate(logprob_items, start=1):
        candidate = _candidate_bytes(item)
        next_value = consumed + candidate
        if not target.startswith(next_value):
            break
        consumed = next_value
        if consumed == target:
            return index
    raise RuntimeError(
        "generated token bytes do not reconstruct message.content: "
        f"reconstructed={len(consumed)} bytes, content={len(target)} bytes"
    )


def _image_messages(*, image_path: Path, prompt: str) -> list[dict[str, object]]:
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _request_with_retry(
    request: Callable[[], Any],
    *,
    label: str,
    max_retries: int,
    retry_base_seconds: float,
    request_limiter: RequestLimiter,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            request_limiter.wait()
            return request()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - API client errors vary by version
            status_code = getattr(exc, "status_code", None)
            retryable_status = status_code in {408, 409, 425, 429} or (
                isinstance(status_code, int) and status_code >= 500
            )
            retryable = status_code is None or retryable_status
            if attempt >= max_retries or not retryable:
                raise
            delay = min(60.0, retry_base_seconds * (2**attempt))
            print(
                f"[qwen-mm-vllm-api] RETRY request={json.dumps(label)} "
                f"attempt={attempt + 1}/{max_retries} delay_seconds={delay:g} "
                f"error={json.dumps(type(exc).__name__ + ': ' + safe_error(exc))}",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _candidate_bytes(candidate: Any) -> bytes:
    values = getattr(candidate, "bytes", None)
    if values is not None:
        return bytes(values)
    return str(getattr(candidate, "token", "")).encode("utf-8")


def _decoded_piece(candidate: Any) -> str:
    values = getattr(candidate, "bytes", None)
    if values is not None:
        try:
            return bytes(values).decode("utf-8")
        except UnicodeDecodeError:
            pass
    return str(getattr(candidate, "token", ""))


def _display_token(value: str) -> str:
    return value.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _stable_candidate_id(raw_token: str) -> int:
    digest = hashlib.sha1(raw_token.encode("utf-8")).digest()[:8]
    return -1 - int.from_bytes(digest, "big")


def _probability(log_probability: float) -> float:
    return math.exp(max(-745.0, min(0.0, log_probability)))
