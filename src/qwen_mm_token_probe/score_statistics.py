"""Chunked exact token statistics for already-computed language-model logits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def collect_token_statistics(
    target_logits: Any,
    response_ids: Sequence[int],
    *,
    probe_token_ids: Sequence[int] | None = None,
    top_k: int,
    chunk_size: int,
) -> dict[str, list[Any] | None]:
    """Collect exact per-position token statistics without chunkwise host syncs.

    ``target_logits`` has shape ``[response_length, vocab_size]`` and is the
    model's existing next-token logits for each response position. The math is
    intentionally identical to the historical scoring implementation: logits
    are cast to float32 per chunk, probabilities use logsumexp, ranks count
    strictly-greater logits, and entropy uses softmax/log_softmax.

    Only compact summaries are retained between chunks. The compact float and
    integer buffers are copied to CPU once each after all chunks are complete.
    """

    import torch

    if target_logits.ndim != 2:
        raise ValueError("target_logits must have shape [response_length, vocab_size]")
    if not response_ids:
        raise ValueError("response_ids must not be empty")
    response_length, vocab_size = target_logits.shape
    if response_length != len(response_ids):
        raise ValueError(
            "target_logits first dimension must match response_ids length: "
            f"logits={response_length} ids={len(response_ids)}"
        )
    if probe_token_ids is not None and len(probe_token_ids) != len(response_ids):
        raise ValueError("probe_token_ids must have the same length as response_ids")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # This utility reports inference statistics, not differentiable outputs.
    # Detaching also ensures the compact buffers cannot retain per-chunk graphs.
    target_logits = target_logits.detach()
    device = target_logits.device
    target_ids = torch.as_tensor(response_ids, dtype=torch.long, device=device)
    probe_ids = (
        torch.as_tensor(probe_token_ids, dtype=torch.long, device=device)
        if probe_token_ids is not None
        else None
    )
    kept_top_k = min(int(top_k), int(vocab_size))
    probe_column = 2 + kept_top_k

    # Float layout: selected log-probability, entropy, top-k log-probabilities,
    # then optional probe log-probability. Integer layout preserves token IDs
    # as int64 rather than round-tripping them through float storage.
    float_summary = torch.empty(
        (response_length, probe_column + (probe_ids is not None)),
        dtype=torch.float32,
        device=device,
    )
    int_summary = torch.empty(
        (response_length, 1 + kept_top_k),
        dtype=torch.long,
        device=device,
    )

    for start in range(0, response_length, chunk_size):
        end = min(response_length, start + chunk_size)
        chunk = target_logits[start:end].float()
        log_normalizer = torch.logsumexp(chunk, dim=-1)
        selected_logits = chunk.gather(1, target_ids[start:end, None]).squeeze(1)
        selected_logp = selected_logits - log_normalizer
        target_ranks = 1 + (chunk > selected_logits[:, None]).sum(dim=-1)
        top_values, top_ids = torch.topk(
            chunk,
            k=kept_top_k,
            dim=-1,
        )
        top_logp = top_values - log_normalizer[:, None]
        probabilities = torch.softmax(chunk, dim=-1)
        entropies = -(probabilities * torch.log_softmax(chunk, dim=-1)).sum(dim=-1)

        float_summary[start:end, 0] = selected_logp
        float_summary[start:end, 1] = entropies
        float_summary[start:end, 2:probe_column] = top_logp
        int_summary[start:end, 0] = target_ranks
        int_summary[start:end, 1:] = top_ids
        if probe_ids is not None:
            probe_logits = chunk.gather(1, probe_ids[start:end, None]).squeeze(1)
            float_summary[start:end, probe_column] = probe_logits - log_normalizer

        # Do not retain vocabulary-sized tensors across chunks.
        del (
            chunk,
            log_normalizer,
            selected_logits,
            selected_logp,
            target_ranks,
            top_values,
            top_ids,
            top_logp,
            probabilities,
            entropies,
        )
        if probe_ids is not None:
            del probe_logits

    float_rows = float_summary.detach().cpu().tolist()
    int_rows = int_summary.detach().cpu().tolist()
    selected_logp_out = [float(row[0]) for row in float_rows]
    entropy_out = [float(row[1]) for row in float_rows]
    top_logp_out = [
        [float(value) for value in row[2:probe_column]] for row in float_rows
    ]
    top_ids_out = [[int(value) for value in row[1:]] for row in int_rows]
    probe_logp_out = (
        [float(row[probe_column]) for row in float_rows]
        if probe_ids is not None
        else None
    )

    return {
        "selected_logp": selected_logp_out,
        "ranks": [int(row[0]) for row in int_rows],
        "entropy": entropy_out,
        "top_ids": top_ids_out,
        "top_logp": top_logp_out,
        "probe_logp": probe_logp_out,
    }
