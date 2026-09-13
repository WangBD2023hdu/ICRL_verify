from __future__ import annotations

import pytest
import torch

from qwen_mm_token_probe.score_statistics import collect_token_statistics


def _reference(
    logits: torch.Tensor,
    response_ids: list[int],
    *,
    probe_token_ids: list[int] | None,
    top_k: int,
    chunk_size: int,
) -> dict[str, list[object] | None]:
    selected_logp: list[float] = []
    ranks: list[int] = []
    entropy: list[float] = []
    top_ids: list[list[int]] = []
    top_logp: list[list[float]] = []
    probe_logp: list[float] | None = [] if probe_token_ids is not None else None

    for start in range(0, len(response_ids), chunk_size):
        end = min(len(response_ids), start + chunk_size)
        chunk = logits[start:end].float()
        targets = torch.tensor(response_ids[start:end], dtype=torch.long)
        normalizer = torch.logsumexp(chunk, dim=-1)
        selected = chunk.gather(1, targets[:, None]).squeeze(1)
        chosen_logp = selected - normalizer
        chosen_ranks = 1 + (chunk > selected[:, None]).sum(dim=-1)
        values, ids = torch.topk(chunk, k=min(top_k, chunk.shape[-1]), dim=-1)
        chosen_top_logp = values - normalizer[:, None]
        probabilities = torch.softmax(chunk, dim=-1)
        chosen_entropy = -(
            probabilities * torch.log_softmax(chunk, dim=-1)
        ).sum(dim=-1)

        selected_logp.extend(chosen_logp.tolist())
        ranks.extend(chosen_ranks.tolist())
        entropy.extend(chosen_entropy.tolist())
        top_ids.extend(ids.tolist())
        top_logp.extend(chosen_top_logp.tolist())
        if probe_token_ids is not None:
            probes = torch.tensor(probe_token_ids[start:end], dtype=torch.long)
            probe_values = chunk.gather(1, probes[:, None]).squeeze(1)
            assert probe_logp is not None
            probe_logp.extend((probe_values - normalizer).tolist())

    return {
        "selected_logp": selected_logp,
        "ranks": ranks,
        "entropy": entropy,
        "top_ids": top_ids,
        "top_logp": top_logp,
        "probe_logp": probe_logp,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_probes", [False, True])
def test_collect_matches_reference_across_chunk_sizes(
    dtype: torch.dtype,
    with_probes: bool,
) -> None:
    torch.manual_seed(23)
    logits = torch.randn(7, 19, dtype=torch.float32).to(dtype)
    response_ids = [0, 4, 2, 12, 7, 1, 18]
    # The minimum-logit token is guaranteed to be outside each row's top 3.
    probes = torch.argmin(logits.float(), dim=-1).tolist() if with_probes else None
    reference = _reference(
        logits,
        response_ids,
        probe_token_ids=probes,
        top_k=3,
        chunk_size=len(response_ids),
    )

    for chunk_size in (1, 2, 4, 20):
        actual = collect_token_statistics(
            logits,
            response_ids,
            probe_token_ids=probes,
            top_k=3,
            chunk_size=chunk_size,
        )
        assert actual.keys() == reference.keys()
        assert actual["top_ids"] == reference["top_ids"]
        assert actual["ranks"] == reference["ranks"]
        for key in ("selected_logp", "entropy", "top_logp", "probe_logp"):
            got = actual[key]
            expected = reference[key]
            if got is None or expected is None:
                assert got is expected
            elif key == "top_logp":
                assert len(got) == len(expected)
                for got_row, expected_row in zip(got, expected):
                    assert got_row == pytest.approx(
                        expected_row, rel=1e-6, abs=1e-6
                    )
            else:
                assert got == pytest.approx(expected, rel=1e-6, abs=1e-6)

        assert all(type(value) is int for value in actual["ranks"])
        assert all(
            type(token_id) is int
            for row in actual["top_ids"]
            for token_id in row
        )
        assert all(type(value) is float for value in actual["selected_logp"])
        assert all(type(value) is float for value in actual["entropy"])
        assert all(
            type(value) is float
            for row in actual["top_logp"]
            for value in row
        )


def test_probe_token_can_be_outside_top_k() -> None:
    logits = torch.tensor(
        [[9.0, 8.0, 7.0, -5.0], [1.0, 4.0, 3.0, -8.0]],
        dtype=torch.float32,
    )
    actual = collect_token_statistics(
        logits,
        [0, 1],
        probe_token_ids=[3, 3],
        top_k=2,
        chunk_size=1,
    )
    assert actual["top_ids"] == [[0, 1], [1, 2]]
    assert actual["probe_logp"] == pytest.approx(
        torch.log_softmax(logits, dim=-1)[:, 3].tolist()
    )


def test_rank_counts_strictly_greater_logits_when_there_are_ties() -> None:
    logits = torch.tensor(
        [[5.0, 5.0, 7.0, 1.0], [1.0, 3.0, 3.0, 3.0]],
        dtype=torch.float32,
    )
    actual = collect_token_statistics(
        logits,
        [0, 1],
        top_k=4,
        chunk_size=2,
    )
    assert actual["ranks"] == [2, 1]
    assert actual["top_ids"] == [
        torch.topk(logits[row], k=4).indices.tolist() for row in range(2)
    ]


def test_rejects_invalid_shapes_and_lengths() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        collect_token_statistics(torch.zeros(1, 3), [], top_k=1, chunk_size=1)
    with pytest.raises(ValueError, match="first dimension"):
        collect_token_statistics(torch.zeros(2, 3), [1], top_k=1, chunk_size=1)
    with pytest.raises(ValueError, match="probe_token_ids"):
        collect_token_statistics(
            torch.zeros(1, 3), [1], probe_token_ids=[1, 2], top_k=1, chunk_size=1
        )
    with pytest.raises(ValueError, match="chunk_size"):
        collect_token_statistics(torch.zeros(1, 3), [1], top_k=1, chunk_size=0)
