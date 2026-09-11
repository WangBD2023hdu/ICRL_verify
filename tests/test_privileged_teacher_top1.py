from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from qwen_mm_token_probe import privileged_probe as probe
from qwen_mm_token_probe.hf_qwen import ModelBundle


class _SyntheticTokenizer:
    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(f"<{int(token_id)}>" for token_id in token_ids)


class _DeterministicModel(torch.nn.Module):
    """Return position-indexed logits while recording the exact model input."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits_by_position = logits.detach().clone().float()
        self.calls: list[torch.Tensor] = []

    def _forward_logits(
        self,
        input_ids: torch.Tensor,
        logits_to_keep: int | None = None,
    ) -> SimpleNamespace:
        self.calls.append(input_ids.detach().cpu().clone())
        sequence_length = int(input_ids.shape[-1])
        logits = self.logits_by_position[:sequence_length].to(input_ids.device)
        logits = logits.unsqueeze(0)
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


class _DeterministicKeepModel(_DeterministicModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        logits_to_keep: int | None = None,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return self._forward_logits(input_ids, logits_to_keep)


class _DeterministicFullLogitModel(_DeterministicModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return self._forward_logits(input_ids)


def _bundle(model: torch.nn.Module) -> ModelBundle:
    return ModelBundle(
        model_id="synthetic",
        model=model,
        processor=SimpleNamespace(),
        tokenizer=_SyntheticTokenizer(),
        device=torch.device("cpu"),
    )


def _synthetic_setup(
    supports_logits_to_keep: bool,
) -> tuple[ModelBundle, dict[str, torch.Tensor], list[int], list[int], torch.Tensor]:
    # The four rows at positions 2:6 predict the four response IDs.  Token 5 is
    # the hypothetical teacher top-1, but is outside the student's top-2 on
    # every row and therefore must be recovered through the probe path.
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 2.0, 3.0, 0.0, -1.0, 1.0],
            [4.5, 0.0, 2.0, 3.5, -1.0, 0.5],
            [4.0, 3.0, 0.0, 2.2, -1.0, 1.3],
            [4.0, 3.0, 0.0, 0.0, 2.1, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model_type = (
        _DeterministicKeepModel
        if supports_logits_to_keep
        else _DeterministicFullLogitModel
    )
    prompt_inputs = {
        "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    return (
        _bundle(model_type(logits)),
        prompt_inputs,
        [1, 2, 3, 4],
        [5, 5, 5, 5],
        logits[2:6],
    )


@pytest.mark.parametrize("supports_logits_to_keep", [False, True])
def test_fixed_response_probe_uses_same_forward_and_exact_log_softmax(
    supports_logits_to_keep: bool,
) -> None:
    bundle, prompt_inputs, response_ids, probe_ids, target_logits = _synthetic_setup(
        supports_logits_to_keep
    )
    original_prompt_ids = prompt_inputs["input_ids"].clone()

    baseline = probe._score_fixed_response_ids(
        model_bundle=bundle,
        prompt_inputs=prompt_inputs,
        response_ids=response_ids,
        top_k=2,
        chunk_size=len(response_ids),
    )
    with_probe_small_chunks = probe._score_fixed_response_ids(
        model_bundle=bundle,
        prompt_inputs=prompt_inputs,
        response_ids=response_ids,
        probe_token_ids=probe_ids,
        top_k=2,
        chunk_size=2,
    )
    with_probe_uneven_chunks = probe._score_fixed_response_ids(
        model_bundle=bundle,
        prompt_inputs=prompt_inputs,
        response_ids=response_ids,
        probe_token_ids=probe_ids,
        top_k=2,
        chunk_size=3,
    )

    assert torch.equal(prompt_inputs["input_ids"], original_prompt_ids)
    expected_model_input = [10, 11, 12, *response_ids]
    assert [call.tolist() for call in bundle.model.calls] == [
        [expected_model_input],
        [expected_model_input],
        [expected_model_input],
    ]

    expected_logp = torch.log_softmax(target_logits, dim=-1)
    expected_response_logp = expected_logp.gather(
        1, torch.tensor(response_ids, dtype=torch.long)[:, None]
    ).squeeze(1)
    expected_probe_logp = expected_logp[:, 5]
    for index, row in enumerate(baseline):
        assert row["log_probability"] == pytest.approx(
            float(expected_response_logp[index])
        )
        assert "probe_token_id" not in row
    for index, row in enumerate(with_probe_small_chunks):
        assert row["token_id"] == response_ids[index]
        assert row["probe_token_id"] == probe_ids[index]
        assert row["log_probability"] == pytest.approx(
            float(expected_response_logp[index])
        )
        assert row["probe_log_probability"] == pytest.approx(
            float(expected_probe_logp[index])
        )
        assert row["probe_probability"] == pytest.approx(
            float(expected_probe_logp[index].exp())
        )
        assert probe_ids[index] not in {
            int(candidate["token_id"]) for candidate in row["top_candidates"]
        }

    # Both sides of the chunk boundary carry distinct probe scores, while the
    # response-token scores remain exactly the same as the no-probe baseline.
    assert with_probe_small_chunks[1]["probe_log_probability"] != pytest.approx(
        with_probe_small_chunks[2]["probe_log_probability"]
    )
    for left, right in zip(with_probe_small_chunks, with_probe_uneven_chunks):
        assert left["probability"] == pytest.approx(right["probability"])
        assert left["log_probability"] == pytest.approx(right["log_probability"])
        assert left["probe_probability"] == pytest.approx(right["probe_probability"])
        assert left["probe_log_probability"] == pytest.approx(
            right["probe_log_probability"]
        )
    assert [row["probability"] for row in baseline] == pytest.approx(
        [row["probability"] for row in with_probe_small_chunks]
    )


def test_fixed_response_probe_length_mismatch_is_rejected() -> None:
    bundle, prompt_inputs, response_ids, _, _ = _synthetic_setup(True)

    with pytest.raises(ValueError, match="probe_token_ids"):
        probe._score_fixed_response_ids(
            model_bundle=bundle,
            prompt_inputs=prompt_inputs,
            response_ids=response_ids,
            probe_token_ids=[5, 5, 5],
            top_k=2,
            chunk_size=2,
        )


def _score_record(
    token_id: int,
    raw_token: str,
    probability: float,
    *,
    top_token_id: int,
    top_raw_token: str,
    top_probability: float,
    top_candidates: list[dict[str, Any]] | None = None,
    probe_token_id: int | None = None,
    probe_probability: float | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "token_id": token_id,
        "token": raw_token,
        "raw_token": raw_token,
        "probability": probability,
        "log_probability": math.log(probability),
        "target_rank": 1,
        "entropy": 0.25,
        "top_token_id": top_token_id,
        "top_token": top_raw_token,
        "top_raw_token": top_raw_token,
        "top_probability": top_probability,
        "top_log_probability": math.log(top_probability),
        "top_candidates": top_candidates or [],
    }
    if probe_token_id is not None:
        assert probe_probability is not None
        record.update(
            {
                "probe_token_id": probe_token_id,
                "probe_probability": probe_probability,
                "probe_log_probability": math.log(probe_probability),
            }
        )
    return record


def _combined_row_for_helper(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "token_id": 7,
        "p_original": 0.7,
        "logp_original": math.log(0.7),
        "top_token_id_original": 8,
        "top_p_original": 0.8,
        "top_logp_original": math.log(0.8),
        "top_token_id_teacher": 9,
        "top_candidates_original": [],
    }
    row.update(overrides)
    return row


def test_student_score_for_teacher_top1_prefers_exact_id_sources() -> None:
    helper = probe._student_score_for_teacher_top1

    assert helper(
        _combined_row_for_helper(
            p_original_teacher_top1=0.11,
            logp_original_teacher_top1=math.log(0.11),
            top_token_id_teacher=99,
        )
    ) == pytest.approx((0.11, math.log(0.11)))
    assert helper(
        _combined_row_for_helper(
            top_token_id_teacher=7,
            p_original=0.21,
            logp_original=math.log(0.21),
        )
    ) == pytest.approx((0.21, math.log(0.21)))
    assert helper(
        _combined_row_for_helper(
            top_token_id_teacher=8,
            top_p_original=0.31,
            top_logp_original=math.log(0.31),
        )
    ) == pytest.approx((0.31, math.log(0.31)))


def test_student_score_for_teacher_top1_fallback_requires_exact_token_id() -> None:
    helper = probe._student_score_for_teacher_top1
    candidates = [
        {
            "token_id": 304,
            "raw_token": "X",
            "probability": 0.77,
            "log_probability": math.log(0.77),
        },
        {
            "token_id": 303,
            "raw_token": "Y",
            "probability": 0.23,
            "log_probability": math.log(0.23),
        },
    ]

    exact_match = helper(
        _combined_row_for_helper(
            top_token_id_teacher=303,
            top_candidates_original=candidates,
        )
    )
    assert exact_match == pytest.approx((0.23, math.log(0.23)))

    same_surface_only = helper(
        _combined_row_for_helper(
            top_token_id_teacher=303,
            top_candidates_original=[candidates[0]],
        )
    )
    assert same_surface_only == (None, None)


def test_combine_scores_records_original_probability_for_teacher_top1_id() -> None:
    response_ids = [7]
    original = _score_record(
        7,
        "A",
        0.7,
        top_token_id=8,
        top_raw_token="X",
        top_probability=0.8,
        probe_token_id=303,
        probe_probability=0.23,
    )
    teacher = _score_record(
        7,
        "A",
        0.6,
        top_token_id=303,
        top_raw_token="X",
        top_probability=0.9,
    )

    row = probe._combine_scores(response_ids, [original], [teacher])[0]

    assert row["p_original_teacher_top1"] == pytest.approx(0.23)
    assert row["logp_original_teacher_top1"] == pytest.approx(math.log(0.23))


def test_combine_scores_does_not_match_teacher_top1_by_surface_only() -> None:
    original = _score_record(
        7,
        "A",
        0.7,
        top_token_id=8,
        top_raw_token="X",
        top_probability=0.8,
        top_candidates=[
            {
                "token_id": 304,
                "raw_token": "X",
                "token": "X",
                "probability": 0.77,
                "log_probability": math.log(0.77),
            }
        ],
    )
    teacher = _score_record(
        7,
        "A",
        0.6,
        top_token_id=303,
        top_raw_token="X",
        top_probability=0.9,
    )

    row = probe._combine_scores([7], [original], [teacher])[0]

    assert row.get("p_original_teacher_top1") is None
    assert row.get("logp_original_teacher_top1") is None
