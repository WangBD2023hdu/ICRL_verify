from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qwen_mm_token_probe import api_privileged_probe as probe
from qwen_mm_token_probe import vllm_api


def _score_record(
    token_id: int,
    raw_token: str,
    probability: float,
    *,
    top_token_id: int,
    top_raw_token: str,
    top_probability: float,
    target_rank: int,
) -> dict[str, object]:
    return {
        "token_id": token_id,
        "token": raw_token,
        "raw_token": raw_token,
        "probability": probability,
        "log_probability": math.log(probability),
        "top_token_id": top_token_id,
        "top_token": top_raw_token,
        "top_raw_token": top_raw_token,
        "top_probability": top_probability,
        "top_log_probability": math.log(top_probability),
        "target_rank": target_rank,
    }


def _generation_response() -> SimpleNamespace:
    items = []
    for raw_token in ("A", "B"):
        item = SimpleNamespace(
            bytes=list(raw_token.encode("utf-8")),
            token=raw_token,
            logprob=math.log(0.5),
            top_logprobs=[],
        )
        item.top_logprobs = [item]
        items.append(item)
    choice = SimpleNamespace(
        message=SimpleNamespace(content="AB"),
        logprobs=SimpleNamespace(content=items),
        model_extra={"token_ids": [10, 11]},
        finish_reason="stop",
    )
    return SimpleNamespace(choices=[choice])


def _prompt_score_response(
    first: dict[str, object],
    second: dict[str, object],
) -> SimpleNamespace:
    entries = [
        {"1": {"logprob": -0.1, "rank": 1, "decoded_token": "P"}},
        {"2": {"logprob": -0.1, "rank": 1, "decoded_token": "Q"}},
        {str(first["token_id"]): first, **dict(first["top_candidate"])},
        {str(second["token_id"]): second, **dict(second["top_candidate"])},
    ]
    entries[2].pop("top_candidate", None)
    entries[3].pop("top_candidate", None)
    return SimpleNamespace(
        model_extra={
            "prompt_token_ids": [1, 2, int(first["token_id"]), int(second["token_id"])],
            "prompt_logprobs": entries,
        }
    )


def _prompt_score_data(
    token_id: int,
    raw_token: str,
    probability: float,
    *,
    top_token_id: int,
    top_raw_token: str,
    top_probability: float,
    target_rank: int,
) -> dict[str, object]:
    return {
        "token_id": token_id,
        "decoded_token": raw_token,
        "logprob": math.log(probability),
        "rank": target_rank,
        "top_candidate": {
            str(top_token_id): {
                "logprob": math.log(top_probability),
                "rank": 1,
                "decoded_token": top_raw_token,
            }
        },
    }


def test_load_release_samples_resolves_pairs_jsonl_relative_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "release"
    image_path = dataset_root / "assets" / "edited.png"
    ground_truth_path = dataset_root / "ground_truth" / "edited.md"
    image_path.parent.mkdir(parents=True)
    ground_truth_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image")
    ground_truth_path.write_text("edited text", encoding="utf-8")
    (dataset_root / "pairs.jsonl").write_text(
        "\n"
        + json.dumps(
            {
                "pair_id": "pair-1",
                "edited_image": "assets/edited.png",
                "edited_markdown": "ground_truth/edited.md",
                "changes": [{"ocr_ans": "x", "origin_ans": "a"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = probe.load_release_samples(dataset_root)

    assert len(samples) == 1
    assert samples[0].pair_id == "pair-1"
    assert samples[0].ordinal == 1
    assert samples[0].image_path == image_path.resolve()
    assert samples[0].ground_truth_path == ground_truth_path.resolve()
    assert samples[0].changes == ({"ocr_ans": "x", "origin_ans": "a"},)


def test_combine_scores_merges_deltas_and_top1_flags() -> None:
    response_ids = [10, 11]
    original_scores = [
        _score_record(
            10,
            "A",
            0.2,
            top_token_id=12,
            top_raw_token="X",
            top_probability=0.6,
            target_rank=2,
        ),
        _score_record(
            11,
            "B",
            0.5,
            top_token_id=11,
            top_raw_token="B",
            top_probability=0.5,
            target_rank=1,
        ),
    ]
    teacher_scores = [
        _score_record(
            10,
            "A",
            0.8,
            top_token_id=10,
            top_raw_token="A",
            top_probability=0.8,
            target_rank=1,
        ),
        _score_record(
            11,
            "B",
            0.25,
            top_token_id=11,
            top_raw_token="B",
            top_probability=0.25,
            target_rank=1,
        ),
    ]

    rows = probe._combine_scores(response_ids, original_scores, teacher_scores)

    assert rows[0]["delta_p_teacher_minus_original"] == pytest.approx(0.6)
    assert rows[0]["delta_logp_teacher_minus_original"] == pytest.approx(math.log(4))
    assert rows[0]["top1_changed"] is True
    assert rows[0]["target_is_top_original"] is False
    assert rows[0]["target_is_top_teacher"] is True
    assert rows[0]["teacher_preference"] == "response_token_top1"
    assert rows[0]["top1_transition"] == "teacher_recovers_response"
    assert rows[1]["delta_p_teacher_minus_original"] == pytest.approx(-0.25)
    assert rows[1]["top1_changed"] is False
    assert rows[1]["top1_transition"] == "response_top1_both"


def test_gt_mutation_alignment_covers_expected_and_opposite_variants() -> None:
    rows = [
        {
            "index": 0,
            "token_id": 10,
            "token": "A",
            "raw_token": "A",
            "p_original": 0.5,
            "p_teacher": 0.5,
            "logp_original": 0.0,
            "logp_teacher": 0.0,
            "delta_logp_teacher_minus_original": 0.0,
        },
        {
            "index": 1,
            "token_id": 11,
            "token": "x",
            "raw_token": "x",
            "p_original": 0.5,
            "p_teacher": 0.5,
            "logp_original": 0.0,
            "logp_teacher": 0.0,
            "delta_logp_teacher_minus_original": 0.0,
        },
        {
            "index": 2,
            "token_id": 12,
            "token": "b",
            "raw_token": "b",
            "p_original": 0.5,
            "p_teacher": 0.5,
            "logp_original": 0.0,
            "logp_teacher": 0.0,
            "delta_logp_teacher_minus_original": 0.0,
        },
    ]
    changes = [
        {"ocr_ans": "x", "origin_ans": "a", "markdown_span": [1, 2]},
        {"ocr_ans": "y", "origin_ans": "b", "markdown_span": [2, 3]},
    ]

    observations = probe._attach_gt_and_mutation_alignment(
        rows=rows,
        response_text="Axb",
        ground_truth="Axy",
        changes=changes,
    )
    mutation_rows = probe._build_mutation_rows(rows, observations, changes)

    assert [observation.relation for observation in observations] == [
        "expected",
        "opposite_variant",
    ]
    assert [observation.predicted for observation in observations] == ["x", "b"]
    assert [row["mutation_relation"] for row in rows[1:]] == [
        "expected",
        "opposite_variant",
    ]
    assert [row["relation"] for row in mutation_rows] == [
        "expected",
        "opposite_variant",
    ]
    assert [row["ocr_ans"] for row in mutation_rows] == ["x", "y"]
    assert [row["origin_ans"] for row in mutation_rows] == ["a", "b"]


def test_run_mock_generates_once_and_reuses_response_ids(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "release"
    data_dir = dataset_root / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "edited.png").write_bytes(b"fake image")
    (data_dir / "edited.md").write_text("AB", encoding="utf-8")
    (dataset_root / "pairs.jsonl").write_text(
        json.dumps(
            {
                "pair_id": "pair-1",
                "edited_image": "data/edited.png",
                "edited_markdown": "data/edited.md",
                "changes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    original_score = [
        _prompt_score_data(
            10,
            "A",
            0.2,
            top_token_id=12,
            top_raw_token="X",
            top_probability=0.6,
            target_rank=2,
        ),
        _prompt_score_data(
            11,
            "B",
            0.5,
            top_token_id=11,
            top_raw_token="B",
            top_probability=0.5,
            target_rank=1,
        ),
    ]
    teacher_score = [
        _prompt_score_data(
            10,
            "A",
            0.8,
            top_token_id=10,
            top_raw_token="A",
            top_probability=0.8,
            target_rank=1,
        ),
        _prompt_score_data(
            11,
            "B",
            0.25,
            top_token_id=13,
            top_raw_token="Y",
            top_probability=0.5,
            target_rank=2,
        ),
    ]
    api_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(
                    side_effect=[
                        _generation_response(),
                        _prompt_score_response(*original_score),
                        _prompt_score_response(*teacher_score),
                    ]
                )
            )
        )
    )

    class FakeClients:
        def __init__(self, **_: object) -> None:
            self.client = api_client

        def get(self) -> object:
            return self.client

        def close(self) -> None:
            return None

    generate_spy = Mock(wraps=probe.generate_response)
    score_spy = Mock(wraps=probe.canonical_prompt_scores)
    monkeypatch.setattr(probe, "ThreadLocalClients", FakeClients)
    monkeypatch.setattr(probe, "generate_response", generate_spy)
    monkeypatch.setattr(probe, "canonical_prompt_scores", score_spy)

    summary = probe.run_api_privileged_probe(
        base_url="http://mock.invalid/v1",
        model="mock-model",
        dataset_root=dataset_root,
        output_dir=tmp_path / "probe-output",
        api_key="test-key",
        prompt="OCR",
        privileged_instruction="GROUND TRUTH",
        max_tokens=8,
        top_logprobs=2,
        max_retries=0,
        retry_base_seconds=0.01,
        heartbeat_seconds=3600,
    )

    assert summary.completed_items == 1
    assert generate_spy.call_count == 1
    assert score_spy.call_count == 2
    score_calls = score_spy.call_args_list
    assert score_calls[0].kwargs["image_path"] is not None
    assert score_calls[1].kwargs["image_path"] is None
    assert score_calls[0].kwargs["expected_token_ids"] == [10, 11]
    assert score_calls[1].kwargs["expected_token_ids"] == [10, 11]
    assert score_calls[0].kwargs["expected_token_ids"] == score_calls[1].kwargs[
        "expected_token_ids"
    ]

    create = api_client.chat.completions.create
    assert create.call_count == 3
    original_messages = create.call_args_list[1].kwargs["messages"]
    privileged_messages = create.call_args_list[2].kwargs["messages"]
    assert original_messages[0]["content"][0]["type"] == "image_url"
    assert privileged_messages[0] == {
        "role": "user",
        "content": "AB\n\nGROUND TRUTH",
    }
    assert privileged_messages[1] == {"role": "assistant", "content": "AB"}


def test_rebuild_report_writes_html_and_csv_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe-output"
    sample_dir = output_dir / "samples" / "001_pair-1"
    sample_dir.mkdir(parents=True)
    rows = probe._combine_scores(
        [10],
        [
            _score_record(
                10,
                "A",
                0.2,
                top_token_id=12,
                top_raw_token="X",
                top_probability=0.6,
                target_rank=2,
            )
        ],
        [
            _score_record(
                10,
                "A",
                0.8,
                top_token_id=10,
                top_raw_token="A",
                top_probability=0.8,
                target_rank=1,
            )
        ],
    )
    mutations = [
        {
            "mutation_id": "m001",
            "ocr_ans": "x",
            "origin_ans": "a",
            "predicted": "x",
            "relation": "expected",
        }
    ]
    result = {
        "pair_id": "pair-1",
        "response": {"finish_reason": "stop"},
        "summary": probe._summarize_rows(rows, mutations),
        "tokens": rows,
        "mutation_observations": mutations,
    }
    (sample_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    summary = probe.rebuild_api_privileged_report(output_dir)

    assert summary["completed_samples"] == 1
    assert summary["total_tokens"] == 1
    assert summary["total_mutations"] == 1
    for filename in (
        "report.html",
        "summary.json",
        "sample_summary.csv",
        "token_probabilities.csv",
        "mutation_probabilities.csv",
    ):
        assert (output_dir / filename).is_file()
    assert "pair-1" in (output_dir / "report.html").read_text(encoding="utf-8")
    with (output_dir / "token_probabilities.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        token_rows = list(csv.DictReader(handle))
    assert token_rows[0]["pair_id"] == "pair-1"
    assert "delta_logp_teacher_minus_original" in token_rows[0]
    with (output_dir / "mutation_probabilities.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        mutation_rows = list(csv.DictReader(handle))
    assert mutation_rows[0]["relation"] == "expected"


def test_canonical_prompt_scores_uses_text_only_messages_and_strict_response_ids() -> None:
    response = SimpleNamespace(
        model_extra={
            "prompt_token_ids": [1, 2, 10, 11],
            "prompt_logprobs": [
                {"1": {"logprob": -0.1, "rank": 1, "decoded_token": "P"}},
                {"2": {"logprob": -0.1, "rank": 1, "decoded_token": "Q"}},
                {
                    "10": {"logprob": math.log(0.7), "rank": 1, "decoded_token": "A"},
                    "12": {"logprob": math.log(0.2), "rank": 2, "decoded_token": "X"},
                },
                {"11": {"logprob": math.log(0.6), "rank": 1, "decoded_token": "B"}},
            ],
        }
    )
    create = Mock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    scores = vllm_api.canonical_prompt_scores(
        client=client,
        model="mock-model",
        image_path=None,
        prompt="OCR prompt",
        response_text="AB",
        prompt_logprobs=2,
        seed=7,
        max_retries=0,
        retry_base_seconds=0.01,
        request_limiter=vllm_api.RequestLimiter(0),
        request_label="text-only-test",
        expected_token_ids=[10, 11],
    )

    assert [score["token_id"] for score in scores] == [10, 11]
    messages = create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "user", "content": "OCR prompt"}
    assert "image_url" not in messages[0]
    assert messages[1] == {"role": "assistant", "content": "AB"}
