from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from PIL import Image

from qwen_mm_token_probe import privileged_probe as probe
from qwen_mm_token_probe.hf_qwen import ModelBundle


class FakeTokenizer:
    all_special_ids: ClassVar[list[int]] = []

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] | None = None

    def decode(self, token_ids: list[int], **_: object) -> str:
        pieces = {7: "A", 8: "B", 9: "X"}
        return "".join(
            pieces.get(int(token_id), f"<{token_id}>") for token_id in token_ids
        )

    def apply_chat_template(
        self, messages: list[dict[str, object]], **_: object
    ) -> dict:
        self.messages = messages
        return {
            "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


class RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        logits_to_keep: int | None = None,
        **_: object,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.calls.append(input_ids.detach().cpu().clone())
        sequence_length = int(input_ids.shape[-1])
        logits = torch.full((1, sequence_length, 32), -12.0)
        for position in range(sequence_length - 1):
            next_id = int(input_ids[0, position + 1])
            logits[0, position, next_id] = 12.0
        logits[0, -1, 0] = 12.0
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


class TrackerStub:
    def set_current(self, **_: object) -> None:
        return None


def _bundle(model: torch.nn.Module | None = None) -> ModelBundle:
    tokenizer = FakeTokenizer()
    return ModelBundle(
        model_id="fake-qwen",
        model=model or RecordingModel(),
        processor=SimpleNamespace(),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )


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
        "target_rank": target_rank,
        "entropy": 0.25,
        "top_token_id": top_token_id,
        "top_token": top_raw_token,
        "top_raw_token": top_raw_token,
        "top_probability": top_probability,
        "top_log_probability": math.log(top_probability),
        "top_candidates": [],
    }


def test_load_release_samples_resolves_release_paths(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "page.png").write_bytes(b"image")
    (data / "page.md").write_text("GT", encoding="utf-8")
    (tmp_path / "pairs.jsonl").write_text(
        json.dumps(
            {
                "pair_id": "p1",
                "edited_image": "data/page.png",
                "edited_markdown": "data/page.md",
                "changes": [{"ocr_ans": "1", "origin_ans": "7"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = probe.load_release_samples(tmp_path)

    assert len(samples) == 1
    assert samples[0].pair_id == "p1"
    assert samples[0].image_path == (data / "page.png").resolve()


def test_text_teacher_prompt_is_text_only_and_verbatim() -> None:
    bundle = _bundle()

    inputs = probe._prepare_text_prompt_inputs(
        model_bundle=bundle,
        prompt="完整 GT\n\n请转写上述文本",
    )

    assert bundle.tokenizer.messages == [
        {"role": "user", "content": "完整 GT\n\n请转写上述文本"}
    ]
    assert inputs["input_ids"].tolist() == [[4, 5, 6]]
    assert "pixel_values" not in inputs


def test_exact_response_ids_are_directly_appended_before_forward() -> None:
    model = RecordingModel()
    bundle = _bundle(model)
    prompt_inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }

    scores = probe._score_fixed_response_ids(
        model_bundle=bundle,
        prompt_inputs=prompt_inputs,
        response_ids=[7, 8],
        top_k=3,
        chunk_size=1,
    )

    assert len(model.calls) == 1
    assert model.calls[0].tolist() == [[1, 2, 3, 7, 8]]
    assert [row["token_id"] for row in scores] == [7, 8]
    assert [row["top_token_id"] for row in scores] == [7, 8]
    assert [row["target_rank"] for row in scores] == [1, 1]
    assert all(float(row["probability"]) > 0.999 for row in scores)


def test_sample_generates_once_then_forwards_same_ids_twice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    gt_path = tmp_path / "page.md"
    gt_path.write_text("AB", encoding="utf-8")
    sample = probe.PrivilegedProbeSample(
        ordinal=1,
        pair_id="p1",
        image_path=image_path,
        ground_truth_path=gt_path,
        changes=(),
    )
    model = RecordingModel()
    bundle = _bundle(model)
    generation_calls: list[list[int]] = []

    def fake_prepare_prompt_inputs(**_: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def fake_teacher_inputs(**_: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def fake_generate(**kwargs: object) -> tuple[list[int], str]:
        generation_calls.append(kwargs["prompt_inputs"]["input_ids"].tolist()[0])
        return [7, 8], "ignored"

    monkeypatch.setattr(probe, "prepare_prompt_inputs", fake_prepare_prompt_inputs)
    monkeypatch.setattr(probe, "_prepare_text_prompt_inputs", fake_teacher_inputs)
    monkeypatch.setattr(probe, "generate_from_prompt", fake_generate)

    result = probe._run_sample(
        sample=sample,
        sample_dir=tmp_path / "sample",
        fingerprint="fingerprint",
        model_bundle=bundle,
        prompt="OCR",
        privileged_instruction="请转写上述文本",
        max_new_tokens=16,
        top_k=3,
        forward_chunk_size=2,
        min_pixels=2048,
        max_pixels=16777216,
        image_patch_size=16,
        seed=7,
        tracker=TrackerStub(),
    )

    assert generation_calls == [[1, 2, 3]]
    assert len(model.calls) == 2
    assert model.calls[0].tolist() == [[1, 2, 3, 7, 8]]
    assert model.calls[1].tolist() == [[4, 5, 6, 7, 8]]
    assert result["response"]["token_ids"] == [7, 8]
    assert result["protocol"]["generation_count"] == 1
    assert result["protocol"]["teacher_forced_forward_count"] == 2
    assert result["protocol"]["response_ids_directly_concatenated"] is True
    assert result["protocol"]["response_text_retokenized"] is False
    assert result["ground_truth"] == "AB"
    assert (tmp_path / "sample" / "privileged_prompt.txt").read_text(
        encoding="utf-8"
    ) == "AB\n\n请转写上述文本"


def test_combine_scores_reports_teacher_alternative() -> None:
    rows = probe._combine_scores(
        [7],
        [
            _score_record(
                7,
                "A",
                0.9,
                top_token_id=7,
                top_raw_token="A",
                top_probability=0.9,
                target_rank=1,
            )
        ],
        [
            _score_record(
                7,
                "A",
                0.01,
                top_token_id=9,
                top_raw_token="X",
                top_probability=0.95,
                target_rank=3,
            )
        ],
    )

    assert rows[0]["teacher_preference"] == "different_surface_top1"
    assert rows[0]["top1_transition"] == "teacher_rejects_response"
    assert rows[0]["delta_logp_teacher_minus_original"] < 0


def test_sample_report_exposes_gt_response_mutation_and_top2() -> None:
    rows = probe._combine_scores(
        [7],
        [
            _score_record(
                7,
                "A",
                0.9,
                top_token_id=7,
                top_raw_token="A",
                top_probability=0.9,
                target_rank=1,
            )
        ],
        [
            _score_record(
                7,
                "A",
                0.1,
                top_token_id=9,
                top_raw_token="X",
                top_probability=0.8,
                target_rank=2,
            )
        ],
    )
    rows[0].update(
        {
            "token_label": "mutation_opposite_variant",
            "is_hallucination": True,
            "mutation_ids": "m001",
            "raw_source_start": 0,
            "raw_source_end": 1,
            "top_candidates_original": [
                {
                    "rank": 1,
                    "token_id": 7,
                    "token": "A",
                    "raw_token": "A",
                    "probability": 0.9,
                },
                {
                    "rank": 2,
                    "token_id": 9,
                    "token": "X",
                    "raw_token": "X",
                    "probability": 0.08,
                },
            ],
            "top_candidates_teacher": [
                {
                    "rank": 1,
                    "token_id": 9,
                    "token": "X",
                    "raw_token": "X",
                    "probability": 0.8,
                },
                {
                    "rank": 2,
                    "token_id": 7,
                    "token": "A",
                    "raw_token": "A",
                    "probability": 0.1,
                },
            ],
        }
    )
    mutation = {
        "mutation_id": "m001",
        "ocr_ans": "A",
        "origin_ans": "X",
        "bbox": [1, 2, 3, 4],
        "predicted": "A",
        "relation": "opposite_variant",
        "response_token_indices": [0],
    }
    result = {
        "pair_id": "pair-1",
        "sample": {
            "image_copy": "/tmp/input.png",
            "changes": [
                {
                    "ocr_ans": "A",
                    "origin_ans": "X",
                    "markdown_span": [0, 1],
                }
            ],
        },
        "response": {"text": "A"},
        "ground_truth": "A",
        "summary": {
            "token_count": 1,
            "mean_p_original": 0.9,
            "mean_p_teacher": 0.1,
            "mean_delta_logp_teacher_minus_original": math.log(0.1) - math.log(0.9),
        },
        "mutation_observations": [mutation],
        "tokens": rows,
    }

    report = probe._render_sample_html(result)

    assert "Ground Truth" in report
    assert "模型 Response" in report
    assert "变异词重点" in report
    assert "图片中的变异词" in report
    assert "Original Top-2" in report
    assert "Teacher Top-2" in report
    assert "ID 9" in report
    assert "p 0.080000" in report
    assert "data-mutation='1'" in report
    assert "class='mutation-mark'" in report


def test_cli_contains_no_api_transport_arguments() -> None:
    parser = probe._build_parser()
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }

    assert "--model-id" in option_strings
    assert "--base-url" not in option_strings
    assert "--api-key" not in option_strings
    assert "--max-new-tokens" in option_strings
