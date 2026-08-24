from __future__ import annotations

import json
import math
from html.parser import HTMLParser
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


class SampleReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.transcripts: list[str] = []
        self.token_rows: list[tuple[str, list[str]]] = []
        self._transcript_parts: list[str] | None = None
        self._token_row: tuple[str, list[str]] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "pre" and attributes.get("class") == "transcript":
            self._transcript_parts = []
        if tag == "tr":
            row_id = attributes.get("id")
            if row_id is not None and row_id.startswith("token-"):
                self._token_row = (row_id, [])
        if tag == "td" and self._token_row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._transcript_parts is not None:
            self._transcript_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self._transcript_parts is not None:
            self.transcripts.append("".join(self._transcript_parts))
            self._transcript_parts = None
        elif tag == "td" and self._token_row is not None:
            self._token_row[1].append("".join(self._cell_parts or []))
            self._cell_parts = None
        elif tag == "tr" and self._token_row is not None:
            self.token_rows.append(self._token_row)
            self._token_row = None


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


def test_sample_report_keeps_complete_response_token_table_in_order() -> None:
    response_ids = [8, 7, 9]
    rows = probe._combine_scores(
        response_ids,
        [
            _score_record(
                8,
                "B",
                0.2,
                top_token_id=7,
                top_raw_token="A",
                top_probability=0.6,
                target_rank=2,
            ),
            _score_record(
                7,
                "A",
                0.9,
                top_token_id=7,
                top_raw_token="A",
                top_probability=0.9,
                target_rank=1,
            ),
            _score_record(
                9,
                "X",
                0.1,
                top_token_id=8,
                top_raw_token="B",
                top_probability=0.7,
                target_rank=3,
            ),
        ],
        [
            _score_record(
                8,
                "B",
                0.4,
                top_token_id=8,
                top_raw_token="B",
                top_probability=0.4,
                target_rank=1,
            ),
            _score_record(
                7,
                "A",
                0.3,
                top_token_id=9,
                top_raw_token="X",
                top_probability=0.5,
                target_rank=2,
            ),
            _score_record(
                9,
                "X",
                0.8,
                top_token_id=9,
                top_raw_token="X",
                top_probability=0.8,
                target_rank=1,
            ),
        ],
    )
    for row, original_candidates, teacher_candidates in zip(
        rows,
        [
            [
                {"rank": 1, "token_id": 7, "token": "A", "raw_token": "A", "probability": 0.6},
                {"rank": 2, "token_id": 8, "token": "B", "raw_token": "B", "probability": 0.2},
            ],
            [
                {"rank": 1, "token_id": 7, "token": "A", "raw_token": "A", "probability": 0.9},
                {"rank": 2, "token_id": 8, "token": "B", "raw_token": "B", "probability": 0.05},
            ],
            [
                {"rank": 1, "token_id": 8, "token": "B", "raw_token": "B", "probability": 0.7},
                {"rank": 2, "token_id": 9, "token": "X", "raw_token": "X", "probability": 0.1},
            ],
        ],
        [
            [
                {"rank": 1, "token_id": 8, "token": "B", "raw_token": "B", "probability": 0.4},
                {"rank": 2, "token_id": 7, "token": "A", "raw_token": "A", "probability": 0.2},
            ],
            [
                {"rank": 1, "token_id": 9, "token": "X", "raw_token": "X", "probability": 0.5},
                {"rank": 2, "token_id": 7, "token": "A", "raw_token": "A", "probability": 0.3},
            ],
            [
                {"rank": 1, "token_id": 9, "token": "X", "raw_token": "X", "probability": 0.8},
                {"rank": 2, "token_id": 7, "token": "A", "raw_token": "A", "probability": 0.1},
            ],
        ],
    ):
        row["top_candidates_original"] = original_candidates
        row["top_candidates_teacher"] = teacher_candidates
    rows[1].update(
        {
            "token_label": "mutation_opposite_variant",
            "is_hallucination": True,
            "mutation_ids": "m001",
            "raw_source_start": 0,
            "raw_source_end": 1,
        }
    )
    mutation = {
        "mutation_id": "m001",
        "ocr_ans": "B",
        "origin_ans": "A",
        "bbox": [1, 2, 3, 4],
        "predicted": "B",
        "relation": "opposite_variant",
        "response_token_indices": [1],
    }
    ground_truth = "A\nGround truth tail"
    response_text = "BAX"
    result = {
        "pair_id": "pair-1",
        "sample": {
            "image_copy": "/tmp/input.png",
            "changes": [
                {
                    "ocr_ans": "B",
                    "origin_ans": "A",
                    "markdown_span": [0, 1],
                }
            ],
        },
        "response": {"text": response_text, "token_ids": response_ids},
        "ground_truth": ground_truth,
        "summary": {
            "token_count": len(rows),
            "mean_p_original": 0.4,
            "mean_p_teacher": 0.5,
            "mean_delta_logp_teacher_minus_original": math.fsum(
                float(row["delta_logp_teacher_minus_original"]) for row in rows
            )
            / len(rows),
        },
        "mutation_observations": [mutation],
        "tokens": rows,
    }

    report = probe._render_sample_html(result)
    parser = SampleReportParser()
    parser.feed(report)

    assert parser.transcripts == [ground_truth, response_text]
    assert "Ground Truth" in report
    assert "模型 Response" in report
    assert "image + prompt" in report
    assert "GT Teacher-Forcing 条件" in report
    assert "p(response token)" in report
    assert "p(same response token)" in report
    assert "Top-1" in report
    assert "Top-2" in report
    assert '<div class="stats">' not in report
    assert "data-token-filter=" not in report
    assert '<article class="mutation-focus">' not in report

    assert [row_id for row_id, _ in parser.token_rows] == [
        "token-0",
        "token-1",
        "token-2",
    ]
    assert len(parser.token_rows) == len(response_ids)
    assert report.count("class='token-row") == len(response_ids)
    expected = [
        (0, 8, "B", 0.2, 0.4),
        (1, 7, "A", 0.9, 0.3),
        (2, 9, "X", 0.1, 0.8),
    ]
    for (row_id, cells), (index, token_id, token_text, p_original, p_teacher) in zip(
        parser.token_rows, expected
    ):
        assert row_id == f"token-{index}"
        assert len(cells) == 12
        assert cells[0] == str(index)
        assert token_text in cells[1]
        assert f"ID {token_id}" in cells[1]
        assert f"{p_original:.6f}" in cells[2]
        assert f"{p_teacher:.6f}" in cells[6]
        assert f"{p_teacher - p_original:+.6f}" in cells[10]
        assert (
            f"{math.log(p_teacher) - math.log(p_original):+.4f}"
            in cells[11]
        )

    assert "0.200000" in parser.token_rows[0][1][2]
    assert "p 0.600000" in parser.token_rows[0][1][4]
    assert "0.400000" in parser.token_rows[0][1][6]
    assert "p 0.400000" in parser.token_rows[0][1][8]


@pytest.mark.parametrize(
    ("tokens", "response_ids"),
    [
        ([{"index": 1, "token_id": 7}], [7]),
        ([{"index": 0, "token_id": 8}], [7]),
    ],
)
def test_response_rows_reject_order_or_id_mismatch(
    tokens: list[dict[str, int]],
    response_ids: list[int],
) -> None:
    with pytest.raises(
        RuntimeError,
        match="report tokens are not in generated response order",
    ):
        probe._response_rows_in_generation_order(
            {"tokens": tokens, "response": {"token_ids": response_ids}}
        )


def test_aggregate_report_has_no_statistics_and_starts_with_first_sample() -> None:
    report = probe._render_aggregate_html(
        [
            {
                "pair_id": "pair-1",
                "report": "samples/001_pair-1/report.html",
                "token_count": 3,
            },
            {
                "pair_id": "pair-2",
                "report": "samples/002_pair-2/report.html",
                "token_count": 5,
            },
        ]
    )

    iframe_start = report.index("<iframe ")
    iframe_end = report.index(">", iframe_start)
    iframe_tag = report[iframe_start:iframe_end]
    assert 'src="samples/001_pair-1/report.html"' in iframe_tag
    assert "samples/002_pair-2/report.html" in report
    assert '<div class="stats">' not in report
    assert "<canvas" not in report
    assert "summary.json" not in report
    assert "<table" not in report


def test_cli_contains_no_api_transport_arguments() -> None:
    parser = probe._build_parser()
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }

    assert "--model-id" in option_strings
    assert "--base-url" not in option_strings
    assert "--api-key" not in option_strings
    assert "--max-new-tokens" in option_strings
