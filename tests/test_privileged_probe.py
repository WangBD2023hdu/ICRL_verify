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

_PRIVILEGED_PROMPT_INSTRUCTION = (
    "请逐字逐符号转写下面边界标记之间的文档。转写不是翻译；不要改变任何字符。"
    "边界标记本身不要输出。"
)


def _expected_privileged_prompt(ground_truth: str) -> str:
    document_end = "" if ground_truth.endswith("\n") else "\n"
    return (
        f"{_PRIVILEGED_PROMPT_INSTRUCTION}\n\n"
        "<<<DOCUMENT_START>>>\n"
        f"{ground_truth}{document_end}"
        "<<<DOCUMENT_END>>>"
    )


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
        self.token_row_attributes: dict[str, dict[str, str | None]] = {}
        self.transcript_marks: list[
            tuple[int, dict[str, str | None], str, int, int]
        ] = []
        self._transcript_parts: list[str] | None = None
        self._transcript_index: int | None = None
        self._token_row: tuple[str, list[str]] | None = None
        self._token_row_attributes: dict[str, str | None] | None = None
        self._cell_parts: list[str] | None = None
        self._mark_attributes: dict[str, str | None] | None = None
        self._mark_parts: list[str] | None = None
        self._mark_transcript_index: int | None = None
        self._mark_start: int | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "pre" and attributes.get("class") == "transcript":
            self._transcript_parts = []
            self._transcript_index = len(self.transcripts)
        if tag == "tr":
            row_id = attributes.get("id")
            if row_id is not None and row_id.startswith("token-"):
                self._token_row = (row_id, [])
                self._token_row_attributes = attributes
        if tag == "td" and self._token_row is not None:
            self._cell_parts = []
        if tag == "mark":
            self._mark_attributes = attributes
            self._mark_parts = []
            self._mark_transcript_index = self._transcript_index
            self._mark_start = (
                len("".join(self._transcript_parts))
                if self._transcript_parts is not None
                else None
            )

    def handle_data(self, data: str) -> None:
        if self._transcript_parts is not None:
            self._transcript_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        if self._mark_parts is not None:
            self._mark_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self._transcript_parts is not None:
            self.transcripts.append("".join(self._transcript_parts))
            self._transcript_parts = None
            self._transcript_index = None
        elif tag == "mark" and self._mark_parts is not None:
            text = "".join(self._mark_parts)
            start = self._mark_start if self._mark_start is not None else -1
            self.transcript_marks.append(
                (
                    self._mark_transcript_index
                    if self._mark_transcript_index is not None
                    else -1,
                    self._mark_attributes or {},
                    text,
                    start,
                    start + len(text) if start >= 0 else -1,
                )
            )
            self._mark_attributes = None
            self._mark_parts = None
            self._mark_transcript_index = None
            self._mark_start = None
        elif tag == "td" and self._token_row is not None:
            self._token_row[1].append("".join(self._cell_parts or []))
            self._cell_parts = None
        elif tag == "tr" and self._token_row is not None:
            self.token_row_attributes[self._token_row[0]] = (
                self._token_row_attributes or {}
            )
            self.token_rows.append(self._token_row)
            self._token_row = None
            self._token_row_attributes = None


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
    ground_truth = "\n# 标题\n\n- 保留 Markdown 的尾部空格  \n"
    prompt = _expected_privileged_prompt(ground_truth)

    inputs = probe._prepare_text_prompt_inputs(
        model_bundle=bundle,
        prompt=prompt,
    )

    assert bundle.tokenizer.messages == [
        {"role": "user", "content": prompt}
    ]
    assert inputs["input_ids"].tolist() == [[4, 5, 6]]
    assert "pixel_values" not in inputs


def test_privileged_prompt_builder_preserves_gt_without_trailing_newline() -> None:
    ground_truth = "# 标题\n\nAB  "

    prompt = probe._build_privileged_prompt(
        ground_truth=ground_truth,
        instruction=_PRIVILEGED_PROMPT_INSTRUCTION,
    )

    assert prompt == _expected_privileged_prompt(ground_truth)
    assert prompt.endswith(f"{ground_truth}\n<<<DOCUMENT_END>>>")
    assert ground_truth in prompt
    assert prompt.count(ground_truth) == 1


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
    ground_truth = "\n# 标题\n\nAB  \n"
    gt_path.write_text(ground_truth, encoding="utf-8")
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

    def fake_generate(**kwargs: object) -> tuple[list[int], str]:
        generation_calls.append(kwargs["prompt_inputs"]["input_ids"].tolist()[0])
        return [7, 8], "ignored"

    monkeypatch.setattr(probe, "prepare_prompt_inputs", fake_prepare_prompt_inputs)
    monkeypatch.setattr(probe, "generate_from_prompt", fake_generate)

    result = probe._run_sample(
        sample=sample,
        sample_dir=tmp_path / "sample",
        fingerprint="fingerprint",
        model_bundle=bundle,
        prompt="OCR",
        privileged_instruction=_PRIVILEGED_PROMPT_INSTRUCTION,
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
    assert result["ground_truth"] == ground_truth
    expected_prompt = _expected_privileged_prompt(ground_truth)
    messages = bundle.tokenizer.messages
    assert messages == [{"role": "user", "content": expected_prompt}]
    assert messages is not None
    teacher_prompt = messages[0]["content"]
    assert isinstance(teacher_prompt, str)
    written_prompt = (tmp_path / "sample" / "privileged_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert written_prompt == expected_prompt
    assert written_prompt == teacher_prompt
    assert (tmp_path / "sample" / "ground_truth.md").read_text(
        encoding="utf-8"
    ) == ground_truth
    assert "<<<DOCUMENT_START>>>" in teacher_prompt
    assert "<<<DOCUMENT_END>>>" in teacher_prompt
    assert result["response"]["token_ids"] == [7, 8]
    assert result["response"]["text"] == "AB"
    assert "<<<DOCUMENT_START>>>" not in result["response"]["text"]
    assert "<<<DOCUMENT_END>>>" not in result["response"]["text"]


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


def test_sample_report_visualizes_mutations_and_keeps_complete_token_order() -> None:
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
    ground_truth = "prefix A suffix"
    response_text = "BAX"
    result = {
        "pair_id": "pair-1",
        "sample": {
            "image_copy": "/tmp/input.png",
            "changes": [
                {
                    "ocr_ans": "B",
                    "origin_ans": "A",
                    "markdown_span": [7, 8],
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

    assert [
        (transcript, attrs["class"], attrs["title"], text, start, end)
        for transcript, attrs, text, start, end in parser.transcript_marks
    ] == [
        (0, "mutation-mark", "m001: A -> B", "A", 7, 8),
        (1, "mutation-mark", "#1 · ID 7 · m001", "A", 1, 2),
    ]

    mutation_section_start = report.index('<section id="mutation-details">')
    token_section_start = report.index('<section id="token-details">')
    mutation_section = report[mutation_section_start:token_section_start]
    assert "变异词对照" in mutation_section
    assert "m001" in mutation_section
    assert "原词" in mutation_section
    assert "图片 / GT 变异词" in mutation_section
    assert "模型读回" in mutation_section
    assert "<code>A</code>" in mutation_section
    assert "<code>B</code>" in mutation_section
    assert "关联 Response token" in mutation_section
    assert "#1 · ID 7" in mutation_section
    assert "0.900000" in mutation_section
    assert "0.300000" in mutation_section
    assert "-0.600000" in mutation_section
    assert "-1.0986" in mutation_section

    assert [row_id for row_id, _ in parser.token_rows] == [
        "token-0",
        "token-1",
        "token-2",
    ]
    assert len(parser.token_rows) == len(response_ids)
    assert report.count("class='token-row") == len(response_ids)
    assert parser.token_row_attributes["token-0"]["class"] == "token-row"
    assert "mutation-token-row" in parser.token_row_attributes["token-1"]["class"]
    assert "m001" in parser.token_rows[1][1][1]
    assert "mutation-token-row" not in parser.token_row_attributes["token-2"]["class"]
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

    no_mutation_rows = [dict(row, mutation_ids="") for row in rows]
    no_mutation_result = {
        **result,
        "sample": {**result["sample"], "changes": []},
        "mutation_observations": [],
        "tokens": no_mutation_rows,
    }
    no_mutation_report = probe._render_sample_html(no_mutation_result)
    assert '<section id="mutation-details">' not in no_mutation_report
    assert "<mark" not in no_mutation_report


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


def _teacher_signal_fixture() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    specs = [
        ("A", "correct", 0.20, 0.32, 101, "A", 0.32, 1),
        ("B", "correct", 0.32, 0.20, 900, "other", 0.70, 2),
        ("C", "hallucinated_insertion", 0.18, 0.30, 103, "C", 0.30, 1),
        ("D", "mutation_opposite_variant", 0.22, 0.34, 903, "other", 0.75, 2),
        ("E", "hallucinated_substitution", 0.40, 0.30, 904, "other", 0.60, 2),
        ("F", "correct", 0.50, 0.52, 106, "F", 0.52, 1),
        (" ", "formatting", 0.10, 0.40, 907, "space", 0.40, 1),
    ]
    rows: list[dict[str, object]] = []
    for index, (
        raw_token,
        token_label,
        p_original,
        p_teacher,
        top_id,
        top_token,
        top_p,
        teacher_rank,
    ) in enumerate(specs):
        token_id = 101 + index
        original = _score_record(
            token_id,
            raw_token,
            p_original,
            top_token_id=token_id,
            top_raw_token=raw_token,
            top_probability=p_original,
            target_rank=1,
        )
        teacher = _score_record(
            token_id,
            raw_token,
            p_teacher,
            top_token_id=top_id,
            top_raw_token=top_token,
            top_probability=top_p,
            target_rank=teacher_rank,
        )
        row = probe._combine_scores([token_id], [original], [teacher])[0]
        excluded = token_label == "formatting"
        row.update(
            {
                "pair_id": "pair-1",
                "report": "samples/001_pair-1/report.html",
                "sample_report": "samples/001_pair-1/report.html",
                "index": index,
                "token_label": token_label,
                "is_hallucination": token_label not in {"correct", "formatting"},
                "lexical_role": "formatting" if excluded else "word_initial",
                "normalized_piece": "" if excluded else raw_token,
                "normalized_char_count": 0 if excluded else 1,
                "script": "formatting" if excluded else "latin",
            }
        )
        rows.append(row)

    result = {
        "pair_id": "pair-1",
        "sample": {"image_copy": "input.png", "changes": []},
        "ground_truth": "ABCDEF",
        "response": {
            "text": "ABCDEF ",
            "token_ids": [int(row["token_id"]) for row in rows],
            "finish_reason": "stop",
        },
        "summary": {"token_count": len(rows)},
        "mutation_observations": [],
        "tokens": [dict(row) for row in rows],
    }
    return [result], rows


def _audit_value(audit: dict[str, object], *keys: str) -> object:
    sections: list[dict[str, object]] = [audit]
    for section_name in (
        "counts",
        "classification_counts",
        "signal_class_counts",
        "class_counts",
        "quadrants",
        "rates",
        "metrics",
        "denominators",
        "overall",
    ):
        section = audit.get(section_name)
        if isinstance(section, dict):
            sections.append(section)
    for key in keys:
        for section in sections:
            if key in section:
                return section[key]
    raise AssertionError(f"audit does not report any of {keys!r}")


def _audit_count(audit: dict[str, object], *keys: str) -> int:
    value = _audit_value(audit, *keys)
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return int(value)


def _relation_count(relation: dict[str, object], *keys: str) -> int:
    for key in keys:
        value = relation.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    raise AssertionError(f"relation does not report any of {keys!r}")


def test_teacher_signal_audit_classifies_lexical_tokens_and_denominators() -> None:
    results, audit_rows = _teacher_signal_fixture()

    audit = probe._build_teacher_signal_audit(
        results,
        audit_rows,
        selected_threshold=0.05,
    )

    assert _audit_value(audit, "selected_threshold") == pytest.approx(0.05)
    assert _audit_count(audit, "total_tokens", "total_rows") == 7
    assert _audit_count(audit, "evaluable_tokens", "lexical_tokens") == 6
    assert _audit_count(audit, "correct_tokens", "correct_token_count") == 3
    assert _audit_count(audit, "incorrect_tokens", "incorrect_token_count") == 3
    assert _audit_count(audit, "excluded_tokens", "excluded_token_count") == 1

    assert _audit_count(audit, "correct_reinforced") == 1
    assert _audit_count(audit, "harmful_correct_suppressed") == 1
    assert _audit_count(audit, "harmful_wrong_promoted") == 2
    assert _audit_count(audit, "wrong_suppressed") == 1
    assert _audit_count(audit, "neutral", "neutral_tokens") == 1

    assert _audit_value(audit, "harmful_signal_rate") == pytest.approx(3 / 5)
    assert _audit_value(audit, "correct_suppression_rate") == pytest.approx(1 / 3)
    assert _audit_value(audit, "wrong_promotion_rate") == pytest.approx(2 / 3)

    threshold_sweep = _audit_value(audit, "threshold_sweep")
    assert isinstance(threshold_sweep, list)
    assert len(threshold_sweep) >= 2
    assert any(
        isinstance(point, dict)
        and math.isclose(float(point["threshold"]), 0.05)
        for point in threshold_sweep
    )

    error_breakdown = _audit_value(
        audit,
        "error_label_breakdown",
        "error_type_breakdown",
    )
    assert isinstance(error_breakdown, list)
    errors_by_label = {
        str(row["token_label"]): row for row in error_breakdown
    }
    assert errors_by_label["hallucinated_insertion"]["incorrect_tokens"] == 1
    assert errors_by_label["mutation_opposite_variant"]["incorrect_tokens"] == 1
    assert errors_by_label["hallucinated_substitution"]["incorrect_tokens"] == 1

    top1_relation = _audit_value(
        audit,
        "harmful_wrong_teacher_top1_relation",
        "harmful_wrong_top1_relation",
        "wrong_promotion_teacher_top1_relation_counts",
    )
    assert isinstance(top1_relation, dict)
    assert _relation_count(
        top1_relation,
        "response_token_top1",
        "same_response_id",
    ) == 1
    assert _relation_count(
        top1_relation,
        "different_surface_top1",
        "different_surface",
    ) == 1


def test_teacher_signal_audit_html_is_standalone_and_links_sample_tokens() -> None:
    results, audit_rows = _teacher_signal_fixture()
    audit = probe._build_teacher_signal_audit(
        results,
        audit_rows,
        selected_threshold=0.05,
    )

    rendered = probe._render_teacher_signal_audit_html(audit, audit_rows)

    assert rendered.lstrip().lower().startswith("<!doctype html>")
    assert "<html" in rendered
    assert "</html>" in rendered
    assert "0.05" in rendered
    for label in (
        "正确强化",
        "有害：正确被抑制",
        "有害：错误被强化",
        "错误被抑制",
    ):
        assert label in rendered
    assert "Teacher Top-1" in rendered
    assert "samples/001_pair-1/report.html#token-2" in rendered
    assert "samples/001_pair-1/report.html#token-3" in rendered


def test_rebuild_privileged_report_writes_teacher_signal_artifacts(
    tmp_path: Path,
) -> None:
    results, _ = _teacher_signal_fixture()
    output_dir = tmp_path / "report"
    sample_dir = output_dir / "samples" / "001_pair-1"
    sample_dir.mkdir(parents=True)
    (sample_dir / "result.json").write_text(
        json.dumps(results[0], ensure_ascii=False),
        encoding="utf-8",
    )

    probe.rebuild_privileged_report(output_dir, teacher_signal_threshold=0.05)

    expected_sample_row = {
        "pair_id": "pair-1",
        "report": "samples/001_pair-1/report.html",
        "finish_reason": "stop",
        **results[0]["summary"],
    }
    assert (output_dir / "report.html").read_text(encoding="utf-8") == probe._render_aggregate_html(
        [expected_sample_row]
    )
    for filename in (
        "teacher_signal_audit.html",
        "teacher_signal_audit.json",
        "teacher_signal_tokens.csv",
        "teacher_signal_sample_summary.csv",
    ):
        assert (output_dir / filename).is_file()

    audit = json.loads(
        (output_dir / "teacher_signal_audit.json").read_text(encoding="utf-8")
    )
    assert _audit_value(audit, "selected_threshold") == pytest.approx(0.05)


def test_cli_exposes_teacher_signal_threshold() -> None:
    args = probe._build_parser().parse_args(
        [
            "--output-dir",
            "outputs/test",
            "--rebuild-report-only",
            "--teacher-signal-threshold",
            "0.125",
        ]
    )

    assert args.teacher_signal_threshold == pytest.approx(0.125)
