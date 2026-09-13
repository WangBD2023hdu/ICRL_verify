from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch
from PIL import Image

from qwen_mm_token_probe import privileged_probe as probe
from qwen_mm_token_probe.hf_qwen import ModelBundle


class BatchTokenizer:
    all_special_ids: ClassVar[list[int]] = []
    pad_token_id = 0

    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []
        self.pieces = {token_id: chr(65 + token_id - 7) for token_id in range(7, 30)}

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return "".join(
            self.pieces.get(int(token_id), f"<{token_id}>") for token_id in token_ids
        )

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **_: Any
    ) -> dict[str, torch.Tensor]:
        self.messages.append(messages)
        text = str(messages[0]["content"])
        match = re.search(r"GT-(\d+)", text)
        assert match is not None
        ordinal = int(match.group(1))
        ids = [20 + ordinal]
        if ordinal % 2 == 0:
            ids.append(40 + ordinal)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


class BatchRecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        logits_to_keep: int | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.calls.append(input_ids.detach().cpu().clone())
        sequence_length = int(input_ids.shape[-1])
        logits = torch.full((1, sequence_length, 128), -12.0)
        for position in range(sequence_length - 1):
            next_id = int(input_ids[0, position + 1])
            logits[0, position, next_id] = 12.0
        logits[0, -1, 0] = 12.0
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


class RecordingTracker:
    instances: ClassVar[list[RecordingTracker]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.completed: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.__class__.instances.append(self)

    def start(self) -> None:
        pass

    def set_current(self, **_: Any) -> None:
        pass

    def note_error(self, **kwargs: Any) -> None:
        self.errors.append(kwargs)

    def complete_unit(self, **kwargs: Any) -> None:
        self.completed.append(kwargs)

    def finish(self, **_: Any) -> None:
        pass


def _bundle(model_id: str) -> ModelBundle:
    return ModelBundle(
        model_id=model_id,
        model=BatchRecordingModel(),
        processor=object(),
        tokenizer=BatchTokenizer(),
        device=torch.device("cpu"),
    )


def _make_release(root: Path, count: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal in range(1, count + 1):
        image_path = root / f"page-{ordinal}.png"
        Image.new("RGB", (12, 12), "white").save(image_path)
        gt_path = root / f"page-{ordinal}.md"
        gt_path.write_text(f"GT-{ordinal}\nBody {ordinal}\n", encoding="utf-8")
        records.append(
            {
                "pair_id": f"p{ordinal}",
                "edited_image": image_path.name,
                "edited_markdown": gt_path.name,
                "changes": [],
            }
        )
    (root / "pairs.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return root


def _ordinal_from_image_path(value: str | Path) -> int:
    path = Path(value)
    match = re.search(r"(?:page-|input)(\d+)", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"\d+_p(\d+)$", path.parent.name)
    assert match is not None, path
    return int(match.group(1))


def _install_run_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_responses: dict[int, list[int]],
    score_hook: Any | None = None,
) -> dict[str, Any]:
    student = _bundle("student")
    teacher = _bundle("teacher")
    generation_calls: list[list[int]] = []
    generation_batches: list[list[int]] = []
    prepared_prompts: list[str] = []

    def prepare_prompt_inputs(**kwargs: Any) -> dict[str, torch.Tensor]:
        ordinal = _ordinal_from_image_path(kwargs["image_path"])
        prepared_prompts.append(kwargs["prompt"])
        ids = [100 + ordinal]
        if ordinal % 2 == 0:
            ids.append(110 + ordinal)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def generate_batch_from_prompts(**kwargs: Any) -> list[tuple[list[int], str]]:
        prompts = kwargs["prompt_inputs"]
        ordinals = [int(item["input_ids"][0, 0]) - 100 for item in prompts]
        generation_calls.append(ordinals)
        generation_batches.append(ordinals)
        return [(expected_responses[ordinal], "unused") for ordinal in ordinals]

    load_calls: list[str] = []

    def load_model_bundle(model_id: str, **_: Any) -> ModelBundle:
        load_calls.append(model_id)
        return student if model_id == "student" else teacher

    original_score = probe._score_fixed_response_ids

    def score_fixed_response_ids(**kwargs: Any) -> list[dict[str, Any]]:
        response_ids = [int(token_id) for token_id in kwargs["response_ids"]]
        if score_hook is not None:
            score_hook(response_ids, kwargs)
        return original_score(**kwargs)

    def write_sample_outputs(
        sample_dir: Path, _: dict[str, Any], *, rendered_html: str | None = None
    ) -> None:
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "report.html").write_text(
            rendered_html or "test report", encoding="utf-8"
        )

    def rebuild_report(output_root: Path, **_: Any) -> dict[str, int]:
        return {
            "completed_samples": len(
                list((output_root / "samples").glob("*/result.json"))
            )
        }

    RecordingTracker.instances.clear()
    monkeypatch.setattr(probe, "ProgressTracker", RecordingTracker)
    monkeypatch.setattr(probe, "load_model_bundle", load_model_bundle)
    monkeypatch.setattr(probe, "prepare_prompt_inputs", prepare_prompt_inputs)
    monkeypatch.setattr(
        probe, "generate_batch_from_prompts", generate_batch_from_prompts
    )
    monkeypatch.setattr(probe, "_score_fixed_response_ids", score_fixed_response_ids)
    monkeypatch.setattr(probe, "_write_sample_outputs", write_sample_outputs)
    monkeypatch.setattr(probe, "rebuild_privileged_report", rebuild_report)
    return {
        "student": student,
        "teacher": teacher,
        "generation_calls": generation_calls,
        "generation_batches": generation_batches,
        "prepared_prompts": prepared_prompts,
        "load_calls": load_calls,
    }


def _run(
    dataset_root: Path,
    output_dir: Path,
    *,
    batch_size: int,
    resume: bool = True,
) -> probe.PrivilegedProbeSummary:
    return probe.run_privileged_probe(
        model_id="student",
        teacher_model_id="teacher",
        dataset_root=dataset_root,
        output_dir=output_dir,
        prompt="OCR_PROMPT_EXACT",
        max_new_tokens=32,
        top_k=3,
        forward_chunk_size=2,
        batch_size=batch_size,
        postprocess_workers=0,
        device_map=None,
        dtype="float32",
        min_pixels=16,
        max_pixels=1024,
        image_patch_size=16,
        heartbeat_seconds=0,
        resume=resume,
    )


def _sample_dir(output_dir: Path, ordinal: int) -> Path:
    return output_dir / "samples" / f"{ordinal:03d}_p{ordinal}"


def test_batch_run_groups_five_pages_and_keeps_ids_and_contexts_aligned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_release(tmp_path / "dataset", 5)
    output = tmp_path / "output"
    responses = {1: [7, 8], 2: [9], 3: [10, 11], 4: [12], 5: [13, 14]}
    response_to_ordinal = {ids[0]: ordinal for ordinal, ids in responses.items()}
    batch_members: dict[int, list[int]] = {}
    first_members: set[int] = set()
    score_ordinals: set[int] = set()
    holder: dict[str, Any] = {}

    def inspect_partial_before_score(
        response_ids: list[int], _: dict[str, Any]
    ) -> None:
        ordinal = response_to_ordinal[response_ids[0]]
        partial_path = _sample_dir(output, ordinal) / "partial.json"
        assert partial_path.is_file(), f"partial missing before scoring p{ordinal}"
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        assert partial["response_ids"] == responses[ordinal]
        score_ordinals.add(ordinal)
        if ordinal in first_members and ordinal not in holder.setdefault(
            "checked_groups", set()
        ):
            for member in batch_members[ordinal]:
                member_partial = _sample_dir(output, member) / "partial.json"
                assert member_partial.is_file(), (
                    f"batch member p{member} not checkpointed"
                )
            holder["checked_groups"].add(ordinal)

    holder.update(
        _install_run_stubs(
            monkeypatch,
            expected_responses=responses,
            score_hook=inspect_partial_before_score,
        )
    )
    generation_batches = holder["generation_batches"]
    # Record the group membership at generation time for the partial-before-score assertion.
    original_generate = probe.generate_batch_from_prompts

    def generate_and_record_groups(**kwargs: Any) -> list[tuple[list[int], str]]:
        values = original_generate(**kwargs)
        group = list(generation_batches[-1])
        for ordinal in group:
            batch_members[ordinal] = group
        first_members.add(group[0])
        return values

    monkeypatch.setattr(
        probe, "generate_batch_from_prompts", generate_and_record_groups
    )

    summary = _run(dataset, output, batch_size=2)

    assert holder["generation_calls"] == [[1, 2], [3, 4], [5]]
    assert summary.total_items == summary.completed_items == 5
    assert summary.failed_items == summary.skipped_items == 0
    assert score_ordinals == set(range(1, 6))
    assert holder["prepared_prompts"] == ["OCR_PROMPT_EXACT"] * 5
    assert [item["status"] for item in RecordingTracker.instances[-1].completed] == [
        "accepted"
    ] * 5
    assert [item["index"] for item in RecordingTracker.instances[-1].completed] == list(
        range(1, 6)
    )

    student_model = holder["student"].model
    teacher_model = holder["teacher"].model
    assert len(student_model.calls) == len(teacher_model.calls) == 5
    for ordinal, response_ids in responses.items():
        expected_original_prefix = [100 + ordinal]
        if ordinal % 2 == 0:
            expected_original_prefix.append(110 + ordinal)
        expected_teacher_prefix = [20 + ordinal]
        if ordinal % 2 == 0:
            expected_teacher_prefix.append(40 + ordinal)
        assert student_model.calls[ordinal - 1].tolist() == [
            expected_original_prefix + response_ids
        ]
        assert teacher_model.calls[ordinal - 1].tolist() == [
            expected_teacher_prefix + response_ids
        ]

        sample_dir = _sample_dir(output, ordinal)
        result = json.loads((sample_dir / "result.json").read_text(encoding="utf-8"))
        assert result["ground_truth"] == f"GT-{ordinal}\nBody {ordinal}\n"
        assert result["response"]["token_ids"] == response_ids
        assert result["protocol"]["generation_batch_size"] == (2 if ordinal < 5 else 1)
        assert result["protocol"]["original_prompt_sha256"] == probe._sha256_text(
            "OCR_PROMPT_EXACT"
        )
        assert (sample_dir / "ground_truth.md").read_text(encoding="utf-8") == result[
            "ground_truth"
        ]
        privileged = (sample_dir / "privileged_prompt.txt").read_text(encoding="utf-8")
        assert result["ground_truth"] in privileged
        assert not (sample_dir / "partial.json").exists()


def test_resume_skips_complete_results_and_no_resume_generates_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_release(tmp_path / "dataset", 2)
    output = tmp_path / "output"
    responses = {1: [7, 8], 2: [9, 10]}
    holder = _install_run_stubs(monkeypatch, expected_responses=responses)

    first = _run(dataset, output, batch_size=2)
    assert first.completed_items == 2
    assert holder["generation_calls"] == [[1, 2]]

    resumed = _run(dataset, output, batch_size=2, resume=True)
    assert resumed.completed_items == 2
    assert resumed.skipped_items == 2
    assert holder["generation_calls"] == [[1, 2]]
    tracker = RecordingTracker.instances[-1]
    assert [item["status"] for item in tracker.completed] == ["skipped", "skipped"]

    forced = _run(dataset, output, batch_size=2, resume=False)
    assert forced.completed_items == 2
    assert holder["generation_calls"] == [[1, 2], [1, 2]]
    for ordinal in (1, 2):
        result = json.loads(
            (_sample_dir(output, ordinal) / "result.json").read_text(encoding="utf-8")
        )
        assert result["protocol"]["generation_performed_this_run"] is True
        assert result["response"]["token_ids"] == responses[ordinal]


def test_partial_generation_is_resumed_without_regenerating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_release(tmp_path / "dataset", 2)
    output = tmp_path / "output"
    responses = {1: [7, 8], 2: [9, 10]}
    fail_first_score = {"pending": True}

    def fail_once(response_ids: list[int], _: dict[str, Any]) -> None:
        if response_ids[0] == responses[1][0] and fail_first_score["pending"]:
            fail_first_score["pending"] = False
            raise RuntimeError("simulated scoring interruption")

    holder = _install_run_stubs(
        monkeypatch,
        expected_responses=responses,
        score_hook=fail_once,
    )
    first = _run(dataset, output, batch_size=2)
    first_partial = _sample_dir(output, 1) / "partial.json"
    assert first.failed_items == 1
    assert first.completed_items == 1
    assert first_partial.is_file()
    saved_partial = json.loads(first_partial.read_text(encoding="utf-8"))
    assert saved_partial["response_ids"] == responses[1]
    assert holder["generation_calls"] == [[1, 2]]

    # The one-shot fault has been consumed; p1 should continue from its response-ID checkpoint.
    resumed = _run(dataset, output, batch_size=2, resume=True)

    assert resumed.completed_items == 2
    assert resumed.skipped_items == 1
    assert holder["generation_calls"] == [[1, 2]]
    result = json.loads(
        (_sample_dir(output, 1) / "result.json").read_text(encoding="utf-8")
    )
    assert result["response"]["token_ids"] == responses[1]
    assert result["protocol"]["generation_performed_this_run"] is False
    assert not first_partial.exists()


def test_result_with_partial_checkpoint_repairs_failed_exports_without_regeneration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_release(tmp_path / "dataset", 2)
    output = tmp_path / "output"
    responses = {1: [7, 8], 2: [9, 10]}
    holder = _install_run_stubs(monkeypatch, expected_responses=responses)
    fail_p1_export = {"pending": True}

    def fail_first_export(
        sample_dir: Path,
        result: dict[str, Any],
        *,
        rendered_html: str | None = None,
    ) -> None:
        if result["pair_id"] == "p1" and fail_p1_export["pending"]:
            fail_p1_export["pending"] = False
            raise OSError("simulated export failure")
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "report.html").write_text(
            rendered_html or "repaired report", encoding="utf-8"
        )

    monkeypatch.setattr(probe, "_write_sample_outputs", fail_first_export)

    failed_run = _run(dataset, output, batch_size=2)
    p1_dir = _sample_dir(output, 1)
    assert failed_run.failed_items == 1
    assert (p1_dir / "result.json").is_file()
    assert (p1_dir / "partial.json").is_file()
    assert holder["generation_calls"] == [[1, 2]]

    repaired_run = _run(dataset, output, batch_size=2, resume=True)

    assert repaired_run.completed_items == 2
    assert repaired_run.skipped_items == 1
    assert holder["generation_calls"] == [[1, 2]]
    assert not (p1_dir / "partial.json").exists()
    assert (p1_dir / "report.html").is_file()
    repaired = json.loads((p1_dir / "result.json").read_text(encoding="utf-8"))
    assert repaired["response"]["token_ids"] == responses[1]
    assert repaired["protocol"]["generation_performed_this_run"] is False


def test_batch_cli_defaults_and_no_resume_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = probe._build_parser()
    defaults = parser.parse_args(["--dataset-root", "data", "--output-dir", "out"])
    assert defaults.batch_size == 1
    assert defaults.postprocess_workers is None
    assert defaults.num_nodes == 1
    assert defaults.node_rank == 0
    assert defaults.num_shards == 1
    assert defaults.shard_index == 0

    received: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> probe.PrivilegedProbeSummary:
        received.update(kwargs)
        return probe.PrivilegedProbeSummary(Path("out"), 4, 4, 0, 0, False)

    monkeypatch.setattr(probe, "run_privileged_probe", fake_run)
    assert (
        probe.main(
            [
                "--dataset-root",
                "data",
                "--output-dir",
                "out",
                "--batch-size",
                "4",
                "--postprocess-workers",
                "0",
                "--num-nodes",
                "2",
                "--node-rank",
                "1",
                "--num-shards",
                "3",
                "--shard-index",
                "2",
                "--worker-mode",
                "--no-resume",
            ]
        )
        == 0
    )
    assert received["batch_size"] == 4
    assert received["postprocess_workers"] == 0
    assert received["resume"] is False
    assert received["num_nodes"] == 2
    assert received["node_rank"] == 1
    assert received["num_shards"] == 3
    assert received["shard_index"] == 2
    assert received["worker_mode"] is True


def test_gpu_launcher_cli_receives_gpu_list_and_node_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}
    launcher_module = ModuleType("qwen_mm_token_probe.multi_gpu_probe")

    def launch_gpu_workers(argv: list[str], **kwargs: Any) -> dict[str, Any]:
        received["argv"] = argv
        received.update(kwargs)
        return {"failed_workers": 0, "interrupted": False}

    launcher_module.launch_gpu_workers = launch_gpu_workers  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, launcher_module.__name__, launcher_module)
    argv = [
        "--dataset-root",
        "data",
        "--output-dir",
        "out",
        "--gpus",
        "0,1,2",
        "--num-nodes",
        "2",
        "--node-rank",
        "1",
    ]

    assert probe.main(argv) == 0
    assert received["argv"] == argv
    assert received["gpu_ids"] == ["0", "1", "2"]
    assert received["num_nodes"] == 2
    assert received["node_rank"] == 1
    assert received["output_dir"] == Path("out")
    assert received["dataset_root"] == Path("data")


def test_partition_uses_different_gpu_counts_per_node_with_complete_disjoint_coverage(
    tmp_path: Path,
) -> None:
    # The sequence represents globally filtered samples and intentionally has ordinal gaps.
    ordinals = [1, 3, 5, 8, 11, 14, 19, 23, 30, 31, 42, 57, 60, 81, 99]
    samples = [
        probe.PrivilegedProbeSample(
            ordinal=ordinal,
            pair_id=f"p{ordinal}",
            image_path=tmp_path / f"{ordinal}.png",
            ground_truth_path=tmp_path / f"{ordinal}.md",
            changes=(),
        )
        for ordinal in ordinals
    ]

    assigned: list[int] = []
    for node_rank, gpu_count in ((0, 2), (1, 3)):
        for shard_index in range(gpu_count):
            shard = probe._partition_samples(
                samples,
                num_nodes=2,
                node_rank=node_rank,
                num_shards=gpu_count,
                shard_index=shard_index,
            )
            shard_ordinals = [sample.ordinal for sample in shard]
            assert shard_ordinals == sorted(shard_ordinals)
            assigned.extend(shard_ordinals)

    assert sorted(assigned) == ordinals
    assert len(assigned) == len(set(assigned))


def test_worker_mode_writes_worker_state_skips_global_report_and_uses_filtered_ordinals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_release(tmp_path / "dataset", 12)
    output = tmp_path / "output"
    responses = {6: [7, 8], 12: [9, 10]}

    def fail_sample_12(response_ids: list[int], _: dict[str, Any]) -> None:
        if response_ids[0] == responses[12][0]:
            raise RuntimeError("simulated worker inference failure")

    holder = _install_run_stubs(
        monkeypatch, expected_responses=responses, score_hook=fail_sample_12
    )

    def unexpected_report(*_: Any, **__: Any) -> dict[str, Any]:
        pytest.fail(
            "worker mode must leave aggregate report construction to the collector"
        )

    monkeypatch.setattr(probe, "rebuild_privileged_report", unexpected_report)
    summary = probe.run_privileged_probe(
        model_id="student",
        teacher_model_id="teacher",
        dataset_root=dataset,
        output_dir=output,
        prompt="OCR_PROMPT_EXACT",
        max_new_tokens=32,
        top_k=3,
        forward_chunk_size=2,
        batch_size=2,
        postprocess_workers=0,
        num_nodes=2,
        node_rank=1,
        num_shards=3,
        shard_index=2,
        worker_mode=True,
        device_map=None,
        dtype="float32",
        min_pixels=16,
        max_pixels=1024,
        image_patch_size=16,
        heartbeat_seconds=0,
    )

    worker_root = output / "workers" / "node_001" / "shard_002"
    assert summary.total_items == 2
    assert summary.completed_items == 1
    assert summary.failed_items == 1
    assert holder["generation_calls"] == [[6, 12]]
    assert set(holder["load_calls"]) == {"student", "teacher"}
    tracker_events = RecordingTracker.instances[-1].completed
    assert sorted(item["status"] for item in tracker_events) == ["accepted", "error"]
    assert sorted(item["index"] for item in tracker_events) == [6, 12]
    assert (worker_root / "config.json").is_file()
    assert (worker_root / "run_summary.json").is_file()
    config = json.loads((worker_root / "config.json").read_text(encoding="utf-8"))
    assert config["num_nodes"] == 2
    assert config["node_rank"] == 1
    assert config["num_shards"] == 3
    assert config["shard_index"] == 2
    assert config["worker_mode"] is True
    worker_summary = json.loads(
        (worker_root / "run_summary.json").read_text(encoding="utf-8")
    )
    assert worker_summary["total_items"] == 2
    assert worker_summary["completed_items"] == 1
    assert worker_summary["failed_items"] == 1
    failure_rows = [
        json.loads(line)
        for line in (worker_root / "failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["pair_id"] for row in failure_rows] == ["p12"]
    assert sorted(path.name for path in (output / "samples").iterdir()) == [
        "006_p6",
        "012_p12",
    ]
    assert (output / "samples" / "012_p12" / "partial.json").is_file()
    assert not (output / "samples" / "012_p12" / "result.json").exists()
    assert not (output / "report.html").exists()


def test_oom_generation_batch_splits_by_ordinal_even_for_duplicate_pair_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from qwen_mm_token_probe.hf_qwen import ModelBundle

    samples = []
    pending = []
    prepared: dict[int, dict[str, torch.Tensor]] = {}
    responses = {1: [7], 2: [8], 3: [9]}
    for ordinal in range(1, 4):
        image = tmp_path / f"page-{ordinal}.png"
        Image.new("RGB", (4, 4), "white").save(image)
        gt = tmp_path / f"page-{ordinal}.md"
        gt.write_text(f"GT-{ordinal}", encoding="utf-8")
        pair_id = "shared-id" if ordinal <= 2 else f"p{ordinal}"
        sample = probe.PrivilegedProbeSample(ordinal, pair_id, image, gt, ())
        sample_dir = tmp_path / "samples" / f"{ordinal:03d}_{pair_id}"
        sample_dir.mkdir(parents=True)
        samples.append(sample)
        pending.append((sample, sample_dir, f"fp-{ordinal}", None))
        prepared[sample.ordinal] = {
            "input_ids": torch.tensor([[100 + ordinal]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

    calls: list[list[int]] = []
    oom_flushes: list[str] = []
    generated_now: set[int] = set()

    def fake_generate_batch(**kwargs: Any) -> list[tuple[list[int], str]]:
        prompts = kwargs["prompt_inputs"]
        ordinals = [int(item["input_ids"][0, 0]) - 100 for item in prompts]
        calls.append(ordinals)
        if len(prompts) > 1:
            raise torch.OutOfMemoryError("synthetic CUDA OOM")
        return [(responses[ordinals[0]], "unused")]

    monkeypatch.setattr(probe, "generate_batch_from_prompts", fake_generate_batch)
    monkeypatch.setattr(
        probe, "_empty_device_cache", lambda device: oom_flushes.append(str(device))
    )
    bundle = ModelBundle(
        model_id="student",
        model=BatchRecordingModel(),
        processor=object(),
        tokenizer=BatchTokenizer(),
        device=torch.device("cpu"),
    )

    probe._generate_pending_batch(
        pending,
        prepared=prepared,
        model_bundle=bundle,
        max_new_tokens=8,
        seed=7,
        tracker=RecordingTracker(),
        generated_now=generated_now,
    )

    assert calls == [[1, 2, 3], [1], [2, 3], [2], [3]]
    assert len(oom_flushes) == 2
    assert generated_now == {1, 2, 3}
    for ordinal, item in enumerate(pending, start=1):
        partial = json.loads((item[1] / "partial.json").read_text(encoding="utf-8"))
        assert partial["response_ids"] == responses[ordinal]
        assert partial["generation_batch_size"] == 1


def _make_raw_deferred_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[probe.PrivilegedProbeSample, Path, dict[str, Any]]:
    image_path = tmp_path / "page-1.png"
    Image.new("RGB", (12, 12), "white").save(image_path)
    gt_path = tmp_path / "page-1.md"
    gt_path.write_text("GT-1\nAB\n", encoding="utf-8")
    sample = probe.PrivilegedProbeSample(1, "p1", image_path, gt_path, ())
    sample_dir = tmp_path / "samples" / "001_p1"

    def prepare_prompt_inputs(**_: Any) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[101]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

    def generate_from_prompt(**_: Any) -> tuple[list[int], str]:
        return [7, 8], "ignored"

    monkeypatch.setattr(probe, "prepare_prompt_inputs", prepare_prompt_inputs)
    monkeypatch.setattr(probe, "generate_from_prompt", generate_from_prompt)
    student = _bundle("student")
    teacher = _bundle("teacher")
    result = probe._run_sample(
        sample=sample,
        sample_dir=sample_dir,
        fingerprint="deferred-fingerprint",
        student_model_bundle=student,
        teacher_model_bundle=teacher,
        prompt="OCR_PROMPT_EXACT",
        privileged_instruction="Transcribe exactly",
        max_new_tokens=16,
        top_k=3,
        forward_chunk_size=2,
        min_pixels=16,
        max_pixels=1024,
        image_patch_size=16,
        seed=7,
        tracker=RecordingTracker(),
        defer_postprocess=True,
        resume=False,
    )
    return sample, sample_dir, result


def test_spawn_postprocess_writer_saves_deferred_result_before_deleting_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample, sample_dir, raw_result = _make_raw_deferred_result(monkeypatch, tmp_path)
    assert raw_result.get("_needs_postprocessing") is True
    partial_path = sample_dir / "partial.json"
    assert partial_path.is_file()

    tracker = RecordingTracker()
    writer = probe._SampleResultWriter(
        tracker=tracker,
        output_root=tmp_path / "output",
        workers=2,
        max_pending=2,
    )
    persist_observations: list[dict[str, Any]] = []
    original_persist = writer._persist

    def observe_persist(
        saved_sample: probe.PrivilegedProbeSample,
        saved_dir: Path,
        final_result: dict[str, Any],
        rendered_html: str | None = None,
    ) -> None:
        assert (saved_dir / "partial.json").is_file()
        assert not (saved_dir / "result.json").exists()
        assert final_result.get("_needs_postprocessing") is None
        assert final_result.get("summary", {}).get("token_count") == 2
        assert rendered_html and "全部 Response Token" in rendered_html
        persist_observations.append(final_result)
        original_persist(saved_sample, saved_dir, final_result, rendered_html)

    writer._persist = observe_persist  # type: ignore[method-assign]
    writer.submit(sample, sample_dir, raw_result)
    writer.close()

    assert writer.completed == 1
    assert writer.failed == 0
    assert len(persist_observations) == 1
    assert [item["status"] for item in tracker.completed] == ["accepted"]
    assert not partial_path.exists()
    saved = json.loads((sample_dir / "result.json").read_text(encoding="utf-8"))
    assert "_needs_postprocessing" not in saved
    assert len(saved["tokens"]) == 2
    assert all(0.0 <= row["p_original"] <= 1.0 for row in saved["tokens"])
    assert all(0.0 <= row["p_teacher"] <= 1.0 for row in saved["tokens"])
    assert all("token_label" in row for row in saved["tokens"])
    assert saved["summary"]["token_count"] == 2
    assert (sample_dir / "report.html").is_file()
    assert (sample_dir / "token_probabilities.csv").is_file()
    assert (sample_dir / "token_category_summary.json").is_file()


def test_writer_export_error_preserves_partial_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample, sample_dir, raw_result = _make_raw_deferred_result(monkeypatch, tmp_path)
    partial_path = sample_dir / "partial.json"
    tracker = RecordingTracker()
    writer = probe._SampleResultWriter(
        tracker=tracker,
        output_root=tmp_path / "output",
        workers=0,
        max_pending=1,
    )

    def fail_export(*_: Any, **__: Any) -> None:
        raise OSError("simulated export failure")

    monkeypatch.setattr(probe, "_write_sample_outputs", fail_export)
    with pytest.raises(OSError, match="simulated export failure"):
        writer.submit(sample, sample_dir, raw_result)

    assert partial_path.is_file()
    assert (sample_dir / "result.json").is_file()
    assert writer.completed == 0
    assert [item["status"] for item in tracker.completed] == []
    writer.close()
