from __future__ import annotations

import io
import json
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from qwen_mm_token_probe import multi_gpu_probe as launcher
from qwen_mm_token_probe import privileged_probe
from qwen_mm_token_probe.progress import ProgressTracker


def _samples(root: Path, count: int) -> list[SimpleNamespace]:
    result = []
    for index in range(count):
        image = root / f"image-{index}.png"
        gt = root / f"gt-{index}.md"
        image.write_bytes(bytes([index % 255]) * (10 + index))
        gt.write_text(f"GT {index}", encoding="utf-8")
        result.append(
            SimpleNamespace(
                ordinal=index + 1,
                pair_id=f"pair-{index + 1}",
                image_path=image,
                ground_truth_path=gt,
                changes=(),
            )
        )
    return result


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _progress_lines(samples: list[Any]) -> str:
    image_bytes = 0
    lines = []
    for index, sample in enumerate(samples, start=1):
        image_bytes += sample.image_path.stat().st_size
        lines.append(
            "[qwen-mm-privileged-probe] CHECKPOINT "
            f'phase="unit-accepted" shard="gpu" '
            f'current_unit={index}/{len(samples)} current="{sample.pair_id}" '
            f"completed={index}/{len(samples)} percentage=100% "
            f"records={index} bytes={image_bytes}/{image_bytes} "
            "throughput_items_s=1.0 throughput_bytes_s=1.0 elapsed=00:01 eta=00:00 "
            f"accepted={index} rejected=0 skipped=0 errors=0 interrupted=false\n"
        )
    return "".join(lines)


class _FakePopen:
    calls: ClassVar[list[dict[str, Any]]] = []
    summaries: ClassVar[dict[int, dict[str, int]]] = {}
    exit_codes: ClassVar[dict[int, int]] = {}

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.shard_index = int(_option(command, "--shard-index"))
        self.returncode = self.exit_codes.get(self.shard_index, 0)
        self.pid = 10000 + self.shard_index
        dataset = privileged_probe.load_release_samples(
            Path(_option(command, "--dataset-root")),
            limit=int(_option(command, "--limit"))
            if "--limit" in command
            else None,
            require_table="--require-table" in command,
        )
        self.assigned = privileged_probe._partition_samples(
            dataset,
            num_nodes=int(_option(command, "--num-nodes")),
            node_rank=int(_option(command, "--node-rank")),
            num_shards=int(_option(command, "--num-shards")),
            shard_index=self.shard_index,
        )
        self.stdout = io.StringIO(_progress_lines(self.assigned))
        output_root = Path(_option(command, "--output-dir"))
        node_rank = int(_option(command, "--node-rank"))
        worker_dir = (
            output_root
            / "workers"
            / f"node_{node_rank:03d}"
            / f"shard_{self.shard_index:03d}"
        )
        worker_dir.mkdir(parents=True, exist_ok=True)
        (worker_dir / "config.json").write_text(
            json.dumps(
                {
                    "node_rank": node_rank,
                    "shard_index": self.shard_index,
                    "num_nodes": int(_option(command, "--num-nodes")),
                    "num_shards": int(_option(command, "--num-shards")),
                    "worker_mode": True,
                    "samples_selected": len(self.assigned),
                    "samples_total": len(dataset),
                    "teacher_signal_threshold": 0.05,
                    "student_response_min_probability": None,
                    "student_response_max_probability": None,
                }
            ),
            encoding="utf-8",
        )
        summary = self.summaries.get(
            self.shard_index,
            {
                "total_items": len(self.assigned),
                "completed_items": len(self.assigned),
                "failed_items": 0,
            },
        )
        (worker_dir / "run_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        self.calls.append(
            {
                "command": command,
                "env": kwargs["env"],
                "assigned_ordinals": [sample.ordinal for sample in self.assigned],
            }
        )

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return int(self.returncode)

    def send_signal(self, sig: int) -> None:
        self.returncode = -int(sig)


def _setup_fake_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakePopen.calls = []
    _FakePopen.summaries = {}
    _FakePopen.exit_codes = {}
    monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)


def _invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    samples: list[Any],
    gpus: list[str],
    num_nodes: int,
    node_rank: int,
) -> dict[str, Any]:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    dataset_root.mkdir(exist_ok=True)
    monkeypatch.setattr(
        privileged_probe,
        "load_release_samples",
        lambda root, *, limit, require_table: samples[:limit]
        if limit is not None
        else samples,
    )
    return launcher.launch_gpu_workers(
        [
            "--gpus",
            "ignored",
            "--num-nodes",
            "99",
            "--node-rank=99",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--model-id",
            "student-model",
        ],
        gpu_ids=gpus,
        num_nodes=num_nodes,
        node_rank=node_rank,
        output_dir=output_dir,
        dataset_root=dataset_root,
        limit=None,
        require_table=False,
        heartbeat_seconds=3600,
    )


def test_launcher_forwards_gpu_and_node_assignment_and_keeps_partial_node_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(tmp_path, 9)
    _setup_fake_workers(monkeypatch)
    rebuild_calls = []
    monkeypatch.setattr(
        privileged_probe,
        "rebuild_privileged_report",
        lambda *args, **kwargs: rebuild_calls.append((args, kwargs)) or {},
    )

    summary = _invoke(
        tmp_path,
        monkeypatch,
        samples=samples,
        gpus=["4", "5", "6"],
        num_nodes=2,
        node_rank=1,
    )

    assert summary["status"] == "success"
    assert summary["assigned_items"] == 4
    assert summary["completed_items"] == 4
    assert summary["accepted"] == 4
    assert summary["failed_workers"] == 0
    assert summary["next_step"]
    assert rebuild_calls == []
    assert len(_FakePopen.calls) == 3

    assigned: list[int] = []
    for call in list(_FakePopen.calls):
        command = call["command"]
        assert "--gpus" not in command
        assert _option(command, "--num-nodes") == "2"
        assert _option(command, "--node-rank") == "1"
        assert call["env"]["CUDA_VISIBLE_DEVICES"] in {"4", "5", "6"}
        assert "--worker-mode" in command
        assigned.extend(call["assigned_ordinals"])
    assert sorted(assigned) == [2, 4, 6, 8]
    assert len(set(assigned)) == len(assigned)
    assert (tmp_path / "output" / "workers" / "node_001" / "node_summary.json").is_file()
    assert not (tmp_path / "output" / "config.json").exists()


def test_single_node_rebuilds_successful_results_even_if_a_worker_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(tmp_path, 4)
    _setup_fake_workers(monkeypatch)
    _FakePopen.exit_codes[1] = 7
    _FakePopen.summaries[1] = {
        "total_items": 2,
        "completed_items": 1,
        "failed_items": 1,
    }
    rebuild_calls = []
    monkeypatch.setattr(
        privileged_probe,
        "rebuild_privileged_report",
        lambda *args, **kwargs: rebuild_calls.append((args, kwargs)) or {"ok": True},
    )

    summary = _invoke(
        tmp_path,
        monkeypatch,
        samples=samples,
        gpus=["0", "1"],
        num_nodes=1,
        node_rank=0,
    )

    assert summary["status"] == "failed"
    assert summary["failed_workers"] == 1
    assert len(rebuild_calls) == 1
    root_config = json.loads((tmp_path / "output" / "config.json").read_text())
    assert root_config["samples_selected"] == 4
    assert root_config["samples_total"] == 4
    assert root_config["num_nodes"] == 1
    assert "node_rank" not in root_config
    assert "shard_index" not in root_config
    assert "worker_mode" not in root_config
    assert "num_shards" not in root_config


def test_error_status_maps_to_parent_error_and_multi_completion_advances_index(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tracker = ProgressTracker(
        task="test-parent-progress",
        total_items=2,
        heartbeat_seconds=3600,
    )
    tracker.start()
    handle = launcher._WorkerHandle(
        shard_index=0,
        gpu_id="0",
        assigned_items=2,
        worker_dir=Path("worker"),
        log_path=Path("worker.log"),
        process=SimpleNamespace(),
        reader=SimpleNamespace(),
        progress=launcher._WorkerProgress.create(),
    )

    launcher._consume_worker_line(
        '[qwen-mm-privileged-probe] CHECKPOINT phase="unit-error" '
        'current="pair-2" completed=2/2 records=2 bytes=10/10 '
        "accepted=1 rejected=0 skipped=0 errors=1 interrupted=false",
        handle=handle,
        tracker=tracker,
        node_rank=0,
        num_nodes=1,
    )
    final = tracker.finish()

    assert final.completed_items == 2
    assert final.accepted == 1
    assert final.errors == 1
    assert tracker.current_index == 2
    assert "current_unit=2/2" in capsys.readouterr().out


def test_sigint_during_worker_startup_stops_already_started_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(tmp_path, 4)
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    dataset_root.mkdir()
    monkeypatch.setattr(
        privileged_probe,
        "load_release_samples",
        lambda root, *, limit, require_table: samples,
    )
    started: list[Any] = []

    class _InterruptSecondStart(_FakePopen):
        attempts = 0

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            type(self).attempts += 1
            if type(self).attempts == 2:
                raise KeyboardInterrupt
            super().__init__(command, **kwargs)
            self.returncode = None
            started.append(self)

    _FakePopen.calls = []
    _FakePopen.summaries = {}
    _FakePopen.exit_codes = {}
    monkeypatch.setattr(launcher.subprocess, "Popen", _InterruptSecondStart)
    monkeypatch.setattr(launcher, "_INTERRUPT_GRACE_SECONDS", 0.0)

    summary = launcher.launch_gpu_workers(
        ["--dataset-root", str(dataset_root), "--output-dir", str(output_dir)],
        gpu_ids=["0", "1"],
        num_nodes=1,
        node_rank=0,
        output_dir=output_dir,
        dataset_root=dataset_root,
        limit=None,
        require_table=False,
        heartbeat_seconds=3600,
    )

    assert summary["interrupted"] is True
    assert summary["status"] == "interrupted"
    assert len(started) == 1
    assert started[0].returncode == -signal.SIGINT


def test_stop_children_waits_after_term_and_escalates_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(returncode=None, pid=123)
    process.poll = lambda: process.returncode
    process.send_signal = lambda _sig: None
    process.wait = lambda timeout=None: _timeout_wait(process)
    signals: list[int] = []

    def signal_group(_process: Any, sig: int) -> None:
        signals.append(sig)
        if sig == signal.SIGKILL:
            process.returncode = -sig

    monkeypatch.setattr(launcher, "_signal_process_group", signal_group)
    monkeypatch.setattr(launcher, "_INTERRUPT_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(launcher, "_TERMINATE_GRACE_SECONDS", 0.0)

    launcher._stop_children([process])

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def _timeout_wait(process: Any) -> int:
    if process.returncode is None:
        raise launcher.subprocess.TimeoutExpired("fake-worker", 0)
    return int(process.returncode)
