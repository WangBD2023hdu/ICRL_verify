"""Local multi-GPU subprocess launcher for the privileged token probe."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .progress import ProgressTracker

_WORKER_LOG_RE = re.compile(r"^\[qwen-mm-privileged-probe\]\s+(\w+)\s+(.*)$")
_FIELD_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:\\.|[^"\\])*"|\S+)')
_COUNTERS = ("completed", "records", "bytes", "accepted", "rejected", "skipped", "errors")
_STATUS_KEYS = ("accepted", "rejected", "skipped", "errors")
_INTERRUPT_GRACE_SECONDS = 30.0
_TERMINATE_GRACE_SECONDS = 5.0


@dataclass
class _WorkerProgress:
    values: dict[str, int]

    @classmethod
    def create(cls) -> _WorkerProgress:
        return cls({key: 0 for key in _COUNTERS})


@dataclass
class _WorkerHandle:
    shard_index: int
    gpu_id: str
    assigned_items: int
    worker_dir: Path
    log_path: Path
    process: Any
    reader: threading.Thread
    progress: _WorkerProgress


def launch_gpu_workers(
    argv: Sequence[str],
    *,
    gpu_ids: Sequence[str],
    num_nodes: int,
    node_rank: int,
    output_dir: Path,
    dataset_root: Path,
    limit: int | None,
    require_table: bool,
    heartbeat_seconds: float,
) -> dict[str, Any]:
    """Launch one independent inference subprocess per nonempty local GPU shard.

    Each worker sees only its assigned GPU via ``CUDA_VISIBLE_DEVICES`` and
    writes sample artifacts into the shared flat ``samples`` tree. Node runs do
    not coordinate or wait for other machines.
    """

    normalized_gpu_ids = [str(gpu).strip() for gpu in gpu_ids]
    if not normalized_gpu_ids or any(not gpu for gpu in normalized_gpu_ids):
        raise ValueError("gpu_ids must contain at least one nonempty identifier")
    if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
        raise ValueError("gpu_ids must be distinct")
    if num_nodes < 1 or not 0 <= node_rank < num_nodes:
        raise ValueError("node_rank must be in [0, num_nodes)")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")

    # Import lazily: privileged_probe imports this launcher from main().
    from .privileged_probe import (
        _partition_samples,
        load_release_samples,
        rebuild_privileged_report,
    )

    all_samples = load_release_samples(
        dataset_root,
        limit=limit,
        require_table=require_table,
    )
    node_samples = _partition_samples(
        all_samples,
        num_nodes=num_nodes,
        node_rank=node_rank,
        num_shards=1,
        shard_index=0,
    )
    shard_samples = [
        _partition_samples(
            all_samples,
            num_nodes=num_nodes,
            node_rank=node_rank,
            num_shards=len(normalized_gpu_ids),
            shard_index=index,
        )
        for index in range(len(normalized_gpu_ids))
    ]

    output_root = Path(output_dir).expanduser().resolve()
    node_dir = output_root / "workers" / f"node_{node_rank:03d}"
    node_dir.mkdir(parents=True, exist_ok=True)
    node_bytes = sum(sample.image_path.stat().st_size for sample in node_samples)
    tracker = ProgressTracker(
        task="qwen-mm-privileged-probe-multi-gpu",
        total_items=len(node_samples),
        total_bytes=node_bytes,
        shard=(
            f"node={node_rank}/{num_nodes}/gpus={','.join(normalized_gpu_ids)}"
            f"/global-items={len(all_samples)}"
        ),
        heartbeat_seconds=heartbeat_seconds,
    )
    tracker.start()

    child_argv = _strip_launcher_options(argv)
    event_queue: queue.Queue[tuple[int, str | None, str | None]] = queue.Queue()
    handles: list[_WorkerHandle] = []
    worker_rows: list[dict[str, Any]] = []
    launch_errors = 0
    interrupted = False
    report_error: str | None = None
    report_summary: dict[str, Any] | None = None
    started_at = time.monotonic()

    for shard_index, (gpu_id, samples_for_gpu) in enumerate(
        zip(normalized_gpu_ids, shard_samples)
    ):
        if not samples_for_gpu:
            continue
        worker_dir = node_dir / f"shard_{shard_index:03d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        log_path = worker_dir / "worker.log"
        command = [
            sys.executable,
            "-u",
            "-m",
            "qwen_mm_token_probe.privileged_probe",
            *child_argv,
            "--num-nodes",
            str(num_nodes),
            "--node-rank",
            str(node_rank),
            "--num-shards",
            str(len(normalized_gpu_ids)),
            "--shard-index",
            str(shard_index),
            "--worker-mode",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                start_new_session=(os.name == "posix"),
            )
        except KeyboardInterrupt:
            interrupted = True
            _stop_children([item.process for item in handles])
            break
        except OSError as exc:
            launch_errors += 1
            message = f"{type(exc).__name__}: {exc}"
            log_path.write_text(f"[launcher-error] {message}\n", encoding="utf-8")
            worker_rows.append(
                {
                    "shard_index": shard_index,
                    "gpu_id": gpu_id,
                    "assigned_items": len(samples_for_gpu),
                    "status": "launch_failed",
                    "exit_code": None,
                    "error": message,
                    "log_path": str(log_path),
                }
            )
            tracker.note_error(
                phase="worker-launch-error",
                name=f"GPU {gpu_id}: {message}",
            )
            continue

        if process.stdout is None:
            process.kill()
            process.wait()
            launch_errors += 1
            message = "worker stdout pipe was not created"
            log_path.write_text(f"[launcher-error] {message}\n", encoding="utf-8")
            worker_rows.append(
                {
                    "shard_index": shard_index,
                    "gpu_id": gpu_id,
                    "assigned_items": len(samples_for_gpu),
                    "status": "launch_failed",
                    "exit_code": None,
                    "error": message,
                    "log_path": str(log_path),
                }
            )
            tracker.note_error(
                phase="worker-launch-error",
                name=f"GPU {gpu_id}: {message}",
            )
            continue

        reader = threading.Thread(
            target=_read_worker_output,
            args=(shard_index, process.stdout, log_path, event_queue),
            name=f"probe-gpu-{gpu_id}-log-reader",
            daemon=True,
        )
        handle = _WorkerHandle(
            shard_index=shard_index,
            gpu_id=gpu_id,
            assigned_items=len(samples_for_gpu),
            worker_dir=worker_dir,
            log_path=log_path,
            process=process,
            reader=reader,
            progress=_WorkerProgress.create(),
        )
        handles.append(handle)
        reader.start()
        worker_rows.append(
            {
                "shard_index": shard_index,
                "gpu_id": gpu_id,
                "assigned_items": len(samples_for_gpu),
                "status": "running",
                "exit_code": None,
                "worker_dir": str(worker_dir),
                "log_path": str(log_path),
            }
        )

    finished_readers: set[int] = set()
    try:
        while len(finished_readers) < len(handles):
            try:
                shard_index, line, reader_error = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                finished_readers.add(shard_index)
                if reader_error:
                    tracker.note_error(
                        phase="worker-log-reader-error",
                        name=f"shard {shard_index}: {reader_error}",
                    )
                continue
            handle = next(item for item in handles if item.shard_index == shard_index)
            _consume_worker_line(
                line,
                handle=handle,
                tracker=tracker,
                node_rank=node_rank,
                num_nodes=num_nodes,
            )
    except KeyboardInterrupt:
        interrupted = True
        _stop_children([handle.process for handle in handles])
        while len(finished_readers) < len(handles):
            try:
                shard_index, line, reader_error = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                finished_readers.add(shard_index)
                if reader_error:
                    tracker.note_error(
                        phase="worker-log-reader-error",
                        name=f"shard {shard_index}: {reader_error}",
                    )
                continue
            handle = next(item for item in handles if item.shard_index == shard_index)
            _consume_worker_line(
                line,
                handle=handle,
                tracker=tracker,
                node_rank=node_rank,
                num_nodes=num_nodes,
            )

    # Readers see EOF after process exit; collect each status and inspect the
    # worker summary as the worker may report sample failures despite exit 0.
    failed_workers = launch_errors
    for handle in handles:
        handle.reader.join()
        exit_code = handle.process.wait()
        worker_summary_path = handle.worker_dir / "run_summary.json"
        worker_summary: dict[str, Any] | None = None
        summary_error: str | None = None
        if worker_summary_path.is_file():
            try:
                worker_summary = json.loads(worker_summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                summary_error = f"{type(exc).__name__}: {exc}"
        failed_items = int(worker_summary.get("failed_items", 0)) if worker_summary else None
        summary_complete = bool(
            worker_summary is not None
            and int(worker_summary.get("total_items", -1)) == handle.assigned_items
            and int(worker_summary.get("completed_items", -1))
            + int(worker_summary.get("failed_items", -1))
            == handle.assigned_items
        )
        worker_failed = exit_code != 0 or failed_items is None or failed_items > 0 or not summary_complete
        if worker_failed:
            failed_workers += 1
            if exit_code != 0:
                tracker.note_error(
                    phase="worker-exit-error",
                    name=f"GPU {handle.gpu_id} shard {handle.shard_index} exit={exit_code}",
                )
        for row in worker_rows:
            if row.get("shard_index") == handle.shard_index:
                row.update(
                    {
                        "status": "failed" if worker_failed else "succeeded",
                        "exit_code": exit_code,
                        "failed_items": failed_items,
                        "summary_path": str(worker_summary_path),
                    }
                )
                if summary_error:
                    row["summary_error"] = summary_error
                break

    # Node 0 owns the shared root config. This is deliberately a one-writer
    # rule for two-machine runs and keeps per-worker node/shard fields out of
    # the aggregate config. Other nodes never race to replace it.
    root_config: dict[str, Any] | None = None
    if not interrupted and node_rank == 0:
        worker_config = next(
            (
                handle.worker_dir / "config.json"
                for handle in handles
                if (handle.worker_dir / "config.json").is_file()
            ),
            None,
        )
        if worker_config is not None:
            try:
                root_config = _build_root_config(
                    worker_config_path=worker_config,
                    output_root=output_root,
                    selected_count=len(all_samples),
                    num_nodes=num_nodes,
                )
            except Exception as exc:  # noqa: BLE001 - preserve completed worker artifacts
                report_error = f"{type(exc).__name__}: {exc}"
                tracker.note_error(phase="aggregate-config-error", name=report_error)
                failed_workers += 1

    if not interrupted and num_nodes == 1 and handles:
        tracker.set_current(
            index=len(node_samples),
            name="aggregate-report",
            phase="building-report",
        )
        try:
            if root_config is None:
                root_config = _build_root_config(
                    worker_config_path=handles[0].worker_dir / "config.json",
                    output_root=output_root,
                    selected_count=len(all_samples),
                    num_nodes=1,
                )
            report_summary = rebuild_privileged_report(
                output_root,
                teacher_signal_threshold=root_config.get("teacher_signal_threshold"),
                student_response_min_probability=root_config.get(
                    "student_response_min_probability"
                ),
                student_response_max_probability=root_config.get(
                    "student_response_max_probability"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - report failure must fail the launcher
            report_error = f"{type(exc).__name__}: {exc}"
            failed_workers += 1
            tracker.note_error(phase="aggregate-report-error", name=report_error)

    tracker_summary = tracker.finish(interrupted=interrupted)
    elapsed = time.monotonic() - started_at
    status = "interrupted" if interrupted else ("failed" if failed_workers else "success")
    next_step = None
    if num_nodes > 1:
        next_step = (
            "After both node ranks finish, run the existing --rebuild-report-only command "
            "against this output directory."
        )
    summary = asdict(tracker_summary) | {
        "status": status,
        "node_rank": node_rank,
        "num_nodes": num_nodes,
        "gpu_ids": normalized_gpu_ids,
        "assigned_items": len(node_samples),
        "workers_started": len(handles),
        "workers_succeeded": sum(row.get("status") == "succeeded" for row in worker_rows),
        "failed_workers": failed_workers,
        "interrupted": interrupted,
        "elapsed_seconds": elapsed,
        "workers": worker_rows,
        "report_summary": report_summary,
        "report_error": report_error,
        "next_step": next_step,
        "node_summary_path": str(node_dir / "node_summary.json"),
    }
    _write_json_atomic(node_dir / "node_summary.json", summary)
    if next_step:
        print(f"[multi-gpu] {next_step}", flush=True)
    return summary


def _split_node_samples(samples: Sequence[Any], num_shards: int) -> list[list[Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    return [list(samples[index::num_shards]) for index in range(num_shards)]


def _strip_launcher_options(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    index = 0
    value_options = {"--gpus", "--num-nodes", "--node-rank"}
    equals_options = tuple(f"{option}=" for option in value_options)
    while index < len(argv):
        value = str(argv[index])
        if value in value_options:
            if index + 1 >= len(argv):
                raise ValueError(f"{value} requires a value")
            index += 2
            continue
        if value.startswith(equals_options):
            index += 1
            continue
        result.append(value)
        index += 1
    return result


def _read_worker_output(
    shard_index: int,
    stream: Any,
    log_path: Path,
    events: queue.Queue[tuple[int, str | None, str | None]],
) -> None:
    reader_error: str | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            for line in stream:
                log_handle.write(line)
                log_handle.flush()
                events.put((shard_index, line, None))
    except OSError as exc:  # keep draining child output if log storage fails
        reader_error = f"{type(exc).__name__}: {exc}"
        try:
            for _ in stream:
                pass
        except OSError as drain_error:
            reader_error += f"; stream drain failed: {type(drain_error).__name__}: {drain_error}"
    finally:
        events.put((shard_index, None, reader_error))


def _parse_worker_progress(line: str) -> tuple[str, dict[str, str]] | None:
    match = _WORKER_LOG_RE.match(line.strip())
    if match is None:
        return None
    event, tail = match.groups()
    fields: dict[str, str] = {}
    for key, encoded in _FIELD_RE.findall(tail):
        if encoded.startswith('"'):
            try:
                fields[key] = str(json.loads(encoded))
            except json.JSONDecodeError:
                fields[key] = encoded[1:-1]
        else:
            fields[key] = encoded
    return event, fields


def _leading_count(value: str | None, *, slash_field: bool = False) -> int | None:
    if value is None:
        return None
    candidate = value.split("/", 1)[0] if slash_field else value
    try:
        return int(candidate)
    except ValueError:
        return None


def _consume_worker_line(
    line: str,
    *,
    handle: _WorkerHandle,
    tracker: ProgressTracker,
    node_rank: int,
    num_nodes: int,
) -> None:
    parsed = _parse_worker_progress(line)
    if parsed is None:
        return
    event, fields = parsed
    phase = fields.get("phase", event.lower())
    current = fields.get("current", "-")
    current_name = f"node {node_rank}/{num_nodes} GPU {handle.gpu_id}: {current}"
    next_index = min(tracker.completed_items + 1, tracker.total_items)
    if tracker.total_items == 0:
        next_index = 0
    tracker.set_current(
        index=next_index,
        name=current_name,
        phase=f"worker-{phase}",
    )

    new_values = dict(handle.progress.values)
    for key in _COUNTERS:
        count = _leading_count(fields.get(key), slash_field=(key in {"completed", "bytes"}))
        if count is not None:
            new_values[key] = count

    delta_completed = max(0, new_values["completed"] - handle.progress.values["completed"])
    delta_records = max(0, new_values["records"] - handle.progress.values["records"])
    delta_bytes = max(0, new_values["bytes"] - handle.progress.values["bytes"])
    delta_status = {
        key: max(0, new_values[key] - handle.progress.values[key])
        for key in _STATUS_KEYS
    }

    if delta_completed:
        status_total = sum(delta_status.values())
        if status_total < delta_completed:
            suffix = phase.removeprefix("unit-")
            fallback = (
                "errors"
                if suffix == "error"
                else (suffix if suffix in _STATUS_KEYS else "accepted")
            )
            delta_status[fallback] += delta_completed - status_total
        elif status_total > delta_completed:
            excess = status_total - delta_completed
            for key in reversed(_STATUS_KEYS):
                reduce = min(excess, delta_status[key])
                delta_status[key] -= reduce
                excess -= reduce
                if not excess:
                    break

        ordered_statuses = [
            "error" if status == "errors" else status
            for status in _STATUS_KEYS
            for _ in range(delta_status[status])
        ][:delta_completed]
        if len(ordered_statuses) < delta_completed:
            ordered_statuses.extend(["accepted"] * (delta_completed - len(ordered_statuses)))
        bytes_per_item, extra_bytes = divmod(delta_bytes, max(1, delta_completed))
        tracker_completed = tracker.completed_items
        for offset, status in enumerate(ordered_statuses):
            bytes_count = bytes_per_item + (1 if offset < extra_bytes else 0)
            tracker.complete_unit(
                status=status,
                records=delta_records if offset == 0 else 0,
                bytes_count=bytes_count,
                index=min(tracker_completed + offset + 1, tracker.total_items),
                name=current_name,
            )
    elif event == "ERROR":
        delta_errors = delta_status["errors"]
        for _ in range(delta_errors):
            tracker.note_error(phase=f"worker-{phase}", name=current_name)

    handle.progress.values = new_values


def _stop_children(processes: Sequence[Any]) -> None:
    running = [process for process in processes if process.poll() is None]
    for process in running:
        try:
            process.send_signal(signal.SIGINT)
        except (OSError, ProcessLookupError):
            pass

    graceful_deadline = time.monotonic() + _INTERRUPT_GRACE_SECONDS
    still_running: list[Any] = []
    for process in running:
        remaining = max(0.0, graceful_deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            still_running.append(process)

    terminating = list(still_running)
    for process in terminating:
        _signal_process_group(process, signal.SIGTERM)
    terminate_deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    still_running = []
    for process in terminating:
        remaining = max(0.0, terminate_deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            still_running.append(process)

    for process in still_running:
        _signal_process_group(process, signal.SIGKILL)
    for process in still_running:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def _signal_process_group(process: Any, sig: int) -> None:
    if os.name == "posix" and getattr(process, "pid", None) is not None:
        try:
            os.killpg(process.pid, sig)
            return
        except (OSError, ProcessLookupError):
            pass
    try:
        process.send_signal(sig)
    except (OSError, ProcessLookupError):
        pass


def _build_root_config(
    *, worker_config_path: Path, output_root: Path, selected_count: int, num_nodes: int
) -> dict[str, Any]:
    config = json.loads(worker_config_path.read_text(encoding="utf-8"))
    config.pop("node_rank", None)
    config.pop("shard_index", None)
    config.pop("worker_mode", None)
    # GPU counts may differ by node, so a single common config must not claim
    # one node's local shard count as a global value.
    config.pop("num_shards", None)
    config.pop("samples_selected", None)
    config["samples_selected"] = selected_count
    config["samples_total"] = selected_count
    config["num_nodes"] = num_nodes
    _write_json_atomic(output_root / "config.json", config)
    return config


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
