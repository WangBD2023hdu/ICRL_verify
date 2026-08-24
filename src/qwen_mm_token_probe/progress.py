from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressSummary:
    task: str
    completed_items: int
    total_items: int
    processed_records: int
    processed_bytes: int
    accepted: int
    rejected: int
    skipped: int
    errors: int
    elapsed_seconds: float
    interrupted: bool


class ProgressTracker:
    """Thread-safe progress logging with a periodic parent-process heartbeat."""

    def __init__(
        self,
        *,
        task: str,
        total_items: int,
        total_bytes: int = 0,
        shard: str = "main",
        heartbeat_seconds: float = 30.0,
    ) -> None:
        if total_items < 0:
            raise ValueError("total_items must be non-negative")
        if total_bytes < 0:
            raise ValueError("total_bytes must be non-negative")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")

        self.task = task
        self.total_items = total_items
        self.total_bytes = total_bytes
        self.shard = shard
        self.heartbeat_seconds = heartbeat_seconds
        self.completed_items = 0
        self.processed_records = 0
        self.processed_bytes = 0
        self.accepted = 0
        self.rejected = 0
        self.skipped = 0
        self.errors = 0
        self.current_index = 0
        self.current_name = "-"
        self.phase = "initializing"
        self._started_at = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._finished = False

    def start(self) -> None:
        with self._lock:
            if self._started_at:
                return
            self._started_at = time.monotonic()
            self._print_locked("START")
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"{self.task}-progress",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def set_current(
        self,
        *,
        index: int,
        name: str,
        phase: str,
    ) -> None:
        with self._lock:
            self.current_index = index
            self.current_name = name
            self.phase = phase

    def complete_unit(
        self,
        *,
        status: str,
        records: int = 1,
        bytes_count: int = 0,
        index: int | None = None,
        name: str | None = None,
    ) -> None:
        if status not in {"accepted", "rejected", "skipped", "error"}:
            raise ValueError(f"unsupported progress status: {status}")
        if records < 0 or bytes_count < 0:
            raise ValueError("records and bytes_count must be non-negative")

        with self._lock:
            self.completed_items += 1
            self.processed_records += records
            self.processed_bytes += bytes_count
            if index is not None:
                self.current_index = index
            if name is not None:
                self.current_name = name
            if status == "accepted":
                self.accepted += 1
            elif status == "rejected":
                self.rejected += 1
            elif status == "skipped":
                self.skipped += 1
            else:
                self.errors += 1
            self.phase = f"unit-{status}"
            self._print_locked("CHECKPOINT")

    def note_error(self, *, phase: str, name: str) -> None:
        with self._lock:
            self.errors += 1
            self.phase = phase
            self.current_name = name
            self._print_locked("ERROR")

    def finish(self, *, interrupted: bool = False) -> ProgressSummary:
        with self._lock:
            if self._finished:
                return self._summary_locked(interrupted=interrupted)
            self._finished = True
            self.phase = "interrupted" if interrupted else "finished"
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=min(2.0, self.heartbeat_seconds))
        with self._lock:
            self._print_locked("FINAL", interrupted=interrupted)
            return self._summary_locked(interrupted=interrupted)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_seconds):
            with self._lock:
                if self._finished:
                    return
                self._print_locked("PROGRESS")

    def _summary_locked(self, *, interrupted: bool) -> ProgressSummary:
        return ProgressSummary(
            task=self.task,
            completed_items=self.completed_items,
            total_items=self.total_items,
            processed_records=self.processed_records,
            processed_bytes=self.processed_bytes,
            accepted=self.accepted,
            rejected=self.rejected,
            skipped=self.skipped,
            errors=self.errors,
            elapsed_seconds=self._elapsed_locked(),
            interrupted=interrupted,
        )

    def _print_locked(self, event: str, *, interrupted: bool = False) -> None:
        elapsed = self._elapsed_locked()
        completed = self.completed_items
        total = self.total_items
        percentage = 100.0 if total == 0 else 100.0 * completed / total
        item_rate = completed / elapsed if elapsed > 0 else 0.0
        byte_rate = self.processed_bytes / elapsed if elapsed > 0 else 0.0
        eta = _eta_seconds(completed=completed, total=total, elapsed=elapsed)
        total_bytes = str(self.total_bytes) if self.total_bytes else "unknown"
        current_unit = f"{self.current_index}/{total}" if total else "0/0"
        print(
            f"[{self.task}] {event} "
            f"phase={_log_value(self.phase)} shard={_log_value(self.shard)} "
            f"current_unit={current_unit} current={_log_value(self.current_name)} "
            f"completed={completed}/{total} percentage={percentage:.2f}% "
            f"records={self.processed_records} bytes={self.processed_bytes}/{total_bytes} "
            f"throughput_items_s={item_rate:.4f} throughput_bytes_s={byte_rate:.2f} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
            f"accepted={self.accepted} rejected={self.rejected} skipped={self.skipped} "
            f"errors={self.errors} interrupted={str(interrupted).lower()}",
            flush=True,
        )

    def _elapsed_locked(self) -> float:
        if not self._started_at:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)


def _eta_seconds(*, completed: int, total: int, elapsed: float) -> float:
    if completed <= 0 or completed >= total or elapsed <= 0:
        return 0.0 if completed >= total else math.inf
    return elapsed * (total - completed) / completed


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _log_value(value: str) -> str:
    compact = " ".join(str(value).split())
    if len(compact) > 180:
        compact = compact[:177] + "..."
    return json_quote(compact)


def json_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
