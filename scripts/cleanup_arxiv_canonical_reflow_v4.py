#!/usr/bin/env python3
"""Remove V4 clean pages and compiler debris while preserving edited data."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

_CONFUSABLE_MARKER = "_confusable_s"
_COMPILER_DIRECTORIES = frozenset({"build", "source"})
_COMPILER_FILENAMES = frozenset(
    {
        "compile.log",
        "pdfinfo.log",
        "pdftotext.log",
        "render.log",
        "pdf_text_reject_only.txt",
        "page.pdf",
        "page.tex",
    }
)
_COMPILER_SUFFIXES = frozenset(
    {
        ".aux",
        ".bbl",
        ".bcf",
        ".blg",
        ".dvi",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".run.xml",
        ".synctex",
        ".toc",
        ".xdv",
    }
)


@dataclass(frozen=True, slots=True)
class DeleteTask:
    path: str
    category: str


@dataclass(frozen=True, slots=True)
class DeleteResult:
    path: str
    category: str
    status: str
    files: int
    directories: int
    bytes: int
    error: str | None = None


def _emit(stage: str, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [{stage}] {message}", flush=True)


def _compiler_artifact(path: Path) -> bool:
    if path.name in _COMPILER_DIRECTORIES or path.name in _COMPILER_FILENAMES:
        return True
    name = path.name
    return any(name.endswith(suffix) for suffix in _COMPILER_SUFFIXES)


def discover_tasks(run_root: Path) -> tuple[DeleteTask, ...]:
    pages_root = run_root / "pages"
    if not pages_root.is_dir():
        raise ValueError(f"run root has no pages directory: {pages_root}")

    tasks: list[DeleteTask] = []
    with os.scandir(pages_root) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            page_dir = Path(entry.path)
            if _CONFUSABLE_MARKER not in entry.name:
                tasks.append(DeleteTask(str(page_dir), "clean_page"))
                continue
            with os.scandir(page_dir) as children:
                for child in children:
                    child_path = Path(child.path)
                    if _compiler_artifact(child_path):
                        tasks.append(
                            DeleteTask(str(child_path), "edited_compile_artifact")
                        )
    return tuple(tasks)


def _unlink_tree(path: Path) -> tuple[int, int, int]:
    if not path.exists() and not path.is_symlink():
        return 0, 0, 0
    if path.is_symlink() or path.is_file():
        size = path.lstat().st_size
        path.unlink()
        return 1, 0, size

    files = 0
    directories = 0
    byte_count = 0
    for root_value, directory_names, file_names in os.walk(
        path, topdown=False, followlinks=False
    ):
        root = Path(root_value)
        for filename in file_names:
            candidate = root / filename
            try:
                byte_count += candidate.lstat().st_size
                candidate.unlink()
                files += 1
            except FileNotFoundError:
                continue
        for directory_name in directory_names:
            candidate = root / directory_name
            try:
                if candidate.is_symlink():
                    byte_count += candidate.lstat().st_size
                    candidate.unlink()
                    files += 1
                else:
                    candidate.rmdir()
                    directories += 1
            except FileNotFoundError:
                continue
    path.rmdir()
    directories += 1
    return files, directories, byte_count


def _delete(task: DeleteTask) -> DeleteResult:
    try:
        files, directories, byte_count = _unlink_tree(Path(task.path))
    except Exception as error:  # noqa: BLE001 - isolate individual deletion failures
        return DeleteResult(
            path=task.path,
            category=task.category,
            status="error",
            files=0,
            directories=0,
            bytes=0,
            error=f"{type(error).__name__}: {error}",
        )
    return DeleteResult(
        path=task.path,
        category=task.category,
        status="deleted",
        files=files,
        directories=directories,
        bytes=byte_count,
    )


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.2f}{unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _validate_run_root(run_root: Path) -> Path:
    resolved = run_root.expanduser().resolve()
    forbidden = {Path("/"), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in forbidden:
        raise ValueError(f"refusing broad cleanup target: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"run root does not exist: {resolved}")
    if len(resolved.parts) < 4:
        raise ValueError(f"refusing shallow cleanup target: {resolved}")
    return resolved


def _run_parallel(
    tasks: Sequence[DeleteTask],
    *,
    workers: int,
) -> tuple[int, int, int, int, int]:
    started = time.monotonic()
    completed = deleted = errors = files = directories = byte_count = 0
    iterator = iter(tasks)
    max_inflight = min(len(tasks), max(1, workers * 4))

    def submit(
        executor: ProcessPoolExecutor, pending: dict[object, DeleteTask]
    ) -> bool:
        try:
            task = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(_delete, task)] = task
        return True

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: dict[object, DeleteTask] = {}
        while len(pending) < max_inflight and submit(executor, pending):
            pass
        last_log = time.monotonic()
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                elapsed = time.monotonic() - started
                _emit(
                    "delete-progress",
                    f"completed={completed}/{len(tasks)} percent="
                    f"{100 * completed / max(1, len(tasks)):.2f}% pending={len(pending)} "
                    f"deleted={deleted} errors={errors} files={files} "
                    f"directories={directories} bytes={byte_count} "
                    f"throughput={completed / max(elapsed, 1e-9):.2f}_tasks/s "
                    f"elapsed={elapsed:.1f}s",
                )
                last_log = time.monotonic()
                continue
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001
                    result = DeleteResult(
                        path=task.path,
                        category=task.category,
                        status="error",
                        files=0,
                        directories=0,
                        bytes=0,
                        error=f"{type(error).__name__}: {error}",
                    )
                completed += 1
                if result.status == "deleted":
                    deleted += 1
                    files += result.files
                    directories += result.directories
                    byte_count += result.bytes
                else:
                    errors += 1
                    _emit(
                        "delete-error",
                        f"category={result.category} path={result.path} "
                        f"error={result.error}",
                    )
                submit(executor, pending)
            now = time.monotonic()
            if completed == len(tasks) or completed % 1000 == 0 or now - last_log >= 30:
                elapsed = now - started
                eta = (len(tasks) - completed) / max(
                    completed / max(elapsed, 1e-9), 1e-9
                )
                _emit(
                    "delete-unit",
                    f"completed={completed}/{len(tasks)} percent="
                    f"{100 * completed / max(1, len(tasks)):.2f}% "
                    f"current={result.path} deleted={deleted} errors={errors} "
                    f"files={files} directories={directories} bytes={byte_count} "
                    f"throughput={completed / max(elapsed, 1e-9):.2f}_tasks/s "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                )
                last_log = now
    return deleted, errors, files, directories, byte_count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform deletion. Without this flag the command is a dry run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.workers <= 256:
        raise SystemExit("--workers must be between 1 and 256")
    try:
        run_root = _validate_run_root(args.run_root)
        started = time.monotonic()
        _emit(
            "start",
            f"run_root={run_root} workers={args.workers} execute={args.execute}",
        )
        tasks = discover_tasks(run_root)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.category] = counts.get(task.category, 0) + 1
    _emit(
        "scan-finish",
        f"tasks={len(tasks)} categories={counts} elapsed="
        f"{time.monotonic() - started:.1f}s",
    )
    if not args.execute:
        _emit(
            "finish",
            "status=dry_run no_files_deleted=true rerun_with=--execute",
        )
        return 0
    if not tasks:
        _emit("finish", "status=passed tasks=0 deleted=0 errors=0")
        return 0

    deleted, errors, files, directories, byte_count = _run_parallel(
        tasks,
        workers=args.workers,
    )
    elapsed = time.monotonic() - started
    _emit(
        "finish",
        f"status={'passed' if errors == 0 else 'partial'} tasks={len(tasks)} "
        f"deleted={deleted} errors={errors} files={files} "
        f"directories={directories} bytes={byte_count} "
        f"freed={_human_bytes(byte_count)} elapsed={elapsed:.1f}s",
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
