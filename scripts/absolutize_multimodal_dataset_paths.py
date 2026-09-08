#!/usr/bin/env python3
"""Rewrite relative image paths in SFT or VERL JSONL files as absolute paths.

Only the top-level ``images`` field is changed.  Relative paths are resolved
from the parent directory of each input JSONL file; absolute paths and URLs are
preserved verbatim.  Output is streamed through a resumable partial file and
atomically promoted when conversion completes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass
class Progress:
    processed_rows: int = 0
    processed_bytes: int = 0
    rewritten_paths: int = 0
    absolute_paths: int = 0
    remote_paths: int = 0
    records_without_images: int = 0
    sft_rows: int = 0
    verl_rows: int = 0
    other_rows: int = 0


def _emit(stage: str, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [{stage}] {message}", flush=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_remote(value: str) -> bool:
    if value.startswith("data:"):
        return True
    return urlparse(value).scheme.lower() in {
        "http",
        "https",
        "s3",
        "oss",
        "gs",
        "hdfs",
    }


def _absolute_local_path(value: str, *, base_dir: Path) -> str:
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        return value
    return os.path.abspath(base_dir / expanded)


def _rewrite_image_item(
    item: Any,
    *,
    base_dir: Path,
    check_exists: bool,
    progress: Progress,
) -> Any:
    if isinstance(item, str):
        if not item:
            raise ValueError("image path is empty")
        if _is_remote(item):
            progress.remote_paths += 1
            return item
        if Path(item).expanduser().is_absolute():
            progress.absolute_paths += 1
            rewritten = item
        else:
            rewritten = _absolute_local_path(item, base_dir=base_dir)
            progress.rewritten_paths += 1
        if check_exists and not Path(rewritten).is_file():
            raise ValueError(f"image file does not exist: {rewritten}")
        return rewritten

    if isinstance(item, dict):
        rewritten = dict(item)
        if isinstance(rewritten.get("path"), str):
            rewritten["path"] = _rewrite_image_item(
                rewritten["path"],
                base_dir=base_dir,
                check_exists=check_exists,
                progress=progress,
            )
            return rewritten
        if isinstance(rewritten.get("url"), str):
            progress.remote_paths += 1
            return rewritten
        raise ValueError("image object must contain a string path or url")

    raise ValueError(
        f"image entry must be a string or object, got {type(item).__name__}"
    )


def _rewrite_record(
    record: dict[str, Any],
    *,
    base_dir: Path,
    check_exists: bool,
    progress: Progress,
) -> dict[str, Any]:
    if "messages" in record:
        progress.sft_rows += 1
    elif "prompt" in record and "reward_model" in record:
        progress.verl_rows += 1
    else:
        progress.other_rows += 1

    if "images" not in record or record["images"] is None:
        progress.records_without_images += 1
        return record

    images = record["images"]
    if isinstance(images, (str, dict)):
        record["images"] = _rewrite_image_item(
            images,
            base_dir=base_dir,
            check_exists=check_exists,
            progress=progress,
        )
        return record
    if not isinstance(images, list):
        raise TypeError("images must be a path, object, or list")
    record["images"] = [
        _rewrite_image_item(
            item,
            base_dir=base_dir,
            check_exists=check_exists,
            progress=progress,
        )
        for item in images
    ]
    return record


def _state_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.progress.json")


def _partial_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.partial")


def _destination_for(
    source: Path,
    *,
    output_dir: Path | None,
    output_suffix: str,
) -> Path:
    parent = source.parent if output_dir is None else output_dir
    suffix = source.suffix or ".jsonl"
    stem = source.name[: -len(source.suffix)] if source.suffix else source.name
    return parent / f"{stem}{output_suffix}{suffix}"


def _load_resume_state(
    source: Path,
    destination: Path,
    *,
    resume: bool,
) -> tuple[Progress, int]:
    state_path = _state_path(destination)
    partial_path = _partial_path(destination)
    if not resume:
        partial_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return Progress(), 0
    if not partial_path.is_file() or not state_path.is_file():
        partial_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return Progress(), 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stat = source.stat()
    if (
        state.get("input") != str(source)
        or state.get("input_bytes") != stat.st_size
        or state.get("input_mtime_ns") != stat.st_mtime_ns
    ):
        raise ValueError(
            f"resume state does not match current input: {state_path}; "
            "rerun without --resume to restart"
        )
    payload = state.get("progress")
    if not isinstance(payload, dict):
        raise TypeError(f"invalid resume state: {state_path}")
    progress = Progress(
        **{
            name: int(payload.get(name, 0))
            for name in Progress.__dataclass_fields__
        }
    )
    partial_output_bytes = int(state.get("partial_output_bytes", -1))
    if partial_output_bytes < 0 or partial_output_bytes > partial_path.stat().st_size:
        raise ValueError(f"invalid partial output size in resume state: {state_path}")
    return progress, partial_output_bytes


def _save_state(
    source: Path,
    destination: Path,
    progress: Progress,
    *,
    partial_output_bytes: int,
) -> None:
    stat = source.stat()
    _atomic_json(
        _state_path(destination),
        {
            "input": str(source),
            "input_bytes": stat.st_size,
            "input_mtime_ns": stat.st_mtime_ns,
            "output": str(destination),
            "partial_output_bytes": partial_output_bytes,
            "progress": asdict(progress),
        },
    )


def convert_file(
    source: Path,
    destination: Path,
    *,
    check_exists: bool,
    overwrite: bool,
    resume: bool,
    progress_every: int,
    file_index: int,
    total_files: int,
    global_started: float,
    global_bytes_before: int,
    global_total_bytes: int,
) -> Progress:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input does not exist: {source}")
    if destination == source:
        raise ValueError("input and output paths must be different")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {destination}; use --overwrite to replace it"
        )
    if destination.exists() and overwrite:
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)

    progress, partial_output_bytes = _load_resume_state(
        source,
        destination,
        resume=resume,
    )
    partial_path = _partial_path(destination)
    source_bytes = source.stat().st_size
    if progress.processed_bytes:
        with partial_path.open("r+b") as partial:
            partial.truncate(partial_output_bytes)
    mode = "ab" if progress.processed_bytes else "wb"
    started = time.monotonic()
    last_log = started
    _emit(
        "file-start",
        f"file={file_index}/{total_files} input={source} output={destination} "
        f"bytes={source_bytes} resumed_rows={progress.processed_rows} "
        f"resumed_bytes={progress.processed_bytes}",
    )

    with source.open("rb") as reader, partial_path.open(mode) as writer:
        if progress.processed_bytes:
            reader.seek(progress.processed_bytes)
        for raw_line in reader:
            line_number = progress.processed_rows + 1
            if not raw_line.strip():
                raise ValueError(f"{source}:{line_number}: blank JSONL row")
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(f"{source}:{line_number}: row is not an object")
            try:
                converted = _rewrite_record(
                    record,
                    base_dir=source.parent,
                    check_exists=check_exists,
                    progress=progress,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            writer.write(
                (json.dumps(converted, ensure_ascii=False) + "\n").encode("utf-8")
            )
            progress.processed_rows += 1
            progress.processed_bytes += len(raw_line)

            now = time.monotonic()
            if (
                progress.processed_rows % progress_every == 0
                or now - last_log >= 30.0
            ):
                writer.flush()
                _save_state(
                    source,
                    destination,
                    progress,
                    partial_output_bytes=writer.tell(),
                )
                global_done = global_bytes_before + progress.processed_bytes
                elapsed = max(now - global_started, 1e-9)
                rate = global_done / elapsed
                eta = (global_total_bytes - global_done) / max(rate, 1e-9)
                _emit(
                    "progress",
                    f"file={file_index}/{total_files} rows={progress.processed_rows} "
                    f"file_bytes={progress.processed_bytes}/{source_bytes} "
                    f"global_bytes={global_done}/{global_total_bytes} "
                    f"percent={global_done / max(1, global_total_bytes):.2%} "
                    f"throughput={rate / (1024 * 1024):.2f}_MiB/s "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
                    f"rewritten={progress.rewritten_paths} current_row={line_number}",
                )
                last_log = now
        writer.flush()
        os.fsync(writer.fileno())

    if progress.processed_bytes != source_bytes:
        raise ValueError(
            f"processed byte count mismatch: {progress.processed_bytes}!={source_bytes}"
        )
    partial_path.replace(destination)
    _state_path(destination).unlink(missing_ok=True)
    _emit(
        "file-finish",
        f"file={file_index}/{total_files} rows={progress.processed_rows} "
        f"rewritten={progress.rewritten_paths} absolute={progress.absolute_paths} "
        f"remote={progress.remote_paths} no_images={progress.records_without_images} "
        f"sft={progress.sft_rows} verl={progress.verl_rows} other={progress.other_rows} "
        f"elapsed={time.monotonic() - started:.1f}s output={destination}",
    )
    return progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
        help=(
            "One or more SFT/VERL JSONL files. The option may be provided "
            "once with multiple paths or repeated for each path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to each input file's directory.",
    )
    parser.add_argument(
        "--output-suffix",
        default=".absolute",
        help="Filename suffix inserted before .jsonl (default: .absolute).",
    )
    parser.add_argument(
        "--check-exists",
        action="store_true",
        help="Check every local image path after conversion (slower on SFS).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching partial conversion after interruption.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")
    if not args.output_suffix:
        raise SystemExit("--output-suffix must not be empty")

    sources = [path.expanduser().resolve() for path in args.input]
    if len(set(sources)) != len(sources):
        raise SystemExit("--input contains duplicate paths")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else None
    )
    destinations = [
        _destination_for(
            source,
            output_dir=output_dir,
            output_suffix=args.output_suffix,
        ).resolve()
        for source in sources
    ]
    if len(set(destinations)) != len(destinations):
        raise SystemExit("multiple inputs map to the same output filename")
    for source in sources:
        if not source.is_file():
            raise SystemExit(f"input does not exist: {source}")

    total_bytes = sum(source.stat().st_size for source in sources)
    started = time.monotonic()
    _emit(
        "start",
        f"files={len(sources)} total_bytes={total_bytes} "
        f"check_exists={args.check_exists} resume={args.resume} "
        f"output_dir={output_dir or 'same_as_input'}",
    )
    totals = Progress()
    completed_bytes = 0
    for index, (source, destination) in enumerate(
        zip(sources, destinations, strict=True),
        start=1,
    ):
        result = convert_file(
            source,
            destination,
            check_exists=args.check_exists,
            overwrite=args.overwrite,
            resume=args.resume,
            progress_every=args.progress_every,
            file_index=index,
            total_files=len(sources),
            global_started=started,
            global_bytes_before=completed_bytes,
            global_total_bytes=total_bytes,
        )
        completed_bytes += source.stat().st_size
        for name in Progress.__dataclass_fields__:
            setattr(totals, name, getattr(totals, name) + getattr(result, name))

    elapsed = time.monotonic() - started
    _emit(
        "finish",
        f"files={len(sources)}/{len(sources)} rows={totals.processed_rows} "
        f"bytes={completed_bytes}/{total_bytes} rewritten={totals.rewritten_paths} "
        f"absolute={totals.absolute_paths} remote={totals.remote_paths} "
        f"no_images={totals.records_without_images} sft={totals.sft_rows} "
        f"verl={totals.verl_rows} other={totals.other_rows} "
        f"elapsed={elapsed:.1f}s",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
