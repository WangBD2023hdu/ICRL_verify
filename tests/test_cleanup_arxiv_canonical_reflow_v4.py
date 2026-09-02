from __future__ import annotations

from pathlib import Path

from scripts.cleanup_arxiv_canonical_reflow_v4 import discover_tasks, main


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_dry_run_and_parallel_execute_preserve_edited_training_data(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "safe" / "v4_run"
    clean = run_root / "pages" / "paper_dense_0001"
    edited = run_root / "pages" / "paper_dense_0002_confusable_s7_deadbeef"
    _write(clean / "build" / "page.pdf", b"clean-pdf")
    _write(clean / "page.png", b"clean-image")
    _write(edited / "build" / "page.pdf", b"edited-pdf")
    _write(edited / "source" / "page.tex", b"edited-tex")
    _write(edited / "compile.log", b"log")
    _write(edited / "page.png", b"edited-image")
    _write(edited / "ground_truth.md", b"edited-gt")
    _write(edited / "terminal_result.json", b"{}")
    training = run_root / "realtime_training" / "verl.jsonl"
    _write(training, b"{}\n")

    tasks = discover_tasks(run_root)
    assert sum(task.category == "clean_page" for task in tasks) == 1
    assert sum(task.category == "edited_compile_artifact" for task in tasks) == 3

    assert main(["--run-root", str(run_root), "--workers", "2"]) == 0
    assert clean.is_dir()
    assert (edited / "build" / "page.pdf").is_file()

    assert (
        main(
            [
                "--run-root",
                str(run_root),
                "--workers",
                "2",
                "--execute",
            ]
        )
        == 0
    )
    assert not clean.exists()
    assert not (edited / "build").exists()
    assert not (edited / "source").exists()
    assert not (edited / "compile.log").exists()
    assert (edited / "page.png").read_bytes() == b"edited-image"
    assert (edited / "ground_truth.md").read_bytes() == b"edited-gt"
    assert (edited / "terminal_result.json").is_file()
    assert training.read_bytes() == b"{}\n"
