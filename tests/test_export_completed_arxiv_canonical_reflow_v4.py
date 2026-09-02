from __future__ import annotations

import json
from pathlib import Path

from scripts import export_completed_arxiv_canonical_reflow_v4 as exporter

PIPELINE_VERSION = exporter.PIPELINE_VERSION


def _write_terminal(
    run_root: Path,
    page_id: str,
    *,
    variant: str = "confusable_edit",
    status: str = "accepted",
    with_image: bool = True,
    result_name: str = "terminal_result.json",
) -> Path:
    page_dir = run_root / "pages" / page_id
    page_dir.mkdir(parents=True)
    image = page_dir / "page.png"
    if with_image:
        image.write_bytes(b"compiled-page-image")
    markdown = "A mutated dcgument paragraph."
    (page_dir / "ground_truth.md").write_text(markdown + "\n", encoding="utf-8")
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "page_id": page_id,
        "paper_id": "2401.00001v1",
        "status": status,
        "reason": None if status == "accepted" else "fixture_rejection",
        "layout": "two_column",
        "has_table": True,
        "markdown": markdown,
        "verifier_recall": 1.0,
        "verifier_precision": 1.0,
        "image": str(image.resolve()),
        "block_ids": ["block-1"],
        "source_node_ids": ["node-1"],
        "content_fill_ratio": 0.82,
        "column_fill_ratios": [0.81, 0.83],
        "page_signature": "signature",
        "elapsed_seconds": 1.5,
        "rescued": False,
        "pack_attempts": 1,
        "mutation_count": 1,
        "changes": [
            {
                "ocr_ans": "dcgument",
                "origin_ans": "document",
                "bbox": [10, 12, 60, 30],
                "from_char": "o",
                "to_char": "g",
                "block_id": "block-1",
            }
        ],
        "clean_page_id": "clean-page-1",
        "max_mutation_vertical_shift_points": 0.0,
        "variant": variant,
    }
    result_path = page_dir / result_name
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exports_only_complete_confusable_pages_in_both_training_formats(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    edited_id = "edited-page_confusable_s1_hash"
    _write_terminal(run_root, edited_id, result_name="result.json")
    _write_terminal(run_root, "clean-page", variant="clean")
    _write_terminal(
        run_root,
        "missing-image_confusable_s1_hash",
        with_image=False,
    )
    _write_terminal(
        run_root,
        "rejected-page_confusable_s1_hash",
        status="rejected",
    )

    output_dir = run_root / "snapshot"
    output_dir.mkdir()
    (output_dir / "manifest.jsonl").write_text("stale\n", encoding="utf-8")
    (output_dir / "SFT_edited_99.jsonl").write_text("stale\n", encoding="utf-8")
    report = exporter.export_completed_snapshot(
        run_root,
        output_dir=output_dir,
        workers=2,
        progress_every=1,
    )

    assert report["status"] == "passed"
    assert report["edited_pages_exported"] == 1
    assert report["terminal_results_skipped"] == 2
    assert report["page_entries_discovered"] == 4
    assert report["confusable_candidates"] == 3

    sft = _jsonl(output_dir / "sft.jsonl")
    assert sft[0]["messages"][1]["content"] == "A mutated dcgument paragraph."
    assert sft[0]["data_source"] == "chaos_document_ocr"
    assert (output_dir / sft[0]["images"][0]).resolve().is_file()

    verl = _jsonl(output_dir / "verl.jsonl")
    assert verl[0]["reward_model"]["ground_truth"] == sft[0]["messages"][1]["content"]
    assert verl[0]["extra_info"]["pair_id"] == edited_id
    assert verl[0]["extra_info"]["changes"] == [
        {
            "ocr_ans": "dcgument",
            "origin_ans": "document",
            "bbox": [10, 12, 60, 30],
        }
    ]
    assert not (output_dir / "manifest.jsonl").exists()
    assert not (output_dir / "SFT_edited_99.jsonl").exists()


def test_only_gt_and_image_are_rechecked_but_changes_are_preserved(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    mismatch = _write_terminal(run_root, "gt-mismatch_confusable_s1_hash")
    (mismatch.parent / "ground_truth.md").write_text("wrong\n", encoding="utf-8")
    outside = _write_terminal(run_root, "bbox-outside_confusable_s1_hash")
    payload = json.loads(outside.read_text(encoding="utf-8"))
    payload["changes"][0]["bbox"] = [10, 12, 999, 1000]
    outside.write_text(json.dumps(payload), encoding="utf-8")

    report = exporter.export_completed_snapshot(run_root, workers=1)

    assert report["status"] == "passed"
    assert report["edited_pages_exported"] == 1
    assert report["skip_reasons"] == {"ground_truth_file_mismatch": 1}
    verl = _jsonl(run_root / "completed_training_data" / "verl.jsonl")
    assert verl[0]["extra_info"]["changes"][0]["bbox"] == [10, 12, 999, 1000]


def test_cli_returns_two_for_a_valid_but_empty_snapshot(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "pages").mkdir(parents=True)
    assert exporter.main(["--run-root", str(run_root), "--workers", "1"]) == 2
