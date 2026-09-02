from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

from scripts import export_completed_arxiv_canonical_reflow_v4 as exporter

PIPELINE_VERSION = exporter.PIPELINE_VERSION


def _png_bytes(width: int = 120, height: int = 80) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _write_terminal(
    run_root: Path,
    page_id: str,
    *,
    variant: str = "confusable_edit",
    status: str = "accepted",
    with_image: bool = True,
) -> Path:
    page_dir = run_root / "pages" / page_id
    page_dir.mkdir(parents=True)
    image = page_dir / "page.png"
    pdf = page_dir / "build" / "page.pdf"
    pdf.parent.mkdir()
    if with_image:
        image.write_bytes(_png_bytes())
    pdf.write_bytes(b"%PDF-1.4\nfixture\n")
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
        "pdf": str(pdf.resolve()),
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
    result_path = page_dir / "terminal_result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exports_only_complete_confusable_pages_in_both_training_formats(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    _write_terminal(run_root, "edited-page")
    _write_terminal(run_root, "clean-page", variant="clean")
    _write_terminal(run_root, "missing-image", with_image=False)
    _write_terminal(run_root, "rejected-page", status="rejected")

    output_dir = run_root / "snapshot"
    report = exporter.export_completed_snapshot(
        run_root,
        output_dir=output_dir,
        workers=2,
        progress_every=1,
    )

    assert report["status"] == "passed"
    assert report["edited_pages_exported"] == 1
    assert report["terminal_results_skipped"] == 3
    assert report["table_pages"] == 1
    assert report["two_column_pages"] == 1

    manifest = _jsonl(output_dir / "manifest.jsonl")
    assert len(manifest) == 1
    assert manifest[0]["pair_id"] == "edited-page"
    assert manifest[0]["variant"] == "confusable_edit"
    assert manifest[0]["ground_truth_source"] == "latex_ast_confusable_edit"
    assert (output_dir / str(manifest[0]["image"])).resolve().is_file()

    sft = _jsonl(output_dir / "sft.jsonl")
    assert sft[0]["messages"][1]["content"] == manifest[0]["markdown"]
    assert sft[0]["images"] == [manifest[0]["image"]]
    assert sft[0]["data_source"] == "chaos_document_ocr"

    verl = _jsonl(output_dir / "verl.jsonl")
    assert verl[0]["reward_model"]["ground_truth"] == manifest[0]["markdown"]
    assert verl[0]["extra_info"]["changes"] == [
        {
            "ocr_ans": "dcgument",
            "origin_ans": "document",
            "bbox": [10, 12, 60, 30],
        }
    ]
    assert (output_dir / "SFT_edited_1.jsonl").is_file()


def test_rejects_gt_mismatch_and_bbox_outside_image(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    mismatch = _write_terminal(run_root, "gt-mismatch")
    (mismatch.parent / "ground_truth.md").write_text("wrong\n", encoding="utf-8")
    outside = _write_terminal(run_root, "bbox-outside")
    payload = json.loads(outside.read_text(encoding="utf-8"))
    payload["changes"][0]["bbox"] = [10, 12, 999, 1000]
    outside.write_text(json.dumps(payload), encoding="utf-8")

    report = exporter.export_completed_snapshot(run_root, workers=1)

    assert report["status"] == "empty"
    assert report["edited_pages_exported"] == 0
    assert report["skip_reasons"] == {
        "invalid_change:0:bbox_out_of_image": 1,
        "ground_truth_file_mismatch": 1,
    }
    assert _jsonl(run_root / "completed_training_data" / "manifest.jsonl") == []


def test_cli_returns_two_for_a_valid_but_empty_snapshot(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "pages").mkdir(parents=True)
    assert exporter.main(["--run-root", str(run_root), "--workers", "1"]) == 2
