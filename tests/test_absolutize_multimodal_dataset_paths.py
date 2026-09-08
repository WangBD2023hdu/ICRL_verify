from __future__ import annotations

import json
from pathlib import Path

from scripts import absolutize_multimodal_dataset_paths as converter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_converts_sft_and_verl_images_relative_to_each_input_parent(
    tmp_path: Path,
) -> None:
    sft_root = tmp_path / "sft_source"
    verl_root = tmp_path / "verl_source"
    output = tmp_path / "absolute"
    (sft_root / "images").mkdir(parents=True)
    (verl_root / "data").mkdir(parents=True)
    (sft_root / "images" / "sft.png").write_bytes(b"sft")
    (verl_root / "data" / "verl.png").write_bytes(b"verl")
    sft_path = sft_root / "sft.jsonl"
    verl_path = verl_root / "verl.jsonl"
    _write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "<image>transcribe"},
                    {"role": "assistant", "content": "mutatcd"},
                ],
                "images": ["images/sft.png"],
                "extra_info": {"pair_id": "sft-1"},
            }
        ],
    )
    _write_jsonl(
        verl_path,
        [
            {
                "prompt": [{"role": "user", "content": "<image>transcribe"}],
                "images": ["data/verl.png"],
                "reward_model": {"style": "rule", "ground_truth": "tcxt"},
                "extra_info": {
                    "pair_id": "verl-1",
                    "changes": [
                        {
                            "ocr_ans": "tcxt",
                            "origin_ans": "text",
                            "bbox": [1, 2, 3, 4],
                        }
                    ],
                },
            }
        ],
    )

    assert (
        converter.main(
            [
                "--input",
                str(sft_path),
                str(verl_path),
                "--output-dir",
                str(output),
                "--check-exists",
            ]
        )
        == 0
    )

    sft = _read_jsonl(output / "sft.absolute.jsonl")[0]
    verl = _read_jsonl(output / "verl.absolute.jsonl")[0]
    assert sft["images"] == [str((sft_root / "images" / "sft.png").resolve())]
    assert verl["images"] == [str((verl_root / "data" / "verl.png").resolve())]
    assert sft["messages"][-1]["content"] == "mutatcd"
    assert verl["reward_model"]["ground_truth"] == "tcxt"
    assert verl["extra_info"]["changes"][0]["bbox"] == [1, 2, 3, 4]


def test_preserves_absolute_and_remote_images(tmp_path: Path) -> None:
    local = tmp_path / "local.png"
    local.write_bytes(b"png")
    source = tmp_path / "dataset.jsonl"
    _write_jsonl(
        source,
        [
            {
                "messages": [],
                "images": [
                    str(local),
                    "https://example.com/image.png",
                    {"url": "s3://bucket/image.png"},
                ],
            }
        ],
    )

    assert converter.main(["--input", str(source)]) == 0
    row = _read_jsonl(tmp_path / "dataset.absolute.jsonl")[0]
    assert row["images"] == [
        str(local),
        "https://example.com/image.png",
        {"url": "s3://bucket/image.png"},
    ]


def test_repeated_input_options_process_both_files(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "page.png").write_bytes(b"png")
    sft = tmp_path / "sft.jsonl"
    verl = tmp_path / "verl.jsonl"
    _write_jsonl(sft, [{"messages": [], "images": ["images/page.png"]}])
    _write_jsonl(
        verl,
        [
            {
                "prompt": [],
                "reward_model": {"ground_truth": "GT"},
                "images": ["images/page.png"],
            }
        ],
    )

    assert (
        converter.main(
            [
                "--input",
                str(sft),
                "--input",
                str(verl),
            ]
        )
        == 0
    )
    assert (tmp_path / "sft.absolute.jsonl").is_file()
    assert (tmp_path / "verl.absolute.jsonl").is_file()
