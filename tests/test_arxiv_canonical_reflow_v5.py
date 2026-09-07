from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image

from arxiv_canonical_reflow_v4 import weighted_mutation as weighted
from arxiv_canonical_reflow_v4.core import CanonicalBlock, CanonicalPage, build_page_tex
from arxiv_canonical_reflow_v4.mutation import apply_page_mutations, markdown_diff_count
from arxiv_canonical_reflow_v4.prompts import (
    DOC2MD_PROMPT,
    DOC2MD_PROMPT_STYLE,
    DOC2MD_SFT_PROMPT,
)
from scripts.experimental import build_arxiv_canonical_reflow_v4 as builder
from scripts.experimental import build_arxiv_canonical_reflow_v5 as v5


def _block(
    text: str, *, kind: str = "paragraph", block_id: str = "body"
) -> CanonicalBlock:
    return CanonicalBlock(
        block_id=block_id,
        node_id=block_id,
        kind=kind,
        markdown=text,
        latex=text + r"\par",
        verifier_text=text,
        weight=len(text),
        source_files=("test.tex",),
        source_char_span=(0, len(text)),
    )


def _page(text: str) -> CanonicalPage:
    return CanonicalPage("example", "test", 1, "one_column", (_block(text),))


def _all_pairs_page() -> CanonicalPage:
    # Each word contains exactly one mutable letter. e has 100 extra words;
    # its character frequency must not inflate its pair sampling probability.
    letters = list(weighted.PAIR_COUNTS) + ["e"] * 100
    words = [
        f"{char}A{chr(65 + i // 26)}{chr(65 + i % 26)}"
        for i, char in enumerate(letters)
    ]
    return _page(" ".join(words))


def _draw_batch(seed_start: int) -> Counter[tuple[str, str]]:
    page = _all_pairs_page()
    counts: Counter[tuple[str, str]] = Counter()
    for seed in range(seed_start, seed_start + 2500):
        (change,) = weighted.choose_weighted_source_page_mutations(
            page,
            seed=seed,
            minimum=1,
            maximum=1,
        )
        counts[change.from_char, change.to_char] += 1
    return counts


def test_pair_distribution_is_not_reweighted_by_source_character_frequency() -> None:
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        batches = list(pool.map(_draw_batch, [0, 2500, 5000, 7500]))
    observed = sum(batches, Counter())
    total = sum(observed.values())
    assert total == 10000
    assert weighted.TOTAL_WEIGHT == 1012
    assert all(a.islower() and b.islower() for a, b in weighted.PAIR_WEIGHTS)
    for pair, weight in weighted.PAIR_WEIGHTS.items():
        p = weight / 1012
        assert abs(observed[pair] - total * p) < 6 * math.sqrt(total * p * (1 - p)) + 3
    assert set(observed) == set(weighted.PAIR_WEIGHTS)


def test_reproducible_one_character_edits_match_all_source_channels() -> None:
    page = _all_pairs_page()
    edits = weighted.choose_weighted_source_page_mutations(page, seed=913)
    assert edits == weighted.choose_weighted_source_page_mutations(page, seed=913)
    assert len(edits) in (3, 4)
    assert len({edit.original_word for edit in edits}) == len(edits)
    changed = apply_page_mutations(page, edits, page_id="edited")
    for before, after in (
        (page.markdown, changed.markdown),
        (page.verifier_text, changed.verifier_text),
        (build_page_tex(page), build_page_tex(changed)),
    ):
        assert markdown_diff_count(before, after) == len(edits)


def test_missing_pairs_resample_and_numeric_math_markup_stay_unchanged() -> None:
    page = _page(
        "MMMM 0000 2026 `example` $formula$ "
        "https://example.org/number [LINK](https://table.org) zzzz qqqq "
        "<em>mXXX</em> mYYY mZZZ"
    )
    page = replace(
        page,
        blocks=page.blocks
        + (
            _block("example heading", kind="heading", block_id="heading"),
            _block(
                "<table><tr><td>number</td></tr></table>",
                kind="table",
                block_id="table",
            ),
        ),
    )
    edits = weighted.choose_weighted_source_page_mutations(
        page,
        seed=7,
        minimum=3,
        maximum=3,
    )
    assert len(edits) == 3
    assert {(e.from_char, e.to_char) for e in edits} == {("m", "n")}
    assert all(e.original_word in {"mXXX", "mYYY", "mZZZ"} for e in edits)
    result = apply_page_mutations(page, edits, page_id="new")
    assert result.blocks[1:] == page.blocks[1:]
    assert "MMMM 0000 2026 `example` $formula$" in result.markdown
    assert "<em>nXXX</em>" in result.markdown


def test_collision_is_not_exported_and_too_few_candidates_return_empty() -> None:
    page = _page("mXYZ nXYZ mABC")
    # mXYZ -> nXYZ would collide. nXYZ and mABC still provide two valid words.
    edits = weighted.choose_weighted_source_page_mutations(
        page, seed=4, minimum=2, maximum=2
    )
    assert len(edits) == 2
    assert all(e.mutated_word not in page.markdown.split() for e in edits)
    assert (
        weighted.choose_weighted_source_page_mutations(
            _page("0000 MMMM qqqq zzzz"),
            seed=9,
        )
        == ()
    )


def test_v5_entry_passes_policy_and_v4_keeps_previous_page_ids(monkeypatch) -> None:
    config = builder.MutationConfig(19, 3, 4, 0.6, 1.25)
    identity = asdict(config)
    identity.pop("policy")
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    assert builder._mutation_page_id("page", config) == f"page_confusable_s19_{digest}"
    assert builder._mutation_page_id("page", config) != builder._mutation_page_id(
        "page",
        replace(config, policy=weighted.POLICY_NAME),
    )

    def fake_main(argv, *, default_mutation_policy, default_prompt_style):
        assert argv == ["--help"]
        assert default_mutation_policy == weighted.POLICY_NAME
        assert default_prompt_style == DOC2MD_PROMPT_STYLE
        return 0

    monkeypatch.setattr(builder, "main", fake_main)
    assert v5.main(["--help"]) == 0
    assert (
        builder._parser()
        .parse_args(
            [
                "--papers-root",
                ".",
                "--output-dir",
                "output",
            ]
        )
        .mutation_policy
        == "v4"
    )


def _result(output: Path) -> builder.WorkerResult:
    folder = output / "pages" / "example_confusable_s1_weighted"
    folder.mkdir(parents=True)
    image = folder / "page.png"
    Image.new("RGB", (20, 20), "white").save(image)
    return builder.WorkerResult(
        page_id=folder.name,
        paper_id="test",
        status="accepted",
        reason=None,
        layout="one_column",
        has_table=False,
        markdown="nXXX",
        verifier_recall=1.0,
        verifier_precision=1.0,
        pdf=None,
        image=str(image),
        block_ids=("body",),
        source_node_ids=("body",),
        content_fill_ratio=0.8,
        column_fill_ratios=(0.8,),
        page_signature="test",
        elapsed_seconds=0.1,
        mutation_count=1,
        changes=({"origin_ans": "mXXX", "ocr_ans": "nXXX", "bbox": [1, 1, 15, 15]},),
        variant="confusable_edit",
    )


def test_weighted_writer_saves_usable_rows_stats_and_resumes_without_duplicates(
    tmp_path, monkeypatch
) -> None:
    result = _result(tmp_path)
    sync_calls = []
    monkeypatch.setattr(builder.os, "fsync", lambda fd: sync_calls.append(fd))
    writer = builder._RealtimeTrainingWriter(
        tmp_path,
        mutation_policy=weighted.POLICY_NAME,
        prompt_style=DOC2MD_PROMPT_STYLE,
    )
    assert writer.add(result)
    assert len(sync_calls) >= 2
    writer.close()
    root = tmp_path / "realtime_training"
    sft = json.loads((root / "sft.jsonl").read_text())
    verl = json.loads((root / "verl.jsonl").read_text())
    assert sft["messages"][0]["content"] == DOC2MD_SFT_PROMPT
    assert verl["prompt"] == [{"role": "user", "content": DOC2MD_PROMPT}]
    assert (
        sft["messages"][1]["content"] == verl["reward_model"]["ground_truth"] == "nXXX"
    )
    assert verl["extra_info"]["changes"] == list(result.changes)
    assert (root / verl["images"][0]).is_file()
    assert verl["images"][0].startswith("images/shard_00000/")
    resumed = builder._RealtimeTrainingWriter(
        tmp_path,
        mutation_policy=weighted.POLICY_NAME,
        prompt_style=DOC2MD_PROMPT_STYLE,
    )
    assert resumed.add(result)
    resumed.close()
    assert len((root / "verl.jsonl").read_text().splitlines()) == 1
    report = json.loads((root / "mutation_distribution.json").read_text())
    assert report["total_weight"] == 1012
    assert report["saved_mutations"] == 1
    assert report["pair_distribution"]["m->n"]["saved_count"] == 1
    assert report["other_pairs"] == {}


def test_doc2md_prompt_matches_confirmed_thread_exactly() -> None:
    assert hashlib.sha256(DOC2MD_PROMPT.encode()).hexdigest() == (
        "37bd2dfdb0514637a85b7be8de52149c32455423a8c03e0f3cac0c4d5e0f9e86"
    )
    assert hashlib.sha256(DOC2MD_SFT_PROMPT.encode()).hexdigest() == (
        "0de57a17ecb6a6a51932cc7ec6781708ce0d3173f0d5742f53db40d214b507b1"
    )


def test_doc2md_prompt_reaches_spawned_worker_export_and_old_parts_recovery(
    tmp_path,
) -> None:
    result = replace(_result(tmp_path), prompt_style=DOC2MD_PROMPT_STYLE)
    root = tmp_path / "realtime_training"
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        sft, verl = pool.submit(builder._realtime_training_rows, result, root).result(
            timeout=30
        )
    assert sft["messages"][0]["content"] == DOC2MD_SFT_PROMPT
    assert verl["prompt"][0]["content"] == DOC2MD_PROMPT
    # An interrupted old worker may have written parts before the prompt update.
    sft["messages"][0]["content"] = "old prompt"
    verl["prompt"][0]["content"] = "old prompt"
    parts = root / "parts"
    parts.mkdir(parents=True)
    for suffix, row in (("sft", sft), ("verl", verl)):
        (parts / f"{result.page_id}.{suffix}.jsonl").write_text(json.dumps(row) + "\n")
    writer = builder._RealtimeTrainingWriter(
        tmp_path,
        mutation_policy=weighted.POLICY_NAME,
        prompt_style=DOC2MD_PROMPT_STYLE,
    )
    writer.close()
    restored_sft = json.loads((root / "sft.jsonl").read_text())
    restored_verl = json.loads((root / "verl.jsonl").read_text())
    assert restored_sft["messages"][0]["content"] == DOC2MD_SFT_PROMPT
    assert restored_verl["prompt"][0]["content"] == DOC2MD_PROMPT
    assert restored_verl["reward_model"] == verl["reward_model"]
    assert restored_verl["extra_info"]["changes"] == verl["extra_info"]["changes"]
