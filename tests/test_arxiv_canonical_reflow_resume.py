from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from arxiv_canonical_reflow_v4.core import CanonicalBlock, CanonicalPage
from arxiv_canonical_reflow_v4.resume import (
    CompileCheckpoint,
    JobResumeSpec,
    ResumeIndex,
)


def _block(block_id: str) -> CanonicalBlock:
    return CanonicalBlock(
        block_id=block_id,
        node_id=f"node-{block_id}",
        kind="paragraph",
        markdown=block_id,
        latex=block_id,
        verifier_text=block_id,
        weight=1,
        source_char_span=(0, len(block_id)),
        source_files=("main.tex",),
    )


def _page(blocks: Sequence[CanonicalBlock]) -> CanonicalPage:
    return CanonicalPage(
        page_id="paper_dense_source_pool_0000",
        paper_id="paper",
        ordinal=1,
        layout="one_column",
        blocks=tuple(blocks),
    )


def _bundle_ids(bundles: Sequence[Sequence[CanonicalBlock]]) -> list[tuple[str, ...]]:
    return [tuple(block.block_id for block in bundle) for bundle in bundles]


def _legacy_pair_id(
    page: CanonicalPage,
    bundles: Sequence[Sequence[CanonicalBlock]],
    *,
    output_ordinal: int = 1,
    mutation_suffix: str = "_confusable_s1_weighted",
) -> str:
    digest = hashlib.sha256(page.layout.encode("utf-8"))
    for bundle in bundles:
        for block in bundle:
            digest.update(b"\0" + block.block_id.encode("utf-8"))
    return (
        f"{page.paper_id}_dense_{page.ordinal}_{output_ordinal}_"
        f"{digest.hexdigest()[:12]}{mutation_suffix}"
    )


def test_restore_recovers_exact_legacy_span_and_leaves_unsaved_bundles(
    tmp_path: Path,
) -> None:
    blocks = tuple(
        _block(block_id) for block_id in ("left", "saved-a", "saved-b", "right")
    )
    page = _page(blocks)
    bundles = tuple((block,) for block in blocks)
    saved_id = _legacy_pair_id(page, bundles[1:3], output_ordinal=4)
    checkpoint = CompileCheckpoint(
        page,
        JobResumeSpec(str(tmp_path / "job.json"), (saved_id,)),
    )

    remaining = checkpoint.restore(bundles)

    assert _bundle_ids(remaining) == [("left",), ("right",)]
    assert checkpoint.records == [
        {"page_id": saved_id, "status": "accepted", "blocks": ["saved-a", "saved-b"]}
    ]
    assert checkpoint.legacy_recovered == 1
    assert checkpoint.next_ordinal == 5


def test_restore_keeps_all_bundles_when_legacy_span_is_not_found(
    tmp_path: Path,
) -> None:
    blocks = tuple(_block(block_id) for block_id in ("first", "second"))
    page = _page(blocks)
    bundles = tuple((block,) for block in blocks)
    missing_id = "paper_dense_1_1_ffffffffffff_confusable_s1_weighted"
    checkpoint = CompileCheckpoint(
        page,
        JobResumeSpec(str(tmp_path / "job.json"), (missing_id,)),
    )

    assert checkpoint.restore(bundles) == list(bundles)
    assert checkpoint.records == []
    assert checkpoint.legacy_recovered == 0


def test_restore_skips_rejected_blocks_and_reports_saved_completion(
    tmp_path: Path,
) -> None:
    blocks = tuple(_block(block_id) for block_id in ("rejected", "saved", "pending"))
    page = _page(blocks)
    bundles = tuple((block,) for block in blocks)
    saved_id = "saved-page"
    checkpoint = CompileCheckpoint(
        page,
        JobResumeSpec(str(tmp_path / "job.json"), (saved_id,)),
    )
    checkpoint.record(
        page_id=saved_id,
        status="accepted",
        blocks=(blocks[1],),
        next_ordinal=2,
    )
    checkpoint.record(
        page_id="rejected-page",
        status="rejected",
        blocks=(blocks[0],),
        next_ordinal=2,
    )

    assert _bundle_ids(checkpoint.restore(bundles)) == [("pending",)]
    assert checkpoint.completion_ids({saved_id}) is None

    checkpoint.record(
        page_id="rejected-pending",
        status="rejected",
        blocks=(blocks[2],),
        next_ordinal=3,
    )
    assert checkpoint.completion_ids({saved_id}) == {saved_id}


def test_restore_retries_accepted_page_missing_from_saved_ids(tmp_path: Path) -> None:
    block = _block("not-exported")
    page = _page((block,))
    bundles = ((block,),)
    checkpoint = CompileCheckpoint(
        page,
        JobResumeSpec(str(tmp_path / "job.json"), ("another-page",)),
    )
    checkpoint.record(
        page_id="accepted-but-not-exported",
        status="accepted",
        blocks=(block,),
        next_ordinal=2,
    )

    assert checkpoint.restore(bundles) == list(bundles)
    assert checkpoint.completion_ids({"another-page"}) is None


def test_checkpoint_restart_preserves_remaining_bundles_and_next_ordinal(
    tmp_path: Path,
) -> None:
    blocks = tuple(_block(block_id) for block_id in ("done", "todo-a", "todo-b"))
    page = _page(blocks)
    bundles = tuple((block,) for block in blocks)
    spec = JobResumeSpec(str(tmp_path / "job.json"), ("done-page",))
    first = CompileCheckpoint(page, spec)
    first.record(
        page_id="done-page",
        status="accepted",
        blocks=(blocks[0],),
        next_ordinal=11,
    )

    restarted = CompileCheckpoint(page, spec)

    assert restarted.next_ordinal == 11
    assert _bundle_ids(restarted.restore(bundles)) == [("todo-a",), ("todo-b",)]


def test_resume_index_requires_matching_namespace_fingerprint_and_dependencies(
    tmp_path: Path,
) -> None:
    settings = {"target_weight": 5200, "min_fill_ratio": 0.7}
    suffix = "_confusable_s1_weighted"
    saved_id = "saved-page"
    index = ResumeIndex(tmp_path, settings, {saved_id}, suffix)
    index.mark_source_complete("paper", "fingerprint-a", {saved_id})

    same = ResumeIndex(tmp_path, settings, {saved_id}, suffix)
    missing_dependency = ResumeIndex(tmp_path, settings, set(), suffix)
    changed_config = ResumeIndex(
        tmp_path,
        {**settings, "target_weight": 5201},
        {saved_id},
        suffix,
    )

    assert same.source_complete("paper", "fingerprint-a")
    assert not same.source_complete("paper", "fingerprint-b")
    assert not missing_dependency.source_complete("paper", "fingerprint-a")
    assert changed_config.root != index.root
    assert not changed_config.source_complete("paper", "fingerprint-a")


def test_resume_index_truncates_incomplete_source_index_tail(tmp_path: Path) -> None:
    settings = {"target_weight": 5200, "min_fill_ratio": 0.7}
    suffix = "_confusable_s1_weighted"
    index = ResumeIndex(tmp_path, settings, set(), suffix)
    complete = {"paper_id": "paper-a", "fingerprint": "fp-a", "accepted_ids": []}
    complete_line = (json.dumps(complete, separators=(",", ":")) + "\n").encode()
    index.source_path.parent.mkdir(parents=True, exist_ok=True)
    index.source_path.write_bytes(
        complete_line + b'{"paper_id":"paper-b","fingerprint":"fp-b","accepted_ids":['
    )

    resumed = ResumeIndex(tmp_path, settings, set(), suffix)

    assert resumed.sources == {"paper-a": complete}
    assert resumed.source_path.read_bytes() == complete_line
    assert resumed.source_complete("paper-a", "fp-a")
    assert not resumed.source_complete("paper-b", "fp-b")
