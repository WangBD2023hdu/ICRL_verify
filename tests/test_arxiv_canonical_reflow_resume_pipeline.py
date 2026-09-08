from __future__ import annotations

import json
import time
from concurrent.futures import Future

import pytest

from arxiv_canonical_reflow_v4 import weighted_mutation
from arxiv_canonical_reflow_v4.core import CanonicalBlock, CanonicalPage
from scripts.experimental import build_arxiv_canonical_reflow_v4 as builder


class ImmediatePool:
    def __init__(self, max_workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - simulate Future's exception transport
            future.set_exception(exc)
        return future


def _setup(tmp_path, monkeypatch):
    output = tmp_path / "output"
    config = builder.WorkerConfig(
        output_dir=str(output),
        latexmk="latexmk",
        pdftoppm="pdftoppm",
        pdftotext="pdftotext",
        pdfinfo="pdfinfo",
        compile_timeout=1,
        render_timeout=1,
        dpi=72,
        max_pack_attempts=3,
        min_page_chars=0,
        target_weight=1000,
        two_column_rate=0,
        target_fill_ratio=0.8,
        min_fill_ratio=0.7,
        stop_file=str(output / "stop"),
    )
    mutation = builder.MutationConfig(
        1, 1, 1, 1.0, 1.0, policy=weighted_mutation.POLICY_NAME
    )
    blocks = tuple(
        CanonicalBlock(
            block_id=f"b{i}",
            node_id=f"n{i}",
            kind="paragraph",
            markdown=f"paragraph {i}",
            latex=f"paragraph {i}",
            verifier_text=f"paragraph {i}",
            weight=1000,
            source_char_span=(i, i + 1),
            source_files=("main.tex",),
        )
        for i in range(3)
    )
    page = CanonicalPage("candidate", "paper", 1, "one_column", blocks)
    archives = [builder.CrawlerArchive("paper", "source.bin", {}, None, 10, 0)]
    calls = {"source": 0, "compile": []}

    def extract(*_):
        calls["source"] += 1
        return (
            (page,),
            {"status": "success"},
            {
                "status": "prepared",
                "cache_cleanup": "deleted",
            },
        )

    def compile_page(candidate, _config, mutation_config):
        calls["compile"].append(tuple(b.block_id for b in candidate.blocks))
        pair_id = builder._mutation_page_id(candidate.page_id, mutation_config)
        image = output / "pages" / pair_id / "page.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"fixture image")
        return builder.WorkerResult(
            page_id=pair_id,
            paper_id="paper",
            status="accepted",
            reason=None,
            layout=candidate.layout,
            has_table=False,
            markdown=candidate.markdown,
            verifier_recall=1,
            verifier_precision=1,
            pdf=None,
            image=str(image),
            block_ids=tuple(b.block_id for b in candidate.blocks),
            source_node_ids=tuple(b.node_id for b in candidate.blocks),
            content_fill_ratio=0.8,
            column_fill_ratios=(0.8,),
            page_signature="fixture",
            elapsed_seconds=0,
            changes=({"origin_ans": "word", "ocr_ans": "wcrd", "bbox": [0, 0, 1, 1]},),
            variant="confusable_edit",
            mutation_count=1,
        )

    monkeypatch.setattr(builder, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(builder, "_prepare_extract_crawler_job", extract)
    monkeypatch.setattr(builder, "_direct_mutate_and_compile", compile_page)

    def run(target):
        builder._run_fused_crawler_direct_pipeline(
            archives,
            cache_root=tmp_path / "work" / "crawler",
            output=output,
            work_root=tmp_path / "work",
            config=config,
            mutation_config=mutation,
            target_count=target,
            workers=1,
            target_weight=1000,
            two_column_rate=0,
            input_root=tmp_path,
            debug_artifacts=False,
            started=time.monotonic(),
        )
        return json.loads((output / "run_summary.json").read_text())

    return output, calls, run, config, mutation, page


def _rows(output, name):
    return [
        json.loads(line)
        for line in (output / "realtime_training" / name).read_text().splitlines()
    ]


def test_target_increase_skips_saved_blocks_and_retries_discarded_overruns(
    tmp_path,
    monkeypatch,
):
    output, calls, run, *_ = _setup(tmp_path, monkeypatch)
    assert run(1)["accepted_count"] == 1
    original_row = _rows(output, "verl.jsonl")[0]
    calls["compile"].clear()
    summary = run(2)
    assert summary["accepted_before_run"] == 1
    assert summary["accepted_added_this_run"] == 1
    assert summary["saved_pages_skipped_before_compile"] == 1
    assert calls["compile"] == [("b1",), ("b2",)]
    assert _rows(output, "verl.jsonl")[0] == original_row
    calls["compile"].clear()
    assert run(3)["accepted_count"] == 3
    assert calls["compile"] == [("b2",)]
    for name in ("sft.jsonl", "verl.jsonl"):
        rows = _rows(output, name)
        assert len(rows) == len({r["extra_info"]["pair_id"] for r in rows}) == 3
        assert all(
            (output / "realtime_training" / r["images"][0]).is_file() for r in rows
        )


def test_completed_source_skips_both_extraction_and_compilation(tmp_path, monkeypatch):
    _output, calls, run, *_ = _setup(tmp_path, monkeypatch)
    assert run(10)["accepted_count"] == 3
    calls["source"] = 0
    calls["compile"].clear()
    summary = run(20)
    assert calls == {"source": 0, "compile": []}
    assert summary["sources_skipped_from_checkpoint"] == 1
    assert summary["saved_pages_skipped_before_compile"] == 3
    assert summary["accepted_added_this_run"] == 0


def test_legacy_dataset_only_migrates_without_recompiling_saved_page(
    tmp_path, monkeypatch
):
    output, calls, run, config, mutation, page = _setup(tmp_path, monkeypatch)
    job = builder._bounded_dense_jobs_from_pages((page,))[0]
    candidate = builder._dense_page(
        job,
        builder.bundle_blocks(job.blocks),
        start=0,
        end=1,
        output_ordinal=1,
        config=config,
    )
    # Simulate old output: only PNG + the two final JSONL records; no resume state.
    result = builder._direct_mutate_and_compile(candidate, config, mutation)
    writer = builder._RealtimeTrainingWriter(output, mutation_policy=mutation.policy)
    writer.add(result)
    writer.close()
    assert not (output / "realtime_training" / "resume").exists()
    assert not (output / "pages" / result.page_id).exists()
    calls["compile"].clear()
    summary = run(3)
    assert calls["compile"] == [("b1",), ("b2",)]
    assert summary["accepted_before_run"] == 1
    assert summary["accepted_added_this_run"] == 2
    assert summary["saved_pages_skipped_before_compile"] == 1


def test_checkpoint_survives_worker_interruption_before_parent_export(
    tmp_path, monkeypatch
):
    output, calls, run, config, mutation, page = _setup(tmp_path, monkeypatch)
    job = builder._bounded_dense_jobs_from_pages((page,))[0]
    index = builder._compile_resume_index(output, config, mutation, set())
    original = builder._direct_mutate_and_compile

    def interrupted(candidate, config, mutation_config):
        if candidate.blocks[0].block_id == "b1":
            raise RuntimeError("simulated worker interruption")
        return original(candidate, config, mutation_config)

    monkeypatch.setattr(builder, "_direct_mutate_and_compile", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        builder._compile_with_rescue(
            job, config, mutation_config=mutation, resume_spec=index.job_spec(job)
        )
    assert list((output / "realtime_training" / "parts").glob("*.verl.jsonl"))
    monkeypatch.setattr(builder, "_direct_mutate_and_compile", original)
    calls["compile"].clear()
    summary = run(3)
    assert summary["accepted_before_run"] == 1
    assert calls["compile"] == [("b1",), ("b2",)]
    assert len(_rows(output, "verl.jsonl")) == 3
