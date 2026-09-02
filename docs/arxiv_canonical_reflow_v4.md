# arXiv canonical reflow V4

`scripts/experimental/build_arxiv_canonical_reflow_v4.py` is an experimental,
independent alternative to the original-page V1/V2/V3 pipelines. It uses an
arXiv project as a source corpus and creates new canonical one-page documents.
It does **not** attempt to reproduce the paper's original page boundaries.

## Correctness contract

- LaTeX source AST is the only ground-truth source.
- Markdown and canonical LaTeX are serialized from the same immutable source
  blocks.
- Citations, bibliography, references, and figures are removed by the existing
  source sanitizer before block extraction.
- Strict tables become clean HTML. Captions remain separate blocks and no
  `Table N` prefix is invented.
- Every candidate is compiled as a standalone one-page XeLaTeX document.
- PDF text and compiler overflow diagnostics are reject-only. They never write
  or repair GT.
- Two-column pages are restricted to sufficiently dense prose/formula groups.
  Table pages are full-width to prevent narrow-column overlap.
- Headings stay with following content; adjacent table/caption blocks are
  indivisible.
- Packing is compiler-driven: source bundles are appended until the rendered
  page reaches the target height. If an append overflows or fails verification,
  only that last bundle is removed. The old midpoint split is not used.
- Rendered height comes from reject-only PDF bounding boxes. By default an
  accepted page must occupy at least 70% of the usable vertical area and the
  packer aims for 82%; two-column pages apply the threshold to both columns.
- The default final dataset is edited-only. Each clean canonical page is used
  as an intermediate proof target, then 3--4 ordinary-prose words receive one
  lower-case, equal-length confusable-character substitution. Headings,
  authors, superscripts, captions, tables, formulas, code, URLs, and numbers
  are not mutated.
- Markdown, canonical LaTeX, and reject-only verifier text receive the same
  substitutions. The edited page is recompiled and accepted only when the
  complete word sequence differs exactly at those declared words, column
  assignments are unchanged, and maximum vertical movement is at most 1.25pt.

## Local pilot

Install the Python dependencies once; LaTeX and Poppler executables are checked
separately by the existing environment checker:

```bash
python -m pip install -r requirements-arxiv-canonical-reflow-v4.txt
```

```bash
PYTHONPATH=src python scripts/experimental/build_arxiv_canonical_reflow_v4.py \
  --papers-root outputs/arxiv_latex_recompile_2000/papers \
  --output-dir output/pdf/arxiv_canonical_reflow_v4 \
  --paper-limit 20 \
  --max-pages 100 \
  --workers 8 \
  --target-weight 5200 \
  --target-fill-ratio 0.82 \
  --min-fill-ratio 0.70 \
  --two-column-rate 0.40 \
  --mutation-mode confusable \
  --min-mutations-per-page 3 \
  --max-mutations-per-page 4
```

Set `--paper-limit 0 --max-pages 0` to process every available paper and page.
The same program can consume the raw output of `crawl_arxiv_sources.py`
directly. `--crawler-root` accepts either the crawler root containing
`results.jsonl` and `papers/`, or the `papers/` directory itself:

```bash
PYTHONPATH=src python scripts/experimental/build_arxiv_canonical_reflow_v4.py \
  --crawler-root /path/to/arxiv_sources/papers \
  --crawler-cache-dir /path/to/work/arxiv_canonical_reflow_v4_unpack \
  --output-dir /path/to/output/arxiv_canonical_reflow_v4_confusable \
  --paper-limit 0 \
  --max-pages 0 \
  --workers 128 \
  --target-fill-ratio 0.82 \
  --min-fill-ratio 0.70 \
  --two-column-rate 0.40 \
  --mutation-mode confusable
```

Raw source archives are SHA-256 checked when the crawler supplied a digest,
extracted with traversal/link/device and expanded-size protections, statically
scanned, and atomically promoted into a resumable `metadata.json + source/`
cache. A download does not need to be globally complete: only final non-empty
`source_archive.bin` files are selected, and `.partial` files are ignored.

`--workers` accepts 1--256. On a 128-core server, `--workers 128` applies to
safe unpacking, source/AST extraction, clean-page compilation, and mutation
recompilation. Every process stage uses a bounded queue of at most twice the
worker count, so a large corpus does not enqueue every archive/page at once.
The parent process prints per-unit aggregate progress and 30-second heartbeats
with totals, bytes, throughput, elapsed time, ETA, and accepted/rejected/error
counts. Existing prepared archives and page `result.json` files are reused only
when their input fingerprints and output contracts match.

## Outputs

- `manifest.jsonl`: page image, PDF, Markdown, layout, table flag, source node
  IDs, actual content/column fill ratios, reject-only verifier metrics, and
  complete mutation provenance.
- `pairs.jsonl`: a compatibility alias of the final edited manifest.
- `sft.jsonl`: ms-swift-style multimodal SFT rows.
- `SFT_edited_<N>.jsonl`: V1-compatible `conversations` SFT rows.
- `verl.jsonl`: VERL-style rows with rule ground truth and compact
  `{ocr_ans, origin_ans, bbox}` mutation records.
- `rejected_pages.jsonl`: terminal page rejections.
- `clean_stage_results.jsonl`: audit-only clean-stage results; clean pages are
  never included in the default final manifest/SFT/VERL datasets.
- `pipeline_report.json`: strict page acceptance and scheduled source-block
  yield, plus per-paper AST audits.
- `crawler_prepare_results.jsonl` and `crawler_prepare_report.json`: raw-bin
  materialization status, bytes, cache paths, failures, and resume state (only
  when `--crawler-root` is used).
- `pages/<pair_id>/`: complete GT, canonical TeX, PDF, PNG, logs, and result.

All exported image paths are relative to the output directory. There is no
`server-root` path rewriting.

## Export a snapshot before the main run finishes

The compile-free snapshot exporter reads only authoritative, atomically written
`pages/*/terminal_result.json` files. It therefore can run while the main V4
job is still active. Only accepted `confusable_edit` pages with matching GT,
PNG/PDF artifacts, mutation counts, and in-image mutation boxes are exported;
clean, rejected, incomplete, or malformed pages are skipped and audited.

```bash
python scripts/export_completed_arxiv_canonical_reflow_v4.py \
  --run-root /path/to/output/arxiv_canonical_reflow_v4_confusable \
  --workers 128
```

The default snapshot directory is
`<run-root>/completed_training_data/`. It contains `manifest.jsonl`,
`pairs.jsonl`, ms-swift `sft.jsonl`, V1-compatible SFT files, `verl.jsonl`,
`skipped_terminal_results.jsonl`, and `completed_export_report.json`. Image and
PDF paths are relative to the snapshot JSONL directory and continue to point
at the existing `<run-root>/pages/` artifacts. Re-running the command replaces
each snapshot file atomically with all pages completed at that time; it does not
compile, mutate, or modify any page artifact.
