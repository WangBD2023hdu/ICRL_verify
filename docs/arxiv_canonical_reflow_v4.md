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
- The default final dataset is edited-only and the default execution path never
  compiles a clean page. Three or four ordinary-prose words receive one
  lower-case, equal-length confusable-character substitution in the source AST
  before compilation. Headings, authors, superscripts, captions, tables,
  formulas, code, URLs, and numbers are not mutated.
- Markdown, canonical LaTeX, and reject-only verifier text receive the same
  substitutions. Only the edited page is compiled; its complete rendered text
  and reading order must match the edited source-derived GT. Mutation boxes are
  located from that edited render. No clean-PDF comparison is required.

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
  --mutation-execution direct \
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
  --output-dir /path/to/output/arxiv_canonical_reflow_v4_confusable \
  --target-count 40000
```

Raw source archives are SHA-256 checked when the crawler supplied a digest,
extracted with traversal/link/device and expanded-size protections, statically
scanned, and converted to immutable AST page candidates. Each temporary
`metadata.json + source/` copy is deleted by the same worker immediately after
AST extraction. A download does not need to be globally complete: only final
non-empty `source_archive.bin` files are selected, and `.partial` files are
ignored.

The normal server command needs only input, output, and `--target-count`.
The target is the number of accepted edited pages that have been durably
appended to **both** the ms-swift and VERL JSONL files. A positive target also
selects the full available input corpus, so `--full-corpus`, `--paper-limit`,
and `--max-pages` are unnecessary. Every newly accepted sample refreshes one
in-place progress bar. Once the target is reached, no new work is scheduled;
any already-running worker output beyond the exact target is deleted.
Mutation mode, direct-edit execution, 70% minimum fill, and CPU worker count are
automatic defaults. `--work-dir` and `--crawler-cache-dir` remain optional
advanced overrides. Temporary LaTeX source, PDFs, auxiliary files, compiler
logs, and reject-only extracted text live outside `--output-dir` and are
deleted as soon as their derived data is safe. The final dataset keeps only
accepted PNGs, Markdown GT, compact result metadata, and training JSONL.

Compact progress is the default. `--verbose` restores detailed stage logs.
`--debug-artifacts` retains candidate manifests, rejected rows, per-paper
reports, and the legacy aggregate export; omit it for production generation.

`--workers` accepts 1--256 and defaults to all detected CPUs, capped at 128. On
a 128-core server it applies to
safe unpacking, source/AST extraction, and direct edited-page compilation.
Every process stage uses a bounded queue of at most twice the
worker count, so a large corpus does not enqueue every archive/page at once.
The parent process rewrites a single compact status line for each completed
source during preparation and each accepted sample during compilation, with a
30-second heartbeat while workers are busy. Existing accepted page
`result.json` files are reused only when their signatures and output contracts
match.

## Production outputs

- `pages/<pair_id>/page.png`: accepted edited page image.
- `pages/<pair_id>/ground_truth.md`: complete source-derived edited Markdown.
- `pages/<pair_id>/result.json`: compact cache/signature and mutation metadata
  required for safe resume; it is not a compiler artifact.
- `realtime_training/sft.jsonl`: append-only ms-swift multimodal SFT data.
- `realtime_training/verl.jsonl`: append-only VERL data with rule GT and
  `{ocr_ans, origin_ans, bbox}` mutation records.
- `realtime_training/progress.json`: atomic resumable row/job counts.
- `run_summary.json`: small final status and target-count report.

Workers briefly create paired one-page files under
`realtime_training/parts/` so an interruption cannot lose a newly accepted
sample. The parent appends and flushes both aggregate rows, then immediately
removes the pair; the directory is absent after a normal run. Rejected page
directories,
compiled PDFs, `.tex`, logs, auxiliary files, unpacked sources, and intermediate
candidate manifests are not retained. Original crawler `source_archive.bin`
files are never modified.

Image paths inside each training JSONL are relative to that JSONL's parent
directory, as required by the dataset reader. There is no `server-root` path
rewriting.

With `--debug-artifacts`, the program additionally writes `manifest.jsonl`,
`pairs.jsonl`, top-level SFT/VERL compatibility files, rejected/clean-stage
rows, candidate/job lists, crawler preparation reports, and
`pipeline_report.json`.

## Export a snapshot before the main run finishes

Production runs already update the two final training JSONL files after every
accepted sample, so no snapshot export is required. For an older/debug run,
the compile-free snapshot exporter first filters page-directory names to the
V4 `_confusable_s` suffix, so it does not open clean-page results. It can run
while the main V4 job is still active. The only export checks are that the
producer marked the confusable page accepted, its image exists, and the
Markdown in `result.json` (or legacy `terminal_result.json`) exactly matches
`ground_truth.md`. It does
not reopen PDFs or revalidate mutation geometry. The producer's
`{ocr_ans, origin_ans, bbox}` records are retained in VERL `extra_info.changes`.

```bash
python scripts/export_completed_arxiv_canonical_reflow_v4.py \
  --run-root /path/to/output/arxiv_canonical_reflow_v4_confusable \
  --workers 16
```

The default snapshot directory is
`<run-root>/completed_training_data/`. It contains only ms-swift `sft.jsonl`,
`verl.jsonl`, and `completed_export_report.json`. Image paths are relative to
the snapshot JSONL directory and continue to point at the existing
`<run-root>/pages/` artifacts. Both training files are written together in one
streaming pass. Re-running the command replaces each file atomically with all
pages completed at that time; it does not compile, mutate, or modify any page
artifact.
