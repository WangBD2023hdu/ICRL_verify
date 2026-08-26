# Experimental source-first v2

## Boundary

The following production files are frozen while v2 is being developed:

- `scripts/run_arxiv_source_bins_to_verl.py`
- `scripts/build_source_first_color_page_gt.py`
- `scripts/build_arxiv_confusable_recompile_pilot.py`
- `scripts/verify_arxiv_confusable_recompile_pilot.py`

V2 has a new contract, a required `EXPERIMENTAL_V2.json` root marker, separate
checkpoints, and separate work/output directories.  It may import stable pure
helpers, but the stable entry point never imports v2 code.  The v2 batch runner
rejects `--builder-script` values resolving to any of the four frozen files,
and re-checks their hashes after all workers finish.  It only writes below the
marked experimental output root.

## Non-negotiable provenance rule

Markdown content is produced only from the LaTeX source and deterministic
compiler metadata such as `.aux` and SyncTeX.  PDF glyphs may supply page,
geometry, and a strict independent visible-character verification signal.  PDF
text is never copied into the Markdown ground truth.

## Algorithm

V2 replaces paragraph-sized placement with source atoms carrying exact source
offsets and Markdown semantics.  Colored glyphs are retained as baseline-level
runs instead of one union bounding box.  A page is partitioned into vertical
bands by spanning blocks; every band infers its own one- or two-column lanes.
Reading-order candidates are constrained by both source ordinal and geometry.
Only a unique candidate whose visible character stream exactly matches the
clean compiled page is accepted.

The intended order for a mixed page is, for example:

```text
full-width title
left column, then right column
full-width table caption and table
left column, then right column
full-width footnote/footer
```

Tables remain source-derived HTML with the caption as a separate Markdown
block.  Display and inline formulas retain source LaTeX.  Figures, plots, and
flowcharts remain outside the data contract and are removed before both clean
and locator compiles.

### Structural IR (experimental v2 only)

The experimental builder also freezes a bounded structural candidate lattice
before reading PDF text:

- literal, unique `\newtheorem` definitions plus one literal theorem label and
  one unique `.aux` number produce theorem-heading candidates;
- static LLNCS `\spnewtheorem`, `\spnewtheorem*`, and `\spn@wtheorem`
  declarations are admitted only for their source caption/environment.  The
  actual number still comes from one unique label and `.aux` record; class
  option counter behavior is never guessed;
- only a single `equation` environment with one literal label and one unique
  `.aux` number produces parenthesized/bare equation-tail candidates;
- dynamic/redefined theorem declarations, ambiguous AUX numbers, nested or
  unbalanced theorem blocks, multi-row displays, explicit `\tag`, and
  `\nonumber`/`\notag` fail closed.

The theorem locator owns only the exact `\begin{...}[optional title]` source
span and is SyncTeX-only; executable color commands are never inserted into a
theorem declaration.  Formula Markdown is kept byte-for-byte before a
source/AUX-derived tail is appended.  The complete audit is written to
`structural_source_ir.json`.  PDF text can select or reject one frozen
candidate through the exact verifier, but cannot create the heading, number,
punctuation, formula, or Markdown.

List serialization is isolated by provable top-level list instance.  A
literal ordinal reset or source-file transition closes an instance, while
nested lists and continuations remain attached to their parent.  An unsafe
dynamic description label rejects only that complete list instance; ordinals
in every accepted instance remain the fixed source-parser ordinals.

The verifier has a source-only visible-flow projection for math styling.  It
removes only a fixed allowlist of style wrappers, preserves TeX control-word
boundaries, and maps the unambiguous operators `\land`, `\lor`, and `\neg` to
their visible characters.  This projection never changes the stored Markdown.

When executable color instrumentation is unsafe for one complete source unit,
v2 may derive one deterministic invariant hybrid shadow.  It atomically reuses
that unit's unique SyncTeX page and geometry only when probe schemas are
identical and the clean/locator compilations have exact text, order, page count,
and zero-shift geometry.  Partial units, multiple pages or lanes, donor
conflicts, external verbatim, and schema/hash disagreements all fail closed.
The donor supplies no text: Markdown remains the original `SourceUnit`, and
PDF text is still read only after the candidate set is frozen.  The derivation
audit is written to `invariant_shadow_geometry_fallback.json`.

## Yield metric

Every successfully clean-compiled page gets exactly one row in
`page_ledger_v2.jsonl`, including rejected pages.  The primary metric is:

```text
strict source-first pages / eligible clean text pages
```

Eligibility is fixed before source-first acceptance and cannot depend on the
result.  Reports also include all-clean-page yield, final edited-pair yield,
and separate buckets for single-column, two-column, mixed spanning/two-column,
other multicolumn, and unknown layouts.  The aggregate report exposes both
`accepted_complex_layout_pages` and the strict `accepted_two_column_pages`
count.

The experimental gate requires all of the following:

- overall eligible-page source-first yield greater than 30%;
- at least one accepted complex-layout page;
- at least one accepted genuine two-column page;
- exact verifier success for every accepted page.

A small local pilot can validate behavior but cannot establish the 30% target.
The final comparison must use the same frozen paper list for stable and v2.

## Reproduced seven-paper gate

The final experimental build was run once with the final code over the fixed
paper IDs `2307.10185v4`, `2308.07591v4`, `2406.06108v2`, `2508.05301v2`,
`2601.04175v2`, `2605.30809v1`, and `2605.31524v1`.  The aggregate result is:

- 73 accepted pages from 235 eligible clean text pages: **31.06%**;
- accepted exact-verifier rate: **1.0**;
- accepted complex-layout pages: **2**;
- accepted genuine two-column pages: **2**;
- paper-processing errors: **0**.

The machine-readable report is
`output/pdf/source_first_v2_fixed7_final_v1/validation_report_v2.json`.  It also
contains the before/after stable-file hash guard and records
`pdf_used_for_generation=false`, `pdf_used_for_verification=true`.

## Fixed-cohort aggregation

Run the isolated batch runner with a new output directory.  Pass each stable
output tree explicitly when it exists; the runner also detects stable markers
in ancestor directories:

```bash
PYTHONPATH=src python scripts/experimental/run_arxiv_source_bins_to_verl_v2.py \
  --input-root /path/to/sources \
  --output-dir /path/to/output/source_first_v2_run \
  --stable-output-root /path/to/stable_v10_output \
  --workers 8 --figure-policy drop --drop-references
```

Exact command used for the reproduced gate (increase `--workers` on the
server):

```bash
PYTHONPATH=src python scripts/experimental/run_arxiv_source_bins_to_verl_v2.py \
  --input-root outputs/arxiv_latex_recompile_2000/papers \
  --output-dir output/pdf/source_first_v2_fixed7_final_v1 \
  --stable-output-root output/pdf/arxiv_confusable_recompile_2000 \
  --paper-ids 2307.10185v4 2308.07591v4 2406.06108v2 \
    2508.05301v2 2601.04175v2 2605.30809v1 2605.31524v1 \
  --workers 3 --figure-policy drop --drop-references \
  --compile-timeout 180 --paper-timeout 1200 --heartbeat-seconds 30
```

To aggregate an explicit fixed cohort without recompilation:

```bash
PYTHONPATH=src python scripts/experimental/evaluate_source_first_v2_cohort.py \
  --paper-output /path/to/output/source_first_v2_run/papers/PAPER_A \
  --paper-output /path/to/output/source_first_v2_run/papers/PAPER_B \
  --output-dir /path/to/output/source_first_v2_fixed_cohort \
  --stable-output-root /path/to/stable_v10_output
```

The evaluator is read-only with respect to paper inputs, validates the current
frozen-file hashes, and writes only the marked cohort output.  Its denominator
is the sum of `eligible_text_page` rows in the supplied paper list; it reports
eligible pages, accepted pages, exact-verifier rate, complex pages, and actual
two-column pages.  Parent progress is emitted per paper and as a 30-second
heartbeat while reading a large ledger.
