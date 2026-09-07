# Text rewrite SFT: answer-only heading changes

Entry: `scripts/build_arxiv_confusable_text_sft.py`.
This text-only task is separate from multimodal V5 synthesis.

## Downloaded source input

`--input-root` points to the crawler output root, not its `papers/` child.
When `results.jsonl` exists, the script keeps using it. When it is absent,
the script reads `papers/*/download.json` instead, using `--workers` processes
for checkpoint discovery and archive checks. The crawler writes these
per-paper records immediately after each download, so the corpus does not
need to finish downloading before synthesis starts.

Only successful (`passed`/`success`) records with a nonempty archive are used.
Failed downloads, missing/malformed checkpoints and incomplete archives with
only a `.partial` file are skipped and counted in the discovery progress.
No replacement `results.jsonl` is created and no download files are modified.
The existing license filter is unchanged; `--allow-all-licenses` remains an
explicit opt-in, not an automatic fallback.

## A/B transformation

The existing source extraction, window selection and approximately 10% word
mutation policy produce input document **A**, which is inserted between
`<<<DOCUMENT_START>>>` and `<<<DOCUMENT_END>>>` without an added code fence.

Answer **B** differs from A in exactly two ways:

1. At the beginning of each extracted text block, an existing prefix of 1–6
   `#` characters is replaced with 1–4 `#` characters, excluding the original
   level. Every eligible heading changes. No heading is added to other text.
   Spaces after the hashes and all other characters remain unchanged.
2. The complete answer is `"```markdown\n" + rewritten_body + "\n```"`.
   Even documents without headings receive this fence.

Heading randomness uses an independent stable seed, so the original A and
its character mutations do not change. This step does not trim whitespace,
normalize newlines or reserialize HTML/LaTeX. Source tables are converted to
HTML before constructing A (see below). Display-math environments remain
excluded; inline LaTeX formulas are retained.

The English prompt explicitly permits only these two formatting changes and
requires preserving every other character, including spelling errors. Its
exact text is in `PROMPT_PREFIX` and `PROMPT_SUFFIX` in the script. No system
message or image field is added.

`extra_info.heading_changes` records `input_offset`, `from_level` and
`to_level`. For each word mutation, `input_char_offset`/`input_char_end` locate
the word in A; `char_offset`/`char_end` locate it in the complete fenced B.
The response token limit includes both fences. The sample validator rebuilds
B from A and the heading changes and requires exact equality.

CLI flags, input/output path conventions and the source-processing worker
pool are unchanged. The pipeline/prompt/schema versions were bumped for this
format; old copy-task checkpoints are not reused. Use a new output directory
to keep old and new datasets separate.

## Source tables → HTML (v3 extraction)

The prompt is **unchanged**, including its version `heading_rewrite_boundary_en_v2`.
Only the pipeline version changes to `arxiv_confusable_text_sft_v3_html_tables`
so old table-free checkpoints are not reused.

- `tabular`, `tabular*` and `tabularx` are parsed from source, including those
  inside `table` / `table*` floats. No PDF, OCR, LaTeX compilation or LLM is used.
- Supported cells retain inline formulas and emphasis. `multicolumn` and
  `multirow` become `colspan` and `rowspan`; row-span placeholders are omitted
  from HTML. Invalid grids and spans extending beyond the table are rejected.
- Captions are separate blocks, not inside `<table>`. No automatic table
  number or `data-*` attribute is invented.
- Unknown macros, unsupported environments such as `longtable`, and malformed
  tables are skipped as whole tables and counted under `table:*` rejection
  reasons. Their cells are never exported as flattened prose or raw TeX.
- Each HTML table is one indivisible window block. Oversized tables are not
  truncated to fit the response limit. This does not guarantee every sample
  contains a table or impose a table quota.
- Character mutation may affect English **cell text**, but not HTML tags,
  attributes, entities, or LaTeX formulas. The resulting table in A and B is
  byte-for-byte identical; only text-block headings and the outer fence differ.

`extra_info.table_count` counts tables per sample; `source_spans[].kind`
distinguishes `text`, `caption` and `table`. The final manifest reports
`merge.table_samples` and `merge.tables_written`.

The script now imports the repository's table parser. Transfer the script
**and** `src/arxiv_source_first_v3/` together (or use the repository checkout);
it is no longer a standalone copied Python file. The table parser itself uses
the Python standard library. Tokenizer dependencies are unchanged.

## Incremental persistence

Each generated sample is immediately appended, flushed and fsynced to its
existing `checkpoints/<fingerprint>/<prefix>/<paper>.jsonl` location. These
are complete MS-Swift `messages` records, usable even during an unfinished run.
Each paper file has exactly one worker writer; no shared JSONL is concurrently
appended. Restarting with the same configuration retains completed sample IDs
and resumes a partially completed paper without duplicating rows.

The existing final balanced corpus selection, paper-level train/val split,
`train/part-*.jsonl`, `val/part-*.jsonl` and optional `--write-merged-jsonl`
outputs are unchanged. Those globally selected exports still run after paper
processing; the per-paper SFT records do not wait for that step. Do not combine
checkpoint files with the final exports, which contain selected copies of the
same samples.
