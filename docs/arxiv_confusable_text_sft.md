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
   `#` characters is replaced with 0–4 `#` characters, excluding the original
   level. Each eligible heading has a **28% probability of removing all hashes**.
   The remaining 72% is split equally among levels 1–4 other than the original:
   three choices at 24% each for original levels 1–4, or four choices at 18%
   each for original levels 5–6. This is per-heading sampling, not a quota of
   entirely heading-free documents. Every eligible heading changes. No heading
   is added to other text. Spaces after the hashes and all other characters
   remain unchanged, including when all hashes are removed.
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
`to_level` (`0` means the heading hashes were removed). For each word mutation,
`input_char_offset`/`input_char_end` locate
the word in A; `char_offset`/`char_end` locate it in the complete fenced B.
The response token limit includes both fences. The sample validator rebuilds
B from A and the heading changes and requires exact equality.

CLI flags, input/output path conventions and the source-processing worker
pool are unchanged. The current pipeline is
`arxiv_confusable_text_sft_v5_weighted_pairs`, with prompt
`heading_rewrite_boundary_en_v3` and heading policy
`block_start_heading_levels_0_to_4_zero28_v2`. Only the approved prompt sentence
changes the allowed hash count from 1–4 to 0–4; the rest is unchanged. Updated
versions isolate checkpoints from earlier heading policies. Use a new output
directory to keep old and new datasets separate.

## V5 weighted letter mutations

Text rewrite and multimodal V5 share the exact `PAIR_COUNTS` / `PAIR_WEIGHTS`
in `src/arxiv_confusable_pairs.py`: 19 source letters, 45 directed pairs and
total weight 1012.
The digit substitution `0 -> e` is excluded. This replaces the old rewrite
letter table; it is not a union of the old and new pairs. For example, `m -> n`,
`n -> m` and `w -> v` are now available, while the old `g -> q` is not.

- First select a valid character pair by its integer weight, then uniformly
  select a matching word occurrence, then a valid character position in that
  occurrence. Do not multiply pair weights by the number of candidate words.
- Pairs with no remaining valid occurrence are removed and the remaining
  weights are renormalized. The exported frequencies depend on available
  source words; these weights are sampling targets, not exact corpus quotas.
- Rewrite keeps its existing approximately 10% occurrence-level budget,
  minimum 3 mutations, and default uncapped maximum. It does **not** switch
  to multimodal V5's 3–4 mutations per page or unique-word requirement.
- Each selected occurrence changes one lower-case character, with no insertion
  or deletion. Numbers, upper-case letters, tags, attributes, formulas, code
  and link targets keep the existing protection. Table cell text may mutate.
  Existing vocabulary collision checks remain in place.

Prompt `heading_rewrite_boundary_en_v3` and the 28% heading-removal rule are
unchanged. A is generated using the new character policy, and B retains the
same character mutations as A. `extra_info.changes` continues to record both
word spellings, the character pair, and offsets in A/B.
The mutation version is `chaos_text_empirical_1012_pair_first_v3`, and
`mutation_pair_policy_fingerprint` is recorded in each sample, the config
fingerprint inputs, and the final manifest. Old-policy checkpoints are not
reused. CLI flags and file paths are unchanged; use a new output directory.
Chinese glyph-pair synthesis is not changed by this English-letter update.

## Source tables → HTML (v3 extraction)

HTML table extraction was introduced in
`arxiv_confusable_text_sft_v3_html_tables` without changing the then-current
prompt. It remains unchanged in the current heading-zero version.

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

The script imports the repository's table parser and shared letter weights.
Transfer the script, `src/arxiv_confusable_pairs.py` **and**
`src/arxiv_source_first_v3/` together (or use the repository checkout);
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
