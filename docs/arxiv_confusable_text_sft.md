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
normalize newlines or reserialize HTML/LaTeX. The existing source extractor
still excludes tables and display-math environments; this change does not
add support for extracting them.

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
