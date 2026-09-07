# Text rewrite SFT: answer-only heading changes

Entry: `scripts/build_arxiv_confusable_text_sft.py`.
This text-only task is separate from multimodal V5 synthesis.

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
