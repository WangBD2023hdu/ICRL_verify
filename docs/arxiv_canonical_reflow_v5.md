# Weighted multimodal synthesis V5

Entry: `scripts/experimental/build_arxiv_canonical_reflow_v5.py`.
This entry reuses the V4 source parser, canonical page layout, multiprocess
edited compilation, source GT verification, and streaming SFT/VERL export.
It depends on the V4 script and repository `src/` modules; copy the repository
to the server rather than this entry file alone.

## Mutation distribution

The user's original counts are in
`src/arxiv_canonical_reflow_v4/weighted_mutation.py:PAIR_COUNTS`.
`0 -> e` is excluded. The remaining 19 source letters have 45 directed
substitutions and a total integer weight of **1012**. Unlisted pairs, uppercase
letters, and digits are not mutated. There is no automatic reverse mapping.

Each page still requests 3 words with probability 40%, or 4 with probability
60%. Each selected word changes exactly one lowercase character. Only unique
visible words of at least four letters in ordinary prose are eligible.
Math, HTML tags, table blocks, headings, captions, code and URL targets keep
their existing content.

For each edit, select a character pair using its integer weight, then a
matching available word, then a position in that word. Do not multiply the
pair weight by character frequency. If no valid word remains for a pair,
remove it from that page's choices and renormalize the others. Thus `e -> c`
has a target share of 214/1012 = 21.15%, `m -> n` 63/1012 = 6.23%, and
`n -> m` 56/1012 = 5.53%. Availability and compile rejection mean saved ratios
are approximate; the code does not guarantee exact corpus-wide quotas.

The same replacement is applied to Markdown, LaTeX and verifier text before
compilation. Only the edited page is compiled. The image and full source GT
must pass the existing verification; mutation bboxes come from that render.

## Server command

Run from the repository directory, with its existing LaTeX/Poppler/Python
dependencies installed (same as V4):

```bash
python scripts/experimental/build_arxiv_canonical_reflow_v5.py \
  --crawler-root /inspire/sfs/project/inf-multimodal/public/wangbaode/06_datasets/03_crawl/outputs/arxiv_sources_balanced_20000 \
  --output-dir /inspire/sfs/project/inf-multimodal/public/wangbaode/06_datasets/03_crawl/arxiv_confusable_v5_weighted \
  --workers 128 \
  --target-count 40000
```

`--target-count` is the desired number of saved edited samples. Change 40000
as needed. New runs should use a new output directory. Re-running the same
command resumes that run. `--papers-root` also accepts normalized local
papers. Existing V4 defaults and V4 sample IDs are preserved.

Source parsing and compilation share the worker pool. Each ready sample is
appended to both datasets, flushed and fsynced before its temporary parts are
removed. Images use subdirectories of at most 20,000 samples. Temporary
compilation and extraction files are removed by the existing V4 workflow.

## Output and prompts

- `realtime_training/sft.jsonl`: ms-swift `messages` plus `images`.
- `realtime_training/verl.jsonl`: VERL `prompt`, `images`, `reward_model`, and
  `extra_info.changes` containing `origin_ans`, `ocr_ans`, `bbox`.
- `realtime_training/images/shard_00000/...png`: edited page images.
- `realtime_training/mutation_distribution.json`: target weights/ratios and
  actual saved counts/ratios, refreshed at checkpoints and shutdown. Resume
  rebuilds counts from saved VERL records, so interrupted runs are included.
- `run_summary.json`: selected policy, generation arguments, saved sample count.

Relative image paths remain relative to the JSONL's directory. Assistant GT
contains the mutated words exactly as printed. Mutation annotations are in
VERL metadata rather than inserted into GT. V5 now uses the exact DOC2MD prompt
from task `01a06b87-a91d-7522-99a2-356430f46b5f` (数据对齐与处理), stored in
`src/arxiv_canonical_reflow_v4/prompts.py`. Both outputs place it in the user
message. SFT prefixes the body with `<image>\n`; VERL uses the body directly and
supplies the image via `images`, matching that task's preprocessing scripts.
No system message, code fence, or additional instruction is added.

Prompt body (SHA-256: `37bd2dfdb0514637a85b7be8de52149c32455423a8c03e0f3cac0c4d5e0f9e86`):

```text
You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.
2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with $ $. For example: This is an inline formula $E = mc^2$
- Enclose block formulas with $$ $$. For example: $$\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

3. Table Processing:
- Convert tables to HTML format.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.
5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
```

This applies to newly exported rows (including unfinished worker parts on
resume). Previously completed JSONL records are not rewritten automatically.
