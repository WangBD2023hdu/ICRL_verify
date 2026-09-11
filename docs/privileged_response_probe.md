# Fixed-response privileged-information probe

`qwen-mm-privileged-probe` is a local Hugging Face Transformers experiment. It
does not call an API, create a train/validation split, or fit a threshold.

By default the student is also the teacher, so only one model is loaded. For
every page it performs exactly:

1. one student `model.generate` call with `image + original prompt`, preserving
   the generated response ID sequence;
2. one teacher forward after directly concatenating that ID tensor to the
   text-only GT prompt shown below;
3. one student forward after directly concatenating the identical ID tensor to
   the original multimodal prompt, scoring both the response IDs and the
   teacher's Top-1 IDs at each position.

The teacher's text-only prompt is:

```text
Please transcribe the document enclosed by the boundary markers verbatim, character by character and symbol by symbol. This is a transcription task, not a translation task. Do not change, correct, add, or omit any character. Output only the document content; do not include the boundary markers.

<<<DOCUMENT_START>>>
{privileged_text}
<<<DOCUMENT_END>>>
```

The complete Markdown GT is inserted verbatim. If it has no trailing newline,
one newline is added only to place `<<<DOCUMENT_END>>>` on its own line.

The response is never decoded and re-tokenized for either forward. Every
`result.json` records `response_ids_directly_concatenated=true` and
`response_text_retokenized=false`.

To use a separate teacher, pass `--teacher-model-id`. The existing `--model-id`
is the student; `--student-model-id` is an equivalent, more explicit spelling.
Both models inherit the same `--dtype`, `--device-map`, and
`--trust-remote-code` settings. Before the teacher forward, the probe verifies
that every generated student token ID and the complete ID sequence decode
identically with the teacher tokenizer. It stops with an error instead of
silently comparing incompatible vocabularies.

```bash
qwen-mm-privileged-probe \
  --student-model-id /path/to/student-model \
  --teacher-model-id /path/to/teacher-model \
  --dataset-root "$DATA_BASE/arxiv_confusable_v10_36_server" \
  --output-dir outputs/arxiv_confusable_two_model_probe \
  --max-new-tokens 4096 \
  --top-k 5 \
  --dtype bfloat16 \
  --device-map auto \
  --trust-remote-code
```

In two-model mode both model instances stay loaded during the run. Ensure that
the selected device map has enough GPU/CPU memory. `config.json` and every
sample `result.json` record the student model ID, teacher model ID, and whether
the same model instance was reused. With different student and teacher models,
`delta_logp = teacher - original` contains both a model-parameter difference
and a context difference. Only the default same-model mode isolates the effect
of adding the privileged GT context while holding model parameters fixed.

```bash
DATA_BASE=/home/ma-user/work/wangbaode/03_innovate/ICRL_verify/exp_v2/data

pip install -e .

tar -xzf "$DATA_BASE/arxiv_confusable_v10_36_server.tar.gz" \
  -C "$DATA_BASE"

qwen-mm-privileged-probe \
  --model-id Qwen/Qwen3.5-4B \
  --dataset-root "$DATA_BASE/arxiv_confusable_v10_36_server" \
  --output-dir outputs/arxiv_confusable_privileged_probe_v1 \
  --max-new-tokens 4096 \
  --top-k 5 \
  --dtype bfloat16 \
  --device-map auto \
  --trust-remote-code \
  --min-pixels 2048 \
  --max-pixels 16777216 \
  --image-patch-size 16 \
  --heartbeat-seconds 30
```

The root `report.html` is a sample browser and opens the first completed sample
directly. It contains no aggregate statistics or filtered token analysis. Each
sample report keeps the complete Ground Truth and model Response visible side by
side, followed by every generated response token in its original ID order.

The last token-table column, `学生 p(教师 Top-1)`, reports the student's
original-image probability for the teacher's GT-conditioned Top-1 **token ID**,
even when it differs from the actual response token or is outside the student's
saved Top-k. Both distributions use the unchanged student response prefix.
New runs score the teacher first, then gather these IDs from the student logits
during the existing original-image forward; this still needs only two scoring
forwards per sample. The prompts and generation are unchanged.

The values are also saved as `p_original_teacher_top1` and
`logp_original_teacher_top1` in the token JSON/CSV. Re-running the same inference
command with resume enabled extends matching old results without regenerating
responses or recomputing teacher scores: known exact-ID values are reused, and
any missing values require one student scoring forward per affected sample.
`--rebuild-report-only` does not run a model; missing values that cannot be
recovered from saved student Top-k entries display as `未记录（需补评分）`, not zero.

To infer only the first 10 table-containing samples, add `--require-table --limit 10`
to the inference command. Selection uses HTML `<table>` opening tags in the GT
(case-insensitive, including tags with attributes), before applying the limit.
The original manifest order, sample ordinals and per-sample seeds are preserved.
Without `--require-table`, selection is unchanged. This does not change either
prompt or restrict the response to table content: the full page is still evaluated
and visualized. Use a separate output directory for a table-only report so that
previously saved non-table results are not included by the report rebuild.

### Three-way token probability statistics

Every scored response token has one `token_category`:

- `formatting`: pure whitespace, common Markdown markers (headings, lists,
  emphasis, fences, link delimiters, pipe-table separators), or HTML tags.
  Tag spans are recognized in the full decoded text, so a BPE piece such as
  `td` inside `<td>` is still formatting, not a body word.
- `body`: remaining document content, including table cell text and LaTeX math.
  This category includes correct and incorrect ordinary text.
- `mutation`: any token aligned to the **whole edited word's GT span**, including
  a model readback of the original spelling or another incorrect spelling.

Tokens containing both syntax/whitespace and content are counted once as `body`,
or as `mutation` if they overlap an edited word. They carry
`mixed_format_content=true`. The existing correctness labels are a separate axis;
the three categories do not replace them. Math and inline-code contents are not
treated as Markdown punctuation. Formatting detection covers common document
syntax, not arbitrary malformed Markdown. `category_surface_matches_response`
reports whether the decoded BPE pieces reproduce the saved response text.

When a release's `changes` only contains `ocr_ans`, `origin_ans`, and `bbox`, the
probe resolves the full edited word in the GT before alignment. A valid supplied
`markdown_span` takes precedence; otherwise the match must be unique and
whole-word. Ambiguous/missing words are listed as unresolved, not assigned an
arbitrary occurrence. Bboxes are not used as text offsets. Their unlocated tokens
cannot be identified as mutation tokens. Deleted words have no response token
and therefore no token probability to average.

New runs write the annotations and per-sample statistics immediately. Existing
results can be re-annotated without inference or prompt changes:

```bash
python -u -m qwen_mm_token_probe.privileged_probe \
  --output-dir /path/to/existing-probe-output \
  --rebuild-report-only
```

Rebuild preserves saved `result.json` inference records, and regenerates exports
and the existing report. The sample report now shows a three-row probability
summary and category labels in the token detail table. Output files:

- `token_probabilities.csv`: all tokens, with `token_category` and unchanged scores.
- `token_category_summary.json` / `.csv`: global three-category statistics.
- `token_category_sample_summary.csv`: three rows per sample, one for each category.
- `samples/<sample>/token_category_summary.json` / `.csv`: per-sample statistics;
  JSON also lists unresolved mutation records.

Each category reports token count, mean `p_original` (student), mean `p_teacher`,
mean changes in probability/log-probability, and the proportion with
`p_teacher < p_original`. The global mean is **token-weighted**, not a mean of
sample means. A category with no tokens has null means/rates, not zero scores.
These descriptive statistics include all tokens and do not use the separate
teacher-audit correctness or student-probability gates. Probability suppression
alone is not a claim that the supervision is harmful.

Teacher-signal quality statistics are written to a separate page so the existing
sample browser and per-token visualization remain unchanged:

- `teacher_signal_audit.html`: mutation-only four-quadrant audit of correct and
  incorrect mutation readbacks versus increasing/decreasing privileged
  log-probability;
- `teacher_signal_audit.json`: aggregate counts, rates, threshold sweep, error
  types, Teacher Top-1 relations, and per-sample summaries;
- `teacher_signal_mutations.csv`: one row per annotated synthetic mutation,
  including the full associated response-token span and the selected-threshold
  signal class;
- `teacher_signal_tokens.csv`: supporting subtoken rows only for annotated
  mutation spans; ordinary response tokens are not included;
- `teacher_signal_sample_summary.csv`: one audit row per sample.

Correct student response tokens have a second, independent teacher-rejection
audit. It includes `token_label=correct` and, by request, formatting tokens:

- `correct_token_teacher_rejection.html`: standalone token-level inspection;
- `correct_token_teacher_rejection.json`: aggregate, group, sample, and
  threshold-sweep metrics;
- `correct_token_teacher_rejection.csv`: every correct/formatting candidate in
  its original order, including whether it passes the optional student response
  probability gate;
- `correct_token_teacher_rejection_sample_summary.csv`: one row per sample.

A third, standalone audit expands the same four-quadrant definition used by
`teacher_signal_audit` to every response content token whose correctness can be
decided by GT alignment:

- `token_teacher_signal_quadrants.html`: correct/incorrect token by Teacher
  increase/decrease matrix, threshold sweep, error-type breakdown, all token
  details, and per-sample metrics;
- `token_teacher_signal_quadrants.json`: aggregate metrics, ungated comparison,
  threshold sweep, error-type breakdown, and sample summaries;
- `token_teacher_signal_quadrants.csv`: every response token in original order,
  including excluded formatting/unknown rows, its quadrant, and probability-gate
  status;
- `token_teacher_signal_quadrants_sample_summary.csv`: one row per sample.

The four active classes are `correct_reinforced`,
`harmful_correct_suppressed`, `harmful_wrong_promoted`, and
`wrong_suppressed`. Signals with `abs(delta_logp) <= threshold` are reported as
neutral. Formatting tokens are excluded because normalization cannot verify
their correctness. Missing GT characters have no response token, so they are
reported separately and cannot enter a token quadrant.

The audit reports four separate notions rather than one ambiguous rejection
rate: any `delta_logp < 0`, suppression beyond
`delta_logp < -teacher_signal_threshold`, Teacher Top-1 using a different token
ID, and Teacher Top-1 decoding to a different surface. Formatting tokens are
included in a separate group, but their correctness is not character-alignment
verified because OCR normalization removes formatting syntax and whitespace.
Use `--student-response-min-probability 0.95` for high-confidence candidates
with `p_original >= 0.95`, or `--student-response-max-probability 0.95` for
low-confidence candidates with `p_original < 0.95`. Supplying both selects the
half-open interval `min <= p_original < max`. This gate applies to both the
correct-token rejection audit and the full-token four-quadrant audit.
`p_original` is the probability assigned to the actual student response token,
not the maximum probability over the vocabulary. Both audits retain ungated
metrics, and their CSV files retain excluded candidates for auditability. These
are report-only filters and can be changed with `--rebuild-report-only` without
another model forward. For the gated correct/formatting candidates, the primary
teacher harmful-signal rate uses `delta_logp < -teacher_signal_threshold`; raw
probability decreases and Teacher Top-1 changes remain separate diagnostics.

Only annotated mutation words are included; ordinary response words never enter
the audit denominator. Each mutation is counted once even when its readback has
multiple tokenizer tokens. `relation=expected` is correct, while
`opposite_variant` and `other` are incorrect. The signal is the delta-logp of
the first associated response token, which avoids allowing later prefix-driven
subtokens to dominate the decision. The default active-signal rule is
`abs(delta_logp) > 0.05`. Deleted mutations are unscored because the fixed
response-ID sequence contains no corresponding token.

For each response token, the table shows its probability and rank under the
original image+prompt context, the probability and rank of that exact same token
under GT teacher forcing, both conditions' Top-1/Top-2 decoded candidates, and
signed `delta_p`/`delta_logp = teacher - original`. Report rebuilding validates
that every row index and token ID still matches the generated `response_ids`;
it never sorts or filters the sequence. Existing results can be rendered with
this layout without another model forward:

Mutation metadata is visualized without filtering the response sequence. The
report highlights each mutation span in Ground Truth and its aligned token span
in the model Response, shows `origin_ans`, image/GT `ocr_ans`, the model readback,
and the aligned tokens' original/teacher probabilities and deltas. The same
`mutation_id` is attached to the corresponding rows in the complete token table.

```bash
qwen-mm-privileged-probe \
  --output-dir outputs/arxiv_confusable_privileged_probe_v1 \
  --teacher-signal-threshold 0.05 \
  --student-response-min-probability 0.95 \
  --rebuild-report-only
```

Replace the minimum option with `--student-response-max-probability 0.95` to
audit the complementary low-confidence group.
