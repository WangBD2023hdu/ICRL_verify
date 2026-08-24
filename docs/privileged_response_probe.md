# Fixed-response privileged-information probe

`qwen-mm-privileged-probe` is a local Hugging Face Transformers experiment. It
does not call an API, create a train/validation split, or fit a threshold.

For every page it performs exactly:

1. one `model.generate` call with `image + original prompt`, preserving the
   generated response ID sequence;
2. one teacher-forced forward after directly concatenating that ID tensor to
   the same multimodal prompt;
3. one teacher-forced forward after directly concatenating the identical ID
   tensor to this text-only prompt:

```text
请逐字逐符号转写下面边界标记之间的文档。转写不是翻译；不要改变任何字符。边界标记本身不要输出。

<<<DOCUMENT_START>>>
{完整 Markdown GT}
<<<DOCUMENT_END>>>
```

The complete Markdown GT is inserted verbatim. If it has no trailing newline,
one newline is added only to place `<<<DOCUMENT_END>>>` on its own line.

The response is never decoded and re-tokenized for either forward. Every
`result.json` records `response_ids_directly_concatenated=true` and
`response_text_retokenized=false`.

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
  --rebuild-report-only
```
