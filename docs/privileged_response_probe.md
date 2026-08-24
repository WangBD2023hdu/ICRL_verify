# Fixed-response privileged-information probe

`qwen-mm-privileged-probe` is a local Hugging Face Transformers experiment. It
does not call an API, create a train/validation split, or fit a threshold.

For every page it performs exactly:

1. one `model.generate` call with `image + original prompt`, preserving the
   generated response ID sequence;
2. one teacher-forced forward after directly concatenating that ID tensor to
   the same multimodal prompt;
3. one teacher-forced forward after directly concatenating the identical ID
   tensor to the text-only prompt `GT + "\n\n" + "请转写上述文本"`.

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

For each response token, the table shows its probability and rank under the
original image+prompt context, the probability and rank of that exact same token
under GT teacher forcing, both conditions' Top-1/Top-2 decoded candidates, and
signed `delta_p`/`delta_logp = teacher - original`. Report rebuilding validates
that every row index and token ID still matches the generated `response_ids`;
it never sorts or filters the sequence. Existing results can be rendered with
this layout without another model forward:

```bash
qwen-mm-privileged-probe \
  --output-dir outputs/arxiv_confusable_privileged_probe_v1 \
  --rebuild-report-only
```
