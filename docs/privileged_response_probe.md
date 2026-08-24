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

The aggregate `report.html` compares original and privileged target-token
probabilities, signed `delta_logp = teacher - original`, entropy, rank, Top-1
transitions, GT alignment, and mutation sites. Each sample directory contains
the copied page image, GT, exact privileged prompt, response text, response IDs,
resumable partial checkpoint, token CSV, mutation CSV, JSON result, and HTML
report.
