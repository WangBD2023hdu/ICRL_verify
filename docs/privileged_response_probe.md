# Fixed-response privileged-information probe

`qwen-mm-api-privileged-probe` reads every row in an
`arxiv_confusable_v10_36_server` release. It does not create a train/validation
split or fit a threshold.

For every page, the command:

1. generates one OCR response from the image and the standard PDF-to-Markdown
   prompt;
2. teacher-forces the exact generated response token IDs under the original
   image condition;
3. teacher-forces the same IDs under a text-only prompt formed as
   `GT + "\n\n" + "请转写上述文本"`.

Both fixed-response calls must return a response suffix whose token IDs match
the generated IDs exactly. The sample fails instead of comparing mismatched
tokenizations.

```bash
DATA_BASE=/home/ma-user/work/wangbaode/03_innovate/ICRL_verify/exp_v2/data

pip install -e '.[api]'

tar -xzf "$DATA_BASE/arxiv_confusable_v10_36_server.tar.gz" \
  -C "$DATA_BASE"

export INF_API_KEY='...'

qwen-mm-api-privileged-probe \
  --base-url https://your-vllm-endpoint.example/v1 \
  --model qwen-4b \
  --dataset-root "$DATA_BASE/arxiv_confusable_v10_36_server" \
  --output-dir outputs/arxiv_confusable_privileged_probe_v1 \
  --max-tokens 4096 \
  --top-logprobs 5 \
  --request-interval-seconds 0.5 \
  --heartbeat-seconds 30 \
  --no-verify-tls
```

The aggregate `report.html` shows the original-versus-privileged probability
scatter, signed `delta_logp = teacher - original`, mutation-site results, and
tokens for which the privileged condition prefers another Top-1 token. Each
sample directory contains its input image, response, response IDs, GT, exact
privileged prompt, token CSV, mutation CSV, JSON checkpoint, and HTML report.
