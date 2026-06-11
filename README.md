# Qwen Multimodal Token Probability Probe

This project runs Hugging Face multimodal inference on both the original image and a degraded copy, then teacher-forces each generated answer twice:

1. under the original image condition
2. under the masked or degraded image condition

It exports per-token probabilities for both generated responses, so you can inspect how the image perturbation changes the response itself and the probability of each generated token.

## Install

Qwen3.5 support currently requires a recent `transformers` build.
Infinity-Parser2-compatible image preprocessing also requires `qwen-vl-utils`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
qwen-mm-token-probe \
  --model-id Qwen/Qwen3.5-2B \
  --image /absolute/path/to/image.jpg \
  --prompt "Describe the image in detail." \
  --output-dir outputs/qwen_probe \
  --max-new-tokens 96 \
  --group-tokens word \
  --min-pixels 2048 \
  --max-pixels 16777216 \
  --image-patch-size 16 \
  --mask-ratio 0.35 \
  --patch-size 32 \
  --seed 7
```

## Mask Strategies

The default is the original hard random patch mask:

```bash
qwen-mm-token-probe ... \
  --mask-strategy patch \
  --mask-effect replace \
  --mask-fill mean \
  --mask-opacity 1.0
```

For document images, a softer degradation is often more realistic:

```bash
qwen-mm-token-probe ... \
  --mask-strategy patch \
  --mask-effect blur_fade \
  --mask-fill white \
  --mask-opacity 0.55 \
  --blur-radius 1.5 \
  --mask-ratio 0.35
```

Word-level masking can use external OCR or annotation boxes:

```bash
qwen-mm-token-probe ... \
  --mask-strategy word \
  --word-boxes words.json \
  --mask-effect blur_fade \
  --mask-fill white \
  --mask-opacity 0.6 \
  --word-padding 3
```

`words.json` may be any nested JSON containing boxes in one of these forms:

```json
[
  {"text": "Example", "bbox": [120, 240, 188, 260]},
  [210, 240, 280, 260],
  {"x": 300, "y": 240, "w": 60, "h": 20}
]
```

If `--mask-strategy word` is used without `--word-boxes`, the script runs a lightweight document-text heuristic that groups dark pixels into line-level word-like boxes. You can tune it with `--text-threshold`, `--word-gap`, and `--word-padding`.

For a larger model, pass another checkpoint:

```bash
qwen-mm-token-probe --model-id Qwen/Qwen3.5-9B ...
```

## Privileged GT Probe

If you want to test whether the original-image response is still likely when the
image is blank but a ground-truth answer is provided as privileged information,
put that answer at `GT.txt` inside the output directory and pass
`--privileged-info-file GT.txt`. The GT text is appended only to the
masked/degraded-image prompt; it is not treated as the response to score.

```bash
qwen-mm-token-probe \
  --model-id /home/ma-user/work/share_base_models/Infinity-Parser2/Infinity-Parser2-Flash \
  --image /absolute/path/to/image.png \
  --prompt "$PDF_PROMPT" \
  --output-dir outputs/qwen_blank_with_gt \
  --dtype bfloat16 \
  --trust-remote-code \
  --min-pixels 2048 \
  --max-pixels 16777216 \
  --image-patch-size 16 \
  --mask-strategy patch \
  --mask-effect replace \
  --mask-fill white \
  --mask-ratio 1.0 \
  --mask-opacity 1.0 \
  --privileged-info-file GT.txt \
  --skip-masked-generation
```

In this mode, `token_probabilities.csv` compares the original-image response
under the clean image (`p_original`) against the same response under the blank
image plus privileged GT (`p_masked`).

## Repeated Unique Inference

To repeatedly sample one image and prompt, saving only responses that have not
appeared before:

```bash
qwen-mm-unique-loop \
  --model-id /home/ma-user/work/share_base_models/Infinity-Parser2/Infinity-Parser2-Flash \
  --image /inspire/sfs/project/inf-multimodal/public/wangbaode/03_innovate/ICRL_verify/yanbao.jpg \
  --prompt "$PDF_PROMPT" \
  --output-dir outputs/qwen_unique_yanbao \
  --max-new-tokens 1024 \
  --no-do-sample \
  --temperature 0.0 \
  --top-p 1.0
```

The loop runs forever by default. Press `Ctrl+C` to stop it. Use `--num-runs N`
for a finite run. Use greedy decoding, as shown above, when you want behavior
closest to Infinity-Parser2. Use `--do-sample --temperature 0.7 --top-p 0.95`
when you intentionally want diverse repeated outputs.

Saved files:

- `unique_000001.txt`, `unique_000002.txt`, ...: unique responses
- `unique_000001.json`, `unique_000002.json`, ...: response metadata
- `manifest.jsonl`: one record per unique response
- `runs.jsonl`: one record per inference attempt, including duplicates

## Outputs

The output directory contains:

- `original.png`: normalized RGB input image
- `masked.png`: masked or degraded image used for the second forward pass
- `generated.txt`: answer generated from the original image
- `masked_generated.txt`: answer generated from the masked/degraded image, unless `--skip-masked-generation` is used
- `token_probabilities.csv`: token probabilities for the original-image response under both image conditions
- `word_probabilities.csv`: word/text-unit scores for the original-image response
- `masked_response_token_probabilities.csv`: token probabilities for the masked-image response under both image conditions, unless `--skip-masked-generation` is used
- `masked_response_word_probabilities.csv`: word/text-unit scores for the masked-image response, unless `--skip-masked-generation` is used
- `token_probabilities.json`: structured run metadata and both response score sets
- `token_probabilities.png`: compact probability comparison chart for the original-image response
- `masked_response_token_probabilities.png`: compact probability comparison chart for the masked-image response, unless `--skip-masked-generation` is used
- `token_probabilities.html`: readable token table for the original-image response
- `masked_response_token_probabilities.html`: readable token table for the masked-image response, unless `--skip-masked-generation` is used
- `word_probabilities.html`: readable word/text-unit table for the original-image response
- `masked_response_word_probabilities.html`: readable word/text-unit table for the masked-image response, unless `--skip-masked-generation` is used

## Method

The script builds the same multimodal chat prompt for the original image and the masked/degraded image, then calls `model.generate(...)` on each. For each generated response, it keeps the generated token ids, appends those exact token ids to the prompt, and runs two teacher-forcing forward passes. For each generated token `x_t`, it reads `softmax(logits[t - 1])[x_t]`.

Image inputs follow Infinity-Parser2's reference Transformers path: load as RGB
PIL, include `min_pixels` and `max_pixels` in the image message, call
`apply_chat_template(tokenize=False, enable_thinking=False)`, process vision
inputs with `qwen_vl_utils.process_vision_info(..., image_patch_size=16)`, then
call the processor with `do_resize=False`.

That gives two complementary views:

- original-image response: whether the answer generated from the clean image remains likely when the image is degraded.
- masked-image response: what the degraded image makes the model generate, and whether that degraded-image answer would also be likely under the clean image.

## Token Grouping

Many words are split into multiple model tokens. The token-level output keeps the exact teacher-forced probabilities:

```text
p(x_t | image, prompt, x_1 ... x_{t-1})
```

By default, `--group-tokens word` also groups subword tokens into word/text units. For each unit, the script reports:

- `image_dependency_logp`: `first_token_logp_original - first_token_logp_masked`. This is the primary word-level image-dependence score.
- `first_token_*`: probability/log probability of the first token in the unit, before later subword tokens make the rest of the word easy to predict.
- `sum_logp_*`: sum of token log probabilities, equivalent to the joint log probability of the full unit. This is useful for exact forced likelihood, but later subword tokens can dilute image-dependence interpretation.
- `mean_logp_*`: average token log probability. Keep this as an auxiliary normalization, not as the main image-dependence score.

The CSV includes `unit_type`, so you can filter to `unit_type=word` when you only want lexical words and not punctuation or whitespace.
