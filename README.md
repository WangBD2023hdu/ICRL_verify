# Qwen Multimodal Token Probability Probe

This project runs Hugging Face multimodal inference on both the original image and a degraded copy, then teacher-forces each generated answer twice:

1. under the original image condition
2. under the masked or degraded image condition

It exports per-token probabilities for both generated responses, so you can inspect how the image perturbation changes the response itself and the probability of each generated token.

## Install

Qwen3.5 support currently requires a recent `transformers` build.

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

## Outputs

The output directory contains:

- `original.png`: normalized RGB input image
- `masked.png`: masked or degraded image used for the second forward pass
- `generated.txt`: answer generated from the original image
- `masked_generated.txt`: answer generated from the masked/degraded image
- `token_probabilities.csv`: token probabilities for the original-image response under both image conditions
- `word_probabilities.csv`: word/text-unit scores for the original-image response
- `masked_response_token_probabilities.csv`: token probabilities for the masked-image response under both image conditions
- `masked_response_word_probabilities.csv`: word/text-unit scores for the masked-image response
- `token_probabilities.json`: structured run metadata and both response score sets
- `token_probabilities.png`: compact probability comparison chart for the original-image response
- `masked_response_token_probabilities.png`: compact probability comparison chart for the masked-image response
- `token_probabilities.html`: readable token table for the original-image response
- `masked_response_token_probabilities.html`: readable token table for the masked-image response
- `word_probabilities.html`: readable word/text-unit table for the original-image response
- `masked_response_word_probabilities.html`: readable word/text-unit table for the masked-image response

## Method

The script builds the same multimodal chat prompt for the original image and the masked/degraded image, then calls `model.generate(...)` on each. For each generated response, it keeps the generated token ids, appends those exact token ids to the prompt, and runs two teacher-forcing forward passes. For each generated token `x_t`, it reads `softmax(logits[t - 1])[x_t]`.

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
