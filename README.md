# Qwen Multimodal Token Probability Probe

This project runs Hugging Face multimodal inference, then teacher-forces the generated answer twice:

1. with the original image
2. with a masked or degraded copy of the same image

It exports per-token probabilities for the generated text, so you can inspect how much the image perturbation changes the probability of each generated token.

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
- `generated.txt`: decoded answer generated from the original image
- `token_probabilities.csv`: token-level probabilities and log probabilities
- `token_probabilities.json`: structured run metadata and token scores
- `token_probabilities.png`: compact probability comparison chart
- `token_probabilities.html`: readable token table and color strips

## Method

The script first builds a multimodal chat prompt and calls `model.generate(...)` on the original image. It keeps the generated token ids, appends those exact token ids to the prompt, and runs two teacher-forcing forward passes. For each generated token `x_t`, it reads `softmax(logits[t - 1])[x_t]`.

That means the masked-image run scores the same generated answer under a different image condition instead of asking the model to generate a new answer.
