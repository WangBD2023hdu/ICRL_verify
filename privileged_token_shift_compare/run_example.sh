#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODEL=${MODEL:-/home/ma-user/work/share_base_models/Qwen3.5/Qwen3.5-2B}
DATASET=${DATASET:-/inspire/sfs/project/inf-multimodal/public/wangbaode/03_innovate/01_datasets/CHAOS-Bench/sft_verl_md/verl_grpo/train_common_filtered.jsonl}
OUTPUT=${OUTPUT:-${SCRIPT_DIR}/outputs/chaos_bench_privileged_token_shift}
SAMPLE_COUNT=${SAMPLE_COUNT:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
DEVICE=${DEVICE:-auto}

echo "START privileged token comparison"
echo "model=${MODEL}"
echo "dataset=${DATASET}"
echo "output=${OUTPUT}"
echo "sample_count=${SAMPLE_COUNT} max_new_tokens=${MAX_NEW_TOKENS}"

python3 "${SCRIPT_DIR}/compare.py" run \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --prompt-key prompt \
  --image-key images \
  --privileged-key reward_model.ground_truth \
  --privileged-template '{privileged_text} 请复写上面的内容。' \
  --selection random \
  --sample-count "${SAMPLE_COUNT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.6 \
  --top-p 1.0 \
  --top-k 10 \
  --max-pixels 4194304 \
  --student-high 0.40 \
  --teacher-low 0.01 \
  --device "${DEVICE}"

echo "DONE report=${OUTPUT}"
echo "OPEN ${OUTPUT}/index.html"
