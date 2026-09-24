#!/usr/bin/env bash
set -euo pipefail

# Example:
# MODEL_ID=/path/to/GRIT-3B \
# IMAGE_ROOT=/path/to/coco/val2014 \
# INPUT=examples/coco_images.example.json \
# OUTPUT=results/mcot_ott.json \
# bash run_mcot_ott.sh

: "${MODEL_ID:?Set MODEL_ID to your local GRIT/Qwen2.5-VL model path or HF id}"
: "${IMAGE_ROOT:?Set IMAGE_ROOT to COCO val2014}"
INPUT="${INPUT:-examples/coco_images.example.json}"
OUTPUT="${OUTPUT:-results/mcot_ott.json}"
GPU="${GPU:-0}"

mkdir -p "$(dirname "$OUTPUT")"

CUDA_VISIBLE_DEVICES="$GPU" python generate_chair_ott.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model_id "$MODEL_ID" \
  --image_root "$IMAGE_ROOT" \
  --num_samples "${NUM_SAMPLES:-10}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --activation_alpha "${ACTIVATION_ALPHA:-0.75}" \
  --activation_info_layer "${ACTIVATION_INFO_LAYER:--1}" \
  --activation_threshold "${ACTIVATION_THRESHOLD:-0.5}" \
  --crc \
  --svc \
  --crc_lambda "${CRC_LAMBDA:-0.1111}" \
  --svc_ratio "${SVC_RATIO:-0.06}"
