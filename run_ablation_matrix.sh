#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE_ROOT:?Set IMAGE_ROOT first}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
MODEL_ID="${MODEL_ID:-yfan1997/GRIT-20-Qwen2.5-VL-3B}"
INPUT="${INPUT:-coco_images.json}"
GPU="${GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

run_one () {
  local name="$1"
  local methods="$2"
  echo "===== $name: $methods ====="
  MODEL_ID="$MODEL_ID" IMAGE_ROOT="$IMAGE_ROOT" INPUT="$INPUT" \
  NUM_SAMPLES="$NUM_SAMPLES" GPU="$GPU" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  METHODS="$methods" OUTPUT="results/${name}_${NUM_SAMPLES}.json" \
  bash run_mcot_ott_saliency.sh
}

run_one mcot "mcot"
run_one ott "ott"
run_one saliency "sgrs,locore"
run_one mcot_ott "mcot,ott"
run_one mcot_saliency "mcot,sgrs,locore"
run_one ott_saliency "ott,sgrs,locore"
run_one all "mcot,ott,sgrs,locore"
