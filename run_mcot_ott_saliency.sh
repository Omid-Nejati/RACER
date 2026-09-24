#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE_ROOT:?Set IMAGE_ROOT to the directory containing COCO val2014 JPG files}"
MODEL_ID="${MODEL_ID:-yfan1997/GRIT-20-Qwen2.5-VL-3B}"
INPUT="${INPUT:-coco_images.json}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
GPU="${GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
METHODS="${METHODS:-mcot,ott,sgrs,locore}"
PRESET="${PRESET:-port}"            # port = previous stacking, ours = all novelties
OUTPUT="${OUTPUT:-results/${PRESET}_${NUM_SAMPLES}.json}"
EXTRA_ARGS="${EXTRA_ARGS:-}"        # e.g. "--router always" for an ablation row

mkdir -p "$(dirname "$OUTPUT")"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPU" python generate_chair_all.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model_id "$MODEL_ID" \
  --image_root "$IMAGE_ROOT" \
  --num_samples "$NUM_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --methods "$METHODS" \
  --preset "$PRESET" \
  --attn_implementation eager \
  --seed "${SEED:-1994}" \
  --activation_alpha "${ACTIVATION_ALPHA:-0.75}" \
  --activation_info_layer="${ACTIVATION_INFO_LAYER:--1}" \
  --activation_threshold "${ACTIVATION_THRESHOLD:-0.5}" \
  --crc_lambda "${CRC_LAMBDA:-0.1111}" \
  --svc_ratio "${SVC_RATIO:-0.06}" \
  --sgrs_top_k "${SGRS_TOP_K:-5}" \
  --sgrs_max_resample "${SGRS_MAX_RESAMPLE:-3}" \
  --sgrs_alpha "${SGRS_ALPHA:-0.6}" \
  --sgrs_history_window "${SGRS_HISTORY_WINDOW:-5}" \
  --sgrs_target_layers "${SGRS_TARGET_LAYERS:-6,7,8}" \
  --saliency_query_mode "${SALIENCY_QUERY_MODE:-predictor}" \
  --locore_beta "${LOCORE_BETA:-0.20}" \
  --locore_window "${LOCORE_WINDOW:-5}" \
  --locore_layers "${LOCORE_LAYERS:-all}" \
  $EXTRA_ARGS
