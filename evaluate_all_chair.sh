#!/usr/bin/env bash
set -euo pipefail

CAP_FILE="${1:?Usage: bash evaluate_all_chair.sh results/file.json}"
SAVE_PATH="${2:-${CAP_FILE%.json}.metrics.json}"

python chair.py \
  --cap_file "$CAP_FILE" \
  --image_id_key image_id \
  --caption_key model_answer \
  --cache chair.pkl \
  --save_path "$SAVE_PATH"
