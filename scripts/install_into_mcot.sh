#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-/data/o_nejati/MCoT-hallucination}"

if [[ ! -f "$TARGET/generate_chair.py" || ! -f "$TARGET/chair.py" ]]; then
  echo "ERROR: $TARGET does not look like the MCoT-hallucination repository." >&2
  exit 1
fi

cp "$HERE/src/ott_qwen.py" "$TARGET/ott_qwen.py"
cp "$HERE/src/saliency_qwen.py" "$TARGET/saliency_qwen.py"
cp "$HERE/src/generate_chair_all.py" "$TARGET/generate_chair_all.py"
cp "$HERE/scripts/check_env.py" "$TARGET/check_all_env.py"
cp "$HERE/scripts/run_full.sh" "$TARGET/run_mcot_ott_saliency.sh"
cp "$HERE/scripts/run_ablation_matrix.sh" "$TARGET/run_ablation_matrix.sh"
cp "$HERE/scripts/evaluate_chair.sh" "$TARGET/evaluate_all_chair.sh"
chmod +x "$TARGET/run_mcot_ott_saliency.sh" "$TARGET/run_ablation_matrix.sh" "$TARGET/evaluate_all_chair.sh"

echo "Installed integration files into: $TARGET"
echo "Original generate_chair.py/chair.py were not overwritten."
