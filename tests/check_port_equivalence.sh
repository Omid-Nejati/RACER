#!/usr/bin/env bash
# Verifies that --preset port reproduces the pre-novelty pipeline token-for-token.
# Needs a copy of the previous src/ files in $ORIG_SRC (e.g. from git history).
set -euo pipefail
cd "$(dirname "$0")"
ORIG_SRC="${ORIG_SRC:?set ORIG_SRC to a directory with the previous ott_qwen.py, saliency_qwen.py, generate_chair_all.py}"
for m in mcot ott sgrs,locore mcot,ott,sgrs,locore; do
  a=$(python run_variant.py orig "$ORIG_SRC" "$m" | tail -1)
  b=$(python run_variant.py new ../src "$m" --preset port | tail -1)
  python - "$a" "$b" "$m" <<'PY'
import json, sys
a, b = json.loads(sys.argv[1]), json.loads(sys.argv[2])
ok = a["ids"] == b["ids"] and a["sal"] == b["sal"]
print(f"{sys.argv[3]:24s} {'IDENTICAL' if ok else 'DIFFERENT'}")
sys.exit(0 if ok else 1)
PY
done
