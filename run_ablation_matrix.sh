#!/usr/bin/env bash
# Two tables for the paper:
#   Table A  (component stacking, previous integration)  --preset port
#   Table B  (novelty ablation: "ours minus X")           --preset ours
set -euo pipefail

: "${IMAGE_ROOT:?Set IMAGE_ROOT first}"
export NUM_SAMPLES="${NUM_SAMPLES:-10}"
export MODEL_ID="${MODEL_ID:-yfan1997/GRIT-20-Qwen2.5-VL-3B}"
export INPUT="${INPUT:-coco_images.json}"
export GPU="${GPU:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TABLE="${TABLE:-both}"   # a | b | both

run_one () {
  local name="$1" preset="$2" methods="$3" extra="${4:-}"
  echo "===== $name | preset=$preset | methods=$methods | $extra ====="
  PRESET="$preset" METHODS="$methods" EXTRA_ARGS="$extra" \
  OUTPUT="results/${name}_${NUM_SAMPLES}.json" bash run_mcot_ott_saliency.sh
}

if [[ "$TABLE" == "a" || "$TABLE" == "both" ]]; then
  run_one A_mcot           port "mcot"
  run_one A_ott            port "ott"
  run_one A_saliency       port "sgrs,locore"
  run_one A_mcot_ott       port "mcot,ott"
  run_one A_all_stacked    port "mcot,ott,sgrs,locore"
fi

if [[ "$TABLE" == "b" || "$TABLE" == "both" ]]; then
  ALL="mcot,ott,sgrs,locore"
  run_one B_ours                 ours "$ALL"
  run_one B_wo_router            ours "$ALL" "--router always"                               # N5
  run_one B_wo_aveg_multilayer   ours "$ALL" "--aveg_info_layers=-1"                          # N1.a
  run_one B_wo_aveg_candgate     ours "$ALL" "--aveg_candidate_k 1 --aveg_soft_gate false"    # N1.b/c
  run_one B_wo_aveg_phase        ours "$ALL" "--aveg_think_scale 1.0"                         # N1.d
  run_one B_wo_ott_schedule      ours "$ALL" "--ott_crc_layers all --ott_svc_layers all"      # N2.a
  run_one B_wo_ott_decode_only   ours "$ALL" "--ott_decode_only false"                        # N2.b
  run_one B_wo_ott_attn_memory   ours "$ALL" "--ott_visual_select uniform"                    # N2.c
  run_one B_wo_ott_mlp_space     ours "$ALL" "--ott_svc_ref_space residual"                   # N2.d
  run_one B_wo_ott_relevance     ours "$ALL" "--ott_relevance_gate false"                     # N2.e
  run_one B_wo_dual_saliency     ours "$ALL" "--sgrs_visual_lambda 0"                         # N3.a
  run_one B_wo_entropy_threshold ours "$ALL" "--sgrs_entropy_kappa 0"                         # N3.b
  run_one B_wo_calib_fallback    ours "$ALL" "--sgrs_fallback_eta none"                       # N3.c
  run_one B_wo_warmstart_fix     ours "$ALL" "--sgrs_exclude_neutral false"                   # N3.d
  run_one B_wo_adaptive_locore   ours "$ALL" "--locore_adaptive false"                        # N4.a
  run_one B_wo_locore_visual     ours "$ALL" "--locore_visual_beta 0"                         # N4.b
fi
