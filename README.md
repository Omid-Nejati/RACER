# MCoT + OTT + LVLMs-Saliency (Qwen2.5-VL / GRIT integration)

This repository is an **overlay** for the existing `ASGO-MM/MCoT-hallucination`
checkout.  It adds the previous OTT Qwen port plus a Qwen2.5-VL port of the
ICLR 2026 paper **Hallucination Begins Where Saliency Drops**.

It intentionally does **not** copy or replace third-party repository source.
The original MCoT `generate_chair.py`, `chair.py`, `chair.pkl`, model, and COCO
data remain the host environment.

## Novelties (see NOVELTY.md)

```bash
# previous plain stacking (bit-identical to the old integration)
PRESET=port METHODS=mcot,ott,sgrs,locore bash run_mcot_ott_saliency.sh
# all novelties: AVEG + TRACE + SGRS+ + LocoRE+ + risk router
PRESET=ours METHODS=mcot,ott,sgrs,locore bash run_mcot_ott_saliency.sh
# paper tables: A = stacking, B = "ours minus X"
TABLE=both NUM_SAMPLES=500 bash run_ablation_matrix.sh
# CPU tests on a tiny random Qwen2.5-VL (no weights needed)
cd tests && python -m pytest -q
```

## What is combined

At every decoding step the effective pipeline is:

1. **OTT (CRC + SVC)** modifies Qwen decoder MLP activations using visual
   references from the prompt.
2. **MCoT activation decoding** computes/uses the visual-entropy penalty and
   conditionally adjusts the next-token logits.
3. **SGRS** takes Top-K candidates from those adjusted logits, computes
   gradient-aware attention saliency for candidates, and rejects candidates
   below the adaptive threshold.
4. **LocoRE** reinforces attention to the most recent generated output tokens
   on the next forward pass.

All components can be independently enabled for ablations.

## Source-method notes

LVLMs-Saliency defines saliency as attention multiplied by its gradient,
followed by head aggregation/L2 normalization.  SGRS uses Top-K candidates,
an adaptive threshold based on recent accepted-token saliency, and a
max-saliency fallback.  The public rejection-sampling script uses defaults
`K=5`, `R=3`, `alpha=0.6`, history window `W=5`, and target layers `6,7,8`.
The paper's Qwen2-VL ablation reports `beta=0.20` for LocoRE.

Upstream sources:
- https://github.com/ASGO-MM/MCoT-hallucination
- https://github.com/Fazhan-cs/OTT
- https://github.com/zhangbaijin/LVLMs-Saliency
- https://arxiv.org/abs/2601.20279

## Important reproducibility caveats

### 1. Qwen2-VL -> Qwen2.5-VL / GRIT is a port

The saliency paper's public Qwen code targets Qwen2-VL, while MCoT/GRIT here
uses `Qwen2_5_VLForConditionalGeneration`.  This overlay is therefore a
method-level port, not byte-for-byte execution of the authors' model fork.

### 2. SGRS predictor-row correction

The public `do_reject_sample.py` appends candidate `c`, forms the candidate loss
from `outputs.logits[:, -2, :]`, but then reads the saliency at the **last**
attention-query row.  In a standard causal decoder that last row cannot affect
the already-computed likelihood of itself.  The default here is therefore:

```text
--saliency_query_mode predictor
```

which reads the preceding causal query row (`T-2`) that actually produces the
candidate logits.  For diagnostic comparison, `--saliency_query_mode repo`
reproduces the public row choice.

### 3. Portable LocoRE

The paper multiplies attention weights to recent output keys by `(1 + beta)`.
To avoid replacing Transformers' Qwen attention source, this overlay adds
`log(1 + beta)` to those key positions in the additive causal mask.  After
softmax this is the **renormalized equivalent** of the gain.  It is portable
across the supported Qwen2.5-VL host version but is not a byte-identical copy of
the authors' custom Qwen2-VL attention code.

For a paper, describe it as a **Qwen2.5-VL port/adaptation** unless you also
validate against the authors' exact Qwen2-VL implementation.

## Install into your current server repository

Your host repository is:

```bash
/data/o_nejati/MCoT-hallucination
```

Extract this overlay somewhere, then:

```bash
cd MCoT_OTT_LVLM_Saliency
bash scripts/install_into_mcot.sh /data/o_nejati/MCoT-hallucination
```

This installs only:

```text
ott_qwen.py
saliency_qwen.py
generate_chair_all.py
check_all_env.py
run_mcot_ott_saliency.sh
run_ablation_matrix.sh
evaluate_all_chair.sh
```

It does not overwrite the original MCoT files.

## Environment

Use the **same `myvenv` that runs MCoT/Qwen2.5-VL**.  Do not install the OTT or
LVLMs-Saliency original environments on top of it.

```bash
cd /data/o_nejati/MCoT-hallucination
source myvenv/bin/activate
python check_all_env.py
python -m pip check
```

Expected key import:

```text
Qwen2_5_VLForConditionalGeneration: OK
MCoT + OTT + Saliency imports: OK
```

The integration is designed around the environment you already repaired:
`transformers==4.49.0` and `huggingface-hub<1.0`.

## Data check

Do not use `examples/coco_images.example.json` for a 500-sample run; it has only
one record.  Use the repository's `coco_images.json` or your fixed 500-image
input.

```bash
python - <<'PY'
import json
with open('coco_images.json') as f:
    x = json.load(f)
print('input records:', len(x))
PY
```

Set your real COCO directory:

```bash
export IMAGE_ROOT=/real/path/to/coco/val2014
```

## First smoke test: Saliency only

Start with **1 image and a short generation** because SGRS performs gradient
forwards for candidate tokens and is much slower than the earlier MCoT+OTT
run.

```bash
cd /data/o_nejati/MCoT-hallucination
source myvenv/bin/activate

export MODEL_ID='yfan1997/GRIT-20-Qwen2.5-VL-3B'
export IMAGE_ROOT='/real/path/to/coco/val2014'
export INPUT='coco_images.json'
export NUM_SAMPLES=1
export MAX_NEW_TOKENS=32
export METHODS='sgrs,locore'
export OUTPUT='results/saliency_smoke.json'
export GPU=0

bash run_mcot_ott_saliency.sh
```

Successful LocoRE execution should finish with a non-zero line such as:

```text
LocoRE mask applications: ...
```

If it reports that no 4-D mask was observed, keep `eager` attention and verify
that your Transformers version is 4.49.0.

## Full combined smoke test

```bash
export METHODS='mcot,ott,sgrs,locore'
export NUM_SAMPLES=2
export MAX_NEW_TOKENS=64
export OUTPUT='results/mcot_ott_saliency_smoke.json'
bash run_mcot_ott_saliency.sh
```

## 500-image experiment

After the smoke test succeeds:

```bash
export INPUT='coco_images.json'
export NUM_SAMPLES=500
export MAX_NEW_TOKENS=256
export METHODS='mcot,ott,sgrs,locore'
export OUTPUT='results/mcot_ott_saliency_500.json'
bash run_mcot_ott_saliency.sh
```

Verify record count before evaluation:

```bash
python - <<'PY'
import json
p='results/mcot_ott_saliency_500.json'
with open(p) as f:
    x=json.load(f)
print('generated:', len(x))
assert len(x)==500
PY
```

## CHAIR evaluation

```bash
bash evaluate_all_chair.sh results/mcot_ott_saliency_500.json
```

Equivalent command:

```bash
python chair.py \
  --cap_file results/mcot_ott_saliency_500.json \
  --image_id_key image_id \
  --caption_key model_answer \
  --cache chair.pkl \
  --save_path results/mcot_ott_saliency_500.metrics.json
```

## Recommended ablation matrix

Use the exact same input JSON and seed for every setting:

| Name | Methods |
|---|---|
| MCoT | `mcot` |
| OTT port | `ott` |
| LVLMs-Saliency port | `sgrs,locore` |
| MCoT + OTT | `mcot,ott` |
| MCoT + Saliency | `mcot,sgrs,locore` |
| OTT + Saliency | `ott,sgrs,locore` |
| Full | `mcot,ott,sgrs,locore` |

For a small validation matrix:

```bash
export NUM_SAMPLES=10
export IMAGE_ROOT='/real/path/to/coco/val2014'
bash run_ablation_matrix.sh
```

Do not launch the 500-image seven-way matrix until the saliency runtime and GPU
memory are measured on 1-10 samples.

## Main hyperparameters

### MCoT
- `ACTIVATION_ALPHA=0.75`
- `ACTIVATION_THRESHOLD=0.5`
- `ACTIVATION_INFO_LAYER=-1`

### OTT Qwen port
- `CRC_LAMBDA=0.1111`
- `SVC_RATIO=0.06`

### LVLMs-Saliency Qwen2.5 port
- `SGRS_TOP_K=5`
- `SGRS_MAX_RESAMPLE=3`
- `SGRS_ALPHA=0.6`
- `SGRS_HISTORY_WINDOW=5`
- `SGRS_TARGET_LAYERS=6,7,8`
- `SALIENCY_QUERY_MODE=predictor`
- `LOCORE_BETA=0.20`
- `LOCORE_WINDOW=5`

## Output traces

Each output record retains the MCoT fields and adds:

```text
methods
activation_gate_trace
saliency_trace
```

`saliency_trace` records the adaptive threshold, candidate token IDs/text,
saliency scores, rejected candidates, selected candidate, and whether the
max-saliency fallback was used.  This is useful for debugging and ablations.
