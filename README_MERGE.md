# MCoT-hallucination + OTT port

Place these files in the root of `ASGO-MM/MCoT-hallucination`:

- `ott_qwen.py`
- `generate_chair_ott.py`
- `run_mcot_ott.sh`

The existing MCoT files remain unchanged.

## Why a port is required

The released OTT implementation is LLaVA-1.5/LlamaMLP-specific. It directly
replaces Llama MLP modules and patches LlamaModel internals. MCoT-hallucination
loads Qwen2.5-VL, so importing OTT's original `llava/` tree into the MCoT
environment is not a compatible merge.

This port keeps MCoT as the host and implements OTT-style CRC and SVC with
Qwen-compatible forward hooks.

## Environment

Use the SAME Python environment that already runs `generate_chair.py`.
Do not install OTT's `environment.yml` over it.

Quick import check:

```bash
python - <<'PY'
import torch, transformers
import generate_chair
import ott_qwen
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("imports OK")
PY
```

## Smoke test

```bash
export MODEL_ID=/path/to/GRIT-3B
export IMAGE_ROOT=/path/to/val2014
export INPUT=examples/coco_images.example.json
export OUTPUT=results/mcot_ott_smoke.json
export NUM_SAMPLES=2
export GPU=0

bash run_mcot_ott.sh
```

## Full generation

```bash
export NUM_SAMPLES=500
export OUTPUT=results/mcot_ott_500.json
bash run_mcot_ott.sh
```

Then evaluate with the original MCoT CHAIR evaluator:

```bash
python chair.py \
  --cap_file results/mcot_ott_500.json \
  --image_id_key image_id \
  --caption_key model_answer \
  --cache chair.pkl \
  --save_path results/mcot_ott_500.metrics.json
```

## Ablations

MCoT only:

```bash
python generate_chair.py ... --activation_alpha 0.75
```

OTT only:

```bash
python generate_chair_ott.py ... \
  --activation_alpha 0 \
  --crc --svc
```

Combined:

```bash
python generate_chair_ott.py ... \
  --activation_alpha 0.75 \
  --activation_threshold 0.5 \
  --crc --svc \
  --crc_lambda 0.1111 \
  --svc_ratio 0.06
```

For a strict OTT-only run, edit the runner to pass `--activation_alpha 0` and
note that zero still constructs MCoT's entropy context. For no MCoT computation
at all, pass no activation value by changing the parser default to `None`.

## Important reproducibility note

This is an architecture port, not a byte-for-byte execution of OTT. The OTT
release obtains CRC/SVC through LLaVA-specific internals. Qwen2.5-VL does not
expose those same interfaces, so the adapter computes visual references from
Qwen hidden states and applies equivalent normalized steering/context retracing
without hard-coding LLaVA's 4096-dimensional hidden size.
