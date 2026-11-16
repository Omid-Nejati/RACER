# RACER directional GSV patch

Targets the public RACER `main` source layout inspected on November 16, 2026:

- `saliency_qwen.py`
- `src/saliency_qwen.py`
- `generate_chair_all.py` (root entrypoint only)

`src/generate_chair_all.py` is an older entrypoint, and is deliberately left unchanged.
Run experiments via the **root** `generate_chair_all.py`.

## Installation

```bash
cd RACER
git switch -c fix/gsv-centered-directional
# Put apply_racer_gsv_centered.py in repository root; put test_gsv_centered.py in tests/
python apply_racer_gsv_centered.py --dry-run
python apply_racer_gsv_centered.py
python -m compileall -q saliency_qwen.py src/saliency_qwen.py generate_chair_all.py
PYTHONPATH=. python -m unittest discover -s tests -p 'test_gsv_centered.py' -v
git diff --check
git diff -- saliency_qwen.py src/saliency_qwen.py generate_chair_all.py
```

## Change in method

For loss L = -log p(candidate), relative to pre-softmax attention logits z,

`dL/dz = A * (dL/dA - sum_over_all_keys(A * dL/dA))`

Supportive positive sensitivity: `relu(-dL/dz)`
Opposing sensitivity: `relu(dL/dz)`

Split signs **before** averaging across heads. Use one shared row normalization
for both channels and reject candidates without polarity-dominant evidence.
The `ours` preset enables this, `port` remains unsigned (original baseline).
Fallback uses the same acceptance gates and abstains (EOS) when necessary.

## Experimental run

Use your existing dataset, model, environment and flags. Set
`--methods "mcot,ott,sgrs,locore" --preset ours --attn_implementation eager`.
To always exercise SGRS during a tiny smoke test, add `--router always`.
Output saliency traces include `chosen_visual_opposing_saliency` and
`chosen_text_opposing_saliency`.

**Limitations:** This patch is a proposed research correction; CPU tests do not
establish Qwen2.5-VL compatibility or CHAIR/POPE accuracy. Compare against the
original branch with equal decoding settings. Do not update results claims
until GPU evaluation is complete.
