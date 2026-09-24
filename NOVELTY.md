# MedicalHall — Novelties over MCoT + OTT + LVLMs-Saliency

The previous version of this repository **stacked** three published methods on
Qwen2.5-VL / GRIT: MCoT activation decoding, OTT (CRC + SVC), and the saliency
method (SGRS + LocoRE). Stacking alone is an engineering contribution, not a
methodological one. This version adds one change per component plus a fusion
mechanism, so the method becomes a single **uncertainty-triggered cascade** in
which the cheap signal of one method decides when the expensive correction of
another is applied.

Everything is switchable. `--preset port` reproduces the previous pipeline
**token-for-token** (verified by `tests/check_port_equivalence.sh`), so earlier
results remain valid as the "naive stacking" baseline. `--preset ours` turns
every novelty on, and each novelty has its own flag for "ours minus X" rows.

| Part | Source method | Novelty name | Code |
|---|---|---|---|
| 1 | MCoT activation decoding | **AVEG** – Adaptive Visual-Entropy Gating | `src/mcot_plus.py` |
| 2 | OTT (CRC + SVC) | **TRACE** – Triggered, Relevance-gated, Attention-prioritised Context rEtracing | `src/ott_qwen.py` |
| 3 | SGRS | **SGRS+** – Grounding-aware, entropy-coupled saliency rejection | `src/saliency_qwen.py` |
| 4 | LocoRE | **LocoRE+** – Saliency-adaptive, visually-anchored reinforcement | `src/saliency_qwen.py` |
| 5 | (fusion) | **Hallucination-Risk Router** | `src/risk_router.py`, `src/generate_chair_all.py` |

---

## Part 1 — AVEG (on MCoT)

**Original.** MCoT computes a per-vocabulary visual entropy
`H[v]` by projecting the image-token hidden states of one layer through the LM
head, then applies `logits − α·H` whenever `H[argmax] > γ` (hard gate, greedy
token only, constant α).

**N1.a Multi-layer logit-lens entropy.**
`H[v] = (1/|L|) Σ_{l∈L} H_l[v]`, where intermediate layers are passed through
the final RMSNorm before the LM head (a proper logit lens; the original applies
the LM head to un-normalised states for any layer other than the last).
Flag: `--aveg_info_layers=-1,-6,-12`. Ablation: `B_wo_aveg_multilayer`.

**N1.b Candidate-expected entropy.** The gate reads
`Ĥ_t = Σ_{c∈TopK} p̂_c · H[c]` instead of `H[argmax]`. Motivation: SGRS later
*samples* from Top-K, so a gate that only inspects the greedy token can miss the
token that is actually emitted.
Flag: `--aveg_candidate_k 5`.

**N1.c Soft gate.** `g_t = σ((Ĥ_t − γ)/T)`, penalty `α·g_t·H`. Removes the
discontinuity at γ and yields a calibrated risk signal in [0,1] consumed by the
router (Part 5).
Flag: `--aveg_soft_gate true --aveg_gate_temperature 0.05`. Ablation (b+c):
`B_wo_aveg_candgate`.

**N1.d Phase-aware strength.** `α_t = α·s_phase`, with separate scales inside
`<think>` and `<answer>` (GRIT's output format). Reasoning text contains many
abstract words whose visual entropy is naturally high; penalising them as hard
as object words in the answer can derail the chain of thought.
Flags: `--aveg_think_scale 0.5 --aveg_answer_scale 1.0`. Ablation:
`B_wo_aveg_phase`.

## Part 2 — TRACE (on OTT)

**Original OTT (LLaVA-1.5).** CRC steering in early layers, one SVC retracing
event at a fixed layer (16 of 32); an entropy-triggered variant and a relevance
gate exist only as commented-out code. The previous Qwen port applied CRC *and*
SVC in all 36 layers to all positions.

**N2.a Layer-scheduled intervention.** Separate layer sets; `auto` maps OTT's
16/32 split to Qwen (CRC in layers 0–17, SVC at 18). Flags:
`--ott_crc_layers auto --ott_svc_layers auto`. Ablation: `B_wo_ott_schedule`.

**N2.b Decode-only steering.** Prompt rows (including every image token) are
never modified; only rows that predict generated tokens are steered. This keeps
the visual evidence that AVEG and SGRS read uncontaminated (unit-tested:
prompt hidden states are bit-identical with and without OTT).
Flag: `--ott_decode_only true`. Ablation: `B_wo_ott_decode_only`.

**N2.c Attention-prioritised visual memory.** The M retracing references are
the image tokens receiving the most attention from the last prompt rows
(`score_i = mean_{l,h,q∈tail} A^{l,h}_{q,i}`), instead of a uniform
`linspace` subsample. The same ranking is exported as LocoRE-V anchors (N4.b),
so one forward pass serves two methods.
Flag: `--ott_visual_select attention`. Ablation: `B_wo_ott_attn_memory`.

**N2.d Space-consistent retracing.** SVC references are the *MLP outputs* of
the visual tokens at that layer, i.e. the same space as the MLP output they are
blended into (the port used residual-stream states).
Flag: `--ott_svc_ref_space mlp`. Ablation: `B_wo_ott_mlp_space`.

**N2.e Relevance gate.** `r_i = r · (1 + cos(h_i, v̄))/2`, making retracing
stronger for positions already aligned with the image.
Flag: `--ott_relevance_gate true`. Ablation: `B_wo_ott_relevance`.

**N2.f Risk-triggered retracing.** SVC strength is a per-position gate `z_p`
written by the router. A row keeps the gate it had when its token was produced,
so the full-sequence recomputation used by the decoder stays self-consistent.
(Part of N5; ablated by `B_wo_router`.)

## Part 3 — SGRS+ (on saliency-guided rejection sampling)

**Original.** Candidate score `s_c = mean_k |A ⊙ ∂L/∂A|` over *previously
generated text* keys; accept if `s_c ≥ α·mean(recent accepted)`; fallback =
max-saliency candidate.

**N3.a Dual (text + visual) grounding saliency.** The same gradient-weighted
attention is also read on *image* keys. Both parts are made scale-free with
their own running means:
`s̃_c = (1−λ)·s^txt_c / μ_txt + λ·s^vis_c / μ_vis`.
The original only asks "is the token consistent with what I wrote?";
this also asks "is it grounded in the image?".
Flag: `--sgrs_visual_lambda 0.3`. Ablation: `B_wo_dual_saliency`.

**N3.b Entropy-coupled per-candidate threshold.**
`α_c = clip(α·(1 + κ·(H[c] − γ)), α_min, α_max)`: tokens that AVEG marks as
visually diffuse must clear a stricter saliency bar. This is the direct
coupling of MCoT's signal with the saliency test.
Flag: `--sgrs_entropy_kappa 0.5`. Ablation: `B_wo_entropy_threshold`.

**N3.c Calibrated fallback.** If every sample is rejected,
`c* = argmax_c log p̂_c + η·log s̃_c` instead of pure max-saliency, so a
near-zero-probability token cannot win on saliency alone.
Flag: `--sgrs_fallback_eta 1.0`. Ablation: `B_wo_calib_fallback`.

**N3.d Warm-start fix (minor; present as an implementation detail).** The
neutral placeholder score used before any output history exists (1.0, far
above real scores) is not written into the history; otherwise it inflates the
threshold for the next W steps.
Flag: `--sgrs_exclude_neutral true`. Ablation: `B_wo_warmstart_fix`.

## Part 4 — LocoRE+ (on local coherence reinforcement)

**Original.** Attention to the last W output tokens is multiplied by `1+β`,
constant β.

**N4.a Saliency-adaptive gain.** Per-row
`β_t = β · clip(μ_txt / s_{t−1}, 1, β_max/β)`, set only after a risky,
saliency-checked step: reinforcement grows exactly when saliency drops.
Flags: `--locore_adaptive true --locore_beta_max 0.6`. Ablation:
`B_wo_adaptive_locore`.

**N4.b Visual anchors (LocoRE-V).** The same additive log-gain
`log(1+β_v)` is applied to the top-k image tokens from N2.c, so the mechanism
reinforces both output coherence and visual grounding.
Flags: `--locore_visual_beta 0.1 --locore_visual_anchor_k 32`. Ablation:
`B_wo_locore_visual`.

(Engineering, not a claim: the bias is now built once per forward pass and
vectorised instead of a Python loop over rows in every layer; unit-tested equal
to the reference loop.)

## Part 5 — Hallucination-Risk Router (main contribution)

Per step,

```
r_t = w_H·g_t  +  w_S·σ((α − s̃_{t−1})/T_S)  +  w_C·(1 − p_max,t)
```

(weights renormalised over the signals that are available). Routing:

* `r_t < ρ` → AVEG-adjusted greedy step (one forward pass);
* `r_t ≥ ρ` → set OTT gate for this row and re-forward (N2.f), run SGRS+ on
  the corrected logits, and raise LocoRE's gain on the next row (N4.a).

Also, SGRS is skipped when `p_max ≥ 0.95` even on risky steps.

The claim for the paper: **the three methods expose complementary
hallucination-risk signals (visual entropy, saliency drop, confidence); using
them to route each other's corrections gives an intervention that is applied
only where needed.** `--router always` is the stacking ablation
(`B_wo_router`). Each output record now contains `cost`
(`forwards`, `retrace_steps`, `sgrs_steps`, `seconds`), and each run writes a
`*_cost.json`, so you can report a quality-vs-compute table directly.

**Calibrating ρ.** Run `--preset ours` on ~20 held-out images, look at the
`risk` values in `activation_gate_trace`, and pick ρ so 20–35 % of steps are
routed. Do not tune ρ on the evaluation images.

---

## Suggested contribution list (for the introduction)

1. A unified, training-free decoding framework that fuses visual-entropy,
   saliency and confidence into a per-token hallucination-risk score and uses it
   to trigger visual retracing and saliency-verified sampling only on risky
   tokens (N5, N2.f).
2. Grounding-aware saliency verification: dual text/visual saliency with an
   entropy-coupled, per-candidate acceptance threshold (N3.a, N3.b).
3. Adaptive visual-entropy gating with multi-layer logit-lens estimation and
   reasoning/answer phase awareness for reasoning LVLMs (N1).
4. A decode-only, attention-prioritised, space-consistent retracing scheme that
   shares its visual anchors with coherence reinforcement (N2.b–d, N4.b).

## Honest caveats

* The unit tests prove the code does what is described and that `port` equals
  the old pipeline. They **do not** show that any novelty improves CHAIR;
  only the ablation matrix on the real model can. Expect some rows to be
  neutral or negative; keep only what helps and report the rest honestly.
* Default hyper-parameters of `ours` are reasonable starting points, not tuned.
* On-disk results produced before this change are "naive stacking" results.
* Values beginning with `-` must be passed as `--aveg_info_layers=-1,-6,-12`
  (argparse limitation).
