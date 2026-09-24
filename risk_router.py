"""
Novelty 5 (core contribution): Hallucination-Risk Router.

The previous integration stacked MCoT, OTT, SGRS and LocoRE and ran every one
of them at every decoding step.  The three source methods, however, each
produce a signal that says *when* a token is at risk:

  * MCoT/AVEG   -> visual entropy of the candidate set (diffuse visual support)
  * SGRS        -> relative saliency of the last accepted token (saliency drop)
  * the LM head -> top-1 confidence

The router fuses them into one per-step risk score

    r_t = w_H * g_H(t) + w_S * g_S(t) + w_C * (1 - p_max(t))      (weights renormalised
                                                                   over available signals)
    g_S(t) = sigmoid((alpha_sgrs - s_rel(t-1)) / T_s)

and uses it to route the expensive interventions:

    r_t <  rho : AVEG-adjusted greedy step (one forward)
    r_t >= rho : OTT visual retracing on this row (second forward, N2.f)
                 + SGRS saliency-verified sampling (N3.*)
                 + stronger LocoRE on the next row (N4.a)

So the combination is not "A + B + C" but an uncertainty-triggered cascade in
which cheap signals of one method decide when the costly correction of another
method is spent.  mode="always" reproduces plain stacking for ablations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class RouterConfig:
    mode: str = "always"            # "always" | "risk"
    w_entropy: float = 0.5
    w_saliency: float = 0.3
    w_conf: float = 0.2
    risk_threshold: float = 0.5     # rho
    saliency_temperature: float = 0.1
    sgrs_alpha: float = 0.6
    confident_skip: float = 1.01    # skip SGRS if p_max >= this (1.01 = never)
    retrace: bool = True


class RiskRouter:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg

    def saliency_signal(self, last_relative: Optional[float]) -> Optional[float]:
        if last_relative is None or not math.isfinite(last_relative):
            return None
        t = max(self.cfg.saliency_temperature, 1e-6)
        z = (self.cfg.sgrs_alpha - last_relative) / t
        z = max(min(z, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-z))

    def risk(self, gate_value: Optional[float], last_relative: Optional[float], p_max: float) -> float:
        terms = []
        if gate_value is not None:
            terms.append((self.cfg.w_entropy, float(gate_value)))
        s = self.saliency_signal(last_relative)
        if s is not None:
            terms.append((self.cfg.w_saliency, s))
        terms.append((self.cfg.w_conf, 1.0 - float(p_max)))
        wsum = sum(w for w, _ in terms)
        if wsum <= 0:
            return 1.0
        return float(sum(w * v for w, v in terms) / wsum)

    def decide(self, risk: float, p_max: float) -> dict:
        if self.cfg.mode == "always":
            return {"risky": True, "retrace": False, "sgrs": True}
        risky = risk >= self.cfg.risk_threshold
        return {
            "risky": risky,
            "retrace": bool(risky and self.cfg.retrace),
            "sgrs": bool(risky and p_max < self.cfg.confident_skip),
        }
