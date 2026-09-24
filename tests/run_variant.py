"""Runs one decoding variant on a tiny random Qwen2.5-VL (CPU) and prints JSON.

usage: python run_variant.py {orig|new} <src_dir> <methods> [extra new-argv ...]
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
which, src_dir, methods = sys.argv[1], sys.argv[2], sys.argv[3]
extra = sys.argv[4:]
sys.path.insert(0, str(HERE))
sys.path.insert(0, src_dir)
sys.path.insert(1, str(HERE.parent))  # generate_chair.py (MCoT host file)

import torch
from tiny_qwen import make_tiny, make_inputs

torch.use_deterministic_algorithms(True)


class Tok:
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{i}>" for i in ids)


class Proc:
    tokenizer = Tok()

    def batch_decode(self, t, **kw):
        return [self.tokenizer.decode(t[0].tolist())]


model = make_tiny(seed=0)
inputs = make_inputs(grid=(1, 8, 8))  # 16 image tokens
common = ["--input", "x", "--output", "x", "--model_id", "x", "--image_root", "x",
          "--methods", methods, "--max_visual_tokens", "8", "--sgrs_target_layers", "1,2",
          "--activation_threshold", "0.5"]

if which == "orig":
    import generate_chair_all as G
    from ott_qwen import OTTConfig, QwenOTTController
    from saliency_qwen import LocoREConfig, LocoREController, SaliencyConfig
    ms = G.parse_methods(methods)
    ott = QwenOTTController(model, OTTConfig(max_visual_tokens=8)) if "ott" in ms else None
    if ott: ott.install()
    loc = LocoREController(model, LocoREConfig()) if "locore" in ms else None
    if loc: loc.install()
    text, gate, sal = G.generate_one(model, Proc(), inputs, ms, 12, 0.75, -1, 0.5, ott, loc,
                                     SaliencyConfig(target_layers="1,2"), seed=7)
    ids = [g["selected_token_id"] for g in gate]
    sal_scores = [c["saliency"] for s in sal for c in s["candidates"]]
    print(json.dumps({"ids": ids, "sal": sal_scores}))
else:
    import generate_chair_all as G
    args = G.resolve_args(G.build_parser().parse_args(common + extra))
    ms = G.parse_methods(methods)
    aveg, ott, loc, cfg, router = G.build_components(model, args, ms)
    text, gate, sal, cost, ids = G.generate_one(model, Proc(), inputs, ms, 12, aveg, ott, loc, cfg, router, seed=7)
    sal_scores = [c["saliency"] for s in sal for c in s["candidates"]]
    print(json.dumps({"ids": ids, "sal": sal_scores, "cost": cost,
                      "risk": [round(g["risk"], 3) for g in gate],
                      "retraced": [g["retraced"] for g in gate], "sgrs": [g["sgrs"] for g in gate],
                      "locore_calls": loc.applied_calls if loc else None,
                      "svc_apps": ott.svc_applications if ott else None}))
