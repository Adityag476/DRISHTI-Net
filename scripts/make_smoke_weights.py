#!/usr/bin/env python3
"""Regenerate weights/drishti_net_smoke.pt against the CURRENT model code.

House law "nothing ships unrun" caught it: the bundled smoke checkpoint predated the
z_norm module and failed strict load. This script instantiates DRISHTINet with its
constructor defaults (the same defaults the README demo assumes) and saves the exact
train.py checkpoint layout, so `evaluate.py` strict-loads it forever:

    python scripts/make_smoke_weights.py
    python evaluate.py --input_dir demo/lq --output_dir demo/restored \
        --weights tests/smoke/drishti_net_smoke.pt   # <- must print the latency line

(GPT round-10: smoke weights live in tests/smoke/, NOT weights/ — the weights/ dir
holds exactly one model: the shipped champion drishti_net.pt. Zero ambiguity.)
"""
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.drishti_net import DRISHTINet

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tests", "smoke", "drishti_net_smoke.pt")


def main():
    sig = inspect.signature(DRISHTINet.__init__)
    cfg = {k: v.default for k, v in sig.parameters.items()
           if k != "self" and v.default is not inspect.Parameter.empty}
    model = DRISHTINet(**cfg)
    cfg = dict(cfg)
    cfg["use_log"] = True
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                "step": 0, "val_psnr": None, "val_ssim": None,
                "note": "smoke placeholder — random init, demo plumbing only "
                        "(real quality = weights/drishti_net.pt)"}, OUT)
    n = sum(p.numel() for p in model.parameters())
    print(f"[smoke] wrote {OUT} ({n:,} params, use_film={cfg['use_film']}, aux_dim={cfg['aux_dim']})")
    # hard proof: the exact judge path must strict-load it right now
    from evaluate import load_model
    m2, c2 = load_model(OUT, torch.device("cpu"))
    print(f"[smoke] strict-load OK via evaluate.load_model (cfg keys: {sorted(c2)})")


if __name__ == "__main__":
    main()
