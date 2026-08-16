#!/usr/bin/env python3
"""LODO OOD probe — eval-only (no training, no GPU budget risk).

    python scripts/lodo_eval.py --weights runs\\M0r_v5\\best.pt,runs\\M3e_30k\\best.pt ^
        --real_gt C:\\...\\GT --kernel 3 --out reports\\lodo_k3.json --md reports\\lodo_k3.md

WHAT THIS MEASURES: the SAME held-out GT frames as train.py/report.py (last val_split
of sorted names) are synthetically degraded with a degradation kernel family that was
NEVER sampled train-side (`lodo_holdout`, item-19 verified) — all other parameters are
drawn from the full engine ranges. Every listed checkpoint sees THE IDENTICAL
synthesized inputs (they are generated once, up front), so per-image comparisons are
paired. Models are driven through evaluate.py's own load_model/preprocess/forward_one.

SCOPE (house law — say exactly what was measured): this is a synthetic-family OOD
proxy. Every candidate this probe has compared (M0r_v5, M3e, M2f) trained on REAL
pairs only (--no_syn), so NO synthetic family was in ANY candidate's training
distribution. Borrowing the engine's held-out kernel corner here therefore tests
generalization to an unseen synthetic degradation family — it does NOT exercise the
synthetic-arm LODO-corner-holdout mechanism of the training protocol (there is no
"corner" being withheld from a model that never saw synthetic data at all — round-11
Claude correction, adopted). It is evidence for OOD-robustness claims, not a
prediction of the KLA test set (which is OOD in image SOURCE).

Metric naming per MASTER_SPEC wording law: flat-region residual σ (lower = better),
SSIM = our Gaussian-window implementation (consistent internal comparator).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import load_model, forward_one, preprocess        # judge-identical path
from data.dataset import imread_gray
from data.degradation import DegradationEngine, KERNELS
from utils.metrics import (psnr, ssim, edge_region_psnr,
                           flat_region_artifact, patch_metrics)

IMG_EXT = (".npy", ".png", ".tif", ".tiff", ".bmp", ".jpg")
KERNEL_NAMES = {0: "INTER_LINEAR", 1: "INTER_CUBIC", 2: "INTER_AREA", 3: "INTER_LANCZOS4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="comma-separated checkpoint paths (identical inputs for all)")
    ap.add_argument("--real_gt", required=True)
    ap.add_argument("--val_split", type=float, default=0.05,
                    help="same holdout rule as train.py (last n of sorted names)")
    ap.add_argument("--n", type=int, default=0, help="cap holdout images (0 = all)")
    ap.add_argument("--kernel", type=int, default=3,
                    help="held-out kernel index (default 3 = LANCZOS4, never sampled train-side)")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.real_gt) if f.lower().endswith(IMG_EXT))
    if not names:
        raise SystemExit(f"[lodo] no images in {args.real_gt}")
    n_hold = max(1, int(len(names) * args.val_split))
    hold = names[-n_hold:]
    if args.n:
        hold = hold[: args.n]
    print(f"[lodo] holdout: {len(hold)} frames ({hold[0]} .. {hold[-1]}) — same rule as train.py")

    eng = DegradationEngine(scale=2, full=True, lodo_holdout={"kernel": args.kernel},
                            seed=args.seed)
    # --- self-audit: prove the corner is actually held out of the train-side draws
    freq = np.zeros(len(KERNELS), dtype=int)
    for _ in range(400):
        _, lab = eng.degrade(np.random.rand(64, 64).astype(np.float32), force_holdout=False)
        freq[lab["kernel"]] += 1
    print(f"[lodo] train-side kernel frequencies over 400 draws: {freq.tolist()} "
          f"(held-out kernel {args.kernel} MUST be 0 — item-19 semantics)")
    assert freq[args.kernel] == 0, "lodo_holdout broken: held-out kernel leaked"

    # --- synthesize the eval set ONCE (identical inputs for every checkpoint)
    eval_set = []
    for name in hold:
        gt, _ = imread_gray(os.path.join(args.real_gt, name))
        lq, lab = eng.degrade(gt, force_holdout=True)
        assert lab["kernel"] == args.kernel
        eval_set.append((name, gt, lq))
    print(f"[lodo] synthesized {len(eval_set)} LQ frames @ kernel={args.kernel} "
          f"({KERNEL_NAMES[args.kernel]}), full ranges, order_p=0.7, seed={args.seed}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = ["psnr", "ssim", "edge_psnr", "flat_sigma", "worst10_ssim"]
    doc = {"kernel": args.kernel, "kernel_name": KERNEL_NAMES[args.kernel],
           "holdout": [hold[0], hold[-1]], "n": len(eval_set), "models": {}}
    md = ["| model | psnr ↑ | ssim ↑ | edge_psnr ↑ | flat_sigma ↓ | worst10_ssim ↑ |",
          "|---|---|---|---|---|---|"]
    for w in args.weights.split(","):
        w = w.strip()
        model, cfg = load_model(w, dev)
        amp_ok = dev.type == "cuda"
        rows = []
        for i, (name, gt, lq) in enumerate(eval_set):
            x, _ = preprocess(lq, cfg)
            with torch.no_grad():
                y = forward_one(model, x.to(dev), amp_ok, False)
            pred = y.squeeze().float().clamp(0.0, 1.0).cpu().numpy()
            if pred.shape != gt.shape:                       # never silently resize
                raise RuntimeError(f"[lodo] pred shape {pred.shape} != GT {gt.shape} on {name}")
            _, wp = patch_metrics(pred, gt)
            rows.append({"psnr": psnr(pred, gt), "ssim": ssim(pred, gt),
                         "edge_psnr": edge_region_psnr(pred, gt),
                         "flat_sigma": flat_region_artifact(pred, gt),
                         "worst10_ssim": wp})
            if (i + 1) % 25 == 0 or i + 1 == len(eval_set):
                print(f"[lodo] {os.path.basename(w)}: {i + 1}/{len(eval_set)}", flush=True)
        agg = {k: {"mean": float(np.mean([r[k] for r in rows])),
                   "median": float(np.median([r[k] for r in rows]))} for k in keys}
        doc["models"][w] = agg
        tag = os.path.basename(os.path.dirname(w)) or w
        md.append("| " + tag + " | " + " | ".join(f"{agg[k]['mean']:.4f}" for k in keys) + " |")
        print(f"[lodo] {tag}: " + ", ".join(f"{k}={agg[k]['mean']:.4f}" for k in keys))

    out = args.out or f"reports/lodo_kernel{args.kernel}.json"
    mdp = args.md or out.replace(".json", ".md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    hdr = (f"# LODO OOD probe — held-out kernel {args.kernel} ({KERNEL_NAMES[args.kernel]}), "
           f"{len(eval_set)} frames ({hold[0]}..{hold[-1]}), seed {args.seed}\n"
           f"Synthetic-family OOD proxy (see script docstring for scope). "
           f"flat_sigma = flat-region residual sigma (lower = better); "
           f"ssim = Gaussian-window internal comparator.\n\n")
    with open(mdp, "w", encoding="utf-8") as f:
        f.write(hdr + "\n".join(md) + "\n")
    print(f"[lodo] -> {out}  |  {mdp}")


if __name__ == "__main__":
    main()
