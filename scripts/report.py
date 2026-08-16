#!/usr/bin/env python3
"""Champion report — the Slide-6 autofill (house law: every number is MEASURED).

    python scripts/report.py --weights runs/M2/best.pt \
        --real_lq C:\\...\\NoisyLR --real_gt C:\\...\\GT \
        --val_split 0.05 --n_time 100 --out reports/M2.json --md reports/M2.md

* Reproduces the EXACT training holdout (last val_split of the sorted matched real
  pairs — identical code path as train.py), so the report numbers are honest
  generalization numbers, never train-set leakage.
* Model is driven through evaluate.py's own functions (load_model / preprocess /
  forward_one) — i.e. the same code the judges run.
* Emits mean/median PSNR & SSIM, edge/metrology metrics, worst-10% patch SSIM,
  LPIPS (if the lpips package is installed), and measured latency medians.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import load_model, forward_one, preprocess          # judge-identical model path
from data.dataset import RealPairsDataset, imread_gray            # identical pairing/loading
from utils.metrics import full_report                              # numpy metric suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--real_lq", required=True)
    ap.add_argument("--real_gt", required=True)
    ap.add_argument("--val_split", type=float, default=0.05,
                    help="must match training (default 0.05 = last 160 of 3200 pairs)")
    ap.add_argument("--n_time", type=int, default=100, help="images timed for latency")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    ds = RealPairsDataset(args.real_lq, args.real_gt)
    n_hold = max(1, int(len(ds.pairs) * args.val_split))
    holdout = ds.pairs[-n_hold:]
    print(f"[report] holdout: {n_hold} pairs (same split train.py held out)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.weights, dev)
    amp_ok = dev.type == "cuda"
    print(f"[report] device={dev}  cfg={cfg}")

    # warmup BEFORE any timing (review fix: first-call latency was being measured)
    lq0, _ = imread_gray(holdout[0][0])
    x0, _ = preprocess(lq0, cfg)
    with torch.no_grad():
        for _ in range(10 if dev.type == "cuda" else 1):
            forward_one(model, x0.to(dev), amp_ok, False)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    per_image, times = [], []
    for i, (lq_p, gt_p) in enumerate(holdout):
        lq, _ = imread_gray(lq_p)     # unit float32, identical loader as training
        gt, _ = imread_gray(gt_p)
        x, _ = preprocess(lq, cfg)
        with torch.no_grad():
            if i < args.n_time:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                y = forward_one(model, x.to(dev), amp_ok, False)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            else:
                y = forward_one(model, x.to(dev), amp_ok, False)
        pred = y.squeeze().float().clamp(0.0, 1.0).cpu().numpy()
        if pred.shape != gt.shape:                       # never silently resize (review fix):
            raise RuntimeError(f"[report] pred shape {pred.shape} != GT shape {gt.shape} on "
                               f"{os.path.basename(str(lq_p))} — a silent resize would corrupt "
                               f"every metric in this report; investigate instead")
        r = full_report(pred, gt)
        r["file"] = os.path.basename(str(lq_p))
        per_image.append(r)
        if (i + 1) % 25 == 0 or i + 1 == len(holdout):
            print(f"[report] {i + 1}/{len(holdout)}", flush=True)

    keys = [k for k in per_image[0] if k != "file"]
    agg = {k: {"mean": float(np.nanmean([r[k] for r in per_image])),
               "median": float(np.nanmedian([r[k] for r in per_image]))} for k in keys}
    worst = sorted(per_image, key=lambda r: r["psnr"])[:10]
    latency = {"n": len(times), "median_ms": float(np.median(times) * 1000) if times else None,
               "mean_ms": float(np.mean(times) * 1000) if times else None,
               "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}

    doc = {"weights": args.weights, "holdout": n_hold,
           "aggregate": agg, "latency": latency,
           "worst10_by_psnr": [{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in r.items()} for r in worst]}
    out = args.out or f"reports/{os.path.basename(os.path.dirname(args.weights))}.json"
    md = args.md or out.replace(".json", ".md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    lines = ["| metric | mean | median |", "|---|---|---|"]
    for k in keys:
        lines.append(f"| {k} | {agg[k]['mean']:.4f} | {agg[k]['median']:.4f} |")
    lines.append(f"| latency ms/img (median, {latency['device']}) | {latency['median_ms']:.2f} |  |")
    lines.append(f"| latency ms/img (mean) | {latency['mean_ms']:.2f} |  |")
    with open(md, "w", encoding="utf-8") as f:
        f.write("## DRISHTI-Net champion report — " + os.path.basename(args.weights) + "\n\n"
                + f"holdout: {n_hold} real pairs (train-excluded)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[report] -> {out}  |  {md}")


if __name__ == "__main__":
    main()
