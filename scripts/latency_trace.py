#!/usr/bin/env python3
"""Latency reconciliation trace — the Claude round-10 / GPT round-10 blocking item.

    python scripts/latency_trace.py --weights weights\\drishti_net.pt ^
        --real_lq C:\\...\\NoisyLR --n 800

FACTS that motivate it (all measured): report.py n=100 -> 47.58 ms median;
evaluate.py on 4 demo images -> 48.97 ms; evaluate.py on 3,200 -> 17.36 ms. Both
scripts time the SAME thing (per-image, torch.cuda.synchronize before t0 and after
forward_one, .to(dev) H2D inside the timed call) — so the difference is time-in-run
state, not methodology. This script records PER-IMAGE sync-timed latency over a long
run plus GPU SM clock samples (nvidia-smi, sampled at segment boundaries), then prints
segment statistics so the two hypotheses are distinguished by data:

  * clock-ramp (Claude): early segments slower, later segments converging lower;
    clock samples rising over the same segments -> report cold-start AND steady-state
    separately (never a single silent number).
  * flat from image 1: the ramp hypothesis is REJECTED and the implementations must be
    inspected again before any latency claim ships.

GPT's protocol is the second half of the output: overall median / mean / p95 / min
over all timed images (>=500 recommended = --n 800 default).
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import load_model, forward_one, preprocess, read_input, EXTS

SEGMENTS = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500), (500, 800)]
CLOCK_EVERY = 10


def gpu_clock_mhz():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            timeout=5, text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/drishti_net.pt")
    ap.add_argument("--real_lq", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.weights, dev)
    amp_ok = dev.type == "cuda"
    paths = sorted(p for e in EXTS for p in __import__("glob").glob(
        os.path.join(args.real_lq, e)))
    paths = paths[: args.n]
    if not paths:
        raise SystemExit(f"[trace] no inputs in {args.real_lq}")

    # warmup on the REAL first tensor (never timed) — identical to evaluate.py
    raw0 = read_input(paths[0])
    x0, _ = preprocess(raw0, cfg)
    with torch.no_grad():
        for _ in range(args.warmup):
            forward_one(model, x0.to(dev), amp_ok, False)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    rows, ms, clocks = [], [], []
    t_run = time.perf_counter()
    with torch.no_grad():
        for i, p in enumerate(paths):
            raw = raw0 if i == 0 else read_input(p)
            x, _ = preprocess(raw, cfg)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            forward_one(model, x.to(dev), amp_ok, False)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            ms.append(dt)
            clk = gpu_clock_mhz() if (i % CLOCK_EVERY == 0 and dev.type == "cuda") else ""
            clocks.append(clk)
            rows.append(f"{i},{os.path.basename(p)},{dt:.3f},{clk}")
    wall = time.perf_counter() - t_run
    ms = np.array(ms)

    print(f"[trace] {len(ms)} images timed, sync per image, "
          f"device={dev.type}, wall={wall:.1f}s")
    print("| segment (images) | median ms | mean ms | p95 ms | SM clock MHz (first/mid/last) |")
    print("|---|---|---|---|---|")
    for lo, hi in SEGMENTS:
        seg = ms[lo:hi]
        if len(seg) == 0:
            continue
        cs = [c for c in clocks[lo:hi] if isinstance(c, int)] or None
        cstr = f"{cs[0]}/{cs[len(cs)//2]}/{cs[-1]}" if cs else "n/a"
        print(f"| {lo}-{hi} | {np.median(seg):.2f} | {seg.mean():.2f} | "
              f"{np.percentile(seg, 95):.2f} | {cstr} |")
    print(f"\n[trace] OVERALL: median {np.median(ms):.2f} ms, mean {ms.mean():.2f}, "
          f"p95 {np.percentile(ms, 95):.2f}, min {ms.min():.2f} (n={len(ms)})")
    first, last = np.median(ms[:max(50, len(ms)//4)]), np.median(ms[len(ms)//2:])
    if last < 0.7 * first:
        print(f"[trace] VERDICT: declining trace (early {first:.1f} -> late {last:.1f} ms) "
              f"=> clock/state ramp CONFIRMED. Report cold-start and steady-state separately.")
    else:
        print(f"[trace] VERDICT: flat trace (early {first:.1f} vs late {last:.1f} ms) "
              f"=> ramp hypothesis REJECTED here; inspect implementations before claiming.")

    out = args.out or "reports/latency_trace.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("idx,name,ms,sm_clock_mhz\n" + "\n".join(rows) + "\n")
    print(f"[trace] -> {out}")


if __name__ == "__main__":
    main()
