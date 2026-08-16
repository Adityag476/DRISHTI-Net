#!/usr/bin/env python3
"""Ladder harvest — reads runs/<rung>/best.pt and prints the adjudication table.

Gate rule (consensus): a rung becomes champion only if it beats the RUNNING BEST
on BOTH val PSNR and val SSIM (160 held-out real pairs, EMA, full-image).

    python scripts/harvest_ladder.py --runs runs
"""
import argparse
import glob
import os

import torch

ORDER = ["M0r", "M0", "M1", "M2", "M3", "M4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()

    rows = {}
    for bp in sorted(glob.glob(os.path.join(a.runs, "*", "best.pt"))):
        rung = os.path.basename(os.path.dirname(bp))
        ck = torch.load(bp, map_location="cpu", weights_only=False)   # our own ckpts
        rows[rung] = (float(ck.get("val_psnr", float("nan"))),
                      float(ck.get("val_ssim", float("nan"))),
                      int(ck.get("step", -1)))
    if not rows:
        print(f"no best.pt found under {a.runs}/<rung>/ — train first")
        return

    rungs = [r for r in ORDER if r in rows] + [r for r in rows if r not in ORDER]
    print(f"{'rung':<6}{'val PSNR':>10}{'val SSIM':>10}{'@step':>7}   verdict")
    best_p = best_s = -1e9
    winner = None
    for rung in rungs:
        p, s, st = rows[rung]
        if winner is None:
            verdict = "(baseline)"
            best_p, best_s, winner = p, s, rung
        else:
            ok = (p > best_p) and (s > best_s)
            verdict = "BEATS BEST -> new champion" if ok else "NO (discarded by gate)"
            if ok:
                best_p, best_s, winner = p, s, rung
        print(f"{rung:<6}{p:>10.2f}{s:>10.4f}{st:>7}   {verdict}")

    print(f"\nchampion: {winner}   {best_p:.2f} dB / {best_s:.4f} SSIM")
    print("next: python scripts/report.py --weights runs/%s/best.pt "
          "--real_lq <NoisyLR> --real_gt <GT>" % winner)


if __name__ == "__main__":
    main()
