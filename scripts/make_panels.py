#!/usr/bin/env python3
"""Render slide-6 panels from the champion's real outputs — SPREAD selection
(Claude + GPT round-10: never let three below-mean frames silently pose as typical,
and never cherry-pick three best cases either).

    python scripts/make_panels.py --lq C:\\...\\NoisyLR --pred outputs\\restored_train ^
        --gt C:\\...\\GT --out docs\\panels

Measures PSNR/SSIM for EVERY held-out frame (same split rule as train.py), then picks:
  A = nearest-to-median PSNR   (the honest typical case)
  B = strongest PSNR           (what the model can do)
  C = 10th-percentile PSNR     (a challenging case, LABELED as such — not the minimum,
                                 which would risk showing a pathological outlier)
Each panel emits three PNGs (lq / pred / gt, display-normalized;
lq nearest-upscaled) and every panel's own measured PSNR/SSIM is printed + written to
panels_index.txt — those exact numbers go under the images on the slide.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import imread_gray
from utils.metrics import psnr, ssim


def to_vis(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, np.uint8)
    return ((x - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lq", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="docs/panels")
    ap.add_argument("--val_split", type=float, default=0.05)
    ap.add_argument("--exclude", default="",
                    help="comma-separated filename stems to drop from panel selection "
                         "(e.g. --exclude 003117 if its GT is near-blank/low-texture "
                         "and would mislead as the 'strongest' showcase — round-11 Claude check)")
    ap.add_argument("--b_texture_pctile", type=float, default=25.0,
                    help="B = strongest PSNR among frames at/above this Sobel-energy "
                         "percentile (round-13 rule: flat near-blank frames earn trivial "
                         "PSNR; excluded by measured rule, skipped frames printed). "
                         "0 disables (pure strongest).")
    args = ap.parse_args()
    import cv2

    def texture_of(a):
        """Sobel gradient energy — the flat-frame detector (round-13): near-blank GTs
        (black OR white field + tiny corner feature) earn trivially high PSNR and would
        masquerade as the 'strongest' showcase. Same operator family as the loss stack."""
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1)
        return float((gx ** 2 + gy ** 2).mean())

    names = sorted(f for f in os.listdir(args.gt) if f.endswith(".npy"))
    n_hold = max(1, int(len(names) * args.val_split))
    hold = names[-n_hold:]
    print(f"[panels] measuring {len(hold)} held-out frames for spread selection...")
    scored = []
    for name in hold:
        gt, _ = imread_gray(os.path.join(args.gt, name))
        pr, _ = imread_gray(os.path.join(args.pred, name))
        if pr.shape != gt.shape:
            raise SystemExit(f"[panels] pred {name} shape {pr.shape} != GT {gt.shape}")
        scored.append((psnr(pr, gt), ssim(pr, gt), name, texture_of(gt)))
    scored.sort()
    def _norm(t):
        # PowerShell quirk (round-12 bug hunt): an UNQUOTED comma list such as
        # --exclude 003117,003119 is parsed by PowerShell as an INTEGER ARRAY, which
        # strips leading zeros (python receives '3117,3119') and silently matches no
        # filename. Normalize leading zeros off BOTH sides so quoted and unquoted
        # forms behave identically.
        t = t.strip().lstrip("0")
        return t or "0"
    excl_raw = args.exclude
    excl = {_norm(e) for e in excl_raw.split(",") if e.strip()}
    excl_note = None
    if excl:
        before = len(scored)
        hit = sorted(s[2] for s in scored if _norm(os.path.splitext(s[2])[0]) in excl)
        scored = [s for s in scored if _norm(os.path.splitext(s[2])[0]) not in excl]
        dropped = before - len(scored)
        names = ", ".join(hit) if hit else excl_raw
        excl_note = (f"(selection excluded {dropped} frame(s): {names} "
                     f"— removed after visual inspection found the GT near-blank/low-texture; "
                     f"selection otherwise identical)")
        print(f"[panels] --exclude '{excl_raw}' dropped {dropped} frame(s): {hit}")
        if dropped == 0:
            print("[panels] WARNING: --exclude matched NO frame — check the stems "
                  "(or quote the list: --exclude \"003117,003119\")")
        if not scored:
            raise SystemExit("[panels] --exclude removed every frame")
    pvals = [s[0] for s in scored]
    med = float(np.median(pvals))
    # --- B rule (round-13): strongest AMONG TEXTURED frames. Frames at/below the
    # 25th-pctile Sobel-energy floor are INELIGIBLE for B — exclusion by measured
    # rule, not by eye. Every higher-PSNR frame skipped by the floor is printed.
    floor = None
    skipped = []
    skip_pcts = []
    if args.b_texture_pctile > 0:
        floor = float(np.percentile([s[3] for s in scored], args.b_texture_pctile))
        eligible = [s for s in scored if s[3] >= floor]
        if not eligible:
            raise SystemExit("[panels] texture floor left no eligible frame for B")
        b_pick = eligible[-1]
        skipped = [s for s in scored if s[3] < floor and s[0] > b_pick[0]]
        print(f"[panels] B texture floor: P{args.b_texture_pctile:g} Sobel energy = {floor:.6f} "
              f"-> B-eligible n={len(eligible)}/{len(scored)}")
        if skipped:
            texs_all = [s[3] for s in scored]
            skip_pcts = [(s[2], round(100.0 * float(np.mean([t < s[3] for t in texs_all])), 1))
                         for s in skipped]
            print(f"[panels] floor skipped {len(skipped)} higher-PSNR flat frame(s): "
                  f"{[(s[2], round(float(s[0]), 2)) for s in skipped[-5:]]}")
            print(f"[panels] skipped-frame texture percentiles: {skip_pcts} "
                  f"(floor = P{args.b_texture_pctile:g} — are flats borderline or deep-tail?)")
    else:
        b_pick = scored[-1]
    picks = {
        "A_typical": min(scored, key=lambda s: abs(s[0] - med)),
        "B_strongest": b_pick,
        "C_challenging": scored[max(0, int(0.10 * len(scored)))],
    }
    os.makedirs(args.out, exist_ok=True)
    lines = []
    for tag, (p, s, name, _tex) in picks.items():
        stem = os.path.splitext(name)[0]
        gt, _ = imread_gray(os.path.join(args.gt, name))
        lq, _ = imread_gray(os.path.join(args.lq, name))
        pr, _ = imread_gray(os.path.join(args.pred, name))
        cv2.imwrite(os.path.join(args.out, f"panel_{tag}_lq.png"),
                    cv2.resize(to_vis(lq), (gt.shape[1], gt.shape[0]),
                               interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(os.path.join(args.out, f"panel_{tag}_pred.png"), to_vis(pr))
        cv2.imwrite(os.path.join(args.out, f"panel_{tag}_gt.png"), to_vis(gt))
        line = f"{tag} = {stem}: measured psnr={p:.2f} dB ssim={s:.4f}"
        lines.append(line)
        print(f"[panels] {line}")
    lines.append(f"(holdout n={len(hold)}, median psnr={med:.2f} dB; PNGs display-normalized)")
    if floor is not None:
        skip_names = ", ".join(s[2] for s in skipped) if skipped else "none"
        pct_str = "; ".join(f"{n} tex=P{p}" for n, p in skip_pcts)
        lines.append(f"(B rule: strongest PSNR among frames >= P{args.b_texture_pctile:g} Sobel "
                     f"energy, floor={floor:.6f}; higher-PSNR flat frame(s) skipped: {skip_names}"
                     + (f"; their texture percentiles: {pct_str}" if pct_str else "") + ")")
    if excl_note:
        lines.append(excl_note)
    with open(os.path.join(args.out, "panels_index.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[panels] -> {args.out}/  (captions: use the printed measured values)")


if __name__ == "__main__":
    main()
