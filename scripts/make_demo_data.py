#!/usr/bin/env python3
"""Generate synthetic stand-in 'wafer' images so the full pipeline can be
smoke-tested before the real KLA pairs arrive.

  python scripts/make_demo_data.py --n 40 --out demo_data
Creates: demo_data/gt/*.png (16-bit) and demo_data/lq/*.png (engine-degraded, 16-bit)
"""
import argparse
import os

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.degradation import DegradationEngine


def wafer_image(rng, size=512):
    img = rng.uniform(0.25, 0.45, (size, size)).astype(np.float32)  # substrate
    # line gratings (poly/metal)
    for _ in range(rng.integers(1, 4)):
        x0, y0 = rng.integers(0, size // 2, 2)
        w, h = rng.integers(size // 4, size // 2, 2)
        pitch = int(rng.integers(8, 24))
        val = rng.uniform(0.6, 0.9)
        if rng.random() < 0.5:                                       # vertical lines
            for x in range(x0, min(x0 + w, size), pitch):
                img[y0:min(y0 + h, size), x:min(x + pitch // 2, size)] = val
        else:                                                        # horizontal
            for y in range(y0, min(y0 + h, size), pitch):
                img[y:min(y + pitch // 2, size), x0:min(x0 + w, size)] = val
    # contact array
    cx, cy = rng.integers(size // 8, size // 2, 2)
    r = int(rng.integers(3, 6)); step = r * 3
    for yy in range(cy, min(cy + 9 * step, size - r), step):
        for xx in range(cx, min(cx + 9 * step, size - r), step):
            cv2.circle(img, (int(xx), int(yy)), r, float(rng.uniform(0.85, 1.0)), -1)
    # a few dark L-shapes
    for _ in range(rng.integers(2, 5)):
        x, y = rng.integers(0, size - 80, 2)
        img[y:y + int(rng.integers(30, 70)), x:x + 10] *= rng.uniform(0.3, 0.6)
        img[y:y + 10, x:x + int(rng.integers(30, 70))] *= rng.uniform(0.3, 0.6)
    img += rng.normal(0, 0.008, (size, size)).astype(np.float32)     # substrate texture
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="demo_data")
    args = ap.parse_args()
    gt_dir = os.path.join(args.out, "gt")
    lq_dir = os.path.join(args.out, "lq")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    eng = DegradationEngine(full=True, sem_artifacts=True, seed=7)
    for i in range(args.n):
        gt = wafer_image(rng, args.size)
        lq, _ = eng.degrade(gt)
        cv2.imwrite(os.path.join(gt_dir, f"wafer_{i:03d}.png"), (gt * 65535).round().astype(np.uint16))
        cv2.imwrite(os.path.join(lq_dir, f"wafer_{i:03d}.png"),
                    (np.clip(lq, 0, None) * 65535).round().astype(np.uint16))
    print(f"wrote {args.n} pairs -> {gt_dir} , {lq_dir} (uint16)")


if __name__ == "__main__":
    main()
