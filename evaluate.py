#!/usr/bin/env python3
"""DRISHTI-Net — KLA i4C PS01 standalone evaluation script.

    python evaluate.py --input_dir <degraded_imgs> --output_dir <restored_out>

Zero manual edits required:
  * CUDA -> CPU fallback, FP16 with FP32 fallback
  * preprocessing read from the checkpoint config (fixed SCALE, never clips)
  * outputs cast to the exact input dtype / range
  * --tta (8-way self-ensemble) exists but is OFF by default (latency benchmark)
  * --weights defaults to weights/drishti_net.pt; a download URL can be baked in below
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch

WEIGHTS_URL = ""  # optional: direct-download URL for the checkpoint (Drive/HF link)
EXTS = ("*.png", "*.tif", "*.tiff", "*.bmp", "*.jpg", "*.jpeg", "*.npy")


def read_input(path):
    """Read image or .npy array; returns 2D numpy array with original dtype."""
    if path.lower().endswith(".npy"):
        arr = np.squeeze(np.load(path))
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected 2D array, got {arr.shape}")
        return arr
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def get_weights(path):
    if os.path.exists(path):
        return path
    if WEIGHTS_URL:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, path)
        return path
    raise FileNotFoundError(f"weights not found at {path} (set WEIGHTS_URL for auto-download)")


def load_model(weights, device):
    from models.drishti_net import DRISHTINet
    # weights_only=False: our checkpoint = dict(model_state_dict, config, step).
    # Only load checkpoints you produced or trust (a .pt can carry code).
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = DRISHTINet(**{k: cfg[k] for k in ("in_ch", "out_ch", "width", "z_dim",
                                              "use_film", "with_uncertainty", "aux_dim") if k in cfg})
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg


_AMP_FALLBACK = False   # runtime switch: fp16 failure on this GPU -> fp32 for the rest of the run


def preprocess(img, cfg):
    """Divisor semantics: normalize by norm_peak (65535/255/1.0). If the checkpoint
    does not record norm_peak, derive it from the input dtype (identical rule as
    data/dataset.imread_gray) — this is exactly equivalent to legacy scale=1.0.
    NEVER clip: speckle legitimately exceeds the GT range."""
    npk = cfg.get("norm_peak", None)
    if isinstance(npk, (int, float)) and npk:
        peak = float(npk)
    else:
        peak = 65535.0 if img.dtype == np.uint16 else (255.0 if img.dtype == np.uint8 else 1.0)
    x = img.astype(np.float32) / peak
    if cfg.get("use_log", True):
        x = np.stack([x, np.log1p(np.clip(x, 0, None))], 0)
    else:
        x = x[None]
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), peak


def forward_one(model, x, amp_ok, tta):
    """amp_ok: fp16 autocast on CUDA. On a RuntimeError under fp16, permanently fall back
    to fp32 for the remainder of the run (old-GPU safety) and retry this call once."""
    global _AMP_FALLBACK

    def single(inp):
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=bool(amp_ok) and not _AMP_FALLBACK):
            return model(inp).float()

    def run():
        if not tta:
            return single(x)
        outs = []
        for k in range(4):
            r = torch.rot90(x, k, (2, 3))
            outs.append(torch.rot90(single(r), -k, (2, 3)))
            outs.append(torch.rot90(single(torch.flip(r, [2])), -k, (2, 3)).flip([2]))
        return torch.stack(outs).mean(0)

    try:
        return run()
    except RuntimeError:
        if not (amp_ok and not _AMP_FALLBACK):
            raise
        _AMP_FALLBACK = True
        print("[eval] fp16 failed on this GPU -> fp32 fallback for the rest of the run")
        return run()


def main():
    ap = argparse.ArgumentParser(description="DRISHTI-Net evaluation (KLA PS01)")
    ap.add_argument("--input_dir", "--input", dest="input_dir", required=True)
    ap.add_argument("--output_dir", "--output", dest="output_dir", required=True)
    ap.add_argument("--weights", default="weights/drishti_net.pt")
    ap.add_argument("--tta", action="store_true", default=False, help="8-way self-ensemble (default OFF)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ok = device.type == "cuda"
    print(f"[eval] device={device}", flush=True)
    model, cfg = load_model(get_weights(args.weights), device)
    os.makedirs(args.output_dir, exist_ok=True)

    paths = []
    for e in EXTS:
        paths += glob.glob(os.path.join(args.input_dir, e)) + \
                 glob.glob(os.path.join(args.input_dir, e.upper()))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(f"[eval] no images found in {args.input_dir}")
    print(f"[eval] {len(paths)} images")

    # ---- warmup on the REAL first input tensor (never timed): per-SHAPE cudnn autotune,
    # allocator growth and lazy CUDA init all happen here, not inside a measured call.
    # (Review fix: warming a fixed 256² uint8 dummy left the real 128² float .npy shape cold.)
    warm_x, warm_key, warm_inv, raw0 = None, None, 1.0, None
    try:
        raw0 = read_input(paths[0])
        if raw0 is not None:
            warm_x, warm_inv = preprocess(raw0, cfg)
            warm_key = paths[0]
    except Exception:
        warm_x, warm_key, raw0 = None, None, None
    if warm_x is None:
        warm_x, _ = preprocess(np.zeros((256, 256), np.uint8), cfg)
    with torch.no_grad():
        for _ in range(10 if device.type == "cuda" else 1):   # 1 iter also warms CPU kernels
            forward_one(model, warm_x.to(device), amp_ok, False)
        # dual-shape warmup (review): the other plausible KLA test shape gets its own
        # autotune pass too, so a mixed-shape test set never eats cold-start latency.
        if device.type == "cuda":
            for hw in (128, 256):
                if warm_x.shape[-1] != hw:
                    other = np.zeros((hw, hw), np.float32)
                    ox, _ = preprocess(other, cfg)
                    for _ in range(10):
                        forward_one(model, ox.to(device), amp_ok, False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for p in paths:
            if p == warm_key:                                   # reuse the already-read tensor
                raw = raw0
                x, inv = warm_x, warm_inv
            else:
                raw = read_input(p)
                if raw is None:
                    print(f"[eval] skip unreadable {p}")
                    continue
                x, inv = preprocess(raw, cfg)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = forward_one(model, x.to(device), amp_ok, args.tta)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            out = y.squeeze().clamp(0.0, 1.0).cpu().numpy() * inv
            name = os.path.basename(p)
            base, ext = os.path.splitext(name)
            if raw.dtype == np.uint8:
                out = np.clip(out, 0, 255).round().astype(np.uint8)
            elif raw.dtype == np.uint16:
                out = np.clip(out, 0, 65535).round().astype(np.uint16)
            else:                                                    # float stays float
                out = out.astype(np.float32)
                if ext.lower() != ".npy":
                    name = base + ".tif"                             # float needs TIFF
            if name.lower().endswith(".npy"):
                np.save(os.path.join(args.output_dir, name), out)    # same name/dtype in->out
            else:
                cv2.imwrite(os.path.join(args.output_dir, name), out)

    if times:
        ms = np.asarray(times) * 1000
        print(f"[eval] latency ms/img — median {np.median(ms):.2f}  mean {ms.mean():.2f}  "
              f"min {ms.min():.2f}  (n={len(ms)})")
    print(f"[eval] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
