#!/usr/bin/env python3
"""DRISHTI-Net training — experiment-ladder driven (MASTER_SPEC §6).

Ladder levels (--level):
  M0 baseline NAFNet + Charbonnier, mild data, no FiLM, no aux
  M1 + full degradation randomization (SEM artifacts still off)
  M2 + supervised degradation encoder + FiLM  (aux loss 0.05)
  M3 + MS-SSIM/Sobel/FFT loss terms (each weight an ablation flag)
  M4 + defect-weighted Charbonnier + SEM artifacts + CutBlur
  M5 hardening step (export/benchmark — not a training config)

Example (demo):
  python train.py --gt_dir demo_data/gt --level M2 --iters 50 --batch 2 --workers 0
"""
import argparse
import copy
import functools
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.dataset import SyntheticDataset, RealPairsDataset, imread_gray, encode_input
from losses.fab_loss import FabLoss
from models.drishti_net import DRISHTINet, NAFBlockFiLM


@torch.no_grad()
def validate(model, items, use_log, dev):
    """Full-image validation on held-out pairs (the actual benchmark task).
    items: list of (lr_np, gt_np) unit-scale float32 arrays. Returns (psnr, ssim)."""
    from utils.metrics import psnr, ssim
    model.eval()
    P, S = [], []
    for lr_np, gt_np in items:
        x = encode_input(lr_np, use_log)[None].to(dev)
        y = model(x)
        pred = y.squeeze().float().clamp(0, 1).cpu().numpy()
        P.append(psnr(pred, gt_np))
        S.append(ssim(pred, gt_np))
    return float(np.mean(P)), float(np.mean(S))

LEVELS = {  # ladder -> (use_film, full_engine, sem, cutblur_p, use_dw, w_ssim, w_sobel, w_fft, w_aux)
    "M0": dict(film=False, full=False, sem=False, cb=0.0, dw=False, ssim=0.0, sob=0.0, fft=0.0, aux=0.0),
    "M1": dict(film=False, full=True,  sem=False, cb=0.0, dw=False, ssim=0.0, sob=0.0, fft=0.0, aux=0.0),
    "M2": dict(film=True,  full=True,  sem=False, cb=0.0, dw=False, ssim=0.0, sob=0.0, fft=0.0, aux=0.05),
    "M3": dict(film=True,  full=True,  sem=False, cb=0.0, dw=False, ssim=0.5, sob=0.2, fft=0.1, aux=0.05),
    "M4": dict(film=True,  full=True,  sem=True,  cb=0.6, dw=True,  ssim=0.5, sob=0.2, fft=0.1, aux=0.05),
    # §11 recovery rungs (post-FiLM-fix):
    "M3r": dict(film=False, full=False, sem=False, cb=0.0, dw=False, ssim=0.5, sob=0.2, fft=0.1, aux=0.0),   # real-only + metric loss
    "M2f": dict(film=True,  full=False, sem=False, cb=0.0, dw=False, ssim=0.0, sob=0.0, fft=0.0, aux=0.05),  # FiLM-fixed, mild engine
    # §11c reviewer-demanded one-term-at-a-time loss isolation (GPT T3: M3a..M3e):
    "M3a": dict(film=False, full=False, sem=False, cb=0.0, dw=False, ssim=0.5, sob=0.0, fft=0.0, aux=0.0),
    "M3b": dict(film=False, full=False, sem=False, cb=0.0, dw=False, ssim=0.0, sob=0.2, fft=0.0, aux=0.0),
    "M3c": dict(film=False, full=False, sem=False, cb=0.0, dw=False, ssim=0.0, sob=0.0, fft=0.1, aux=0.0),
    "M3d": dict(film=False, full=False, sem=False, cb=0.0, dw=False, ssim=0.5, sob=0.2, fft=0.0, aux=0.0),  # pair combo
    # (M3e == M3r: SSIM+Sobel+FFT, already defined above)
}


def _seed_worker(worker_id, base_seed):
    """Multi-worker diversity fix (review P0) — MODULE-LEVEL + functools.partial on purpose:
    Windows spawns DataLoader workers, so the init fn must be picklable (a closure inside
    main() would crash with 'Can't pickle local object' the moment workers>0). The worker's
    own dataset copies are reached via get_worker_info().dataset — no closure needed."""
    np.random.seed(torch.initial_seed() % 2**32)
    info = torch.utils.data.get_worker_info()
    ds = info.dataset if info is not None else None
    for j, d in enumerate(getattr(ds, "datasets", None) or ([ds] if ds is not None else [])):
        if hasattr(d, "worker_seed"):
            d.worker_seed(base_seed + worker_id * 7919 + j * 131)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", default=None, help="clean GT images (synthetic engine input)")
    ap.add_argument("--no_syn", action="store_true",
                    help="disable synthetic engine data — real-pairs-only baseline arm (M0r)")
    ap.add_argument("--real_lq", default=None, help="optional KLA degraded dir")
    ap.add_argument("--real_gt", default=None, help="optional KLA GT dir (pairs by filename)")
    ap.add_argument("--level", default="M2", choices=list(LEVELS))
    ap.add_argument("--use_log", action="store_true", default=True)
    ap.add_argument("--no_log", dest="use_log", action="store_false")
    ap.add_argument("--patch_hr", type=int, default=256, help="HR crop (LR = /2); both scales via engine")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--out", default="runs/M2")
    ap.add_argument("--fft_weight", type=float, default=None, help="ablation override for λ_fft")
    ap.add_argument("--init_from", default=None,
                    help="checkpoint to initialize/continue weights from. M3 ablation arms "
                         "MUST share one start point (GPT round-4): pass M0r-v5 latest.pt "
                         "so the question is 'what does this loss term add to the same model?'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=1000, help="0 disables validation")
    ap.add_argument("--probe_every", type=int, default=500,
                    help="FiLM/z/grad telemetry cadence (0 disables); evidence for §11c root cause")
    ap.add_argument("--val_split", type=float, default=0.05,
                    help="fraction of real pairs held out for validation (sorted order)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    L = LEVELS[args.level]
    fft_w = L["fft"] if args.fft_weight is None else args.fft_weight
    # Claude round-9 / GPT round-10: a CLI loss override must never silently carry the
    # base level's name inside the checkpoint — make the run self-describing.
    overrides = []
    if args.fft_weight is not None and float(args.fft_weight) != float(L["fft"]):
        overrides.append(f"fft{args.fft_weight}")
    level_tag = args.level + ("+" + "+".join(overrides) if overrides else "")
    loss_formula = (f"Charbonnier + {L['ssim']}*SSIM + {L['sob']}*Sobel + {fft_w}*FFT-mag"
                    + (" (edge-weighted)" if L["dw"] else ""))
    cfg = dict(in_ch=2 if args.use_log else 1, out_ch=1, width=32, z_dim=64,
               use_film=L["film"], with_uncertainty=False, use_log=args.use_log,
               norm_peak=None, level=level_tag, aux_dim=10,   # 5 reg + 4 kernel + 1 order
               loss_formula=loss_formula)
    # auditable experiment header (GPT round-5): logs must stand alone as evidence
    print(f"[cfg] level={level_tag} (base={args.level}) seed={args.seed} lr={args.lr} warmup={args.warmup} "
          f"iters={args.iters} batch={args.batch} film={L['film']} use_log={args.use_log} "
          f"synthetic={'OFF (--no_syn)' if args.no_syn else 'on'} init_from={args.init_from}")
    print(f"[cfg] loss_formula: {loss_formula}")
    print(f"[cfg] loss_weights: charb=1.0 dw={L['dw']} ssim={L['ssim']} sobel={L['sob']} "
          f"fft={fft_w} aux={L['aux']} sem={L['sem']} cutblur_p={L['cb']}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRISHTINet(**{k: cfg[k] for k in ("in_ch", "out_ch", "width", "z_dim", "use_film",
                                              "with_uncertainty", "aux_dim")}).to(dev)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        try:
            model.load_state_dict(sd)                       # strict — the preferred path
            note = "strict load OK (all tensors transferred)"
        except RuntimeError:
            # LOUD non-strict fallback (GPT round-5): allowed ONLY when the sole missing
            # tensors are FiLM-branch params (film=False checkpoint -> film=True model,
            # whose FiLM layers correctly keep their own zero-init). Any unexpected keys
            # or non-FiLM missing keys = incompatible experiment -> crash loudly.
            inc = model.load_state_dict(sd, strict=False)
            bad_missing = [k for k in inc.missing_keys if ".film." not in k]
            if inc.unexpected_keys or bad_missing:
                raise RuntimeError(
                    f"--init_from {args.init_from}: INCOMPATIBLE checkpoint — "
                    f"unexpected keys={inc.unexpected_keys[:4]}  "
                    f"non-FiLM missing keys={bad_missing[:4]} (silent strict=False is forbidden)")
            note = (f"non-strict load: {len(inc.missing_keys)} missing FiLM-branch tensors "
                    f"left at own zero-init (e.g. {inc.missing_keys[0]})")
        print(f"[init] weights from {args.init_from} "
              f"(step {ck.get('step', '?')}, val_psnr {ck.get('val_psnr', '?')}); {note}")
        print("[init] optimizer + scheduler are FRESH — no AdamW momentum state carried over")
    loss_fn = FabLoss(use_dw=L["dw"], w_ssim=L["ssim"], w_sobel=L["sob"], w_fft=fft_w, w_aux=L["aux"])

    sets, syn = [], None
    if not args.no_syn:
        assert args.gt_dir, "--gt_dir is required unless --no_syn"
        syn = SyntheticDataset(args.gt_dir, patch_hr=args.patch_hr, use_log=args.use_log,
                               engine_kw=dict(full=L["full"], sem_artifacts=L["sem"], cutblur_p=L["cb"]),
                               epoch_len=max(args.batch * 4, args.batch * args.iters if args.iters < 200 else 2000),
                               seed=args.seed)
        sets.append(syn)
    if args.real_lq and args.real_gt:
        sets.append(RealPairsDataset(args.real_lq, args.real_gt, patch_hr=args.patch_hr,
                                     use_log=args.use_log, seed=args.seed + 7))
    assert sets, "no training data: provide --gt_dir and/or --real_lq + --real_gt"

    # ---- validation items BEFORE DataLoader (holdout must be split before concat)
    val_items = []
    if args.val_every > 0:
        rp = next((s for s in sets if isinstance(s, RealPairsDataset)), None)
        if rp is not None:
            allp = list(rp.pairs)
            n_val = max(1, int(len(allp) * args.val_split))
            if len(allp) > n_val:                           # honest holdout: excluded from training
                rp.pairs = allp[:-n_val]
            print(f"[val] holding out {min(n_val, len(allp))} real pairs (excluded from training)")
            # filename transparency (reviewer ask: the tail split is deterministic —
            # print the exact range so acquisition-order correlation can be inspected)
            if 0 < n_val <= len(allp):
                print(f"[val] holdout names: {allp[-n_val][0].name} .. {allp[-1][0].name}")
            for lq_p, gt_p in allp[-n_val:]:
                g, _ = imread_gray(gt_p)
                l, _ = imread_gray(lq_p)
                val_items.append((l.astype(np.float32), g.astype(np.float32)))
        else:
            from data.degradation import DegradationEngine
            eng = DegradationEngine(scale=2, full=L["full"], sem_artifacts=L["sem"], seed=12345)
            for p in syn.paths[-min(16, len(syn.paths)):]:
                g, _ = imread_gray(p)
                ph = min(256, g.shape[0], g.shape[1])
                y = (g.shape[0] - ph) // 2
                x = (g.shape[1] - ph) // 2
                g = np.ascontiguousarray(g[y:y + ph, x:x + ph])
                l, _ = eng.degrade(g)
                val_items.append((l.astype(np.float32), g.astype(np.float32)))
            print(f"[val] no real pairs -> {len(val_items)} fixed synthetic validation images")

    dl = DataLoader(ConcatDataset(sets), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    worker_init_fn=functools.partial(_seed_worker,
                                                     base_seed=(args.seed + 1) * 100003))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    # protocol fix (review): cosine phase must span iters AFTER warmup, not the full count
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.iters - max(args.warmup, 0)), eta_min=1e-6)
    if args.warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=args.warmup)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, sched], [args.warmup])
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    ema = EMA(model)

    it = iter(dl)
    best_psnr = -1e9
    t0 = time.time()
    for step in range(1, args.iters + 1):
        try:
            lr, gt, reg, ker, order, has = next(it)
        except StopIteration:
            it = iter(dl)
            lr, gt, reg, ker, order, has = next(it)
        lr, gt, reg, ker, order, has = (t.to(dev, non_blocking=True)
                                        for t in (lr, gt, reg, ker, order, has))
        probe = bool(args.probe_every) and (step % args.probe_every == 0)
        # FiLM/pred telemetry: converts "root cause" from assertion to measured evidence
        model.probe = probe
        for m in model.modules():
            if isinstance(m, NAFBlockFiLM):
                m.probe = probe
        with amp:
            if cfg["use_film"]:
                pred, aux = model(lr, need_aux=True)
            else:
                pred, aux = model(lr), None
            # MS-SSIM ramps in over the first 1000 iters (H-MS isolation, M3r family)
            ssim_scale = min(1.0, step / 1000.0) if loss_fn.w["ssim"] > 0 else 0.0
            loss, parts = loss_fn(pred.float(), gt.float(), aux=aux, reg_labels=reg,
                                  kernel_idx=ker.long(), has_labels=has,
                                  order_labels=order.float(), ssim_scale=ssim_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        last_gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))  # PRE-clip norm
        opt.step()
        sched.step()
        ema.update(model)

        if step % args.log_every == 0:
            msg = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
            print(f"[{step}/{args.iters}] loss={loss.item():.4f} ({msg}) "
                  f"lr={sched.get_last_lr()[0]:.2e} {(time.time()-t0)/step:.3f}s/it", flush=True)
        if probe:
            try:
                pmin, pmax = pred.min().item(), pred.max().item()
            except KeyboardInterrupt:
                raise
            except Exception:
                pmin = pmax = float("nan")
            zmax = getattr(model, "_z_absmax", float("nan"))
            zmean = getattr(model, "_z_absmean", float("nan"))
            gs, bs = [], []
            for m in model.modules():
                if isinstance(m, NAFBlockFiLM) and m.use_film and hasattr(m, "_g_flat"):
                    gs.append(m._g_flat)
                    bs.append(m._b_flat)
            if gs:                                       # distribution stats (GPT round-3):
                g_all = torch.cat(gs).float()            # a single max can hide p95 drift
                b_all = torch.cat(bs).float()
                gdesc = f"{g_all.mean():.3f}/{torch.quantile(g_all, 0.95):.3f}/{g_all.max():.3f}"
                bdesc = f"{b_all.mean():.3f}/{torch.quantile(b_all, 0.95):.3f}/{b_all.max():.3f}"
                # per-STAGE p95 (GPT round-4): one deep stage may saturate while the rest
                stage_g, stage_b = {}, {}
                for m in model.modules():
                    st = getattr(m, "_stage", None)
                    if st is not None and getattr(m, "use_film", False) and hasattr(m, "_g_flat"):
                        stage_g.setdefault(st, []).append(m._g_flat)
                        stage_b.setdefault(st, []).append(m._b_flat)
                order = ["enc1", "enc2", "enc3", "mid", "dec3", "dec2", "dec1"]
                keys = [s for s in order if s in stage_g] + sorted(set(stage_g) - set(order))
                sdesc = " | g95 " + " ".join(
                    f"{s}={torch.quantile(torch.cat(stage_g[s]).float(), 0.95):.3f}" for s in keys
                ) + " | b95 " + " ".join(
                    f"{s}={torch.quantile(torch.cat(stage_b[s]).float(), 0.95):.3f}" for s in keys)
            else:
                gdesc = bdesc = "off"
                sdesc = ""
            print(f"  [probe @{step}] pred[{pmin:.3f},{pmax:.3f}] zAbs mean/max={zmean:.3f}/{zmax:.3f} "
                  f"film|g| mean/p95/max={gdesc} film|b| mean/p95/max={bdesc} "
                  f"gradNormPreClip={last_gn:.3f}{sdesc}", flush=True)
        if args.val_every > 0 and val_items and (step % args.val_every == 0 or step == args.iters):
            vp, vs = validate(ema.shadow, val_items, args.use_log, dev)
            print(f"  [val @{step}] PSNR={vp:.2f} dB  SSIM={vs:.4f}", flush=True)
            if vp > best_psnr:
                best_psnr = vp
                torch.save({"model_state_dict": ema.shadow.state_dict(), "config": cfg,
                            "step": step, "val_psnr": vp, "val_ssim": vs},
                           os.path.join(args.out, "best.pt"))
        if step % args.save_every == 0 or step == args.iters:
            torch.save({"model_state_dict": ema.shadow.state_dict(), "config": cfg,
                        "step": step}, os.path.join(args.out, "latest.pt"))
    print(f"done -> {args.out}/latest.pt")


if __name__ == "__main__":
    main()
