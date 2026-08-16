#!/usr/bin/env python3
"""DAY-0 DATA AUDIT — DRISHTI-Net / KLA i4C PS01 (MASTER_SPEC §7, Day 0).

Run this on the KLA-provided dataset BEFORE training. It measures, never assumes:
  A) inventory      : dtype, shapes, value range, GT:LQ size ratio, pair matching
  B) paired stats   : speckle-family fit (multiplicative vs additive), blur-sigma
                      fingerprint (spectral MTF), range-exceed check (speckle tail),
                      histogram overlap
  C) engine fit     : random-search OUR DegradationEngine params per pair and keep
                      the best-matching configs -> the ranges we lock into training.

Outputs: <out>/audit.json + paper-grade PNG plots + a printed MASTER_SPEC block.

Usage:
  python scripts/day0_audit.py --gt_dir DATA/gt --lq_dir DATA/degraded --out audit_report --fit_engine
  python scripts/day0_audit.py --lq_dir DATA/test --out audit_test          # single-dir inventory
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
from data.degradation import DegradationEngine                  # noqa: E402

EXTS = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".npy")


# ----------------------------------------------------------------------------- io
def list_images(d):
    return sorted(p for p in Path(d).rglob("*")
                  if p.suffix.lower() in EXTS and not p.name.startswith("._"))


def imread_gray(path):
    path = str(path)
    if path.lower().endswith(".npy"):                       # KLA train set format
        arr = np.squeeze(np.load(path))
        assert arr.ndim == 2, f"{path}: expected 2D, got {arr.shape}"
        return arr
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def to_unit(img):
    """-> (float64 in ~[0,1], peak). NEVER clip (speckle exceeds GT range by design)."""
    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0, 65535
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0, 255
    f = img.astype(np.float64)
    peak = 1.0 if f.max() <= 2.0 else (255.0 if f.max() <= 300.0 else 65535)
    return f / peak, int(peak)


# ------------------------------------------------------------------ phase A: inventory
def inventory(name, paths, max_hist_files=200):
    rec = {"dir": name, "n_files": len(paths), "dtypes": {}, "shapes": {},
           "min": float("inf"), "max": float("-inf")}
    hist = np.zeros(512, np.int64)
    for i, p in enumerate(paths):
        img = imread_gray(p)
        rec["dtypes"][str(img.dtype)] = rec["dtypes"].get(str(img.dtype), 0) + 1
        shp = f"{img.shape[0]}x{img.shape[1]}"
        rec["shapes"][shp] = rec["shapes"].get(shp, 0) + 1
        rec["min"] = min(rec["min"], float(img.min()))
        rec["max"] = max(rec["max"], float(img.max()))
        if i < max_hist_files:
            f, peak = to_unit(img)
            hist += np.histogram(np.clip(f, 0, None), bins=512, range=(0, max(1.5, f.max())))[0]
    rec["hist"] = hist
    return rec


# --------------------------------------------------------------- phase B: paired stats
def flat_mask(gt, patch=32):
    """bool map: True where GT is locally flat (blur-robust speckle estimation).
    Adaptive: prefers std<2% of full scale, falls back to the flattest 10% of pixels."""
    mu = cv2.blur(gt, (patch, patch))
    mu2 = cv2.blur(gt * gt, (patch, patch))
    std = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    m = std < 0.02
    if m.sum() < 200:
        thr = np.percentile(std, 10)
        m = std <= max(thr, 1e-6)
    return m, mu


def radial_power(img):
    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fftshift(np.fft.fft2(img * win))) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.indices(img.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int).ravel()
    maxr = min(cy, cx)
    binc = np.bincount(r, P.ravel())[:maxr]
    cnt = np.bincount(r)[:maxr]
    freqs = np.arange(maxr) / (2.0 * maxr)                      # cycles/pixel
    return freqs, binc / np.maximum(cnt, 1)


def paired_stats(pairs, scale_guess=2):
    out = {"n_pairs": len(pairs), "size_ratios": {}, "range_exceed_frac": 0.0,
           "speckle_cv": [], "blur_sigma": [], "add_sigma": [],
           "mu_bins": [], "resid_std_vs_mu": []}
    n_exceed = 0
    spec_acc, spec_cnt = None, 0
    for lq_p, gt_p in pairs:
        gt_raw, lq_raw = imread_gray(gt_p), imread_gray(lq_p)
        gt, _ = to_unit(gt_raw)
        lq, _ = to_unit(lq_raw)
        ratio = f"{gt.shape[0]/lq.shape[0]:.3g}"
        out["size_ratios"][ratio] = out["size_ratios"].get(ratio, 0) + 1

        # align: downsample GT to LQ size (area), compare in LQ space
        gt_ds = cv2.resize(gt, (lq.shape[1], lq.shape[0]), interpolation=cv2.INTER_AREA)
        if lq.max() > gt_ds.max() * 1.02:
            n_exceed += 1

        # -- speckle family: residual std vs local mean (flat regions only)
        fm, mu = flat_mask(gt_ds)
        if fm.sum() > 200:
            r = (lq + 1e-4) / (gt_ds + 1e-4)
            cv_est = float(np.std(r[fm]) / max(np.mean(r[fm]), 1e-6))
            out["speckle_cv"].append(cv_est)
            resid = lq - gt_ds
            bins = np.linspace(0, max(gt_ds.max(), 1e-3), 9)
            for b0, b1 in zip(bins[:-1], bins[1:]):
                m = fm & (gt_ds >= b0) & (gt_ds < b1)
                if m.sum() > 100:
                    out["mu_bins"].append(float((b0 + b1) / 2))
                    out["resid_std_vs_mu"].append(float(np.std(resid[m])))

        # -- blur sigma from spectral MTF ratio — LOW band only (0.02..0.12 c/px):
        # at mid/high frequencies speckle+noise floor lifts P_lq above P_gt and the
        # naive MTF fit inverts (this caused the earlier None). Low band is blur-dominated.
        f_gt, P_gt = radial_power(gt_ds)
        f_lq, P_lq = radial_power(lq)
        band = (f_gt >= 0.02) & (f_gt <= 0.12) & (P_gt > 0) & (P_lq > 0)
        if band.sum() > 4:
            logr = np.log(P_lq[band] / P_gt[band])              # = -(2 pi sigma f)^2 (power)
            x = (2 * np.pi * f_gt[band]) ** 2
            A = np.vstack([x, np.ones_like(x)]).T
            slope, _ = np.linalg.lstsq(A, logr, rcond=None)[0]
            if -slope > 0:
                out["blur_sigma"].append(float(np.sqrt(-slope)))

        # -- additive floor: residual std in flat, dark-ish regions after blur is weak
        if fm.sum() > 200:
            resid = lq - cv2.GaussianBlur(gt_ds, (0, 0), 1.0)
            out["add_sigma"].append(float(np.std(resid[fm])))

        acc = np.log(np.maximum(P_lq, 1e-12)) - np.log(np.maximum(P_gt, 1e-12))
        spec_acc = acc if spec_acc is None else spec_acc + acc
        spec_cnt += 1

    out["range_exceed_frac"] = n_exceed / max(len(pairs), 1)
    out["speckle_cv_med"] = float(np.median(out["speckle_cv"])) if out["speckle_cv"] else None
    out["blur_sigma_med"] = float(np.median(out["blur_sigma"])) if out["blur_sigma"] else None
    out["add_sigma_med"] = float(np.median(out["add_sigma"])) if out["add_sigma"] else None
    if spec_cnt:
        out["mean_log_spec_ratio"] = (spec_acc / spec_cnt).tolist()
    return out


# -------------------------------------------------------------- phase C: engine fit
# Widened search space (§10c): GT-scale terms can go to TRUE ZERO, and the
# detector-scale (post-downsample) module is on the ballot this time.
FIT_RANGES = {"spk": (0.0, 0.15), "blur": (0.3, 3.0), "noise": (0.0, 0.06),
              "post_spk": (0.0, 0.30), "post_noise": (0.0, 0.04)}
FIT_KEYS = ("speckle", "blur", "noise", "kernel", "order", "post_spk", "post_noise")


def fit_engine(pairs, iters=150, seed=0):
    """For each pair: random-search our engine; keep best-PSNR config. -> distributions."""
    rng = np.random.default_rng(seed)
    winners = {"speckle": [], "blur": [], "noise": [], "kernel": [], "order": [],
               "post_spk": [], "post_noise": [], "psnr": [], "ceiling": [], "records": []}
    try:
        from tqdm import tqdm
        pair_iter = tqdm(pairs, desc="engine fit", unit="img")
    except ImportError:
        pair_iter = pairs
    for lq_p, gt_p in pair_iter:
        gt, _ = to_unit(imread_gray(gt_p))
        lq, _ = to_unit(imread_gray(lq_p))
        lq = lq.astype(np.float32)
        best = (-1.0, None)
        eng = DegradationEngine(scale=round(gt.shape[0] / lq.shape[0]), full=True, seed=None,
                                ranges=FIT_RANGES)
        for _ in range(iters):
            sim, p = eng.degrade(gt.astype(np.float32))
            if sim.shape != lq.shape:
                continue
            mse = float(np.mean((sim - lq) ** 2))
            psnr = 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)
            if psnr > best[0]:
                best = (psnr, p)
        if best[1]:
            winners["psnr"].append(best[0])
            for k in FIT_KEYS:
                winners[k].append(best[1][k])
            winners["records"].append({"lq": lq_p.name, "gt": gt_p.name,
                                       **{k: float(best[1][k]) for k in FIT_KEYS},
                                       "psnr": float(best[0])})
            # noise-realization ceiling: PSNR between INDEPENDENT draws at the SAME
            # fitted params — a perfect engine cannot beat this, so compare fit to it.
            fixed = dict(best[1])
            ceils = []
            gtf = gt.astype(np.float32)
            for t in range(4):
                ea = DegradationEngine(scale=eng.scale, full=True, seed=1000 + t)
                eb = DegradationEngine(scale=eng.scale, full=True, seed=2000 + t)
                la, _ = ea.degrade(gtf, fixed=fixed)
                lb, _ = eb.degrade(gtf, fixed=fixed)
                mse = float(np.mean((la - lb) ** 2))
                ceils.append(99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse))
            cmed = float(np.median(ceils))
            winners["ceiling"].append(cmed)
            winners["records"][-1]["ceiling"] = cmed
    rng = rng  # noqa: F841
    agg = {}
    for k, v in winners.items():
        if v and k != "records":
            agg[k + "_p25"] = float(np.percentile(v, 25))
            agg[k + "_med"] = float(np.median(v))
            agg[k + "_p75"] = float(np.percentile(v, 75))
    agg["kernel_votes"] = {int(k): int(v) for k, v in zip(*np.unique(winners["kernel"], return_counts=True))} if winners["kernel"] else {}
    agg["order_votes"] = {int(k): int(v) for k, v in zip(*np.unique(winners["order"], return_counts=True))} if winners["order"] else {}
    agg["n_pairs_fit"] = len(winners["psnr"])
    # bound-pinning: counts of winners sitting within 2% of a search-range edge.
    # >10% at one edge = the truth lives outside the cage; widen or rethink.
    pin = {}
    for key, (lo, hi) in {"speckle": FIT_RANGES["spk"], "blur": FIT_RANGES["blur"],
                          "noise": FIT_RANGES["noise"], "post_spk": FIT_RANGES["post_spk"],
                          "post_noise": FIT_RANGES["post_noise"]}.items():
        ws = winners[key]
        if ws:
            span = hi - lo
            pin[key] = {"at_lo": int(sum(1 for v in ws if v - lo < 0.02 * span)),
                        "at_hi": int(sum(1 for v in ws if hi - v < 0.02 * span))}
    agg["bound_pinning"] = pin
    return agg, winners


def residual_autopsy(pairs, recs, outdir, k=6):
    """Worst-fit autopsy: re-render the winning synthetic per worst pair and inspect the
    residual (real LQ - synthetic). Edge-correlated residual => blur/kernel mismatch;
    uncorrelated => extra unstructured noise process (e.g. detector-scale speckle)."""
    if not recs:
        return None
    lut = {lq.name: (lq, gt) for lq, gt in pairs}
    rows, corrs = [], []
    for r in sorted(recs, key=lambda z: z["psnr"])[:k]:
        lq_p, gt_p = lut[r["lq"]]
        gt, _ = to_unit(imread_gray(gt_p))
        lq, _ = to_unit(imread_gray(lq_p))
        gt = gt.astype(np.float32); lq = lq.astype(np.float32)
        eng = DegradationEngine(scale=round(gt.shape[0] / lq.shape[0]), full=True, seed=123)
        fixed = {kk: r[kk] for kk in FIT_KEYS}
        sim, _ = eng.degrade(gt, fixed=fixed)
        sim = np.asarray(sim, np.float32)
        if sim.shape != lq.shape:
            continue
        resid = lq - sim
        gt_ds = cv2.resize(gt, (lq.shape[1], lq.shape[0]), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(gt_ds, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gt_ds, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(gx, gy)
        c = np.corrcoef(np.abs(resid).ravel(), edge.ravel())[0, 1]
        corrs.append(float(c))
        rows.append((gt_ds, lq, sim, resid))
    if not rows:
        return None
    fig, axs = plt.subplots(len(rows), 4, figsize=(12, 2.6 * len(rows)))
    axs = np.atleast_2d(axs)
    vlim = max(1e-6, float(np.percentile(np.abs(rows[0][3]), 99)))
    for i, (gd, lq, sim, resid) in enumerate(rows):
        for j, (im, ttl, cmap, rng_) in enumerate(
                ((gd, "GT↓ (ref)", "gray", (0, 1)), (lq, "real LQ", "gray", None),
                 (sim, "best synthetic", "gray", None),
                 (resid, "residual (LQ−syn)", "RdBu_r", (-vlim, vlim)))):
            kw = dict(cmap=cmap)
            if rng_:
                kw.update(vmin=rng_[0], vmax=rng_[1])
            axs[i][j].imshow(im, **kw); axs[i][j].set_title(ttl, fontsize=8)
            axs[i][j].axis("off")
    fig.tight_layout(); fig.savefig(Path(outdir) / "residual_autopsy.png", dpi=140)
    plt.close(fig)
    return float(np.median(corrs))


def synth_check(gt_paths, real_stats, deg_hist, outdir, n=24, scale=2, seed=77):
    """The CORRECT family-adequacy gate (§10c): per-pixel PSNR fitting of stochastic
    noise is degenerate (fit collapses to the smooth conditional-mean corner), so we
    compare DISTRIBUTION-LEVEL statistics instead: render synthetic LQ from real GTs
    with the TRAINING engine defaults and check flat-CV / sigma-vs-mu slope / bright
    tail against the REAL LQ population. MATCH => synthetic distribution ≈ real."""
    outdir = Path(outdir)
    tmp = outdir / "_synth_probe"; tmp.mkdir(exist_ok=True)
    eng = DegradationEngine(scale=scale, full=True, seed=seed)
    spairs, pix = [], []
    for i, gt_p in enumerate(gt_paths[:n]):
        gt, _ = to_unit(imread_gray(gt_p)); gt = gt.astype(np.float32)
        lr, _ = eng.degrade(gt)
        lp = tmp / f"s{i:06d}.npy"; np.save(lp, lr.astype(np.float32))
        spairs.append((lp, gt_p))
        pix.append(lr[::8, ::8].ravel())
    s = paired_stats(spairs)
    pix = np.concatenate(pix)
    s_tail, s_max = float(np.percentile(pix, 99.99)), float(pix.max())
    r_slope = r_int = s_slope = s_int = None
    if real_stats.get("mu_bins"):
        r_slope, r_int = np.polyfit(real_stats["mu_bins"], real_stats["resid_std_vs_mu"], 1)
    if s.get("mu_bins"):
        s_slope, s_int = np.polyfit(s["mu_bins"], s["resid_std_vs_mu"], 1)
    # real bright tail from the degraded inventory histogram (512 bins over [0, 1.5])
    r_tail = None
    if deg_hist is not None:
        hc = np.cumsum(deg_hist) / max(deg_hist.sum(), 1)
        r_tail = float(np.searchsorted(hc, 0.9999) / len(deg_hist) * 1.5)
    print(f"[D] synth-check — engine TRAINING defaults vs REAL LQ (n={len(spairs)} synth pairs)")
    print(f"[D]   flat CV:        real {real_stats['speckle_cv_med']:.4f} vs synth {s['speckle_cv_med']:.4f}")
    if r_slope is not None and s_slope is not None:
        print(f"[D]   slope/intercept: real {r_slope:.3f}/{r_int:.4f} vs synth {s_slope:.3f}/{s_int:.4f}"
              f"   (slope = multiplicative strength at LQ scale)")
    print(f"[D]   bright tail:     real p99.99={r_tail} vs synth p99.99={s_tail:.3f} max={s_max:.3f}"
          f"   (must EXCEED 1.0 — clipping is forbidden)")
    ok = True
    if real_stats["speckle_cv_med"] and s["speckle_cv_med"]:
        ok &= abs(s["speckle_cv_med"] - real_stats["speckle_cv_med"]) / real_stats["speckle_cv_med"] < 0.25
    if r_slope is not None and s_slope is not None and r_slope > 0:
        ok &= abs(s_slope - r_slope) / r_slope < 0.25
    ok &= s_tail > 1.1
    print(f"[D]   verdict: {'MATCH (rel. deltas < 25%) — family adequate, M-series GO' if ok else 'MISMATCH — recalibrate ranges before M-series'}")
    return {"synth_cv": s["speckle_cv_med"], "synth_slope": None if s_slope is None else float(s_slope),
            "synth_tail_p9999": s_tail, "real_tail_p9999": r_tail, "match": bool(ok)}


# ------------------------------------------------------------------------ plots
def plot_all(outdir, invs, stats, winners):
    outdir = Path(outdir)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rec in invs.items():
        h = rec["hist"] / max(rec["hist"].sum(), 1)
        x = np.linspace(0, 1.5, 512)
        ax.plot(x, h, label=name, lw=1.2)
    ax.set_yscale("log"); ax.set_xlabel("intensity (unit)"); ax.set_ylabel("density (log)")
    ax.set_title("Histograms — does degraded EXCEED GT range?"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "histograms.png", dpi=150); plt.close(fig)

    if stats and stats["mu_bins"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        mu, sd = np.array(stats["mu_bins"]), np.array(stats["resid_std_vs_mu"])
        ax.scatter(mu, sd, s=8, alpha=0.35, label="flat-patch residuals")
        if len(mu) > 5:
            a, b = np.polyfit(mu, sd, 1)
            xs = np.linspace(mu.min(), mu.max(), 50)
            ax.plot(xs, a * xs + b, "r-", lw=2, label=f"fit: slope={a:.3f}, int={b:.4f}")
        ax.set_xlabel("local GT mean"); ax.set_ylabel("residual std")
        ax.set_title("Speckle family: slope>0 & slope>>intercept -> multiplicative")
        ax.legend(); fig.tight_layout()
        fig.savefig(outdir / "sigma_vs_mu.png", dpi=150); plt.close(fig)

    if winners and winners["psnr"]:
        cols = [("speckle", "speckle var (GT)"), ("blur", "blur σ (GT)"),
                ("noise", "additive σ (GT)"), ("post_spk", "post-spk σ (LQ std)"),
                ("post_noise", "post-noise σ (LQ)")]
        cols = [c for c in cols if winners.get(c[0])]
        fig, axs = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.4))
        for ax, (k, t) in zip(np.atleast_1d(axs), cols):
            ax.hist(winners[k], bins=15, color="#0B3C5D"); ax.set_title(f"engine fit: {t}")
        fig.tight_layout(); fig.savefig(outdir / "engine_fit.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir"); ap.add_argument("--lq_dir")
    ap.add_argument("--out", default="audit_report")
    ap.add_argument("--max_pairs", type=int, default=48)
    ap.add_argument("--fit_engine", action="store_true")
    ap.add_argument("--fit_iters", type=int, default=150)
    ap.add_argument("--fit_pairs", type=int, default=12)
    ap.add_argument("--synth_check", action="store_true",
                    help="distribution-level family gate (§10c); doesn't need --fit_engine")
    ap.add_argument("--synth_pairs", type=int, default=24)
    a = ap.parse_args()
    assert a.gt_dir or a.lq_dir, "give at least one of --gt_dir / --lq_dir"
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    report, invs = {}, {}
    gt_paths = list_images(a.gt_dir) if a.gt_dir else []
    lq_paths = list_images(a.lq_dir) if a.lq_dir else []
    if a.gt_dir:
        invs["GT"] = inventory(a.gt_dir, gt_paths)
    if a.lq_dir:
        invs["DEGRADED"] = inventory(a.lq_dir, lq_paths)
    report["inventory"] = {k: {kk: vv for kk, vv in v.items() if kk != "hist"}
                           for k, v in invs.items()}
    print(f"[A] GT files: {len(gt_paths)}  DEG files: {len(lq_paths)}")
    for k, v in report["inventory"].items():
        print(f"    {k}: dtypes={v['dtypes']} shapes={v['shapes']} range=[{v['min']},{v['max']}]")

    stats, winners = None, None
    if a.gt_dir and a.lq_dir:
        lut = {p.name: p for p in lq_paths}
        pairs = [(lut[p.name], p) for p in gt_paths if p.name in lut]
        print(f"[A] matched pairs by filename: {len(pairs)} "
              f"(GT-only: {len(gt_paths)-len(pairs)}, DEG-only: {len(lq_paths)-len(pairs)})")
        report["n_matched_pairs"] = len(pairs)
        if pairs:
            stats = paired_stats(pairs[:a.max_pairs])
            report["paired_stats"] = {k: v for k, v in stats.items()
                                      if not isinstance(v, list) or len(v) < 50}
            n_stat = min(len(pairs), a.max_pairs)
            print(f"[B] size GT/LQ ratios: {stats['size_ratios']}  (n={n_stat} of {len(pairs)} pairs)")
            print(f"[B] LQ brighter than GT-downsampled in {stats['range_exceed_frac']*100:.0f}% pairs "
                  f"(resolution-matched; >0 => speckle tail past GT range)")
            print(f"[B] speckle CV (flat regions): {stats['speckle_cv_med']}")
            print(f"[B] blur sigma (spectral MTF, low band): {stats['blur_sigma_med']}")
            print(f"[B] flat-residual σ TOTAL (speckle + blur-mismatch + additive — "
                  f"an UPPER BOUND on additive σ, not additive itself): {stats['add_sigma_med']}")
            if a.fit_engine:
                agg, winners = fit_engine(pairs[:a.fit_pairs], iters=a.fit_iters)
                report["engine_fit"] = agg
                report["engine_fit_records"] = winners.get("records", [])
                print(f"[C] engine fit over {agg['n_pairs_fit']} pairs, best-PSNR "
                      f"med={agg.get('psnr_med'):.2f} dB | smooth-corner ceiling "
                      f"med={agg.get('ceiling_med'):.2f} dB")
                print(f"[C] NOTE: PSNR-fit of stochastic noise collapses to the smooth "
                      f"conditional-mean corner — a fit-vs-ceiling gap is EXPECTED and is NOT "
                      f"a family verdict (§10c). Family gate = --synth_check below.")
                print(f"[C] speckle var p25/med/p75: {agg.get('speckle_p25'):.4f}/"
                      f"{agg.get('speckle_med'):.4f}/{agg.get('speckle_p75'):.4f}  (GT scale; std=√var)")
                print(f"[C] blur    σ p25/med/p75: {agg.get('blur_p25'):.3f}/"
                      f"{agg.get('blur_med'):.3f}/{agg.get('blur_p75'):.3f}")
                print(f"[C] noise   σ p25/med/p75: {agg.get('noise_p25'):.4f}/"
                      f"{agg.get('noise_med'):.4f}/{agg.get('noise_p75'):.4f}")
                print(f"[C] POST-spk σ p25/med/p75: {agg.get('post_spk_p25'):.4f}/"
                      f"{agg.get('post_spk_med'):.4f}/{agg.get('post_spk_p75'):.4f}  (detector/LQ scale std)")
                print(f"[C] POST-noise σ p25/med/p75: {agg.get('post_noise_p25'):.4f}/"
                      f"{agg.get('post_noise_med'):.4f}/{agg.get('post_noise_p75'):.4f}")
                print(f"[C] kernel votes {agg['kernel_votes']}  order votes {agg['order_votes']} "
                      f"(0=blur-first,1=noise-first)")
                pin = agg.get("bound_pinning", {})
                print(f"[C] bound-pinning (of {agg['n_pairs_fit']}): " +
                      "  ".join(f"{k}: lo={v['at_lo']} hi={v['at_hi']}" for k, v in pin.items()) +
                      "   (>10% at an edge = truth outside search range)")
                em = residual_autopsy(pairs[:a.fit_pairs], winners.get("records"), outdir)
                if em is not None:
                    report["engine_fit"]["resid_edge_corr_med"] = em
                    print(f"[C] residual edge-correlation med = {em:+.3f}  (≈0 => unstructured "
                          f"extra noise; >>0 => blur/kernel mismatch)  -> residual_autopsy.png")
            if a.synth_check:
                report["synth_check"] = synth_check(
                    gt_paths, stats, invs.get("DEGRADED", {}).get("hist"),
                    outdir, n=a.synth_pairs)

    plot_all(outdir, invs, stats, winners)

    # ---- SCALE + spec recommendation
    peak = 1
    for v in report["inventory"].values():
        if "uint16" in v["dtypes"]:
            peak = 65535
        elif "uint8" in v["dtypes"]:
            peak = max(peak, 255)
    report["recommendation"] = {
        "norm_peak": "auto (divide by dtype peak: 65535 uint16 / 255 uint8 / 1.0 float .npy) "
                     "— applied automatically by data/dataset.py and evaluate.py; "
                     "set cfg['norm_peak'] only to override",
        "dominant_dtype_peak_detected": peak,
        "log1p_input": True,
        "note": "paste measured numbers into MASTER_SPEC §2/§3"}
    with open(outdir / "audit.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[✓] audit.json + plots -> {outdir}/")
    print(f"[✓] norm_peak: auto by dtype (detected {peak}); never clip")
    print("NEXT: numbers -> MASTER_SPEC §10. Family gate: re-run with --synth_check; M0 launches only after a MATCH print.")


if __name__ == "__main__":
    main()
