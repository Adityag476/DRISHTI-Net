#!/usr/bin/env python3
"""Metric suite (MASTER_SPEC §4): pSNR/SSIM + metrology metrics + LODO tooling.

Numpy/OpenCV implementations to keep the eval environment lean. LPIPS is optional
(import guarded) since it requires the 'lpips' package + VGG weights.
"""
import cv2
import numpy as np


def psnr(pred, gt, peak=1.0):
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * np.log10(peak * peak / mse)


def ssim(pred, gt, peak=1.0):
    C1, C2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    mu_p, mu_g = cv2.GaussianBlur(p, (0, 0), 1.5), cv2.GaussianBlur(g, (0, 0), 1.5)
    va_p = cv2.GaussianBlur(p * p, (0, 0), 1.5) - mu_p ** 2
    va_g = cv2.GaussianBlur(g * g, (0, 0), 1.5) - mu_g ** 2
    cov = cv2.GaussianBlur(p * g, (0, 0), 1.5) - mu_p * mu_g
    s = ((2 * mu_p * mu_g + C1) * (2 * cov + C2)) / ((mu_p ** 2 + mu_g ** 2 + C1) * (va_p + va_g + C2))
    return float(s.mean())


def sobel_mag(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def edge_ssim(pred, gt):
    return ssim(sobel_mag(pred), sobel_mag(gt), peak=max(sobel_mag(gt).max(), 1e-6))


def edge_region_psnr(pred, gt, pct=0.15):
    """pSNR restricted to the top-`pct` GT-gradient pixels (defect/edge regions)."""
    m = sobel_mag(gt)
    thr = np.quantile(m, 1 - pct)
    mask = m >= thr
    mse = float(np.mean((pred[mask].astype(np.float64) - gt[mask].astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


def flat_region_artifact(pred, gt, pct=0.30):
    """Std of pred residual in the flattest GT regions — hallucination detector."""
    m = sobel_mag(gt)
    mask = m <= np.quantile(m, pct)
    return float(np.std((pred - gt)[mask]))


def patch_metrics(pred, gt, patch=64, stride=32, worst=0.10):
    """(mean_ssim, worst-10%-quantile SSIM) over sliding patches."""
    H, W = gt.shape
    vals = []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            vals.append(ssim(pred[y:y + patch, x:x + patch], gt[y:y + patch, x:x + patch]))
    vals = np.asarray(vals)
    return float(vals.mean()), float(np.quantile(vals, worst))


def lpips_score(pred, gt, net="vgg"):
    try:
        import lpips, torch
        if not hasattr(lpips_score, "_net"):
            lpips_score._net = lpips.LPIPS(net=net, verbose=False)
        t = lambda a: torch.from_numpy(a).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            return float(lpips_score._net(t(pred), t(gt)).item())
    except ImportError:
        return float("nan")


def full_report(pred, gt):
    pm, wp = patch_metrics(pred, gt)
    return {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "edge_ssim": edge_ssim(pred, gt),
            "edge_psnr": edge_region_psnr(pred, gt), "flat_artifact": flat_region_artifact(pred, gt),
            "patch_ssim_mean": pm, "patch_ssim_worst10": wp, "lpips": lpips_score(pred, gt)}
