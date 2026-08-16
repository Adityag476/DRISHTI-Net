#!/usr/bin/env python3
"""FabLoss (MASTER_SPEC §3): DW-Charbonnier + MS-SSIM + Sobel + FFT-L1 + masked aux.

Two-batch-type rule: aux regression/CE losses are multiplied by has_labels
(1.0 for synthetic pairs, 0.0 for real KLA pairs) — never crash, never silently
disable supervision on all data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_kernels(device, dtype):
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


def charbonnier(pred, gt, eps=1e-6):
    return torch.mean(torch.sqrt((pred - gt) ** 2 + eps))


def dw_charbonnier(pred, gt, alpha=2.0, eps=1e-6):
    kx, ky = _sobel_kernels(gt.device, gt.dtype)
    g = torch.sqrt(F.conv2d(gt, kx, padding=1) ** 2 + F.conv2d(gt, ky, padding=1) ** 2 + 1e-12)
    g = g / g.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    w = 1.0 + alpha * g
    return torch.mean(w * torch.sqrt((pred - gt) ** 2 + eps))


def sobel_edge_loss(pred, gt):
    kx, ky = _sobel_kernels(gt.device, gt.dtype)
    return (F.l1_loss(F.conv2d(pred, kx, padding=1), F.conv2d(gt, kx, padding=1)) +
            F.l1_loss(F.conv2d(pred, ky, padding=1), F.conv2d(gt, ky, padding=1)))


def fft_l1_loss(pred, gt):
    return F.l1_loss(torch.fft.rfft2(pred, norm="ortho").abs(),
                     torch.fft.rfft2(gt, norm="ortho").abs())


class FabLoss(nn.Module):
    def __init__(self, use_dw=False, alpha_edge=2.0, w_ssim=0.0, w_sobel=0.0,
                 w_fft=0.0, w_aux=0.05):
        super().__init__()
        self.use_dw, self.alpha = use_dw, alpha_edge
        self.w = dict(ssim=w_ssim, sobel=w_sobel, fft=w_fft, aux=w_aux)
        self._ms_ssim = self._ssim = None
        if w_ssim > 0:
            try:
                from pytorch_msssim import MS_SSIM, SSIM
                self._ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=1)
                self._ssim = SSIM(data_range=1.0, size_average=True, channel=1)
            except ImportError:
                print("[FabLoss] pytorch-msssim missing — MS-SSIM term disabled")

    def forward(self, pred, gt, aux=None, reg_labels=None, kernel_idx=None, has_labels=None,
                order_labels=None, ssim_scale=1.0):
        """ssim_scale: training ramp hook (review-hypothesis H-MS): full MS-SSIM weight from
        iter 0 on unclamped residuals was implicated in the M3/M4 collapse; the ramp and the
        clamp below isolate that. Clamp is LOCAL to the MS-SSIM term only — all other terms
        keep the raw prediction (pixel loss on raw signal = honest large-error gradient)."""
        parts = {}
        total = dw_charbonnier(pred, gt, self.alpha) if self.use_dw else charbonnier(pred, gt)
        parts["pixel"] = total.item()
        if self.w["ssim"] > 0 and self._ms_ssim is not None and ssim_scale > 0:
            # MS-SSIM needs >=160px (4 downsamplings); fall back to SSIM on small crops
            fn = self._ms_ssim if min(pred.shape[-2:]) >= 176 else self._ssim
            l = (1 - fn(pred.clamp(0.0, 1.0), gt)) * (self.w["ssim"] * ssim_scale)
            total = total + l; parts["ssim"] = l.item()
            parts["ss_scale"] = round(float(ssim_scale), 4)   # GPT round-3: log EFFECTIVE weight
        if self.w["sobel"] > 0:
            l = sobel_edge_loss(pred, gt) * self.w["sobel"]; total = total + l; parts["sobel"] = l.item()
        if self.w["fft"] > 0:
            l = fft_l1_loss(pred, gt) * self.w["fft"]; total = total + l; parts["fft"] = l.item()
        if aux is not None and has_labels is not None and float(has_labels.sum()) > 0:
            hmm = has_labels.view(-1)
            n_reg = int(reg_labels.shape[-1])           # 3 (legacy runs) or 5 (post terms added)
            reg = (F.mse_loss(aux[:, :n_reg], reg_labels, reduction="none").mean(1) * hmm).sum() / hmm.sum()
            cls = (F.cross_entropy(aux[:, n_reg:n_reg + 4], kernel_idx, reduction="none") * hmm).sum() / hmm.sum()
            ordr = 0.0
            if order_labels is not None:
                ordr = (F.binary_cross_entropy_with_logits(
                    aux[:, n_reg + 4], order_labels.float(), reduction="none") * hmm).sum() / hmm.sum()
            l = (reg + 0.5 * cls + 0.5 * ordr) * self.w["aux"]
            total = total + l; parts["aux"] = l.item()
        return total, parts
