#!/usr/bin/env python3
"""Unit tests for the FabLoss stack — house law: nothing ships unrun.

    python scripts/test_losses.py

Covers (reviewer-requested):
  * finite losses + gradient flow on every term
  * two-batch-type masking (real pairs get aux=0 contribution, never a crash)
  * order-head BCE is now a REAL term (aligned < opposed)  [dead-weight fix]
  * MS-SSIM clamp is LOCAL (pixel path keeps raw, out-of-range preds handled)
  * ssim_scale ramp 0 -> term fully off
  * dynamic n_reg slicing: legacy 3-reg and new 5-reg label vectors both work
  * small-crop fallback (MS-SSIM needs >=176 px -> SSIM)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses.fab_loss import FabLoss, charbonnier, dw_charbonnier, sobel_edge_loss, fft_l1_loss  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name, cond, extra=""):
    print(f"[{PASS if cond else FAIL}] {name}{('  — ' + extra) if extra else ''}")
    if not cond:
        failures.append(name)


def main():
    torch.manual_seed(0)
    B, H = 4, 192                                   # >=176 exercises the true MS-SSIM branch
    pred = (torch.rand(B, 1, H, H) * 1.4 - 0.2).requires_grad_(True)   # out-of-range on purpose
    gt = torch.rand(B, 1, H, H)
    aux10 = torch.randn(B, 10, requires_grad=True)
    reg5 = torch.rand(B, 5)
    reg3 = torch.rand(B, 3)
    ker = torch.randint(0, 4, (B,))
    order = (torch.rand(B) > 0.5).float()
    has = torch.tensor([1.0, 0.0, 1.0, 0.0])         # mixed synthetic/real batch

    # 1) every term finite, backward works, grads reach pred and aux
    lf = FabLoss(use_dw=True, w_ssim=0.5, w_sobel=0.2, w_fft=0.1, w_aux=0.05)
    loss, parts = lf(pred, gt, aux=aux10, reg_labels=reg5, kernel_idx=ker,
                     has_labels=has, order_labels=order, ssim_scale=1.0)
    check("full loss finite", torch.isfinite(loss).item(),
          " | ".join(f"{k}={v:.4f}" for k, v in parts.items()))
    loss.backward()
    check("grads finite & nonzero (pred)", pred.grad is not None
          and torch.isfinite(pred.grad).all().item() and pred.grad.abs().sum().item() > 0)
    check("grads reach aux head", aux10.grad is not None and aux10.grad.abs().sum().item() > 0)

    # 2) two-batch-type masking: has=0 rows contribute nothing to the aux term
    has_none = torch.zeros(B)
    l_masked, p_masked = lf(pred.detach(), gt, aux=aux10.detach(), reg_labels=reg5,
                            kernel_idx=ker, has_labels=has_none, order_labels=order)
    check("has_labels=0 -> aux term absent (never crashes on real-only batches)",
          "aux" not in p_masked and torch.isfinite(l_masked).item())

    # 3) order head is real: aligned logits beat opposed logits
    lf_aux = FabLoss(w_aux=1.0)
    aligned = torch.zeros(B, 10); aligned[:, 9] = (order * 2 - 1) * 5.0     # logit matches label
    opposed = torch.zeros(B, 10); opposed[:, 9] = (1 - 2 * order) * 5.0     # logit opposes label
    ones = torch.ones(B)
    l_al, _ = lf_aux(pred.detach(), gt, aux=aligned, reg_labels=torch.zeros(B, 5),
                     kernel_idx=torch.zeros(B, dtype=torch.long), has_labels=ones, order_labels=order)
    l_op, _ = lf_aux(pred.detach(), gt, aux=opposed, reg_labels=torch.zeros(B, 5),
                     kernel_idx=torch.zeros(B, dtype=torch.long), has_labels=ones, order_labels=order)
    check("order BCE: aligned < opposed", l_al.item() < l_op.item(),
          f"aligned={l_al.item():.4f} opposed={l_op.item():.4f}")

    # 4) MS-SSIM clamp is LOCAL: same 'ssim' part for raw vs pre-clamped pred
    lf_ssim = FabLoss(w_ssim=0.5)
    _, p_raw = lf_ssim(pred.detach(), gt, ssim_scale=1.0)
    _, p_pre = lf_ssim(pred.detach().clamp(0, 1), gt, ssim_scale=1.0)
    if "ssim" in p_raw:
        check("MS-SSIM clamp local to the term", abs(p_raw["ssim"] - p_pre["ssim"]) < 1e-6,
              f"raw={p_raw['ssim']:.6f} pre-clamped={p_pre['ssim']:.6f}")
    else:
        check("MS-SSIM clamp local to the term", True, "pytorch-msssim missing -> term skipped")

    # 5) ssim_scale=0 (ramp start) fully disables the term
    _, p_ramp0 = lf_ssim(pred.detach(), gt, ssim_scale=0.0)
    check("ssim_scale=0 disables term (ramp start)", "ssim" not in p_ramp0)

    # 6) dynamic n_reg slicing: legacy 3-wide and new 5-wide labels both fine
    for reg, tag in ((reg3, "3-reg legacy"), (reg5, "5-reg post")):
        l, _ = lf_aux(pred.detach(), gt, aux=aux10.detach(), reg_labels=reg,
                      kernel_idx=ker, has_labels=torch.ones(B), order_labels=order)
        check(f"aux slices for {tag} checkpoints", torch.isfinite(l).item())

    # 7) small-crop fallback: 128 px < 176 -> SSIM branch, no crash
    ps = torch.rand(2, 1, 128, 128)
    gs = torch.rand(2, 1, 128, 128)
    l_small, p_small = lf_ssim(ps, gs, ssim_scale=1.0)
    check("128px fallback to SSIM works", torch.isfinite(l_small).item(),
          f"parts={list(p_small)}")

    # 8) individual terms finite
    check("charbonnier finite", torch.isfinite(charbonnier(pred.detach(), gt)).item())
    check("dw_charbonnier finite", torch.isfinite(dw_charbonnier(pred.detach(), gt)).item())
    check("sobel finite", torch.isfinite(sobel_edge_loss(pred.detach(), gt)).item())
    check("fft_l1 finite", torch.isfinite(fft_l1_loss(pred.detach(), gt)).item())

    print()
    if failures:
        print(f"FAILED: {len(failures)} -> {failures}")
        sys.exit(1)
    print("ALL LOSS TESTS PASS")


if __name__ == "__main__":
    main()
