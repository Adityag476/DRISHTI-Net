# DRISHTI-Net v5/v6 — Reviewer Evidence Pack (2026-08-16)
Every line below is raw tool output or raw code. Nothing is paraphrased.
CPU sandbox (2 GB RAM, torch 2.13+cpu); GPU numbers come from the RTX-4060 runs attached separately.

---

## 1. Claude demand #1 — `python scripts/test_losses.py` raw stdout

```
[PASS] full loss finite  — pixel=0.6784 | ssim=0.4914 | sobel=0.5526 | fft=0.0197 | aux=0.0842
[PASS] grads finite & nonzero (pred)
[PASS] grads reach aux head
[PASS] has_labels=0 -> aux term absent (never crashes on real-only batches)
[PASS] order BCE: aligned < opposed  — aligned=1.1070 opposed=3.6070
[PASS] MS-SSIM clamp local to the term  — raw=0.491438 pre-clamped=0.491438
[PASS] ssim_scale=0 disables term (ramp start)
[PASS] aux slices for 3-reg legacy checkpoints
[PASS] aux slices for 5-reg post checkpoints
[PASS] 128px fallback to SSIM works  — parts=['pixel', 'ssim']
[PASS] charbonnier finite
[PASS] dw_charbonnier finite
[PASS] sobel finite
[PASS] fft_l1 finite

ALL LOSS TESTS PASS
```

## 2. Claude demand #2 — literal probe telemetry lines (M2f smoke, CPU, full run log excerpt)

```
[val] holding out 1 real pairs (excluded from training)
[val] holdout names: 000007.npy .. 000007.npy
[8/24] loss=0.1504 (pixel=0.0907 aux=0.0637) lr=8.99e-06 0.535s/it
  [probe @8] pred[-0.109,1.627] zAbs mean/max=0.878/1.932 film|g| mean/p95/max=0.000/0.000/0.000 film|b| mean/p95/max=0.000/0.000/0.001 gradNormPreClip=1.080
  [val @8] PSNR=19.86 dB  SSIM=0.2027
[16/24] loss=0.1301 (pixel=0.0655 aux=0.0646) lr=1.70e-05 0.536s/it
  [probe @16] pred[-0.035,1.536] zAbs mean/max=0.877/1.825 film|g| mean/p95/max=0.000/0.000/0.002 film|b| mean/p95/max=0.001/0.003/0.003 gradNormPreClip=0.884
  [val @16] PSNR=19.86 dB  SSIM=0.2029
[24/24] loss=0.1188 (pixel=0.0546 aux=0.0641) lr=2.50e-05 0.542s/it
  [probe @24] pred[-0.041,1.336] zAbs mean/max=0.875/1.892 film|g| mean/p95/max=0.000/0.001/0.006 film|b| mean/p95/max=0.003/0.007/0.008 gradNormPreClip=0.616
  [val @24] PSNR=19.88 dB  SSIM=0.2035
```
Distribution stats (mean/p95/max) are GPT round-3's refinement — live in this build.

## 3. Claude demand #3 — `_seed_worker` implementation (module-level, spawn-picklable) + item-19 output

```python
def _seed_worker(worker_id, base_seed):
    """MODULE-LEVEL + functools.partial on purpose: Windows spawns DataLoader workers,
    so the init fn must be picklable (a closure inside main() would crash with
    'Can't pickle local object' the moment workers>0)."""
    np.random.seed(torch.initial_seed() % 2**32)
    info = torch.utils.data.get_worker_info()
    ds = info.dataset if info is not None else None
    for j, d in enumerate(getattr(ds, "datasets", None) or ([ds] if ds is not None else [])):
        if hasattr(d, "worker_seed"):
            d.worker_seed(base_seed + worker_id * 7919 + j * 131)

# DataLoader(..., worker_init_fn=functools.partial(_seed_worker, base_seed=(args.seed + 1) * 100003))
```

item-19 (32 samples, FORCED SPAWN = Windows semantics):
```
unique param vectors: w0 32/32, w2(spawn) 32/32
  spk: w0 mean 0.115  w2 mean 0.102
  blur: w0 mean 0.593  w2 mean 0.661
  noise: w0 mean 0.581  w2 mean 0.561
  post_spk: w0 mean 0.646  w2 mean 0.652
  post_noise: w0 mean 0.634  w2 mean 0.606
ITEM-19 SANITY: PASS (spawn semantics = user's Windows)
```

item-19b — GPT's distribution-comparison demand (n=1024, spawn, workers 0/2/4):
```
workers=0: unique 1024/1024
workers=2: unique 1024/1024
workers=4: unique 1024/1024

param        |   w0 mean/std   |   w2 mean/std   |   w4 mean/std   | max|dmean|
spk          | 0.107/0.055      | 0.105/0.056      | 0.105/0.056      | 0.002
blur         | 0.602/0.234      | 0.592/0.230      | 0.592/0.230      | 0.011
noise        | 0.605/0.231      | 0.603/0.229      | 0.603/0.229      | 0.002
post_spk     | 0.668/0.193      | 0.663/0.194      | 0.663/0.194      | 0.006
post_noise   | 0.578/0.253      | 0.575/0.249      | 0.575/0.249      | 0.003
kernel freq rows w0/w2/w4:
[[0.244 0.253 0.239 0.264]
 [0.264 0.241 0.245 0.25 ]
 [0.264 0.241 0.245 0.25 ]]
order freq rows w0/w2/w4:
[[0.303 0.697]
 [0.287 0.713]
 [0.287 0.713]]
ITEM-19b (n=1024, spawn, workers 0/2/4): PASS
```
Side confirmation: order freq ≈ 0.30/0.70 = the engine's `order_p=0.7` (Day-0 measured 140–150/150
noise-first; 0.7 keeps 30% blur-first for OOD cover). All regression targets land in O(0.1–1)
(GPT's normalization-stability requirement: max|Δmean| = 0.011).

## 4. Claude demand #4 — the aux-head structural change, as code

models/drishti_net.py — DegradationEncoder:
```python
def __init__(self, in_ch=2, z_dim=64, aux_dim=8):          # old default preserved
    ...
    self.aux = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(True), nn.Linear(64, aux_dim))
```
data/dataset.py — SyntheticDataset.__getitem__ (5-wide regression targets, all NORM-divided):
```python
reg = np.array([p["speckle"] / NORM["speckle"], p["blur"] / NORM["blur"],
                p["noise"] / NORM["noise"],
                p["post_spk"] / NORM["post_spk"], p["post_noise"] / NORM["post_noise"]],
               np.float32)
```
train.py: `cfg = dict(..., aux_dim=10)   # 5 reg + 4 kernel + 1 order`
losses/fab_loss.py — dynamic slicing (3-reg legacy vectors also accepted, unit-tested):
```python
n_reg = int(reg_labels.shape[-1])
reg  = ... aux[:, :n_reg] ...
cls  = ... aux[:, n_reg:n_reg + 4] ...
ordr = ... aux[:, n_reg + 4] ...          # order BCE — previously dead weight
```

## 5. Claude's two-line check — `evaluate.py` load whitelist (predicted silent-reconstruct bug)

```python
model = DRISHTINet(**{k: cfg[k] for k in ("in_ch", "out_ch", "width", "z_dim",
                                          "use_film", "with_uncertainty", "aux_dim") if k in cfg})
```
`aux_dim` is IN the whitelist. Proof by execution, not assertion — zero-edit roundtrip:
```
[eval] device=cpu
[eval] 8 images
[eval] latency ms/img — median 149.85  mean 150.37  min 146.64  (n=8)
[eval] done -> /tmp/fxout
8/8 outputs: same names, float32, 256x256 (2x SR)         # checkpoint trained with aux_dim=10
legacy ckpt loads + forwards: torch.Size([1, 1, 128, 128]) OK   # cfg WITHOUT aux_dim (old runs)
```

## 6. FiLM tanh bound exists (GPT: "≤0.5 only meaningful if the bound exists")

models/drishti_net.py, NAFBlockFiLM.forward:
```python
g, b = self.film(z).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
g, b = 0.5 * torch.tanh(g), 0.5 * torch.tanh(b)   # bounded modulation (§11 fix)
```
Probe evidence above confirms: after 24 iters |g|max=0.006, |b|max=0.008 ≪ 0.5.

## 7. Correction to the record — LODO loud-warning timestamp (Claude is right)

My earlier table said "already fixed (loud warning) pre-review." Inaccurate: the file Claude
reviewed did NOT contain the warning; it was added AFTER his flag in the code-update cycle.
Current code (degradation.py, degrade(), loop-else):
```python
else:
    # retry budget exhausted inside a holdout corner: log loudly, never silently leak
    print(f"[engine] LODO retry budget ({max_retry}) exhausted (force_holdout={force_holdout}); "
          f"using last draw — widen ranges or raise max_retry")
```
Sandbox check: train-side exclusion holds — 40 draws with holdout={kernel:3} sampled kernels [0,1,2] only.

## 8. Windows-spawn end-to-end proof (the cat-1 the closure version would have caused)

Forced-spawn (mp.set_start_method("spawn")) training run, workers=2:
```
[4/12] loss=0.1429 (pixel=0.0805 ssim=0.0017 sobel=0.0607) lr=5.00e-06 1.110s/it
  [probe @6] pred[0.000,1.486] zAbs mean/max=nan/nan film|g| mean/p95/max=off film|b| mean/p95/max=off gradNormPreClip=1.056
  [val @6] PSNR=19.41 dB  SSIM=0.1903
[12/12] loss=0.1400 (pixel=0.0764 ssim=0.0048 sobel=0.0589) lr=1.30e-05 0.696s/it
done -> /tmp/run_spawn/latest.pt
```
(ssim ≈ 0.00x = ramp `min(1, step/1000)` near zero at step ≤12 — by design; term is alive.)

## 9. Third-time ask — day0_audit.py + audit outputs

`scripts/day0_audit.py` ships in this pack, together with the n=150 outputs the team has:
`audit.json`, `engine_fit.png`, `histograms.png`, `sigma_vs_mu.png`.
(These are the files as produced on the training machine.)
