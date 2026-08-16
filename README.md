# DRISHTI-Net (दृष्टि)

**D**egradation-aware **R**estoration via **I**mplicit **S**upervision for **H**igh-fidelity **T**echnology **I**nspection.
KLA i4C Hackathon — PS01: AI-Based Restoration of Degraded Images for Semiconductor Inspection.

One model, one forward pass: joint speckle denoise + Gaussian deblur + 2× super-resolution
(256²→512² or 128²→256²), grayscale, deterministic, OOD-hardened. ~3.47M parameters.

## 1. Setup (laptop, ~2 minutes)

Needs Python 3.9+ (3.10/3.11 recommended). Works on Windows, Linux, macOS; NVIDIA GPU optional.

```bash
git clone <this-repo-url> && cd DRISHTI-Net
pip install -r requirements.txt
```

> CPU-only PyTorch is enough for inference and the smoke test. For training,
> an NVIDIA GPU is strongly recommended (free Google Colab T4 works).

Verify the install in 30 seconds with the bundled demo + placeholder smoke weights:

```bash
python evaluate.py --input_dir demo/lq --output_dir demo/restored --weights tests/smoke/drishti_net_smoke.pt
# expected: 4 restored 256x256 uint16 images in demo/restored + a latency line
```

## 2. Inference (what the benchmark runs)

```bash
python evaluate.py --input_dir /path/to/degraded --output_dir /path/to/restored
# flags: --weights weights/drishti_net.pt    --tta (8x self-ensemble, default OFF)
```

- Zero manual edits: auto CUDA→CPU, FP16 with FP32 fallback, weights auto-download if URL set.
- Reads `.png/.tif/.bmp/.jpg` **and `.npy`** (the KLA train format). `.npy` inputs are
  restored as `.npy` with the same filename and dtype.
- Outputs use the **same filenames and dtype/range as inputs** (uint8→uint8, uint16→uint16, float→float/.tif).
- Preprocessing constants are read from the checkpoint config — fixed scale, never clips
  (speckle legitimately exceeds the GT range; clipping would destroy measurements).

## 3. Day-0 data audit — run FIRST on the KLA dataset

Measures (never assumes) dtype/bit-depth, GT↔degraded size ratio, speckle family
(multiplicative vs additive), blur σ (spectral MTF), additive noise floor, and
random-searches our degradation engine to fingerprint the real corruption parameters:

```bash
python scripts/day0_audit.py --gt_dir DATA/gt --lq_dir DATA/degraded --out audit_report --fit_engine
# -> audit_report/audit.json + plots + recommended SCALE/ranges for MASTER_SPEC.md
```

## 4. Training

```bash
# synthetic-only start (after the Day-0 audit locks engine ranges):
python train.py --gt_dir DATA/gt --level M0 --iters 30000 --batch 16

# with real KLA pairs mixed in (aux loss auto-masked on real pairs — two-batch-type rule):
python train.py --gt_dir DATA/gt --real_lq DATA/degraded --real_gt DATA/gt --level M4 --iters 30000 --batch 16
```

Levels follow the experiment ladder (M0 baseline → M4 full + SEM/CutBlur/DW-Charb → M5 hardening);
a component ships only if it beats the previous level on the validation suite (see `MASTER_SPEC.md`).
Checkpoints land in `runs/<level>/latest.pt` — copy the final one to `weights/drishti_net.pt`.

## 5. Repo map

```
evaluate.py             zero-edit CLI inference (benchmark-critical)
train.py                ladder-driven training: EMA, bf16 AMP, cosine LR, two-batch-type loader
models/drishti_net.py   FiLM-conditioned NAFNet + supervised degradation encoder (3.47M params)
data/degradation.py     synthetic degradation engine (returns true params for aux supervision)
data/dataset.py         SyntheticDataset + RealPairsDataset (aux masked on real pairs)
losses/fab_loss.py      FabLoss: Charbonnier (V1) · Charb+Sobel+FFT-mag (V2, shipped) · ablation terms kept for evidence
utils/metrics.py        PSNR / SSIM / edge / flat-region residual σ / worst-patch / LPIPS diagnostic
scripts/report.py       holdout metrics + timing harness (both champions, reports/*.md)
scripts/latency_trace.py  per-image latency trace (cold-start vs steady-state reconciliation)
scripts/lodo_eval.py    unseen-synthetic-family OOD generalization probe (eval-only; scope note in docstring)
scripts/day0_audit.py   dataset fingerprinting (see §3)
scripts/make_demo_data.py  synthetic wafer stand-in generator
weights/                drishti_net.pt — THE shipped champion (exactly one model file)
tests/smoke/            drishti_net_smoke.pt — smoke-test placeholder (development only)
demo/                   4 sample pairs + restored outputs for a no-dataset install check
docs/                   hackathon deck (KLA 9-slide template) + audit report + panels
MASTER_SPEC.md          build-locked decisions, ablation ladder, validation protocol, vetoes
```

Status: SHIPPED — champion = FabLoss V2 (Charbonnier + 0.2·Sobel + 0.1·FFT-magnitude),
gate-verified on the 160-frame held-out split (exact checkpoint values, full
trajectories, and the dual report.py evidence are in reports/ and MASTER_SPEC.md §11).
Every number in this repo is measured on recorded runs — nothing estimated.
