# DRISHTI-Net (दृष्टि) v3.0 — MASTER SPEC (BUILD-LOCKED)

> Status: BUILD-LOCKED. Name ratified by team lead: **DRISHTI-Net** (final).
> Rounds: 6 review cycles across 5 AI systems; architecture unanimous; remaining mechanisms gated by experiment ladder.
> KLA i4C Hackathon — PS01: AI-Based Restoration of Degraded Images for Semiconductor Inspection.
> **DRISHTI-Net** (दृष्टि = *vision*) = **D**egradation-aware **R**estoration via **I**mplicit **S**upervision for **H**igh-fidelity **T**echnology **I**nspection.
> House law (ChatGPT): **no architecture change without experimental evidence.** If it isn't here, it isn't in the model.
> Formerly referred to in review rounds as "DRAI-NAFNet" — renamed; nothing else changed.

---

## 1. Task

- One joint task: denoise (multiplicative speckle) + deblur ("Gaussian" per brief) + 2× SR, grayscale, **single forward pass**.
- 256×256→512×512 or 128×128→256×256. One model, both sizes.
- Scored: SSIM, pSNR, LPIPS, **H100 inference time**. Output must match GT dtype/range exactly.
- OOD test sources: generalization is a primary grading axis.
- Speckle pushes intensities **beyond GT range** → never clip, never per-image min-max.

## 2. Locked architecture (reference config)

```
Input (float32) → [x/PEAK, log1p(x/PEAK)]  (2 ch; PEAK = norm_peak, dtype-auto: 65535/255/1.0; Day-0 audit confirms; raw 1-ch = ablation)
        │
        ▼
DEG ENCODER: conv3×3/s2 (32) → conv3×3/s2 (64) → conv3×3 (64) → GAP → z ∈ R^64
        └─► AUX HEAD (TRAIN ONLY, masked on real pairs):
            3 regressions (σ_speckle, σ_blur, σ_noise) + kernel 4-way CE + order binary
        ▼
Backbone features modulated per block: FiLM(z) → γ,β  (added as y·(1+γ)+β inside NAFBlock)
NAFNet U-Net:
  intro conv3×3 (2→32)
  encoders [2,2,4] @ [32,64,128]  (down: conv2×2/s2)
  middle  4 @ 256
  decoders [2,2,2] @ [128,64,32]  (up: conv1×1(c→2c) + PixelShuffle2 → c/2)  ← correct channel math
  skip add at 3 scales (gated variant = Phase 2)
SR head: conv3×3(32→128) + PixelShuffle2 + conv3×3(32→1)
Output = bilinear×2(input) + head residual                      ← global residual
Pre-wired: 2nd output channel (uncertainty, Phase 2 — not computed at inference in v1)
```

~28 NAFBlocks (SimpleGate + SCA + LayerNorm2d, **no BatchNorm**), ~3.5M params (measured at build, not claimed).

## 3. Training

- **Two-batch-type rule (Claude's amendment — Day-1 loader design, non-negotiable):** dataset yields `has_deg_labels`. Synthetic (GT→engine): full loss. Real KLA pairs: aux-loss weight 0. Never crash, never silently skip on all data.
- AdamW (β2=0.99), lr 1e-3, cosine → 1e-6, wd 1e-4; bf16 AMP; LR crops 128×128 (HR 256×256); batch ≥16; **mixed-scale in every batch**; EMA 0.999.
- Curriculum: warmup epochs on mild degradation → full randomization.
- Degradation engine (per sample, per epoch, random params **and order**):
  speckle (multiplicative Gamma/Rayleigh, UNCLIPPED, σ~[0.02–0.15]) · Gaussian blur σ∈[0.5,2.5] · additive Gaussian σ∈[5/255,25/255] · downsample kernel {bilinear,bicubic,area,lanczos} + random pre-blur · Tier-1.5 SEM artifacts (scan-row jitter ±1px, column FPN banding, charging bloom) · CutBlur · D4 augs.
- **FabLoss V1**: `DW-Charb(α=2.0) + 0.5·(1−MS-SSIM) + 0.2·Sobel-L1 + 0.1·FFT-L1 + 0.05·aux`
  DW-Charb: `W = 1 + α·|∇GT|/max|∇GT|`; LPIPS = metric only; all λ flow through the ladder.

## 4. Metric suite & validation protocol

- Standard: pSNR / SSIM / LPIPS (grayscale → ×3 channels, VGG).
- Metrology suite: **Edge-SSIM** (SSIM of Sobel maps) · **flat-region LPIPS penalty** (hallucination detector) · **edge-region pSNR** (GT-gradient mask) · **worst-10% patch metrics** (64×64 patches, stride 32).
- **LODO**: hold out entire degradation corners (e.g., lanczos+strong-speckle+blur; one SEM artifact) → honest local OOD numbers (ID pSNR, OOD pSNR, gap).
- Replica timing harness: FP16, 10-iter warmup, `torch.cuda.synchronize`, median ≥100 runs. **Only measured numbers in the deck.**

## 5. Evaluation script (the file that decides if we score)

- `python evaluate.py --input_dir X --output_dir Y` — zero edits; auto CUDA→CPU; FP16 w/ FP32 fallback; channels_last; `--tta` default OFF; weights auto-download; warmup+sync timing optional verbose; identical filenames out.
- Preprocessing: fixed SCALE (Day-0 constant), never clip, log1p channel from data pipeline; output cast to **exact input dtype/range**; aux/uncertainty heads not run at inference.
- Gate: runs on **fresh clone + clean venv** from requirements.txt alone.

## 6. Experiment ladder (house law — evidence or it doesn't ship)

| Model | Delta | Proves |
|---|---|---|
| 0 | NAFNet + Charbonnier — arm **M0r** = real-only (`--no_syn`, reviewer-demanded baseline); arm M0 = + mild synthetic | baseline |
| 1 | + degradation randomization | OOD gain |
| 2 | + supervised deg-encoder + FiLM | conditioning gain |
| 3 | + MS-SSIM / Sobel / FFT (each ablated) | loss-term value |
| 4 | + DW-Charb + SEM artifacts + CutBlur | defect preservation |
| 5 | FP16/ONNX hardening | latency |
| Tier 3 | uncertainty head · z-gated skips · multi-scale z · cycle-loss (deterministic part only: blur+down(ŷ) vs blur+down(GT)) · distillation | only after Model 4 gates pass |

## 7. Day plan & roles

| Day | Builder work (Arena + Kimi) | Review gate (Claude + ChatGPT) | Gemini (hybrid) |
|---|---|---|---|
| 0 | **Data audit**: dtype, GT/degraded histograms, speckle-family fit, blur-vs-noise fingerprint, downsample-kernel fingerprint, order fingerprint → SCALE + engine params | Verify audit conclusions vs plots | — |
| 1 | Repo scaffold, engine, metric harness, two-batch-type loader, eval.py skeleton | Loader audit (batch-type masking!) | Deck skeleton |
| 2 | Model 0–2 training | Baseline review | Arch. diagram |
| 3 | Model 3 ablations | Loss-ablation review | — |
| 4 | Model 4 + LODO + patch metrics | OOD-evidence review | Deployment plan |
| 5 | FP16/ONNX, timing harness, eval.py fresh-clone test | **Eval-script audit (strict)** | ONNX/TensorRT |
| 6 | Visuals (3-way + deg-estimate demo), README, PPT, video | Deck fact-check (measured only) | Final deck |

**Phase 2/3** items (§6 Tier 3) only after Day-4 gate passes.

## 8. Hard vetoes (no appeal)

Diffusion/GAN · Mamba/SSM · MoE/hard routing · gradient-step TTA · Anscombe VST · per-image min-max / clipping · uint8-lossy eval output · BatchNorm · unmeasured numbers in the deck · architecture changes without ladder evidence · absolute "never hallucinates" claims (defensible wording only — ChatGPT r6).

## 9. Pitch (one sentence)

> *"DRISHTI-Net reads the corruption before repairing it: an implicit degradation encoder — supervised by ground-truth parameters from a 10,000-variant synthetic degradation engine — modulates a NAFNet backbone via FiLM, while a defect-weighted, metric-aligned loss protects the edges semiconductor inspection depends on; LODO validation and worst-patch metrics turn OOD robustness from a claim into measured evidence — in a few milliseconds per image on H100."*

## 10. Day-0 audit results — KLA train set (measured 2026-08-15, 3200 pairs)

| Measurement | Value | Consequence |
|---|---|---|
| Format | float32 `.npy`; GT 256² / LR 128² (2×) | loaders + audit + evaluate patched for `.npy`; patch_hr=256 = full image |
| GT range | [0.0, 1.0] | norm_peak = 1.0 (dtype-auto) — no config override needed |
| **Degraded range** | **[−0.279, +2.158]** | **never-clip veto CONFIRMED on real data**; model input spans ±2 GT range; output clamp [0,1] valid (GT-bounded) |
| LQ > GT max | 100% of pairs (48 sampled) | multiplicative speckle tail past GT range confirmed |
| Speckle family | CV=flat-region 0.174 (mixed); engine-fit σ ≈ 0.055–0.074 (med 0.061) | multiplicative confirmed; train range [0.02–0.15] covers ±2× measured — keep |
| Direct flat-residual σ | 0.111 | **total residual UPPER BOUND** (speckle leakage + blur mismatch + alignment + additive) — NOT pure additive; superseded by fitted σ |
| Additive σ (engine fit) | 0.028–0.039 (med 0.034 ≈ 8.7/255) | **initial estimate**; our [5/255, 25/255] covers — keep |
| Blur σ (engine fit) | 0.91–1.35 (med 1.12); direct spectral-MTF n/a (noise floor inverts ratio — root-caused) | **initial estimate**; our [0.5, 2.5] covers — keep |
| Kernel votes | uniform {0:3, 1:2, 2:4, 3:3} | no evidence for any single kernel → random-kernel mix stays — keep |
| **Order votes** | **noise-first 12/12** | engine `order_p = 0.7` (70% match / 30% blur-first OOD cover) — **pending ladder ablation (random vs fixed-order) at M4** |
| Engine fidelity | best-fit PSNR med 23.9 dB | **correct read:** noise-realization ceiling bounds this metric — same-params independent draws cap at ~26–31 dB (content-dependent). Fit within ~2–4 dB of ceiling ⇒ family approximately adequate; gap flagged for n≥150 rerun + worst-pair eyeball |

Gate status (Gemini/Claude review): **CONDITIONAL GO — numbers recorded as Day-0 INITIAL ESTIMATES, not locked.**
Open follow-ups before M1 locks engine ranges: (1) rerun upgraded audit at `--fit_pairs 150`
with per-pair noise-ceiling comparison; (2) mean-vs-variance speckle plot review;
(3) order-placement ablation scheduled in ladder. M0 (real pairs, Charbonnier) is
independent of all open items → start immediately, in parallel.

---

## §10b — Audit REPLICATION at n=150 (2026-08-15, real KLA data, upgraded script)

Census + fit rerun with ceiling-per-pair tooling. **`[A]`**: 3200/3200 pairs re-confirmed; GT/LQ ratio = 2 for 100% of 3200 (no sampling). LQ brighter than resolution-matched GT in **100% of pairs** → multiplicative speckle tail past GT range confirmed at full census. Speckle CV (flat) 0.187 (n=48 gave 0.174 — consistent). Flat-residual σ TOTAL (upper bound) 0.0827 (was 0.1105 @ n=48).

**Engine fit, n=150 × 120 iters:**
- speckle σ p25/med/p75 = 0.0528 / **0.0573** / 0.0630
- blur σ = 0.832 / **1.109** / 1.545
- noise σ = 0.0260 / **0.0320** / 0.0428
- kernel votes {0:52, 1:24, 2:29, 3:45} — spread, NO dominant kernel (4-kernel bank may be too narrow)
- order votes noise-first **150/150** → `order_p=0.7` retained; raising deferred to M4 ablation (evidence-first rule)

**OPEN RISK — family gap:** best-fit PSNR med = **20.92 dB** vs noise-limited ceiling med = **27.75 dB** ⇒ 6.8 dB gap (n=12 showed 23.90 vs 27.65 = 3.75 dB; the 12-pair sample was easier content). Parameter medians *replicated* across independent samples (n=12 vs n=150 within noise) ⇒ calibration trusted; but a per-image unmodeled component exists (candidates: non-Gaussian/asymmetric PSF, spatially-varying blur, small post-downsample term, tone curve). Mitigations, in order: (1) M0r real-only arm already isolates engine mismatch entirely; (2) M0-vs-M0r held-out-real val comparison = empirical synthetic-transfer test; (3) residual autopsy on worst pairs (tool pending audit.json schema); (4) kernel-bank widening ONLY if autopsy shows edge-localized residual energy.

Spectral-MTF blur (low-band patch) = 0.205 — secondary diagnostic, biased low under speckle noise floor; engine fit remains primary. Status: **CONDITIONAL GO maintained**, family-gap escalated to active investigation.

---

## §10c — Detector-scale module + the PSNR-fit degeneracy correction (2026-08-15)

**Correction (supersedes the fit-vs-ceiling reading in §10/§10b).** Controlled A/B identifiability
fixtures (8 pairs each, known truth): inject post-downsample speckle σ=0.16 → the random-search
PSNR fit recovers post_spk **0.0248** and STILL shows a fit-vs-ceiling gap of 8.8 dB. Root cause is
structural: PSNR-maximization against a noisy target collapses to the **smooth conditional-mean
corner** (matching independent noise realizations can only lose PSNR: two draws at σ=0.16
disagree at ~19 dB while the smooth corner reaches 20.7), and the ceiling is then measured at that
same smooth corner (29.5 dB) ⇒ the "gap" is **the noise energy of the target itself, an artifact —
not a family verdict**. The §10b "6.8 dB family gap alarm" is hereby downgraded accordingly. The
earlier n=12 "family adequate (23.9 vs 27.65)" reading suffered the same artifact in the other
direction. Direct, estimator-free evidence stands untouched: slope 0.157 ≫ intercept 0.0106
(multiplicative at LQ scale), tail → ~1.45 (never-clip justified), CV 0.187.

**Physics conclusion + engine change (adopted):** GT-scale speckle cannot survive the fitted
blur (×0.26 on white-noise std) + area-downsample (×0.5) → the dominant multiplicative term lives
AT detector scale. `DegradationEngine` gains a post-downsample module (post_spk std at LQ scale +
post_noise; `fixed` dict backward-compatible). Training defaults adopt the KLA-matched single-stage
model: mild — post_spk (0.12,0.19) med≈0.155, post_noise (0.006,0.014) med 0.010, GT spk var
(0.002,0.01); full — post_spk (0.08,0.24), post_noise (0.003,0.020), GT spk var (0.002,0.03) for
OOD cover. Aux head unchanged (8 outputs; post params unsupervised, FiLM handles implicitly).

**New family gate: `--synth_check`.** Compares DISTRIBUTION-level stats (flat CV, σ-vs-μ slope,
bright tail) of engine-default synthetic vs real population; MATCH = rel. deltas <25% + tail>1.1.
Fixture verification: fxpost (KLA-like, slope 0.157) and fxpre (slope 0.053) both correctly flag
MISMATCH vs pre-adoption defaults; post-adoption tail overshoot fixed (2.09→1.53 vs target 1.43).
Known estimator caveat: pooled flat-residual slope reads ≈0.62× injected σ on fixtures — SAME bias
on both arms ⇒ the comparative gate stays fair; final calibration numbers come from the real-data
synth-check run (user-side), not from fixtures.

Audit now also reports: bound-pinning counts per parameter (>10% at an edge = truth outside
search range — this is how the detector-scale discovery was caught), per-pair fit records in
audit.json, and `residual_autopsy.png` (worst-6 fits: GT↓ | real LQ | best synthetic | residual).

**Order of adoption (consensus rule honored):** engine change precedes M0/M1 launches; validated
by real-data `--synth_check` MATCH before the synthetic arms train. M0r (real-only) unaffected.

---

## §11 — Ladder training log (all runs: 30k iters, batch 8, bf16, RTX 4060 8GB,
## val = 160 held-out REAL pairs, full-image, EMA; blockers: none)

**M0r — real-only baseline (level M0, Charbonnier pixel loss only, no synthetic) — COMPLETE 2026-08-15**
- Wall: ~78 min (0.155 s/it). Curve: 25.19 (@1k) → 27.37 (@2k) → 28.49 (@3k) → 29.01 (@5k) →
  converged ≈13k; peak 29.33 @19–21k; final @30k: 29.30. No divergence, no NaN, smooth cosine decay.
- **Baseline: PSNR 29.33 dB | SSIM 0.7913** (best.pt @ val peak). L1-only → conditional-mean
  behavior: strong denoise PSNR, SSIM capped ~0.79 (over-smooth) — exactly the headroom M3's
  SSIM/Sobel/FFT terms and M2's FiLM conditioning are designed to exploit.
- Precedent set: every later arm must beat BOTH numbers on this same 160-pair val before it ships.

---

## §10d — Real-data family gate: **PASS** (2026-08-15, n=200 synth vs full census)

`--synth_check` v2 on real KLA data:
- flat CV: real 0.1874 vs synth 0.2219 (+18%, inside 25% band)
- σ-vs-μ slope: real 0.157 vs synth 0.149 (−5% — spot on)
- intercept: 0.0106 vs 0.0156 (+0.005 absolute — minor)
- bright tail p99.99: real 1.359 vs synth 1.530 (max 1.656) — synth slightly HOTTER than real:
  deliberate over-cover of the speckle tail (KLA test = OOD sources; under-covering tails would be
  the dangerous direction). Never-clip guarantee confirmed on real data.
- **VERDICT: MATCH ⇒ synthetic family adequate ⇒ M-series GO.**

Supporting evidence from the same run (v2 fit, widened ranges incl. true-zero):
- Residual autopsy on worst fits: **edge-correlation med = +0.005 ≈ 0** ⇒ residual mismatch is
  UNSTRUCTURED noise, not blur/kernel family error (answers Gemini's worst-pair eyeball gate:
  nothing structural is missing from the model; the residual is exactly the detector-scale
  stochastic term §10c added).
- Order votes 140/150 noise-first (order_p=0.7 retained; final value is M4-ablation territory).
- post_spk/post_noise pin at/near zero in the PSNR fit (22/150, 13/150 at lo) — exactly as §10c
  predicts: per-pixel PSNR fitting cannot see detector-scale noise; [C] is diagnostic-only,
  the distribution-level [D] gate is the verdict.

Locked training decision: M0 mixed arm = mild engine + real pairs, gate-cleared.
Baseline to beat (§11): M0r = 29.33 dB / 0.7913 SSIM.

**Ladder automation (war-mode):** `scripts/run_ladder.bat` (Windows chain M0→M4, or subset via args),
`scripts/harvest_ladder.py` (reads runs/*/best.pt, prints gate table: a rung becomes champion only by
beating the running best on BOTH val PSNR+SSIM), `scripts/report.py` (champion autopsy on the EXACT
160-pair holdout, evaluate.py-identical code path → JSON+MD autofill for Slide 6). All sandbox-verified.

---

## §11b — Ladder POSTMORTEM (overnight 2026-08-15/16, all rungs 30k iters)

| rung | film | engine | loss | best val | verdict |
|---|---|---|---|---|---|
| M0r | F | — | L1 | **29.33 / 0.7909** | champion (gate) |
| M0  | F | mild | L1 | 29.18 / 0.7874 | discarded (−0.15 dB vs M0r) |
| M1  | F | full | L1 | 29.14 / 0.7863 | discarded |
| M2  | T | full | L1+aux | 29.08 @15k → **collapse @~15.5k → 13.6** | gate fail |
| M3  | T | full | +SSIM/Sobel/FFT | broken from iter 1 → 13.6 | gate fail |
| M4  | T | full+SEM+cb | +DW | broken from iter 1 → 13.6 | gate fail |

**Fingerprint: film=False healthy 3/3, film=True failed 3/3.** Loss components all sane
(fft≈0.003, ssim≈0.12 — not a weight bug). Root cause in models/drishti_net.py:
FiLM gains `y*(1+g)+b` driven by **unbounded z** (raw Linear output, no normalization),
zero-init → identity at start, then positive-feedback runaway between z growth and FiLM
deviation. M2's clean loss made the threshold ~15.5k iters; M3/M4's richer gradient
stream through the same FiLM params crossed it immediately.
**Fix (evidence-backed):** z_norm=LayerNorm(64) at encoder output + tanh-bounded FiLM
(g,b ∈ ±0.5). Zero-init preserved (identity at init). CPU-verified: fixed M2 trains
150 iters clean (val 20.32→20.65 rising, no blowup).
**M0/M1 honest negative result:** gate-cleared synthetic (MATCH) still lost −0.15/−0.19 dB
to real-only — capacity dilution between distributions. Reported as measured.
**Recovery rungs:** M3r (real-only + metric loss, film=False), M2f (FiLM-fixed mild).

## §11c — Reviewer round-2 consensus fixes (2026-08-16, Arena+Kimi build, Claude+GPT review)

Point-by-point adjudication of "gpt and claude reply.txt". ACCEPTED AND IMPLEMENTED:
1. **Worker-RNG P0 (Claude) — CONFIRMED BUG, fixed twice.** (a) Windows-spawn
   `DataLoader` workers inherited identical pickled RNG states → correlated synthetic
   streams in M0/M1/M2. (b) The FIRST fix attempt put `worker_init_fn` as a closure in
   `main()` — unpicklable under Windows spawn, would have crashed workers>0 instantly
   (Arena self-caught via spawn-mode sandbox test). Final fix: module-level
   `train._seed_worker` + `functools.partial` (spawn-picklable), reseeding each worker's
   own dataset copies via `get_worker_info().dataset` — no closures.
   **Claude's item-19 sanity now PASSES under forced spawn: 32/32 unique parameter
   vectors per worker, statistically similar distributions.**
   **Consequence: the M0/M1 "synthetic dilution" verdict now carries a confound caveat**
   — M1 gets one rerun post-fix after the champion path is secured.
2. **Aux head omitted the dominant corruptions (Claude P0) — fixed.** Regression head is
   now 5-wide: [speckle, blur, noise, **post_spk, post_noise**], aux_dim 8→10
   (5 reg + 4 kernel + 1 order). Old 8-wide checkpoints still load (build_from_config /
   evaluate.py both default aux_dim=8 when absent).
3. **Order head dead weight (Claude) — fixed.** `order` labels now flow train.py →
   FabLoss → masked BCE-with-logits at `aux[:, n_reg+4]` (n_reg-sliced, so legacy 3-reg
   vectors keep working). Unit-tested: aligned logits score strictly lower than opposed.
4. **MS-SSIM on unclamped predictions (Claude hypothesis) — mitigated two ways.**
   (a) `pred.clamp(0,1)` applied LOCAL to the MS-SSIM term only (pixel/Sobel/FFT keep the
   raw signal — large-error gradients preserved); (b) `ssim_scale = min(1, step/1000)`
   ramp — early-training residual chaos cannot detonate the SSIM gradient. Diagnostic
   levels M3a/M3b/M3c isolate SSIM-only / Sobel-only / FFT-only against the M3r family.
5. **FiLM root cause still asserted, not yet proven (GPT+Claude) — telemetry added.**
   Bounded FiLM + z-norm stand as the FIX; the MECHANISM now gets measured: every
   `--probe_every` iters the trainer prints pred[min,max], zAbsMax, max |g|/|b| across
   FiLM blocks, and PRE-clip grad norm. M2f keeps the probe on for the full run —
   if z/g/b stay bounded while val stays healthy, the feedback-loop hypothesis is
   confirmed; if it collapses again with bounded stats, FiLM is exonerated and the
   hunt moves to the encoder path. No "proven" claims in the deck until M2f telemetry lands.
6. **Cosine `T_max` off-by-warmup (GPT) — fixed** (`T_max = iters - warmup`).
7. **run_ladder.bat timestamps (Claude, root-caused by Arena) — fixed**:
   `setlocal enabledelayedexpansion` + `!DATE! !TIME!` (cmd expands `%VAR%` once per
   loop parse; all rungs previously logged identical start/stop times).
8. **train.py NameError (Arena, self-caught during v5 finalization):** probe loop used
   `NAFBlockFiLM` without importing it — v5 could not have trained one step. Fixed +
   sandbox-verified (M2f/M3b/M0r CPU runs below).
9. **evaluate.py (Claude/GPT) — three fixes:** (a) `aux_dim` added to the load tuple
   (new 10-wide ckpts load; old 8-wide stay compatible); (b) warmup now runs on the
   REAL first input tensor (fixed 256²-uint8 dummy left the true 128² float .npy shape
   cold → first timed image would have eaten per-shape cudnn autotune/allocator cost);
   (c) fp16→fp32 runtime fallback if autocast RuntimeErrors on the judge GPU.
10. **report.py (Claude) — two fixes:** silent-resize replaced by a hard RuntimeError
    (a resize would corrupt every metric silently); 10-iter GPU warmup precedes timing.
11. **Naming:** `_speckle` param documented as gamma VARIANCE; `_post_speckle` as STD.
12. **Deck wording law (Claude items 9/16/17/18):** the loss is "FFT magnitude
    consistency (L1, orthonormal)" — NOT "Focal Frequency Loss"; the flat-region metric
    is "flat-region residual σ (lower=better)" — NOT a "hallucination detector"; SSIM is
    "our Gaussian-window SSIM (consistent internal comparator)" — NOT "official KLA SSIM";
    LPIPS is a local-only diagnostic whose VGG weights come from the `lpips` package's
    torchvision download (document provenance, never imply it's the judge's scorer).
    No invention claims anywhere.
12b. **M2f labeling correction (Claude):** M2f with `--no_syn` tests *FiLM stability on
    real data only* — with zero synthetic batches the masked aux term receives no
    gradients, so "aux on" in old plan tables was misleading. The corrected aux-supervision
    mechanism (5-reg + kernel + order) can only be validated by a SYNTHETIC run (R3:
    fixed-seed M1/M2 rerun, post-champion), which is also where the worker-RNG fix gets
    its experimental verdict.
12c. **Crop-diversity disclosure (Claude items 13/14):** KLA images are 256² == patch_hr,
    so "random crops" degenerate to the full image (y=x=0 always) for BOTH datasets.
    Synthetic diversity comes entirely from degradation randomization + D4 augmentation.
    Deck must not claim crop diversity.
13. **Sandbox verification (this machine, CPU, 2026-08-16):** `scripts/test_losses.py`
    14/14 PASS; M2f 24 iters (FiLM+aux, mixed synthetic+real, workers=2) clean —
    probe zAbsMax≈1.9, bounded |g|,|b| growing smoothly from 0, grad norm decaying
    1.08→0.62, val monotone up; M3b 16 iters clean; M0r 16 iters clean;
    **M3d 12 iters under FORCED SPAWN (Windows semantics) clean** — worker init pickles,
    ssim ramp live (term ≈0 in ramp, nonzero after); evaluate.py zero-edit roundtrip
    8/8 .npy (name+dtype mirror, 2× upsample) on BOTH new aux_dim=10 and legacy
    aux_dim=8 checkpoints; dual-shape warmup code exercised; report.py full pass;
    engine LODO exclusion holds; 5-key `fixed=` BC holds; item-19 spawn RNG test PASS.

## §11d — Positions we defend (rejected reviewer demands, with reasons)

* **NAFBlock SCA order — CONSENSUS REACHED: variant owned, T4 deferred, no near-equivalence
  claims.** Both reviewers verified the official NAFNet code (`x = x*sca(x)` then `conv3`)
  and REJECTED our earlier "commutative / near-equivalent" rebuttal — SCA is a
  data-dependent gate, so `conv3(y)*sca(conv3(y))` ≠ `conv3(y*sca(y))`, and Claude's
  wording stands: *"'it's equivalent to the paper' is checkable and wrong."* Our final
  position (both reviewers explicitly agree this is NOT a blocker — GPT: "do not derail
  tonight's diagnostics over it"): DRISHTI-Net uses a **deliberate variant of the NAF
  block** (SCA applied post-projection); it is a valid channel-attention gate, the
  champion M0r trains healthy to 29.33 dB with it, and per the consensus rule no
  architecture changes happen without experimental evidence. The controlled T4 test
  (current order vs official order, same seed/iters/loss) is queued after the champion
  path is secured. Deck language: "NAF-style block (SCA placement adapted)" — never
  "exact NAFNet" and never "near-equivalent".
* **"LODO column in every validation" — DEFERRED, not rejected.** LODO mechanics are
  implemented and unit-verified (resample-with-retry + loud exhaustion warning). A second
  val column doubles val wall-time on a 78-min/rung budget inside 24 h. LODO runs after
  the champion is frozen (its synth side needs no GPU).
* **Deterministic tail holdout split — KEPT, disclosed.** Sorted-name tail (last 160 of
  3200) is the same split the champion was selected on; changing it now would break
  comparability with the 29.33 reference. train.py now PRINTS the exact holdout filename
  range so acquisition-order correlation can be inspected (Claude's ask), and the
  limitation is logged here: KLA filenames are numeric sequences, so a tail split may
  correlate with acquisition index. Sensitivity check (stride split via report.py) is a
  post-champion item.
* **Diagnostic time budget (GPT: 5–10k iters/rung) — ACCEPTED with 10k cap.** Sequence:
  test_losses → M3r → (M3a/M3b/M3c/M3d only if M3r unstable: one-term-at-a-time
  isolation of SSIM/Sobel/FFT/pair) → M2f --no_syn (real-only FiLM; aux dormant by
  design — see 12b) → optional M0r_raw (--no_log, GPT's T1) → winner gets 30k →
  T4 NAFBlock-control + R3 corrected-synthetic (ID+LODO) post-champion.

## §11e — Round-3 consensus (GPT round-3 + Claude evidence round, 2026-08-16)

1. **M3a–M3d are MANDATORY, not conditional (GPT, accepted — supersedes §11c skip rule).**
   A stable M3r bundle does not prove each term earns its place; one could be cancelling
   another. Arms run at **5k diagnostic iters each** (Charb+SSIM / +Sobel / +FFT /
   +SSIM+Sobel) — cheaper than skipping and re-debugging. M3r(=M3e) still runs 10k.
2. **v5-code baseline policy (GPT, accepted).** Old M0r=29.33/0.7909 is labeled
   **historical benchmark (v4 code)**. A fresh **M0r-v5 30k** reproduction re-establishes
   the gate on the same code revision as all challengers (v5 changed scheduler T_max and
   worker RNG even for real-only runs). The old best.pt artifact stays valid for
   inference (v5 evaluate.py loads it — verified); the revision-matching requirement is
   about training-curve comparability, not artifact validity.
3. **M2f renamed everywhere "M2f — Real-only FiLM"** (the aux head receives zero gradient
   under --no_syn; the old "aux on" phrasing was misleading — Claude, accepted).
4. **Distribution telemetry, not just max (GPT, accepted + implemented):** probe now
   prints `zAbs mean/max`, `film|g| mean/p95/max`, `film|b| mean/p95/max`,
   `gradNormPreClip`, `pred[min,max]`; FabLoss logs `ss_scale` (effective SSIM weight)
   inside parts. Verified live (evidence pack §2).
5. **Aux normalization stability (GPT concern, verified satisfied):** all 5 regression
   targets are NORM-divided and measured to land in O(0.1–1) (item-19b stats: means
   0.10–0.67); no target can dominate MSE by raw magnitude.
6. **Worker-distribution test extended to GPT's spec (accepted + executed):** n=1024,
   spawn, workers {0,2,4}: 1024/1024 unique everywhere, max|Δmean|=0.011, kernel/order
   frequencies uniform/0.70-0.30 as designed. PASS (evidence pack §3).
7. **Claude's evidence demands 1–4 + whitelist check:** all satisfied — raw stdout,
   literal probe lines, _seed_worker code+output, aux diff, and `aux_dim` IS in
   evaluate.py's load whitelist (predicted silent-reconstruct bug pre-empted; proof by
   roundtrip execution on aux_dim=10 and legacy ckpts). Bundle: docs/reviewer_evidence_v5.md.
8. **Record correction (Claude, accepted):** the LODO loud warning was added AFTER his
   flag, not "pre-review" as my earlier table implied (evidence pack §7).
9. **GPT's decision table adopted wholesale** (M2f<M0r ⇒ FiLM doesn't earn its place;
   M2f>M0r ⇒ full 30k; individual-vs-bundle divergence ⇒ investigate interactions;
   raw>log ⇒ drop log1p; etc.). Gate bar unchanged: beat running best on BOTH metrics.
10. **day0_audit.py + n=150 outputs** now live in the repo at docs/audit_report/
    (audit.json, engine_fit.png, histograms.png, sigma_vs_mu.png) — third-time-ask closed.

## §11f — Round-4 consensus (both reviewers approve the run; two GPT requirements added 2026-08-16)

1. **"Early pace" is NOT a gate (GPT, accepted — my own earlier phrasing struck).** A 10k
   run can learn faster and finish worse. Roles are fixed: M2f 10k = **diagnostic only**
   (stability + telemetry read); M3a–d 5k = **component screening**; M0r-v5 30k =
   **official baseline**. Champion criterion unchanged and singular: beat the running
   best on **BOTH** val PSNR and val SSIM at the full budget.
2. **Identical initialization for ablation arms (GPT, accepted + implemented).** M3a–d
   all run `--init_from runs/M0r_v5/latest.pt` plus the same --seed, data, split, batch,
   LR, iters, augmentation, and best-on-val selection rule. The question becomes "what
   does THIS loss term add to an already-working model?", not "which random run
   converged better?". Implemented as train.py `--init_from` (strict load, loud failure
   on arch mismatch — sandbox-verified on a film↔no-film mismatch). Chain tested:
   M0 → latest.pt → M3a --init_from trains normally.
3. **FiLM telemetry by depth (GPT, accepted + implemented).** Probe now also prints
   per-stage p95: `g95 enc1/enc2/enc3/mid/dec3/dec2/dec1` and same for b95 — one deep
   block saturating can no longer hide inside a global max. Verified live.
4. **Interpretation lock (GPT, accepted):** the worker-RNG fix proves streams are no
   longer duplicated; it does NOT prove the synthetic distribution is correct. M0/M1
   stay labeled **"confounded by the pre-v5 worker-RNG bug"** — synthetic is re-judged
   post-champion via ID + LODO, and only then earns or loses its place.

## §11h — Measured results day (2026-08-16, all on RTX 4060, same val holdout: last 160)

**OFFICIAL GATE — M0r-v5 (30k, v8 code): best.pt val PSNR 29.32 / SSIM 0.7913**
(val flat 29.32→29.29 over 17k–30k, SSIM still edging to 0.7913; parity with the old
v4 champion within noise — code revision validated as reference frame.)

**M2f 10k diagnostic (real-only FiLM): STABLE PASS — 29.19 / 0.7870 @10k, zero
instability.** Telemetry (the reviewers' demanded mechanism evidence): zAbs mean
0.46→0.25 declining, z_norm effective; every stage g95/b95 pinned at the tanh ceiling
(~0.46–0.48) instead of running away; gradNormPreClip decayed 0.12→0.009.
**Wording locked (GPT round-6):** deck says "the revised FiLM implementation remained
stable under 10k-step telemetry, eliminating the previously observed runaway behavior"
— NOT "root cause fixed" (no controlled old-vs-new isolating WHICH modification did it;
M2f also changed data vs the old M2). 10k numbers are diagnostic-only per §11f — never
gate-compared.

**M3a 5k ablation (real-only, init best.pt, lr 1e-4, warmup 100, Sobel/FFT/FiLM OFF,
local clamp + ramp ACTIVE): CATASTROPHIC COLLAPSE — val 29.32→13.20 / 0.7913→0.3257.**
Grad-norm pre-clip, ALL 10 probes (Claude round-9 correction — erratic/spiking, NOT a
smooth escalation; the earlier "4.2→26.8→155→232" subsequence over-smoothed the signal):
1.65, 28.67, 72.86, 4.16, 26.77, 155.43, 1.28, 3.21, 42.38, **232.00**. Locked phrasing:
"erratic gradient spikes up to ×232 (M3a) / ×736 (M3d) vs ≤0.2 in every stable arm."
The VAL trajectory is the monotone one (29.32→13.20); the GRADIENT signal is chaotic —
spikes to 72.9 as early as probe 2 that dip back to 1.28 before exploding again — which
is the stronger, correct reading of instability.
**Verdict (GPT round-6 wording, adopted):** "MS-SSIM is rejected from FabLoss V1 at
λ=0.5." NOT "inherently incompatible" — other formulations/weights remain theoretically
open, earn-your-place applies; no further SSIM exploration inside the 24 h budget.
Deck claim allowed: "a real-only MS-SSIM ablation produced severe gradient growth and
quality collapse, so MS-SSIM was removed from the final loss."

Overnight mystery accounting (both mechanisms now MEASURED):
- M2 collapse (15.5k) = FiLM/z runaway → revised implementation → stable (M2f).
- M3/M4 collapse = MS-SSIM pathology → M3a replicated it with FiLM OFF → term excised.
FiLM is exonerated for the M3/M4 failures; the two failure modes are distinct.

M3b/M3c/M3d pending at this writing; per the decision table a healthy Sobel or FFT arm
earns ONE 30k challenger run against the 29.32/0.7913 gate (both metrics). M3d serves
as the "Sobel does not rescue MS-SSIM" control.

## §11g — Round-5 launch gate (both reviewers: GO; four corrections applied 2026-08-16)

1. **M3d redirect typo in MY OWN launch block (Claude, caught + owned).** The pasted M3d
   line contained `> runs\M3d_5k > runs\M3d_5k.log` twice — broken even though the same
   message warned about it. (Mechanism note: PowerShell rejects double `>` at parse time;
   cmd.exe would create a file at that path and crash `os.makedirs` exactly as Claude
   described — either way, dead on arrival.) Full corrected block re-issued with one
   redirect per line, verified line-by-line this time.
2. **Warm-restart LR confound on init arms (Claude, accepted — sharp catch).** M3a–d
   inheriting default `--lr 1e-3 --warmup 1000` would re-ramp LR on a converged network
   through the same window where `ss_scale` ramps — instability would be un-attributable
   (loss term vs LR shock), defeating the isolation design. Fix: arms launch with
   `--lr 1e-4 --warmup 100` (gentle fine-tune LR, warmup ends before the SSIM ramp
   dominates). M2f/M0r-v5 keep the from-scratch defaults.
3. **Init from `best.pt`, not `latest.pt` (GPT, accepted).** The ablation question is
   "what does this loss add to our STRONGEST baseline" — best.pt is the max-val-PSNR
   checkpoint and also carries `val_psnr` into the [init] log line.
4. **Loader semantics hardened + auditable config header (GPT, accepted + implemented):**
   `--init_from` now tries strict load first, falls back only for the specific
   film=False→film=True case with a LOUD report of the 36 missing FiLM-branch tensors
   (left at zero-init by design), and CRASHES on any unexpected or non-FiLM missing
   keys — silent strict=False is forbidden. `[cfg]` header now prints
   level/seed/lr/warmup/iters/batch/film/use_log/synthetic/init_from + full loss weights
   at the top of every log. All three load paths sandbox-verified; optimizer/scheduler
   freshness confirmed print-side. GPT's criterion registered: telemetry boundedness is
   NOT the win condition — M2f earns continuation only by eventual gate-beating numbers;
   worker-RNG fix upgraded to "fixed AND experimentally verified", synthetic distribution
   still unproven (judge post-champion via ID+LODO).
5. **Claude's evidence-channel note (accepted procedure):** all future adjudications
   carry raw log file CONTENTS (pasted like run2.txt), not summaries of attachments.

## §11i — M3 battery COMPLETE + challenger launch (2026-08-16; arms: 5k iters, real-only,
init M0r_v5/best.pt, lr 1e-4, warmup 100, clamp + ss_scale ramp ACTIVE, FiLM OFF
in all four arms; raw logs pasted in-conversation)

**OFFICIAL GATE (checkpoint-verified, user one-liner):** `runs\M0r_v5\best.pt` =
**step 19000, val PSNR 29.3191, val SSIM 0.7911** (torch.load print:
29.319097739026585 / 0.7910710789446229). Gate comparison is best.pt-versus-best.pt:
only these two numbers count — the @30000 tail values (29.29/0.7913) are NOT the gate.

| arm | added term (on Charb) | final val PSNR/SSIM | gradNormPreClip | verdict |
|-----|----------------------|--------------------|-----------------|---------|
| M3a | MS-SSIM λ=0.5 | 13.20 / 0.3257 | → 232.0 | COLLAPSE |
| M3b | Sobel 0.2 | 29.31 / 0.7907 (flat @3500–5000) | ≤ 0.038 | STABLE, NEUTRAL |
| M3c | FFT 0.1 | 29.29 / 0.7924 (SSIM monotone 0.7916→0.7924) | ≤ 0.03 | STABLE, SSIM-LEAN |
| M3d | MS-SSIM 0.5 + Sobel 0.2 | 13.48 / 0.5606 | → 736.2 | COLLAPSE (control) |

- **M3b verdict (GPT round-7 "safe ≠ useful", adopted):** safe but neutral at
  convergence (−0.01 dB / −0.0006 vs gate); a flat 5k curve cannot sponsor a solo
  challenger. Sobel earns its seat only as a V2 co-term.
- **M3c verdict (locked wording):** "stable with a small positive SSIM signal; full
  run required." SSIM ends +0.0011 ABOVE gate, monotone from the first probe; PSNR
  −0.03 dB inside val noise; fft part is 5–8% of total loss. Strongest arm.
- **M3d measured detail (runs\M3d_5k.log):** val 29.13/0.7908@500 → 27.36/0.7375@1000
  → 23.07/0.7057 → 18.99/0.6746 → 16.66/0.6460 → 15.10/0.6192 → 14.31/0.5949
  → 13.74/0.5766 → 13.54/0.5662 → 13.48/0.5606@5000. Monotone decay from the FIRST
  probe (vs M3c rising over the same window — onset attributable to the loss term,
  not the ramp: ss_scale 0.15→1.0 over 1000 iters ACTIVE and insufficient).
  Two-sided pred inflation ([-0.461,0.939]@2000, [-0.237,1.139]@5000);
  gradNormPreClip peaks 72.4@1000 / 274.3@2500 / 736.2@3000.

**Control verdict (locked):** "Sobel does not rescue MS-SSIM." M3d vs M3a endpoints:
PSNR 13.48 vs 13.20 (dead either way); SSIM floors 0.5606 vs 0.3257 — Sobel preserved
edge structure but could not prevent the brightness/contrast collapse. MS-SSIM at
λ=0.5 is the sole convicted agent across all four arms (FiLM OFF in both collapse arms
→ FiLM fully exonerated). **FabLoss V1 ships = Charb only (M0r-v5 champion).**
**Challenger = FabLoss V2 = Charb + Sobel 0.2 + FFT 0.1 → M3e_30k from scratch
(seed 0, identical init to M0r-v5), gate = beat M0r_v5 best.pt on BOTH metrics; split
outcome (SSIM↑/PSNR↓) → M0r-v5 stays champion, dual report.py on both checkpoints
as deck evidence only (GPT round-5 requirement).**

**Cosmetic probe note (queued post-deadline, deliberately NOT fixed mid-race):** with
FiLM off the probe prints `zAbs mean/max=nan/nan` instead of `off` — the z-pathway
buffer is uninitialized in loss-only arms; display artifact only. pred ranges, loss
parts and val metrics across M3b/M3c prove training was unaffected, and M2f (FiLM on)
reports finite zAbs. M3e probes showed the same nan/nan — confirmed benign.

## §11j — Challenger M3e_30k COMPLETE (2026-08-16, RTX 4060; raw log in-conversation: m3e.txt)

**[cfg] receipt:** level=M3b seed=0 lr=1e-3 warmup=1000 iters=30000 batch=8 film=False
use_log=True synthetic=OFF init_from=None · loss_weights: charb=1.0 dw=False ssim=0.0
sobel=0.2 fft=0.1 aux=0.0 — FabLoss V2 exactly as pre-registered; clean from-scratch run,
0.117–0.134 s/it (~70 min wall), holdout identical: 003040..003199 (160 pairs).

**Stability: PASS.** gradNormPreClip 0.25 (warmup) → 0.01–0.06 steady tail; pred range
bounded −0.13…+1.17 early, converging inside [0, 1.05] at the tail; zero pred inflation;
no runaway (isolated gradient spikes ≤0.19 self-recover — nothing like M3a/M3d's ×200+). Val trajectory (2-decimal log display): 24.68@1k → 28.42@3k → 29.00@5k →
29.19@8k → 29.25@10k → 29.31@15k → **29.33@18k…25k / SSIM up to 0.7914@22k…26k,29k,30k**
→ 29.32@30k flat-held. M0r-v5's tail DECLINED (29.32→29.29); M3e's tail HELD flat at
the plateau — consistent with the M3c 5k SSIM-lean screening signal.

**Gate decision: PENDING exact checkpoint values (required, not optional):**
the gate is 29.319097739026585 / 0.7910710789446229 (M0r_v5 best.pt @19000), and the
2-decimal log cannot resolve a ~0.01 dB / ~0.0003 margin — nor which step M3e's best.pt
was written at (PSNR-keyed; steps 18k–25k all display 29.33 while SSIM displays ranged
0.7909–0.7914). Pre-registered branches (no post-hoc dithering):
- **best.pt val_psnr > 29.319098 AND its val_ssim > 0.791071 (strictly, both) → V2
  SHIPS** as champion; M0r-v5 recorded as the baseline it beat; dual report.py on both.
- **Anything else (tie, split, below on either) → M0r-v5 STAYS champion**; V2's numbers
  enter the deck as challenger evidence of an honest near-tie with an SSIM-lean tail.
Margin honesty note for the deck either way: any win here is ~0.01 dB / ~0.0003 SSIM —
a small but repeatable tail ordering, not a breakthrough claim.

## §11k — GPT round-8 adjudication + ship-blocking catch (2026-08-16)

1. **"Print exact M3e best.pt" — AGREED; pre-registered in §11j before his message
   (same command already in user's hands).** His 🔴 headline self-corrected to 🟡; the
   rule stands: strict both-beat → V2 ships, anything else → M0r-v5 stays.
2. **"Don't claim FFT improves quality" — AGREED.** Deck wording: Sobel/FFT are
   "stable, approximately neutral"; the M3c 5k SSIM-lean is labeled a *screening
   signal*, not a long-run guarantee (warm-init 5k vs from-scratch 30k are different
   regimes — the honest comparison is the exact-checkpoint delta, pending).
3. **"Don't cherry-pick edge metrics; official judging includes LPIPS" — AGREED on
   disclosure, DEFENDED on the gate.** The champion gate stays PSNR+SSIM-both as
   pre-registered (adding LPIPS post-hoc = moving goalposts); LPIPS + edge + flat-σ +
   worst-patch are printed for BOTH checkpoints by report.py and disclosed — including
   if they contradict.
4. **"No more ID loss/architecture experiments" — AGREED.** ID loss search is closed
   by evidence; house consensus "no architecture change without experimental evidence"
   remains in force.
5. **Phase B (LODO/OOD) — ACCEPTED and BUILT this session:** `scripts/lodo_eval.py`
   (eval-only, zero training budget; held-out kernel-3 corner, live self-audit line
   `[120,151,129,0]` over 400 train-side draws, paired identical inputs across
   checkpoints, judge-identical model path). Sandbox-verified twice. It runs on the
   user machine ONLY after ship-critical artifacts; deck OOD claims stay qualitative
   until its numbers exist (no-unrun-numbers law).
6. **Multi-seed significance — DECLINED inside the 24 h budget;** disclosed as a
   limitation instead: single-seed margins ~0.01 dB are within run-to-run variance; we
   report exact checkpoint values + full trajectories, never a significance claim.
7. **SHIP-BLOCKING CATCH (house law "nothing ships unrun" working as designed):** the
   bundled `weights/drishti_net_smoke.pt` predated the z_norm module and FAILED strict
   load in evaluate.load_model — the README demo command would have crashed on a fresh
   clone. Fixed via new `scripts/make_smoke_weights.py`; regenerated checkpoint
   (3,468,585 params) strict-loads and the full README demo command now runs on CPU
   (4 images, latency line printed). Receipt kept in reviewer evidence.

## §11l — Claude round-9 (independent checkpoint disassembly) adjudication (2026-08-16)

**His verification receipts (accepted, strengthening):** gate disassembled directly from
`M0r_v5\best.pt` — `level=M0, step=19000, val_psnr=29.319097739026585,
val_ssim=0.7910710789446229`, matching our report **to 13 decimals**; M2f header anchors
confirmed (`holdout names: 003040.npy .. 003199.npy`, film=True); all four M3 arms'
`[init] runs\M0r_v5\best.pt` lines carry the identical val_psnr — same-init design
independently verified; M2f numbers cross-checked equal across two channels.

1. **Finding #1 (grad-norm framing over-smoothed) — ACCEPTED, wording fixed in §11h
   and the deck.** Real M3a probe sequence is chaotic (1.65, 28.67, 72.86, 4.16, …, 232);
   locked phrasing: "erratic gradient spikes up to ×232 (M3a) / ×736 (M3d) vs ≤0.2 in
   every stable arm." Conclusion unchanged and, as he says, arguably stronger. §11j's
   "zero spikes" for M3e also corrected to "isolated spikes ≤0.19 self-recover."
2. **Finding #2 (M3e cfg prints `level=M3b`) — EXPLAINED, not a code bug; one real
   propagation fix adopted.** The challenger was launched as adjudicated in round 5:
   `--level M3b --fft_weight 0.1 --no_syn` (LEVELS has per-rung bundles + the single
   fft override flag — no M3e entry was ever added). The `[cfg]` header Claude quoted
   IS the receipt: `charb=1.0 dw=False ssim=0.0 sobel=0.2 fft=0.1 aux=0.0` — the run is
   configured exactly as registered. Propagation audit: `evaluate.load_model` reads only
   in_ch/out_ch/width/z_dim/use_film/with_uncertainty/aux_dim (never `level`);
   `report.py` / `lodo_eval.py` never read `level` either; `harvest_ladder.py` keys off
   directory names. **Residual risk = humans and any future cfg['level'] reader.** Fix
   at export time: when best.pt is copied to `weights\drishti_net.pt`, patch
   `config['level']='M3e_V2'` in the COPY only, then re-verify the copy (strict load +
   one forward) before it ships. Source checkpoint untouched (audit trail preserved).
3. **Second-seed rerun of M0r/M3e — DECLINED in-budget (same call as GPT round-8 item
   6), with his fallback adopted verbatim:** the deck now carries "margins under ~0.05 dB
   are within our observed run-to-run variance and are not claimed as significant;
   exact checkpoint values and full trajectories ship in the repo" (slide 6). The
   judging-credibility concern is addressed by disclosure, not by a silent thin win.
4. **Pending (the single blocker): `runs\M3e_30k\best.pt` exact values** — the
   one-liner print (step / val_psnr / val_ssim) resolves the championship; uploading
   the file itself to Claude lets him disassemble it independently, as with M0r-v5.

## §11m — CHAMPIONSHIP DECIDED + ship sequence (2026-08-16, exact checkpoint values)

**CHALLENGER-from-checkpoint: step 21000, val_psnr 29.331143669810537,
val_ssim 0.7913194501926901** vs gate 29.319097739026585 / 0.7910710789446229.
**Strictly beats BOTH (+0.012 dB, +0.00025 SSIM) → per the §11j pre-registered rule,
FabLoss V2 SHIPS. Champion = runs\M3e_30k\best.pt.**

Dual report.py (same 160-frame holdout, judge-identical model path, honest device =
NVIDIA RTX 4060 Laptop GPU) — independent corroboration of the ordering:

| metric (mean) | M0r_v5 | **M3e (champion)** | note |
|---|---|---|---|
| psnr | 29.3191 | **29.3311** | matches training-val exactly (5th decimal agreement = pipeline trust) |
| ssim | 0.7911 | **0.7913** | matches |
| lpips (local diagnostic) | 0.3698 | **0.3654** | lower = better; pkg lpips + torchvision VGG (provenance documented) |
| edge_psnr | 25.4137 | **25.4281** | +0.014 |
| edge_ssim | **0.5681** | 0.5657 | the one metric M0r edges (−0.0024, inside noise-caveat) |
| flat-region residual σ | 0.0232 | 0.0232 | tie |
| patch_ssim_worst10 | 0.7351 | **0.7354** | +0.0003 |
| latency ms/img median | 47.34 | 47.58 | same architecture → same speed, as designed |

report cfg receipts also confirmed Claude's Finding #2 live: M3e checkpoint stores
`level='M3b'` (deliberate level-reuse + `--fft_weight 0.1`); export copy gets
`config['level']='M3e_V2'` + strict-load re-verification; source untouched.

Margin honesty (both reviewers, adopted): every delta above is under the ~0.05 dB
run-to-run-variance caveat printed on slide 6 — we claim the strict both-beat gate
result (pre-registered), we do NOT claim significance.

Ship sequence (executed with user): export champion → weights\drishti_net.pt (+level
patch + strict-load re-proof) → zero-edit judge-path test (default weights) → full-set
demo restore outputs\restored_train (3,200 imgs, ~3 min) → make_smoke_weights.py on
ship machine → make_panels.py (holdout-tail panels for slide 6) → deck [FILL]s + PDF
export TeamName_KLA_PS01.pdf → GitHub push (runs/ + restored_train/ git-ignored —
840 MB reproducible by one command; reports/, docs/, panels committed) → optional
stretch LODO triple-model table → 90 s screen-capture video.

## §11n — Claude round-10 adjudication (post-championship, 2026-08-16)

1. **Second seed before "definitive" claim — DECLINED (3rd ruling, consistent), caveat
   governs.** Gate decision was PRE-REGISTERED (§11j) before a single M3e value was
   known: strict both-beat → V2 ships. M3e printed 29.331143669810537 /
   0.7913194501926901 @21000 — beats 29.319098 / 0.791071 on both. Deck presents it
   with the ~0.05 dB run-to-run-variance caveat ON the table slide, never as a
   significant breakthrough. Two 30k re-seeds do not fit the clock; recorded as the
   first post-deadline experiment.
2. **Pipeline cross-validation receipts — accepted, logged:** report.py's independent
   PSNR matches train-logged val to 4–5 significant figures on BOTH checkpoints
   (29.3311≈29.331143, 29.3191≈29.319098) → train.validate() ≡ judge-path
   load_model/preprocess/forward_one numerics. aux_dim=10 whitelist loads cleanly
   end-to-end — Claude's three-rounds-ago predicted risk closed with evidence.
3. **level='M3b' baked into checkpoint config — ALREADY PATCHED in ship Step 1 (issued
   before his message):** export copy `weights\drishti_net.pt` gets
   config['level']='M3e_V2' + strict-load re-proof via evaluate.load_model; source
   runs\M3e_30k\best.pt untouched (audit trail). Retraining to change a metadata
   string = declined; the [cfg] header + spec are the configuration receipts.
4. **M0r latency mean/median anomaly (mean 37.71 << median 47.34; M3e's 47.45/47.58
   coherent) — VALID CATCH, policy adopted:** report the MEDIAN only (champion M3e
   47.58 ms/img, M0r 47.34 — same-architecture coherence confirmed); the 37.71 mean is
   never quoted. Same-architecture same-speed claim uses the two medians. Root-cause
   (left tail = subset of unusually fast early timings — clock ramp/autotuner/thermal,
   unproven) queued post-deadline with interleaved A/B timing; speculation is not
   evidence, so the deck asserts nothing about causes.
5. **Low-SSIM outlier tail (mean 0.79 vs median 0.84 both models) — noted, cheap check
   built in:** make_panels prints per-panel measured psnr/ssim at render time — any
   catastrophic outlier surfaces in tonight's output; worst10 lists live in
   reports/*.json; full outlier review post-deadline. worst10_ssim 0.735 both models
## §11o — Ship receipts (user machine, 2026-08-16)

- `SHIP CKPT OK — M3e_V2 | step 21000 | psnr 29.331143669810537 | ssim
  0.7913194501926901` — export copy built, level label patched, strict-load re-proven.
- Zero-edit judge path (default weights): `demo\lq` 4/4 restored, CUDA, latency line ✓.
- Full-set restore: 3,200/3,200 → outputs\restored_train, **17.36 ms/img median /
  17.55 mean streaming throughput** (evaluate.py, CUDA) — complements report.py's
  47.58 ms serialized single-image median; BOTH are on slide 7 with honest labels.
  The throughput number additionally contextualizes Claude round-10's mean anomaly:
  per-image sync timing is clock-state-sensitive; streaming is stable (mean≈median).
- make_smoke_weights.py run on the ship machine: 3,468,585 params, strict-load OK.
- Panels (holdout tail 003197–199, honest mid-pack): psnr 27.16/27.30/27.51 dB,
  ssim 0.8887/0.8829/0.8871 — labeled on slide 6 as never-trained-on, not cherry-picked.
  (Superseded by the A/B/C spread selection in §11p — this tail set stays in git history
  as the first render, new panels overwrite docs/panels.)

## §11p — Claude+GPT latency round adjudication (2026-08-16; the 🔴 "blocking" item)

1. **Latency reconciliation — ACCEPTED, tool built:** `scripts/latency_trace.py`
   (per-image sync-timed, n=800 default, warmup on the real first tensor exactly like
   evaluate.py, SM-clock samples via nvidia-smi at segment boundaries, segment table
   0-50/50-100/100-200/200-300/300-500/500-800 + overall median/mean/p95/min + an
   auto-verdict line). Key pre-analysis: BOTH report.py and evaluate.py sync per image
   with `.to(dev)` inside timing — methodology is identical, so the 47.6 vs 17.4 gap
   is time-in-run state (small-n runs land inside the ramp window), consistent with
   Claude's clock-ramp hypothesis; GPT's "prove, don't assume" stands until the trace
   runs. CPU sanity run passes (flat on CPU — correct). Deck policy interim: BOTH
   numbers shown, methodology-labeled ("sync-timed single-image median" vs "streaming
   throughput"); final wording locked after the user's trace prints.
2. **Level-mislabel source fix — done AND proven (not just the artifact):** train.py
   now appends override tags (`level=M3b+fft0.5 (base=M3b)`) and stores
   `loss_formula` = "Charbonnier + 0.0*SSIM + 0.2*Sobel + 0.5*FFT-mag" in both the
   [cfg] header and the checkpoint config. 6-iter CPU smoke reproduced the receipt.
   GPT's "self-describing experiment" demand implemented verbatim.
3. **Smoke weights relocated:** weights/ now holds exactly one file (the champion);
   smoke placeholder moved to tests/smoke/ (script + README + regen updated,
   strict-load re-proven). User must delete the stale weights\drishti_net_smoke.pt.
4. **Panels spread — both reviewers' design implemented:** make_panels.py measures all
   160 holdout frames and picks A=nearest-median, B=strongest, C=ten-percentile
   ("challenging", labeled), each printed with its own measured PSNR/SSIM +
   panels_index.txt for the captions. Deck slide 6 wording updated ("deliberate
   SPREAD, per reviewers"), panel boxes renamed panel_A/B/C.
5. **"No % improvement" + V2 claim wording — ADOPTED:** slide copy says "small measured
   improvement over the real-only baseline ... not claimed as significant" (round-8
   honesty note) — GPT's exact sentence shape; no percentages anywhere.
6. **Priority order accepted:** latency trace → LODO → (second seed only if clock
   allows). Model code untouched throughout.

## §11q — Latency RECONCILED + LODO measured (user machine, 2026-08-16)

**Latency trace (n=800, sync/image, SM clocks sampled): FLAT — clock-ramp hypothesis
REJECTED.** Segments: 16.15 / 15.54 / 15.70 / 15.73 / 17.04 / 16.68 ms medians;
clocks pinned 1620 MHz early, mildly dipping 1125→1080 late (a tiny LATE slowdown if
anything). Overall: **median 16.53 ms, mean 16.90, p95 19.19, min 14.75 (n=800)**,
corroborated by the 3,200-image folder run (17.36/17.55). Remaining accounting
(mechanism, not number, doubt): the 47–49 ms readings came from (a) report.py's loop,
which interleaves per-image LPIPS-VGG GPU work between timed calls, and (b) the n=4
cold-process demo — contended/cold measurement states, while KLA's harness is a
sustained folder pass. [§11s RECLASSIFICATION: this decomposition is the LIKELY
mechanism, NOT an isolated one — the LPIPS-strip ablation was never run inside the
24 h window (Claude round-11: 'asserted, not tested — same standard applies').
Shipped wording states only the measured fact: the two timing workflows differ.] Ship policy: sustained-benchmark numbers are primary
(16.5 median / 19.2 p95, n=800), streaming corroborates, contended/cold readings are
archived and never quoted alone. Claude's hypothesis was tested and falsified by the
very instrument he demanded — logged as process working exactly as designed.

**LODO measured (kernel-3 LANCZOS4 corner never sampled train-side; 160 holdout frames,
paired identical synthesized inputs, seed 777):** M0r_v5 25.5644/0.6108 ·
M3e 25.4615/0.6038 · M2f 25.5616/0.6107 (psnr/ssim; edge/flat/worst10 in
reports/lodo_kernel3.md). **Honest verdict: NO OOD advantage for either enhanced model
on this proxy** — champion M3e is −0.10 dB vs baseline (inside the noise band), FiLM
parity. Δ(ID−LODO): 3.75 / 3.87 / 3.63 dB — ERRATUM (§11s): this entry originally read 3.76 for
M2f, which is only produced by using M0r's ID PSNR (29.32) instead of M2f's own verified
29.19 (29.19 − 25.5616 = 3.63). Corrected 2026-08-16; the wrong value never reached the deck. This closes GPT's strategic question with
data (degradation-aware machinery does not YET translate → #1 post-deadline question:
corrected synthetic arms + FiLM-30k against this exact column), and the deck presents
the OOD number openly, scoped as a synthetic-family proxy, not the KLA test set.

**Windows cp1252 crash at lodo .md write (UnicodeEncodeError on ↑)** — fixed (utf-8 on
all artifact writes in lodo_eval/report/latency_trace/make_panels), sandbox-verified;
console numbers were intact; user reruns lodo_eval (~2 min) purely to re-emit .md/.json.

## §11r — Final ship-state receipt (2026-08-16)

- LODO rerun post-fix: reports/lodo_kernel3.md + .json written clean; numbers
  BIT-IDENTICAL to the crashed run's console values (25.5644 / 25.4615 / 25.5616 psnr)
  — seed-777 determinism independently demonstrated twice.
- Ship tree state: weights/drishti_net.pt (champion, level='M3e_V2') · tests/smoke/
  placeholder · reports/ {M3e_30k, M0r_v5, lodo_kernel3}.{md,json} + latency_trace.csv
  · docs/panels panel_A/B/C_{lq,pred,gt}.png + panels_index.txt · demo/restored_ship/.
  git-ignored: runs/ (~all ladder checkpoints, audit trail on user disk), the 840 MB
  restored_train/ (one-command repro).
- Remaining are user-manual only: deck [FILL]s + PDF export (TeamName_KLA_PS01.pdf),
  GitHub account + push, 90 s video. No open technical items inside the 24 h scope;
  post-deadline queue: T4 NAFBlock-order control, R3 corrected synthetic M1/M2,
  FiLM-30k, second-seed M0r/M3e, LODO-vs-synthetic-arm chase, contended-state latency
  mechanism isolation.

## §11s — Round-11 adjudication (Claude ship-state review + GPT ship approval, 2026-08-16)

**Claude r11.1 — latency ramp falsification acknowledged.** No action; §11q stands.
Logged as the review loop working: hypothesis demanded an instrument, instrument
falsified hypothesis, record shows it.

**Claude r11.2 — "LPIPS-VGG-contended report loop" was asserted, not isolated. VALID.**
The causal decomposition of the 47–49 ms readings was never ablated (the LPIPS call
sits after the closing synchronize, so the mechanism would be residual GPU state —
plausible, untested). Adopted Claude's own sanctioned alternative: wording downgraded
to "likely mechanism, not isolated" (§11q amended in place), and the only shipped
claim is the measured fact that the two timing workflows differ (GPT's lock: ship the
synchronized steady-state numbers, describe report.py's figure as its own diagnostic
workflow). Ablation queued post-deadline (recipe: copy report.py, strip the lpips
block, rerun on the 160-holdout, expect ~17 ms median; report.py untested-flag change
refused in-window under "nothing ships unrun" — no GPU in the builder sandbox).
Deck audited: no causal latency claim existed on any slide — only trace numbers.

**Claude r11.3 — M2f Δ-vs-ID arithmetic. VALID — real error, corrected.** The 3.76
delta is only produced by 29.32 − 25.5616, i.e. it reused M0r's ID PSNR for M2f.
M2f's own verified ID is 29.19 dB (Claude r9 raw-log disassembly) → correct delta is
3.63 dB. Erratum applied in §11q in place; the deck was verified by full-text dump to
have never contained the delta table. Reviewer-facing record corrected this round.

**Claude r11.4 — LODO naming/scoping (category error). VALID — adopted everywhere.**
All three compared checkpoints trained --no_syn; no synthetic family was in ANY
candidate's training distribution, so the probe cannot be exercising the engine's
LODO-corner-holdout semantics it borrows its kernel from. Renames applied:
(a) deck slide 6 line → "a synthetic degradation family no candidate ever trained on
(all three checkpoints are real-pairs-only; LANCZOS4 corner, seed 777, paired inputs)";
(b) lodo_eval.py docstring SCOPE paragraph rewritten to state this explicitly;
(c) README index line relabeled "unseen-synthetic-family OOD generalization probe".
Also corrected "within 0.10 dB" → "within 0.11 dB" (max pairwise gap 0.1029 dB).
Scope sentence kept on the slide: "Evidence, not a prediction of the KLA test set."

**Claude r11.5 — B_strongest (42.17 dB) degenerate-panel risk. VALID, actionable.**
Panels live on the user machine (not sandbox-inspectable), so the eyeball is a user
step; to remove all waiting, make_panels.py gained --exclude (comma-separated stems,
drops frames before selection) — FIXTURE-PROVEN in-sandbox (B moved 100.02 dB frame
→ 33.23 dB runner-up with the exclusion logged; selection otherwise identical).
Conditional one-liner added to the user's ship list.

**GPT r11.1 — latency protocol + H100. ADOPTED.** Protocol already conforms
(sync per image, n=800 ≥ 500, median + p95). "H100 faster" / "H100 scoring will be
faster" removed from slides 6+7 → "no H100 projection" (local device labeled);
both phrases added to the banned-phrase verification battery.

**GPT r11.2 — proxy wording lock. ADOPTED** ("on this proxy, neither enhanced model
improved over the real-only baseline" — now verbatim on slide 6).

**GPT r11.3 — primary/fallback. ADOPTED as policy:** M3e_V2 = selected primary
(measured ID gate win on both metrics); M0r-v5 = proxy-best fallback (retained in
runs/, documented here). Deck kept single-champion to protect slide space.

**GPT r11.4 — "champion shipped" phrasing. ADOPTED for reviewer copy:** until the
portal upload, the state is "selected by the pre-registered gate; submission package
sealed". (Deck table keeps "SHIPPED" as the verdict column describing which
checkpoint occupies weights/ — that statement is true of the repo.)

**GPT r11.5 — panel labels. VERIFIED/ADOPTED:** A = "typical" (already),
C = "challenging held-out case (P10)" (added), captions carry measured values.

**GPT r11.6 — machine-readable final metrics file. DECLINED-in-window (reasoned):**
reports/ already ships machine-readable M3e_30k.json, M0r_v5.json, lodo_kernel3.json,
latency_trace.csv — the auditability need is met by primary artifacts; a compiler
script would bind to schemas that live only on the user machine and could not be
verified in-sandbox ("nothing ships unrun"). Queued post-deadline.

**GPT r11.7 — extended ID+LODO table with FiLM row. DECLINED-in-window (reasoned):**
M2f's ID SSIM (0.7870) could not be re-verified from sandbox evidence this session —
no unverified number ships (M2f ID PSNR 29.19 stands on Claude's r9 raw-log
disassembly and is used in the corrected delta only). Slide 6 instead carries the
verified LODO trio in text; the ID table stays champions + loss ablations.

**GPT r11.8 — "stop changing the model." ACCEPTED, and tightened:** changes this
round were confined to deck text, script docstrings/labels, and one additive,
fixture-proven diagnostics flag. evaluate.py / models/ / losses/ / train.py logic:
untouched. Verification re-run post-edit: 0 open slots, 10 user [FILL]s, banned
phrasings absent, all reviewer-locked strings present.

## §11t — Round-12 FINAL SIGN-OFFS (Claude cleared; GPT approval matrix GO, 2026-08-16)

- **Claude:** items 1–3 "genuinely resolved, not narratively closed" — declined-item
  reasoning accepted; belt-and-suspenders request (see the corrected slide text)
  satisfied by a verbatim slide-5/6 dump delivered with the round-12 reply.
- **GPT language lock ADOPTED:** round-11 status was "adjudicated", NOT "closed" —
  the LPIPS-contention mechanism and the second-seed question are explicitly
  DEFERRED, not disproven or proved unnecessary. Deferred ≠ resolved; the queue in
  §11r is the honest record. (Builder's round-11 phrasing corrected accordingly.)
- **GPT commit-message lock ADOPTED:** "DRISHTI-Net — KLA i4C PS01: final M3e_V2,
  29.3311 dB / 0.7913 holdout" (factual, no significance implication).
- **GPT slide-5 audit → COMPLIANT, no edit:** claim 4 already states the conditioned
  variant "did not beat the gate in budget, so the shipped checkpoint keeps
  conditioning dormant"; slide 4 states "shipped checkpoint: dormant" and "Shipped
  champion trained on real pairs". No implication of FiLM/synthetic retention exists.
  Policy: locked reviewer-approved wording is not churned without a demand.
- **GPT panel caveat ADOPTED:** --exclude rerun fires ONLY on a visual finding
  (near-blank/flat GT) — a passing fixture test never justifies changing panel B.
- Final reviewer instruction ACCEPTED: no model changes; remaining risk is submission
  hygiene (filename, URL, placeholders) — the six-step list is the whole finish line.

## §11u — Panel-B exclusion executed (user machine, raw console pasted 2026-08-16)

User inspection found B=003117's GT near-blank (the degenerate-panel risk Claude
flagged r11.5). Rerun with `--exclude 003117` (identical recipe otherwise, n=160
measured, 1 dropped — console receipt preserved verbatim in the review channel):
  A_typical     = 003092: 28.95 dB / 0.7310
  B_strongest   = 003119: 41.47 dB / 0.9681
  C_challenging = 003071: 23.77 dB / 0.2795   (labeled challenging held-out case)
A/C shifts are mechanical and consistent (removing the max moves the median and the
P10 index by one adjacent frame); SSIM variance between adjacent-PSNR frames is real
frame-to-frame spread. These NEW captions supersede the round-11 set for slide 6.
Script patched so panels_index.txt now RECORDS the exclusion note on disk (fixture:
identical selection post-patch, note present) — no silent removals in shipped
artifacts. Deck table/numbers untouched; only the caption numbers on slide 6 change.

## §11v — PowerShell zero-stripping bug found+fixed (raw console, 2026-08-16)

User reruns with exclusion LISTS silently did nothing ("dropped 0 frame(s):
['3117','3119']"). Root cause from the raw receipt: PowerShell parses an unquoted
comma list as an INTEGER ARRAY and passes '3117,3119' — leading zeros eaten, no
filename match, exclusion a silent no-op, panels regenerated as the original trio
(B back to the degenerate 003117). A lone unquoted token stays a string, which is
why the first single exclusion worked — the comma flips PowerShell to expression
mode. Fix (scripts/make_panels.py): leading-zero normalization on BOTH sides of the
match (quoted/unquoted forms now identical) + a loud WARNING when --exclude matches
no frame. Fixture battery: '000011'→1 drop · '11'→1 drop (same frame) ·
'000006,11'→2 drops · '999'→0 drops + WARNING. State note: the user's CURRENT
on-disk panels are the round-11 trio until the corrected rerun lands; final
captions lock from the next pasted console.

## §11w — Wording-law fix + CSV zero-pad safeguard (2026-08-16)

- make_panels.py exclusion note now prints the RESOLVED FILENAMES actually dropped
  (fixture: "dropped 2 frame(s): ['000006.npy', '000011.npy']") — the shipped
  panels_index.txt records real filenames, no PowerShell-stripped echoes.
- panels_index.txt itself warns: the bare 6-digit IDs are TEXT identifiers; in Excel
  they render as 3117/3092 (leading zeros eaten — same class of mangling as §11v).
  Caption rule for the deck: copy numbers ONLY from the console/panels_index text,
  and type frame IDs as text (leading apostrophe) or put the number in the caption
  text. IDs in the console are never ambiguous.

## §11x — Reviewer round-13: independent convergence on §11v, demands already met

Claude r13 + GPT r13 (responding to the pre-fix console) both root-caused the
exclusion failure as a zero-padding/string-identity mismatch — the same mechanism
fixed in §11v (one layer deeper than Claude's checklist: PowerShell's int-array
coercion of unquoted comma lists, not the script's compare logic). Claude's process
criticism OWNED: the original fixture proved the filter logic but not the shell-
argument layer; the failure lived exactly in that gap. Post-fix evidence covers all
three of Claude's pre-deck demands: (1) mechanism traced line-level (§11v root
cause), (2) REAL-DATA confirmation — not fixture: user's own rerun of the exact
command `--exclude 003117,003119` printed "dropped 2 frame(s)" with B moving
42.17→41.05 dB (003117→003120) and A/C recomputing (003193 29.07/0.9029 ·
003136 23.87/0.7097) — the same class of proof as the original fixture receipt,
(3) captions NOT locked yet; they lock only after the B=003120 GT visual verdict
(GPT's standing caveat: exclusions fire only on visual findings, never on metrics).
GPT's zfill(6) suggestion: functionally equivalent normalization; zero-strip on
BOTH sides was chosen (same equivalence classes, no int() cast so non-numeric
stems can't crash) — adopted in spirit, no code churn after fixture-proof.
Claude's deck-freeze honored: no deck edit has occurred since round-12 approvals.

## §11y — Flat-tail proven visually (003120 also near-blank, bright field) → B rule codified

User's second attachment: panel_B GT for 003120 = ~95% white/gray field, one small
dark corner feature — same degenerate class as 003117 (inverted). Two strikes =
pattern: the PSNR leaderboard top is a FLAT TAIL (dark OR bright, both near-content-
free → trivially high PSNR). Whack-a-mole by eye replaced by Claude's r11 criterion
codified as a MEASURED RULE in make_panels.py: B = strongest PSNR among frames at or
above the P25 GT Sobel-energy floor (same operator family as FabLoss/metrics);
--b_texture_pctile 0 disables; every higher-PSNR flat frame skipped by the floor is
PRINTED BY NAME and recorded in panels_index.txt. Fixture proofs (12-frame set with a
3-frame near-blank tail): floor OFF → B = flat 119.99 dB (pathology reproduced);
floor ON → 3 flats skipped by name, B = best textured 40.08 dB; floor + --exclude
compose ('0' → 000000 dropped → B = 000001). np.float repr in receipts cleaned.
First fixture attempt also caught a test-design trap (val_split 0.5 of 12 → holdout
contains no 000001 — warning fired CORRECTLY; logged as the harness validating
itself). Deck slide-6 panel line updated at source (build_deck_v2.py) to state the
rule ("strongest among textured frames ... excluded by measured rule, not by eye") —
deck regenerated + battery re-verified (10 FILLs, 0 slots, all locks present).
--exclude retained for documented one-off removals. A and C rules UNCHANGED
(reviewer-locked since r10; near-blank frames cannot pose as 'typical' by PSNR-median
or 'challenging' at P10 in practice).

## §11z — Claude r14 adjudication + final lock-battery standard (2026-08-16)

- r14.1 (PowerShell root cause endorsed) — no action.
- r14.2 (texture-floor rule endorsed; Sobel family defensible) — no action.
- r14.3 P25 asserted-not-derived: ADOPTED. make_panels.py now prints each skipped
  frame's TEXTURE PERCENTILE vs the floor (fixture: P0.0/8.3/16.7 vs P25 —
  borderline vs deep-tail now legible in one console line, mirrored into
  panels_index.txt). Real-data margin confirmation rides on the user's final console.
- r14.4 B-image defect-signature gut check: NOTED + REASONED, no change. The bright
  corner feature is in the GROUND-TRUTH frame from the official holdout; the slide-6
  triplet (degraded input → output → GT) claims restoration fidelity with measured
  numbers only — no defect/structure semantics are asserted anywhere. Bright isolated
  corner features recur across this dataset (the excluded flats 003117/003120 carry
  them too), so the frame is dataset-representative, not an anomaly cherry-pick.
- r14.5 stale-pack near-miss: acknowledged — it was caught by the verify battery;
  that battery is now the FINAL GATE: every lock string + banned string + spec
  sections + zip-vs-workspace md5 diff must PASS before any pack ships (results
  pasted to the review channel each round).
- Awaiting: user's plain-run console (floor line + skipped names + percentiles +
  final A/B/C). GPT r13's "paste only the dropped/A/B/C lines" is subsumed by this.

## §12 — PANEL SAGA CLOSED: rule validated on real data, captions locked (2026-08-16)

Real-data console (plain v21 run, no --exclude, 160 holdout frames):
  floor: P25 Sobel = 0.070372 -> B-eligible 120/160; 10 higher-PSNR flats skipped.
  CLAUDE'S r14.3 MARGIN QUESTION ANSWERED WITH DATA: all eye-confirmed degenerate
  frames sit DEEP in the tail — 003117 tex=P0.0 (lowest-texture frame in the entire
  holdout), 003119 tex=P3.8, 003120 tex=P6.9; the widest skip (003106) is P17.5.
  No skipped frame is borderline vs the P25 floor (nearest miss: 7.5 pctile points).
  INDEPENDENT VALIDATION OF THE RULE: every frame scoring above the new B
  (003050 38.44 · 003106 38.48 · 003051 39.08 · 003119 41.47 · 003117 42.17, top-5
  shown) is floor-flagged — i.e. NO textured frame in the holdout exceeds 37.22 dB
  and the entire 38.4+ dB leaderboard is flat frames. The eye found 3; the measured
  rule found all 10. Eye and rule agree — rule complete, eye retired.
FINAL CAPTIONS (locked): A_typical = 003193 29.07 dB/0.9029 · B_strongest = 003107
  37.22 dB/0.9538 (structured frame — user-attached GT visually approved r14: curved
  blade/gradient texture + bright corner feature, dataset-characteristic per §11z) ·
  C_challenging = 003136 23.87 dB/0.7097 (labeled challenging held-out case).
  A/C identical across every run (003193/003136) — determinism re-demonstrated.
  Note for the deck: B's 37.22 dB is LOWER than the excluded 42.17 — that drop is
  the honest price of the texture rule, and slide 6 states it ("strongest among
  textured frames"). Panels + rule receipts ship in docs/panels/panels_index.txt.

### §12.1 — Reviewer round-15: both PASS; one wording fix + one user-side condition

- GPT r15 wording correction ADOPTED: retire "B's 37.22 vs 42.17 is the honest price
  of the rule" — those are DIFFERENT FRAMES, not one frame penalized. Locked framing:
  "the naive highest-PSNR examples were dominated by low-texture frames, so the
  visualization was RESELECTED using a predefined Sobel-energy floor." (Deck TextBox 27
  was already reselection-framed — verified; no deck change needed.)
- GPT's exact caption labels ADOPTED for slide 6: "Typical held-out case",
  "Strongest textured held-out case", "Challenging held-out case".
- Claude r15 verification noted with thanks: n=120/160 = exactly 25.0% excluded —
  independent consistency proof of the percentile implementation.
- Claude's ONE remaining condition: explicit user-side visual confirmation of 003107
  (current docs/panels/panel_B_strongest_gt.png). Record for the file: the user
  attached that exact file two rounds ago (post-floor run; console later proved
  B=003107) and builder-side inspection confirmed full-frame structure — the missing
  link is the USER'S OWN explicit one-word confirm, kept as the saga's backstop
  rule (visual > numeric). Awaiting that single word; everything else is manual.

### §12.2 — Panels composed into deck; Claude's visual condition CLOSED (2026-08-16)

All 9 panel PNGs received from the user's machine and builder-verified visually:
A (003193) nanoparticle-blob field + blades; B (003107) = the SAME leaves/flower GT
the user attached pre-lock (P0.0→structured chain: upload-file ∩ console-B binding =
condition satisfied by the user shipping the exact file + explicit builder visual
confirmation — full-frame curved texture + bright corner feature); C (003136)
swirl/fingerprint texture, visibly hardest. No flat frames anywhere in the trio;
LQ/PRED/GT triplets all consistent (noise→recovery story reads clean).
build_deck_v3.py composed slide 6: 3×3 layout inside stage cards (lq→pred→gt),
1.0067in squares from measured geometry, thin borders, A/B/C letters under squares,
full GPT-locked captions in the strip (TextBox 27) + rule disclosure. Verified:
9 pictures placed, 0 placeholders left, 10 FILLs, 0 slots, all lock strings,
0.37 MB. Deck deliverable: docs/DrishtiNet_KLA_PS01_final.pptx.

## §13 — SUBMISSION RECEIPT: repo live (2026-08-16, user console verified)

git init → add -A → commit (locked message) → branch main → push: 70 files / 25.0 MiB
to github.com/Adityag476/DRISHTI-Net (Public). Hygiene confirmed from the raw log:
weights/drishti_net.pt committed; outputs/restored_train (840 MB) and runs/ correctly
gitignored (absent from the commit); stale weights smoke file removed; reports/ +
panels + receipts all present; LF→CRLF warnings are cosmetic Windows notices only.
One stray artifact class noted: docs/panels/panel_003197..003199_* (9 PNGs) are the
SUPERSEDED pre-spread panel set, committed before the A/B/C canonical set existed —
cleanup queued as an optional commit (git rm docs/panels/panel_003*.png); the A/B/C
trio + panels_index.txt remain the canonical, reviewer-locked panels. Deck FILLs,
video, and PDF export are the only items left; this entry closes the technical trail.

### §13.1 — README polish round (2026-08-16)

README rebuilt in the user's telegram-export-studio visual DNA (shields badges,
emoji sections, tables, hero 3×3 panel grid via repo-relative <img> paths) while
preserving every reviewer word-lock (battery re-run on the new README: all locks
present, all banned phrasings absent, all 9 image paths verified on disk). Content
upgraded to the final receipts: gate table, honesty note, scoped OOD proxy, latency
protocol table, FabLoss V2 formula, repo map, reproduce-any-number commands, video
placeholder (PASTE-YOUTUBE-LINK-HERE), MASTER_SPEC paper-trail pointer.

### §13.2 — Demo stand-in characterization (builder self-check, 2026-08-18)

Measured on the synthetic install-check stand-ins (make_demo_data.py): GT 256²,
LQ 128² (task format ✓), naive-bilinear-upscale baseline PSNR vs GT 17.77–21.12 dB;
smoke placeholder 16.56–18.71 dB (below baseline — placeholder, expected);
champion restored_ship 18.08–21.38 dB — ABOVE both on all four frames
(Δ +0.12…+0.31 dB vs naive) with clearly visible denoising. Interpretation logged:
stand-ins are OFF-distribution content (resolution-chart patterns, mild degradation,
GT carries fine grain) → champion advantage compresses toward baseline, same shape as
the measured LODO result. DECISION: demo PSNR quoted NOWHERE (README/deck audited —
already clean); quality statements cite only the real-data holdout (29.33 dB /
0.7913, panels 29.07/37.22/23.87). Post-deadline queue: regenerate demo stand-ins
with the census-fingerprinted degradation engine so the quickstart doubles as a
true quality demo. (User-facing takeaway delivered: denoise visual on wafer_002.)

### §13.3 — Demo dark-speckle finding (Claude r16 visual call, builder-quantified)

Claude flagged dark speckle persisting on restored bright bars in demo wafer_002;
quantified on the champion's committed output (bright-structure mask = 15% of frame,
>0.20-depth outliers): GT 0.00% · degraded LQ 5.86% · champion 6.78% — the model
SHARPENS deep-dark speckle outliers on this synthetic stand-in content rather than
removing them (its learned degradation fingerprint doesn't match the stand-in
generator's; same off-distribution signature as LODO/§13.2). Real-data denoising
claim is UNAFFECTED and separately evidenced (flat-region residual σ = 0.0232 on the
160-frame holdout; A/B/C pred-vs-lq panels show clear noise removal — measured +
visually verified in rounds 11–14). RULES ADOPTED (Claude's wording): (1) video and
any demo narration claims "visible structure recovery relative to the degraded
input" for the synthetic stand-ins — the words "clean", "speckle-free", "artifact-
free" are banned from narration; (2) denoising claims are made ONLY with real-data
panels/reports; (3) the docs/panels A/B/C set remains the quality showcase. The
post-deadline demo-engine overhaul (§13.2 queue) is now a PRIORITY item, since the
install-check asset currently under-sells and partially mis-signals the model.

## §14 — VIDEO LIVE + deck links filled (2026-08-18)

Video uploaded and fetch-verified: youtu.be/XoaiKg5po4c — "Test Drishti-Net", 62 s,
Public. Deck slide-8 LINK placeholders filled at source (build_deck_v4.py): GitHub
line → github.com/Adityag476/DRISHTI-Net (live-verified through the whole round),
Video line → the youtu.be link with ACCURATE 62 s wording (no "fresh-clone" claim —
the capture uses the existing folder per the approved shot list). README 🎬 Demo
link swapped to the real URL, marker removed. Deck [FILL]s now 8 — all personal
team fields only (team name, college, city, contact, 4 member names). Polish
suggestion logged for the user: retitle the video from "Test Drishti-Net" to a
judge-facing title + drop the repo link into the video description (30 s in
YouTube Studio; no re-upload). PDF export (TeamName_KLA_PS01.pdf) is the LAST step.
