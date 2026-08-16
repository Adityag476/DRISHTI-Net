@echo off
REM ============================================================
REM  DRISHTI-Net - unattended overnight ladder (M0 -> M4)
REM  Usage:   scripts\run_ladder.bat            (all rungs)
REM           scripts\run_ladder.bat M1 M2 M3 M4 (subset - e.g. M0 already done)
REM  Runs sequentially on the single GPU. ~80 min per rung.
REM  A rung only ships if it beats the running best (see harvest_ladder.py).
REM ============================================================
setlocal enabledelayedexpansion
cd /d C:\Users\Adity\Desktop\DRISHTI-Net
set GT=C:\Users\Adity\Desktop\dataset\train\GT
set LQ=C:\Users\Adity\Desktop\dataset\train\NoisyLR

if "%~1"=="" (set LEVELS=M0 M1 M2 M3 M4) else (set LEVELS=%*)

for %%L in (%LEVELS%) do (
  echo ============================================================
  echo  STARTING RUNG %%L   -   !DATE! !TIME!
  echo ============================================================
  python train.py --gt_dir %GT% --real_lq %LQ% --real_gt %GT% --level %%L --iters 30000 --batch 8 --workers 4 --val_every 1000 --out runs/%%L
  echo  RUNG %%L FINISHED   -   !DATE! !TIME!
)

echo ============================================================
echo  LADDER COMPLETE - harvest:
python scripts\harvest_ladder.py --runs runs
endlocal
