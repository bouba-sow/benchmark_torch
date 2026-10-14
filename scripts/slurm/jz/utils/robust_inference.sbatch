#!/bin/bash
# Robust inference for the matched-scale constraint regressions (paper Table 2): family/lineage random effects,
# unit random slopes, joint behavioral x neural model, robustness variants, lineage cluster bootstrap, per-unit fits.
# Reads saved last-layer scores only. Output: reports/robust_inference/ (override: OUT=...; N_BOOT=...).
#   sbatch scripts/slurm/jz/utils/robust_inference.sbatch
#SBATCH --account=ldh@cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=robust_inference
#SBATCH --output=outputs_jeanzay/logs/robust_inference_%j.out
set -euo pipefail
: "${WORK:?}"; export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"; cd "$BENCHMARK_ROOT"
source scripts/slurm/jz/jz_cache_env.sh 2>/dev/null || true
module load r/4.5.2 2>/dev/null || true
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=1
OUT="${OUT:-reports/robust_inference}"; N_BOOT="${N_BOOT:-999}"
mkdir -p "$OUT"
.venv/bin/python scripts/analysis/robust_inference_export.py --out "$OUT"
Rscript scripts/analysis/robust_inference.R "$OUT" "$N_BOOT" "${SLURM_CPUS_PER_TASK:-8}"
Rscript scripts/analysis/robust_inference_stepB.R "$OUT"
cat "$OUT/summary.txt"
