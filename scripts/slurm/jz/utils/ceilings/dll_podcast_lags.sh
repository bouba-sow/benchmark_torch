#!/bin/bash
#SBATCH -A ldh@cpu
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ceil_pod_lag
#SBATCH --time=01:00:00
#SBATCH --array=7-25
#SBATCH --output=outputs_jeanzay/logs/ceil_pod_lag_%A_%a.out
#
# Per-lag split-half DELTA-LL ceilings for Podcast (one array task per lag index, 20 workers =
# 10 splits x 2 directions). Lags 7-25 = 0.019-0.581 s (0-600 ms post-onset, 31.25 ms grid).
# Writes to outputs_jeanzay/noise_ceilings/podcast_lags/ (NOT outputs/noise_ceilings).
#   sbatch scripts/slurm/jz/utils/ceilings/dll_podcast_lags.sh
set -euo pipefail
: "${WORK:?Set WORK}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if ! command -v R >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load r/4.5.2 2>/dev/null || module load r 2>/dev/null || module load R 2>/dev/null || true
fi
command -v R >/dev/null 2>&1 || { echo "ERROR: R not found" >&2; exit 1; }
export R_HOME="${R_HOME:-$(R RHOME)}"
export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
[[ -d "${R_HOME}/lib" ]] && export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
if command -v uv >/dev/null 2>&1; then PY=(uv run --frozen --no-sync --no-dev python); else source .venv/bin/activate; PY=(python); fi
"${PY[@]}" scripts/ceilings/dll_ecog.py --dataset podcast --lag "${SLURM_ARRAY_TASK_ID}" \
  --n_splits 10 --n_workers 20 --seed 0 --out_dir outputs_jeanzay/noise_ceilings/podcast_lags
