#!/bin/bash
#SBATCH --account=torch_pr_287_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ceil_pod_reg
#SBATCH --time=00:30:00
#SBATCH --array=7-25
#SBATCH --output=outputs_torch/logs/ceil_pod_reg_%A_%a.out
#
# Per-lag, per-REGION (region-averaged, (1|subject) only) split-half DELTA-LL ceilings for Podcast (lag = array index;
# Lags 7-25 = 0.019-0.581 s. Output: outputs_torch/noise_ceilings/podcast_lags_region/
#   sbatch scripts/slurm/torch/utils/ceilings/dll_podcast_region.sh
set -euo pipefail
: "${WORK:?Set WORK}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"
cd "$BENCHMARK_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if ! command -v R >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load r/4.6.1 2>/dev/null || module load r 2>/dev/null || module load R 2>/dev/null || true
fi
command -v R >/dev/null 2>&1 || { echo "ERROR: R not found" >&2; exit 1; }
export R_HOME="${R_HOME:-$(R RHOME)}"
export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
[[ -d "${R_HOME}/lib" ]] && export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
if command -v uv >/dev/null 2>&1; then PY=(uv run --frozen --no-sync --no-dev python); else source .venv/bin/activate; PY=(python); fi
"${PY[@]}" scripts/ceilings/dll_podcast_region.py --lag "${SLURM_ARRAY_TASK_ID}" \
  --n_splits 10 --n_workers 20 --seed 0 --out_dir outputs_torch/noise_ceilings/podcast_lags_region
