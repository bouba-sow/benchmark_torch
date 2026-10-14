#!/bin/bash
#SBATCH -A ldh@cpu
#SBATCH --cpus-per-task=30
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ceil_beh_bins
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=outputs_jeanzay/logs/ceil_beh_bins_%A_%a.out
# Per-DLT-bin split-half delta-LL ceilings for the 12 behavioral entries of the DLT figures. sbatch this file.
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
ENTRIES=(geco geco_gp meco meco_gp meco_sp provo provo_gp provo_sp ucl_reading zuco_reading zuco_reading_sr zuco_reading_tsr)
"${PY[@]}" scripts/ceilings/dll_behavioral_bins.py --entry "${ENTRIES[$SLURM_ARRAY_TASK_ID]}" --n_workers 20
