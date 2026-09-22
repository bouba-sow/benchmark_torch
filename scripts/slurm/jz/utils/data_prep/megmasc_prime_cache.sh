#!/bin/bash
#SBATCH -A ldh@cpu
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=mm_prime
#SBATCH --time=02:00:00
#SBATCH --qos=qos_cpu-dev
#SBATCH --output=outputs_jeanzay/logs/megmasc_prime_%A_%a.out
#SBATCH --array=0-9
#
# MEG-MASC epoch-cache prime — one subject per array task, CPU only.
# Reads raw KIT .con, bandpass [0.1,40] Hz @ 1000 Hz -> resample 50 Hz ->
# RobustScaler+clamp -> word-locked epochs -> one .npz per subject under
# $MEGMASC_EPOCH_CACHE_DIR.  No GPU, no LLM, no regression.
#
#   sbatch scripts/slurm/jz/utils/data_prep/megmasc_prime_cache.sh              # subjects[0..9]
#   sbatch --array=10 scripts/slurm/jz/utils/data_prep/megmasc_prime_cache.sh   # the 11th, after a slot frees

set -euo pipefail
: "${WORK:?Set WORK (e.g. export WORK=/lustre/fswork/projects/rech/ldh/una68ug)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT"

if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
fi
export MEGMASC_BIDS_ROOT="${MEGMASC_BIDS_ROOT:-${SCRATCH:-$BENCHMARK_ROOT/..}/data/gwilliams}"
export MEGMASC_EPOCH_CACHE_DIR="${MEGMASC_EPOCH_CACHE_DIR:-${BENCHMARK_CACHE_ROOT}/megmasc_epochs}"
mkdir -p outputs_jeanzay/logs "$MEGMASC_EPOCH_CACHE_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  source .venv/bin/activate
  PY=(python)
fi

echo "Job $SLURM_JOB_ID  array_task=$IDX  host=$(hostname)  $(date)"
echo "MEGMASC_BIDS_ROOT=$MEGMASC_BIDS_ROOT"
echo "MEGMASC_EPOCH_CACHE_DIR=$MEGMASC_EPOCH_CACHE_DIR"

"${PY[@]}" scripts/data/megmasc_epoch_cache.py \
  --config configs/embedding/neural/megmasc.yaml \
  --array-index "$IDX" \
  --epoch_cache_dir "$MEGMASC_EPOCH_CACHE_DIR"

echo "Done $(date)"
