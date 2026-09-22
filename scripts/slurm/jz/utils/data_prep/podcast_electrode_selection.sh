#!/bin/bash
#SBATCH -A ldh@h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=podcast_elec_sel
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/podcast_electrode_selection_%A_%a.out
#SBATCH --array=0-8
#
# Podcast electrode selection — GloVe-proxy encoding + phase-randomisation null.
# One SLURM array job per subject (sub-01 … sub-09).
# Follows Bhattacharjee et al. (2024, Nat Comput Sci) §2.3.
#
#   sbatch scripts/slurm/jz/utils/data_prep/podcast/electrode_selection.sh
#
# Estimated wall time per subject: ~50 min (worst case sub-03, 235 channels).
# Time limit set to 4 h to include safety margin.

set -euo pipefail
: "${WORK:?Set WORK (e.g. export WORK=/lustre/fswork/projects/rech/ldh/una68ug)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

cd "$BENCHMARK_ROOT" || exit 1

# ── Subject mapping: array index → BIDS subject ID ───────────────────────────
SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09)
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

# ── Paths ─────────────────────────────────────────────────────────────────────
BIDS_ROOT="${WORK}/benchmark/data/podcast"
OUT_DIR="${BENCHMARK_ROOT}/outputs_jeanzay/podcast/electrode_selection"

mkdir -p "$OUT_DIR" outputs_jeanzay/logs

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MNE_USE_NUMBA=false

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  source .venv/bin/activate
  PY=(python)
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}  Subject: ${SUBJECT}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-?}  nproc: $(nproc)"
echo "BIDS_ROOT: ${BIDS_ROOT}"
echo "OUT_DIR:   ${OUT_DIR}"

"${PY[@]}" scripts/data/podcast/electrode_selection.py \
    --bids_root  "$BIDS_ROOT"  \
    --out_dir    "$OUT_DIR"    \
    --subjects   "$SUBJECT"   \
    --n_rand     5000          \
    --n_jobs     80            \
    --skip_done

echo "Job completed at $(date)"
