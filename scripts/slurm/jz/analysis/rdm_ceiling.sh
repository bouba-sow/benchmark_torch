#!/usr/bin/env bash
# Split-half RDM noise ceiling + half-target model scores, last layer (scripts/analysis/rdm_ceiling_run.py), CPU only.
#
#   sbatch --array=0-39 scripts/slurm/jz/analysis/rdm_ceiling.sh provo
#   sbatch --dependency=afterany:<jobid> scripts/slurm/jz/analysis/rdm_ceiling.sh provo summarize
#
# Array task I of N (N = array size) scores every N-th model of both tracks. Resumable (a model is done
# when its ceil_*.npz exists). Env: RDM_CEIL_OUT (default $SCRATCH/rdm_ceiling),
# RDM_WORKERS (2).
#SBATCH -A ldh@cpu
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=04:00:00
#SBATCH --output=outputs_jeanzay/logs/rdm_ceiling_%x_%A_%a.out
#SBATCH --error=outputs_jeanzay/logs/rdm_ceiling_%x_%A_%a.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/ldh/una68ug}"
export WORK="${WORK:-/lustre/fswork/projects/rech/ldh/una68ug}"
source scripts/slurm/jz/jz_cache_env.sh
export MPLCONFIGDIR="/tmp/rdm-ceiling-mpl-${SLURM_JOB_ID}"
export TMPDIR="${RDM_TMPDIR:-$SCRATCH/rdm_regression_tmp}"; mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
dataset="${1:?dataset}"
phase="${2:-run}"
out="${RDM_CEIL_OUT:-$SCRATCH/rdm_ceiling}"
if [[ "$phase" == summarize ]]; then
  .venv/bin/python scripts/analysis/rdm_ceiling_run.py "$dataset" --out-root "$out" --summarize-only
else
  n="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:?submit as an array}}"
  .venv/bin/python scripts/analysis/rdm_ceiling_run.py "$dataset" --out-root "$out" \
    --workers "${RDM_WORKERS:-2}" --track "${RDM_TRACK:-both}" --shard "${SLURM_ARRAY_TASK_ID}/${n}"
fi
