#!/usr/bin/env bash
# Stimulus bootstrap of the last-layer RDM delta LL (scripts/analysis/rdm_bootstrap_run.py), CPU only.
#
#   sbatch --array=0-39 scripts/slurm/torch/analysis/rdm_bootstrap.sh provo
#   sbatch --dependency=afterany:<jobid> scripts/slurm/torch/analysis/rdm_bootstrap.sh provo summarize
#
# Array task I of N (N = array size) scores every N-th model of both tracks. Resumable (a model is done
# when its boot_*.npz exists). Env: RDM_BOOT_OUT (default $SCRATCH/rdm_bootstrap_r200), RDM_NBOOT (200),
# RDM_WORKERS (2).
#SBATCH --account=torch_pr_287_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=04:00:00
#SBATCH --output=outputs_torch/logs/rdm_bootstrap_%x_%A_%a.out
#SBATCH --error=outputs_torch/logs/rdm_bootstrap_%x_%A_%a.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/ldh/una68ug}"
export WORK="${WORK:-/lustre/fswork/projects/rech/ldh/una68ug}"
source scripts/slurm/torch/torch_cache_env.sh
export MPLCONFIGDIR="/tmp/rdm-bootstrap-mpl-${SLURM_JOB_ID}"
export TMPDIR="${RDM_TMPDIR:-$SCRATCH/rdm_regression_tmp}"; mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
dataset="${1:?dataset}"
phase="${2:-run}"
out="${RDM_BOOT_OUT:-$SCRATCH/rdm_bootstrap_r200}"
if [[ "$phase" == summarize ]]; then
  .venv/bin/python scripts/analysis/rdm_bootstrap_run.py "$dataset" --out-root "$out" --summarize-only
else
  n="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:?submit as an array}}"
  .venv/bin/python scripts/analysis/rdm_bootstrap_run.py "$dataset" --out-root "$out" \
    --workers "${RDM_WORKERS:-2}" --n-boot "${RDM_NBOOT:-200}" --track "${RDM_TRACK:-both}" --shard "${SLURM_ARRAY_TASK_ID}/${n}"
fi
