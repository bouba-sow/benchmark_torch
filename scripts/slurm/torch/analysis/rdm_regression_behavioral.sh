#!/usr/bin/env bash
# RDM-regression (held-out delta LL) track, Provo / English MECO, CPU only.
#
#   sbatch --array=0-15 scripts/slurm/torch/analysis/rdm_regression_behavioral.sh provo both
#   sbatch --dependency=afterok:<jobid> scripts/slurm/torch/analysis/rdm_regression_behavioral.sh provo both summarize
#
# Array task I of N (N = SHARD_COUNT, default = array size) scores every N-th model.
# qos_cpu-dev: max 2 h and 10 submitted jobs per user; runs are resumable, resubmit to continue.
# Item-partition repeats: $RDM_REPEATS (default 30). Output root: $RDM_REGRESSION_OUT (default $SCRATCH/rdm_regression). Resumable.
#SBATCH --account=torch_pr_287_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs_torch/logs/rdm_regression_%x_%A_%a.out
#SBATCH --error=outputs_torch/logs/rdm_regression_%x_%A_%a.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/ldh/una68ug}"
export WORK="${WORK:-/lustre/fswork/projects/rech/ldh/una68ug}"
source scripts/slurm/torch/torch_cache_env.sh
export MPLCONFIGDIR="/tmp/rdm-regression-mpl-${SLURM_JOB_ID}"
export TMPDIR="${RDM_TMPDIR:-$SCRATCH/rdm_regression_tmp}"; mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
dataset="${1:?dataset}"
case "$dataset" in
  pereira|tuckute|fedorenko2016|broderick|megmasc|nieuwland|szewczyk|podcast|nieuwland_avg|szewczyk_avg)
    driver=scripts/analysis/rdm_regression_neural.py ;;
  *) driver=scripts/analysis/rdm_regression_behavioral.py ;;
esac
track="${2:-both}"
phase="${3:-score}"
out="${RDM_REGRESSION_OUT:-$SCRATCH/rdm_regression}"
if [[ "$phase" == summarize ]]; then
  .venv/bin/python "$driver" "$dataset" --track "$track" \
    --out-root "$out" --summarize-only
else
  n="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:?submit as an array}}"
  .venv/bin/python "$driver" "$dataset" --track "$track" \
    --out-root "$out" --workers "${RDM_WORKERS:-4}" --n-repeats "${RDM_REPEATS:-30}" --shard "${SLURM_ARRAY_TASK_ID}/${n}"
fi
