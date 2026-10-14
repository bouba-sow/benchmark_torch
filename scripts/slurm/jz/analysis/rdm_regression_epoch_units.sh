#!/usr/bin/env bash
#SBATCH -A ldh@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/rdm_epoch_units_%A_%a.out
#SBATCH --error=outputs_jeanzay/logs/rdm_epoch_units_%A_%a.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/ldh/una68ug}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then SH="--shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"; else SH=""; fi
.venv/bin/python scripts/analysis/rdm_regression_epoch_units.py --out-root "${OUT_ROOT:-$SCRATCH/rdm_epoch_units}" $SH "$@"
