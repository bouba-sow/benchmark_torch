#!/usr/bin/env bash
# Leave-one-subject-out RDM noise ceiling for one per-subject neural dataset (one job, no shards).
#   sbatch scripts/slurm/jz/analysis/rdm_ceiling_neural.sh broderick
#SBATCH -A ldh@cpu
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=04:00:00
#SBATCH --output=outputs_jeanzay/logs/rdm_ceiling_loso_%x_%j.out
#SBATCH --error=outputs_jeanzay/logs/rdm_ceiling_loso_%x_%j.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/ldh/una68ug}"
export WORK="${WORK:-/lustre/fswork/projects/rech/ldh/una68ug}"
source scripts/slurm/jz/jz_cache_env.sh
export MPLCONFIGDIR="/tmp/rdm-ceiling-loso-mpl-${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
.venv/bin/python scripts/analysis/rdm_ceiling_neural.py "${1:?dataset}" --out-root "${RDM_CEIL_OUT:-$SCRATCH/rdm_ceiling}"
