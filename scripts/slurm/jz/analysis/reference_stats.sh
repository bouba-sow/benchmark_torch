#!/usr/bin/env bash
#SBATCH -A ldh@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/refstats_%A_%a.out
#SBATCH --error=outputs_jeanzay/logs/refstats_%A_%a.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/ldh/una68ug} WORK=${WORK:-/lustre/fswork/projects/rech/ldh/una68ug}
source scripts/slurm/jz/jz_cache_env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
.venv/bin/python scripts/features/reference_stats.py "$@"
