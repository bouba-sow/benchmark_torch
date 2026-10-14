#!/usr/bin/env bash
#SBATCH -A ldh@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/podcast_lagwise_std_%A_%a.out
#SBATCH --error=outputs_jeanzay/logs/podcast_lagwise_std_%A_%a.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 RSA_STANDARDIZE=1
idx="${SLURM_ARRAY_TASK_ID:?submit as an array}"; stride="${STRIDE:-4}"
for ((m=idx; m<118; m+=stride)); do
  .venv/bin/python scripts/analysis/podcast_lagwise_rsa.py embedding score --model-index "$m"
done
