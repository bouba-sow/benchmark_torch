#!/usr/bin/env bash
#SBATCH -A ldh@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/podcast_lagwise_%A_%a.out
#SBATCH --error=outputs_jeanzay/logs/podcast_lagwise_%A_%a.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export MPLCONFIGDIR=/tmp/podcast-lagwise-mpl-${SLURM_JOB_ID}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
track="${1:?embedding or surprisal}"
phase="${2:?prepare or score}"
idx="${SLURM_ARRAY_TASK_ID:?submit as an array}"
if [[ "$phase" == prepare ]]; then
  printf -v subject 'sub-%02d' "$((idx + 1))"
  .venv/bin/python scripts/analysis/podcast_lagwise_rsa.py "$track" prepare --subject "$subject"
elif [[ "$phase" == score ]]; then
  stride="${PODCAST_RSA_SCORE_STRIDE:-8}"
  for ((model_idx=idx; model_idx<118; model_idx+=stride)); do
    .venv/bin/python scripts/analysis/podcast_lagwise_rsa.py "$track" score --model-index "$model_idx"
  done
else
  exit 2
fi
