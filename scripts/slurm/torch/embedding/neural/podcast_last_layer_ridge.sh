#!/usr/bin/env bash
#SBATCH -A ldh@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs_torch/logs/podcast_last_layer_ridge_%A_%a.out
#SBATCH --error=outputs_torch/logs/podcast_last_layer_ridge_%A_%a.err
#SBATCH --array=0-117%4
set -euo pipefail
# Fit only the final layer (2026-10-12; caches keep every layer). EMBED_LAST_LAYER_ONLY=0 for the full sweep.
export EMBED_LAST_LAYER_ONLY="${EMBED_LAST_LAYER_ONLY:-1}"
: "${WORK:=/lustre/fswork/projects/rech/ldh/una68ug}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"
cd "$BENCHMARK_ROOT"
if [[ -n "${SCRATCH:-}" ]]; then
  source scripts/slurm/torch/torch_cache_env.sh
fi
export MPLCONFIGDIR="/tmp/podcast-last-layer-ridge-${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/analysis/podcast_last_layer_ridge.py --model-index "$SLURM_ARRAY_TASK_ID"
