#!/bin/bash
# ONE job per model-size class that loops over every model of every family, both Szewczyk studies per
# model (single-study item-mean ridge; see scripts/run/embedding/neural/szewczyk_item_mean.py).
# Resumable: finished (study, model) pairs are skipped.
#   small (<20B, 1 GPU):   sbatch --gres=gpu:1 --time=01:00:00 --export=ALL,MAXP=20e9  <this file>
#   large (>=20B, 2 GPU):  sbatch --gres=gpu:2 --cpus-per-task=10 --time=01:00:00 --export=ALL,MINP=20e9  <this file>
#SBATCH --account=ldh@h100
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=5
#SBATCH --hint=nomultithread
#SBATCH --output=outputs_jeanzay/logs/szewczyk_item_mean_all_%j.out
set -euo pipefail
: "${WORK:?}"; export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"; cd "$BENCHMARK_ROOT"
source scripts/slurm/jz/jz_cache_env.sh
export SZEWCZYK_FEATURE_CACHE_DIR="${SZEWCZYK_FEATURE_CACHE_DIR:-${BENCHMARK_CACHE_ROOT}/szewczyk_features}"
mkdir -p outputs_jeanzay/logs "$SZEWCZYK_FEATURE_CACHE_DIR"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# forgetting-transformer / stickbreaking are vendored and needed by the Lanni-ni constrained models
export PYTHONPATH="$BENCHMARK_ROOT/forgetting-transformer/src:$BENCHMARK_ROOT/stickbreaking-attention:${PYTHONPATH:-}"
echo "host=$(hostname) start=$(date) MINP=${MINP:-0} MAXP=${MAXP:-inf}"
.venv/bin/python scripts/run/embedding/neural/szewczyk_item_mean.py --from-registry --all-models \
  --studies f2007 h2019 --min-params "${MINP:-0}" --max-params "${MAXP:-inf}" \
  --out_dir outputs_jeanzay --feature_cache_dir "$SZEWCZYK_FEATURE_CACHE_DIR" --device cuda
echo "done $(date)"
