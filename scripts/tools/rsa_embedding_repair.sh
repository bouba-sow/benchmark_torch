#!/bin/bash
set -euo pipefail
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-${WORK}/benchmark}"
export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-${SCRATCH}/.cache}"
cd "$BENCHMARK_ROOT"
export UCL_READING_EMBED_CACHE_DIR="$BENCHMARK_CACHE_ROOT/ucl_reading_features"
export ZUCO_READING_EMBED_CACHE_DIR="$BENCHMARK_CACHE_ROOT/zuco_reading_features/nr"
export BRODERICK_FEATURE_CACHE_DIR="$BENCHMARK_CACHE_ROOT/broderick_features"
export BRODERICK_EPOCH_CACHE_DIR="$BENCHMARK_CACHE_ROOT/broderick_epochs"
export PEREIRA_EMBED_FEATURE_CACHE_DIR="$BENCHMARK_CACHE_ROOT/pereira_embed_features"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  PY=("$BENCHMARK_ROOT/.venv/bin/python")
fi
"${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py
exec "${PY[@]}" scripts/tools/rsa_embedding_repair.py "$@"
