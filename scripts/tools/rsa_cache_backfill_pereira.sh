#!/bin/bash
set -uo pipefail
: "${WORK:?Set WORK}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT" || exit 1
source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
export PEREIRA_EMBED_FEATURE_CACHE_DIR="${PEREIRA_EMBED_FEATURE_CACHE_DIR:-$SCRATCH/.cache/pereira_embed_features}"
mkdir -p "$PEREIRA_EMBED_FEATURE_CACHE_DIR" outputs_jeanzay/logs
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORK/hf_datasets}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if command -v uv >/dev/null 2>&1; then PY=(uv run --frozen --no-sync --no-dev python); else source .venv/bin/activate; PY=(python); fi
"${PY[@]}" scripts/tools/rsa_cache_backfill_pereira_run.py --models-file scripts/tools/rsa_cache_backfill_pereira.json --device cuda
