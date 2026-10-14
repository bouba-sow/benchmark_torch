#!/bin/bash
# Throwaway RSA raw-embedding-cache backfill launcher -- see
# rsa_cache_backfill_run.py's docstring. One array task per model group (see
# rsa_cache_backfill_groups.py). Exports every cache-dir env var the six
# target datasets' real run scripts expect (matching the REAL production
# .sh wrappers under scripts/slurm/jz/embedding/..., not the stale/legacy
# "_embed_features" dirs some of them also have on disk), then hands off to
# rsa_cache_backfill_run.py which loads each model once and runs it through
# every dataset where it's missing.
#
# Usage (after generating the group file once):
#   python scripts/tools/rsa_cache_backfill_groups.py --n-groups 14
#   sbatch --array=0-13 ... scripts/tools/rsa_cache_backfill.sh
#
# Safe to re-submit after a timeout: every underlying extraction path's own
# on-disk cache lookup makes an already-cached (model, dataset) pair a fast
# no-op, and rsa_cache_backfill_groups.json fixes each group's model list.

set -uo pipefail  # NOT -e: one dataset/model failing must not kill the whole group

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT" || exit 1

source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"

# fedorenko2016, nieuwland, podcast: jz_cache_env.sh already exports these
# correctly (FEDORENKO_FEATURE_CACHE_DIR, NIEUWLAND_FEATURE_CACHE_DIR,
# PODCAST_FEATURE_CACHE_DIR) -- nothing to override.

# ucl_reading (embedding track): shares the surprisal track's cache dir by
# convention -- NOT ucl_reading_embed_features (stale/legacy).
export UCL_READING_EMBED_CACHE_DIR="${UCL_READING_EMBED_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/ucl_reading_features}"

# zuco_reading NR (embedding track): NOT zuco_reading_embed_features.
export ZUCO_READING_EMBED_CACHE_DIR="${ZUCO_READING_EMBED_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/zuco_reading_features/nr}"

# naturalstories (embedding track): configs/embedding/behavioral/naturalstories.yaml
# hardcodes ${BENCHMARK_ROOT}/.cache/naturalstories_features -- not an env var,
# so nothing to export, just make sure the dir exists.
mkdir -p "${BENCHMARK_ROOT}/.cache/naturalstories_features"

mkdir -p "$UCL_READING_EMBED_CACHE_DIR" "$ZUCO_READING_EMBED_CACHE_DIR" outputs_jeanzay/logs

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORK/hf_datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  source .venv/bin/activate
  PY=(python)
fi

"${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

echo "Job started at $(date)  group=${IDX}  node=$(hostname)"
"${PY[@]}" scripts/tools/rsa_cache_backfill_run.py \
  --group "${IDX}" \
  --groups-file "${GROUPS_FILE:-scripts/tools/rsa_cache_backfill_groups.json}" \
  --device cuda
echo "Job completed at $(date)"
