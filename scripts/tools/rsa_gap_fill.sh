#!/bin/bash
# Throwaway RSA gap-fill launcher -- see rsa_gap_fill_run.py's docstring.
# One array task per model group (see rsa_gap_fill_groups.py). Exports every
# cache-dir env var the six target datasets' real run scripts expect, then
# hands off to rsa_gap_fill_run.py which loads each model once and runs it
# through all six.
#
# Usage (after generating the group file once):
#   python scripts/tools/rsa_gap_fill_groups.py --n-groups 5
#   sbatch --array=0-4 ... scripts/tools/rsa_gap_fill.sh
#
# Safe to re-submit after a timeout: every underlying script's own
# already_computed/already_done check makes finished (model, dataset) cells
# a fast no-op, and rsa_gap_fill_groups.json fixes each group's model list.

set -uo pipefail  # NOT -e: one dataset/model failing must not kill the whole group

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT" || exit 1

source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"

export PODCAST_FEATURE_CACHE_DIR="${PODCAST_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_features}"
export PODCAST_ELECTRODE_SELECTION_DIR="${PODCAST_ELECTRODE_SELECTION_DIR:-$PWD/outputs_jeanzay/podcast/electrode_selection}"

export BRODERICK_FEATURE_CACHE_DIR="${BRODERICK_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_features}"
export BRODERICK_EPOCH_CACHE_DIR="${BRODERICK_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_epochs}"
export BRODERICK_LME_FEATURE_CACHE_DIR="${BRODERICK_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_lme_features}"

export NIEUWLAND_FEATURE_CACHE_DIR="${NIEUWLAND_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_features}"
export NIEUWLAND_EPOCH_CACHE_DIR="${NIEUWLAND_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_epochs}"

export UCL_READING_FEATURE_CACHE_DIR="${UCL_READING_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/ucl_reading_features}"
export UCL_READING_DATA_ROOT="${UCL_READING_DATA_ROOT:-$WORK/benchmark/data/ucl_reading}"

export NS_FEATURE_CACHE_DIR="${NS_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/naturalstories_features}"

export ZUCO_TASK="NR"
export ZUCO_READING_EMBED_CACHE_DIR="${ZUCO_READING_EMBED_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/zuco_reading_features}/nr"

mkdir -p "$PODCAST_FEATURE_CACHE_DIR" "$PODCAST_ELECTRODE_SELECTION_DIR" \
  "$BRODERICK_FEATURE_CACHE_DIR" "$BRODERICK_EPOCH_CACHE_DIR" "$BRODERICK_LME_FEATURE_CACHE_DIR" \
  "$NIEUWLAND_FEATURE_CACHE_DIR" "$NIEUWLAND_EPOCH_CACHE_DIR" \
  "$UCL_READING_FEATURE_CACHE_DIR" "$NS_FEATURE_CACHE_DIR" "$ZUCO_READING_EMBED_CACHE_DIR" \
  outputs_jeanzay/logs

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
"${PY[@]}" scripts/tools/rsa_gap_fill_run.py \
  --group "${IDX}" \
  --groups-file "${GROUPS_FILE:-scripts/tools/rsa_gap_fill_groups.json}" \
  --device cuda
echo "Job completed at $(date)"
