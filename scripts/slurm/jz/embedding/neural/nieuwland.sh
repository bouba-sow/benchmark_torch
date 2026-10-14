# Shared runtime for Jean Zay Nieuwland EEG N400 per-family checkpoint sweeps.
# Source from scripts/slurm/jz/nieuwland_<family>.sh (do not run directly).
#
# Required before source:
#   NIEUWLAND_FAMILY    — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, NIEUWLAND_SWEEP_ROOT
#   BENCHMARK_CACHE_ROOT         — default: $SCRATCH/.cache on Jean Zay
#   NIEUWLAND_FEATURE_CACHE_DIR  — default: $BENCHMARK_CACHE_ROOT/nieuwland_features
#   NIEUWLAND_EPOCH_CACHE_DIR    — default: $BENCHMARK_CACHE_ROOT/nieuwland_epochs
#   N_LAYER_WORKERS              — parallel GPU layer workers (default 36)
#   SAVE_WORD_PREDICTIONS=1      — also save each channel's best-layer OOF
#                                  item-mean ERP prediction, for the DLT-cost
#                                  difficulty-split analysis
#
# NOTE: this reads NIEUWLAND_SWEEP_ROOT specifically, NOT the generic
# SWEEP_ROOT — a stale `export SWEEP_ROOT=...` left over from another
# dataset's session would otherwise silently leak in and write output to the
# wrong place (this bit Podcast once already).

: "${NIEUWLAND_FAMILY:?Set NIEUWLAND_FAMILY before sourcing job_nieuwland_jz_common.sh}"

set -euo pipefail
# Fit only the final layer (2026-10-12; caches keep every layer). EMBED_LAST_LAYER_ONLY=0 for the full sweep.
export EMBED_LAST_LAYER_ONLY="${EMBED_LAST_LAYER_ONLY:-1}"

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export NIEUWLAND_FEATURE_CACHE_DIR="${NIEUWLAND_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_features}"
  export NIEUWLAND_EPOCH_CACHE_DIR="${NIEUWLAND_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_epochs}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" \
    "$NIEUWLAND_FEATURE_CACHE_DIR" "$NIEUWLAND_EPOCH_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/embedding/neural/nieuwland.yaml}"
SWEEP_ROOT="${NIEUWLAND_SWEEP_ROOT:-$PWD/outputs_jeanzay/nieuwland/embedding_sweep}"
N_LAYER_WORKERS="${N_LAYER_WORKERS:-36}"

# On by default — this sweep's output feeds the DLT-cost difficulty-split
# analysis. Set SAVE_WORD_PREDICTIONS=0 to disable.
export SAVE_WORD_PREDICTIONS="${SAVE_WORD_PREDICTIONS:-1}"
SAVE_WORD_PREDICTIONS_FLAG=()
if [[ "${SAVE_WORD_PREDICTIONS}" == "1" ]]; then
  SAVE_WORD_PREDICTIONS_FLAG=(--save_word_predictions)
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

nieuwland_jz_check_data() {
  if [[ ! -f "${NIEUWLAND_BIDS_ROOT}/participants.tsv" ]]; then
    echo "ERROR: Nieuwland BIDS root not found: ${NIEUWLAND_BIDS_ROOT}/participants.tsv"
    return 1
  fi
}

nieuwland_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export SAFETENSORS_FAST_GPU=1   # memory-map safetensors directly to GPU (no CPU staging)
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export MNE_USE_NUMBA=false

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORK/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  export NIEUWLAND_FEATURE_CACHE_DIR="${NIEUWLAND_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_features}"
  export NIEUWLAND_EPOCH_CACHE_DIR="${NIEUWLAND_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_epochs}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "$NIEUWLAND_FEATURE_CACHE_DIR" \
    "$NIEUWLAND_EPOCH_CACHE_DIR" \
    "${SWEEP_ROOT}/${NIEUWLAND_FAMILY}" \
    outputs_jeanzay/logs
}

nieuwland_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${NIEUWLAND_FAMILY}"
}

nieuwland_jz_run_array_task() {
  local _n
  _n="$(nieuwland_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${NIEUWLAND_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${NIEUWLAND_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "NIEUWLAND_BIDS_ROOT=${NIEUWLAND_BIDS_ROOT}"
  echo "N_LAYER_WORKERS=${N_LAYER_WORKERS}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
  nvidia-smi

  nieuwland_jz_check_data || exit 1
  nieuwland_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${NIEUWLAND_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/embedding/neural/nieuwland.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NIEUWLAND_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --feature_cache_dir "${NIEUWLAND_FEATURE_CACHE_DIR}" \
    --epoch_cache_dir "${NIEUWLAND_EPOCH_CACHE_DIR}" \
    --regression_device cuda \
    --n_layer_workers "${N_LAYER_WORKERS}" \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}"

  echo "Job completed at $(date)"
}

nieuwland_jz_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  nieuwland_jz_setup_env
  "${PY[@]}" scripts/run/embedding/neural/nieuwland.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NIEUWLAND_FAMILY}" \
    --summary_only \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
