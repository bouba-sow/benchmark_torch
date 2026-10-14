# Shared runtime for Torch Podcast ECoG per-family checkpoint sweeps.
# Source from scripts/slurm/torch/podcast_<family>.sh (do not run directly).
#
# Required before source:
#   PODCAST_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, PODCAST_SWEEP_ROOT, PODCAST_BIDS_ROOT
#   BENCHMARK_CACHE_ROOT  — default: $SCRATCH/.cache on Torch
#   PODCAST_FEATURE_CACHE_DIR — default: $BENCHMARK_CACHE_ROOT/podcast_features
#   N_JOBS            — parallel workers for per-electrode regression (default 10)
#
# UNTRAINED=1 — run the untrained-baseline path instead of pretrained weights
#   (random init, same architecture, via --untrained). Results and the
#   activation cache are written under a separate "_untrained" tree so they
#   never mix with trained-model runs. Set UNTRAINED_SEED to change the
#   random-init seed (default 0).
#   UNTRAINED=1 python scripts/submit.py podcast embedding fit --family gpt2
#
# NOTE: this reads PODCAST_SWEEP_ROOT specifically, NOT the generic SWEEP_ROOT —
# a stale `export SWEEP_ROOT=...` left over from a Broderick/GECO/etc. session
# previously leaked into Podcast jobs and silently wrote its output into the
# wrong dataset's directory. Namespacing per-dataset closes that off.

: "${PODCAST_FAMILY:?Set PODCAST_FAMILY before sourcing job_podcast_torch_common.sh}"

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export PODCAST_FEATURE_CACHE_DIR="${PODCAST_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$PODCAST_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/embedding/neural/podcast.yaml}"
UNTRAINED="${UNTRAINED:-0}"
UNTRAINED_SEED="${UNTRAINED_SEED:-0}"
SWEEP_ROOT="${PODCAST_SWEEP_ROOT:-$PWD/outputs_torch/podcast/embedding_sweep}"
PODCAST_BIDS_ROOT="${PODCAST_BIDS_ROOT:-$SCRATCH/Documents/benchmark/data/podcast}"
PODCAST_ELECTRODE_SELECTION_DIR="${PODCAST_ELECTRODE_SELECTION_DIR:-$PWD/outputs_torch/podcast/electrode_selection}"
N_JOBS="${N_JOBS:-80}"

# NOTE: --out_dir/--feature_cache_dir passed to the python script below are
# the *plain* (non-suffixed) paths -- run/embedding/neural/podcast.py itself
# appends "_untrained" when --untrained is set. Don't also suffix here, or
# the two suffixes stack (…_untrained_untrained).
UNTRAINED_ARGS=()
if [[ "${UNTRAINED}" == "1" ]]; then
  UNTRAINED_ARGS=(--untrained --untrained_seed "${UNTRAINED_SEED}")
fi

# On by default — this sweep's output feeds the DLT-cost difficulty-split
# analysis, which needs each electrode's best-layer out-of-fold predicted lag
# matrix (.npz per subject). Set SAVE_WORD_PREDICTIONS=0 to disable.
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

podcast_torch_check_data() {
  local _missing=0
  if [[ ! -f "${PODCAST_BIDS_ROOT}/stimuli/podcast_transcript.csv" ]]; then
    echo "ERROR: Podcast transcript not found: ${PODCAST_BIDS_ROOT}/stimuli/podcast_transcript.csv"
    _missing=1
  fi
  if [[ ! -d "${PODCAST_BIDS_ROOT}/derivatives/ecogprep/sub-01" ]]; then
    echo "ERROR: Podcast derivatives not found: ${PODCAST_BIDS_ROOT}/derivatives/ecogprep/"
    _missing=1
  fi
  return "$_missing"
}

podcast_torch_setup_env() {
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
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${PODCAST_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_features}" \
    "${SWEEP_ROOT}/${PODCAST_FAMILY}" outputs_torch/logs
}

podcast_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${PODCAST_FAMILY}"
}

podcast_torch_run_array_task() {
  local _n
  _n="$(podcast_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${PODCAST_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${PODCAST_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT} (effective out_dir gets _untrained suffix from the python script when UNTRAINED=1)"
  echo "UNTRAINED=${UNTRAINED} UNTRAINED_SEED=${UNTRAINED_SEED}"
  echo "PODCAST_BIDS_ROOT=${PODCAST_BIDS_ROOT}"
  echo "N_JOBS=${N_JOBS}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
  nvidia-smi

  podcast_torch_check_data || exit 1
  podcast_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${PODCAST_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/embedding/neural/podcast.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${PODCAST_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --n_jobs "${N_JOBS}" \
    --electrode_selection_dir "${PODCAST_ELECTRODE_SELECTION_DIR}" \
    --regression_device cuda \
    --skip_summary \
    --no_plots \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}" \
    "${UNTRAINED_ARGS[@]}"

  echo "Job completed at $(date)"
}

# Stage 1 — GPU: compute and cache activations for one array task (one checkpoint).
# Cache hit = model never loaded; cache miss = forward pass + write + free GPU.
podcast_torch_prefetch_activations() {
  local _n
  _n="$(podcast_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] — exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Stage 1 — activations only (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_task=${IDX}  family=${PODCAST_FAMILY}"
  podcast_torch_setup_env

  "${PY[@]}" scripts/run/embedding/neural/podcast.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${PODCAST_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --activations_only \
    --skip_summary \
    "${UNTRAINED_ARGS[@]}"

  echo "Stage 1 completed at $(date)"
}

# Stage 2 — CPU: ridge regression only (activations read from cache, no GPU).
podcast_torch_run_regression() {
  local _n
  _n="$(podcast_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] — exiting."
    exit 0
  fi

  echo "Stage 2 — regression only (CPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_task=${IDX}  family=${PODCAST_FAMILY}"
  echo "N_JOBS=${N_JOBS}  N_LAYER_WORKERS=${N_LAYER_WORKERS:-36}"
  
  podcast_torch_setup_env

  "${PY[@]}" scripts/run/embedding/neural/podcast.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${PODCAST_FAMILY}" \
    --array-index "${IDX}" \
    --device cpu \
    --out_dir "${SWEEP_ROOT}" \
    --n_jobs "${N_JOBS}" \
    --electrode_selection_dir "${PODCAST_ELECTRODE_SELECTION_DIR}" \
    --regression_device cuda \
    --n_layer_workers "${N_LAYER_WORKERS:-36}" \
    --skip_summary \
    --no_plots \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}" \
    "${UNTRAINED_ARGS[@]}"

  echo "Stage 2 completed at $(date)"
}

podcast_torch_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  podcast_torch_setup_env
  "${PY[@]}" scripts/run/embedding/neural/podcast.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${PODCAST_FAMILY}" \
    --summary_only \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
