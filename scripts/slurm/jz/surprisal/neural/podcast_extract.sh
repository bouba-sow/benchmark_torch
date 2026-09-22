# Shared runtime for Jean Zay Podcast ECoG surprisal (logit-lens) checkpoint sweeps.
# Source from scripts/submit.py (see scripts/slurm/stages.yaml) (do not run directly).
#
# Required before source:
#   PODCAST_FAMILY   — registry key (gpt2, pythia, olmo2, …) — standard LLM
#                       architectures only (logit-lens scope, see
#                       neuroai/features/logit_lens.py).
#
# Optional overrides:
#   CONFIG, PODCAST_SURPRISAL_SWEEP_ROOT, PODCAST_BIDS_ROOT
#   BENCHMARK_CACHE_ROOT  — default: $SCRATCH/.cache on Jean Zay
#   PODCAST_FEATURE_CACHE_DIR — default: $BENCHMARK_CACHE_ROOT/podcast_features
#     (shared with the embeddings sweep's cache dir — logit-lens surprisal
#     uses a distinct "logitlens_all" filename tag, so the two never collide.)
#
# UNTRAINED=1 — run the untrained-baseline path instead of pretrained weights.
#   UNTRAINED=1 python scripts/submit.py podcast surprisal extract --family gpt2
#
# Single-stage only: unlike the embeddings sweep (which has an activations-only
# GPU stage + a ridge-only CPU/GPU stage split), ridge regression here needs no
# R and is far cheaper than the embeddings case (scalar predictor per layer,
# not a full hidden-state vector) — so logit-lens extraction (GPU, all layers
# in one forward pass) and himalaya GPU ridge regression run together in one
# H100 job, always with --regression_device cuda.
#
# NOTE: this reads PODCAST_SURPRISAL_SWEEP_ROOT specifically, NOT the generic
# SWEEP_ROOT or PODCAST_SWEEP_ROOT (the embeddings sweep's variable) — so a
# stale env var from another session never leaks output into the wrong tree.

: "${PODCAST_FAMILY:?Set PODCAST_FAMILY before sourcing job_podcast_surprisal_jz_common.sh}"

set -euo pipefail

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
  export PODCAST_FEATURE_CACHE_DIR="${PODCAST_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$PODCAST_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/neural/podcast_extract.yaml}"
UNTRAINED="${UNTRAINED:-0}"
UNTRAINED_SEED="${UNTRAINED_SEED:-0}"
SWEEP_ROOT="${PODCAST_SURPRISAL_SWEEP_ROOT:-$PWD/outputs_jeanzay/podcast/checkpoint_sweep_surprisal}"
PODCAST_BIDS_ROOT="${PODCAST_BIDS_ROOT:-$WORK/benchmark/data/podcast}"
PODCAST_ELECTRODE_SELECTION_DIR="${PODCAST_ELECTRODE_SELECTION_DIR:-$PWD/outputs_jeanzay/podcast/electrode_selection}"
N_JOBS="${N_JOBS:-80}"

# NOTE: --out_dir/--feature_cache_dir passed to the python script below are
# the *plain* (non-suffixed) paths -- run/embedding/neural/podcast.py itself
# appends "_untrained" when --untrained is set. Don't also suffix here, or
# the two suffixes stack (…_untrained_untrained).
UNTRAINED_ARGS=()
if [[ "${UNTRAINED}" == "1" ]]; then
  UNTRAINED_ARGS=(--untrained --untrained_seed "${UNTRAINED_SEED}")
fi

# On by default — same as the embeddings sweep: this feeds the DLT-cost
# difficulty-split analysis, which needs each electrode's best-layer
# out-of-fold predicted lag matrix (.npz per subject). For surprisal mode
# "best-layer" picks among the 12 (or however many) logit-lens layers.
# Set SAVE_WORD_PREDICTIONS=0 to disable.
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

podcast_surprisal_jz_check_data() {
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

podcast_surprisal_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export SAFETENSORS_FAST_GPU=1
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

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${PODCAST_FEATURE_CACHE_DIR}" \
    "${SWEEP_ROOT}/${PODCAST_FAMILY}" outputs_jeanzay/logs
}

podcast_surprisal_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${PODCAST_FAMILY}"
}

podcast_surprisal_jz_run_array_task() {
  local _n
  _n="$(podcast_surprisal_jz_array_size)"
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

  podcast_surprisal_jz_check_data || exit 1
  podcast_surprisal_jz_setup_env

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

podcast_surprisal_jz_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  podcast_surprisal_jz_setup_env
  "${PY[@]}" scripts/run/embedding/neural/podcast.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${PODCAST_FAMILY}" \
    --summary_only \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
