# Shared runtime for Torch Fedorenko2016 iEEG surprisal extraction jobs
# (GPU stage). Source from
# scripts/submit.py (see scripts/slurm/stages.yaml) (do not run
# directly).
#
# GPU-only purpose: populate the per-layer logit-lens fulltext surprisal
# cache (one forward pass per sentence, 52 sentences × 8 words) consumed by
# the CPU-only LME fitting job (job_fedorenko2016_lme_torch_common.sh /
# scripts/run/surprisal/neural/fedorenko2016_fit.py). Split into its own job because R
# (pymer4/lme4, needed for the LME fit) is only available on CPU partitions
# on Torch — extraction (wants a GPU for larger models) and fitting
# (CPU-only) can't share a job, same as the N400 datasets' pooled LMEs.
#
# Much lighter than a narrative-scale GPU stage: 52 sentences, one forward
# pass each — not sliding-window chunks over a long document. ONE task per
# model (array), no per-layer/per-lag parallelism needed here at all.
#
# Required before source:
#   FEDORENKO2016_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, FEDORENKO2016_LME_FEATURE_CACHE_DIR
#
# Idempotent: logit_lens_surprisal_all_layers checks the cache BEFORE
# loading the model, so a re-run with the same cache dir is a fast no-op per
# model.

: "${FEDORENKO2016_FAMILY:?Set FEDORENKO2016_FAMILY before sourcing job_fedorenko2016_surprisal_torch_common.sh}"

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
  export FEDORENKO2016_LME_FEATURE_CACHE_DIR="${FEDORENKO2016_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/fedorenko2016_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export FEDORENKO2016_LME_FEATURE_CACHE_DIR="${FEDORENKO2016_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/fedorenko2016_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$FEDORENKO2016_LME_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/surprisal/neural/fedorenko2016.yaml}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

fedorenko2016_surprisal_torch_check_data() {
  if [[ ! -f "${BENCHMARK_ROOT}/data/fedorenko2016/stimuli.csv" ]]; then
    echo "ERROR: Fedorenko2016 stimuli.csv not found under ${BENCHMARK_ROOT}/data/fedorenko2016/"
    return 1
  fi
  return 0
}

fedorenko2016_surprisal_torch_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${FEDORENKO2016_LME_FEATURE_CACHE_DIR}" outputs_torch/logs
}

fedorenko2016_surprisal_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${FEDORENKO2016_FAMILY}"
}

fedorenko2016_surprisal_torch_run_array_task() {
  local _n
  _n="$(fedorenko2016_surprisal_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${FEDORENKO2016_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${FEDORENKO2016_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "FEDORENKO2016_LME_FEATURE_CACHE_DIR=${FEDORENKO2016_LME_FEATURE_CACHE_DIR}"

  fedorenko2016_surprisal_torch_check_data || exit 1
  fedorenko2016_surprisal_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${FEDORENKO2016_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/fedorenko2016_extract.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${FEDORENKO2016_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --feature_cache_dir "${FEDORENKO2016_LME_FEATURE_CACHE_DIR}"

  echo "Job completed at $(date)"
}
