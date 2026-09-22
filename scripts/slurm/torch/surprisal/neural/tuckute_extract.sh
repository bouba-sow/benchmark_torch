# Shared runtime for Torch Tuckute fMRI surprisal extraction jobs
# (GPU stage). Source from
# scripts/submit.py (see scripts/slurm/stages.yaml) (do not run
# directly).
#
# GPU-only purpose: populate the per-layer last-word logit-lens surprisal
# cache (one forward pass per sentence, up to 2,000 sentences, 6 words each)
# consumed by the CPU-only LME fitting job (job_tuckute_lme_torch_common.sh /
# scripts/run/surprisal/neural/tuckute_fit.py). Split into its own job because R
# (pymer4/lme4, needed for the LME fit) is only available on CPU partitions
# on Torch — extraction (wants a GPU for larger models) and fitting
# (CPU-only) can't share a job, same as every other pooled LME here.
#
# Much lighter than a narrative-scale GPU stage: up to 2,000 sentences, one
# forward pass each — not sliding-window chunks over a long document. ONE
# task per model (array), no per-layer/per-lag parallelism needed here.
#
# Required before source:
#   TUCKUTE_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, TUCKUTE_LME_FEATURE_CACHE_DIR
#
# Idempotent: logit_lens_surprisal_all_layers checks the cache BEFORE
# loading the model, so a re-run with the same cache dir is a fast no-op per
# model.

: "${TUCKUTE_FAMILY:?Set TUCKUTE_FAMILY before sourcing job_tuckute_surprisal_torch_common.sh}"

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
  export TUCKUTE_LME_FEATURE_CACHE_DIR="${TUCKUTE_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/tuckute_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export TUCKUTE_LME_FEATURE_CACHE_DIR="${TUCKUTE_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/tuckute_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$TUCKUTE_LME_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/surprisal/neural/tuckute.yaml}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

tuckute_surprisal_torch_check_data() {
  # TuckuteDataset reads brain-lang-data_participant_20230728.csv from EITHER
  # an extracted data/ subdirectory OR data.tar (see
  # neuroai/datasets/tuckute.py::_read_csv_from_dir_or_tar) — this OSF
  # deposit ships only the tar, never extracted, so both must be accepted.
  local _root="${BENCHMARK_ROOT}/data/tuckute_fmri/osfstorage"
  if [[ ! -f "${_root}/data/brain-lang-data_participant_20230728.csv" && ! -f "${_root}/data.tar" ]]; then
    echo "ERROR: Tuckute data not found — need either ${_root}/data/brain-lang-data_participant_20230728.csv or ${_root}/data.tar"
    return 1
  fi
  return 0
}

tuckute_surprisal_torch_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${TUCKUTE_LME_FEATURE_CACHE_DIR}" outputs_torch/logs
}

tuckute_surprisal_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${TUCKUTE_FAMILY}"
}

tuckute_surprisal_torch_run_array_task() {
  local _n
  _n="$(tuckute_surprisal_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${TUCKUTE_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${TUCKUTE_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "TUCKUTE_LME_FEATURE_CACHE_DIR=${TUCKUTE_LME_FEATURE_CACHE_DIR}"

  tuckute_surprisal_torch_check_data || exit 1
  tuckute_surprisal_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${TUCKUTE_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/tuckute_extract.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${TUCKUTE_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --feature_cache_dir "${TUCKUTE_LME_FEATURE_CACHE_DIR}"

  echo "Job completed at $(date)"
}
