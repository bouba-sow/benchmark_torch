# Shared runtime for Jean Zay Nieuwland N400 surprisal extraction jobs (GPU
# stage). Source from scripts/submit.py (see scripts/slurm/stages.yaml)
# (do not run directly).
#
# GPU-only purpose: populate the per-layer logit-lens trial surprisal cache
# (noun + previous-word spillover control, ~160 items) consumed by the
# CPU-only LME fitting job (job_nieuwland_lme_jz_common.sh /
# scripts/run/surprisal/neural/nieuwland_fit.py). Split into its own job because R
# (pymer4/lme4, needed for the LME fit) is only available on CPU partitions
# on Jean Zay — extraction (wants a GPU for larger models) and fitting
# (CPU-only) can't share a job, same as Podcast's pooled LME.
#
# EEG-FREE and much lighter than Podcast's GPU stage: uses load_all_items
# (pools events.tsv across subjects), never touches raw EEG — ~160 short
# trial items, two forward passes each, not a full narrative's
# sliding-window chunks. ONE task per model (array), no per-layer/per-lag
# parallelism needed here at all.
#
# Required before source:
#   NIEUWLAND_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, NIEUWLAND_LME_FEATURE_CACHE_DIR
#
# Idempotent: logit_lens_surprisal_trials checks the cache BEFORE loading the
# model, so a re-run with the same cache dir is a fast no-op per model.

: "${NIEUWLAND_FAMILY:?Set NIEUWLAND_FAMILY before sourcing job_nieuwland_surprisal_jz_common.sh}"

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
  export NIEUWLAND_LME_FEATURE_CACHE_DIR="${NIEUWLAND_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export NIEUWLAND_LME_FEATURE_CACHE_DIR="${NIEUWLAND_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$NIEUWLAND_LME_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/neural/nieuwland.yaml}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

nieuwland_surprisal_jz_check_data() {
  local bids_root="${NIEUWLAND_BIDS_ROOT:-${BENCHMARK_ROOT}/data/nieuwland}"
  if [[ ! -f "${bids_root}/participants.tsv" ]]; then
    echo "ERROR: Nieuwland participants.tsv not found under ${bids_root}"
    return 1
  fi
  return 0
}

nieuwland_surprisal_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${NIEUWLAND_LME_FEATURE_CACHE_DIR}" outputs_jeanzay/logs
}

nieuwland_surprisal_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${NIEUWLAND_FAMILY}"
}

nieuwland_surprisal_jz_run_array_task() {
  local _n
  _n="$(nieuwland_surprisal_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${NIEUWLAND_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${NIEUWLAND_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "NIEUWLAND_LME_FEATURE_CACHE_DIR=${NIEUWLAND_LME_FEATURE_CACHE_DIR}"

  nieuwland_surprisal_jz_check_data || exit 1
  nieuwland_surprisal_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${NIEUWLAND_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/nieuwland_extract.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NIEUWLAND_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --feature_cache_dir "${NIEUWLAND_LME_FEATURE_CACHE_DIR}"

  echo "Job completed at $(date)"
}
