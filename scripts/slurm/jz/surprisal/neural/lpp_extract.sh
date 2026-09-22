# Shared runtime for Jean Zay Le Petit Prince fMRI surprisal extraction jobs
# (GPU stage). Source from
# scripts/submit.py (see scripts/slurm/stages.yaml) (do not run directly).
#
# GPU-only purpose: populate the per-layer logit-lens surprisal cache (one
# forward pass per LPP section, 9 sections) consumed by the CPU-only LME
# fitting job (job_lpp_lme_jz_common.sh / scripts/run/surprisal/neural/lpp_fit.py).
# Split into its own job because R (pymer4/lme4, needed for the LME fit) is
# only available on CPU partitions on Jean Zay — extraction (wants a GPU for
# larger models) and fitting (CPU-only) can't share a job, same as every
# other pooled LME here.
#
# ONE task per (model, checkpoint) — same registry / array-indexing helper
# (scripts/slurm/family_tasks.py) as the Tuckute pooled LME.
#
# Required before source:
#   LPP_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, LPP_LME_FEATURE_CACHE_DIR
#
# Idempotent: logit_lens_surprisal_all_layers checks the cache BEFORE
# loading the model, so a re-run with the same cache dir is a fast no-op per
# model.

: "${LPP_FAMILY:?Set LPP_FAMILY before sourcing job_lpp_surprisal_jz_common.sh}"

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
  export LPP_LME_FEATURE_CACHE_DIR="${LPP_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/lpp_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export LPP_LME_FEATURE_CACHE_DIR="${LPP_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/lpp_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$LPP_LME_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/neural/lpp.yaml}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

lpp_surprisal_jz_check_data() {
  local _root="${BENCHMARK_ROOT}/data/lpp_fmri"
  if [[ ! -f "${_root}/annotation/EN/lppEN_word_information.csv" ]]; then
    echo "ERROR: LPP data not found — missing ${_root}/annotation/EN/lppEN_word_information.csv"
    return 1
  fi
  return 0
}

lpp_surprisal_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${LPP_LME_FEATURE_CACHE_DIR}" outputs_jeanzay/logs
}

lpp_surprisal_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${LPP_FAMILY}"
}

lpp_surprisal_jz_run_array_task() {
  local _n
  _n="$(lpp_surprisal_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${LPP_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${LPP_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "LPP_LME_FEATURE_CACHE_DIR=${LPP_LME_FEATURE_CACHE_DIR}"

  lpp_surprisal_jz_check_data || exit 1
  lpp_surprisal_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${LPP_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/lpp_extract.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${LPP_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --feature_cache_dir "${LPP_LME_FEATURE_CACHE_DIR}"

  echo "Job completed at $(date)"
}
