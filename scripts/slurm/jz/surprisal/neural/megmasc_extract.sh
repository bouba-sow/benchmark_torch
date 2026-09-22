# Shared runtime for Jean Zay MEG-MASC MEG surprisal extraction jobs (GPU
# stage). Source from
# scripts/submit.py (see scripts/slurm/stages.yaml) (do not run
# directly).
#
# GPU-only purpose: populate the per-layer logit-lens fulltext surprisal
# cache (one forward pass per audiobook story, 36 stories) consumed by the
# CPU-only LME fitting job (job_megmasc_lme_jz_common.sh /
# scripts/run/surprisal/neural/megmasc_fit.py). Split into its own job because R
# (pymer4/lme4, needed for the LME fit) is only available on CPU partitions
# on Jean Zay — extraction (wants a GPU for larger models) and fitting
# (CPU-only) can't share a job, same as every other pooled LME here.
#
# Much lighter than a narrative-scale GPU stage: 36 stories, one forward
# pass each. ONE task per model (array), no per-layer parallelism needed
# here.
#
# Required before source:
#   MEGMASC_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, MEGMASC_LME_FEATURE_CACHE_DIR
#
# Idempotent: logit_lens_surprisal_all_layers checks the cache BEFORE
# loading the model, so a re-run with the same cache dir is a fast no-op per
# model.

: "${MEGMASC_FAMILY:?Set MEGMASC_FAMILY before sourcing job_megmasc_surprisal_jz_common.sh}"

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1
export PYTHONPATH="${BENCHMARK_ROOT}/forgetting-transformer/src:${BENCHMARK_ROOT}/stickbreaking-attention${PYTHONPATH:+:$PYTHONPATH}"  # constrained-arch Lanni-ni models

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
  export MEGMASC_LME_FEATURE_CACHE_DIR="${MEGMASC_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export MEGMASC_LME_FEATURE_CACHE_DIR="${MEGMASC_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$MEGMASC_LME_FEATURE_CACHE_DIR"
fi

export MEGMASC_BIDS_ROOT="${MEGMASC_BIDS_ROOT:-${SCRATCH:-$BENCHMARK_ROOT/..}/data/gwilliams}"
mkdir -p outputs_jeanzay/logs outputs_jeanzay/megmasc/perplexity

CONFIG="${CONFIG:-configs/surprisal/neural/megmasc.yaml}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

megmasc_surprisal_jz_check_data() {
  if [[ ! -d "${MEGMASC_BIDS_ROOT:-}" ]]; then
    echo "ERROR: MEGMASC_BIDS_ROOT not found: ${MEGMASC_BIDS_ROOT:-<unset>}"
    return 1
  fi
  return 0
}

megmasc_surprisal_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${MEGMASC_LME_FEATURE_CACHE_DIR}" outputs_jeanzay/logs
}

megmasc_surprisal_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${MEGMASC_FAMILY}"
}

megmasc_surprisal_jz_run_array_task() {
  local _n
  _n="$(megmasc_surprisal_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${MEGMASC_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${MEGMASC_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "MEGMASC_LME_FEATURE_CACHE_DIR=${MEGMASC_LME_FEATURE_CACHE_DIR}"

  megmasc_surprisal_jz_check_data || exit 1
  megmasc_surprisal_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${MEGMASC_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/megmasc_extract.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${MEGMASC_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --feature_cache_dir "${MEGMASC_LME_FEATURE_CACHE_DIR}"

  echo "Job completed at $(date)"
}
