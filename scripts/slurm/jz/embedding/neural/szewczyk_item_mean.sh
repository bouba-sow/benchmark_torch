# Shared runtime for the single-study Szewczyk N400 item-mean embedding fit (LLM forward pass for
# the study's critical words + Himalaya GPU ridge on the subject-averaged N400). Source from
# scripts/submit.py (see stages.yaml); do not run directly.
#
# Required before source:  SZEWCZYK_FAMILY, SZEWCZYK_STUDY (f2007 | h2019)
# Optional overrides:      SZEWCZYK_ITEM_MEAN_ROOT (output root), SZEWCZYK_FEATURE_CACHE_DIR

: "${SZEWCZYK_FAMILY:?Set SZEWCZYK_FAMILY before sourcing szewczyk_item_mean.sh}"
: "${SZEWCZYK_STUDY:?Set SZEWCZYK_STUDY (f2007|h2019)}"

set -euo pipefail
# Fit only the final layer (2026-10-12; caches keep every layer). EMBED_LAST_LAYER_ONLY=0 for the full sweep.
export EMBED_LAST_LAYER_ONLY="${EMBED_LAST_LAYER_ONLY:-1}"

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT" || exit 1

if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
fi
export SZEWCZYK_FEATURE_CACHE_DIR="${SZEWCZYK_FEATURE_CACHE_DIR:-${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}/szewczyk_features}"
OUT_ROOT="${SZEWCZYK_ITEM_MEAN_ROOT:-$PWD/outputs_jeanzay/szewczyk_${SZEWCZYK_STUDY}/embedding_sweep}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p outputs_jeanzay/logs "$OUT_ROOT/${SZEWCZYK_FAMILY}" "$SZEWCZYK_FEATURE_CACHE_DIR"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

szewczyk_item_mean_jz_run_array_task() {
  local _n
  _n="$("${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${SZEWCZYK_FAMILY}")"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${SZEWCZYK_FAMILY}; exiting."
    exit 0
  fi
  echo "Job started at $(date)  study=${SZEWCZYK_STUDY} family=${SZEWCZYK_FAMILY} index=${IDX} node=$(hostname)"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${SZEWCZYK_FAMILY}" --index "${IDX}"
  "${PY[@]}" scripts/run/embedding/neural/szewczyk_item_mean.py --study "${SZEWCZYK_STUDY}" \
    --from-registry --family "${SZEWCZYK_FAMILY}" --array-index "${IDX}" \
    --out_dir "${OUT_ROOT}" --feature_cache_dir "${SZEWCZYK_FEATURE_CACHE_DIR}" --device cuda
  echo "Job completed at $(date)"
}
