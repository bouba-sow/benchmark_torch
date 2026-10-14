# Shared runtime for the Nieuwland paper-N400 item-mean embedding fit (Himalaya GPU ridge on
# cached all-layer features; no LLM is loaded). Source from scripts/submit.py or
# scripts/slurm/run_pool.py (see stages.yaml); do not run directly.
#
# Required before source:  NIEUWLAND_FAMILY
# Optional overrides:      NIEUWLAND_PAPER_EMB_ROOT (output root), NIEUWLAND_FEATURE_CACHE_DIR

: "${NIEUWLAND_FAMILY:?Set NIEUWLAND_FAMILY before sourcing nieuwland_paper.sh}"

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
export NIEUWLAND_FEATURE_CACHE_DIR="${NIEUWLAND_FEATURE_CACHE_DIR:-${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}/nieuwland_features}"
OUT_ROOT="${NIEUWLAND_PAPER_EMB_ROOT:-$PWD/outputs_jeanzay/nieuwland/embedding_sweep_paper_n400}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p outputs_jeanzay/logs "$OUT_ROOT/${NIEUWLAND_FAMILY}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

nieuwland_paper_jz_run_array_task() {
  local _n
  _n="$("${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${NIEUWLAND_FAMILY}")"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${NIEUWLAND_FAMILY}; exiting."
    exit 0
  fi
  echo "Job started at $(date)  family=${NIEUWLAND_FAMILY} index=${IDX} node=$(hostname)"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${NIEUWLAND_FAMILY}" --index "${IDX}"
  "${PY[@]}" scripts/run/embedding/neural/nieuwland_paper_item_mean.py \
    --from-registry --family "${NIEUWLAND_FAMILY}" --array-index "${IDX}" \
    --out_dir "${OUT_ROOT}" --feature_cache_dir "${NIEUWLAND_FEATURE_CACHE_DIR}" --device cuda
  echo "Job completed at $(date)"
}
