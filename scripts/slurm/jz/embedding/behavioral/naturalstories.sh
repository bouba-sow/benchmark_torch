# Shared runtime for Jean Zay Natural Stories GPU/himalaya embedding
# checkpoint sweeps. Source from
# scripts/submit.py (see scripts/slurm/stages.yaml) (do not run
# directly).
#
# Single-stage, GPU-only: unlike the LME pipeline (R/pymer4 needs CPU on
# Jean Zay, so extraction/fit are split into separate GPU/CPU jobs), ridge
# regression here runs on GPU via himalaya
# (cv_ridge_score_multioutput_gpu, one CUDA stream per layer — see
# neuroai/benchmarks/natural_stories_embed_gpu.py) — embedding extraction
# and the ridge fit both happen in this one job. No CPU variant needed or
# provided.
#
# Chosen ridge alpha is logged per layer in the job output. Every layer's
# out-of-fold predictions are saved by default (save_word_predictions in
# configs/embedding/behavioral/naturalstories.yaml).
#
# Required before source:
#   NS_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, SWEEP_ROOT, NS_DATA_ROOT
#   BENCHMARK_CACHE_ROOT  — default: $SCRATCH/.cache on Jean Zay
#   NS_FEATURE_CACHE_DIR  — default: $BENCHMARK_CACHE_ROOT/naturalstories_features

: "${NS_FAMILY:?Set NS_FAMILY before sourcing job_naturalstories_embed_gpu_jz_common.sh}"

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
  export NS_FEATURE_CACHE_DIR="${NS_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/naturalstories_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export NS_FEATURE_CACHE_DIR="${NS_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/naturalstories_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$NS_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/embedding/behavioral/naturalstories.yaml}"
SWEEP_ROOT="${SWEEP_ROOT:-$PWD/outputs_jeanzay/naturalstories/embedding_sweep}"
NS_DATA_ROOT="${NS_DATA_ROOT:-$WORK/benchmark/data/naturalstories}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

ns_embed_gpu_jz_check_data() {
  if [[ ! -f "${NS_DATA_ROOT}/words.tsv" ]]; then
    echo "ERROR: Natural Stories data not found: ${NS_DATA_ROOT}/words.tsv"
    return 1
  fi
  if [[ ! -f "${NS_DATA_ROOT}/naturalstories_RTS/processed_RTs.tsv" ]]; then
    echo "ERROR: SPR RT file missing: ${NS_DATA_ROOT}/naturalstories_RTS/processed_RTs.tsv"
    return 1
  fi
  return 0
}

ns_embed_gpu_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export SAFETENSORS_FAST_GPU=1   # memory-map safetensors directly to GPU (no CPU staging)
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORK/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${NS_FEATURE_CACHE_DIR}" \
    "${SWEEP_ROOT}/${NS_FAMILY}" outputs_jeanzay/logs
}

ns_embed_gpu_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${NS_FAMILY}"
}

ns_embed_gpu_jz_run_array_task() {
  local _n
  _n="$(ns_embed_gpu_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${NS_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Natural Stories embed ridge sweep (GPU/himalaya)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${NS_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "NS_DATA_ROOT=${NS_DATA_ROOT}"
  echo "NS_FEATURE_CACHE_DIR=${NS_FEATURE_CACHE_DIR}"

  ns_embed_gpu_jz_check_data || exit 1
  ns_embed_gpu_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${NS_FAMILY}" --index "${IDX}"

  # A saved ridge score does not imply that its feature cache still exists.
  # Repair missing activations before the fit's existing-score shortcut.
  if [[ "${NS_ENSURE_ACTIVATIONS:-0}" == "1" ]]; then
    "${PY[@]}" scripts/run/embedding/behavioral/naturalstories.py \
      --config "${CONFIG}" --from-registry --family "${NS_FAMILY}" \
      --array-index "${IDX}" --activations_only
  fi

  "${PY[@]}" scripts/run/embedding/behavioral/naturalstories.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NS_FAMILY}" \
    --array-index "${IDX}" \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots

  echo "Job completed at $(date)"
}

ns_embed_gpu_jz_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  ns_embed_gpu_jz_setup_env
  "${PY[@]}" scripts/run/embedding/behavioral/naturalstories.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${NS_FAMILY}" \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
