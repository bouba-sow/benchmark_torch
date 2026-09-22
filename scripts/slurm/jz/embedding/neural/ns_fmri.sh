# Shared runtime for Jean Zay Natural Stories fMRI per-family checkpoint sweeps.
# Source from scripts/slurm/jz/ns_fmri_<family>.sh
# Do not run directly.
#
# Required before source:
#   NS_FMRI_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, SWEEP_ROOT
#   NS_FMRI_DATA_ROOT          — default: $BENCHMARK_ROOT/data/naturalstories_fmri/osfstorage
#   NS_FMRI_FEATURE_CACHE_DIR  — default: $BENCHMARK_CACHE_ROOT/ns_fmri_features
#   N_JOBS                     — joblib workers for CPU layer regression (default 40)
#   N_LAYER_WORKERS            — CUDA stream workers for GPU path (default 8)
#
# Two-stage pipeline (same as ZuCo):
#   Stage 1 — GPU  (ns_fmri_prefetch_<family>.sh)
#     Runs the LLM forward pass for all 8 stories and all checkpoints.
#     Writes .npz files to $NS_FMRI_FEATURE_CACHE_DIR.  Cache hit = no GPU used.
#   Stage 2 — CPU  (ns_fmri_<family>_cpu.sh)
#     Reads cached embeddings; computes HRF design matrices; runs per-subject
#     ridge regression (joblib, n_jobs workers).  No GPU needed.
#   GPU all-in-one  (ns_fmri_<family>_gpu.sh)
#     Prefetch + himalaya GPU ridge in one H100 job.  No MPS needed.

: "${NS_FMRI_FAMILY:?Set NS_FMRI_FAMILY before sourcing job_ns_fmri_jz_common.sh}"

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

if [[ -n "${SCRATCH:-}" ]]; then
  source "${_COMMON_DIR}/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT"
fi

# Ensure NS fMRI vars are always set, even when jz_cache_env.sh was sourced.
export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
export NS_FMRI_FEATURE_CACHE_DIR="${NS_FMRI_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/ns_fmri_features}"
export NS_FMRI_DATA_ROOT="${NS_FMRI_DATA_ROOT:-$BENCHMARK_ROOT/data/naturalstories_fmri/osfstorage}"
mkdir -p "$NS_FMRI_FEATURE_CACHE_DIR"

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/embedding/neural/ns_fmri.yaml}"
SWEEP_ROOT="${SWEEP_ROOT:-$PWD/outputs_jeanzay/ns_fmri/embedding_sweep}"
N_JOBS="${N_JOBS:-40}"
N_LAYER_WORKERS="${N_LAYER_WORKERS:-8}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

ns_fmri_jz_check_data() {
  local _missing=0
  if [[ ! -f "${NS_FMRI_DATA_ROOT}/LANG_MD_ts_X.csv" ]]; then
    echo "ERROR: NS fMRI X table not found: ${NS_FMRI_DATA_ROOT}/LANG_MD_ts_X.csv"
    _missing=1
  fi
  if [[ ! -f "${NS_FMRI_DATA_ROOT}/LANG_MD_y_train.csv" ]]; then
    echo "ERROR: NS fMRI Y table not found: ${NS_FMRI_DATA_ROOT}/LANG_MD_y_train.csv"
    _missing=1
  fi
  return "$_missing"
}

ns_fmri_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export SAFETENSORS_FAST_GPU=1   # memory-map safetensors directly to GPU (no CPU staging)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  local _ncpu="${SLURM_CPUS_PER_TASK:-1}"
  export OMP_NUM_THREADS="${_ncpu}"
  export MKL_NUM_THREADS="${_ncpu}"
  export OPENBLAS_NUM_THREADS="${_ncpu}"
  export NUMEXPR_NUM_THREADS="${_ncpu}"

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORK/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  export BENCHMARK_ROOT="${BENCHMARK_ROOT}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${NS_FMRI_FEATURE_CACHE_DIR}" \
    "${SWEEP_ROOT}/${NS_FMRI_FAMILY}" outputs_jeanzay/logs
}

ns_fmri_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${NS_FMRI_FAMILY}"
}

# Stage 1 — GPU: compute and cache activations for all 8 stories (one model per array task).
ns_fmri_jz_prefetch_activations() {
  local _n
  _n="$(ns_fmri_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] — exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Stage 1 — activations only (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_task=${IDX}  family=${NS_FMRI_FAMILY}"
  echo "NS_FMRI_FEATURE_CACHE_DIR=${NS_FMRI_FEATURE_CACHE_DIR}"

  ns_fmri_jz_check_data || exit 1
  ns_fmri_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${NS_FMRI_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/embedding/neural/ns_fmri.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NS_FMRI_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --activations_only \
    --feature_cache_dir "${NS_FMRI_FEATURE_CACHE_DIR}" \
    --skip_summary

  echo "Stage 1 completed at $(date)"
}

# Stage 2 — CPU: HRF convolution + ridge regression (reads cached embeddings).
ns_fmri_jz_run_regression() {
  local _n
  _n="$(ns_fmri_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] — exiting."
    exit 0
  fi

  echo "Stage 2 — regression only (CPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_task=${IDX}  family=${NS_FMRI_FAMILY}"
  echo "CPUS=${SLURM_CPUS_PER_TASK:-?}  N_JOBS=${N_JOBS}"

  ns_fmri_jz_check_data || exit 1
  ns_fmri_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${NS_FMRI_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/embedding/neural/ns_fmri.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NS_FMRI_FAMILY}" \
    --array-index "${IDX}" \
    --device cpu \
    --out_dir "${SWEEP_ROOT}" \
    --feature_cache_dir "${NS_FMRI_FEATURE_CACHE_DIR}" \
    --n_jobs "${N_JOBS}" \
    --skip_summary \
    --no_plots

  echo "Stage 2 completed at $(date)"
}

# GPU all-in-one — prefetch activations then run himalaya GPU ridge.
ns_fmri_jz_run_gpu() {
  local _n
  _n="$(ns_fmri_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] — exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "GPU — prefetch + himalaya regression (H100)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_task=${IDX}  family=${NS_FMRI_FAMILY}"
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "N_LAYER_WORKERS=${N_LAYER_WORKERS}"

  ns_fmri_jz_check_data || exit 1
  ns_fmri_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${NS_FMRI_FAMILY}" --index "${IDX}"

  # Stage 1: extract and cache LLM activations (no-op on cache hit).
  "${PY[@]}" scripts/run/embedding/neural/ns_fmri.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NS_FMRI_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --activations_only \
    --feature_cache_dir "${NS_FMRI_FEATURE_CACHE_DIR}" \
    --skip_summary

  # Stage 2: himalaya GPU ridge (reads cached activations).
  "${PY[@]}" scripts/run/embedding/neural/ns_fmri.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NS_FMRI_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --feature_cache_dir "${NS_FMRI_FEATURE_CACHE_DIR}" \
    --n_layer_workers "${N_LAYER_WORKERS}" \
    --skip_summary \
    --no_plots

  echo "GPU job completed at $(date)"
}

ns_fmri_jz_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  ns_fmri_jz_setup_env
  "${PY[@]}" scripts/run/embedding/neural/ns_fmri.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NS_FMRI_FAMILY}" \
    --summary_only \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
