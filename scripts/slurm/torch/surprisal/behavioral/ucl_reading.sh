# Shared runtime for Torch UCL Reading LME checkpoint sweeps.
# Source from scripts/slurm/torch/ucl_reading_<family>_lme.sh (do not run directly).
#
# UCL_FAMILY — registry key (gpt2, pythia, olmo2, …). Required for LME run/summary jobs;
#              optional for ucl_reading_lme_prefetch_all.sh (global-index mode).
#
# Optional overrides:
#   CONFIG, UCL_READING_SWEEP_ROOT, UCL_READING_DATA_ROOT
#   BENCHMARK_CACHE_ROOT, UCL_READING_FEATURE_CACHE_DIR, UCL_READING_PERPLEXITY_DIR
#   N_WORKERS    — concurrent layer-fitting worker processes for the CPU LME
#                  stage (default 20) — pymer4/rpy2 embeds one R interpreter
#                  per process, so layer-level parallelism must be process-based.
#   LAYER_STRIDE — fit every Nth layer (default 1 = all layers)
#   SAVE_WORD_PREDICTIONS — 1 (default) saves per-word predictions to
#                  *_lme_per_word.csv; set to 0 to disable
#
# NOTE: this reads UCL_READING_SWEEP_ROOT specifically, NOT the generic
# SWEEP_ROOT — a stale `export SWEEP_ROOT=...` left over from another
# dataset's session would otherwise leak in and silently write output into
# the wrong directory.

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export UCL_READING_FEATURE_CACHE_DIR="${UCL_READING_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/ucl_reading_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$UCL_READING_FEATURE_CACHE_DIR"
fi
export UCL_READING_FEATURE_CACHE_DIR="${UCL_READING_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/ucl_reading_features}"

mkdir -p outputs_torch/logs

export UCL_READING_PERPLEXITY_DIR="${UCL_READING_PERPLEXITY_DIR:-$PWD/outputs_torch/ucl_reading/perplexity}"
mkdir -p "${UCL_READING_PERPLEXITY_DIR}"

CONFIG="${CONFIG:-configs/surprisal/behavioral/ucl_reading.yaml}"
SWEEP_ROOT="${UCL_READING_SWEEP_ROOT:-$PWD/outputs_torch/ucl_reading/surprisal_sweep}"
UCL_READING_DATA_ROOT="${UCL_READING_DATA_ROOT:-$SCRATCH/Documents/benchmark/data/ucl_reading}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

# Concurrent layer-fitting worker processes (pymer4/rpy2 embeds one R
# interpreter per process, so layer-level parallelism must be process-based
# — same design as the neural LME datasets' job_nieuwland_lme_torch_common.sh /
# job_ns_fmri_lme_torch_common.sh).
N_WORKERS="${N_WORKERS:-20}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"

# On by default: save each held-out word's true value + baseline/full model
# predictions and per-word delta log-likelihood (per layer) for post-hoc DLT-
# difficulty analysis — same convention as the neural LME datasets
# (job_broderick_torch_common.sh / job_podcast_torch_common.sh). Set
# SAVE_WORD_PREDICTIONS=0 to disable.
export SAVE_WORD_PREDICTIONS="${SAVE_WORD_PREDICTIONS:-1}"
SAVE_WORD_PREDICTIONS_FLAG=()
if [[ "${SAVE_WORD_PREDICTIONS}" == "1" ]]; then
  SAVE_WORD_PREDICTIONS_FLAG=(--save_word_predictions)
else
  SAVE_WORD_PREDICTIONS_FLAG=(--no-save_word_predictions)
fi

# Fit only the deepest layer (skips --layer_stride entirely) -- cuts the
# CPU LME-fitting stage from n_layers separate R/lme4 fits down to 1.
# Set LAST_LAYER_ONLY=0 to fit every layer_stride'd layer instead.
export LAST_LAYER_ONLY="${LAST_LAYER_ONLY:-0}"
LAST_LAYER_ONLY_FLAG=()
if [[ "${LAST_LAYER_ONLY}" == "1" ]]; then
  LAST_LAYER_ONLY_FLAG=(--last_layer_only)
fi

# Off by default: set PAPER_REPLICATION=1 to switch to Kuribayashi et al.
# 2025's exact UCL/SPR baseline+full predictor set (arXiv:2502.01615) instead
# of this pipeline's own convention -- see --paper_replication's help text in
# run/surprisal/behavioral/ucl_reading_fit.py. Point UCL_READING_SWEEP_ROOT at a separate
# output dir when using this so results don't mix with the default sweep.
export PAPER_REPLICATION="${PAPER_REPLICATION:-0}"
PAPER_REPLICATION_FLAG=()
if [[ "${PAPER_REPLICATION}" == "1" ]]; then
  PAPER_REPLICATION_FLAG=(--paper_replication)
fi

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

ucl_lme_torch_export_rpy2_env() {
  # pymer4 → rpy2 → lme4; ABI mode matches the login-node rpy2 install.
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

ucl_lme_torch_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    ucl_lme_torch_export_rpy2_env
    return 0
  fi
  # Always try `module load` before trusting an R_HOME inherited via
  # --export=ALL: the submitting (login-node) shell's R_HOME can be set
  # by a default profile that doesn't resolve the same way on the
  # compute node (seen in practice: R_HOME pointed at a real, executable
  # $R_HOME/bin/R on the login node, but the equivalent check silently
  # failed on the compute node) -- unconditionally skipping `module load`
  # whenever *some* R_HOME string happens to be present left the job with
  # no working R at all. `module load` itself is cheap/idempotent, so
  # trying it first even when R_HOME looks present is strictly safer.
  if command -v module >/dev/null 2>&1; then
    module load r/4.6.1 2>/dev/null || module load r 2>/dev/null || module load R 2>/dev/null || true
  fi
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    ucl_lme_torch_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      ucl_lme_torch_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

ucl_lme_torch_check_data() {
  if [[ ! -f "${UCL_READING_DATA_ROOT}/stimuli.txt" ]]; then
    echo "ERROR: UCL Reading data not found: ${UCL_READING_DATA_ROOT}/stimuli.txt"
    return 1
  fi
  if [[ ! -f "${UCL_READING_DATA_ROOT}/selfpacedreading.RT.txt" ]]; then
    echo "ERROR: SPR RT file missing: ${UCL_READING_DATA_ROOT}/selfpacedreading.RT.txt"
    return 1
  fi
  return 0
}


ucl_lme_torch_setup_base_env() {
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
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${UCL_READING_FEATURE_CACHE_DIR}" \
    outputs_torch/logs
  if [[ -n "${UCL_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${UCL_FAMILY}"
  fi
}

ucl_lme_torch_setup_env() {
  ucl_lme_torch_setup_base_env
  ucl_lme_torch_load_r || exit 1
}

ucl_lme_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${UCL_FAMILY}"
}

ucl_lme_torch_run_array_task() {
  : "${UCL_FAMILY:?Set UCL_FAMILY before calling ucl_lme_torch_run_array_task}"
  local _n
  _n="$(ucl_lme_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${UCL_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${UCL_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "UCL_READING_DATA_ROOT=${UCL_READING_DATA_ROOT}"
  echo "UCL_READING_FEATURE_CACHE_DIR=${UCL_READING_FEATURE_CACHE_DIR:-unset}"
  echo "R_HOME=${R_HOME:-unset}"

  ucl_lme_torch_check_data || exit 1
  ucl_lme_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${UCL_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/behavioral/ucl_reading_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${UCL_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --no_plots

  echo "Job completed at $(date)"
}

# Stage 2 — CPU: LME fitting only (surprisal read from cache, no GPU / no model load).
# Requires Stage 1 (ucl_reading_lme_prefetch_all.sh) to have completed so that
# surprisal CSVs are present in $UCL_READING_FEATURE_CACHE_DIR.
# Already-converged checkpoints are skipped automatically by the Python sweep script.
ucl_lme_torch_run_array_task_cpu() {
  : "${UCL_FAMILY:?Set UCL_FAMILY before calling ucl_lme_torch_run_array_task_cpu}"
  local _n
  _n="$(ucl_lme_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${UCL_FAMILY}; exiting."
    exit 0
  fi

  echo "Stage 2 — LME fitting (CPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${UCL_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "UCL_READING_DATA_ROOT=${UCL_READING_DATA_ROOT}"
  echo "UCL_READING_FEATURE_CACHE_DIR=${UCL_READING_FEATURE_CACHE_DIR:-unset}"
  echo "R_HOME=${R_HOME:-unset}"

  ucl_lme_torch_check_data || exit 1
  ucl_lme_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${UCL_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/behavioral/ucl_reading_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${UCL_FAMILY}" \
    --array-index "${IDX}" \
    --device cpu \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    "${LAST_LAYER_ONLY_FLAG[@]}" \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}" \
    "${PAPER_REPLICATION_FLAG[@]}"

  echo "Stage 2 completed at $(date)"
}

# Stage 1 — GPU: compute and cache surprisal for one array task (one checkpoint).
# Cache hit = model never loaded; cache miss = forward pass + write + free GPU.
# Call from ucl_reading_lme_prefetch_all.sh (global index) or per-family prefetch scripts.
#
# Set UCL_PREFETCH_INDEX to override the global index (default: SLURM_ARRAY_TASK_ID).
ucl_lme_torch_prefetch_surprisal() {
  IDX="${UCL_PREFETCH_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Stage 1 — UCL Reading surprisal prefetch (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_task=${IDX}  family=${UCL_FAMILY:-all}"
  echo "UCL_READING_DATA_ROOT=${UCL_READING_DATA_ROOT}"
  echo "UCL_READING_FEATURE_CACHE_DIR=${UCL_READING_FEATURE_CACHE_DIR}"
  echo "UCL_READING_PERPLEXITY_DIR=${UCL_READING_PERPLEXITY_DIR}"

  ucl_lme_torch_setup_base_env

  if [[ -n "${UCL_FAMILY:-}" ]]; then
    # Per-family mode (mirrors podcast_torch_prefetch_activations)
    "${PY[@]}" scripts/run/surprisal/behavioral/ucl_reading_extract.py \
      --config "${CONFIG}" \
      --family "${UCL_FAMILY}" \
      --array-index "${IDX}" \
      --dataset-root "${UCL_READING_DATA_ROOT}" \
      --cache-dir "${UCL_READING_FEATURE_CACHE_DIR}" \
      --perplexity-dir "${UCL_READING_PERPLEXITY_DIR}" \
      --device cuda
  else
    # Global mode — flat index across all families
    "${PY[@]}" scripts/run/surprisal/behavioral/ucl_reading_extract.py \
      --config "${CONFIG}" \
      --global-index "${IDX}" \
      --dataset-root "${UCL_READING_DATA_ROOT}" \
      --cache-dir "${UCL_READING_FEATURE_CACHE_DIR}" \
      --perplexity-dir "${UCL_READING_PERPLEXITY_DIR}" \
      --device cuda
  fi

  echo "Stage 1 completed at $(date)"
}

ucl_lme_torch_run_summary() {
  : "${UCL_FAMILY:?Set UCL_FAMILY before calling ucl_lme_torch_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  ucl_lme_torch_setup_env
  "${PY[@]}" scripts/run/surprisal/behavioral/ucl_reading_fit.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${UCL_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}" \
    --summary_out_dir "${UCL_READING_LME_SUMMARY_DIR:-${SWEEP_ROOT}}"
  echo "Summary completed at $(date)"
}
