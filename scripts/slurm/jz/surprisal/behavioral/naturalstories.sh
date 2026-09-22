# Shared runtime for Jean Zay Natural Stories LME checkpoint sweeps.
# Source from scripts/slurm/jz/naturalstories_<family>_lme.sh (do not run directly).
#
# NS_FAMILY — registry key (gpt2, pythia, olmo2, …). Required for LME run/summary jobs;
#             optional for naturalstories_lme_prefetch_all.sh (global-index mode).
#
# Optional overrides:
#   CONFIG, NATURALSTORIES_SWEEP_ROOT, NS_DATA_ROOT
#   BENCHMARK_CACHE_ROOT, NS_FEATURE_CACHE_DIR
#   N_WORKERS    — concurrent layer-fitting worker processes for the CPU LME
#                  stage (default 20) — pymer4/rpy2 embeds one R interpreter
#                  per process, so layer-level parallelism must be process-based.
#   LAYER_STRIDE — fit every Nth layer (default 1 = all layers)
#   SAVE_WORD_PREDICTIONS — 1 (default) saves per-word predictions to
#                  *_lme_per_word.csv; set to 0 to disable
#
# NOTE: this reads NATURALSTORIES_SWEEP_ROOT specifically, NOT the generic
# SWEEP_ROOT — a stale `export SWEEP_ROOT=...` left over from another
# dataset's session would otherwise leak in and silently write output into
# the wrong directory.

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export NS_FEATURE_CACHE_DIR="${NS_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/naturalstories_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$NS_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/behavioral/naturalstories.yaml}"
SWEEP_ROOT="${NATURALSTORIES_SWEEP_ROOT:-$PWD/outputs_jeanzay/naturalstories/surprisal_sweep}"
NS_DATA_ROOT="${NS_DATA_ROOT:-$WORK/benchmark/data/naturalstories}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

# Concurrent layer-fitting worker processes (pymer4/rpy2 embeds one R
# interpreter per process, so layer-level parallelism must be process-based
# — same design as the neural LME datasets' job_nieuwland_lme_jz_common.sh /
# job_ns_fmri_lme_jz_common.sh).
N_WORKERS="${N_WORKERS:-20}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"

# On by default: save each held-out word's true value + baseline/full model
# predictions and per-word delta log-likelihood (per layer) for post-hoc DLT-
# difficulty analysis — same convention as the neural LME datasets
# (job_broderick_jz_common.sh / job_podcast_jz_common.sh). Set
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

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

ns_lme_jz_export_rpy2_env() {
  # pymer4 → rpy2 → lme4; ABI mode matches the login-node rpy2 install.
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

ns_lme_jz_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    ns_lme_jz_export_rpy2_env
    return 0
  fi
  # Always try `module load` before trusting an R_HOME inherited via
  # --export=ALL: the submitting (login-node) shell's R_HOME can be set by a
  # default profile that doesn't resolve the same way on the compute node
  # (seen in practice: R_HOME pointed at a real, executable
  # $R_HOME/bin/R on the login node, but the equivalent check silently
  # failed on the compute node) -- unconditionally skipping `module load`
  # whenever *some* R_HOME string happens to be present left the job with
  # no working R at all. `module load` itself is cheap/idempotent, so
  # trying it first even when R_HOME looks present is strictly safer.
  if command -v module >/dev/null 2>&1; then
    module load r/4.5.2 2>/dev/null || module load r 2>/dev/null || module load R 2>/dev/null || true
  fi
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    ns_lme_jz_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its bin/R
  # actually usable *on this node* — re-verify with command -v after adding
  # to PATH rather than trusting the -x test and returning unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      ns_lme_jz_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

ns_lme_jz_check_data() {
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


ns_lme_jz_setup_base_env() {
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
    outputs_jeanzay/logs
  if [[ -n "${NS_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${NS_FAMILY}"
  fi
}

ns_lme_jz_setup_env() {
  ns_lme_jz_setup_base_env
  ns_lme_jz_load_r || exit 1
}

ns_lme_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${NS_FAMILY}"
}

ns_lme_jz_run_array_task() {
  : "${NS_FAMILY:?Set NS_FAMILY before calling ns_lme_jz_run_array_task}"
  local _n
  _n="$(ns_lme_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${NS_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${NS_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "NS_DATA_ROOT=${NS_DATA_ROOT}"
  echo "NS_FEATURE_CACHE_DIR=${NS_FEATURE_CACHE_DIR:-unset}"
  echo "R_HOME=${R_HOME:-unset}"

  ns_lme_jz_check_data || exit 1
  ns_lme_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${NS_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/behavioral/naturalstories_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NS_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    "${LAST_LAYER_ONLY_FLAG[@]}" \
    --out_dir "${SWEEP_ROOT}" \
    --no_plots \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}"

  echo "Job completed at $(date)"
}

# Stage 2 — CPU: LME fitting only (surprisal read from cache, no GPU / no model load).
# Requires Stage 1 (naturalstories_lme_prefetch_all.sh) to have completed so that
# surprisal CSVs are present in $NS_FEATURE_CACHE_DIR.
# Already-converged checkpoints are skipped automatically by the Python sweep script.
ns_lme_jz_run_array_task_cpu() {
  : "${NS_FAMILY:?Set NS_FAMILY before calling ns_lme_jz_run_array_task_cpu}"
  local _n
  _n="$(ns_lme_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${NS_FAMILY}; exiting."
    exit 0
  fi

  echo "Stage 2 — LME fitting (CPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${NS_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "NS_DATA_ROOT=${NS_DATA_ROOT}"
  echo "NS_FEATURE_CACHE_DIR=${NS_FEATURE_CACHE_DIR:-unset}"
  echo "R_HOME=${R_HOME:-unset}"

  ns_lme_jz_check_data || exit 1
  ns_lme_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${NS_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/behavioral/naturalstories_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NS_FAMILY}" \
    --array-index "${IDX}" \
    --device cpu \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    "${LAST_LAYER_ONLY_FLAG[@]}" \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}"

  echo "Stage 2 completed at $(date)"
}

# Stage 1 — GPU: compute and cache surprisal for one array task (one checkpoint).
# Cache hit = model never loaded; cache miss = forward pass + write + free GPU.
# Call from naturalstories_lme_prefetch_all.sh (global index) or per-family prefetch scripts.
#
# Set NS_PREFETCH_INDEX to override the global index (default: SLURM_ARRAY_TASK_ID).
ns_lme_jz_prefetch_surprisal() {
  IDX="${NS_PREFETCH_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Stage 1 — NS surprisal prefetch (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_task=${IDX}  family=${NS_FAMILY:-all}"
  echo "NS_DATA_ROOT=${NS_DATA_ROOT}"
  echo "NS_FEATURE_CACHE_DIR=${NS_FEATURE_CACHE_DIR}"

  ns_lme_jz_setup_base_env

  if [[ -n "${NS_FAMILY:-}" ]]; then
    # Per-family mode (mirrors podcast_jz_prefetch_activations)
    "${PY[@]}" scripts/run/surprisal/behavioral/naturalstories_extract.py \
      --config "${CONFIG}" \
      --family "${NS_FAMILY}" \
      --array-index "${IDX}" \
      --dataset-root "${NS_DATA_ROOT}" \
      --cache-dir "${NS_FEATURE_CACHE_DIR}" \
      --device cuda
  else
    # Global mode — flat index across all families
    "${PY[@]}" scripts/run/surprisal/behavioral/naturalstories_extract.py \
      --config "${CONFIG}" \
      --global-index "${IDX}" \
      --dataset-root "${NS_DATA_ROOT}" \
      --cache-dir "${NS_FEATURE_CACHE_DIR}" \
      --device cuda
  fi

  echo "Stage 1 completed at $(date)"
}

ns_lme_jz_run_summary() {
  : "${NS_FAMILY:?Set NS_FAMILY before calling ns_lme_jz_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  ns_lme_jz_setup_env
  "${PY[@]}" scripts/run/surprisal/behavioral/naturalstories_fit.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${NS_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}" \
    --summary_out_dir "${NS_LME_SUMMARY_DIR:-${SWEEP_ROOT}}" \
    --perplexity_dir "${LPP_PERPLEXITY_DIR:-${WORK}/benchmark/outputs_jeanzay/lpp/perplexity}"
  echo "Summary completed at $(date)"
}
