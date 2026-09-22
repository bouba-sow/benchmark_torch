# Shared runtime for Jean Zay ZuCo Reading (behavioral TRT) LME checkpoint sweeps.
# Source from scripts/slurm/jz/zuco_reading_<family>_gpu.sh (Stage 1) or
# scripts/slurm/jz/zuco_reading_<family>_lme_cpu.sh (Stage 2) — do not run directly.
#
# Two-stage design, same as UCL Reading:
#   Stage 1 (GPU, zuco_lme_jz_prefetch_gpu)       — surprisal + perplexity, cached.
#   Stage 2 (CPU, zuco_lme_jz_run_array_task_cpu) — LME fit, reads cache, needs R.
# Originally single-stage (GPU did both), but R was found to be unavailable on
# Jean Zay's H100 partition even after a clean `module load r/4.5.2` (works fine
# on the CPU partition) — split to sidestep that rather than chase the module issue.
#
# ZUCO_FAMILY — registry key (gpt2, pythia, olmo2, …). Required for LME run/summary jobs.
# ZUCO_TASK   — ZuCo reading task: SR | NR | TSR (default: NR). Selects both the
#               config file (configs/surprisal/behavioral/zuco_reading_<task>.yaml) and the
#               sweep/cache subdirectories — SR/NR/TSR never share a cache dir
#               because their (story_id, zone) ranges overlap.
#
# Optional overrides:
#   CONFIG, ZUCO_READING_SWEEP_ROOT, ZUCO_READING_DATA_ROOT, ZUCO_READING_PERPLEXITY_DIR
#   BENCHMARK_CACHE_ROOT, ZUCO_READING_FEATURE_CACHE_DIR
#   N_WORKERS    — concurrent layer-fitting worker processes for the CPU LME
#                  stage (default 20) — pymer4/rpy2 embeds one R interpreter
#                  per process, so layer-level parallelism must be process-based.
#   LAYER_STRIDE — fit every Nth layer (default 1 = all layers)
#   SAVE_WORD_PREDICTIONS — 1 (default) saves per-word predictions to
#                  *_lme_per_word.csv; set to 0 to disable
#
# NOTE: this reads ZUCO_READING_SWEEP_ROOT specifically, NOT the generic
# SWEEP_ROOT — a stale `export SWEEP_ROOT=...` left over from another
# dataset's session would otherwise leak in and silently write output into
# the wrong directory.

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
export ZUCO_TASK="${ZUCO_TASK:-NR}"
ZUCO_TASK_LOWER="$(echo "${ZUCO_TASK}" | tr '[:upper:]' '[:lower:]')"

cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT"
fi
# Per-task subdirectory — see note above on why SR/NR/TSR must not share a cache dir.
export ZUCO_READING_FEATURE_CACHE_DIR="${ZUCO_READING_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/zuco_reading_features}/${ZUCO_TASK_LOWER}"
mkdir -p "${ZUCO_READING_FEATURE_CACHE_DIR}"

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/behavioral/zuco_reading_${ZUCO_TASK_LOWER}.yaml}"
SWEEP_ROOT="${ZUCO_READING_SWEEP_ROOT:-$PWD/outputs_jeanzay/zuco_reading/surprisal_sweep/${ZUCO_TASK_LOWER}}"
ZUCO_READING_DATA_ROOT="${ZUCO_READING_DATA_ROOT:-$WORK/benchmark/data/zuco_reading/osfstorage}"
export ZUCO_READING_PERPLEXITY_DIR="${ZUCO_READING_PERPLEXITY_DIR:-$PWD/outputs_jeanzay/zuco_reading/perplexity}"
mkdir -p "${ZUCO_READING_PERPLEXITY_DIR}"

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

# Explicit layer index list (space-separated, e.g. "3 11") -- overrides
# LAYER_STRIDE/LAST_LAYER_ONLY entirely. For the cheap second pass that
# saves word predictions for only {last, best} layer per model, after a
# first pass (SAVE_WORD_PREDICTIONS=0) already identified each model's best
# layer from delta_loglik_test -- see run/surprisal/behavioral/zuco_reading_fit.py's
# --layers flag / UCLReadingLmeConfig.layers.
LAYERS_FLAG=()
if [[ -n "${ZUCO_LAYERS:-}" ]]; then
  # shellcheck disable=SC2206 # intentional word-splitting of a space-separated list
  LAYERS_FLAG=(--layers ${ZUCO_LAYERS})
fi

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

zuco_lme_jz_export_rpy2_env() {
  # pymer4 → rpy2 → lme4; ABI mode matches the login-node rpy2 install.
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

zuco_lme_jz_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    zuco_lme_jz_export_rpy2_env
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
    module load r/4.5.2 2>/dev/null || module load r 2>/dev/null || module load R 2>/dev/null || true
  fi
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    zuco_lme_jz_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      zuco_lme_jz_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

zuco_lme_jz_task_dir() {
  case "${ZUCO_TASK}" in
    SR)  echo "task1- SR" ;;
    NR)  echo "task2 - NR" ;;
    TSR) echo "task3 - TSR" ;;
    *) echo "ERROR: Unknown ZUCO_TASK=${ZUCO_TASK} (expected SR|NR|TSR)" >&2; return 1 ;;
  esac
}

zuco_lme_jz_check_data() {
  local task_dir
  task_dir="$(zuco_lme_jz_task_dir)" || return 1
  if [[ ! -d "${ZUCO_READING_DATA_ROOT}/${task_dir}/Matlab files" ]]; then
    echo "ERROR: ZuCo data not found: ${ZUCO_READING_DATA_ROOT}/${task_dir}/Matlab files"
    return 1
  fi
  return 0
}


zuco_lme_jz_setup_base_env() {
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
    "${ZUCO_READING_FEATURE_CACHE_DIR}" \
    outputs_jeanzay/logs
  if [[ -n "${ZUCO_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${ZUCO_FAMILY}"
  fi
}

zuco_lme_jz_setup_env() {
  zuco_lme_jz_setup_base_env
  zuco_lme_jz_load_r || exit 1
}

zuco_lme_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${ZUCO_FAMILY}"
}

# Stage 1 — GPU: surprisal + perplexity prefetch only (no R/LME fit — R was found
# to be unavailable/non-functional on Jean Zay's H100 partition even after a clean
# `module load r/4.5.2`, unlike the CPU partition where it works fine). Uses
# zuco_lme_jz_setup_base_env (no R load attempt).
# Already-converged/cached surprisal+perplexity are skipped without reloading the model.
zuco_lme_jz_prefetch_gpu() {
  : "${ZUCO_FAMILY:?Set ZUCO_FAMILY before calling zuco_lme_jz_prefetch_gpu}"
  local _n
  _n="$(zuco_lme_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${ZUCO_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Stage 1 — ZuCo Reading surprisal + perplexity prefetch (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${ZUCO_FAMILY}  task=${ZUCO_TASK}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "ZUCO_READING_DATA_ROOT=${ZUCO_READING_DATA_ROOT}"
  echo "ZUCO_READING_FEATURE_CACHE_DIR=${ZUCO_READING_FEATURE_CACHE_DIR:-unset}"
  echo "ZUCO_READING_PERPLEXITY_DIR=${ZUCO_READING_PERPLEXITY_DIR}"

  zuco_lme_jz_check_data || exit 1
  zuco_lme_jz_setup_base_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${ZUCO_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/behavioral/zuco_reading_extract.py \
    --config "${CONFIG}" \
    --family "${ZUCO_FAMILY}" \
    --array-index "${IDX}" \
    --mat-root "${ZUCO_READING_DATA_ROOT}" \
    --task "${ZUCO_TASK}" \
    --cache-dir "${ZUCO_READING_FEATURE_CACHE_DIR}" \
    --perplexity-dir "${ZUCO_READING_PERPLEXITY_DIR}" \
    --device cuda

  echo "Stage 1 completed at $(date)"
}

# Stage 2 — CPU: LME fitting only (surprisal read from cache, no GPU / no model
# load). Requires Stage 1 to have already written the surprisal cache.
# Already-converged checkpoints are skipped automatically by the Python sweep script.
zuco_lme_jz_run_array_task_cpu() {
  : "${ZUCO_FAMILY:?Set ZUCO_FAMILY before calling zuco_lme_jz_run_array_task_cpu}"
  local _n
  _n="$(zuco_lme_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${ZUCO_FAMILY}; exiting."
    exit 0
  fi

  echo "Stage 2 — ZuCo Reading LME fitting (CPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${ZUCO_FAMILY}  task=${ZUCO_TASK}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "ZUCO_READING_FEATURE_CACHE_DIR=${ZUCO_READING_FEATURE_CACHE_DIR:-unset}"
  echo "R_HOME=${R_HOME:-unset}"

  zuco_lme_jz_check_data || exit 1
  zuco_lme_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${ZUCO_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/behavioral/zuco_reading_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${ZUCO_FAMILY}" \
    --array-index "${IDX}" \
    --device cpu \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    "${LAST_LAYER_ONLY_FLAG[@]}" \
    "${LAYERS_FLAG[@]}" \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}"

  echo "Stage 2 completed at $(date)"
}

zuco_lme_jz_run_summary() {
  : "${ZUCO_FAMILY:?Set ZUCO_FAMILY before calling zuco_lme_jz_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  zuco_lme_jz_setup_env
  "${PY[@]}" scripts/run/surprisal/behavioral/zuco_reading_fit.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${ZUCO_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}" \
    --summary_out_dir "${ZUCO_READING_LME_SUMMARY_DIR:-${SWEEP_ROOT}}" \
    --perplexity_dir "${ZUCO_READING_PERPLEXITY_DIR}"
  echo "Summary completed at $(date)"
}
