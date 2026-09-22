# Shared runtime for Jean Zay MECO (English subset) LME checkpoint sweeps.
# Source from scripts/slurm/jz/meco_<family>_gpu.sh (Stage 1) or
# scripts/slurm/jz/meco_<family>_lme_cpu.sh (Stage 2) — do not run directly.
#
# Two-stage design, same as Provo / GECO / ZuCo Reading / UCL Reading:
#   Stage 1 (GPU, meco_lme_jz_prefetch_gpu)       — surprisal + perplexity, cached.
#   Stage 2 (CPU, meco_lme_jz_run_array_task_cpu) — LME fit, reads cache, needs R.
# R was found to be unavailable on Jean Zay's H100 partition even after a clean
# `module load r/4.5.2` (works fine on the CPU partition) — split to sidestep
# that rather than chase the module issue.
#
# MECO_FAMILY   — registry key (gpt2, pythia, olmo2, …). Required for all jobs.
# MECO_MEASURE  — MECO eye-tracking measure: SP | FP | GP (default: FP). Selects
#                 the config file (configs/surprisal/behavioral/meco_<measure>.yaml) and the
#                 sweep output subdirectory — but NOT the feature cache dir:
#                 MECO's SP/FP/GP share the exact same word set (per language),
#                 so Stage 1 (surprisal prefetch) only needs to run ONCE per
#                 (family, checkpoint) — not once per measure. Run Stage 1 with
#                 any single MECO_MEASURE value; all three Stage 2 sweeps reuse
#                 that one cache.
# MECO_LANGUAGE — MECO language code (default: en). Only English is wired up by
#                 neuroai.datasets.meco today — see its module docstring.
#
# Optional overrides:
#   CONFIG, MECO_SWEEP_ROOT, MECO_DATA_ROOT, MECO_PERPLEXITY_DIR, MECO_RDA_FILENAME
#   BENCHMARK_CACHE_ROOT, MECO_FEATURE_CACHE_DIR
#   N_WORKERS    — concurrent layer-fitting worker processes for the CPU LME
#                  stage (default 20) — pymer4/rpy2 embeds one R interpreter
#                  per process, so layer-level parallelism must be process-based.
#   LAYER_STRIDE — fit every Nth layer (default 1 = all layers)
#   SAVE_WORD_PREDICTIONS — 1 (default) saves per-word predictions to
#                  *_lme_per_word.csv; set to 0 to disable
#
# NOTE: this reads MECO_SWEEP_ROOT specifically, NOT the generic SWEEP_ROOT —
# a stale `export SWEEP_ROOT=...` left over from another dataset's session
# would otherwise leak in and silently write output into the wrong directory.

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
export MECO_MEASURE="${MECO_MEASURE:-FP}"
export MECO_LANGUAGE="${MECO_LANGUAGE:-en}"
MECO_MEASURE_LOWER="$(echo "${MECO_MEASURE}" | tr '[:upper:]' '[:lower:]')"
MECO_RDA_FILENAME="${MECO_RDA_FILENAME:-joint_l1_data_trimmed_version2.1.rda}"

cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT"
fi
# Shared across SP/FP/GP — deliberately NOT nested per measure (see note above).
export MECO_FEATURE_CACHE_DIR="${MECO_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/meco_features}"
mkdir -p "${MECO_FEATURE_CACHE_DIR}"

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/behavioral/meco_${MECO_MEASURE_LOWER}.yaml}"
SWEEP_ROOT="${MECO_SWEEP_ROOT:-$PWD/outputs_jeanzay/meco/surprisal_sweep/${MECO_MEASURE_LOWER}}"
# NOTE: this path contains spaces (OSF's released directory names) — always
# reference it quoted ("${MECO_DATA_ROOT}"), never bare.
MECO_DATA_ROOT="${MECO_DATA_ROOT:-$WORK/benchmark/data/meco/osfstorage/release 2.0/version 2.1/wave 1/primary data/eye tracking data}"
export MECO_PERPLEXITY_DIR="${MECO_PERPLEXITY_DIR:-$PWD/outputs_jeanzay/meco/perplexity}"
mkdir -p "${MECO_PERPLEXITY_DIR}"

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

meco_lme_jz_export_rpy2_env() {
  # pymer4 → rpy2 → lme4; ABI mode matches the login-node rpy2 install.
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

meco_lme_jz_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    meco_lme_jz_export_rpy2_env
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
    meco_lme_jz_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      meco_lme_jz_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

meco_lme_jz_check_data() {
  if [[ ! -f "${MECO_DATA_ROOT}/${MECO_RDA_FILENAME}" ]]; then
    echo "ERROR: MECO word-level .rda file not found: ${MECO_DATA_ROOT}/${MECO_RDA_FILENAME}"
    return 1
  fi
  return 0
}


meco_lme_jz_setup_base_env() {
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
    "${MECO_FEATURE_CACHE_DIR}" \
    outputs_jeanzay/logs
  if [[ -n "${MECO_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${MECO_FAMILY}"
  fi
}

meco_lme_jz_setup_env() {
  meco_lme_jz_setup_base_env
  meco_lme_jz_load_r || exit 1
}

meco_lme_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${MECO_FAMILY}"
}

# Stage 1 — GPU: surprisal + perplexity prefetch only (no R/LME fit). Runs ONCE
# per (family, checkpoint) regardless of MECO_MEASURE — the cache it writes is
# shared by all three Stage-2 measure sweeps (see header note).
# Already-cached surprisal+perplexity are skipped without reloading the model.
meco_lme_jz_prefetch_gpu() {
  : "${MECO_FAMILY:?Set MECO_FAMILY before calling meco_lme_jz_prefetch_gpu}"
  local _n
  _n="$(meco_lme_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${MECO_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Stage 1 — MECO surprisal + perplexity prefetch (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${MECO_FAMILY}  language=${MECO_LANGUAGE}  array_size=${_n}  (measure-agnostic — shared cache)"
  echo "CONFIG=${CONFIG}"
  echo "MECO_DATA_ROOT=${MECO_DATA_ROOT}"
  echo "MECO_FEATURE_CACHE_DIR=${MECO_FEATURE_CACHE_DIR:-unset}"
  echo "MECO_PERPLEXITY_DIR=${MECO_PERPLEXITY_DIR}"

  meco_lme_jz_check_data || exit 1
  meco_lme_jz_setup_base_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${MECO_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/behavioral/meco_extract.py \
    --config "${CONFIG}" \
    --family "${MECO_FAMILY}" \
    --array-index "${IDX}" \
    --meco-root "${MECO_DATA_ROOT}" \
    --cache-dir "${MECO_FEATURE_CACHE_DIR}" \
    --perplexity-dir "${MECO_PERPLEXITY_DIR}" \
    --device cuda

  echo "Stage 1 completed at $(date)"
}

# Stage 2 — CPU: LME fitting only (surprisal read from cache, no GPU / no model
# load). Requires Stage 1 to have already written the surprisal cache (for ANY
# MECO_MEASURE — the cache is shared). Already-converged checkpoints are
# skipped automatically by the Python sweep script.
meco_lme_jz_run_array_task_cpu() {
  : "${MECO_FAMILY:?Set MECO_FAMILY before calling meco_lme_jz_run_array_task_cpu}"
  local _n
  _n="$(meco_lme_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${MECO_FAMILY}; exiting."
    exit 0
  fi

  echo "Stage 2 — MECO LME fitting (CPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${MECO_FAMILY}  measure=${MECO_MEASURE}  language=${MECO_LANGUAGE}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "MECO_FEATURE_CACHE_DIR=${MECO_FEATURE_CACHE_DIR:-unset}"
  echo "R_HOME=${R_HOME:-unset}"

  meco_lme_jz_check_data || exit 1
  meco_lme_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${MECO_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/behavioral/meco_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${MECO_FAMILY}" \
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

meco_lme_jz_run_summary() {
  : "${MECO_FAMILY:?Set MECO_FAMILY before calling meco_lme_jz_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  meco_lme_jz_setup_env
  "${PY[@]}" scripts/run/surprisal/behavioral/meco_fit.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${MECO_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}" \
    --summary_out_dir "${MECO_LME_SUMMARY_DIR:-${SWEEP_ROOT}}"
  echo "Summary completed at $(date)"
}
