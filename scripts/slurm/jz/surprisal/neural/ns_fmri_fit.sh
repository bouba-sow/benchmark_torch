# Shared runtime for Jean Zay Natural Stories fMRI LME checkpoint sweeps.
# Used through scripts/submit.py (see scripts/slurm/stages.yaml); do not run directly.
#
# GPU nodes on Jean Zay have no R module (pymer4/lme4 only works on the CPU partition), so the
# GPU stage (ns_fmri_extract.sh) only extracts and caches surprisal, and the LME fit
# (ns_fmri_fit.sh) always runs on CPU.
#
# NS_FAMILY: registry key (gpt2, pythia, olmo2, ...).
# Optional overrides: CONFIG, NS_FMRI_LME_SWEEP_ROOT, NS_FMRI_DATA_ROOT, BENCHMARK_CACHE_ROOT,
#   NS_FEATURE_CACHE_DIR, N_WORKERS (concurrent (fROI, layer) fits, default 20), LAYER_STRIDE
#   (fit every Nth layer), UNTRAINED=1 (untrained-baseline path, results under a separate
#   "_untrained" tree; UNTRAINED_SEED, default 0), LAST_LAYER_ONLY=1.
# NS_FMRI_LME_SWEEP_ROOT is read specifically, NOT the generic SWEEP_ROOT: a stale export from
# another dataset's session would otherwise leak in and write into the wrong directory.
# Unlike the Natural Stories SPR LME there is no unigram baseline here (Lin & Schuler's fMRI
# formula uses surprisal + word length + sentence position only).

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

CONFIG="${CONFIG:-configs/surprisal/neural/ns_fmri.yaml}"
UNTRAINED="${UNTRAINED:-0}"
UNTRAINED_SEED="${UNTRAINED_SEED:-0}"
SWEEP_ROOT="${NS_FMRI_LME_SWEEP_ROOT:-$PWD/outputs_jeanzay/ns_fmri/surprisal_sweep}"
NS_FMRI_DATA_ROOT="${NS_FMRI_DATA_ROOT:-$WORK/benchmark/data/naturalstories_fmri/osfstorage}"
N_WORKERS="${N_WORKERS:-20}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"
LAST_LAYER_ONLY="${LAST_LAYER_ONLY:-0}"

# NOTE: --out_dir passed to the python script below (and NS_FEATURE_CACHE_DIR,
# which the YAML config's cache.features_dir expands to) are the *plain*
# (non-suffixed) paths -- run/surprisal/neural/ns_fmri_fit.py itself appends "_untrained"
# when --untrained is set. Don't also suffix here, or the two suffixes stack
# (…_untrained_untrained).
UNTRAINED_ARGS=()
if [[ "${UNTRAINED}" == "1" ]]; then
  UNTRAINED_ARGS=(--untrained --untrained_seed "${UNTRAINED_SEED}")
fi

LAST_LAYER_ONLY_ARGS=()
if [[ "${LAST_LAYER_ONLY}" == "1" ]]; then
  LAST_LAYER_ONLY_ARGS=(--last_layer_only)
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

ns_fmri_lme_jz_export_rpy2_env() {
  # pymer4 → rpy2 → lme4; ABI mode matches the login-node rpy2 install.
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

ns_fmri_lme_jz_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    ns_fmri_lme_jz_export_rpy2_env
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
    ns_fmri_lme_jz_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      ns_fmri_lme_jz_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

ns_fmri_lme_jz_check_data() {
  if [[ ! -f "${NS_FMRI_DATA_ROOT}/LANG_MD_ts_X.csv" ]]; then
    echo "ERROR: Natural Stories fMRI X table not found: ${NS_FMRI_DATA_ROOT}/LANG_MD_ts_X.csv"
    return 1
  fi
  if [[ ! -f "${NS_FMRI_DATA_ROOT}/LANG_MD_y_train.csv" ]]; then
    echo "ERROR: Natural Stories fMRI Y table missing: ${NS_FMRI_DATA_ROOT}/LANG_MD_y_train.csv"
    return 1
  fi
  return 0
}

ns_fmri_lme_jz_setup_base_env() {
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

ns_fmri_lme_jz_setup_env() {
  ns_fmri_lme_jz_setup_base_env
  ns_fmri_lme_jz_load_r || exit 1
}

ns_fmri_lme_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${NS_FAMILY}"
}

ns_fmri_lme_jz_run_array_task_cpu() {
  : "${NS_FAMILY:?Set NS_FAMILY before calling ns_fmri_lme_jz_run_array_task_cpu}"
  local _n
  _n="$(ns_fmri_lme_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${NS_FAMILY}; exiting."
    exit 0
  fi

  echo "fMRI LME fitting (CPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${NS_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT} (effective out_dir gets _untrained suffix from the python script when UNTRAINED=1)"
  echo "UNTRAINED=${UNTRAINED} UNTRAINED_SEED=${UNTRAINED_SEED}"
  echo "NS_FMRI_DATA_ROOT=${NS_FMRI_DATA_ROOT}"
  echo "NS_FEATURE_CACHE_DIR=${NS_FEATURE_CACHE_DIR:-unset}"
  echo "R_HOME=${R_HOME:-unset}"
  echo "N_WORKERS=${N_WORKERS}  LAYER_STRIDE=${LAYER_STRIDE}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

  ns_fmri_lme_jz_check_data || exit 1
  ns_fmri_lme_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${NS_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/ns_fmri_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NS_FAMILY}" \
    --array-index "${IDX}" \
    --device cpu \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    "${LAST_LAYER_ONLY_ARGS[@]}" \
    "${UNTRAINED_ARGS[@]}"

  echo "Fit completed at $(date)"
}

ns_fmri_lme_jz_run_summary() {
  : "${NS_FAMILY:?Set NS_FAMILY before calling ns_fmri_lme_jz_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  ns_fmri_lme_jz_setup_env
  "${PY[@]}" scripts/run/surprisal/neural/ns_fmri_fit.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${NS_FAMILY}" \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
