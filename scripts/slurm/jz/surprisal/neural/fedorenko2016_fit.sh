# Shared runtime for Jean Zay Fedorenko2016 iEEG pooled-LME jobs (one mixed
# model per LM layer, pooling all subjects/electrodes, electrode NESTED
# within subject — subject:electrode — since Fedorenko2016's SEEG/ECoG
# electrodes are patient-specific grids with no shared coordinate system
# across subjects, same as Podcast's ECoG). Source from
# scripts/submit.py (see scripts/slurm/stages.yaml) (do not run directly).
#
# CPU-only, same as the N400 datasets' pooled LMEs: R (pymer4/lme4) is only
# available on CPU partitions on Jean Zay, so this job reads per-layer
# logit-lens surprisal from a cache already populated by
# scripts/run/surprisal/neural/fedorenko2016_extract.py (GPU job, see
# job_fedorenko2016_surprisal_jz_common.sh in this same directory) — it
# never loads the LLM itself. A cache miss silently falls back to loading
# the LLM on CPU (slow, not wrong) rather than hard-erroring — same soft
# contract as the N400 datasets.
#
# ONE JOB PER MODEL, not one job per layer: all layers for a given model are
# fit concurrently inside this single job via a spawned Python process pool
# (--n_workers, default 20, 1 BLAS thread each) — pymer4/rpy2 embeds one R
# interpreter per process, so layer-level parallelism must be process-based.
#
# Fedorenko2016's pooled table (at most 416 words × 97 electrodes ≈ 40K rows
# before filtering) is far smaller than Michaelov's (~55K rows) or Podcast's
# (~1.4M rows, 285 nested groups) — expect fast fits, but this is still
# unvalidated: start with a single array index before trusting the full
# family's time budget.
#
# Required before source:
#   FEDORENKO2016_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, FEDORENKO2016_LME_SWEEP_ROOT
#   N_WORKERS         — concurrent layer-fitting worker processes (default 20)
#   LAYER_STRIDE      — fit every Nth layer (default 1 = all layers)
#   FEDORENKO2016_LME_FEATURE_CACHE_DIR — must already contain this model's
#                       logit-lens fulltext surprisal cache (see
#                       job_fedorenko2016_surprisal_jz_common.sh)
#
# Resumable: fedorenko2016_lme_by_layer.csv is written incrementally, one row
# per completed layer — a killed/requeued job picks up only the remaining
# layers.

: "${FEDORENKO2016_FAMILY:?Set FEDORENKO2016_FAMILY before sourcing job_fedorenko2016_lme_jz_common.sh}"

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
  export FEDORENKO2016_LME_FEATURE_CACHE_DIR="${FEDORENKO2016_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/fedorenko2016_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export FEDORENKO2016_LME_FEATURE_CACHE_DIR="${FEDORENKO2016_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/fedorenko2016_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$FEDORENKO2016_LME_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/neural/fedorenko2016.yaml}"
SWEEP_ROOT="${FEDORENKO2016_LME_SWEEP_ROOT:-$PWD/outputs_jeanzay/fedorenko2016/surprisal_sweep}"
N_WORKERS="${N_WORKERS:-20}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"
# LAST_LAYER_ONLY=1 fits only the final layer (the model's own logits).
LAST_LAYER_ONLY="${LAST_LAYER_ONLY:-0}"
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

fedorenko2016_lme_jz_export_rpy2_env() {
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

fedorenko2016_lme_jz_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    fedorenko2016_lme_jz_export_rpy2_env
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
    fedorenko2016_lme_jz_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      fedorenko2016_lme_jz_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

fedorenko2016_lme_jz_check_data() {
  if [[ ! -f "${BENCHMARK_ROOT}/data/fedorenko2016/stimuli.csv" ]]; then
    echo "ERROR: Fedorenko2016 stimuli.csv not found under ${BENCHMARK_ROOT}/data/fedorenko2016/"
    return 1
  fi
  return 0
}

fedorenko2016_lme_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  # Single-threaded BLAS per worker process — same rationale as the N400
  # datasets' LME jobs: lme4's fit is dominated by sparse (CHOLMOD) Cholesky
  # updates, not dense BLAS-heavy ops, so multi-threading BLAS per worker
  # buys little.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${FEDORENKO2016_LME_FEATURE_CACHE_DIR}" \
    "${SWEEP_ROOT}/${FEDORENKO2016_FAMILY}" outputs_jeanzay/logs
}

fedorenko2016_lme_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${FEDORENKO2016_FAMILY}"
}

fedorenko2016_lme_jz_run_array_task() {
  local _n
  _n="$(fedorenko2016_lme_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${FEDORENKO2016_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${FEDORENKO2016_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "FEDORENKO2016_LME_FEATURE_CACHE_DIR=${FEDORENKO2016_LME_FEATURE_CACHE_DIR}"
  echo "N_WORKERS=${N_WORKERS}  LAYER_STRIDE=${LAYER_STRIDE}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

  fedorenko2016_lme_jz_check_data || exit 1
  fedorenko2016_lme_jz_setup_env
  fedorenko2016_lme_jz_load_r || exit 1

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${FEDORENKO2016_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/fedorenko2016_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${FEDORENKO2016_FAMILY}" \
    --array-index "${IDX}" \
    --out_dir "${SWEEP_ROOT}" \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    "${LAST_LAYER_ONLY_ARGS[@]}" \
    --device cpu \
    --feature_cache_dir "${FEDORENKO2016_LME_FEATURE_CACHE_DIR}"

  echo "Job completed at $(date)"
}
