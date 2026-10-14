# Shared runtime for Jean Zay Podcast PER-REGION LME jobs (region-averaged electrodes, one model per
# region x lag, lags 0-600 ms, final-layer logit-lens surprisal read from the cache). Source from
# scripts/submit.py (stage podcast.surprisal.region_fit); do not run directly. ONE JOB PER MODEL,
# all (region, lag) fits concurrently via a spawned process pool (N_WORKERS, 1 BLAS thread each).
# Needs R (pymer4/lme4); CPU-only. Resumable: podcast_region_lme_by_lag.csv is appended per fit.
#
# Required before source:  PODCAST_FAMILY   -- registry key (gpt2, pythia, olmo2, ...)
# Optional overrides:      CONFIG, PODCAST_REGION_SWEEP_ROOT, PODCAST_BIDS_ROOT, N_WORKERS

: "${PODCAST_FAMILY:?Set PODCAST_FAMILY before sourcing job_podcast_region_jz_common.sh}"

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
  export PODCAST_FEATURE_CACHE_DIR="${PODCAST_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_features}"
  export PODCAST_EPOCH_CACHE_DIR="${PODCAST_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_epochs}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$PODCAST_FEATURE_CACHE_DIR" "$PODCAST_EPOCH_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/neural/podcast.yaml}"
SWEEP_ROOT="${PODCAST_REGION_SWEEP_ROOT:-$PWD/outputs_jeanzay/podcast/surprisal_region_sweep}"
PODCAST_BIDS_ROOT="${PODCAST_BIDS_ROOT:-$WORK/benchmark/data/podcast}"
PODCAST_ELECTRODE_SELECTION_DIR="${PODCAST_ELECTRODE_SELECTION_DIR:-$PWD/outputs_jeanzay/podcast/electrode_selection}"
N_WORKERS="${N_WORKERS:-20}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

podcast_region_jz_export_rpy2_env() {
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

podcast_region_jz_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    podcast_region_jz_export_rpy2_env
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
    podcast_region_jz_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      podcast_region_jz_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

podcast_region_jz_check_data() {
  if [[ ! -f "${PODCAST_BIDS_ROOT}/stimuli/podcast_transcript.csv" ]]; then
    echo "ERROR: Podcast transcript not found: ${PODCAST_BIDS_ROOT}/stimuli/podcast_transcript.csv"
    return 1
  fi
  if [[ ! -d "${PODCAST_BIDS_ROOT}/derivatives/ecogprep/sub-01" ]]; then
    echo "ERROR: Podcast derivatives not found: ${PODCAST_BIDS_ROOT}/derivatives/ecogprep/"
    return 1
  fi
  return 0
}

podcast_region_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  # Single-threaded BLAS per worker process. lme4's nested-model fit is
  # dominated by sparse (CHOLMOD) Cholesky updates inside the optimizer loop,
  # not dense BLAS-heavy ops, so multi-threading BLAS per worker buys little
  # — better to trade those threads for more concurrent workers (N_WORKERS
  # is set to match cpus-per-task with 1 thread each). Bump these only if you
  # lower N_WORKERS well below cpus-per-task and want each fit multi-threaded.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${PODCAST_FEATURE_CACHE_DIR}" \
    "${SWEEP_ROOT}/${PODCAST_FAMILY}" outputs_jeanzay/logs
}

podcast_region_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${PODCAST_FAMILY}"
}

podcast_region_jz_run_array_task() {
  local _n
  _n="$(podcast_region_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${PODCAST_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${PODCAST_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "N_WORKERS=${N_WORKERS}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

  podcast_region_jz_check_data || exit 1
  podcast_region_jz_setup_env
  podcast_region_jz_load_r || exit 1

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${PODCAST_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/podcast_region_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${PODCAST_FAMILY}" \
    --array-index "${IDX}" \
    --out_dir "${SWEEP_ROOT}" \
    --n_workers "${N_WORKERS}"

  echo "Job completed at $(date)"
}
