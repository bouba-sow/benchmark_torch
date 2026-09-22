# Shared runtime for Torch Podcast pooled-LME jobs (Design B — one mixed
# model per lag, pooling all subjects/electrodes, electrode nested within
# subject). Source from scripts/submit.py (see scripts/slurm/stages.yaml)
# (do not run directly).
#
# Unlike the ridge-based surprisal sweep (job_podcast_surprisal_torch_common.sh),
# this DOES need R (pymer4/lme4) — the LME fit itself runs here, not just
# surprisal extraction. It's still CPU-only: surprisal is read from the
# existing logit-lens cache (already computed by the ridge sweep), never
# recomputed, so no GPU/model load ever happens in this job.
#
# ONE JOB PER MODEL, not one job per lag: all lags for a given model are fit
# concurrently inside this single job via a spawned Python process pool
# (--n_workers, default 20, 1 BLAS thread each) — pymer4/rpy2 embeds one R
# interpreter per process, so lag-level parallelism must be process-based,
# and each SLURM job's cost is per-job, not per-lag, given limited Torch
# hours.
#
# N_WORKERS=40 (one per cpus-per-task, 1 BLAS thread each) was tried and
# OOM-killed the whole job ~9 min in (all 40 R/lme4 processes fitting the
# ~1.4M-row nested model concurrently exceeded the node's default memory
# allocation for cpus-per-task=40) — 10 workers ran cleanly. 20 is an
# untested middle ground; watch the first array task's log for
# "oom_kill event" before trusting a full family run, and back off toward
# 10 (or add an explicit --mem/--mem-per-cpu request) if it recurs.
#
# Required before source:
#   PODCAST_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, PODCAST_LME_SWEEP_ROOT, PODCAST_BIDS_ROOT
#   N_WORKERS         — concurrent lag-fitting worker processes (default 20;
#                       set so N_WORKERS × threads-per-worker ≈ cpus-per-task)
#   LAG_STRIDE        — fit every Nth lag (default 1 = all lags). Use a
#                       larger stride for a cheap first pass across the grid.
#   PODCAST_FEATURE_CACHE_DIR — must already contain this model's logit-lens
#                       surprisal cache (see job_podcast_surprisal_torch_common.sh).
#
# Resumable: podcast_lme_by_lag.csv is written incrementally, one row per
# completed lag — a killed/requeued job picks up only the remaining lags.

: "${PODCAST_FAMILY:?Set PODCAST_FAMILY before sourcing job_podcast_lme_torch_common.sh}"

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export PODCAST_FEATURE_CACHE_DIR="${PODCAST_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_features}"
  export PODCAST_EPOCH_CACHE_DIR="${PODCAST_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_epochs}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$PODCAST_FEATURE_CACHE_DIR" "$PODCAST_EPOCH_CACHE_DIR"
fi

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/surprisal/neural/podcast.yaml}"
SWEEP_ROOT="${PODCAST_LME_SWEEP_ROOT:-$PWD/outputs_torch/podcast/surprisal_sweep}"
PODCAST_BIDS_ROOT="${PODCAST_BIDS_ROOT:-$SCRATCH/Documents/benchmark/data/podcast}"
PODCAST_ELECTRODE_SELECTION_DIR="${PODCAST_ELECTRODE_SELECTION_DIR:-$PWD/outputs_torch/podcast/electrode_selection}"
N_WORKERS="${N_WORKERS:-20}"
LAG_STRIDE="${LAG_STRIDE:-1}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

podcast_lme_torch_export_rpy2_env() {
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

podcast_lme_torch_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    podcast_lme_torch_export_rpy2_env
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
    podcast_lme_torch_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      podcast_lme_torch_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

podcast_lme_torch_check_data() {
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

podcast_lme_torch_setup_env() {
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
    "${SWEEP_ROOT}/${PODCAST_FAMILY}" outputs_torch/logs
}

podcast_lme_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${PODCAST_FAMILY}"
}

podcast_lme_torch_run_array_task() {
  local _n
  _n="$(podcast_lme_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${PODCAST_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${PODCAST_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "N_WORKERS=${N_WORKERS}  LAG_STRIDE=${LAG_STRIDE}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

  podcast_lme_torch_check_data || exit 1
  podcast_lme_torch_setup_env
  podcast_lme_torch_load_r || exit 1

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${PODCAST_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/podcast_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${PODCAST_FAMILY}" \
    --array-index "${IDX}" \
    --out_dir "${SWEEP_ROOT}" \
    --n_workers "${N_WORKERS}" \
    --lag_stride "${LAG_STRIDE}"

  echo "Job completed at $(date)"
}
