# Shared runtime for Torch MEG-MASC MEG per-subject LME jobs (one mixed
# model per (subject, LM layer), 4 scalp regions — left/right/posterior/
# occipital, MEG-MASC's 64-channel BioSemi montage collapsed via
# channel_region and averaged BEFORE fitting). See
# neuroai/regressors/megmasc_lme.py for the full formula rationale.
# Source from scripts/submit.py (see scripts/slurm/stages.yaml) (do not run
# directly).
#
# CPU-only, same as every other pooled LME in this codebase: R (pymer4/lme4)
# is only available on CPU partitions on Torch, so this job reads
# per-layer logit-lens surprisal from a cache already populated by
# scripts/run/surprisal/neural/megmasc_extract.py (GPU job, see
# job_megmasc_surprisal_torch_common.sh in this same directory) — it never
# loads the LLM itself. A cache miss silently falls back to loading the LLM
# on CPU (slow, not wrong) rather than hard-erroring — same soft contract as
# every other pooled LME here.
#
# ONE JOB PER MODEL, not one job per layer: all (subject, layer) pairs for a
# given model are fit concurrently inside this single job via a spawned
# Python process pool (--n_workers, default 40, 1 BLAS thread each) —
# pymer4/rpy2 embeds one R interpreter per process, so parallelism must be
# process-based.
#
# MEG-MASC has ~5x more subjects than Broderick (85-105 vs 19), so the
# (subject, layer) task grid is proportionally larger even though each
# individual fit is just as cheap (one subject's own region-averaged rows,
# one random-effect grouping factor). Walltime below is a first estimate
# scaled up from Broderick's own (also unvalidated) numbers — start with a
# single array index and watch slope_dropped in the output CSV before
# trusting a full family run.
#
# Required before source:
#   MEGMASC_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, MEGMASC_LME_SWEEP_ROOT
#   N_WORKERS         — concurrent lme4 worker processes (default 12 — 40 OOM-killed the node on 32+ layer models, n_obs~890k/fit)
#   LAYER_STRIDE      — fit every Nth layer (default 1 = all layers)
#   MEGMASC_LME_FEATURE_CACHE_DIR — must already contain this model's
#                       logit-lens fulltext surprisal cache (see
#                       job_megmasc_surprisal_torch_common.sh)
#
# Resumable: megmasc_lme_by_subject_layer.csv is written incrementally,
# one row per completed (subject, layer) pair — a killed/requeued job picks
# up only the remaining pairs.

: "${MEGMASC_FAMILY:?Set MEGMASC_FAMILY before sourcing job_megmasc_lme_torch_common.sh}"

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1
export PYTHONPATH="${BENCHMARK_ROOT}/forgetting-transformer/src:${BENCHMARK_ROOT}/stickbreaking-attention${PYTHONPATH:+:$PYTHONPATH}"  # constrained-arch Lanni-ni models

# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
  export MEGMASC_LME_FEATURE_CACHE_DIR="${MEGMASC_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export MEGMASC_LME_FEATURE_CACHE_DIR="${MEGMASC_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$MEGMASC_LME_FEATURE_CACHE_DIR"
fi

export MEGMASC_BIDS_ROOT="${MEGMASC_BIDS_ROOT:-${SCRATCH:-$BENCHMARK_ROOT/..}/data/gwilliams}"
mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/surprisal/neural/megmasc.yaml}"
SWEEP_ROOT="${MEGMASC_LME_SWEEP_ROOT:-$PWD/outputs_torch/megmasc/surprisal_sweep}"
N_WORKERS="${N_WORKERS:-12}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

megmasc_lme_torch_export_rpy2_env() {
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

megmasc_lme_torch_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    megmasc_lme_torch_export_rpy2_env
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
    megmasc_lme_torch_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      megmasc_lme_torch_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

megmasc_lme_torch_check_data() {
  if [[ ! -d "${MEGMASC_BIDS_ROOT:-}" ]]; then
    echo "ERROR: MEGMASC_BIDS_ROOT not found: ${MEGMASC_BIDS_ROOT:-<unset>}"
    return 1
  fi
  return 0
}

megmasc_lme_torch_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  # Single-threaded BLAS per worker process — lme4's fit is dominated by
  # sparse (CHOLMOD) Cholesky updates, not dense BLAS-heavy ops, so
  # multi-threading BLAS per worker buys little (same rationale as every
  # other pooled LME's CPU job here).
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${MEGMASC_LME_FEATURE_CACHE_DIR}" \
    "${SWEEP_ROOT}/${MEGMASC_FAMILY}" outputs_torch/logs
}

megmasc_lme_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${MEGMASC_FAMILY}"
}

megmasc_lme_torch_run_array_task() {
  local _n
  _n="$(megmasc_lme_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${MEGMASC_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${MEGMASC_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "MEGMASC_LME_FEATURE_CACHE_DIR=${MEGMASC_LME_FEATURE_CACHE_DIR}"
  echo "N_WORKERS=${N_WORKERS}  LAYER_STRIDE=${LAYER_STRIDE}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

  megmasc_lme_torch_check_data || exit 1
  megmasc_lme_torch_setup_env
  megmasc_lme_torch_load_r || exit 1

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${MEGMASC_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/megmasc_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${MEGMASC_FAMILY}" \
    --array-index "${IDX}" \
    --out_dir "${SWEEP_ROOT}" \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    --device cpu \
    --feature_cache_dir "${MEGMASC_LME_FEATURE_CACHE_DIR}" \
    --epoch_cache_dir "${MEGMASC_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_epochs}"

  echo "Job completed at $(date)"
}
