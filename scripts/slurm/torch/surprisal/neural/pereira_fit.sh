# Shared runtime for Torch Pereira fMRI pooled-LME jobs (one mixed model
# per LM layer, pooling all subjects/language-atlas ROIs/sentences). ROI is
# a CROSSED random effect (fixed canonical 12-label language-atlas set,
# present in all 10 subjects, like Tuckute's LANG_ROIS/a scalp
# electrode), plus a third crossed grouping factor, sentence_id — see
# neuroai/regressors/pereira_pooled_lme.py for the full formula rationale.
# Source from scripts/submit.py (see scripts/slurm/stages.yaml) (do not run
# directly). Modeled directly on
# scripts/slurm/torch/surprisal/neural/lpp_fit.sh /
# scripts/slurm/torch/surprisal/neural/tuckute_fit.sh.
#
# CPU-only, same as every other pooled LME in this codebase: R (pymer4/lme4)
# is only available on CPU partitions on Torch, so this job reads
# per-layer last-word logit-lens surprisal from a cache already populated by
# python scripts/submit.py pereira surprisal extract --all-families /
# surprisal_opt66b.sh (GPU jobs) — it never loads the LLM itself. A cache
# miss silently falls back to loading the LLM on CPU (slow, not wrong)
# rather than hard-erroring — same soft contract as every other pooled LME
# here.
#
# ONE JOB PER MODEL (one SLURM array task = one model), not one job per
# layer: all layers for a given model are fit concurrently inside this
# single task via a spawned Python process pool (N_WORKERS, default 30 per
# 2026-09-01 request — matches the LPP/NS-fMRI convention of parallelizing
# LAYERS within one model's job, not just models against each other).
#
# train_frac=0.75 (75/25 train/test) is the configs/surprisal/neural/pereira.yaml default —
# same convention as every other pooled LME (LPP, NS-fMRI, Tuckute, ...).
#
# Required before source:
#   PEREIRA_FAMILY   — registry key (gpt2, pythia, olmo2, opt, …)
#
# Optional overrides:
#   CONFIG, PEREIRA_LME_SWEEP_ROOT
#   N_WORKERS         — concurrent layer-fitting worker processes (default 30)
#   LAYER_STRIDE      — fit every Nth layer (default 1 = all layers)
#   PEREIRA_LME_FEATURE_CACHE_DIR — must already contain this model's
#                       logit-lens fulltext surprisal cache (see
#                       surprisal_all_families.sh / surprisal_opt66b.sh)
#
# Resumable: pereira_lme_by_layer.csv is written incrementally, one row
# per completed layer — a killed/requeued job picks up only the remaining
# layers.

: "${PEREIRA_FAMILY:?Set PEREIRA_FAMILY before sourcing job_pereira_lme_torch_common.sh}"

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

export PEREIRA_LME_FEATURE_CACHE_DIR="${PEREIRA_LME_FEATURE_CACHE_DIR:-$SCRATCH/.cache/pereira_lme_features}"

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/surprisal/neural/pereira.yaml}"
SWEEP_ROOT="${PEREIRA_LME_SWEEP_ROOT:-$PWD/outputs_torch/pereira/surprisal_sweep}"
N_WORKERS="${N_WORKERS:-30}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

pereira_lme_torch_export_rpy2_env() {
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

pereira_lme_torch_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    pereira_lme_torch_export_rpy2_env
    return 0
  fi
  if command -v module >/dev/null 2>&1; then
    module load r/4.6.1 2>/dev/null || module load r 2>/dev/null || module load R 2>/dev/null || true
  fi
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    pereira_lme_torch_export_rpy2_env
    return 0
  fi
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      pereira_lme_torch_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

pereira_lme_torch_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  # Single-threaded BLAS per worker process — lme4's fit is dominated by
  # sparse (CHOLMOD) Cholesky updates, not dense BLAS-heavy ops.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "${PEREIRA_LME_FEATURE_CACHE_DIR}" "${SWEEP_ROOT}/${PEREIRA_FAMILY}" outputs_torch/logs
}

pereira_lme_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${PEREIRA_FAMILY}"
}

pereira_lme_torch_run_array_task() {
  local _n
  _n="$(pereira_lme_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${PEREIRA_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${PEREIRA_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "PEREIRA_LME_FEATURE_CACHE_DIR=${PEREIRA_LME_FEATURE_CACHE_DIR}"
  echo "N_WORKERS=${N_WORKERS}  LAYER_STRIDE=${LAYER_STRIDE}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

  pereira_lme_torch_setup_env
  pereira_lme_torch_load_r || exit 1

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${PEREIRA_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/pereira_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${PEREIRA_FAMILY}" \
    --array-index "${IDX}" \
    --out_dir "${SWEEP_ROOT}" \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    --device cpu \
    --feature_cache_dir "${PEREIRA_LME_FEATURE_CACHE_DIR}"

  echo "Job completed at $(date)"
}
