# Shared runtime for Torch Tuckute fMRI pooled-LME jobs (one mixed model
# per LM layer, pooling all subjects/ROIs/sentences for one condition —
# default "B", the naturalistic baseline). ROI is a CROSSED random effect
# (fixed canonical LANG_ROIS labels shared across subjects, like Michaelov's
# scalp electrode), plus a third crossed grouping factor, sentence_id — see
# neuroai/regressors/tuckute_pooled_lme.py for the full formula rationale.
# Source from scripts/submit.py (see scripts/slurm/stages.yaml) (do not run
# directly).
#
# CPU-only, same as every other pooled LME in this codebase: R (pymer4/lme4)
# is only available on CPU partitions on Torch, so this job reads
# per-layer last-word logit-lens surprisal from a cache already populated by
# scripts/run/surprisal/neural/tuckute_extract.py (GPU job, see
# job_tuckute_surprisal_torch_common.sh in this same directory) — it
# never loads the LLM itself. A cache miss silently falls back to loading
# the LLM on CPU (slow, not wrong) rather than hard-erroring — same soft
# contract as every other pooled LME here.
#
# ONE JOB PER MODEL, not one job per layer: all layers for a given model are
# fit concurrently inside this single job via a spawned Python process pool
# (--n_workers, default 20, 1 BLAS thread each) — pymer4/rpy2 embeds one R
# interpreter per process, so layer-level parallelism must be process-based.
#
# Tuckute's pooled table (condition "B": up to 10 subjects × 6 ROIs × 1,000
# sentences ≈ 60K rows before filtering) is comparable to Michaelov's
# (~55K rows) and far smaller than Podcast's (~1.4M rows, 285 nested groups)
# — expect fast fits, but this is still unvalidated: start with a single
# array index before trusting the full family's time budget.
#
# Required before source:
#   TUCKUTE_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, TUCKUTE_LME_SWEEP_ROOT
#   N_WORKERS         — concurrent layer-fitting worker processes (default 20)
#   LAYER_STRIDE      — fit every Nth layer (default 1 = all layers)
#   TUCKUTE_LME_FEATURE_CACHE_DIR — must already contain this model's
#                       logit-lens fulltext surprisal cache (see
#                       job_tuckute_surprisal_torch_common.sh)
#
# Resumable: tuckute_lme_by_layer.csv is written incrementally, one row
# per completed layer — a killed/requeued job picks up only the remaining
# layers.

: "${TUCKUTE_FAMILY:?Set TUCKUTE_FAMILY before sourcing job_tuckute_lme_torch_common.sh}"

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
  export TUCKUTE_LME_FEATURE_CACHE_DIR="${TUCKUTE_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/tuckute_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export TUCKUTE_LME_FEATURE_CACHE_DIR="${TUCKUTE_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/tuckute_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$TUCKUTE_LME_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/surprisal/neural/tuckute.yaml}"
SWEEP_ROOT="${TUCKUTE_LME_SWEEP_ROOT:-$PWD/outputs_torch/tuckute/surprisal_sweep}"
N_WORKERS="${N_WORKERS:-20}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

tuckute_lme_torch_export_rpy2_env() {
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

tuckute_lme_torch_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    tuckute_lme_torch_export_rpy2_env
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
    tuckute_lme_torch_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      tuckute_lme_torch_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

tuckute_lme_torch_check_data() {
  # TuckuteDataset reads brain-lang-data_participant_20230728.csv from EITHER
  # an extracted data/ subdirectory OR data.tar (see
  # neuroai/datasets/tuckute.py::_read_csv_from_dir_or_tar) — this OSF
  # deposit ships only the tar, never extracted, so both must be accepted.
  local _root="${BENCHMARK_ROOT}/data/tuckute_fmri/osfstorage"
  if [[ ! -f "${_root}/data/brain-lang-data_participant_20230728.csv" && ! -f "${_root}/data.tar" ]]; then
    echo "ERROR: Tuckute data not found — need either ${_root}/data/brain-lang-data_participant_20230728.csv or ${_root}/data.tar"
    return 1
  fi
  return 0
}

tuckute_lme_torch_setup_env() {
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
    "$BENCHMARK_CACHE_ROOT" "${TUCKUTE_LME_FEATURE_CACHE_DIR}" \
    "${SWEEP_ROOT}/${TUCKUTE_FAMILY}" outputs_torch/logs
}

tuckute_lme_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${TUCKUTE_FAMILY}"
}

tuckute_lme_torch_run_array_task() {
  local _n
  _n="$(tuckute_lme_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${TUCKUTE_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${TUCKUTE_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "TUCKUTE_LME_FEATURE_CACHE_DIR=${TUCKUTE_LME_FEATURE_CACHE_DIR}"
  echo "N_WORKERS=${N_WORKERS}  LAYER_STRIDE=${LAYER_STRIDE}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

  tuckute_lme_torch_check_data || exit 1
  tuckute_lme_torch_setup_env
  tuckute_lme_torch_load_r || exit 1

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${TUCKUTE_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/tuckute_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${TUCKUTE_FAMILY}" \
    --array-index "${IDX}" \
    --out_dir "${SWEEP_ROOT}" \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    --device cpu \
    --feature_cache_dir "${TUCKUTE_LME_FEATURE_CACHE_DIR}"

  echo "Job completed at $(date)"
}
