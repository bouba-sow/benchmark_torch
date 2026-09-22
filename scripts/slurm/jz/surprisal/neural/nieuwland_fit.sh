# Shared runtime for Jean Zay Nieuwland N400 pooled-LME jobs (one mixed
# model per LM layer, pooling all subjects/electrodes, electrode CROSSED —
# not nested, unlike Podcast's subject-specific ECoG grid — since Nieuwland's
# scalp electrodes are the same physical 10-20 locations for every
# subject). Source from
# scripts/submit.py (see scripts/slurm/stages.yaml) (do not run directly).
#
# CPU-only, same as Podcast's pooled LME: R (pymer4/lme4) is only available
# on CPU partitions on Jean Zay, so this job reads per-layer logit-lens
# surprisal from a cache already populated by
# scripts/run/surprisal/neural/nieuwland_extract.py (GPU job, see
# job_nieuwland_surprisal_jz_common.sh in this same directory) — it never
# loads the LLM itself. A cache miss silently falls back to loading the LLM
# on CPU (slow, not wrong) rather than hard-erroring — same soft contract as
# Podcast's LME.
#
# ONE JOB PER MODEL, not one job per layer: all layers for a given model are
# fit concurrently inside this single job via a spawned Python process pool
# (--n_workers, default 20, 1 BLAS thread each) — pymer4/rpy2 embeds one R
# interpreter per process, so layer-level parallelism must be process-based.
#
# Nieuwland's pooled table (up to ~300 subjects x 6 electrodes x ~160 items
# ≈ 300K+ rows, likely less after artefact/trial exclusion) is bigger than
# Michaelov's (~55K) and Ryskin's (~123K), but its random-effects structure
# is simpler than Podcast's — electrode is CROSSED (6 levels total), not
# Podcast's 285-group nested subject:electrode covariance. Podcast tolerated
# 25 workers cleanly (33 OOM'd) on the same cpus-per-task=40 allocation
# under a harder problem (1.4M rows, nested random slope), so N_WORKERS=20
# here (matching the earlier N400 datasets' default) is a reasoned choice, not
# just a guess — but still empirically unvalidated for Nieuwland specifically:
# start with a single array index before trusting the full family's time
# budget, and watch that job's log for "oom_kill event".
#
# This job ALSO epochs raw EEG (expensive, MNE-based) the first time it
# runs for a given set of EEG-preprocessing params — subsequent runs (any
# model) hit the per-subject NIEUWLAND_EPOCH_CACHE_DIR cache. If the
# existing Nieuwland ridge checkpoint sweep has already populated that cache
# with matching params, this job gets it for free.
#
# Required before source:
#   NIEUWLAND_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, NIEUWLAND_LME_SWEEP_ROOT
#   N_WORKERS         — concurrent layer-fitting worker processes (default 20)
#   LAYER_STRIDE      — fit every Nth layer (default 1 = all layers)
#   NIEUWLAND_LME_FEATURE_CACHE_DIR — must already contain this model's
#                       logit-lens trial surprisal cache (see
#                       job_nieuwland_surprisal_jz_common.sh)
#   NIEUWLAND_EPOCH_CACHE_DIR — per-subject EEG epoch cache (shared with the
#                       existing ridge sweep)
#
# Resumable: nieuwland_lme_by_layer.csv is written incrementally, one row per
# completed layer — a killed/requeued job picks up only the remaining layers.

: "${NIEUWLAND_FAMILY:?Set NIEUWLAND_FAMILY before sourcing job_nieuwland_lme_jz_common.sh}"

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
  # NIEUWLAND_EPOCH_CACHE_DIR is already exported by jz_cache_env.sh — reused
  # as-is (shared with the ridge sweep), not redefined here.
  export NIEUWLAND_LME_FEATURE_CACHE_DIR="${NIEUWLAND_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export NIEUWLAND_EPOCH_CACHE_DIR="${NIEUWLAND_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_epochs}"
  export NIEUWLAND_LME_FEATURE_CACHE_DIR="${NIEUWLAND_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$NIEUWLAND_EPOCH_CACHE_DIR" "$NIEUWLAND_LME_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/neural/nieuwland.yaml}"
SWEEP_ROOT="${NIEUWLAND_LME_SWEEP_ROOT:-$PWD/outputs_jeanzay/nieuwland/surprisal_sweep}"
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

nieuwland_lme_jz_export_rpy2_env() {
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

nieuwland_lme_jz_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    nieuwland_lme_jz_export_rpy2_env
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
    nieuwland_lme_jz_export_rpy2_env
    return 0
  fi
  # Last-resort fallback: R_HOME inherited via --export=ALL with its
  # bin/R actually usable *on this node* -- re-verify with command -v
  # after adding to PATH rather than trusting the -x test and returning
  # unconditionally.
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      nieuwland_lme_jz_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

nieuwland_lme_jz_check_data() {
  local bids_root="${NIEUWLAND_BIDS_ROOT:-${BENCHMARK_ROOT}/data/nieuwland}"
  if [[ ! -f "${bids_root}/participants.tsv" ]]; then
    echo "ERROR: Nieuwland participants.tsv not found under ${bids_root}"
    return 1
  fi
  return 0
}

nieuwland_lme_jz_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  # Single-threaded BLAS per worker process — same rationale as Podcast's LME
  # job: lme4's fit is dominated by sparse (CHOLMOD) Cholesky updates, not
  # dense BLAS-heavy ops, so multi-threading BLAS per worker buys little.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${NIEUWLAND_LME_FEATURE_CACHE_DIR}" "${NIEUWLAND_EPOCH_CACHE_DIR}" \
    "${SWEEP_ROOT}/${NIEUWLAND_FAMILY}" outputs_jeanzay/logs
}

nieuwland_lme_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${NIEUWLAND_FAMILY}"
}

nieuwland_lme_jz_run_array_task() {
  local _n
  _n="$(nieuwland_lme_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${NIEUWLAND_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${NIEUWLAND_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "NIEUWLAND_LME_FEATURE_CACHE_DIR=${NIEUWLAND_LME_FEATURE_CACHE_DIR}"
  echo "NIEUWLAND_EPOCH_CACHE_DIR=${NIEUWLAND_EPOCH_CACHE_DIR}"
  echo "N_WORKERS=${N_WORKERS}  LAYER_STRIDE=${LAYER_STRIDE}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

  nieuwland_lme_jz_check_data || exit 1
  nieuwland_lme_jz_setup_env
  nieuwland_lme_jz_load_r || exit 1

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${NIEUWLAND_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/nieuwland_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${NIEUWLAND_FAMILY}" \
    --array-index "${IDX}" \
    --out_dir "${SWEEP_ROOT}" \
    --n_workers "${N_WORKERS}" \
    --layer_stride "${LAYER_STRIDE}" \
    --device cpu \
    --feature_cache_dir "${NIEUWLAND_LME_FEATURE_CACHE_DIR}" \
    --epoch_cache_dir "${NIEUWLAND_EPOCH_CACHE_DIR}"

  echo "Job completed at $(date)"
}
