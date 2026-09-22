# Shared runtime for Torch Provo Corpus embedding-ridge checkpoint sweeps.
# Source from scripts/submit.py (see scripts/slurm/stages.yaml) — do not
# run directly.
#
# Single-stage design (unlike provo_lme's GPU/CPU split): ridge regression is
# plain sklearn, no R/pymer4 dependency, so embedding extraction + ridge fit
# both run in one GPU job.
#
# PROVO_FAMILY  — registry key (gpt2, pythia, olmo2, …). Required for all jobs.
# PROVO_MEASURE — Provo eye-tracking measure: SP | FP | GP (default: FP). Selects
#                 the config file (configs/embedding/behavioral/provo_<measure>.yaml) and
#                 the sweep output subdirectory — but NOT the embedding cache dir:
#                 Provo's SP/FP/GP measures share the exact same word set, so the
#                 embedding cache built by any one PROVO_MEASURE run is reused by
#                 the other two (see neuroai/benchmarks/provo_embed.py docstring).
#
# Optional overrides:
#   CONFIG, PROVO_SWEEP_ROOT, PROVO_DATA_ROOT
#   BENCHMARK_CACHE_ROOT, PROVO_EMBED_CACHE_DIR
#   PROVO_AVERAGE_ACROSS_SUBJECTS=1 — additionally fit RT averaged across
#     subjects per item (see neuroai/benchmarks/subject_average.py); written
#     to a sibling "<PROVO_SWEEP_ROOT>_average" directory, does not affect
#     the normal per-subject output.
#
# NOTE: this reads PROVO_SWEEP_ROOT specifically, NOT the generic SWEEP_ROOT —
# a stale `export SWEEP_ROOT=...` left over from another dataset's session
# would otherwise leak in and silently write output into the wrong directory.

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"
export PROVO_MEASURE="${PROVO_MEASURE:-FP}"
PROVO_MEASURE_LOWER="$(echo "${PROVO_MEASURE}" | tr '[:upper:]' '[:lower:]')"

export PROVO_OUTPUT_FAMILY="${PROVO_OUTPUT_FAMILY:-$PROVO_FAMILY}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT"
fi
# Shared across SP/FP/GP — deliberately NOT nested per measure (see note above).
export PROVO_EMBED_CACHE_DIR="${PROVO_EMBED_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/provo_embed_features}"
# Per-word/per-item OOF prediction .npz files (save_word_predictions=True) go
# here instead of under SWEEP_ROOT — many small files, kept out of the
# per-model sweep directory. Set to "" to fall back to the old behavior (saved
# alongside the CSVs under SWEEP_ROOT).
export PROVO_WORD_PREDICTIONS_ROOT="${PROVO_WORD_PREDICTIONS_ROOT-$BENCHMARK_CACHE_ROOT/provo_word_predictions}"
[[ -n "${PROVO_WORD_PREDICTIONS_ROOT}" ]] && mkdir -p "${PROVO_WORD_PREDICTIONS_ROOT}"
mkdir -p "${PROVO_EMBED_CACHE_DIR}"

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/embedding/behavioral/provo_${PROVO_MEASURE_LOWER}.yaml}"
SWEEP_ROOT="${PROVO_SWEEP_ROOT:-$PWD/outputs_torch/provo/embedding_sweep/${PROVO_MEASURE_LOWER}}"
PROVO_DATA_ROOT="${PROVO_DATA_ROOT:-$SCRATCH/Documents/benchmark/data/provo/osfstorage}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

provo_torch_check_data() {
  if [[ ! -f "${PROVO_DATA_ROOT}/Provo_Corpus-Eyetracking_Data.csv" || \
        ! -f "${PROVO_DATA_ROOT}/Provo_Corpus-Predictability_Norms.csv" ]]; then
    echo "ERROR: Provo data not found under ${PROVO_DATA_ROOT}"
    return 1
  fi
  return 0
}

provo_torch_setup_env() {
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
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${PROVO_EMBED_CACHE_DIR}" \
    outputs_torch/logs
  if [[ -n "${PROVO_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${PROVO_FAMILY}"
  fi
}

provo_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${PROVO_FAMILY}"
}

# GPU: embedding extraction (cached) + per-layer ridge fit, one array task per
# (family, checkpoint). Already-computed checkpoints are skipped automatically.
provo_torch_run_array_task_gpu() {
  : "${PROVO_FAMILY:?Set PROVO_FAMILY before calling provo_torch_run_array_task_gpu}"
  local _n
  _n="$(provo_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${PROVO_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Provo embed ridge sweep (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${PROVO_FAMILY}  measure=${PROVO_MEASURE}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "PROVO_DATA_ROOT=${PROVO_DATA_ROOT}"
  echo "PROVO_EMBED_CACHE_DIR=${PROVO_EMBED_CACHE_DIR:-unset}  (measure-agnostic — shared cache)"
  echo "PROVO_WORD_PREDICTIONS_ROOT=${PROVO_WORD_PREDICTIONS_ROOT:-unset (falls back to SWEEP_ROOT)}"
  echo "PROVO_AVERAGE_ACROSS_SUBJECTS=${PROVO_AVERAGE_ACROSS_SUBJECTS:-0}"
  echo "PROVO_AVERAGE_ONLY=${PROVO_AVERAGE_ONLY:-0}"

  provo_torch_check_data || exit 1
  provo_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${PROVO_FAMILY}" --index "${IDX}"

  local _avg_flag=()
  if [[ "${PROVO_AVERAGE_ONLY:-0}" == "1" ]]; then
    # Implies --average_across_subjects; skip the per-subject fits entirely.
    # (The shipped configs also set 'average_only: true', so this is belt-and-
    # suspenders / lets a run flip it off by editing the config.)
    _avg_flag=(--average_only)
  elif [[ "${PROVO_AVERAGE_ACROSS_SUBJECTS:-0}" == "1" ]]; then
    _avg_flag=(--average_across_subjects)
  fi

  "${PY[@]}" scripts/run/embedding/behavioral/provo.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${PROVO_FAMILY}" \
    --output_family "${PROVO_OUTPUT_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${_avg_flag[@]}"

  echo "Job completed at $(date)"
}


provo_torch_run_summary() {
  : "${PROVO_FAMILY:?Set PROVO_FAMILY before calling provo_torch_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  provo_torch_setup_env
  "${PY[@]}" scripts/run/embedding/behavioral/provo.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${PROVO_FAMILY}" \
    --output_family "${PROVO_OUTPUT_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}" \
    --summary_out_dir "${PROVO_EMBED_SUMMARY_DIR:-${SWEEP_ROOT}}"
  echo "Summary completed at $(date)"
}
