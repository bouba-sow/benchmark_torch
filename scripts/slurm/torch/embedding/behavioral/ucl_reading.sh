# Shared runtime for Torch UCL Self-Paced Reading embedding-ridge
# checkpoint sweeps. Source from
# scripts/submit.py (see scripts/slurm/stages.yaml) — do not
# run directly.
#
# Single-stage design (unlike ucl_reading_lme's GPU/CPU split): ridge
# regression is plain sklearn, no R/pymer4 dependency, so embedding
# extraction + ridge fit both run in one GPU job. Unlike Provo/GECO/MECo, UCL
# Reading has only one measure (self-paced RT) — no per-measure config/env var.
#
# UCL_READING_FAMILY — registry key (gpt2, pythia, olmo2, …). Required for all jobs.
#
# Optional overrides:
#   CONFIG, UCL_READING_SWEEP_ROOT, UCL_READING_DATA_ROOT
#   BENCHMARK_CACHE_ROOT, UCL_READING_EMBED_CACHE_DIR
#   UCL_READING_AVERAGE_ACROSS_SUBJECTS=1 — additionally fit RT averaged
#     across subjects per item (see neuroai/benchmarks/subject_average.py);
#     written to a sibling "<UCL_READING_SWEEP_ROOT>_average" directory,
#     does not affect the normal per-subject output.
#
# NOTE: this reads UCL_READING_SWEEP_ROOT specifically, NOT the generic
# SWEEP_ROOT — a stale `export SWEEP_ROOT=...` left over from another
# dataset's session would otherwise leak in and silently write output into
# the wrong directory.

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

export UCL_READING_OUTPUT_FAMILY="${UCL_READING_OUTPUT_FAMILY:-$UCL_READING_FAMILY}"

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
export UCL_READING_EMBED_CACHE_DIR="${UCL_READING_EMBED_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/ucl_reading_features}"
# Per-word/per-item OOF prediction .npz files (save_word_predictions=True) go
# here instead of under SWEEP_ROOT — many small files, kept out of the
# per-model sweep directory. Set to "" to fall back to the old behavior (saved
# alongside the CSVs under SWEEP_ROOT).
export UCL_READING_WORD_PREDICTIONS_ROOT="${UCL_READING_WORD_PREDICTIONS_ROOT-$BENCHMARK_CACHE_ROOT/ucl_reading_word_predictions}"
[[ -n "${UCL_READING_WORD_PREDICTIONS_ROOT}" ]] && mkdir -p "${UCL_READING_WORD_PREDICTIONS_ROOT}"
mkdir -p "${UCL_READING_EMBED_CACHE_DIR}"

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/embedding/behavioral/ucl_reading.yaml}"
SWEEP_ROOT="${UCL_READING_SWEEP_ROOT:-$PWD/outputs_torch/ucl_reading/embedding_sweep}"
UCL_READING_DATA_ROOT="${UCL_READING_DATA_ROOT:-$SCRATCH/Documents/benchmark/data/ucl_reading}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

ucl_reading_torch_check_data() {
  if [[ ! -f "${UCL_READING_DATA_ROOT}/stimuli.txt" || \
        ! -f "${UCL_READING_DATA_ROOT}/selfpacedreading.RT.txt" ]]; then
    echo "ERROR: UCL Reading data not found under ${UCL_READING_DATA_ROOT}"
    return 1
  fi
  return 0
}

ucl_reading_torch_setup_env() {
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
    "${UCL_READING_EMBED_CACHE_DIR}" \
    outputs_torch/logs
  if [[ -n "${UCL_READING_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${UCL_READING_FAMILY}"
  fi
}

ucl_reading_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${UCL_READING_FAMILY}"
}

# GPU: embedding extraction (cached) + per-layer ridge fit, one array task per
# (family, checkpoint). Already-computed checkpoints are skipped automatically.
ucl_reading_torch_run_array_task_gpu() {
  : "${UCL_READING_FAMILY:?Set UCL_READING_FAMILY before calling ucl_reading_torch_run_array_task_gpu}"
  local _n
  _n="$(ucl_reading_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${UCL_READING_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "UCL Reading embed ridge sweep (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${UCL_READING_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "UCL_READING_DATA_ROOT=${UCL_READING_DATA_ROOT}"
  echo "UCL_READING_EMBED_CACHE_DIR=${UCL_READING_EMBED_CACHE_DIR:-unset}"
  echo "UCL_READING_WORD_PREDICTIONS_ROOT=${UCL_READING_WORD_PREDICTIONS_ROOT:-unset (falls back to SWEEP_ROOT)}"
  echo "UCL_READING_AVERAGE_ACROSS_SUBJECTS=${UCL_READING_AVERAGE_ACROSS_SUBJECTS:-0}"

  ucl_reading_torch_check_data || exit 1
  ucl_reading_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${UCL_READING_FAMILY}" --index "${IDX}"

  local _avg_flag=()
  if [[ "${UCL_READING_AVERAGE_ONLY:-0}" == "1" ]]; then
    _avg_flag=(--average_only)
  elif [[ "${UCL_READING_AVERAGE_ACROSS_SUBJECTS:-0}" == "1" ]]; then
    _avg_flag=(--average_across_subjects)
  fi

  "${PY[@]}" scripts/run/embedding/behavioral/ucl_reading.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${UCL_READING_FAMILY}" \
    --output_family "${UCL_READING_OUTPUT_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${_avg_flag[@]}"

  echo "Job completed at $(date)"
}


ucl_reading_torch_run_summary() {
  : "${UCL_READING_FAMILY:?Set UCL_READING_FAMILY before calling ucl_reading_torch_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  ucl_reading_torch_setup_env
  "${PY[@]}" scripts/run/embedding/behavioral/ucl_reading.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${UCL_READING_FAMILY}" \
    --output_family "${UCL_READING_OUTPUT_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}" \
    --summary_out_dir "${UCL_READING_EMBED_SUMMARY_DIR:-${SWEEP_ROOT}}"
  echo "Summary completed at $(date)"
}
