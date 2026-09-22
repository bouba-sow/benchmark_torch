# Shared runtime for Jean Zay MECo Corpus (English subset) embedding-ridge checkpoint sweeps.
# Source from scripts/submit.py (see scripts/slurm/stages.yaml) — do not
# run directly.
#
# Single-stage design (unlike meco_lme's GPU/CPU split): ridge regression is
# plain sklearn, no R/pymer4 dependency, so embedding extraction + ridge fit
# both run in one GPU job.
#
# MECO_FAMILY   — registry key (gpt2, pythia, olmo2, …). Required for all jobs.
# MECO_MEASURE  — MECo eye-tracking measure: SP | FP | GP (default: FP). Selects
#                 the config file (configs/embedding/behavioral/meco_<measure>.yaml) and
#                 the sweep output subdirectory — but NOT the embedding cache dir:
#                 MECo's SP/FP/GP measures share the exact same word set, so the
#                 embedding cache built by any one MECO_MEASURE run is reused by
#                 the other two (see neuroai/benchmarks/meco_embed.py docstring).
# MECO_LANGUAGE — MECo language code (default: en). Only English is wired up by
#                 neuroai.datasets.meco today — see its module docstring.
#
# Optional overrides:
#   CONFIG, MECO_SWEEP_ROOT, MECO_DATA_ROOT, MECO_RDA_FILENAME
#   BENCHMARK_CACHE_ROOT, MECO_EMBED_CACHE_DIR
#   MECO_AVERAGE_ACROSS_SUBJECTS=1 — additionally fit RT averaged across
#     subjects per item (see neuroai/benchmarks/subject_average.py); written
#     to a sibling "<MECO_SWEEP_ROOT>_average" directory, does not affect
#     the normal per-subject output.
#
# NOTE: this reads MECO_SWEEP_ROOT specifically, NOT the generic SWEEP_ROOT —
# a stale `export SWEEP_ROOT=...` left over from another dataset's session
# would otherwise leak in and silently write output into the wrong directory.

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
export MECO_MEASURE="${MECO_MEASURE:-FP}"
export MECO_LANGUAGE="${MECO_LANGUAGE:-en}"
MECO_MEASURE_LOWER="$(echo "${MECO_MEASURE}" | tr '[:upper:]' '[:lower:]')"
MECO_RDA_FILENAME="${MECO_RDA_FILENAME:-joint_l1_data_trimmed_version2.1.rda}"

export MECO_OUTPUT_FAMILY="${MECO_OUTPUT_FAMILY:-$MECO_FAMILY}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT"
fi
# Shared across SP/FP/GP — deliberately NOT nested per measure (see note above).
export MECO_EMBED_CACHE_DIR="${MECO_EMBED_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/meco_embed_features}"
# Per-word/per-item OOF prediction .npz files (save_word_predictions=True) go
# here instead of under SWEEP_ROOT (on $WORK) — many small files, better
# suited to $SCRATCH. Set to "" to fall back to the old behavior (saved
# alongside the CSVs under SWEEP_ROOT).
export MECO_WORD_PREDICTIONS_ROOT="${MECO_WORD_PREDICTIONS_ROOT-$BENCHMARK_CACHE_ROOT/meco_word_predictions}"
[[ -n "${MECO_WORD_PREDICTIONS_ROOT}" ]] && mkdir -p "${MECO_WORD_PREDICTIONS_ROOT}"
mkdir -p "${MECO_EMBED_CACHE_DIR}"

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/embedding/behavioral/meco_${MECO_MEASURE_LOWER}.yaml}"
SWEEP_ROOT="${MECO_SWEEP_ROOT:-$PWD/outputs_jeanzay/meco/embedding_sweep/${MECO_MEASURE_LOWER}}"
MECO_DATA_ROOT="${MECO_DATA_ROOT:-$WORK/benchmark/data/meco/osfstorage/release 2.0/version 2.1/wave 1/primary data/eye tracking data}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

meco_jz_check_data() {
  if [[ ! -f "${MECO_DATA_ROOT}/${MECO_RDA_FILENAME}" ]]; then
    echo "ERROR: MECo .rda file not found: ${MECO_DATA_ROOT}/${MECO_RDA_FILENAME}"
    return 1
  fi
  return 0
}

meco_jz_setup_env() {
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
    "${MECO_EMBED_CACHE_DIR}" \
    outputs_jeanzay/logs
  if [[ -n "${MECO_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${MECO_FAMILY}"
  fi
}

meco_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${MECO_FAMILY}"
}

# GPU: embedding extraction (cached) + per-layer ridge fit, one array task per
# (family, checkpoint). Already-computed checkpoints are skipped automatically.
meco_jz_run_array_task_gpu() {
  : "${MECO_FAMILY:?Set MECO_FAMILY before calling meco_jz_run_array_task_gpu}"
  local _n
  _n="$(meco_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${MECO_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "MECO embed ridge sweep (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${MECO_FAMILY}  measure=${MECO_MEASURE}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "MECO_DATA_ROOT=${MECO_DATA_ROOT}"
  echo "MECO_EMBED_CACHE_DIR=${MECO_EMBED_CACHE_DIR:-unset}  (measure-agnostic — shared cache)"
  echo "MECO_WORD_PREDICTIONS_ROOT=${MECO_WORD_PREDICTIONS_ROOT:-unset (falls back to SWEEP_ROOT)}"
  echo "MECO_AVERAGE_ACROSS_SUBJECTS=${MECO_AVERAGE_ACROSS_SUBJECTS:-0}"
  echo "MECO_AVERAGE_ONLY=${MECO_AVERAGE_ONLY:-0}"

  meco_jz_check_data || exit 1
  meco_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${MECO_FAMILY}" --index "${IDX}"

  local _avg_flag=()
  if [[ "${MECO_AVERAGE_ONLY:-0}" == "1" ]]; then
    _avg_flag=(--average_only)
  elif [[ "${MECO_AVERAGE_ACROSS_SUBJECTS:-0}" == "1" ]]; then
    _avg_flag=(--average_across_subjects)
  fi

  "${PY[@]}" scripts/run/embedding/behavioral/meco.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${MECO_FAMILY}" \
    --output_family "${MECO_OUTPUT_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${_avg_flag[@]}"

  echo "Job completed at $(date)"
}


meco_jz_run_summary() {
  : "${MECO_FAMILY:?Set MECO_FAMILY before calling meco_jz_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  meco_jz_setup_env
  "${PY[@]}" scripts/run/embedding/behavioral/meco.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${MECO_FAMILY}" \
    --output_family "${MECO_OUTPUT_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}" \
    --summary_out_dir "${MECO_EMBED_SUMMARY_DIR:-${SWEEP_ROOT}}"
  echo "Summary completed at $(date)"
}
