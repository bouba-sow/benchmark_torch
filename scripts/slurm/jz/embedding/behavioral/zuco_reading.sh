# Shared runtime for Jean Zay ZuCo Reading (behavioral TRT) embedding-ridge
# checkpoint sweeps. Source from
# scripts/submit.py (see scripts/slurm/stages.yaml) — do not
# run directly.
#
# Single-stage design (unlike zuco_reading_lme's GPU/CPU split): ridge
# regression is plain sklearn, no R/pymer4 dependency, so embedding
# extraction + ridge fit both run in one GPU job.
#
# ZUCO_READING_FAMILY — registry key (gpt2, pythia, olmo2, …). Required for all jobs.
# ZUCO_TASK           — ZuCo reading task: SR | NR | TSR (default: NR). Selects
#                        both the config file
#                        (configs/embedding/behavioral/zuco_reading_<task>.yaml) and the
#                        sweep/cache subdirectories — SR/NR/TSR never share a
#                        cache dir because their (story_id, zone) ranges overlap
#                        (different sentence sets, unlike Provo/MECo's SP/FP/GP).
#
# Optional overrides:
#   CONFIG, ZUCO_READING_SWEEP_ROOT, ZUCO_READING_DATA_ROOT
#   BENCHMARK_CACHE_ROOT, ZUCO_READING_EMBED_CACHE_DIR
#   ZUCO_READING_AVERAGE_ACROSS_SUBJECTS=1 — additionally fit RT averaged
#     across subjects per item (see neuroai/benchmarks/subject_average.py);
#     written to a sibling "<ZUCO_READING_SWEEP_ROOT>_average" directory,
#     does not affect the normal per-subject output.
#
# NOTE: this reads ZUCO_READING_SWEEP_ROOT specifically, NOT the generic
# SWEEP_ROOT — a stale `export SWEEP_ROOT=...` left over from another
# dataset's session would otherwise leak in and silently write output into
# the wrong directory.

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
export ZUCO_TASK="${ZUCO_TASK:-NR}"
ZUCO_TASK_LOWER="$(echo "${ZUCO_TASK}" | tr '[:upper:]' '[:lower:]')"

export ZUCO_READING_OUTPUT_FAMILY="${ZUCO_READING_OUTPUT_FAMILY:-$ZUCO_READING_FAMILY}"

# On by default — off for the pythia_checkpoints sweep (110 tasks would write
# a per-word .npz per checkpoint; not needed there, see pythia_ckpt.sh).
export ZUCO_READING_SAVE_WORD_PREDICTIONS="${ZUCO_READING_SAVE_WORD_PREDICTIONS:-1}"
SAVE_WORD_PREDICTIONS_FLAG=()
if [[ "${ZUCO_READING_SAVE_WORD_PREDICTIONS}" == "1" ]]; then
  SAVE_WORD_PREDICTIONS_FLAG=(--save_word_predictions)
else
  SAVE_WORD_PREDICTIONS_FLAG=(--no-save_word_predictions)
fi

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
# Per-task subdirectory — SR/NR/TSR must never share a cache dir (see note above).
export ZUCO_READING_EMBED_CACHE_DIR="${ZUCO_READING_EMBED_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/zuco_reading_features}/${ZUCO_TASK_LOWER}"
# Per-word/per-item OOF prediction .npz files (save_word_predictions=True) go
# here instead of under SWEEP_ROOT (on $WORK) — many small files, better
# suited to $SCRATCH. Set to "" to fall back to the old behavior (saved
# alongside the CSVs under SWEEP_ROOT).
export ZUCO_READING_WORD_PREDICTIONS_ROOT="${ZUCO_READING_WORD_PREDICTIONS_ROOT-$BENCHMARK_CACHE_ROOT/zuco_reading_word_predictions}"
[[ -n "${ZUCO_READING_WORD_PREDICTIONS_ROOT}" ]] && mkdir -p "${ZUCO_READING_WORD_PREDICTIONS_ROOT}"
mkdir -p "${ZUCO_READING_EMBED_CACHE_DIR}"

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/embedding/behavioral/zuco_reading_${ZUCO_TASK_LOWER}.yaml}"
SWEEP_ROOT="${ZUCO_READING_SWEEP_ROOT:-$PWD/outputs_jeanzay/zuco_reading/embedding_sweep/${ZUCO_TASK_LOWER}}"
ZUCO_READING_DATA_ROOT="${ZUCO_READING_DATA_ROOT:-$WORK/benchmark/data/zuco_reading/osfstorage}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

zuco_reading_jz_check_data() {
  if [[ ! -d "${ZUCO_READING_DATA_ROOT}" ]]; then
    echo "ERROR: ZuCo data root not found: ${ZUCO_READING_DATA_ROOT}"
    return 1
  fi
  return 0
}

zuco_reading_jz_setup_env() {
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
    "${ZUCO_READING_EMBED_CACHE_DIR}" \
    outputs_jeanzay/logs
  if [[ -n "${ZUCO_READING_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${ZUCO_READING_FAMILY}"
  fi
}

zuco_reading_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${ZUCO_READING_FAMILY}"
}

# GPU: embedding extraction (cached) + per-layer ridge fit, one array task per
# (family, checkpoint). Already-computed checkpoints are skipped automatically.
zuco_reading_jz_run_array_task_gpu() {
  : "${ZUCO_READING_FAMILY:?Set ZUCO_READING_FAMILY before calling zuco_reading_jz_run_array_task_gpu}"
  local _n
  _n="$(zuco_reading_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${ZUCO_READING_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "ZuCo reading embed ridge sweep (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${ZUCO_READING_FAMILY}  task=${ZUCO_TASK}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "ZUCO_READING_DATA_ROOT=${ZUCO_READING_DATA_ROOT}"
  echo "ZUCO_READING_EMBED_CACHE_DIR=${ZUCO_READING_EMBED_CACHE_DIR:-unset}"
  echo "ZUCO_READING_WORD_PREDICTIONS_ROOT=${ZUCO_READING_WORD_PREDICTIONS_ROOT:-unset (falls back to SWEEP_ROOT)}"
  echo "ZUCO_READING_AVERAGE_ACROSS_SUBJECTS=${ZUCO_READING_AVERAGE_ACROSS_SUBJECTS:-0}"

  zuco_reading_jz_check_data || exit 1
  zuco_reading_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${ZUCO_READING_FAMILY}" --index "${IDX}"

  local _avg_flag=()
  if [[ "${ZUCO_READING_AVERAGE_ONLY:-0}" == "1" ]]; then
    _avg_flag=(--average_only)
  elif [[ "${ZUCO_READING_AVERAGE_ACROSS_SUBJECTS:-0}" == "1" ]]; then
    _avg_flag=(--average_across_subjects)
  fi

  # EXTRACT_ONLY=1 -- GPU half of the two-stage pipeline: populate the
  # embedding cache only, skip the ridge fit (see
  # zuco_reading_jz_run_array_task_cpu, the CPU half that reads this cache).
  local _extract_only_flag=()
  [[ "${EXTRACT_ONLY:-0}" == "1" ]] && _extract_only_flag=(--extract_only)

  "${PY[@]}" scripts/run/embedding/behavioral/zuco_reading.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${ZUCO_READING_FAMILY}" \
    --output_family "${ZUCO_READING_OUTPUT_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}" \
    "${_avg_flag[@]}" \
    "${_extract_only_flag[@]}"

  echo "Job completed at $(date)"
}

# CPU: ridge fit only, reusing the embedding cache a prior EXTRACT_ONLY=1 GPU
# job already populated -- contextual_word_embeddings_hf_fulltext checks that
# cache before touching torch/CUDA at all, so this needs no GPU whatsoever as
# long as the cache is warm (submit with --dependency=afterok on the GPU
# job). Same array-task/checkpoint resolution as the GPU function; only the
# device and the missing GPU-arch check differ.
zuco_reading_jz_run_array_task_cpu() {
  : "${ZUCO_READING_FAMILY:?Set ZUCO_READING_FAMILY before calling zuco_reading_jz_run_array_task_cpu}"
  local _n
  _n="$(zuco_reading_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${ZUCO_READING_FAMILY}; exiting."
    exit 0
  fi

  echo "ZuCo reading embed ridge fit (CPU-only, cache must already be warm)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${ZUCO_READING_FAMILY}  task=${ZUCO_TASK}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "ZUCO_READING_EMBED_CACHE_DIR=${ZUCO_READING_EMBED_CACHE_DIR:-unset}"

  zuco_reading_jz_check_data || exit 1
  zuco_reading_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${ZUCO_READING_FAMILY}" --index "${IDX}"

  local _avg_flag=()
  if [[ "${ZUCO_READING_AVERAGE_ONLY:-0}" == "1" ]]; then
    _avg_flag=(--average_only)
  elif [[ "${ZUCO_READING_AVERAGE_ACROSS_SUBJECTS:-0}" == "1" ]]; then
    _avg_flag=(--average_across_subjects)
  fi

  "${PY[@]}" scripts/run/embedding/behavioral/zuco_reading.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${ZUCO_READING_FAMILY}" \
    --output_family "${ZUCO_READING_OUTPUT_FAMILY}" \
    --array-index "${IDX}" \
    --device cpu \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}" \
    "${_avg_flag[@]}"

  echo "Job completed at $(date)"
}


zuco_reading_jz_run_summary() {
  : "${ZUCO_READING_FAMILY:?Set ZUCO_READING_FAMILY before calling zuco_reading_jz_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  zuco_reading_jz_setup_env
  "${PY[@]}" scripts/run/embedding/behavioral/zuco_reading.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${ZUCO_READING_FAMILY}" \
    --output_family "${ZUCO_READING_OUTPUT_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}" \
    --summary_out_dir "${ZUCO_READING_EMBED_SUMMARY_DIR:-${SWEEP_ROOT}}"
  echo "Summary completed at $(date)"
}
