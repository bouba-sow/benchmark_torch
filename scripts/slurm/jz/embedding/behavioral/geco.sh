# Shared runtime for Jean Zay GECO Corpus embedding-ridge checkpoint sweeps.
# Source from scripts/submit.py (see scripts/slurm/stages.yaml) — do not
# run directly.
#
# Single-stage design (unlike geco_lme's GPU/CPU split): ridge regression is
# plain sklearn, no R/pymer4 dependency, so embedding extraction + ridge fit
# both run in one GPU job.
#
# --gres=gpu:2 / --cpus-per-task=48 (not 1 GPU / 10 cpus): Jean Zay's H100
# nodes have no separate --mem knob — host RAM scales with cores requested
# (468GB / 96 cores per 4-GPU node, so 1 GPU's share = 24 cores ~= 117GB).
# _embed_words() holds EVERY layer's embedding for the full GECO word list
# (~53,500 words) in host RAM at once, before any per-subject/per-layer
# fitting even starts — that footprint scales with n_words x n_layers x
# hidden_dim, not with subject_n_jobs/n_jobs, so it OOM'd even on small
# models (e.g. Ministral-3B) at the 1-GPU/~49GB share. The 2nd GPU sits
# idle for compute; it's requested purely to double the RAM allocation.
#
# GECO_FAMILY  — registry key (gpt2, pythia, olmo2, …). Required for all jobs.
# GECO_MEASURE — GECO eye-tracking measure: FP | GP (default: FP; GECO has no
#                 raw fixation report, so no SP measure exists, unlike Provo).
#                 Selects the config file (configs/embedding/behavioral/geco_<measure>.yaml)
#                 and the sweep output subdirectory — but NOT the embedding cache
#                 dir: GECO's FP/GP measures share the exact same word set, so
#                 the embedding cache built by any one GECO_MEASURE run is reused
#                 by the other (see neuroai/benchmarks/geco_embed.py docstring).
#
# Optional overrides:
#   CONFIG, GECO_SWEEP_ROOT, GECO_DATA_ROOT
#   BENCHMARK_CACHE_ROOT, GECO_EMBED_CACHE_DIR
#   GECO_AVERAGE_ACROSS_SUBJECTS=1 — additionally fit RT averaged across
#     subjects per item (see neuroai/benchmarks/subject_average.py); written
#     to a sibling "<GECO_SWEEP_ROOT>_average" directory, does not affect
#     the normal per-subject output.
#
# NOTE: this reads GECO_SWEEP_ROOT specifically, NOT the generic SWEEP_ROOT —
# a stale `export SWEEP_ROOT=...` left over from another dataset's session
# would otherwise leak in and silently write output into the wrong directory.

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
export GECO_MEASURE="${GECO_MEASURE:-FP}"
GECO_MEASURE_LOWER="$(echo "${GECO_MEASURE}" | tr '[:upper:]' '[:lower:]')"

export GECO_OUTPUT_FAMILY="${GECO_OUTPUT_FAMILY:-$GECO_FAMILY}"

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
# Shared across FP/GP — deliberately NOT nested per measure (see note above).
export GECO_EMBED_CACHE_DIR="${GECO_EMBED_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/geco_embed_features}"
mkdir -p "${GECO_EMBED_CACHE_DIR}"
# Per-word/per-item OOF prediction .npz files (save_word_predictions=True) go
# here instead of under SWEEP_ROOT (on $WORK) — many small files, better
# suited to $SCRATCH. Set to "" to fall back to the old behavior (saved
# alongside the CSVs under SWEEP_ROOT).
export GECO_WORD_PREDICTIONS_ROOT="${GECO_WORD_PREDICTIONS_ROOT-$BENCHMARK_CACHE_ROOT/geco_word_predictions}"
[[ -n "${GECO_WORD_PREDICTIONS_ROOT}" ]] && mkdir -p "${GECO_WORD_PREDICTIONS_ROOT}"

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/embedding/behavioral/geco_${GECO_MEASURE_LOWER}.yaml}"
SWEEP_ROOT="${GECO_SWEEP_ROOT:-$PWD/outputs_jeanzay/geco/embedding_sweep/${GECO_MEASURE_LOWER}}"
GECO_DATA_ROOT="${GECO_DATA_ROOT:-$WORK/benchmark/data/geco}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

geco_jz_check_data() {
  if [[ ! -f "${GECO_DATA_ROOT}/MonolingualReadingData.xlsx" || \
        ! -f "${GECO_DATA_ROOT}/EnglishMaterial.xlsx" ]]; then
    echo "ERROR: GECO data not found under ${GECO_DATA_ROOT}"
    return 1
  fi
  return 0
}

geco_jz_setup_env() {
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
    "${GECO_EMBED_CACHE_DIR}" \
    outputs_jeanzay/logs
  if [[ -n "${GECO_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${GECO_FAMILY}"
  fi
}

geco_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${GECO_FAMILY}"
}

# GPU: embedding extraction (cached) + per-layer ridge fit, one array task per
# (family, checkpoint). Already-computed checkpoints are skipped automatically.
geco_jz_run_array_task_gpu() {
  : "${GECO_FAMILY:?Set GECO_FAMILY before calling geco_jz_run_array_task_gpu}"
  local _n
  _n="$(geco_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${GECO_FAMILY}; exiting."
    exit 0
  fi

  # GECO_DEVICE=cpu: CPU-partition run for jobs whose embeddings are already
  # fully cached (no model is ever loaded) -- ridge is CPU anyway.
  GECO_DEVICE="${GECO_DEVICE:-cuda}"
  if [[ "${GECO_DEVICE}" == "cuda" ]]; then
    "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1
  fi

  echo "GECO embed ridge sweep (device=${GECO_DEVICE})"
  echo "GECO_LAYER_CHUNK_SIZE=${GECO_LAYER_CHUNK_SIZE:-16 (default)}  GECO_EMBED_N_JOBS=${GECO_EMBED_N_JOBS:-cfg}"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${GECO_FAMILY}  measure=${GECO_MEASURE}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "GECO_DATA_ROOT=${GECO_DATA_ROOT}"
  echo "GECO_EMBED_CACHE_DIR=${GECO_EMBED_CACHE_DIR:-unset}  (measure-agnostic — shared cache)"
  echo "GECO_WORD_PREDICTIONS_ROOT=${GECO_WORD_PREDICTIONS_ROOT:-unset (falls back to SWEEP_ROOT)}"
  echo "GECO_AVERAGE_ACROSS_SUBJECTS=${GECO_AVERAGE_ACROSS_SUBJECTS:-0}"
  echo "GECO_AVERAGE_ONLY=${GECO_AVERAGE_ONLY:-0}"
  echo "GECO_AVERAGE_RIDGE_DEVICE=${GECO_AVERAGE_RIDGE_DEVICE:-cpu}"

  geco_jz_check_data || exit 1
  geco_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${GECO_FAMILY}" --index "${IDX}"

  local _avg_flag=()
  if [[ "${GECO_AVERAGE_ONLY:-0}" == "1" ]]; then
    # Implies --average_across_subjects; skip the (expensive) per-subject
    # fits entirely for models where they alone are too slow to finish
    # (e.g. opt-66b) — see GecoEmbedBenchmarkConfig.average_only docstring.
    _avg_flag=(--average_only)
  elif [[ "${GECO_AVERAGE_ACROSS_SUBJECTS:-0}" == "1" ]]; then
    _avg_flag=(--average_across_subjects)
  fi
  if [[ -n "${GECO_AVERAGE_RIDGE_DEVICE:-}" ]]; then
    _avg_flag+=(--average_ridge_device "${GECO_AVERAGE_RIDGE_DEVICE}")
  fi

  "${PY[@]}" scripts/run/embedding/behavioral/geco.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${GECO_FAMILY}" \
    --output_family "${GECO_OUTPUT_FAMILY}" \
    --array-index "${IDX}" \
    --device "${GECO_DEVICE}" \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${_avg_flag[@]}"

  echo "Job completed at $(date)"
}


geco_jz_run_summary() {
  : "${GECO_FAMILY:?Set GECO_FAMILY before calling geco_jz_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  geco_jz_setup_env
  "${PY[@]}" scripts/run/embedding/behavioral/geco.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${GECO_FAMILY}" \
    --output_family "${GECO_OUTPUT_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}" \
    --summary_out_dir "${GECO_EMBED_SUMMARY_DIR:-${SWEEP_ROOT}}"
  echo "Summary completed at $(date)"
}
