# Shared runtime for Jean Zay OneStop Corpus embedding-ridge checkpoint sweeps.
# Source from scripts/submit.py (see scripts/slurm/stages.yaml) — do not
# run directly.
#
# Single-stage design (unlike the onestop_lme GPU/CPU split): ridge regression is
# plain sklearn, no R/pymer4 dependency, so embedding extraction + ridge fit
# both run in one GPU job.
#
# --gres=gpu:1 / --cpus-per-task=24 for every family EXCEPT opt_66b.sh (2 GPUs,
# needed for the device_map-sharded forward pass of facebook/opt-66b):
#
#   IMPORTANT — do NOT bump these to 2 GPUs for the smaller families. When
#   contextual_word_embeddings_hf_fulltext() sees >1 GPU *and* the model fits
#   on one (est_model_gib <= ~70), it takes the `use_multigpu` branch, which
#   spawns fresh ProcessPoolExecutor workers that call from_pretrained() FROM
#   SCRATCH — and this function is called once per OneStop passage (~324x). The
#   per-passage model reload turned even a 2-layer ALiBi model into a >4h
#   extraction that hit the 2h wall. With 1 GPU the resident-model cache
#   (_get_cached_causal_lm) is used and the whole corpus extracts in minutes.
#   opt-66b is >70 GiB so `use_multigpu` is False even with 2 GPUs -> it uses
#   the resident device_map path, no reload -> safe on 2 GPUs.
#
# Averaged-only mode does NOT hold per-subject embedding copies in host RAM
# (the old per-subject-fit rationale for 2 GPUs), so 24 cores / ~117 GB is
# plenty for _embed_words()'s single corpus-wide accumulation.
#
# ONESTOP_FAMILY  — registry key (gpt2, pythia, olmo2, …). Required for all jobs.
# ONESTOP_MEASURE — OneStop eye-tracking measure: FP | GP | SP (default: FP).
#                 Selects the config file (configs/embedding/behavioral/onestop_<measure>.yaml)
#                 and the sweep output subdirectory — but NOT the embedding cache
#                 dir: OneStop's SP/FP/GP measures share the exact same word set,
#                 so the embedding cache built by any one ONESTOP_MEASURE run is
#                 reused by the others (see neuroai/benchmarks/onestop_embed.py).
#
# Optional overrides:
#   CONFIG, ONESTOP_SWEEP_ROOT, ONESTOP_DATA_ROOT
#   BENCHMARK_CACHE_ROOT, ONESTOP_EMBED_CACHE_DIR
#   ONESTOP_AVERAGE_ACROSS_SUBJECTS=1 — additionally fit RT averaged across
#     subjects per item (see neuroai/benchmarks/subject_average.py); written
#     to a sibling "<ONESTOP_SWEEP_ROOT>_average" directory, does not affect
#     the normal per-subject output.
#
# NOTE: this reads ONESTOP_SWEEP_ROOT specifically, NOT the generic SWEEP_ROOT —
# a stale `export SWEEP_ROOT=...` left over from another dataset's session
# would otherwise leak in and silently write output into the wrong directory.

set -euo pipefail
# Fit only the final layer (2026-10-12; caches keep every layer). EMBED_LAST_LAYER_ONLY=0 for the full sweep.
export EMBED_LAST_LAYER_ONLY="${EMBED_LAST_LAYER_ONLY:-1}"

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
export ONESTOP_MEASURE="${ONESTOP_MEASURE:-FP}"
ONESTOP_MEASURE_LOWER="$(echo "${ONESTOP_MEASURE}" | tr '[:upper:]' '[:lower:]')"

export ONESTOP_OUTPUT_FAMILY="${ONESTOP_OUTPUT_FAMILY:-$ONESTOP_FAMILY}"

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
export ONESTOP_EMBED_CACHE_DIR="${ONESTOP_EMBED_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/onestop_embed_features}"
mkdir -p "${ONESTOP_EMBED_CACHE_DIR}"
# Per-word/per-item OOF prediction .npz files (save_word_predictions=True) go
# here instead of under SWEEP_ROOT (on $WORK) — many small files, better
# suited to $SCRATCH. Set to "" to fall back to the old behavior (saved
# alongside the CSVs under SWEEP_ROOT).
export ONESTOP_WORD_PREDICTIONS_ROOT="${ONESTOP_WORD_PREDICTIONS_ROOT-$BENCHMARK_CACHE_ROOT/onestop_word_predictions}"
[[ -n "${ONESTOP_WORD_PREDICTIONS_ROOT}" ]] && mkdir -p "${ONESTOP_WORD_PREDICTIONS_ROOT}"

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/embedding/behavioral/onestop_${ONESTOP_MEASURE_LOWER}.yaml}"
SWEEP_ROOT="${ONESTOP_SWEEP_ROOT:-$PWD/outputs_jeanzay/onestop/embedding_sweep/${ONESTOP_MEASURE_LOWER}}"
ONESTOP_DATA_ROOT="${ONESTOP_DATA_ROOT:-$WORK/benchmark/data/onestop}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

onestop_jz_check_data() {
  if [[ ! -f "${ONESTOP_DATA_ROOT}/precomputed_reading_measures/ia_Paragraph.csv" ]]; then
    echo "ERROR: OneStop data not found under ${ONESTOP_DATA_ROOT}"
    return 1
  fi
  return 0
}

onestop_jz_setup_env() {
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
    "${ONESTOP_EMBED_CACHE_DIR}" \
    outputs_jeanzay/logs
  if [[ -n "${ONESTOP_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${ONESTOP_FAMILY}"
  fi
}

onestop_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${ONESTOP_FAMILY}"
}

# GPU: embedding extraction (cached) + per-layer ridge fit, one array task per
# (family, checkpoint). Already-computed checkpoints are skipped automatically.
onestop_jz_run_array_task_gpu() {
  : "${ONESTOP_FAMILY:?Set ONESTOP_FAMILY before calling onestop_jz_run_array_task_gpu}"
  local _n
  _n="$(onestop_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${ONESTOP_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "OneStop embed ridge sweep (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${ONESTOP_FAMILY}  measure=${ONESTOP_MEASURE}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "ONESTOP_DATA_ROOT=${ONESTOP_DATA_ROOT}"
  echo "ONESTOP_EMBED_CACHE_DIR=${ONESTOP_EMBED_CACHE_DIR:-unset}  (measure-agnostic — shared cache)"
  echo "ONESTOP_WORD_PREDICTIONS_ROOT=${ONESTOP_WORD_PREDICTIONS_ROOT:-unset (falls back to SWEEP_ROOT)}"
  echo "ONESTOP_AVERAGE_ACROSS_SUBJECTS=${ONESTOP_AVERAGE_ACROSS_SUBJECTS:-0}"
  echo "ONESTOP_AVERAGE_ONLY=${ONESTOP_AVERAGE_ONLY:-1}"
  echo "ONESTOP_AVERAGE_RIDGE_DEVICE=${ONESTOP_AVERAGE_RIDGE_DEVICE:-cuda}"

  onestop_jz_check_data || exit 1
  onestop_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${ONESTOP_FAMILY}" --index "${IDX}"

  # OneStop embed is grand-average-subject-only by design (de Varda et al. 2026):
  # ONESTOP_AVERAGE_ONLY defaults to 1. The shipped configs also set
  # 'average_only: true', so even ONESTOP_AVERAGE_ONLY=0 stays averaged-only
  # unless the config is edited.
  local _avg_flag=()
  if [[ "${ONESTOP_AVERAGE_ONLY:-1}" == "1" ]]; then
    _avg_flag=(--average_only)
  elif [[ "${ONESTOP_AVERAGE_ACROSS_SUBJECTS:-0}" == "1" ]]; then
    _avg_flag=(--average_across_subjects)
  fi
  if [[ -n "${ONESTOP_AVERAGE_RIDGE_DEVICE:-cuda}" ]]; then
    _avg_flag+=(--average_ridge_device "${ONESTOP_AVERAGE_RIDGE_DEVICE:-cuda}")
  fi

  "${PY[@]}" scripts/run/embedding/behavioral/onestop.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${ONESTOP_FAMILY}" \
    --output_family "${ONESTOP_OUTPUT_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --skip_summary \
    --no_plots \
    "${_avg_flag[@]}"

  echo "Job completed at $(date)"
}


onestop_jz_run_summary() {
  : "${ONESTOP_FAMILY:?Set ONESTOP_FAMILY before calling onestop_jz_run_summary}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  onestop_jz_setup_env
  # Averaged-only: real scores live in the sibling "<SWEEP_ROOT>_average" tree.
  "${PY[@]}" scripts/run/embedding/behavioral/onestop.py \
    --config "${CONFIG}" \
    --summary_only \
    --family "${ONESTOP_FAMILY}" \
    --output_family "${ONESTOP_OUTPUT_FAMILY}" \
    --sweep_dir "${SWEEP_ROOT}_average" \
    --summary_out_dir "${ONESTOP_EMBED_SUMMARY_DIR:-${SWEEP_ROOT}_average}"
  echo "Summary completed at $(date)"
}
