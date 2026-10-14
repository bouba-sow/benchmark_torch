# Shared runtime for Jean Zay LPP per-family checkpoint sweeps.
# Source from scripts/slurm/jz/lpp_<family>.sh (do not run directly).
#
# Required before source:
#   LPP_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, SWEEP_ROOT, LPP_BIDS_ROOT, LBG_ROOT
#   RIDGE_N_JOBS, LAYER_N_JOBS, REGRESSION_DEVICE, N_LAYER_WORKERS, SLURM_ARRAY_MAX
#   BENCHMARK_CACHE_ROOT  — default: $SCRATCH/.cache on Jean Zay
#   LPP_FEATURE_CACHE_DIR — default: $BENCHMARK_CACHE_ROOT/lpp_features
#
# UNTRAINED=1 — run the untrained-baseline path instead of pretrained weights
#   (random init, same architecture, via --untrained). Results and the
#   feature cache are written under a separate "_untrained" tree so they
#   never mix with trained-model runs. Set UNTRAINED_SEED to change the
#   random-init seed (default 0).
#   UNTRAINED=1 python scripts/submit.py lpp embedding fit --family gpt2
#
# NOTE: --out_dir/--feature_cache_dir passed to the python script below (and
# LPP_FEATURE_CACHE_DIR) are the *plain* (non-suffixed) paths --
# run/embedding/neural/lpp.py itself appends "_untrained" when --untrained is
# set. Don't also suffix here, or the two suffixes stack (…_untrained_untrained).

: "${LPP_FAMILY:?Set LPP_FAMILY before sourcing job_lpp_jz_common.sh}"

set -euo pipefail
# Fit only the final layer (2026-10-12; caches keep every layer). EMBED_LAST_LAYER_ONLY=0 for the full sweep.
export EMBED_LAST_LAYER_ONLY="${EMBED_LAST_LAYER_ONLY:-1}"

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/jz/jz_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export LPP_FEATURE_CACHE_DIR="${LPP_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/lpp_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$LPP_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/embedding/neural/lpp.yaml}"
UNTRAINED="${UNTRAINED:-0}"
UNTRAINED_SEED="${UNTRAINED_SEED:-0}"
SWEEP_ROOT="${SWEEP_ROOT:-$PWD/outputs_jeanzay/lpp/embedding_sweep}"
LPP_BIDS_ROOT="${LPP_BIDS_ROOT:-$WORK/benchmark/data/lpp_fmri}"
LBG_ROOT="${LBG_ROOT:-$WORK/benchmark/data/lpp_fmri_extra}"

RIDGE_N_JOBS="${RIDGE_N_JOBS:-1}"
LAYER_N_JOBS="${LAYER_N_JOBS:-1}"
REGRESSION_DEVICE="${REGRESSION_DEVICE:-cuda}"
N_LAYER_WORKERS="${N_LAYER_WORKERS:-8}"
SLURM_ARRAY_MAX="${SLURM_ARRAY_MAX:-4}"

UNTRAINED_ARGS=()
UNTRAINED_REGISTRY_FLAG=()
if [[ "${UNTRAINED}" == "1" ]]; then
  UNTRAINED_ARGS=(--untrained --untrained_seed "${UNTRAINED_SEED}")
  UNTRAINED_REGISTRY_FLAG=(--untrained)
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

lpp_jz_check_data() {
  local _missing=0
  if [[ ! -d "${LPP_BIDS_ROOT}/annotation" ]]; then
    echo "ERROR: LPP BIDS not found (need annotation/): ${LPP_BIDS_ROOT}"
    _missing=1
  fi
  if [[ ! -f "${LBG_ROOT}/masks/mask_lpp_en.nii.gz" ]]; then
    echo "ERROR: brain mask missing: ${LBG_ROOT}/masks/mask_lpp_en.nii.gz"
    _missing=1
  fi
  if [[ ! -d "${LBG_ROOT}/average_subject" ]]; then
    echo "ERROR: precomputed averages missing: ${LBG_ROOT}/average_subject"
    _missing=1
  fi
  return "$_missing"
}

lpp_jz_setup_env() {
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
  export NILEARN_DATA="${NILEARN_DATA:-$WORK/nilearn_data}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export LPP_CHECKPOINT_SHARD=1

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$NILEARN_DATA" "$BENCHMARK_CACHE_ROOT" \
    "$LPP_FEATURE_CACHE_DIR" "${SWEEP_ROOT}/${LPP_FAMILY}" outputs_jeanzay/logs
}

lpp_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${LPP_FAMILY}" "${UNTRAINED_REGISTRY_FLAG[@]}"
}

lpp_jz_run_array_task() {
  local _n
  _n="$(lpp_jz_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${LPP_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${LPP_FAMILY}  array_size=${_n}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
  nvidia-smi

  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT} (effective out_dir gets _untrained suffix from the python script when UNTRAINED=1)"
  echo "UNTRAINED=${UNTRAINED} UNTRAINED_SEED=${UNTRAINED_SEED}"
  echo "LPP_BIDS_ROOT=${LPP_BIDS_ROOT}"
  echo "RIDGE_N_JOBS=${RIDGE_N_JOBS}  LAYER_N_JOBS=${LAYER_N_JOBS}"
  echo "REGRESSION_DEVICE=${REGRESSION_DEVICE}  N_LAYER_WORKERS=${N_LAYER_WORKERS}"
  echo "BENCHMARK_CACHE_ROOT=${BENCHMARK_CACHE_ROOT}"
  echo "LPP_FEATURE_CACHE_DIR=${LPP_FEATURE_CACHE_DIR}"

  lpp_jz_check_data || exit 1
  lpp_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve --family "${LPP_FAMILY}" --index "${IDX}" "${UNTRAINED_REGISTRY_FLAG[@]}"

  # LPP_DEVICE defaults to cuda; set to cpu when the feature cache is warm
  # and you want to run on a CPU-only partition (no GPU needed for cached features).
  _lpp_device="${LPP_DEVICE:-cuda}"

  "${PY[@]}" scripts/run/embedding/neural/lpp.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${LPP_FAMILY}" \
    --array-index "${IDX}" \
    --device "${_lpp_device}" \
    --out_dir "${SWEEP_ROOT}" \
    --ridge_n_jobs "${RIDGE_N_JOBS}" \
    --layer_n_jobs "${LAYER_N_JOBS}" \
    --regression_device "${REGRESSION_DEVICE}" \
    --n_layer_workers "${N_LAYER_WORKERS}" \
    --skip_summary \
    --no_plots \
    "${UNTRAINED_ARGS[@]}"

  echo "Job completed at $(date)"
}

lpp_jz_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  lpp_jz_setup_env
  "${PY[@]}" scripts/run/embedding/neural/lpp.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${LPP_FAMILY}" \
    --summary_only \
    --out_dir "${SWEEP_ROOT}" \
    "${UNTRAINED_ARGS[@]}"
  # summarize/lpp_families.py has no --untrained suffixing of its own, so pass
  # the already-effective directory explicitly (same "_untrained" rule as
  # run/embedding/neural/lpp.py's --out_dir handling).
  _summary_sweep_root="${SWEEP_ROOT}"
  if [[ "${UNTRAINED}" == "1" ]]; then
    _summary_sweep_root="${SWEEP_ROOT}_untrained"
  fi
  "${PY[@]}" scripts/summarize/lpp_families.py \
    --sweep_dir "${_summary_sweep_root}" \
    --families "${LPP_FAMILY}"
  echo "Summary completed at $(date)"
}
