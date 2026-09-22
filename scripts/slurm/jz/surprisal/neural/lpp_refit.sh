# Shared runtime for the LPP pooled-LME REFIT with saved word predictions: recovers per-ROI
# held-out surprisal scores for the ROI scaling figure (scripts/figures/figure_roi_scaling.py)
# without a full resweep. Reuses the cached surprisal and pooled table, so only the CPU-side
# mixed-model fit runs for one or two layers. Used through scripts/submit.py:
#
#   python scripts/submit.py lpp surprisal refit --family gpt2 --submit                      # last + best layer
#   python scripts/submit.py lpp surprisal refit --family gpt2 --env REFIT_LAYERS=last --submit   # last layer only
#
# REFIT_LAYERS=best (default): relies on maintenance/prep_lpp_refit_last.py + maintenance/prep_lpp_refit_best.py
#   having invalidated exactly the last-layer and best-layer row per model, so the sweep's resume logic
#   refits only those (at most two) layers; N_WORKERS defaults to 2.
# REFIT_LAYERS=last: adds --last_layer_only; needs maintenance/prep_lpp_refit_last.py; N_WORKERS defaults to 1.
# Run the prep script(s) FIRST. Not for pythia_checkpoints / babylm_gpt2_checkpoints.
#
# Families whose remote-code model imports `fla` (stickbreaking, alibi, forgetting, hard_{2,3,5}gram) cannot
# run on a CPU-only node (fla's Triton backend needs a CUDA driver just to import). Submit those with a
# GPU allocation: --account ldh@h100 --gres gpu:1 --constraint h100.
#
# Optional overrides: CONFIG, LPP_LME_SWEEP_ROOT, N_WORKERS, LPP_LME_FEATURE_CACHE_DIR (must already
# contain the model's logit-lens surprisal cache).

: "${LPP_FAMILY:?Set LPP_FAMILY before sourcing job_lpp_lme_lpp_refit_jz_common.sh}"

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
  export LPP_LME_FEATURE_CACHE_DIR="${LPP_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/lpp_lme_features}"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export LPP_LME_FEATURE_CACHE_DIR="${LPP_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/lpp_lme_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$LPP_LME_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/surprisal/neural/lpp.yaml}"
SWEEP_ROOT="${LPP_LME_SWEEP_ROOT:-$PWD/outputs_jeanzay/lpp/surprisal_sweep}"
REFIT_LAYERS="${REFIT_LAYERS:-best}"   # best = last + best layer, last = last layer only
if [[ "$REFIT_LAYERS" == "last" ]]; then N_WORKERS="${N_WORKERS:-1}"; else N_WORKERS="${N_WORKERS:-2}"; fi
LAST_ONLY_ARGS=(); [[ "$REFIT_LAYERS" == "last" ]] && LAST_ONLY_ARGS=(--last_layer_only)

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  source .venv/bin/activate
  PY=(python)
fi

lpp_refit_export_rpy2_env() {
  export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
  if [[ -n "${R_HOME:-}" && -d "${R_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

lpp_refit_load_r() {
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    lpp_refit_export_rpy2_env
    return 0
  fi
  if command -v module >/dev/null 2>&1; then
    module load r/4.5.2 2>/dev/null || module load r 2>/dev/null || module load R 2>/dev/null || true
  fi
  if command -v R >/dev/null 2>&1; then
    export R_HOME="${R_HOME:-$(R RHOME)}"
    lpp_refit_export_rpy2_env
    return 0
  fi
  if [[ -n "${R_HOME:-}" && -x "${R_HOME}/bin/R" ]]; then
    export PATH="${R_HOME}/bin:${PATH}"
    if command -v R >/dev/null 2>&1; then
      lpp_refit_export_rpy2_env
      return 0
    fi
  fi
  echo "ERROR: R not found (required by pymer4 / lme4). Load an R module before submitting."
  return 1
}

lpp_refit_check_data() {
  local _root="${BENCHMARK_ROOT}/data/lpp_fmri"
  if [[ ! -f "${_root}/annotation/EN/lppEN_word_information.csv" ]]; then
    echo "ERROR: LPP data not found — missing ${_root}/annotation/EN/lppEN_word_information.csv"
    return 1
  fi
  if [[ ! -d "${BENCHMARK_ROOT}/data/lpp_fmri_extra/average_subject/bold" ]]; then
    echo "ERROR: l-bg precomputed average-subject BOLD not found at ${BENCHMARK_ROOT}/data/lpp_fmri_extra/average_subject/bold"
    return 1
  fi
  if [[ ! -f "${BENCHMARK_ROOT}/data/lpp_fmri_extra/masks/mask_lpp_en.nii.gz" ]]; then
    echo "ERROR: LPP brain mask not found at ${BENCHMARK_ROOT}/data/lpp_fmri_extra/masks/mask_lpp_en.nii.gz"
    return 1
  fi
  if [[ ! -f "${BENCHMARK_ROOT}/data/lpp_fmri_extra/masks/allParcels-language-SN220-hgwm.nii" ]]; then
    echo "ERROR: Fedorenko SN220 atlas not found at ${BENCHMARK_ROOT}/data/lpp_fmri_extra/masks/allParcels-language-SN220-hgwm.nii"
    return 1
  fi
  return 0
}

lpp_refit_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$BENCHMARK_CACHE_ROOT" "${LPP_LME_FEATURE_CACHE_DIR}" \
    "${SWEEP_ROOT}/${LPP_FAMILY}" outputs_jeanzay/logs
}

lpp_refit_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${LPP_FAMILY}"
}

lpp_refit_run_array_task() {
  local _n
  _n="$(lpp_refit_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${LPP_FAMILY}; exiting."
    exit 0
  fi

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${LPP_FAMILY}  array_size=${_n}  (refit=${REFIT_LAYERS}, save_word_predictions)"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "LPP_LME_FEATURE_CACHE_DIR=${LPP_LME_FEATURE_CACHE_DIR}"
  echo "N_WORKERS=${N_WORKERS}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

  lpp_refit_check_data || exit 1
  lpp_refit_setup_env
  lpp_refit_load_r || exit 1

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${LPP_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/surprisal/neural/lpp_fit.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${LPP_FAMILY}" \
    --array-index "${IDX}" \
    --out_dir "${SWEEP_ROOT}" \
    --n_workers "${N_WORKERS}" \
    --save_word_predictions \
    "${LAST_ONLY_ARGS[@]}" \
    --device cpu \
    --feature_cache_dir "${LPP_LME_FEATURE_CACHE_DIR}"

  echo "Job completed at $(date)"
}
