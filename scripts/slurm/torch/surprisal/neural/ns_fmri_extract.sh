# Shared runtime for Torch Natural Stories fMRI LME checkpoint sweeps.
# Used through scripts/submit.py (see scripts/slurm/stages.yaml); do not run directly.
#
# GPU nodes on Torch have no R module (pymer4/lme4 only works on the CPU partition), so the
# GPU stage (this file) only extracts and caches surprisal, and the LME fit
# (ns_fmri_fit.sh) always runs on CPU.
#
# NS_FAMILY: registry key (gpt2, pythia, olmo2, ...).
# Optional overrides: CONFIG, NS_FMRI_LME_SWEEP_ROOT, NS_FMRI_DATA_ROOT, BENCHMARK_CACHE_ROOT,
#   NS_FEATURE_CACHE_DIR, N_WORKERS (concurrent (fROI, layer) fits, default 20), LAYER_STRIDE
#   (fit every Nth layer), UNTRAINED=1 (untrained-baseline path, results under a separate
#   "_untrained" tree; UNTRAINED_SEED, default 0), LAST_LAYER_ONLY=1.
# NS_FMRI_LME_SWEEP_ROOT is read specifically, NOT the generic SWEEP_ROOT: a stale export from
# another dataset's session would otherwise leak in and write into the wrong directory.
# Unlike the Natural Stories SPR LME there is no unigram baseline here (Lin & Schuler's fMRI
# formula uses surprisal + word length + sentence position only).

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export NS_FEATURE_CACHE_DIR="${NS_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/naturalstories_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$NS_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/surprisal/neural/ns_fmri.yaml}"
UNTRAINED="${UNTRAINED:-0}"
UNTRAINED_SEED="${UNTRAINED_SEED:-0}"
SWEEP_ROOT="${NS_FMRI_LME_SWEEP_ROOT:-$PWD/outputs_torch/ns_fmri/surprisal_sweep}"
NS_FMRI_DATA_ROOT="${NS_FMRI_DATA_ROOT:-$SCRATCH/Documents/benchmark/data/naturalstories_fmri/osfstorage}"
N_WORKERS="${N_WORKERS:-20}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"
LAST_LAYER_ONLY="${LAST_LAYER_ONLY:-0}"

# NOTE: --out_dir passed to the python script below (and NS_FEATURE_CACHE_DIR,
# which the YAML config's cache.features_dir expands to) are the *plain*
# (non-suffixed) paths -- run/surprisal/neural/ns_fmri_fit.py itself appends "_untrained"
# when --untrained is set. Don't also suffix here, or the two suffixes stack
# (…_untrained_untrained).
UNTRAINED_ARGS=()
if [[ "${UNTRAINED}" == "1" ]]; then
  UNTRAINED_ARGS=(--untrained --untrained_seed "${UNTRAINED_SEED}")
fi

LAST_LAYER_ONLY_ARGS=()
if [[ "${LAST_LAYER_ONLY}" == "1" ]]; then
  LAST_LAYER_ONLY_ARGS=(--last_layer_only)
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

ns_fmri_lme_torch_setup_base_env() {
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
    "${NS_FEATURE_CACHE_DIR}" \
    outputs_torch/logs
  if [[ -n "${NS_FAMILY:-}" ]]; then
    mkdir -p "${SWEEP_ROOT}/${NS_FAMILY}"
  fi
}

# GPU: compute and cache surprisal (all 8 documents) for one array task (one registry model/checkpoint).
# Cache hit = the model is never loaded; cache miss = forward pass + write + free the GPU.
ns_fmri_surprisal_torch_run_array_task() {
  : "${NS_FAMILY:?Set NS_FAMILY before calling ns_fmri_surprisal_torch_run_array_task}"
  IDX="${SLURM_ARRAY_TASK_ID:-0}"

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "NS-fMRI surprisal extraction (GPU)"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_task=${IDX}  family=${NS_FAMILY}"
  echo "NS_FMRI_DATA_ROOT=${NS_FMRI_DATA_ROOT}"
  echo "NS_FEATURE_CACHE_DIR=${NS_FEATURE_CACHE_DIR}"

  ns_fmri_lme_torch_setup_base_env

  "${PY[@]}" scripts/run/surprisal/neural/ns_fmri_extract.py \
    --config "${CONFIG}" \
    --family "${NS_FAMILY}" \
    --array-index "${IDX}" \
    --data-root "${NS_FMRI_DATA_ROOT}" \
    --cache-dir "${NS_FEATURE_CACHE_DIR}" \
    --device cuda

  echo "Extraction completed at $(date)"
}
