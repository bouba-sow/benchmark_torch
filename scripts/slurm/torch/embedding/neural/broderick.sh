# Shared runtime for Torch Broderick EEG per-family checkpoint sweeps.
# Source from scripts/slurm/torch/broderick_<family>.sh (do not run directly).
#
# Required before source:
#   BRODERICK_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, BRODERICK_SWEEP_ROOT, BRODERICK_BIDS_ROOT
#   BENCHMARK_CACHE_ROOT        — default: $SCRATCH/.cache on Torch
#   BRODERICK_FEATURE_CACHE_DIR — default: $BENCHMARK_CACHE_ROOT/broderick_features
#   N_JOBS                      — joblib workers for per-channel regression (default 40)
#
# NOTE: this reads BRODERICK_SWEEP_ROOT specifically, NOT the generic SWEEP_ROOT —
# a stale `export SWEEP_ROOT=...` left over from another dataset's session would
# otherwise silently leak in and write output to the wrong place.
#
# CUDA MPS note (Jean Zay only — NOT available on Torch)
# ---------------------------------------------------------------------
# Subjects are processed via spawned worker processes (ProcessPoolExecutor);
# each spawned process creates its own CUDA context, and without MPS,
# context-switching serialises GPU kernels and the frequent init/teardown
# crashes the driver (CUDA error 805 "MPS not running").
#
# Jean Zay activated MPS at the SLURM level by passing -C mps and --exclusive
# to sbatch. Torch's `sinfo -o "%P %f"` lists no `mps` feature on any
# partition (checked 2026-09-22) — there is no SLURM-managed MPS daemon here,
# and manually starting one (nvidia-cuda-mps-control) is explicitly NOT a
# substitute (that caused crashes in this pipeline previously, on Jean Zay,
# when tried by hand).
#
# Consequence: N_SUBJECTS_BATCH is forced to 1 below (sequential per-subject,
# one CUDA context at a time) rather than the concurrent multi-subject batch
# JZ's H100 nodes could run under MPS. Slower per job, but avoids the crash.
# Revisit if Torch adds an MPS-capable partition/feature.

: "${BRODERICK_FAMILY:?Set BRODERICK_FAMILY before sourcing job_broderick_torch_common.sh}"

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

if [[ -n "${SCRATCH:-}" ]]; then
  source "${_COMMON_DIR}/torch_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export BRODERICK_FEATURE_CACHE_DIR="${BRODERICK_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_features}"
  export BRODERICK_EPOCH_CACHE_DIR="${BRODERICK_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_epochs}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$BRODERICK_FEATURE_CACHE_DIR" "$BRODERICK_EPOCH_CACHE_DIR"
fi

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/embedding/neural/broderick.yaml}"
SWEEP_ROOT="${BRODERICK_SWEEP_ROOT:-$PWD/outputs_torch/broderick/embedding_sweep}"
BRODERICK_BIDS_ROOT="${BRODERICK_BIDS_ROOT:-$SCRATCH/Documents/benchmark/data/broderick}"
N_JOBS="${N_JOBS:-40}"

# On by default — this sweep's output feeds the DLT-cost difficulty-split
# analysis, which needs each channel's best-layer out-of-fold predicted lag
# matrix (.npz per subject). Set SAVE_WORD_PREDICTIONS=0 to disable. Forces
# the sequential per-subject path in Python regardless of N_SUBJECTS_BATCH
# (OOF isn't wired through the multiprocess flat-batch path).
export SAVE_WORD_PREDICTIONS="${SAVE_WORD_PREDICTIONS:-1}"
SAVE_WORD_PREDICTIONS_FLAG=()
if [[ "${SAVE_WORD_PREDICTIONS}" == "1" ]]; then
  SAVE_WORD_PREDICTIONS_FLAG=(--save_word_predictions)
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

broderick_torch_check_data() {
  local _missing=0
  if [[ ! -f "${BRODERICK_BIDS_ROOT}/stimuli/audio01.TextGrid" ]]; then
    echo "ERROR: Broderick TextGrid not found: ${BRODERICK_BIDS_ROOT}/stimuli/audio01.TextGrid"
    _missing=1
  fi
  if [[ ! -d "${BRODERICK_BIDS_ROOT}/sub-001/eeg" ]]; then
    echo "ERROR: Broderick EEG not found: ${BRODERICK_BIDS_ROOT}/sub-001/eeg/"
    _missing=1
  fi
  return "$_missing"
}

broderick_torch_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export SAFETENSORS_FAST_GPU=1   # memory-map safetensors directly to GPU (no CPU staging)
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export MNE_USE_NUMBA=false

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${BRODERICK_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_features}" \
    "${BRODERICK_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_epochs}" \
    "${SWEEP_ROOT}/${BRODERICK_FAMILY}" outputs_torch/logs
}

broderick_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${BRODERICK_FAMILY}"
}

broderick_torch_run_array_task() {
  local _n
  _n="$(broderick_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${BRODERICK_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${BRODERICK_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "BRODERICK_BIDS_ROOT=${BRODERICK_BIDS_ROOT}"
  echo "N_JOBS=${N_JOBS}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
  nvidia-smi

  broderick_torch_check_data || exit 1
  broderick_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${BRODERICK_FAMILY}" --index "${IDX}"

  # Sanity check CUDA is up before the real work starts (no MPS wait-loop needed
  # here — see the CUDA MPS note above for why).
  if ! "${PY[@]}" -c "import torch; torch.zeros(1).cuda()" 2>/dev/null; then
    echo "ERROR: CUDA not available on node $(hostname)."
    exit 1
  fi

  # GPU monitor: one line per GPU per minute.
  # nvidia-smi --query-gpu returns one row per GPU, so we parse with awk.
  # No SLURM-managed MPS on Torch (see note above) -> forced sequential (1 subject
  # at a time per GPU context) rather than JZ's concurrent N_SUBJECTS_BATCH=4.
  N_SUBJECTS_BATCH="${N_SUBJECTS_BATCH:-1}"
  if [[ "${N_SUBJECTS_BATCH}" != "1" ]]; then
    echo "WARNING: N_SUBJECTS_BATCH=${N_SUBJECTS_BATCH} without MPS on Torch — spawned"
    echo "  processes will fight over one CUDA context and may crash (CUDA error 805)."
  fi
  (while true; do
    _ts="$(date '+%H:%M:%S')"
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null \
    | awk -v ts="$_ts" -F', ' \
        '{printf "[%s] GPU%s: util=%s%%  mem=%s/%s MiB\n", ts, $1, $2, $3, $4}'
    sleep 60
  done) &
  _GPU_MONITOR_PID=$!
  trap 'kill "${_GPU_MONITOR_PID:-}" 2>/dev/null || true' EXIT

  LAYER_CHUNK_FLAG=()
  if [[ -n "${LAYER_CHUNK_SIZE:-}" ]]; then
    LAYER_CHUNK_FLAG=(--layer_chunk_size "${LAYER_CHUNK_SIZE}")
  fi

  "${PY[@]}" scripts/run/embedding/neural/broderick.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${BRODERICK_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --n_jobs "${N_JOBS}" \
    --regression_device cuda \
    --n_layer_workers "${N_LAYER_WORKERS:-56}" \
    --n_subjects_batch "${N_SUBJECTS_BATCH}" \
    --epoch_cache_dir "${BRODERICK_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_epochs}" \
    --skip_summary \
    --no_plots \
    "${SAVE_WORD_PREDICTIONS_FLAG[@]}" \
    "${LAYER_CHUNK_FLAG[@]}"

  kill "${_GPU_MONITOR_PID:-}" 2>/dev/null || true
  echo "Job completed at $(date)"
}

broderick_torch_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  broderick_torch_setup_env
  "${PY[@]}" scripts/run/embedding/neural/broderick.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${BRODERICK_FAMILY}" \
    --summary_only \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
