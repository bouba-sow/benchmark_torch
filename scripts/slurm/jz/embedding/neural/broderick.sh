# Shared runtime for Jean Zay Broderick EEG per-family checkpoint sweeps.
# Source from scripts/slurm/jz/broderick_<family>.sh (do not run directly).
#
# Required before source:
#   BRODERICK_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, BRODERICK_SWEEP_ROOT, BRODERICK_BIDS_ROOT
#   BENCHMARK_CACHE_ROOT        — default: $SCRATCH/.cache on Jean Zay
#   BRODERICK_FEATURE_CACHE_DIR — default: $BENCHMARK_CACHE_ROOT/broderick_features
#   N_JOBS                      — joblib workers for per-channel regression (default 40)
#
# NOTE: this reads BRODERICK_SWEEP_ROOT specifically, NOT the generic SWEEP_ROOT —
# a stale `export SWEEP_ROOT=...` left over from another dataset's session would
# otherwise silently leak in and write output to the wrong place.
#
# CUDA MPS note
# -------------
# Subjects are processed via spawned worker processes (ProcessPoolExecutor);
# each spawned process creates its own CUDA context, and without MPS,
# context-switching serialises GPU kernels and the frequent init/teardown
# crashes the driver (CUDA error 805 "MPS not running").
#
# On Jean Zay, MPS is activated at the SLURM level by passing -C mps and
# --exclusive to sbatch. SLURM starts the MPS daemon before the job script
# runs and shuts it down cleanly on exit. The Python code is unchanged.
#
# DO NOT start the MPS daemon manually (nvidia-cuda-mps-control) — that is
# what caused crashes in this pipeline previously. The SLURM-managed daemon
# is the only supported path on Jean Zay (IDRIS documentation, June 2026).
#
# Billing note: --exclusive charges for the full node even if only one GPU is
# used. This is mandatory for MPS; there is no workaround.

: "${BRODERICK_FAMILY:?Set BRODERICK_FAMILY before sourcing job_broderick_jz_common.sh}"

set -euo pipefail

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

if [[ -n "${SCRATCH:-}" ]]; then
  source "${_COMMON_DIR}/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export BRODERICK_FEATURE_CACHE_DIR="${BRODERICK_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_features}"
  export BRODERICK_EPOCH_CACHE_DIR="${BRODERICK_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_epochs}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$BRODERICK_FEATURE_CACHE_DIR" "$BRODERICK_EPOCH_CACHE_DIR"
fi

mkdir -p outputs_jeanzay/logs

CONFIG="${CONFIG:-configs/embedding/neural/broderick.yaml}"
SWEEP_ROOT="${BRODERICK_SWEEP_ROOT:-$PWD/outputs_jeanzay/broderick/embedding_sweep}"
BRODERICK_BIDS_ROOT="${BRODERICK_BIDS_ROOT:-$WORK/benchmark/data/broderick}"
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

broderick_jz_check_data() {
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

broderick_jz_setup_env() {
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
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORK/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${BRODERICK_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_features}" \
    "${BRODERICK_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_epochs}" \
    "${SWEEP_ROOT}/${BRODERICK_FAMILY}" outputs_jeanzay/logs
}

broderick_jz_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${BRODERICK_FAMILY}"
}

broderick_jz_run_array_task() {
  local _n
  _n="$(broderick_jz_array_size)"
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

  broderick_jz_check_data || exit 1
  broderick_jz_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${BRODERICK_FAMILY}" --index "${IDX}"

  # Verify MPS is active (SLURM-managed) — fail fast rather than silently serialising.
  # If -C mps + --exclusive were omitted from the job header, CUDA from spawned processes
  # will get "Error 805: MPS daemon not running" and crash.
  _mps_ok=0
  for _i in $(seq 1 30); do
    if "${PY[@]}" -c "import torch; torch.zeros(1).cuda()" 2>/dev/null; then
      _mps_ok=1
      break
    fi
    sleep 1
  done
  if (( _mps_ok == 0 )); then
    echo "ERROR: CUDA not available after 30s on node $(hostname). Check -C mps + --exclusive."
    exit 1
  fi
  echo "[MPS] CUDA client OK (SLURM-managed MPS daemon)"

  # GPU monitor: one line per GPU per minute — handles all 4 H100s.
  # nvidia-smi --query-gpu returns one row per GPU, so we parse with awk.
  N_SUBJECTS_BATCH="${N_SUBJECTS_BATCH:-4}"
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

broderick_jz_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  broderick_jz_setup_env
  "${PY[@]}" scripts/run/embedding/neural/broderick.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${BRODERICK_FAMILY}" \
    --summary_only \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
