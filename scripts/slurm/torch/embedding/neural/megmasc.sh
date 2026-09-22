# Shared runtime for Torch MEG-MASC MEG per-family checkpoint sweeps.
# Source from scripts/slurm/torch/megmasc_<family>.sh (do not run directly).
#
# Required before source:
#   MEGMASC_FAMILY   — registry key (gpt2, pythia, olmo2, …)
#
# Optional overrides:
#   CONFIG, SWEEP_ROOT, MEGMASC_BIDS_ROOT
#   BENCHMARK_CACHE_ROOT         — default: $SCRATCH/.cache on Torch
#   MEGMASC_FEATURE_CACHE_DIR    — default: $BENCHMARK_CACHE_ROOT/megmasc_features
#   MEGMASC_EPOCH_CACHE_DIR      — default: $BENCHMARK_CACHE_ROOT/megmasc_epochs
#   N_JOBS                       — joblib workers for the CPU fallback path (default 24)
#
# GPU model
# ---------
# Only 11 subjects — no need for a multi-GPU MPS (subject x layer) fan-out.
# Podcast-style: 1 GPU, -C h100 (no mps), no --exclusive. Subjects run
# sequentially; all 208 channels are batched per layer in one GPU call
# (cv_ridge_score_lag_matrix_gpu_batched). Prime the epoch cache first with
# scripts/slurm/torch/utils/data_prep/megmasc_prime_cache.sh so this job does no raw-MEG I/O.

: "${MEGMASC_FAMILY:?Set MEGMASC_FAMILY before sourcing job_megmasc_torch_common.sh}"

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1
export PYTHONPATH="${BENCHMARK_ROOT}/forgetting-transformer/src:${BENCHMARK_ROOT}/stickbreaking-attention${PYTHONPATH:+:$PYTHONPATH}"  # constrained-arch Lanni-ni models

if [[ -n "${SCRATCH:-}" ]]; then
  source "${_COMMON_DIR}/torch_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
  export MEGMASC_FEATURE_CACHE_DIR="${MEGMASC_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_features}"
  export MEGMASC_EPOCH_CACHE_DIR="${MEGMASC_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_epochs}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" \
    "$MEGMASC_FEATURE_CACHE_DIR" "$MEGMASC_EPOCH_CACHE_DIR"
fi

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/embedding/neural/megmasc.yaml}"
SWEEP_ROOT="${SWEEP_ROOT:-$PWD/outputs_torch/megmasc/embedding_sweep}"
export MEGMASC_BIDS_ROOT="${MEGMASC_BIDS_ROOT:-${SCRATCH:-$BENCHMARK_ROOT/..}/data/gwilliams}"
export MEGMASC_FEATURE_CACHE_DIR="${MEGMASC_FEATURE_CACHE_DIR:-${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}/megmasc_features}"
export MEGMASC_EPOCH_CACHE_DIR="${MEGMASC_EPOCH_CACHE_DIR:-${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}/megmasc_epochs}"
N_JOBS="${N_JOBS:-24}"
N_SUBJECTS_BATCH="${N_SUBJECTS_BATCH:-5}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-12}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

megmasc_torch_check_data() {
  local _missing=0
  if [[ ! -d "${MEGMASC_BIDS_ROOT}/sub-01/ses-0/meg" ]]; then
    echo "ERROR: MEG-MASC subject data not found: ${MEGMASC_BIDS_ROOT}/sub-01/ses-0/meg/"
    _missing=1
  fi
  if ! ls "${MEGMASC_BIDS_ROOT}"/sub-01/ses-0/meg/*_meg.con >/dev/null 2>&1; then
    echo "ERROR: no .con recordings under ${MEGMASC_BIDS_ROOT}/sub-01/ses-0/meg/"
    _missing=1
  fi
  return "$_missing"
}

megmasc_torch_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export SAFETENSORS_FAST_GPU=1   # memory-map safetensors directly to GPU (no CPU staging)
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  # Reduce GPU memory fragmentation: allows the allocator to grow segments
  # on demand rather than pre-allocating large contiguous blocks.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${MEGMASC_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_features}" \
    "${MEGMASC_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_epochs}" \
    "${SWEEP_ROOT}/${MEGMASC_FAMILY}" outputs_torch/logs
}

megmasc_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${MEGMASC_FAMILY}"
}

megmasc_torch_run_array_task() {
  local _n
  _n="$(megmasc_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] for family=${MEGMASC_FAMILY}; exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_job=${SLURM_ARRAY_JOB_ID:-?}  array_task=${IDX}"
  echo "family=${MEGMASC_FAMILY}  array_size=${_n}"
  echo "CONFIG=${CONFIG}"
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "MEGMASC_BIDS_ROOT=${MEGMASC_BIDS_ROOT}"
  echo "N_JOBS=${N_JOBS}"
  echo "nprocs: $(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
  nvidia-smi

  # Fail fast if the GPU isn't usable rather than falling back silently.
  _cuda_ok=0
  for _i in $(seq 1 30); do
    if "${PY[@]}" -c "import torch; torch.zeros(1).cuda()" 2>/dev/null; then
      _cuda_ok=1
      break
    fi
    sleep 1
  done
  if (( _cuda_ok == 0 )); then
    echo "ERROR: CUDA not available after 30s on node $(hostname)."
    exit 1
  fi
  echo "[CUDA] client OK"

  megmasc_torch_check_data || exit 1
  megmasc_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${MEGMASC_FAMILY}" --index "${IDX}"

  # GPU monitor: one line per GPU per minute — handles all 4 H100s.
  # nvidia-smi --query-gpu returns one row per GPU, so we parse with awk.
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

  # Podcast-style single-GPU path: 11 subjects is far too few to justify
  # multi-GPU MPS (subject x layer) fan-out. Sequential subjects
  # (--n_subjects_batch 0), all 208 channels batched per layer on one GPU via
  # cv_ridge_score_lag_matrix_gpu_batched (reached when regression_device=cuda,
  # use_lag_matrix, and save_word_predictions is off).
  "${PY[@]}" scripts/run/embedding/neural/megmasc.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${MEGMASC_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --out_dir "${SWEEP_ROOT}" \
    --n_jobs "${N_JOBS}" \
    --regression_device cuda \
    --n_subjects_batch 0 \
    --no-save_word_predictions \
    --epoch_cache_dir "${MEGMASC_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_epochs}" \
    --feature_cache_dir "${MEGMASC_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_features}" \
    --skip_summary \
    --no_plots

  kill "${_GPU_MONITOR_PID:-}" 2>/dev/null || true
  echo "Job completed at $(date)"
}

megmasc_torch_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  megmasc_torch_setup_env
  "${PY[@]}" scripts/run/embedding/neural/megmasc.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${MEGMASC_FAMILY}" \
    --summary_only \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
