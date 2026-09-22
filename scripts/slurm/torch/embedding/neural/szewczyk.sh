# Shared runtime for Torch Szewczyk & Federmeier N400 EEG per-family checkpoint sweeps.
# Source from scripts/slurm/torch/szewczyk_<family>.sh — do not run directly.
#
# Required before source:
#   SZEWCZYK_FAMILY   — registry key (pythia, olmo2, gpt2, qwen3, llama32, mistral, …)
#
# Optional overrides:
#   CONFIG, SWEEP_ROOT
#   SZEWCZYK_FEATURE_CACHE_DIR   — default: $BENCHMARK_CACHE_ROOT/szewczyk_features
#   N_GPU_WORKERS                 — flattened (subject, layer) concurrent CUDA-stream
#                                    workers (default 32)
#
# Single-stage GPU pipeline:
#   LLM forward pass (~2,300 items x all checkpoints) on H100, then the
#   flattened (subject, layer) ridge sweep (~138 subjects x n_layers
#   independent single-output fits, dispatched concurrently — see
#   neuroai/benchmarks/n400_szewczyk.py docstring). Feature .npz files are
#   cached so re-runs skip the forward pass.

: "${SZEWCZYK_FAMILY:?Set SZEWCZYK_FAMILY before sourcing job_szewczyk_torch_common.sh}"

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
  export SZEWCZYK_FEATURE_CACHE_DIR="${SZEWCZYK_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/szewczyk_features}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  mkdir -p "$UV_CACHE_DIR" "$BENCHMARK_CACHE_ROOT" "$SZEWCZYK_FEATURE_CACHE_DIR"
fi

mkdir -p outputs_torch/logs

CONFIG="${CONFIG:-configs/embedding/neural/szewczyk.yaml}"
SWEEP_ROOT="${SWEEP_ROOT:-$PWD/outputs_torch/szewczyk/embedding_sweep}"
N_GPU_WORKERS="${N_GPU_WORKERS:-32}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

szewczyk_torch_check_data() {
  local _data="${SZEWCZYK_DATA_PATH:-$BENCHMARK_ROOT/data/szewczyk/osfstorage/bind-5-dset_300-500.rds}"
  if [[ ! -f "${_data}" ]]; then
    echo "ERROR: Szewczyk .rds not found: ${_data}"
    echo "  Set SZEWCZYK_DATA_PATH or place data at data/szewczyk/osfstorage/bind-5-dset_300-500.rds"
    return 1
  fi
}

szewczyk_torch_setup_env() {
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export SAFETENSORS_FAST_GPU=1   # memory-map safetensors directly to GPU (no CPU staging)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # BLAS thread cap. Originally set to cpus-per-task on the assumption the
  # only CPU-heavy work is the small sequential OOF-prediction pass at the
  # end (sklearn RidgeCV) -- but the main (subject, layer) flattened sweep
  # (N_GPU_WORKERS=32 concurrent ThreadPoolExecutor tasks, each doing some
  # CPU-side numpy/sklearn work per fit) runs under the SAME process-wide
  # env var. With cpus-per-task=24, that's up to 32*24=768 BLAS threads
  # fighting over 24 real cores -- confirmed empirically (2026-09-13):
  # bloom/rwkv4 szewczyk_embed jobs at cpus=24 sat with ZERO subject-level
  # progress for 75+ minutes on the "3450 ridge fits, 32 concurrent workers"
  # step, while an otherwise-identical gpt2 job at cpus=8 finished a whole
  # checkpoint (138 subjects) in well under a minute. Each individual fit is
  # tiny (<=300 rows, single-output) so BLAS threading barely helps a single
  # fit anyway -- the real parallelism comes from N_GPU_WORKERS. Cap at 2
  # regardless of cpus-per-task; the sequential OOF-prediction pass (only 2
  # cheap passes per subject) doesn't need more.
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2

  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" "$BENCHMARK_CACHE_ROOT" \
    "${SZEWCZYK_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/szewczyk_features}" \
    "${SWEEP_ROOT}/${SZEWCZYK_FAMILY}" outputs_torch/logs
}

szewczyk_torch_array_size() {
  "${PY[@]}" scripts/slurm/family_tasks.py array-size --family "${SZEWCZYK_FAMILY}"
}

# GPU job: LLM forward pass (cached) + flattened (subject, layer) ridge sweep.
szewczyk_torch_run_gpu() {
  local _n
  _n="$(szewczyk_torch_array_size)"
  if (( IDX < 0 || IDX >= _n )); then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((_n - 1))] — exiting."
    exit 0
  fi

  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

  echo "Szewczyk N400 — GPU (H100) feature extraction + CPU (ProcessPoolExecutor) flattened ridge sweep"
  echo "Job started at $(date)"
  echo "Node: $(hostname)  array_task=${IDX}  family=${SZEWCZYK_FAMILY}"
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "N_GPU_WORKERS=${N_GPU_WORKERS}"

  szewczyk_torch_check_data || exit 1
  szewczyk_torch_setup_env

  "${PY[@]}" scripts/slurm/family_tasks.py resolve \
    --family "${SZEWCZYK_FAMILY}" --index "${IDX}"

  "${PY[@]}" scripts/run/embedding/neural/szewczyk.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${SZEWCZYK_FAMILY}" \
    --array-index "${IDX}" \
    --device cuda \
    --regression_device cpu \
    --out_dir "${SWEEP_ROOT}" \
    --feature_cache_dir "${SZEWCZYK_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/szewczyk_features}" \
    --n_gpu_workers "${N_GPU_WORKERS}" \
    --skip_summary \
    --no_plots

  echo "Job completed at $(date)"
}

szewczyk_torch_run_summary() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "SUMMARY_ONLY=1: only array task 0 runs summary — exiting."
    exit 0
  fi
  szewczyk_torch_setup_env
  "${PY[@]}" scripts/run/embedding/neural/szewczyk.py \
    --config "${CONFIG}" \
    --from-registry \
    --family "${SZEWCZYK_FAMILY}" \
    --summary_only \
    --out_dir "${SWEEP_ROOT}"
  echo "Summary completed at $(date)"
}
