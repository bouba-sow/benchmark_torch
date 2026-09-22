# Shared runtime for Torch Fedorenko2016 checkpoint sweeps.
# Source from family wrappers under scripts/slurm/torch/fedorenko2016_<family>.sh.

set -euo pipefail

: "${SCRATCH:?Set SCRATCH to your Torch scratch project}"
export SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT" || exit 1

if [[ -n "${SCRATCH:-}" ]]; then
  # shellcheck source=scripts/slurm/torch/torch_cache_env.sh
  source "${_COMMON_DIR}/torch_cache_env.sh"
fi

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SAFETENSORS_FAST_GPU=1   # memory-map safetensors directly to GPU (no CPU staging)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-$BENCHMARK_ROOT/configs/embedding/neural/fedorenko2016.yaml}"
OUT_DIR="${OUT_DIR:-$BENCHMARK_ROOT/outputs_torch/fedorenko2016/embedding_sweep}"
mkdir -p "$OUT_DIR" outputs_torch/logs

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  # shellcheck source=/dev/null
  source .venv/bin/activate
  PY=(python)
fi

"${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1
echo "Job started at $(date)  node=$(hostname)  job=${SLURM_JOB_ID:-?}  task=${SLURM_ARRAY_TASK_ID:-0}"

if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
  "${PY[@]}" scripts/run/embedding/neural/fedorenko2016.py \
    --config "$CONFIG" \
    --family "${FEDORENKO_FAMILY}" \
    --summary_only \
    --out_dir "$OUT_DIR"
else
  "${PY[@]}" scripts/run/embedding/neural/fedorenko2016.py \
    --config "$CONFIG" \
    --from-registry \
    --family "${FEDORENKO_FAMILY}" \
    --array-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --device cuda \
    --regression_device cuda \
    --n_layer_workers 32 \
    --n_jobs 20 \
    --out_dir "$OUT_DIR"
fi

echo "Job completed at $(date)"

