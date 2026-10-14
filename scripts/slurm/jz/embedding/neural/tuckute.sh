#!/bin/bash
# Shared runner: Tuckute fMRI embedding encoding (ridge on hidden states).
# One array task = one registry (model, checkpoint); submit via scripts/submit.py.
#
# Required env: TUCKUTE_FAMILY (registry family), SLURM_ARRAY_TASK_ID.
# Optional env: OUT_DIR, SUMMARY_ONLY=1 (run tuckute_jz_run_summary instead).

# Fit only the final layer (2026-10-12; caches keep every layer). EMBED_LAST_LAYER_ONLY=0 for the full sweep.
export EMBED_LAST_LAYER_ONLY="${EMBED_LAST_LAYER_ONLY:-1}"
tuckute_jz_prepare() {
  : "${TUCKUTE_FAMILY:?Set TUCKUTE_FAMILY (registry family, e.g. gpt2)}"
  cd "$BENCHMARK_ROOT" || return 1
  if [[ -n "${SCRATCH:-}" ]]; then
    source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
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
  export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  OUT_DIR="${OUT_DIR:-$BENCHMARK_ROOT/outputs_jeanzay/tuckute/embedding_sweep}"
  mkdir -p "$OUT_DIR" outputs_jeanzay/logs
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  if command -v uv >/dev/null 2>&1; then PY=(uv run --frozen --no-sync --no-dev python); else source .venv/bin/activate; PY=(python); fi
}

tuckute_jz_run_array_task() {
  tuckute_jz_prepare || return 1
  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || return 1
  echo "Job started at $(date)  node=$(hostname)  job=${SLURM_JOB_ID:-?}  task=${SLURM_ARRAY_TASK_ID:-0}"
  nvidia-smi | grep -E "GPU|MiB" | head -4
  (while true; do
      _ts="$(date '+%H:%M:%S')"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk -v ts="$_ts" -F', ' '{printf "[%s] GPU%s: util=%s%%  mem=%s/%s MiB\n", ts, $1, $2, $3, $4}'
      sleep 60
    done) &
  _GPU_MONITOR_PID=$!
  trap 'kill "${_GPU_MONITOR_PID:-}" 2>/dev/null || true' EXIT
  "${PY[@]}" scripts/run/embedding/neural/tuckute.py \
      --config      "$BENCHMARK_ROOT/configs/embedding/neural/tuckute.yaml" \
      --family      "$TUCKUTE_FAMILY" \
      --array-index "${SLURM_ARRAY_TASK_ID:-0}" \
      --batch_size  128 \
      --device      cuda \
      --n_layer_workers 32 \
      --out_dir     "$OUT_DIR"
  echo "Job completed at $(date)"
}

tuckute_jz_run_summary() {
  tuckute_jz_prepare || return 1
  "${PY[@]}" scripts/run/embedding/neural/tuckute.py \
      --config  "$BENCHMARK_ROOT/configs/embedding/neural/tuckute.yaml" \
      --family  "$TUCKUTE_FAMILY" \
      --summary-only \
      --out_dir "$OUT_DIR"
}
