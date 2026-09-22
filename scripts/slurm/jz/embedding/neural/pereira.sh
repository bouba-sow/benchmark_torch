#!/bin/bash
# Shared runner: Pereira fMRI voxelwise embedding encoding (himalaya ridge, GroupKFold).
# One array task = one registry (model, checkpoint); extraction AND ridge fit run in the
# same GPU task. Submit via scripts/submit.py.
#
# Required env: PEREIRA_FAMILY (registry family), SLURM_ARRAY_TASK_ID.

pereira_embed_jz_run_array_task() {
  : "${PEREIRA_FAMILY:?Set PEREIRA_FAMILY (registry family, e.g. gpt2)}"
  cd "$BENCHMARK_ROOT" || return 1
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  PEREIRA_EMBED_FEATURE_CACHE_DIR="${PEREIRA_EMBED_FEATURE_CACHE_DIR:-$SCRATCH/.cache/pereira_embed_features}"
  SWEEP_ROOT="${SWEEP_ROOT:-$PWD/outputs_jeanzay/pereira/embedding_sweep}"
  mkdir -p "$PEREIRA_EMBED_FEATURE_CACHE_DIR" "$SWEEP_ROOT" outputs_jeanzay/logs
  if command -v uv >/dev/null 2>&1; then
    PY=(uv run --frozen --no-sync --no-dev python)
  else
    source .venv/bin/activate
    PY=(python)
  fi
  local IDX="${SLURM_ARRAY_TASK_ID:-0}"
  echo "Job started at $(date) on $(hostname)  array_task=${IDX}"
  "${PY[@]}" scripts/run/embedding/neural/pereira.py \
    --config configs/embedding/neural/pereira.yaml \
    --from-registry --family "$PEREIRA_FAMILY" --array-index "${IDX}" \
    --device cuda \
    --feature_cache_dir "${PEREIRA_EMBED_FEATURE_CACHE_DIR}" \
    --out_dir "${SWEEP_ROOT}"
  echo "Job completed at $(date)"
}
