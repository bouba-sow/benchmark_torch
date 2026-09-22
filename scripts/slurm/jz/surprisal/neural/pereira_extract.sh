#!/bin/bash
# Shared runner: Pereira fMRI surprisal extraction (GPU stage; the CPU LME fit is pereira_fit.sh).
# One array task = one registry (model, checkpoint); logit-lens surprisal for the 627 sentences.
# Submit via scripts/submit.py.  Required env: PEREIRA_FAMILY, SLURM_ARRAY_TASK_ID.

pereira_surprisal_jz_run_array_task() {
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
  PEREIRA_LME_FEATURE_CACHE_DIR="${PEREIRA_LME_FEATURE_CACHE_DIR:-$SCRATCH/.cache/pereira_lme_features}"
  mkdir -p "$PEREIRA_LME_FEATURE_CACHE_DIR" outputs_jeanzay/logs
  if command -v uv >/dev/null 2>&1; then
    PY=(uv run --frozen --no-sync --no-dev python)
  else
    source .venv/bin/activate
    PY=(python)
  fi
  "${PY[@]}" scripts/run/surprisal/neural/pereira_extract.py \
    --config configs/surprisal/neural/pereira.yaml \
    --from-registry --family "$PEREIRA_FAMILY" --array-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --device cuda \
    --feature_cache_dir "${PEREIRA_LME_FEATURE_CACHE_DIR}"
}
