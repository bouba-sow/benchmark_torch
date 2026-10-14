#!/bin/bash
# Eight balanced model groups; raw surprisal extraction only (no LME fits).
set -uo pipefail

: "${WORK:=/lustre/fswork/projects/rech/ldh/una68ug}"
: "${SCRATCH:=/lustre/fsn1/projects/rech/ldh/una68ug}"
export WORK SCRATCH
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$SCRATCH/.cache}"
cd "$BENCHMARK_ROOT" || exit 1

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORK/hf_datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$BENCHMARK_CACHE_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" outputs_jeanzay/logs

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  source .venv/bin/activate
  PY=(python)
fi

"${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1
echo "Extraction-only RSA surprisal gap fill: group=$IDX host=$(hostname) start=$(date)"
"${PY[@]}" scripts/tools/rsa_surprisal_gapfill_run.py --group "$IDX" \
  --groups-file "${GROUPS_FILE:-scripts/tools/rsa_surprisal_gapfill_groups.json}"
status=$?
echo "Group $IDX finished at $(date) status=$status"
exit "$status"
