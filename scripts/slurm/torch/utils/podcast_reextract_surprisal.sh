#!/bin/bash
# Re-extract Podcast final-layer logit-lens surprisal for a list of registry task indices
# (global index of scripts/run/surprisal/neural/podcast_extract.py) into a FRESH cache dir,
# one model after another in one GPU job (2026-10-08: the 2026-09-28 caches have a constant
# first-chunk block for 79 models). Does not touch the old cache.
#   TASKS="0 9 17" CACHE_DIR=$SCRATCH/.cache/podcast_features_v2 sbatch <this script> (see sbatch flags in the caller)
set -uo pipefail
cd "${BENCHMARK_ROOT:-/lustre/fswork/projects/rech/ldh/una68ug/benchmark}" || exit 1
: "${TASKS:?space-separated global task indices}"
: "${CACHE_DIR:?target cache dir}"
export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/ldh/una68ug}" WORK="${WORK:-/lustre/fswork/projects/rech/ldh/una68ug}"
source scripts/slurm/torch/torch_cache_env.sh
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="forgetting-transformer/src:${PYTHONPATH:-}"
source .venv/bin/activate
mkdir -p "$CACHE_DIR"
nvidia-smi -L
fail=0
for i in $TASKS; do
  echo "=== task $i  $(date +%T)"
  python scripts/run/surprisal/neural/podcast_extract.py --global-index "$i" --device cuda \
    --cache-dir "$CACHE_DIR" --skip_perplexity 2>&1 | grep -v "Loading weights" | tail -8 \
    || true
  [[ ${PIPESTATUS[0]} -ne 0 ]] && { echo "TASK $i FAILED"; fail=1; }
done
echo "done fail=$fail"; exit $fail
