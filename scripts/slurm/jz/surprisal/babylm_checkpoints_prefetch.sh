#!/bin/bash
#SBATCH -A ldh@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --job-name=blm_gpt2_prefetch
#SBATCH --output=outputs_jeanzay/logs/babylm_gpt2_prefetch_%A_%a.out
#SBATCH --array=0-6%7
#
# babylm_gpt2_checkpoints: last-layer-only surprisal + perplexity prefetch, one array task per
# group of datasets. Each task is ONE process (scripts/run/surprisal/babylm_checkpoints_prefetch.py): it loops the
# 29 checkpoints and loads each model once for all datasets of its group. Idempotent: cached
# (dataset, checkpoint) pairs are skipped, so a re-run after a time-out just continues.
#
#   sbatch scripts/slurm/jz/surprisal/babylm_checkpoints_prefetch.sh                # all groups
#   sbatch --array=3 scripts/slurm/jz/surprisal/babylm_checkpoints_prefetch.sh      # nieuwland only
#   DATASETS="geco,provo" sbatch --array=0 scripts/slurm/jz/surprisal/babylm_checkpoints_prefetch.sh
set -uo pipefail
: "${WORK:?Set WORK (e.g. export WORK=/lustre/fswork/projects/rech/ldh/una68ug)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT" || exit 1
mkdir -p outputs_jeanzay/logs
GROUPS_=(
  "geco,meco"
  "naturalstories,provo"
  "ucl_reading,zuco_reading"
  "nieuwland"
  "lpp,tuckute"
  "ns_fmri"
  "fedorenko2016,podcast"
)
DATASETS="${DATASETS:-${GROUPS_[${SLURM_ARRAY_TASK_ID:-0}]}}"
echo "datasets [${DATASETS}] started at $(date)  node=$(hostname)"
.venv/bin/python scripts/run/surprisal/babylm_checkpoints_prefetch.py --datasets "$DATASETS" --device cuda
rc=$?
echo "finished at $(date)  rc=$rc"
exit 0
