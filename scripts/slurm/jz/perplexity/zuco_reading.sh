#!/bin/bash
#SBATCH -A ldh@h100
#SBATCH --gres=gpu:1
#SBATCH -C h100
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ppl_zuco_reading
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/perplexity_zuco_reading_%j.out
#
# ZuCo Reading perplexity for every registry model + CBR-RNN (resumable: finished models are skipped).
#   sbatch scripts/slurm/jz/perplexity/zuco_reading.sh
# See _common.sh for PPL_EXTRA_ARGS (e.g. "--family opt") and how to request 2 GPUs for the 20B+ models.
set -euo pipefail
: "${WORK:?Set WORK (e.g. export WORK=/lustre/fswork/projects/rech/ldh/una68ug)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
source "${BENCHMARK_ROOT}/scripts/slurm/jz/perplexity/_common.sh"
DATA_ROOT="${ZUCO_MAT_ROOT:-$BENCHMARK_ROOT/data/zuco_reading/osfstorage}"
OUT_DIR="${OUT_DIR:-$BENCHMARK_ROOT/outputs_jeanzay/zuco_reading/perplexity}"
ppl_setup
ppl_run perplexity/zuco_reading.py --mat_root "$DATA_ROOT"
