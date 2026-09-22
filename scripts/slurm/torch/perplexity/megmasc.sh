#!/bin/bash
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ppl_megmasc
#SBATCH --time=02:00:00
#SBATCH --output=outputs_torch/logs/perplexity_megmasc_%j.out
#
# MEG-MASC (Gwilliams) perplexity for every registry model (resumable: finished models are skipped).
#   sbatch scripts/slurm/torch/perplexity/megmasc.sh
# See _common.sh for PPL_EXTRA_ARGS (e.g. "--family opt") and how to request 2 GPUs for the 20B+ models.
set -euo pipefail
: "${SCRATCH:?Set SCRATCH (e.g. export SCRATCH=/scratch/$USER)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"
source "${BENCHMARK_ROOT}/scripts/slurm/torch/perplexity/_common.sh"
DATA_ROOT="${MEGMASC_BIDS_ROOT:-$BENCHMARK_ROOT/data/gwilliams}"
OUT_DIR="${OUT_DIR:-$BENCHMARK_ROOT/outputs_torch/megmasc/perplexity}"
export PPL_NO_CBR=1   # perplexity/megmasc.py has no CBR-RNN path
ppl_setup
ppl_run perplexity/megmasc.py --bids_root "$DATA_ROOT"
