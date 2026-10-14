#!/bin/bash
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ppl_szewczyk
#SBATCH --time=02:00:00
#SBATCH --output=outputs_torch/logs/perplexity_szewczyk_%j.out
#
# Szewczyk & Federmeier perplexity for every registry model + CBR-RNN (resumable: finished models are skipped).
#   STUDY=f2007 sbatch scripts/slurm/torch/perplexity/szewczyk.sh      # or STUDY=h2019
# See _common.sh for PPL_EXTRA_ARGS (e.g. "--family pythia_checkpoints").
set -euo pipefail
: "${SCRATCH:?Set SCRATCH (e.g. export SCRATCH=/scratch/$USER)}"
: "${STUDY:?Set STUDY=f2007 or STUDY=h2019}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"
source "${BENCHMARK_ROOT}/scripts/slurm/torch/perplexity/_common.sh"
OUT_DIR="${OUT_DIR:-$BENCHMARK_ROOT/outputs_torch/szewczyk_${STUDY}/perplexity}"
ppl_setup
# not ppl_run: its final --summarize call does not forward --study (required by szewczyk.py)
# shellcheck disable=SC2086
"${PY[@]}" scripts/perplexity/szewczyk.py --study "$STUDY" --all ${PPL_EXTRA_ARGS:-} --out_dir "$OUT_DIR" --device cuda
ppl_cbr_rnn scripts/perplexity/szewczyk.py --study "$STUDY"
"${PY[@]}" scripts/perplexity/szewczyk.py --study "$STUDY" --summarize --out_dir "$OUT_DIR"
echo "Job completed at $(date)"
