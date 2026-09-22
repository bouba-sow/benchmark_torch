#!/bin/bash
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ppl_pythia_ckpt
#SBATCH --time=04:00:00
#SBATCH --output=outputs_torch/logs/perplexity_pythia_checkpoints_%A_%a.out
#SBATCH --array=0-9%10
#
# Perplexity of every Pythia TRAINING CHECKPOINT (11 steps) for the size selected by the array index
# (14m ... 12b, 10 sizes), on four datasets: Natural Stories SPR, UCL Reading, ZuCo Reading and
# Nieuwland. It feeds the checkpoint-scaling figure (scripts/build_paper_figure9_...). Provo, GECO and
# MECO are not listed because their sweeps already stored per-checkpoint perplexity. Other datasets'
# final-model perplexity comes from the per-dataset scripts in this folder. Resumable.
# (Runs perplexity/pythia_checkpoints_3datasets.py for the first three, then ..._n400.py.)
#
#   sbatch scripts/slurm/torch/perplexity/pythia_checkpoints.sh
#   sbatch --array=9 scripts/slurm/torch/perplexity/pythia_checkpoints.sh       # one size (12b)
#   COLLECT=1 sbatch --array=0 --time=00:20:00 scripts/slurm/torch/perplexity/pythia_checkpoints.sh   # rebuild the summaries
set -euo pipefail
: "${SCRATCH:?Set SCRATCH (e.g. export SCRATCH=/scratch/$USER)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"
cd "$BENCHMARK_ROOT"
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
fi
MODEL_IDS=(
  "EleutherAI/pythia-14m-deduped" "EleutherAI/pythia-31m-deduped" "EleutherAI/pythia-70m-deduped"
  "EleutherAI/pythia-160m-deduped" "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-1b-deduped"
  "EleutherAI/pythia-1.4b-deduped" "EleutherAI/pythia-2.8b-deduped" "EleutherAI/pythia-6.9b-deduped"
  "EleutherAI/pythia-12b-deduped"
)
source .venv/bin/activate 2>/dev/null || true
if [[ "${COLLECT:-0}" == "1" ]]; then
  .venv/bin/python scripts/perplexity/pythia_checkpoints_3datasets.py --collect_summary
  .venv/bin/python scripts/perplexity/pythia_checkpoints_n400.py --collect_summary || true
  exit 0
fi
MODEL_ID="${MODEL_IDS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "Job started at $(date)  node=$(hostname)  MODEL_ID=${MODEL_ID}"
.venv/bin/python scripts/perplexity/pythia_checkpoints_3datasets.py --model_id "$MODEL_ID" --device cuda
.venv/bin/python scripts/perplexity/pythia_checkpoints_n400.py --model_id "$MODEL_ID" --device cuda
echo "Job completed at $(date)"
