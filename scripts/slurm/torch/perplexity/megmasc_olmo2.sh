#!/bin/bash
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ppl_megmasc_olmo2
#SBATCH --time=02:00:00
#SBATCH --output=outputs_torch/logs/perplexity_megmasc_olmo2_%j.out
#
# MEG-MASC perplexity for the 3 OLMo-2 models. Only their final `stage1-step...` revision is in the HF cache
# (the `main` snapshot is not), so load that revision; the result is stored as the model's main entry.
#   sbatch scripts/slurm/torch/perplexity/megmasc_olmo2.sh
set -euo pipefail
: "${SCRATCH:?Set SCRATCH (e.g. export SCRATCH=/scratch/$USER)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"
source "${BENCHMARK_ROOT}/scripts/slurm/torch/perplexity/_common.sh"
DATA_ROOT="${MEGMASC_BIDS_ROOT:-${SCRATCH:-$BENCHMARK_ROOT}/data/gwilliams}"
OUT_DIR="${OUT_DIR:-$BENCHMARK_ROOT/outputs_torch/megmasc/perplexity}"
ppl_setup
for spec in "allenai/OLMo-2-0425-1B 1.0e9 stage1-step1907359-tokens4001B" \
            "allenai/OLMo-2-1124-7B 7.0e9 stage1-step928646-tokens3896B" \
            "allenai/OLMo-2-1124-13B 13.0e9 stage1-step596057-tokens5001B"; do
  read -r model nparams rev <<<"$spec"
  "${PY[@]}" scripts/perplexity/megmasc.py --model_id "$model" --n_params "$nparams" --load_revision "$rev" \
    --bids_root "$DATA_ROOT" --out_dir "$OUT_DIR" --device cuda
done
"${PY[@]}" scripts/perplexity/megmasc.py --summarize --out_dir "$OUT_DIR"
echo "Job completed at $(date)"
