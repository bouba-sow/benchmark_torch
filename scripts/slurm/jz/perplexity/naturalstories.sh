#!/bin/bash
#SBATCH -A ldh@h100
#SBATCH --gres=gpu:1
#SBATCH -C h100
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ppl_naturalstories
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/perplexity_naturalstories_%j.out
#
# Natural Stories perplexity for every registry family (resumable: finished models are skipped).
#   sbatch scripts/slurm/jz/perplexity/naturalstories.sh
#   FAMILIES="opt gptneo" sbatch --gres=gpu:2 scripts/slurm/jz/perplexity/naturalstories.sh   # 20B+ models
# See _common.sh for PPL_EXTRA_ARGS. Context length: 4096 tokens, 8192 for Mamba/Mamba2.
set -euo pipefail
: "${WORK:?Set WORK (e.g. export WORK=/lustre/fswork/projects/rech/ldh/una68ug)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
source "${BENCHMARK_ROOT}/scripts/slurm/jz/perplexity/_common.sh"
DATA_ROOT="${NS_DATASET_ROOT:-$BENCHMARK_ROOT/data/naturalstories}"
OUT_DIR="${OUT_DIR:-$BENCHMARK_ROOT/outputs_jeanzay/naturalstories/perplexity}"
ppl_setup
if [[ -z "${FAMILIES:-}" ]]; then
  FAMILIES="$("${PY[@]}" -c "from neuroai.models.registry import LPP_FAMILIES; print(' '.join(f for f in LPP_FAMILIES if not f.endswith('_checkpoints')))")"
fi
for FAMILY in $FAMILIES; do
  MAX_LENGTH=4096
  [[ "$FAMILY" == mamba || "$FAMILY" == mamba2 ]] && MAX_LENGTH=8192
  echo "=== family=${FAMILY} (max_length=${MAX_LENGTH}) ==="
  # shellcheck disable=SC2086
  "${PY[@]}" scripts/perplexity/naturalstories.py --all --family "$FAMILY" --max_length "$MAX_LENGTH" \
      ${PPL_EXTRA_ARGS:-} --dataset_root "$DATA_ROOT" --out_dir "$OUT_DIR" --device cuda
done
"${PY[@]}" scripts/perplexity/naturalstories.py --summarize --out_dir "$OUT_DIR"
echo "Job completed at $(date)"
