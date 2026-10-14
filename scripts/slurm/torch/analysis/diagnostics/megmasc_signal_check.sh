#!/bin/bash
#SBATCH --account=torch_pr_287_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=mm_sigchk
#SBATCH --time=01:30:00
#SBATCH --output=outputs_torch/logs/megmasc_sigchk_%A_%a.out
#SBATCH --array=0-10
# MEG-MASC signal sanity check, one subject per task (raw .con -> diagnostics).
# Output -> $SCRATCH/megmasc_signal_check/ (not outputs/). Then run the reduce script.
set -euo pipefail
: "${WORK:?}"; export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"; cd "$BENCHMARK_ROOT"
source scripts/slurm/torch/torch_cache_env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
SUB=$(printf "sub-%02d" $((SLURM_ARRAY_TASK_ID + 1)))
.venv/bin/python scripts/diagnostics/megmasc_signal_check.py \
  --subject "$SUB" --bids-root "$MEGMASC_BIDS_ROOT" --out-dir "$SCRATCH/megmasc_signal_check"
