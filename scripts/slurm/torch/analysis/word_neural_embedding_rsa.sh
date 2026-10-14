#!/usr/bin/env bash
#SBATCH --account=torch_pr_287_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs_torch/logs/word_neural_rsa_%j.out
#SBATCH --error=outputs_torch/logs/word_neural_rsa_%j.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
.venv/bin/python scripts/analysis/word_neural_embedding_rsa.py "$@"
