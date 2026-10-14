#!/usr/bin/env bash
#SBATCH -A ldh@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/other_neural_rsa_%j.out
#SBATCH --error=outputs_jeanzay/logs/other_neural_rsa_%j.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/ldh/una68ug}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
.venv/bin/python scripts/analysis/other_neural_rsa.py "$@"
