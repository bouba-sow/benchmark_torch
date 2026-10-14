#!/usr/bin/env bash
# Run both cache-only RSA sweeps inside one 10-CPU Slurm allocation.
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/neural-rsa-matplotlib
.venv/bin/python scripts/analysis/other_neural_rsa.py tuckute --workers 8
.venv/bin/python scripts/analysis/other_neural_rsa.py fedorenko2016 --workers 8
