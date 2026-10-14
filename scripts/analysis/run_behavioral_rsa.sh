#!/usr/bin/env bash
# Run Provo and English MECO cache-only RSA in one 10-CPU Slurm allocation.
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/behavioral-rsa-matplotlib
.venv/bin/python scripts/analysis/behavioral_rsa.py provo --workers 8
.venv/bin/python scripts/analysis/behavioral_rsa.py meco --workers 8
