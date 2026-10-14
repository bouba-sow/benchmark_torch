#!/usr/bin/env bash
# Cached word-level surprisal RSA for the remaining neural datasets.
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/analysis/surprisal_rsa.py broderick --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py megmasc --workers 8
