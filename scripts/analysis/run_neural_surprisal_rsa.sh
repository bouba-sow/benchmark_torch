#!/usr/bin/env bash
# Run the cached item-level N400 surprisal RSA tracks in one CPU allocation.
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/analysis/surprisal_rsa.py nieuwland --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py szewczyk --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py broderick --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py megmasc --workers 8
