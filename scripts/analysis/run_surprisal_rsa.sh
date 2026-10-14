#!/usr/bin/env bash
# Cache-only surprisal RSA for the datasets with straightforward current RDM alignment.
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/surprisal-rsa-matplotlib
.venv/bin/python scripts/analysis/surprisal_rsa.py provo --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py meco --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py tuckute --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py fedorenko2016 --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py geco --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py onestop --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py naturalstories --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py ucl_reading --workers 8
.venv/bin/python scripts/analysis/surprisal_rsa.py zuco_reading --workers 8
