#!/usr/bin/env bash
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/analysis/word_neural_embedding_rsa.py nieuwland --limit 1 --workers 1
.venv/bin/python scripts/analysis/word_neural_embedding_rsa.py szewczyk --limit 1 --workers 1
