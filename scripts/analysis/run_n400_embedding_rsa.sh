#!/usr/bin/env bash
# Cache-only last-layer embedding RSA for item-level N400 datasets.
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/analysis/word_neural_embedding_rsa.py nieuwland --workers 2
.venv/bin/python scripts/analysis/word_neural_embedding_rsa.py szewczyk --workers 2
