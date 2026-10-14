#!/usr/bin/env bash
# Cache-only Podcast RSA. Submit/run this from a CPU dev allocation (10 cores).
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/analysis/podcast_embedding_rsa.py --workers 2
.venv/bin/python scripts/analysis/surprisal_rsa.py podcast --workers 8
