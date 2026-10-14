#!/usr/bin/env bash
# Rebuild cache-only embedding/surprisal RSA outputs, confound RDMs, and report.
# Run inside a 10-CPU qos_cpu-dev allocation (dev QoS limit: 2 hours).
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

SURPRISAL_DATASETS=(
  provo meco geco onestop naturalstories ucl_reading zuco_reading
  pereira tuckute fedorenko2016 nieuwland szewczyk broderick megmasc podcast
)
for dataset in "${SURPRISAL_DATASETS[@]}"; do
  echo "=== Surprisal RSA: ${dataset} ==="
  .venv/bin/python scripts/analysis/surprisal_rsa.py "${dataset}" --workers 8
done

for dataset in provo meco; do
  echo "=== Embedding RSA: ${dataset} ==="
  .venv/bin/python scripts/analysis/behavioral_rsa.py "${dataset}" --workers 8
done

for dataset in geco onestop naturalstories ucl_reading zuco_reading; do
  echo "=== Embedding RSA: ${dataset} ==="
  .venv/bin/python scripts/analysis/additional_behavioral_embedding_rsa.py "${dataset}" --workers 8
done

echo "=== Embedding RSA: Pereira ==="
.venv/bin/python scripts/analysis/pereira_rsa.py --all-scored

for dataset in tuckute fedorenko2016; do
  echo "=== Embedding RSA: ${dataset} ==="
  .venv/bin/python scripts/analysis/other_neural_rsa.py "${dataset}" --workers 8
done

for dataset in nieuwland szewczyk broderick megmasc; do
  echo "=== Embedding RSA: ${dataset} ==="
  .venv/bin/python scripts/analysis/word_neural_embedding_rsa.py "${dataset}" --workers 2
done

echo "=== Embedding RSA: Podcast ==="
.venv/bin/python scripts/analysis/podcast_embedding_rsa.py --workers 2

echo "=== Refresh lexical-confound RDMs ==="
.venv/bin/python scripts/analysis/save_lexical_confounds_rdms.py
.venv/bin/python scripts/analysis/save_neural_lexical_confounds_rdms.py

echo "=== Rebuild unified RSA report ==="
.venv/bin/python scripts/analysis/build_rsa_report.py
