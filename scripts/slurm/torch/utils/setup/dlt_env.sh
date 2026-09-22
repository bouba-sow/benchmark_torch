#!/bin/bash
# One-time setup of the isolated spaCy environment (.venv-dlt) used by scripts/slurm/torch/utils/dlt/costs.sh.
# It is separate from the main .venv because spaCy's dependencies conflict with it. Needs internet:
# run on a login node (the compute nodes then load the model offline).
#
#   bash scripts/slurm/torch/utils/setup/dlt_env.sh
set -euo pipefail
: "${SCRATCH:?Set SCRATCH (e.g. export SCRATCH=/scratch/$USER)}"
: "${SCRATCH:?Set SCRATCH (lustre project dir) -- required so downloads don't fill $HOME's quota}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"
cd "$BENCHMARK_ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/.cache/pip}"
DLT_VENV="${DLT_VENV:-$BENCHMARK_ROOT/.venv-dlt}"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required to create ${DLT_VENV}." >&2; exit 1; }
echo "Creating the isolated DLT/spaCy venv at ${DLT_VENV} ..."
uv venv --python 3.12 "$DLT_VENV"
echo "Installing torch (CPU), spaCy + transformers and en_core_web_trf (main .venv untouched) ..."
uv pip install --python "${DLT_VENV}/bin/python" --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python "${DLT_VENV}/bin/python" \
  "spacy[transformers]>=3.8.14" pandas click \
  "en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"
echo
echo "Verifying offline load (simulates the compute-node environment) ..."
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${DLT_VENV}/bin/python" -c "
import spacy
nlp = spacy.load('en_core_web_trf')
doc = nlp('The reporter who the senator attacked disliked the editor.')
print('OK -', len(doc), 'tokens parsed offline.')
"
echo "Done. .venv-dlt is ready for scripts/slurm/torch/utils/dlt/costs.sh."
