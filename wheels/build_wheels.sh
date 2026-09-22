#!/bin/bash
set -euo pipefail
module load cuda/12.4.1
source /lustre/fswork/projects/rech/ldh/una68ug/benchmark/.venv/bin/activate
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
cd /lustre/fswork/projects/rech/ldh/una68ug/mamba_build/causal-conv1d
python setup.py bdist_wheel -d /lustre/fswork/projects/rech/ldh/una68ug/wheels
cd /lustre/fswork/projects/rech/ldh/una68ug/mamba_build_v226/mamba-ssm
python setup.py bdist_wheel -d /lustre/fswork/projects/rech/ldh/una68ug/wheels
