#!/usr/bin/env bash
#SBATCH --account=torch_pr_287_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=outputs_torch/logs/build_rsa_report_%j.out
#SBATCH --error=outputs_torch/logs/build_rsa_report_%j.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export MPLCONFIGDIR=/tmp/rsa-report-mpl-${SLURM_JOB_ID} PYTHONUNBUFFERED=1
.venv/bin/python scripts/analysis/build_rsa_report.py
