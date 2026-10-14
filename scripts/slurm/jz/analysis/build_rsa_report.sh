#!/usr/bin/env bash
#SBATCH -A ldh@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=outputs_jeanzay/logs/build_rsa_report_%j.out
#SBATCH --error=outputs_jeanzay/logs/build_rsa_report_%j.err
set -euo pipefail
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
export MPLCONFIGDIR=/tmp/rsa-report-mpl-${SLURM_JOB_ID} PYTHONUNBUFFERED=1
.venv/bin/python scripts/analysis/build_rsa_report.py
