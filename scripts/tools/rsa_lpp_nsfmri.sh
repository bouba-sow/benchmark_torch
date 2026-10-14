#!/bin/bash
# Throwaway RSA gap-fill launcher for LPP fMRI + Natural Stories fMRI raw
# embedding caches -- see rsa_lpp_nsfmri_run.py's docstring.
#
# Usage (after generating the group file once):
#   python scripts/tools/rsa_lpp_nsfmri_groups.py --n-groups 8
#   sbatch --array=0-7 ... scripts/tools/rsa_lpp_nsfmri.sh

set -uo pipefail  # NOT -e: one dataset/model failing must not kill the whole group

: "${WORK:?Set WORK to your Jean Zay fswork project}"
export WORK
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT" || exit 1

source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"

mkdir -p "$LPP_FEATURE_CACHE_DIR" "$NS_FMRI_FEATURE_CACHE_DIR" outputs_jeanzay/logs

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORK/hf_datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --frozen --no-sync --no-dev python)
else
  source .venv/bin/activate
  PY=(python)
fi

"${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || exit 1

echo "Job started at $(date)  group=${IDX}  node=$(hostname)"
"${PY[@]}" scripts/tools/rsa_lpp_nsfmri_run.py \
  --group "${IDX}" \
  --groups-file "${GROUPS_FILE:-scripts/tools/rsa_lpp_nsfmri_groups.json}" \
  --device cuda
echo "Job completed at $(date)"
