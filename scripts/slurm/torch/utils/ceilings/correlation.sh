#!/bin/bash
#SBATCH --account=torch_pr_287_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ceiling_corr
#SBATCH --time=02:00:00
#SBATCH --output=outputs_torch/logs/ceiling_correlation_%j.out
#
# Split-half noise ceilings on the CORRELATION scale (what the embedding track is normalised by).
#
#   sbatch scripts/slurm/torch/utils/ceilings/correlation.sh                 # every dataset
#   sbatch scripts/slurm/torch/utils/ceilings/correlation.sh geco provo      # chosen datasets
#   sbatch scripts/slurm/torch/utils/ceilings/correlation.sh megmasc         # MEG-MASC session test-retest
#
# Datasets are the slugs of scripts/ceilings/correlation.py (--help lists them). Results go to
# outputs/noise_ceilings/, the folder the reports read; MEG-MASC keeps its own
# outputs_torch/noise_ceilings/megmasc/. Memory-heavy datasets (Nieuwland, Podcast, LPP, Broderick)
# may need `sbatch --mem=...` or `--time=...`.
set -euo pipefail
: "${SCRATCH:?Set SCRATCH (e.g. export SCRATCH=/scratch/$USER)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRATCH/Documents/benchmark}"
cd "$BENCHMARK_ROOT"
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
fi
mkdir -p outputs_torch/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if command -v uv >/dev/null 2>&1; then PY=(uv run --frozen --no-sync --no-dev python); else source .venv/bin/activate; PY=(python); fi

DATASETS=()
DO_MEGMASC=0
for d in "$@"; do
  if [[ "$d" == "megmasc" ]]; then DO_MEGMASC=1; else DATASETS+=("$d"); fi
done
if [[ $# -eq 0 ]]; then DATASETS=(all); fi
echo "Job started at $(date)  node=$(hostname)  datasets=${DATASETS[*]:-} megmasc=${DO_MEGMASC}"

if [[ ${#DATASETS[@]} -gt 0 ]]; then
  "${PY[@]}" scripts/ceilings/correlation.py --dataset "${DATASETS[@]}" --out-dir outputs/noise_ceilings
fi

if [[ "$DO_MEGMASC" == "1" ]]; then
  export MEGMASC_BIDS_ROOT="${MEGMASC_BIDS_ROOT:-${SCRATCH:-$BENCHMARK_ROOT/..}/data/gwilliams}"
  export MEGMASC_EPOCH_CACHE_DIR="${MEGMASC_EPOCH_CACHE_DIR:-${BENCHMARK_CACHE_ROOT}/megmasc_epochs}"
  mkdir -p "$MEGMASC_EPOCH_CACHE_DIR"
  # One subject per array index; SUBJECTS overrides the default 0-9 (or use `sbatch --array=3 ... megmasc`).
  for IDX in ${SLURM_ARRAY_TASK_ID:-${SUBJECTS:-0 1 2 3 4 5 6 7 8 9}}; do
    echo "--- MEG-MASC subject index ${IDX} ---"
    "${PY[@]}" scripts/ceilings/correlation_megmasc.py --config configs/embedding/neural/megmasc.yaml \
      --array-index "$IDX" --out_dir "$BENCHMARK_ROOT/outputs_torch/noise_ceilings/megmasc"
  done
fi
echo "Done $(date)"
