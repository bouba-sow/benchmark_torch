#!/bin/bash
#SBATCH -A ldh@cpu
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=dlt_costs
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/dlt_costs_%j.out
#
# Dependency-locality (Gibson 2000) integration costs per word, with spaCy en_core_web_trf.
# Two stages, in two environments (spaCy conflicts with the main one):
#   export : word lists, in the main .venv        parse : spaCy parse, in the isolated .venv-dlt
# .venv-dlt must exist first: run scripts/slurm/jz/utils/setup/dlt_env.sh on a login node once.
#
#   sbatch scripts/slurm/jz/utils/dlt/costs.sh                          # every dataset, both stages
#   sbatch scripts/slurm/jz/utils/dlt/costs.sh meco provo lpp           # chosen datasets
#   STAGE=parse sbatch scripts/slurm/jz/utils/dlt/costs.sh meco         # only the parse (words already exported)
#   sbatch scripts/slurm/jz/utils/dlt/costs.sh nieuwland                # costs at the critical noun (N400 items)
#
# Datasets: naturalstories geco provo meco onestop ucl_reading zuco_reading_{sr,nr,tsr} podcast broderick lpp
# fedorenko2016 (see scripts/dlt/costs.py) and nieuwland. Results: outputs/dlt/<dataset>_dlt_cost.csv.
set -euo pipefail
: "${WORK:?Set WORK (e.g. export WORK=/lustre/fswork/projects/rech/ldh/una68ug)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT"
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
fi
mkdir -p outputs_jeanzay/logs
STAGE="${STAGE:-both}"
OUT_DIR="${OUT_DIR:-$BENCHMARK_ROOT/outputs/dlt}"
DLT_VENV="${DLT_VENV:-$BENCHMARK_ROOT/.venv-dlt}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 MNE_USE_NUMBA=false
if command -v uv >/dev/null 2>&1; then MAIN_PY=(uv run --frozen --no-sync --no-dev python); else MAIN_PY=(.venv/bin/python); fi
if [[ "$STAGE" != "export" && ! -x "${DLT_VENV}/bin/python" ]]; then
  echo "ERROR: ${DLT_VENV} not found. Run scripts/slurm/jz/utils/setup/dlt_env.sh on a login node first." >&2
  exit 1
fi

DATASETS=(); NIEUWLAND=0
for d in "$@"; do [[ "$d" == "nieuwland" ]] && NIEUWLAND=1 || DATASETS+=("$d"); done
[[ $# -eq 0 ]] && { DATASETS=(); ALL=1; } || ALL=0
DATASET_ARGS=(); [[ ${#DATASETS[@]} -gt 0 ]] && DATASET_ARGS=(--datasets "${DATASETS[@]}")
echo "Job started at $(date)  node=$(hostname)  stage=${STAGE}  OUT_DIR=${OUT_DIR}"

if [[ "$ALL" == "1" || ${#DATASETS[@]} -gt 0 ]]; then
  if [[ "$STAGE" == "both" || "$STAGE" == "export" ]]; then
    echo "--- export-words (main .venv) ---"
    "${MAIN_PY[@]}" scripts/dlt/costs.py export-words "${DATASET_ARGS[@]}" --out_dir "${OUT_DIR}"
  fi
  if [[ "$STAGE" == "both" || "$STAGE" == "parse" ]]; then
    echo "--- parse (.venv-dlt, en_core_web_trf) ---"
    "${DLT_VENV}/bin/python" scripts/dlt/costs.py parse "${DATASET_ARGS[@]}" --spacy_model en_core_web_trf --out_dir "${OUT_DIR}"
  fi
fi

if [[ "$NIEUWLAND" == "1" || "$ALL" == "1" ]]; then
  NIEUWLAND_BIDS_ROOT="${NIEUWLAND_BIDS_ROOT:-$BENCHMARK_ROOT/data/nieuwland}"
  ITEMS_OUT_DIR="${ITEMS_OUT_DIR:-$OUT_DIR/_items_export}"
  if [[ "$STAGE" == "both" || "$STAGE" == "export" ]]; then
    if [[ -f "${NIEUWLAND_BIDS_ROOT}/dataset_description.json" ]]; then
      echo "--- Nieuwland export-items (real EEG events from ${NIEUWLAND_BIDS_ROOT}) ---"
      "${MAIN_PY[@]}" scripts/dlt/costs_nieuwland.py export-items --bids_root "${NIEUWLAND_BIDS_ROOT}" --out_dir "${ITEMS_OUT_DIR}"
    else
      NIEUWLAND_SWEEP_ROOT="${NIEUWLAND_SWEEP_ROOT:-$BENCHMARK_ROOT/outputs_jeanzay/nieuwland/embedding_sweep}"
      echo "--- Nieuwland export-items (from saved predictions in ${NIEUWLAND_SWEEP_ROOT}; no BIDS/EEG needed) ---"
      "${MAIN_PY[@]}" scripts/dlt/export_nieuwland_items.py --sweep_root "${NIEUWLAND_SWEEP_ROOT}" --out_dir "${ITEMS_OUT_DIR}"
    fi
  fi
  if [[ "$STAGE" == "both" || "$STAGE" == "parse" ]]; then
    echo "--- Nieuwland parse (.venv-dlt) ---"
    "${DLT_VENV}/bin/python" scripts/dlt/costs_nieuwland.py parse --spacy_model en_core_web_trf --out_dir "${ITEMS_OUT_DIR}"
  fi
fi
echo "Job completed at $(date)"
ls -la "${OUT_DIR}"/*_dlt_cost*.csv 2>/dev/null || true
