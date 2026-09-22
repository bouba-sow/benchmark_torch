#!/bin/bash
#SBATCH -A ldh@cpu
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=ceiling_dll
#SBATCH --time=04:00:00
#SBATCH --output=outputs_jeanzay/logs/ceiling_dll_%j.out
#
# Split-half DELTA-LOG-LIKELIHOOD noise ceilings (Timkey et al. 2026, adapted): what the surprisal
# track is normalised by. CPU only (R / pymer4 / lme4); no LLM is run.
#
#   sbatch scripts/slurm/jz/utils/ceilings/dll.sh geco                       # out-of-sample (default)
#   sbatch scripts/slurm/jz/utils/ceilings/dll.sh geco --measure GP
#   sbatch scripts/slurm/jz/utils/ceilings/dll.sh pereira --in-sample
#   sbatch scripts/slurm/jz/utils/ceilings/dll.sh megmasc --mode split_half
#   sbatch scripts/slurm/jz/utils/ceilings/dll.sh all                        # every dataset/measure, one after the other
#
# Datasets: naturalstories_spr geco meco provo ucl_reading zuco_reading onestop (behavioral;
# --measure FP|GP|SP, ZuCo Reading NR|SR|TSR), pereira tuckute ns_fmri (fMRI), fedorenko2016 podcast
# (ECoG), lpp, nieuwland szewczyk (N400), megmasc (--mode loso|split_half).
# Options: --n-splits N (15) --n-workers N (30; MEG-MASC 12) --seed N. Slow ones (Natural Stories
# ~1 h per split, MEG-MASC) may need `sbatch --time=...`. Out-of-sample results go to
# outputs/noise_ceilings/ (what the reports read); in-sample to outputs/noise_ceilings_insample/.
set -euo pipefail
: "${WORK:?Set WORK (e.g. export WORK=/lustre/fswork/projects/rech/ldh/una68ug)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT"
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
else
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$BENCHMARK_ROOT/.cache}"
fi

DATASET="${1:?usage: dll.sh <dataset|all> [--measure M] [--in-sample] [--mode loso|split_half] [--n-splits N] [--n-workers N]}"; shift
MEASURE=""; INSAMPLE=0; MODE="loso"; N_SPLITS=15; N_WORKERS=""; SEED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --measure) MEASURE="$2"; shift 2 ;;
    --in-sample) INSAMPLE=1; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --n-splits) N_SPLITS="$2"; shift 2 ;;
    --n-workers) N_WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# R for pymer4 / lme4
if ! command -v R >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load r/4.5.2 2>/dev/null || module load r 2>/dev/null || module load R 2>/dev/null || true
fi
command -v R >/dev/null 2>&1 || { echo "ERROR: R not found (required by pymer4 / lme4). Load an R module first." >&2; exit 1; }
export R_HOME="${R_HOME:-$(R RHOME)}"
export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
[[ -d "${R_HOME}/lib" ]] && export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
if command -v uv >/dev/null 2>&1; then PY=(uv run --frozen --no-sync --no-dev python); else source .venv/bin/activate; PY=(python); fi
mkdir -p outputs_jeanzay/logs outputs/noise_ceilings_insample

OUT_ARGS=(); SPLIT_FLAG=()
if [[ "$INSAMPLE" == "1" ]]; then OUT_ARGS=(--out_dir outputs/noise_ceilings_insample); fi

# run_dll <dataset> [measure]
run_dll() {
  local ds="$1" measure="${2:-}" nw="${N_WORKERS:-30}" common
  common=(--n_splits "$N_SPLITS" --seed "$SEED")
  echo "=== $(date +%T)  dataset=${ds} measure=${measure:-default} $([[ $INSAMPLE == 1 ]] && echo in-sample || echo out-of-sample) ==="
  case "$ds" in
    naturalstories_spr|geco|meco|provo|ucl_reading|zuco_reading|onestop)
      local flag=(); [[ "$INSAMPLE" == "1" ]] || flag=(--out_of_sample)   # this driver is in-sample by default
      "${PY[@]}" scripts/ceilings/dll_behavioral.py --dataset "$ds" ${measure:+--measure "$measure"} \
        "${common[@]}" --n_workers "$nw" "${flag[@]}" "${OUT_ARGS[@]}" ;;
    pereira|tuckute|ns_fmri)
      "${PY[@]}" scripts/ceilings/dll_fmri.py --dataset "$ds" "${common[@]}" --n_workers "$nw" \
        $([[ $INSAMPLE == 1 ]] && echo --in_sample) "${OUT_ARGS[@]}" ;;
    fedorenko2016|podcast)
      "${PY[@]}" scripts/ceilings/dll_ecog.py --dataset "$ds" "${common[@]}" --n_workers "$nw" \
        $([[ $INSAMPLE == 1 ]] && echo --in_sample) "${OUT_ARGS[@]}" ;;
    lpp)
      "${PY[@]}" scripts/ceilings/dll_lpp.py "${common[@]}" --n_workers "$nw" \
        $([[ $INSAMPLE == 1 ]] && echo --in_sample) "${OUT_ARGS[@]}" ;;
    nieuwland)
      "${PY[@]}" scripts/ceilings/dll_n400.py --dataset nieuwland_paper --re_structure nieuwland_paper \
        "${common[@]}" --n_workers "$nw" $([[ $INSAMPLE == 1 ]] && echo --in_sample) "${OUT_ARGS[@]}" ;;
    szewczyk)
      "${PY[@]}" scripts/ceilings/dll_n400.py --dataset szewczyk --re_structure nieuwland_paper \
        "${common[@]}" --n_workers "$nw" $([[ $INSAMPLE == 1 ]] && echo --in_sample) "${OUT_ARGS[@]}" ;;
    megmasc)
      # Reuses a model's megmasc_lme_pooled_table.parquet; 12 workers keeps RAM under the node (890k-row fits).
      "${PY[@]}" scripts/ceilings/dll_megmasc.py --mode "$MODE" --n_splits "$N_SPLITS" --seed "$SEED" \
        --n_workers "${N_WORKERS:-12}" --nuisances "${NUISANCES:-worddur_z + wlen_z + freq_z + prev_worddur_z}" \
        --out_dir "$BENCHMARK_ROOT/outputs_jeanzay/noise_ceilings/megmasc" ;;
    *) echo "unknown dataset '$ds' (see the header of this script)" >&2; return 2 ;;
  esac
}

echo "Job started at $(date)  node=$(hostname)  nprocs=$(nproc)  R_HOME=${R_HOME}"
if [[ "$DATASET" == "all" ]]; then
  for ds in naturalstories_spr ucl_reading; do run_dll "$ds"; done
  for ds in geco; do for m in "" GP; do run_dll "$ds" "$m"; done; done
  for ds in meco provo onestop; do for m in "" GP SP; do run_dll "$ds" "$m"; done; done
  for m in "" SR TSR; do run_dll zuco_reading "$m"; done
  for ds in pereira tuckute ns_fmri fedorenko2016 podcast lpp nieuwland szewczyk; do run_dll "$ds"; done
else
  run_dll "$DATASET" "$MEASURE"
fi
echo "Done $(date)"
