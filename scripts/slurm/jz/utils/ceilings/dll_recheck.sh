#!/bin/bash
#SBATCH -A ldh@cpu
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/ceiling_recheck_%x_%j.out
# Re-run a DLL ceiling after a fix, writing to $SCRATCH/ceilings_recheck (NOT outputs/) so the
# published file is only replaced once the new value is validated.
#   sbatch --qos=qos_cpu-dev --job-name=cr_broderick scripts/slurm/jz/utils/ceilings/dll_recheck.sh broderick
#   sbatch --qos=qos_cpu-dev --job-name=cr_szewczyk  scripts/slurm/jz/utils/ceilings/dll_recheck.sh szewczyk
set -euo pipefail
: "${WORK:?}"; export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"; cd "$BENCHMARK_ROOT"
source scripts/slurm/jz/jz_cache_env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if ! command -v R >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then module load r/4.5.2 2>/dev/null || module load r 2>/dev/null || true; fi
export R_HOME="${R_HOME:-$(R RHOME)}" RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-ABI}"
[[ -d "${R_HOME}/lib" ]] && export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH:-}"
OUT="$SCRATCH/ceilings_recheck"; mkdir -p "$OUT"
case "${1:?broderick|szewczyk|fedorenko_loso|broderick_loso|szewczyk_loso|szewczyk_sf|szewczyk_sf_loso|szewczyk_sub|szewczyk_sub_loso|szewczyk_sub_extrap|szewczyk_sub_insample|diag}" in
  broderick) .venv/bin/python scripts/ceilings/dll_broderick.py --n_splits 15 --n_workers 30 --out_dir "$OUT" ;;
  szewczyk)  .venv/bin/python scripts/ceilings/dll_n400.py --dataset szewczyk_paper --re_structure szewczyk_paper \
               --n_splits 15 --n_workers 30 --out_dir "$OUT" ;;
  fedorenko_loso) .venv/bin/python scripts/ceilings/dll_ecog.py --dataset fedorenko2016 --mode loso \
               --n_repeats "${N_REPEATS:-20}" --n_workers 40 --out_dir "$OUT" ;;
  broderick_loso) .venv/bin/python scripts/ceilings/dll_broderick.py --mode loso --n_repeats "${N_REPEATS:-20}" \
               --n_workers 20 --out_dir "$OUT" ;;
  szewczyk_loso) .venv/bin/python scripts/ceilings/dll_n400.py --dataset szewczyk_paper --re_structure szewczyk_paper \
               --mode loso --n_repeats "${N_REPEATS:-20}" --n_workers 20 --out_dir "$OUT" ;;
  szewczyk_sf) .venv/bin/python scripts/ceilings/dll_n400.py --dataset szewczyk_paper --re_structure szewczyk_paper \
               --sub_experiment "Szewczyk & Federmeier (2022)" --n_splits 15 --n_workers 30 --out_dir "$OUT" ;;
  szewczyk_sf_loso) .venv/bin/python scripts/ceilings/dll_n400.py --dataset szewczyk_paper --re_structure szewczyk_paper \
               --sub_experiment "Szewczyk & Federmeier (2022)" --mode loso --n_repeats "${N_REPEATS:-20}" --n_workers 20 --out_dir "$OUT" ;;
  # any single Szewczyk sub-experiment: SUB="Federmeier et al. (2007)" sbatch ... dll_recheck.sh szewczyk_sub[_loso]
  szewczyk_sub) .venv/bin/python scripts/ceilings/dll_n400.py --dataset szewczyk_paper --re_structure szewczyk_paper \
               --sub_experiment "${SUB:?set SUB}" --n_splits 15 --n_workers 30 --out_dir "$OUT" ;;
  szewczyk_sub_loso) .venv/bin/python scripts/ceilings/dll_n400.py --dataset szewczyk_paper --re_structure szewczyk_paper \
               --sub_experiment "${SUB:?set SUB}" --mode loso --n_repeats "${N_REPEATS:-20}" --n_workers 20 --out_dir "$OUT" ;;
  szewczyk_sub_extrap) .venv/bin/python scripts/ceilings/dll_n400.py --dataset szewczyk_paper --re_structure szewczyk_paper \
               --sub_experiment "${SUB:?set SUB}" --mode extrap --target_size "${TARGET_SIZE:-8}" --n_draws "${N_DRAWS:-20}" \
               --n_workers 40 ${INSAMPLE:+--in_sample} --out_dir "$OUT" ;;
  szewczyk_sub_insample) .venv/bin/python scripts/ceilings/dll_n400.py --dataset szewczyk_paper --re_structure szewczyk_paper \
               --sub_experiment "${SUB:?set SUB}" --n_splits 15 --n_workers 30 --in_sample --out_dir "$OUT" ;;
  diag) .venv/bin/python "${DIAG_PY:?set DIAG_PY}" ;;
  *) echo "unknown $1" >&2; exit 2 ;;
esac
