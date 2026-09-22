#!/bin/bash
#SBATCH -A ldh@cpu
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --job-name=dlt_analysis
#SBATCH --time=02:00:00
#SBATCH --output=outputs_jeanzay/logs/dlt_analysis_%j.out
#
# Analyses that combine the DLT costs (costs.sh) with the sweep results. CPU only.
#
#   sbatch scripts/slurm/jz/utils/dlt/analysis.sh corr geco           # DLT cost vs. surprisal-effect correlation
#   sbatch scripts/slurm/jz/utils/dlt/analysis.sh loglik provo        # delta log-likelihood per DLT bin
#   sbatch scripts/slurm/jz/utils/dlt/analysis.sh difficulty          # easy/hard difficulty-scaling CSVs, every dataset
#   sbatch scripts/slurm/jz/utils/dlt/analysis.sh difficulty podcast_neural provo_embed   # chosen ones
#
# corr/loglik datasets: the keys of scripts/dlt/surprisal_corr.py / dlt/delta_loglik.py
# (podcast has its own scripts). difficulty runs the entries one after the other (a parallel run was
# once OOM-killed): naturalstories_lme geco_lme provo_lme ucl_reading_lme meco_lme zuco_reading_nr_lme
# podcast_neural podcast_lme broderick_neural fedorenko2016 provo_embed meco_embed ucl_reading_embed
# zuco_reading_nr_embed naturalstories_embed. Hard thresholds: THR="0 1 2 3 4 5 7" (env).
set -uo pipefail
: "${WORK:?Set WORK (e.g. export WORK=/lustre/fswork/projects/rech/ldh/una68ug)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"
cd "$BENCHMARK_ROOT"
if [[ -n "${SCRATCH:-}" ]]; then
  source "${BENCHMARK_ROOT}/scripts/slurm/jz/jz_cache_env.sh"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p outputs_jeanzay/logs
PY=".venv/bin/python"
MODE="${1:?usage: analysis.sh corr|loglik <dataset>  |  difficulty [name ...]}"; shift
echo "Job started at $(date) on $(hostname)  mode=${MODE}"

case "$MODE" in
  corr)
    DS="${1:?dataset required}"
    if [[ "$DS" == "podcast" ]]; then "$PY" scripts/dlt/surprisal_corr_podcast.py --n_jobs 6 --force
    else "$PY" scripts/dlt/surprisal_corr.py --dataset "$DS" --n_jobs 20 --force; fi ;;
  loglik)
    DS="${1:?dataset required}"
    if [[ "$DS" == "podcast" ]]; then "$PY" scripts/dlt/delta_loglik_podcast.py --n_jobs 20 --force
    else "$PY" scripts/dlt/delta_loglik.py --dataset "$DS" --n_jobs 20 --force; fi ;;
  difficulty)
    THR="${THR:-0 1 2 3 4 5 7}"
    OUT_DIR="outputs/dlt_difficulty"; LOG_DIR="${OUT_DIR}/logs"; mkdir -p "$LOG_DIR"
    D="outputs/dlt"; S="outputs_jeanzay"
    # behavioral LME / neural score tables: name dataset sweep_root dlt_cost perplexity_dir [n_jobs]
    run_plot () {
      local name="$1" dataset="$2" sweep_root="$3" dlt_cost="$4" perp_dir="$5" njobs="${6:-}"
      echo "=== [$(date +%T)] ${name} ==="
      "$PY" scripts/dlt/difficulty_scaling.py --dataset "$dataset" --sweep_root "$sweep_root" --dlt_cost "$dlt_cost" \
        --perplexity_dir "$perp_dir" --out_dir "$OUT_DIR" ${njobs:+--n_jobs "$njobs"} --hard_gt_thresholds $THR \
        > "${LOG_DIR}/${name}.log" 2>&1
      echo "  exit=$? -> ${LOG_DIR}/${name}.log"
    }
    # embedding-track score tables: name --sweep_root .. --dlt_cost .. [--perplexity_dir ..]
    run_new_schema () {
      local name="$1"; shift
      echo "=== [$(date +%T)] ${name} ==="
      "$PY" scripts/dlt/difficulty_scores.py --dataset "$name" --out_dir "$OUT_DIR" --hard_gt_thresholds $THR "$@" \
        > "${LOG_DIR}/${name}.log" 2>&1
      echo "  exit=$? -> ${LOG_DIR}/${name}.log"
    }
    declare -A JOBS=(
      [naturalstories_lme]="run_plot naturalstories_lme naturalstories $S/naturalstories/surprisal_sweep $D/naturalstories_dlt_cost.csv $S/naturalstories/perplexity"
      [geco_lme]="run_plot geco_lme geco $S/geco/surprisal_sweep/fp $D/geco_dlt_cost.csv $S/geco/perplexity"
      [provo_lme]="run_plot provo_lme provo $S/provo/surprisal_sweep $D/provo_dlt_cost.csv $S/provo/perplexity"
      [ucl_reading_lme]="run_plot ucl_reading_lme ucl_reading $S/ucl_reading/surprisal_sweep $D/ucl_reading_dlt_cost.csv $S/ucl_reading/perplexity"
      [meco_lme]="run_plot meco_lme meco $S/meco/surprisal_sweep $D/meco_dlt_cost.csv $S/meco/perplexity"
      [zuco_reading_nr_lme]="run_plot zuco_reading_nr_lme zuco_reading_nr $S/zuco_reading/surprisal_sweep/nr $D/zuco_reading_nr_dlt_cost.csv $S/zuco_reading/perplexity"
      [podcast_neural]="run_plot podcast_neural podcast $S/podcast/embedding_sweep $D/podcast_dlt_cost.csv $S/podcast/perplexity 8"
      [podcast_lme]="run_plot podcast_lme podcast_lme $S/podcast/surprisal_sweep $D/podcast_dlt_cost.csv $S/podcast/perplexity"
      [broderick_neural]="run_plot broderick_neural broderick $S/broderick/embedding_sweep $D/broderick_dlt_cost.csv $S/broderick/perplexity 8"
      [fedorenko2016]="run_new_schema fedorenko2016 --sweep_root $S/fedorenko2016/embedding_sweep --dlt_cost $D/fedorenko2016_dlt_cost.csv"
      [provo_embed]="run_new_schema provo_embed --sweep_root $S/provo/embedding_sweep --dlt_cost $D/provo_dlt_cost.csv --perplexity_dir $S/provo/perplexity"
      [meco_embed]="run_new_schema meco_embed --sweep_root $S/meco/embedding_sweep --dlt_cost $D/meco_dlt_cost.csv --perplexity_dir $S/meco/perplexity"
      [ucl_reading_embed]="run_new_schema ucl_reading_embed --sweep_root $S/ucl_reading/embedding_sweep --dlt_cost $D/ucl_reading_dlt_cost.csv --perplexity_dir $S/ucl_reading/perplexity"
      [zuco_reading_nr_embed]="run_new_schema zuco_reading_nr_embed --sweep_root $S/zuco_reading/embedding_sweep/nr --dlt_cost $D/zuco_reading_nr_dlt_cost.csv --perplexity_dir $S/zuco_reading/perplexity"
      [naturalstories_embed]="run_new_schema naturalstories_embed --sweep_root $S/naturalstories/embedding_sweep --dlt_cost $D/naturalstories_dlt_cost.csv --perplexity_dir $S/naturalstories/perplexity"
    )
    ORDER=(naturalstories_lme geco_lme provo_lme ucl_reading_lme meco_lme zuco_reading_nr_lme podcast_neural podcast_lme broderick_neural fedorenko2016 provo_embed meco_embed ucl_reading_embed zuco_reading_nr_embed naturalstories_embed)
    NAMES=("$@"); [[ ${#NAMES[@]} -eq 0 ]] && NAMES=("${ORDER[@]}")
    for n in "${NAMES[@]}"; do
      [[ -n "${JOBS[$n]:-}" ]] || { echo "unknown difficulty entry '$n'" >&2; continue; }
      # shellcheck disable=SC2086
      ${JOBS[$n]}
    done
    ls -la "${OUT_DIR}"/*_dlt_difficulty_scores_multi_threshold.csv 2>/dev/null ;;
  *) echo "unknown mode '$MODE'" >&2; exit 2 ;;
esac
echo "Job completed at $(date)"
