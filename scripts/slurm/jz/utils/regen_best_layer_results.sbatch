#!/bin/bash
# Regenerate every result that depends on the best-layer surprisal score after the 2026-10-01 switch to
# TRAIN-selected layers (neuroai.analysis.benchmark_common.LAYER_SELECTION). Each step is independent: a
# failure is logged and the next step runs. Logs: $SCRATCH/archives/regen_logs/. Old results: tar in
# $SCRATCH/archives/summary_and_reports_before_train_selection_20261001.tar
#SBATCH --account=ldh@cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
set -uo pipefail
: "${WORK:?}"; export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORK/benchmark}"; cd "$BENCHMARK_ROOT"
source scripts/slurm/jz/jz_cache_env.sh 2>/dev/null || true
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=4
L="$SCRATCH/archives/regen_logs"; mkdir -p "$L"; : > "$L/_status.txt"
step() { name="$1"; shift; echo "=== $(date +%T) $name"; if .venv/bin/python "$@" > "$L/$name.log" 2>&1; then echo "OK    $name" >> "$L/_status.txt"; else echo "FAIL  $name (exit $?)" >> "$L/_status.txt"; fi; }
step 01_summarize_benchmark   scripts/summarize/benchmark.py
step 02_report                scripts/summarize/report.py
step 03_pareto_figures        scripts/figures/pareto_figures.py
step 04_figure3_family        scripts/figures/figure3_family_scaling.py
step 05_feature_mode          scripts/figures/feature_mode_comparison.py
step 06_matched_scale_table   scripts/analysis/matched_scale_master_table.py
step 07_perplexity_matched    scripts/analysis/perplexity_matched_scaling.py
step 08_figure8_scale_matched scripts/figures/figure8_scale_matched_comparison.py
step 09_figure9_pythia_ckpt   scripts/figures/figure9_pythia_checkpoint_scaling.py
step 10_figure10_provo        scripts/figures/figure10_provo_ohschuler.py
step 11_figure11_provo_form   scripts/figures/figure11_provo_formula_ablation.py
step 12_babylm_ckpt_scaling   scripts/figures/figure_babylm_checkpoint_scaling.py
step 13_benchmark_overview    scripts/figures/benchmark_overview.py
echo "=== done $(date +%T)"; cat "$L/_status.txt"
