#!/bin/bash
# Shared helpers for the per-dataset perplexity scripts (sourced, not submitted).
#
# Each perplexity/<dataset>.sh computes corpus perplexity for EVERY model in the registry
# (already-computed models are skipped, so a re-run after a time-out just continues),
# then the two CBR-RNN checkpoints, then rebuilds the summary CSV.
#
# Restrict or tune a run from the command line, e.g.
#   PPL_EXTRA_ARGS="--family opt" sbatch --gres=gpu:2 scripts/slurm/torch/perplexity/nieuwland.sh
#   PPL_EXTRA_ARGS="--force" sbatch scripts/slurm/torch/perplexity/lpp.sh

ppl_setup() {
  cd "$BENCHMARK_ROOT" || return 1
  if [[ -n "${SCRATCH:-}" ]]; then
    source "${BENCHMARK_ROOT}/scripts/slurm/torch/torch_cache_env.sh"
  fi
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
  if command -v uv >/dev/null 2>&1; then
    PY=(uv run --frozen --no-sync --no-dev python)
  else
    source .venv/bin/activate
    PY=(python)
  fi
  "${PY[@]}" scripts/maintenance/check_torch_gpu_arch.py || return 1
  echo "Job started at $(date)  node=$(hostname)  job=${SLURM_JOB_ID:-?}"
  nvidia-smi | grep -E "GPU|MiB" | head -6
  mkdir -p "$OUT_DIR"
}

# ppl_run <compute_script.py> [dataset args...]   e.g.  ppl_run perplexity/lpp.py --bids_root "$ROOT"
ppl_run() {
  local script="scripts/$1"; shift
  # shellcheck disable=SC2086
  "${PY[@]}" "$script" --all ${PPL_EXTRA_ARGS:-} "$@" --out_dir "$OUT_DIR" --device cuda
  ppl_cbr_rnn "$script" "$@"
  "${PY[@]}" "$script" --summarize --out_dir "$OUT_DIR"
  echo "Job completed at $(date)"
}

# CBR-RNN is a custom architecture, loaded by model id rather than through the registry.
ppl_cbr_rnn() {
  [[ "${PPL_NO_CBR:-0}" == "1" ]] && return 0   # datasets whose script has no CBR-RNN path (MEG-MASC)
  local script="$1"; shift
  local model
  for model in cclark/cbr-rnn cclark/cbr-rnn-m; do
    "${PY[@]}" "$script" --architecture cbr_rnn --model_id "$model" "$@" --out_dir "$OUT_DIR" --device cuda
  done
}
