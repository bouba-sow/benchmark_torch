#!/usr/bin/env bash
# Refill the qos_gpu-dev user queue up to its ten-task limit while the
# last-layer Podcast ridge tasks run. Completed outputs are skipped by the worker.
set -euo pipefail
# Fit only the final layer (2026-10-12; caches keep every layer). EMBED_LAST_LAYER_ONLY=0 for the full sweep.
export EMBED_LAST_LAYER_ONLY="${EMBED_LAST_LAYER_ONLY:-1}"
cd /lustre/fswork/projects/rech/ldh/una68ug/benchmark
next=0
submission_log=outputs_torch/logs/podcast_last_layer_ridge_submissions.tsv
mkdir -p outputs_torch/logs
while (( next < 118 )); do
  queued="$(squeue --array -u "$USER" -h -o '%q' | awk '$1 == "qos_gpu-dev" {n++} END {print n+0}')"
  if (( queued < 10 )); then
    if job_id="$(sbatch --parsable --array="$next" scripts/slurm/torch/embedding/neural/podcast_last_layer_ridge.sh 2>&1)"; then
      printf '%s\t%s\n' "$next" "$job_id" >> "$submission_log"
      printf 'Submitted model task %s as job %s (%s/10 queued)\n' "$next" "$job_id" "$((queued + 1))"
      ((next += 1))
    else
      printf 'Submission deferred for model task %s: %s\n' "$next" "$job_id"
      sleep 20
    fi
  else
    sleep 20
  fi
done
printf 'Submitted all 118 model tasks.\n'
