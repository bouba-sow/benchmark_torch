# The NYU "Torch" cluster

This benchmark was originally written for **Jean Zay** (`docs/running.md`, `scripts/slurm/jz/`).
This doc is the reference for adapting and running it on NYU's **Torch** HPC cluster instead.
It is documentation only — `scripts/slurm/jz/` and `stages.yaml` still hard-code Jean Zay
paths/accounts/modules; the actual `scripts/slurm/torch/` job tree and a `torch` profile in
`stages.yaml` are future work (see "Adapting the benchmark" at the end).

## Access

* Login host: `torch-login-a-2` (there are multiple `torch-login-*` nodes; SSH round-robins).
* Repo: `/scratch/bs5813/Documents/benchmark`.
* **SLURM accounts** (`--account=`, required — the bare value `users` is rejected):
  * `torch_pr_287_cds`
  * `torch_pr_287_general`

  Both are valid; pick explicitly with `--account=` per job rather than relying on one default:
  * `_cds` — the CDS group's own **dedicated, always-available slice**: 24 H200 GPUs and 24
    A100 GPUs group-wide (`h200_cds`/`a100_cds` partitions, verified via `sacctmgr show qos` —
    see Partitions and QOS below). Small compared to bigger stakeholders on the same hardware
    (Tandon has 112 H200 GPUs reserved, Courant 88), but nobody outside CDS contends for it.
  * `_general` — draws from the **full shared/idle pool across all GPU types**
    (272 H200, 272 L40S, 172 A100, 120 B200, 60 H100, 48 RTX Pro 6000 cluster-wide), capped
    **per user** rather than per group (16 concurrent GPUs under the `gpu48` QOS — see below),
    at lower priority than groups with their own dedicated reservation on that hardware.

  Rule of thumb: use `_cds` when you need guaranteed, contention-free H200/A100 capacity for a
  modest number of concurrent jobs; use `_general` for larger sweeps that need more than 24
  concurrent GPUs, accepting it may queue behind dedicated-reservation groups.
  `torch_pr_287_general` was the account used for prior NYU work on this repo (ZuCo migration,
  model prefetch).
* List your available accounts: `my_slurm_accounts` (or `sacctmgr show assoc user=$USER`).

## Hardware (June 2025 spec sheet)

| Node type | Nodes | GPUs | Cores/node | Mem/node |
|---|---|---|---|---|
| Standard memory (CPU) | 186 | — | 128 | 512 GB |
| Large memory (CPU) | 8 | — | 128 | 3072 GB |
| H200 | 34 | 8 | 128 | 2048 GB |
| L40S | 68 | 4 | 128 | 512 GB |
| B200 | 15 | 8 | 128 | 512 GB |
| H100 | 15 | 4 | 96 | 1536 GB |
| A100 | 43 | 4 | 76 | 512–1024 GB |
| RTX Pro 6000 | 6 | 8 | 128 | 1536 GB |

Total: 45,248 CPU cores, 944 GPUs, ~332 TB memory, Infiniband NDR400. Ranked #133 on Top500 /
#40 Green500 (June 2025).

**GPU type matters for kernels, not just capacity**: the main `.venv`'s CUDA 12.4 PyTorch
wheel has no `sm_89` kernels, so **L40S GPUs fail** for this repo (`check_torch_gpu_arch.py`
raises). Constrain GPU jobs to `h200|h100|a100` (or the newer `b200`/`rtx6000` once verified),
e.g. `--constraint='h100|h200|a100'`. Do not use L40S until the wheel is rebuilt with `sm_89`
kernels or a separate wheel is used for those nodes.

## Partitions and QOS

Pulled live from `sinfo` / `sacctmgr show qos` / `scontrol show partition` on 2026-09-22 —
treat as ground truth over anything inferred from docs alone, but re-check if it's been a
while (`sinfo`, `sacctmgr show qos` are free to run any time).

**How this cluster is actually organized**: for a given GPU type, every group-suffixed
partition (`h200_cds`, `h200_tandon`, `h200_public`, `h200`, `h200_plus`, ...) points at the
*same physical nodes* (e.g. all `h200*` partitions list `gh[101-134]`, the same 34 nodes).
The partition name doesn't carve out separate hardware — it pairs (via the site's submission
plugin) with a matching QOS that caps *your* concurrent usage and sets your priority tier. The
real caps live in `sacctmgr show qos`, not in `scontrol show partition` (partitions themselves
show `AllowAccounts=ALL AllowQos=ALL` — no partition-level restriction). Practically: pick the
GPU type with `--constraint=` and let `--account=` (+ whatever the plugin auto-assigns) put you
in the right pool; **don't pass `--qos=` by hand** — my own association only lists `QOS=normal`
even though `_cds`/`gpu48`/etc. clearly govern real jobs, so the actual QOS is assigned behind
the scenes from the account/partition combo, not chosen client-side.

**GPU inventory** (`sinfo`, node×GPU/node):

| Type | Nodes | GPUs/node | Total GPUs | Kernel note |
|---|---|---|---|---|
| H200 | 34 | 8 | 272 | OK |
| L40S | 68 | 4 | 272 | **sm_89 unsupported by this repo's CUDA 12.4 wheel — do not use** |
| A100 | 43 | 4 | 172 | OK |
| B200 | 15 | 8 | 120 | unverified with this repo's stack |
| H100 | 15 | 4 | 60 | OK |
| RTX Pro 6000 | 6 | 8 | 48 | unverified with this repo's stack |

**Group-dedicated GPU reservations** (`GrpTRES` on the group-suffixed QOS — a hard cap on that
group's *total* concurrent usage, shared across everyone in the group, no queueing against
other groups):

| QOS (≈ partition) | GPUs reserved | CPUs | Mem |
|---|---|---|---|
| `h200_cds` | 24 | 384 | 6000G |
| `h200_bpeher` | 24 | 384 | 6000G |
| `h200_public` | 24 | 384 | 6000G |
| `h200_courant` | 88 | 1408 | 2200G+ |
| `h200_tandon` | 112 | 1792 | 2800G+ |
| `a100_cds` | 24 | 480 | 6000G |
| `a100_cilvr` | 32 | 512 | 4000G |
| `a100_chemistry` | 56 | 1120 | 1400G+ |
| `a100_tandon` | 60 | 1184 | 1450G+ |
| `l40s_mren` | 4 | 128 | 500G |
| `l40s_lpinto` | 8 | 256 | 1000G |
| `l40s_courant` | 52 | 1664 | 6500G |
| `l40s_public` | 208 | 6656 | 26000G+ |
| `rtx6000_*` (lzanna/bpeher/courant/economics/xwang/humanities) | 8 each | 128 | 1400G |

This is the concrete version of "other groups have priority" from the Access section: CDS's
own guaranteed slice (24 GPUs on H200/A100) is small next to Tandon's or Courant's, but it's
exclusively yours — no cross-group contention.

**No per-user cap within `_cds`** (`MaxTRESPerUser` is blank on `h200_cds`/`a100_cds`) — if
you're the only CDS member running anything, you can burst up to the full 24-GPU group ceiling
yourself. But the ceiling is shared with every other CDS user, so someone else's jobs can leave
you queued well before you'd hit "your" 24. Check current usage before assuming it's free:
```bash
squeue -A torch_pr_287_cds -o "%.9i %.10P %.10u %.9T %.10M %.6D %.14b %R"  # everyone on _cds, readable columns
squeue -A torch_pr_287_cds -u $USER                                       # just your own
```
Aliased to `sq_cds` in `~/.bashrc`:
```bash
alias sq_cds='squeue -A torch_pr_287_cds -o "%.9i %.10P %.10u %.9T %.10M %.6D %.14b %R"'
```
Pending jobs show a `NODELIST(REASON)` — `QOSGrpGRES` means the group/QOS GPU ceiling is
currently full (see the group-reservation table above); `ReqNodeNotAvail, UnavailableNodes:...`
means specific nodes are down/drained, unrelated to any cap. `squeue --start -j <jobid>` gives
the scheduler's own ETA for a pending job.

**Per-user QOS** (`MaxTRESPU` — caps one user, drawn from the shared/idle pool across *all*
groups' unclaimed capacity; this is what `_general` jobs land in):

| QOS | Max wall | Per-user cap | Max submitted jobs/user |
|---|---|---|---|
| `cpu_short` | 6h | cpu=32, mem=120G | — |
| `cpu48` | 2d | cpu=3000, mem=6000G | 3000 |
| `cpu168` | 7d | cpu=1000, mem=2000G | 500 |
| `cpuprem` (preemptible) | 2d | cpu=20000 | 5000 |
| `cpuplus` | unlimited | cpu=6000 | — |
| `interactive` | 6h | cpu=16, mem=60G | 20 |
| `gpu48` | 2d | **gpu=16** | 500 |
| `gpu168` | 7d | gpu=4 | 50 |
| `gpuplus` (preemptible/burst) | 7d | gpu=64 | 50 |

This resolves the earlier ambiguity against the general NYU docs' "24-GPU/<48h" language:
**the real per-user concurrent-GPU cap on Torch is 16** (`gpu48`), confirmed directly from
`sacctmgr show qos`, not 24 — plan sweep throttling around 16.

* **CPU jobs**: `cs` for general-purpose (184 nodes, no GPU, no walltime cap seen at the
  partition level — real limit comes from whichever CPU QOS gets assigned, `cpu48`/`cpu168`).
  Prior empirical testing on this repo saw `cpu_short`/`cpu_prem` reject jobs with
  `--time > 4h`, which sits *below* `cpu_short`'s documented 6h QOS max — re-validate with
  `sbatch --test-only` rather than trusting either number blindly, since site config can drift.
* **GPU partitions**: match the GPU-inventory table above; use `--constraint=` for the type,
  not a `_public`/`_plus`/`_cds` partition suffix, unless deliberately targeting a specific
  group's reservation.
* Always validate a resource combo before a real submission: `sbatch --test-only <script>`.

## Environment setup

* **Do not use the login shell's Anaconda base directly.** It pollutes `LD_LIBRARY_PATH` with
  an old `libcrypto`, which breaks `import hashlib` under this repo's `.venv` (built against
  system `/usr/bin/python3.12`). Any job script sourced here must scrub the environment before
  activating the venv:
  ```bash
  unset LD_LIBRARY_PATH
  unset PYTHONPATH
  # strip conda entries from PATH before activating .venv
  ```
  Do **not** `sbatch --export=ALL` a login shell's polluted environment unscrubbed.
* Main environment is still `uv sync` per `docs/dependencies.md` — conda is only relevant if
  you follow NYU's own conda-env docs for something outside this repo's `.venv` (NYU
  recommends prefix envs: `conda create -p ./penv python`, `source activate ./penv`, not
  `conda activate`, stored under `/scratch` since HPC discourages home-directory envs).
* R via `pymer4` (surprisal track) needs an R module — module name/version on Torch is
  **unconfirmed**; on Jean Zay it was `r/4.5.2` + `gcc/11.5.0` (`scripts/slurm/jz/jz_cache_env.sh`).
  Check `module avail r` on Torch before assuming the same versions exist.
* **Login-node `/tmp` is a 2 GB tmpfs, chronically full** — breaks `git`, `pwd`, anything
  using `/tmp` by default, including `.git` discovery in this repo. Set
  `TMPDIR=$SCRATCH/tmp` (or any scratch path) before running git/build tooling on the login
  node.

## Filesystems and caches

Unlike Jean Zay's `$WORK`/`$SCRATCH` split (`scripts/slurm/jz/jz_cache_env.sh`), Torch has a
single project scratch:

* `$SCRATCH=/scratch/bs5813` — 5 TB quota (~18% used as of last check; cluster-wide usage
  reported ~82% full, so don't assume headroom).
* `$HOME` — 50 GB quota, small; do not put caches or data here.
* HF cache: point `HF_HOME=/scratch/bs5813/hf_cache` explicitly — the default login env
  otherwise resolves `HF_HOME` to `/scratch/bs5813/.huggingface`, which is not what job
  scripts expect.
* There is no `jz_cache_env.sh` equivalent yet for Torch. When building one, every
  `BENCHMARK_CACHE_ROOT`-derived path in `jz_cache_env.sh` should resolve under
  `$SCRATCH/.cache` the same way, just without the `$WORK` distinction — i.e.
  `BENCHMARK_ROOT=$SCRATCH/Documents/benchmark` (or wherever the repo actually lives) instead
  of `${WORK}/benchmark`.

## Downloading / prefetching model weights

* **Login nodes SIGKILL large or parallel downloads** (cgroup memory limit; made worse by
  `HF_HUB_ENABLE_HF_TRANSFER=1`). Never run bulk HF downloads on a login node.
* Prefetch via `sbatch` on `cpu_short` (short jobs only, see partition note above). The
  existing prefetch tooling lives at `scripts/slurm/jz/prefetch_models_hf.sh` /
  `submit_prefetch_array.sh` (107 model repos, `main` revision only, `xargs -P $JOBS`
  fan-out, per-repo retry with backoff, refuses to run on a `*login*` host unless
  `ALLOW_LOGIN_NODE=1`) — reusable as-is once the account/partition flags are swapped to
  Torch's (`--account=torch_pr_287_general --partition=cpu_short`).
* Array form (preferred): `--array=0-106%16`, 2 cpus / 16 GB / 8h per task — fits under the
  4h-or-less requirement only if trimmed; check current per-repo timing before reusing the 8h
  default, since `cpu_short` here caps at 4h (see Partitions above — this may need
  `--partition=cs` instead if a single task legitimately needs more than 4h).

## SLURM command cheat sheet

| Command | Purpose |
|---|---|
| `sbatch script.sh` | submit a batch job (`#SBATCH` directives in the script) |
| `sbatch --test-only script.sh` | validate a resource request without submitting |
| `srun --pty --gres=gpu:1 --cpus-per-task=N --mem=MG --time=HH:MM:SS /bin/bash` | interactive session |
| `salloc ...` | interactive allocation (same flags as `sbatch`/`srun`) |
| `squeue --me` | your queued/running jobs |
| `sinfo` | partition/node availability |
| `sacct -u $USER --format=JobID,JobName,State,Elapsed,AllocCPUs` | job accounting/history |
| `scancel <jobid>` / `scancel --me` | cancel one job / all your jobs |
| `my_slurm_accounts` | list accounts you can submit under (Torch-specific) |

Minimal `sbatch` header:
```bash
#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --output=output.out
#SBATCH --account=torch_pr_287_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=HH:MM:SS
```
GPU job: add `#SBATCH --gres=gpu:1` and `#SBATCH --constraint='h100|h200|a100'` (not L40S, see
above).

## Scheduling etiquette (from NYU's own docs)

* Never run heavy compute on the login node — this is also why downloads there get SIGKILLed;
  schedule everything, use Data Transfer Nodes/Globus for large transfers.
* Test on a small case before submitting a large sweep; check `sacct -u $USER` to calibrate
  `--time`/`--mem` from real usage, then pad by ~20–30% rather than guessing generously —
  over-padded requests wait longer for the scheduler to find room.
* Archive/compress (`tar`, `gzip`) before large transfers.
* No assumption of automatic backups — this repo's `outputs*/` and `data/` are exactly the
  kind of expensive-to-regenerate content that should get its own backup plan (see
  `CLAUDE.md`'s rule against touching `outputs/`, `outputs_jeanzay/`, `data/`).

## Known gotchas hit during prior NYU work on this repo

1. **Env scrub required** (see Environment setup above) — otherwise `import hashlib` dies
   under `.venv`.
2. **GPU `--constraint` must exclude L40S** (`sm_89` not in the CUDA 12.4 wheel's kernels).
3. **`gpu48` QOS**: 16 concurrent running GPUs per user / 500 max submitted array tasks / 2-day
   wall (confirmed via `sacctmgr show qos`, see Partitions and QOS) — sweep submission needs a
   wave/throttle strategy (submit in batches, or a wave-submission job) rather than one giant
   array, once the array would exceed either cap. `_cds` jobs instead share a 24-GPU
   *group-wide* cap (`h200_cds`/`a100_cds`), so sweeps launched under `_cds` throttle even
   lower if others in the group are also running.
4. **Triton `Python.h: No such file` (fatal)** for `alibi`/`forgetting`/`stickbreaking`
   families (flash-linear-attention JIT-compiles a CUDA shim at runtime): compute nodes have
   no Python dev headers, and none are available via `module load` either. Unresolved — needs
   either an Apptainer/container with headers, or copying a prebuilt shim from Jean Zay.
5. **HF offline on compute nodes** — prefetch weights from a login/DTN node first
   (`ALLOW_LOGIN_NODE=1`, short job, see above). Checkpointed models (e.g. OLMo-2) need their
   specific revision tags, not just `main`.
6. **Login-node `/tmp`** is a shared 2 GB tmpfs, chronically full — route anything that writes
   temp files (including `git`) to `$SCRATCH` via `TMPDIR`.

## Adapting the benchmark to Torch (done)

`scripts/slurm/torch/` (mirrors `scripts/slurm/jz/`, same runner/function split, `_jz_` ->
`_torch_`) plus `torch_cache_env.sh` (single `$SCRATCH`, `$WORK`/`$SCRATCH` split collapsed,
env scrub from Environment setup baked in) now exist, converted 2026-09-22. `stages.yaml`'s
`cpu`/`gpu` profiles point at `torch_pr_287_general` (`--partition=cs` for CPU,
`--constraint=h100|h200|a100` for GPU — no separate `qos:`, per the "don't pass `--qos` by
hand" note above) and its `runner:` paths point into `scripts/slurm/torch/`. R is
`module load r/4.6.1` (confirmed working, already the default-loaded module — no separate gcc
module needed, unlike Jean Zay's `r/4.5.2` + `gcc/11.5.0`). `scripts/slurm/jz/` is kept
unmodified for provenance; nothing in `stages.yaml` references it anymore.

Known gaps carried over from this conversion, not yet resolved:

* **No Torch equivalent of Jean Zay's SLURM-managed MPS** (`sinfo -o "%P %f"` lists no `mps`
  feature on any partition, checked 2026-09-22). The Broderick embedding-fit job
  (`embedding/neural/broderick.sh`) is forced to `N_SUBJECTS_BATCH=1` (sequential per subject)
  as the safe default instead of Jean Zay's concurrent multi-subject batching under MPS —
  revisit if Torch ever adds an MPS-capable partition.
* The `_general` vs `_cds` mapping suggested here (large GPU sweeps -> `_general`, smaller
  CPU/refit stages -> `_cds`) was **not** applied to `stages.yaml` — both profiles currently
  use `_general` throughout (simplest default, matches prior NYU sessions on this repo). Revisit
  per-stage if `_cds`'s guaranteed slice turns out to matter for turnaround.
* Whether Jean Zay's empirical R-on-H100-partition unavailability (see the two-stage
  GPU/CPU split in e.g. `surprisal/behavioral/zuco_reading.sh`, `ucl_reading.sh`) also applies
  to Torch is untested — the split was kept as a no-cost precaution, not re-verified here.

## Sources

NYU RT HPC docs (fetched 2026-09-22): `submitting_jobs/slurm_submitting_jobs`,
`submitting_jobs/slurm_main_commands`, `tools_and_software/conda_environments`, `spec_sheet`,
`tutorial_intro_hpc/using_resources_effectively`, `tutorial_intro_hpc/scheduler_fundamentals`,
`tutorial_intro_hpc/using_resources_responsibly` (all under `services.rt.nyu.edu/docs/hpc/`);
live `sinfo` / `sacctmgr show qos` / `sacctmgr show assoc` / `scontrol show partition` output
from `torch-login-b-1` (2026-09-22, this session); plus empirical findings from prior NYU
Torch sessions on this repo (ZuCo migration, model prefetch).
