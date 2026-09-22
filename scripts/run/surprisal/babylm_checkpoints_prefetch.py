#!/usr/bin/env python3
"""Unified last-layer-only surprisal + perplexity prefetch for the
``babylm_gpt2_checkpoints`` family (BabyLM-community/babylm-baseline-100m-gpt2,
29 training checkpoints), across a GROUP of behavioural/neural datasets.

Why this exists
---------------
The per-dataset prefetch entrypoints, run as one SLURM task each, (a) reload
the model for every (dataset, checkpoint) pair and (b) compute the full
per-layer logit-lens surprisal (12 vocab projections per word) even though
every downstream LME fit here is ``--last_layer_only``. This driver fixes both
in ONE process:

* **native final-layer-only surprisal** — monkeypatches
  ``logit_lens_surprisal_trials`` / ``logit_lens_surprisal_all_layers`` to force
  ``native_only=True`` (reads the model's own ``.logits``; skips the 11 wasted
  per-earlier-layer reconstructions). Cache path is unchanged; the resulting
  ``(N, 1)`` cache is read identically by a ``last_layer_only`` LME fit
  (``layer_indices = [n_layers - 1] = [0]``).
* **one model load per checkpoint** — loops ``for checkpoint: for dataset``.
  Datasets whose extractor does not call ``clear_hf_model_cache`` (the N400 /
  LPP / Tuckute benchmarks) reuse the ``_get_cached_causal_lm`` LRU slot for
  free within a checkpoint; the reading-dataset prefetch scripts still clear
  after themselves, so those reload (~7 s, cheap for a 100 M model).

Each per-dataset entrypoint is invoked in-process via ``runpy.run_path`` with a
constructed ``sys.argv`` — no per-dataset code is modified, and every
already-cached (dataset, checkpoint) is skipped by the entrypoint's own cache
check.

Usage
-----
    python scripts/run/surprisal/babylm_checkpoints_prefetch.py --datasets geco,meco --device cuda
"""
from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys
import traceback
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("babylm_gpt2_group")

FAMILY = "babylm_gpt2_checkpoints"

# --------------------------------------------------------------------------- #
# Environment defaults — mirror scripts/slurm/jz/surprisal/*/job_*_common.sh   #
# so this can run from a bare SLURM allocation that only exports WORK/SCRATCH. #
# --------------------------------------------------------------------------- #
_WORK = os.environ.get("WORK", "/lustre/fswork/projects/rech/ldh/una68ug")
_SCRATCH = os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug")
os.environ.setdefault("BENCHMARK_ROOT", f"{_WORK}/benchmark")
_BENCH = os.environ["BENCHMARK_ROOT"]
os.environ.setdefault("BENCHMARK_CACHE_ROOT", f"{_SCRATCH}/.cache")
_CR = os.environ["BENCHMARK_CACHE_ROOT"]
os.chdir(_BENCH)


def _d(key: str, val: str) -> None:
    os.environ.setdefault(key, val)


_d("HF_HOME", f"{_SCRATCH}/hf_cache")
_d("HF_HUB_CACHE", f"{_SCRATCH}/hf_cache/hub")
_d("HF_HUB_OFFLINE", "1")
_d("TRANSFORMERS_OFFLINE", "1")
_d("TOKENIZERS_PARALLELISM", "false")
_d("OMP_NUM_THREADS", "1")

# feature caches (shared with the two-stage per-dataset pipelines)
_d("GECO_FEATURE_CACHE_DIR", f"{_CR}/geco_features")
_d("MECO_FEATURE_CACHE_DIR", f"{_CR}/meco_features")
_d("PROVO_FEATURE_CACHE_DIR", f"{_CR}/provo_features")
_d("NS_FEATURE_CACHE_DIR", f"{_CR}/naturalstories_features")
_d("UCL_READING_FEATURE_CACHE_DIR", f"{_CR}/ucl_reading_features")
_d("ZUCO_TASK", "NR")
_d("ZUCO_READING_FEATURE_CACHE_DIR", f"{_CR}/zuco_reading_features/nr")
_d("NIEUWLAND_LME_FEATURE_CACHE_DIR", f"{_CR}/nieuwland_lme_features")
_d("NIEUWLAND_EPOCH_CACHE_DIR", f"{_CR}/nieuwland_epochs")
_d("LPP_LME_FEATURE_CACHE_DIR", f"{_CR}/lpp_lme_features")
_d("TUCKUTE_LME_FEATURE_CACHE_DIR", f"{_CR}/tuckute_lme_features")
_d("FEDORENKO2016_LME_FEATURE_CACHE_DIR", f"{_CR}/fedorenko2016_lme_features")
_d("PODCAST_FEATURE_CACHE_DIR", f"{_CR}/podcast_features")

# per-dataset perplexity dirs (outputs live under the repo, not $SCRATCH)
_d("GECO_PERPLEXITY_DIR", f"{_BENCH}/outputs_jeanzay/geco/perplexity")
_d("MECO_PERPLEXITY_DIR", f"{_BENCH}/outputs_jeanzay/meco/perplexity")
_d("PROVO_PERPLEXITY_DIR", f"{_BENCH}/outputs_jeanzay/provo/perplexity")
_d("UCL_READING_PERPLEXITY_DIR", f"{_BENCH}/outputs_jeanzay/ucl_reading/perplexity")
_d("ZUCO_READING_PERPLEXITY_DIR", f"{_BENCH}/outputs_jeanzay/zuco_reading/perplexity")

# dataset roots
_d("GECO_DATA_ROOT", f"{_WORK}/benchmark/data/geco")
_d("MECO_DATA_ROOT", f"{_WORK}/benchmark/data/meco/osfstorage/release 2.0/version 2.1/wave 1/primary data/eye tracking data")
_d("PROVO_DATA_ROOT", f"{_WORK}/benchmark/data/provo/osfstorage")
_d("NS_DATA_ROOT", f"{_WORK}/benchmark/data/naturalstories")
_d("UCL_READING_DATA_ROOT", f"{_WORK}/benchmark/data/ucl_reading")
_d("ZUCO_READING_DATA_ROOT", f"{_WORK}/benchmark/data/zuco_reading/osfstorage")
_d("NS_FMRI_DATA_ROOT", f"{_WORK}/benchmark/data/naturalstories_fmri/osfstorage")

for _p in (
    "GECO_FEATURE_CACHE_DIR", "MECO_FEATURE_CACHE_DIR", "PROVO_FEATURE_CACHE_DIR",
    "NS_FEATURE_CACHE_DIR", "UCL_READING_FEATURE_CACHE_DIR",
    "ZUCO_READING_FEATURE_CACHE_DIR", "NIEUWLAND_LME_FEATURE_CACHE_DIR",
    "NIEUWLAND_EPOCH_CACHE_DIR", "LPP_LME_FEATURE_CACHE_DIR",
    "TUCKUTE_LME_FEATURE_CACHE_DIR", "GECO_PERPLEXITY_DIR", "MECO_PERPLEXITY_DIR",
    "PROVO_PERPLEXITY_DIR", "UCL_READING_PERPLEXITY_DIR", "ZUCO_READING_PERPLEXITY_DIR",
    "FEDORENKO2016_LME_FEATURE_CACHE_DIR", "PODCAST_FEATURE_CACHE_DIR",
):
    Path(os.environ[_p]).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Monkeypatch: force native (final-layer-only) logit-lens everywhere.         #
# --------------------------------------------------------------------------- #
import neuroai.features.logit_lens as _ll  # noqa: E402


def _force_native(orig):
    def wrapped(*args, **kwargs):
        kwargs.setdefault("native_only", True)
        return orig(*args, **kwargs)

    wrapped.__name__ = getattr(orig, "__name__", "wrapped")
    return wrapped


_ll.logit_lens_surprisal_trials = _force_native(_ll.logit_lens_surprisal_trials)
_ll.logit_lens_surprisal_all_layers = _force_native(_ll.logit_lens_surprisal_all_layers)
log.info("Patched logit_lens_surprisal_{trials,all_layers} → native_only=True")

from neuroai.models.registry import flatten_family_tasks  # noqa: E402

_TASKS = flatten_family_tasks(FAMILY)  # 29 FamilyTask, .array_index 0..28


# --------------------------------------------------------------------------- #
# Per-dataset in-process invocation                                           #
# --------------------------------------------------------------------------- #
def _argv(ds: str, idx: int, device: str) -> list[str]:
    E = os.environ
    common_reg = ["--from-registry", "--family", FAMILY, "--array-index", str(idx),
                  "--device", device]
    if ds == "geco":
        return ["scripts/run/surprisal/behavioral/geco_extract.py",
                "--config", "configs/surprisal/behavioral/geco_fp.yaml",
                "--family", FAMILY, "--array-index", str(idx),
                "--geco-root", E["GECO_DATA_ROOT"],
                "--cache-dir", E["GECO_FEATURE_CACHE_DIR"],
                "--perplexity-dir", E["GECO_PERPLEXITY_DIR"], "--device", device]
    if ds == "meco":
        return ["scripts/run/surprisal/behavioral/meco_extract.py",
                "--config", "configs/surprisal/behavioral/meco_fp.yaml",
                "--family", FAMILY, "--array-index", str(idx),
                "--meco-root", E["MECO_DATA_ROOT"],
                "--cache-dir", E["MECO_FEATURE_CACHE_DIR"],
                "--perplexity-dir", E["MECO_PERPLEXITY_DIR"], "--device", device]
    if ds == "provo":
        return ["scripts/run/surprisal/behavioral/provo_extract.py",
                "--config", "configs/surprisal/behavioral/provo_fp.yaml",
                "--family", FAMILY, "--array-index", str(idx),
                "--provo-root", E["PROVO_DATA_ROOT"],
                "--cache-dir", E["PROVO_FEATURE_CACHE_DIR"],
                "--perplexity-dir", E["PROVO_PERPLEXITY_DIR"], "--device", device]
    if ds == "naturalstories":
        return ["scripts/run/surprisal/behavioral/naturalstories_extract.py",
                "--config", "configs/surprisal/behavioral/naturalstories.yaml",
                "--family", FAMILY, "--array-index", str(idx),
                "--dataset-root", E["NS_DATA_ROOT"],
                "--cache-dir", E["NS_FEATURE_CACHE_DIR"], "--device", device]
    if ds == "ucl_reading":
        return ["scripts/run/surprisal/behavioral/ucl_reading_extract.py",
                "--config", "configs/surprisal/behavioral/ucl_reading.yaml",
                "--family", FAMILY, "--array-index", str(idx),
                "--dataset-root", E["UCL_READING_DATA_ROOT"],
                "--cache-dir", E["UCL_READING_FEATURE_CACHE_DIR"],
                "--perplexity-dir", E["UCL_READING_PERPLEXITY_DIR"], "--device", device]
    if ds == "zuco_reading":
        return ["scripts/run/surprisal/behavioral/zuco_reading_extract.py",
                "--config", "configs/surprisal/behavioral/zuco_reading_nr.yaml",
                "--family", FAMILY, "--array-index", str(idx),
                "--mat-root", E["ZUCO_READING_DATA_ROOT"], "--task", "NR",
                "--cache-dir", E["ZUCO_READING_FEATURE_CACHE_DIR"],
                "--perplexity-dir", E["ZUCO_READING_PERPLEXITY_DIR"], "--device", device]
    if ds == "nieuwland":
        return ["scripts/run/surprisal/neural/nieuwland_extract.py",
                "--config", "configs/surprisal/neural/nieuwland.yaml", *common_reg,
                "--feature_cache_dir", E["NIEUWLAND_LME_FEATURE_CACHE_DIR"]]
    if ds == "lpp":
        return ["scripts/run/surprisal/neural/lpp_extract.py",
                "--config", "configs/surprisal/neural/lpp.yaml", *common_reg,
                "--feature_cache_dir", E["LPP_LME_FEATURE_CACHE_DIR"]]
    if ds == "tuckute":
        return ["scripts/run/surprisal/neural/tuckute_extract.py",
                "--config", "configs/surprisal/neural/tuckute.yaml", *common_reg,
                "--feature_cache_dir", E["TUCKUTE_LME_FEATURE_CACHE_DIR"]]
    if ds == "ns_fmri":
        return ["scripts/run/surprisal/neural/ns_fmri_extract.py",
                "--config", "configs/surprisal/neural/ns_fmri.yaml",
                "--family", FAMILY, "--array-index", str(idx),
                "--data-root", E["NS_FMRI_DATA_ROOT"],
                "--cache-dir", E["NS_FEATURE_CACHE_DIR"], "--device", device]
    if ds == "fedorenko2016":
        return ["scripts/run/surprisal/neural/fedorenko2016_extract.py",
                "--config", "configs/surprisal/neural/fedorenko2016.yaml", *common_reg,
                "--feature_cache_dir", E["FEDORENKO2016_LME_FEATURE_CACHE_DIR"]]
    if ds == "podcast":
        return ["scripts/run/surprisal/neural/podcast_extract.py",
                "--config", "configs/surprisal/neural/podcast.yaml",
                "--family", FAMILY, "--array-index", str(idx),
                "--cache-dir", E["PODCAST_FEATURE_CACHE_DIR"], "--device", device]
    raise KeyError(f"unknown dataset {ds!r}")


ALL_DATASETS = [
    "geco", "meco", "naturalstories", "provo", "ucl_reading", "zuco_reading",
    "nieuwland", "lpp", "tuckute", "ns_fmri",
    "fedorenko2016", "podcast",
]

# Datasets whose prefetch entrypoint does NOT compute corpus perplexity
# (naturalstories_spr's was historically a separate job — see
# scripts/perplexity/pythia_checkpoints_3datasets.py). We fill that gap
# here, reusing the model the surprisal pass just loaded.
_NEEDS_PERPLEXITY_HOOK = {"naturalstories"}


def _naturalstories_perplexity(task, device: str) -> None:
    import json

    from neuroai.features.llm import _get_cached_causal_lm
    from scripts.perplexity.lpp import compute_perplexity, _max_length_for_model
    from scripts.perplexity.naturalstories import load_naturalstories_text

    rev = task.revision if task.revision not in (None, "", "main") else None
    rev_safe = (task.revision or "main").replace("/", "_")
    safe = task.load_model_id.replace("/", "_")
    out_path = Path("outputs_jeanzay") / "naturalstories/spr_perplexity" / safe / f"{rev_safe}_naturalstories_spr_perplexity.json"
    if out_path.is_file():
        log.info("naturalstories_spr perplexity already cached for %s — skipping", task.revision)
        return
    try:
        tokenizer, model = _get_cached_causal_lm(
            task.load_model_id, revision=rev, device=device,
            load_kw={"trust_remote_code": False}, tokenizer_no_revision=True,
            n_params=98_425_344,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_length = _max_length_for_model(model, task.load_model_id)
        texts = load_naturalstories_text(Path("data/naturalstories"))
        res = compute_perplexity(model, tokenizer, texts, device, max_length)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model_id": task.load_model_id, "revision": task.revision or "main",
            "n_params": 98_425_344, "dataset": "naturalstories_spr",
            "perplexity": res["perplexity"], "n_tokens": res["n_tokens"],
            "max_length": max_length,
        }, indent=2))
        log.info("Saved naturalstories_spr perplexity %.4f for %s", res["perplexity"], task.revision or "main")
    except Exception:  # noqa: BLE001
        log.error("naturalstories_spr perplexity FAILED for %s:\n%s", task.revision, traceback.format_exc())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True,
                    help="comma-separated subset of: " + ",".join(ALL_DATASETS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--start-index", type=int, default=0, help="first checkpoint array-index (default 0)")
    ap.add_argument("--end-index", type=int, default=len(_TASKS) - 1, help="last checkpoint array-index inclusive")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    bad = [d for d in datasets if d not in ALL_DATASETS]
    if bad:
        ap.error(f"unknown datasets: {bad}")

    tasks = [t for t in _TASKS if args.start_index <= t.array_index <= args.end_index]
    log.info("group datasets=%s  checkpoints=%d (idx %d..%d)  device=%s",
             datasets, len(tasks), args.start_index, args.end_index, args.device)

    n_ok = n_fail = 0
    for task in tasks:
        idx = task.array_index
        for ds in datasets:
            argv = _argv(ds, idx, args.device)
            script = argv[0]
            log.info("═══ checkpoint %-9s (idx=%2d) → %-14s ═══", task.revision, idx, ds)
            saved_argv = sys.argv
            sys.argv = argv
            try:
                runpy.run_path(script, run_name="__main__")
                n_ok += 1
            except SystemExit as exc:
                if exc.code not in (0, None):
                    n_fail += 1
                    log.error("%s @ %s exited with code %s", ds, task.revision, exc.code)
                else:
                    n_ok += 1
            except Exception:  # noqa: BLE001
                n_fail += 1
                log.error("%s @ %s FAILED:\n%s", ds, task.revision, traceback.format_exc())
            finally:
                sys.argv = saved_argv

            if ds in _NEEDS_PERPLEXITY_HOOK:
                _naturalstories_perplexity(task, args.device)

    log.info("DONE  ok=%d  failed=%d", n_ok, n_fail)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
