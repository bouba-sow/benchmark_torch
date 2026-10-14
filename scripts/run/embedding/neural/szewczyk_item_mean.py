#!/usr/bin/env python3
"""Single-study Szewczyk & Federmeier N400 embedding encoding on the subject-averaged N400.

One study of ``szewczyk_2022.tsv`` per run (``--study f2007|h2019``): the five experiments in that
file are separate datasets and are NOT pooled.

Target : per-item mean over the study's subjects of the paper's N400 (300-500 ms, 4 centro-parietal
         ROI clusters) with the pre-stimulus baseline regressed out (neuroai.datasets.szewczyk_paper).
X      : hidden states of the critical word (every layer), extracted here on the GPU and cached.
Model  : Himalaya ``RidgeCV`` on GPU, one alpha per layer/fold chosen by inner group-CV (same code as
         the Nieuwland item-mean track, imported from nieuwland_paper_item_mean.py).
CV     : outer 10-fold GroupKFold over frames (``dataset_Item``), inner 10-fold over the same groups.
Score  : Spearman / Pearson of out-of-fold predictions vs item means, per layer. Noise ceiling:
         split-half subject reliability of the same target (scripts/ceilings/correlation.py).

    python scripts/run/embedding/neural/szewczyk_item_mean.py --study f2007 \\
        --from-registry --family gpt2 --array-index 0 --out_dir <root> --feature_cache_dir <dir>
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from neuroai.benchmarks.eeg_erp import _feature_cache_path, _items_hash, extract_sentence_features
from neuroai.datasets.szewczyk_paper import STUDIES, TSV, item_means, load_study, study_items
from neuroai.models.registry import registry_model_configs, resolve_family_task
from neuroai.sweep import safe_model_dir, setup_logging
from nieuwland_paper_item_mean import _pearson, _spearman, fit_layers

setup_logging()
logger = logging.getLogger("szewczyk_item_mean")

# 7 alphas, 10^1.5 ... 10^5.25. Calibrated on one model per family (both studies, the wide Nieuwland
# grid 10^-2..10^14): the per-layer alpha chosen by the inner CV ranged 10^2.0-10^4.6 (mode 10^2.75
# and 10^3.5, independent of model width), so this brackets that range by ~0.5 decade each side.
ALPHAS = [10 ** e for e in np.linspace(1.5, 5.25, 7)]


OLD_RDS = "data/szewczyk/osfstorage/bind-5-dset_300-500.rds"


def features_from_pooled_cache(items: pd.DataFrame, model_name: str, hf_rev: str | None, cache_dir: Path):
    """Slice this study's items out of the features cached by the OLD pooled Szewczyk embedding run
    (keyed by that run's item list, ``neuroai.datasets.szewczyk``): every (context, critical word)
    of the single-study tables occurs there with the identical strings, so the hidden states are
    bit-identical to a fresh extraction and no model has to be loaded. Returns None if the old
    cache for this model is missing or any item is not found."""
    from neuroai.datasets.szewczyk import build_items_df, load_data

    rds = build_items_df(load_data(REPO / OLD_RDS))
    ctx, wrd = rds["context"].tolist(), rds["critical_word"].tolist()
    path = _feature_cache_path(cache_dir, model_name, hf_rev, "all", _items_hash(ctx, wrd))
    if not path.is_file():
        return None
    pos: dict[tuple[str, str], int] = {}
    for i, key in enumerate(zip(ctx, wrd)):
        pos.setdefault(key, i)
    rows = [pos.get((c, w)) for c, w in zip(items["context"], items["critical_word"])]
    if any(r is None for r in rows):
        return None
    arr = np.load(path)["features"]
    if arr.shape[0] != len(ctx) or not np.isfinite(arr).all():
        return None
    logger.info("features sliced from pooled-run cache %s (no model load)", path.name)
    return arr[rows]


def run(study: str, family: str, index: int, out_root: Path, cache_dir: Path, device: str) -> None:
    task = resolve_family_task(family, index)
    model_cfg = next((m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id), {})
    out_dir = out_root / family / safe_model_dir(task.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    revision = task.revision or "main"
    scores_csv = out_dir / f"szewczyk_{study}_item_mean_scores.csv"
    if scores_csv.is_file() and (pd.read_csv(scores_csv)["revision"].astype(str) == revision).any():
        logger.info("%s@%s already done", task.model_id, revision)
        return

    df = load_study(REPO / TSV, study)
    items = item_means(df, study_items(df))
    hf_rev = None if revision in ("", "main") else revision
    new_cache = _feature_cache_path(cache_dir, task.load_model_id, hf_rev, "all",
                                    _items_hash(items["context"].tolist(), items["critical_word"].tolist()))
    features = None if new_cache.is_file() else features_from_pooled_cache(items, task.load_model_id, hf_rev, cache_dir)
    if features is None:
        features = extract_sentence_features(
            items["context"].tolist(), items["critical_word"].tolist(), task.load_model_id,
            revision=hf_rev, layer_mode="all", device=device, batch_size=8,
            tokenizer_no_revision=True, cache_dir=cache_dir,
            n_params=float(model_cfg.get("n_params", 0.0)) or None,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
    if features.ndim == 2:
        features = features[:, None, :]
    from neuroai.features.llm import select_embedding_layers

    features = select_embedding_layers(features)
    if not np.isfinite(features).all():
        raise ValueError("non-finite features")
    y = items["y"].to_numpy(dtype=np.float64)
    frames = items["chunk"].to_numpy()
    logger.info("%s [%s]: features %s, %d items, %d frames, device=%s", task.model_id, study,
                features.shape, len(y), len(np.unique(frames)), device)

    t0 = time.time()
    oof, alphas = fit_layers(features, y, frames, device, alphas_grid=ALPHAS)
    layers = pd.DataFrame([dict(family=family, model_id=task.model_id, revision=revision, layer=li,
                                spearmanr=_spearman(oof[li], y), pearsonr=_pearson(oof[li], y),
                                median_alpha=float(np.median(alphas[li])), n_items=len(y),
                                n_layers=features.shape[1]) for li in range(features.shape[1])])
    layers.to_csv(out_dir / f"szewczyk_{study}_item_mean_layerwise.csv", index=False)
    best = int(layers["spearmanr"].idxmax())
    np.savez_compressed(out_dir / f"szewczyk_{study}_item_mean_oof.npz", oof=oof.astype(np.float32), y=y,
                        chunk=frames.astype(str))
    pd.DataFrame([dict(family=family, model_id=task.model_id, revision=revision,
                       n_params=float(model_cfg.get("n_params", 0.0)), n_layers=features.shape[1],
                       best_layer=best, best_spearmanr=float(layers.loc[best, "spearmanr"]),
                       best_pearsonr=float(layers.loc[best, "pearsonr"]),
                       last_layer_spearmanr=float(layers["spearmanr"].iloc[-1]),
                       last_layer_pearsonr=float(layers["pearsonr"].iloc[-1]),
                       n_items=len(y), fit_seconds=round(time.time() - t0, 1))]).to_csv(scores_csv, index=False)
    logger.info("%s [%s]: best layer %d spearman=%.3f pearson=%.3f (%.0fs)", task.model_id, study, best,
                layers.loc[best, "spearmanr"], layers.loc[best, "pearsonr"], time.time() - t0)


def run_all(studies: list[str], out_roots: dict[str, Path], cache_dir: Path, device: str,
            min_params: float, max_params: float, families: list[str] | None) -> int:
    """Every model of every registry family, all ``studies`` per model (the model is loaded once:
    the LLM cache is size-1 and both studies' features come from the same loaded model). One failing
    model is logged and skipped. Returns the number of failures."""
    from neuroai.models.registry import LPP_FAMILIES, flatten_family_tasks

    fams = families or list(LPP_FAMILIES)
    failures = 0
    for family in fams:
        tasks = flatten_family_tasks(family)
        for idx in range(len(tasks)):
            task = tasks[idx]
            cfg = next((m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id), {})
            n_params = float(cfg.get("n_params", 0.0))
            if not (min_params <= n_params < max_params):
                continue
            for study in studies:
                try:
                    run(study, family, idx, out_roots[study], cache_dir, device)
                except Exception:
                    failures += 1
                    logger.error("FAILED %s %s [%s]:\n%s", family, task.model_id, study, traceback.format_exc())
    logger.info("run_all finished, %d failed (study, model) fits", failures)
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", choices=sorted(STUDIES), help="single study (with --family/--array-index)")
    p.add_argument("--from-registry", action="store_true", required=True)
    p.add_argument("--family")
    p.add_argument("--array-index", type=int)
    p.add_argument("--all-models", action="store_true", help="loop over every model of every family")
    p.add_argument("--studies", nargs="+", choices=sorted(STUDIES), default=sorted(STUDIES))
    p.add_argument("--families", nargs="+", default=None, help="restrict --all-models to these families")
    p.add_argument("--min-params", type=float, default=0.0)
    p.add_argument("--max-params", type=float, default=float("inf"))
    p.add_argument("--out_dir", required=True, help="output root; with --all-models, '<root>/szewczyk_<study>/embedding_sweep'")
    p.add_argument("--feature_cache_dir", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = p.parse_args()
    try:
        if a.all_models:
            roots = {s: Path(a.out_dir) / f"szewczyk_{s}" / "embedding_sweep" for s in a.studies}
            sys.exit(1 if run_all(a.studies, roots, Path(a.feature_cache_dir), a.device,
                                  a.min_params, a.max_params, a.families) else 0)
        if not (a.study and a.family and a.array_index is not None):
            p.error("need --study, --family and --array-index (or --all-models)")
        run(a.study, a.family, a.array_index, Path(a.out_dir), Path(a.feature_cache_dir), a.device)
    except SystemExit:
        raise
    except Exception:
        logger.error("Failed:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
