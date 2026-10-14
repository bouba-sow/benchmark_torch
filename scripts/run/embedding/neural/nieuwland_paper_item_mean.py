#!/usr/bin/env python3
"""Nieuwland N400 embedding encoding on the paper's item-mean N400 (160 items).

Target : the paper's per-trial ``n400`` (data/n400_paper_formula/.../nieuwland_2018.tsv;
         6-channel, 200-500 ms mean), averaged over subjects for each (frame, condition) item.
X      : cached last-noun hidden states of the 160 (context, noun) items, every layer
         (same feature caches as the epoch-based sweep -- no LLM is loaded here).
Model  : Himalaya ``RidgeCV`` on GPU, one alpha per target chosen by inner group-CV.
CV     : outer GroupKFold over frames (both nouns of one context stay together); the inner
         CV is a 10-fold GroupKFold over the same frame groups, so alpha selection never
         sees a train item whose sibling condition is in the validation fold.
Score  : Spearman / Pearson of out-of-fold predictions vs item means, per layer; the noise
         ceiling for this target is the split-half item reliability on the same table
         (scripts/ceilings/correlation.py --dataset nieuwland_paper).

    python scripts/run/embedding/neural/nieuwland_paper_item_mean.py \\
        --from-registry --family gpt2 --array-index 0 --out_dir <root> --feature_cache_dir <dir>
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from neuroai.benchmarks.eeg_erp import _feature_cache_path, _items_hash
from neuroai.datasets.nieuwland import load_all_items
from neuroai.models.registry import registry_model_configs, resolve_family_task
from neuroai.sweep import safe_model_dir, setup_logging

setup_logging()
logger = logging.getLogger("nieuwland_paper_item_mean")

PAPER_TSV = REPO / "data/n400_paper_formula/osfstorage/datasets/nieuwland_2018.tsv"
# Grid 10^-2 ... 10^14. Starts from the previously calibrated values [3162, 42170, 562341, 1e8, 1e9, 1e10]
# (10^3.5 ... 10^10, calibrated for the epoch-based trial-level fit, not this target), plus extra values
# between them; the first item-mean test on GPT-2 chose the old edges (10^2 and 10^12), so the grid
# is extended by 4 decades at the bottom and 2 at the top.
ALPHAS = [10 ** e for e in (-2.0, -1.0, 0.0, 1.0, 2.0, 2.75, 3.5, 4.0, 4.625, 5.2, 5.75, 6.4, 7.2,
                            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)]
N_OUTER, N_INNER = 10, 10


def _key(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text).lower())


def item_means(items: pd.DataFrame) -> pd.DataFrame:
    """Per-item mean paper N400 (uV), aligned to ``items`` row order."""
    paper = pd.read_csv(PAPER_TSV, sep="\t").drop_duplicates()
    paper = paper.dropna(subset=["n400", "FullText", "TargetWords"])
    paper["k"] = paper["FullText"].map(_key).str[:20] + "|" + paper["TargetWords"].map(_key)
    it = items.reset_index(drop=True).copy()
    it["k"] = it["sentence_context"].map(_key).str[:20] + "|" + it["noun"].map(_key)
    if it["k"].duplicated().any():
        raise ValueError("item (context, noun) keys are not unique")
    if set(paper["k"]) - set(it["k"]):
        raise ValueError("paper trials without a matching item")
    g = paper.groupby("k")["n400"].agg(y="mean", n_trials="size")
    it = it.join(g, on="k")
    if it["y"].isna().any():
        raise ValueError("items without any paper trial")
    return it


def _spearman(a, b):
    return float(spearmanr(a, b).statistic)


def _pearson(a, b):
    return float(pearsonr(a, b).statistic)


def fit_layers(features: np.ndarray, y: np.ndarray, frames: np.ndarray, device: str, alphas_grid=None):
    """Out-of-fold predictions for every layer. features: (n_items, n_layers, D)."""
    import warnings

    use_gpu = device == "cuda"
    from himalaya.backend import get_backend, set_backend
    from himalaya.ridge import RidgeCV

    set_backend("torch_cuda" if use_gpu else "numpy")
    b = get_backend()
    n_items, n_layers, _ = features.shape
    outer = list(GroupKFold(n_splits=min(N_OUTER, len(np.unique(frames)))).split(features[:, 0], y, frames))
    oof = np.full((n_layers, n_items), np.nan)
    alphas = np.full((n_layers, len(outer)), np.nan)
    for li in range(n_layers):
        X = features[:, li, :].astype(np.float64)
        for fi, (tr, te) in enumerate(outer):
            sc = RobustScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            # himalaya's RidgeCV fits no intercept: centre X and y on the training fold
            # (the target mean is ~2.9 uV; without centring most models chose alpha=1e14).
            x_mu, y_mu = Xtr.mean(axis=0), float(y[tr].mean())
            Xtr, Xte = Xtr - x_mu, Xte - x_mu
            inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(frames[tr])))).split(Xtr, y[tr], frames[tr]))
            model = RidgeCV(alphas=ALPHAS if alphas_grid is None else alphas_grid, cv=inner)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(b.asarray(Xtr), b.asarray((y[tr] - y_mu)[:, None]))
            oof[li, te] = b.to_numpy(model.predict(b.asarray(Xte)))[:, 0] + y_mu
            alphas[li, fi] = float(np.median(b.to_numpy(model.best_alphas_)))
    return oof, alphas


def run(family: str, index: int, out_root: Path, cache_dir: Path, device: str) -> None:
    task = resolve_family_task(family, index)
    model_cfg = next((m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id), {})
    out_dir = out_root / family / safe_model_dir(task.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    revision = task.revision or "main"
    scores_csv = out_dir / "nieuwland_paper_item_mean_scores.csv"
    if scores_csv.is_file():
        done = pd.read_csv(scores_csv)
        if (done["revision"].astype(str) == revision).any():
            logger.info("%s@%s already done", task.model_id, revision)
            return

    items = load_all_items(REPO / "data/nieuwland", included_in_paper_only=True, exclude_labs=["glas"])
    items = item_means(items)
    ih = _items_hash(items["sentence_context"].tolist(), items["noun"].tolist())
    hf_rev = None if revision in ("", "main") else revision
    cache = _feature_cache_path(cache_dir, task.load_model_id, hf_rev, "all", ih)
    if not cache.is_file():
        raise FileNotFoundError(f"No cached all-layer features: {cache}")
    from neuroai.features.llm import select_embedding_layers

    features = select_embedding_layers(np.load(cache)["features"])
    if features.ndim == 2:
        features = features[:, None, :]
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite features in {cache}")
    y = items["y"].to_numpy(dtype=np.float64)
    frames = items["item_number"].to_numpy()
    logger.info("%s: features %s, %d items, %d frames, device=%s",
                task.model_id, features.shape, len(y), len(np.unique(frames)), device)

    t0 = time.time()
    oof, alphas = fit_layers(features, y, frames, device)
    layer_rows = [dict(family=family, model_id=task.model_id, revision=revision, layer=li,
                       spearmanr=_spearman(oof[li], y), pearsonr=_pearson(oof[li], y),
                       median_alpha=float(np.median(alphas[li])),
                       n_items=len(y), n_layers=features.shape[1])
                  for li in range(features.shape[1])]
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(out_dir / "nieuwland_paper_item_mean_layerwise.csv", index=False)
    best = int(layers["spearmanr"].idxmax())
    np.savez_compressed(out_dir / "nieuwland_paper_item_mean_oof.npz", oof=oof.astype(np.float32),
                        y=y, item_number=frames, condition=items["condition"].to_numpy(dtype=str))
    summary = dict(family=family, model_id=task.model_id, revision=revision,
                   n_params=float(model_cfg.get("n_params", 0.0)), n_layers=features.shape[1],
                   best_layer=best, best_spearmanr=float(layers.loc[best, "spearmanr"]),
                   best_pearsonr=float(layers.loc[best, "pearsonr"]),
                   last_layer_spearmanr=float(layers["spearmanr"].iloc[-1]),
                   last_layer_pearsonr=float(layers["pearsonr"].iloc[-1]),
                   n_items=len(y), fit_seconds=round(time.time() - t0, 1))
    pd.DataFrame([summary]).to_csv(scores_csv, index=False)
    logger.info("%s: best layer %d spearman=%.3f pearson=%.3f (%.0fs)", task.model_id, best,
                summary["best_spearmanr"], summary["best_pearsonr"], summary["fit_seconds"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-registry", action="store_true", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--array-index", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--feature_cache_dir", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = p.parse_args()
    try:
        run(a.family, a.array_index, Path(a.out_dir), Path(a.feature_cache_dir), a.device)
    except Exception:
        logger.error("Failed:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
