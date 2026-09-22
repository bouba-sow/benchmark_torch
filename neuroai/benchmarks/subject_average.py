"""Shared helper: fit ridge-per-layer on RT averaged across subjects.

This is the "grand-average" branch of the per-subject embed-ridge behavioral
benchmarks (``geco_embed``, ``meco_embed``, ``provo_embed``, ``ucl_embed``,
``zuco_reading_embed``, ``natural_stories_embed_gpu``). Those benchmarks fit
one ridge CV per subject (subject plays the role of "electrode" in
``NeuralBenchmark``) and report the mean score across subjects. This module
instead **averages the RT target across subjects first** (per item, i.e. per
word/zone), then fits a single ridge CV per layer on that averaged target —
the item-level analogue of an ERP grand average: subject-specific noise
(idiosyncratic reading strategy, RT scale/variance) cancels out, so this
branch is expected to score higher than any individual subject and is cheap
(one ridge fit per layer instead of N).

Available-case mean: an item's averaged RT is the mean over whichever
subjects have a valid (post per-subject-filtering) trial for that item — NOT
a strict intersection across every subject. Several of these datasets'
module docstrings note that per-subject RT filtering leaves very different
valid-trial sets per subject (e.g. GECO: 14 subjects' strict intersection
collapses ~19.7k words/subject down to ~288 shared words) — requiring every
subject would make the averaged branch nearly empty for those datasets.
``min_subjects_per_item`` is a knob to trade off item count vs.
per-item averaging quality (1 = keep every item any subject has, at the
cost of some items being a "average" of just one subject).
"""

from __future__ import annotations

import logging
import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..regressors.ridge import cv_ridge_score, cv_ridge_score_multioutput_gpu

logger = logging.getLogger(__name__)


def average_items_across_subjects(
    trial_df: pd.DataFrame,
    *,
    item_key_cols: tp.Sequence[str],
    rt_col: str = "rt",
    min_subjects_per_item: int = 1,
) -> pd.DataFrame:
    """Available-case mean RT per item, across whichever subjects have it.

    Returns one row per unique ``item_key_cols`` combination with an ``rt``
    column (mean across contributing subjects) and ``n_subjects_avg`` (how
    many subjects contributed). Items with fewer than
    ``min_subjects_per_item`` contributing subjects are dropped.
    """
    key_cols = list(item_key_cols)
    agg = (
        trial_df.groupby(key_cols)[rt_col]
        .agg(rt="mean", n_subjects_avg="count")
        .reset_index()
    )
    if min_subjects_per_item > 1:
        agg = agg.loc[agg["n_subjects_avg"] >= min_subjects_per_item].reset_index(drop=True)
    return agg


def fit_averaged_layers(
    item_df: pd.DataFrame,
    emb_lookup: dict[tuple, np.ndarray],
    *,
    item_key_cols: tp.Sequence[str],
    group_col: str | None,
    n_splits: int,
    alpha,
    scale: bool,
    random_state: int,
    print_chosen_alpha: bool,
    thread_n_jobs: int,
    layer_offset: int = 0,
    total_n_layers: int | None = None,
    extra_fields: dict | None = None,
    return_oof_for_layer: int | None = None,
    fast_gcv: bool = False,
) -> tuple[list[dict], pd.DataFrame | None]:
    """Fit one ridge CV per layer on RT averaged across subjects.

    ``emb_lookup`` may hold only a layer **chunk** (mirrors the per-subject
    chunking in e.g. ``geco_embed.py``) — ``layer_offset``/``total_n_layers``
    translate the chunk-local layer index to the true global layer number,
    same convention as ``_fit_subject_layers`` in the per-subject modules.

    ``return_oof_for_layer`` (chunk-local index), when given, additionally
    refits that one layer with out-of-fold predictions and returns a
    per-item DataFrame of observed vs. predicted averaged RT alongside the
    per-layer score rows (mirrors ``save_word_predictions`` in the
    per-subject path, at the item level instead of the subject level).

    Builds one layer's ``(n_items, hidden)`` matrix at a time from
    ``emb_lookup`` rather than stacking the full ``(n_items,
    n_layers_chunk, hidden)`` array up front — for a large corpus (tens of
    thousands of items) and a wide layer chunk this full stack can rival
    the size of ``emb_lookup`` itself, doubling this call's peak memory on
    top of whatever the per-subject fits in the same chunk are already
    using (see e.g. geco_embed.py's job script comments on host-RAM sizing).
    """
    key_cols = list(item_key_cols)
    rt = item_df["rt"].to_numpy(dtype=float)
    item_embs = [emb_lookup[key] for key in item_df[key_cols].itertuples(index=False, name=None)]
    groups = item_df[group_col].to_numpy() if group_col else None

    n_layers_chunk = item_embs[0].shape[0]
    n_layers = total_n_layers if total_n_layers is not None else n_layers_chunk

    def _layer_matrix(li: int) -> np.ndarray:
        return np.stack([e[li] for e in item_embs], axis=0)

    def _fit_layer(li: int) -> dict:
        s = cv_ridge_score(
            _layer_matrix(li), rt,
            n_splits=n_splits, alpha=alpha, scale=scale,
            random_state=random_state, groups=groups,
            print_chosen_alpha=print_chosen_alpha, fast_gcv=fast_gcv,
        )
        global_li = li + layer_offset
        row = dict(
            layer=global_li, layer_rel=global_li / max(1, n_layers - 1),
            n_layers=n_layers, n_items=len(rt), **s.to_dict(),
        )
        if extra_fields:
            row.update(extra_fields)
        return row

    if thread_n_jobs > 1:
        rows = Parallel(n_jobs=thread_n_jobs, prefer="threads")(
            delayed(_fit_layer)(li) for li in range(n_layers_chunk)
        )
    else:
        rows = [_fit_layer(li) for li in range(n_layers_chunk)]
    rows.sort(key=lambda r: r["layer"])

    per_item: pd.DataFrame | None = None
    if return_oof_for_layer is not None:
        _, oof, valid_mask = cv_ridge_score(
            _layer_matrix(return_oof_for_layer), rt,
            n_splits=n_splits, alpha=alpha, scale=scale,
            random_state=random_state, groups=groups,
            print_chosen_alpha=False, return_oof_predictions=True, fast_gcv=fast_gcv,
        )
        per_item = item_df.loc[valid_mask, key_cols].reset_index(drop=True).copy()
        per_item["layer"] = return_oof_for_layer + layer_offset
        per_item["n_layers"] = n_layers
        per_item["rt_true_avg"] = rt[valid_mask]
        per_item["n_subjects_avg"] = item_df.loc[valid_mask, "n_subjects_avg"].to_numpy()
        per_item["rt_pred_oof"] = oof
        if extra_fields:
            for k, v in extra_fields.items():
                per_item[k] = v

    return rows, per_item


def himalaya_gpu_warmup() -> None:
    """Register the himalaya torch_cuda backend and force the one-time,
    non-thread-safe CUDA/MAGMA SVD backend registration up front, single-
    threaded — must happen before ``fit_averaged_layers_gpu``'s per-layer
    thread pool starts calling ``torch.linalg.svd`` concurrently, or PyTorch
    raises "lazy wrapper should be called at most once" (same fix as
    e.g. ``meco_embed.py``'s ``_himalaya_gpu_warmup``, minus the multi-device
    warmup — the averaged-RT fit is always a single fit on one device, never
    fanned out across GPUs the way per-subject fitting is)."""
    from himalaya.backend import set_backend as _set_backend
    _set_backend("torch_cuda")
    import torch
    _svd_warmup = torch.eye(2, device="cuda")
    torch.linalg.svd(_svd_warmup, full_matrices=False)
    del _svd_warmup


def fit_averaged_layers_gpu(
    item_df: pd.DataFrame,
    emb_lookup: dict[tuple, np.ndarray],
    *,
    item_key_cols: tp.Sequence[str],
    group_col: str | None,
    n_splits: int,
    alpha,
    scale: bool,
    random_state: int,
    layer_offset: int = 0,
    total_n_layers: int | None = None,
    extra_fields: dict | None = None,
    return_oof_for_layer: int | None = None,
    n_layer_workers: int = 8,
) -> tuple[list[dict], pd.DataFrame | None]:
    """GPU counterpart to :func:`fit_averaged_layers` — one CUDA stream per
    layer (``cv_ridge_score_multioutput_gpu``, SVD-based GCV over the alpha
    grid), same pattern as e.g. ``meco_embed.py``'s ``_fit_subject_layers_gpu``.

    Only ever fits **one** target (the averaged RT), so unlike the per-
    subject GPU paths this never needs cross-GPU/cross-subject fan-out —
    call :func:`himalaya_gpu_warmup` once before the first call in a process.
    Meant for datasets whose per-subject fit runs on CPU (e.g. GECO, where a
    nested-CV sklearn ``RidgeCV`` grid search over a ~50k-word, thousands-of-
    dims matrix is what makes a single averaged-RT layer fit slow — this
    swaps that for the much faster SVD-based GPU GCV, which is safe to use
    here specifically because there is no per-subject GPU contention to
    worry about (see geco_embed.py module docstring for why the per-subject
    path itself stays on CPU)."""
    import torch

    key_cols = list(item_key_cols)
    rt = item_df["rt"].to_numpy(dtype=float)
    item_embs = [emb_lookup[key] for key in item_df[key_cols].itertuples(index=False, name=None)]
    groups = item_df[group_col].to_numpy() if group_col else None
    Y = rt.reshape(-1, 1).astype(np.float32)

    n_layers_chunk = item_embs[0].shape[0]
    n_layers = total_n_layers if total_n_layers is not None else n_layers_chunk

    def _layer_matrix(li: int) -> np.ndarray:
        return np.stack([e[li] for e in item_embs], axis=0).astype(np.float32)

    def _fit_layer(li: int) -> tuple[dict, np.ndarray | None]:
        X_li = _layer_matrix(li)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            if li == return_oof_for_layer:
                scores, oof = cv_ridge_score_multioutput_gpu(
                    X_li, Y, groups=groups, n_splits=n_splits, alpha=alpha,
                    scale=scale, standardize_target=True, random_state=random_state,
                    return_oof_predictions=True,
                )
                oof_li = oof[:, 0]
            else:
                scores = cv_ridge_score_multioutput_gpu(
                    X_li, Y, groups=groups, n_splits=n_splits, alpha=alpha,
                    scale=scale, standardize_target=True, random_state=random_state,
                )
                oof_li = None
        stream.synchronize()
        global_li = li + layer_offset
        row = dict(
            layer=global_li, layer_rel=global_li / max(1, n_layers - 1),
            n_layers=n_layers, n_items=len(rt), **scores[0].to_dict(),
        )
        if extra_fields:
            row.update(extra_fields)
        return row, oof_li

    rows: list[dict | None] = [None] * n_layers_chunk
    oof_by_layer: list[np.ndarray | None] = [None] * n_layers_chunk
    n_workers = min(n_layer_workers, n_layers_chunk)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fit_layer, li): li for li in range(n_layers_chunk)}
        for fut in as_completed(futures):
            li = futures[fut]
            row, oof_li = fut.result()
            rows[li] = row
            oof_by_layer[li] = oof_li
    torch.cuda.synchronize()

    per_item: pd.DataFrame | None = None
    if return_oof_for_layer is not None:
        per_item = item_df[key_cols].reset_index(drop=True).copy()
        per_item["layer"] = return_oof_for_layer + layer_offset
        per_item["n_layers"] = n_layers
        per_item["rt_true_avg"] = rt
        per_item["n_subjects_avg"] = item_df["n_subjects_avg"].to_numpy()
        per_item["rt_pred_oof"] = oof_by_layer[return_oof_for_layer]
        if extra_fields:
            for k, v in extra_fields.items():
                per_item[k] = v

    return rows, per_item
