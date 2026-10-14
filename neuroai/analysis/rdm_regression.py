"""Held-out delta log-likelihood for RDM regression (RSA-style "variance partitioning").

Predicts a response RDM (neural or behavioral) from a model RDM (embedding
distance or |delta surprisal|), with or without confound RDMs, and scores the
gain in held-out Gaussian log-likelihood::

    base:  y ~ 1 [+ confound RDMs]
    full:  y ~ 1 [+ confound RDMs] + model RDM
    delta_ll_per_obs = (LL_full - LL_base) / n_test_pairs     (held-out)

RDM entries are pairs of items and are NOT independent (each item appears in
n-1 pairs). Splitting pairs at random would leak, so folds are made over
ITEMS: a pair is used for training only if both its items are training items
and for testing only if both are in the held-out fold; cross-fold pairs are
discarded. See docs/rdm_regression.md for the literature this follows
(Schutt et al. 2023; Kaniuth & Hebart) and for what is our own choice.

Pair layout: ``pairs`` is an (n_pairs, 2) int array of item indices. ``None``
means the full upper triangle in ``scipy.spatial.distance.pdist`` order, which
matches the ``pair_indices`` convention of the cached RDMs
(``scripts/analysis/surprisal_rsa.py``: sampled pairs when n(n-1)/2 > 250k).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


def standardize_features(x: np.ndarray) -> np.ndarray:
    """z-score every feature across items (rows). Raw LLM hidden states carry a few massive-activation
    dimensions that dominate a correlation/cosine distance (the RDM then collapses to ~0 with a few
    extreme items); standardising each feature across the item set removes that. Constant features
    become 0. Setting RDM_EMBED_STANDARDIZE=0 disables it (raw states; used for the no-standardization
    comparison runs, which write to separate output roots)."""
    x = np.asarray(x, dtype=np.float32)
    if os.environ.get("RDM_EMBED_STANDARDIZE") == "0":
        return x
    mu = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = x.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd <= 1e-12] = 1.0
    return (x - mu) / sd


def n_folds_for(n_items: int) -> int:
    """Number of item folds, following the schedule of Schutt et al. (2023,
    Methods): 2 folds for few conditions, 3 for >=12, 4 for >=24, 5 for >=40
    (their values are "chosen ad hoc"). Fewer than 6 items cannot be split."""
    if n_items < 6:
        raise ValueError(f"Need at least 6 items to cross-validate an RDM, got {n_items}")
    if n_items >= 40:
        return 5
    if n_items >= 24:
        return 4
    if n_items >= 12:
        return 3
    return 2


def all_pairs(n_items: int) -> np.ndarray:
    """Upper-triangle pairs in ``pdist`` order."""
    return np.column_stack(np.triu_indices(n_items, k=1)).astype(np.int32)


def item_fold_assignment(n_items: int, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Random balanced assignment of items to folds (fold id per item)."""
    return rng.permutation(np.arange(n_items) % n_folds)


def _zscore(v: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"{name} RDM is constant or non-finite")
    return (v - v.mean()) / sd


def _cell_sums(cell: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(cell, weights=w, minlength=k * k).reshape(k, k)


def _solve(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(xtx, xty, rcond=None)[0]


def _test_ll(xtx_tr, xty_tr, yy_tr, n_tr, xtx_te, xty_te, yy_te, n_te) -> float:
    """Fit OLS from training sufficient statistics (ML noise variance) and return
    the Gaussian log-likelihood of the test pairs, also from sufficient statistics."""
    beta = _solve(xtx_tr, xty_tr)
    sigma2 = max((yy_tr - 2 * beta @ xty_tr + beta @ xtx_tr @ beta) / n_tr, 1e-12)
    rss = yy_te - 2 * beta @ xty_te + beta @ xtx_te @ beta
    return float(-0.5 * (n_te * (_LOG_2PI + np.log(sigma2)) + rss / sigma2))


@dataclass
class RdmDeltaLL:
    delta_ll_per_obs: float       # mean over repeats of (LL_full - LL_base) / n_test_pairs
    delta_ll_per_obs_sd: float    # sd over repeats (split noise only, not sampling uncertainty)
    ll_full_per_obs: float
    ll_base_per_obs: float
    n_test_pairs: float           # mean over repeats
    frac_pairs_tested: float      # mean n_test_pairs / n_pairs (share left after discarding cross-fold pairs)
    n_folds: int
    n_repeats: int


class RdmCvPlan:
    """Item-fold partitions plus the model-independent sufficient statistics for one
    response RDM (and optional confounds). Build once per (dataset, measure,
    confound set), then call ``delta_ll(model_rdm)`` for every model/layer.

    OLS train fits and test log-likelihoods only need sums of products of
    (1, confounds, model, response) over the training pairs (both items outside the
    fold) and over the test pairs (both items inside the fold). These are accumulated
    per (fold of item a, fold of item b) cell with one ``bincount`` per product, so
    cost per model is O(n_pairs) per repeat and independent of the number of folds.
    The result equals refitting with ``lstsq`` on the selected rows fold by fold
    (checked in tests/test_rdm_regression.py).

    All RDM vectors are z-scored once; this is a linear reparametrisation of the same
    OLS models and shifts base and full log-likelihood by the same constant, so
    delta LL per observation is unchanged up to the scale of the response.
    """

    def __init__(self, response, n_items, pairs=None, confounds=None,
                 n_repeats=10, n_folds=None, seed=0):
        y = np.asarray(response, dtype=np.float64)
        if y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError("response must be a finite 1-D RDM vector")
        if pairs is None:
            pairs = all_pairs(n_items)
        pairs = np.asarray(pairs)
        if pairs.shape != (len(y), 2):
            raise ValueError(f"{len(pairs)} pairs but {len(y)} RDM entries")
        self.n_items, self.n_pairs, self.n_repeats = n_items, len(y), n_repeats
        self.k = n_folds or n_folds_for(n_items)
        self._y = _zscore(y, "response")
        cols = [np.ones(len(y))]
        if confounds is not None:
            c = np.asarray(confounds, dtype=np.float64).reshape(len(y), -1)
            if not np.isfinite(c).all():
                raise ValueError("Non-finite confound RDM entries")
            cols += [_zscore(c[:, j], f"confound {j}") for j in range(c.shape[1])]
        self._B = cols
        rng = np.random.default_rng(seed)
        self._cells, self._fixed = [], []
        for _ in range(n_repeats):
            fold = item_fold_assignment(n_items, self.k, rng)
            cell = (fold[pairs[:, 0]] * self.k + fold[pairs[:, 1]]).astype(np.int8)
            self._cells.append(cell)
            q = len(cols)
            btb = np.empty((q, q, self.k, self.k))
            for i in range(q):
                for j in range(i, q):
                    btb[i, j] = btb[j, i] = _cell_sums(cell, cols[i] * cols[j], self.k)
            bty = np.stack([_cell_sums(cell, cols[i] * self._y, self.k) for i in range(q)])
            yy = _cell_sums(cell, self._y * self._y, self.k)
            n = _cell_sums(cell, np.ones(len(y)), self.k)
            self._fixed.append((btb, bty, yy, n))

    def pearson(self, model) -> float:
        """Plain (fit-free) RSA correlation between the model and response RDMs."""
        return float(np.mean(self._y * _zscore(model, "model")))

    def delta_ll(self, model) -> RdmDeltaLL:
        m = _zscore(model, "model")
        if m.shape != self._y.shape:
            raise ValueError(f"model RDM has {m.shape[0]} entries, expected {self._y.shape[0]}")
        q, k = len(self._B), self.k
        d_obs, ll_f, ll_b, n_te = [], [], [], []
        for cell, (btb, bty, yy, n) in zip(self._cells, self._fixed):
            bm = np.stack([_cell_sums(cell, self._B[i] * m, k) for i in range(q)])
            my = _cell_sums(cell, m * self._y, k)
            mm = _cell_sums(cell, m * m, k)
            tot_f = tot_b = 0.0
            tot_n = 0
            for f in range(k):
                keep = np.array([g for g in range(k) if g != f])
                ix = np.ix_(keep, keep)
                sel = lambda a: a[..., ix[0], ix[1]].sum(axis=(-2, -1))  # noqa: E731
                tr_btb, tr_bty, tr_yy, tr_n = sel(btb), sel(bty), sel(yy), sel(n)
                tr_bm, tr_my, tr_mm = sel(bm), sel(my), sel(mm)
                te_btb, te_bty, te_yy, te_n = btb[..., f, f], bty[..., f, f], yy[f, f], n[f, f]
                te_bm, te_my, te_mm = bm[..., f, f], my[f, f], mm[f, f]
                if te_n < 2 or tr_n <= q + 1:
                    continue
                xtx_tr = np.block([[tr_btb, tr_bm[:, None]], [tr_bm[None, :], np.array([[tr_mm]])]])
                xtx_te = np.block([[te_btb, te_bm[:, None]], [te_bm[None, :], np.array([[te_mm]])]])
                tot_f += _test_ll(xtx_tr, np.append(tr_bty, tr_my), tr_yy, tr_n,
                                  xtx_te, np.append(te_bty, te_my), te_yy, te_n)
                tot_b += _test_ll(tr_btb, tr_bty, tr_yy, tr_n, te_btb, te_bty, te_yy, te_n)
                tot_n += int(te_n)
            if tot_n == 0:
                raise ValueError("No test pairs after item-fold split; too few sampled pairs per fold")
            d_obs.append((tot_f - tot_b) / tot_n)
            ll_f.append(tot_f / tot_n)
            ll_b.append(tot_b / tot_n)
            n_te.append(tot_n)
        return RdmDeltaLL(
            delta_ll_per_obs=float(np.mean(d_obs)),
            delta_ll_per_obs_sd=float(np.std(d_obs, ddof=1)) if self.n_repeats > 1 else float("nan"),
            ll_full_per_obs=float(np.mean(ll_f)),
            ll_base_per_obs=float(np.mean(ll_b)),
            n_test_pairs=float(np.mean(n_te)),
            frac_pairs_tested=float(np.mean(n_te) / self.n_pairs),
            n_folds=k,
            n_repeats=self.n_repeats,
        )


def cv_delta_ll(response, model, n_items, pairs=None, confounds=None,
                n_repeats=10, n_folds=None, seed=0) -> RdmDeltaLL:
    """Item-fold cross-validated delta log-likelihood of ``model`` over the base model.

    response, model: (n_pairs,) RDM vectors; confounds: (n_pairs,) or (n_pairs, k).
    Each repeat draws a new random item partition (Kaniuth & Hebart repeat the
    outer 5-fold 10 times so that many pairs get to be test pairs); LL is summed
    over folds within a repeat, then per-observation values are averaged over repeats.
    Base and full models are scored on identical test pairs. For many models on the
    same response, build an ``RdmCvPlan`` once instead.
    """
    m = np.asarray(model)
    y = np.asarray(response)
    if y.shape != m.shape or y.ndim != 1:
        raise ValueError(f"response/model must be equal-length vectors, got {y.shape} vs {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("Non-finite model RDM entries")
    if confounds is not None and len(confounds) != len(y):
        raise ValueError("confounds must have one row per RDM entry")
    return RdmCvPlan(y, n_items, pairs, confounds, n_repeats, n_folds, seed).delta_ll(m)
