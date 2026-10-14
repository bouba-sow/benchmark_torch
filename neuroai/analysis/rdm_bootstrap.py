"""Stimulus (item) bootstrap of the held-out RDM delta log-likelihood.

Uncertainty of a model's score, and of score *differences between models*, over
experiments with different stimulus samples (Schutt et al. 2023, "Bootstrapping
conditions"): resample items with replacement, rebuild the RDMs for the resample,
rescore. Items repeated in a resample have dissimilarity 0 with their own copy in the
data and in every model, which would bias every score upward, so those self-pairs are
excluded (as in the paper).

Implementation (ours): a bootstrap sample is a vector of item counts ``c``. Every pair
(i, j), i != j, of the ORIGINAL item set gets weight ``c_i * c_j`` (its multiplicity in
the resampled RDM); self-pairs are simply not in the pair list. Fits are weighted least
squares, the noise variance is the weighted ML estimate, and test log-likelihoods are
weight-summed, which equals fitting and scoring on the explicitly duplicated rows
(checked in tests/test_rdm_bootstrap.py). Cross-validation folds are assigned to the
ORIGINAL items, so copies of one item always share a fold (no train/test leakage).
Everything runs on the same per-fold sufficient statistics as ``RdmCvPlan``.

All models evaluated on one ``RdmBootstrap`` object see identical resamples and fold
partitions, so differences between models are paired.

Variance: ``n_partitions`` fold partitions are drawn per resample (Schutt et al.
recommend 2). The spread of resample scores then contains partition noise; ``se``
subtracts its expected share (mean within-resample partition variance / n_partitions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroai.analysis.rdm_regression import (
    _cell_sums, _test_ll, _zscore, all_pairs, item_fold_assignment, n_folds_for)


def _dll_from_stats(btb, bty, yy, nn, bm, my, mm, k: int, idx: list[int]) -> float:
    """Weighted held-out (LL_full - LL_base) / sum of test weights for one partition.

    ``idx`` selects the base-design columns (intercept only, or intercept + confounds)."""
    sub = np.ix_(idx, idx)
    tot_f = tot_b = tot_n = 0.0
    for f in range(k):
        keep = [g for g in range(k) if g != f]
        ix = np.ix_(keep, keep)

        def tr(a):
            return a[..., ix[0], ix[1]].sum(axis=(-2, -1))

        tr_btb, tr_bty, tr_yy, tr_n = tr(btb)[sub], tr(bty)[idx], tr(yy), tr(nn)
        tr_bm, tr_my, tr_mm = tr(bm)[idx], tr(my), tr(mm)
        te_btb, te_bty, te_yy, te_n = btb[..., f, f][sub], bty[..., f, f][idx], yy[f, f], nn[f, f]
        te_bm, te_my, te_mm = bm[..., f, f][idx], my[f, f], mm[f, f]
        if te_n < 2 or tr_n <= len(idx) + 1:
            continue
        xtx_tr = np.block([[tr_btb, tr_bm[:, None]], [tr_bm[None, :], np.array([[tr_mm]])]])
        xtx_te = np.block([[te_btb, te_bm[:, None]], [te_bm[None, :], np.array([[te_mm]])]])
        tot_f += _test_ll(xtx_tr, np.append(tr_bty, tr_my), tr_yy, tr_n,
                          xtx_te, np.append(te_bty, te_my), te_yy, te_n)
        tot_b += _test_ll(tr_btb, tr_bty, tr_yy, tr_n, te_btb, te_bty, te_yy, te_n)
        tot_n += float(te_n)
    return (tot_f - tot_b) / tot_n if tot_n > 0 else float("nan")


@dataclass
class BootResult:
    delta: np.ndarray            # (n_models, n_variants, n_replicates, n_partitions) delta LL per pair
    variants: tuple[str, ...]
    include_original: bool       # replicate 0 is the original (all-ones) item weights

    def theta(self) -> np.ndarray:
        """Resample scores (mean over fold partitions): (n_models, n_variants, n_replicates)."""
        return np.nanmean(self.delta, axis=-1)

    @property
    def point(self) -> np.ndarray:
        if not self.include_original:
            raise ValueError("built without include_original=True")
        return self.theta()[..., 0]

    def _boot(self, t: np.ndarray) -> np.ndarray:
        return t[..., 1:] if self.include_original else t

    def se(self) -> np.ndarray:
        """Bootstrap standard error with the partition-noise share subtracted: (n_models, n_variants)."""
        d = self.delta[..., 1:, :] if self.include_original else self.delta
        p = d.shape[-1]
        theta = np.nanmean(d, axis=-1)
        var_total = np.nanvar(theta, axis=-1, ddof=1)
        if p < 2:
            return np.sqrt(var_total)
        excess = np.nanmean(np.nanvar(d, axis=-1, ddof=1), axis=-1) / p
        return np.sqrt(np.maximum(var_total - excess, 0.0))

    def se_naive(self) -> np.ndarray:
        """Spread of the resample scores WITHOUT the partition-noise correction (an upper bound on the
        bootstrap SE): (n_models, n_variants). ``se()`` is 0 when the estimated partition noise exceeds
        the total spread (near-null models); the percentile interval is then the quantity to report."""
        t = self._boot(self.theta())
        return np.sqrt(np.nanvar(t, axis=-1, ddof=1))

    def ci(self, alpha: float = 0.05) -> np.ndarray:
        """Percentile interval of the resample scores: (2, n_models, n_variants)."""
        t = self._boot(self.theta())
        return np.nanpercentile(t, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=-1)

    def diff(self, a: int, b: int, alpha: float = 0.05) -> dict:
        """Paired difference model a - model b per variant: mean over resamples, corrected SE,
        percentile CI and the share of resamples where a > b."""
        d = self.delta[a] - self.delta[b]                      # (variants, replicates, partitions)
        res = BootResult(d[None], self.variants, self.include_original)
        t = self._boot(res.theta())[0]
        lo, hi = np.nanpercentile(t, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=-1)
        return {"mean": np.nanmean(t, axis=-1), "se": res.se()[0], "ci_low": lo, "ci_high": hi,
                "p_a_greater": np.nanmean(t > 0, axis=-1)}


class RdmBootstrap:
    """Item-bootstrap of one response RDM (+ optional confound RDMs); score many models."""

    def __init__(self, response, n_items: int, pairs=None, confounds=None, n_boot: int = 200,
                 n_partitions: int = 2, n_folds: int | None = None, seed: int = 0,
                 counts: np.ndarray | None = None, include_original: bool = False):
        y = np.asarray(response, dtype=np.float64)
        if y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError("response must be a finite 1-D RDM vector")
        pairs = all_pairs(n_items) if pairs is None else np.asarray(pairs)
        if pairs.shape != (len(y), 2):
            raise ValueError(f"{len(pairs)} pairs but {len(y)} RDM entries")
        if (pairs[:, 0] == pairs[:, 1]).any():
            raise ValueError("self-pairs must not be in the pair list")
        self.n_items, self.n_pairs, self.P = n_items, len(y), n_partitions
        self.k = n_folds or n_folds_for(n_items)
        self._i, self._j = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
        self._y = _zscore(y, "response")
        cols = [np.ones(len(y))]
        if confounds is not None:
            c = np.asarray(confounds, dtype=np.float64).reshape(len(y), -1)
            cols += [_zscore(c[:, j], f"confound {j}") for j in range(c.shape[1])]
        self._cols = cols
        self.variants = ("none", "len_freq") if confounds is not None else ("none",)
        self._idx = {"none": [0], "len_freq": list(range(len(cols)))}
        if counts is None:
            rng = np.random.default_rng([seed, 0])
            counts = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=n_boot)
        counts = np.asarray(counts)
        if counts.ndim != 2 or counts.shape[1] != n_items or (counts < 0).any():
            raise ValueError("counts must be a non-negative (n_replicates, n_items) array")
        self.include_original = include_original
        if include_original:
            counts = np.vstack([np.ones((1, n_items), dtype=counts.dtype), counts])
        self.counts = counts.astype(np.float64)
        self.n_rep = len(counts)
        self._folds = [[item_fold_assignment(n_items, self.k, np.random.default_rng([seed, 1, b, p]))
                        for p in range(n_partitions)] for b in range(self.n_rep)]
        q = len(cols)
        self._fixed = []
        for b, p, cell, w in self._iter_cells():
            btb = np.empty((q, q, self.k, self.k))
            for a in range(q):
                for c_ in range(a, q):
                    btb[a, c_] = btb[c_, a] = _cell_sums(cell, w * cols[a] * cols[c_], self.k)
            bty = np.stack([_cell_sums(cell, w * cols[a] * self._y, self.k) for a in range(q)])
            self._fixed.append((btb, bty, _cell_sums(cell, w * self._y * self._y, self.k),
                                _cell_sums(cell, w, self.k)))

    def _iter_cells(self):
        for b in range(self.n_rep):
            c = self.counts[b]
            w = c[self._i] * c[self._j]
            for p in range(self.P):
                fold = self._folds[b][p]
                yield b, p, fold[self._i] * self.k + fold[self._j], w

    def delta_ll_batch(self, models) -> BootResult:
        """Score models (array-like of shape (n_models, n_pairs)) on every resample/partition."""
        zm = np.stack([_zscore(m, "model") for m in np.asarray(models, dtype=np.float64).reshape(-1, self.n_pairs)])
        q, n_models = len(self._cols), len(zm)
        out = np.full((n_models, len(self.variants), self.n_rep, self.P), np.nan)
        for (b, p, cell, w), (btb, bty, yy, nn) in zip(self._iter_cells(), self._fixed):
            for mi in range(n_models):
                wm = w * zm[mi]
                bm = np.stack([_cell_sums(cell, wm * self._cols[a], self.k) for a in range(q)])
                my = _cell_sums(cell, wm * self._y, self.k)
                mm = _cell_sums(cell, wm * zm[mi], self.k)
                for vi, variant in enumerate(self.variants):
                    out[mi, vi, b, p] = _dll_from_stats(btb, bty, yy, nn, bm, my, mm, self.k,
                                                        self._idx[variant])
        return BootResult(out, self.variants, self.include_original)
