"""Dense (all-pairs) version of ``rdm_regression.RdmCvPlan``: the same item-fold cross-validated delta log-likelihood, computed with
matrix products instead of bincounts over the pair vector.

For ALL pairs i < j of n items, an RDM is the upper triangle of an n x n matrix W. The sum of W over the pairs whose items fall in the
folds (a, b) is (F^T W F)[a, b] with F the n x K fold-indicator matrix, exactly the ``_cell_sums`` of ``RdmCvPlan`` (cell = fold_i * K +
fold_j). With all the repeats stacked in one indicator matrix this is one (n x n)(n x R K) matrix product per weight vector (BLAS), about 5-10x
faster than ``np.bincount`` over the n(n-1)/2 pairs for n of several thousand, and needs no pair-index arrays. The statistics, the OLS fits
and the Gaussian test log-likelihoods are the ones of ``RdmCvPlan`` (same item partitions for the same seed); tests/test_rdm_regression_dense.py
checks equality with ``RdmCvPlan``. Only the case "every pair is used" is covered.
"""
from __future__ import annotations

import numpy as np

from neuroai.analysis.rdm_regression import (
    RdmDeltaLL, _test_ll, _zscore, item_fold_assignment, n_folds_for)


def fill_upper(v: np.ndarray, n: int) -> np.ndarray:
    """Pair vector (all_pairs order: i < j, row-major) -> dense n x n matrix with the values above the diagonal, zeros elsewhere."""
    out = np.zeros((n, n), dtype=np.float64)
    o = 0
    for i in range(n - 1):
        m = n - i - 1
        out[i, i + 1:] = v[o:o + m]
        o += m
    return out


class DenseDesign:
    """Item partitions and the model-independent statistics shared by all units that use the same confounds and items."""

    def __init__(self, n_items: int, confounds: list[np.ndarray], n_repeats: int = 10, n_folds: int | None = None, seed: int = 0):
        self.n, self.n_repeats = n_items, n_repeats
        self.k = n_folds or n_folds_for(n_items)
        n, k, r = self.n, self.k, n_repeats
        self.n_pairs = n * (n - 1) // 2
        rng = np.random.default_rng(seed)
        self.folds = [item_fold_assignment(n, k, rng) for _ in range(r)]          # same draws as RdmCvPlan
        self.F = np.zeros((n, r * k))
        for t, fold in enumerate(self.folds):
            self.F[np.arange(n), t * k + fold] = 1.0
        # dense z-scored design columns (ones, confounds)
        self.C = [fill_upper(_zscore(np.asarray(c, dtype=np.float64), f"confound {j}"), n) for j, c in enumerate(confounds)]
        U = fill_upper(np.ones(self.n_pairs), n)
        self.q = 1 + len(self.C)
        cols = [U] + self.C                                                         # B_0 = ones (restricted to i < j)
        self.btb = np.empty((r, self.q, self.q, k, k))
        for i in range(self.q):
            for j in range(i, self.q):
                s = self._cells(cols[i] * cols[j] if (i or j) else U)
                self.btb[:, i, j] = self.btb[:, j, i] = s
        self.nn = self._cells(U)
        self._U = None
        self._model_stats: dict = {}   # model key -> (bm, mm): the fold sums that do not depend on the response (shared by all units)

    def _cells(self, w: np.ndarray) -> np.ndarray:
        """(R, K, K) sums of the upper-triangular weight matrix over the fold cells of every repeat."""
        r, k = self.n_repeats, self.k
        t = w @ self.F                                                              # n x (R K)
        return np.stack([self.F[:, a * k:(a + 1) * k].T @ t[:, a * k:(a + 1) * k] for a in range(r)])

    def plan(self, response: np.ndarray) -> "DensePlan":
        return DensePlan(self, response)


class DensePlan:
    """One response RDM (one unit): call ``delta_ll(model_vector)`` for every model."""

    def __init__(self, design: DenseDesign, response: np.ndarray):
        d = design
        self.d, self.n_pairs = d, d.n_pairs
        self._y = _zscore(np.asarray(response, dtype=np.float64), "response")
        Y = fill_upper(self._y, d.n)
        U = fill_upper(np.ones(d.n_pairs), d.n)
        cols = [U] + d.C
        self.bty = np.stack([d._cells(c * Y if i else Y) for i, c in enumerate(cols)], axis=1)   # (R, q, K, K)
        self.yy = d._cells(Y * Y)
        self.Y = Y
        del U

    def pearson(self, model) -> float:
        return float(np.mean(self._y * _zscore(np.asarray(model, dtype=np.float64), "model")))

    def delta_ll(self, model, model_key=None) -> RdmDeltaLL:
        """``model_key`` (hashable, optional): the sums bm = sum B_i m and mm = sum m^2 do not depend on the response, so they are cached on
        the design under this key and reused by every other unit scored with the same model (only my = sum y m is recomputed)."""
        d = self.d
        n, k, q = d.n, d.k, d.q
        m = _zscore(np.asarray(model, dtype=np.float64), "model")
        M = fill_upper(m, n)
        cached = d._model_stats.get(model_key) if model_key is not None else None
        if cached is None:
            bm = np.stack([d._cells(M)] + [d._cells(c * M) for c in d.C], axis=1)    # (R, q, K, K)  sum B_i * m
            mm = d._cells(M * M)
            if model_key is not None:
                d._model_stats[model_key] = (bm, mm)
        else:
            bm, mm = cached
        my = d._cells(self.Y * M)                                                    # (R, K, K)
        d_obs, ll_f, ll_b, n_te = [], [], [], []
        for t in range(d.n_repeats):
            btb, bty, yy, nn = d.btb[t], self.bty[t], self.yy[t], d.nn[t]
            tot_f = tot_b = 0.0
            tot_n = 0
            for f in range(k):
                keep = np.array([g for g in range(k) if g != f])
                ix = np.ix_(keep, keep)
                sel = lambda a: a[..., ix[0], ix[1]].sum(axis=(-2, -1))  # noqa: E731
                tr_btb, tr_bty, tr_yy, tr_n = sel(btb), sel(bty), sel(yy), sel(nn)
                tr_bm, tr_my, tr_mm = sel(bm[t]), sel(my[t]), sel(mm[t])
                te_btb, te_bty, te_yy, te_n = btb[..., f, f], bty[..., f, f], yy[f, f], nn[f, f]
                te_bm, te_my, te_mm = bm[t][..., f, f], my[t][f, f], mm[t][f, f]
                if te_n < 2 or tr_n <= q + 1:
                    continue
                xtx_tr = np.block([[tr_btb, tr_bm[:, None]], [tr_bm[None, :], np.array([[tr_mm]])]])
                xtx_te = np.block([[te_btb, te_bm[:, None]], [te_bm[None, :], np.array([[te_mm]])]])
                tot_f += _test_ll(xtx_tr, np.append(tr_bty, tr_my), tr_yy, tr_n,
                                  xtx_te, np.append(te_bty, te_my), te_yy, te_n)
                tot_b += _test_ll(tr_btb, tr_bty, tr_yy, tr_n, te_btb, te_bty, te_yy, te_n)
                tot_n += int(te_n)
            if tot_n == 0:
                raise ValueError("No test pairs after item-fold split")
            d_obs.append((tot_f - tot_b) / tot_n)
            ll_f.append(tot_f / tot_n)
            ll_b.append(tot_b / tot_n)
            n_te.append(tot_n)
        return RdmDeltaLL(
            delta_ll_per_obs=float(np.mean(d_obs)),
            delta_ll_per_obs_sd=float(np.std(d_obs, ddof=1)) if d.n_repeats > 1 else float("nan"),
            ll_full_per_obs=float(np.mean(ll_f)), ll_base_per_obs=float(np.mean(ll_b)),
            n_test_pairs=float(np.mean(n_te)), frac_pairs_tested=float(np.mean(n_te) / d.n_pairs),
            n_folds=k, n_repeats=d.n_repeats)
