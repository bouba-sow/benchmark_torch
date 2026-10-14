"""Noise ceilings for the RDM delta-LL track, scored like-for-like with the models.

Timkey et al. (2026, PNAS supplement, "variance ceiling") bound delta LL by randomly splitting the
participants into halves A and B, using the per-word mean of one half as the predictor of the other
half's data in the same model that uses surprisal, summing both directions and averaging over 15
random splits. This module is the RDM-space version: the predictor is one half's item-mean RDM, the
target the other half's item-mean RDM, scored with the same item-fold cross-validated Gaussian pair
likelihood as the models (docs/rdm_regression.md).

Like-for-like: a half-split target RDM is noisier than the full-sample grand-average RDM the models are
usually scored on, so a ceiling built on halves must be compared with model scores on the SAME half
targets. ``RdmTargetReplicates`` therefore scores (a) fixed model RDMs and (b) the partner half's RDM
on every target, with identical item folds; ``combine_halves`` then sums the two directions of a split
(total delta LL over total test pairs, as in Timkey et al.).

A target may be defined on a subset of the items (items without an observation in a half): pairs get
weight v_i * v_j with v the 0/1 item validity, so all statistics are weighted per-fold sums exactly as
in rdm_bootstrap.py (the same machinery; checked against a brute-force reference in
tests/test_rdm_ceiling.py).
"""
from __future__ import annotations

import numpy as np

from neuroai.analysis.rdm_regression import (
    _cell_sums, _test_ll, _zscore, all_pairs, item_fold_assignment, n_folds_for)


def _dll_and_n(btb, bty, yy, nn, bm, my, mm, k: int, idx: list[int]) -> tuple[float, float]:
    """(LL_full - LL_base) per test pair and the summed test weight, for one partition."""
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
    return ((tot_f - tot_b) / tot_n if tot_n > 0 else float("nan")), tot_n


def _masked_z(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """z-score over the pairs with weight > 0 (values elsewhere are irrelevant and set to 0)."""
    m = w > 0
    mu, sd = v[m].mean(), v[m].std()
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("constant RDM on the valid pairs")
    out = np.zeros_like(v, dtype=np.float64)
    out[m] = (v[m] - mu) / sd
    return out


class RdmTargetReplicates:
    """Many target RDMs (each on its own valid items) scored with shared pair list and confounds."""

    def __init__(self, responses, n_items: int, pairs=None, confounds=None, item_valid=None,
                 n_partitions: int = 5, n_folds: int | None = None, seed: int = 0, pair_valid=None):
        responses = np.asarray(responses, dtype=np.float64)
        pairs = all_pairs(n_items) if pairs is None else np.asarray(pairs)
        if responses.ndim != 2 or responses.shape[1] != len(pairs):
            raise ValueError("responses must be (n_targets, n_pairs)")
        self.T, self.n_items, self.n_pairs, self.P = len(responses), n_items, len(pairs), n_partitions
        self.k = n_folds or n_folds_for(n_items)
        valid = np.ones((self.T, n_items)) if item_valid is None else np.asarray(item_valid, dtype=np.float64)
        if valid.shape != (self.T, n_items):
            raise ValueError("item_valid must be (n_targets, n_items)")
        self._i, self._j = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
        # weights are exactly 0/1 (or small counts), so float32 loses nothing and halves the memory
        self._w = (valid[:, self._i] * valid[:, self._j]).astype(np.float32)   # (T, n_pairs)
        if pair_valid is not None:       # extra pair-level mask, e.g. pairs no other subject observed
            pair_valid = np.asarray(pair_valid, dtype=np.float32)
            if pair_valid.shape != self._w.shape:
                raise ValueError("pair_valid must be (n_targets, n_pairs)")
            self._w = self._w * pair_valid
        if not np.isfinite(responses[self._w > 0]).all():
            raise ValueError("non-finite response on a valid pair")
        self._y = np.stack([_masked_z(np.nan_to_num(r), w) for r, w in zip(responses, self._w)])
        cols = [np.ones(self.n_pairs)]
        if confounds is not None:
            c = np.asarray(confounds, dtype=np.float64).reshape(self.n_pairs, -1)
            cols += [_zscore(c[:, j], f"confound {j}") for j in range(c.shape[1])]
        self._cols = cols
        self.variants = ("none", "len_freq") if confounds is not None else ("none",)
        self._idx = {"none": [0], "len_freq": list(range(len(cols)))}
        self._folds = [[item_fold_assignment(n_items, self.k, np.random.default_rng([seed, 2, t, p]))
                        for p in range(n_partitions)] for t in range(self.T)]
        q = len(cols)
        self._fixed = {}
        for t, p, cell in self._iter_cells():
            w, y = self._w[t], self._y[t]
            btb = np.empty((q, q, self.k, self.k))
            for a in range(q):
                for b in range(a, q):
                    btb[a, b] = btb[b, a] = _cell_sums(cell, w * cols[a] * cols[b], self.k)
            bty = np.stack([_cell_sums(cell, w * cols[a] * y, self.k) for a in range(q)])
            self._fixed[t, p] = (btb, bty, _cell_sums(cell, w * y * y, self.k), _cell_sums(cell, w, self.k))

    def _iter_cells(self):
        for t in range(self.T):
            for p in range(self.P):
                fold = self._folds[t][p]
                yield t, p, fold[self._i] * self.k + fold[self._j]

    def _evaluate(self, predictors_for_target, n_models: int):
        """predictors_for_target(t) -> (n_models, n_pairs) z-scored predictor vectors for target t.
        Returns (delta (n_models, n_variants, T, P), n_test (T, P))."""
        q = len(self._cols)
        delta = np.full((n_models, len(self.variants), self.T, self.P), np.nan)
        n_test = np.zeros((self.T, self.P))
        for t in range(self.T):
            preds = predictors_for_target(t)
            w, y = self._w[t], self._y[t]
            for p in range(self.P):
                fold = self._folds[t][p]
                cell = fold[self._i] * self.k + fold[self._j]
                btb, bty, yy, nn = self._fixed[t, p]
                for mi in range(n_models):
                    m = preds[mi]
                    wm = w * m
                    bm = np.stack([_cell_sums(cell, wm * self._cols[a], self.k) for a in range(q)])
                    my = _cell_sums(cell, wm * y, self.k)
                    mm = _cell_sums(cell, wm * m, self.k)
                    for vi, variant in enumerate(self.variants):
                        d, n = _dll_and_n(btb, bty, yy, nn, bm, my, mm, self.k, self._idx[variant])
                        delta[mi, vi, t, p] = d
                        n_test[t, p] = n
        return delta, n_test

    def score_models(self, models):
        """Fixed model RDMs (n_models, n_pairs) scored on every target: (delta, n_test) with
        delta (n_models, n_variants, T, P) per test pair and n_test (T, P) test weight."""
        models = np.asarray(models, dtype=np.float64).reshape(-1, self.n_pairs)
        return self._evaluate(lambda t: np.stack([_masked_z(m, self._w[t]) for m in models]), len(models))

    def score_partners(self, partner):
        """Ceiling: the predictor of target t is the response of target partner[t].
        Returns (delta (1, n_variants, T, P), n_test (T, P))."""
        partner = np.asarray(partner)
        return self._evaluate(lambda t: _masked_z(self._y[partner[t]], self._w[t])[None], 1)


def combine_halves(delta: np.ndarray, n_test: np.ndarray, half_pairs: list[tuple[int, int]]) -> np.ndarray:
    """Sum the two directions of each split (Timkey et al.): total delta LL / total test pairs.

    delta (..., T, P) per test pair, n_test (T, P); ``half_pairs`` lists (t_A, t_B) target indices per
    split. Returns (..., n_splits, P)."""
    out = []
    for ta, tb in half_pairs:
        num = delta[..., ta, :] * n_test[ta] + delta[..., tb, :] * n_test[tb]
        out.append(num / (n_test[ta] + n_test[tb]))
    return np.stack(out, axis=-2)


def split_half_targets(values, subject_of_row, item_of_row, n_items: int, n_splits: int = 15, seed: int = 0):
    """Random participant splits -> per-half item means (Timkey et al.: two equally sized halves).

    values/subject_of_row/item_of_row: one entry per trial (subject index 0..S-1, item index 0..n-1).
    Returns item_means (2*n_splits, n_items) with NaN where a half has no trial for the item, valid
    (2*n_splits, n_items) 0/1 (an item is valid for a split only if BOTH halves have it, so both
    directions use the same pairs) and half_pairs [(2s, 2s+1)]."""
    from scipy import sparse
    values = np.asarray(values, dtype=np.float64)
    subj = np.asarray(subject_of_row)
    item = np.asarray(item_of_row)
    n_sub = int(subj.max()) + 1
    v = sparse.csr_matrix((values, (item, subj)), shape=(n_items, n_sub))
    c = sparse.csr_matrix((np.ones(len(values)), (item, subj)), shape=(n_items, n_sub))
    rng = np.random.default_rng([seed, 11])
    means = np.full((2 * n_splits, n_items), np.nan)
    valid = np.zeros((2 * n_splits, n_items))
    for s in range(n_splits):
        perm = rng.permutation(n_sub)
        halves = np.zeros((n_sub, 2))
        halves[perm[:n_sub // 2], 0] = 1
        halves[perm[n_sub // 2:], 1] = 1
        sums, cnts = v @ halves, c @ halves
        ok = (cnts > 0).all(axis=1)
        for h in range(2):
            with np.errstate(invalid="ignore", divide="ignore"):
                means[2 * s + h] = np.where(cnts[:, h] > 0, sums[:, h] / cnts[:, h], np.nan)
            valid[2 * s + h] = ok
    return means, valid, [(2 * s, 2 * s + 1) for s in range(n_splits)]
