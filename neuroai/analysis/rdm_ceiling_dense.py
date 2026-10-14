"""Dense (all-pairs) split-half RDM ceilings: the computation of ``rdm_ceiling.RdmTargetReplicates.score_partners`` + ``combine_halves`` +
the split-half RDM reliability, done with matrix products instead of bincounts over the pair list (see rdm_regression_dense.py).

For one unit, ``split_half_targets`` gives 2 x n_splits half-sample item means (NaN where a half has no data) and an item validity per
split (an item counts only if BOTH halves have it). A pair (i, j) of a target is valid iff both items are, i.e. its weight is
v_i v_j, which is absorbed into the fold-indicator matrix (rows scaled by v): weighted cell sums of an upper-triangular matrix X are
F_v^T X F_v. Everything else (the z-scoring of the response over the valid pairs, the item folds drawn per (target, partition) from
default_rng([seed, 2, t, p]), the OLS fits, the Gaussian test log-likelihoods, the two directions of a split summed) is that of
RdmTargetReplicates; tests/test_rdm_ceiling_dense.py checks equality on random data.

Memory: about 6 dense n x n float64 matrices (n = 5,000: 1.2 GB; n = 11,000: 6 GB). Only all-pairs designs are covered.
"""
from __future__ import annotations

import numpy as np

from neuroai.analysis.rdm_ceiling import _dll_and_n, combine_halves, split_half_targets
from neuroai.analysis.rdm_regression import _zscore, item_fold_assignment, n_folds_for
from neuroai.analysis.rdm_regression_dense import fill_upper


def _masked_z_matrix(m: np.ndarray, v: np.ndarray):
    """Upper-triangular dense matrix of the response RDM |m_i - m_j|, z-scored over the VALID pairs (v_i v_j = 1), zeros elsewhere
    (the zeros outside the valid pairs are removed by the validity weights). Returns (matrix, n_valid_pairs)."""
    n = len(m)
    mm = np.where(v > 0, m, 0.0)
    nv = float(v.sum())
    n_valid = nv * (nv - 1) / 2
    s1 = s2 = 0.0
    for i in range(n - 1):
        if v[i] > 0:
            d = np.abs(mm[i] - mm[i + 1:]) * v[i + 1:]
            s1 += d.sum()
            s2 += (d * d).sum()
    mu = s1 / n_valid
    sd = np.sqrt(max(s2 / n_valid - mu * mu, 0.0))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("constant RDM on the valid pairs")
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        out[i, i + 1:] = (np.abs(mm[i] - mm[i + 1:]) - mu) / sd
    return out, n_valid


class DenseCeilingDesign:
    """Confound matrices (z-scored over ALL pairs, as RdmTargetReplicates does) shared by the units of a dataset."""

    def __init__(self, n_items: int, confounds: list[np.ndarray] | None, n_partitions: int = 5, n_folds: int | None = None):
        self.n, self.P = n_items, n_partitions
        self.k = n_folds or n_folds_for(n_items)
        self.n_pairs = n_items * (n_items - 1) // 2
        self.C = [fill_upper(_zscore(np.asarray(c, dtype=np.float64), f"confound {j}"), n_items) for j, c in enumerate(confounds or [])]
        self.q = 1 + len(self.C)
        self.variants = ("none", "len_freq") if confounds is not None else ("none",)
        self._idx = {"none": [0], "len_freq": list(range(self.q))}

    def _folds(self, t: int, v: np.ndarray, seed: int) -> np.ndarray:
        """(n, P K) fold indicators of target t with the rows scaled by the item validity."""
        n, k, P = self.n, self.k, self.P
        F = np.zeros((n, P * k))
        for p in range(P):
            fold = item_fold_assignment(n, k, np.random.default_rng([seed, 2, t, p]))
            F[np.arange(n), p * k + fold] = v
        return F

    def _cells(self, X: np.ndarray, F: np.ndarray) -> np.ndarray:
        k = self.k
        T_ = X @ F
        return np.stack([F[:, p * k:(p + 1) * k].T @ T_[:, p * k:(p + 1) * k] for p in range(self.P)])

    def _cells_ones(self, F: np.ndarray) -> np.ndarray:
        """cells of the all-ones upper triangle without building it: (U F)[i] = sum_{j > i} F[j]."""
        k = self.k
        UF = F[::-1].cumsum(axis=0)[::-1] - F
        return np.stack([F[:, p * k:(p + 1) * k].T @ UF[:, p * k:(p + 1) * k] for p in range(self.P)])

    def _direction(self, t: int, v: np.ndarray, Y: np.ndarray, M: np.ndarray, seed: int):
        """delta LL per test pair and test weight of predicting target t (response Y) from predictor M, per partition -> (variants, P), (P,)."""
        k, q, P = self.k, self.q, self.P
        F = self._folds(t, v, seed)
        nn = self._cells_ones(F)                                                   # (P, K, K)
        btb = np.empty((P, q, q, k, k))
        btb[:, 0, 0] = nn
        for a in range(1, q):
            btb[:, 0, a] = btb[:, a, 0] = self._cells(self.C[a - 1], F)
            for b in range(a, q):
                btb[:, a, b] = btb[:, b, a] = self._cells(self.C[a - 1] * self.C[b - 1], F)
        bty = np.stack([self._cells(Y, F)] + [self._cells(Ca * Y, F) for Ca in self.C], axis=1)
        yy = self._cells(Y * Y, F)
        bm = np.stack([self._cells(M, F)] + [self._cells(Ca * M, F) for Ca in self.C], axis=1)
        my = self._cells(Y * M, F)
        mm = self._cells(M * M, F)
        delta = np.full((len(self.variants), P), np.nan)
        n_test = np.zeros(P)
        for p in range(P):
            for vi, variant in enumerate(self.variants):
                d, n = _dll_and_n(btb[p], bty[p], yy[p], nn[p], bm[p], my[p], mm[p], k, self._idx[variant])
                delta[vi, p] = d
                n_test[p] = n
        return delta, n_test

    def run(self, means: np.ndarray, valid: np.ndarray, half_pairs, seed: int):
        """All split-half targets of one unit. Returns (comb (variants, splits, P) delta LL per test pair with the two directions of a split
        combined, n_test (T, P), r_half per split)."""
        T = len(means)
        delta = np.full((1, len(self.variants), T, self.P), np.nan)
        n_test = np.zeros((T, self.P))
        rs = []
        for ta, tb in half_pairs:
            v = valid[ta]
            ZA, nv = _masked_z_matrix(means[ta], v)
            ZB, _ = _masked_z_matrix(means[tb], v)
            if nv > 10:
                rs.append(float(v @ (ZA * ZB) @ v / nv))   # corr over the valid pairs (both z-scored over the same pairs)
            for t, Y, M in ((ta, ZA, ZB), (tb, ZB, ZA)):
                d, n = self._direction(t, v, Y, M, seed)
                delta[0, :, t, :] = d
                n_test[t] = n
            del ZA, ZB
        return combine_halves(delta, n_test, half_pairs)[0], n_test, rs

    def unit_ceiling(self, means: np.ndarray, valid: np.ndarray, half_pairs, n_subjects: int, seed: int) -> dict:
        """Ceilings of one unit from its split-half targets (``split_half_targets`` output). Same keys as rdm_ceiling_units.unit_ceiling."""
        comb, _, rs = self.run(means, valid, half_pairs, seed)
        per_split = np.nanmean(comb[self.variants.index("len_freq")], axis=-1)
        r_half = float(np.mean(rs)) if rs else np.nan
        r_sb = 2 * r_half / (1 + r_half) if np.isfinite(r_half) and r_half > -1 else np.nan
        return dict(r_half=r_half, rsa_ceiling=float(np.sqrt(r_sb)) if np.isfinite(r_sb) and r_sb > 0 else np.nan,
                    dll_ceiling=float(np.mean(per_split)), dll_ceiling_sd=float(np.std(per_split, ddof=1)),
                    n_subjects=n_subjects, n_items=self.n, n_pairs=self.n_pairs)


def dense_unit_ceiling(values, subj, item, design: DenseCeilingDesign, n_splits: int, seed: int) -> dict:
    """Drop-in for rdm_ceiling_units.unit_ceiling on all pairs."""
    means, valid, half_pairs = split_half_targets(values, subj, item, design.n, n_splits, seed)
    return design.unit_ceiling(means, valid, half_pairs, int(np.max(subj)) + 1, seed)
