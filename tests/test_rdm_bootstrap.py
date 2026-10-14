"""Correctness checks for neuroai.analysis.rdm_bootstrap (synthetic data only)."""
import numpy as np
import pytest
from scipy.spatial.distance import pdist

from neuroai.analysis.rdm_bootstrap import BootResult, RdmBootstrap
from neuroai.analysis.rdm_regression import all_pairs

LOG2PI = np.log(2 * np.pi)


def _z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


def _data(n=20, seed=0, with_conf=True):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    pairs = all_pairs(n)
    y = pdist(z[:, None]) + 0.6 * rng.normal(size=len(pairs))
    m = 0.8 * pdist(z[:, None]) + 0.6 * rng.normal(size=len(pairs))
    conf = np.column_stack([pdist(rng.normal(size=(n, 1))), pdist(rng.normal(size=(n, 1)))]) if with_conf else None
    return n, pairs, y, m, conf


def _design(m, conf, variant):
    cols = [np.ones_like(m)]
    if variant == "len_freq":
        cols += [_z(conf[:, j]) for j in range(conf.shape[1])]
    base = np.column_stack(cols)
    return base, np.column_stack(cols + [_z(m)])


def _ll_folds(X_tr, y_tr, w_tr, X_te, y_te, w_te):
    sw = np.sqrt(w_tr)
    beta = np.linalg.lstsq(X_tr * sw[:, None], y_tr * sw, rcond=None)[0]
    s2 = max(float((w_tr * (y_tr - X_tr @ beta) ** 2).sum() / w_tr.sum()), 1e-12)
    r = y_te - X_te @ beta
    return -0.5 * (w_te.sum() * (LOG2PI + np.log(s2)) + (w_te * r ** 2).sum() / s2)


def _reference_weighted(y, m, conf, pairs, counts, fold, k, variant):
    """Brute-force weighted lstsq on row-selected pairs (what the sufficient-statistics code must equal)."""
    Xb, Xf = _design(m, conf, variant)
    yz = _z(y)
    i, j = pairs[:, 0], pairs[:, 1]
    w = counts[i] * counts[j]
    tf = tb = tn = 0.0
    for f in range(k):
        te = (fold[i] == f) & (fold[j] == f)
        tr = (fold[i] != f) & (fold[j] != f)
        if w[te].sum() < 2 or w[tr].sum() <= Xf.shape[1] + 1:
            continue
        tf += _ll_folds(Xf[tr], yz[tr], w[tr], Xf[te], yz[te], w[te])
        tb += _ll_folds(Xb[tr], yz[tr], w[tr], Xb[te], yz[te], w[te])
        tn += w[te].sum()
    return (tf - tb) / tn if tn > 0 else float("nan")


def _reference_explicit(y, m, conf, n, counts, fold, k, variant):
    """Literally duplicate the resampled items, drop self-pairs, fit plain OLS on the rows."""
    Xb_orig, Xf_orig = _design(m, conf, variant)
    yz = _z(y)
    index = {}
    for r, (a, b) in enumerate(all_pairs(n)):
        index[(a, b)] = r
    items = np.repeat(np.arange(n), counts.astype(int))
    rows, ia, ib = [], [], []
    for p in range(len(items)):
        for q in range(p + 1, len(items)):
            a, b = items[p], items[q]
            if a == b:
                continue                       # self-dissimilarity: excluded
            rows.append(index[(min(a, b), max(a, b))])
            ia.append(a)
            ib.append(b)
    rows, ia, ib = np.array(rows), np.array(ia), np.array(ib)
    tf = tb = tn = 0.0
    ones = np.ones(len(rows))
    for f in range(k):
        te = (fold[ia] == f) & (fold[ib] == f)
        tr = (fold[ia] != f) & (fold[ib] != f)
        if te.sum() < 2 or tr.sum() <= Xf_orig.shape[1] + 1:
            continue
        for X, tag in ((Xf_orig[rows], "f"), (Xb_orig[rows], "b")):
            ll = _ll_folds(X[tr], yz[rows][tr], ones[tr], X[te], yz[rows][te], ones[te])
            if tag == "f":
                tf += ll
            else:
                tb += ll
        tn += te.sum()
    return (tf - tb) / tn if tn > 0 else float("nan")


@pytest.mark.parametrize("variant", ["none", "len_freq"])
def test_matches_weighted_bruteforce(variant):
    n, pairs, y, m, conf = _data()
    boot = RdmBootstrap(y, n, pairs, conf, n_boot=6, n_partitions=2, seed=3)
    res = boot.delta_ll_batch(m[None])
    vi = boot.variants.index(variant)
    for b in range(boot.n_rep):
        for p in range(2):
            want = _reference_weighted(y, m, conf, pairs, boot.counts[b], boot._folds[b][p], boot.k, variant)
            np.testing.assert_allclose(res.delta[0, vi, b, p], want, rtol=1e-7, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("variant", ["none", "len_freq"])
def test_weights_equal_explicit_resampling(variant):
    """The weight c_i*c_j representation == duplicating the resampled items (self-pairs removed),
    with copies of an item kept in the same fold."""
    n, pairs, y, m, conf = _data(n=20, seed=1)
    boot = RdmBootstrap(y, n, pairs, conf, n_boot=5, n_partitions=2, seed=11)
    res = boot.delta_ll_batch(m[None])
    vi = boot.variants.index(variant)
    assert (boot.counts.sum(axis=1) == n).all()
    assert (boot.counts == 0).any() and (boot.counts > 1).any()     # the test really exercises duplicates
    for b in range(boot.n_rep):
        for p in range(2):
            want = _reference_explicit(y, m, conf, n, boot.counts[b], boot._folds[b][p], boot.k, variant)
            np.testing.assert_allclose(res.delta[0, vi, b, p], want, rtol=1e-7, atol=1e-10, equal_nan=True)


def test_original_replicate_uses_unit_weights():
    n, pairs, y, m, conf = _data(seed=2)
    boot = RdmBootstrap(y, n, pairs, conf, n_boot=2, n_partitions=2, seed=5, include_original=True)
    res = boot.delta_ll_batch(m[None])
    assert (boot.counts[0] == 1).all()
    for p in range(2):
        want = _reference_weighted(y, m, conf, pairs, np.ones(n), boot._folds[0][p], boot.k, "len_freq")
        assert res.delta[0, 1, 0, p] == pytest.approx(want, rel=1e-7, abs=1e-10)
    assert res.point.shape == (1, 2)


def test_sampled_pairs_supported():
    rng = np.random.default_rng(4)
    n = 40
    z = rng.normal(size=n)
    pairs = all_pairs(n)
    pairs = pairs[np.sort(rng.choice(len(pairs), 400, replace=False))]
    y = np.abs(z[pairs[:, 0]] - z[pairs[:, 1]]) + 0.5 * rng.normal(size=len(pairs))
    m = np.abs(z[pairs[:, 0]] - z[pairs[:, 1]])
    boot = RdmBootstrap(y, n, pairs, None, n_boot=4, n_partitions=2, seed=1)
    res = boot.delta_ll_batch(m[None])
    want = _reference_weighted(y, m, None, pairs, boot.counts[2], boot._folds[2][1], boot.k, "none")
    assert res.delta[0, 0, 2, 1] == pytest.approx(want, rel=1e-7, abs=1e-10)
    assert np.nanmean(res.theta()) > 0


def test_models_are_paired_and_deterministic():
    n, pairs, y, m, conf = _data(seed=6)
    m2 = np.random.default_rng(9).normal(size=len(y))
    a = RdmBootstrap(y, n, pairs, conf, n_boot=5, seed=2).delta_ll_batch(np.stack([m, m2]))
    b = RdmBootstrap(y, n, pairs, conf, n_boot=5, seed=2).delta_ll_batch(np.stack([m2, m]))
    assert np.allclose(a.delta[0], b.delta[1]) and np.allclose(a.delta[1], b.delta[0])


def test_se_correction_recovers_true_sd():
    """delta = theta_b + partition noise: the corrected SE must recover sd(theta_b), not the inflated one."""
    rng = np.random.default_rng(0)
    B, P, true_sd, noise_sd = 6000, 2, 0.010, 0.010
    theta = rng.normal(0.02, true_sd, size=B)
    delta = theta[:, None] + rng.normal(0, noise_sd, size=(B, P))
    r = BootResult(delta[None, None], ("none",), include_original=False)
    naive = float(np.sqrt(np.nanvar(r.theta(), axis=-1, ddof=1))[0, 0])
    assert naive > 1.15 * true_sd                          # uncorrected is visibly inflated
    assert r.se()[0, 0] == pytest.approx(true_sd, rel=0.06)


def test_diff_api():
    rng = np.random.default_rng(1)
    base = rng.normal(0.02, 0.01, size=(400, 1))
    delta = np.stack([base + 0.01 + rng.normal(0, 0.002, size=(400, 2)),
                      base + rng.normal(0, 0.002, size=(400, 2))])[:, None]
    r = BootResult(delta, ("none",), include_original=False)
    d = r.diff(0, 1)
    assert d["mean"][0] == pytest.approx(0.01, abs=0.001)
    assert d["p_a_greater"][0] > 0.99 and d["ci_low"][0] > 0
    assert d["se"][0] < 0.004                              # paired: shared resample variation cancels


def test_se_naive_is_upper_bound():
    rng = np.random.default_rng(3)
    theta = rng.normal(0.02, 0.01, size=500)
    delta = theta[:, None] + rng.normal(0, 0.01, size=(500, 2))
    r = BootResult(delta[None, None], ("none",), include_original=False)
    assert r.se_naive()[0, 0] >= r.se()[0, 0] >= 0
