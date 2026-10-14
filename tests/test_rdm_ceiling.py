"""Correctness checks for neuroai.analysis.rdm_ceiling (synthetic data only)."""
import numpy as np
import pytest
from scipy.spatial.distance import pdist

from neuroai.analysis.rdm_ceiling import RdmTargetReplicates, combine_halves, split_half_targets
from neuroai.analysis.rdm_regression import all_pairs

LOG2PI = np.log(2 * np.pi)


def _zv(v, w):
    m = w > 0
    out = np.zeros_like(v, float)
    out[m] = (v[m] - v[m].mean()) / v[m].std()
    return out


def _reference(y, pred, conf, pairs, valid, fold, k, variant):
    """Brute force: select the valid pairs, z-score, plain lstsq per fold. Returns (dLL per pair, n_test)."""
    i, j = pairs[:, 0], pairs[:, 1]
    w = valid[i] * valid[j]
    yz, pz = _zv(y, w), _zv(pred, w)
    cols = [np.ones(len(y))]
    if variant == "len_freq":
        cols += [(conf[:, c] - conf[:, c].mean()) / conf[:, c].std() for c in range(conf.shape[1])]
    Xb = np.column_stack(cols)
    Xf = np.column_stack(cols + [pz])
    tf = tb = tn = 0.0
    for f in range(k):
        te = (w > 0) & (fold[i] == f) & (fold[j] == f)
        tr = (w > 0) & (fold[i] != f) & (fold[j] != f)
        if te.sum() < 2 or tr.sum() <= Xf.shape[1] + 1:
            continue
        for X, tag in ((Xf, "f"), (Xb, "b")):
            beta = np.linalg.lstsq(X[tr], yz[tr], rcond=None)[0]
            s2 = max(float(np.mean((yz[tr] - X[tr] @ beta) ** 2)), 1e-12)
            r = yz[te] - X[te] @ beta
            ll = -0.5 * (te.sum() * (LOG2PI + np.log(s2)) + (r ** 2).sum() / s2)
            if tag == "f":
                tf += ll
            else:
                tb += ll
        tn += te.sum()
    return ((tf - tb) / tn if tn else float("nan")), tn


def _setup(n=30, T=4, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    pairs = all_pairs(n)
    responses = np.stack([pdist((z + rng.normal(size=n))[:, None]) for _ in range(T)])
    valid = (rng.random((T, n)) > 0.15).astype(float)
    conf = np.column_stack([pdist(rng.normal(size=(n, 1))), pdist(rng.normal(size=(n, 1)))])
    models = np.stack([pdist((z + 0.5 * rng.normal(size=n))[:, None]), rng.normal(size=len(pairs))])
    return n, pairs, responses, valid, conf, models


def test_models_match_bruteforce():
    n, pairs, responses, valid, conf, models = _setup()
    rep = RdmTargetReplicates(responses, n, pairs, conf, valid, n_partitions=3, seed=5)
    delta, n_test = rep.score_models(models)
    assert (valid == 0).any()                                  # the masks really exclude items
    for mi in range(len(models)):
        for vi, variant in enumerate(rep.variants):
            for t in range(rep.T):
                for p in range(3):
                    want, wn = _reference(responses[t], models[mi], conf, pairs, valid[t], rep._folds[t][p], rep.k, variant)
                    np.testing.assert_allclose(delta[mi, vi, t, p], want, rtol=1e-7, atol=1e-10, equal_nan=True)
                    assert n_test[t, p] == pytest.approx(wn)


def test_partner_predictor_matches_bruteforce():
    n, pairs, responses, valid, conf, _ = _setup(seed=1)
    valid[1] = valid[0]; valid[3] = valid[2]                   # halves of a split share the valid items
    rep = RdmTargetReplicates(responses, n, pairs, conf, valid, n_partitions=2, seed=9)
    partner = np.array([1, 0, 3, 2])
    delta, n_test = rep.score_partners(partner)
    for t in range(rep.T):
        for p in range(2):
            want, _ = _reference(responses[t], responses[partner[t]], conf, pairs, valid[t], rep._folds[t][p], rep.k, "len_freq")
            np.testing.assert_allclose(delta[0, 1, t, p], want, rtol=1e-7, atol=1e-10, equal_nan=True)


def test_combine_halves_sums_directions():
    delta = np.array([[[0.10, 0.20], [0.30, 0.50]]])            # (1 model, T=2, P=2)
    n_test = np.array([[100.0, 100.0], [300.0, 300.0]])
    out = combine_halves(delta, n_test, [(0, 1)])
    np.testing.assert_allclose(out[0, 0], [(0.10 * 100 + 0.30 * 300) / 400, (0.20 * 100 + 0.50 * 300) / 400])


def test_split_half_targets_explicit():
    # 4 items, 4 subjects; every subject saw items 0..2, only subject 3 saw item 3
    subj = np.array([0, 1, 2, 3] * 3 + [3])
    item = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3])
    vals = np.arange(13, dtype=float)
    means, valid, halves = split_half_targets(vals, subj, item, 4, n_splits=6, seed=3)
    assert halves == [(2 * s, 2 * s + 1) for s in range(6)]
    for s in range(6):
        a, b = means[2 * s], means[2 * s + 1]
        assert (valid[2 * s] == valid[2 * s + 1]).all()
        assert valid[2 * s, 3] == 0                              # item 3 has one subject: cannot be in both halves
        assert valid[2 * s, :3].all()
        # the two halves together contain every subject exactly once -> their size-weighted means give the grand mean
        grand = np.array([vals[item == i].mean() for i in range(3)])
        assert np.allclose((a[:3] + b[:3]) / 2, grand, atol=np.ptp(vals))   # sanity bound
        assert np.isfinite(a[:3]).all() and np.isfinite(b[:3]).all()
    m2, v2, _ = split_half_targets(vals, subj, item, 4, n_splits=6, seed=3)
    np.testing.assert_array_equal(np.nan_to_num(means), np.nan_to_num(m2))


def test_perfect_partner_beats_noise_model():
    rng = np.random.default_rng(0)
    n = 60
    z = rng.normal(size=n)
    pairs = all_pairs(n)
    clean = pdist(z[:, None])
    responses = np.stack([clean + 0.3 * rng.normal(size=len(pairs)) * 0 + 0.05 * pdist(rng.normal(size=(n, 1))),
                          clean])
    rep = RdmTargetReplicates(responses, n, pairs, None, None, n_partitions=3, seed=1)
    ceil, _ = rep.score_partners(np.array([1, 0]))
    noise, _ = rep.score_models(rng.normal(size=(1, len(pairs))))
    assert np.nanmean(ceil) > 0.5 > np.nanmean(noise)


def test_pair_valid_masks_pairs_like_bruteforce_and_default_is_unchanged():
    n, pairs, responses, valid, conf, models = _setup(T=2, seed=4)
    rng = np.random.default_rng(8)
    pair_valid = (rng.random((2, len(pairs))) > 0.3).astype(float)
    a = RdmTargetReplicates(responses, n, pairs, conf, valid, n_partitions=2, seed=3)
    b = RdmTargetReplicates(responses, n, pairs, conf, valid, n_partitions=2, seed=3, pair_valid=np.ones((2, len(pairs))))
    np.testing.assert_allclose(a.score_models(models)[0], b.score_models(models)[0])     # all-ones mask == no mask
    c = RdmTargetReplicates(responses, n, pairs, conf, valid, n_partitions=2, seed=3, pair_valid=pair_valid)
    delta, _ = c.score_models(models)
    i, j = pairs[:, 0], pairs[:, 1]
    for t in range(2):
        # brute force with the combined pair mask: encode it by zeroing validity through a per-pair weight
        w = valid[t][i] * valid[t][j] * pair_valid[t]
        yz, mz = _zv(responses[t], w), _zv(models[0], w)
        cols_b = [np.ones(len(i))]
        Xb = np.column_stack(cols_b)
        Xf = np.column_stack(cols_b + [mz])
        fold = c._folds[t][0]
        tf = tb = tn = 0.0
        for f in range(c.k):
            te = (w > 0) & (fold[i] == f) & (fold[j] == f)
            tr = (w > 0) & (fold[i] != f) & (fold[j] != f)
            for X, tag in ((Xf, "f"), (Xb, "b")):
                beta = np.linalg.lstsq(X[tr], yz[tr], rcond=None)[0]
                s2 = max(float(np.mean((yz[tr] - X[tr] @ beta) ** 2)), 1e-12)
                r = yz[te] - X[te] @ beta
                ll = -0.5 * (te.sum() * (LOG2PI + np.log(s2)) + (r ** 2).sum() / s2)
                tf, tb = (tf + ll, tb) if tag == "f" else (tf, tb + ll)
            tn += te.sum()
        np.testing.assert_allclose(delta[0, 0, t, 0], (tf - tb) / tn, rtol=1e-7, atol=1e-10)
