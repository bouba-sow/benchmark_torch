"""Synthetic checks for neuroai.analysis.rdm_regression (no repo outputs touched)."""
import numpy as np
import pytest
from scipy.spatial.distance import pdist

from neuroai.analysis.rdm_regression import all_pairs, cv_delta_ll, n_folds_for


def _rdms(n=60, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    conf = rng.normal(size=n)
    return rng, z, conf


def test_pairs_match_pdist_order():
    x = np.random.default_rng(1).normal(size=9)
    p = all_pairs(9)
    assert np.allclose(np.abs(x[p[:, 0]] - x[p[:, 1]]), pdist(x[:, None]))


def test_fold_schedule():
    assert [n_folds_for(n) for n in (6, 12, 24, 40, 500)] == [2, 3, 4, 5, 5]
    with pytest.raises(ValueError):
        n_folds_for(5)


def test_informative_model_beats_noise_model():
    rng, z, _ = _rdms()
    n = len(z)
    y = pdist(z[:, None]) + 0.5 * rng.normal(size=n * (n - 1) // 2)
    good = pdist(z[:, None])
    noise = pdist(rng.normal(size=(n, 1)))
    d_good = cv_delta_ll(y, good, n).delta_ll_per_obs
    d_noise = cv_delta_ll(y, noise, n).delta_ll_per_obs
    assert d_good > 0.1
    assert d_noise < 0.02 < d_good


def test_confound_absorbs_shared_signal():
    rng, z, conf = _rdms()
    n = len(z)
    conf_rdm = pdist(conf[:, None])
    # model RDM is a noisy copy of the confound; response is driven by the confound only
    y = conf_rdm + 0.3 * rng.normal(size=len(conf_rdm))
    model = conf_rdm + 0.3 * rng.normal(size=len(conf_rdm))
    without = cv_delta_ll(y, model, n).delta_ll_per_obs
    with_conf = cv_delta_ll(y, model, n, confounds=conf_rdm).delta_ll_per_obs
    assert without > 0.1
    assert with_conf < without / 3


def test_sampled_pairs_and_coverage():
    rng, z, _ = _rdms(n=80)
    n = len(z)
    pairs = all_pairs(n)
    keep = rng.choice(len(pairs), 1500, replace=False)
    pairs = pairs[np.sort(keep)]
    x = np.abs(z[pairs[:, 0]] - z[pairs[:, 1]])
    y = x + 0.5 * rng.normal(size=len(x))
    res = cv_delta_ll(y, x, n, pairs=pairs, n_repeats=4)
    assert res.delta_ll_per_obs > 0
    # 5 folds: within-fold pairs are about 1/5 of all pairs, test pairs one fold at a time
    assert 0.1 < res.frac_pairs_tested < 0.3


def test_deterministic_given_seed():
    rng, z, _ = _rdms()
    n = len(z)
    y = pdist(z[:, None]) + rng.normal(size=n * (n - 1) // 2)
    a = cv_delta_ll(y, pdist(z[:, None]), n, seed=3)
    b = cv_delta_ll(y, pdist(z[:, None]), n, seed=3)
    assert a.delta_ll_per_obs == b.delta_ll_per_obs


def _reference(y, m, n, pairs, conf, n_repeats, seed):
    """Brute-force per-fold lstsq on row-selected pairs (what RdmCvPlan must equal)."""
    from neuroai.analysis.rdm_regression import item_fold_assignment
    y = (y - y.mean()) / y.std()
    m = (m - m.mean()) / m.std()
    cols_b = [np.ones_like(y)]
    if conf is not None:
        c = np.asarray(conf, float).reshape(len(y), -1)
        cols_b += [(c[:, j] - c[:, j].mean()) / c[:, j].std() for j in range(c.shape[1])]
    Xb, Xf = np.column_stack(cols_b), np.column_stack(cols_b + [m])
    k = n_folds_for(n)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_repeats):
        fold = item_fold_assignment(n, k, rng)
        fa, fb = fold[pairs[:, 0]], fold[pairs[:, 1]]
        tf = tb = 0.0
        tn = 0
        for f in range(k):
            te, tr = (fa == f) & (fb == f), (fa != f) & (fb != f)
            for X, tag in ((Xf, "f"), (Xb, "b")):
                beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
                s2 = max(np.mean((y[tr] - X[tr] @ beta) ** 2), 1e-12)
                r = y[te] - X[te] @ beta
                ll = -0.5 * (te.sum() * (np.log(2 * np.pi) + np.log(s2)) + (r ** 2).sum() / s2)
                if tag == "f":
                    tf += ll
                else:
                    tb += ll
            tn += te.sum()
        out.append((tf - tb) / tn)
    return float(np.mean(out))


@pytest.mark.parametrize("with_conf", [False, True])
def test_matches_bruteforce_reference(with_conf):
    rng, z, conf = _rdms(n=45, seed=5)
    n = len(z)
    pairs = all_pairs(n)
    c = np.column_stack([pdist(conf[:, None]), pdist(rng.normal(size=(n, 1)))]) if with_conf else None
    y = pdist(z[:, None]) + rng.normal(size=len(pairs))
    m = pdist(z[:, None]) * 0.7 + rng.normal(size=len(pairs))
    got = cv_delta_ll(y, m, n, confounds=c, n_repeats=3, seed=7).delta_ll_per_obs
    assert got == pytest.approx(_reference(y, m, n, pairs, c, 3, 7), rel=1e-6, abs=1e-9)


def test_plan_pearson_matches_numpy():
    from neuroai.analysis.rdm_regression import RdmCvPlan
    rng, z, _ = _rdms(n=30)
    y = pdist(z[:, None]) + rng.normal(size=30 * 29 // 2)
    m = pdist(z[:, None])
    assert RdmCvPlan(y, 30, n_repeats=1).pearson(m) == pytest.approx(np.corrcoef(y, m)[0, 1])


def test_driver_layer_rdm_matches_pdist_correlation():
    from scripts.analysis.rdm_regression_behavioral import _layer_rdm
    x = np.random.default_rng(2).normal(size=(25, 40)).astype(np.float32)
    assert np.allclose(_layer_rdm(x, all_pairs(25)), pdist(x, metric="correlation"), atol=1e-5)


def test_driver_layer_rdm_blockwise_matches_gram():
    from scripts.analysis.rdm_regression_behavioral import _layer_rdm
    x = np.random.default_rng(4).normal(size=(30, 40)).astype(np.float32)
    pairs = all_pairs(30)[::3]
    full = _layer_rdm(x, pairs)
    blocked = _layer_rdm(x, pairs, gram_max_items=10, block=7)
    assert np.allclose(full, blocked, atol=1e-5)


def test_standardize_features_removes_massive_dimension_and_scale():
    from neuroai.analysis.rdm_regression import standardize_features
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 40)).astype(np.float32)
    big = x.copy()
    big[:, 3] = 1000.0 + 50.0 * rng.normal(size=60)          # a massive-activation dimension
    scaled = x.copy()
    scaled[:, 3] *= 1000.0                                    # per-feature rescaling
    np.testing.assert_allclose(standardize_features(x)[:, 3], standardize_features(scaled)[:, 3], atol=1e-4)
    z = standardize_features(big)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4) and np.allclose(z.std(axis=0), 1, atol=1e-3)
    const = np.ones((10, 3), dtype=np.float32)
    assert np.all(standardize_features(const) == 0)            # constant features become 0, no NaN
