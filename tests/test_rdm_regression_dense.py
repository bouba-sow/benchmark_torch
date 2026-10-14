"""The dense all-pairs plan must reproduce RdmCvPlan (same item partitions, same sums, same log-likelihoods)."""
import numpy as np
import pytest

from neuroai.analysis.rdm_regression import RdmCvPlan
from neuroai.analysis.rdm_regression_dense import DenseDesign, fill_upper


def pair_abs_diff(v):
    n = len(v)
    return np.concatenate([np.abs(v[i] - v[i + 1:]) for i in range(n - 1)])


@pytest.mark.parametrize("n,repeats,seed", [(17, 3, 0), (50, 4, 5), (83, 2, 11)])
def test_dense_matches_sparse(n, repeats, seed):
    rng = np.random.default_rng(seed)
    wlen, freq = rng.normal(size=n), rng.normal(size=n)
    y = rng.normal(size=n)
    s1, s2 = rng.normal(size=n), rng.normal(size=n)
    conf = np.column_stack([pair_abs_diff(wlen), pair_abs_diff(freq)])
    resp = pair_abs_diff(y + 0.3 * s1)
    sparse = RdmCvPlan(resp, n, None, conf, repeats, seed=seed)
    design = DenseDesign(n, [conf[:, 0], conf[:, 1]], repeats, seed=seed)
    dense = design.plan(resp)
    for s in (s1, s2):
        model = pair_abs_diff(s)
        a, b = sparse.delta_ll(model), dense.delta_ll(model)
        for f in ("delta_ll_per_obs", "delta_ll_per_obs_sd", "ll_full_per_obs", "ll_base_per_obs", "n_test_pairs", "frac_pairs_tested"):
            assert getattr(a, f) == pytest.approx(getattr(b, f), rel=1e-9, abs=1e-12), f
        assert a.n_folds == b.n_folds
        assert sparse.pearson(model) == pytest.approx(dense.pearson(model), rel=1e-12)


def test_fill_upper_order():
    v = np.arange(1.0, 7.0)  # n = 4: pairs (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
    m = fill_upper(v, 4)
    assert m[0, 1] == 1 and m[0, 3] == 3 and m[1, 2] == 4 and m[2, 3] == 6 and m[3, 0] == 0 and m[1, 1] == 0


def test_dense_without_confounds_matches_sparse():
    """The 'none' variant of the behavioral runs: intercept + model only."""
    n, repeats, seed = 57, 3, 4
    rng = np.random.default_rng(seed)
    y, s = rng.normal(size=n), rng.normal(size=n)
    resp = pair_abs_diff(y + 0.4 * s)
    model = pair_abs_diff(s)
    sparse = RdmCvPlan(resp, n, None, None, repeats, seed=seed)
    dense = DenseDesign(n, [], repeats, seed=seed).plan(resp)
    a, b = sparse.delta_ll(model), dense.delta_ll(model)
    for f in ("delta_ll_per_obs", "delta_ll_per_obs_sd", "ll_full_per_obs", "ll_base_per_obs", "n_test_pairs", "frac_pairs_tested"):
        assert getattr(a, f) == pytest.approx(getattr(b, f), rel=1e-9, abs=1e-12), f
    assert sparse.pearson(model) == pytest.approx(dense.pearson(model), rel=1e-12)


def test_model_stats_cache_gives_identical_results():
    """Scoring several units with a shared model key (cached bm, mm) equals scoring each without the cache."""
    n, repeats, seed = 52, 3, 9
    rng = np.random.default_rng(seed)
    conf = [pair_abs_diff(rng.normal(size=n)), pair_abs_diff(rng.normal(size=n))]
    design = DenseDesign(n, conf, repeats, seed=seed)
    models = [pair_abs_diff(rng.normal(size=n)) for _ in range(3)]
    plans = [design.plan(pair_abs_diff(rng.normal(size=n))) for _ in range(3)]
    for key, model in enumerate(models):
        for plan in plans:
            a = plan.delta_ll(model)
            b = plan.delta_ll(model, key)
            assert a.delta_ll_per_obs == pytest.approx(b.delta_ll_per_obs, rel=1e-12, abs=1e-15)
            assert a.n_test_pairs == b.n_test_pairs
    assert len(design._model_stats) == 3
