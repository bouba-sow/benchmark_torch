"""The dense all-pairs ceiling must reproduce rdm_ceiling_units.unit_ceiling (RdmTargetReplicates) on random data."""
import numpy as np
import pytest

from neuroai.analysis.rdm_ceiling_dense import DenseCeilingDesign, dense_unit_ceiling
from neuroai.analysis.rdm_regression import all_pairs
from scripts.analysis import rdm_ceiling_units as U


def pair_abs_diff(v):
    return np.concatenate([np.abs(v[i] - v[i + 1:]) for i in range(len(v) - 1)])


@pytest.mark.parametrize("n,n_sub,missing,seed", [(48, 12, 0.0, 3), (61, 9, 0.35, 8)])
def test_dense_ceiling_matches_sparse(n, n_sub, missing, seed):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    rows = []
    for s in range(n_sub):
        for i in range(n):
            if rng.random() >= missing:
                rows.append((s, i, base[i] + 0.8 * rng.normal()))
    subj, item, val = (np.array(c) for c in zip(*rows))
    wlen, freq = rng.normal(size=n), rng.normal(size=n)
    pairs = all_pairs(n)
    conf = np.column_stack([pair_abs_diff(wlen), pair_abs_diff(freq)])
    U.N_SPLITS = 5
    ref = U.unit_ceiling(val.astype(float), subj, item, n, pairs, conf, seed)
    design = DenseCeilingDesign(n, [conf[:, 0], conf[:, 1]])
    got = dense_unit_ceiling(val.astype(float), subj, item, design, 5, seed)
    for key in ("r_half", "dll_ceiling", "dll_ceiling_sd"):
        assert got[key] == pytest.approx(ref[key], rel=1e-8, abs=1e-12), key
    assert got["rsa_ceiling"] == pytest.approx(ref["rsa_ceiling"], rel=1e-8) or (np.isnan(got["rsa_ceiling"]) and np.isnan(ref["rsa_ceiling"]))
    assert got["n_pairs"] == ref["n_pairs"] and got["n_subjects"] == ref["n_subjects"]
