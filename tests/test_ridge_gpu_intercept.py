"""GPU ridge == sklearn ridge with intercept (2026-10-12 audit).

``cv_ridge_score_multioutput_gpu`` solves ridge without an intercept; before the audit X was
only median-centred (RobustScaler), so it differed from the CPU path. The CUDA code is run
on CPU by redirecting ``torch.device``; the equations are unchanged.
"""
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from neuroai.regressors import ridge  # noqa: E402


def _gpu_on_cpu(fn, *args, **kwargs):
    real_device = torch.device
    with patch.object(torch.cuda, "current_device", return_value=0), \
            patch.object(torch.cuda, "empty_cache"), \
            patch.object(torch, "device", side_effect=lambda s: real_device("cpu") if str(s).startswith("cuda") else real_device(s)):
        return fn(*args, **kwargs)


@pytest.mark.parametrize("standardize_target", [True, False])
def test_gpu_multioutput_matches_cpu(standardize_target):
    rng = np.random.default_rng(4)
    X = rng.exponential(size=(160, 12)) + 3.0  # skewed, non-zero mean features
    Y = X @ rng.normal(size=(12, 2)) + 50.0 + rng.normal(size=(160, 2))
    groups = np.repeat(np.arange(16), 10)
    cpu, cpu_pred = ridge.cv_ridge_score_multioutput(
        X, Y, groups=groups, alpha=[10.0], return_oof_predictions=True)
    gpu, gpu_pred = _gpu_on_cpu(
        ridge.cv_ridge_score_multioutput_gpu, X, Y, groups=groups, alpha=[10.0],
        standardize_target=standardize_target, return_oof_predictions=True)
    if standardize_target:
        # alpha acts on the standardised target there, so only check the fit quality
        assert abs(cpu[0].pearsonr - gpu[0].pearsonr) < 0.02
    else:
        np.testing.assert_allclose(gpu_pred, cpu_pred, rtol=1e-3, atol=1e-3)
