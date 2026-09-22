"""Evaluation utilities: bootstrap confidence intervals and result tables."""

from __future__ import annotations

import typing as tp

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def bootstrap_pearsonr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    ci: float = 95.0,
    random_state: int = 0,
) -> dict[str, float]:
    """Bootstrap confidence interval for Pearson r.

    Returns
    -------
    dict with keys ``r``, ``ci_low``, ``ci_high``.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_r = np.array(
        [
            pearsonr(
                y_true[idx := rng.integers(0, n, size=n)],
                y_pred[idx],
            ).statistic
            for _ in range(n_boot)
        ]
    )
    alpha = (100.0 - ci) / 2
    return {
        "r": float(pearsonr(y_true, y_pred).statistic),
        "ci_low": float(np.nanpercentile(boot_r, alpha)),
        "ci_high": float(np.nanpercentile(boot_r, 100.0 - alpha)),
    }


def summarize_results(
    results: list[dict[str, tp.Any]],
) -> pd.DataFrame:
    """Convert a list of result dicts into a tidy DataFrame.

    Each dict should contain at minimum: ``model_name``, ``layer``, ``pearsonr``.

    Returns a long-format DataFrame ready for plotting.
    """
    return pd.DataFrame(results).sort_values(
        ["model_name", "layer"], ignore_index=True
    )
