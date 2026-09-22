"""Shared OLS helper for the matched-scale report -- no statsmodels dependency
(not installed in this venv), just numpy + scipy.stats for the p-value.

Fits y ~ 1 + log10(x) + log10(x)^2 + group_bin (group_bin=1 for the
"reference" group being tested against the baseline, e.g. LLM, 0 otherwise),
within a caller-supplied (already-windowed) subset. The group_bin
coefficient is the score shift attributable to group membership AFTER
controlling for the quadratic trend in log10(x) -- i.e. the "does the
reference group still win at matched x" test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class OlsResult:
    n: int
    coef: dict[str, float]
    se: dict[str, float]
    t: dict[str, float]
    p: dict[str, float]
    r2: float


def fit_quadratic_group_ols(log10x: np.ndarray, y: np.ndarray, group_bin: np.ndarray) -> OlsResult | None:
    """group_bin: 1.0 for the reference group, 0.0 for the comparison group."""
    n = len(y)
    names = ["intercept", "log10x", "log10x2", "group"]
    p = len(names)
    if n <= p + 1:
        return None
    X = np.column_stack([np.ones(n), log10x, log10x ** 2, group_bin])
    coefs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < p:
        return None
    resid = y - X @ coefs
    dof = n - p
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))
    t = coefs / np.where(se > 0, se, np.nan)
    pvals = 2 * stats.t.sf(np.abs(t), df=dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return OlsResult(
        n=n,
        coef=dict(zip(names, coefs.tolist())),
        se=dict(zip(names, se.tolist())),
        t=dict(zip(names, t.tolist())),
        p=dict(zip(names, pvals.tolist())),
        r2=r2,
    )
