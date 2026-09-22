"""Ridge regression with nested cross-validation for NeuroAI benchmarking.

Provides:
  * ``cv_ridge_score``                        — scalar metrics (Pearson r, Spearman r, R², MSE)
  * ``cv_ridge_score_windowed``               — time-resolved scores via sliding windows
  * ``cv_ridge_score_multioutput``            — group-k-fold multi-output ridge (CPU)
  * ``cv_ridge_score_multioutput_gpu``        — group-k-fold multi-output ridge (GPU, SVD-GCV)
  * ``cv_ridge_score_lag_matrix``             — lag-matrix TRF ridge (CPU, single channel)
  * ``cv_ridge_score_lag_matrix_gpu_batched`` — lag-matrix TRF ridge (GPU, all channels)
  * ``cv_ridge_predict``                      — out-of-fold predictions
  * ``CVScore``                               — scalar summary per electrode / layer
  * ``TimeResolvedScore``                     — temporal encoding profile (n_times,)
  * ``LayerwiseRidgeResults``                 — aggregates results across all transformer layers
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy.linalg import LinAlgWarning
from scipy.stats import pearsonr, spearmanr as _spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import RobustScaler

# Grid or scalar regularization strength (anything ``np.asarray`` accepts).
# ``Union[...]`` for Python 3.9 (PEP 604 not valid in runtime alias assignment).
AlphaInput = Union[float, Sequence[float], np.ndarray]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    alpha: float


@dataclass
class CVScore:
    """Scalar regression metrics averaged across outer CV folds."""
    pearsonr: float
    r2: float
    mse: float
    spearmanr: float = field(default=float("nan"))
    pearsonr_folds: list[float] = field(default_factory=list)
    spearmanr_folds: list[float] = field(default_factory=list)
    # Ridge penalty: mean of ``RidgeCV.alpha_`` across folds (fixed ``Ridge.alpha`` if scalar grid).
    ridge_alpha: float = field(default=float("nan"))
    ridge_alpha_folds: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {
            "pearsonr": self.pearsonr,
            "spearmanr": self.spearmanr,
            "r2": self.r2,
            "mse": self.mse,
            "ridge_alpha": self.ridge_alpha,
        }


@dataclass
class TimeResolvedScore:
    """Temporal encoding profile from sliding-window ridge regression.

    Attributes
    ----------
    window_centers : (n_windows,)
        Centre of each time window in seconds relative to event onset.
    pearsonr : (n_windows,)
        Median Pearson r across outer CV folds.
    spearmanr : (n_windows,)
        Median Spearman r across outer CV folds.
    r2 : (n_windows,)
        Median R² across outer CV folds.
    std : (n_windows,)
        Fold std of the primary metric (``metric`` arg in the regression call).
    ridge_alpha : (n_windows,) | None
        Median ``RidgeCV.alpha_`` across outer folds per window (NaN if unavailable).
    """
    window_centers: np.ndarray
    pearsonr: np.ndarray
    spearmanr: np.ndarray
    r2: np.ndarray
    std: np.ndarray
    ridge_alpha: np.ndarray | None = None
    # Out-of-fold predicted lag matrix (n_words, n_times), float32, NaN for any
    # word never selected into a test fold. Only populated when the caller
    # passes ``return_oof_predictions=True`` — used to save per-word predicted
    # responses (e.g. for post-hoc DLT-cost difficulty splits), not for scoring.
    oof_predictions: np.ndarray | None = None

    def peak_score(self, metric: str = "spearmanr") -> float:
        arr = getattr(self, metric)
        return float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else float("nan")

    def peak_time(self, metric: str = "spearmanr") -> float:
        arr = getattr(self, metric)
        if not np.any(np.isfinite(arr)):
            return float("nan")
        return float(self.window_centers[int(np.nanargmax(arr))])

    def to_summary_cvScore(self, metric: str = "spearmanr") -> CVScore:
        """Return a CVScore at the peak time window (for brain-map plots)."""
        arr_sp = self.spearmanr
        arr_pr = self.pearsonr
        arr_r2 = self.r2
        if not np.any(np.isfinite(arr_sp)):
            return CVScore(pearsonr=float("nan"), r2=float("nan"), mse=float("nan"))
        peak_idx = int(np.nanargmax(getattr(self, metric)))
        ra = float("nan")
        if self.ridge_alpha is not None and len(self.ridge_alpha) > peak_idx:
            ra = float(self.ridge_alpha[peak_idx])
        return CVScore(
            pearsonr=float(arr_pr[peak_idx]),
            spearmanr=float(arr_sp[peak_idx]),
            r2=float(arr_r2[peak_idx]),
            mse=float("nan"),
            ridge_alpha=ra,
        )

    def to_mean_cvScore(self) -> CVScore:
        """Return a CVScore with Pearson / Spearman / R² **mean** over windows.

        Matches Encoder-style summaries that average the time-resolved curve
        rather than taking the single best window.
        """
        if not np.any(np.isfinite(self.spearmanr)):
            return CVScore(
                pearsonr=float("nan"),
                r2=float("nan"),
                mse=float("nan"),
                spearmanr=float("nan"),
                ridge_alpha=float("nan"),
            )
        ra = float("nan")
        if self.ridge_alpha is not None and len(self.ridge_alpha):
            ra = float(np.nanmean(self.ridge_alpha))
        return CVScore(
            pearsonr=float(np.nanmean(self.pearsonr)),
            spearmanr=float(np.nanmean(self.spearmanr)),
            r2=float(np.nanmean(self.r2)),
            mse=float("nan"),
            ridge_alpha=ra,
        )


# ---------------------------------------------------------------------------
# Alpha grid → Ridge vs RidgeCV
# ---------------------------------------------------------------------------

def _alphas_arr(alpha: AlphaInput) -> np.ndarray:
    a = np.asarray(alpha, dtype=float).ravel()
    if a.size < 1:
        raise ValueError("alpha must specify at least one value")
    # RidgeCV expects strictly positive alphas; sklearn Ridge allows 0.0.
    return a


def _fit_ridge_for_fold(X_tr: np.ndarray, y_tr: np.ndarray, alphas_arr: np.ndarray,
                        inner_cv_splits: int = 5, fast_gcv: bool = False) -> Ridge | RidgeCV:
    # Suppress repetitive linear-algebra warnings from sklearn ridge internals
    # for ill-conditioned/singular systems; Ridge/RidgeCV already fall back to
    # least-squares and continue, so this is mostly log noise in large sweeps.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=LinAlgWarning,
            module=r"sklearn\.linear_model\._ridge",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Singular matrix in solving dual problem\..*",
            category=UserWarning,
            module=r"sklearn\.linear_model\._ridge",
        )
        if alphas_arr.size == 1:
            return Ridge(alpha=float(alphas_arr[0])).fit(X_tr, y_tr)
        # Use nested k-fold inner CV for more robust alpha selection (GCV can
        # be unreliable for small sample sizes and structured features).
        # Fall back to GCV (cv=None) when n_samples < inner_cv_splits * 2.
        # fast_gcv=True forces the cv=None (efficient leave-one-out GCV via a
        # single SVD, all alphas evaluated ~for free) path regardless of
        # sample size -- an explicit integer cv makes RidgeCV do a brute-force
        # refit for every (alpha, inner-fold) pair instead (n_alphas *
        # inner_cv_splits separate solves), which is what made a handful of
        # very-wide-hidden-dim checkpoints (opt-66b, hidden=9216, etc.) take
        # hours per layer in the ZuCo reading embed sweep. GCV is the
        # standard/preferred choice for exactly this regime (large n, large
        # p) -- the nested-CV "more robust for small samples" rationale above
        # doesn't apply here. Opt-in only (default False) so every other
        # caller of this shared function keeps its existing alpha-selection
        # behavior unchanged.
        if fast_gcv:
            return RidgeCV(alphas=alphas_arr, cv=None).fit(X_tr, y_tr)
        n_inner = len(X_tr)
        cv = inner_cv_splits if n_inner >= inner_cv_splits * 3 else None
        return RidgeCV(alphas=alphas_arr, cv=cv).fit(X_tr, y_tr)


def _ridge_chosen_alpha(model: Ridge | RidgeCV) -> float:
    """Scalar penalty chosen for this fit (``RidgeCV.alpha_`` or ``Ridge.alpha``)."""
    if isinstance(model, RidgeCV):
        a = model.alpha_
        arr = np.asarray(a, dtype=float).ravel()
        if arr.size == 0:
            return float("nan")
        return float(np.nanmedian(arr))
    return float(np.asarray(model.alpha, dtype=float).ravel()[0])


def _nanmedian_along_axis0(a: np.ndarray) -> np.ndarray:
    """Column-wise median of finite values; NaN if a column is all-NaN.

    Avoids ``RuntimeWarning: All-NaN slice encountered`` from ``np.nanmedian(..., axis=0)``.
    """
    out = np.empty(a.shape[1], dtype=np.float64)
    for j in range(a.shape[1]):
        col = a[:, j]
        m = np.isfinite(col)
        out[j] = float(np.median(col[m])) if m.any() else np.nan
    return out


def _nanstd_along_axis0(a: np.ndarray, *, ddof: int = 0) -> np.ndarray:
    """Column-wise std of finite values; NaN if too few points for ``ddof``.

    Avoids ``Degrees of freedom <= 0`` from ``np.nanstd(..., axis=0)``.
    """
    out = np.empty(a.shape[1], dtype=np.float64)
    for j in range(a.shape[1]):
        v = a[:, j][np.isfinite(a[:, j])]
        out[j] = float(np.std(v, ddof=ddof)) if v.size > ddof else np.nan
    return out


# ---------------------------------------------------------------------------
# Scalar regression (single time point / mean window)
# ---------------------------------------------------------------------------

def cv_ridge_score(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    alpha: AlphaInput = 1000.0,
    scale: bool = True,
    random_state: int = 0,
    metric: str = "spearmanr",
    print_chosen_alpha: bool = False,
    groups: np.ndarray | None = None,
    return_oof_predictions: bool = False,
    fast_gcv: bool = False,
) -> CVScore | tuple[CVScore, np.ndarray, np.ndarray]:
    """K-fold ridge regression with fixed ``alpha`` or ``RidgeCV`` over a grid.

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,)
    n_splits : int
        Outer CV folds.
    alpha : float or sequence of floats
        If a single value, uses ``Ridge``.
        If multiple values, fits ``RidgeCV(alphas=..., cv=inner_cv)``
        (k-fold or GCV) **within each outer training fold**, then
        evaluates on the held-out fold (same outer split as
        fixed-``alpha`` mode).
    scale : bool
        Apply ``RobustScaler`` to X within each fold.
    random_state : int
        Random seed for reproducibility.
    metric : str
        Primary summary metric — "spearmanr" or "pearsonr".
    print_chosen_alpha : bool
        If True, print mean and per-fold chosen regularization strength to stdout.
    groups : (n_samples,), optional
        Integer group index (e.g. story id) for each sample. When given, folds
        are built with ``GroupKFold`` so that no group straddles the train/test
        split (e.g. leave-N-stories-out), instead of a plain shuffled ``KFold``.
    return_oof_predictions : if True, ALSO return ``(oof, valid_mask)`` — an
        ``(n_valid_samples,)`` out-of-fold prediction array (NaN for any row
        whose fold was skipped) and the boolean mask (relative to the
        ORIGINAL, pre-filtering ``X``/``y``) selecting the finite-``y`` rows
        those predictions correspond to — needed for post-hoc per-word
        difficulty splits (e.g. DLT cost). Default False, so every existing
        caller's ``score = cv_ridge_score(...)`` call is unaffected; only
        pass ``True`` when you unpack the ``(score, oof, valid_mask)`` tuple.

    Returns
    -------
    CVScore with Spearman r, Pearson r, R², MSE averaged across folds.
    """
    valid_mask = np.isfinite(y)
    X, y = X[valid_mask], y[valid_mask]
    if groups is not None:
        groups = np.asarray(groups)[valid_mask]
    if len(y) < n_splits * 2:
        empty = CVScore(pearsonr=float("nan"), r2=float("nan"), mse=float("nan"),
                       spearmanr=float("nan"), ridge_alpha=float("nan"))
        if return_oof_predictions:
            return empty, np.full(len(y), np.nan, dtype=np.float64), valid_mask
        return empty

    oof: np.ndarray | None = (
        np.full(len(y), np.nan, dtype=np.float64) if return_oof_predictions else None
    )

    alphas_arr = _alphas_arr(alpha)
    pearsonr_folds: list[float] = []
    spearmanr_folds: list[float] = []
    r2_folds: list[float] = []
    mse_folds: list[float] = []
    alpha_folds: list[float] = []

    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))
        kf_splits = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        kf_splits = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X)

    for train_idx, test_idx in kf_splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if scale:
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        model = _fit_ridge_for_fold(X_tr, y_tr, alphas_arr, fast_gcv=fast_gcv)
        alpha_folds.append(_ridge_chosen_alpha(model))
        y_pred = model.predict(X_te)
        if oof is not None:
            oof[test_idx] = y_pred

        # Guard against near-constant inputs (avoids NearConstantInputWarning)
        if np.std(y_te) > 1e-8 and np.std(y_pred) > 1e-8:
            pearsonr_folds.append(float(pearsonr(y_te, y_pred).statistic))
            spearmanr_folds.append(float(_spearmanr(y_te, y_pred).statistic))
        else:
            pearsonr_folds.append(float("nan"))
            spearmanr_folds.append(float("nan"))

        ss_res = np.sum((y_te - y_pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        r2_folds.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        mse_folds.append(float(np.mean((y_te - y_pred) ** 2)))

    mean_alpha = float(np.nanmean(alpha_folds)) if alpha_folds else float("nan")
    score = CVScore(
        pearsonr=float(np.nanmean(pearsonr_folds)),
        spearmanr=float(np.nanmean(spearmanr_folds)),
        r2=float(np.nanmean(r2_folds)),
        mse=float(np.nanmean(mse_folds)),
        pearsonr_folds=pearsonr_folds,
        spearmanr_folds=spearmanr_folds,
        ridge_alpha=mean_alpha,
        ridge_alpha_folds=alpha_folds,
    )
    if print_chosen_alpha:
        finite_alphas = [a for a in alpha_folds if np.isfinite(a)]
        min_alpha = min(finite_alphas) if finite_alphas else float("nan")
        med_alpha = float(np.median(finite_alphas)) if finite_alphas else float("nan")
        max_alpha = max(finite_alphas) if finite_alphas else float("nan")
        print(
            f"[ridge] chosen alpha: min={min_alpha:.6g}  median={med_alpha:.6g}  "
            f"max={max_alpha:.6g}  mean={mean_alpha:.6g}  per_fold={alpha_folds}",
            flush=True,
        )
    if return_oof_predictions:
        return score, oof, valid_mask
    return score


# ---------------------------------------------------------------------------
# Time-resolved regression (sliding window, matching reference script)
# ---------------------------------------------------------------------------

def cv_ridge_score_windowed(
    X: np.ndarray,
    y_timeseries: np.ndarray,
    times: np.ndarray,
    *,
    window_size: float = 0.2,
    step_size: float = 0.05,
    n_splits: int = 5,
    alpha: AlphaInput = 1000.0,
    scale: bool = True,
    random_state: int = 0,
    metric: str = "spearmanr",
    print_chosen_alpha: bool = False,
) -> TimeResolvedScore:
    """Time-resolved encoding via sliding-window ridge regression.

    For each sliding window ``[t0, t0+window_size)``, the regression target is
    the mean neural activity within that window (as in the reference encoding script).

    Parameters
    ----------
    X : (n_trials, n_features)
        LLM embeddings for each trial.
    y_timeseries : (n_trials, n_times)
        Pre-processed iEEG data (high-gamma envelope, possibly baseline-corrected
        and IQR-clipped).
    times : (n_times,)
        Time axis in seconds relative to event onset.
    window_size : float
        Duration of each analysis window in seconds (default: 0.2 s).
    step_size : float
        Step between consecutive windows in seconds (default: 0.05 s).
    n_splits : int
        Number of outer cross-validation folds.
    alpha : float or sequence of floats
        Single value → ``Ridge``; multiple → ``RidgeCV`` on outer-train splits.
    scale : bool
        Apply ``RobustScaler`` to embeddings within each fold.
    random_state : int
        Reproducibility seed.
    metric : str
        Primary metric for ``std`` and peak-time computation.
    print_chosen_alpha : bool
        If True, print a summary of median chosen ``alpha`` across time windows.

    Returns
    -------
    TimeResolvedScore

    Notes
    -----
    **Where NaNs in ``pearsonr`` / ``spearmanr`` come from**

    1. **Non-finite window targets** — ``y_win = y_ts[:, time_mask].mean(axis=1)`` uses
       ordinary ``mean``; any NaN inside the window makes that trial’s ``y_win`` NaN,
       so the fold is skipped (no correlation). Missing iEEG samples or masking in
       ``y_timeseries`` therefore propagate.
    2. **Near-constant test targets or predictions** — correlations are filled only when
       ``std(y_te) > 1e-8`` and ``std(y_pred) > 1e-8`` (same guard as :func:`cv_ridge_score`).
       If the test fold’s mean activity is almost constant, or ridge predicts a near-flat
       vector, that fold’s r is left as NaN.
    3. **All folds NaN for a time window** — then the column-wise median is NaN for that
       window (expected; enable DEBUG logging on ``neuroai.regressors.ridge`` for counts).
    """
    # Drop trials with NaN in embeddings
    valid = np.isfinite(X).all(axis=-1) if X.ndim == 2 else np.ones(len(X), dtype=bool)
    X = X[valid]
    y_ts = y_timeseries[valid]

    win_starts = np.arange(times[0], times[-1] - window_size + step_size * 0.5, step_size)
    win_ends = win_starts + window_size
    win_centers = win_starts + window_size / 2.0
    n_windows = len(win_starts)

    alphas_arr = _alphas_arr(alpha)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_splits = list(kf.split(X))

    fold_pr = np.full((n_splits, n_windows), np.nan)
    fold_sr = np.full((n_splits, n_windows), np.nan)
    fold_r2 = np.full((n_splits, n_windows), np.nan)
    fold_alpha = np.full((n_splits, n_windows), np.nan)

    skip_nonfinite_y = 0
    skip_low_std_corr = 0
    fill_corr = 0

    # Scale X once per fold (not once per window) to avoid data leakage
    if scale:
        X_tr_scaled = {}
        X_te_scaled = {}
        for fi, (tr_idx, te_idx) in enumerate(outer_splits):
            sc = RobustScaler()
            X_tr_scaled[fi] = sc.fit_transform(X[tr_idx])
            X_te_scaled[fi] = sc.transform(X[te_idx])

    for wi, (t0, t1) in enumerate(zip(win_starts, win_ends)):
        time_mask = (times >= t0) & (times < t1)
        if time_mask.sum() == 0:
            continue
        y_win = y_ts[:, time_mask].mean(axis=1)   # (n_trials,)

        for fi, (tr_idx, te_idx) in enumerate(outer_splits):
            y_tr, y_te = y_win[tr_idx], y_win[te_idx]
            if not (np.isfinite(y_tr).all() and np.isfinite(y_te).all()):
                skip_nonfinite_y += 1
                continue

            Xtr = X_tr_scaled[fi] if scale else X[tr_idx]
            Xte = X_te_scaled[fi] if scale else X[te_idx]

            model = _fit_ridge_for_fold(Xtr, y_tr, alphas_arr)
            fold_alpha[fi, wi] = _ridge_chosen_alpha(model)
            y_pred = model.predict(Xte)

            if np.std(y_te) > 1e-8 and np.std(y_pred) > 1e-8:
                fold_pr[fi, wi] = float(pearsonr(y_te, y_pred).statistic)
                fold_sr[fi, wi] = float(_spearmanr(y_te, y_pred).statistic)
                fill_corr += 1
            else:
                skip_low_std_corr += 1

            ss_res = np.sum((y_te - y_pred) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            fold_r2[fi, wi] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    primary = fold_sr if metric == "spearmanr" else fold_pr
    if logger.isEnabledFor(logging.DEBUG):
        y_bad = float(np.mean(~np.isfinite(y_ts))) if y_ts.size else 0.0
        pr_per_win = np.isfinite(fold_pr).sum(axis=0)
        n_dead_win = int(np.sum(pr_per_win == 0))
        logger.debug(
            "cv_ridge_score_windowed: trials=%d n_windows=%d y_nonfinite_frac=%.5g | "
            "fold_skips: nonfinite_y=%d low_std_te_or_pred=%d corr_filled=%d | "
            "windows_with_all_nan_pearson=%d",
            len(X),
            n_windows,
            y_bad,
            skip_nonfinite_y,
            skip_low_std_corr,
            fill_corr,
            n_dead_win,
        )

    ridge_alpha_win = _nanmedian_along_axis0(fold_alpha)
    out = TimeResolvedScore(
        window_centers=win_centers,
        pearsonr=_nanmedian_along_axis0(fold_pr),
        spearmanr=_nanmedian_along_axis0(fold_sr),
        r2=_nanmedian_along_axis0(fold_r2),
        std=_nanstd_along_axis0(primary, ddof=0),
        ridge_alpha=ridge_alpha_win,
    )
    if print_chosen_alpha:
        finite_ra = ridge_alpha_win[np.isfinite(ridge_alpha_win)]
        med_all = float(np.median(finite_ra)) if finite_ra.size else float("nan")
        if finite_ra.size:
            ra_min, ra_max = float(finite_ra.min()), float(finite_ra.max())
        else:
            ra_min = ra_max = float("nan")
        print(
            f"[ridge windowed] chosen alpha: median over windows={med_all:.6g}  "
            f"min={ra_min:.6g} max={ra_max:.6g}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# Lag-matrix TRF ridge — group k-fold, multi-output per time bin
# ---------------------------------------------------------------------------

def _pearsonr_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between matching columns of A and B, shape (n, k) → (k,)."""
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)
    num = (A_c * B_c).sum(axis=0)
    std_A = np.sqrt((A_c ** 2).sum(axis=0))
    std_B = np.sqrt((B_c ** 2).sum(axis=0))
    valid = (std_A > 1e-8) & (std_B > 1e-8)
    denom = np.where(valid, std_A * std_B, 1.0)
    return np.where(valid, num / denom, np.nan)


def _spearmanr_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spearman r between matching columns, computed via rank-then-Pearson."""
    # argsort twice gives ordinal ranks (0-indexed); no special tie handling needed
    # for continuous EEG data where exact ties are vanishingly rare.
    A_r = np.argsort(np.argsort(A, axis=0), axis=0).astype(np.float32)
    B_r = np.argsort(np.argsort(B, axis=0), axis=0).astype(np.float32)
    return _pearsonr_columns(A_r, B_r)


def _r2_columns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """R² for each column."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)


def cv_ridge_score_lag_matrix(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    *,
    groups: np.ndarray,
    n_splits: int = 5,
    alpha: AlphaInput | None = None,
    scale: bool = True,
    standardize_target: bool = True,
    random_state: int = 0,
    device: str = "cpu",
    return_oof_predictions: bool = False,
) -> TimeResolvedScore:
    """Group-k-fold ridge regression predicting EEG at all time lags (CPU, single channel).

    Unlike ``cv_ridge_score_windowed``, the target ``y`` (n_words × T) is fit
    with one multi-output model per fold; ``groups`` (run indices) controls the
    fold assignment via ``GroupKFold``.

    Parameters
    ----------
    X : (n_words, n_features)
    y : (n_words, n_times) — EEG epoch at T time bins relative to word onset
    times : (n_times,) — time axis in seconds
    groups : (n_words,) — integer run / segment index for each word
    n_splits : outer CV folds (GroupKFold)
    alpha : ridge penalty grid (default: 5-value log-spaced [10^3.5 … 10^8])
    scale : apply RobustScaler to X within each fold
    standardize_target : z-score each time-bin column of y on the training fold
    device : ignored (kept for call-site compatibility with the GPU path)

    Returns
    -------
    TimeResolvedScore with per-time-bin Pearson r / Spearman r / R²
    """
    n_times = y.shape[1]
    if alpha is None:
        alpha = np.logspace(3.5, 8, 5)
    alphas_arr = _alphas_arr(alpha)

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=groups))

    fold_pr = np.full((n_splits, n_times), np.nan)
    fold_sr = np.full((n_splits, n_times), np.nan)
    fold_r2 = np.full((n_splits, n_times), np.nan)
    fold_alpha = np.full(n_splits, np.nan)
    oof_predictions = (
        np.full((X.shape[0], n_times), np.nan, dtype=np.float32)
        if return_oof_predictions else None
    )

    for fi, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if scale:
            sc = RobustScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        if standardize_target:
            y_std = y_tr.std(axis=0)
            y_std[y_std < 1e-8] = 1.0
            y_mean = y_tr.mean(axis=0)
            y_tr_fit = (y_tr - y_mean) / y_std
        else:
            y_tr_fit = y_tr
            y_mean = np.zeros(n_times, dtype=y_tr.dtype)
            y_std = np.ones(n_times, dtype=y_tr.dtype)

        model = _fit_ridge_for_fold(X_tr, y_tr_fit, alphas_arr)
        fold_alpha[fi] = _ridge_chosen_alpha(model)
        y_pred = model.predict(X_te)

        if standardize_target:
            y_pred = y_pred * y_std + y_mean

        fold_pr[fi] = _pearsonr_columns(y_te, y_pred)
        fold_sr[fi] = _spearmanr_columns(y_te, y_pred)
        fold_r2[fi] = _r2_columns(y_te, y_pred)
        if oof_predictions is not None:
            oof_predictions[te_idx] = y_pred.astype(np.float32)

    alpha_med = float(np.nanmedian(fold_alpha))
    return TimeResolvedScore(
        window_centers=np.asarray(times, dtype=float),
        pearsonr=_nanmedian_along_axis0(fold_pr),
        spearmanr=_nanmedian_along_axis0(fold_sr),
        r2=_nanmedian_along_axis0(fold_r2),
        std=_nanstd_along_axis0(fold_pr, ddof=0),
        ridge_alpha=np.full(n_times, alpha_med),
        oof_predictions=oof_predictions,
    )


def cv_ridge_score_lag_matrix_gpu_batched(
    X: np.ndarray,
    eeg: np.ndarray,
    times: np.ndarray,
    *,
    groups: np.ndarray,
    n_splits: int = 5,
    alpha: AlphaInput | None = None,
    scale: bool = True,
    standardize_target: bool = True,
    random_state: int = 0,
    return_oof_predictions: bool = False,
) -> list[TimeResolvedScore]:
    """GPU-batched lag-matrix ridge for all channels simultaneously.

    Fits all C channels and T time bins in one multi-output GPU solve per fold.
    Alpha is selected via GCV (SVD-based, computed once per fold in O(n·D) ops).

    Parameters
    ----------
    X : (n_words, n_features)
    eeg : (n_words, n_channels, n_times)
    times : (n_times,)
    groups : (n_words,) — run indices for GroupKFold
    n_splits, alpha, scale, standardize_target, random_state — same as CPU version

    Returns
    -------
    list of TimeResolvedScore, one per channel
    """
    import torch
    from sklearn.model_selection import GroupKFold as _GKF

    n, C, T = eeg.shape
    if alpha is None:
        alpha = np.logspace(3.5, 8, 5)
    alphas_arr = _alphas_arr(alpha)

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    gkf = _GKF(n_splits=n_splits)
    splits = list(gkf.split(X, groups=groups))

    fold_pr = np.full((n_splits, C, T), np.nan)
    fold_sr = np.full((n_splits, C, T), np.nan)
    fold_r2 = np.full((n_splits, C, T), np.nan)
    fold_alpha = np.full(n_splits, np.nan)
    oof_predictions = (
        np.full((n, C, T), np.nan, dtype=np.float32) if return_oof_predictions else None
    )

    Y_flat = eeg.reshape(n, C * T).astype(np.float32)

    for fi, (tr_idx, te_idx) in enumerate(splits):
        X_tr = X[tr_idx].astype(np.float32)
        X_te = X[te_idx].astype(np.float32)
        Y_tr = Y_flat[tr_idx]
        Y_te = Y_flat[te_idx]

        if scale:
            sc = RobustScaler()
            X_tr = sc.fit_transform(X_tr).astype(np.float32)
            X_te = sc.transform(X_te).astype(np.float32)

        # A non-finite feature (e.g. a degenerate predictor, or — for
        # logit-lens surprisal — an early layer whose activations overflow
        # under the final unembed) makes every alpha's GCV score NaN, so
        # `best_W` in the alpha-search loop below never gets assigned and the
        # subsequent `Xte_t @ best_W` crashes with a TypeError. Skip the fold
        # instead (leaves NaN for this fold; other folds/channels still
        # score normally), matching the guard in cv_ridge_score_multioutput.
        if not (np.isfinite(X_tr).all() and np.isfinite(X_te).all()):
            logger.warning(
                "cv_ridge_score_lag_matrix_gpu_batched: fold %d/%d has "
                "non-finite features after scaling — skipping this fold.",
                fi + 1, len(splits),
            )
            continue

        if standardize_target:
            y_mean = Y_tr.mean(axis=0)
            y_std = Y_tr.std(axis=0)
            y_std[y_std < 1e-8] = 1.0
            Y_tr_fit = ((Y_tr - y_mean) / y_std).astype(np.float32)
        else:
            Y_tr_fit = Y_tr
            y_mean = np.zeros(C * T, dtype=np.float32)
            y_std = np.ones(C * T, dtype=np.float32)

        Xt = torch.from_numpy(X_tr).to(dev)
        Yt = torch.from_numpy(Y_tr_fit).to(dev)

        # Eigh-based GCV: eigendecompose X^TX (D×D) instead of SVD of X (n_tr×D).
        # This avoids storing U (n_tr × D) — saves ~1 GB/worker for D≥4096 (large models).
        # Peak GPU memory: Xt + V(D×D) + VtXtY + Yt + HY vs Xt + U(n_tr×D) + Vt + Yt + HY.
        n_tr = Xt.shape[0]
        XtX = Xt.T @ Xt           # (D, D)
        XtY = Xt.T @ Yt           # (D, C*T)
        eigvals, V = torch.linalg.eigh(XtX)  # ascending order; eigvals=s², V: (D, D)
        eigvals = eigvals.clamp(min=0.0)      # numerical safety for near-zero eigenvalues
        del XtX
        VtXtY = V.T @ XtY         # (D, C*T)
        del XtY

        best_gcv = float("inf")
        best_alpha_val = float(alphas_arr[0])
        best_W: torch.Tensor | None = None

        for a_val in alphas_arr.tolist():
            a = torch.tensor(a_val, dtype=Xt.dtype, device=dev)
            d = 1.0 / (eigvals + a)                       # (D,)
            W_a = V @ (d.unsqueeze(-1) * VtXtY)           # (D, C*T)
            HY = Xt @ W_a                                  # (n_tr, C*T)
            res = Yt - HY
            trace_H = (eigvals / (eigvals + a)).sum()
            denom = (1.0 - trace_H / n_tr).clamp(min=1e-8) ** 2
            # nanmean, not mean: this scalar is shared across all C*T target
            # columns for alpha selection (the efficient batched-GCV trick),
            # so even one non-finite value in a single (channel, lag) column
            # of Yt — e.g. a logratio-baseline edge case — would otherwise
            # poison gcv to NaN for every alpha, leaving best_W unassigned
            # and crashing the later `Xte_t @ best_W`. nanmean lets the
            # majority of valid columns still pick a sensible alpha; the
            # handful of bad columns simply keep scoring NaN downstream
            # (already handled by the nanmedian aggregation below).
            gcv = float(torch.nanmean(res ** 2) / denom)
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha_val = a_val
                best_W = W_a

        fold_alpha[fi] = best_alpha_val

        Xte_t = torch.from_numpy(X_te).to(dev)
        Y_pred_t = Xte_t @ best_W          # (n_te, C*T)
        Y_pred = Y_pred_t.cpu().numpy().astype(np.float64)

        # Release GPU tensors before numpy work
        del Xt, Yt, eigvals, V, VtXtY, HY, best_W, Y_pred_t, Xte_t, res
        torch.cuda.empty_cache()

        if standardize_target:
            Y_pred = Y_pred * y_std.astype(np.float64) + y_mean.astype(np.float64)

        # Reshape to (n_te, C, T) for per-channel metrics
        Y_pred_ct = Y_pred.reshape(len(te_idx), C, T)
        Y_te_ct = Y_te.astype(np.float64).reshape(len(te_idx), C, T)

        for ci in range(C):
            yt_c = Y_te_ct[:, ci, :]    # (n_te, T)
            yp_c = Y_pred_ct[:, ci, :]  # (n_te, T)
            fold_pr[fi, ci] = _pearsonr_columns(yt_c, yp_c)
            fold_sr[fi, ci] = _spearmanr_columns(yt_c, yp_c)
            fold_r2[fi, ci] = _r2_columns(yt_c, yp_c)

        if oof_predictions is not None:
            oof_predictions[te_idx] = Y_pred_ct.astype(np.float32)

    alpha_med = float(np.nanmedian(fold_alpha))
    alpha_arr_t = np.full(T, alpha_med)

    results: list[TimeResolvedScore] = []
    for ci in range(C):
        results.append(TimeResolvedScore(
            window_centers=np.asarray(times, dtype=float),
            pearsonr=np.nanmedian(fold_pr[:, ci, :], axis=0),
            spearmanr=np.nanmedian(fold_sr[:, ci, :], axis=0),
            r2=np.nanmedian(fold_r2[:, ci, :], axis=0),
            std=np.nanstd(fold_pr[:, ci, :], axis=0),
            ridge_alpha=alpha_arr_t.copy(),
            oof_predictions=oof_predictions[:, ci, :] if oof_predictions is not None else None,
        ))
    return results


# ---------------------------------------------------------------------------
# Multi-output scalar regression (one CVScore per target column)
# ---------------------------------------------------------------------------

def cv_ridge_score_multioutput(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    groups: np.ndarray,
    n_splits: int = 5,
    alpha: AlphaInput | None = None,
    scale: bool = True,
    standardize_target: bool = True,
    random_state: int = 0,
    return_oof_predictions: bool = False,
) -> list[CVScore] | tuple[list[CVScore], np.ndarray]:
    """Group-k-fold multi-output ridge regression (CPU).

    Returns one :class:`CVScore` per target column in ``Y``.  All targets are
    solved in a single ``RidgeCV`` call per fold so the X gram matrix is
    factorised only once per (fold, alpha) instead of once per target.

    Parameters
    ----------
    X : (n_samples, n_features)
    Y : (n_samples, n_targets)
    groups : (n_samples,) integer group index for ``GroupKFold``
    n_splits : outer GroupKFold folds
    alpha : ridge penalty grid (default: 6-value log-spaced [10⁻² … 10⁴])
    scale : apply ``RobustScaler`` to X within each fold
    standardize_target : z-score each Y column on the training fold
    return_oof_predictions : if True, ALSO return an ``(n_samples, n_targets)``
        out-of-fold prediction array (NaN for any row whose fold was skipped)
        — needed for post-hoc per-word difficulty splits (e.g. DLT cost).
        Default False, so every existing caller's ``scores = cv_ridge_score_
        multioutput(...)`` call is unaffected; only pass ``True`` when you
        unpack the ``(scores, oof)`` tuple.
    """
    if alpha is None:
        alpha = np.logspace(-2, 4, 6)
    alphas_arr = _alphas_arr(alpha)
    n_targets = Y.shape[1]

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, groups=groups))

    fold_pr = np.full((len(splits), n_targets), np.nan)
    fold_sr = np.full((len(splits), n_targets), np.nan)
    fold_alpha = np.full(len(splits), np.nan)
    oof: np.ndarray | None = (
        np.full((X.shape[0], n_targets), np.nan, dtype=np.float64)
        if return_oof_predictions else None
    )

    for fi, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx].astype(np.float64), Y[te_idx].astype(np.float64)

        if scale:
            sc = RobustScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        if standardize_target:
            y_mean = Y_tr.mean(axis=0)
            y_std = Y_tr.std(axis=0)
            y_std[y_std < 1e-8] = 1.0
            Y_tr_fit = (Y_tr - y_mean) / y_std
        else:
            Y_tr_fit = Y_tr
            y_mean = np.zeros(n_targets, dtype=np.float64)
            y_std = np.ones(n_targets, dtype=np.float64)

        # See the matching guard in cv_ridge_score_multioutput_gpu: a NaN/Inf
        # fold is a data problem (e.g. a custom LM emitting non-finite hidden
        # states), not something a solver retry can fix — skip it rather than
        # raising and losing every target's score for this checkpoint.
        if not (np.isfinite(X_tr).all() and np.isfinite(X_te).all()):
            logger.warning(
                "cv_ridge_score_multioutput: fold %d/%d has non-finite "
                "features after scaling — skipping this fold.",
                fi + 1, len(splits),
            )
            continue

        model = _fit_ridge_for_fold(X_tr, Y_tr_fit, alphas_arr)
        fold_alpha[fi] = _ridge_chosen_alpha(model)
        Y_pred = np.asarray(model.predict(X_te), dtype=np.float64)

        if standardize_target:
            Y_pred = Y_pred * y_std + y_mean

        fold_pr[fi] = _pearsonr_columns(Y_te, Y_pred)
        fold_sr[fi] = _spearmanr_columns(Y_te, Y_pred)
        if oof is not None:
            oof[te_idx] = Y_pred

    alpha_med = float(np.nanmedian(fold_alpha))
    results: list[CVScore] = []
    for j in range(n_targets):
        results.append(CVScore(
            pearsonr=float(np.nanmedian(fold_pr[:, j])),
            spearmanr=float(np.nanmedian(fold_sr[:, j])),
            r2=float("nan"),
            mse=float("nan"),
            ridge_alpha=alpha_med,
        ))
    if return_oof_predictions:
        return results, oof
    return results


def cv_ridge_score_multioutput_gpu(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    groups: np.ndarray,
    n_splits: int = 5,
    alpha: AlphaInput | None = None,
    scale: bool = True,
    standardize_target: bool = True,
    random_state: int = 0,
    return_oof_predictions: bool = False,
) -> list[CVScore] | tuple[list[CVScore], np.ndarray]:
    """Group-k-fold multi-output ridge regression (GPU, SVD-based GCV).

    Identical contract to :func:`cv_ridge_score_multioutput` but runs on the
    current CUDA device via ``torch``.  Alpha is selected per fold with
    SVD-based GCV: the SVD of X_train is computed once and all alpha candidates
    are evaluated in O(k) ops each, amortised across **all** n_targets columns.

    Call from within a ``torch.cuda.stream()`` context for layer-parallel
    sweeps (see the Podcast/Nieuwland encoders).

    The ``himalaya`` ``torch_cuda`` backend must be activated in the calling
    process before the first call (``set_backend("torch_cuda")``).

    Pass ``return_oof_predictions=True`` to additionally return an
    ``(n_samples, n_targets)`` out-of-fold prediction array — same opt-in,
    backward-compatible contract as :func:`cv_ridge_score_multioutput`.
    """
    import torch
    from sklearn.model_selection import GroupKFold as _GKF

    if alpha is None:
        alpha = np.logspace(-2, 4, 6)
    alphas_arr = _alphas_arr(alpha)
    n_targets = Y.shape[1]

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    gkf = _GKF(n_splits=n_splits)
    splits = list(gkf.split(X, groups=groups))

    fold_pr = np.full((len(splits), n_targets), np.nan)
    fold_sr = np.full((len(splits), n_targets), np.nan)
    fold_alpha = np.full(len(splits), np.nan)
    oof: np.ndarray | None = (
        np.full((X.shape[0], n_targets), np.nan, dtype=np.float64)
        if return_oof_predictions else None
    )

    for fi, (tr_idx, te_idx) in enumerate(splits):
        X_tr = X[tr_idx].astype(np.float32)
        X_te = X[te_idx].astype(np.float32)
        Y_tr = Y[tr_idx].astype(np.float32)
        Y_te = Y[te_idx].astype(np.float64)

        if scale:
            sc = RobustScaler()
            X_tr = sc.fit_transform(X_tr).astype(np.float32)
            X_te = sc.transform(X_te).astype(np.float32)

        if standardize_target:
            y_mean = Y_tr.mean(axis=0)
            y_std = Y_tr.std(axis=0)
            y_std[y_std < 1e-8] = 1.0
            Y_tr_fit = ((Y_tr - y_mean) / y_std).astype(np.float32)
        else:
            Y_tr_fit = Y_tr
            y_mean = np.zeros(n_targets, dtype=np.float32)
            y_std = np.ones(n_targets, dtype=np.float32)

        Xt = torch.from_numpy(X_tr).to(dev)
        Yt = torch.from_numpy(Y_tr_fit).to(dev)

        # A NaN/Inf-valued fold (e.g. a small custom LM emits non-finite
        # hidden states for some layer/checkpoint, or scaling degenerates a
        # constant feature) makes SVD unrecoverable on both GPU and CPU — it's
        # a data problem, not a numerical-precision one. Skip the fold rather
        # than crashing the whole checkpoint; fold_pr/fold_sr/fold_alpha are
        # NaN-initialized and aggregated via nanmedian, so this degrades
        # gracefully to "fewer folds" instead of losing every layer's score.
        if not (torch.isfinite(Xt).all() and np.isfinite(X_te).all()):
            logger.warning(
                "cv_ridge_score_multioutput_gpu: fold %d/%d has non-finite "
                "features after scaling — skipping this fold.",
                fi + 1, len(splits),
            )
            del Xt, Yt
            torch.cuda.empty_cache()
            continue

        # SVD-based GCV: compute once per fold, evaluate all alphas in O(k).
        try:
            U, S, Vt = torch.linalg.svd(Xt, full_matrices=False)
        except RuntimeError as e:
            # Two distinct GPU SVD failure modes land here, both worth a CPU
            # retry rather than losing the whole layer's score:
            #  - torch._C._LinAlgError: cuSOLVER fails to numerically converge
            #    on an ill-conditioned fold (e.g. highly collinear features).
            #  - plain RuntimeError "cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR":
            #    a driver-level failure (not a convergence issue) empirically
            #    reproducible when many CUDA streams call torch.linalg.svd
            #    concurrently (layer-parallel fitting) on large hidden dims —
            #    unrelated RuntimeErrors (e.g. OOM) are re-raised, not retried.
            if not (isinstance(e, torch._C._LinAlgError) or "cusolver" in str(e).lower()):
                raise
            logger.warning(
                "cv_ridge_score_multioutput_gpu: GPU SVD failed (fold %d/%d): %s "
                "— retrying on CPU",
                fi + 1, len(splits), e,
            )
            U, S, Vt = torch.linalg.svd(Xt.cpu(), full_matrices=False)
            U, S, Vt = U.to(dev), S.to(dev), Vt.to(dev)
        UtY = U.T @ Yt   # (k, n_targets)

        best_gcv = float("inf")
        best_alpha_val = float(alphas_arr[0])
        best_W: torch.Tensor | None = None

        for a_val in alphas_arr.tolist():
            a = torch.tensor(a_val, dtype=Xt.dtype, device=dev)
            d = S / (S ** 2 + a)
            W_a = Vt.T @ (d.unsqueeze(-1) * UtY)    # (D, n_targets)
            hat_s = S ** 2 / (S ** 2 + a)
            HY = U @ (hat_s.unsqueeze(-1) * UtY)     # (n_tr, n_targets)
            res = Yt - HY
            trace_H = hat_s.sum()
            n_tr = Xt.shape[0]
            denom = (1.0 - trace_H / n_tr).clamp(min=1e-8) ** 2
            gcv = float((res ** 2).mean() / denom)
            if np.isfinite(gcv) and gcv < best_gcv:
                best_gcv = gcv
                best_alpha_val = a_val
                best_W = W_a

        # Fallback: if all alphas produced NaN/inf GCV (e.g. Y contains NaN),
        # use the largest alpha (maximum regularization) as a safe default.
        if best_W is None:
            a_fb = torch.tensor(float(alphas_arr[-1]), dtype=Xt.dtype, device=dev)
            d_fb = S / (S ** 2 + a_fb)
            best_W = Vt.T @ (d_fb.unsqueeze(-1) * UtY)
            best_alpha_val = float(alphas_arr[-1])

        fold_alpha[fi] = best_alpha_val

        Xte_t = torch.from_numpy(X_te).to(dev)
        Y_pred_t = Xte_t @ best_W               # (n_te, n_targets)
        Y_pred = Y_pred_t.cpu().numpy().astype(np.float64)

        del Xt, Yt, U, S, Vt, UtY, HY, best_W, Y_pred_t, Xte_t, res
        torch.cuda.empty_cache()

        if standardize_target:
            Y_pred = Y_pred * y_std.astype(np.float64) + y_mean.astype(np.float64)

        fold_pr[fi] = _pearsonr_columns(Y_te, Y_pred)
        fold_sr[fi] = _spearmanr_columns(Y_te, Y_pred)
        if oof is not None:
            oof[te_idx] = Y_pred

    alpha_med = float(np.nanmedian(fold_alpha))
    results: list[CVScore] = []
    for j in range(n_targets):
        results.append(CVScore(
            pearsonr=float(np.nanmedian(fold_pr[:, j])),
            spearmanr=float(np.nanmedian(fold_sr[:, j])),
            r2=float("nan"),
            mse=float("nan"),
            ridge_alpha=alpha_med,
        ))
    if return_oof_predictions:
        return results, oof
    return results


def lpp_loso_multioutput_ridge_gpu(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    ridge_alphas: np.ndarray,
    return_predictions: bool = False,
) -> tuple[float, np.ndarray, np.ndarray] | tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """LOSO multi-output ridge on GPU for pre-zscored LPP design matrices.

    Alpha is chosen by nested validation (``x_tr``/``y_tr`` vs ``x_val``/``y_val``),
    then the model is refit on ``x_train_full``/``y_train_full`` (all training runs)
    and evaluated on ``x_test``.  No intercept; no additional scaling.

    Returns ``(best_alpha, pearson_r_per_voxel, spearman_r_per_voxel)``, plus the
    held-out ``(n_test_TRs, n_voxels)`` prediction array as a 4th element when
    ``return_predictions=True`` — used for post-hoc per-TR difficulty splits
    (e.g. DLT cost), same opt-in contract as ``cv_ridge_score_multioutput``.
    """
    import torch

    alphas = np.asarray(ridge_alphas, dtype=np.float64)
    default_alpha = float(alphas[len(alphas) // 2])
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    def _fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_te: np.ndarray, alpha: float) -> np.ndarray:
        xt = torch.from_numpy(x_fit.astype(np.float32)).to(dev)
        yt = torch.from_numpy(y_fit.astype(np.float32)).to(dev)
        u, s, vt = torch.linalg.svd(xt, full_matrices=False)
        uty = u.T @ yt
        a = torch.tensor(float(alpha), dtype=xt.dtype, device=dev)
        d = s / (s ** 2 + a)
        w = vt.T @ (d.unsqueeze(-1) * uty)
        xte = torch.from_numpy(x_te.astype(np.float32)).to(dev)
        pred = (xte @ w).cpu().numpy().astype(np.float64)
        del xt, yt, u, s, vt, uty, w, xte
        return pred

    best_alpha = default_alpha
    if x_val is not None and y_val is not None and x_val.shape[0] > 0 and x_tr.shape[0] > 0:
        best_score = -np.inf
        for alpha in alphas:
            pred_val = _fit_predict(x_tr, y_tr, x_val, float(alpha))
            sc = float(np.nanmean(_pearsonr_columns(y_val, pred_val)))
            if np.isfinite(sc) and sc > best_score:
                best_score = sc
                best_alpha = float(alpha)

    pred_te = _fit_predict(x_train_full, y_train_full, x_test, best_alpha)
    corrs = _pearsonr_columns(y_test, pred_te)
    corrs_sp = _spearmanr_columns(y_test, pred_te)
    if return_predictions:
        result = best_alpha, corrs, corrs_sp, pred_te
    else:
        result = best_alpha, corrs, corrs_sp
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Out-of-fold predictions
# ---------------------------------------------------------------------------

def cv_ridge_predict(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    alpha: AlphaInput = 1000.0,
    scale: bool = True,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """K-fold ridge regression returning out-of-fold predictions.

    Returns
    -------
    y_true : (n_valid_samples,) — ground-truth y (NaN rows dropped)
    y_pred : (n_valid_samples,) — out-of-fold predictions
    """
    valid_mask = np.isfinite(y)
    X, y = X[valid_mask], y[valid_mask]
    n = len(y)
    y_pred_all = np.full(n, np.nan)
    alphas_arr = _alphas_arr(alpha)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        if scale:
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
        model = _fit_ridge_for_fold(X_tr, y_tr, alphas_arr)
        y_pred_all[test_idx] = model.predict(X_te)

    return y, y_pred_all


# ---------------------------------------------------------------------------
# Layerwise container
# ---------------------------------------------------------------------------

@dataclass
class LayerwiseRidgeResults:
    """Container for layerwise regression results across one model.

    Attributes
    ----------
    model_name : str
    layer_scores : list of CVScore, one per layer
    layer_indices : list of int
    """

    model_name: str
    layer_scores: list[CVScore]
    layer_indices: list[int]

    def best_layer_score(self, metric: str = "spearmanr") -> CVScore:
        vals = [getattr(s, metric) for s in self.layer_scores]
        best_idx = int(np.nanargmax(vals))
        return self.layer_scores[best_idx]

    def scores_array(self, metric: str = "spearmanr") -> np.ndarray:
        return np.array([getattr(s, metric) for s in self.layer_scores])
