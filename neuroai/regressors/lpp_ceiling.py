"""Split-half delta-log-likelihood noise ceiling (Timkey et al. 2026,
adapted) for the LPP fMRI pooled LME, split by SUBJECT -- unlike the real
production formula (``lpp_pooled_lme.py``), which has no subject term at all
(average-subject BOLD only; see that module's docstring). ``subject`` is
added here as an intercept-only crossed random effect alongside the real
formula's ``region`` (random slope on surprisal_z) and ``section``
(intercept-only) terms -- kept intercept-only rather than adding a subject
random slope, matching this codebase's convention of not stacking two
still-untested random slopes in one ceiling formula (Podcast's
subject:electrode ceiling made the same choice for `subject`).

See ``neuroai/regressors/lpp_persubject.py`` for how the required per-subject
``bold`` column is built (the real pipeline has none), and its module
docstring for the item-key convention.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fmri_lme import split_train_test_chunks
from .lme import _fit_stats, _partition_loglik_detailed

logger = logging.getLogger(__name__)


def _lpp_ceiling_formulas() -> tuple[str, str]:
    nuisances = "wlen_z + freq_z + sentpos_z"
    baseline = f"bold ~ {nuisances} + (1 | region) + (1 | section) + (1 | subject)"
    full = f"bold ~ {nuisances} + surprisal_z + (1 + surprisal_z | region) + (1 | section) + (1 | subject)"
    return baseline, full


def _lpp_ceiling_direction_task(
    df: pd.DataFrame, source_ids: set, target_ids: set, item_col: str,
) -> pd.DataFrame | None:
    src = df[df["subject"].astype(str).isin(source_ids)]
    tgt = df[df["subject"].astype(str).isin(target_ids)].copy()
    item_mean = src.groupby(item_col)["bold"].mean().rename("surprisal").reset_index()
    tgt = tgt.drop(columns=["surprisal"], errors="ignore").merge(item_mean, on=item_col, how="left")
    tgt = tgt.dropna(subset=["surprisal"])
    return tgt if not tgt.empty else None


def _prepare_lpp_ceiling_frame(df: pd.DataFrame):
    from .lme import zscore_series
    import polars as pl

    out = df.copy()
    out["region"] = out["region"].astype(str)
    out["section"] = out["section"].astype(str)
    out["subject"] = out["subject"].astype(str)
    out["chunk"] = out["section"]  # reuse split_train_test_chunks as-is (no chunk column in raw data)
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["sentpos_z"] = zscore_series(out["sentpos"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["bold"] = out["bold"].astype(float)
    cols = ["subject", "region", "section", "chunk", "wlen_z", "freq_z", "sentpos_z", "surprisal_z", "bold"]
    return pl.from_pandas(out[cols].copy())


def _lpp_ceiling_direction_worker(
    iteration: int,
    direction: int,
    tgt: pd.DataFrame,
    *,
    out_of_sample: bool,
    train_frac: float,
    random_state: int,
    verbose: bool,
) -> tuple[int, int, float | None, int, bool]:
    """Process-pool worker: fit one (iteration, direction)'s LPP ceiling model."""
    from pymer4.models import lmer

    try:
        data = _prepare_lpp_ceiling_frame(tgt)

        if out_of_sample:
            train_data, test_data = split_train_test_chunks(
                data, train_frac=train_frac, random_state=random_state,
            )
        else:
            train_data = test_data = data
        if test_data.height < 10:
            return iteration, direction, None, 0, False

        baseline_formula, full_formula = _lpp_ceiling_formulas()

        baseline_model = lmer(baseline_formula, data=train_data)
        baseline_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, base_ok = _fit_stats(baseline_model)

        full_model = lmer(full_formula, data=train_data)
        full_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, full_ok = _fit_stats(full_model)

        base_ll, _ = _partition_loglik_detailed(baseline_model, test_data, outcome="bold")
        full_ll, _ = _partition_loglik_detailed(full_model, test_data, outcome="bold")
    except Exception as exc:
        logger.warning("LPP ceiling fit failed on split %d direction %d: %s", iteration, direction, exc)
        return iteration, direction, None, 0, False

    converged = bool(base_ok and full_ok)
    delta = full_ll - base_ll if np.isfinite(base_ll) and np.isfinite(full_ll) else None
    ok = delta is not None and np.isfinite(delta) and converged
    return iteration, direction, (delta if ok else None), int(test_data.height), converged


@dataclass
class LppDeltaLoglikCeilingResult:
    ceiling_delta_loglik_per_obs: float
    ceiling_delta_loglik_per_obs_sd: float
    n_iterations_used: int
    n_iterations_requested: int
    per_iteration_delta_loglik: list[float] = field(default_factory=list)
    per_iteration_n_obs: list[int] = field(default_factory=list)
    per_direction_records: list[dict] = field(default_factory=list)


def compute_lpp_ceiling_delta_loglik(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    n_splits: int = 15,
    train_frac: float = 0.75,
    seed: int = 0,
    verbose: bool = False,
    n_workers: int = 1,
    out_of_sample: bool = True,
) -> LppDeltaLoglikCeilingResult:
    """Split-half noise ceiling for the LPP pooled LME's delta log-likelihood,
    split by subject. ``df`` must have one row per (subject, region, section,
    TR) with columns ``subject, region, section, wlen, freq, sentpos, bold``
    plus ``item_col`` (default ``"item_id"`` = ``f"{section}_{tr_idx}"``)."""
    required = {"subject", "region", "section", "wlen", "freq", "sentpos", "bold", item_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for LPP delta-LL ceiling: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(df["subject"].astype(str).unique()))

    tasks: list[tuple[int, int, pd.DataFrame]] = []
    for it in range(n_splits):
        perm = rng.permutation(subjects)
        half = len(perm) // 2
        halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
        for direction, (source_ids, target_ids) in enumerate((halves, halves[::-1])):
            tgt = _lpp_ceiling_direction_task(df, source_ids, target_ids, item_col)
            if tgt is not None:
                tasks.append((it, direction, tgt))

    iter_delta: dict[int, float] = {}
    iter_nobs: dict[int, int] = {}
    iter_failed: set[int] = set()
    per_direction_records: list[dict] = []

    def _record(iteration: int, direction: int, delta: float | None, n_obs: int, converged: bool) -> None:
        per_direction_records.append({
            "iteration": iteration, "direction": direction,
            "delta_loglik": delta, "n_obs": n_obs, "converged": converged,
        })
        if iteration in iter_failed:
            return
        if delta is None:
            iter_failed.add(iteration)
            iter_delta.pop(iteration, None)
            iter_nobs.pop(iteration, None)
            return
        iter_delta[iteration] = iter_delta.get(iteration, 0.0) + delta
        iter_nobs[iteration] = iter_nobs.get(iteration, 0) + n_obs

    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _lpp_ceiling_direction_worker, it, direction, tgt,
                    out_of_sample=out_of_sample, train_frac=train_frac,
                    random_state=seed, verbose=verbose,
                ): (it, direction)
                for it, direction, tgt in tasks
            }
            for fut in as_completed(futures):
                iteration, direction, delta, n_obs, converged = fut.result()
                _record(iteration, direction, delta, n_obs, converged)
    else:
        for it, direction, tgt in tasks:
            iteration, direction, delta, n_obs, converged = _lpp_ceiling_direction_worker(
                it, direction, tgt, out_of_sample=out_of_sample,
                train_frac=train_frac, random_state=seed, verbose=verbose,
            )
            _record(iteration, direction, delta, n_obs, converged)

    per_direction_records.sort(key=lambda r: (r["iteration"], r["direction"]))
    per_iter_delta = [iter_delta[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]
    per_iter_nobs = [iter_nobs[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]

    if not per_iter_delta:
        return LppDeltaLoglikCeilingResult(
            ceiling_delta_loglik_per_obs=float("nan"),
            ceiling_delta_loglik_per_obs_sd=float("nan"),
            n_iterations_used=0,
            n_iterations_requested=n_splits,
            per_direction_records=per_direction_records,
        )

    per_obs = [d / n for d, n in zip(per_iter_delta, per_iter_nobs)]
    return LppDeltaLoglikCeilingResult(
        ceiling_delta_loglik_per_obs=float(np.mean(per_obs)),
        ceiling_delta_loglik_per_obs_sd=float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan"),
        n_iterations_used=len(per_obs),
        n_iterations_requested=n_splits,
        per_iteration_delta_loglik=per_iter_delta,
        per_iteration_n_obs=per_iter_nobs,
        per_direction_records=per_direction_records,
    )
