"""Split-half delta-log-likelihood noise ceiling (Timkey et al. 2026,
adapted) for the two nested subject:electrode ECoG/iEEG LME datasets --
Fedorenko2016 (5 subjects) and Podcast (9 subjects). See
neuroai.regressors.n400_pooled_lme's module docstring for the general
split-half method; this module generalizes it to the nested-electrode
formula family (``ieeg_pooled_lme.py`` / ``podcast_lme.py``) instead of
N400's crossed-electrode one.

Both datasets have far fewer subjects than any dataset already
ceiling-tested (5 and 9, vs the already-noisy smaller EEG datasets) --
expect a much noisier estimate, quite possibly not distinguishable from
zero. Report accordingly rather than treating a point estimate as reliable.

``prev_surprisal_z`` is dropped from Fedorenko2016's ceiling formula for the
same reason as the N400 datasets: no response is measured at the preceding
word's position to build a split-half mean from (Podcast's real formula
never had a spillover term to begin with).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fmri_lme import split_train_test_chunks
from .ieeg_pooled_lme import prepare_ieeg_lme_frame
from .lme import _fit_stats, _partition_loglik_detailed
from .podcast_lme import prepare_podcast_lme_frame

logger = logging.getLogger(__name__)

DATASETS = ("fedorenko2016", "podcast")


def _ecog_ceiling_formulas(dataset: str) -> tuple[str, str]:
    """Baseline/full formulas -- same as ``ieeg_lme_formulas``/
    ``podcast_lme_formulas`` but with ``prev_surprisal_z`` dropped from
    Fedorenko2016's full model (Podcast's real formula never had one)."""
    if dataset == "fedorenko2016":
        nuisances = "wlen_z + freq_z"
        baseline = f"y ~ {nuisances} + (1 + {nuisances} | subject) + (1 | subject:electrode)"
        full = (
            f"y ~ {nuisances} + surprisal_z + (1 + {nuisances} + surprisal_z | subject) + "
            "(1 + surprisal_z | subject:electrode)"
        )
    elif dataset == "podcast":
        baseline = "y ~ wlen_z + freq_z + (1 | subject) + (1 | subject:electrode)"
        full = "y ~ wlen_z + freq_z + surprisal_z + (1 | subject) + (1 + surprisal_z | subject:electrode)"
    else:
        raise ValueError(f"Unknown ECoG dataset: {dataset!r}")
    return baseline, full


def _ecog_ceiling_direction_task(
    df: pd.DataFrame, source_ids: set, target_ids: set, item_col: str,
) -> pd.DataFrame | None:
    """Build one direction's target-half frame with the split-half item-mean
    ECoG signal (``y``) merged in as a column literally named ``"surprisal"``
    -- required so ``prepare_ieeg_lme_frame``/``prepare_podcast_lme_frame``
    z-score it into ``surprisal_z``, the term the formulas reference."""
    src = df[df["subject"].astype(str).isin(source_ids)]
    tgt = df[df["subject"].astype(str).isin(target_ids)].copy()
    item_mean = src.groupby(item_col)["y"].mean().rename("surprisal").reset_index()
    tgt = tgt.drop(columns=["surprisal"], errors="ignore").merge(item_mean, on=item_col, how="left")
    tgt = tgt.dropna(subset=["surprisal"])
    return tgt if not tgt.empty else None


def _ecog_ceiling_direction_worker(
    iteration: int,
    direction: int,
    tgt: pd.DataFrame,
    *,
    dataset: str,
    out_of_sample: bool,
    train_frac: float,
    random_state: int,
    verbose: bool,
) -> tuple[int, int, float | None, int, bool]:
    """Process-pool worker: fit one (iteration, direction)'s ECoG ceiling
    model. Module-level (not a closure) so it can be pickled for spawn/fork.
    Returns (iteration, direction, delta_loglik or None on failure, n_obs, converged)."""
    from pymer4.models import lmer

    try:
        if dataset == "fedorenko2016":
            frame_in = tgt.assign(prev_surprisal=0.0)
            data = prepare_ieeg_lme_frame(frame_in)
        elif dataset == "podcast":
            data = prepare_podcast_lme_frame(tgt)
        else:
            raise ValueError(f"Unknown ECoG dataset: {dataset!r}")

        if out_of_sample:
            train_data, test_data = split_train_test_chunks(
                data, train_frac=train_frac, random_state=random_state,
            )
        else:
            train_data = test_data = data
        if test_data.height < 10:
            return iteration, direction, None, 0, False

        baseline_formula, full_formula = _ecog_ceiling_formulas(dataset)

        baseline_model = lmer(baseline_formula, data=train_data)
        baseline_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, base_ok = _fit_stats(baseline_model)

        full_model = lmer(full_formula, data=train_data)
        full_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, full_ok = _fit_stats(full_model)

        base_ll, _ = _partition_loglik_detailed(baseline_model, test_data, outcome="y")
        full_ll, _ = _partition_loglik_detailed(full_model, test_data, outcome="y")
    except Exception as exc:
        logger.warning("ECoG ceiling fit failed on split %d direction %d: %s", iteration, direction, exc)
        return iteration, direction, None, 0, False

    converged = bool(base_ok and full_ok)
    delta = full_ll - base_ll if np.isfinite(base_ll) and np.isfinite(full_ll) else None
    ok = delta is not None and np.isfinite(delta) and converged
    return iteration, direction, (delta if ok else None), int(test_data.height), converged


@dataclass
class ECoGDeltaLoglikCeilingResult:
    """Split-half theoretical upper bound for a nested-electrode ECoG LME's
    delta log-likelihood (Timkey et al. 2026 method, adapted)."""

    ceiling_delta_loglik_per_obs: float
    ceiling_delta_loglik_per_obs_sd: float
    n_iterations_used: int
    n_iterations_requested: int
    per_iteration_delta_loglik: list[float] = field(default_factory=list)
    per_iteration_n_obs: list[int] = field(default_factory=list)
    per_direction_records: list[dict] = field(default_factory=list)


def compute_ecog_ceiling_delta_loglik(
    df: pd.DataFrame,
    *,
    dataset: str,
    item_col: str,
    n_splits: int = 15,
    train_frac: float = 0.75,
    seed: int = 0,
    verbose: bool = False,
    n_workers: int = 1,
    out_of_sample: bool = True,
) -> ECoGDeltaLoglikCeilingResult:
    """Split-half noise ceiling for a nested subject:electrode ECoG LME's
    delta log-likelihood. ``df`` must have one row per (subject, electrode,
    word) with columns ``subject, electrode, wlen, freq, y`` plus
    ``item_col`` (the word-level grouping the split-half substitute is
    computed over) and, depending on ``dataset``: ``sentence_id``
    (fedorenko2016, renamed to ``chunk`` internally) or ``chunk``
    (podcast, already present in its raw table)."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    chunk_col = "sentence_id" if dataset == "fedorenko2016" else "chunk"
    required = {"subject", "electrode", "wlen", "freq", "y", chunk_col, item_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for ECoG delta-LL ceiling ({dataset}): {sorted(missing)}")

    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(df["subject"].astype(str).unique()))

    tasks: list[tuple[int, int, pd.DataFrame]] = []
    for it in range(n_splits):
        perm = rng.permutation(subjects)
        half = len(perm) // 2
        halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
        for direction, (source_ids, target_ids) in enumerate((halves, halves[::-1])):
            tgt = _ecog_ceiling_direction_task(df, source_ids, target_ids, item_col)
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
                    _ecog_ceiling_direction_worker, it, direction, tgt,
                    dataset=dataset, out_of_sample=out_of_sample,
                    train_frac=train_frac, random_state=seed, verbose=verbose,
                ): (it, direction)
                for it, direction, tgt in tasks
            }
            for fut in as_completed(futures):
                iteration, direction, delta, n_obs, converged = fut.result()
                _record(iteration, direction, delta, n_obs, converged)
    else:
        for it, direction, tgt in tasks:
            iteration, direction, delta, n_obs, converged = _ecog_ceiling_direction_worker(
                it, direction, tgt, dataset=dataset, out_of_sample=out_of_sample,
                train_frac=train_frac, random_state=seed, verbose=verbose,
            )
            _record(iteration, direction, delta, n_obs, converged)

    per_direction_records.sort(key=lambda r: (r["iteration"], r["direction"]))
    per_iter_delta = [iter_delta[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]
    per_iter_nobs = [iter_nobs[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]

    if not per_iter_delta:
        return ECoGDeltaLoglikCeilingResult(
            ceiling_delta_loglik_per_obs=float("nan"),
            ceiling_delta_loglik_per_obs_sd=float("nan"),
            n_iterations_used=0,
            n_iterations_requested=n_splits,
            per_direction_records=per_direction_records,
        )

    per_obs = [d / n for d, n in zip(per_iter_delta, per_iter_nobs)]
    return ECoGDeltaLoglikCeilingResult(
        ceiling_delta_loglik_per_obs=float(np.mean(per_obs)),
        ceiling_delta_loglik_per_obs_sd=float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan"),
        n_iterations_used=len(per_obs),
        n_iterations_requested=n_splits,
        per_iteration_delta_loglik=per_iter_delta,
        per_iteration_n_obs=per_iter_nobs,
        per_direction_records=per_direction_records,
    )
