"""Split-half delta-log-likelihood noise ceiling (Timkey et al. 2026,
adapted) for Broderick's pooled, CROSSED subject x region LME -- see
``neuroai.regressors.ecog_ceiling`` for the same method applied to the
NESTED subject:electrode design (Fedorenko2016/Podcast); this module is
that same template adapted to Broderick's crossed structure (region is the
same 4 labels -- left/right/posterior/occipital -- for every subject, not
subject-specific), matching the real sweep's formula in
``neuroai.regressors.broderick_lme.broderick_lme_formulas``:

    baseline: eeg ~ wlen_z + freq_z + (1 | subject) + (1 | region)
    full:     eeg ~ wlen_z + freq_z + surprisal_z + prev_surprisal_z +
                     (1 | subject) + (1 + surprisal_z | region)

``prev_surprisal_z`` is dropped from the ceiling's full formula for the
same reason as every other ceiling in this codebase (Fedorenko2016/N400
datasets): there is no real per-layer surprisal here at all -- the ceiling
tests a split-half-derived substitute predictor, which has no
"preceding word" analogue to build a spillover term from.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fmri_lme import split_train_test_chunks
from .broderick_lme import prepare_broderick_lme_frame
from .lme import _fit_stats, _partition_loglik_detailed

logger = logging.getLogger(__name__)


def _broderick_ceiling_formulas() -> tuple[str, str]:
    """Baseline/full formulas -- same as ``broderick_lme_formulas`` but with
    ``prev_surprisal_z`` dropped from the full model (see module docstring)."""
    nuisances = "wlen_z + freq_z"
    re_subject = "(1 | subject)"
    baseline = f"eeg ~ {nuisances} + {re_subject} + (1 | region)"
    full = f"eeg ~ {nuisances} + surprisal_z + {re_subject} + (1 + surprisal_z | region)"
    return baseline, full


def _broderick_ceiling_direction_task(
    df: pd.DataFrame, source_ids: set, target_ids: set, item_col: str,
) -> pd.DataFrame | None:
    """Build one direction's target-half frame with the split-half item-mean
    EEG signal (``eeg``) merged in as a column literally named
    ``"surprisal"`` -- required so ``prepare_broderick_lme_frame`` z-scores
    it into ``surprisal_z``, the term the formulas reference."""
    src = df[df["subject"].astype(str).isin(source_ids)]
    tgt = df[df["subject"].astype(str).isin(target_ids)].copy()
    # Oracle is per (region, word): a region-blind word mean averages four regions whose
    # word-level EEG patterns differ in sign/scale (cross-subject r 0.0005 vs 0.024
    # per-region on the real table), which erased the signal.
    keys = ["region", item_col]
    item_mean = src.groupby(keys)["eeg"].mean().rename("surprisal").reset_index()
    tgt = tgt.drop(columns=["surprisal"], errors="ignore").merge(item_mean, on=keys, how="left")
    tgt = tgt.dropna(subset=["surprisal"])
    return tgt if not tgt.empty else None


def _broderick_ceiling_direction_worker(
    iteration: int,
    direction: int,
    tgt: pd.DataFrame,
    *,
    out_of_sample: bool,
    train_frac: float,
    random_state: int,
    verbose: bool,
) -> tuple[int, int, float | None, int, bool]:
    """Process-pool worker: fit one (iteration, direction)'s Broderick
    ceiling model. Module-level (not a closure) so it can be pickled for
    spawn/fork. Returns (iteration, direction, delta_loglik or None on
    failure, n_obs, converged)."""
    from pymer4.models import lmer

    try:
        frame_in = tgt.assign(prev_surprisal=0.0)
        data = prepare_broderick_lme_frame(frame_in)

        if out_of_sample:
            train_data, test_data = split_train_test_chunks(
                data, train_frac=train_frac, random_state=random_state,
            )
        else:
            train_data = test_data = data
        if test_data.height < 10:
            return iteration, direction, None, 0, False

        baseline_formula, full_formula = _broderick_ceiling_formulas()

        baseline_model = lmer(baseline_formula, data=train_data)
        baseline_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, base_ok = _fit_stats(baseline_model)

        full_model = lmer(full_formula, data=train_data)
        full_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, full_ok = _fit_stats(full_model)

        base_ll, _ = _partition_loglik_detailed(baseline_model, test_data, outcome="eeg")
        full_ll, _ = _partition_loglik_detailed(full_model, test_data, outcome="eeg")
    except Exception as exc:
        logger.warning("Broderick ceiling fit failed on split %d direction %d: %s", iteration, direction, exc)
        return iteration, direction, None, 0, False

    converged = bool(base_ok and full_ok)
    delta = full_ll - base_ll if np.isfinite(base_ll) and np.isfinite(full_ll) else None
    ok = delta is not None and np.isfinite(delta) and converged
    return iteration, direction, (delta if ok else None), int(test_data.height), converged


@dataclass
class BroderickDeltaLoglikCeilingResult:
    """Split-half theoretical upper bound for Broderick's pooled crossed
    subject x region LME's delta log-likelihood (Timkey et al. 2026 method,
    adapted)."""

    ceiling_delta_loglik_per_obs: float
    ceiling_delta_loglik_per_obs_sd: float
    n_iterations_used: int
    n_iterations_requested: int
    per_iteration_delta_loglik: list[float] = field(default_factory=list)
    per_iteration_n_obs: list[int] = field(default_factory=list)
    per_direction_records: list[dict] = field(default_factory=list)


def compute_broderick_ceiling_delta_loglik(
    df: pd.DataFrame,
    *,
    item_col: str = "word_idx",
    n_splits: int = 15,
    train_frac: float = 0.75,
    seed: int = 0,
    verbose: bool = False,
    n_workers: int = 1,
    out_of_sample: bool = True,
    mode: str = "split_half",
    n_repeats: int = 1,
) -> BroderickDeltaLoglikCeilingResult:
    """Split-half noise ceiling for Broderick's pooled crossed subject x
    region LME's delta log-likelihood. ``df`` must have one row per
    (subject, region, word) with columns ``subject, region, chunk,
    word_idx, wlen, freq, eeg`` -- exactly the base pooled table
    ``neuroai.benchmarks.broderick_lme.BroderickLmeBenchmark.run`` writes
    (``broderick_lme_table.parquet``, before any layer's surprisal is
    merged in -- identical across every model/layer in a sweep)."""
    required = {"subject", "region", "wlen", "freq", "eeg", "chunk", item_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Broderick delta-LL ceiling: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(df["subject"].astype(str).unique()))

    tasks: list[tuple[int, int, pd.DataFrame, int]] = []   # (iteration, direction, frame, split seed)
    if mode == "loso":
        # ONE pooled fit per repeat; each row's oracle = item mean over all OTHER subjects.
        n_splits = n_repeats
        keys = ["region", item_col]
        g = df.groupby(keys)["eeg"]
        tot, cnt = g.transform("sum"), g.transform("count")
        gs = df.groupby(keys + [df["subject"].astype(str)])["eeg"]
        own_sum, own_cnt = gs.transform("sum"), gs.transform("count")
        oracle = (tot - own_sum) / (cnt - own_cnt).where(cnt - own_cnt > 0)
        tgt = df.assign(surprisal=oracle).dropna(subset=["surprisal"])
        for rep in range(n_repeats):
            tasks.append((rep, 0, tgt, seed + rep))
    else:
        for it in range(n_splits):
            perm = rng.permutation(subjects)
            half = len(perm) // 2
            halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
            for direction, (source_ids, target_ids) in enumerate((halves, halves[::-1])):
                tgt = _broderick_ceiling_direction_task(df, source_ids, target_ids, item_col)
                if tgt is not None:
                    tasks.append((it, direction, tgt, seed))
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
                    _broderick_ceiling_direction_worker, it, direction, tgt,
                    out_of_sample=out_of_sample,
                    train_frac=train_frac, random_state=rs, verbose=verbose,
                ): (it, direction)
                for it, direction, tgt, rs in tasks
            }
            for fut in as_completed(futures):
                iteration, direction, delta, n_obs, converged = fut.result()
                _record(iteration, direction, delta, n_obs, converged)
    else:
        for it, direction, tgt, rs in tasks:
            iteration, direction, delta, n_obs, converged = _broderick_ceiling_direction_worker(
                it, direction, tgt, out_of_sample=out_of_sample,
                train_frac=train_frac, random_state=rs, verbose=verbose,
            )
            _record(iteration, direction, delta, n_obs, converged)

    per_direction_records.sort(key=lambda r: (r["iteration"], r["direction"]))
    per_iter_delta = [iter_delta[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]
    per_iter_nobs = [iter_nobs[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]

    if not per_iter_delta:
        return BroderickDeltaLoglikCeilingResult(
            ceiling_delta_loglik_per_obs=float("nan"),
            ceiling_delta_loglik_per_obs_sd=float("nan"),
            n_iterations_used=0,
            n_iterations_requested=n_splits,
            per_direction_records=per_direction_records,
        )

    per_obs = [d / n for d, n in zip(per_iter_delta, per_iter_nobs)]
    return BroderickDeltaLoglikCeilingResult(
        ceiling_delta_loglik_per_obs=float(np.mean(per_obs)),
        ceiling_delta_loglik_per_obs_sd=float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan"),
        n_iterations_used=len(per_obs),
        n_iterations_requested=n_splits,
        per_iteration_delta_loglik=per_iter_delta,
        per_iteration_n_obs=per_iter_nobs,
        per_direction_records=per_direction_records,
    )
