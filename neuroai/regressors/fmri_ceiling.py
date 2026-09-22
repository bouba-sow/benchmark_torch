"""Split-half delta-log-likelihood noise ceiling (Timkey et al. 2026,
adapted) for the per-subject fMRI LME datasets -- Natural Stories fMRI (real
subject count in the cached table; see loader), Tuckute (10 subjects), and
Pereira (10 subjects). See neuroai.regressors.n400_pooled_lme's module
docstring for the general split-half method; this module generalizes it to
the fMRI formula families (``fmri_lme.py`` / ``tuckute_pooled_lme.py`` /
``pereira_pooled_lme.py``) instead of N400's crossed-electrode one. LPP
fMRI is NOT covered here -- its real pipeline pools an AVERAGE-SUBJECT BOLD
target with no subject term in the model at all, so this split-half-by-
subject method doesn't apply without first rebuilding a per-subject pooled
table from raw voxel-level BOLD (out of scope for this module; see
neuroai/lpp_encoding.py's fmri_unit="single_subject" path for where that
would start).

Pereira is structurally identical to Tuckute (isolated-sentence BOLD, per
subject/roi/sentence, no HRF) except for one extra nuisance term --
``sentlen_z`` (word count), which Tuckute couldn't use (all its sentences
are exactly 6 words, so sentlen has zero variance there) but Pereira's
variable-length sentences can and do (see pereira_pooled_lme.py's module
docstring for why).

Natural Stories fMRI caveat: the cached nuisance table has no explicit
TR-index or word-index column identifying which row is "the same timepoint"
across subjects -- only ``subject, story, chunk`` (chunk = story, used for
train/test splitting) plus per-row covariates. The split-half item key here
is therefore CONSTRUCTED as row-order-within-(subject, story), assuming (a)
every subject's rows for a given story appear in chronological TR order in
the cached table, and (b) the same story has the same number of usable TRs
across subjects (both standard for an HRF-aligned per-TR design matrix, but
not independently verified here -- if rows were filtered/dropped
differently per subject upstream, this alignment could be off by a few
TRs). Tuckute has no such ambiguity: ``sentence_id`` is an explicit,
unambiguous item key.
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

DATASETS = ("ns_fmri", "tuckute", "pereira")


def _fmri_ceiling_formulas(dataset: str) -> tuple[str, str]:
    """Baseline/full formulas -- exact copies of ``fmri_lme_formulas``/
    ``tuckute_lme_formulas``/``pereira_lme_formulas`` (none has a spillover
    term to drop)."""
    if dataset == "ns_fmri":
        nuisances = "wlen_z + freq_z"
        baseline = f"bold ~ {nuisances} + sentpos_z + (1 + {nuisances} | subject) + (1 | story)"
        full = (
            f"bold ~ {nuisances} + sentpos_z + surprisal_z + "
            f"(1 + {nuisances} + surprisal_z | subject) + (1 + surprisal_z | story)"
        )
    elif dataset == "tuckute":
        nuisances = "mean_wlen_z + mean_freq_z"
        baseline = f"y ~ {nuisances} + (1 + {nuisances} | subject) + (1 | roi) + (1 | sentence_id)"
        full = (
            f"y ~ {nuisances} + surprisal_z + (1 + {nuisances} + surprisal_z | subject) + "
            "(1 | roi) + (1 + surprisal_z | sentence_id)"
        )
    elif dataset == "pereira":
        nuisances = "sentlen_z + mean_wlen_z + mean_freq_z"
        baseline = f"y ~ {nuisances} + (1 + {nuisances} | subject) + (1 | roi) + (1 | sentence_id)"
        full = (
            f"y ~ {nuisances} + surprisal_z + (1 + {nuisances} + surprisal_z | subject) + "
            "(1 | roi) + (1 + surprisal_z | sentence_id)"
        )
    else:
        raise ValueError(f"Unknown fMRI dataset: {dataset!r}")
    return baseline, full


def _fmri_ceiling_direction_task(
    df: pd.DataFrame, source_ids: set, target_ids: set, item_col: str, outcome: str,
) -> pd.DataFrame | None:
    """Build one direction's target-half frame with the split-half item-mean
    outcome merged in as a column literally named ``"surprisal"``."""
    src = df[df["subject"].astype(str).isin(source_ids)]
    tgt = df[df["subject"].astype(str).isin(target_ids)].copy()
    item_mean = src.groupby(item_col)[outcome].mean().rename("surprisal").reset_index()
    tgt = tgt.drop(columns=["surprisal"], errors="ignore").merge(item_mean, on=item_col, how="left")
    tgt = tgt.dropna(subset=["surprisal"])
    return tgt if not tgt.empty else None


def _prepare_ns_fmri_ceiling_frame(df: pd.DataFrame):
    from .lme import zscore_series
    import polars as pl

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["story"] = out["story"].astype(str)
    out["chunk"] = out["chunk"].astype(str)
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["sentpos_z"] = zscore_series(out["sentpos"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["bold"] = out["bold"].astype(float)
    cols = ["subject", "story", "chunk", "wlen_z", "freq_z", "sentpos_z", "surprisal_z", "bold"]
    return pl.from_pandas(out[cols].copy())


def _prepare_tuckute_ceiling_frame(df: pd.DataFrame):
    from .lme import zscore_series
    import polars as pl

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["roi"] = out["roi"].astype(str)
    out["sentence_id"] = out["sentence_id"].astype(str)
    out["chunk"] = out["sentence_id"]  # reuse split_train_test_chunks as-is
    out["mean_wlen_z"] = zscore_series(out["mean_wlen"])
    out["mean_freq_z"] = zscore_series(out["mean_freq"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["y"] = out["y"].astype(float)
    cols = ["subject", "roi", "sentence_id", "chunk", "mean_wlen_z", "mean_freq_z", "surprisal_z", "y"]
    return pl.from_pandas(out[cols].copy())


def _prepare_pereira_ceiling_frame(df: pd.DataFrame):
    from .lme import zscore_series
    import polars as pl

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["roi"] = out["roi"].astype(str)
    out["sentence_id"] = out["sentence_id"].astype(str)
    out["chunk"] = out["sentence_id"]  # reuse split_train_test_chunks as-is
    out["sentlen_z"] = zscore_series(out["sentlen"])
    out["mean_wlen_z"] = zscore_series(out["mean_wlen"])
    out["mean_freq_z"] = zscore_series(out["mean_freq"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["y"] = out["y"].astype(float)
    cols = [
        "subject", "roi", "sentence_id", "chunk",
        "sentlen_z", "mean_wlen_z", "mean_freq_z", "surprisal_z", "y",
    ]
    return pl.from_pandas(out[cols].copy())


def _fmri_ceiling_direction_worker(
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
    """Process-pool worker: fit one (iteration, direction)'s fMRI ceiling
    model. Module-level (not a closure) so it can be pickled for spawn/fork."""
    from pymer4.models import lmer

    outcome = "bold" if dataset == "ns_fmri" else "y"
    try:
        if dataset == "ns_fmri":
            data = _prepare_ns_fmri_ceiling_frame(tgt)
        elif dataset == "tuckute":
            data = _prepare_tuckute_ceiling_frame(tgt)
        elif dataset == "pereira":
            data = _prepare_pereira_ceiling_frame(tgt)
        else:
            raise ValueError(f"Unknown fMRI dataset: {dataset!r}")

        if out_of_sample:
            train_data, test_data = split_train_test_chunks(
                data, train_frac=train_frac, random_state=random_state,
            )
        else:
            train_data = test_data = data
        if test_data.height < 10:
            return iteration, direction, None, 0, False

        baseline_formula, full_formula = _fmri_ceiling_formulas(dataset)

        baseline_model = lmer(baseline_formula, data=train_data)
        baseline_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, base_ok = _fit_stats(baseline_model)

        full_model = lmer(full_formula, data=train_data)
        full_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, full_ok = _fit_stats(full_model)

        base_ll, _ = _partition_loglik_detailed(baseline_model, test_data, outcome=outcome)
        full_ll, _ = _partition_loglik_detailed(full_model, test_data, outcome=outcome)
    except Exception as exc:
        logger.warning("fMRI ceiling fit failed on split %d direction %d: %s", iteration, direction, exc)
        return iteration, direction, None, 0, False

    converged = bool(base_ok and full_ok)
    delta = full_ll - base_ll if np.isfinite(base_ll) and np.isfinite(full_ll) else None
    ok = delta is not None and np.isfinite(delta) and converged
    return iteration, direction, (delta if ok else None), int(test_data.height), converged


@dataclass
class FmriDeltaLoglikCeilingResult:
    ceiling_delta_loglik_per_obs: float
    ceiling_delta_loglik_per_obs_sd: float
    n_iterations_used: int
    n_iterations_requested: int
    per_iteration_delta_loglik: list[float] = field(default_factory=list)
    per_iteration_n_obs: list[int] = field(default_factory=list)
    per_direction_records: list[dict] = field(default_factory=list)


def compute_fmri_ceiling_delta_loglik(
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
) -> FmriDeltaLoglikCeilingResult:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    outcome = "bold" if dataset == "ns_fmri" else "y"
    required = {"subject", outcome, item_col}
    if dataset == "ns_fmri":
        required |= {"story", "chunk", "wlen", "freq", "sentpos"}
    elif dataset == "pereira":
        required |= {"roi", "sentence_id", "sentlen", "mean_wlen", "mean_freq"}
    else:
        required |= {"roi", "sentence_id", "mean_wlen", "mean_freq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for fMRI delta-LL ceiling ({dataset}): {sorted(missing)}")

    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(df["subject"].astype(str).unique()))

    tasks: list[tuple[int, int, pd.DataFrame]] = []
    for it in range(n_splits):
        perm = rng.permutation(subjects)
        half = len(perm) // 2
        halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
        for direction, (source_ids, target_ids) in enumerate((halves, halves[::-1])):
            tgt = _fmri_ceiling_direction_task(df, source_ids, target_ids, item_col, outcome)
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
                    _fmri_ceiling_direction_worker, it, direction, tgt,
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
            iteration, direction, delta, n_obs, converged = _fmri_ceiling_direction_worker(
                it, direction, tgt, dataset=dataset, out_of_sample=out_of_sample,
                train_frac=train_frac, random_state=seed, verbose=verbose,
            )
            _record(iteration, direction, delta, n_obs, converged)

    per_direction_records.sort(key=lambda r: (r["iteration"], r["direction"]))
    per_iter_delta = [iter_delta[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]
    per_iter_nobs = [iter_nobs[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]

    if not per_iter_delta:
        return FmriDeltaLoglikCeilingResult(
            ceiling_delta_loglik_per_obs=float("nan"),
            ceiling_delta_loglik_per_obs_sd=float("nan"),
            n_iterations_used=0,
            n_iterations_requested=n_splits,
            per_direction_records=per_direction_records,
        )

    per_obs = [d / n for d, n in zip(per_iter_delta, per_iter_nobs)]
    return FmriDeltaLoglikCeilingResult(
        ceiling_delta_loglik_per_obs=float(np.mean(per_obs)),
        ceiling_delta_loglik_per_obs_sd=float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan"),
        n_iterations_used=len(per_obs),
        n_iterations_requested=n_splits,
        per_iteration_delta_loglik=per_iter_delta,
        per_iteration_n_obs=per_iter_nobs,
        per_direction_records=per_direction_records,
    )
