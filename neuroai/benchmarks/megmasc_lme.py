"""MEG-MASC MEG surprisal LME — one mixed-effects model per LM layer,
region-averaged, fit PER SUBJECT then averaged across subjects.

Second-stage (CPU) half of a two-stage pipeline: a GPU job
(``scripts/run/surprisal/neural/megmasc_extract.py``) first caches per-layer
logit-lens surprisal for every canonical word; this benchmark reads that
cache, builds each subject's (region × word) MEG table by averaging the
**250-450 ms** post-onset amplitude within six KIT sensor groups, and fits

    baseline: y ~ nuisances + (1 | region)
    full:     y ~ nuisances + surprisal_z + prev_surprisal_z + (1 + surprisal_z | region)

per (subject, layer). See ``neuroai/regressors/megmasc_lme.py`` and
``docs/megmasc_design.md`` §4 for the formula rationale (default
``nuisances = worddur_z + wlen_z + freq_z + prev_worddur_z``; a
``wlen_z + freq_z`` parity run is also available).

Primary metric: Δ log-likelihood (test) = full − baseline, per (subject,
layer), averaged across subjects.

Outputs (under ``output_dir``):
  megmasc_lme_by_subject_layer.csv  — one row per (subject, layer), incremental
  megmasc_lme_by_layer.csv          — one row per layer, averaged across subjects
  megmasc_lme_per_word.csv          — held-out rows + baseline/full loglik (DLT)
"""

from __future__ import annotations

import logging
import time
import typing as tp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.megmasc import (
    MEGMASC_N_CHANNELS,
    MegMascDataset,
    load_meg_run,
    load_run_words,
    load_story_words,
    megmasc_sensor_region,
)
from ..regressors.megmasc_lme import DEFAULT_NUISANCES
from ..sweep import csv_file_lock
from .base import ModelConfig


def _pw_dir(out_dir):
    """Per-word prediction directory: mirrors ``out_dir`` under $SCRATCH (word_predictions_root)
    so large per-word CSVs never land on $WORK (2026-10-12)."""
    from .base import word_predictions_root

    d = word_predictions_root(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d
from .megmasc import (
    MegMascDataConfig as _EncDataConfig,
    _bandpass_eeg,
    _epoch_cache_key,
    _extract_word_epochs,
    _load_epoch_cache,
    _robust_scale_clamp,
)

logger = logging.getLogger(__name__)

_SUBJECT_ROW_COLUMNS = (
    "subject", "layer", "beta", "se", "t_stat", "p_value", "loglik", "aic", "bic",
    "baseline_loglik", "delta_loglik_train", "delta_loglik_test",
    "baseline_loglik_test", "loglik_test", "lrt_p_value",
    "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "slope_dropped", "n_obs", "n_obs_test", "error", "fit_wall_time_s",
)
_POOLED_ROW_COLUMNS = (
    "layer", "beta", "se", "t_stat", "p_value", "loglik", "aic", "bic",
    "baseline_loglik", "delta_loglik_train", "delta_loglik_test",
    "baseline_loglik_test", "loglik_test", "lrt_p_value",
    "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "slope_dropped", "n_obs", "n_obs_test", "error", "fit_wall_time_s",
)
_AVERAGE_METRIC_COLUMNS = (
    "beta", "se", "t_stat", "p_value", "loglik", "aic", "bic",
    "baseline_loglik", "delta_loglik_train", "delta_loglik_test",
    "baseline_loglik_test", "loglik_test", "lrt_p_value",
    "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr",
    "n_obs", "n_obs_test", "fit_wall_time_s",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MegMascLmeDataConfig:
    bids_root: str
    subjects: list[str] | None = None
    stories: list[str] | None = None
    # Scalar-outcome window (s rel. word onset) — mean MEG amplitude here is
    # the LME's `y`. 250-450 ms from the encoding time course (docs §6).
    window_tmin: float = 0.25
    window_tmax: float = 0.45
    # Epoch window must cover baseline + outcome window.
    epoch_tmin: float = -0.2
    epoch_tmax: float = 0.8
    baseline: tuple[float, float] | None = (-0.2, 0.0)
    # Same preprocessing as the ridge encoder (benchmarks/megmasc.py).
    l_freq: float | None = 0.1
    h_freq: float | None = 40.0
    target_sfreq: float = 50.0
    robust_scale: bool = True
    clamp: float | None = 5.0
    min_word_duration: float = 0.050
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    wordfreq_language: str = "en"
    lang: str = "en"
    device: str = "cpu"
    feature_cache_dir: str | None = None
    # Model-independent word-locked epoch cache built by
    # scripts/data/megmasc_epoch_cache.py / the ridge encoder
    # (benchmarks/megmasc.py) — when set and a hit, avoids re-reading raw
    # MEG entirely (the epoch/preprocessing params above must match the
    # cache's, which the config defaults are designed to do).
    epoch_cache_dir: str | None = None


@dataclass
class MegMascLmeConfig:
    train_frac: float = 0.75
    split_random_state: int = 0
    verbose: bool = False
    allow_slope_fallback: bool = True
    layer_stride: int = 1
    last_layer_only: bool = False  # fit only the final layer (overrides layer_stride)
    n_workers: int = 10
    require_cached_surprisal: bool = False
    save_word_predictions: bool = True
    # "worddur_z + wlen_z + freq_z + prev_worddur_z" (default) or
    # "wlen_z + freq_z" for a Broderick-parity run.
    nuisances: str = DEFAULT_NUISANCES
    # "subject"     — ONE pooled fit per layer, all subjects together, with
    #                 (1+surprisal_z|subject) crossed with (1+surprisal_z|region)
    #                 (region is the same fixed 6-way grouping for every
    #                 subject, so this is well-posed; the only risk is lme4
    #                 convergence time on the larger crossed-RE model).
    # "none"        — the old two-stage design: one small LME per subject,
    #                 then average across subjects (safe fallback if pooled
    #                 fits don't converge in reasonable time).
    pooling: str = "subject"


@dataclass
class MegMascLmeBenchmarkConfig:
    data: MegMascLmeDataConfig
    model: ModelConfig
    lme: MegMascLmeConfig = field(default_factory=MegMascLmeConfig)
    output_dir: str = "outputs"
    canonical_model_name: str | None = None


# ---------------------------------------------------------------------------
# Spawned worker
# ---------------------------------------------------------------------------

def _fit_one_subject_layer_worker(
    base_parquet_path: str,
    surprisal_npy_path: str,
    prev_surprisal_npy_path: str,
    subject: str,
    layer_idx: int,
    *,
    nuisances: str,
    train_frac: float,
    random_state: int,
    verbose: bool,
    allow_slope_fallback: bool,
    save_word_predictions: bool = False,
) -> dict[str, tp.Any]:
    import numpy as _np
    import pandas as _pd

    from ..regressors.megmasc_lme import fit_megmasc_lme_one_layer

    t0 = time.time()
    row: dict[str, tp.Any] = {"subject": subject, "layer": layer_idx}
    per_word_out: pd.DataFrame | None = None
    try:
        base_df = _pd.read_parquet(base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        prev_surp_all = _np.load(prev_surprisal_npy_path, mmap_mode="r")
        base_df["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)
        base_df["prev_surprisal"] = _np.asarray(prev_surp_all[:, layer_idx], dtype=_np.float64)

        fit = fit_megmasc_lme_one_layer(
            base_df,
            nuisances=nuisances,
            train_frac=train_frac,
            random_state=random_state,
            verbose=verbose,
            allow_slope_fallback=allow_slope_fallback,
            save_per_word=save_word_predictions,
        )
        if fit.per_word is not None and not fit.per_word.empty:
            per_word_out = fit.per_word.copy()
            per_word_out.insert(0, "layer", layer_idx)
            per_word_out.insert(0, "subject", subject)
        row.update({
            "beta": fit.beta, "se": fit.se, "t_stat": fit.t_stat, "p_value": fit.p_value,
            "loglik": fit.loglik, "aic": fit.aic, "bic": fit.bic,
            "baseline_loglik": fit.baseline_loglik,
            "delta_loglik_train": fit.delta_loglik_train,
            "delta_loglik_test": fit.delta_loglik_test,
            "baseline_loglik_test": fit.baseline_loglik_test,
            "loglik_test": fit.loglik_test,
            "lrt_p_value": fit.lrt_p_value,
            "test_pearsonr_full_model": fit.test_pearsonr_full_model,
            "test_spearmanr_full_model": fit.test_spearmanr_full_model,
            "test_pearsonr_baseline_lme": fit.test_pearsonr_baseline_lme,
            "test_spearmanr_baseline_lme": fit.test_spearmanr_baseline_lme,
            "test_delta_pearsonr": fit.test_delta_pearsonr,
            "test_delta_spearmanr": fit.test_delta_spearmanr,
            "converged": fit.converged, "slope_dropped": fit.slope_dropped,
            "n_obs": fit.n_obs, "n_obs_test": fit.n_obs_test, "error": None,
        })
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] Layer %d failed", subject, layer_idx, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return {"row": row, "per_word": per_word_out}


def _fit_one_layer_pooled_worker(
    base_parquet_path: str,
    surprisal_npy_path: str,
    prev_surprisal_npy_path: str,
    layer_idx: int,
    *,
    nuisances: str,
    train_frac: float,
    random_state: int,
    verbose: bool,
    allow_slope_fallback: bool,
    save_word_predictions: bool = False,
) -> dict[str, tp.Any]:
    """Pooled counterpart of :func:`_fit_one_subject_layer_worker`: ONE fit
    per layer, spanning every subject's rows at once."""
    import numpy as _np
    import pandas as _pd

    from ..regressors.megmasc_lme import fit_megmasc_lme_pooled_one_layer

    t0 = time.time()
    row: dict[str, tp.Any] = {"layer": layer_idx}
    per_word_out: pd.DataFrame | None = None
    try:
        base_df = _pd.read_parquet(base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        prev_surp_all = _np.load(prev_surprisal_npy_path, mmap_mode="r")
        base_df["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)
        base_df["prev_surprisal"] = _np.asarray(prev_surp_all[:, layer_idx], dtype=_np.float64)

        fit = fit_megmasc_lme_pooled_one_layer(
            base_df,
            nuisances=nuisances,
            train_frac=train_frac,
            random_state=random_state,
            verbose=verbose,
            allow_slope_fallback=allow_slope_fallback,
            save_per_word=save_word_predictions,
        )
        if fit.per_word is not None and not fit.per_word.empty:
            per_word_out = fit.per_word.copy()
            per_word_out.insert(0, "layer", layer_idx)
        row.update({
            "beta": fit.beta, "se": fit.se, "t_stat": fit.t_stat, "p_value": fit.p_value,
            "loglik": fit.loglik, "aic": fit.aic, "bic": fit.bic,
            "baseline_loglik": fit.baseline_loglik,
            "delta_loglik_train": fit.delta_loglik_train,
            "delta_loglik_test": fit.delta_loglik_test,
            "baseline_loglik_test": fit.baseline_loglik_test,
            "loglik_test": fit.loglik_test,
            "lrt_p_value": fit.lrt_p_value,
            "test_pearsonr_full_model": fit.test_pearsonr_full_model,
            "test_spearmanr_full_model": fit.test_spearmanr_full_model,
            "test_pearsonr_baseline_lme": fit.test_pearsonr_baseline_lme,
            "test_spearmanr_baseline_lme": fit.test_spearmanr_baseline_lme,
            "test_delta_pearsonr": fit.test_delta_pearsonr,
            "test_delta_spearmanr": fit.test_delta_spearmanr,
            "converged": fit.converged, "slope_dropped": fit.slope_dropped,
            "n_obs": fit.n_obs, "n_obs_test": fit.n_obs_test, "error": None,
        })
    except Exception as exc:  # noqa: BLE001
        logger.error("Pooled layer %d failed", layer_idx, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return {"row": row, "per_word": per_word_out}


# ---------------------------------------------------------------------------
# Canonical word list
# ---------------------------------------------------------------------------

def _load_all_words(bids_root: str, stories: list[str] | None) -> pd.DataFrame:
    """Canonical word list across stories, global ``word_idx`` running index
    (mirrors benchmarks/megmasc.py::_load_all_words)."""
    ds = MegMascDataset(bids_root, stories=stories)
    dfs, g = [], 0
    for story in ds.available_stories():
        d = load_story_words(bids_root, story).sort_values("word_index").reset_index(drop=True)
        d["word_idx"] = np.arange(g, g + len(d), dtype=np.int64)
        g += len(d)
        dfs.append(d)
    if not dfs:
        raise RuntimeError(
            f"No MEG-MASC stories found under {bids_root!r} — check bids_root / "
            "MEGMASC_BIDS_ROOT."
        )
    return pd.concat(dfs, ignore_index=True)


def _build_story_groups(words_df: pd.DataFrame) -> list[tuple[str, list[str], list[int]]]:
    groups = []
    for story, g in words_df.sort_values(["story", "word_idx"]).groupby("story"):
        groups.append((str(story), g["word"].astype(str).tolist(),
                       g["word_idx"].astype(int).tolist()))
    return groups


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class MegMascLmeBenchmark:
    def __init__(self, cfg: MegMascLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def extract_and_cache_surprisal(self) -> tuple[np.ndarray, np.ndarray]:
        """GPU-stage entry point: populate the surprisal cache only."""
        words_df = _load_all_words(self.cfg.data.bids_root, self.cfg.data.stories)
        return self._extract_story_surprisal(words_df)

    def run(self) -> pd.DataFrame:
        """Dispatch to the pooled (default) or per-subject LME sweep — see
        ``MegMascLmeConfig.pooling``."""
        if self.cfg.lme.pooling == "subject":
            return self._run_pooled()
        return self._run_per_subject()

    # ------------------------------------------------------------------
    # Shared setup (both pooling modes)
    # ------------------------------------------------------------------

    def _prepare_common(self) -> dict[str, tp.Any]:
        dcfg = self.cfg.data
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name

        words_df = _load_all_words(dcfg.bids_root, dcfg.stories)
        ds = MegMascDataset(dcfg.bids_root, subjects=dcfg.subjects, stories=dcfg.stories)
        subjects = ds.available_subjects()
        runs_df = ds.runs()

        surprisal_items, prev_surprisal_items = self._extract_story_surprisal(words_df)
        wlen_freq = self._load_baseline_covariates(words_df["word"].astype(str).tolist())
        dur_cov = self._load_duration_covariates(words_df)
        n_layers = surprisal_items.shape[1]

        item_finite = (
            np.isfinite(surprisal_items).all(axis=1)
            & np.isfinite(prev_surprisal_items).all(axis=1)
            & np.isfinite(wlen_freq).all(axis=1)
            & np.isfinite(dur_cov).all(axis=1)
        )
        if not item_finite.all():
            logger.warning(
                "[%s] dropping %d/%d words with non-finite features (story-initial "
                "words have no prev_surprisal/prev_worddur)", model_name,
                int((~item_finite).sum()), len(words_df),
            )
        keep_word_idx = set(words_df.loc[item_finite, "word_idx"].astype(int).tolist())
        word_idx_to_pos = {int(w): i for i, w in enumerate(words_df["word_idx"].astype(int))}

        region_groups: dict[str, list[int]] = {}
        for ci in range(MEGMASC_N_CHANNELS):
            region_groups.setdefault(megmasc_sensor_region(ci), []).append(ci)

        return dict(
            model_name=model_name, words_df=words_df, subjects=subjects, runs_df=runs_df,
            surprisal_items=surprisal_items, prev_surprisal_items=prev_surprisal_items,
            wlen_freq=wlen_freq, dur_cov=dur_cov, n_layers=n_layers,
            keep_word_idx=keep_word_idx, word_idx_to_pos=word_idx_to_pos,
            region_groups=region_groups,
        )

    def _merge_covariates(self, df: pd.DataFrame, c: dict[str, tp.Any]) -> pd.DataFrame:
        """Attach wlen/freq/worddur/prev_worddur (canonical, word_idx-indexed)
        to a (region, chunk, word_idx, y[, subject]) table."""
        row_pos = df["word_idx"].map(c["word_idx_to_pos"]).to_numpy(dtype=np.int64)
        df = df.copy()
        df["wlen"] = c["wlen_freq"][row_pos, 0]
        df["freq"] = c["wlen_freq"][row_pos, 1]
        df["worddur"] = c["dur_cov"][row_pos, 0]
        df["prev_worddur"] = c["dur_cov"][row_pos, 1]
        return df, row_pos

    # ------------------------------------------------------------------
    # Pooled sweep (default): ONE fit per layer, all subjects together
    # ------------------------------------------------------------------

    def _run_pooled(self) -> pd.DataFrame:
        lcfg = self.cfg.lme
        c = self._prepare_common()
        model_name = c["model_name"]

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        layer_csv = out_dir / "megmasc_lme_by_layer.csv"
        per_word_csv = _pw_dir(out_dir) / "megmasc_lme_per_word.csv"

        n_layers = c["n_layers"]
        layer_indices = [n_layers - 1] if lcfg.last_layer_only else list(range(0, n_layers, max(1, lcfg.layer_stride)))
        done_layers: set[int] = set()
        if layer_csv.exists():
            try:
                existing = pd.read_csv(layer_csv)
                good = existing.loc[existing["error"].isna()]
                done_layers = set(good["layer"].astype(int).tolist())
                if done_layers:
                    logger.info("Skipping %d already-fit layer(s)", len(done_layers))
                if int(existing["error"].notna().sum()):
                    good.reindex(columns=_POOLED_ROW_COLUMNS).to_csv(layer_csv, index=False)
            except Exception:
                logger.warning("Could not read existing layer CSV; re-fitting all.")
        pending_layers = [li for li in layer_indices if li not in done_layers]
        if not pending_layers:
            logger.info("[%s] all %d layers already fit", model_name, len(layer_indices))
            return pd.DataFrame()

        parts: list[pd.DataFrame] = []
        for subject in c["subjects"]:
            subj_runs = c["runs_df"].loc[c["runs_df"]["subject"] == subject]
            t = self._build_subject_region_table(
                subject, subj_runs, c["words_df"], c["keep_word_idx"], c["region_groups"],
            )
            if t is None or t.empty:
                logger.warning("[%s | %s] no valid rows — skipping", model_name, subject)
                continue
            t["subject"] = subject
            parts.append(t)
        if not parts:
            raise RuntimeError(f"[{model_name}] no subject produced any rows — nothing to pool")
        pooled = pd.concat(parts, ignore_index=True)
        pooled, row_pos = self._merge_covariates(pooled, c)
        logger.info("[%s] pooled table: %d rows, %d subjects, %d regions",
                    model_name, len(pooled), pooled["subject"].nunique(), pooled["region"].nunique())

        bp = out_dir / "megmasc_lme_pooled_table.parquet"
        sp = out_dir / "megmasc_lme_pooled_surprisal.npy"
        pp = out_dir / "megmasc_lme_pooled_prev_surprisal.npy"
        pooled.to_parquet(bp, index=False)
        np.save(sp, c["surprisal_items"][row_pos].astype(np.float32))
        np.save(pp, c["prev_surprisal_items"][row_pos].astype(np.float32))
        del pooled

        logger.info("[%s] fitting %d layer(s) (pooled, all subjects), %d workers",
                    model_name, len(pending_layers), lcfg.n_workers)

        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        rows_out: list[dict] = []
        with ProcessPoolExecutor(
            max_workers=min(lcfg.n_workers, len(pending_layers)), mp_context=ctx,
            max_tasks_per_child=1,
        ) as pool:
            futs = {
                pool.submit(
                    _fit_one_layer_pooled_worker,
                    str(bp), str(sp), str(pp), li,
                    nuisances=lcfg.nuisances,
                    train_frac=lcfg.train_frac,
                    random_state=lcfg.split_random_state,
                    verbose=lcfg.verbose,
                    allow_slope_fallback=lcfg.allow_slope_fallback,
                    save_word_predictions=lcfg.save_word_predictions,
                ): li
                for li in pending_layers
            }
            for fut in as_completed(futs):
                li = futs[fut]
                try:
                    res = fut.result()
                    row, per_word = res["row"], res["per_word"]
                except Exception as exc:
                    logger.error("layer %d worker crashed: %s", li, exc, exc_info=True)
                    row, per_word = {"layer": li, "error": str(exc)}, None
                rows_out.append(row)
                pd.DataFrame([row]).reindex(columns=_POOLED_ROW_COLUMNS).to_csv(
                    layer_csv, mode="a", header=not layer_csv.exists(), index=False,
                )
                if per_word is not None:
                    with csv_file_lock(per_word_csv):
                        per_word.to_csv(per_word_csv, mode="a",
                                        header=not per_word_csv.exists(), index=False)
                logger.info("  [layer %d] %.1fs — delta_loglik_test=%s converged=%s%s",
                            li, row.get("fit_wall_time_s", float("nan")),
                            f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                            row.get("converged"),
                            " (slope dropped)" if row.get("slope_dropped") else "")
        return pd.DataFrame(rows_out)

    # ------------------------------------------------------------------
    # Per-subject sweep (fallback: MegMascLmeConfig.pooling = "none")
    # ------------------------------------------------------------------

    def _run_per_subject(self) -> pd.DataFrame:
        lcfg = self.cfg.lme
        c = self._prepare_common()
        model_name = c["model_name"]
        words_df, subjects, runs_df = c["words_df"], c["subjects"], c["runs_df"]
        surprisal_items, prev_surprisal_items = c["surprisal_items"], c["prev_surprisal_items"]
        keep_word_idx, region_groups = c["keep_word_idx"], c["region_groups"]
        n_layers = c["n_layers"]

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        subject_csv = out_dir / "megmasc_lme_by_subject_layer.csv"
        summary_csv = out_dir / "megmasc_lme_by_layer.csv"

        done_pairs: set[tuple[str, int]] = set()
        if subject_csv.exists():
            try:
                existing = pd.read_csv(subject_csv)
                good = existing.loc[existing["error"].isna()]
                done_pairs = set(zip(good["subject"], good["layer"]))
                if done_pairs:
                    logger.info("Skipping %d already-fit (subject, layer) pair(s)", len(done_pairs))
                if int(existing["error"].notna().sum()):
                    good.reindex(columns=_SUBJECT_ROW_COLUMNS).to_csv(subject_csv, index=False)
            except Exception:
                logger.warning("Could not read existing subject CSV; re-fitting all.")

        layer_indices = list(range(0, n_layers, max(1, lcfg.layer_stride)))

        tasks: list[tuple[str, int, Path, Path, Path]] = []
        for subject in subjects:
            pending = [li for li in layer_indices if (subject, li) not in done_pairs]
            if not pending:
                continue
            subj_runs = runs_df.loc[runs_df["subject"] == subject]
            subj_df = self._build_subject_region_table(
                subject, subj_runs, words_df, keep_word_idx, region_groups,
            )
            if subj_df is None or subj_df.empty:
                logger.warning("[%s | %s] no valid rows — skipping", model_name, subject)
                continue

            subj_df, row_pos = self._merge_covariates(subj_df, c)
            surprisal_by_row = surprisal_items[row_pos]
            prev_surprisal_by_row = prev_surprisal_items[row_pos]

            logger.info("[%s | %s] table: %d rows (%d regions), %d layers pending",
                        model_name, subject, len(subj_df), subj_df["region"].nunique(), len(pending))

            safe = str(subject).replace("/", "_")
            bp = out_dir / f"megmasc_lme_table_{safe}.parquet"
            sp = out_dir / f"megmasc_lme_surprisal_{safe}.npy"
            pp = out_dir / f"megmasc_lme_prev_surprisal_{safe}.npy"
            subj_df.to_parquet(bp, index=False)
            np.save(sp, surprisal_by_row.astype(np.float32))
            np.save(pp, prev_surprisal_by_row.astype(np.float32))
            del subj_df, surprisal_by_row, prev_surprisal_by_row
            for li in pending:
                tasks.append((subject, li, bp, sp, pp))

        logger.info("[%s] fitting %d (subject, layer) pair(s), %d workers",
                    model_name, len(tasks), lcfg.n_workers)

        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        per_word_csv = _pw_dir(out_dir) / "megmasc_lme_per_word.csv"
        rows_out: list[dict] = []
        if tasks:
            with ProcessPoolExecutor(
                max_workers=lcfg.n_workers, mp_context=ctx, max_tasks_per_child=1,
            ) as pool:
                futs = {
                    pool.submit(
                        _fit_one_subject_layer_worker,
                        str(bp), str(sp), str(pp), subject, li,
                        nuisances=lcfg.nuisances,
                        train_frac=lcfg.train_frac,
                        random_state=lcfg.split_random_state,
                        verbose=lcfg.verbose,
                        allow_slope_fallback=lcfg.allow_slope_fallback,
                        save_word_predictions=lcfg.save_word_predictions,
                    ): (subject, li)
                    for (subject, li, bp, sp, pp) in tasks
                }
                for fut in as_completed(futs):
                    subject, li = futs[fut]
                    try:
                        res = fut.result()
                        row, per_word = res["row"], res["per_word"]
                    except Exception as exc:
                        logger.error("[%s] layer %d worker crashed: %s", subject, li, exc, exc_info=True)
                        row, per_word = {"subject": subject, "layer": li, "error": str(exc)}, None
                    rows_out.append(row)
                    pd.DataFrame([row]).reindex(columns=_SUBJECT_ROW_COLUMNS).to_csv(
                        subject_csv, mode="a", header=not subject_csv.exists(), index=False,
                    )
                    if per_word is not None:
                        with csv_file_lock(per_word_csv):
                            per_word.to_csv(per_word_csv, mode="a",
                                            header=not per_word_csv.exists(), index=False)
                    logger.info("  [%s | layer %d] %.1fs — delta_loglik_test=%s converged=%s%s",
                                subject, li, row.get("fit_wall_time_s", float("nan")),
                                f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                                row.get("converged"),
                                " (slope dropped)" if row.get("slope_dropped") else "")

        self._write_averaged_summary(subject_csv, summary_csv)
        return pd.DataFrame(rows_out)

    # ------------------------------------------------------------------

    def _write_averaged_summary(self, subject_csv: Path, summary_csv: Path) -> None:
        if not subject_csv.exists():
            return
        df = pd.read_csv(subject_csv)
        ok = df.loc[df["error"].isna()].copy()
        if ok.empty:
            logger.warning("No successful subject fits to average.")
            return
        rows = []
        for layer, g in ok.groupby("layer"):
            r: dict[str, tp.Any] = {"layer": layer}
            for col in _AVERAGE_METRIC_COLUMNS:
                r[col] = float(g[col].mean()) if col in g else float("nan")
            r["n_subjects"] = int(len(g))
            r["n_converged"] = int(g["converged"].fillna(False).astype(bool).sum())
            r["n_slope_dropped"] = int(g["slope_dropped"].fillna(False).astype(bool).sum())
            rows.append(r)
        pd.DataFrame(rows).sort_values("layer").to_csv(summary_csv, index=False)
        logger.info("Wrote averaged summary (%d layers) to %s", len(rows), summary_csv)

    # ------------------------------------------------------------------

    def _build_subject_region_table(
        self,
        subject: str,
        subj_runs: pd.DataFrame,
        words_df: pd.DataFrame,
        keep_word_idx: set[int],
        region_groups: dict[str, list[int]],
    ) -> pd.DataFrame | None:
        """One row per (region, word) for THIS subject: the window-mean MEG
        amplitude, averaged over channels in each sensor group. Columns:
        region, chunk (story), word_idx, y — wlen/freq/worddur/prev_worddur
        are canonical and merged centrally (see ``_merge_covariates``).

        Fast path: read the model-independent word-locked epoch cache
        already built by the ridge encoder / ``data/megmasc_epoch_cache.py``
        (no raw MEG I/O). Falls back to a fresh raw-MEG read (slow, ~30s-2min
        per recording) only on a cache miss.
        """
        cached = self._table_from_epoch_cache(subject, words_df, keep_word_idx, region_groups)
        if cached is not None:
            return cached
        return self._build_subject_region_table_raw(
            subject, subj_runs, words_df, keep_word_idx, region_groups,
        )

    def _epoch_cache_path(self, subject: str) -> Path | None:
        dcfg = self.cfg.data
        if not dcfg.epoch_cache_dir:
            return None
        enc_cfg = _EncDataConfig(
            bids_root=dcfg.bids_root,
            tmin=dcfg.epoch_tmin, tmax=dcfg.epoch_tmax, baseline=dcfg.baseline,
            l_freq=dcfg.l_freq, h_freq=dcfg.h_freq, target_sfreq=dcfg.target_sfreq,
            robust_scale=dcfg.robust_scale, clamp=dcfg.clamp,
            min_word_duration=dcfg.min_word_duration,
        )
        key = _epoch_cache_key(enc_cfg)
        return Path(dcfg.epoch_cache_dir) / f"{subject}_{key}.npz"

    def _table_from_epoch_cache(
        self,
        subject: str,
        words_df: pd.DataFrame,
        keep_word_idx: set[int],
        region_groups: dict[str, list[int]],
    ) -> pd.DataFrame | None:
        dcfg = self.cfg.data
        path = self._epoch_cache_path(subject)
        if path is None or not path.exists():
            return None
        try:
            epochs, _groups, times, word_indices = _load_epoch_cache(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] epoch cache unreadable (%s) — falling back to raw MEG: %s",
                            subject, path.name, exc)
            return None

        win = (times >= dcfg.window_tmin) & (times < dcfg.window_tmax)
        if not win.any():
            logger.warning("[%s] window [%.2f,%.2f) outside cached epoch range — "
                            "falling back to raw MEG", subject, dcfg.window_tmin, dcfg.window_tmax)
            return None
        wi = word_indices.astype(np.int64)
        mask = np.fromiter((w in keep_word_idx for w in wi), dtype=bool, count=len(wi))
        if not mask.any():
            return None
        amp = epochs[mask][:, :, win].mean(axis=-1)  # (n, 208)
        wi = wi[mask]

        story_by_idx = dict(zip(words_df["word_idx"].astype(int), words_df["story"]))
        stories = np.array([story_by_idx.get(w) for w in wi], dtype=object)

        parts = [
            pd.DataFrame({
                "region": region_name, "chunk": stories, "word_idx": wi,
                "y": amp[:, ch_idxs].mean(axis=1),
            })
            for region_name, ch_idxs in region_groups.items()
        ]
        out = pd.concat(parts, ignore_index=True).dropna(subset=["y", "chunk"])
        if out.empty:
            return None
        logger.info("[%s] epoch cache hit (%s) — %d rows, no raw MEG read",
                    subject, path.name, len(out))
        return out.reset_index(drop=True)

    def _build_subject_region_table_raw(
        self,
        subject: str,
        subj_runs: pd.DataFrame,
        words_df: pd.DataFrame,
        keep_word_idx: set[int],
        region_groups: dict[str, list[int]],
    ) -> pd.DataFrame | None:
        """Slow path: read raw MEG directly (epoch cache miss). Same output
        contract as :meth:`_build_subject_region_table`."""
        dcfg = self.cfg.data
        if subj_runs.empty or not region_groups:
            return None
        logger.warning("[%s] no epoch cache hit — reading raw MEG (slow); "
                        "set data.epoch_cache_dir to avoid this", subject)

        canon_by_story = {
            s: g[["word_index", "word_idx"]].astype({"word_index": int, "word_idx": int})
            for s, g in words_df.groupby("story")
        }
        parts: list[pd.DataFrame] = []

        for run in subj_runs.itertuples():
            story = run.story
            canon = canon_by_story.get(story)
            if canon is None:
                continue
            rw = load_run_words(run.events_tsv)
            if dcfg.min_word_duration > 0:
                rw = rw[rw["duration"] >= dcfg.min_word_duration]
            rw = rw.merge(canon, on="word_index", how="inner")
            rw = rw[rw["word_idx"].isin(keep_word_idx)].sort_values("word_index")
            if rw.empty:
                continue

            try:
                meg, sf = load_meg_run(run.con)
            except (FileNotFoundError, OSError) as exc:
                logger.warning("[%s] story %s: MEG unreadable: %s", subject, story, exc)
                continue
            if dcfg.l_freq is not None and dcfg.h_freq is not None:
                meg = _bandpass_eeg(meg, sf, dcfg.l_freq, dcfg.h_freq)
            if dcfg.target_sfreq < sf:
                from math import gcd as _gcd
                from scipy.signal import resample_poly as _rp
                up, down = int(round(dcfg.target_sfreq)), int(round(sf))
                g = _gcd(up, down)
                meg = _rp(meg, up // g, down // g, axis=-1).astype(np.float32)
            if dcfg.robust_scale:
                meg = _robust_scale_clamp(meg, dcfg.clamp)

            epochs, valid, etimes = _extract_word_epochs(
                meg, dcfg.target_sfreq, rw["onset"].to_numpy(np.float64),
                dcfg.epoch_tmin, dcfg.epoch_tmax,
                tuple(dcfg.baseline) if dcfg.baseline else None,
            )
            if not valid.any():
                continue
            win = (etimes >= dcfg.window_tmin) & (etimes < dcfg.window_tmax)
            if not win.any():
                logger.warning("[%s] story %s: window [%.2f,%.2f) outside epoch",
                               subject, story, dcfg.window_tmin, dcfg.window_tmax)
                continue
            amp = epochs[:, :, win].mean(axis=-1)  # (n_valid, n_ch)

            rw_v = rw.iloc[np.flatnonzero(valid)]
            widx = rw_v["word_idx"].to_numpy(np.int64)
            for region_name, ch_idxs in region_groups.items():
                parts.append(pd.DataFrame({
                    "region": region_name,
                    "chunk": story,
                    "word_idx": widx,
                    "y": amp[:, ch_idxs].mean(axis=1),
                }))

        if not parts:
            return None
        return pd.concat(parts, ignore_index=True).dropna(subset=["y"]).reset_index(drop=True)

    # ------------------------------------------------------------------

    def _load_baseline_covariates(self, words: list[str]) -> np.ndarray:
        from ..features.word_frequency import compute_wlen_wordfreq_covariates
        return compute_wlen_wordfreq_covariates(words, language=self.cfg.data.wordfreq_language)

    # ------------------------------------------------------------------

    def _load_duration_covariates(self, words_df: pd.DataFrame) -> np.ndarray:
        """Canonical [worddur, prev_worddur] (seconds), one row per
        ``words_df`` row, in the same order. Word duration is essentially
        identical across sessions (same rendered audio; verified inter-word
        gap correlation ~0.994) so ONE representative recording per story is
        enough — no MEG, just that story's own events.tsv. ``prev_worddur``
        is the same array shifted one word within each story (NaN for each
        story's first word, matching ``prev_surprisal``)."""
        bids_root = self.cfg.data.bids_root
        ds = MegMascDataset(bids_root, stories=self.cfg.data.stories)
        runs = ds.runs()
        dur_by_idx: dict[int, float] = {}
        for story in ds.available_stories():
            story_runs = runs[runs["story"] == story]
            if story_runs.empty:
                continue
            rw = load_run_words(story_runs.iloc[0]["events_tsv"]).sort_values("word_index")
            canon = words_df.loc[words_df["story"] == story, ["word_index", "word_idx"]]
            merged = rw.merge(canon.astype({"word_index": int}), on="word_index", how="inner")
            for wi, dur in zip(merged["word_idx"].astype(int), merged["duration"].astype(float)):
                dur_by_idx[wi] = dur

        word_idx_order = words_df["word_idx"].astype(int).tolist()
        worddur = np.array([dur_by_idx.get(w, np.nan) for w in word_idx_order], dtype=np.float64)
        prev_worddur = np.full_like(worddur, np.nan)
        for story, g in words_df.groupby("story"):
            pos = g.index.to_numpy()
            order = np.argsort(g["word_idx"].to_numpy())
            ordered_pos = pos[order]
            prev_worddur[ordered_pos[1:]] = worddur[ordered_pos[:-1]]
        return np.stack([worddur, prev_worddur], axis=1).astype(np.float32)

    # ------------------------------------------------------------------

    def _extract_story_surprisal(self, words_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per-layer surprisal for every canonical word + its within-story
        predecessor, in ``word_idx`` order. Logit-lens for standard causal
        LMs (incl. the constrained ``Lanni-ni/*`` alibi / stickbreaking /
        forgetting / hard-*gram HF models and BabyLM); the dedicated CBR-RNN
        path for that one custom architecture (single hidden state, no layer
        axis)."""
        mcfg, dcfg = self.cfg.model, self.cfg.data
        require_cache_hit = self.cfg.lme.require_cached_surprisal
        is_cbr_rnn = "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower()

        groups = _build_story_groups(words_df)
        n_layers: int | None = None
        surp_by_idx: dict[int, np.ndarray] = {}
        prev_by_idx: dict[int, np.ndarray] = {}

        if is_cbr_rnn:
            from ..features.cbr_rnn import batch_surprisal_cbr_rnn
            contexts: list[str] = []
            targets: list[str] = []
            for _story, wlist, _idxs in groups:
                for wi, w in enumerate(wlist):
                    contexts.append(" ".join(wlist[:wi]))
                    targets.append(w)
            # CBR-RNN is a 256-unit custom RNN (not an HF AutoModel) and this
            # flat all-stories call never matches the per-section Stage-1 cache
            # anyway, so always let it compute inline — the `require_cache_hit`
            # guard exists only to stop huge transformers loading on a CPU node,
            # which does not apply here (loads/runs in seconds on CPU).
            surp_flat = batch_surprisal_cbr_rnn(
                mcfg.model_name, contexts, targets,
                device=dcfg.device, n_params=mcfg.n_params,
                cache_dir=dcfg.feature_cache_dir, require_cache_hit=False,
            )[:, 0]
            n_layers = 1
            pos = 0
            for _story, wlist, idxs in groups:
                n = len(wlist)
                seg = surp_flat[pos:pos + n]
                pos += n
                prev = np.full(n, np.nan, dtype=np.float32)
                prev[1:] = seg[:-1]
                for i, wi in enumerate(idxs):
                    surp_by_idx[wi] = np.array([seg[i]], dtype=np.float32)
                    prev_by_idx[wi] = np.array([prev[i]], dtype=np.float32)
        else:
            from ..features.logit_lens import logit_lens_surprisal_all_layers
            surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None
            kw = dict(
                revision=mcfg.model_revision, device=dcfg.device, max_length=dcfg.max_length,
                tokenizer_no_revision=dcfg.tokenizer_no_revision, n_params=mcfg.n_params,
                trust_remote_code=mcfg.trust_remote_code, cache_dir=dcfg.feature_cache_dir,
                lang=dcfg.lang, untrained=mcfg.untrained, untrained_seed=mcfg.untrained_seed,
                require_cache_hit=require_cache_hit, dtype=surp_dtype,
            )
            for story_idx, (story, wlist, idxs) in enumerate(groups):
                arr = logit_lens_surprisal_all_layers(wlist, mcfg.model_name, section_id=story_idx, **kw)
                surp = arr[:, :, 0]  # (n_words_in_story, n_layers)
                if n_layers is None:
                    n_layers = surp.shape[1]
                prev = np.full_like(surp, np.nan)
                prev[1:] = surp[:-1]
                for i, wi in enumerate(idxs):
                    surp_by_idx[wi] = surp[i]
                    prev_by_idx[wi] = prev[i]
        assert n_layers is not None
        order = words_df["word_idx"].astype(int).tolist()
        S = np.stack([surp_by_idx.get(w, np.full(n_layers, np.nan, np.float32)) for w in order]).astype(np.float32)
        P = np.stack([prev_by_idx.get(w, np.full(n_layers, np.nan, np.float32)) for w in order]).astype(np.float32)
        return S, P
