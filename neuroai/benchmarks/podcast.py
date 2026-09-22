"""Podcast ECoG encoding benchmark.

Predicts high-gamma power (70–200 Hz Hilbert envelope) from LLM hidden states
using per-electrode ridge regression with temporal 3-fold CV (contiguous chunks,
matching the standard approach for continuous podcast paradigms, Zada et al. 2025).

Pipeline (per subject)
----------------------
1. Load preprocessed high-gamma .fif from ``derivatives/ecogprep/``.
2. Resample to ``downsample_to`` Hz (default 32 Hz, matching Zada et al. 2025).
3. Keep only "good" channels from channels.tsv.
4. Extract word-locked epochs ``[tmin_extract, tmax]`` for all words in transcript.
5. IQR clipping (5×) on the raw Hilbert envelope.
6. No pre-onset baseline correction by default (``PodcastDataConfig.baseline
   = None``, matching the reference Hasson-lab tutorial and the pooled
   Podcast LME pipeline) — the outcome is z-scored per electrode on
   TRAIN-fold-only statistics inside the ridge CV loop instead (see
   ``PodcastDataConfig.baseline``'s docstring).
7. Crop to analysis window ``[tmin, tmax]`` (default −2 to +2 s).
8. Per-electrode lag-matrix ridge regression with temporal 3-fold CV.

Features
--------
LLM hidden states extracted via :func:`~neuroai.features.llm.contextual_word_embeddings_hf_fulltext`:
the full transcript is tokenized once and processed in overlapping fixed-length chunks
(one forward pass per chunk, not one per word), then each word's hidden state is
mean-pooled from the sub-tokens of the chunk where the word has maximum left context.
For models whose context window exceeds the transcript length (e.g. Llama-3.2, Mistral)
this reduces to a single forward pass.  Falls back to per-word extraction for models
with slow (non-fast) tokenizers.  All 9 subjects share the same transcript, so features
are extracted once and reused.

Outputs
-------
Per-checkpoint:
  podcast_checkpoint_scores.csv     — best-layer summary per model/checkpoint
  podcast_checkpoint_layerwise.csv  — per-layer Pearson r / Spearman r
  podcast_by_electrode.csv          — per-electrode scores (all subjects)
  podcast_by_subject.csv            — per-subject mean
  podcast_by_region.csv             — per-region mean (averaged across subjects)
  podcast_training_curve.png
  podcast_layer_heatmap.png
  podcast_brain_map.png             — scatter of MNI coords coloured by score
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..benchmarks.base import ModelConfig, RegressionConfig, word_predictions_dir
from ..datasets.podcast import (
    PodcastDataset,
    extract_word_epochs,
    highgamma_fif_path,
    load_transcript,
)
from ..features.cbr_rnn import contextual_word_embeddings_cbr_rnn
from ..features.llm import (
    contextual_word_embeddings_hf,
    contextual_word_embeddings_hf_fulltext,
)
from ..regressors.ridge import (
    CVScore,
    TimeResolvedScore,
    cv_ridge_score_lag_matrix,
    cv_ridge_score_lag_matrix_gpu_batched,
    cv_ridge_score_windowed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PodcastDataConfig:
    """Dataset location and epoch-extraction parameters."""
    bids_root: str
    subjects: list[str] | None = None
    # Epoch window (relative to word onset)
    tmin: float = -2.0          # analysis window start (s)
    tmax: float = 2.0           # analysis window end (s)
    tmin_extract: float = -2.1  # extraction start (needs 100 ms baseline margin)
    downsample_to: float = 32.0
    drop_bads: bool = True
    # Preprocessing
    clip_iqr: float | None = 5.0
    # No pre-onset baseline correction by default — matches the reference
    # Hasson-lab tutorial (mne.Epochs(..., baseline=None)) and the pooled
    # Podcast LME pipeline (regressors/podcast_lme.py). The outcome is
    # instead z-scored per electrode using TRAIN-fold-only statistics inside
    # the ridge CV loop (``standardize_target=True`` in
    # ``cv_ridge_score_lag_matrix`` / ``cv_ridge_score_windowed``, always on
    # for this benchmark — see reg_kwargs below), so no test-fold
    # information leaks into the normalization, same guarantee the LME's
    # ``_zscore_y_by_group_train_test`` gives.
    baseline: tuple[float, float] | None = None
    baseline_mode: str = "logratio"
    # LLM feature options — None means use the full preceding context up to the
    # model's own context length (no word-level cap applied).
    context_words: int | None = None
    max_length: int | None = None
    feature_cache_dir: str | None = None
    # Path to output dir of data/podcast/electrode_selection.py.
    # When set, only channels in selected_electrodes.csv (selected=True) are kept.
    electrode_selection_dir: str | None = None
    # wordfreq lexical-frequency baseline language (only used when
    # PodcastBenchmarkConfig.compute_unique_surprisal=True).
    wordfreq_language: str = "en"


@dataclass
class PodcastBenchmarkConfig:
    """Top-level config for the Podcast ECoG encoding benchmark."""
    data: PodcastDataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    output_dir: str = "outputs"
    layerwise: bool = True
    # "embeddings" — full LLM hidden-state vectors (existing pipeline).
    # "surprisal" — per-layer logit-lens surprisal (Kuribayashi et al. 2025,
    # arXiv:2502.01615): a single scalar per (word, layer), reusing the model's
    # own final norm + unembedding matrix at every layer. Forces layerwise
    # regression regardless of the `layerwise` flag above (there's no "last
    # layer only" concept for logit-lens — the whole point is comparing layers).
    feature_mode: tp.Literal["embeddings", "surprisal"] = "embeddings"
    # Lag-matrix: predict all time samples simultaneously (recommended).
    # Falls back to windowed-mean regression when False.
    use_lag_matrix: bool = True
    # Number of contiguous CV folds (splits the word sequence into equal chunks).
    n_cv_folds: int = 3
    # Fallback windowed-mean parameters (only used when use_lag_matrix=False)
    window_size: float = 0.2
    step_size: float = 0.05
    n_jobs: int = 80
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    # Device for ridge regression: "cuda" uses himalaya GPU backend, "cpu" uses sklearn
    regression_device: str = "cpu"
    # Number of layers to fit in parallel (GPU threads). Each thread runs one
    # himalaya call; the GPU scheduler overlaps them. 36 covers all layers of
    # the largest Pythia/OLMo models on 80 GB H100. Reduce if OOM.
    n_layer_workers: int = 36
    # Save the best-layer out-of-fold predicted lag matrix per electrode
    # (n_words × n_times, float32) alongside the usual scores — needed for
    # post-hoc per-word difficulty splits (e.g. DLT cost). Off by default:
    # adds one .npz per subject per model (all electrodes stacked).
    save_word_predictions: bool = False
    # Only meaningful when feature_mode="surprisal". Fits BOLD ~ [wlen,
    # freq] (baseline) and BOLD ~ [wlen, freq, surprisal_layer]
    # (full) per layer, reporting the delta held-out Pearson r as surprisal's
    # UNIQUE contribution once word length + lexical frequency are already
    # accounted for. This is the confound Oh et al. (2024) propose for the
    # inverse-scaling effect (larger models' surprisal tracks frequency more
    # tightly) — lets you check whether a positive raw-correlation scaling
    # trend survives controlling for it, or is explained by it. Adds
    # podcast_unique_surprisal_by_electrode.csv.
    compute_unique_surprisal: bool = False


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ElectrodeScore:
    """Encoding score for one electrode."""
    subject: str
    name: str
    x: float | None
    y: float | None
    z: float | None
    region: str | None
    hemisphere: str | None
    score: CVScore                            # best-layer or flat
    time_course: TimeResolvedScore | None = None
    layer_scores: list[CVScore] | None = None
    layer_time_courses: list[TimeResolvedScore] | None = None

    def best_layer_index(self, metric: str = "pearsonr") -> int | None:
        if self.layer_scores:
            vals = np.asarray([getattr(s, metric) for s in self.layer_scores], dtype=float)
            if vals.size > 0 and np.any(np.isfinite(vals)):
                return int(np.nanargmax(vals))
        return None

    def best_layer_pearsonr(self) -> float:
        if self.layer_scores:
            vals = np.asarray([s.pearsonr for s in self.layer_scores], dtype=float)
            return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else float("nan")
        return self.score.pearsonr

    def best_layer_spearmanr(self) -> float:
        if self.layer_scores:
            vals = np.asarray([s.spearmanr for s in self.layer_scores], dtype=float)
            return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else float("nan")
        return self.score.spearmanr

    def timecourse_rows(self) -> list[dict]:
        """One dict per time bin with electrode metadata + Pearson/Spearman r."""
        if self.time_course is None:
            return []
        tc = self.time_course
        base = {
            "subject": self.subject,
            "electrode": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "region": self.region,
            "hemisphere": self.hemisphere,
        }
        return [
            {**base, "time_s": float(tc.window_centers[t]),
             "pearsonr": float(tc.pearsonr[t]),
             "spearmanr": float(tc.spearmanr[t])}
            for t in range(len(tc.window_centers))
        ]

    def to_dict(self, include_layers: bool = False) -> dict:
        d: dict[str, tp.Any] = {
            "subject": self.subject,
            "electrode": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "region": self.region,
            "hemisphere": self.hemisphere,
            **self.score.to_dict(),
            "best_pearsonr": self.best_layer_pearsonr(),
            "best_spearmanr": self.best_layer_spearmanr(),
            "best_layer": self.best_layer_index("pearsonr"),
        }
        if self.time_course is not None:
            d["peak_pearsonr"] = self.time_course.peak_score("pearsonr")
            d["peak_spearmanr"] = self.time_course.peak_score("spearmanr")
            d["peak_time_s"] = self.time_course.peak_time("pearsonr")
        if include_layers and self.layer_scores:
            for li, ls in enumerate(self.layer_scores):
                d[f"layer_{li}_pearsonr"] = ls.pearsonr
                d[f"layer_{li}_spearmanr"] = ls.spearmanr
        return d


@dataclass
class PodcastEncoderResult:
    """Aggregated results for one model checkpoint."""
    model_name: str
    model_revision: str | None
    n_words: int
    electrode_scores: list[ElectrodeScore]
    n_subjects_expected: int = 0  # total subjects in dataset; 0 = unknown

    def mean_pearsonr(self) -> float:
        vals = [e.best_layer_pearsonr() for e in self.electrode_scores]
        return float(np.nanmean(vals)) if vals else float("nan")

    def mean_spearmanr(self) -> float:
        vals = [e.best_layer_spearmanr() for e in self.electrode_scores]
        return float(np.nanmean(vals)) if vals else float("nan")

    def to_dataframe(self, include_layers: bool = False) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict(include_layers=include_layers)
                             for e in self.electrode_scores])

    def layerwise_matrix(self, metric: str = "pearsonr") -> np.ndarray | None:
        if not self.electrode_scores or self.electrode_scores[0].layer_scores is None:
            return None
        n_layers = max(
            len(e.layer_scores) for e in self.electrode_scores if e.layer_scores
        )
        mat = np.full((len(self.electrode_scores), n_layers), np.nan)
        for i, e in enumerate(self.electrode_scores):
            if e.layer_scores:
                for li, s in enumerate(e.layer_scores):
                    mat[i, li] = getattr(s, metric)
        return mat

    def electrode_timecourse_df(self) -> pd.DataFrame:
        """Long-form time courses: one row per (electrode × time_bin).

        Columns: subject, electrode, region, hemisphere, x, y, z, time_s,
        pearsonr, spearmanr.  Useful for plotting per-electrode or per-region
        temporal profiles.
        """
        rows = []
        for e in self.electrode_scores:
            rows.extend(e.timecourse_rows())
        return pd.DataFrame(rows)

    def mean_timecourse_df(self) -> pd.DataFrame:
        """Mean ± SEM brain score per time bin, averaged across all electrodes.

        Columns: time_s, mean_pearsonr, sem_pearsonr, std_pearsonr,
        mean_spearmanr, sem_spearmanr, std_spearmanr, n_electrodes.
        """
        tc_df = self.electrode_timecourse_df()
        if tc_df.empty:
            return pd.DataFrame()
        agg = (
            tc_df.groupby("time_s")[["pearsonr", "spearmanr"]]
            .agg(["mean", "std", "count"])
        )
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        agg = agg.reset_index()
        n = agg["pearsonr_count"]
        agg["sem_pearsonr"] = agg["pearsonr_std"] / np.sqrt(n.clip(lower=1))
        agg["sem_spearmanr"] = agg["spearmanr_std"] / np.sqrt(n.clip(lower=1))
        agg = agg.rename(columns={
            "pearsonr_mean": "mean_pearsonr",
            "pearsonr_std": "std_pearsonr",
            "pearsonr_count": "n_electrodes",
            "spearmanr_mean": "mean_spearmanr",
            "spearmanr_std": "std_spearmanr",
            "spearmanr_count": "n_electrodes_sp",
        }).drop(columns=["n_electrodes_sp"])
        return agg[["time_s", "mean_pearsonr", "sem_pearsonr", "std_pearsonr",
                    "mean_spearmanr", "sem_spearmanr", "std_spearmanr", "n_electrodes"]]

    def by_subject(self) -> pd.DataFrame:
        df = self.to_dataframe()
        cols = ["best_pearsonr", "best_spearmanr", "pearsonr", "spearmanr"]
        agg = df.groupby("subject")[cols].agg(["mean", "std", "count"])
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        return agg.reset_index()

    def by_region(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if "region" not in df.columns or df["region"].isna().all():
            return pd.DataFrame()
        cols = ["best_pearsonr", "best_spearmanr", "pearsonr", "spearmanr"]
        agg = (
            df.dropna(subset=["region"])
            .groupby("region")[cols]
            .agg(["mean", "std", "count"])
        )
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        return agg.reset_index()

    def by_hemisphere(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if "hemisphere" not in df.columns or df["hemisphere"].isna().all():
            return pd.DataFrame()
        cols = ["best_pearsonr", "best_spearmanr", "pearsonr", "spearmanr"]
        agg = (
            df.dropna(subset=["hemisphere"])
            .groupby("hemisphere")[cols]
            .agg(["mean", "std", "count"])
        )
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        return agg.reset_index()


# ---------------------------------------------------------------------------
# Ridge alpha diagnostics
# ---------------------------------------------------------------------------

def _log_chosen_alphas(
    subject: str,
    results: list["ElectrodeScore"],
    alpha_grid: list[float],
    print_chosen: bool,
) -> None:
    """Print the distribution of ridge alphas actually selected across electrodes.

    Uses each electrode's *best* layer (the one its reported score comes
    from) when layerwise, matching what ``print_chosen_ridge_alpha`` already
    does for the other ridge regression paths — this option was
    accepted into ``RegressionConfig`` and threaded through the Podcast YAML
    config, but ``podcast.py`` never actually read it or logged anything.
    """
    if not print_chosen:
        return
    alphas: list[float] = []
    for r in results:
        if r.layer_scores:
            li = r.best_layer_index("pearsonr")
            a = r.layer_scores[li].ridge_alpha if li is not None else r.score.ridge_alpha
        else:
            a = r.score.ridge_alpha
        if np.isfinite(a):
            alphas.append(float(a))
    if not alphas:
        return

    lo, hi = min(alpha_grid), max(alpha_grid)
    med = float(np.nanmedian(alphas))
    p25, p75 = float(np.percentile(alphas, 25)), float(np.percentile(alphas, 75))
    at_boundary = sum(1 for a in alphas if a <= lo * 1.0001 or a >= hi * 0.9999)
    msg = (
        f"  [{subject}] Ridge chosen alpha (best layer per electrode): "
        f"median={med:.6g} (25-75 pct: {p25:.6g}-{p75:.6g}, n={len(alphas)}), "
        f"grid=[{lo:.3g}, {hi:.3g}]"
    )
    if at_boundary:
        msg += (
            f" — {at_boundary}/{len(alphas)} electrodes picked a boundary alpha; "
            "consider widening the grid"
        )
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Per-electrode regression worker (module-level for joblib pickling)
# ---------------------------------------------------------------------------

def _regress_electrode(
    ci: int,
    emb: np.ndarray,               # (n_valid_words, [n_layers,] D)
    y_ci: np.ndarray,              # (n_valid_words, n_times)
    times: np.ndarray,             # (n_times,) seconds relative to onset
    groups: np.ndarray,            # (n_valid_words,)
    ch_name: str,
    meta: dict,
    subject: str,
    use_lag_matrix: bool,
    window_size: float,
    step_size: float,
    reg_kwargs: dict,
    save_word_predictions: bool = False,
) -> ElectrodeScore:
    layerwise = emb.ndim == 3

    def _best_li(scores: list[CVScore]) -> int:
        vals = np.asarray([s.pearsonr for s in scores], dtype=float)
        if vals.size == 0 or not np.any(np.isfinite(vals)):
            return 0
        return int(np.nanargmax(vals))

    rk = {k: v for k, v in reg_kwargs.items()
          if k not in ("window_size", "step_size", "windowed", "use_lag_matrix")}

    if use_lag_matrix:
        # Lag-matrix: one multi-output ridge maps LLM features → all time samples
        # simultaneously, giving a TRF at full sampling resolution (e.g. 10 ms at 100 Hz).
        if layerwise:
            n_layers = emb.shape[1]
            layer_tcs: list[TimeResolvedScore] = []
            for li in range(n_layers):
                tc = cv_ridge_score_lag_matrix(
                    emb[:, li, :], y_ci, times,
                    groups=groups, **rk,
                )
                layer_tcs.append(tc)
            layer_scores = [tc.to_mean_cvScore() for tc in layer_tcs]
            best_li = _best_li(layer_scores)
            score = layer_scores[-1]           # last layer as flat baseline
            time_course: TimeResolvedScore | None = layer_tcs[best_li]
            layer_time_courses: list[TimeResolvedScore] | None = layer_tcs
            if save_word_predictions:
                # Refit only the winning layer with OOF predictions kept — avoids
                # holding a (n_words × n_times) array per layer in memory/CSV.
                time_course = cv_ridge_score_lag_matrix(
                    emb[:, best_li, :], y_ci, times,
                    groups=groups, return_oof_predictions=True, **rk,
                )
        else:
            tc = cv_ridge_score_lag_matrix(
                emb, y_ci, times, groups=groups,
                return_oof_predictions=save_word_predictions, **rk,
            )
            score = tc.to_mean_cvScore()
            time_course = tc
            layer_scores = None
            layer_time_courses = None
    else:
        # Fallback: windowed-mean regression 
        if layerwise:
            n_layers = emb.shape[1]
            layer_tcs_w: list[TimeResolvedScore] = []
            for li in range(n_layers):
                tc = cv_ridge_score_windowed(
                    emb[:, li, :], y_ci, times,
                    window_size=window_size, step_size=step_size,
                    groups=groups, **rk,
                )
                layer_tcs_w.append(tc)
            layer_scores = [tc.to_mean_cvScore() for tc in layer_tcs_w]
            best_li = _best_li(layer_scores)
            score = layer_scores[-1]
            time_course = layer_tcs_w[best_li]
            layer_time_courses = layer_tcs_w
        else:
            tc = cv_ridge_score_windowed(
                emb, y_ci, times,
                window_size=window_size, step_size=step_size,
                groups=groups, **rk,
            )
            score = tc.to_mean_cvScore()
            time_course = tc
            layer_scores = None
            layer_time_courses = None

    return ElectrodeScore(
        subject=subject,
        name=ch_name,
        x=meta.get("x"),
        y=meta.get("y"),
        z=meta.get("z"),
        region=meta.get("region"),
        hemisphere=meta.get("hemisphere"),
        score=score,
        time_course=time_course,
        layer_scores=layer_scores,
        layer_time_courses=layer_time_courses,
    )


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class PodcastEcogEncoder:
    """Predict Podcast ECoG high-gamma power from LLM hidden states.

    Parameters
    ----------
    cfg : PodcastBenchmarkConfig
    """

    def __init__(self, cfg: PodcastBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(self) -> PodcastEncoderResult:
        words_df = self._load_transcript()
        words = words_df["word"].astype(str).tolist()
        onsets = words_df["onset"].values.astype(np.float64)

        emb = self._extract_features(words)
        logger.info(
            "[%s] features shape: %s", self.cfg.model.model_name,
            "×".join(str(d) for d in emb.shape),
        )

        # Logit-lens surprisal is undefined for the very first word (no left
        # context) and is NaN there in every layer by construction (see
        # ``logit_lens_surprisal_all_layers``). Left in place, that single NaN
        # row poisons every CV fold downstream (RobustScaler doesn't remove
        # NaNs, and a NaN training or test row makes
        # ``cv_ridge_score_lag_matrix_gpu_batched`` skip the whole fold — for a
        # 1-feature column, the fold containing that row in *either* split is
        # unusable), so drop it here before it ever reaches per-subject
        # regression rather than losing entire folds to it.
        feat_finite = np.isfinite(emb.reshape(emb.shape[0], -1)).all(axis=1)
        if not feat_finite.all():
            n_dropped = int((~feat_finite).sum())
            logger.warning(
                "[%s] dropping %d word(s) with non-finite features "
                "(e.g. no-left-context word 0) before regression: indices %s",
                self.cfg.model.model_name, n_dropped,
                np.where(~feat_finite)[0].tolist(),
            )
            emb = emb[feat_finite]
            words = [w for w, keep in zip(words, feat_finite) if keep]
            onsets = onsets[feat_finite]

        n_words_feat = emb.shape[0]

        baseline_feats: np.ndarray | None = None
        if self.cfg.feature_mode == "surprisal" and self.cfg.compute_unique_surprisal:
            baseline_feats = self._compute_baseline_covariates(words)

        ds = PodcastDataset(
            self.cfg.data.bids_root,
            subjects=self.cfg.data.subjects,
            drop_bads=self.cfg.data.drop_bads,
        )
        subjects = ds.available_subjects()
        if not subjects:
            raise RuntimeError(
                f"No subjects with highgamma .fif found under {self.cfg.data.bids_root}"
            )
        logger.info("[%s] running on %d subjects: %s",
                    self.cfg.model.model_name, len(subjects), subjects)

        # Prepare incremental output paths — written after each subject so
        # results are available for analysis before all subjects finish.
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        elec_csv = out_dir / "podcast_by_electrode.csv"
        model_revision = self.cfg.model.model_revision or "main"

        # Load already-completed subjects from the incremental CSV so we can skip them.
        done_subjects: set[str] = set()
        if elec_csv.exists():
            try:
                _existing = pd.read_csv(elec_csv, usecols=["subject", "revision"])
                done_subjects = set(
                    _existing.loc[_existing["revision"] == model_revision, "subject"].unique()
                )
                if done_subjects:
                    logger.info("Skipping %d already-completed subject(s): %s",
                                len(done_subjects), sorted(done_subjects))
            except Exception:
                logger.warning("Could not read existing electrode CSV; re-running all subjects.",
                               exc_info=True)

        all_scores: list[ElectrodeScore] = []
        subj_csv = out_dir / "podcast_by_subject.csv"
        reg_csv = out_dir / "podcast_by_region.csv"
        elec_tc_csv = out_dir / "podcast_electrode_timecourse.csv"
        elec_lw_csv = out_dir / "podcast_by_electrode_layerwise.csv"
        unique_csv = out_dir / "podcast_unique_surprisal_by_electrode.csv"

        for subj in subjects:
            if subj in done_subjects:
                logger.info("--- Subject: %s --- (skipped, already in CSV)", subj)
                continue
            logger.info("--- Subject: %s ---", subj)
            try:
                scores, unique_rows = self._run_subject(
                    subj, ds, emb, onsets, words=words, baseline_feats=baseline_feats,
                )
                all_scores.extend(scores)

                if unique_rows:
                    u_df = pd.DataFrame(unique_rows)
                    u_df["revision"] = model_revision
                    u_df.to_csv(unique_csv, mode="a", header=not unique_csv.exists(), index=False)
                    logger.info("  Saved %d unique-surprisal-contribution rows → %s",
                                len(u_df), unique_csv)

                if scores:
                    mean_r = float(np.nanmean([e.best_layer_pearsonr() for e in scores]))
                    logger.info("  %d electrodes, mean best-layer Pearson r = %.4f",
                                len(scores), mean_r)

                    # ---- flush electrode rows ----
                    rows = [e.to_dict(include_layers=False) for e in scores]
                    for r in rows:
                        r["revision"] = model_revision
                    subj_df = pd.DataFrame(rows)
                    subj_df.to_csv(elec_csv, mode="a", header=not elec_csv.exists(), index=False)
                    logger.info("  Saved %d electrode rows → %s", len(subj_df), elec_csv)

                    # ---- per-subject summary row ----
                    score_cols = ["best_pearsonr", "best_spearmanr", "pearsonr", "spearmanr"]
                    subj_agg = subj_df[score_cols].agg(["mean", "std", "count"])
                    subj_row: dict = {"subject": subj, "revision": model_revision}
                    for col in score_cols:
                        subj_row[f"{col}_mean"] = subj_agg.loc["mean", col]
                        subj_row[f"{col}_std"] = subj_agg.loc["std", col]
                        subj_row[f"{col}_count"] = int(subj_agg.loc["count", col])
                    pd.DataFrame([subj_row]).to_csv(
                        subj_csv, mode="a", header=not subj_csv.exists(), index=False
                    )

                    # ---- per-region summary rows ----
                    if "region" in subj_df.columns and subj_df["region"].notna().any():
                        reg_agg = (
                            subj_df.dropna(subset=["region"])
                            .groupby("region")[score_cols]
                            .agg(["mean", "std", "count"])
                        )
                        reg_agg.columns = [f"{c}_{s}" for c, s in reg_agg.columns]
                        reg_agg = reg_agg.reset_index()
                        reg_agg.insert(0, "subject", subj)
                        reg_agg.insert(1, "revision", model_revision)
                        reg_agg.to_csv(
                            reg_csv, mode="a", header=not reg_csv.exists(), index=False
                        )

                    # ---- per-electrode timecourse rows ----
                    tc_rows = []
                    for e in scores:
                        tc_rows.extend(e.timecourse_rows())
                    if tc_rows:
                        tc_df = pd.DataFrame(tc_rows)
                        tc_df["revision"] = model_revision
                        tc_df.to_csv(
                            elec_tc_csv, mode="a",
                            header=not elec_tc_csv.exists(), index=False,
                        )

                    # ---- per-electrode per-layer scores ----
                    lw_rows = []
                    for e in scores:
                        if e.layer_scores:
                            for li, ls in enumerate(e.layer_scores):
                                lw_rows.append({
                                    "subject": subj,
                                    "electrode": e.name,
                                    "revision": model_revision,
                                    "layer": li,
                                    "pearsonr": ls.pearsonr,
                                    "spearmanr": ls.spearmanr,
                                })
                    if lw_rows:
                        lw_df = pd.DataFrame(lw_rows)
                        lw_df.to_csv(
                            elec_lw_csv, mode="a",
                            header=not elec_lw_csv.exists(), index=False,
                        )

            except Exception:
                logger.error("Subject %s failed", subj, exc_info=True)

        # ---- mean timecourse across all subjects (read from CSV to include skipped ones) ----
        if elec_tc_csv.exists():
            try:
                tc_all = pd.read_csv(elec_tc_csv)
                tc_all = tc_all[tc_all["revision"] == model_revision]
                if not tc_all.empty:
                    mean_tc = (
                        tc_all.groupby("time_s")
                        .agg(
                            pearsonr=("pearsonr", "mean"),
                            sem_pearsonr=("pearsonr", "sem"),
                            spearmanr=("spearmanr", "mean"),
                            sem_spearmanr=("spearmanr", "sem"),
                            n_electrodes=("pearsonr", "count"),
                        )
                        .reset_index()
                    )
                    mean_tc["revision"] = model_revision
                    mean_tc_csv = out_dir / "podcast_mean_timecourse.csv"
                    mean_tc.to_csv(
                        mean_tc_csv, mode="a",
                        header=not mean_tc_csv.exists(), index=False,
                    )
                    logger.info("Saved mean time course (%d bins) → %s",
                                len(mean_tc), mean_tc_csv)
            except Exception:
                logger.warning("Mean timecourse computation failed", exc_info=True)

        return PodcastEncoderResult(
            model_name=self.cfg.model.model_name,
            model_revision=self.cfg.model.model_revision,
            n_words=n_words_feat,
            electrode_scores=all_scores,
            n_subjects_expected=len(subjects),
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _load_transcript(self) -> pd.DataFrame:
        return load_transcript(self.cfg.data.bids_root)

    def _compute_baseline_covariates(self, words: list[str]) -> np.ndarray:
        """Word length + wordfreq lexical frequency — the confound Oh et al.
        (2024) propose for the inverse-scaling effect.

        Returns
        -------
        np.ndarray  shape (n_words, 2): [wlen, zipf_freq]
        """
        from ..features.word_frequency import compute_wlen_wordfreq_covariates

        return compute_wlen_wordfreq_covariates(words, language=self.cfg.data.wordfreq_language)

    def _compute_unique_surprisal_contribution(
        self,
        subject: str,
        emb: np.ndarray,             # (n, n_layers, 1) surprisal, one layer's worth
        baseline_feats: np.ndarray,  # (n, 2) [wlen, zipf_freq]
        epochs: np.ndarray,          # (n, C, T)
        times: np.ndarray,
        groups: np.ndarray,
        electrode_meta: dict[str, dict],
    ) -> list[dict]:
        """Fit the [wlen, freq] baseline once, then [wlen, freq, surprisal]
        per layer; report delta held-out Pearson r as surprisal's unique
        contribution over the baseline. Every layer's delta is saved (not
        averaged/dropped) so a scaling-law-vs-model-size plot can be built
        post-hoc, matching the rest of this pipeline's "save everything"
        convention.
        """
        rcfg = self.cfg.regression
        n_channels = epochs.shape[1]
        n_layers = emb.shape[1]
        channel_names = list(electrode_meta.keys())

        reg_kwargs = dict(
            n_splits=self.cfg.n_cv_folds,
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            standardize_target=True,
            random_state=rcfg.random_state,
        )

        def _fit(X: np.ndarray) -> list[TimeResolvedScore]:
            if self.cfg.regression_device != "cpu":
                from himalaya.backend import set_backend as _set_backend
                _set_backend("torch_cuda")
                return cv_ridge_score_lag_matrix_gpu_batched(
                    X, epochs, times, groups=groups, **reg_kwargs
                )
            return [
                cv_ridge_score_lag_matrix(
                    X, epochs[:, ci, :], times, groups=groups, device="cpu", **reg_kwargs
                )
                for ci in range(n_channels)
            ]

        baseline_tcs = _fit(baseline_feats)
        baseline_r = [float(np.nanmean(tc.pearsonr)) for tc in baseline_tcs]

        rows: list[dict] = []
        for li in range(n_layers):
            X_full = np.concatenate([baseline_feats, emb[:, li, :]], axis=1)
            full_tcs = _fit(X_full)
            for ci, ch_name in enumerate(channel_names):
                full_r = float(np.nanmean(full_tcs[ci].pearsonr))
                meta = electrode_meta.get(ch_name, {})
                rows.append({
                    "subject": subject,
                    "electrode": ch_name,
                    "x": meta.get("x"), "y": meta.get("y"), "z": meta.get("z"),
                    "region": meta.get("region"), "hemisphere": meta.get("hemisphere"),
                    "layer": li,
                    "baseline_pearsonr": baseline_r[ci],
                    "full_pearsonr": full_r,
                    "delta_pearsonr": full_r - baseline_r[ci],
                })
        logger.info(
            "  [%s] unique-surprisal-contribution: %d layers × %d electrodes, "
            "mean delta Pearson r = %.4f",
            subject, n_layers, n_channels,
            float(np.nanmean([r["delta_pearsonr"] for r in rows])) if rows else float("nan"),
        )
        return rows

    def _extract_features(self, words: list[str]) -> np.ndarray:
        dcfg = self.cfg.data
        mcfg = self.cfg.model
        layer_index: str | int = "all" if self.cfg.layerwise else "last"
        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        if self.cfg.feature_mode == "surprisal":
            if "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower():
                # CBR-RNN is a custom RNN, not a transformer — it can't reuse
                # the sliding-window transformer forward pass logit-lens uses
                # for fulltext extraction. Instead, same bounded moving-window
                # context (last 48 words) as batch_surprisal_cbr_rnn already
                # uses for Natural Stories (natural_stories_lme.py), fed
                # word-by-word — cheap for this ~17M-param model even over a
                # narrative-length transcript.
                from ..datasets.podcast import build_word_context
                from ..features.cbr_rnn import batch_surprisal_cbr_rnn

                contexts = build_word_context(words)
                surp = batch_surprisal_cbr_rnn(
                    mcfg.model_name, contexts, words,
                    device=mcfg.device, n_params=mcfg.n_params,
                )  # (n_words, 1)
                return surp.reshape(len(words), 1, 1).astype(np.float32)
            from ..features.logit_lens import logit_lens_surprisal_all_layers

            return logit_lens_surprisal_all_layers(
                words,
                mcfg.model_name,
                revision=mcfg.model_revision,
                device=mcfg.device,
                max_length=dcfg.max_length,
                cache_dir=dcfg.feature_cache_dir,
                section_id=0,
                lang="en",
                tokenizer_no_revision=self.cfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=self.cfg.trust_remote_code,
                untrained=mcfg.untrained,
                untrained_seed=mcfg.untrained_seed,
                dtype=surp_dtype,
            )  # (n_words, n_layers, 1)

        if "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower():
            if mcfg.untrained:
                raise NotImplementedError(
                    "untrained baseline not supported for CBR-RNN (custom architecture)."
                )
            # CBR-RNN is a custom RNN (not a HuggingFace AutoModel) with a single
            # hidden-state layer — no layer dimension, shape (n_words, d_hid).
            return contextual_word_embeddings_cbr_rnn(
                words,
                mcfg.model_name,
                device=mcfg.device,
                context_words=dcfg.context_words if dcfg.context_words is not None else 48,
                cache_dir=dcfg.feature_cache_dir,
                section_id=0,
                lang="en",
            )

        try:
            emb = contextual_word_embeddings_hf_fulltext(
                words,
                mcfg.model_name,
                layer_index=layer_index,
                revision=mcfg.model_revision,
                device=mcfg.device,
                max_length=dcfg.max_length,
                cache_dir=dcfg.feature_cache_dir,
                section_id=0,
                lang="en",
                tokenizer_no_revision=self.cfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=self.cfg.trust_remote_code,
                untrained=mcfg.untrained,
                untrained_seed=mcfg.untrained_seed,
                dtype=surp_dtype,
            )
        except ValueError as exc:
            if "fast tokenizer" not in str(exc):
                raise
            logger.warning(
                "Falling back to per-word extraction (slow tokenizer): %s", exc
            )
            emb = contextual_word_embeddings_hf(
                words,
                mcfg.model_name,
                layer_index=layer_index,
                revision=mcfg.model_revision,
                device=mcfg.device,
                context_words=dcfg.context_words,
                max_length=dcfg.max_length,
                batch_size=mcfg.batch_size,
                cache_dir=dcfg.feature_cache_dir,
                section_id=0,
                lang="en",
                tokenizer_no_revision=self.cfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=self.cfg.trust_remote_code,
                untrained=mcfg.untrained,
                untrained_seed=mcfg.untrained_seed,
                dtype=surp_dtype,
            )
        return emb  # (n_words, [n_layers,] D)

    # ------------------------------------------------------------------
    # Per-subject pipeline
    # ------------------------------------------------------------------

    def _load_subject_epochs(self, subject: str, ds: PodcastDataset, word_onsets: np.ndarray):
        return load_subject_epochs(
            ds, subject, word_onsets,
            tmin=self.cfg.data.tmin, tmax=self.cfg.data.tmax,
            tmin_extract=self.cfg.data.tmin_extract,
            downsample_to=self.cfg.data.downsample_to,
            clip_iqr=self.cfg.data.clip_iqr,
            baseline=self.cfg.data.baseline,
            baseline_mode=self.cfg.data.baseline_mode,
            electrode_selection_dir=self.cfg.data.electrode_selection_dir,
        )

    def _run_subject(
        self,
        subject: str,
        ds: PodcastDataset,
        emb: np.ndarray,
        word_onsets: np.ndarray,
        words: list[str] | None = None,
        baseline_feats: np.ndarray | None = None,
    ) -> tuple[list[ElectrodeScore], list[dict]]:
        dcfg = self.cfg.data

        loaded = self._load_subject_epochs(subject, ds, word_onsets)
        if loaded is None:
            return [], []
        epochs, valid_mask, analysis_times, electrode_meta = loaded

        n_valid = int(valid_mask.sum())
        emb_valid = emb[valid_mask]  # (n_valid, [L,] D)

        # ---- contiguous k-fold CV groups (avoids temporal data leakage) ----
        n_cv = self.cfg.n_cv_folds
        chunk = n_valid // n_cv
        groups = np.zeros(n_valid, dtype=int)
        for k in range(1, n_cv):
            groups[k * chunk:] = k
        group_sizes = np.bincount(groups)
        logger.info(
            "  CV groups: %d contiguous folds, sizes=%s (words per fold)",
            n_cv, group_sizes.tolist(),
        )

        # ---- per-electrode regression ----
        scores = self._regress_electrodes(
            subject, emb_valid, epochs, analysis_times, groups, electrode_meta,
        )

        if self.cfg.save_word_predictions and words is not None:
            self._save_word_predictions(
                subject, scores,
                words=[w for w, keep in zip(words, valid_mask) if keep],
                word_idx=np.where(valid_mask)[0],
                onsets=word_onsets[valid_mask],
                times=analysis_times,
                epochs=epochs,
                channel_names=list(electrode_meta.keys()),
            )

        unique_rows: list[dict] = []
        if baseline_feats is not None:
            baseline_valid = baseline_feats[valid_mask]
            unique_rows = self._compute_unique_surprisal_contribution(
                subject, emb_valid, baseline_valid, epochs, analysis_times,
                groups, electrode_meta,
            )

        return scores, unique_rows

    def _save_word_predictions(
        self,
        subject: str,
        scores: list[ElectrodeScore],
        *,
        words: list[str],
        word_idx: np.ndarray,
        onsets: np.ndarray,
        times: np.ndarray,
        epochs: np.ndarray,
        channel_names: list[str],
    ) -> None:
        """Save each electrode's best-layer out-of-fold predicted lag matrix
        AND the actual observed iEEG (needed to score any post-hoc word subset,
        e.g. an easy/hard DLT-cost split — a prediction alone isn't scoreable).

        One compressed ``.npz`` per (subject, model revision): all electrodes
        stacked as ``(n_electrodes, n_words, n_times)`` for both ``oof_predictions``
        and ``y_true``, plus the word text / onset / electrode-name / time-axis
        metadata needed to join against a per-word DLT-cost table later.
        """
        electrodes_with_oof = [
            e for e in scores
            if e.time_course is not None and e.time_course.oof_predictions is not None
        ]
        if not electrodes_with_oof:
            logger.warning("  save_word_predictions=True but no OOF predictions captured for %s", subject)
            return

        n_words = len(words)
        stacked = np.stack(
            [e.time_course.oof_predictions for e in electrodes_with_oof], axis=0
        )  # (n_electrodes, n_words, n_times)
        if stacked.shape[1] != n_words:
            logger.warning(
                "  Word-count mismatch for %s (%d words vs %d OOF rows) — skipping save",
                subject, n_words, stacked.shape[1],
            )
            return

        name_to_idx = {name: i for i, name in enumerate(channel_names)}
        y_true = np.stack(
            [epochs[:, name_to_idx[e.name], :] for e in electrodes_with_oof], axis=0
        ).astype(np.float32)  # (n_electrodes, n_words, n_times)

        out_dir = word_predictions_dir(self.cfg.output_dir, "podcast_word_predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (self.cfg.model.model_revision or "main").replace("/", "_")
        safe_model = self.cfg.model.model_name.replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}_{subject}.npz"
        np.savez_compressed(
            out_path,
            oof_predictions=stacked,
            y_true=y_true,
            electrode_names=np.array([e.name for e in electrodes_with_oof]),
            words=np.array(words),
            word_idx=word_idx,
            onsets=onsets,
            times=times,
            model=np.array(self.cfg.model.model_name),
            revision=np.array(self.cfg.model.model_revision or "main"),
            n_params=np.array(self.cfg.model.n_params),
        )
        logger.info("  Saved word-level predictions (%s) → %s", stacked.shape, out_path)

    def _regress_electrodes(
        self,
        subject: str,
        emb: np.ndarray,      # (n, [L,] D)
        ieeg: np.ndarray,     # (n, C, T)
        times: np.ndarray,    # (T,)
        groups: np.ndarray,   # (n,)
        electrode_meta: dict[str, dict],
    ) -> list[ElectrodeScore]:
        if self.cfg.regression_device != "cpu" and self.cfg.use_lag_matrix:
            return self._regress_electrodes_gpu_batched(
                subject, emb, ieeg, times, groups, electrode_meta
            )
        return self._regress_electrodes_cpu(
            subject, emb, ieeg, times, groups, electrode_meta
        )

    def _regress_electrodes_gpu_batched(
        self,
        subject: str,
        emb: np.ndarray,      # (n, [L,] D)
        ieeg: np.ndarray,     # (n, C, T)
        times: np.ndarray,    # (T,)
        groups: np.ndarray,   # (n,)
        electrode_meta: dict[str, dict],
    ) -> list[ElectrodeScore]:
        """Fit all electrodes in one himalaya GPU call per (layer, fold).

        Eliminates the 80-worker joblib → GPU contention that causes OOM when
        many electrodes all try to allocate CUDA tensors simultaneously.
        """
        rcfg = self.cfg.regression
        n_channels = ieeg.shape[1]
        n_lags = ieeg.shape[2]
        channel_names = list(electrode_meta.keys())
        layerwise = emb.ndim == 3
        n_layers = emb.shape[1] if layerwise else 1
        n_features = emb.shape[-1]

        alpha_grid = rcfg.alpha if isinstance(rcfg.alpha, list) else [rcfg.alpha]
        logger.info(
            "  [%s] GPU batched regression:\n"
            "    inputs         : n_words=%d  n_features=%d  n_channels=%d  n_lags=%d\n"
            "    layers         : n_layers=%d  (layerwise=%s)\n"
            "    alpha grid     : [%s]  n_alphas=%d\n"
            "    CV             : n_outer_folds=%d  scale=%s  standardize_target=True\n"
            "    total outputs  : %d  (= %d channels × %d lags, per-output alpha)",
            subject,
            emb.shape[0], n_features, n_channels, n_lags,
            n_layers, layerwise,
            ", ".join(f"{a:.3g}" for a in alpha_grid), len(alpha_grid),
            self.cfg.n_cv_folds, rcfg.scale,
            n_channels * n_lags, n_channels, n_lags,
        )

        batched_kwargs = dict(
            n_splits=self.cfg.n_cv_folds,
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            standardize_target=True,
            random_state=rcfg.random_state,
            groups=groups,
            return_oof_predictions=self.cfg.save_word_predictions,
        )

        # Set himalaya backend once in the main thread before spawning workers.
        try:
            from himalaya.backend import set_backend as _set_himalaya_backend
            _set_himalaya_backend("torch_cuda")
        except Exception as exc:
            raise RuntimeError(f"himalaya GPU backend unavailable: {exc}") from exc

        # Fit layers in parallel: each thread submits an independent himalaya
        # call (different X per layer, same Y). PyTorch CUDA ops are thread-safe
        # so the GPU scheduler overlaps execution across threads.
        n_layer_workers = min(self.cfg.n_layer_workers, n_layers)
        layer_tcs: list[list[TimeResolvedScore] | None] = [None] * n_layers

        import torch

        # torch.linalg.eigh (used inside cv_ridge_score_lag_matrix_gpu_batched)
        # does a one-time, non-thread-safe CUDA/MAGMA backend registration on its
        # first call in the process. If two layer-worker threads race to trigger
        # that first call concurrently, PyTorch raises
        # "RuntimeError: lazy wrapper should be called at most once" and the
        # subject's regression is lost. Force that one-time init here, single
        # threaded, before spawning the pool below.
        _eigh_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
        torch.linalg.eigh(_eigh_warmup)
        del _eigh_warmup

        def _fit_layer(li: int) -> tuple[int, list[TimeResolvedScore]]:
            X_li = emb[:, li, :] if layerwise else emb
            # Each thread gets its own CUDA stream so GPU ops run in parallel
            # rather than serialising on the default stream.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                tcs = cv_ridge_score_lag_matrix_gpu_batched(
                    X_li, ieeg, times, **batched_kwargs
                )
            stream.synchronize()
            return li, tcs

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_layer_workers) as pool:
            futures = {pool.submit(_fit_layer, li): li for li in range(n_layers)}
            for fut in as_completed(futures):
                li, tcs = fut.result()
                layer_tcs[li] = tcs
                logger.info("  [%s] layer %d/%d done", subject, li + 1, n_layers)

        results: list[ElectrodeScore] = []
        for ci in range(n_channels):
            ch_name = channel_names[ci] if ci < len(channel_names) else f"ch_{ci}"
            meta = electrode_meta.get(ch_name, {})

            if layerwise:
                per_layer: list[TimeResolvedScore] = [layer_tcs[li][ci] for li in range(n_layers)]  # type: ignore[index]
                layer_scores = [tc.to_mean_cvScore() for tc in per_layer]
                # An electrode where every layer scored NaN (e.g. all folds
                # hit non-finite features for this electrode/checkpoint) has
                # no valid "best" layer — np.nanargmax would raise
                # ValueError: All-NaN slice encountered and crash the whole
                # subject. Fall back to layer 0 (matches ElectrodeScore.
                # best_layer_index / the CPU path's _best_li helper above);
                # this electrode's scores stay NaN either way.
                pearsonrs = np.asarray([s.pearsonr for s in layer_scores], dtype=float)
                best_li = (
                    int(np.nanargmax(pearsonrs)) if np.any(np.isfinite(pearsonrs)) else 0
                )
                if self.cfg.save_word_predictions:
                    # Keep OOF predictions only for the winning layer — the full
                    # (n_layers × n_words × n_times) set would be enormous.
                    for li, tc in enumerate(per_layer):
                        if li != best_li:
                            tc.oof_predictions = None
                results.append(ElectrodeScore(
                    subject=subject,
                    name=ch_name,
                    x=meta.get("x"), y=meta.get("y"), z=meta.get("z"),
                    region=meta.get("region"), hemisphere=meta.get("hemisphere"),
                    score=layer_scores[-1],
                    time_course=per_layer[best_li],
                    layer_scores=layer_scores,
                    layer_time_courses=per_layer,
                ))
            else:
                tc = layer_tcs[0][ci]
                results.append(ElectrodeScore(
                    subject=subject,
                    name=ch_name,
                    x=meta.get("x"), y=meta.get("y"), z=meta.get("z"),
                    region=meta.get("region"), hemisphere=meta.get("hemisphere"),
                    score=tc.to_mean_cvScore(),
                    time_course=tc,
                ))

        mean_r = float(np.nanmean([r.best_layer_pearsonr() for r in results]))
        logger.info(
            "  [%s] %d electrodes — mean best-layer Pearson r = %.4f (GPU batched)",
            subject, len(results), mean_r,
        )
        _log_chosen_alphas(
            subject, results, rcfg.alpha, self.cfg.regression.print_chosen_ridge_alpha,
        )
        return results

    def _regress_electrodes_cpu(
        self,
        subject: str,
        emb: np.ndarray,
        ieeg: np.ndarray,
        times: np.ndarray,
        groups: np.ndarray,
        electrode_meta: dict[str, dict],
    ) -> list[ElectrodeScore]:
        """Original per-electrode joblib path (CPU sklearn ridge)."""
        rcfg = self.cfg.regression
        n_channels = ieeg.shape[1]
        channel_names = list(electrode_meta.keys())

        reg_kwargs = dict(
            n_splits=self.cfg.n_cv_folds,
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            standardize_target=True,
            random_state=rcfg.random_state,
            device="cpu",
        )

        jobs = (
            delayed(_regress_electrode)(
                ci=ci,
                emb=emb,
                y_ci=ieeg[:, ci, :],
                times=times,
                groups=groups,
                ch_name=channel_names[ci] if ci < len(channel_names) else f"ch_{ci}",
                meta=electrode_meta.get(
                    channel_names[ci] if ci < len(channel_names) else f"ch_{ci}", {}
                ),
                subject=subject,
                use_lag_matrix=self.cfg.use_lag_matrix,
                window_size=self.cfg.window_size,
                step_size=self.cfg.step_size,
                reg_kwargs=reg_kwargs,
                save_word_predictions=self.cfg.save_word_predictions,
            )
            for ci in range(n_channels)
        )

        results: list[ElectrodeScore] = Parallel(
            n_jobs=self.cfg.n_jobs, backend="loky",
        )(jobs)

        mean_r = float(np.nanmean([r.best_layer_pearsonr() for r in results]))
        logger.info(
            "  [%s] %d electrodes — mean best-layer Pearson r = %.4f",
            subject, len(results), mean_r,
        )
        _log_chosen_alphas(
            subject, results, rcfg.alpha, self.cfg.regression.print_chosen_ridge_alpha,
        )
        return results


# ---------------------------------------------------------------------------
# Shared per-subject epoch loading (used by PodcastEcogEncoder and by the
# pooled Podcast LME benchmark, neuroai/benchmarks/podcast_lme.py)
# ---------------------------------------------------------------------------

def load_subject_epochs(
    ds: PodcastDataset,
    subject: str,
    word_onsets: np.ndarray,
    *,
    tmin: float,
    tmax: float,
    tmin_extract: float,
    downsample_to: float,
    clip_iqr: float | None,
    baseline: tuple[float, float] | None,
    baseline_mode: str,
    electrode_selection_dir: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict]] | None:
    """Load, filter, and preprocess one subject's word-locked ECoG epochs.

    Returns ``(epochs, valid_mask, analysis_times, electrode_meta)`` or
    ``None`` if the subject should be skipped (missing file, no channels,
    too few valid epochs, etc — already logged at the warning that explains
    why).

    ``epochs`` : (n_valid_words, n_channels, n_times), preprocessed
    (IQR-clipped, baseline-corrected, cropped to ``[tmin, tmax)``, resampled
    to ``downsample_to`` Hz). ``valid_mask`` : (n_words,) boolean, indexes
    into the original (pre-epoch) word list. ``electrode_meta`` keyed by the
    same channel order as ``epochs``' second axis.
    """
    fif = highgamma_fif_path(ds.bids_root, subject)
    if not fif.is_file():
        logger.warning("  highgamma .fif missing for %s — skipping", subject)
        return None

    # ---- load at native sfreq (512 Hz) — do NOT resample yet ----
    try:
        import mne
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
    except Exception as exc:
        logger.warning("  Failed to load .fif for %s: %s", subject, exc)
        return None

    sfreq_native = raw.info["sfreq"]
    recording_tmin = raw.times[0]

    # ---- channel selection ----
    good_ch = ds.good_channel_names(subject)
    if not good_ch:
        logger.warning("  No good channels for %s — skipping", subject)
        return None

    # ---- electrode selection filter (language-responsive only) ----
    if electrode_selection_dir:
        sel_path = Path(electrode_selection_dir) / "selected_electrodes.csv"
        if sel_path.exists():
            sel_df = pd.read_csv(sel_path)
            # selected_electrodes.csv already contains only the selected rows
            sel_names = set(sel_df.loc[sel_df["subject"] == subject, "channel"])
            n_before = len(good_ch)
            good_ch = [ch for ch in good_ch if ch in sel_names]
            logger.info(
                "  Electrode selection: %d → %d language-responsive channels",
                n_before, len(good_ch),
            )
            if not good_ch:
                logger.warning("  No selected electrodes for %s — skipping", subject)
                return None
        else:
            logger.warning(
                "  electrode_selection_dir set but %s not found — using all channels",
                sel_path,
            )

    try:
        raw.pick_channels(good_ch, ordered=True)
    except Exception:
        present = [ch for ch in good_ch if ch in raw.ch_names]
        if not present:
            logger.warning("  No good channels found in .fif for %s", subject)
            return None
        raw.pick_channels(present, ordered=True)
        good_ch = present

    data = raw.get_data().astype(np.float32)  # (n_ch, n_t) at 512 Hz
    n_channels = data.shape[0]
    logger.info("  %d good channels, native sfreq=%.1f Hz", n_channels, sfreq_native)

    # ---- extract epochs at native 512 Hz ----
    epochs, valid_mask, epoch_times = extract_word_epochs(
        data, sfreq_native, recording_tmin,
        word_onsets, tmin_extract, tmax,
    )
    n_valid = int(valid_mask.sum())
    logger.info("  %d / %d words extracted as epochs", n_valid, len(word_onsets))

    if n_valid < 20:
        logger.warning("  Too few valid epochs (%d) for %s — skipping", n_valid, subject)
        return None

    # ---- preprocessing at native 512 Hz ----
    # IQR clip and baseline must happen before resampling so the baseline
    # window has enough samples (e.g. 51 samples at 512 Hz vs 3 at 32 Hz).
    if clip_iqr is not None:
        logger.info("  IQR clip: threshold=%.1f × IQR per channel", clip_iqr)
        epochs = _apply_iqr_clip(epochs, clip_iqr)

    if baseline is not None and baseline_mode:
        logger.info(
            "  Baseline: window=[%.3f, %.3f] s  mode=%s  (at native %.1f Hz)",
            baseline[0], baseline[1], baseline_mode, sfreq_native,
        )
        epochs = _apply_baseline(epochs, epoch_times, baseline, baseline_mode)

    # ---- crop to analysis window (still at 512 Hz) ----
    t_mask = (epoch_times >= tmin) & (epoch_times < tmax)
    epochs = epochs[:, :, t_mask]
    epoch_times = epoch_times[t_mask]
    logger.info(
        "  Cropped to [%.2f, %.2f] s → %d samples at %.1f Hz",
        tmin, tmax, epoch_times.shape[0], sfreq_native,
    )

    # ---- resample epochs to target sfreq ----
    target_sfreq = downsample_to or sfreq_native
    if sfreq_native > target_sfreq + 1:
        logger.info("  Resampling: %.1f Hz → %.1f Hz", sfreq_native, target_sfreq)
        epochs = _resample_epochs(epochs, sfreq_native, target_sfreq)

    n_times = epochs.shape[-1]
    analysis_times = np.linspace(tmin, tmax, n_times, endpoint=False)
    logger.info(
        "  Final epoch shape: (%d words, %d channels, %d time bins)  "
        "@ %.1f Hz  (%.1f ms / bin)",
        n_valid, n_channels, n_times, target_sfreq, 1000.0 / target_sfreq,
    )

    # ---- electrode metadata ----
    electrode_meta = ds.electrode_meta(subject, good_ch)

    return epochs, valid_mask, analysis_times, electrode_meta


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _resample_epochs(epochs: np.ndarray, sfreq_in: float, sfreq_out: float) -> np.ndarray:
    """Resample epoch tensor (n, C, T) along the time axis using scipy FFT resampling."""
    from scipy.signal import resample
    n_times_out = int(round(epochs.shape[-1] * sfreq_out / sfreq_in))
    return resample(epochs, n_times_out, axis=-1).astype(np.float32)


def _apply_iqr_clip(ieeg: np.ndarray, clip_iqr: float) -> np.ndarray:
    """Clip each channel to median ± clip_iqr × IQR across all trials/times."""
    out = ieeg.copy()
    n_trials, n_ch, _ = ieeg.shape
    for ci in range(n_ch):
        x = out[:, ci, :].ravel()
        q25, median, q75 = np.percentile(x, [25, 50, 75])
        iqr = q75 - q25
        out[:, ci, :] = np.clip(out[:, ci, :],
                                 median - clip_iqr * iqr,
                                 median + clip_iqr * iqr)
    return out


def _apply_baseline(
    ieeg: np.ndarray,
    times: np.ndarray,
    baseline: tuple[float, float],
    mode: str,
    eps: float = 1e-12,
) -> np.ndarray:
    """Baseline-correct high-gamma envelope (logratio or mean/percent)."""
    t0, t1 = baseline
    mask = (times >= t0) & (times < t1)
    if not mask.any():
        logger.warning("Baseline window %s has no samples in times [%.3f, %.3f]",
                       baseline, times[0], times[-1])
        return ieeg
    ref = ieeg[:, :, mask].mean(axis=2, keepdims=True)
    out = ieeg.astype(np.float64, copy=True)
    if mode == "mean":
        out -= ref
    elif mode == "logratio":
        out = np.log10(np.maximum(out, eps) / np.maximum(ref, eps))
    elif mode == "ratio":
        out /= np.maximum(ref, eps)
    elif mode == "percent":
        out = (out - ref) / np.maximum(ref, eps) * 100.0
    else:
        raise ValueError(f"Unknown baseline mode: {mode!r}")
    return out.astype(np.float32, copy=False)
