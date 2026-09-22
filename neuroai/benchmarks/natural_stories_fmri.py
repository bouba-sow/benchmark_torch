"""Natural Stories fMRI encoding benchmark.

Predicts ROI-averaged BOLD from LLM hidden states using HRF-convolved
ridge regression with leave-one-story-out cross-validation.

Pipeline (per model checkpoint)
---------------------------------
1. For each of the 8 stories, load the canonical word list + onset times.
2. Extract LLM features for the canonical words (full-text pass, cached to disk
   keyed by model / revision / doc_idx).
3. Precompute the HRF basis matrix B (n_TRs × n_words, shared across subjects).
4. Convolve all layers at once: design_doc = einsum("tw,wld->tld", B, emb)
   → shape (n_TRs_doc, n_layers, D).
5. For each of the 78 subjects:
   a. Load BOLD per fROI (26 targets: 6 LANG + 20 MD) for all 8 stories.
   b. Stack design matrices and BOLD across stories; build group labels
      (group = story index per TR) for leave-one-story-out CV.
   c. Run multi-output ridge with CUDA-stream layer parallelism (GPU) or
      joblib thread parallelism (CPU):
        - himalaya torch_cuda backend: one CUDA stream per layer
        - All 26 fROI targets solved in a single kernel call per (layer, fold)
   d. Append per-fROI results to CSVs immediately (crash-safe restart).
6. After all subjects, write checkpoint summary CSV.

Cross-validation
----------------
Leave-one-story-out: groups[t] = doc_idx of the story containing TR t.
GroupKFold(n_splits=n_docs) = 8 folds, each holding out one complete story.
This prevents any within-story autocorrelation leakage and matches the
independent-context assumption of the stimulus design.

Primary metric: Spearman r (consistent with the rest of the benchmark suite).
Pearson r is also stored in CSVs as a secondary metric.

Outputs (written to output_dir)
---------------------------------
  ns_fmri_checkpoint_scores.csv     — one row per revision: headline Spearman r
  ns_fmri_checkpoint_layerwise.csv  — one row per (revision, layer, network)
  ns_fmri_by_roi.csv                — one row per (subject, fROI, revision)
  ns_fmri_by_subject.csv            — one row per (subject, network, revision)

Reference
---------
Shain et al. (2020). fMRI reveals language-specific predictive coding during
naturalistic sentence comprehension. Neuropsychologia, 138, 107307.
"""

from __future__ import annotations

import logging
import threading
import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..benchmarks.base import ModelConfig, RegressionConfig, word_predictions_dir
from ..datasets.natural_stories_fmri import (
    NS_FMRI_DOCS,
    NS_FMRI_TR,
    NS_FMRI_WORD_DUR,
    available_frois,
    available_subjects,
    load_bold_subject,
    load_canonical_words,
    word_hrf_basis_matrix,
)
from ..features.llm import (
    contextual_word_embeddings_hf,
    contextual_word_embeddings_hf_fulltext,
)
from ..regressors.ridge import (
    CVScore,
    cv_ridge_score_multioutput,
    cv_ridge_score_multioutput_gpu,
)

logger = logging.getLogger(__name__)

_csv_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NsFmriDataConfig:
    """Dataset location and loading options."""

    data_root: str
    froi_table: str = "LANG_MD"          # "LANG", "MD", or "LANG_MD"
    docs: list[str] | None = None        # None = all 8 stories
    subjects: list[str] | None = None   # None = all 78 subjects
    t_r: float = NS_FMRI_TR             # repetition time in seconds
    word_duration: float = NS_FMRI_WORD_DUR  # HRF event duration
    hrf_model: str = "glover"           # double-gamma HRF (nilearn)
    n_splits: int = 8                   # leave-one-story-out (= n_docs)
    feature_cache_dir: str | None = None
    max_length: int | None = None       # None = model default
    context_words: int | None = None    # None = full-text pass


@dataclass
class NsFmriBenchmarkConfig:
    """Top-level config for the Natural Stories fMRI encoding benchmark."""

    data: NsFmriDataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    output_dir: str = "outputs"
    layerwise: bool = True
    n_jobs: int = 20              # CPU joblib layer workers
    n_layer_workers: int = 8     # GPU CUDA stream workers
    tokenizer_no_revision: bool = True
    # Save each held-out TR's true BOLD + out-of-fold predicted BOLD (LAST
    # layer only -- a per-target best-layer refit isn't worth the extra
    # compute, same call as ``zuco_reading_embed.py``'s word-prediction
    # convention) to ns_fmri_embed_word_predictions/<subject>.parquet. On by
    # default.
    save_word_predictions: bool = True


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NsFmriRoiScore:
    """Encoding score for one (subject, fROI)."""

    subject: str
    froi: str
    network: str      # "LANG" or "MD"
    hemisphere: str   # "L" or "R" (derived from fROI name)
    score: CVScore    # last-layer flat score
    layer_scores: list[CVScore] | None = None

    def best_layer_spearmanr(self) -> float:
        if self.layer_scores:
            vals = np.asarray([s.spearmanr for s in self.layer_scores], dtype=float)
            return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else float("nan")
        return self.score.spearmanr

    def best_layer_pearsonr(self) -> float:
        if self.layer_scores:
            vals = np.asarray([s.pearsonr for s in self.layer_scores], dtype=float)
            return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else float("nan")
        return self.score.pearsonr

    def best_layer_index(self, metric: str = "spearmanr") -> int | None:
        if self.layer_scores:
            vals = np.asarray([getattr(s, metric) for s in self.layer_scores], dtype=float)
            if np.any(np.isfinite(vals)):
                return int(np.nanargmax(vals))
        return None

    def to_dict(self, include_layers: bool = False) -> dict:
        d: dict[str, tp.Any] = {
            "subject": self.subject,
            "froi": self.froi,
            "network": self.network,
            "hemisphere": self.hemisphere,
            **self.score.to_dict(),
            "best_spearmanr": self.best_layer_spearmanr(),
            "best_pearsonr": self.best_layer_pearsonr(),
            "best_layer": self.best_layer_index("spearmanr"),
        }
        if include_layers and self.layer_scores:
            for li, ls in enumerate(self.layer_scores):
                d[f"layer_{li}_spearmanr"] = ls.spearmanr
                d[f"layer_{li}_pearsonr"] = ls.pearsonr
        return d


@dataclass
class NsFmriEncoderResult:
    """Aggregated encoding results for one model checkpoint."""

    model_name: str
    model_revision: str | None
    roi_scores: list[NsFmriRoiScore]

    def mean_spearmanr(self, network: str | None = None) -> float:
        scores = self.roi_scores
        if network:
            scores = [s for s in scores if s.network == network]
        vals = [s.best_layer_spearmanr() for s in scores]
        return float(np.nanmean(vals)) if vals else float("nan")

    def by_subject(self) -> pd.DataFrame:
        rows = [s.to_dict() for s in self.roi_scores]
        df = pd.DataFrame(rows)
        return (
            df.groupby(["subject", "network"])[["best_spearmanr", "best_pearsonr"]]
            .mean()
            .reset_index()
        )

    def by_network(self) -> pd.DataFrame:
        rows = [s.to_dict() for s in self.roi_scores]
        df = pd.DataFrame(rows)
        return (
            df.groupby("network")[["best_spearmanr", "best_pearsonr"]]
            .agg(["mean", "std", "count"])
            .pipe(lambda x: x.set_axis([f"{c}_{s}" for c, s in x.columns], axis=1))
            .reset_index()
        )


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class NsFmriEncoder:
    """Predict Natural Stories fMRI BOLD from LLM hidden states.

    Embeddings are extracted once per story, HRF-convolved (shared across all
    78 subjects), then ridge-regressed per subject against 26 fROI targets
    using leave-one-story-out CV.  Himalaya GPU backend is used when
    ``model.device != "cpu"``, with one CUDA stream per transformer layer.
    """

    def __init__(self, cfg: NsFmriBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(self) -> NsFmriEncoderResult:
        dcfg = self.cfg.data
        docs = dcfg.docs or NS_FMRI_DOCS
        subjects = dcfg.subjects or available_subjects(dcfg.data_root, dcfg.froi_table)

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        roi_csv  = out_dir / "ns_fmri_by_roi.csv"
        subj_csv = out_dir / "ns_fmri_by_subject.csv"
        lw_csv   = out_dir / "ns_fmri_checkpoint_layerwise.csv"
        # NOT "ns_fmri_checkpoint_scores.csv" -- that filename is owned by
        # run/embedding/neural/ns_fmri.py's per-checkpoint _summary_row()
        # (step/tokens_seen/mean_best_spearmanr schema, read by
        # summarize/ns_fmri.py / summarize/benchmark.py). This used to
        # collide with it: two incompatible schemas appended to the same
        # CSV, corrupting it for any downstream pd.read_csv.
        ck_csv   = out_dir / "ns_fmri_checkpoint_by_network.csv"

        model_revision = self.cfg.model.model_revision or "main"

        # Skip subjects already computed for this revision
        done_subjects: set[str] = set()
        if roi_csv.exists():
            try:
                _ex = pd.read_csv(roi_csv, usecols=["subject", "revision"])
                done_subjects = set(
                    _ex.loc[_ex["revision"] == model_revision, "subject"]
                )
                if done_subjects:
                    logger.info(
                        "[NS-fMRI] Skipping %d already-done subjects for revision=%s",
                        len(done_subjects), model_revision,
                    )
            except Exception:
                logger.warning("[NS-fMRI] Could not read existing ROI CSV; re-running all.")

        # ── Step 1: extract embeddings + precompute design matrices per story ──
        design_docs: dict[str, np.ndarray] = {}  # doc → (n_TRs, n_layers, D)
        n_trs_per_doc: dict[str, int] = {}
        froi_meta: dict[str, tuple[list[str], list[str]]] | None = None

        for doc_idx, doc in enumerate(docs):
            logger.info("=== Story: %s (idx=%d) ===", doc, doc_idx)

            words_df = load_canonical_words(dcfg.data_root, doc, dcfg.froi_table)
            words = words_df["word"].tolist()
            onsets = words_df["time"].values

            emb = self._extract_features(words, doc_idx)
            # emb: (n_words, n_layers, D) if layerwise else (n_words, D)

            # HRF basis matrix: (n_TRs_doc, n_words) — shared across subjects
            last_onset = float(onsets[-1])
            n_trs = int(np.ceil((last_onset + 20.0) / dcfg.t_r)) + 1
            frame_times = np.arange(n_trs) * dcfg.t_r
            durations = np.full(len(onsets), dcfg.word_duration)
            B = word_hrf_basis_matrix(onsets, durations, frame_times, dcfg.hrf_model)
            # B: (n_TRs_doc, n_words)

            # HRF-convolved design: (n_TRs_doc, n_layers, D) or (n_TRs_doc, D)
            if emb.ndim == 3:
                # einsum "tw,wld->tld": faster than looping over layers
                design = np.einsum("tw,wld->tld", B, emb.astype(np.float64))
            else:
                design = B @ emb.astype(np.float64)  # (n_TRs_doc, D)

            design_docs[doc] = design.astype(np.float32)
            n_trs_per_doc[doc] = n_trs

            logger.info(
                "[%s | %s] design shape: %s",
                self.cfg.model.model_name, doc,
                "×".join(str(d) for d in design.shape),
            )

        # ── Step 2: loop subjects ──
        all_scores: list[NsFmriRoiScore] = []

        for subj in subjects:
            if subj in done_subjects:
                logger.info("  [%s] skipped (already done)", subj)
                continue

            logger.info("  --- Subject: %s ---", subj)
            try:
                scores = self._run_subject(
                    subj, docs, design_docs, n_trs_per_doc, froi_meta, out_dir
                )
                all_scores.extend(scores)
                self._save_scores(subj, scores, out_dir, model_revision, roi_csv, subj_csv, lw_csv)

                # Cache fROI metadata from first successful subject
                if froi_meta is None and scores:
                    froi_meta = {}
                    for s in scores:
                        froi_meta[s.froi] = (s.network, s.hemisphere)

            except Exception:
                logger.error("  [%s] failed", subj, exc_info=True)

        from ..features.llm import clear_hf_model_cache
        clear_hf_model_cache()

        self._save_checkpoint_summary(all_scores, ck_csv, lw_csv, model_revision)

        return NsFmriEncoderResult(
            model_name=self.cfg.model.model_name,
            model_revision=self.cfg.model.model_revision,
            roi_scores=all_scores,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, words: list[str], doc_idx: int) -> np.ndarray:
        dcfg = self.cfg.data
        mcfg = self.cfg.model
        layer_index: str | int = "all" if self.cfg.layerwise else "last"
        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        try:
            emb = contextual_word_embeddings_hf_fulltext(
                words,
                mcfg.model_name,
                layer_index=layer_index,
                revision=mcfg.model_revision,
                device=mcfg.device,
                max_length=dcfg.max_length,
                cache_dir=dcfg.feature_cache_dir,
                section_id=doc_idx,
                lang="en",
                tokenizer_no_revision=self.cfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=mcfg.trust_remote_code,
                dtype=surp_dtype,
            )
        except ValueError as exc:
            if "fast tokenizer" not in str(exc):
                raise
            logger.warning("[NS-fMRI] falling back to per-word extraction: %s", exc)
            emb = contextual_word_embeddings_hf(
                words,
                mcfg.model_name,
                layer_index=layer_index,
                revision=mcfg.model_revision,
                device=mcfg.device,
                context_words=dcfg.context_words,
                max_length=dcfg.max_length,
                cache_dir=dcfg.feature_cache_dir,
                section_id=doc_idx,
                lang="en",
                tokenizer_no_revision=self.cfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=mcfg.trust_remote_code,
                dtype=surp_dtype,
            )
        return emb  # (n_words, [n_layers,] D)

    # ------------------------------------------------------------------
    # Per-subject pipeline
    # ------------------------------------------------------------------

    def _run_subject(
        self,
        subject: str,
        docs: list[str],
        design_docs: dict[str, np.ndarray],
        n_trs_per_doc: dict[str, int],
        froi_meta: dict[str, tuple[str, str]] | None,
        out_dir: Path | None = None,
    ) -> list[NsFmriRoiScore]:
        dcfg = self.cfg.data

        bold_data = load_bold_subject(
            dcfg.data_root, subject, dcfg.froi_table, docs=docs
        )
        if not bold_data:
            logger.warning("  [%s] no BOLD data — skipping", subject)
            return []

        # Align design matrices and BOLD by story
        X_parts: list[np.ndarray] = []
        Y_parts: list[np.ndarray] = []
        group_parts: list[np.ndarray] = []
        story_parts: list[np.ndarray] = []
        scan_time_parts: list[np.ndarray] = []
        froi_names: list[str] = []
        networks: list[str] = []

        for doc_idx, doc in enumerate(docs):
            if doc not in bold_data or doc not in design_docs:
                continue

            bold, scan_times, doc_frois, doc_networks = bold_data[doc]
            design = design_docs[doc]   # (n_TRs_design, n_layers, D) or (n_TRs_design, D)

            # Trim design to match BOLD TR count (BOLD may be shorter than design window)
            n_trs_bold = len(scan_times)
            n_trs_design = design.shape[0]
            n_trs = min(n_trs_bold, n_trs_design)
            if n_trs < 20:
                logger.warning(
                    "  [%s %s] only %d aligned TRs — skipping story", subject, doc, n_trs
                )
                continue

            X_parts.append(design[:n_trs])
            Y_parts.append(bold[:n_trs])
            group_parts.append(np.full(n_trs, doc_idx, dtype=np.int32))
            story_parts.append(np.full(n_trs, doc, dtype=object))
            scan_time_parts.append(np.asarray(scan_times[:n_trs], dtype=np.float64))

            if not froi_names:
                froi_names = doc_frois
                networks = doc_networks

        if not X_parts:
            return []

        X_all = np.concatenate(X_parts, axis=0)   # (ΣTRs, n_layers, D) or (ΣTRs, D)
        Y_all = np.concatenate(Y_parts, axis=0).astype(np.float32)   # (ΣTRs, n_fROIs)
        groups = np.concatenate(group_parts, axis=0)
        story_all = np.concatenate(story_parts, axis=0)
        scan_time_all = np.concatenate(scan_time_parts, axis=0)

        n_unique_groups = int(np.unique(groups).size)
        n_splits = min(dcfg.n_splits, n_unique_groups)
        if n_splits < 2:
            logger.warning("  [%s] fewer than 2 story groups — skipping", subject)
            return []

        # Drop fROIs with any NaN across the concatenated BOLD
        valid_froi = ~np.any(np.isnan(Y_all), axis=0)
        if valid_froi.sum() == 0:
            logger.warning("  [%s] all fROIs have NaN — skipping", subject)
            return []

        valid_idx = np.where(valid_froi)[0]
        Y_valid = Y_all[:, valid_idx]
        froi_names_valid = [froi_names[j] for j in valid_idx]
        networks_valid = [networks[j] for j in valid_idx]

        logger.info(
            "  [%s] %d TRs, %d stories, %d/%d valid fROIs",
            subject, len(groups), n_unique_groups, len(valid_idx), Y_all.shape[1],
        )

        rcfg = self.cfg.regression
        reg_kwargs = dict(
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            random_state=getattr(rcfg, "random_state", 0),
        )

        use_gpu = self.cfg.model.device != "cpu"
        if use_gpu:
            layer_scores_all = self._regress_gpu(X_all, Y_valid, groups, n_splits, reg_kwargs)
        else:
            layer_scores_all = self._regress_cpu(X_all, Y_valid, groups, n_splits, reg_kwargs)
        # layer_scores_all[j] = list[CVScore] over layers for target j

        layerwise = X_all.ndim == 3
        n_layers = X_all.shape[1] if layerwise else 1

        alpha_meds = np.array(
            [ls[-1].ridge_alpha for ls in layer_scores_all if np.isfinite(ls[-1].ridge_alpha)],
            dtype=np.float64,
        )
        if alpha_meds.size:
            logger.info(
                "  [%s] chosen alpha (last layer, %d fROIs): median=%.3g min=%.3g max=%.3g",
                subject, alpha_meds.size,
                float(np.median(alpha_meds)),
                float(alpha_meds.min()),
                float(alpha_meds.max()),
            )

        nan_score = CVScore(
            pearsonr=float("nan"), r2=float("nan"),
            mse=float("nan"), spearmanr=float("nan"),
        )
        roi_scores: list[NsFmriRoiScore] = []
        for j, (froi, net) in enumerate(zip(froi_names_valid, networks_valid)):
            ls = layer_scores_all[j]
            hemi = "L" if "L" in froi[4:6] else "R"
            roi_scores.append(NsFmriRoiScore(
                subject=subject,
                froi=froi,
                network=net,
                hemisphere=hemi,
                score=ls[-1],
                layer_scores=ls if layerwise else None,
            ))

        mean_sr = float(np.nanmean([s.best_layer_spearmanr() for s in roi_scores]))
        mean_lang = float(np.nanmean(
            [s.best_layer_spearmanr() for s in roi_scores if s.network == "LANG"]
        ))
        mean_md = float(np.nanmean(
            [s.best_layer_spearmanr() for s in roi_scores if s.network == "MD"]
        ))
        logger.info(
            "  [%s] mean best-layer Spearman r: overall=%.4f  LANG=%.4f  MD=%.4f",
            subject, mean_sr, mean_lang, mean_md,
        )

        if self.cfg.save_word_predictions and out_dir is not None:
            self._save_word_predictions(
                subject, X_all, Y_valid, groups, story_all, scan_time_all,
                froi_names_valid, n_splits, reg_kwargs, use_gpu, out_dir,
            )

        return roi_scores

    # ------------------------------------------------------------------
    # Held-out per-TR predictions (last layer only, see save_word_predictions)
    # ------------------------------------------------------------------

    def _save_word_predictions(
        self,
        subject: str,
        X_all: np.ndarray,
        Y_valid: np.ndarray,
        groups: np.ndarray,
        story_all: np.ndarray,
        scan_time_all: np.ndarray,
        froi_names_valid: list[str],
        n_splits: int,
        reg_kwargs: dict,
        use_gpu: bool,
        out_dir: Path,
    ) -> None:
        X_last = X_all[:, -1, :] if X_all.ndim == 3 else X_all
        try:
            if use_gpu:
                import torch
                from himalaya.backend import set_backend as _set_backend

                _set_backend("torch_cuda")
                torch.cuda.init()
                _, oof = cv_ridge_score_multioutput_gpu(
                    X_last.astype(np.float32), Y_valid,
                    groups=groups, n_splits=n_splits,
                    standardize_target=True, return_oof_predictions=True,
                    **reg_kwargs,
                )
            else:
                _, oof = cv_ridge_score_multioutput(
                    X_last, Y_valid,
                    groups=groups, n_splits=n_splits,
                    standardize_target=True, return_oof_predictions=True,
                    **reg_kwargs,
                )
        except Exception:
            logger.error("  [%s] last-layer OOF refit for word predictions failed", subject, exc_info=True)
            return

        n_trs, n_targets = oof.shape
        pred_dir = word_predictions_dir(out_dir, "ns_fmri_embed_word_predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "subject": np.full(n_trs * n_targets, subject, dtype=object),
            "story": np.repeat(story_all, n_targets),
            "scan_time": np.repeat(scan_time_all, n_targets),
            "froi": np.tile(np.asarray(froi_names_valid, dtype=object), n_trs),
            # Y_valid/oof are (n_trs, n_targets) C-contiguous — row-major
            # .reshape(-1) walks fROI fastest, matching np.tile(..., n_trs)
            # above and np.repeat(story_all/scan_time_all, n_targets).
            "bold_true": Y_valid.astype(np.float64).reshape(-1),
            "bold_pred_oof": oof.reshape(-1),
        })
        out_path = pred_dir / f"{subject}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("  [%s] saved %d held-out TR×fROI predictions -> %s", subject, len(df), out_path.name)

    # ------------------------------------------------------------------
    # Regression — CPU
    # ------------------------------------------------------------------

    def _regress_cpu(
        self,
        X_all: np.ndarray,    # (ΣTRs, n_layers, D) or (ΣTRs, D)
        Y_valid: np.ndarray,  # (ΣTRs, n_fROIs)
        groups: np.ndarray,
        n_splits: int,
        reg_kwargs: dict,
    ) -> list[list[CVScore]]:
        try:
            import threadpoolctl as _tc
            _HAS_TC = True
        except ImportError:
            _tc = None  # type: ignore[assignment]
            _HAS_TC = False

        layerwise = X_all.ndim == 3
        n_layers = X_all.shape[1] if layerwise else 1
        n_targets = Y_valid.shape[1]

        n_jobs = self.cfg.n_jobs
        n_layer_jobs = min(n_jobs, n_layers)
        n_blas = max(1, n_jobs // n_layer_jobs)

        def _fit_layer(li: int) -> list[CVScore]:
            X_li = X_all[:, li, :] if layerwise else X_all
            return cv_ridge_score_multioutput(
                X_li, Y_valid,
                groups=groups,
                n_splits=n_splits,
                standardize_target=True,
                **reg_kwargs,
            )

        if n_layer_jobs > 1:
            import contextlib
            from joblib import Parallel, delayed
            ctx = _tc.threadpool_limits(limits=n_blas, user_api="blas") if _HAS_TC else contextlib.nullcontext()
            with ctx:
                layer_results: list[list[CVScore]] = Parallel(
                    n_jobs=n_layer_jobs, prefer="threads"
                )(delayed(_fit_layer)(li) for li in range(n_layers))
        else:
            layer_results = [_fit_layer(li) for li in range(n_layers)]

        # Transpose: layer_results[li][j] → result[j][li]
        return [[layer_results[li][j] for li in range(n_layers)] for j in range(n_targets)]

    # ------------------------------------------------------------------
    # Regression — GPU (himalaya, CUDA streams per layer)
    # ------------------------------------------------------------------

    def _regress_gpu(
        self,
        X_all: np.ndarray,    # (ΣTRs, n_layers, D) or (ΣTRs, D)
        Y_valid: np.ndarray,  # (ΣTRs, n_fROIs) float32
        groups: np.ndarray,
        n_splits: int,
        reg_kwargs: dict,
    ) -> list[list[CVScore]]:
        import torch
        from himalaya.backend import set_backend as _set_backend

        _set_backend("torch_cuda")
        torch.cuda.init()

        # torch.linalg.svd (used inside cv_ridge_score_multioutput_gpu) does a
        # one-time, non-thread-safe CUDA/MAGMA backend registration on its
        # first call in the process — separate from torch.cuda.init() above.
        # If two layer-worker threads race to trigger that first call
        # concurrently, PyTorch raises "RuntimeError: lazy wrapper should be
        # called at most once" and the layer's fit is lost (same fix as
        # podcast.py / broderick.py's eigh-based GPU-batched regression).
        _svd_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
        torch.linalg.svd(_svd_warmup, full_matrices=False)
        del _svd_warmup

        layerwise = X_all.ndim == 3
        n_layers = X_all.shape[1] if layerwise else 1
        n_targets = Y_valid.shape[1]
        n_layer_workers = min(self.cfg.n_layer_workers, n_layers)

        def _fit_layer(li: int) -> tuple[int, list[CVScore]]:
            X_li = (X_all[:, li, :] if layerwise else X_all).astype(np.float32)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                scores = cv_ridge_score_multioutput_gpu(
                    X_li, Y_valid,
                    groups=groups,
                    n_splits=n_splits,
                    standardize_target=True,
                    **reg_kwargs,
                )
            stream.synchronize()
            return li, scores

        layer_scores_by_layer: list[list[CVScore] | None] = [None] * n_layers
        with ThreadPoolExecutor(max_workers=n_layer_workers) as pool:
            futures = {pool.submit(_fit_layer, li): li for li in range(n_layers)}
            for fut in as_completed(futures):
                li, scores = fut.result()
                layer_scores_by_layer[li] = scores

        # Transpose: layer_scores_by_layer[li][j] → result[j][li]
        return [
            [layer_scores_by_layer[li][j] for li in range(n_layers)]  # type: ignore[index]
            for j in range(n_targets)
        ]

    # ------------------------------------------------------------------
    # Incremental CSV saving
    # ------------------------------------------------------------------

    def _save_scores(
        self,
        subject: str,
        scores: list[NsFmriRoiScore],
        out_dir: Path,
        model_revision: str,
        roi_csv: Path,
        subj_csv: Path,
        lw_csv: Path,
    ) -> None:
        if not scores:
            return

        rows = [s.to_dict() for s in scores]
        for r in rows:
            r["revision"] = model_revision
        df = pd.DataFrame(rows)

        # Per-subject mean by network
        subj_rows: list[dict] = []
        for net in df["network"].unique():
            ndf = df[df["network"] == net]
            subj_rows.append({
                "subject": subject,
                "network": net,
                "revision": model_revision,
                "best_spearmanr_mean": ndf["best_spearmanr"].mean(),
                "best_spearmanr_std": ndf["best_spearmanr"].std(),
                "best_pearsonr_mean": ndf["best_pearsonr"].mean(),
                "n_rois": len(ndf),
            })

        # Per-layer rows
        lw_rows: list[dict] = []
        for s in scores:
            if s.layer_scores:
                for li, ls in enumerate(s.layer_scores):
                    lw_rows.append({
                        "subject": subject,
                        "froi": s.froi,
                        "network": s.network,
                        "hemisphere": s.hemisphere,
                        "revision": model_revision,
                        "layer": li,
                        "spearmanr": ls.spearmanr,
                        "pearsonr": ls.pearsonr,
                    })

        with _csv_write_lock:
            df.to_csv(roi_csv, mode="a", header=not roi_csv.exists(), index=False)
            pd.DataFrame(subj_rows).to_csv(
                subj_csv, mode="a", header=not subj_csv.exists(), index=False,
            )
            if lw_rows:
                pd.DataFrame(lw_rows).to_csv(
                    lw_csv, mode="a", header=not lw_csv.exists(), index=False,
                )

        logger.info(
            "  [%s] saved %d ROI rows → %s", subject, len(df), roi_csv.name,
        )

    def _save_checkpoint_summary(
        self,
        scores: list[NsFmriRoiScore],
        ck_csv: Path,
        lw_csv: Path,
        model_revision: str,
    ) -> None:
        if not scores:
            return

        df = pd.DataFrame([s.to_dict() for s in scores])
        lang_sr = float(np.nanmean(df.loc[df["network"] == "LANG", "best_spearmanr"]))
        md_sr   = float(np.nanmean(df.loc[df["network"] == "MD",   "best_spearmanr"]))
        overall = float(np.nanmean(df["best_spearmanr"]))

        row: dict = {
            "revision": model_revision,
            "mean_best_spearmanr": overall,
            "lang_best_spearmanr": lang_sr,
            "md_best_spearmanr": md_sr,
            "lang_minus_md_spearmanr": lang_sr - md_sr,
            "mean_best_pearsonr": float(np.nanmean(df["best_pearsonr"])),
            "lang_best_pearsonr": float(np.nanmean(df.loc[df["network"] == "LANG", "best_pearsonr"])),
            "n_subjects": df["subject"].nunique(),
            "n_rois": len(df),
        }

        # Per-fROI breakdown
        for froi, gdf in df.groupby("froi"):
            row[f"{froi}_spearmanr"] = float(np.nanmean(gdf["best_spearmanr"]))

        with _csv_write_lock:
            pd.DataFrame([row]).to_csv(
                ck_csv, mode="a", header=not ck_csv.exists(), index=False,
            )
        logger.info(
            "[NS-fMRI] checkpoint summary: overall=%.4f LANG=%.4f MD=%.4f → %s",
            overall, lang_sr, md_sr, ck_csv.name,
        )

        # Layerwise summary (mean Spearman r per layer × network)
        if lw_csv.exists():
            try:
                lw = pd.read_csv(lw_csv)
                lw = lw[lw["revision"] == model_revision]
                if not lw.empty:
                    lw_summary = (
                        lw.groupby(["layer", "network"])["spearmanr"]
                        .agg(["mean", "std", "count"])
                        .reset_index()
                    )
                    lw_summary["revision"] = model_revision
                    # NOT "ns_fmri_checkpoint_layerwise.csv" -- that's lw_csv
                    # itself (raw per-subject/layer rows, read by
                    # summarize/ns_fmri.py). Writing the aggregate back into
                    # the same file it was aggregated from would corrupt it
                    # with a second, incompatible schema.
                    lw_summary_csv = ck_csv.parent / "ns_fmri_checkpoint_layerwise_summary.csv"
                    lw_summary.to_csv(
                        lw_summary_csv, mode="a",
                        header=not lw_summary_csv.exists(), index=False,
                    )
            except Exception:
                logger.warning("[NS-fMRI] Layerwise summary failed", exc_info=True)
