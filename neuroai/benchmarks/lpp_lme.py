"""Le Petit Prince fMRI encoding benchmark — pooled LME per LM layer.

Predicts average-subject, reliable-voxel-filtered, Fedorenko-parcel-averaged
BOLD from PER-LAYER LLM surprisal (logit-lens, Kuribayashi et al. 2025) using
``pymer4``/``lme4`` — see ``neuroai/regressors/lpp_pooled_lme.py`` for the
exact formula:

    bold ~ wlen_z + freq_z + sentpos_z + surprisal_z +
           (1 + surprisal_z | region) + (1 | section)

fit ONCE PER LM LAYER, pooling every one of the 10 regions (5 Fedorenko
language-network parcels × L/R hemisphere) and all 9 LPP sections into one
mixed-effects model — unlike Natural Stories fMRI/Tuckute, which fit one
model per (ROI, layer) pair, ``region`` here is a RANDOM EFFECT inside a
single per-layer model, not a second axis of the fitting grid. This is the
one genuinely new statistical idea in this dataset (see
``regressors/lpp_pooled_lme.py`` module docstring for the random-slope-vs-
fixed-interaction rationale) and it makes this pipeline far cheaper than
Natural Stories fMRI's (n_layers fits total, not n_regions × n_layers).

Scope note — average-subject BOLD, precomputed only
-----------------------------------------------------
This benchmark REQUIRES ``data.average_subject_precomputed_dir`` (l-bg's
``average_subject_run-<0..8>.gz`` files, already used in production by
``neuroai/lpp_encoding.py``). The alignment between the precomputed arrays'
voxel order and the Fedorenko-parcel masks (both derived from the same
canonical ``mask_lpp_en.nii.gz``-style brain mask, see ``neuroai/roi/lpp.py``)
only holds for that path — the on-the-fly ``average_subject_subjects``
alternative in ``lpp_encoding.py`` derives its own EPI mask per call, which is
not guaranteed to align with ``data.brain_mask_path``. Region averaging would
silently be wrong if voxel order didn't match, so this is enforced rather
than silently falling back.

Pipeline (per model checkpoint)
--------------------------------
1. For each of the 9 sections: load word rows (onset/offset/word/logfreq),
   attach a sentence-position covariate derived from the constituency trees
   (``lpp*_tree.csv`` — one sentence per line, story order; LPP word tokens
   carry almost no punctuation, so sentence boundaries come from the tree
   file's own segmentation, not punctuation heuristics), compute per-layer
   logit-lens surprisal (single model load per section, disk-cached — same
   ``logit_lens_surprisal_all_layers`` cache used by Fedorenko2016/Tuckute/NS
   fMRI), then HRF-convolve
   ``[surprisal_layer_0, ..., surprisal_layer_{L-1}, wlen, logfreq, sentpos]``
   together via the same ``word_hrf_basis_matrix`` machinery the ridge
   encoder (``lpp_encoding.py``) uses, onto the precomputed average-subject
   BOLD's own TR grid (``stimulus_aligned_frame_times``), then trim the same
   ``trim_trs_start``/``trim_trs_end`` TRs the ridge encoder trims.
2. Load the precomputed average-subject BOLD for that section
   (``LppFmriEncoder._load_averaged_masked_bold``), optionally restricted to
   the top ``reliable_voxel_fraction`` of voxels by ISC
   (``_select_voxel_indices`` — reused as-is, no ``max_voxels`` subsampling
   here since parcel averaging wants every reliable voxel, not a random
   subset of them).
3. Average BOLD across the voxels in each of the 10 Fedorenko-parcel ×
   hemisphere masks (``neuroai/roi/lpp.py::load_fedorenko_parcelization``,
   restricted to the same voxel selection) to collapse per-voxel BOLD into
   one scalar per (region, TR) — the one genuinely new aggregation step.
4. Cut each section's TRs into contiguous ``chunk_seconds``-long chunks
   (default 30 s, matching Natural Stories fMRI / Shain et al. 2020) — the
   atomic unit for the train/held-out split, so no chunk straddles a
   boundary. Broadcast the section's convolved surprisal/covariates onto
   every region's row for that (section, TR).
5. Append every (region, section, TR) row to one long pooled table
   (region, section, chunk, wlen, freq, sentpos, bold) plus a separate
   (n_rows, n_layers) surprisal array — written to parquet/npy ONCE, then
   read (mmap'd) by worker processes, same pattern as every other pooled LME
   in this codebase.
6. For each LM layer: fit baseline/full LME, compute train delta-loglik
   (LRT) and held-out delta-loglik + Pearson/Spearman r
   (``fit_lpp_lme_one_layer``) — via a spawned process pool
   (``max_tasks_per_child=1``: lme4/rpy2's embedded R interpreter leaks
   memory across repeated fits in one process).
7. Append each layer's result to CSV the moment it completes (crash-safe
   restart — already-completed layers are skipped on re-run).

Output: ``lpp_lme_by_layer.csv`` — one row per layer, written incrementally.
"""

from __future__ import annotations

import logging
import re
import time
import typing as tp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.le_petit_prince import (
    LPP_N_ANNOTATION_SECTIONS,
    LPP_TR_SECONDS,
    load_sentence_trees,
    load_word_information_table,
    words_for_section,
)
from ..lpp_encoding import (
    LppFmriEncoder,
    LppFmriEncodingConfig,
    _select_voxel_indices,
    stimulus_aligned_frame_times,
    word_hrf_basis_matrix,
)
from ..roi.lpp import load_fedorenko_parcelization
from ..sweep import csv_file_lock
from .base import ModelConfig

logger = logging.getLogger(__name__)

# Matches terminal (leaf) tokens in a constituency tree line: "(POS word)"
# where "word" itself contains no nested parens — i.e. NOT a higher
# constituent like "(S (NP ...) ...)".
_LEAF_RE = re.compile(r"\(([^\s()]+)\s+([^()]+?)\)")

_LAYER_ROW_COLUMNS = (
    "layer", "beta", "se", "t_stat", "p_value", "loglik", "aic", "bic",
    "baseline_loglik", "delta_loglik_train", "delta_loglik_test",
    "baseline_loglik_test", "loglik_test", "lrt_p_value",
    "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "slope_dropped", "n_obs", "n_obs_test", "error", "fit_wall_time_s", "revision",
)


# ---------------------------------------------------------------------------
# Sentence-position covariate (from constituency trees, not punctuation)
# ---------------------------------------------------------------------------

def _leaf_tokens_per_sentence(bids_root: str | Path, lang: str) -> list[list[str]]:
    lines = load_sentence_trees(bids_root, lang)
    return [[m.group(2) for m in _LEAF_RE.finditer(line)] for line in lines]


def _filter_valid_word_rows(sec_df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``word`` isn't usable text (NaN, blank, single space).

    The LPP word table has a handful of genuine NaN entries (e.g. row 334 of
    section 1) — coercing them to the string ``"nan"`` via ``.astype(str)``
    would feed a fake token to the LM (and is fragile: whether
    ``.astype(str)`` stringifies NaN vs. leaves it as a float depends on the
    pandas string-dtype backend). Dropping them instead mirrors
    ``le_petit_prince.py::build_word_events_frame``'s own ``_keep`` filter.
    """
    word = sec_df["word"]
    keep = word.map(lambda w: isinstance(w, str) and w.strip() not in ("", " "))
    return sec_df.loc[keep].copy()


def compute_sentpos_covariate(bids_root: str | Path, lang: str) -> np.ndarray:
    """Position-within-sentence (1-based), one value per word, story order.

    Derived from ``lpp*_tree.csv`` (one sentence per line, in the same
    story-linear order as ``lpp*_word_information.csv``'s row order) rather
    than punctuation: LPP's word tokens have almost no punctuation attached
    (already stripped in the annotation), so a punctuation heuristic would
    barely detect any sentence boundaries. Counting leaf tokens per tree line
    gives an exact sentence segmentation instead.

    Must be aligned POSITIONALLY (not by any shared key) with
    ``load_word_information_table(..., apply_lbg_fixes=False)`` — the caller
    is responsible for truncating both to the same length if they mismatch.
    """
    sentences = _leaf_tokens_per_sentence(bids_root, lang)
    sentpos: list[int] = []
    for toks in sentences:
        sentpos.extend(range(1, len(toks) + 1))
    return np.asarray(sentpos, dtype=np.float64)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LppLmeDataConfig:
    """Dataset location and loading options."""

    bids_root: str
    lang: str = "en"
    sections: list[int] | None = None  # None = all 9 LPP chapters
    t_r: float = LPP_TR_SECONDS
    min_word_duration_sec: float = 0.05
    hrf_model: str = "glover"
    trim_trs_start: int = 10
    trim_trs_end: int = 10
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    feature_cache_dir: str | None = None
    # l-bg precomputed average-subject BOLD directory (average_subject_run-
    # <0..8>.gz). Required — see module docstring for why the on-the-fly
    # average_subject_subjects alternative isn't supported here.
    average_subject_precomputed_dir: str | None = None
    # NeurIPS 2024 / l-bg: keep only the top reliable_voxel_fraction of
    # voxels by precomputed split-half BOLD reliability (ISC), aligned to
    # brain_mask_path's voxel order. null -> use every voxel.
    reliable_voxel_fraction: float | None = None
    reliable_voxel_scores_path: str | None = None
    # Canonical LPP brain mask (l-bg voxel order) — must match the voxel
    # order of both average_subject_precomputed_dir and
    # reliable_voxel_scores_path.
    brain_mask_path: str = ""
    fedorenko_atlas_nii_path: str = ""
    fedorenko_labels_txt_path: str | None = None


@dataclass
class LppLmeConfig:
    """LME fitting options (see ``regressors/lpp_pooled_lme.py`` for the formula)."""

    train_frac: float = 0.75
    split_random_state: int = 0
    chunk_seconds: float = 30.0
    verbose: bool = False
    # Barr et al. (2013)-style simplification ladder if the full model fails
    # to converge — drop the region slope-intercept correlation, then drop
    # the region random slope on surprisal_z entirely. See
    # fit_lpp_lme_one_layer's docstring.
    allow_slope_fallback: bool = True
    # Which layers to fit: 1 = all layers; N>1 = every Nth layer.
    layer_stride: int = 1
    # If True, fit ONLY the final layer (index n_layers-1), ignoring layer_stride.
    last_layer_only: bool = False
    # Fit per-subject BOLD (English LPP subjects, ``(1 | subject)`` random
    # intercept added to the formula) instead of the average-subject BOLD
    # this benchmark originally shipped with. The expensive part -- per-
    # subject, Fedorenko-parcel-averaged BOLD extraction -- is a ONE-TIME,
    # model-independent cost already cached at ``persubject_bold_path`` (see
    # ``neuroai/regressors/lpp_persubject.py``, built for the split-half
    # noise ceiling in ``neuroai/regressors/lpp_ceiling.py``), so switching
    # to per-subject does not add per-layer/per-checkpoint cost.
    use_persubject: bool = True
    persubject_bold_path: str = "data/lpp_fmri_extra/lpp_en_persubject_bold.parquet"
    # Process-pool size for fitting layers concurrently — each worker is a
    # separate process with its own embedded R/pymer4 interpreter.
    n_workers: int = 10
    # CPU sweep runs are the LME-only half of a two-stage pipeline — a cache
    # miss there should fail fast rather than silently loading a model.
    require_cached_surprisal: bool = False
    # Save each held-out TR's observed BOLD + baseline/full model predictions
    # (lpp_lme_per_word.csv, one row per (layer, region, section, TR)) — same
    # DLT-difficulty-style convention as every other pooled LME here, see
    # ``fit_spr_lme(..., save_per_word=True)``. On by default.
    save_word_predictions: bool = True


@dataclass
class LppLmeBenchmarkConfig:
    data: LppLmeDataConfig
    model: ModelConfig
    lme: LppLmeConfig = field(default_factory=LppLmeConfig)
    output_dir: str = "outputs"
    canonical_model_name: str | None = None


# ---------------------------------------------------------------------------
# Worker (module-level for spawn-based multiprocessing pickling)
# ---------------------------------------------------------------------------

def _fit_one_layer_worker(
    base_parquet_path: str,
    surprisal_npy_path: str,
    layer_idx: int,
    *,
    train_frac: float,
    random_state: int,
    verbose: bool,
    allow_slope_fallback: bool,
    save_word_predictions: bool = False,
    persubject: bool = False,
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: load the shared table, fit one layer."""
    import numpy as _np
    import pandas as _pd

    from ..regressors.lpp_pooled_lme import fit_lpp_lme_one_layer

    t0 = time.time()
    row: dict[str, tp.Any] = {"layer": layer_idx}
    per_word_out: _pd.DataFrame | None = None
    try:
        base_df = _pd.read_parquet(base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        base_df["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)

        fit = fit_lpp_lme_one_layer(
            base_df,
            train_frac=train_frac,
            random_state=random_state,
            verbose=verbose,
            allow_slope_fallback=allow_slope_fallback,
            save_per_word=save_word_predictions,
            persubject=persubject,
        )
        if fit.per_word is not None and not fit.per_word.empty:
            per_word_out = fit.per_word.copy()
            per_word_out.insert(0, "layer", layer_idx)
        row.update({
            "beta": fit.beta,
            "se": fit.se,
            "t_stat": fit.t_stat,
            "p_value": fit.p_value,
            "loglik": fit.loglik,
            "aic": fit.aic,
            "bic": fit.bic,
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
            "converged": fit.converged,
            "slope_dropped": fit.slope_dropped,
            "n_obs": fit.n_obs,
            "n_obs_test": fit.n_obs_test,
            "error": None,
        })
    except Exception as exc:  # noqa: BLE001 — one bad layer must not kill the pool
        logger.error("Layer %d failed", layer_idx, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return {"row": row, "per_word": per_word_out}


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class LppLmeBenchmark:
    """Pooled LPP fMRI LME: one mixed-effects model per LM layer.

    ``region`` (10 Fedorenko-parcel × hemisphere labels) is a random effect
    INSIDE each per-layer model, not a second fitting-grid axis — see module
    docstring.
    """

    def __init__(self, cfg: LppLmeBenchmarkConfig) -> None:
        self.cfg = cfg
        if not cfg.data.average_subject_precomputed_dir:
            raise ValueError(
                "LppLmeBenchmark requires data.average_subject_precomputed_dir "
                "(l-bg precomputed average-subject BOLD) — see module docstring "
                "for why the on-the-fly average_subject_subjects path isn't "
                "supported here (voxel-order alignment with the Fedorenko "
                "parcel masks is only guaranteed for the precomputed arrays)."
            )

    def _encoder(self) -> LppFmriEncoder:
        dcfg = self.cfg.data
        enc_cfg = LppFmriEncodingConfig(
            bids_root=dcfg.bids_root,
            subject="average_precomputed",  # unused by the precomputed-dir path
            lang=dcfg.lang,
            fmri_unit="average_subject",
            average_subject_precomputed_dir=dcfg.average_subject_precomputed_dir,
            reliable_voxel_fraction=dcfg.reliable_voxel_fraction,
            reliable_voxel_scores_path=dcfg.reliable_voxel_scores_path,
            max_voxels=None,  # region averaging wants every reliable voxel
            brain_mask_path=dcfg.brain_mask_path,
        )
        return LppFmriEncoder(enc_cfg)

    def extract_and_cache_surprisal(self) -> None:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only.

        Loads the LLM, computes per-layer logit-lens surprisal for every LPP
        section, and returns — does not touch BOLD data, does not fit
        anything. Run once per model (GPU job) before the CPU-only ``run()``
        LME sweep, same two-stage split as every other pooled LME here.
        """
        dcfg = self.cfg.data
        sections = dcfg.sections or list(range(1, LPP_N_ANNOTATION_SECTIONS + 1))
        words_full, _ = self._load_words_with_sentpos()
        for section in sections:
            sec_df = words_for_section(words_full, section)
            words = sec_df["word"].astype(str).tolist()
            self._word_surprisal_all_layers(words, section)

    def _load_words_with_sentpos(self) -> tuple[pd.DataFrame, np.ndarray]:
        dcfg = self.cfg.data
        words_full = load_word_information_table(
            dcfg.bids_root, dcfg.lang, apply_lbg_fixes=False,
        )
        sentpos_full = compute_sentpos_covariate(dcfg.bids_root, dcfg.lang)
        n = min(len(words_full), len(sentpos_full))
        if len(words_full) != len(sentpos_full):
            logger.warning(
                "LPP LME: word table (%d rows) and tree-derived sentence "
                "positions (%d rows) disagree in length — truncating both to "
                "%d rows (positional alignment assumed).",
                len(words_full), len(sentpos_full), n,
            )
        words_full = words_full.iloc[:n].copy()
        words_full["sentpos"] = sentpos_full[:n]
        return words_full, sentpos_full

    def run(self, extra_fields: dict[str, tp.Any] | None = None) -> pd.DataFrame:
        dcfg = self.cfg.data
        lcfg = self.cfg.lme
        sections = dcfg.sections or list(range(1, LPP_N_ANNOTATION_SECTIONS + 1))
        model_revision = self.cfg.model.model_revision or "main"
        safe_revision = model_revision.replace("/", "_")
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        layer_csv = out_dir / "lpp_lme_by_layer.csv"

        encoder = self._encoder()
        n_vox_full = encoder._precomputed_n_voxels()
        rng = np.random.default_rng(lcfg.split_random_state)
        voxel_ix = _select_voxel_indices(encoder.cfg, n_vox_full, rng)
        logger.info(
            "[%s] LPP LME: %d mask voxels -> %d selected (reliable_voxel_fraction=%s)",
            model_name, n_vox_full, voxel_ix.size, dcfg.reliable_voxel_fraction,
        )

        parcelization = load_fedorenko_parcelization(
            dcfg.brain_mask_path, dcfg.fedorenko_atlas_nii_path, dcfg.fedorenko_labels_txt_path,
        )
        region_names: list[str] = []
        region_masks_selected: list[np.ndarray] = []
        for ri, name in enumerate(parcelization.region_names):
            for hemi, full_mask in (("L", parcelization.left_masks[ri]), ("R", parcelization.right_masks[ri])):
                mask_sel = full_mask[voxel_ix]
                if not mask_sel.any():
                    logger.warning(
                        "[%s] region %s_%s has 0 selected voxels after reliability "
                        "filtering — dropping.", model_name, hemi, name,
                    )
                    continue
                region_names.append(f"{hemi}_{name}")
                region_masks_selected.append(mask_sel)
        if not region_names:
            raise RuntimeError("No Fedorenko-parcel region has any selected voxel")

        words_full, _ = self._load_words_with_sentpos()

        region_parts: list[np.ndarray] = []
        section_parts: list[np.ndarray] = []
        chunk_parts: list[np.ndarray] = []
        wlen_parts: list[np.ndarray] = []
        freq_parts: list[np.ndarray] = []
        sentpos_parts: list[np.ndarray] = []
        bold_parts: list[np.ndarray] = []
        surp_parts: list[np.ndarray] = []
        n_layers: int | None = None

        for section in sections:
            logger.info("=== [%s] LPP section %d ===", model_name, section)
            sec_df = words_for_section(words_full, section)
            words = sec_df["word"].astype(str).tolist()

            surprisal_layers = self._word_surprisal_all_layers(words, section)  # (n_words, n_layers)
            if n_layers is None:
                n_layers = surprisal_layers.shape[1]

            onsets = sec_df["onset"].to_numpy(dtype=np.float64)
            offsets = sec_df["offset"].to_numpy(dtype=np.float64)
            min_dur = max(0.0, float(dcfg.min_word_duration_sec))
            durations = np.maximum(min_dur, offsets - onsets)
            wlen = sec_df["word"].astype(str).str.len().to_numpy(dtype=np.float64)
            freq = sec_df["logfreq"].to_numpy(dtype=np.float64)
            sentpos = sec_df["sentpos"].to_numpy(dtype=np.float64)

            raw_feats = np.concatenate(
                [
                    surprisal_layers,
                    wlen[:, None],
                    freq[:, None],
                    sentpos[:, None],
                ],
                axis=1,
            )  # (n_words, n_layers + 3)

            # Same NaN-poisoning guard as Natural Stories fMRI: a single
            # non-finite word-level feature must not reach the HRF
            # convolution matmul, or it wipes the entire section's design.
            bad = ~np.isfinite(raw_feats)
            if bad.any():
                n_bad = int(bad.any(axis=1).sum())
                logger.warning(
                    "[%s | section %d] %d/%d word(s) had non-finite feature(s) "
                    "— zeroing before HRF convolution.",
                    model_name, section, n_bad, len(raw_feats),
                )
                raw_feats = np.where(bad, 0.0, raw_feats)

            bold_full = encoder._load_averaged_masked_bold(section, None, voxel_ix)
            n_scans = bold_full.shape[0]
            frame_times = stimulus_aligned_frame_times(n_scans, tr=dcfg.t_r)
            B = word_hrf_basis_matrix(onsets, durations, frame_times, dcfg.hrf_model)
            design = B @ raw_feats  # (n_scans, n_layers + 3)

            ts, te = dcfg.trim_trs_start, dcfg.trim_trs_end
            sl = slice(ts, None if te <= 0 else -te)
            design_trim = design[sl]
            bold_trim = bold_full[sl]
            n_trs = design_trim.shape[0]
            if n_trs < 5:
                logger.warning(
                    "[%s | section %d] only %d TRs after trimming — skipping.",
                    model_name, section, n_trs,
                )
                continue

            chunk_len_trs = max(1, int(round(lcfg.chunk_seconds / dcfg.t_r)))
            chunk_ids = np.arange(n_trs) // chunk_len_trs
            chunk_labels = np.array([f"{section}_{c}" for c in chunk_ids], dtype=object)

            surp_trim = design_trim[:, :n_layers]
            wlen_trim = design_trim[:, n_layers]
            freq_trim = design_trim[:, n_layers + 1]
            sentpos_trim = design_trim[:, n_layers + 2]

            for region_name, mask_sel in zip(region_names, region_masks_selected):
                region_bold = np.nanmean(bold_trim[:, mask_sel], axis=1)

                region_parts.append(np.full(n_trs, region_name, dtype=object))
                section_parts.append(np.full(n_trs, str(section), dtype=object))
                chunk_parts.append(chunk_labels)
                wlen_parts.append(wlen_trim)
                freq_parts.append(freq_trim)
                sentpos_parts.append(sentpos_trim)
                bold_parts.append(region_bold)
                surp_parts.append(surp_trim)

        if not bold_parts:
            raise RuntimeError("No (region, section) rows assembled for LPP LME")

        assert n_layers is not None
        nuisance_df = pd.DataFrame({
            "region": np.concatenate(region_parts),
            "section": np.concatenate(section_parts),
            "chunk": np.concatenate(chunk_parts),
            "wlen": np.concatenate(wlen_parts),
            "freq": np.concatenate(freq_parts),
            "sentpos": np.concatenate(sentpos_parts),
            "bold": np.concatenate(bold_parts),
        })
        surprisal_all = np.concatenate(surp_parts, axis=0).astype(np.float32)  # (N, n_layers)
        logger.info(
            "[%s] pooled table (average-subject): %d rows across %d regions x %d sections, %d layers",
            model_name, len(nuisance_df), len(region_names), len(sections), n_layers,
        )

        if lcfg.use_persubject:
            # Reuse the ONE-TIME, model-independent per-subject BOLD table
            # (see LppLmeConfig.use_persubject's docstring) -- join it onto
            # this model's own covariates/surprisal by (region, section,
            # tr_idx), where tr_idx = row-order-within-(region, section),
            # matching neuroai/regressors/lpp_ceiling.py's already-validated
            # convention (that ceiling converged 15/15 with this same join).
            nuisance_df["tr_idx"] = nuisance_df.groupby(["region", "section"]).cumcount()
            nuisance_df["_orig_row"] = np.arange(len(nuisance_df))

            persubj = pd.read_parquet(lcfg.persubject_bold_path)
            persubj["region"] = persubj["region"].astype(str)
            persubj["section"] = persubj["section"].astype(str)

            merged = persubj.merge(
                nuisance_df[["region", "section", "tr_idx", "chunk", "wlen", "freq", "sentpos", "_orig_row"]],
                on=["region", "section", "tr_idx"], how="inner",
            )
            dropped = len(persubj) - len(merged)
            if dropped:
                logger.warning(
                    "[%s] %d/%d per-subject rows had no matching (region, section, tr_idx) "
                    "in this model's covariate table (TR-count mismatch) -- dropped.",
                    model_name, dropped, len(persubj),
                )
            surprisal_all = surprisal_all[merged["_orig_row"].to_numpy()]
            nuisance_df = pd.DataFrame({
                "subject": merged["subject"].astype(str).values,
                "region": merged["region"].values,
                "section": merged["section"].values,
                "chunk": merged["chunk"].values,
                "wlen": merged["wlen"].values,
                "freq": merged["freq"].values,
                "sentpos": merged["sentpos"].values,
                "bold": merged["bold"].astype(float).values,
            })
            logger.info(
                "[%s] pooled table (per-subject): %d rows across %d subjects x %d regions x %d sections, %d layers",
                model_name, len(nuisance_df), nuisance_df["subject"].nunique(),
                len(region_names), len(sections), n_layers,
            )

        base_parquet_path = out_dir / f"lpp_lme_pooled_table_{safe_revision}.parquet"
        surprisal_npy_path = out_dir / f"lpp_lme_pooled_surprisal_{safe_revision}.npy"
        nuisance_df.to_parquet(base_parquet_path, index=False)
        np.save(surprisal_npy_path, surprisal_all)
        del nuisance_df, surprisal_all

        if lcfg.last_layer_only:
            layer_indices = [n_layers - 1]
        else:
            layer_indices = list(range(0, n_layers, max(1, lcfg.layer_stride)))
        logger.info(
            "[%s] Fitting %d layers (stride=%d) with %d worker processes",
            model_name, len(layer_indices), lcfg.layer_stride, lcfg.n_workers,
        )

        out_columns = list(_LAYER_ROW_COLUMNS) + [
            k for k in (extra_fields or {}) if k not in _LAYER_ROW_COLUMNS
        ]
        done_layers: set[int] = set()
        if layer_csv.exists():
            try:
                existing = pd.read_csv(layer_csv)
                existing_revision = existing["revision"] if "revision" in existing.columns else "main"
                same_revision = existing_revision == model_revision
                good = existing.loc[same_revision & existing["error"].isna()]
                done_layers = set(good["layer"].tolist())
                n_failed = int((same_revision & existing["error"].notna()).sum())
                if done_layers:
                    logger.info("Skipping %d already-fit layer(s) for revision=%s", len(done_layers), model_revision)
                if n_failed:
                    logger.warning(
                        "%d previously-crashed layer(s) found in %s for revision=%s — will retry",
                        n_failed, layer_csv, model_revision,
                    )
                    keep = existing.loc[~(same_revision & existing["error"].notna())]
                    with csv_file_lock(layer_csv):
                        keep.reindex(columns=out_columns).to_csv(layer_csv, index=False)
            except Exception:
                logger.warning("Could not read existing layer CSV; re-fitting all layers.")

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        per_word_csv = out_dir / "lpp_lme_per_word.csv"
        pending = [li for li in layer_indices if li not in done_layers]
        rows_out: list[dict] = []
        with ProcessPoolExecutor(
            max_workers=lcfg.n_workers, mp_context=ctx, max_tasks_per_child=1,
        ) as pool:
            futures = {
                pool.submit(
                    _fit_one_layer_worker,
                    str(base_parquet_path), str(surprisal_npy_path), li,
                    train_frac=lcfg.train_frac,
                    random_state=lcfg.split_random_state,
                    verbose=lcfg.verbose,
                    allow_slope_fallback=lcfg.allow_slope_fallback,
                    save_word_predictions=lcfg.save_word_predictions,
                    persubject=lcfg.use_persubject,
                ): li
                for li in pending
            }
            for fut in as_completed(futures):
                li = futures[fut]
                try:
                    result = fut.result()
                    row, per_word = result["row"], result["per_word"]
                except Exception as exc:
                    logger.error("Layer %d worker crashed: %s", li, exc, exc_info=True)
                    row, per_word = {"layer": li, "error": str(exc)}, None
                row["revision"] = model_revision
                if extra_fields:
                    row.update(extra_fields)
                rows_out.append(row)
                with csv_file_lock(layer_csv):
                    pd.DataFrame([row]).reindex(columns=out_columns).to_csv(
                        layer_csv, mode="a", header=not layer_csv.exists(), index=False,
                    )
                if per_word is not None:
                    with csv_file_lock(per_word_csv):
                        per_word.to_csv(
                            per_word_csv, mode="a", header=not per_word_csv.exists(), index=False,
                        )
                logger.info(
                    "  [layer %d] done in %.1fs — delta_loglik_test=%s converged=%s%s",
                    li, row.get("fit_wall_time_s", float("nan")),
                    f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                    row.get("converged"),
                    " (slope dropped)" if row.get("slope_dropped") else "",
                )

        return pd.DataFrame(rows_out)

    # ------------------------------------------------------------------
    # Surprisal extraction (per-layer logit-lens, one model load per section loop, disk-cached)
    # ------------------------------------------------------------------

    def _word_surprisal_all_layers(self, words: list[str], section: int) -> np.ndarray:
        """Per-layer logit-lens surprisal for every word in one LPP section.

        One forward pass (sliding-window chunked internally) via
        :func:`logit_lens_surprisal_all_layers` — same extractor already used
        by Fedorenko2016/Tuckute/Natural Stories fMRI. ``section`` (1-9) is
        used as ``section_id`` so all 9 chapters share one cache directory
        without collisions.

        CBR-RNN is a special case: bypasses ``AutoModelForCausalLM`` entirely
        and has a single hidden-state vector per token (no layer dimension),
        so ``batch_surprisal_cbr_rnn`` is used instead, returning a single
        "layer" of surprisal — the rest of the pipeline treats ``n_layers``
        generically, so this just means one fit instead of a sweep.
        """
        mcfg = self.cfg.model
        dcfg = self.cfg.data
        require_cache_hit = self.cfg.lme.require_cached_surprisal

        if "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower():
            from ..features.cbr_rnn import batch_surprisal_cbr_rnn

            contexts = [" ".join(words[: i + 1]) for i in range(len(words))]
            targets = words
            surp = batch_surprisal_cbr_rnn(
                mcfg.model_name,
                contexts,
                targets,
                device=mcfg.device,
                cache_dir=dcfg.feature_cache_dir,
                section_id=section,
                require_cache_hit=require_cache_hit,
            )  # (n_words, 1)
            return surp.astype(np.float64)

        from ..features.logit_lens import logit_lens_surprisal_all_layers

        # See podcast_lme.py's `surp_dtype` — BabyLM-community's gpt-bert-*
        # models need float32 (their custom DWA accumulator isn't bf16-safe).
        surp_dtype: str | None = None
        if "gpt-bert" in mcfg.model_name.lower():
            surp_dtype = "float32"

        arr = logit_lens_surprisal_all_layers(
            words,
            mcfg.model_name,
            revision=mcfg.model_revision,
            device=mcfg.device,
            max_length=dcfg.max_length,
            tokenizer_no_revision=dcfg.tokenizer_no_revision,
            cache_dir=dcfg.feature_cache_dir,
            section_id=section,
            lang=dcfg.lang,
            n_params=mcfg.n_params,
            trust_remote_code=mcfg.trust_remote_code,
            untrained=mcfg.untrained,
            untrained_seed=mcfg.untrained_seed,
            require_cache_hit=require_cache_hit,
            dtype=surp_dtype,
        )  # (n_words, n_layers, 1)
        return arr[:, :, 0].astype(np.float64)
