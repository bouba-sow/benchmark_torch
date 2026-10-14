"""Le Petit Prince voxelwise fMRI encoding (language features → BOLD).

Uses **HRF convolved design matrices** (nilearn ``compute_regressor``, Glover HRF by
default) aligned to **preprocessed BOLD** timelines where volume 0 corresponds to
**auditory section onset** (first four pre-stimulus TRs removed upstream; see
``docs/lpp_sprint1_spec.md``).

Feature modes
-------------
* ``glove``       — static GloVe embeddings from ``annotation/<LANG>/lpp*_word_embeddings_GloVe.csv``.
* ``huggingface`` — contextual hidden states via
  :mod:`neuroai.lpp_llm_features` (GPU recommended).

Layer modes (``layer_mode``, HuggingFace only)
----------------------------------------------
* ``"last"``  — last hidden layer only (default; backward compatible).
* ``"all"``   — extract every layer, run ridge per layer; returns
  :class:`LppLayerwiseEncodingResult`.
* ``int N``   — single specific layer N (0 = token embedding).

Lateralization (``compute_lateralization=True``)
-----------------------------------------------
After per-voxel ridge, split selected voxels by MNI x-coordinate (x > 0 →
right hemisphere, x < 0 → left hemisphere in RAS/MNI space) and report
mean Pearson r per hemisphere alongside the overall mean.  The asymmetry
index is ``mean_left − mean_right``.

Optional **picture-block** nuisance regressors for **section 1, English/Chinese**
follow Li et al., *Sci Data* 2022 (onsets 10 / 35 / 60 s, durations 15 / 20 / 15 s).
"""

from __future__ import annotations

import logging
import re
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from nilearn.glm.first_level import compute_regressor
from nilearn.image import resample_to_img, smooth_img
from nilearn.masking import compute_epi_mask
from nilearn.signal import clean
from sklearn.linear_model import Ridge

from .regressors.ridge import lpp_loso_multioutput_ridge_gpu

from neuroai.datasets.le_petit_prince import (
    LPP_TR_SECONDS,
    discover_lp_timelines,
    load_word_information_table,
    words_for_section,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GloVe helpers (unchanged)
# ---------------------------------------------------------------------------

def parse_glove_vector(s: str) -> np.ndarray:
    """Parse the bracketed float vector string in LPPC GloVe CSV cells."""
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
    return np.asarray([float(x) for x in nums], dtype=np.float32)


def load_glove_features(
    bids_root: Path | str, lang: str, word_indices: pd.Index
) -> np.ndarray:
    """Rows of GloVe vectors in ``word_indices`` order."""
    lang = lang.lower()
    sub = {"en": "EN", "fr": "FR", "cn": "CN"}[lang]
    path = Path(bids_root) / "annotation" / sub / f"lpp{sub}_word_embeddings_GloVe.csv"
    glove_df = pd.read_csv(path, index_col=0)
    rows: list[np.ndarray] = []
    for ix in word_indices:
        if ix not in glove_df.index:
            raise KeyError(f"Word table index {ix} missing in {path.name}")
        rows.append(parse_glove_vector(str(glove_df.loc[ix, "GloVe"])))
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def stimulus_aligned_frame_times(n_scans: int, tr: float = LPP_TR_SECONDS) -> np.ndarray:
    """Sample times (seconds) at mid-TR; t=0 is auditory onset for preproc series."""
    return np.arange(n_scans, dtype=np.float64) * tr + 0.5 * tr


# ---------------------------------------------------------------------------
# HRF design helpers
# ---------------------------------------------------------------------------

def word_hrf_basis_matrix(
    onsets: np.ndarray,
    durations: np.ndarray,
    frame_times: np.ndarray,
    hrf_model: str = "glover",
) -> np.ndarray:
    """Basis ``B`` of shape ``(n_scans, n_words)``: column ``w`` is the Glen HRF regressor
    for word ``w`` alone at **unit** amplitude (duration ``durations[w]``).

    For linear HRF models (Glover, etc.), the amplitude-modulated multi-event design
    satisfies ``design = B @ features`` for ``features`` of shape ``(n_words, k)``,
    equivalent to calling ``compute_regressor`` separately for every feature column
    but **~O(n_feat)** times faster when ``n_feat ≫ n_words``.
    """
    onsets = np.asarray(onsets, dtype=np.float64).ravel()
    durations = np.asarray(durations, dtype=np.float64).ravel()
    n_words = int(onsets.shape[0])
    if durations.shape[0] != n_words:
        raise ValueError("onsets and durations must have the same length")
    n_scans = len(frame_times)
    B = np.zeros((n_scans, n_words), dtype=np.float64)
    for w in range(n_words):
        exp_condition = np.asarray(([onsets[w]], [durations[w]], [1.0]))
        signal, _ = compute_regressor(exp_condition, hrf_model, frame_times)
        B[:, w] = signal[:, 0]
    return B


def features_to_hrf_design(
    onsets: np.ndarray,
    durations: np.ndarray,
    features: np.ndarray,
    frame_times: np.ndarray,
    hrf_model: str = "glover",
) -> np.ndarray:
    """Build ``(n_scans, n_features)`` design via amplitude-modulated event regressors."""
    B = word_hrf_basis_matrix(onsets, durations, frame_times, hrf_model)
    return B @ np.asarray(features, dtype=np.float64)


# Li et al. 2022 — section 1 visual illustrations (EN/CN only), stimulus-aligned seconds.
LPP_SECTION1_PICTURE_BLOCKS_SEC: tuple[tuple[float, float], ...] = (
    (10.0, 15.0),
    (35.0, 20.0),
    (60.0, 15.0),
)


def picture_block_nuisance_design(
    section: int,
    lang: str,
    frame_times: np.ndarray,
    hrf_model: str = "glover",
) -> np.ndarray:
    """HRF-convolved boxcar regressors for picture blocks (section 1, EN/CN), or empty."""
    key = lang.lower()
    if section != 1 or key not in ("en", "cn"):
        return np.zeros((len(frame_times), 0), dtype=np.float64)
    out = np.zeros((len(frame_times), len(LPP_SECTION1_PICTURE_BLOCKS_SEC)), dtype=np.float64)
    for j, (onset, dur) in enumerate(LPP_SECTION1_PICTURE_BLOCKS_SEC):
        exp_condition = np.asarray(([onset], [dur], [1.0]))
        signal, _ = compute_regressor(exp_condition, hrf_model, frame_times)
        out[:, j] = signal[:, 0]
    return out


# ---------------------------------------------------------------------------
# NIfTI / mask helpers
# ---------------------------------------------------------------------------

def correlations_to_masked_nifti(
    corr_sel: np.ndarray,
    voxel_ix: np.ndarray,
    mask_bool: np.ndarray,
    affine: np.ndarray,
) -> nib.Nifti1Image:
    """Map length-``len(voxel_ix)`` correlations into mask space (NaN outside)."""
    brain_flat_ix = np.flatnonzero(mask_bool.ravel())
    full = np.full(brain_flat_ix.shape[0], np.nan, dtype=np.float32)
    full[voxel_ix] = corr_sel.astype(np.float32, copy=False)
    vol = np.full(mask_bool.shape, np.nan, dtype=np.float32)
    vol.ravel()[brain_flat_ix] = full
    return nib.Nifti1Image(vol, affine)


def voxel_mni_x_coords(
    mask_bool: np.ndarray,
    affine: np.ndarray,
    voxel_ix: np.ndarray,
) -> np.ndarray:
    """Return MNI x-coordinates (mm) for the selected masked voxels.

    Parameters
    ----------
    mask_bool : (X, Y, Z) bool array — brain mask.
    affine    : (4, 4) voxel-to-MNI affine from the NIfTI image.
    voxel_ix  : 1-D indices into the flattened ``mask_bool`` mask
                (i.e. indices into ``mask_bool.ravel()[mask_bool.ravel()]``).

    Returns
    -------
    (len(voxel_ix),) float array — MNI x in mm.
        Positive x → right hemisphere (RAS convention).
        Negative x → left hemisphere.
    """
    # Flat voxel indices in the full 3D grid
    flat_in_brain = np.flatnonzero(mask_bool.ravel())
    flat_in_vol = flat_in_brain[voxel_ix]  # shape (n_sel,)

    # Convert flat → (i,j,k) in the 3D grid
    shape = mask_bool.shape
    k_idx = flat_in_vol % shape[2]
    j_idx = (flat_in_vol // shape[2]) % shape[1]
    i_idx = flat_in_vol // (shape[2] * shape[1])
    ijk = np.stack([i_idx, j_idx, k_idx, np.ones_like(i_idx)], axis=0).astype(float)

    # Apply affine: rows = x,y,z,1
    mni = affine @ ijk  # (4, n_sel)
    return mni[0]  # MNI x-coordinate


def _zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    m = np.nanmean(x, axis=axis, keepdims=True)
    s = np.nanstd(x, axis=axis, keepdims=True)
    s = np.where(s < eps, 1.0, s)
    return (x - m) / s


def _standardize_lbg(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Exact l-bg standardize: (x - mean) / std, no epsilon guard."""
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, keepdims=True)
    return (x - m) / s


def _mask_to_bool_for_img(mask_img: nib.Nifti1Image, reference: nib.Nifti1Image) -> np.ndarray:
    """Boolean mask resampled to the 3D grid of ``reference``."""
    if mask_img.shape[:3] != reference.shape[:3]:
        mask_img = resample_to_img(mask_img, reference, interpolation="nearest")
    return mask_img.get_fdata() > 0.5


def _extract_timeseries_masked(
    bold_path: Path,
    mask_img: nib.Nifti1Image,
    voxel_ix: np.ndarray,
    *,
    smoothing_fwhm: float | None,
    high_pass_hz: float | None,
    tr: float,
) -> np.ndarray:
    """Return ``(n_timepoints, n_voxels_sel)`` after optional preprocessing.

    When ``high_pass_hz`` is ``None`` or ``0``, this returns the raw masked
    series (no detrend/filter). Standardization is applied later after TR
    trimming in :meth:`_prepare_run`, which matches strict l-bg parity mode.
    Otherwise ``nilearn.signal.clean`` runs with detrend + high-pass and zscore.
    ``smoothing_fwhm=None`` skips spatial smoothing (also matching l-bg).
    """
    img = nib.load(str(bold_path))
    if smoothing_fwhm is not None and float(smoothing_fwhm) > 0:
        img = smooth_img(img, fwhm=float(smoothing_fwhm))
    d = np.asarray(img.dataobj, dtype=np.float32)
    if d.ndim != 4:
        raise ValueError(f"Expected 4D BOLD at {bold_path}, got shape {d.shape}")
    mask_bool = _mask_to_bool_for_img(mask_img, img)
    flat = d.reshape(-1, d.shape[-1])
    m = mask_bool.ravel()
    brain = flat[m, :].T.astype(np.float64)
    y = brain[:, voxel_ix]
    if not high_pass_hz:
        return y
    return clean(
        y,
        t_r=tr,
        detrend=True,
        standardize="zscore_sample",
        high_pass=float(high_pass_hz),
    )


def _subsample_voxels(
    n_voxels: int,
    max_voxels: int | None,
    rng: np.random.Generator,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Subsample voxel column indices into masked BOLD.

    If ``candidate_indices`` is set (e.g. top-ISC voxels), only those indices are
    considered; ``n_voxels`` is still the upper bound for valid index range.
    """
    if candidate_indices is None:
        pool = np.arange(int(n_voxels), dtype=int)
    else:
        pool = np.asarray(candidate_indices, dtype=int).ravel()
        if pool.size and (pool.min() < 0 or pool.max() >= int(n_voxels)):
            raise ValueError(
                f"candidate_indices out of range [0, {n_voxels}); "
                f"min={int(pool.min())}, max={int(pool.max())}"
            )
        pool = np.unique(pool)
    n_pool = int(pool.size)
    if max_voxels is None or n_pool <= int(max_voxels):
        return pool
    chosen = rng.choice(pool, size=int(max_voxels), replace=False)
    return np.sort(chosen)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LppFmriEncodingConfig:
    bids_root: str | Path
    subject: str
    lang: str = "en"
    bold_source: tp.Literal["raw", "derivatives"] = "derivatives"
    echo: int | None = 2
    # Feature extraction
    feature_mode: tp.Literal["glove", "huggingface", "cbr_rnn"] = "glove"
    # HuggingFace options
    hf_model_name: str = "openai-community/gpt2"
    hf_revision: str | None = None
    hf_device: str = "cpu"
    hf_batch_size: int = 4
    # ``context_window`` — cumulative left words + last-token vector (cheap).
    # ``lbg_fulltext`` — l-bg transcript + tokenizer stride + token-mean per word.
    hf_extraction_mode: tp.Literal["context_window", "lbg_fulltext"] = "context_window"
    hf_context_words: int = 48
    hf_max_length: int = 512
    # l-bg replay (only if ``hf_extraction_mode == \"lbg_fulltext\"``)
    hf_lbg_story_text_zip: str | Path | None = None
    hf_lbg_stride_delta: int = 64
    hf_lbg_max_length: int | None = None
    # Layer selection: "last" | "all" | int N
    layer_mode: tp.Union[tp.Literal["last", "all"], int] = "last"
    # Feature caching: save/load pre-extracted features per (model, revision, section)
    feature_cache_dir: str | Path | None = None
    # Nuisance
    picture_nuisance_regressors: bool = False
    hrf_model: str = "glover"
    # Preprocessing
    trim_trs_start: int = 10
    trim_trs_end: int = 10
    ridge_alphas: np.ndarray = field(
        default_factory=lambda: np.logspace(2.0, 7.0, 8)
    )
    smoothing_fwhm: float | None = 4.0
    high_pass_hz: float | None = 1.0 / 128.0
    # When True the tokenizer is loaded from the base model (no revision/step),
    # matching l-bg which always calls AutoTokenizer.from_pretrained(base_name).
    hf_tokenizer_no_revision: bool = False
    hf_trust_remote_code: bool = False
    # If True, skip pretrained weight loading and use a randomly initialised
    # model of the same architecture (untrained baseline for brain scores).
    hf_untrained: bool = False
    hf_untrained_seed: int = 0
    # Approximate parameter count — enables multi-GPU device_map sharding for
    # large models (e.g. OPT-66B) that don't fit on a single GPU.
    hf_n_params: float | None = None
    random_state: int = 0
    # Word table
    apply_lbg_fixes: bool = False
    # Match llms_brain_lateralization numerical behavior as closely as possible:
    # - no pre-trim z-scoring when high_pass_hz is disabled
    # - trim first, then standardize with plain mean/std (no epsilon guard)
    strict_lbg_parity: bool = False
    # Minimum duration floor for word events. Set to 0.0 for strict l-bg parity.
    min_word_duration_sec: float = 0.05
    # Voxel sampling
    max_voxels: int | None = 4000
    # NeurIPS 2024 / l-bg: keep only the top ``reliable_voxel_fraction`` of voxels
    # by precomputed split-half BOLD reliability (ISC). Requires a 1-D score vector
    # (or l-bg ``isc_*trials_*.gz``) aligned to the same in-mask voxel order as this
    # encoder's EPI mask. null → use all mask voxels before ``max_voxels``.
    reliable_voxel_fraction: float | None = None
    reliable_voxel_scores_path: str | Path | None = None
    mask_from_runs: int | None = 3
    max_runs: int | None = None
    # Output
    save_corr_maps_dir: Path | str | None = None
    # Lateralization analysis
    compute_lateralization: bool = False
    # run_layerwise() only: save each LOSO fold's held-out true/predicted BOLD
    # for the LAST layer only (mirrors every other pooled prediction-saving
    # convention in this codebase -- a per-fold best-layer refit isn't worth
    # the extra compute) to lpp_embed_word_predictions/<section>.npz. On by
    # default.
    save_word_predictions: bool = True
    # fMRI target: ``single_subject`` (default) or ``average_subject`` (l-bg
    # ``fit_average_subject.py`` — mean masked BOLD across listed subjects per section).
    fmri_unit: tp.Literal["single_subject", "average_subject"] = "single_subject"
    # Subjects to average when ``fmri_unit == \"average_subject\"`` (non-empty).
    # ``cfg.subject`` must be one of these (typically the first); used only to
    # define run↔section pairing via :func:`discover_lp_timelines`.
    average_subject_subjects: tuple[str, ...] | None = None
    # Optional l-bg precomputed average-subject directory containing
    # ``average_subject_run-<0..8>.gz`` joblib files.
    average_subject_precomputed_dir: str | Path | None = None
    # Parallelism (joblib/loky). Avoid setting both high: see :meth:`LppFmriEncoder._ridge_n_jobs`.
    ridge_n_jobs: int = 1
    layer_n_jobs: int = 1
    # ``cuda`` — himalaya/torch multi-output ridge on GPU; layers parallel via CUDA streams.
    regression_device: tp.Literal["cpu", "cuda"] = "cpu"
    n_layer_workers: int = 8
    # l-bg Harvard-Oxford sphere ROIs in mask voxel order (for granular ROI CSVs).
    brain_mask_path: str | Path | None = None
    roi_masks_dir: str | Path | None = None


def _score_ridge_alpha_nested_cv(
    alpha: float,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    """Module-level worker for parallel nested-CV alpha search."""
    m = Ridge(alpha=float(alpha), fit_intercept=False)
    m.fit(x_tr, y_tr)
    pred = m.predict(x_val)
    corrs = LppFmriEncoder._voxelwise_pearson(y_val, pred)
    return float(alpha), float(np.nanmean(corrs))


def _build_layer_designs(
    li: int,
    section_feats: list[np.ndarray],
    basis_sl_list: list[np.ndarray],
    pn_sl_list: list[np.ndarray],
    *,
    strict_lbg_parity: bool,
) -> list[np.ndarray]:
    """Per-run design matrices for one hidden layer (trimmed, standardized)."""
    x_for_layer: list[np.ndarray] = []
    for ri in range(len(section_feats)):
        feats_2d = section_feats[ri][:, li, :].astype(np.float64, copy=False)
        x_core = basis_sl_list[ri] @ feats_2d
        pn_sl = pn_sl_list[ri]
        if pn_sl.shape[1] > 0:
            x_d = np.hstack([x_core, pn_sl])
        else:
            x_d = x_core
        if strict_lbg_parity:
            x_d = _standardize_lbg(x_d, axis=0)
        else:
            x_d = _zscore(x_d, axis=0)
        x_for_layer.append(x_d)
    return x_for_layer


def _ensure_himalaya_cuda() -> None:
    """Activate himalaya torch_cuda backend (once per process)."""
    from himalaya.backend import set_backend as _set_backend

    import torch

    _set_backend("torch_cuda")
    if torch.cuda.is_available():
        torch.cuda.init()


def _loso_nested_alpha_arrays(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    train_idx: list[int],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    """Split stacked training runs for nested alpha selection vs full train fit."""
    x_train_full = np.vstack([x_list[i] for i in train_idx])
    y_train_full = np.vstack([y_list[i] for i in train_idx])
    if len(train_idx) < 2:
        return None, None, x_train_full, y_train_full
    val_idx = train_idx[0]
    tr_idx = [i for i in train_idx if i != val_idx]
    x_tr = np.vstack([x_list[i] for i in tr_idx])
    y_tr = np.vstack([y_list[i] for i in tr_idx])
    return x_tr, y_tr, x_train_full, y_train_full


def _fit_layer_loso_fold(
    li: int,
    section_feats: list[np.ndarray],
    basis_sl_list: list[np.ndarray],
    pn_sl_list: list[np.ndarray],
    y_list: list[np.ndarray],
    train_idx: list[int],
    test_k: int,
    ridge_alphas: np.ndarray,
    ridge_n_jobs: int,
    strict_lbg_parity: bool,
    compute_lateralization: bool,
    use_vector_hemisphere_split: bool,
    voxel_ix: np.ndarray,
    all_mask_bools: list[np.ndarray],
    all_affines: list[np.ndarray],
    *,
    regression_device: tp.Literal["cpu", "cuda"] = "cpu",
    return_predictions: bool = False,
) -> dict[str, tp.Any]:
    """Fit one layer for one LOSO fold; returns scores and per-voxel correlations.

    ``return_predictions=True`` additionally includes the held-out
    ``(n_test_TRs, n_voxels)`` true/predicted BOLD arrays in the returned dict
    (``y_test``/``pred``) — used for post-hoc per-TR difficulty splits (e.g.
    DLT cost). Only pass this for the layer(s) whose predictions you actually
    want to keep in memory/on disk (e.g. the last layer only, mirroring every
    other pooled prediction-saving convention in this codebase) — every layer
    is fit regardless, this flag only controls whether its prediction array
    is additionally materialised and returned.
    """
    x_for_layer = _build_layer_designs(
        li,
        section_feats,
        basis_sl_list,
        pn_sl_list,
        strict_lbg_parity=strict_lbg_parity,
    )
    x_tr, y_tr, x_train_full, y_train_full = _loso_nested_alpha_arrays(
        x_for_layer, y_list, train_idx,
    )
    x_test = x_for_layer[test_k]
    y_test = y_list[test_k]

    pred: np.ndarray | None = None
    if regression_device == "cuda":
        x_val = x_for_layer[train_idx[0]] if len(train_idx) >= 2 else None
        y_val = y_list[train_idx[0]] if len(train_idx) >= 2 else None
        gpu_out = lpp_loso_multioutput_ridge_gpu(
            x_tr if x_tr is not None else x_train_full,
            y_tr if y_tr is not None else y_train_full,
            x_val,
            y_val,
            x_train_full,
            y_train_full,
            x_test,
            y_test,
            ridge_alphas=ridge_alphas,
            return_predictions=return_predictions,
        )
        if return_predictions:
            best_alpha, corrs, corrs_sp, pred = gpu_out
        else:
            best_alpha, corrs, corrs_sp = gpu_out
    else:
        best_alpha = float(ridge_alphas[len(ridge_alphas) // 2])
        if x_tr is not None and y_tr is not None and len(train_idx) >= 2:
            val_idx = train_idx[0]
            x_val = x_for_layer[val_idx]
            y_val = y_list[val_idx]
            n_alpha_jobs = max(1, int(ridge_n_jobs))
            if n_alpha_jobs > 1 and len(ridge_alphas) > 1:
                scored = Parallel(n_jobs=n_alpha_jobs, backend="loky")(
                    delayed(_score_ridge_alpha_nested_cv)(
                        float(alpha), x_tr, y_tr, x_val, y_val
                    )
                    for alpha in ridge_alphas
                )
                best_alpha, _best_score = max(scored, key=lambda t: t[1])
            else:
                best_score = -np.inf
                for alpha in ridge_alphas:
                    _, sc = _score_ridge_alpha_nested_cv(float(alpha), x_tr, y_tr, x_val, y_val)
                    if np.isfinite(sc) and sc > best_score:
                        best_score = sc
                        best_alpha = float(alpha)

        ridge = Ridge(alpha=best_alpha, fit_intercept=False)
        ridge.fit(x_train_full, y_train_full)
        pred = ridge.predict(x_test)
        corrs = LppFmriEncoder._voxelwise_pearson(y_test, pred)
        corrs_sp = LppFmriEncoder._voxelwise_spearman(y_test, pred)

    mean_r = float(np.nanmean(corrs))
    mean_sp = float(np.nanmean(corrs_sp))

    left_r = right_r = float("nan")
    if compute_lateralization:
        if use_vector_hemisphere_split:
            n = int(corrs.shape[0])
            h = n // 2
            left_r = float(np.nanmean(corrs[:h]))
            right_r = float(np.nanmean(corrs[h:]))
        elif all_mask_bools and test_k < len(all_mask_bools):
            mb = all_mask_bools[test_k]
            aff = all_affines[test_k]
            x_coords = voxel_mni_x_coords(mb, aff, voxel_ix)
            left_r = float(np.nanmean(corrs[x_coords < 0]))
            right_r = float(np.nanmean(corrs[x_coords > 0]))

    out: dict[str, tp.Any] = {
        "layer": li,
        "best_alpha": best_alpha,
        "mean_pearson": mean_r,
        "mean_pearson_left": left_r,
        "mean_pearson_right": right_r,
        "mean_spearman": mean_sp,
        "corrs": corrs,
        "corrs_spearman": corrs_sp,
    }
    if return_predictions:
        out["y_test"] = y_test
        out["pred"] = pred
    return out


def _load_reliable_voxel_scores(path: Path | str) -> np.ndarray:
    """Load one score per in-mask voxel (same order as ``mask_bool`` brain columns).

    Supports:

    * ``.npz`` with array ``scores`` (or a single array key) — shape ``(n_vox,)``.
    * l-bg ``isc_{k}trials_<lang>.gz`` from ``joblib.dump``: shape ``(k, n_vox)``
      (mean over trials is used) or 1-D ``(n_vox,)``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Reliable-voxel scores file not found: {p}")
    if p.suffix.lower() == ".npz":
        z = np.load(p, allow_pickle=False)
        if "scores" in z.files:
            arr = z["scores"]
        elif len(z.files) == 1:
            arr = z[z.files[0]]
        else:
            raise ValueError(
                f"{p}: .npz must contain 'scores' or exactly one array key; got {z.files}"
            )
    else:
        arr = joblib.load(p)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.nanmean(arr, axis=0)
    elif arr.ndim != 1:
        raise ValueError(f"Expected 1-D or 2-D scores from {p}, got shape {arr.shape}")
    return arr.astype(np.float64, copy=False).ravel()


def _top_reliable_voxel_indices(scores: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Indices of the highest-scoring ``keep_fraction`` of voxels (inclusive rounding)."""
    n = int(scores.size)
    f = float(keep_fraction)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    k = max(1, min(n, int(round(f * n))))
    order = np.argsort(-scores, kind="stable")
    return np.sort(order[:k])


def _select_voxel_indices(
    cfg: LppFmriEncodingConfig,
    n_vox_full: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply optional ISC / reliability filter, then ``max_voxels`` subsampling."""
    candidates: np.ndarray | None = None
    if cfg.reliable_voxel_fraction is not None:
        if cfg.reliable_voxel_scores_path is None:
            logger.warning(
                "reliable_voxel_fraction=%s set but reliable_voxel_scores_path is None; "
                "using all %d mask voxels.",
                cfg.reliable_voxel_fraction,
                n_vox_full,
            )
        else:
            scores = _load_reliable_voxel_scores(cfg.reliable_voxel_scores_path)
            if scores.size != n_vox_full:
                raise ValueError(
                    f"Reliable scores length {scores.size} != EPI mask voxel count "
                    f"{n_vox_full} (file {cfg.reliable_voxel_scores_path}). "
                    "Scores must be aligned to this pipeline's mask voxel order "
                    "(``compute_epi_mask`` / ``_extract_timeseries_masked`` columns)."
                )
            candidates = _top_reliable_voxel_indices(scores, cfg.reliable_voxel_fraction)
            logger.info(
                "Reliable-voxel filter: keep top %.1f%% → %d / %d voxels (from %s).",
                100.0 * float(cfg.reliable_voxel_fraction),
                candidates.size,
                n_vox_full,
                Path(cfg.reliable_voxel_scores_path).name,
            )
    elif cfg.reliable_voxel_scores_path is not None:
        logger.warning(
            "reliable_voxel_scores_path is set but reliable_voxel_fraction is None; "
            "ignoring scores file."
        )
    return _subsample_voxels(n_vox_full, cfg.max_voxels, rng, candidate_indices=candidates)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LppEncodingRunResult:
    section: int
    bids_run: int
    n_words: int
    n_tr_used: int
    n_voxels: int
    mean_pearson: float
    best_alpha: float
    mean_pearson_left: float = float("nan")
    mean_pearson_right: float = float("nan")
    lateralization_index: float = float("nan")
    mean_spearman: float = float("nan")

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            "section": self.section,
            "bids_run": self.bids_run,
            "n_words": self.n_words,
            "n_tr_used": self.n_tr_used,
            "n_voxels": self.n_voxels,
            "mean_pearson": self.mean_pearson,
            "mean_spearman": self.mean_spearman,
            "best_alpha": self.best_alpha,
            "mean_pearson_left": self.mean_pearson_left,
            "mean_pearson_right": self.mean_pearson_right,
            "lateralization_index": self.lateralization_index,
        }


@dataclass
class LppEncodingResult:
    subject: str
    lang: str
    runs: list[LppEncodingRunResult]
    overall_mean_pearson: float
    overall_mean_pearson_left: float = float("nan")
    overall_mean_pearson_right: float = float("nan")
    overall_lateralization_index: float = float("nan")
    overall_mean_spearman: float = float("nan")


@dataclass
class LppLayerwiseRunResult:
    """Per-fold results for every layer when ``layer_mode="all"``."""
    section: int
    bids_run: int
    n_words: int
    n_tr_used: int
    n_voxels: int
    layer_mean_pearson: list[float] = field(default_factory=list)
    layer_mean_pearson_left: list[float] = field(default_factory=list)
    layer_mean_pearson_right: list[float] = field(default_factory=list)
    layer_best_alphas: list[float] = field(default_factory=list)
    best_alpha: float = float("nan")
    layer_mean_spearman: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, tp.Any]:
        d: dict[str, tp.Any] = {
            "section": self.section,
            "bids_run": self.bids_run,
            "n_words": self.n_words,
            "n_tr_used": self.n_tr_used,
            "n_voxels": self.n_voxels,
            "best_alpha": self.best_alpha,
        }
        for li, r in enumerate(self.layer_mean_pearson):
            d[f"layer_{li}_pearson"] = r
        for li, r in enumerate(self.layer_mean_spearman):
            d[f"layer_{li}_spearman"] = r
        for li, r in enumerate(self.layer_mean_pearson_left):
            d[f"layer_{li}_pearson_left"] = r
        for li, r in enumerate(self.layer_mean_pearson_right):
            d[f"layer_{li}_pearson_right"] = r
        for li, a in enumerate(self.layer_best_alphas):
            d[f"layer_{li}_alpha"] = a
        return d


@dataclass
class LppLayerwiseFoldDetail:
    """One LOSO fold × layer row for granular checkpoint CSVs."""
    section: int
    bids_run: int
    layer: int
    best_alpha: float
    mean_pearson: float
    mean_pearson_left: float = float("nan")
    mean_pearson_right: float = float("nan")
    lateralization_index: float = float("nan")
    n_voxels: int = 0
    mean_spearman: float = float("nan")

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            "section": self.section,
            "bids_run": self.bids_run,
            "layer": self.layer,
            "best_alpha": self.best_alpha,
            "mean_pearson": self.mean_pearson,
            "mean_spearman": self.mean_spearman,
            "mean_pearson_left": self.mean_pearson_left,
            "mean_pearson_right": self.mean_pearson_right,
            "lateralization_index": self.lateralization_index,
            "n_voxels": self.n_voxels,
        }


@dataclass
class LppLayerwiseEncodingResult:
    """Full per-layer encoding results across all LOSO folds."""
    subject: str
    lang: str
    n_layers: int
    runs: list[LppLayerwiseRunResult]
    # Mean across folds per layer
    layer_mean_pearson: list[float] = field(default_factory=list)
    layer_mean_pearson_left: list[float] = field(default_factory=list)
    layer_mean_pearson_right: list[float] = field(default_factory=list)
    layer_mean_alphas: list[float] = field(default_factory=list)
    layer_mean_spearman: list[float] = field(default_factory=list)
    # Granular exports (mask-space indices, LOSO-averaged voxel correlations per layer)
    voxel_global_ix: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    fold_details: list[LppLayerwiseFoldDetail] = field(default_factory=list)
    layer_voxel_mean_pearson: list[np.ndarray] = field(default_factory=list)
    voxel_roi_labels: np.ndarray | None = None
    # Held-out per-TR true/predicted BOLD, LAST layer only, one entry per LOSO
    # fold -- populated only when cfg.save_word_predictions=True. Each dict:
    # {"section", "bids_run", "layer", "y_test", "pred"} ((n_test_TRs,
    # n_voxels) arrays, aligned to voxel_global_ix). None unless requested;
    # the caller (e.g. run/embedding/neural/lpp.py) is responsible for
    # persisting these to disk -- see module docstring / save_word_predictions.
    fold_predictions: list[dict[str, tp.Any]] | None = None

    @property
    def best_layer(self) -> int:
        if not self.layer_mean_pearson:
            return 0
        return int(np.nanargmax(self.layer_mean_pearson))

    @property
    def overall_mean_pearson(self) -> float:
        if not self.layer_mean_pearson:
            return float("nan")
        return float(np.nanmax(self.layer_mean_pearson))

    @property
    def overall_mean_spearman(self) -> float:
        """Spearman at the Pearson-selected best layer (same layer as the headline)."""
        if not self.layer_mean_spearman:
            return float("nan")
        best = self.best_layer
        return self.layer_mean_spearman[best] if best < len(self.layer_mean_spearman) else float("nan")

    @property
    def overall_lateralization_index(self) -> float:
        if not self.layer_mean_pearson_left or not self.layer_mean_pearson_right:
            return float("nan")
        best = self.best_layer
        left = self.layer_mean_pearson_left[best] if best < len(self.layer_mean_pearson_left) else float("nan")
        right = self.layer_mean_pearson_right[best] if best < len(self.layer_mean_pearson_right) else float("nan")
        if np.isnan(left) or np.isnan(right):
            return float("nan")
        return left - right


# ---------------------------------------------------------------------------
# LppFmriEncoder
# ---------------------------------------------------------------------------

class LppFmriEncoder:
    """Leave-one-run-out ridge encoding: word-level features → whole-brain TR series.

    Standard usage (last layer or GloVe):

    .. code-block:: python

        encoder = LppFmriEncoder(cfg)
        result  = encoder.run_leave_one_run_out()

    Per-layer usage (HuggingFace with ``layer_mode="all"``):

    .. code-block:: python

        cfg.layer_mode = "all"
        result = encoder.run_layerwise()
    """

    def __init__(self, cfg: LppFmriEncodingConfig) -> None:
        self.cfg = cfg
        self._precomputed_n_voxels_cache: int | None = None

    def _words_df_for_features(self, sec_df: pd.DataFrame) -> pd.DataFrame:
        """Subset word table rows that receive LM embeddings (aligned with regressors).

        l-bg aligns only non-empty, non-single-space stimulus words — mirror that when
        using ``hf_extraction_mode=\"lbg_fulltext\"``.
        """
        if (
            self.cfg.feature_mode != "huggingface"
            or self.cfg.hf_extraction_mode != "lbg_fulltext"
        ):
            return sec_df
        wc = sec_df["word"]

        def _keep(w: object) -> bool:
            if not isinstance(w, str):
                return False
            return w.strip() != "" and w != " "

        return sec_df.loc[wc.map(_keep)].copy()

    # ------------------------------------------------------------------
    # Brain mask
    # ------------------------------------------------------------------

    def _brain_mask(self, timelines: list) -> nib.Nifti1Image:
        n = len(timelines)
        k = self.cfg.mask_from_runs
        if k is not None:
            k = min(k, n)
            paths = [str(timelines[i].bold_path) for i in range(k)]
        else:
            paths = [str(t.bold_path) for t in timelines]
        return compute_epi_mask(paths)

    def _precomputed_n_voxels(self) -> int:
        """Number of voxels in l-bg precomputed average-subject arrays."""
        if self.cfg.average_subject_precomputed_dir is None:
            raise ValueError("average_subject_precomputed_dir is not set")
        if self._precomputed_n_voxels_cache is not None:
            return self._precomputed_n_voxels_cache
        p = (
            Path(self.cfg.average_subject_precomputed_dir)
            / "average_subject_run-0.gz"
        )
        if not p.is_file():
            raise FileNotFoundError(f"Missing precomputed average file: {p}")
        arr = np.asarray(joblib.load(p))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D precomputed array in {p}, got {arr.shape}")
        self._precomputed_n_voxels_cache = int(arr.shape[1])
        return self._precomputed_n_voxels_cache

    def _result_subject_label(self) -> str:
        if (
            self.cfg.fmri_unit == "average_subject"
            and self.cfg.average_subject_precomputed_dir is not None
        ):
            return f"average_{self.cfg.lang}_precomputed"
        if (
            self.cfg.fmri_unit == "average_subject"
            and self.cfg.average_subject_subjects
        ):
            n = len(self.cfg.average_subject_subjects)
            return f"average_{self.cfg.lang}_{n}subj"
        return str(self.cfg.subject)

    def _load_averaged_masked_bold(
        self,
        section: int,
        mask_img: nib.Nifti1Image,
        voxel_ix: np.ndarray,
    ) -> np.ndarray:
        """Mean ``(n_TR, n_voxel)`` BOLD across ``average_subject_subjects`` (l-bg style)."""
        if self.cfg.average_subject_precomputed_dir is not None:
            p = (
                Path(self.cfg.average_subject_precomputed_dir)
                / f"average_subject_run-{int(section) - 1}.gz"
            )
            if not p.is_file():
                raise FileNotFoundError(
                    f"Precomputed average-subject run not found for section={section}: {p}"
                )
            arr = np.asarray(joblib.load(p), dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2-D array in {p}, got shape {arr.shape}"
                )
            if voxel_ix.size and int(np.max(voxel_ix)) >= arr.shape[1]:
                raise ValueError(
                    "Precomputed average-subject voxel count is smaller than selected "
                    f"indices: max(voxel_ix)={int(np.max(voxel_ix))}, n_vox={arr.shape[1]} "
                    f"(file={p}). Ensure mask / voxel selection matches l-bg ordering."
                )
            return arr[:, voxel_ix]

        subs = self.cfg.average_subject_subjects
        if not subs:
            raise ValueError(
                "average_subject_subjects must be set when fmri_unit='average_subject'"
            )
        ys: list[np.ndarray] = []
        for subj in subs:
            tls = discover_lp_timelines(
                self.cfg.bids_root,
                subj,
                self.cfg.lang,
                source=self.cfg.bold_source,
                echo=self.cfg.echo,
            )
            hit = [t for t in tls if int(t.section) == int(section)]
            if len(hit) != 1:
                raise ValueError(
                    f"Subject {subj!r}: expected one timeline for section {section}, "
                    f"found {len(hit)} (missing runs / misaligned BIDS)."
                )
            ys.append(
                _extract_timeseries_masked(
                    Path(hit[0].bold_path),
                    mask_img,
                    voxel_ix,
                    smoothing_fwhm=self.cfg.smoothing_fwhm,
                    high_pass_hz=self.cfg.high_pass_hz,
                    tr=LPP_TR_SECONDS,
                )
            )
        lengths = [y.shape[0] for y in ys]
        min_t = min(lengths)
        if min(lengths) != max(lengths):
            logger.warning(
                "Average-subject BOLD: TR mismatch across subjects for section %s "
                "(min=%s max=%s) — using first %d TRs.",
                section, min(lengths), max(lengths), min_t,
            )
        stack = np.stack([y[:min_t].astype(np.float64, copy=False) for y in ys], axis=0)
        return np.mean(stack, axis=0)

    # ------------------------------------------------------------------
    # Feature extraction (single run)
    # ------------------------------------------------------------------

    def extract_and_cache_only(self) -> None:
        """Extract + cache raw per-word features for every story section,
        skipping BOLD loading, HRF design and ridge encoding entirely.

        Used by the RSA raw-embedding-cache backfill (2026-09-28): a plain
        ``run_layerwise()``/``run_leave_one_run_out()`` call pays for a full
        LOSO ridge-fit sweep just to get the (cache-side-effect) feature
        extraction as a byproduct -- 40+ minutes for a large model. This
        method does only the extraction, which is what ``feature_cache_dir``
        actually needs; no BOLD/mask/ridge machinery is touched.
        """
        cfg = self.cfg
        timelines = discover_lp_timelines(
            cfg.bids_root, cfg.subject, cfg.lang,
            source=cfg.bold_source, echo=cfg.echo,
        )
        if cfg.max_runs is not None:
            timelines = timelines[: cfg.max_runs]
        words_full = load_word_information_table(
            cfg.bids_root, cfg.lang, apply_lbg_fixes=cfg.apply_lbg_fixes
        )
        for tl in timelines:
            sec_df = words_for_section(words_full, tl.section)
            sec_lm = self._words_df_for_features(sec_df)
            self._extract_features(tl.section, sec_lm)

    def _extract_features(
        self,
        section: int,
        words_df: pd.DataFrame,
    ) -> np.ndarray:
        """Extract features for one section.

        Returns
        -------
        np.ndarray
            ``(n_words, hidden)`` for last/specific layer or GloVe.
            ``(n_words, n_layers, hidden)`` when ``layer_mode="all"``.
        """
        if self.cfg.feature_mode == "glove":
            return load_glove_features(self.cfg.bids_root, self.cfg.lang, words_df.index)

        if self.cfg.feature_mode == "cbr_rnn":
            # CBR-RNN is a custom PyTorchModelHubMixin architecture, not an
            # AutoModelForCausalLM -- it needs its own loader/extractor
            # (neuroai.features.cbr_rnn), not the generic HuggingFace path
            # below. It has a single hidden-state output (n_layers=1), so
            # "layer_mode='all'" just gets one layer's worth of features.
            from neuroai.features.cbr_rnn import contextual_word_embeddings_cbr_rnn

            words = [str(w) for w in words_df["word"].tolist()]
            emb = contextual_word_embeddings_cbr_rnn(
                words,
                self.cfg.hf_model_name,
                device=self.cfg.hf_device,
                context_words=self.cfg.hf_context_words,
                cache_dir=self.cfg.feature_cache_dir,
                section_id=section,
                lang=self.cfg.lang,
            )
            if self.cfg.layer_mode == "all":
                return emb[:, None, :]  # (n_words, 1, hidden)
            return emb

        from neuroai.lpp_llm_features import (
            contextual_word_embeddings_hf,
            contextual_word_embeddings_hf_lbg,
        )

        surp_dtype = "float32" if "gpt-bert" in self.cfg.hf_model_name.lower() else None

        if self.cfg.hf_extraction_mode == "lbg_fulltext":
            return contextual_word_embeddings_hf_lbg(
                words_df,
                self.cfg.hf_model_name,
                bids_root=self.cfg.bids_root,
                lang=self.cfg.lang,
                section_id=section,
                revision=self.cfg.hf_revision,
                device=self.cfg.hf_device,
                layer_index=self.cfg.layer_mode,
                lbg_story_text_zip=self.cfg.hf_lbg_story_text_zip,
                hf_lbg_stride_delta=self.cfg.hf_lbg_stride_delta,
                hf_lbg_max_length=self.cfg.hf_lbg_max_length,
                tokenizer_no_revision=self.cfg.hf_tokenizer_no_revision,
                cache_dir=self.cfg.feature_cache_dir,
                trust_remote_code=self.cfg.hf_trust_remote_code,
                n_params=self.cfg.hf_n_params,
                dtype=surp_dtype,
            )

        words = [str(w) for w in words_df["word"].tolist()]
        return contextual_word_embeddings_hf(
            words,
            self.cfg.hf_model_name,
            layer_index=self.cfg.layer_mode,
            revision=self.cfg.hf_revision,
            device=self.cfg.hf_device,
            context_words=self.cfg.hf_context_words,
            max_length=self.cfg.hf_max_length,
            batch_size=self.cfg.hf_batch_size,
            n_params=self.cfg.hf_n_params,
            cache_dir=self.cfg.feature_cache_dir,
            section_id=section,
            lang=self.cfg.lang,
            tokenizer_no_revision=self.cfg.hf_tokenizer_no_revision,
            trust_remote_code=self.cfg.hf_trust_remote_code,
            untrained=self.cfg.hf_untrained,
            untrained_seed=self.cfg.hf_untrained_seed,
            dtype=surp_dtype,
        )

    # ------------------------------------------------------------------
    # Design matrix construction (single layer view of features)
    # ------------------------------------------------------------------

    def _build_design(
        self,
        features_2d: np.ndarray,
        onsets: np.ndarray,
        durations: np.ndarray,
        frame_times: np.ndarray,
        section: int,
    ) -> np.ndarray:
        """Build HRF design from ``(n_words, n_features)`` features."""
        x_design = features_to_hrf_design(
            onsets, durations, features_2d, frame_times, self.cfg.hrf_model
        )
        if self.cfg.picture_nuisance_regressors:
            pn = picture_block_nuisance_design(
                section, self.cfg.lang, frame_times, self.cfg.hrf_model
            )
            if pn.shape[1] > 0:
                x_design = np.hstack([x_design, pn])
        return x_design

    # ------------------------------------------------------------------
    # Single-run preparation
    # ------------------------------------------------------------------

    def _prepare_run(
        self,
        section: int,
        words_df: pd.DataFrame,
        bold_path: Path,
        mask_img: nib.Nifti1Image,
        voxel_ix: np.ndarray,
        features: np.ndarray,
        *,
        y_masked_full: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_design, y_bold) after trim / z-score.

        ``features`` must be ``(n_words, n_features)`` — pass a 2-D slice when
        the caller has a per-layer array of shape ``(n_words, n_layers, hidden)``.

        If ``y_masked_full`` is set (e.g. group-mean BOLD), it must be
        ``(n_timepoints, n_voxels_sel)`` and ``bold_path`` is ignored for loading.
        """
        onsets = words_df["onset"].to_numpy(dtype=np.float64)
        offsets = words_df["offset"].to_numpy(dtype=np.float64)
        min_dur = max(0.0, float(self.cfg.min_word_duration_sec))
        raw_durations = offsets - onsets
        durations = (
            raw_durations
            if min_dur <= 0.0
            else np.maximum(min_dur, raw_durations)
        )

        if y_masked_full is not None:
            y_full = np.asarray(y_masked_full, dtype=np.float64)
        else:
            y_full = _extract_timeseries_masked(
                bold_path,
                mask_img,
                voxel_ix,
                smoothing_fwhm=self.cfg.smoothing_fwhm,
                high_pass_hz=self.cfg.high_pass_hz,
                tr=LPP_TR_SECONDS,
            )
        n_scans = y_full.shape[0]
        frame_times = stimulus_aligned_frame_times(n_scans)

        x_design = self._build_design(features, onsets, durations, frame_times, section)

        ts, te = self.cfg.trim_trs_start, self.cfg.trim_trs_end
        sl = slice(ts, None if te <= 0 else -te)
        x_trim = x_design[sl]
        y_trim = y_full[sl]
        if self.cfg.strict_lbg_parity:
            x_design = _standardize_lbg(x_trim, axis=0)
            y_full = _standardize_lbg(y_trim, axis=0)
        else:
            x_design = _zscore(x_trim, axis=0)
            y_full = _zscore(y_trim, axis=0)
        return x_design, y_full

    # ------------------------------------------------------------------
    # Alpha selection helper
    # ------------------------------------------------------------------

    def _ridge_n_jobs(self) -> int:
        """Effective parallel workers for the alpha grid (avoid nested oversubscription)."""
        if self.cfg.layer_n_jobs > 1 and self.cfg.ridge_n_jobs > 1:
            return 1
        return max(1, int(self.cfg.ridge_n_jobs))

    def _best_alpha(
        self,
        x_list: list[np.ndarray],
        y_list: list[np.ndarray],
        train_indices: list[int],
    ) -> float:
        """Nested CV: hold out first training run as validation, select alpha."""
        if len(train_indices) < 2:
            return float(self.cfg.ridge_alphas[len(self.cfg.ridge_alphas) // 2])

        val_idx = train_indices[0]
        tr_idx = [i for i in train_indices if i != val_idx]
        x_tr = np.vstack([x_list[i] for i in tr_idx])
        y_tr = np.vstack([y_list[i] for i in tr_idx])
        x_val = x_list[val_idx]
        y_val = y_list[val_idx]

        n_jobs = self._ridge_n_jobs()
        if n_jobs > 1 and len(self.cfg.ridge_alphas) > 1:
            scored = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_score_ridge_alpha_nested_cv)(
                    float(alpha), x_tr, y_tr, x_val, y_val
                )
                for alpha in self.cfg.ridge_alphas
            )
            return max(scored, key=lambda t: t[1])[0]

        best_alpha = float(self.cfg.ridge_alphas[len(self.cfg.ridge_alphas) // 2])
        best_score = -np.inf
        for alpha in self.cfg.ridge_alphas:
            _, sc = _score_ridge_alpha_nested_cv(float(alpha), x_tr, y_tr, x_val, y_val)
            if np.isfinite(sc) and sc > best_score:
                best_score = sc
                best_alpha = float(alpha)
        return best_alpha

    # ------------------------------------------------------------------
    # Hemisphere scoring helper
    # ------------------------------------------------------------------

    def _hemisphere_scores(
        self,
        corrs: np.ndarray,
        voxel_ix: np.ndarray,
        mask_bool: np.ndarray,
        affine: np.ndarray,
    ) -> tuple[float, float]:
        """Return (mean_left, mean_right) Pearson r split by MNI x-coordinate."""
        if self.cfg.average_subject_precomputed_dir is not None:
            # l-bg notebooks split hemisphere by vector halves on averaged-mask order.
            n = int(corrs.shape[0])
            h = n // 2
            left_vals = corrs[:h]
            right_vals = corrs[h:]
            return float(np.nanmean(left_vals)), float(np.nanmean(right_vals))

        x_coords = voxel_mni_x_coords(mask_bool, affine, voxel_ix)
        left_mask = x_coords < 0
        right_mask = x_coords > 0

        def _mean(m: np.ndarray) -> float:
            vals = corrs[m]
            return float(np.nanmean(vals)) if m.any() else float("nan")

        return _mean(left_mask), _mean(right_mask)

    # ------------------------------------------------------------------
    # Per-voxel score computation
    # ------------------------------------------------------------------

    @staticmethod
    def _voxelwise_pearson(y_test: np.ndarray, pred_te: np.ndarray) -> np.ndarray:
        yc = y_test - y_test.mean(axis=0, keepdims=True)
        pc = pred_te - pred_te.mean(axis=0, keepdims=True)
        num = (yc * pc).sum(axis=0)
        den = np.sqrt((yc * yc).sum(axis=0) * (pc * pc).sum(axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corrs = num / den
        corrs[den < 1e-8] = np.nan
        return corrs.astype(np.float64, copy=False)

    @staticmethod
    def _voxelwise_spearman(y_test: np.ndarray, pred_te: np.ndarray) -> np.ndarray:
        """Per-voxel Spearman r = Pearson on the column-wise TR ranks."""
        y_r = np.argsort(np.argsort(y_test, axis=0), axis=0).astype(np.float64)
        p_r = np.argsort(np.argsort(pred_te, axis=0), axis=0).astype(np.float64)
        return LppFmriEncoder._voxelwise_pearson(y_r, p_r)

    # ------------------------------------------------------------------
    # Standard LOSO (single-layer / GloVe)
    # ------------------------------------------------------------------

    def run_leave_one_run_out(self) -> LppEncodingResult:
        """Run leave-one-run-out ridge encoding (single-layer or GloVe mode).

        For per-layer analysis set ``cfg.layer_mode = "all"`` and call
        :meth:`run_layerwise` instead.
        """
        cfg = self.cfg
        timelines = discover_lp_timelines(
            cfg.bids_root, cfg.subject, cfg.lang,
            source=cfg.bold_source, echo=cfg.echo,
        )
        if cfg.max_runs is not None:
            timelines = timelines[: cfg.max_runs]
        if len(timelines) < 2:
            raise ValueError("Need at least two BOLD runs for leave-one-run-out CV.")

        if cfg.fmri_unit == "average_subject":
            if (
                cfg.average_subject_subjects is None
                and cfg.average_subject_precomputed_dir is None
            ):
                raise ValueError(
                    "fmri_unit='average_subject' requires non-empty "
                    "average_subject_subjects or average_subject_precomputed_dir "
                    "in LppFmriEncodingConfig."
                )
            if cfg.average_subject_subjects and cfg.subject not in cfg.average_subject_subjects:
                logger.warning(
                    "cfg.subject=%r is not in average_subject_subjects; "
                    "run↔section pairing still uses this subject's timelines.",
                    cfg.subject,
                )

        words_full = load_word_information_table(
            cfg.bids_root, cfg.lang, apply_lbg_fixes=cfg.apply_lbg_fixes
        )
        mask_img = self._brain_mask(timelines)
        rng = np.random.default_rng(cfg.random_state)

        # Reference image for affine and mask shape
        if cfg.average_subject_precomputed_dir is not None:
            n_vox_full = self._precomputed_n_voxels()
        else:
            ref_img = nib.load(str(timelines[0].bold_path))
            mask_bool = _mask_to_bool_for_img(mask_img, ref_img)
            n_vox_full = int(np.sum(mask_bool))
        voxel_ix = _select_voxel_indices(cfg, n_vox_full, rng)

        sub_lbl = self._result_subject_label()
        logger.info(
            "LPP encoding [%s/%s]: %d timelines, %d mask voxels → using %d",
            sub_lbl, cfg.lang, len(timelines), n_vox_full, voxel_ix.size,
        )
        if cfg.fmri_unit == "average_subject":
            n_avg = (
                len(cfg.average_subject_subjects)
                if cfg.average_subject_subjects
                else -1
            )
            logger.info(
                "LPP fMRI: fmri_unit=average_subject (%s).",
                (
                    f"precomputed_dir={cfg.average_subject_precomputed_dir}"
                    if cfg.average_subject_precomputed_dir is not None
                    else f"averaging {n_avg} subjects from derivatives"
                ),
            )

        # Pre-extract features and BOLD for all runs
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        meta: list[tuple[int, int, int, int]] = []
        all_mask_bools: list[np.ndarray] = []
        all_affines: list[np.ndarray] = []

        for tl in timelines:
            sec_df = words_for_section(words_full, tl.section)
            sec_lm = self._words_df_for_features(sec_df)
            feats = self._extract_features(tl.section, sec_lm)
            # For layer_mode="all", use the LAST layer as the single-mode fallback
            if feats.ndim == 3:
                feats = feats[:, -1, :]
            elif feats.ndim != 2:
                raise ValueError(f"Unexpected feature shape {feats.shape}")

            y_pre = (
                self._load_averaged_masked_bold(tl.section, mask_img, voxel_ix)
                if cfg.fmri_unit == "average_subject"
                else None
            )
            x_i, y_i = self._prepare_run(
                tl.section,
                sec_lm,
                tl.bold_path,
                mask_img,
                voxel_ix,
                feats,
                y_masked_full=y_pre,
            )
            x_list.append(x_i)
            y_list.append(y_i)
            meta.append((tl.section, tl.bids_run, sec_lm.shape[0], x_i.shape[0]))

            if cfg.compute_lateralization:
                img = nib.load(str(tl.bold_path))
                mb = _mask_to_bool_for_img(mask_img, img)
                all_mask_bools.append(mb)
                all_affines.append(img.affine)

        maps_dir = Path(cfg.save_corr_maps_dir) if cfg.save_corr_maps_dir else None
        if maps_dir is not None:
            maps_dir.mkdir(parents=True, exist_ok=True)

        use_gpu = cfg.regression_device == "cuda"
        if use_gpu:
            _ensure_himalaya_cuda()

        n = len(timelines)
        run_results: list[LppEncodingRunResult] = []
        fold_scores: list[float] = []
        fold_scores_sp: list[float] = []
        fold_scores_left: list[float] = []
        fold_scores_right: list[float] = []

        for k in range(n):
            train_idx = [i for i in range(n) if i != k]
            x_tr, y_tr, x_train_full, y_train_full = _loso_nested_alpha_arrays(
                x_list, y_list, train_idx,
            )
            x_test = x_list[k]
            y_test = y_list[k]

            if use_gpu:
                x_val = x_list[train_idx[0]] if len(train_idx) >= 2 else None
                y_val = y_list[train_idx[0]] if len(train_idx) >= 2 else None
                best_alpha, corrs_te, corrs_te_sp = lpp_loso_multioutput_ridge_gpu(
                    x_tr if x_tr is not None else x_train_full,
                    y_tr if y_tr is not None else y_train_full,
                    x_val,
                    y_val,
                    x_train_full,
                    y_train_full,
                    x_test,
                    y_test,
                    ridge_alphas=cfg.ridge_alphas,
                )
            else:
                best_alpha = self._best_alpha(x_list, y_list, train_idx)
                model = Ridge(alpha=best_alpha, fit_intercept=False)
                model.fit(x_train_full, y_train_full)
                pred_te = model.predict(x_test)
                corrs_te = self._voxelwise_pearson(y_test, pred_te)
                corrs_te_sp = self._voxelwise_spearman(y_test, pred_te)

            mean_r = float(np.nanmean(corrs_te))
            mean_sp = float(np.nanmean(corrs_te_sp))
            fold_scores.append(mean_r)
            fold_scores_sp.append(mean_sp)

            left_r, right_r = float("nan"), float("nan")
            lat_idx = float("nan")
            if cfg.compute_lateralization and all_mask_bools:
                mb = all_mask_bools[k]
                aff = all_affines[k]
                left_r, right_r = self._hemisphere_scores(corrs_te, voxel_ix, mb, aff)
                if np.isfinite(left_r) and np.isfinite(right_r):
                    lat_idx = left_r - right_r
                fold_scores_left.append(left_r)
                fold_scores_right.append(right_r)

            sec, br, nw, ntr = meta[k]
            run_results.append(
                LppEncodingRunResult(
                    section=sec, bids_run=br, n_words=nw, n_tr_used=ntr,
                    n_voxels=voxel_ix.size, mean_pearson=mean_r,
                    best_alpha=best_alpha,
                    mean_pearson_left=left_r,
                    mean_pearson_right=right_r,
                    lateralization_index=lat_idx,
                    mean_spearman=mean_sp,
                )
            )

            # Correlation maps
            if maps_dir is not None:
                img_te = nib.load(str(timelines[k].bold_path))
                mask_te = _mask_to_bool_for_img(mask_img, img_te)
                nii = correlations_to_masked_nifti(
                    corrs_te.astype(np.float32), voxel_ix, mask_te, img_te.affine
                )
                out_nii = maps_dir / f"{cfg.subject}_fold{k}_section{sec}_run{br}_corr_pearson.nii.gz"
                nib.save(nii, str(out_nii))

            logger.info(
                "LOSO fold test section=%d bids_run=%d mean_pearson=%.4f"
                " left=%.4f right=%.4f alpha=%.2g",
                sec, br, mean_r, left_r, right_r, best_alpha,
            )

        overall = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
        overall_sp = float(np.nanmean(fold_scores_sp)) if fold_scores_sp else float("nan")
        overall_left = float(np.nanmean(fold_scores_left)) if fold_scores_left else float("nan")
        overall_right = float(np.nanmean(fold_scores_right)) if fold_scores_right else float("nan")
        overall_lat = (
            overall_left - overall_right
            if np.isfinite(overall_left) and np.isfinite(overall_right)
            else float("nan")
        )
        return LppEncodingResult(
            subject=self._result_subject_label(),
            lang=cfg.lang,
            runs=run_results,
            overall_mean_pearson=overall,
            overall_mean_pearson_left=overall_left,
            overall_mean_pearson_right=overall_right,
            overall_lateralization_index=overall_lat,
            overall_mean_spearman=overall_sp,
        )

    # ------------------------------------------------------------------
    # Layerwise LOSO (layer_mode="all")
    # ------------------------------------------------------------------

    def run_layerwise(self) -> LppLayerwiseEncodingResult:
        """Run per-layer LOSO ridge encoding when ``layer_mode='all'``.

        Extracts all hidden layers from the HF model, builds separate
        HRF design matrices per layer, and runs ridge regression per layer.
        Optionally computes left/right hemisphere scores.

        This method is significantly slower than ``run_leave_one_run_out``
        because it repeats ridge fitting ``n_layers`` times per LOSO fold.
        GPU extraction is strongly recommended.
        """
        cfg = self.cfg
        if cfg.feature_mode == "glove":
            raise ValueError(
                "layer_mode='all' requires feature_mode='huggingface'. "
                "GloVe embeddings have no layer dimension."
            )

        timelines = discover_lp_timelines(
            cfg.bids_root, cfg.subject, cfg.lang,
            source=cfg.bold_source, echo=cfg.echo,
        )
        if cfg.max_runs is not None:
            timelines = timelines[: cfg.max_runs]
        if len(timelines) < 2:
            raise ValueError("Need at least two BOLD runs for leave-one-run-out CV.")

        if cfg.fmri_unit == "average_subject":
            if (
                cfg.average_subject_subjects is None
                and cfg.average_subject_precomputed_dir is None
            ):
                raise ValueError(
                    "fmri_unit='average_subject' requires non-empty "
                    "average_subject_subjects or average_subject_precomputed_dir "
                    "in LppFmriEncodingConfig."
                )
            if cfg.average_subject_subjects and cfg.subject not in cfg.average_subject_subjects:
                logger.warning(
                    "cfg.subject=%r is not in average_subject_subjects; "
                    "run↔section pairing still uses this subject's timelines.",
                    cfg.subject,
                )

        words_full = load_word_information_table(
            cfg.bids_root, cfg.lang, apply_lbg_fixes=cfg.apply_lbg_fixes
        )
        mask_img = self._brain_mask(timelines)
        rng = np.random.default_rng(cfg.random_state)

        if cfg.average_subject_precomputed_dir is not None:
            n_vox_full = self._precomputed_n_voxels()
        else:
            ref_img = nib.load(str(timelines[0].bold_path))
            mask_bool_ref = _mask_to_bool_for_img(mask_img, ref_img)
            n_vox_full = int(np.sum(mask_bool_ref))
        voxel_ix = _select_voxel_indices(cfg, n_vox_full, rng)

        sub_lbl = self._result_subject_label()
        logger.info(
            "LPP layerwise encoding [%s/%s]: %d timelines, %d mask voxels → %d",
            sub_lbl, cfg.lang, len(timelines), n_vox_full, voxel_ix.size,
        )
        if cfg.fmri_unit == "average_subject":
            logger.info(
                "LPP fMRI layerwise: fmri_unit=average_subject (%s).",
                (
                    f"precomputed_dir={cfg.average_subject_precomputed_dir}"
                    if cfg.average_subject_precomputed_dir is not None
                    else f"averaging {len(cfg.average_subject_subjects or ())} subjects from derivatives"
                ),
            )

        # Pre-extract per-layer features (n_words, n_layers, hidden) per run
        # and preprocess BOLD
        section_feats: list[np.ndarray] = []  # each: (n_words, n_layers, hidden)
        section_meta: list[tuple[int, int, int]] = []  # (section, bids_run, n_words)
        y_list: list[np.ndarray] = []
        onsets_list: list[np.ndarray] = []
        offsets_list: list[np.ndarray] = []
        n_scans_list: list[int] = []
        all_mask_bools: list[np.ndarray] = []
        all_affines: list[np.ndarray] = []

        n_layers: int = 0

        for tl in timelines:
            sec_df = words_for_section(words_full, tl.section)
            sec_lm = self._words_df_for_features(sec_df)
            # Extract ALL layers: (n_words, n_layers, hidden)
            feats_all = self._extract_features(tl.section, sec_lm)
            if feats_all.ndim != 3:
                raise ValueError(
                    f"Expected (n_words, n_layers, hidden) from layer_mode='all', "
                    f"got shape {feats_all.shape}"
                )
            n_layers = feats_all.shape[1]
            section_feats.append(feats_all)
            section_meta.append((tl.section, tl.bids_run, sec_lm.shape[0]))

            onsets_list.append(sec_lm["onset"].to_numpy(dtype=np.float64))
            offsets_list.append(sec_lm["offset"].to_numpy(dtype=np.float64))

            if cfg.fmri_unit == "average_subject":
                y_full = self._load_averaged_masked_bold(
                    tl.section, mask_img, voxel_ix,
                )
            else:
                y_full = _extract_timeseries_masked(
                    tl.bold_path, mask_img, voxel_ix,
                    smoothing_fwhm=cfg.smoothing_fwhm,
                    high_pass_hz=cfg.high_pass_hz,
                    tr=LPP_TR_SECONDS,
                )
            n_scans_list.append(y_full.shape[0])
            ts, te = cfg.trim_trs_start, cfg.trim_trs_end
            sl = slice(ts, None if te <= 0 else -te)
            if cfg.strict_lbg_parity:
                y_trimmed = _standardize_lbg(y_full[sl], axis=0)
            else:
                y_trimmed = _zscore(y_full[sl], axis=0)
            y_list.append(y_trimmed)

            if cfg.compute_lateralization:
                img = nib.load(str(tl.bold_path))
                all_mask_bools.append(_mask_to_bool_for_img(mask_img, img))
                all_affines.append(img.affine)

        logger.info("Detected n_layers=%d from model", n_layers)

        # Feature extraction (the only step that needs the LLM) is done — all
        # layers/sections are already materialised in `section_feats` as numpy
        # arrays. Free the model from GPU memory before the LOSO×layer ridge
        # loop below: for large models (device_map="auto" across multiple
        # GPUs, e.g. OPT-66B needing ~69GiB/GPU) the model would otherwise
        # stay resident throughout ridge fitting, leaving too little headroom
        # for the GPU-batched ridge regression's own tensors and causing an
        # OOM that more GPUs would only partially mask (ridge workers all run
        # on the default CUDA device regardless of how many GPUs are visible).
        if cfg.feature_mode == "huggingface":
            from neuroai.lpp_llm_features import clear_hf_model_cache as _clear_lpp_hf_cache

            _clear_lpp_hf_cache()

        # Precompute trimmed word→TR HRF bases once per run (shared across layers & folds).
        # Without this, each layer×fold×run called compute_regressor ~hidden times → millions
        # of Python/nilearn calls (~hours per checkpoint).  Here: ~n_words singleton calls per
        # run, then only cheap matmuls ``B @ feats_layer``.
        ts_tr, te_tr = cfg.trim_trs_start, cfg.trim_trs_end
        trim_sl = slice(ts_tr, None if te_tr <= 0 else -te_tr)
        basis_sl_list: list[np.ndarray] = []
        pn_sl_list: list[np.ndarray] = []
        for ri, tl in enumerate(timelines):
            ons = onsets_list[ri]
            min_dur = max(0.0, float(cfg.min_word_duration_sec))
            raw_durs = offsets_list[ri] - ons
            durs = raw_durs if min_dur <= 0.0 else np.maximum(min_dur, raw_durs)
            ft = stimulus_aligned_frame_times(n_scans_list[ri])
            B_full = word_hrf_basis_matrix(ons, durs, ft, cfg.hrf_model)
            B_sl = B_full[trim_sl].astype(np.float64, copy=False)
            basis_sl_list.append(B_sl)
            if cfg.picture_nuisance_regressors:
                pn_full = picture_block_nuisance_design(
                    tl.section, cfg.lang, ft, cfg.hrf_model
                )
                if pn_full.shape[1] > 0:
                    pn_sl_list.append(pn_full[trim_sl].astype(np.float64, copy=False))
                else:
                    pn_sl_list.append(
                        np.zeros((B_sl.shape[0], 0), dtype=np.float64),
                    )
            else:
                pn_sl_list.append(
                    np.zeros((B_sl.shape[0], 0), dtype=np.float64),
                )
        logger.info(
            "Precomputed word HRF bases for %d runs (%d words/run median); "
            "layer/fold loop uses matmul only",
            len(timelines),
            int(np.median([section_feats[i].shape[0] for i in range(len(timelines))]))
            if timelines
            else 0,
        )

        n = len(timelines)
        fold_layer_scores: list[list[float]] = [[] for _ in range(n_layers)]
        fold_layer_scores_left: list[list[float]] = [[] for _ in range(n_layers)]
        fold_layer_scores_right: list[list[float]] = [[] for _ in range(n_layers)]
        fold_layer_spearman: list[list[float]] = [[] for _ in range(n_layers)]
        fold_layer_alphas: list[list[float]] = [[] for _ in range(n_layers)]
        fold_voxel_corrs: list[list[np.ndarray]] = [[] for _ in range(n_layers)]

        run_results: list[LppLayerwiseRunResult] = []
        fold_details: list[LppLayerwiseFoldDetail] = []
        fold_predictions: list[dict[str, tp.Any]] = []

        use_vector_hemi = cfg.average_subject_precomputed_dir is not None
        use_gpu = cfg.regression_device == "cuda"
        if use_gpu:
            _ensure_himalaya_cuda()
            import torch

        layer_jobs = max(1, int(cfg.n_layer_workers if use_gpu else cfg.layer_n_jobs))
        ridge_jobs = self._ridge_n_jobs()
        if use_gpu:
            logger.info(
                "Layer parallelism (GPU): n_layer_workers=%d (ridge on CUDA, no joblib layers)",
                layer_jobs,
            )
        elif layer_jobs > 1:
            logger.info(
                "Layer parallelism: layer_n_jobs=%d (ridge_n_jobs capped to %d per layer)",
                layer_jobs,
                ridge_jobs,
            )

        fold_common = dict(
            section_feats=section_feats,
            basis_sl_list=basis_sl_list,
            pn_sl_list=pn_sl_list,
            y_list=y_list,
            ridge_alphas=cfg.ridge_alphas,
            ridge_n_jobs=ridge_jobs,
            strict_lbg_parity=cfg.strict_lbg_parity,
            compute_lateralization=cfg.compute_lateralization,
            use_vector_hemisphere_split=use_vector_hemi,
            voxel_ix=voxel_ix,
            all_mask_bools=all_mask_bools,
            all_affines=all_affines,
            regression_device=cfg.regression_device,
        )

        def _fit_one_layer(li: int, train_idx: list[int], test_k: int) -> dict[str, tp.Any]:
            want_pred = cfg.save_word_predictions and li == n_layers - 1
            if use_gpu:
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    out = _fit_layer_loso_fold(
                        li,
                        train_idx=train_idx,
                        test_k=test_k,
                        return_predictions=want_pred,
                        **fold_common,
                    )
                stream.synchronize()
                return out
            return _fit_layer_loso_fold(
                li,
                train_idx=train_idx,
                test_k=test_k,
                return_predictions=want_pred,
                **fold_common,
            )

        for k in range(n):
            train_idx = [i for i in range(n) if i != k]
            sec, br, nw = section_meta[k]
            n_tr_used = y_list[k].shape[0]

            layer_scores_this_fold: list[float] = []
            layer_sp_this_fold: list[float] = []
            layer_left: list[float] = []
            layer_right: list[float] = []
            layer_alphas_this_fold: list[float] = []

            best_alpha_overall = float(cfg.ridge_alphas[len(cfg.ridge_alphas) // 2])

            if layer_jobs > 1 and n_layers > 1:
                if use_gpu:
                    layer_out: list[dict[str, tp.Any]] = [None] * n_layers  # type: ignore[list-item]
                    with ThreadPoolExecutor(max_workers=min(layer_jobs, n_layers)) as pool:
                        futures = {
                            pool.submit(_fit_one_layer, li, train_idx, k): li
                            for li in range(n_layers)
                        }
                        for fut in as_completed(futures):
                            out = fut.result()
                            layer_out[int(out["layer"])] = out
                    layer_out = [d for d in layer_out if d is not None]
                else:
                    layer_out = Parallel(n_jobs=min(layer_jobs, n_layers), backend="loky")(
                        delayed(_fit_layer_loso_fold)(
                            li,
                            train_idx=train_idx,
                            test_k=k,
                            return_predictions=(cfg.save_word_predictions and li == n_layers - 1),
                            **fold_common,
                        )
                        for li in range(n_layers)
                    )
                    layer_out.sort(key=lambda d: d["layer"])
            else:
                layer_out = [
                    _fit_one_layer(li, train_idx, k)
                    for li in range(n_layers)
                ]

            for out in layer_out:
                li = int(out["layer"])
                mean_r = float(out["mean_pearson"])
                mean_sp = float(out.get("mean_spearman", float("nan")))
                alpha_li = float(out["best_alpha"])
                layer_scores_this_fold.append(mean_r)
                layer_sp_this_fold.append(mean_sp)
                layer_alphas_this_fold.append(alpha_li)
                fold_layer_scores[li].append(mean_r)
                fold_layer_spearman[li].append(mean_sp)
                fold_layer_alphas[li].append(alpha_li)
                fold_voxel_corrs[li].append(out["corrs"])

                if cfg.save_word_predictions and out.get("pred") is not None:
                    fold_predictions.append({
                        "section": sec,
                        "bids_run": br,
                        "layer": li,
                        "y_test": out["y_test"],
                        "pred": out["pred"],
                    })

                if li == n_layers // 2:
                    best_alpha_overall = alpha_li

                lr = float(out["mean_pearson_left"])
                rr = float(out["mean_pearson_right"])
                if cfg.compute_lateralization:
                    layer_left.append(lr)
                    layer_right.append(rr)
                    fold_layer_scores_left[li].append(lr)
                    fold_layer_scores_right[li].append(rr)

                lat = float("nan")
                if np.isfinite(lr) and np.isfinite(rr):
                    lat = lr - rr
                fold_details.append(
                    LppLayerwiseFoldDetail(
                        section=sec,
                        bids_run=br,
                        layer=li,
                        best_alpha=alpha_li,
                        mean_pearson=mean_r,
                        mean_pearson_left=lr,
                        mean_pearson_right=rr,
                        lateralization_index=lat,
                        n_voxels=voxel_ix.size,
                        mean_spearman=mean_sp,
                    )
                )

            run_results.append(
                LppLayerwiseRunResult(
                    section=sec, bids_run=br, n_words=nw, n_tr_used=n_tr_used,
                    n_voxels=voxel_ix.size,
                    layer_mean_pearson=layer_scores_this_fold,
                    layer_mean_pearson_left=layer_left,
                    layer_mean_pearson_right=layer_right,
                    layer_best_alphas=layer_alphas_this_fold,
                    best_alpha=best_alpha_overall,
                    layer_mean_spearman=layer_sp_this_fold,
                )
            )
            if layer_alphas_this_fold:
                arr = np.asarray(layer_alphas_this_fold, dtype=float)
                counts = np.unique(arr, return_counts=True)
                mode_idx = int(np.argmax(counts[1]))
                mode_a = float(counts[0][mode_idx])
                logger.info(
                    "LOSO layerwise fold section=%d bids_run=%d best_layer_pearson=%.4f | "
                    "chosen alphas: mode=%.4g median=%.4g min=%.4g max=%.4g",
                    sec,
                    br,
                    max(layer_scores_this_fold) if layer_scores_this_fold else float("nan"),
                    mode_a,
                    float(np.median(arr)),
                    float(np.min(arr)),
                    float(np.max(arr)),
                )
                top = sorted(
                    zip(counts[0].tolist(), counts[1].tolist()),
                    key=lambda t: t[1],
                    reverse=True,
                )[:5]
                top_str = ", ".join(f"{a:.4g}×{n}" for a, n in top)
                logger.info(
                    "  fold section=%d top alphas (layer×1 pick each): %s",
                    sec,
                    top_str,
                )
            else:
                logger.info(
                    "LOSO layerwise fold section=%d bids_run=%d best_layer_pearson=%.4f",
                    sec,
                    br,
                    max(layer_scores_this_fold) if layer_scores_this_fold else float("nan"),
                )

        layer_mean_pearson = [
            float(np.nanmean(fold_layer_scores[li])) for li in range(n_layers)
        ]
        layer_mean_spearman = [
            float(np.nanmean(fold_layer_spearman[li])) if fold_layer_spearman[li] else float("nan")
            for li in range(n_layers)
        ]
        layer_mean_left = [
            float(np.nanmean(fold_layer_scores_left[li])) if fold_layer_scores_left[li] else float("nan")
            for li in range(n_layers)
        ]
        layer_mean_right = [
            float(np.nanmean(fold_layer_scores_right[li])) if fold_layer_scores_right[li] else float("nan")
            for li in range(n_layers)
        ]
        layer_mean_alphas = [
            float(np.nanmean(fold_layer_alphas[li])) if fold_layer_alphas[li] else float("nan")
            for li in range(n_layers)
        ]
        layer_voxel_mean_pearson = [
            np.nanmean(np.stack(fold_voxel_corrs[li], axis=0), axis=0)
            if fold_voxel_corrs[li]
            else np.array([], dtype=np.float64)
            for li in range(n_layers)
        ]

        voxel_roi_labels: np.ndarray | None = None
        if cfg.brain_mask_path and cfg.roi_masks_dir:
            try:
                from neuroai.roi.lpp import load_lbg_voxel_roi_labels

                full_labels = load_lbg_voxel_roi_labels(
                    cfg.brain_mask_path, cfg.roi_masks_dir
                )
                voxel_roi_labels = full_labels[voxel_ix]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load ROI labels: %s", exc)

        return LppLayerwiseEncodingResult(
            subject=self._result_subject_label(),
            lang=cfg.lang,
            n_layers=n_layers,
            runs=run_results,
            layer_mean_pearson=layer_mean_pearson,
            layer_mean_pearson_left=layer_mean_left,
            layer_mean_pearson_right=layer_mean_right,
            layer_mean_alphas=layer_mean_alphas,
            layer_mean_spearman=layer_mean_spearman,
            voxel_global_ix=voxel_ix.astype(np.int64, copy=False),
            fold_details=fold_details,
            layer_voxel_mean_pearson=layer_voxel_mean_pearson,
            voxel_roi_labels=voxel_roi_labels,
            fold_predictions=fold_predictions or None,
        )


# ---------------------------------------------------------------------------
# DataFrame serialization
# ---------------------------------------------------------------------------

def result_to_dataframe(res: LppEncodingResult) -> pd.DataFrame:
    rows = [r.to_dict() for r in res.runs]
    df = pd.DataFrame(rows)
    df["subject"] = res.subject
    df["lang"] = res.lang
    return df


def layerwise_result_to_dataframe(res: LppLayerwiseEncodingResult) -> pd.DataFrame:
    rows = [r.to_dict() for r in res.runs]
    df = pd.DataFrame(rows)
    df["subject"] = res.subject
    df["lang"] = res.lang
    return df


def layerwise_summary_dataframe(res: LppLayerwiseEncodingResult) -> pd.DataFrame:
    """Per-layer summary (mean across folds) as a tidy DataFrame."""
    rows = []
    for li in range(res.n_layers):
        row: dict[str, tp.Any] = {
            "subject": res.subject,
            "lang": res.lang,
            "layer": li,
            "mean_pearson": res.layer_mean_pearson[li] if li < len(res.layer_mean_pearson) else float("nan"),
            "mean_spearman": res.layer_mean_spearman[li] if li < len(res.layer_mean_spearman) else float("nan"),
        }
        if res.layer_mean_pearson_left:
            row["mean_pearson_left"] = res.layer_mean_pearson_left[li] if li < len(res.layer_mean_pearson_left) else float("nan")
            row["mean_pearson_right"] = res.layer_mean_pearson_right[li] if li < len(res.layer_mean_pearson_right) else float("nan")
            if np.isfinite(row["mean_pearson_left"]) and np.isfinite(row["mean_pearson_right"]):
                row["lateralization_index"] = row["mean_pearson_left"] - row["mean_pearson_right"]
            else:
                row["lateralization_index"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "LPP_SECTION1_PICTURE_BLOCKS_SEC",
    "word_hrf_basis_matrix",
    "LppEncodingResult",
    "LppEncodingRunResult",
    "LppFmriEncoder",
    "LppFmriEncodingConfig",
    "LppLayerwiseEncodingResult",
    "LppLayerwiseRunResult",
    "correlations_to_masked_nifti",
    "features_to_hrf_design",
    "layerwise_result_to_dataframe",
    "layerwise_summary_dataframe",
    "load_glove_features",
    "parse_glove_vector",
    "picture_block_nuisance_design",
    "result_to_dataframe",
    "stimulus_aligned_frame_times",
    "voxel_mni_x_coords",
]
