"""Per-subject, Fedorenko-parcel-averaged LPP fMRI BOLD extraction.

The real production LPP LME pipeline (``neuroai/benchmarks/lpp_lme.py``)
only ever pools AVERAGE-SUBJECT BOLD (l-bg's precomputed
``average_subject_run-*.gz``) -- there is no ``subject`` random-effect
grouping factor anywhere in that formula (see
``neuroai/regressors/lpp_pooled_lme.py``'s module docstring). Building a
genuine split-half-by-SUBJECT noise ceiling (Timkey et al. 2026) requires a
per-subject BOLD signal that does not exist in any cached table, so this
module rebuilds it directly from each subject's own preprocessed NIfTI runs
-- the one piece of this session's noise-ceiling work that touches raw
voxel-level data instead of a cached pooled table.

English subjects only (``sub-EN*``): LPP's canonical brain mask
(``mask_lpp_en.nii.gz``), Fedorenko parcel atlas, and ISC reliability scores
are all English-language-specific (``*_en.*`` file naming) -- French/Chinese
subjects use different masks/atlases and cannot be pooled into the same
voxel space.

Alignment strategy
-------------------
The real pipeline's ``fmri_unit="average_subject"`` fallback path (no
precomputed dir) derives its own per-call EPI mask
(``LppFmriEncoder._brain_mask`` -> ``compute_epi_mask``), which is NOT
guaranteed to align with ``brain_mask_path``'s voxel order -- that's why
``LppLmeBenchmark`` refuses to run without the precomputed-average path (see
its module docstring). This module avoids that problem by using
``brain_mask_path`` itself (the same canonical mask whose voxel order
defines both the precomputed average files and the Fedorenko region masks)
as the ``mask_img`` passed to ``_extract_timeseries_masked`` for every
subject -- that function internally resamples the mask onto each subject's
own EPI grid (nearest-neighbor), the same resampling step already used
throughout ``lpp_encoding.py``'s single-subject and on-the-fly
average-subject paths. ``voxel_ix`` (the ISC reliable-voxel selection) is
recomputed with the exact same deterministic call
(``_select_voxel_indices``, ``reliable_voxel_fraction=0.25``,
``max_voxels=None`` -> no randomness actually used) as the real sweep, so
Fedorenko region masks built the same way as
``LppLmeBenchmark.run()`` line up with this module's voxel columns.

Item key for the split-half ceiling
------------------------------------
Row order within a (subject, region, section) block is TR order after
trimming -- ``tr_idx = 0..n_trs-1``. The split-half "item" is
``f"{section}_{tr_idx}"`` (matching Natural Stories fMRI's row-order
item-key convention), collapsed across region and subject when computing
the item-mean substitute -- the same coarser-than-ideal but established
convention Tuckute's ceiling already uses (its ``sentence_id`` item key
also collapses across ``roi``). Caveat: if subjects' raw (pre-trim) TR
counts differ slightly, ``tr_idx`` alignment across subjects for the same
section could be off by a few TRs -- not independently verified here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LPP_ENGLISH_SUBJECT_PREFIX = "sub-EN"


def discover_english_lpp_subjects(bids_root: str | Path) -> list[str]:
    """List every English (``sub-EN*``) LPP subject directory under ``bids_root``."""
    root = Path(bids_root)
    subs = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.startswith(LPP_ENGLISH_SUBJECT_PREFIX)
    )
    if not subs:
        raise RuntimeError(f"No {LPP_ENGLISH_SUBJECT_PREFIX}* subject directories found under {root}")
    return subs


def build_lpp_persubject_bold_table(
    *,
    bids_root: str | Path,
    subjects: list[str],
    lang: str = "en",
    bold_source: str = "derivatives",
    echo: int | None = 2,
    sections: list[int] | None = None,
    t_r: float = 2.0,
    trim_trs_start: int = 10,
    trim_trs_end: int = 10,
    smoothing_fwhm: float | None = 4.0,
    high_pass_hz: float | None = 0.0078125,
    brain_mask_path: str,
    reliable_voxel_fraction: float | None = 0.25,
    reliable_voxel_scores_path: str | None = None,
    fedorenko_atlas_nii_path: str,
    fedorenko_labels_txt_path: str | None = None,
    split_random_state: int = 0,
) -> pd.DataFrame:
    """Long-format (subject, region, section, tr_idx, bold) table.

    One row per (subject, region-hemisphere, section, TR) after the same
    trim used in production. Does NOT include wlen/freq/sentpos/surprisal --
    merge those in from an existing cached ``lpp_lme_pooled_table.parquet``
    (they only depend on words/HRF design, identical for every subject) by
    ``(region, section, tr_idx)``. See module docstring for the alignment
    strategy and item-key convention.
    """
    import nibabel as nib

    from ..datasets.le_petit_prince import LPP_N_ANNOTATION_SECTIONS, discover_lp_timelines
    from ..lpp_encoding import (
        LppFmriEncodingConfig,
        _extract_timeseries_masked,
        _select_voxel_indices,
    )
    from ..roi.lpp import load_fedorenko_parcelization

    secs = sections or list(range(1, LPP_N_ANNOTATION_SECTIONS + 1))

    mask_img = nib.load(str(brain_mask_path))
    n_vox_full = int((mask_img.get_fdata() > 0.5).sum())

    sel_cfg = LppFmriEncodingConfig(
        bids_root=str(bids_root),
        subject=subjects[0],
        lang=lang,
        fmri_unit="single_subject",
        reliable_voxel_fraction=reliable_voxel_fraction,
        reliable_voxel_scores_path=reliable_voxel_scores_path,
        max_voxels=None,
        brain_mask_path=str(brain_mask_path),
    )
    rng = np.random.default_rng(split_random_state)
    voxel_ix = _select_voxel_indices(sel_cfg, n_vox_full, rng)
    logger.info("LPP per-subject extraction: %d/%d voxels selected", voxel_ix.size, n_vox_full)

    parcelization = load_fedorenko_parcelization(
        str(brain_mask_path), str(fedorenko_atlas_nii_path), fedorenko_labels_txt_path,
    )
    region_names: list[str] = []
    region_masks_selected: list[np.ndarray] = []
    for ri, name in enumerate(parcelization.region_names):
        for hemi, full_mask in (("L", parcelization.left_masks[ri]), ("R", parcelization.right_masks[ri])):
            mask_sel = full_mask[voxel_ix]
            if not mask_sel.any():
                logger.warning("region %s_%s has 0 selected voxels -- dropping.", hemi, name)
                continue
            region_names.append(f"{hemi}_{name}")
            region_masks_selected.append(mask_sel)
    if not region_names:
        raise RuntimeError("No Fedorenko-parcel region has any selected voxel")

    subject_parts: list[np.ndarray] = []
    region_parts: list[np.ndarray] = []
    section_parts: list[np.ndarray] = []
    tr_idx_parts: list[np.ndarray] = []
    bold_parts: list[np.ndarray] = []

    ts, te = trim_trs_start, trim_trs_end
    sl = slice(ts, None if te <= 0 else -te)

    for si, subject in enumerate(subjects):
        timelines = discover_lp_timelines(bids_root, subject, lang, source=bold_source, echo=echo)
        by_section = {int(t.section): t for t in timelines}
        for section in secs:
            tl = by_section.get(int(section))
            if tl is None:
                logger.warning("[%s] no timeline for section %d -- skipping.", subject, section)
                continue
            try:
                y = _extract_timeseries_masked(
                    Path(tl.bold_path), mask_img, voxel_ix,
                    smoothing_fwhm=smoothing_fwhm, high_pass_hz=high_pass_hz, tr=t_r,
                )
            except Exception as exc:
                logger.warning("[%s | section %d] BOLD extraction failed: %s", subject, section, exc)
                continue
            y_trim = y[sl]
            n_trs = y_trim.shape[0]
            if n_trs < 5:
                logger.warning("[%s | section %d] only %d TRs after trim -- skipping.", subject, section, n_trs)
                continue
            tr_idx = np.arange(n_trs)
            for region_name, mask_sel in zip(region_names, region_masks_selected):
                region_bold = np.nanmean(y_trim[:, mask_sel], axis=1)
                subject_parts.append(np.full(n_trs, subject, dtype=object))
                region_parts.append(np.full(n_trs, region_name, dtype=object))
                section_parts.append(np.full(n_trs, str(section), dtype=object))
                tr_idx_parts.append(tr_idx)
                bold_parts.append(region_bold)
        logger.info("[%d/%d] %s done", si + 1, len(subjects), subject)

    if not bold_parts:
        raise RuntimeError("No (subject, region, section) rows assembled")

    return pd.DataFrame({
        "subject": np.concatenate(subject_parts),
        "region": np.concatenate(region_parts),
        "section": np.concatenate(section_parts),
        "tr_idx": np.concatenate(tr_idx_parts),
        "bold": np.concatenate(bold_parts),
    })
