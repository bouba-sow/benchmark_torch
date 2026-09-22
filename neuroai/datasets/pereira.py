"""Pereira et al. (2018) fMRI dataset loader.

Paper: "Toward a universal decoder of linguistic meaning from brain
activation" — Nature Communications (2018). https://doi.org/10.1038/s41467-018-03068-4

Dataset: 627 isolated sentences (243sentences + 384sentences experiments,
3-4-sentence passages, one sentence presented at a time) x up to 10
participants x per-voxel BOLD, annotated with a 5-atlas ROI parcellation
(visual/MD/language/DMN/auditory). Same "isolated sentence, per-trial
scalar-per-region BOLD" shape as Tuckute (see ``neuroai/datasets/tuckute.py``)
-- no HRF needed, one response per (subject, region, sentence).

Data layout expected under ``nc_path`` (default
``data/pereira_fmri/Pereira2018_merged.nc``): a single xarray DataArray with dims
``(presentation, neuroid)``, coords ``stimulus_id``/``sentence``/``experiment``
on ``presentation`` and ``subject``/``roi``/``atlas`` on ``neuroid`` -- see
``data/pereira_fmri/merge_pereira.py`` for how this was built (brainio's own
``data/pereira_fmri/Pereira2018.nc`` + ``data/pereira_fmri/Pereira2018-stimulus_set.csv``, merged and split).

12 language-network ROI labels (``LH_AntTemp``, ``RH_PostTemp``, ...) are
present in EVERY one of the 10 subjects -- a genuinely crossed random effect,
like Tuckute's ``LANG_ROIS`` (6 labels) or a scalp electrode, not a
per-subject electrode/voxel grid. Pereira keeps hundreds of voxels per
(subject, roi) pair; this loader averages across voxels within each
(subject, roi) group to collapse to one BOLD scalar per (subject, roi,
sentence) -- same fROI-averaging convention as Natural Stories fMRI /
Fedorenko-parcel LPP.

Usage::

    ds = PereiraDataset(Path("data/pereira_fmri/Pereira2018_merged.nc"))
    sentences, item_ids = ds.get_sentences()
    # sentences : list[str]  (n_sentences,) -- 627 unique sentences
    # item_ids  : np.ndarray (n_sentences,) str -- stimulus_id, stable ordering key

    ds.rois  # ['LH_AngG', 'LH_AntTemp', ..., 'RH_PostTemp'] (12, atlas="language")

    per_subject = ds.get_condition_per_subject(atlas="language")
    # dict: subject_id (str) -> (bold (n_seen, n_rois), item_ids (n_seen,))
"""

from __future__ import annotations

import logging
import warnings
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ATLAS = "language"


class PereiraDataset:
    def __init__(self, nc_path: str | Path = "data/pereira_fmri/Pereira2018_merged.nc") -> None:
        self.nc_path = Path(nc_path)
        if not self.nc_path.is_file():
            raise FileNotFoundError(
                f"{self.nc_path} not found -- see data/pereira_fmri/merge_pereira.py to build it "
                "from brainio's data/pereira_fmri/Pereira2018.nc + data/pereira_fmri/Pereira2018-stimulus_set.csv."
            )

    @cached_property
    def _assembly(self):
        import xarray as xr

        return xr.open_dataarray(self.nc_path)

    @cached_property
    def _sentence_by_stimulus_id(self) -> dict[str, str]:
        a = self._assembly
        return dict(zip(a["stimulus_id"].values.tolist(), a["sentence"].values.tolist()))

    @cached_property
    def _passage_group_by_stimulus_id(self) -> dict[str, str]:
        """``stimulus_id -> "<experiment>_p<passage_index>"``.

        ``passage_index`` alone is NOT globally unique (it restarts within
        each of the two sub-experiments, 243sentences/384sentences) -- see
        module docstring's "Passages (4 sentences)" grouping. This is the
        natural leave-one-passage-out CV group key (sentences in the same
        passage share topical/lexical content and must stay on one side of
        a train/test split), the direct Pereira analogue of LPP's
        leave-one-section-out grouping.
        """
        a = self._assembly
        exps = a["experiment"].values
        pidx = a["passage_index"].values
        stim = a["stimulus_id"].values
        return {
            str(s): f"{e}_p{int(p)}"
            for s, e, p in zip(stim.tolist(), exps.tolist(), pidx.tolist())
        }

    def get_passage_groups(self, item_ids: np.ndarray) -> np.ndarray:
        """Passage-group key for each of ``item_ids`` (see
        ``_passage_group_by_stimulus_id``) -- for ``GroupKFold`` splitting."""
        mapping = self._passage_group_by_stimulus_id
        return np.array([mapping[str(i)] for i in item_ids.tolist()])

    def get_sentences(self) -> tuple[list[str], np.ndarray]:
        """All 627 unique sentences, in a stable ``stimulus_id``-sorted order.

        Returns
        -------
        sentences : list[str]  length n_sentences
        item_ids  : np.ndarray (n_sentences,) str -- ``stimulus_id`` (e.g.
            ``"384sentences.0"``), the stable join key back to per-subject BOLD.
        """
        item_ids = np.array(sorted(self._sentence_by_stimulus_id))
        sentences = [self._sentence_by_stimulus_id[i] for i in item_ids.tolist()]
        return sentences, item_ids

    def rois(self, atlas: str = DEFAULT_ATLAS) -> list[str]:
        """Canonical, sorted ROI labels for one atlas (12 for "language")."""
        a = self._assembly
        mask = a["atlas"].values == atlas
        if not mask.any():
            raise ValueError(
                f"No neuroids with atlas={atlas!r}; available atlases: "
                f"{sorted(set(a['atlas'].values.tolist()))}"
            )
        return sorted(set(a["roi"].values[mask].tolist()))

    def get_condition_per_subject(
        self, atlas: str = DEFAULT_ATLAS,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-subject, per-ROI-averaged BOLD for every sentence that subject saw.

        Averages across voxels within each (subject, roi) group (hundreds of
        voxels per group -- see module docstring) to collapse per-voxel BOLD
        into one scalar per (subject, roi, sentence), then drops presentation
        rows that subject never saw (NaN across every ROI -- e.g. a subject
        who only ran one of the two sub-experiments).

        Returns
        -------
        dict mapping subject_id (str) -> (bold, item_ids):
            bold     : np.ndarray (n_seen, n_rois) float64, ROI columns in
                       ``self.rois(atlas)`` order.
            item_ids : np.ndarray (n_seen,) str -- ``stimulus_id``, aligned
                       with ``bold``'s rows.
        """
        a = self._assembly
        atlas_mask = a["atlas"].values == atlas
        sub = a.isel(neuroid=atlas_mask)
        values = np.asarray(sub.values, dtype=np.float64)  # (presentation, neuroid_subset)
        subjects = sub["subject"].values
        rois = sub["roi"].values
        stim_ids = sub["stimulus_id"].values

        roi_names = self.rois(atlas)
        roi_idx = {r: i for i, r in enumerate(roi_names)}

        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for subj in sorted(set(subjects.tolist())):
            col_mask = subjects == subj
            sub_vals = values[:, col_mask]  # (presentation, n_subj_voxels)
            sub_rois = rois[col_mask]

            roi_avg = np.full((sub_vals.shape[0], len(roi_names)), np.nan, dtype=np.float64)
            for roi, ri in roi_idx.items():
                rmask = sub_rois == roi
                if rmask.any():
                    # Presentation rows this subject never ran (the OTHER
                    # sub-experiment) are all-NaN across every voxel of
                    # theirs — nanmean legitimately warns "Mean of empty
                    # slice" per such row before correctly returning NaN;
                    # harmless (those rows get dropped by `valid` below),
                    # but silence the expected warning rather than let it
                    # fire once per subject per ROI.
                    with np.errstate(invalid="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        roi_avg[:, ri] = np.nanmean(sub_vals[:, rmask], axis=1)

            valid = ~np.isnan(roi_avg).all(axis=1)
            if not valid.any():
                continue
            result[str(subj)] = (roi_avg[valid], stim_ids[valid].astype(str))
        return result

    def get_voxelwise_per_subject(
        self, atlas: str = DEFAULT_ATLAS,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per-subject, per-VOXEL (not ROI-averaged) BOLD for every sentence
        that subject saw — for voxelwise ridge encoding (unlike
        ``get_condition_per_subject``, used by the pooled LME, which averages
        within each ROI to get one scalar per ROI/sentence).

        Returns
        -------
        dict mapping subject_id (str) -> (bold, item_ids, roi_per_voxel):
            bold          : np.ndarray (n_seen, n_subj_voxels) float64.
            item_ids      : np.ndarray (n_seen,) str -- ``stimulus_id``,
                             aligned with ``bold``'s rows.
            roi_per_voxel : np.ndarray (n_subj_voxels,) str -- this subject's
                             own voxel-to-ROI-label assignment, aligned with
                             ``bold``'s columns (for per-ROI aggregation of
                             per-voxel encoding scores downstream).
        """
        a = self._assembly
        atlas_mask = a["atlas"].values == atlas
        sub = a.isel(neuroid=atlas_mask)
        values = np.asarray(sub.values, dtype=np.float64)  # (presentation, neuroid_subset)
        subjects = sub["subject"].values
        rois = sub["roi"].values
        stim_ids = sub["stimulus_id"].values

        result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for subj in sorted(set(subjects.tolist())):
            col_mask = subjects == subj
            sub_vals = values[:, col_mask]  # (presentation, n_subj_voxels)
            sub_rois = rois[col_mask].astype(str)

            valid_rows = ~np.isnan(sub_vals).all(axis=1)
            if not valid_rows.any():
                continue
            sub_vals = sub_vals[valid_rows]
            sub_stim_ids = stim_ids[valid_rows].astype(str)

            # Drop voxels with ANY remaining NaN (scattered per-trial dropout
            # -- a handful of voxels per subject, e.g. QC-excluded trials --
            # distinct from the whole-row NaNs already removed above, which
            # were just "this subject didn't run that sub-experiment"). A
            # single NaN voxel would otherwise poison
            # cv_ridge_score_multioutput_gpu's shared per-fold GCV scalar
            # (one mean over the WHOLE samples x voxels residual tensor) for
            # EVERY voxel in this subject, not just the affected one.
            voxel_valid = ~np.isnan(sub_vals).any(axis=0)
            n_dropped = int((~voxel_valid).sum())
            if n_dropped:
                logger.info(
                    "Pereira subject %s: dropping %d/%d language-atlas voxel(s) "
                    "with scattered NaN (per-trial dropout)",
                    subj, n_dropped, sub_vals.shape[1],
                )
                sub_vals = sub_vals[:, voxel_valid]
                sub_rois = sub_rois[voxel_valid]

            result[str(subj)] = (sub_vals, sub_stim_ids, sub_rois)
        return result
