"""ROI and atlas parcel labels for l-bg LPP brain-mask voxel order (``mask_lpp_en.nii.gz``)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker
from scipy import stats


@dataclass(frozen=True)
class HarvardOxfordParcelization:
    """48 homologous cortical regions (Harvard–Oxford, symmetric L/R pairs)."""

    region_names: tuple[str, ...]
    left_masks: np.ndarray  # (n_regions, n_voxels) bool, NiftiMasker order
    right_masks: np.ndarray  # (n_regions, n_voxels) bool

    @property
    def n_regions(self) -> int:
        return len(self.region_names)


def _mask_cache_dir() -> Path:
    import os

    cache_root = os.environ.get("BENCHMARK_CACHE_ROOT", "").strip()
    if cache_root:
        cache = Path(cache_root) / "harvard_oxford_parcels"
    else:
        repo = Path(__file__).resolve().parents[2]  # neuroai/roi/lpp.py -> repo root
        cache = repo / ".cache" / "harvard_oxford_parcels"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _normalize_ho_region_name(label: str) -> str:
    return (
        label.replace("Left ", "")
        .replace("Right ", "")
        .replace("(formerly Supplementary Motor Cortex", "")
        .strip()
    )


def load_harvard_oxford_parcelization(
    brain_mask_path: str | Path,
    *,
    atlas_name: str = "cort-maxprob-thr0-2mm",
    cache: bool = True,
) -> HarvardOxfordParcelization:
    """Harvard–Oxford cortical parcels aligned to ``brain_mask_path`` (l-bg Fig. B.1).

    Uses Nilearn ``fetch_atlas_harvard_oxford(..., symmetric_split=True)`` and the
    same resampling / ``NiftiMasker`` pipeline as the LPP encoding scripts.
    """
    brain_mask_path = Path(brain_mask_path).resolve()
    if not brain_mask_path.is_file():
        raise FileNotFoundError(f"Brain mask not found: {brain_mask_path}")

    cache_key = hashlib.sha256(
        f"{brain_mask_path}|{atlas_name}|{brain_mask_path.stat().st_mtime_ns}".encode()
    ).hexdigest()[:16]
    cache_npz = _mask_cache_dir() / f"ho_{cache_key}.npz"
    cache_meta = _mask_cache_dir() / f"ho_{cache_key}.json"

    if cache and cache_npz.is_file() and cache_meta.is_file():
        data = np.load(cache_npz, allow_pickle=False)
        names = tuple(json.loads(cache_meta.read_text(encoding="utf-8"))["region_names"])
        return HarvardOxfordParcelization(
            region_names=names,
            left_masks=data["left_masks"].astype(bool),
            right_masks=data["right_masks"].astype(bool),
        )

    from nilearn.datasets import fetch_atlas_harvard_oxford
    from nilearn.image import resample_img

    atlas = fetch_atlas_harvard_oxford(atlas_name, symmetric_split=True)
    mask_img = nib.load(str(brain_mask_path))
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    n_vox = int(masker.n_elements_)

    parcels = resample_img(
        atlas["maps"],
        interpolation="nearest",
        target_affine=mask_img.affine,
        target_shape=mask_img.shape,
    )
    parcels_t = np.asarray(masker.transform(parcels)).ravel().astype(np.int32)
    n_parcels = int(np.unique(parcels_t).size) - 1  # drop background 0

    idx_parcels = [
        np.flatnonzero(parcels_t == i).astype(np.int64) for i in range(1, n_parcels + 1)
    ]
    region_names = tuple(
        _normalize_ho_region_name(label)
        for label in atlas["labels"][1:][::2]
    )
    n_regions = len(region_names)
    if n_regions * 2 != len(idx_parcels):
        raise RuntimeError(
            f"Expected {n_regions * 2} parcel indices, got {len(idx_parcels)}"
        )

    left_masks = np.zeros((n_regions, n_vox), dtype=bool)
    right_masks = np.zeros((n_regions, n_vox), dtype=bool)
    for ri in range(n_regions):
        left_masks[ri, idx_parcels[2 * ri]] = True
        right_masks[ri, idx_parcels[2 * ri + 1]] = True

    if cache:
        np.savez_compressed(
            cache_npz,
            left_masks=left_masks,
            right_masks=right_masks,
        )
        cache_meta.write_text(
            json.dumps({"region_names": list(region_names)}, indent=2),
            encoding="utf-8",
        )

    return HarvardOxfordParcelization(
        region_names=region_names,
        left_masks=left_masks,
        right_masks=right_masks,
    )


def load_fedorenko_parcelization(
    brain_mask_path: str | Path,
    atlas_nii_path: str | Path,
    labels_txt_path: str | Path | None = None,
    *,
    cache: bool = True,
) -> HarvardOxfordParcelization:
    """Fedorenko language parcels aligned to ``brain_mask_path``.

    Loads the EvLab SN220 language atlas (``allParcels-language-SN220-hgwm.nii``)
    and resamples it to the LPP brain mask, producing L/R paired masks that are
    compatible with ``parcel_summary_from_correlations``.

    The atlas has 5 bilateral regions (IFGorb, IFG, MFG, AntTemp, PostTemp);
    LH voxels carry labels 1–5 and RH voxels carry labels 7–11 in the NIfTI.

    ``labels_txt_path`` is the companion ``allParcels-language-SN220.txt``;
    if omitted, region names are inferred from the NIfTI label order.
    """
    brain_mask_path = Path(brain_mask_path).resolve()
    atlas_nii_path = Path(atlas_nii_path).resolve()
    if not brain_mask_path.is_file():
        raise FileNotFoundError(f"Brain mask not found: {brain_mask_path}")
    if not atlas_nii_path.is_file():
        raise FileNotFoundError(f"Fedorenko atlas not found: {atlas_nii_path}")

    cache_key = hashlib.sha256(
        f"{brain_mask_path}|{atlas_nii_path}|{brain_mask_path.stat().st_mtime_ns}|{atlas_nii_path.stat().st_mtime_ns}".encode()
    ).hexdigest()[:16]
    cache_npz = _mask_cache_dir() / f"fedorenko_{cache_key}.npz"
    cache_meta = _mask_cache_dir() / f"fedorenko_{cache_key}.json"

    if cache and cache_npz.is_file() and cache_meta.is_file():
        data = np.load(cache_npz, allow_pickle=False)
        names = tuple(json.loads(cache_meta.read_text(encoding="utf-8"))["region_names"])
        return HarvardOxfordParcelization(
            region_names=names,
            left_masks=data["left_masks"].astype(bool),
            right_masks=data["right_masks"].astype(bool),
        )

    from nilearn.image import resample_img

    # Parse region names from companion txt if provided
    if labels_txt_path is not None and Path(labels_txt_path).is_file():
        raw_names = Path(labels_txt_path).read_text(encoding="utf-8").splitlines()
        raw_names = [n.strip() for n in raw_names if n.strip()]
    else:
        raw_names = []

    atlas_img = nib.load(str(atlas_nii_path))
    mask_img = nib.load(str(brain_mask_path))
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    n_vox = int(masker.n_elements_)

    resampled = resample_img(
        atlas_img,
        interpolation="nearest",
        target_affine=mask_img.affine,
        target_shape=mask_img.shape,
    )
    labels_vec = np.asarray(masker.transform(resampled)).ravel().astype(np.int32)

    # Identify all non-background label values, split into LH (smaller half) and RH
    label_vals = sorted(int(v) for v in np.unique(labels_vec) if v > 0)
    n_total = len(label_vals)
    if n_total == 0:
        raise RuntimeError("No non-background voxels found in Fedorenko atlas after resampling")
    if n_total % 2 != 0:
        raise RuntimeError(
            f"Expected an even number of parcels (L/R pairs), got {n_total} labels: {label_vals}"
        )
    n_regions = n_total // 2
    lh_labels = label_vals[:n_regions]
    rh_labels = label_vals[n_regions:]

    # Derive region names: strip LH_/RH_ prefix from txt names, or use index
    lh_raw = [n for n in raw_names if n.startswith("LH_")] if raw_names else []
    region_names: tuple[str, ...]
    if len(lh_raw) == n_regions:
        region_names = tuple(n[3:] for n in lh_raw)  # strip "LH_"
    else:
        region_names = tuple(f"region_{i + 1}" for i in range(n_regions))

    left_masks = np.zeros((n_regions, n_vox), dtype=bool)
    right_masks = np.zeros((n_regions, n_vox), dtype=bool)
    for ri, (lv, rv) in enumerate(zip(lh_labels, rh_labels)):
        left_masks[ri] = labels_vec == lv
        right_masks[ri] = labels_vec == rv

    if cache:
        np.savez_compressed(cache_npz, left_masks=left_masks, right_masks=right_masks)
        cache_meta.write_text(
            json.dumps({"region_names": list(region_names)}, indent=2),
            encoding="utf-8",
        )

    return HarvardOxfordParcelization(
        region_names=region_names,
        left_masks=left_masks,
        right_masks=right_masks,
    )


def load_lbg_voxel_roi_labels(
    brain_mask_path: str | Path,
    roi_masks_dir: str | Path,
) -> np.ndarray:
    """Assign each in-mask voxel (NiftiMasker order) to at most one ROI name.

    Sphere ROI NIfTIs from ``data/lpp_fmri_extra/average_subject/roi_masks/`` are intersected
    with the English LPP mask. When a voxel falls in multiple ROIs, the first ROI
    in sorted filename order wins.
    """
    brain_mask_path = Path(brain_mask_path)
    roi_masks_dir = Path(roi_masks_dir)
    if not brain_mask_path.is_file():
        raise FileNotFoundError(f"Brain mask not found: {brain_mask_path}")
    if not roi_masks_dir.is_dir():
        raise FileNotFoundError(f"ROI masks directory not found: {roi_masks_dir}")

    mask_img = nib.load(str(brain_mask_path))
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    ones_vol = nib.Nifti1Image(np.ones(mask_img.shape[:3]), mask_img.affine)
    n_vox = int(masker.transform(ones_vol).shape[0])
    labels = np.full(n_vox, "", dtype=object)

    roi_paths = sorted(roi_masks_dir.glob("*.nii.gz"))
    if not roi_paths:
        raise FileNotFoundError(f"No ROI masks (*.nii.gz) in {roi_masks_dir}")

    for roi_path in roi_paths:
        roi_name = roi_path.name.replace(".nii.gz", "")
        roi_img = nib.load(str(roi_path))
        vec = np.asarray(masker.transform(roi_img)).ravel()
        in_roi = vec > 0.5
        unassigned = (labels == "") & in_roi
        labels[unassigned] = roi_name

    return labels


def parcel_summary_from_correlations(
    corrs: np.ndarray,
    voxel_global_ix: np.ndarray,
    parcelization: HarvardOxfordParcelization,
) -> list[dict[str, float | str | int]]:
    """Per homologous region: mean r in L/R parcels and L−R (Fig. B.1c-style stats).

    ``voxel_global_ix`` maps each entry in ``corrs`` to an index on the full
    brain-mask axis (same order as ``NiftiMasker`` / ``HarvardOxfordParcelization``).
    """
    if corrs.shape[0] != voxel_global_ix.shape[0]:
        raise ValueError(
            f"corrs length {corrs.shape[0]} != voxel_global_ix length {voxel_global_ix.shape[0]}"
        )

    rows: list[dict[str, float | str | int]] = []
    gix = voxel_global_ix.astype(np.int64, copy=False)

    for ri, name in enumerate(parcelization.region_names):
        lm = parcelization.left_masks[ri]
        rm = parcelization.right_masks[ri]
        left_vals = corrs[lm[gix]]
        right_vals = corrs[rm[gix]]

        row: dict[str, float | str | int] = {
            "parcel": name,
            "n_voxels_left": int(left_vals.size),
            "n_voxels_right": int(right_vals.size),
            "mean_pearson_left": float("nan"),
            "mean_pearson_right": float("nan"),
            "lateralization_index": float("nan"),
            "lateralization_pvalue": float("nan"),
            "mean_pearson": float("nan"),
        }
        if left_vals.size:
            row["mean_pearson_left"] = float(np.nanmean(left_vals))
        if right_vals.size:
            row["mean_pearson_right"] = float(np.nanmean(right_vals))
        if left_vals.size and right_vals.size:
            row["lateralization_index"] = (
                float(row["mean_pearson_left"]) - float(row["mean_pearson_right"])
            )
            if left_vals.size >= 2 and right_vals.size >= 2:
                _, p = stats.ttest_ind(left_vals, right_vals, equal_var=False)
                row["lateralization_pvalue"] = float(p)
            row["mean_pearson"] = float(np.nanmean(np.concatenate([left_vals, right_vals])))
        rows.append(row)

    return rows


def roi_summary_from_correlations(
    corrs: np.ndarray,
    roi_labels: np.ndarray,
    *,
    left_mask: np.ndarray | None = None,
    right_mask: np.ndarray | None = None,
) -> list[dict[str, float | str | int]]:
    """Aggregate Pearson r per ROI (``roi_labels`` aligned with ``corrs``).

    For lateralization, pass boolean ``left_mask`` / ``right_mask`` with the same
    length as ``corrs``. For sphere ROIs (left hemisphere only), omit these masks.
    Use ``parcel_summary_from_correlations`` for Harvard–Oxford L/R homologues.
    """
    if corrs.shape[0] != roi_labels.shape[0]:
        raise ValueError(
            f"corrs length {corrs.shape[0]} != roi_labels length {roi_labels.shape[0]}"
        )
    rows: list[dict[str, float | str | int]] = []
    for roi in sorted({str(x) for x in roi_labels if str(x)}):
        m = roi_labels == roi
        if not m.any():
            continue
        vals = corrs[m]
        row: dict[str, float | str | int] = {
            "roi": roi,
            "mean_pearson": float(np.nanmean(vals)),
            "std_pearson": float(np.nanstd(vals)),
            "n_voxels": int(m.sum()),
        }
        if left_mask is not None and right_mask is not None:
            if left_mask.shape[0] != corrs.shape[0] or right_mask.shape[0] != corrs.shape[0]:
                raise ValueError("hemisphere masks must match corrs length")
            lm = m & left_mask
            rm = m & right_mask
            row["mean_pearson_left"] = (
                float(np.nanmean(corrs[lm])) if lm.any() else float("nan")
            )
            row["mean_pearson_right"] = (
                float(np.nanmean(corrs[rm])) if rm.any() else float("nan")
            )
            if np.isfinite(row["mean_pearson_left"]) and np.isfinite(row["mean_pearson_right"]):
                row["lateralization_index"] = (
                    float(row["mean_pearson_left"]) - float(row["mean_pearson_right"])
                )
            else:
                row["lateralization_index"] = float("nan")
        rows.append(row)
    return rows
