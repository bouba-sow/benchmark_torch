"""Podcast iEEG electrode -> Fedorenko language-network ROI assignment.

Unlike LPP (voxel-level BOLD, masked directly against the EvLab SN220 atlas
volume via ``neuroai.roi.lpp.load_fedorenko_parcelization``), Podcast gives
individual electrode locations (MNI152NLin2009aSym x/y/z, mm) rather than a
full volume -- there's no voxel grid to resample the atlas onto. Instead,
each electrode is assigned to the nearest labelled atlas voxel by Euclidean
distance in MNI space, subject to a maximum-distance cutoff (electrodes
recorded from the cortical surface routinely sit a few mm outside the
atlas's gray-matter mask, so exact voxel containment is too strict).

The two spaces are both "MNI" but not identical (the SN220 atlas ships on
the classic FSL MNI152 2mm linear-template grid; Podcast's electrodes.tsv
is in ICBM 2009c Nonlinear Asymmetric). Treating them as interchangeable
for point-in-atlas assignment is standard practice in the iEEG-atlas
literature -- the two templates differ by only a few mm on average, well
within typical electrode-localization error, and is the same order of
tolerance already built into the nearest-voxel cutoff below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# Same label -> (region, hemisphere) convention as
# neuroai.roi.lpp.load_fedorenko_parcelization: LH labels 1-5, RH labels 7-11,
# in the allParcels-language-SN220.txt order (IFGorb, IFG, MFG, AntTemp, PostTemp).
_REGION_ORDER = ("IFGorb", "IFG", "MFG", "AntTemp", "PostTemp")
LABEL_TO_REGION: dict[int, str] = {i + 1: name for i, name in enumerate(_REGION_ORDER)}
LABEL_TO_REGION.update({i + 7: name for i, name in enumerate(_REGION_ORDER)})
LABEL_TO_HEMI: dict[int, str] = {i + 1: "L" for i in range(5)}
LABEL_TO_HEMI.update({i + 7: "R" for i in range(5)})


@dataclass(frozen=True)
class FedorenkoAtlasPoints:
    """Flattened (mm, label) representation of every non-background atlas voxel."""

    coords_mm: np.ndarray  # (n_voxels, 3)
    labels: np.ndarray     # (n_voxels,) int


def load_fedorenko_atlas_points(atlas_nii_path: str | Path) -> FedorenkoAtlasPoints:
    """Load the SN220 atlas and flatten every labelled voxel to (x, y, z) mm."""
    img = nib.load(str(atlas_nii_path))
    data = np.asarray(img.get_fdata())
    ijk = np.argwhere(data > 0)
    labels = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.int32)
    ijk_h = np.concatenate([ijk, np.ones((len(ijk), 1))], axis=1)
    coords_mm = (img.affine @ ijk_h.T).T[:, :3]
    return FedorenkoAtlasPoints(coords_mm=coords_mm, labels=labels)


def assign_electrodes_to_fedorenko_parcels(
    electrode_meta: dict[str, dict],
    atlas_nii_path: str | Path,
    *,
    max_dist_mm: float = 10.0,
    atlas_points: FedorenkoAtlasPoints | None = None,
) -> pd.DataFrame:
    """Assign each electrode to its nearest Fedorenko language parcel.

    ``electrode_meta`` is the ``{channel_name: {"x", "y", "z", ...}}`` dict
    from ``PodcastDataset.electrode_meta`` (MNI152NLin2009aSym mm). Returns
    one row per electrode with the nearest parcel's region/hemisphere, the
    distance to it, and ``assigned=False`` (region/hemisphere left as
    ``None``) for electrodes farther than ``max_dist_mm`` from every parcel.
    """
    from scipy.spatial import cKDTree

    if atlas_points is None:
        atlas_points = load_fedorenko_atlas_points(atlas_nii_path)
    tree = cKDTree(atlas_points.coords_mm)

    rows = []
    for name, meta in electrode_meta.items():
        x, y, z = meta.get("x"), meta.get("y"), meta.get("z")
        row = {
            "channel": name, "x": x, "y": y, "z": z,
            "region": None, "hemisphere": None, "label": None,
            "dist_mm": np.nan, "assigned": False,
        }
        if x is not None and y is not None and z is not None:
            dist, idx = tree.query([x, y, z])
            row["dist_mm"] = float(dist)
            if dist <= max_dist_mm:
                lab = int(atlas_points.labels[idx])
                row["label"] = lab
                row["region"] = LABEL_TO_REGION[lab]
                row["hemisphere"] = LABEL_TO_HEMI[lab]
                row["assigned"] = True
        rows.append(row)
    return pd.DataFrame(rows)
