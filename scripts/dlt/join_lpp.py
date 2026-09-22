#!/usr/bin/env python3
"""Reconstruct a per-TR DLT-cost covariate for LPP fMRI and join it onto an
already-computed ``lpp_lme_pooled_table.parquet``.

Same rationale as ``dlt/join_ns_fmri.py``: the pooled table's ``wlen``/
``freq``/``sentpos`` columns are HRF-CONVOLVED per-TR values (``B @
raw_feats`` in ``neuroai/benchmarks/lpp_lme.py``'s ``LppLmeBenchmark.run``),
not discrete word ids, so a DLT-cost column can only be attached by
recomputing the identical per-section design (region loop inside a section
loop, same chunking) with DLT total cost substituted for surprisal, using
the SAME config as the production sweep (``configs/surprisal/neural/lpp.yaml`` —
deterministic, data-only, no per-model dependence) and reusing
``LppLmeBenchmark``'s own helper functions rather than reimplementing them.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DLT_DIR = REPO_ROOT / "outputs" / "dlt"
JOINED_DIR = DLT_DIR / "joined"


def main() -> None:
    from neuroai.benchmarks.lpp_lme import (
        LppLmeBenchmark, LppLmeBenchmarkConfig, LppLmeConfig, LppLmeDataConfig,
        words_for_section,
    )
    from neuroai.benchmarks.base import ModelConfig
    from neuroai.datasets.le_petit_prince import LPP_N_ANNOTATION_SECTIONS
    from neuroai.lpp_encoding import _select_voxel_indices, stimulus_aligned_frame_times, word_hrf_basis_matrix
    from neuroai.roi.lpp import load_fedorenko_parcelization

    os.environ.setdefault("BENCHMARK_ROOT", str(REPO_ROOT))

    # Exact same values as configs/surprisal/neural/lpp.yaml (dataset section) — data
    # config only, no per-model dependence, so any lpp_lme_pooled_table.parquet
    # from the checkpoint sweep was built with these same settings.
    dcfg = LppLmeDataConfig(
        bids_root=str(REPO_ROOT / "data" / "Lepetitprince_fMRI"),
        lang="en",
        sections=None,
        t_r=2.0,
        min_word_duration_sec=0.05,
        hrf_model="glover",
        trim_trs_start=10,
        trim_trs_end=10,
        average_subject_precomputed_dir=str(REPO_ROOT / "data" / "lpp_en_average_subject"),
        reliable_voxel_fraction=0.25,
        reliable_voxel_scores_path=str(REPO_ROOT / "data" / "isc" / "isc_10trials_en.gz"),
        brain_mask_path=str(REPO_ROOT / "data" / "masks" / "mask_lpp_en.nii.gz"),
        fedorenko_atlas_nii_path=str(REPO_ROOT / "data" / "masks" / "allParcels-language-SN220-hgwm.nii"),
        fedorenko_labels_txt_path=str(REPO_ROOT / "data" / "masks" / "allParcels-language-SN220.txt"),
    )
    lcfg = LppLmeConfig(chunk_seconds=30.0, split_random_state=0)
    cfg = LppLmeBenchmarkConfig(
        data=dcfg, model=ModelConfig(model_name="unused"), lme=lcfg,
    )
    bench = LppLmeBenchmark(cfg)

    dlt = pd.read_csv(DLT_DIR / "lpp_dlt_cost.csv")

    sections = list(range(1, LPP_N_ANNOTATION_SECTIONS + 1))
    encoder = bench._encoder()
    n_vox_full = encoder._precomputed_n_voxels()
    rng = np.random.default_rng(lcfg.split_random_state)
    voxel_ix = _select_voxel_indices(encoder.cfg, n_vox_full, rng)
    logger.info("%d mask voxels -> %d selected", n_vox_full, voxel_ix.size)

    parcelization = load_fedorenko_parcelization(
        dcfg.brain_mask_path, dcfg.fedorenko_atlas_nii_path, dcfg.fedorenko_labels_txt_path,
    )
    region_names: list[str] = []
    region_masks_selected: list[np.ndarray] = []
    for ri, name in enumerate(parcelization.region_names):
        for hemi, full_mask in (("L", parcelization.left_masks[ri]), ("R", parcelization.right_masks[ri])):
            mask_sel = full_mask[voxel_ix]
            if not mask_sel.any():
                continue
            region_names.append(f"{hemi}_{name}")
            region_masks_selected.append(mask_sel)
    logger.info("%d regions", len(region_names))

    words_full, _ = bench._load_words_with_sentpos()

    region_parts, section_parts, chunk_parts, dlt_parts = [], [], [], []
    for section in sections:
        sec_df = words_for_section(words_full, section)
        dlt_sec = dlt[dlt["section"] == section].sort_values("position").reset_index(drop=True)
        if len(sec_df) != len(dlt_sec):
            raise RuntimeError(
                f"section {section}: word count mismatch ({len(sec_df)} word-table rows "
                f"vs {len(dlt_sec)} DLT rows) — lpp_dlt_cost.csv's own exporter "
                "(_export_lpp) filters blank/NaN words the same way "
                "_load_words_with_sentpos does, so these should match exactly."
            )
        dlt_total = dlt_sec["total"].to_numpy(dtype=np.float64)

        onsets = sec_df["onset"].to_numpy(dtype=np.float64)
        offsets = sec_df["offset"].to_numpy(dtype=np.float64)
        min_dur = max(0.0, float(dcfg.min_word_duration_sec))
        durations = np.maximum(min_dur, offsets - onsets)

        bold_full = encoder._load_averaged_masked_bold(section, None, voxel_ix)
        n_scans = bold_full.shape[0]
        frame_times = stimulus_aligned_frame_times(n_scans, tr=dcfg.t_r)
        B = word_hrf_basis_matrix(onsets, durations, frame_times, dcfg.hrf_model)
        design_dlt = B @ dlt_total[:, None]  # (n_scans, 1)

        ts, te = dcfg.trim_trs_start, dcfg.trim_trs_end
        sl = slice(ts, None if te <= 0 else -te)
        design_trim = design_dlt[sl]
        n_trs = design_trim.shape[0]
        if n_trs < 5:
            continue

        chunk_len_trs = max(1, int(round(lcfg.chunk_seconds / dcfg.t_r)))
        chunk_ids = np.arange(n_trs) // chunk_len_trs
        chunk_labels = np.array([f"{section}_{c}" for c in chunk_ids], dtype=object)

        for region_name in region_names:
            region_parts.append(np.full(n_trs, region_name, dtype=object))
            section_parts.append(np.full(n_trs, str(section), dtype=object))
            chunk_parts.append(chunk_labels)
            dlt_parts.append(design_trim[:, 0])
        logger.info("[section %d] %d TRs x %d regions", section, n_trs, len(region_names))

    recon = pd.DataFrame({
        "region": np.concatenate(region_parts),
        "section": np.concatenate(section_parts),
        "chunk": np.concatenate(chunk_parts),
        "dlt_total": np.concatenate(dlt_parts),
    })
    logger.info("Reconstructed %d rows", len(recon))

    JOINED_DIR.mkdir(parents=True, exist_ok=True)
    recon_path = JOINED_DIR / "lpp_tr_dlt.parquet"
    recon.to_parquet(recon_path, index=False)
    logger.info("Wrote %s", recon_path)

    ref_path = (REPO_ROOT / "outputs_jeanzay" / "lpp/surprisal_sweep" / "babylm"
                / "colinglab_CLASS_IT-140M" / "lpp_lme_pooled_table.parquet")
    if not ref_path.is_file():
        logger.error("Reference table %s not found — writing standalone table only.", ref_path)
        return
    ref = pd.read_parquet(ref_path)
    if len(ref) != len(recon):
        logger.error(
            "Row count mismatch: reference=%d, reconstruction=%d — not attaching positionally.",
            len(ref), len(recon),
        )
        return
    same_region = (ref["region"].to_numpy() == recon["region"].to_numpy()).all()
    same_section = (ref["section"].to_numpy() == recon["section"].to_numpy()).all()
    same_chunk = (ref["chunk"].to_numpy() == recon["chunk"].to_numpy()).all()
    logger.info("Row-order check: region=%s, section=%s, chunk=%s", same_region, same_section, same_chunk)
    if not (same_region and same_section and same_chunk):
        logger.error("Row order mismatch — not attaching positionally.")
        return

    out = ref.copy()
    out["dlt_total"] = recon["dlt_total"].to_numpy()
    out_path = JOINED_DIR / "lpp_with_dlt.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %s — row-aligned with %s", out_path, ref_path)


if __name__ == "__main__":
    main()
