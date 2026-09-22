#!/usr/bin/env python3
"""Reconstruct a per-TR DLT-cost covariate for Natural Stories fMRI and join
it onto an already-computed ``ns_fmri_lme_nuisance_table.parquet``.

Why this can't be a plain key-merge
------------------------------------
Unlike Podcast (one row per word, exact ``word_idx`` join), the NS-fMRI
pooled table is TR-level: ``wlen``/``freq``/``sentpos`` are HRF-CONVOLVED
values (``B @ raw_feats`` in ``neuroai/benchmarks/natural_stories_fmri_lme.
py``'s Step 1), so there is no discrete per-row word/TR id to merge on. The
only way to attach a DLT-cost column that lines up with the existing table's
rows is to recompute the SAME per-doc HRF design (same onsets/durations/
frame_times/chunking, same subject x doc iteration order as Step 2) with DLT
total cost substituted for surprisal as the "raw feature" column, then
concatenate in the same order.

Story-name mapping (fMRI uses 8 of the original 10 Natural Stories docs,
named e.g. "Aqua" instead of a numeric story_id) was established by matching
per-doc word COUNTS and the first 8 words against ``naturalstories_dlt_cost.
csv`` (see this script's ``_STORY_ID_BY_DOC``) — Roswell (story_id 8) and
Tourette (story_id 10) are not among the 8 fMRI docs.

The full ``LANG_MD_ts_X.csv`` (~900MB, 5.5M rows = word x subject x fROI) is
too large to load with pandas under this environment's memory limits, so the
canonical (docid, time, wlen, sentpos, sentid) word list is instead extracted
with ``awk`` for ONE representative (subject, fROI) pair first (words are
identical across subjects/fROIs for a given doc — same convention as
``load_canonical_words``).
"""

from __future__ import annotations

import logging
import subprocess
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

NS_FMRI_DATA_ROOT = REPO_ROOT / "data" / "naturalstories_fmri" / "osfstorage"
NS_FMRI_TR = 2.0
NS_FMRI_WORD_DUR = 0.1
CHUNK_SECONDS = 30.0
NS_FMRI_DOCS = ["Aqua", "Boar", "Elvis", "HighSchool", "KingOfBirds",
                "MatchstickSeller", "MrSticky", "Tulips"]

# Verified by matching per-doc word count + first-8-words against
# naturalstories_dlt_cost.csv (see module docstring).
_STORY_ID_BY_DOC = {
    "Boar": 1, "Aqua": 2, "MatchstickSeller": 3, "KingOfBirds": 4,
    "Elvis": 5, "MrSticky": 6, "HighSchool": 7, "Tulips": 9,
}


def _extract_canonical_words_awk(subject: str = "s061", froi: str = "LangLAngG") -> pd.DataFrame:
    """One representative (subject, fROI)'s word rows, via awk (avoids loading
    the full 900MB/5.5M-row X table into pandas)."""
    # Tab-separated: word tokens can contain a literal comma (e.g. the token
    # "thirty-seven," — punctuation is attached to the word, not stripped in
    # this table), which collides with a comma-delimited output.
    tmp_path = Path("/tmp/ns_fmri_extract_canonical_words.tsv")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    awk_prog = (
        r'NR==1{print "word\ttime\twlen\tsentpos\tsentid\tdocid"; next} '
        f'$44=="{subject}" && $45=="{froi}"'
        r'{print $1"\t"$42"\t"$41"\t"$25"\t"$26"\t"$43}'
    )
    x_path = NS_FMRI_DATA_ROOT / "LANG_MD_ts_X.csv"
    with tmp_path.open("w") as f:
        subprocess.run(["awk", awk_prog, str(x_path)], stdout=f, check=True)
    df = pd.read_csv(tmp_path, sep="\t")
    return df


def _word_hrf_basis_matrix(onsets, durations, frame_times, hrf_model="glover"):
    from nilearn.glm.first_level import compute_regressor

    onsets = np.asarray(onsets, dtype=np.float64).ravel()
    durations = np.asarray(durations, dtype=np.float64).ravel()
    n_words = len(onsets)
    n_scans = len(frame_times)
    B = np.zeros((n_scans, n_words), dtype=np.float64)
    for w in range(n_words):
        exp_condition = np.asarray([[onsets[w]], [durations[w]], [1.0]])
        signal, _ = compute_regressor(exp_condition, hrf_model, frame_times)
        B[:, w] = signal[:, 0]
    return B


def main() -> None:
    from neuroai.datasets.natural_stories_fmri import available_subjects, load_bold_subject

    words_all = _extract_canonical_words_awk()
    dlt = pd.read_csv(DLT_DIR / "naturalstories_dlt_cost.csv")

    conv_by_doc: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for doc in NS_FMRI_DOCS:
        wdf = words_all[words_all["docid"] == doc].sort_values("time").reset_index(drop=True)
        story_id = _STORY_ID_BY_DOC[doc]
        dlt_doc = dlt[dlt["story_id"] == story_id].sort_values("position").reset_index(drop=True)
        if len(wdf) != len(dlt_doc):
            raise RuntimeError(
                f"{doc} (story_id={story_id}): word count mismatch "
                f"({len(wdf)} fMRI words vs {len(dlt_doc)} DLT words)"
            )
        n_mismatch = int((wdf["word"].astype(str).str.strip(".,;:!?") !=
                           dlt_doc["word"].astype(str).str.strip(".,;:!?")).sum())
        if n_mismatch:
            logger.warning("[%s] %d/%d word text mismatches after punctuation-strip (tokenization drift)",
                            doc, n_mismatch, len(wdf))

        onsets = wdf["time"].to_numpy(dtype=np.float64)
        durations = np.full(len(onsets), NS_FMRI_WORD_DUR)
        dlt_total = dlt_doc["total"].to_numpy(dtype=np.float64)

        last_onset = float(onsets[-1])
        n_trs = int(np.ceil((last_onset + 20.0) / NS_FMRI_TR)) + 1
        frame_times = np.arange(n_trs) * NS_FMRI_TR
        B = _word_hrf_basis_matrix(onsets, durations, frame_times)
        design_dlt = B @ dlt_total[:, None]  # (n_trs, 1)

        chunk_len_trs = max(1, int(round(CHUNK_SECONDS / NS_FMRI_TR)))
        chunk_ids = np.arange(n_trs) // chunk_len_trs
        conv_by_doc[doc] = (design_dlt[:, 0], chunk_ids)
        logger.info("[%s] story_id=%d, %d words, %d TRs", doc, story_id, len(onsets), n_trs)

    subjects = available_subjects(str(NS_FMRI_DATA_ROOT), "LANG_MD")
    logger.info("%d subjects", len(subjects))

    subj_parts, story_parts, chunk_parts, dlt_parts = [], [], [], []
    for subj in subjects:
        bold_data = load_bold_subject(str(NS_FMRI_DATA_ROOT), subj, "LANG_MD", docs=NS_FMRI_DOCS)
        for doc in NS_FMRI_DOCS:
            if doc not in bold_data or doc not in conv_by_doc:
                continue
            _, scan_times, _, _ = bold_data[doc]
            dlt_series, chunk_ids = conv_by_doc[doc]
            n_trs = min(len(scan_times), len(dlt_series))
            if n_trs < 20:
                continue
            subj_parts.append(np.full(n_trs, subj, dtype=object))
            story_parts.append(np.full(n_trs, doc, dtype=object))
            chunk_parts.append(np.array([f"{doc}_{c}" for c in chunk_ids[:n_trs]], dtype=object))
            dlt_parts.append(dlt_series[:n_trs])

    recon = pd.DataFrame({
        "subject": np.concatenate(subj_parts),
        "story": np.concatenate(story_parts),
        "chunk": np.concatenate(chunk_parts),
        "dlt_total": np.concatenate(dlt_parts),
    })
    logger.info("Reconstructed %d TR-rows", len(recon))

    JOINED_DIR.mkdir(parents=True, exist_ok=True)
    recon_path = JOINED_DIR / "ns_fmri_tr_dlt.parquet"
    recon.to_parquet(recon_path, index=False)
    logger.info("Wrote %s", recon_path)

    ref_path = (REPO_ROOT / "outputs_jeanzay" / "ns_fmri/surprisal_sweep" / "gpt2"
                / "openai-community_gpt2-xl" / "ns_fmri_lme_nuisance_table.parquet")
    ref = pd.read_parquet(ref_path)
    if len(ref) != len(recon):
        logger.error(
            "Row count mismatch: reference table has %d rows, reconstruction has %d — "
            "cannot safely attach positionally. Writing the standalone per-TR DLT table only.",
            len(ref), len(recon),
        )
        return
    same_subj = (ref["subject"].to_numpy() == recon["subject"].to_numpy()).all()
    same_story = (ref["story"].to_numpy() == recon["story"].to_numpy()).all()
    same_chunk = (ref["chunk"].to_numpy() == recon["chunk"].to_numpy()).all()
    logger.info("Row-order check vs reference: subject match=%s, story match=%s, chunk match=%s",
                same_subj, same_story, same_chunk)
    if not (same_subj and same_story and same_chunk):
        logger.error("Row order does not match the reference table — not attaching positionally.")
        return

    out = ref.copy()
    out["dlt_total"] = recon["dlt_total"].to_numpy()
    out_path = JOINED_DIR / "ns_fmri_with_dlt.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%.1f MB) — row-aligned with %s", out_path, out_path.stat().st_size / 1e6, ref_path)


if __name__ == "__main__":
    main()
