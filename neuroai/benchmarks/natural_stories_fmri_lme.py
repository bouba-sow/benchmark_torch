"""Natural Stories fMRI encoding benchmark — linear mixed-effects (LME) version.

Predicts per-fROI BOLD from PER-LAYER LLM surprisal (logit-lens, Kuribayashi
et al. 2025) using ``pymer4``/``lme4``, extending Lin & Schuler (2025,
arXiv:2506.11338) with ``freq_z`` — see ``neuroai/regressors/fmri_lme.py``
for the exact formula:

    BOLD ~ wlen_z + freq_z + sentpos_z + surprisal_z +
           (1 + wlen_z + freq_z + surprisal_z | subject) + (1 + surprisal_z | story)

fit ONCE PER (fROI, LAYER) PAIR — same per-layer sweep as the N400 datasets
(Nieuwland) and Fedorenko2016/Tuckute, generalized to a
second grouping dimension (26 fROIs) that those pooled-single-DV datasets
don't have. This replaces the previous single-top-layer-surprisal design
(``batch_surprisal_one_model``): logit-lens reads out every transformer
layer's own next-token distribution via the model's own final norm +
unembedding, in the SAME forward pass already needed for one layer's
surprisal, so the per-layer sweep costs about the same to EXTRACT — the
added cost is entirely in FITTING (fROI × layer instead of just fROI), which
is why this is fit via a process pool (pymer4/rpy2 embeds one R interpreter
per process, so parallelism must be process-based), same mitigation as every
other pooled LME here.

Scope: every registry family, including ``cbr_rnn`` (see
``_word_surprisal_all_layers``: it bypasses ``AutoModelForCausalLM`` and has
a single hidden-state vector per token, so it gets ``batch_surprisal_cbr_rnn``
instead of logit-lens — a genuine ``n_layers=1`` case, not a sweep, but
handled by the same (fROI, layer) task-grid machinery). Also includes the
FLA-based ``trust_remote_code`` families (stickbreaking, alibi, forgetting,
hard_2gram/3gram/5gram) via ``logit_lens.py``'s hook-based hidden-states
fallback — see ``docs/history/logit_lens_constrained_models_handoff.md`` (verified on Jean Zay H100). One
narrow exception within ``babylm``: the MoE model ``Jtapsa/moep`` specifically
fails (variable residual-stream width across layers), while the rest of that
family works.

Pipeline (per model checkpoint)
--------------------------------
1. For each of the 8 stories: load the canonical word list, compute
   per-layer logit-lens surprisal (single model load per doc, disk-cached —
   same ``logit_lens_surprisal_all_layers`` cache used by Fedorenko2016/
   Tuckute) and wordfreq Zipf frequency, then HRF-convolve
   ``[surprisal_layer_0, ..., surprisal_layer_{L-1}, wlen, freq, sentpos]``
   together via the same ``word_hrf_basis_matrix`` machinery the ridge
   encoder uses — one ``B @ features`` matmul per doc (cost barely changes
   with the extra layer columns; a dense matmul this small is sub-ms
   regardless), shared across all 78 subjects.
2. Cut each doc's TRs into contiguous ``chunk_seconds``-long chunks (default
   30 s, matching Shain et al. 2020 / Lin & Schuler) — the atomic unit for the
   train/held-out split, so no chunk straddles the boundary.
3. For each subject, load BOLD per fROI for all 8 stories, align with the
   convolved design, and append to one long TR-level table (subject, story,
   chunk, wlen, freq, sentpos, bold_<fROI>) plus a separate (n_rows,
   n_layers) surprisal array — written to parquet/npy ONCE, then read
   (mmap'd) by worker processes, same pattern as Fedorenko2016/Tuckute.
4. For each (fROI, layer) pair with enough valid TRs: fit baseline/full LME,
   compute train delta-loglik (LRT) and held-out delta-loglik +
   Pearson/Spearman r (``fit_fmri_lme``) — via a spawned process pool
   (``max_tasks_per_child=1``: lme4/rpy2's embedded R interpreter leaks
   memory across repeated fits in one process).
5. Append each (fROI, layer) result to CSV the moment it completes
   (crash-safe restart — already-completed (fROI, layer, revision) triples
   are skipped on re-run).

Cross-validation
-----------------
Train/held-out split is by contiguous TR-chunk (not leave-story-out, and not
the SPR LME's ``(subject + sentid) % mod`` hash — BOLD is autocorrelated
within a story, so held-out partitions must respect chunk contiguity). Both
``subject`` and ``story`` random-effect levels appear in both partitions, so
held-out predictions use the model's actual fitted subject/story slopes.

Outputs (written to output_dir)
---------------------------------
  ns_fmri_lme_by_roi.csv       — one row per (fROI, layer, revision)
  ns_fmri_lme_by_network.csv   — one row per (network, layer, revision)
  ns_fmri_lme_checkpoint_scores.csv — one row per revision (headline summary,
                                       computed from the LAST layer — see
                                       ``scripts/run/surprisal/neural/ns_fmri_fit.py``)

Reference
---------
Lin, Y.-C. & Schuler, W. (2025). Surprisal from Larger Transformer-based
Language Models Predicts fMRI Data More Poorly. arXiv:2506.11338.
Shain et al. (2020). fMRI reveals language-specific predictive coding during
naturalistic sentence comprehension. Neuropsychologia, 138, 107307.
"""

from __future__ import annotations

import logging
import time
import typing as tp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

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
from ..features.llm import clear_hf_model_cache
from ..features.word_frequency import compute_wlen_wordfreq_covariates
from ..sweep import csv_file_lock
from .base import ModelConfig

logger = logging.getLogger(__name__)

_LAYER_ROW_COLUMNS = (
    "froi", "network", "hemisphere", "layer", "beta", "se", "t_stat", "p_value",
    "loglik", "aic", "bic", "baseline_loglik", "delta_loglik", "delta_loglik_train",
    "delta_loglik_test", "baseline_loglik_test", "loglik_test", "lrt_p_value",
    "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "n_obs", "n_obs_test", "error", "fit_wall_time_s", "revision",
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NsFmriLmeDataConfig:
    """Dataset location and loading options."""

    data_root: str
    froi_table: str = "LANG_MD"
    docs: list[str] | None = None       # None = all 8 stories
    subjects: list[str] | None = None   # None = all 78 subjects
    t_r: float = NS_FMRI_TR
    word_duration: float = NS_FMRI_WORD_DUR
    hrf_model: str = "glover"
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    feature_cache_dir: str | None = None   # logit-lens surprisal cache (per doc)


@dataclass
class NsFmriLmeConfig:
    """LME fitting options (Lin & Schuler 2025 formula + freq_z, per-layer sweep)."""

    chunk_seconds: float = 30.0
    train_frac: float = 0.75
    split_random_state: int = 0
    min_valid_trs: int = 50   # minimum valid (non-NaN) TRs to attempt a fit
    verbose: bool = False
    # Which layers to fit: 1 = all layers; N>1 = every Nth layer.
    layer_stride: int = 1
    # If True, fit ONLY the final layer (index n_layers-1), ignoring layer_stride.
    last_layer_only: bool = False
    # Process-pool size for fitting (fROI, layer) pairs concurrently. Each
    # worker is a separate process with its own embedded R/pymer4 interpreter.
    n_workers: int = 20
    # CPU sweep runs are the Stage-2 LME-only half of a two-stage pipeline —
    # a cache miss there should fail fast rather than silently loading a model.
    require_cached_surprisal: bool = False
    # False reproduces Lin & Schuler (2025)'s original formula exactly (no
    # freq_z nuisance term in either baseline or full model) — see
    # ``neuroai/regressors/fmri_lme.py`` module docstring.
    include_freq: bool = True
    # Save each held-out TR's observed BOLD + baseline/full model predictions
    # (ns_fmri_lme_per_word.csv, one row per (fROI, layer, TR) — same
    # DLT-difficulty-style convention as every other pooled LME here, see
    # ``fit_spr_lme(..., save_per_word=True)``. On by default.
    save_word_predictions: bool = True


@dataclass
class NsFmriLmeBenchmarkConfig:
    data: NsFmriLmeDataConfig
    model: ModelConfig
    lme: NsFmriLmeConfig = field(default_factory=NsFmriLmeConfig)
    output_dir: str = "outputs"
    # Grouping label for saved CSVs when HF load repo differs (e.g. OLMo early-training).
    canonical_model_name: str | None = None


# ---------------------------------------------------------------------------
# Worker (module-level for spawn-based multiprocessing pickling)
# ---------------------------------------------------------------------------

def _fit_one_froi_layer_worker(
    nuisance_parquet_path: str,
    surprisal_npy_path: str,
    bold_npy_path: str,
    froi_idx: int,
    froi: str,
    network: str,
    hemisphere: str,
    layer_idx: int,
    *,
    train_frac: float,
    random_state: int,
    verbose: bool,
    include_freq: bool = True,
    save_word_predictions: bool = False,
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: load the shared tables, fit one (fROI, layer)."""
    import numpy as _np
    import pandas as _pd

    from ..regressors.fmri_lme import fit_fmri_lme

    t0 = time.time()
    row: dict[str, tp.Any] = {
        "froi": froi, "network": network, "hemisphere": hemisphere, "layer": layer_idx,
    }
    per_word_out: _pd.DataFrame | None = None
    try:
        nuisance_df = _pd.read_parquet(nuisance_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        bold_all = _np.load(bold_npy_path, mmap_mode="r")

        col = _np.asarray(bold_all[:, froi_idx], dtype=_np.float64)
        surp_col = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)
        # A non-finite surprisal value at even one word can, after HRF
        # convolution (~20s-wide kernel), NaN-poison many nearby TRs — must
        # be dropped explicitly here, not left to R's silent na.omit, which
        # can otherwise collapse a story/subject down to too few rows for
        # lme4 to estimate that grouping factor's variance at all ("grouping
        # factors must have > 1 sampled level").
        valid = _np.isfinite(col) & _np.isfinite(surp_col)
        n_dropped_surp = int((_np.isfinite(col) & ~_np.isfinite(surp_col)).sum())
        if n_dropped_surp:
            logger.warning(
                "[%s | layer %d] dropping %d/%d TR(s) with non-finite surprisal",
                froi, layer_idx, n_dropped_surp, int(_np.isfinite(col).sum()),
            )
        df_local = nuisance_df.loc[valid].copy()
        df_local["surprisal"] = surp_col[valid]
        df_local["bold"] = col[valid]

        fit = fit_fmri_lme(
            df_local,
            train_frac=train_frac,
            random_state=random_state,
            verbose=verbose,
            include_freq=include_freq,
            save_per_word=save_word_predictions,
        )
        if fit.per_word is not None and not fit.per_word.empty:
            per_word_out = fit.per_word.copy()
            per_word_out.insert(0, "layer", layer_idx)
            per_word_out.insert(0, "froi", froi)
        row.update({
            "beta": fit.beta,
            "se": fit.se,
            "t_stat": fit.t_stat,
            "p_value": fit.p_value,
            "loglik": fit.loglik,
            "aic": fit.aic,
            "bic": fit.bic,
            "baseline_loglik": fit.baseline_loglik,
            "delta_loglik": fit.delta_loglik,
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
            "n_obs": fit.n_obs,
            "n_obs_test": fit.n_obs_test,
            "error": None,
        })
    except Exception as exc:  # noqa: BLE001 — one bad (fROI, layer) must not kill the pool
        logger.error("[%s | layer %d] fit failed", froi, layer_idx, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return {"row": row, "per_word": per_word_out}


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class NsFmriLmeBenchmark:
    """Predict Natural Stories fMRI BOLD from per-layer LLM surprisal via LME.

    Per-layer logit-lens surprisal is extracted once per story (single model
    load), HRF-convolved together with word length, frequency, and sentence
    position (shared across all 78 subjects), then one mixed-effects model
    is fit per (fROI, layer) pair, pooling every subject and every story.
    """

    def __init__(self, cfg: NsFmriLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(self) -> pd.DataFrame:
        dcfg = self.cfg.data
        docs = dcfg.docs or NS_FMRI_DOCS
        subjects = dcfg.subjects or available_subjects(dcfg.data_root, dcfg.froi_table)
        model_revision = self.cfg.model.model_revision or "main"
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        roi_csv = out_dir / "ns_fmri_lme_by_roi.csv"
        net_csv = out_dir / "ns_fmri_lme_by_network.csv"

        # ── Step 1: per-doc convolved [surprisal_layers..., wlen, freq, sentpos] + chunk ids ──
        conv_by_doc: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        n_words_total = 0
        n_layers: int | None = None
        for doc_idx, doc in enumerate(docs):
            logger.info("=== Story: %s (idx=%d) ===", doc, doc_idx)
            words_df = load_canonical_words(dcfg.data_root, doc, dcfg.froi_table)
            n_words_total += len(words_df)

            surprisal_layers = self._word_surprisal_all_layers(words_df, doc_idx)  # (n_words, n_layers)
            if n_layers is None:
                n_layers = surprisal_layers.shape[1]
            freq = compute_wlen_wordfreq_covariates(
                words_df["word"].astype(str).tolist(),
            )[:, 1].astype(np.float64)
            raw_feats = np.concatenate(
                [
                    surprisal_layers,
                    words_df["wlen"].to_numpy(dtype=np.float64)[:, None],
                    freq[:, None],
                    words_df["sentpos"].to_numpy(dtype=np.float64)[:, None],
                ],
                axis=1,
            )  # (n_words, n_layers + 3): [surprisal_layer_0..L-1, wlen, freq, sentpos]

            # A single non-finite word-level feature (e.g. logit-lens surprisal
            # is structurally undefined for a story's first word — no preceding
            # context to condition on) must NOT reach the B @ raw_feats HRF
            # convolution below: under IEEE float rules 0 * NaN = NaN, so even
            # an HRF regressor column that's exactly zero far from that word's
            # onset still turns the *entire* matmul output NaN once summed —
            # not just "nearby TRs". That silently wipes an entire story's
            # convolved design, which downstream (validity filtering by
            # isfinite) collapses the `story` random-effect grouping factor to
            # too few levels for lme4 ("grouping factors must have > 1 sampled
            # level"). Zero out the bad word's contribution instead — one
            # word's amplitude out of ~1000 lost is far cheaper than an entire
            # story's TRs.
            bad = ~np.isfinite(raw_feats)
            if bad.any():
                n_bad_words = int(bad.any(axis=1).sum())
                logger.warning(
                    "[%s | %s] %d/%d word(s) had non-finite feature(s) "
                    "(e.g. undefined first-word surprisal) — zeroing before "
                    "HRF convolution to avoid NaN-poisoning the whole story",
                    model_name, doc, n_bad_words, len(raw_feats),
                )
                raw_feats = np.where(bad, 0.0, raw_feats)

            onsets = words_df["time"].to_numpy(dtype=np.float64)
            last_onset = float(onsets[-1])
            n_trs = int(np.ceil((last_onset + 20.0) / dcfg.t_r)) + 1
            frame_times = np.arange(n_trs) * dcfg.t_r
            durations = np.full(len(onsets), dcfg.word_duration)
            B = word_hrf_basis_matrix(onsets, durations, frame_times, dcfg.hrf_model)
            design = B @ raw_feats  # (n_trs, n_layers + 3), HRF-convolved raw values

            chunk_len_trs = max(1, int(round(self.cfg.lme.chunk_seconds / dcfg.t_r)))
            chunk_ids = np.arange(n_trs) // chunk_len_trs

            conv_by_doc[doc] = (design.astype(np.float64), chunk_ids)
            logger.info(
                "[%s | %s] design shape: %s, %d chunks",
                model_name, doc, "×".join(str(d) for d in design.shape),
                int(chunk_ids.max()) + 1,
            )

        assert n_layers is not None
        clear_hf_model_cache()

        # ── Step 2: assemble one long TR-level table across subjects × docs ──
        # Not every subject/doc has BOLD for every fROI (some localizer parcels
        # fail per subject), so column *count* can differ across subject/doc
        # pairs (e.g. 25 vs 26 valid fROIs). Align every block onto one
        # canonical fROI order by name — inserting NaN for missing fROIs —
        # before concatenating across subjects, otherwise np.concatenate raises
        # a shape mismatch (or silently misaligns fROI identity by position).
        canonical_frois = available_frois(dcfg.data_root, dcfg.froi_table)
        froi_to_col = {f: i for i, f in enumerate(canonical_frois)}
        froi_to_network: dict[str, str] = {}

        subj_parts: list[np.ndarray] = []
        story_parts: list[np.ndarray] = []
        chunk_parts: list[np.ndarray] = []
        surp_parts: list[np.ndarray] = []
        wlen_parts: list[np.ndarray] = []
        freq_parts: list[np.ndarray] = []
        sentpos_parts: list[np.ndarray] = []
        bold_parts: list[np.ndarray] = []

        for subj in subjects:
            bold_data = load_bold_subject(dcfg.data_root, subj, dcfg.froi_table, docs=docs)
            if not bold_data:
                continue
            for doc in docs:
                if doc not in bold_data or doc not in conv_by_doc:
                    continue
                bold, scan_times, doc_frois, doc_networks = bold_data[doc]
                for f, net in zip(doc_frois, doc_networks):
                    froi_to_network.setdefault(f, net)

                design, chunk_ids = conv_by_doc[doc]
                n_trs = min(len(scan_times), design.shape[0])
                if n_trs < 20:
                    continue

                bold_aligned = np.full((n_trs, len(canonical_frois)), np.nan, dtype=np.float64)
                for local_j, f in enumerate(doc_frois):
                    gj = froi_to_col.get(f)
                    if gj is not None:
                        bold_aligned[:, gj] = bold[:n_trs, local_j]

                subj_parts.append(np.full(n_trs, subj, dtype=object))
                story_parts.append(np.full(n_trs, doc, dtype=object))
                chunk_parts.append(np.array([f"{doc}_{c}" for c in chunk_ids[:n_trs]], dtype=object))
                surp_parts.append(design[:n_trs, :n_layers])
                wlen_parts.append(design[:n_trs, n_layers])
                freq_parts.append(design[:n_trs, n_layers + 1])
                sentpos_parts.append(design[:n_trs, n_layers + 2])
                bold_parts.append(bold_aligned)

        if not bold_parts:
            raise RuntimeError("No subject/story BOLD data aligned with convolved design")

        froi_names = canonical_frois
        networks = [froi_to_network.get(f, "NA") for f in canonical_frois]

        nuisance_df = pd.DataFrame({
            "subject": np.concatenate(subj_parts),
            "story": np.concatenate(story_parts),
            "chunk": np.concatenate(chunk_parts),
            "wlen": np.concatenate(wlen_parts),
            "freq": np.concatenate(freq_parts),
            "sentpos": np.concatenate(sentpos_parts),
        })
        surprisal_all = np.concatenate(surp_parts, axis=0)  # (N, n_layers)
        bold_all = np.concatenate(bold_parts, axis=0)  # (N, n_fROIs)
        logger.info(
            "[NS-fMRI-LME] pooled table: %d TR-rows across %d subjects × %d stories, "
            "%d fROIs, %d layers",
            len(nuisance_df), len(subjects), len(docs), len(froi_names), n_layers,
        )

        # Revision-namespaced so concurrent per-checkpoint sweeps sharing this
        # out_dir (e.g. `xargs -P` / a SLURM array over the *_checkpoints
        # family) never overwrite each other's pooled table before the
        # per-fROI worker processes reload it — the same fix applied to
        # ucl_reading_lme.py / natural_stories_lme.py on 2026-08-26, missed
        # here. Without it, most checkpoints' fits silently reuse whichever
        # revision's surprisal.npy landed on disk last.
        _rev_tag = str(model_revision).replace("/", "_")
        nuisance_parquet_path = out_dir / f"ns_fmri_lme_nuisance_table__{_rev_tag}.parquet"
        surprisal_npy_path = out_dir / f"ns_fmri_lme_surprisal__{_rev_tag}.npy"
        bold_npy_path = out_dir / f"ns_fmri_lme_bold__{_rev_tag}.npy"
        nuisance_df.to_parquet(nuisance_parquet_path, index=False)
        np.save(surprisal_npy_path, surprisal_all.astype(np.float32))
        np.save(bold_npy_path, bold_all.astype(np.float32))
        del nuisance_df, surprisal_all

        # ── Step 3: build the (fROI, layer) task grid, skipping already-done pairs ──
        valid_frois: list[tuple[int, str, str, str]] = []  # (froi_idx, froi, network, hemisphere)
        for j, froi in enumerate(froi_names):
            valid = np.isfinite(bold_all[:, j])
            if valid.sum() < self.cfg.lme.min_valid_trs:
                logger.warning("  [%s] only %d valid TRs — skipping", froi, int(valid.sum()))
                continue
            net = networks[j] if networks else "NA"
            hemi = "L" if "L" in froi[4:6] else "R"
            valid_frois.append((j, froi, net, hemi))

        if self.cfg.lme.last_layer_only:
            layer_indices = [n_layers - 1]
        else:
            layer_indices = list(range(0, n_layers, max(1, self.cfg.lme.layer_stride)))
        logger.info(
            "Fitting %d fROIs × %d layers (stride=%d) = %d tasks with %d worker processes",
            len(valid_frois), len(layer_indices), self.cfg.lme.layer_stride,
            len(valid_frois) * len(layer_indices), self.cfg.lme.n_workers,
        )

        done_pairs: set[tuple[str, int]] = set()
        if roi_csv.exists():
            try:
                existing = pd.read_csv(roi_csv)
                existing_rev = existing[existing["revision"] == model_revision]
                good = existing_rev[existing_rev["error"].isna()]
                done_pairs = set(zip(good["froi"], good["layer"]))
                if done_pairs:
                    logger.info("Skipping %d already-fit (fROI, layer) pair(s)", len(done_pairs))
            except Exception:
                logger.warning("Could not read existing ROI CSV; re-fitting all pairs.")

        tasks = [
            (froi_idx, froi, net, hemi, li)
            for (froi_idx, froi, net, hemi) in valid_frois
            for li in layer_indices
            if (froi, li) not in done_pairs
        ]

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        per_word_csv = out_dir / f"ns_fmri_lme_per_word__{_rev_tag}.csv"
        rows: list[dict] = []
        # max_tasks_per_child=1: recycle each worker after ONE fit — lme4/rpy2's
        # embedded R interpreter leaks memory across repeated fits in one
        # process (same mitigation as every other pooled LME in this codebase).
        with ProcessPoolExecutor(
            max_workers=self.cfg.lme.n_workers, mp_context=ctx, max_tasks_per_child=1,
        ) as pool:
            futures = {
                pool.submit(
                    _fit_one_froi_layer_worker,
                    str(nuisance_parquet_path), str(surprisal_npy_path), str(bold_npy_path),
                    froi_idx, froi, net, hemi, li,
                    train_frac=self.cfg.lme.train_frac,
                    random_state=self.cfg.lme.split_random_state,
                    verbose=self.cfg.lme.verbose,
                    include_freq=self.cfg.lme.include_freq,
                    save_word_predictions=self.cfg.lme.save_word_predictions,
                ): (froi, li)
                for (froi_idx, froi, net, hemi, li) in tasks
            }
            for fut in as_completed(futures):
                froi, li = futures[fut]
                try:
                    result = fut.result()
                    row, per_word = result["row"], result["per_word"]
                except Exception as exc:
                    logger.error("[%s | layer %d] worker crashed: %s", froi, li, exc, exc_info=True)
                    row, per_word = {"froi": froi, "layer": li, "error": str(exc)}, None
                row["revision"] = model_revision
                rows.append(row)
                with csv_file_lock(roi_csv):
                    pd.DataFrame([row]).reindex(columns=_LAYER_ROW_COLUMNS).to_csv(
                        roi_csv, mode="a", header=not roi_csv.exists(), index=False,
                    )
                if per_word is not None:
                    per_word = per_word.copy()
                    per_word.insert(0, "revision", model_revision)
                    with csv_file_lock(per_word_csv):
                        per_word.to_csv(
                            per_word_csv, mode="a", header=not per_word_csv.exists(), index=False,
                        )
                logger.info(
                    "  [%s | layer %d] done in %.1fs — delta_loglik_test=%s converged=%s",
                    froi, li, row.get("fit_wall_time_s", float("nan")),
                    f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                    row.get("converged"),
                )

        self._save_network_summary(roi_csv, net_csv, model_revision)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Surprisal extraction (per-layer logit-lens, one model load per doc loop, disk-cached)
    # ------------------------------------------------------------------

    def _word_surprisal_all_layers(self, words_df: pd.DataFrame, doc_idx: int) -> np.ndarray:
        """Per-layer logit-lens surprisal for every word in one story.

        One forward pass (sliding-window chunked internally for long
        documents) via :func:`logit_lens_surprisal_all_layers` — same
        extractor already used by Fedorenko2016/Tuckute. ``doc_idx``
        (0-7, the story's position in ``docs``) is used as ``section_id``
        so all 8 stories can share one cache directory without collisions.

        CBR-RNN is a special case: it bypasses ``AutoModelForCausalLM``
        entirely and has a single hidden-state vector per token (no layer
        dimension), so logit-lens doesn't apply — ``batch_surprisal_cbr_rnn``
        is used instead, returning a single "layer" of surprisal. The rest
        of the pipeline (design matrix, (fROI, layer) task grid) treats
        ``n_layers`` generically, so ``n_layers=1`` for this family just
        means each fROI gets exactly one fit instead of a sweep.

        Returns
        -------
        np.ndarray  shape (n_words, n_layers).
        """
        mcfg = self.cfg.model
        dcfg = self.cfg.data
        # require_cache_hit is enforced inside batch_surprisal_cbr_rnn /
        # logit_lens_surprisal_all_layers themselves (covers both "no
        # cache_dir configured" and "cache_dir configured but this specific
        # item missing" — no separate pre-check needed here).
        require_cache_hit = self.cfg.lme.require_cached_surprisal

        words = words_df["word"].astype(str).tolist()

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
                section_id=doc_idx,
                require_cache_hit=require_cache_hit,
            )  # (n_words, 1)
            return surp.astype(np.float64)

        from ..features.logit_lens import logit_lens_surprisal_all_layers

        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None
        arr = logit_lens_surprisal_all_layers(
            words,
            mcfg.model_name,
            revision=mcfg.model_revision,
            device=mcfg.device,
            max_length=dcfg.max_length,
            tokenizer_no_revision=dcfg.tokenizer_no_revision,
            cache_dir=dcfg.feature_cache_dir,
            section_id=doc_idx,
            n_params=mcfg.n_params,
            trust_remote_code=mcfg.trust_remote_code,
            untrained=mcfg.untrained,
            untrained_seed=mcfg.untrained_seed,
            require_cache_hit=require_cache_hit,
            dtype=surp_dtype,
        )  # (n_words, n_layers, 1)
        return arr[:, :, 0].astype(np.float64)

    # ------------------------------------------------------------------
    # Network-level summary
    # ------------------------------------------------------------------

    def _save_network_summary(
        self,
        roi_csv: Path,
        net_csv: Path,
        model_revision: str,
    ) -> None:
        if not roi_csv.exists():
            return
        df = pd.read_csv(roi_csv)
        df = df[(df["revision"] == model_revision) & df["error"].isna()]
        if df.empty:
            return
        rows: list[dict] = []
        for (net, layer), gdf in df.groupby(["network", "layer"]):
            rows.append({
                "network": net,
                "layer": layer,
                "revision": model_revision,
                "beta_mean": gdf["beta"].mean(),
                "delta_loglik_test_mean": gdf["delta_loglik_test"].mean(),
                "test_spearmanr_full_model_mean": gdf["test_spearmanr_full_model"].mean(),
                "test_pearsonr_full_model_mean": gdf["test_pearsonr_full_model"].mean(),
                "n_frois": len(gdf),
            })
        with csv_file_lock(net_csv):
            pd.DataFrame(rows).to_csv(
                net_csv, mode="a", header=not net_csv.exists(), index=False,
            )
