"""Szewczyk & Federmeier N400 EEG encoding benchmark — pooled LME per LM layer.

Unlike the per-subject ridge encoder (``benchmarks/n400_szewczyk.py``, which
predicts N400 from full hidden-state embeddings via per-subject ridge), this
pools every subject into ONE mixed-effects model per layer, using per-layer
logit-lens surprisal as the predictor of interest — same "two paths" split as
Nieuwland (embeddings-ridge vs. LME-surprisal) in this
codebase. See ``neuroai/regressors/n400_pooled_lme.py``'s
``_n400_ceiling_formulas_szewczyk_paper``/``prepare_szewczyk_paper_frame`` for
the formula/fitting details, matched exactly to
``data/n400_paper_formula/osfstorage/calculate_fits/calculate_aics_szewczyk.R``.

Unlike Nieuwland, this dataset's N400 is already pre-averaged
over 4 centro-parietal ROI clusters into ONE scalar per trial (see
``neuroai/datasets/szewczyk.py``) — there is no per-electrode long table, so
no electrode random effect and no per-(subject, electrode) broadcasting.

Data source: rather than ``neuroai.datasets.szewczyk``'s own ``.rds`` loader
(which lacks the paper's bline/logfreq/pos_start/old20/concr covariates
entirely), this loads
``data/n400_paper_formula/osfstorage/datasets/szewczyk_2022.tsv`` directly — the OSF
replication package's own trial-level table, which already carries n400,
bline, and every fixed-effect covariate the paper formula needs (same
approach the R script itself takes: ``read_tsv("../datasets/szewczyk_2022.tsv")``).
138 subjects, 5 sub-experiments ("dataset" column), 1,330 unique
(FullText, TargetWords) context/critical-word pairs, 27,762 trials.

Pipeline
--------
1. Load ``szewczyk_2022.tsv``; extract per-layer logit-lens surprisal at each
   unique (context, target) pair's critical word, plus the previous word's
   surprisal (spillover, unused by the paper formula but computed for parity
   with the other N400 LME benchmarks / possible future generic-formula use).
2. Broadcast surprisal back onto all 27,762 trial rows by
   (FullText, TargetWords).
3. ``chunk`` = ``dataset + "_" + Item`` (Item numbers repeat across the 5
   sub-experiments -- see szewczyk.py's module docstring -- so the train/test
   split key must disambiguate by sub-experiment).
4. Write the pooled base table (parquet) and the (n_rows, n_layers) surprisal
   array (npy) to disk once, then fit one LME per layer via a process pool
   (pymer4/rpy2 embeds one R interpreter per process, so parallelism must be
   process-based) using the paper-exact formula.

Output: ``szewczyk_lme_by_layer.csv`` — one row per layer, written
incrementally as each layer's fit completes.
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

from ..sweep import csv_file_lock
from .base import ModelConfig, word_predictions_dir

logger = logging.getLogger(__name__)

# Fixed column order for szewczyk_lme_by_layer.csv — a crash-fallback row
# (only a few keys) is reindexed to this before writing, so it can never
# misalign the CSV's columns.
_LAYER_ROW_COLUMNS = (
    "layer", "beta", "se", "t_stat", "p_value", "loglik", "aic",
    "bic", "baseline_loglik", "delta_loglik_train", "delta_loglik_test",
    "baseline_loglik_test", "loglik_test", "lrt_p_value",
    "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "slope_dropped", "n_obs", "n_obs_test", "error", "fit_wall_time_s", "revision",
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SzewczykLmeDataConfig:
    # OSF replication-package trial table -- carries n400, bline, logfreq,
    # pos_start, old20, concr natively (unlike neuroai.datasets.szewczyk's
    # own .rds loader). See this module's docstring.
    data_path: str | Path = "data/n400_paper_formula/osfstorage/datasets/szewczyk_2022.tsv"
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    # Device for the (cheap, ~1,330-item) surprisal extraction forward passes.
    device: str = "cpu"
    # Surprisal cache dir. On Jean Zay, R (pymer4/lme4) is only available on
    # CPU partitions, not GPU — so, like Podcast's pooled LME,
    # extraction and LME fitting must run as separate jobs: a GPU job
    # populates this cache, then the CPU-only LME job reads from it.
    feature_cache_dir: str | None = None
    # Restrict the pooled table to ONE sub-experiment (the tsv's ``dataset`` column, e.g.
    # "Szewczyk & Federmeier (2022)"); None = pool all five (the original behaviour). The
    # surprisal items/cache still cover every item (the cache is keyed by model only, so
    # filtering items before extraction would misalign it) -- only the TRIAL ROWS are filtered.
    sub_experiment: str | None = None


@dataclass
class SzewczykLmeConfig:
    train_frac: float = 0.75
    split_random_state: int = 0
    verbose: bool = False
    # Barr et al. (2013)-style simplification ladder if the full model (with
    # the uncorrelated random slopes on subject/item) fails to converge:
    # drop the slopes entirely (intercept-only). Applied identically to
    # every layer so comparisons stay valid.
    allow_slope_fallback: bool = True
    # Which layers to fit: 1 = all layers; N>1 = every Nth layer.
    layer_stride: int = 1
    # If True, fit ONLY the final layer (index n_layers-1), ignoring layer_stride.
    last_layer_only: bool = False
    # Explicit layer index list — when set, overrides layer_stride AND
    # last_layer_only entirely (cheap second pass after a first pass
    # identified each model's best layer).
    layers: tuple[int, ...] | None = None
    # Process-pool size for fitting layers concurrently. Each worker is a
    # separate Python process (spawn) with its own embedded R/pymer4
    # interpreter — rpy2 is not thread-safe, so this must be process-based.
    n_workers: int = 10
    # Save held-out per-trial predictions alongside the usual per-layer
    # summary — needed for post-hoc per-word difficulty splits. On by default.
    save_word_predictions: bool = True
    # Write the pooled surprisal table/array (what scripts/analysis/
    # surprisal_rsa.py actually reads), then return immediately -- skip the
    # entire per-layer LME regression loop. RSA only needs the cached
    # surprisal values, not the benchmark's own statistical fit; the LME
    # loop is by far the most expensive part of this pipeline and is pure
    # overhead when only backfilling the RSA-facing cache.
    surprisal_only: bool = False


@dataclass
class SzewczykLmeBenchmarkConfig:
    data: SzewczykLmeDataConfig
    model: ModelConfig
    lme: SzewczykLmeConfig = field(default_factory=SzewczykLmeConfig)
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
    per_word_dir: str | None = None,
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: load the shared table, fit one layer."""
    import numpy as _np
    import pandas as _pd

    from ..regressors.n400_pooled_lme import fit_n400_lme_one_layer_paper

    t0 = time.time()
    row: dict[str, tp.Any] = {"layer": layer_idx}
    try:
        base_df = _pd.read_parquet(base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        base_df["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)

        fit = fit_n400_lme_one_layer_paper(
            base_df,
            dataset="szewczyk",
            train_frac=train_frac,
            random_state=random_state,
            verbose=verbose,
            allow_slope_fallback=allow_slope_fallback,
            save_per_word=save_word_predictions,
        )
        if save_word_predictions and fit.per_word is not None and per_word_dir is not None:
            _pd.DataFrame(fit.per_word).to_parquet(
                f"{per_word_dir}/layer{layer_idx:03d}.parquet", index=False,
            )
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
    return row


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class SzewczykLmeBenchmark:
    """Pooled Szewczyk N400 LME: one mixed-effects model per LM layer."""

    def __init__(self, cfg: SzewczykLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def _load_trial_table(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.data.data_path, sep="\t")
        df["chunk"] = df["dataset"].astype(str) + "_" + df["Item"].astype(str)
        return df

    def extract_and_cache_surprisal(self) -> np.ndarray:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only.

        Loads the LLM, computes per-layer logit-lens surprisal for all 1,330
        unique (context, critical-word) pairs, and returns — does not touch
        the N400 trial table, does not fit anything. Run this once per model
        (GPU job) before the CPU-only ``run()`` LME sweep, same two-stage
        split as Podcast's pooled LME (R/pymer4 is CPU-only on
        Jean Zay, so extraction and fitting can't share a job)."""
        df = self._load_trial_table()
        items = df[["FullText", "TargetWords"]].drop_duplicates().reset_index(drop=True)
        contexts = items["FullText"].astype(str).tolist()
        targets = items["TargetWords"].astype(str).tolist()
        return self._extract_item_surprisal(contexts, targets)

    def run(self, extra_fields: dict[str, tp.Any] | None = None) -> pd.DataFrame:
        dcfg = self.cfg.data
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        revision = self.cfg.model.model_revision or "main"

        df = self._load_trial_table()
        items = df[["FullText", "TargetWords"]].drop_duplicates().reset_index(drop=True)
        items["item_id"] = items.index
        contexts = items["FullText"].astype(str).tolist()
        targets = items["TargetWords"].astype(str).tolist()

        surprisal_items = self._extract_item_surprisal(contexts, targets)

        df_merged = df.merge(items, on=["FullText", "TargetWords"], how="left")
        item_finite = np.isfinite(surprisal_items).all(axis=1)
        if not item_finite.all():
            n_dropped = int((~item_finite).sum())
            logger.warning(
                "Dropping %d/%d unique item(s) with non-finite surprisal before pooling",
                n_dropped, len(items),
            )
        keep_item_ids = set(items.loc[item_finite, "item_id"].tolist())
        row_mask = df_merged["item_id"].isin(keep_item_ids) & df_merged["n400"].notna()
        if dcfg.sub_experiment:
            known = sorted(df_merged["dataset"].unique())
            if dcfg.sub_experiment not in known:
                raise ValueError(f"sub_experiment {dcfg.sub_experiment!r} not in {known}")
            row_mask &= df_merged["dataset"] == dcfg.sub_experiment
        df_merged = df_merged.loc[row_mask].reset_index(drop=True)
        row_item_idx = df_merged["item_id"].values.astype(np.int64)

        n_layers = surprisal_items.shape[1]
        surprisal_by_row = surprisal_items[row_item_idx]  # (n_rows, n_layers)

        base_df = pd.DataFrame({
            "subject": df_merged["Subject"].astype(str).values,
            "chunk": df_merged["chunk"].astype(str).values,
            "bline": df_merged["bline"].astype(float).values,
            "freq": df_merged["logfreq"].astype(float).values,
            "pos_start": df_merged["pos_start"].astype(float).values,
            "old20": df_merged["old20"].astype(float).values,
            "concr": df_merged["concr"].astype(float).values,
            "n400": df_merged["n400"].astype(float).values,
        })
        logger.info(
            "Pooled table: %d rows (%d subjects, %d chunks), %d layers",
            len(base_df), base_df["subject"].nunique(), base_df["chunk"].nunique(), n_layers,
        )

        safe_revision = revision.replace("/", "_")
        base_parquet_path = out_dir / f"szewczyk_lme_pooled_table_{safe_revision}.parquet"
        surprisal_npy_path = out_dir / f"szewczyk_lme_pooled_surprisal_{safe_revision}.npy"
        base_df.to_parquet(base_parquet_path, index=False)
        np.save(surprisal_npy_path, surprisal_by_row.astype(np.float32))
        del base_df, surprisal_by_row

        if self.cfg.lme.surprisal_only:
            logger.info(
                "surprisal_only: wrote %s and %s, skipping the LME regression loop.",
                base_parquet_path, surprisal_npy_path,
            )
            return pd.DataFrame()

        if self.cfg.lme.layers is not None:
            layer_indices = [li for li in self.cfg.lme.layers if 0 <= li < n_layers]
        elif self.cfg.lme.last_layer_only:
            layer_indices = [n_layers - 1]
        else:
            layer_indices = list(range(0, n_layers, max(1, self.cfg.lme.layer_stride)))
        logger.info(
            "Fitting %d layers (stride=%d) with %d worker processes",
            len(layer_indices), self.cfg.lme.layer_stride, self.cfg.lme.n_workers,
        )

        out_csv = out_dir / "szewczyk_lme_by_layer.csv"
        out_columns = list(_LAYER_ROW_COLUMNS) + [
            k for k in (extra_fields or {}) if k not in _LAYER_ROW_COLUMNS
        ]
        done_layers: set[int] = set()
        if out_csv.exists():
            try:
                existing = pd.read_csv(out_csv)
                existing_revision = existing["revision"] if "revision" in existing.columns else "main"
                same_revision = existing_revision == revision
                good = existing.loc[same_revision & existing["error"].isna()]
                done_layers = set(good["layer"].tolist())
                n_failed = int((same_revision & existing["error"].notna()).sum())
                if done_layers:
                    logger.info("Skipping %d already-fit layer(s) for revision=%s", len(done_layers), revision)
                if n_failed:
                    logger.warning(
                        "%d previously-crashed layer(s) found in %s for revision=%s — will retry",
                        n_failed, out_csv, revision,
                    )
                    keep = existing.loc[~(same_revision & existing["error"].notna())]
                    with csv_file_lock(out_csv):
                        keep.reindex(columns=out_columns).to_csv(out_csv, index=False)
            except Exception:
                logger.warning("Could not read existing layer CSV; re-fitting all layers.")

        per_word_dir = word_predictions_dir(out_dir, "szewczyk_lme_per_word")
        if self.cfg.lme.save_word_predictions:
            per_word_dir.mkdir(parents=True, exist_ok=True)

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        pending = [li for li in layer_indices if li not in done_layers]
        rows: list[dict] = []
        # max_tasks_per_child=1: recycle each worker after ONE layer fit —
        # lme4/rpy2's embedded R interpreter leaks memory across repeated
        # fits within a long-lived process (same mitigation as Nieuwland's).
        with ProcessPoolExecutor(
            max_workers=self.cfg.lme.n_workers, mp_context=ctx, max_tasks_per_child=1,
        ) as pool:
            futures = {
                pool.submit(
                    _fit_one_layer_worker,
                    str(base_parquet_path), str(surprisal_npy_path), li,
                    train_frac=self.cfg.lme.train_frac,
                    random_state=self.cfg.lme.split_random_state,
                    verbose=self.cfg.lme.verbose,
                    allow_slope_fallback=self.cfg.lme.allow_slope_fallback,
                    save_word_predictions=self.cfg.lme.save_word_predictions,
                    per_word_dir=str(per_word_dir) if self.cfg.lme.save_word_predictions else None,
                ): li
                for li in pending
            }
            for fut in as_completed(futures):
                li = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    logger.error("Layer %d worker crashed: %s", li, exc, exc_info=True)
                    row = {"layer": li, "error": str(exc)}
                row["revision"] = revision
                if extra_fields:
                    row.update(extra_fields)
                rows.append(row)
                with csv_file_lock(out_csv):
                    pd.DataFrame([row]).reindex(columns=out_columns).to_csv(
                        out_csv, mode="a", header=not out_csv.exists(), index=False,
                    )
                logger.info(
                    "  [layer %d] done in %.1fs — delta_loglik_test=%s converged=%s%s",
                    li, row.get("fit_wall_time_s", float("nan")),
                    f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                    row.get("converged"),
                    " (slope dropped)" if row.get("slope_dropped") else "",
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Feature extraction (per-layer logit-lens surprisal)
    # ------------------------------------------------------------------

    def _extract_item_surprisal(self, contexts: list[str], targets: list[str]) -> np.ndarray:
        """Per-layer surprisal for the critical word.

        CBR-RNN is single-layer — bypasses logit-lens entirely and uses
        plain final-output surprisal instead (same as Nieuwland/Natural
        Stories/Provo/GECO/MECO/etc.). Its output is ``(n_items, 1)``, which
        fits the ``(n_items, n_layers)`` convention unchanged (n_layers=1).

        Returns
        -------
        surprisal : (n_items, n_layers)
        """
        mcfg = self.cfg.model
        dcfg = self.cfg.data
        is_cbr_rnn = "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower()

        if is_cbr_rnn:
            from ..features.cbr_rnn import batch_surprisal_cbr_rnn

            return batch_surprisal_cbr_rnn(
                mcfg.model_name, contexts, targets,
                device=dcfg.device, n_params=mcfg.n_params,
            )

        from ..features.logit_lens import logit_lens_surprisal_trials

        surp_dtype = None
        if "gpt-bert" in mcfg.model_name.lower():
            surp_dtype = "float32"

        return logit_lens_surprisal_trials(
            contexts, targets, mcfg.model_name,
            cache_key="target",
            revision=mcfg.model_revision,
            device=dcfg.device,
            tokenizer_no_revision=dcfg.tokenizer_no_revision,
            n_params=mcfg.n_params,
            trust_remote_code=dcfg.trust_remote_code,
            cache_dir=dcfg.feature_cache_dir,
            dtype=surp_dtype,
        )
