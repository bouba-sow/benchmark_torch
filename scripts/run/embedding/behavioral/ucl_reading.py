#!/usr/bin/env python3
"""UCL Self-Paced Reading checkpoint sweep with ridge-on-embeddings regression.

Uses the same model registry/checkpoints as ``run/surprisal/behavioral/ucl_reading_fit.py``
but fits per-layer ridge regression (embeddings -> word-level reading time)
instead of an LME on scalar surprisal — the behavioral analogue of
``NeuralBenchmark``'s per-layer iEEG ridge regression (see
``neuroai.benchmarks.ucl_embed`` module docstring). UCL Reading has only one
measure (self-paced RT), unlike Provo/MECo's SP/FP/GP — no ``--measure`` flag.

Fit independently per subject (mirrors ``NeuralBenchmark`` fitting independently
per electrode) — ``ucl_reading_embed_by_subject.csv`` has one row per
(checkpoint, subject, layer); ``ucl_reading_embed_checkpoint_scores.csv`` has
the mean-across-subjects score per (checkpoint, layer), analogous to averaging
across electrodes.

Output structure
----------------
<output_dir>/
  <family>/
    <model_short>/
      ucl_reading_embed_checkpoint_scores.csv   # one row per (checkpoint, layer) — mean across subjects
      ucl_reading_embed_by_subject.csv          # one row per (checkpoint, subject, layer)
      ucl_reading_embed_training_curve.png
      ucl_reading_embed_layerwise_heatmap.png
    summary/
      all_models_ucl_reading_embed_scores.csv
      ucl_reading_embed_layerwise_heatmaps_grid.png
      scaling_law_final_checkpoint.png

Examples
--------
python scripts/run/embedding/behavioral/ucl_reading.py \\
    --config configs/embedding/behavioral/ucl_reading.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda

python scripts/run/embedding/behavioral/ucl_reading.py \\
    --config configs/embedding/behavioral/ucl_reading.yaml \\
    --summary_only --out_dir outputs/ucl_reading/embedding_sweep
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from neuroai.benchmarks.base import ModelConfig, RegressionConfig
from neuroai.benchmarks.ucl_embed import (
    UclReadingEmbedBenchmark,
    UclReadingEmbedBenchmarkConfig,
    UclReadingEmbedDataConfig,
)
from neuroai.datasets.ucl_reading import sentence_word_lists
from neuroai.features.llm import clear_hf_model_cache
from neuroai.perplexity import save_checkpoint_perplexity
from neuroai.models.registry import (
    build_checkpoint_items,
    registry_model_configs,
    resolve_family_task,
    resolve_load_model_id,
)
from neuroai.plotting.scaling import plot_layerwise_heatmap, plot_scaling_curve
from neuroai.sweep import (
    CheckpointItem,
    already_computed,
    append_row,
    free_gpu,
    load_existing_csv,
    safe_model_dir,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

SCORES_FILENAME = "ucl_reading_embed_checkpoint_scores.csv"
BY_SUBJECT_FILENAME = "ucl_reading_embed_by_subject.csv"


def _parse_alpha(raw: Any) -> Any:
    """Evaluate ``"np.logspace(2.5, 7, 6)"``-style alpha strings from YAML."""
    if isinstance(raw, str):
        return eval(raw, {"np": np})  # noqa: S307 — trusted config file, not user input
    return raw


def load_config(config_path: str | Path) -> dict[str, Any]:
    repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    os.environ.setdefault("BENCHMARK_ROOT", str(repo_root))
    scratch = os.environ.get("SCRATCH", "").strip()
    os.environ.setdefault(
        "BENCHMARK_CACHE_ROOT",
        f"{scratch}/.cache" if scratch else f"{os.environ['BENCHMARK_ROOT']}/.cache",
    )
    os.environ.setdefault(
        "UCL_READING_EMBED_CACHE_DIR",
        f"{os.environ['BENCHMARK_CACHE_ROOT']}/ucl_reading_embed_features",
    )
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    cfg = yaml.safe_load(raw)

    cache = cfg.setdefault("cache", {})
    feat_dir = os.environ.get("UCL_READING_EMBED_CACHE_DIR", "").strip()
    if feat_dir:
        cache.setdefault("features_dir", feat_dir)
    return cfg


def _model_output_dir(
    output_dir: Path,
    model_id: str,
    *,
    family: str | None = None,
    model_cfg: dict | None = None,
) -> Path:
    fam = family
    if fam is None and model_cfg is not None:
        fam = model_cfg.get("registry_family") or model_cfg.get("family")
    base = output_dir / fam if fam else output_dir
    return base / safe_model_dir(model_id)


def _discover_model_score_dirs(sweep_dir: Path) -> list[Path]:
    found = [scores.parent for scores in sorted(sweep_dir.glob(f"**/{SCORES_FILENAME}"))]
    found_set = set(found)
    return sorted(p for p in found if not any(q != p and q.is_relative_to(p) for q in found_set))


def _model_id_from_score_dir(score_dir: Path) -> str:
    return score_dir.name.replace("_", "/", 1)


def _family_from_score_dir(score_dir: Path, sweep_root: Path) -> str | None:
    if score_dir.parent != sweep_root and score_dir.parent.name != "summary":
        return score_dir.parent.name
    return None


def _checkpoint_items_from_task(task) -> list[CheckpointItem]:
    model_cfg = next(m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id)
    return [
        CheckpointItem(revision=rev, step=step, tokens_seen=tokens)
        for rev, step, tokens in build_checkpoint_items(model_cfg)
        if rev == task.revision
    ]


def _benchmark_config(
    cfg: dict,
    *,
    model_id: str,
    revision: str,
    n_params: float,
    batch_size: int,
    device: str,
    feature_cache_dir: Path | None,
    load_model_id: str | None = None,
    trust_remote_code: bool = False,
    output_dir: Path | None = None,
) -> UclReadingEmbedBenchmarkConfig:
    ds = cfg["dataset"]
    reg = cfg.get("regression", {})

    hf_model_id = load_model_id or model_id
    word_pred_root = os.environ.get("UCL_READING_WORD_PREDICTIONS_ROOT", "").strip()
    word_predictions_dir = str(Path(word_pred_root) / safe_model_dir(model_id)) if word_pred_root else None
    return UclReadingEmbedBenchmarkConfig(
        data=UclReadingEmbedDataConfig(
            root=ds["root"],
            sentences=ds.get("sentences"),
            min_accuracy=float(ds.get("min_accuracy", 0.0)),
            min_rt=float(ds.get("min_rt", 100.0)),
            max_rt=float(ds.get("max_rt", 3000.0)),
            context_words=int(ds.get("context_words", 48)),
            max_length=ds.get("max_length", 512),
            feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
        ),
        model=ModelConfig(
            model_name=hf_model_id,
            model_revision=revision if revision not in ("", "main") else None,
            device=device,
            batch_size=batch_size,
            n_params=n_params,
            trust_remote_code=trust_remote_code,
        ),
        regression=RegressionConfig(
            n_splits=int(reg.get("n_splits", 5)),
            alpha=_parse_alpha(reg.get("alpha", "np.logspace(2.5, 9, 8)")),
            scale=bool(reg.get("scale", True)),
            random_state=int(reg.get("random_state", 0)),
            print_chosen_ridge_alpha=bool(reg.get("print_chosen_ridge_alpha", True)),
        ),
        filter_sentence_edges=bool(cfg.get("filter_sentence_edges", True)),
        group_cv_by_story=bool(cfg.get("group_cv_by_story", True)),
        min_trials_per_subject=int(cfg.get("min_trials_per_subject", 20)),
        # Env overrides (not CLI flags): let a specific family's SLURM script
        # dial down concurrency for large models without touching the shared
        # config every other family in this sweep relies on for speed.
        # subject_n_jobs>1 spawns a ProcessPoolExecutor and PICKLES each
        # in-flight subject's full (n_words, n_layers, hidden) embedding
        # array across the IPC boundary -- for a big model (e.g. OPT-13B/
        # 66B) that's multiplied by subject_n_jobs concurrent copies in host
        # RAM, which OOM'd on the default cpus-per-task RAM share (see
        # neuroai.benchmarks.ucl_embed module docstring for the analogous
        # Qwen3-8B case that motivated the lazy generator this feeds).
        n_jobs=int(os.environ.get("UCL_READING_EMBED_N_JOBS", cfg.get("n_jobs", 10))),
        subject_n_jobs=int(os.environ.get("UCL_READING_EMBED_SUBJECT_N_JOBS", cfg.get("subject_n_jobs", 1))),
        canonical_model_name=model_id,
        output_dir=str(output_dir) if output_dir is not None else None,
        word_predictions_dir=word_predictions_dir,
        average_across_subjects=bool(cfg.get("average_across_subjects", False)),
        min_subjects_per_item=int(cfg.get("min_subjects_per_item", 1)),
        average_only=bool(cfg.get("average_only", False)),
    )


def _ckpt_rows(ckpt: CheckpointItem, df: pd.DataFrame) -> list[dict[str, Any]]:
    base = {"step": ckpt.step, "revision": ckpt.revision, "tokens_seen": ckpt.tokens_seen}
    return [{**base, **row} for row in df.to_dict("records")]


_corpus_text_cache: dict[str, str] = {}


def _ucl_reading_corpus_text(root: str) -> str:
    """Full UCL self-paced-reading corpus text, in sentence order — cached."""
    if root not in _corpus_text_cache:
        sent_lists = sentence_word_lists(root)
        parts = [" ".join(sent_lists[sid]) for sid in sorted(sent_lists)]
        _corpus_text_cache[root] = " ".join(parts)
    return _corpus_text_cache[root]


def run_model_sweep(
    checkpoints: list[CheckpointItem],
    *,
    model_id: str,
    n_params: float,
    batch_size: int,
    cfg: dict,
    output_dir: Path,
    device: str,
    feature_cache_dir: Path | None,
    no_plots: bool = False,
    model_cfg: dict[str, Any] | None = None,
    family: str | None = None,
    perplexity_dir: str | None = None,
    average_output_dir: Path | None = None,
) -> pd.DataFrame:
    data_root = cfg["dataset"]["root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / SCORES_FILENAME
    by_subject_csv = output_dir / BY_SUBJECT_FILENAME

    if average_output_dir is not None:
        average_output_dir.mkdir(parents=True, exist_ok=True)
    average_scores_csv = (average_output_dir / SCORES_FILENAME) if average_output_dir is not None else None

    existing = load_existing_csv(scores_csv)
    rows = existing.to_dict("records") if not existing.empty else []

    average_only = bool(cfg.get("average_only", False))
    for ckpt in checkpoints:
        already_main = True if average_only else already_computed(existing, ckpt.revision)
        already_avg = True
        if average_scores_csv is not None:
            already_avg = already_computed(load_existing_csv(average_scores_csv), ckpt.revision)
        if already_main and already_avg:
            logger.info("[%s] %s already done — skipping", model_id, ckpt.revision)
            continue

        logger.info("\n[%s] UCL Reading embed ── %s (step=%d) ──", model_id, ckpt.revision, ckpt.step)

        load_model_id = resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        bench_cfg = _benchmark_config(
            cfg,
            model_id=model_id,
            load_model_id=load_model_id,
            revision=ckpt.revision,
            n_params=n_params,
            batch_size=batch_size,
            device=device,
            feature_cache_dir=feature_cache_dir,
            trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
            output_dir=output_dir,
        )

        # Subject-level resumption: if a previous (crashed/timed-out) run for
        # this exact checkpoint already wrote some subjects' rows, skip them
        # and only fit the remaining subjects.
        by_subject_existing = load_existing_csv(by_subject_csv)
        existing_subject_rows = None
        if not by_subject_existing.empty and "revision" in by_subject_existing.columns:
            rev_rows = by_subject_existing[by_subject_existing["revision"] == ckpt.revision]
            if not rev_rows.empty:
                existing_subject_rows = rev_rows.drop(
                    columns=["step", "revision", "tokens_seen"], errors="ignore",
                ).to_dict("records")

        def _on_subject_done(subj_rows: list[dict], ckpt=ckpt) -> None:
            # Persist each subject's rows immediately (before the next subject
            # is fit) so a crash mid-checkpoint loses at most the in-progress
            # subject, not every subject fit so far this checkpoint.
            base = {"step": ckpt.step, "revision": ckpt.revision, "tokens_seen": ckpt.tokens_seen}
            for row in subj_rows:
                append_row(by_subject_csv, {**base, **row})

        try:
            result = UclReadingEmbedBenchmark(bench_cfg).run(
                existing_subject_rows=existing_subject_rows,
                on_subject_done=_on_subject_done,
            )
            # Per-subject rows were already written incrementally above; only
            # the mean-across-subjects summary is written here.
            if not already_main:
                new_rows = _ckpt_rows(ckpt, result.summary_df)
                for row in new_rows:
                    append_row(scores_csv, row)
                rows.extend(new_rows)
                existing = load_existing_csv(scores_csv)

            if average_scores_csv is not None and not already_avg and result.average_summary_df is not None:
                for row in _ckpt_rows(ckpt, result.average_summary_df):
                    append_row(average_scores_csv, row)
        except Exception:
            logger.error("[%s] UCL Reading embed failed at %s", model_id, ckpt.revision, exc_info=True)
        finally:
            clear_hf_model_cache()
            free_gpu()

        if perplexity_dir is not None:
            try:
                save_checkpoint_perplexity(
                    model_id=load_model_id,
                    revision=ckpt.revision,
                    label=(model_cfg or {}).get("label", model_id),
                    n_params=n_params,
                    family=family,
                    device=device,
                    text=_ucl_reading_corpus_text(data_root),
                    perplexity_dir=perplexity_dir,
                    dataset_label="ucl_reading",
                    trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
                )
            except Exception:
                logger.error("[%s] perplexity save failed at %s", model_id, ckpt.revision, exc_info=True)

    df = pd.DataFrame(rows)
    if not df.empty and not no_plots:
        _plot_training_curve(df, output_dir, model_id)
        _plot_layerwise_heatmap_one_model(df, output_dir, model_id)
    return df


def _plot_training_curve(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if "step" not in df.columns:
        return
    best = df.loc[df.groupby("step")["spearmanr"].idxmax()].sort_values("step")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, ylab in [(axes[0], "pearsonr", "Pearson r"), (axes[1], "spearmanr", "Spearman r")]:
        ax.plot(best["step"], best[col], "o-")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylab)
        ax.set_title(f"Best-layer {ylab}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"UCL Reading embed — {model_id}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "ucl_reading_embed_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layerwise_heatmap_one_model(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if "step" not in df.columns or "layer" not in df.columns:
        return
    steps = sorted(df["step"].unique())
    n_layers = int(df["layer"].max()) + 1
    mat = np.full((len(steps), n_layers), np.nan)
    for i, step in enumerate(steps):
        sub = df[df["step"] == step]
        for _, r in sub.iterrows():
            mat[i, int(r["layer"])] = r["spearmanr"]
    plot_layerwise_heatmap(
        mat, [f"step {s}" for s in steps],
        metric="Spearman r", title=f"UCL Reading embed layerwise — {model_id}",
        save_path=str(output_dir / "ucl_reading_embed_layerwise_heatmap.png"),
    )
    plt.close("all")


def run_summary(
    sweep_dir: Path,
    *,
    summary_out_dir: Path | None = None,
    family: str | None = None,
    registry_family: str | None = None,
    models_filter: list[str] | None = None,
) -> None:
    sweep_dir = sweep_dir.resolve()
    summary_root = (summary_out_dir or sweep_dir).resolve()
    search_root = sweep_dir / family if family else sweep_dir
    summary_dir = (summary_root / family / "summary") if family else (summary_root / "summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for sub in _discover_model_score_dirs(search_root):
        csv_path = sub / SCORES_FILENAME
        model_id = _model_id_from_score_dir(sub)
        fam = _family_from_score_dir(sub, search_root)
        if models_filter and model_id not in models_filter:
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "model_id", model_id)
        if fam:
            df.insert(0, "family", fam)
        all_rows.append(df)

    if not all_rows:
        logger.warning("No %s found under %s", SCORES_FILENAME, search_root)
        return

    combined = pd.concat(all_rows, ignore_index=True)
    reg_fam = registry_family or family
    families = [reg_fam] if reg_fam else sorted(combined.get("family", pd.Series(dtype=str)).dropna().unique())
    n_params_map: dict[str, float] = {}
    for fam in families:
        try:
            n_params_map.update({m["model_id"]: m["n_params"] for m in registry_model_configs(fam)})
        except KeyError:
            continue
    if n_params_map:
        combined["n_params"] = combined["model_id"].map(n_params_map)
    combined.to_csv(summary_dir / "all_models_ucl_reading_embed_scores.csv", index=False)
    logger.info("UCL Reading embed summary CSV written to %s", summary_dir)

    final = combined.loc[combined.groupby("model_id")["step"].idxmax()]
    model_ids = sorted(combined["model_id"].unique())
    n_layers = int(combined["layer"].max()) + 1
    mat = np.full((len(model_ids), n_layers), np.nan)
    for i, model_id in enumerate(model_ids):
        sub = combined[(combined["model_id"] == model_id) & (combined["step"] == final.set_index("model_id").loc[model_id, "step"])]
        for _, r in sub.iterrows():
            mat[i, int(r["layer"])] = r["spearmanr"]
    plot_layerwise_heatmap(
        mat, model_ids, metric="Spearman r",
        title="UCL Reading embed layerwise (final checkpoint)",
        save_path=str(summary_dir / "ucl_reading_embed_layerwise_heatmaps_grid.png"),
    )
    plt.close("all")

    if "n_params" in final.columns:
        best_per_model = (
            combined.loc[combined.groupby(["model_id", "step"])["spearmanr"].idxmax()]
            .loc[lambda d: d.groupby("model_id")["step"].idxmax()]
        )
        if not best_per_model.empty:
            plot_scaling_curve(
                best_per_model, x="n_params", y="spearmanr",
                hue="family" if "family" in best_per_model.columns else "model_id",
                title="UCL Reading embed scaling law (final checkpoint, best layer)",
                save_path=str(summary_dir / "scaling_law_final_checkpoint.png"),
            )
            plt.close("all")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/embedding/behavioral/ucl_reading.yaml")
    p.add_argument("--out_dir", default=None, help="Sweep output dir (for running checkpoints)")
    p.add_argument(
        "--sweep_dir",
        default=os.environ.get("UCL_READING_EMBED_SWEEP_DIR", "outputs/ucl_reading/embedding_sweep"),
        help="Read sweep CSVs from here when using --summary_only",
    )
    p.add_argument("--summary_out_dir", default=os.environ.get("UCL_READING_EMBED_SUMMARY_DIR") or None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument(
        "--output_family", default=None,
        help="Override the output subfolder name (defaults to --family). Use e.g. "
             "\'pythia_checkpoints\' to keep a full checkpoint sweep separate from the "
             "regular \'pythia\' folder.",
    )
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--perplexity_dir",
        default=os.environ.get("UCL_READING_PERPLEXITY_DIR", "outputs_jeanzay/ucl_reading/perplexity"),
        help="Save each checkpoint's corpus perplexity here too (merges into the "
             "existing ucl_reading_perplexity_summary.csv tree). Set to '' to disable.",
    )
    p.add_argument(
        "--average_across_subjects", action="store_true",
        help="Additionally fit a single ridge CV per layer on RT averaged across "
             "subjects per item (available-case mean; see neuroai.benchmarks."
             "subject_average). Written alongside the per-subject results to a "
             "sibling '<out_dir>_average' directory; does not change the "
             "existing per-subject outputs. Can also be set via the config's "
             "top-level 'average_across_subjects: true'.",
    )
    p.add_argument(
        "--average_only", action="store_true",
        help="Skip the per-subject ridge fits entirely and only compute the "
             "averaged-RT branch (implies --average_across_subjects) — for "
             "models where per-subject fitting alone is too slow to finish "
             "(e.g. opt-66b). Can also be set via the config's top-level "
             "'average_only: true'.",
    )
    args = p.parse_args()
    perplexity_dir = args.perplexity_dir or None
    output_family = args.output_family or args.family

    cfg = load_config(args.config)
    if args.average_across_subjects:
        cfg["average_across_subjects"] = True
    if args.average_only:
        cfg["average_across_subjects"] = True
        cfg["average_only"] = True
    default_save_dir = "outputs/ucl_reading/embedding_sweep"
    out_dir = Path(args.out_dir or cfg.get("output", {}).get("save_dir", default_save_dir))
    average_out_dir = out_dir.with_name(out_dir.name + "_average") if cfg.get("average_across_subjects") else None

    cache_raw = (cfg.get("cache") or {}).get("features_dir")
    feature_cache_dir = Path(os.path.expandvars(str(cache_raw))) if cache_raw else None
    if feature_cache_dir:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summary_out = Path(args.summary_out_dir or out_dir)
        run_summary(
            Path(args.sweep_dir), summary_out_dir=summary_out,
            family=output_family, registry_family=args.family, models_filter=args.models,
        )
        return

    if args.from_registry:
        if not args.family:
            p.error("--from-registry requires --family")

        if args.array_index is not None:
            task = resolve_family_task(args.family, args.array_index)
            model_cfgs = [m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id]
            checkpoints = _checkpoint_items_from_task(task)
            if args.dry_run:
                print(f"task: {task}")
                return
            run_model_sweep(
                checkpoints,
                model_id=task.model_id,
                n_params=model_cfgs[0]["n_params"],
                batch_size=model_cfgs[0]["batch_size"],
                cfg=cfg,
                output_dir=_model_output_dir(out_dir, task.model_id, family=output_family, model_cfg=model_cfgs[0]),
                average_output_dir=(
                    _model_output_dir(average_out_dir, task.model_id, family=output_family, model_cfg=model_cfgs[0])
                    if average_out_dir is not None else None
                ),
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                no_plots=args.no_plots,
                model_cfg=model_cfgs[0],
                family=args.family,
                perplexity_dir=perplexity_dir,
            )
            return

        all_models = registry_model_configs(args.family)
        if args.models:
            all_models = [m for m in all_models if m["model_id"] in args.models]
        if args.dry_run:
            for m in all_models:
                n = len(list(build_checkpoint_items(m)))
                print(f"{m['model_id']}: {n} checkpoints")
            return

        for m in all_models:
            ckpts = [CheckpointItem(rev, step, tok) for rev, step, tok in build_checkpoint_items(m)]
            run_model_sweep(
                ckpts,
                model_id=m["model_id"],
                n_params=m["n_params"],
                batch_size=m["batch_size"],
                cfg=cfg,
                output_dir=_model_output_dir(out_dir, m["model_id"], family=output_family, model_cfg=m),
                average_output_dir=(
                    _model_output_dir(average_out_dir, m["model_id"], family=output_family, model_cfg=m)
                    if average_out_dir is not None else None
                ),
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                no_plots=args.no_plots,
                model_cfg=m,
                family=args.family,
                perplexity_dir=perplexity_dir,
            )

        if not args.skip_summary:
            summary_out = Path(args.summary_out_dir or out_dir)
            run_summary(out_dir, summary_out_dir=summary_out, family=output_family, registry_family=args.family, models_filter=args.models)
        return

    p.error("Specify --from-registry with --family (and optional --array-index)")


if __name__ == "__main__":
    main()
