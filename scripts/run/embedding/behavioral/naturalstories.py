#!/usr/bin/env python3
"""Natural Stories reading-time checkpoint sweep — GPU/himalaya ridge.

Ridge
regression runs on GPU via himalaya (one CUDA stream per layer), not
sklearn on CPU — see ``neuroai/benchmarks/natural_stories_embed_gpu.py`` for
the full rationale. Single stage, no CPU/prefetch split (embedding
extraction and regression both run in the same GPU job).

Chosen ridge alpha is logged per layer in the job output — read the log to
pick the best alpha. Word-level out-of-fold predictions are saved for every
layer by default (save_word_predictions).

Output structure
----------------
<output_dir>/
  <family>/                    # e.g. gpt2, pythia, mamba (requires --family)
    <model_short>/
      spr_checkpoint_scores.csv
      spr_checkpoint_layerwise.csv
      spr_training_curve.png
      spr_layerwise_heatmap.png
      naturalstories_embed_word_predictions/
        <model>_<revision>.npz
    summary/
      all_models_spr_scores.csv
      all_models_spr_training_curves.png

Examples
--------
# Single array task (Jean Zay):
python scripts/run/embedding/behavioral/naturalstories.py \\
    --config configs/embedding/behavioral/naturalstories.yaml \\
    --from-registry --family pythia --array-index 0

# Full family locally:
python scripts/run/embedding/behavioral/naturalstories.py \\
    --config configs/embedding/behavioral/naturalstories.yaml \\
    --from-registry --family gpt2

# Merge summary after all SLURM jobs:
python scripts/run/embedding/behavioral/naturalstories.py \\
    --config configs/embedding/behavioral/naturalstories.yaml \\
    --summary_only --out_dir outputs/naturalstories/embedding_sweep
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
from neuroai.benchmarks.natural_stories_embed_gpu import (
    NaturalStoriesEmbedGpuBenchmark,
    NaturalStoriesEmbedGpuBenchmarkConfig,
    NaturalStoriesEmbedGpuDataConfig,
    natural_stories_embed_gpu_ridge_alphas,
)
from neuroai.features.llm import clear_hf_model_cache
from neuroai.models.registry import (
    build_checkpoint_items,
    registry_model_configs,
    resolve_family_task,
    resolve_load_model_id,
)
from neuroai.sweep import (
    CheckpointItem,
    append_row,
    already_computed,
    free_gpu,
    load_existing_csv,
    safe_model_dir,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

SCORES_FILENAME = "spr_checkpoint_scores.csv"
LAYERWISE_FILENAME = "spr_checkpoint_layerwise.csv"


def load_config(config_path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)


def _resolve_alphas(raw: Any) -> list[float]:
    if raw in (None, "null", []):
        return natural_stories_embed_gpu_ridge_alphas()
    return [float(a) for a in raw]


def _model_output_dir(
    output_dir: Path,
    model_id: str,
    *,
    family: str | None = None,
    model_cfg: dict | None = None,
) -> Path:
    fam = family
    if model_cfg is not None:
        fam = model_cfg.get("registry_family") or model_cfg.get("family") or fam
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
    revision: str | None,
    n_params: float | None,
    batch_size: int,
    output_dir: Path,
    feature_cache_dir: Path | None,
    load_model_id: str | None = None,
    trust_remote_code: bool = False,
) -> NaturalStoriesEmbedGpuBenchmarkConfig:
    ds = cfg["dataset"]
    reg = cfg.get("regression", {})
    bench = cfg.get("benchmark", {})

    hf_model_id = load_model_id or model_id
    return NaturalStoriesEmbedGpuBenchmarkConfig(
        data=NaturalStoriesEmbedGpuDataConfig(
            dataset_root=ds["root"],
            stories=ds.get("stories"),
            min_correct=int(ds.get("min_correct", 0)),
            context_words=int(ds.get("context_words", 48)),
            max_length=int(ds.get("max_length", 512)),
            log_rt=bool(ds.get("log_rt", True)),
            feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
        ),
        model=ModelConfig(
            model_name=hf_model_id,
            model_revision=revision if revision not in (None, "", "main") else None,
            device="cuda",
            batch_size=batch_size,
            n_params=n_params,
            trust_remote_code=trust_remote_code,
        ),
        regression=RegressionConfig(
            n_splits=int(reg.get("n_splits", 10)),
            alpha=_resolve_alphas(reg.get("alpha")),
            scale=bool(reg.get("scale", True)),
            random_state=int(reg.get("random_state", 0)),
        ),
        output_dir=str(output_dir),
        layerwise=bool(bench.get("layerwise", True)),
        tokenizer_no_revision=bool(bench.get("tokenizer_no_revision", True)),
        n_layer_workers=int(bench.get("n_layer_workers", 8)),
        save_word_predictions=bool(bench.get("save_word_predictions", True)),
        canonical_model_name=model_id,
    )


def _score_row(ckpt: CheckpointItem, result) -> dict[str, Any]:
    lw_pearson = result.layerwise.scores_array(metric="pearsonr")
    lw_spearman = result.layerwise.scores_array(metric="spearmanr")
    best_li_pearson = int(np.nanargmax(lw_pearson))
    best_li_spearman = int(np.nanargmax(lw_spearman))
    return {
        "step": ckpt.step,
        "revision": ckpt.revision,
        "tokens_seen": ckpt.tokens_seen,
        "n_words": result.n_words,
        "flat_pearsonr": result.flat_score.pearsonr,
        "flat_spearmanr": result.flat_score.spearmanr,
        "best_pearsonr": float(lw_pearson[best_li_pearson]),
        "best_spearmanr": float(lw_spearman[best_li_spearman]),
        "best_layer_pearson": best_li_pearson,
        "best_layer_spearman": best_li_spearman,
    }


def run_activations_only(
    checkpoints: list[CheckpointItem],
    *,
    model_id: str,
    n_params: float,
    batch_size: int,
    cfg: dict,
    feature_cache_dir: Path | None,
    model_cfg: dict | None = None,
) -> None:
    """Extract and cache LLM word embeddings for every checkpoint, skipping
    the ridge fit entirely -- mirrors podcast.py/broderick.py's
    run_activations_only. NaturalStoriesEmbedGpuBenchmark's cache only
    depends on word text (via _load_data/_extract_embeddings), not on the
    ridge fit that follows."""
    if feature_cache_dir is None:
        raise ValueError(
            "--activations_only requires a feature cache directory. "
            "Set cache.features_dir in the YAML or pass --feature_cache_dir."
        )
    for ckpt in checkpoints:
        logger.info("\n[%s] activations ── %s (step=%d) ──", model_id, ckpt.revision, ckpt.step)
        load_model_id = resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        bench_cfg = _benchmark_config(
            cfg,
            model_id=model_id,
            load_model_id=load_model_id,
            revision=ckpt.revision,
            n_params=n_params,
            batch_size=batch_size,
            output_dir=feature_cache_dir,  # unused for activations, just needs a Path
            feature_cache_dir=feature_cache_dir,
            trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
        )
        encoder = NaturalStoriesEmbedGpuBenchmark(bench_cfg)
        try:
            df = encoder._load_data()
            emb = encoder._extract_embeddings(df)
            logger.info("[%s] %s — done, shape %s", model_id, ckpt.revision,
                        "×".join(str(d) for d in emb.shape))
        except Exception:
            logger.error("[%s] %s — failed", model_id, ckpt.revision, exc_info=True)
        finally:
            clear_hf_model_cache()
            free_gpu()


def run_model_sweep(
    checkpoints: list[CheckpointItem],
    *,
    model_id: str,
    n_params: float,
    batch_size: int,
    cfg: dict,
    output_dir: Path,
    feature_cache_dir: Path | None,
    no_plots: bool = False,
    model_cfg: dict | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / SCORES_FILENAME
    layer_csv = output_dir / LAYERWISE_FILENAME

    existing = load_existing_csv(scores_csv)
    rows = existing.to_dict("records") if not existing.empty else []
    layer_rows: list[dict] = []
    if layer_csv.exists():
        try:
            layer_rows = pd.read_csv(layer_csv).to_dict("records")
        except Exception:
            layer_rows = []

    for ckpt in checkpoints:
        if already_computed(existing, ckpt.revision):
            logger.info("[%s] %s already done — skipping", model_id, ckpt.revision)
            continue

        logger.info(
            "\n[%s] Natural Stories (GPU embed) ── %s (step=%d, tokens≈%s) ──",
            model_id, ckpt.revision, ckpt.step, ckpt.tokens_seen,
        )

        load_model_id = resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        bench_cfg = _benchmark_config(
            cfg,
            model_id=model_id,
            load_model_id=load_model_id,
            revision=ckpt.revision,
            n_params=n_params,
            batch_size=batch_size,
            output_dir=output_dir,
            feature_cache_dir=feature_cache_dir,
            trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
        )

        try:
            result = NaturalStoriesEmbedGpuBenchmark(bench_cfg).run()
            row = _score_row(ckpt, result)
            append_row(scores_csv, row)
            rows.append(row)
            existing = load_existing_csv(scores_csv)

            for li, s in zip(result.layerwise.layer_indices, result.layerwise.layer_scores):
                layer_rows.append({
                    "step": ckpt.step,
                    "revision": ckpt.revision,
                    "tokens_seen": ckpt.tokens_seen,
                    "layer": li,
                    "layer_rel": li / max(1, result.n_layers - 1),
                    **s.to_dict(),
                })
            pd.DataFrame(layer_rows).to_csv(layer_csv, index=False)
        except Exception:
            logger.error("[%s] failed at %s", model_id, ckpt.revision, exc_info=True)
        finally:
            clear_hf_model_cache()
            free_gpu()

    df = pd.DataFrame(rows)
    if not df.empty and not no_plots:
        _plot_training_curve(df, output_dir, model_id)
        if layer_csv.exists():
            try:
                _plot_layer_heatmap(pd.read_csv(layer_csv), output_dir, model_id)
            except Exception:
                logger.warning("[%s] layer heatmap skipped", model_id, exc_info=True)
    return df


def _plot_training_curve(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if "step" not in df.columns:
        return
    plot_df = df.sort_values("step")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(plot_df["step"], plot_df["best_pearsonr"], "o-")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Natural Stories SPR (GPU embed) — {model_id}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "spr_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_heatmap(layer_df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if layer_df.empty:
        return
    pivot = layer_df.pivot_table(index="layer", columns="step", values="pearsonr")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer")
    ax.set_title(f"Natural Stories activations (GPU embed) — {model_id}")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_dir / "spr_layerwise_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_summary(
    output_dir: Path,
    *,
    family: str | None = None,
    models_filter: list[str] | None = None,
) -> None:
    search_root = output_dir / family if family else output_dir
    summary_dir = (output_dir / family / "summary") if family else (output_dir / "summary")
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
    combined.to_csv(summary_dir / "all_models_spr_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    title_family = f" ({family})" if family else ""
    for model_id, grp in combined.groupby("model_id"):
        g = grp.sort_values("step")
        ax.plot(g["step"], g["best_pearsonr"], "o-", label=model_id)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Natural Stories SPR (GPU embed) — all models{title_family}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(summary_dir / "all_models_spr_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary written to %s", summary_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/embedding/behavioral/naturalstories.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--activations_only", action="store_true",
                   help="Stage 1: extract and cache LLM word embeddings only (no ridge fit). "
                        "Run on GPU; follow with the full sweep once cache is warm.")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/naturalstories/embedding_sweep"))

    cache_raw = (cfg.get("cache") or {}).get("features_dir")
    feature_cache_dir = Path(os.path.expandvars(str(cache_raw))).resolve() if cache_raw else None
    if feature_cache_dir:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        run_summary(out_dir, family=args.family, models_filter=args.models)
        return

    if args.from_registry:
        if not args.family:
            p.error("--from-registry requires --family (e.g. gpt2, pythia, mamba)")

        if args.array_index is not None:
            task = resolve_family_task(args.family, args.array_index)
            model_cfgs = [m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id]
            checkpoints = _checkpoint_items_from_task(task)
            if args.dry_run:
                print(f"task: {task}")
                return
            if args.activations_only:
                run_activations_only(
                    checkpoints,
                    model_id=task.model_id,
                    n_params=model_cfgs[0]["n_params"],
                    batch_size=model_cfgs[0]["batch_size"],
                    cfg=cfg,
                    feature_cache_dir=feature_cache_dir,
                    model_cfg=model_cfgs[0],
                )
                return
            run_model_sweep(
                checkpoints,
                model_id=task.model_id,
                n_params=model_cfgs[0]["n_params"],
                batch_size=model_cfgs[0]["batch_size"],
                cfg=cfg,
                output_dir=_model_output_dir(out_dir, task.model_id, family=args.family, model_cfg=model_cfgs[0]),
                feature_cache_dir=feature_cache_dir,
                no_plots=args.no_plots,
                model_cfg=model_cfgs[0],
            )
            return

        all_models = registry_model_configs(args.family)
        if args.models:
            all_models = [m for m in all_models if m["model_id"] in args.models]
        if args.dry_run:
            for m in all_models:
                ckpts = [CheckpointItem(rev, step, tok) for rev, step, tok in build_checkpoint_items(m)]
                print(f"{m['model_id']}: {len(ckpts)} checkpoints")
            return

        for m in all_models:
            ckpts = [CheckpointItem(rev, step, tok) for rev, step, tok in build_checkpoint_items(m)]
            run_model_sweep(
                ckpts,
                model_id=m["model_id"],
                n_params=m["n_params"],
                batch_size=m["batch_size"],
                cfg=cfg,
                output_dir=_model_output_dir(out_dir, m["model_id"], model_cfg=m),
                feature_cache_dir=feature_cache_dir,
                no_plots=args.no_plots,
                model_cfg=m,
            )

        if not args.skip_summary:
            run_summary(out_dir, family=args.family, models_filter=args.models)
        return

    p.error("Specify --from-registry with --family (and optional --array-index)")


if __name__ == "__main__":
    main()
