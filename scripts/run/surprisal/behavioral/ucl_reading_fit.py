#!/usr/bin/env python3
"""UCL self-paced reading checkpoint sweep with trial-level LME (pymer4).

Uses the same model registry/checkpoints as other sweeps but fits a by-subject
mixed-effects model (raw RT ~ surprisal + nuisances) on the Frank et al. (2013)
broad-coverage self-paced reading corpus (``data/ucl_reading``).

Output structure
----------------
<output_dir>/
  <family>/
    <model_short>/
      ucl_reading_spr_lme_checkpoint_scores.csv
    summary/
      all_models_ucl_reading_spr_lme_scores.csv

Examples
--------
python scripts/run/surprisal/behavioral/ucl_reading_fit.py \\
    --config configs/surprisal/behavioral/ucl_reading.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda

python scripts/run/surprisal/behavioral/ucl_reading_fit.py \\
    --config configs/surprisal/behavioral/ucl_reading.yaml \\
    --summary_only --out_dir outputs/ucl_reading/surprisal_sweep
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
import pandas as pd
import yaml

from neuroai.benchmarks.base import ModelConfig
from neuroai.benchmarks.ucl_reading_lme import (
    UCLReadingDataConfig,
    UCLReadingLmeBenchmark,
    UCLReadingLmeBenchmarkConfig,
    UCLReadingLmeConfig,
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
    free_gpu,
    load_existing_csv,
    safe_model_dir,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

SCORES_FILENAME = "ucl_reading_spr_lme_checkpoint_scores.csv"


def load_config(config_path: str | Path) -> dict[str, Any]:
    repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    os.environ.setdefault("BENCHMARK_ROOT", str(repo_root))
    scratch = os.environ.get("SCRATCH", "").strip()
    os.environ.setdefault(
        "BENCHMARK_CACHE_ROOT",
        f"{scratch}/.cache" if scratch else f"{os.environ['BENCHMARK_ROOT']}/.cache",
    )
    os.environ.setdefault(
        "UCL_READING_FEATURE_CACHE_DIR",
        f"{os.environ['BENCHMARK_CACHE_ROOT']}/ucl_reading_features",
    )
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    cfg = yaml.safe_load(raw)

    cache = cfg.setdefault("cache", {})
    feat_dir = os.environ.get("UCL_READING_FEATURE_CACHE_DIR", "").strip()
    if feat_dir:
        cache["features_dir"] = feat_dir

    return cfg


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
    found: list[Path] = []
    for scores in sorted(sweep_dir.glob(f"**/{SCORES_FILENAME}")):
        found.append(scores.parent)
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
    output_dir: Path,
    feature_cache_dir: Path | None,
    load_model_id: str | None = None,
    trust_remote_code: bool = False,
    n_workers: int = 10,
    layer_stride: int = 1,
    last_layer_only: bool = False,
    save_word_predictions: bool = True,
    paper_replication: bool = False,
) -> UCLReadingLmeBenchmarkConfig:
    ds = cfg["dataset"]
    lme = cfg.get("lme", {})
    cache = cfg.get("cache", {})

    hf_model_id = load_model_id or model_id
    return UCLReadingLmeBenchmarkConfig(
        data=UCLReadingDataConfig(
            dataset_root=ds["root"],
            sentences=ds.get("sentences"),
            min_accuracy=float(ds.get("min_accuracy", 0.0)),
            min_rt=float(ds.get("min_rt", 100.0)),
            max_rt=float(ds.get("max_rt", 3000.0)),
            feature_cache_dir=str(feature_cache_dir or cache.get("features_dir", "") or ""),
        ),
        model=ModelConfig(
            model_name=hf_model_id,
            model_revision=revision if revision not in ("", "main") else None,
            device=device,
            batch_size=batch_size,
            n_params=n_params,
            trust_remote_code=trust_remote_code,
        ),
        lme=UCLReadingLmeConfig(
            include_surprisal=bool(lme.get("include_surprisal", True)),
            include_prev_surprisal=bool(lme.get("include_prev_surprisal", True)),
            include_freq=bool(lme.get("include_freq", True)),
            filter_sentence_edges=bool(lme.get("filter_sentence_edges", True)),
            compare_baseline=bool(lme.get("compare_baseline", True)),
            out_of_sample=bool(lme.get("out_of_sample", True)),
            partition_mod=int(lme.get("partition_mod", 4)),
            partition_fit_indices=tuple(int(i) for i in lme.get("partition_fit_indices", [0, 1])),
            partition_test_indices=tuple(int(i) for i in lme.get("partition_test_indices", [2, 3])),
            verbose=bool(lme.get("verbose", False)),
            # CPU sweep runs are the Stage-2 LME-only half of the two-stage Jean
            # Zay pipeline — they must never need to load a model. A cache miss
            # there should fail fast, not silently fall back to a slow (and, for
            # trust_remote_code models depending on Triton, GPU-driver-probing
            # and therefore crashing) CPU forward pass.
            require_cached_surprisal=bool(lme.get("require_cached_surprisal", device == "cpu")),
            layer_stride=layer_stride,
            last_layer_only=last_layer_only,
            n_workers=n_workers,
            save_word_predictions=save_word_predictions,
            paper_replication=paper_replication,
        ),
        output_dir=str(output_dir),
        canonical_model_name=model_id,
    )


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
    n_workers: int = 10,
    layer_stride: int = 1,
    last_layer_only: bool = False,
    save_word_predictions: bool = True,
    paper_replication: bool = False,
) -> pd.DataFrame:
    """Fit every layer for every checkpoint.

    No outer "is this checkpoint done" check: ``UCLReadingLmeBenchmark.run()``
    already skips layers it finds completed for a given revision, and a cache
    hit in the per-layer surprisal extraction never touches the model — so
    re-running a fully-done checkpoint here is a cheap no-op.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / SCORES_FILENAME

    for ckpt in checkpoints:
        logger.info("\n[%s] UCL Reading LME ── %s (step=%d) ──", model_id, ckpt.revision, ckpt.step)

        load_model_id = resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        bench_cfg = _benchmark_config(
            cfg,
            model_id=model_id,
            load_model_id=load_model_id,
            revision=ckpt.revision,
            n_params=n_params,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            feature_cache_dir=feature_cache_dir,
            trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
            n_workers=n_workers,
            layer_stride=layer_stride,
            last_layer_only=last_layer_only,
            save_word_predictions=save_word_predictions,
            paper_replication=paper_replication,
        )

        try:
            UCLReadingLmeBenchmark(bench_cfg).run(
                extra_fields={"step": ckpt.step, "tokens_seen": ckpt.tokens_seen},
            )
        except Exception:
            logger.error("[%s] LME failed at %s", model_id, ckpt.revision, exc_info=True)
        finally:
            clear_hf_model_cache()
            free_gpu()

    df = load_existing_csv(scores_csv)
    if not df.empty and not no_plots:
        _plot_training_curve(df, output_dir, model_id)
    return df


def _plot_training_curve(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    """Training curve for the DEEPEST fitted layer (closest analog to the old
    final-layer-only metric)."""
    if "step" not in df.columns or "layer" not in df.columns or df.empty:
        return
    deepest_layer = df["layer"].max()
    plot_df = df[(df["layer"] == deepest_layer) & df["error"].isna()].sort_values("step")
    if plot_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, label, ylab in [
        (axes[0], "beta", "Surprisal β", "Fixed-effect β"),
        (axes[1], "delta_loglik_test", "ΔLogLik (test)", "ΔLogLik"),
    ]:
        if col not in plot_df.columns:
            ax.set_visible(False)
            continue
        ax.plot(plot_df["step"], plot_df[col], "o-", label=label)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylab)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"UCL Reading LME — {model_id} (layer {deepest_layer})", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "ucl_reading_spr_lme_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_summary(
    sweep_dir: Path,
    *,
    summary_out_dir: Path | None = None,
    family: str | None = None,
    models_filter: list[str] | None = None,
    perplexity_dir: Path | str | None = None,
) -> None:
    """Write combined CSV and scaling / training plots for finished LME sweeps."""
    _repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from scripts.summarize.ucl_reading_lme_families import DEFAULT_SUMMARY_OUT_DIR, run_lme_summary

    sweep_dir = sweep_dir.resolve()
    summary_root = (summary_out_dir or DEFAULT_SUMMARY_OUT_DIR).resolve()
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
    combined.to_csv(summary_dir / "all_models_ucl_reading_spr_lme_scores.csv", index=False)
    logger.info("UCL Reading LME summary CSV written to %s", summary_dir)

    families = [family] if family else None
    run_lme_summary(sweep_dir, summary_root, families=families, perplexity_dir=perplexity_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/surprisal/behavioral/ucl_reading.yaml")
    p.add_argument("--out_dir", default=None, help="Sweep output dir (for running checkpoints)")
    p.add_argument(
        "--sweep_dir",
        default=os.environ.get("UCL_READING_LME_SWEEP_DIR", "outputs/ucl_reading/surprisal_sweep"),
        help="Read sweep CSVs from here when using --summary_only",
    )
    p.add_argument("--summary_out_dir", default=os.environ.get("UCL_READING_LME_SUMMARY_DIR") or None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--n_workers", type=int, default=10,
                   help="Concurrent worker processes fitting layers (default 10)")
    p.add_argument("--layer_stride", type=int, default=1,
                   help="Fit every Nth layer (default 1 = all layers)")
    p.add_argument("--last_layer_only", action="store_true",
                   help="Fit only the deepest layer, ignoring --layer_stride. "
                        "Cuts the CPU LME-fitting stage to 1 layer per checkpoint.")
    p.add_argument("--save_word_predictions", action=argparse.BooleanOptionalAction, default=True,
                   help="Save each held-out word's true value + baseline/full predictions "
                        "and per-word delta log-likelihood to ucl_reading_spr_lme_per_word.csv. "
                        "On by default; pass --no-save_word_predictions to disable.")
    p.add_argument("--paper_replication", action="store_true",
                   help="Use Kuribayashi et al. 2025's exact UCL/SPR baseline+full predictor set "
                        "(arXiv:2502.01615) instead of this pipeline's default -- adds lag-1/lag-2 "
                        "word length+frequency to the baseline and moves lag-1/lag-2 surprisal into "
                        "the baseline too, so the delta isolates only the current word's surprisal. "
                        "Off by default (this pipeline's own convention).")
    p.add_argument(
        "--perplexity_dir",
        default=os.environ.get("UCL_READING_PERPLEXITY_DIR", "outputs_jeanzay/ucl_reading/perplexity"),
        help="Perplexity JSON/CSV root for scaling-vs-PPL plots",
    )
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/ucl_reading/surprisal_sweep"))

    cache_raw = (cfg.get("cache") or {}).get("features_dir")
    # NOTE: deliberately not .resolve()'d — see the matching comment in
    # run/surprisal/behavioral/zuco_reading_fit.py. Must stay byte-for-byte identical to the raw
    # path run/surprisal/behavioral/ucl_reading_extract.py used to write the cache.
    feature_cache_dir = Path(os.path.expandvars(str(cache_raw))) if cache_raw else None
    if feature_cache_dir:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summary_out = Path(args.summary_out_dir or "outputs/ucl_reading/surprisal_sweep")
        run_summary(
            Path(args.sweep_dir),
            summary_out_dir=summary_out,
            family=args.family,
            models_filter=args.models,
            perplexity_dir=args.perplexity_dir,
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
                output_dir=_model_output_dir(out_dir, task.model_id, family=args.family, model_cfg=model_cfgs[0]),
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                no_plots=args.no_plots,
                model_cfg=model_cfgs[0],
                n_workers=args.n_workers,
                layer_stride=args.layer_stride,
                last_layer_only=args.last_layer_only,
                save_word_predictions=args.save_word_predictions,
                paper_replication=args.paper_replication,
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
                output_dir=_model_output_dir(out_dir, m["model_id"], model_cfg=m),
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                no_plots=args.no_plots,
                model_cfg=m,
                n_workers=args.n_workers,
                layer_stride=args.layer_stride,
                last_layer_only=args.last_layer_only,
                save_word_predictions=args.save_word_predictions,
                paper_replication=args.paper_replication,
            )

        if not args.skip_summary:
            summary_out = Path(args.summary_out_dir or out_dir)
            run_summary(
                out_dir,
                summary_out_dir=summary_out,
                family=args.family,
                models_filter=args.models,
                perplexity_dir=args.perplexity_dir,
            )
        return

    p.error("Specify --from-registry with --family (and optional --array-index)")


if __name__ == "__main__":
    main()
