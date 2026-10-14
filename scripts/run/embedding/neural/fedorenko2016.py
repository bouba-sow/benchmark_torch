#!/usr/bin/env python3
"""Fedorenko2016 iEEG encoding checkpoint sweep.

Runs layerwise word-level encoding (scalar electrode responses) across registry
model families/checkpoints with grouped sentence-level CV.

Output structure
----------------
<output_dir>/
  <family>/
    <model_short>/
      fedorenko2016_checkpoint_scores.csv
      fedorenko2016_checkpoint_layerwise.csv
      fedorenko2016_by_electrode.csv
      fedorenko2016_by_subject.csv
      fedorenko2016_sentence_ppl.csv
      perplexity.json
      fedorenko2016_training_curve.png
      fedorenko2016_layer_heatmap.png
    summary/
      all_models_fedorenko2016_scores.csv
      all_models_fedorenko2016_training_curves.png
"""

from __future__ import annotations

import argparse
import json
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
from neuroai.benchmarks.fedorenko2016 import (
    Fedorenko2016BenchmarkConfig,
    Fedorenko2016DataConfig,
    Fedorenko2016Encoder,
    Fedorenko2016Result,
)
from neuroai.datasets.fedorenko2016 import Fedorenko2016Dataset
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


def load_config(config_path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)


def _resolve_alphas(raw: Any) -> list[float]:
    if raw in (None, "null", []):
        return np.logspace(3.5, 8, 5).tolist()
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


def _build_benchmark_config(
    cfg: dict,
    *,
    model_id: str,
    revision: str,
    n_params: float,
    batch_size: int,
    device: str,
    feature_cache_dir: Path | None,
    n_jobs: int,
    n_layer_workers: int,
    regression_device: str,
    load_model_id: str | None = None,
    trust_remote_code: bool = False,
    output_dir: Path | None = None,
) -> Fedorenko2016BenchmarkConfig:
    ds = cfg["dataset"]
    reg = cfg.get("regression", {})
    bench = cfg.get("benchmark", {})

    data_cfg = Fedorenko2016DataConfig(
        root=ds["root"],
        subjects=ds.get("subjects") or None,
        electrodes=ds.get("electrodes") or None,
        context_words=int(ds["context_words"]) if ds.get("context_words") is not None else None,
        max_length=int(ds["max_length"]) if ds.get("max_length") is not None else None,
        zscore_targets=bool(ds.get("zscore_targets", True)),
        drop_constant_electrodes=bool(ds.get("drop_constant_electrodes", True)),
        feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
    )
    hf_model_id = load_model_id or model_id
    model_cfg_obj = ModelConfig(
        model_name=hf_model_id,
        model_revision=revision if revision not in ("", "main") else None,
        device=device,
        batch_size=batch_size,
        n_params=n_params,
        trust_remote_code=trust_remote_code,
    )
    reg_cfg = RegressionConfig(
        n_splits=int(reg.get("n_splits", 5)),
        alpha=_resolve_alphas(reg.get("alpha")),
        scale=bool(reg.get("scale", True)),
        random_state=int(reg.get("random_state", 0)),
        print_chosen_ridge_alpha=bool(reg.get("print_chosen_ridge_alpha", False)),
    )
    return Fedorenko2016BenchmarkConfig(
        data=data_cfg,
        model=model_cfg_obj,
        regression=reg_cfg,
        layerwise=bool(bench.get("layerwise", True)),
        n_splits=int(reg.get("n_splits", 5)),
        n_jobs=n_jobs,
        n_layer_workers=n_layer_workers,
        tokenizer_no_revision=bool(bench.get("tokenizer_no_revision", True)),
        regression_device=regression_device,
        output_dir=str(output_dir) if output_dir is not None else None,
    )


def _save_sentence_ppl(
    *,
    out_dir: Path,
    model_id: str,
    revision: str,
    dataset_root: Path,
    sentence_nll: np.ndarray,
    sentence_ntoks: np.ndarray,
) -> float:
    sent_df = Fedorenko2016Dataset(dataset_root).load_sentences()
    sent_df = sent_df.sort_values("sentence_id").reset_index(drop=True)
    n = min(len(sent_df), len(sentence_nll), len(sentence_ntoks))
    sent_df = sent_df.iloc[:n].copy()

    rows = pd.DataFrame(
        {
            "model_id": model_id,
            "revision": revision,
            "sentence_id": sent_df["sentence_id"].astype(int).values,
            "sentence": sent_df["sentence"].astype(str).values,
            "mean_nll": sentence_nll[:n],
            "n_tokens": sentence_ntoks[:n],
            "perplexity": np.exp(sentence_nll[:n]),
        }
    )
    ppl_csv = out_dir / "fedorenko2016_sentence_ppl.csv"
    if ppl_csv.exists():
        prev = pd.read_csv(ppl_csv)
        prev = prev[prev["revision"] != revision] if "revision" in prev.columns else prev
        rows = pd.concat([prev, rows], ignore_index=True)
    rows.to_csv(ppl_csv, index=False)

    valid = np.isfinite(sentence_nll[:n]) & np.isfinite(sentence_ntoks[:n])
    total_nll = float((sentence_nll[:n][valid] * sentence_ntoks[:n][valid]).sum())
    total_tok = int(sentence_ntoks[:n][valid].sum())
    corpus_ppl = float(np.exp(total_nll / total_tok)) if total_tok > 0 else float("nan")
    ppl_json = out_dir / "perplexity.json"
    ppl_json.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "revision": revision,
                "corpus": "fedorenko2016",
                "perplexity": round(corpus_ppl, 4) if np.isfinite(corpus_ppl) else None,
                "n_sentences": int(valid.sum()),
                "n_tokens": total_tok,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return corpus_ppl


def _summary_row(ckpt: CheckpointItem, result: Fedorenko2016Result, corpus_ppl: float) -> dict[str, Any]:
    df = result.to_dataframe()
    row: dict[str, Any] = {
        "step": ckpt.step,
        "revision": ckpt.revision,
        "tokens_seen": ckpt.tokens_seen,
        "n_words": result.n_words,
        "n_sentences": result.n_sentences,
        "n_electrodes": len(df),
        "n_subjects": int(df["subject_uid"].nunique()) if not df.empty else 0,
        "mean_best_spearmanr": float(np.nanmean(df["best_spearmanr"])) if not df.empty else np.nan,
        "mean_flat_spearmanr": float(np.nanmean(df["spearmanr"])) if not df.empty else np.nan,
        "mean_best_pearsonr": float(np.nanmean(df["best_pearsonr"])) if not df.empty else np.nan,
        "mean_flat_pearsonr": float(np.nanmean(df["pearsonr"])) if not df.empty else np.nan,
        "perplexity": corpus_ppl,
    }
    if result.electrode_scores and result.electrode_scores[0].layer_scores:
        mat = np.array(
            [[ls.spearmanr for ls in s.layer_scores] for s in result.electrode_scores],
            dtype=float,
        )
        mean_per_layer = np.nanmean(mat, axis=0)
        if np.any(np.isfinite(mean_per_layer)):
            row["best_layer"] = int(np.nanargmax(mean_per_layer))
            row["best_layer_mean_spearmanr"] = float(np.nanmax(mean_per_layer))
    return row


def _layerwise_rows(ckpt: CheckpointItem, result: Fedorenko2016Result) -> list[dict[str, Any]]:
    if not result.electrode_scores or not result.electrode_scores[0].layer_scores:
        return []
    n_layers = len(result.electrode_scores[0].layer_scores)
    rows: list[dict[str, Any]] = []
    for li in range(n_layers):
        sr = np.array([s.layer_scores[li].spearmanr for s in result.electrode_scores], dtype=float)
        pr = np.array([s.layer_scores[li].pearsonr for s in result.electrode_scores], dtype=float)
        rows.append(
            {
                "step": ckpt.step,
                "revision": ckpt.revision,
                "tokens_seen": ckpt.tokens_seen,
                "layer": li,
                "layer_rel": li / max(1, n_layers - 1),
                "mean_spearmanr": float(np.nanmean(sr)),
                "mean_pearsonr": float(np.nanmean(pr)),
            }
        )
    return rows


def _append_result_tables(out_dir: Path, result: Fedorenko2016Result, revision: str) -> None:
    elec_csv = out_dir / "fedorenko2016_by_electrode.csv"
    subj_csv = out_dir / "fedorenko2016_by_subject.csv"

    elec_df = result.to_dataframe()
    elec_df["revision"] = revision
    for _, row in elec_df.iterrows():
        append_row(elec_csv, row.to_dict())

    subj_df = result.by_subject()
    subj_df["revision"] = revision
    for _, row in subj_df.iterrows():
        append_row(subj_csv, row.to_dict())


def run_activations_only(
    checkpoints: list[CheckpointItem],
    *,
    model_id: str,
    n_params: float,
    batch_size: int,
    cfg: dict,
    device: str,
    feature_cache_dir: Path | None,
    output_dir: Path,
    model_cfg: dict | None = None,
) -> None:
    for ckpt in checkpoints:
        logger.info("[%s] activations/ppl — %s", model_id, ckpt.revision)
        load_model_id = resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        bench_cfg = _build_benchmark_config(
            cfg,
            model_id=model_id,
            load_model_id=load_model_id,
            revision=ckpt.revision,
            n_params=n_params,
            batch_size=batch_size,
            device=device,
            feature_cache_dir=feature_cache_dir,
            n_jobs=1,
            n_layer_workers=1,
            regression_device="cpu",
            trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
        )
        encoder = Fedorenko2016Encoder(bench_cfg)
        _, sentence_nll, sentence_ntoks, n_words, n_sent = encoder.extract_features_only()
        ppl = _save_sentence_ppl(
            out_dir=output_dir,
            model_id=model_id,
            revision=ckpt.revision,
            dataset_root=Path(bench_cfg.data.root),
            sentence_nll=sentence_nll,
            sentence_ntoks=sentence_ntoks,
        )
        logger.info(
            "[%s] %s done — words=%d sentences=%d ppl=%.3f",
            model_id,
            ckpt.revision,
            n_words,
            n_sent,
            ppl,
        )
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
    device: str,
    feature_cache_dir: Path | None,
    n_jobs: int,
    n_layer_workers: int,
    regression_device: str,
    no_plots: bool = False,
    model_cfg: dict | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / "fedorenko2016_checkpoint_scores.csv"
    layer_csv = output_dir / "fedorenko2016_checkpoint_layerwise.csv"
    existing = load_existing_csv(scores_csv)
    summary_rows = existing.to_dict("records") if not existing.empty else []

    for ckpt in checkpoints:
        if already_computed(existing, ckpt.revision):
            logger.info("[%s] %s already done — skipping", model_id, ckpt.revision)
            continue

        logger.info("[%s] Fedorenko2016 ── %s (step=%d) ──", model_id, ckpt.revision, ckpt.step)
        load_model_id = resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        bench_cfg = _build_benchmark_config(
            cfg,
            model_id=model_id,
            load_model_id=load_model_id,
            revision=ckpt.revision,
            n_params=n_params,
            batch_size=batch_size,
            device=device,
            feature_cache_dir=feature_cache_dir,
            n_jobs=n_jobs,
            n_layer_workers=n_layer_workers,
            regression_device=regression_device,
            trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
            output_dir=output_dir,
        )

        try:
            result = Fedorenko2016Encoder(bench_cfg).run()
            corpus_ppl = _save_sentence_ppl(
                out_dir=output_dir,
                model_id=model_id,
                revision=ckpt.revision,
                dataset_root=Path(bench_cfg.data.root),
                sentence_nll=result.sentence_nll,
                sentence_ntoks=result.sentence_ntoks,
            )
            _append_result_tables(output_dir, result, ckpt.revision)
            row = _summary_row(ckpt, result, corpus_ppl)
            append_row(scores_csv, row)
            summary_rows.append(row)
            for lw in _layerwise_rows(ckpt, result):
                append_row(layer_csv, lw)
            existing = load_existing_csv(scores_csv)
        except Exception:
            logger.error("[%s] failed at %s", model_id, ckpt.revision, exc_info=True)
        finally:
            clear_hf_model_cache()
            free_gpu()

    df = pd.DataFrame(summary_rows)
    if not df.empty and not no_plots:
        _plot_training_curve(df, output_dir, model_id)
        if layer_csv.exists():
            _plot_layer_heatmap(pd.read_csv(layer_csv), output_dir, model_id)
    return df


def _plot_training_curve(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if "step" not in df.columns:
        return
    plot_df = df.sort_values("step")
    fig, ax = plt.subplots(figsize=(9, 4))
    if "mean_best_spearmanr" in plot_df.columns:
        ax.plot(plot_df["step"], plot_df["mean_best_spearmanr"], "o-", label="Best layer")
    if "mean_flat_spearmanr" in plot_df.columns:
        ax.plot(plot_df["step"], plot_df["mean_flat_spearmanr"], "s--", alpha=0.7, label="Last layer")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean Spearman r")
    ax.set_title(f"Fedorenko2016 — {model_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fedorenko2016_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_heatmap(layer_df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if layer_df.empty:
        return
    pivot = layer_df.pivot_table(index="layer", columns="step", values="mean_spearmanr")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.6), 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer")
    ax.set_title(f"Fedorenko2016 layerwise — {model_id}")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(int(s)) for s in pivot.columns], rotation=45, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean Spearman r")
    fig.tight_layout()
    fig.savefig(output_dir / "fedorenko2016_layer_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_summary(output_dir: Path, *, family: str | None = None) -> None:
    search_root = output_dir / family if family else output_dir
    summary_dir = (output_dir / family / "summary") if family else (output_dir / "summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for csv in sorted(search_root.glob("**/fedorenko2016_checkpoint_scores.csv")):
        model_id = csv.parent.name.replace("_", "/", 1)
        fam = csv.parent.parent.name if csv.parent.parent != search_root else None
        df = pd.read_csv(csv)
        if df.empty:
            continue
        df.insert(0, "model_id", model_id)
        if fam:
            df.insert(0, "family", fam)
        all_rows.append(df)

    if not all_rows:
        logger.warning("No fedorenko2016_checkpoint_scores.csv found under %s", search_root)
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(summary_dir / "all_models_fedorenko2016_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    title_family = f" ({family})" if family else ""
    for model_id, grp in combined.groupby("model_id"):
        g = grp.sort_values("step")
        if "mean_best_spearmanr" in g.columns:
            ax.plot(g["step"], g["mean_best_spearmanr"], "o-", label=model_id)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean best-layer Spearman r")
    ax.set_title(f"Fedorenko2016 — all models{title_family}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(summary_dir / "all_models_fedorenko2016_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary written to %s", summary_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/embedding/neural/fedorenko2016.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n_jobs", type=int, default=20)
    p.add_argument("--n_layer_workers", type=int, default=8)
    p.add_argument("--regression_device", default="cuda")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--activations_only", action="store_true")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg.get("output", {}).get("save_dir", "outputs_jeanzay/fedorenko2016/embedding_sweep"))

    cache_raw = (cfg.get("cache") or {}).get("features_dir")
    feature_cache_dir = Path(os.path.expandvars(str(cache_raw))).resolve() if cache_raw else None
    if feature_cache_dir:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        run_summary(out_dir, family=args.family)
        return

    if not args.from_registry:
        p.error("Specify --from-registry with --family (and optional --array-index)")
    if not args.family:
        p.error("--from-registry requires --family")

    if args.array_index is not None:
        task = resolve_family_task(args.family, args.array_index)
        model_cfgs = [m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id]
        model_cfg = model_cfgs[0]
        checkpoints = [CheckpointItem(r, s, t) for r, s, t in build_checkpoint_items(model_cfg) if r == task.revision]
        if args.dry_run:
            print(f"task: {task}")
            return
        if args.activations_only:
            run_activations_only(
                checkpoints,
                model_id=task.model_id,
                n_params=float(model_cfg.get("n_params", 0)),
                batch_size=int(model_cfg.get("batch_size", 8)),
                cfg=cfg,
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                output_dir=_model_output_dir(out_dir, task.model_id, family=args.family, model_cfg=model_cfg),
                model_cfg=model_cfg,
            )
        else:
            run_model_sweep(
                checkpoints,
                model_id=task.model_id,
                n_params=float(model_cfg.get("n_params", 0)),
                batch_size=int(model_cfg.get("batch_size", 8)),
                cfg=cfg,
                output_dir=_model_output_dir(out_dir, task.model_id, family=args.family, model_cfg=model_cfg),
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                n_jobs=args.n_jobs,
                n_layer_workers=args.n_layer_workers,
                regression_device=args.regression_device,
                no_plots=args.no_plots,
                model_cfg=model_cfg,
            )
        return

    all_models = registry_model_configs(args.family)
    if args.models:
        all_models = [m for m in all_models if m["model_id"] in args.models]
    if args.dry_run:
        for m in all_models:
            ckpts = [CheckpointItem(r, s, t) for r, s, t in build_checkpoint_items(m)]
            print(f"{m['model_id']}: {len(ckpts)} checkpoints")
        return

    for m in all_models:
        ckpts = [CheckpointItem(r, s, t) for r, s, t in build_checkpoint_items(m)]
        if args.activations_only:
            run_activations_only(
                ckpts,
                model_id=m["model_id"],
                n_params=float(m.get("n_params", 0)),
                batch_size=int(m.get("batch_size", 8)),
                cfg=cfg,
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                output_dir=_model_output_dir(out_dir, m["model_id"], model_cfg=m),
                model_cfg=m,
            )
        else:
            run_model_sweep(
                ckpts,
                model_id=m["model_id"],
                n_params=float(m.get("n_params", 0)),
                batch_size=int(m.get("batch_size", 8)),
                cfg=cfg,
                output_dir=_model_output_dir(out_dir, m["model_id"], model_cfg=m),
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                n_jobs=args.n_jobs,
                n_layer_workers=args.n_layer_workers,
                regression_device=args.regression_device,
                no_plots=args.no_plots,
                model_cfg=m,
            )

    if not args.skip_summary:
        run_summary(out_dir, family=args.family)


if __name__ == "__main__":
    main()

