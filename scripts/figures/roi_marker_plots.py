#!/usr/bin/env python3
"""ROI marker plots for last-layer surprisal (main) and embeddings (appendix).

Each row is a dataset. Within each ROI there is one bar per model class (group
mean, white diamond = mean +/- SEM); each dot is a model's held-out score in
that ROI, with marker area denoting log parameter count. The
script reads the ROI-wise surprisal reruns and existing ROI-resolved embedding
outputs, then writes both figures and the plotted long-form tables.

Run with ``uv run --frozen --no-sync python scripts/figures/roi_marker_plots.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from neuroai.analysis.benchmark_common import (
    EXCLUDED_MODEL_IDS, GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, REGISTRY_INDEX,
    OUTPUTS_JEANZAY, REPO_ROOT, dll_ceiling_for, lookup_model_meta, model_dir_to_model_id,
)
from neuroai.analysis.paper_style import log_size_map, use_paper_style
from scripts.figures.figure5_roi_comparison import ROI_ORDER as FEDORENKO_ROIS
from scripts.figures.figure5_roi_comparison import _iter_lpp_final
from scripts.summarize.benchmark import (
    _iter_model_score_files, _pick_final_row, _read_csv_safe,
)

OUT_DIR = REPO_ROOT / "reports" / "paper"
PODCAST_ROI_MAP = REPO_ROOT / "outputs" / "podcast" / "electrode_roi_map_language_responsive.csv"
DATASETS = ("lpp", "ns_fmri", "podcast", "tuckute", "pereira")
# Report slug whose delta-log-likelihood noise ceiling normalises each dataset.
DLL_CEILING_SLUGS = {
    "lpp": "lpp_fmri__surprisal", "ns_fmri": "ns_fmri__surprisal",
    "podcast": "podcast_ieeg__surprisal", "tuckute": "tuckute_fmri__surprisal",
    "pereira": "pereira_fmri",
}
DATASET_LABELS = {
    "lpp": "LPP fMRI",
    "ns_fmri": "Natural Stories fMRI",
    "podcast": "Podcast iEEG",
    "tuckute": "Tuckute fMRI",
    "pereira": "Pereira fMRI",
}


def _read_final_rows(files: list[Path]) -> pd.DataFrame:
    frames = [d for f in files if (d := _read_csv_safe(f)) is not None]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Results may be split over checkpoint subdirectories. Use the latest
    # trained checkpoint before selecting the final layer.
    key = "tokens_seen" if "tokens_seen" in df.columns else ("step" if "step" in df.columns else None)
    if key is not None and df[key].notna().any():
        final = _pick_final_row(df)
        df = df[pd.to_numeric(df[key], errors="coerce") == pd.to_numeric(final[key], errors="coerce")]
        if "revision" in df.columns and pd.notna(final.get("revision")):
            df = df[df["revision"].astype(str) == str(final["revision"])]
    return df


def _model_results(root: Path, filename: str):
    for family, model_dir, files in _iter_model_score_files(root, filename):
        model_id = model_dir_to_model_id(model_dir.name)
        if model_id in EXCLUDED_MODEL_IDS:
            continue
        meta = lookup_model_meta(model_id, family)
        if not np.isfinite(meta.n_params) or meta.n_params <= 0:
            continue
        scores = _read_final_rows(files)
        if not scores.empty:
            yield model_id, meta, model_dir, files, scores


def _finite(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df:
        return df.iloc[0:0].copy()
    out = df.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[np.isfinite(out[column])]


def _left_roi(value: object, *, prefix: str) -> str | None:
    """Return a short ROI name only for the left hemisphere."""
    name = str(value)
    if prefix == "lpp":
        return name[2:] if name.startswith("L_") else None
    if prefix == "ns":
        return name.removeprefix("LangL") if name.startswith("LangL") else None
    if prefix == "tuckute":
        if not name.startswith("lang_LH_"):
            return None
        roi = name.removeprefix("lang_LH_")
        return "Network" if roi == "netw" else roi
    if prefix == "pereira":
        return name.removeprefix("LH_") if name.startswith("LH_") else None
    return name


def _append_rows(out: list[dict], model_id: str, meta, roi: str, score: float) -> None:
    out.append({
        "dataset": "", "model_id": model_id, "family": meta.family,
        "group": meta.group, "n_params": float(meta.n_params),
        "roi": str(roi), "score": float(score),
    })


def load_surprisal(dataset: str) -> pd.DataFrame:
    roots = {
        "lpp": (OUTPUTS_JEANZAY / "lpp/surprisal_roi_lastlayer", "lpp_lme_by_roi.csv", "region", "dll_test_per_obs"),
        "ns_fmri": (OUTPUTS_JEANZAY / "ns_fmri/surprisal_sweep", "ns_fmri_lme_by_roi.csv", "froi", "dll_test_per_obs"),
        "podcast": (OUTPUTS_JEANZAY / "podcast/surprisal_roi_lastlayer", "podcast_lme_by_roi.csv", "region", "dll_test_per_obs"),
        "tuckute": (OUTPUTS_JEANZAY / "tuckute/surprisal_roi_lastlayer", "tuckute_lme_by_roi.csv", "roi", "dll_test_per_obs"),
        "pereira": (OUTPUTS_JEANZAY / "pereira/surprisal_roi_lastlayer", "pereira_lme_by_roi.csv", "roi", "dll_test_per_obs"),
    }
    root, filename, roi_col, score_col = roots[dataset]
    rows: list[dict] = []
    for model_id, meta, _, _, scores in _model_results(root, filename):
        if dataset == "ns_fmri":
            scores = scores[(scores.get("network", "") == "LANG") & (scores.get("hemisphere", "") == "L")]
        scores = scores.assign(dll_test_per_obs=pd.to_numeric(scores["delta_loglik_test"], errors="coerce")
                               / pd.to_numeric(scores["n_obs_test"], errors="coerce"))
        scores = _finite(scores, score_col)
        if "converged" in scores:
            scores = scores[scores["converged"].astype(str).str.lower().isin(("true", "1"))]
        if roi_col not in scores:
            continue
        if "layer" in scores:
            scores["layer"] = pd.to_numeric(scores["layer"], errors="coerce")
            scores = scores[scores["layer"] == scores.groupby(roi_col)["layer"].transform("max")]
        # Podcast's ROI fit returns one row per ROI at lag zero (the configured
        # 0–800 ms average); if a lag grid is present, retain its configured
        # initial lag rather than selecting by held-out score.
        if dataset == "podcast" and "lag" in scores:
            scores["lag"] = pd.to_numeric(scores["lag"], errors="coerce")
            scores = scores[scores["lag"] == scores.groupby(roi_col)["lag"].transform("min")]
        if dataset == "ns_fmri":
            roi_names = scores[roi_col].map(lambda x: _left_roi(x, prefix="ns"))
            roi_names = roi_names.where(roi_names.isin(FEDORENKO_ROIS))
        elif dataset == "lpp":
            roi_names = scores[roi_col].map(lambda x: _left_roi(x, prefix="lpp"))
        elif dataset == "tuckute":
            roi_names = scores[roi_col].map(lambda x: _left_roi(x, prefix="tuckute"))
        elif dataset == "pereira":
            roi_names = scores[roi_col].map(lambda x: _left_roi(x, prefix="pereira"))
        else:
            roi_names = scores[roi_col].astype(str)
        temp = scores.assign(roi=roi_names).dropna(subset=["roi"])
        for roi, g in temp.groupby("roi"):
            _append_rows(rows, model_id, meta, roi, float(g[score_col].mean()))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["dataset"] = dataset
        ceiling = dll_ceiling_for(DLL_CEILING_SLUGS[dataset])
        if ceiling is None or ceiling <= 0:
            raise RuntimeError(f"no delta-log-likelihood ceiling for {dataset}")
        result["ceiling"] = ceiling
        result["score_raw"] = result["score"]
        result["score"] = result["score_raw"] / ceiling
    return result


def _layer_summary(scores: pd.DataFrame, roi_col: str, score_col: str,
                   roi_mapper) -> pd.DataFrame:
    scores = _finite(scores, score_col)
    scores = scores.dropna(subset=[roi_col, "layer"]).copy()
    scores["layer"] = pd.to_numeric(scores["layer"], errors="coerce")
    scores = scores.dropna(subset=["layer"])
    if scores.empty:
        return pd.DataFrame(columns=["roi", "score"])
    scores = scores[scores["layer"] == scores.groupby(roi_col)["layer"].transform("max")]
    scores["roi"] = scores[roi_col].map(roi_mapper)
    return scores.dropna(subset=["roi"]).groupby("roi", as_index=False)[score_col].mean().rename(columns={score_col: "score"})


def _load_pereira_embedding_rows(model_id: str, meta, model_dir: Path, files: list[Path]) -> list[dict]:
    """Aggregate final-layer voxel correlations within each ROI, then subjects.

    Pereira's standard score CSV pools over voxels. Its per-voxel parquet files
    retain ROI labels, so the ROI score is the mean voxel r within subject/ROI,
    averaged equally over available subjects, matching the dataset's existing
    subject-averaged summary convention.
    """
    summaries = [d for f in files if (d := _read_csv_safe(f)) is not None]
    if not summaries:
        return []
    summary = pd.concat(summaries, ignore_index=True)
    final = _pick_final_row(summary)
    layer = pd.to_numeric(summary["layer"], errors="coerce")
    if "revision" in summary and pd.notna(final.get("revision")):
        summary = summary[summary["revision"].astype(str) == str(final["revision"])]
    if "tokens_seen" in summary and pd.notna(final.get("tokens_seen")):
        summary = summary[pd.to_numeric(summary["tokens_seen"], errors="coerce") == float(final["tokens_seen"])]
    elif "step" in summary and pd.notna(final.get("step")):
        summary = summary[pd.to_numeric(summary["step"], errors="coerce") == float(final["step"])]
    summary_layers = pd.to_numeric(summary["layer"], errors="coerce").dropna()
    if summary_layers.empty:
        return []
    last_layer = int(summary_layers.max())

    # Find the revision-specific voxel score directory next to the summary CSV.
    source = next((f for f in files if f.parent.name == str(final.get("revision", ""))), files[-1])
    candidates = [source.parent / "voxel_scores", model_dir / "voxel_scores"]
    voxel_dir = next((p for p in candidates if p.is_dir()), None)
    if voxel_dir is None:
        return []
    revision = str(final.get("revision", "main")).replace("/", "_")
    voxel_files = sorted(voxel_dir.glob(f"*_layer{last_layer}_{revision}.parquet"))
    subject_scores = []
    for path in voxel_files:
        try:
            vox = pd.read_parquet(path, columns=["roi", "pearsonr"])
        except (OSError, ValueError, ImportError):
            continue
        vox["pearsonr"] = pd.to_numeric(vox["pearsonr"], errors="coerce")
        vox = vox[np.isfinite(vox["pearsonr"])]
        vox["roi"] = vox["roi"].map(lambda x: _left_roi(x, prefix="pereira"))
        vox = vox.dropna(subset=["roi"])
        if not vox.empty:
            subject_scores.append(vox.groupby("roi")["pearsonr"].mean().rename(path.name.split("_layer", 1)[0]))
    if not subject_scores:
        return []
    by_roi = pd.concat(subject_scores, axis=1).mean(axis=1, skipna=True)
    return [{"dataset": "pereira", "model_id": model_id, "family": meta.family,
             "group": meta.group, "n_params": float(meta.n_params), "roi": roi,
             "score": float(score)} for roi, score in by_roi.items() if np.isfinite(score)]


def load_embeddings(dataset: str) -> pd.DataFrame:
    roots = {
        "lpp": (OUTPUTS_JEANZAY / "lpp/embedding_sweep", "lpp_checkpoint_by_fedorenko_parcel.csv"),
        "ns_fmri": (OUTPUTS_JEANZAY / "ns_fmri/embedding_sweep", "ns_fmri_checkpoint_layerwise.csv"),
        "podcast": (OUTPUTS_JEANZAY / "podcast/embedding_sweep", "podcast_by_electrode_layerwise.csv"),
        "tuckute": (OUTPUTS_JEANZAY / "tuckute/embedding_sweep", "tuckute_by_roi.csv"),
        "pereira": (OUTPUTS_JEANZAY / "pereira/embedding_sweep", "pereira_embed_by_layer.csv"),
    }
    root, filename = roots[dataset]
    rows: list[dict] = []
    if dataset == "lpp":
        # LPP has a mix of nested registry outputs and flattened memory-model
        # directories; use the same comprehensive walker as the existing ROI
        # comparison so the latter are not silently omitted.
        for model_id, _, scores in _iter_lpp_final(filename):
            meta = REGISTRY_INDEX.get(model_id)
            if meta is None:
                continue
            layer_rows = _layer_summary(
                scores, "parcel", "mean_pearson_left",
                lambda x: str(x) if str(x) in FEDORENKO_ROIS else None,
            )
            for row in layer_rows.to_dict("records"):
                _append_rows(rows, model_id, meta, row["roi"], row["score"])
        result = pd.DataFrame(rows)
        if not result.empty:
            result["dataset"] = dataset
        return result
    podcast_map = None
    if dataset == "podcast":
        podcast_map = pd.read_csv(PODCAST_ROI_MAP)
        podcast_map = podcast_map[podcast_map["assigned"].astype(bool) & (podcast_map["hemisphere"] == "L")]
        podcast_map = podcast_map[["subject", "channel", "region"]].rename(columns={"channel": "electrode"})
    results_iter = list(_model_results(root, filename))
    if dataset == "podcast":
        # cbr-rnn models are single-layer: no *_layerwise.csv, only by_electrode.
        have = {r[0] for r in results_iter}
        for mid, meta, mdir, files, scores in _model_results(root, "podcast_by_electrode.csv"):
            if mid not in have:
                results_iter.append((mid, meta, mdir, files, scores.drop(columns=["region", "hemisphere"], errors="ignore").assign(layer=0)))
    for model_id, meta, model_dir, files, scores in results_iter:
        if dataset == "pereira":
            rows.extend(_load_pereira_embedding_rows(model_id, meta, model_dir, files))
            continue
        if dataset == "lpp":
            scores = _finite(scores, "mean_pearson_left")
            layer_rows = _layer_summary(scores, "parcel", "mean_pearson_left", lambda x: str(x) if str(x) in FEDORENKO_ROIS else None)
        elif dataset == "ns_fmri":
            scores = scores[(scores.get("network", "") == "LANG") & (scores.get("hemisphere", "") == "L")]
            layer_rows = _layer_summary(scores, "froi", "pearsonr",
                                        lambda x: (_left_roi(x, prefix="ns")
                                                   if _left_roi(x, prefix="ns") in FEDORENKO_ROIS else None))
        elif dataset == "podcast":
            scores = _finite(scores, "pearsonr")
            scores = scores.dropna(subset=["subject", "electrode", "layer"])
            scores = scores.merge(podcast_map, on=["subject", "electrode"], how="inner")
            layer_rows = _layer_summary(scores, "region", "pearsonr", lambda x: str(x) if str(x) in FEDORENKO_ROIS else None)
        else:  # Tuckute naturalistic baseline condition B
            if "cond" in scores:
                scores = scores[scores["cond"].astype(str) == "B"]
            layer_rows = _layer_summary(scores, "roi", "pearsonr", lambda x: _left_roi(x, prefix="tuckute"))
        for row in layer_rows.to_dict("records"):
            _append_rows(rows, model_id, meta, row["roi"], row["score"])
    result = pd.DataFrame(rows)
    if not result.empty:
        result["dataset"] = dataset
    return result


def _roi_order(dataset: str, frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    vals = set(frame["roi"].astype(str))
    if dataset in ("lpp", "ns_fmri", "podcast"):
        ordered = list(FEDORENKO_ROIS)
    elif dataset == "tuckute":
        ordered = ["IFGorb", "IFG", "MFG", "AntTemp", "PostTemp", "Network"]
    else:
        ordered = ["IFGorb", "IFG", "MFG", "AntTemp", "PostTemp", "AngG"]
    return [x for x in ordered if x in vals] + sorted(vals.difference(ordered))


def _draw_track(frames: dict[str, pd.DataFrame], *, track: str, out_path: Path) -> None:
    active = [(ds, frames.get(ds, pd.DataFrame())) for ds in DATASETS]
    fig, axes = plt.subplots(len(active), 1, figsize=(13.2, 3.4 * len(active)), squeeze=False)
    axes = axes[:, 0]
    all_sizes = pd.concat([d["n_params"] for _, d in active if not d.empty], ignore_index=True)
    log_params = np.log10(pd.to_numeric(all_sizes, errors="coerce").replace(0, np.nan).dropna())
    lo, hi = (float(np.floor(log_params.min())), float(np.ceil(log_params.max()))) if len(log_params) else (7.0, 11.0)
    min_size, max_size = 16.0, 72.0

    for ax, (dataset, frame) in zip(axes, active):
        if frame.empty:
            ax.text(0.5, 0.5, "No ROI-resolved scores found", transform=ax.transAxes,
                    ha="center", va="center", color="#777777")
            ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="semibold")
            ax.set_axisbelow(True)
            continue
        rois = _roi_order(dataset, frame)
        frame = frame[frame["roi"].isin(rois) & np.isfinite(frame["score"])].copy()
        frame["_size"] = log_size_map(frame["n_params"], lo, hi, s_min=min_size, s_max=max_size)
        groups = [g for g in GROUP_ORDER if (frame["group"] == g).any()]
        slot = 0.8 / max(len(groups), 1)
        rng = np.random.default_rng(0)
        for ri, roi in enumerate(rois):
            for gi, group in enumerate(groups):
                sub = frame[(frame["roi"] == roi) & (frame["group"] == group)]
                if sub.empty:
                    continue
                x0 = ri + (gi - (len(groups) - 1) / 2) * slot
                vals = sub["score"].to_numpy(dtype=float)
                m = float(vals.mean())
                se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                ax.bar(x0, m, width=slot * 0.9, color=GROUP_COLORS[group], alpha=0.20,
                       edgecolor=GROUP_COLORS[group], linewidth=1.0, zorder=1)
                sizes = sub["_size"].to_numpy()
                order = np.argsort(-sizes)
                xj = x0 + rng.uniform(-0.33, 0.33, len(sub)) * slot
                ax.scatter(xj[order], vals[order], s=sizes[order], color=GROUP_COLORS[group],
                           alpha=0.55, edgecolors="white", linewidths=0.35, zorder=2,
                           rasterized=True)
                ax.errorbar(x0, m, yerr=se, fmt="D", color="black", markersize=4.5,
                            markerfacecolor="white", markeredgewidth=1.1, capsize=3,
                            elinewidth=1.2, zorder=4)
        ax.set_xticks(np.arange(len(rois)), rois)
        ax.set_xlim(-0.5, len(rois) - 0.5)
        ax.set_title(f"{DATASET_LABELS[dataset]}  (n={frame['model_id'].nunique()} models)",
                     loc="left", fontweight="semibold", fontsize=10.5)
        ax.set_ylabel("Pearson r" if track == "embedding" else r"$\Delta$LL / ceiling")
        ax.axhline(0, color="#555555", lw=0.65, alpha=0.7, zorder=1)
        ax.grid(axis="y", alpha=0.32)
        ax.set_axisbelow(True)

    fig.suptitle(
        "ROI-wise last-layer surprisal" if track == "surprisal" else "ROI-wise last-layer embeddings",
        x=0.075, y=0.995, ha="left", fontweight="semibold", fontsize=14,
    )
    group_handles = [Line2D([], [], marker="o", ls="None", color=GROUP_COLORS[g],
                            markeredgecolor="white", markersize=7,
                            label=GROUP_DISPLAY.get(g, g))
                     for g in GROUP_ORDER if any((not d.empty and (d["group"] == g).any()) for _, d in active)]
    size_values = [1e7, 1e8, 1e9, 1e10]
    size_handles = [plt.scatter([], [], s=log_size_map(pd.Series([v]), lo, hi,
                                                        s_min=min_size, s_max=max_size)[0],
                                color="#777777", edgecolors="white", linewidths=0.35,
                                label=(f"{v / 1e9:g}B" if v >= 1e9 else f"{v / 1e6:g}M"))
                    for v in size_values if 10 ** lo <= v <= 10 ** hi]
    handles = group_handles + size_handles
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
               bbox_to_anchor=(0.5, 0.002), frameon=False,
               title="Model class                                      Parameters")
    fig.supxlabel("Left-hemisphere ROI", y=0.037)
    fig.tight_layout(rect=(0.035, 0.075, 0.99, 0.975), h_pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    use_paper_style()
    results: dict[str, pd.DataFrame] = {}
    for dataset in DATASETS:
        s = load_surprisal(dataset)
        e = load_embeddings(dataset)
        results[f"surprisal_{dataset}"] = s
        results[f"embedding_{dataset}"] = e
        print(f"{dataset:8s}: surprisal {s['model_id'].nunique() if not s.empty else 0} models / "
              f"{len(s)} model-ROI rows; embedding {e['model_id'].nunique() if not e.empty else 0} models / {len(e)} model-ROI rows")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for track in ("surprisal", "embedding"):
        frames = {ds: results[f"{track}_{ds}"] for ds in DATASETS}
        table = pd.concat([d for d in frames.values() if not d.empty], ignore_index=True) if any(not d.empty for d in frames.values()) else pd.DataFrame()
        table.to_csv(OUT_DIR / f"roi_{track}_lastlayer_points.csv", index=False)
        output = OUT_DIR / ("roi_surprisal_main.pdf" if track == "surprisal" else "roi_embedding_appendix.pdf")
        _draw_track(frames, track=track, out_path=output)


if __name__ == "__main__":
    main()
