#!/usr/bin/env python3
"""Pareto fronts of PEARSON RSA / noise ceiling (neural vs behavioral), language regions only for the regional datasets.

Score per (model, unit) = plain Pearson RSA between the model RDM and the unit's response RDM (last layer; embedding = per-feature
standardized states), divided by that unit's OWN RSA noise ceiling = sqrt(Spearman-Brown corrected split-half RDM correlation)
(neural: rdm_ceiling_units.py; behavioral: rdm_ceiling_rsa_behavioral.py). A neural dataset's score is the median over its units of RSA_u / ceiling_u (units without a positive ceiling are dropped); a model's composite on each axis is the MEDIAN over datasets (neural) or measures (behavioral).

Units: neural = Pereira (language atlas ROIs only), Tuckute (language ROIs only), Podcast (Fedorenko language parcels),
Broderick and MEG-MASC (regions; all pairs), the three N400 sets; behavioral = the 16 reading measures.
Fedorenko is excluded (no reliable RDM-level ceiling). Output folder: reports/rsa_summary/rsa_pareto/.

    .venv/bin/python scripts/figures/rsa_pareto.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, family_group
from neuroai.analysis.paper_style import log_size_map, use_paper_style
from scripts.analysis import rdm_regression_neural_summary as NS
from scripts.figures import rdm_regression_figures as F

# RSA_PARETO_ROOT: results root to read instead of $SCRATCH (Jean Zay resets SCRATCH inside jobs), e.g. $SCRATCH/rdm_v2
SCR = Path(os.environ.get("RSA_PARETO_ROOT") or os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
OUT = ROOT / "reports/rsa_summary/rsa_pareto"
PAPER = ROOT / "paperwriting/figures"
_SIZE_REFS = (1e7, 1e8, 1e9, 1e10, 7e10)
# legends inside the axes (corner, as in the LME / ridge Pareto figures of the paper): surprisal bottom right, embedding top right
LEGEND_CORNER = {"surprisal": "lower right", "embedding": "upper right"}
KEYS = ["family", "model_id", "revision"]
BEHAV = {"provo": "Provo", "meco": "MECO", "geco": "GECO", "onestop": "OneStop", "naturalstories": "Natural Stories",
         "ucl_reading": "UCL Reading", "zuco_reading": "ZuCo"}
# dataset -> (loader arguments atlas, kind) restricting the regional datasets to language regions
NEURAL = {"pereira": (["language"], None), "tuckute": (["lang"], None), "podcast": (None, None),
          "broderick": (None, "region"), "megmasc": (None, "region"),
          "nieuwland": (None, None), "szewczyk_f2007": (None, None), "szewczyk_h2019": (None, None)}
CEIL_KEY = {"pereira": "pereira", "tuckute": "tuckute", "podcast": "podcast", "broderick": "broderick", "megmasc": "megmasc",
            "nieuwland": "nieuwland", "szewczyk_f2007": "f2007", "szewczyk_h2019": "h2019"}
TITLE = {"pereira": "Pereira (language)", "tuckute": "Tuckute (language)", "podcast": "Podcast", "broderick": "Broderick",
         "megmasc": "MEG-MASC", "nieuwland": "Nieuwland", "szewczyk_f2007": "Federmeier 2007", "szewczyk_h2019": "Hubbard 2019"}


def _group(family, model_id, stored):
    """Current registry group (neuroai.analysis.benchmark_common.family_group); falls back to the stored one for non-registry rows."""
    try:
        return family_group(family, model_id)
    except KeyError:
        return stored


def unit_ceilings(ds: str) -> tuple[pd.DataFrame, list[str]]:
    """Per-unit RSA ceilings (rdm_ceiling_units.py) and the columns that identify a unit in both the results and the ceiling file."""
    if ds in NEW_NEURAL:
        c = pd.concat([pd.read_csv(f) for f in sorted((ALLP / "ceilings").glob(f"ceilings_{ds}*.csv"))], ignore_index=True)
    else:
        c = pd.read_csv(SCR / f"rdm_ceiling_units/ceilings_{CEIL_KEY[ds]}.csv")
    if ds == "pereira":
        c, key = c[c.atlas == "language"], ["passage_set", "roi"]
    elif ds == "tuckute":
        c, key = c[c.atlas == "lang"], ["roi"]
    elif ds == "podcast":
        c, key = c[c.roi != "netw"], ["roi", "lag"]
    elif ds in ("broderick", "megmasc"):
        c, key = c.rename(columns={"channel": "unit"}), ["unit"]
    else:
        key = ["unit"]
    return c[key + ["rsa_ceiling", "dll_ceiling"]], key



ALLP = SCR / "rdm_allpairs/allpairs"          # all-pairs neural runs (Podcast parcels, MEG-MASC / Broderick regions) + their ceilings
NEW_NEURAL = ("podcast", "megmasc", "broderick")
# behavioral datasets: (results root, ceiling root, rsa-ceiling csv glob). Pair counts: Provo/MECO/UCL/ZuCo/Natural Stories = all pairs,
# GECO/OneStop = 30M sampled pairs.
BEH_SRC = {"provo": "old", "meco": "old",
           "ucl_reading": ("rdm_regression_allpairs", "rdm_ceiling_allpairs"), "zuco_reading": ("rdm_regression_allpairs", "rdm_ceiling_allpairs"),
           "naturalstories": ("rdm_regression_allpairs", "rdm_ceiling_allpairs"),
           "geco": ("rdm_regression_30m", "rdm_ceiling_30m"), "onestop": ("rdm_regression_30m", "rdm_ceiling_30m")}
BEH_MEASURES = {"provo": ("FP", "GP", "SP"), "meco": ("FP", "GP", "SP"), "geco": ("FP", "GP"), "onestop": ("FP", "GP", "SP"),
                "naturalstories": ("SPR",), "ucl_reading": ("SPR",), "zuco_reading": ("NR", "SR", "TSR")}


def load_neural(ds, atlas, kind):
    """Unit-level results: all-pairs runs for NEW_NEURAL, the earlier runs for the other datasets."""
    if ds not in NEW_NEURAL:
        return NS.load(ds, SCR, atlas, kind, None)
    d = NS.read([f for t in ("embedding", "surprisal") for f in (ALLP / ds / t).glob("units_*.csv")])
    if kind:
        d = d[d.kind == kind]
    if ds == "podcast":
        d = d[d.roi != "netw"]
    return d


def behavioral_ceilings() -> tuple[dict, dict]:
    """(rsa ceiling, dLL ceiling) per (dataset, measure): split-half ceilings on the same pairs as the scores."""
    rsa, dll = {}, {}
    old = pd.read_csv(SCR / "rdm_ceiling_rsa/behavioral_rsa_ceiling.csv")
    for r in old.itertuples():
        if BEH_SRC[r.dataset] == "old":
            rsa[(r.dataset, r.unit)] = r.rsa_ceiling
    for ds, src in BEH_SRC.items():
        croot = SCR / ("rdm_ceiling_std" if src == "old" else src[1])
        if src != "old":
            files = list((croot / ds).glob("rsa_ceiling_*.csv")) + list(croot.glob(f"rsa_ceiling_{ds}.csv"))
            for f in files:
                for r in pd.read_csv(f).itertuples():
                    rsa[(ds, r.unit)] = r.rsa_ceiling
        for m in BEH_MEASURES[ds]:
            z = np.load(croot / ds / f"ceiling_{m}.npz")
            c = z["ceiling"][list(z["variants"]).index("len_freq")]
            dll[(ds, m)] = float(np.mean(np.nanmean(c, axis=-1)))
    return rsa, dll


def behavioral_scores(track: str) -> pd.DataFrame:
    """Last-layer Pearson RSA and held-out dLL (confounds len_freq) of every model on every behavioral measure."""
    rows = []
    for ds, src in BEH_SRC.items():
        if src == "old":
            root = SCR / ("rdm_regression_std" if track == "embedding" else "rdm_regression_r30")
            t = pd.read_csv(root / ds / track / "model_summary.csv", dtype={"revision": str})
            t = t[t.confounds == "len_freq"]
            rows.append(pd.DataFrame(dict(dataset=ds, unit=t.measure, family=t.family, model_id=t.model_id, revision=t.revision,
                                          rsa=t.best_layer_rsa_pearson, dll=t.last_layer_delta_ll_per_obs, group=t.group, n_params=t.n_params)))
            continue
        frames = [pd.read_csv(f, dtype={"revision": str}) for f in (SCR / src[0] / ds / track).glob("*/*/*/rdm_regression_*.csv")]
        t = pd.concat(frames, ignore_index=True)
        t = t[t.confounds == "len_freq"]
        t = t[t.layer == t.groupby(["measure", "family", "model_id", "revision"]).layer.transform("max")]
        rows.append(pd.DataFrame(dict(dataset=ds, unit=t.measure, family=t.family, model_id=t.model_id, revision=t.revision,
                                      rsa=t.rsa_pearson, dll=t.delta_ll_per_obs, group=None, n_params=np.nan)))
    d = pd.concat(rows, ignore_index=True)
    meta = d.dropna(subset=["n_params"]).drop_duplicates(["family", "model_id", "revision"]).set_index(["family", "model_id", "revision"]).n_params
    d["n_params"] = [meta.get((f, m, v), np.nan) for f, m, v in zip(d.family, d.model_id, d.revision)]
    return d


def _fmt_params(n: float) -> str:
    return f"{n/1e9:.0f}B" if n >= 1e9 else f"{n/1e6:.0f}M"


def draw_paper_style(m: pd.DataFrame, track: str, front: pd.DataFrame, out_paths: list[Path], group_corner: str | None = None) -> None:
    """Same layout as scripts/figures/pareto_figures.py (paper Pareto figures); legends in the corner given by LEGEND_CORNER."""
    v = m.n_params[m.n_params > 0].astype(float)
    lo, hi = float(np.log10(v.min())), float(np.log10(v.max()))
    sizes = log_size_map(m.n_params, lo, hi, s_min=25, s_max=400)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for g in GROUP_ORDER:
        mask = (m.group == g).to_numpy()
        if mask.any():
            ax.scatter(m.neural[mask], m.behav[mask], s=sizes[mask], color=GROUP_COLORS[g], alpha=0.6, edgecolors="white",
                       linewidths=0.5, zorder=2, marker="o")
    if len(front) > 1:
        ax.plot(front.neural, front.behav, color="black", lw=1.8, zorder=3, alpha=0.9)
    for _, r in front.iterrows():
        ax.scatter([r.neural], [r.behav], s=90, facecolors="none", edgecolors="black", linewidths=1.8, zorder=4, marker="o")
    corner = LEGEND_CORNER[track]
    top = corner.startswith("upper")
    if top:  # headroom so that the legends in the top-right corner sit above the data
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.28 * (y1 - y0))
    else:  # room on the right for the labels of the crowded end of the front
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + 0.22 * (x1 - x0))
    fig.canvas.draw()
    placed = []  # window extents of the labels already drawn
    cand = [(5, 6), (5, -13), (-5, 6), (-5, -13), (5, 20), (5, -27), (-5, 20), (-5, -27), (5, 34), (5, -41), (-5, 34), (-5, -41),
            (5, 48), (5, -55), (-5, 48), (-5, -55),
            (14, 8), (14, -10), (14, 26), (14, -28), (14, 44), (14, -46), (14, 62), (14, -64), (14, 80), (14, -82)]
    axbox = ax.get_window_extent()
    from matplotlib.transforms import Bbox
    pt = fig.dpi / 72.0

    def box(r, dx, dy, w, h):  # label bbox in display pixels (text anchored at bottom-left / bottom-right of the offset point)
        px, py = ax.transData.transform((r.neural, r.behav))
        x0 = px + dx * pt - (w if dx < 0 else 0)
        return Bbox.from_bounds(x0, py + dy * pt, w, h)

    for _, r in front.sort_values("neural").iterrows():
        t = ax.text(0, 0, r.label, fontsize=7)
        ext = matplotlib.text.Text.get_window_extent(t)
        w, h = ext.width * 1.06, ext.height * 1.25
        t.remove()
        pts = [(r2.neural, r2.behav) for _, r2 in front.iterrows()]
        for dx, dy in cand:
            bb = box(r, dx, dy, w, h)
            if bb.x0 < axbox.x0 or bb.x1 > axbox.x1 or bb.y1 > axbox.y1:
                continue
            if any(bb.overlaps(q) for q in placed):
                continue
            if any(bb.contains(*ax.transData.transform(q)) for q in pts):  # do not cover a front marker
                continue
            placed.append(bb)
            ax.annotate(r.label, (r.neural, r.behav), fontsize=7, color="black", xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx > 0 else "right", va="bottom", zorder=5,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.35", shrinkA=0, shrinkB=4) if abs(dy) > 14 or abs(dx) > 6 else None)
            break
        else:
            raise RuntimeError(f"no free label slot for {r.label}")
    ax.set_xlabel("BrainScore", fontsize=13, color="black")
    ax.set_ylabel("BehavScore", fontsize=13, color="black")
    ax.tick_params(axis="both", labelsize=9.5, colors="black", width=1.4)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.6)
    gh = [Line2D([], [], marker="o", linestyle="None", color=GROUP_COLORS[g], markeredgecolor="white", markersize=8,
                 label=GROUP_DISPLAY.get(g, g)) for g in GROUP_ORDER if (m.group == g).any()]
    sh = [Line2D([], [], marker="o", linestyle="None", markerfacecolor="grey", markeredgecolor="white", alpha=0.65,
                 markersize=float(np.sqrt(log_size_map(pd.Series([r]), lo, hi)[0])), label=_fmt_params(r))
          for r in _SIZE_REFS if lo <= np.log10(r) <= hi]
    # the two legends are stacked in the chosen corner (group legend on the outer side)
    (x_size, y_group, y_size) = (0.70, 1.0, 1.0) if top else (1.0, 0.30, 0.0)  # top: side by side in the headroom; bottom: stacked
    if group_corner == "lower left":  # group legend alone in the bottom-left corner, size legend at the bottom right
        gl = ax.legend(handles=gh, fontsize=8, title_fontsize=8, loc="lower left", frameon=True, title="Model group")
    else:
        gl = ax.legend(handles=gh, fontsize=8, title_fontsize=8, loc=corner, bbox_to_anchor=(1.0, y_group), frameon=True, title="Model group")
    ax.add_artist(gl)
    ax.legend(handles=sh, title="model size", fontsize=8, title_fontsize=8, loc=corner, bbox_to_anchor=(x_size, y_size), frameon=True,
              labelspacing=1.2, borderpad=0.9)
    ax.add_artist(gl)
    fig.tight_layout()
    for pth in out_paths:
        pth.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pth, dpi=600 if pth.suffix == ".png" else None, facecolor="white")
    plt.close(fig)


def build(track: str, score: str = "rsa") -> pd.DataFrame:
    """score = \"rsa\": Pearson RSA / RSA ceiling; \"dll\": held-out dLL / split-half dLL ceiling (both per unit, median over a dataset's units)."""
    if score == "dll":
        return build_dll(track)
    meta = {}
    rows = []
    for ds, (atlas, kind) in NEURAL.items():
        d = load_neural(ds, atlas, kind)
        d = d[(d.track == track) & d.subject.str.startswith("GROUP")]  # subject-averaged units, as in the headline aggregation
        c, key = unit_ceilings(ds)
        d = d.merge(c, on=key, how="inner")
        d["norm_u"] = np.where(d.rsa_ceiling > 0, d.rsa_pearson / d.rsa_ceiling, np.nan)  # each unit divided by its OWN ceiling
        n_units = d.drop_duplicates(key)[["rsa_ceiling"]]
        print(f"  {track} {ds}: {len(n_units)} units with a ceiling, {int((n_units.rsa_ceiling > 0).sum())} usable (ceiling > 0)")
        s = d.groupby(KEYS, as_index=False).agg(norm=("norm_u", "median"), rsa=("rsa_pearson", "median"),
                                                 ceiling=("rsa_ceiling", "median"), n_units=("norm_u", "count"))
        for r in s.itertuples():  # one row per (model, dataset): the dataset's median over units of RSA_u / ceiling_u
            rows.append(dict(dataset=ds, unit="all", family=r.family, model_id=r.model_id, revision=r.revision,
                             rsa=r.rsa, ceiling=r.ceiling, norm=r.norm, n_units=r.n_units, is_neural=True))
    cb, _ = behavioral_ceilings()
    bs = behavioral_scores(track)
    for r in bs.itertuples():
        meta[(r.family, r.model_id, r.revision)] = (r.group, r.n_params)
        rows.append(dict(dataset=r.dataset, unit=r.unit, family=r.family, model_id=r.model_id, revision=r.revision,
                         rsa=r.rsa, ceiling=float(cb[(r.dataset, r.unit)]), is_neural=False))
    d = pd.DataFrame(rows)
    beh = ~d.is_neural
    d.loc[beh, "norm"] = np.where(d.ceiling[beh] > 0, d.rsa[beh] / d.ceiling[beh], np.nan)
    for k in KEYS:
        d[k] = d[k].astype(str)
    gm = {k: v for k, v in meta.items()}
    d["group"] = [_group(f, m, None) for f, m in zip(d.family, d.model_id)]  # current registry grouping
    d["n_params"] = [gm.get((f, m, v), (None, None))[1] for f, m, v in zip(d.family, d.model_id, d.revision)]
    # neural rows carry no group/n_params: fill from the behavioral rows of the same model
    d["track"], d["confounds"] = track, "len_freq"
    d["unit_key"] = d.dataset + "/" + d.unit
    return d


DLL_TABLE = SCR / "rdm_report_std/rdm_last_layer_results.csv"
DLL_CEIL = ROOT / "reports/rsa_summary/rdm_regression/rdm_ceiling_values.csv"


def build_dll(track: str) -> pd.DataFrame:
    """Pareto input for the held-out dLL track: neural datasets = median over units of dLL_u / dLL ceiling_u (units with a non-positive
    ceiling dropped); behavioral measures = full-sample last-layer dLL / split-half dLL ceiling (rdm_regression_figures.apply_ceilings)."""
    _, cd = behavioral_ceilings()
    bs = behavioral_scores(track)
    ceil = np.array([cd[(ds, u)] for ds, u in zip(bs.dataset, bs.unit)])
    bs["norm"] = np.where(ceil > 0, bs.dll / ceil, np.nan)
    bs["is_neural"], bs["unit_key"] = False, bs.dataset + "/" + bs.unit
    bs["group"] = [_group(f, m, None) for f, m in zip(bs.family, bs.model_id)]
    beh = bs[["dataset", "unit", "family", "model_id", "revision", "norm", "group", "n_params", "is_neural", "unit_key"]].copy()
    meta = {(r.family, r.model_id, r.revision): r.n_params for r in beh.itertuples()}
    rows = []
    for ds, (atlas, kind) in NEURAL.items():
        d = load_neural(ds, atlas, kind)
        d = d[(d.track == track) & d.subject.str.startswith("GROUP")]
        c, key = unit_ceilings(ds)
        d = d.merge(c, on=key, how="inner")
        d["norm_u"] = np.where(d.dll_ceiling > 0, d.delta_ll_per_obs / d.dll_ceiling, np.nan)
        u = d.drop_duplicates(key)
        print(f"  dll {track} {ds}: {len(u)} units with a ceiling, {int((u.dll_ceiling > 0).sum())} usable (ceiling > 0)")
        s = d.groupby(KEYS, as_index=False).agg(norm=("norm_u", "median"))
        for r in s.itertuples():
            rows.append(dict(dataset=ds, unit="all", family=r.family, model_id=r.model_id, revision=str(r.revision), norm=r.norm, is_neural=True,
                             unit_key=f"{ds}/all"))
    neu = pd.DataFrame(rows)
    neu["group"] = [_group(f, m, None) for f, m in zip(neu.family, neu.model_id)]
    neu["n_params"] = [meta.get((f, m, v)) for f, m, v in zip(neu.family, neu.model_id, neu.revision)]
    d = pd.concat([beh, neu], ignore_index=True)
    d["track"], d["confounds"] = track, "len_freq"
    return d


def main():
    use_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)
    F.FIG = F.FIGPDF = OUT
    F.RSA_NOTE = ("median over units of last-layer Pearson RSA / that unit's own RSA noise ceiling (sqrt of Spearman-Brown corrected split-half RDM "
                  "reliability; units with a non-positive ceiling dropped, neural datasets summarized by the median over their units). Neural: Pereira and Tuckute language ROIs only, Podcast language parcels, Broderick/MEG-MASC channels, 3 N400 sets; Fedorenko excluded.")
    F.VARIANT_TEXT["len_freq"] = "Pearson RSA"
    allrows, fronts = [], []
    for score, out_dir, stem in (("rsa", OUT, "rsa"), ("dll", OUT.parent / "dll_pareto", "dll")):
        out_dir.mkdir(parents=True, exist_ok=True)
        allrows = []
        for track in ("embedding", "surprisal"):
            d = build(track, score)
            d.to_csv(out_dir / f"{stem}_ratios_{track}.csv", index=False)
            m = F.composites(d, track, "len_freq")
            front = F.pareto_front(m, "neural", "behav")
            draw_paper_style(m, track, front, [out_dir / f"{track}_pareto_{stem}.png", out_dir / f"{track}_pareto_{stem}.pdf",
                                               PAPER / f"figure_{stem}_pareto_{track}.pdf"],
                             group_corner="lower left" if (score, track) == ("rsa", "surprisal") else None)
            allrows.append(m.assign(track=track, on_front=m.index.isin(front.index)))
            print(f"{score} {track}: {len(m)} models, front {len(front)} ({m.attrs['n_neural_units']} neural, {m.attrs['n_behav_units']} behavioral units)")
            print(front.sort_values("behav", ascending=False)[["label", "group", "n_params", "neural", "behav"]].round(4).to_string(index=False))
        pd.concat(allrows).to_csv(out_dir / f"{stem}_pareto_scores.csv", index=False)


if __name__ == "__main__":
    main()
