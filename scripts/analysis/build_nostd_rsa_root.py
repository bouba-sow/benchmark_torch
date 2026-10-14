#!/usr/bin/env python3
"""Mirror of outputs/<dataset>/pearson_rsa holding the RAW (non-standardized) embedding RSA summaries.

The standardization switch (2026-10-02/03) overwrote the embedding RSA summaries in outputs/; the raw ones were kept in
$SCRATCH/rsa_std_backup_old (tuckute: $SCRATCH/tuckute_rsa_backup_old; N400: item-mean ``model_summary_unstandardized.csv``).
This lays them out in the structure scripts/analysis/build_rsa_report.py reads, next to symlinks to the (unchanged)
surprisal summaries, so the report can be rebuilt with  RSA_OUTPUTS_ROOT=<root>  RSA_REPORT_OUT=<folder>.

    .venv/bin/python scripts/analysis/build_nostd_rsa_root.py [--root $SCRATCH/rsa_nostd_root]
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
B, O = S / "rsa_std_backup_old", ROOT / "outputs"

# (dataset, destination below <root>/<dataset>/pearson_rsa, source file)
COPIES = [("pereira", "model_summary.csv", B / "pereira/model_summary.csv"),
          ("tuckute", "model_summary.csv", S / "tuckute_rsa_backup_old/model_summary.csv"),
          ("fedorenko2016", "model_summary.csv", B / "fedorenko2016/model_summary.csv"),
          ("broderick", "embedding/model_summary.csv", B / "broderick/model_summary.csv"),
          ("megmasc", "embedding/model_summary.csv", B / "megmasc/model_summary.csv"),
          ("podcast", "embedding/lagwise/model_summary.csv", B / "podcast/model_summary.csv"),
          ("podcast", "embedding/lagwise/model_summary_by_lag.csv", B / "podcast/model_summary_by_lag.csv")]
for ds in ("nieuwland", "szewczyk_f2007", "szewczyk_h2019"):
    COPIES.append((ds, "embedding/model_summary.csv", O / ds / "pearson_rsa/embedding/model_summary_unstandardized.csv"))
for ds, ms, sub in [("provo", ["fp", "gp", "sp"], ""), ("meco", ["fp", "gp", "sp"], ""), ("geco", ["fp", "gp"], "embedding"),
                    ("onestop", ["fp", "gp", "sp"], "embedding"), ("naturalstories", ["spr"], "embedding"),
                    ("ucl_reading", ["spr"], "embedding"), ("zuco_reading", ["nr", "sr", "tsr"], "embedding")]:
    for m in ms:
        COPIES.append((ds, f"{sub}/{m}/model_summary.csv".lstrip("/"), B / ds / m / "model_summary.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=S / "rsa_nostd_root")
    root = ap.parse_args().root
    for ds, dest, src in COPIES:
        if not src.is_file():
            raise FileNotFoundError(src)
        d = root / ds / "pearson_rsa" / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
    for ds in {c[0] for c in COPIES}:  # surprisal summaries are unaffected: link to the real ones
        real = O / ds / "pearson_rsa" / "surprisal"
        link = root / ds / "pearson_rsa" / "surprisal"
        if real.is_dir() and not link.exists():
            link.symlink_to(real)
    print(f"{len(COPIES)} embedding summaries + surprisal links -> {root}")


if __name__ == "__main__":
    main()
