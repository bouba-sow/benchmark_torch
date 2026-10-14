#!/usr/bin/env python3
"""RSA-correlation ceilings (split-half reliability of the item-mean RDM) of a SUBSET of the behavioral datasets, on the shared RDMs
read through RDM_RSA_DIR (e.g. the 1,000,000-pair ones). Same computation as rdm_ceiling_rsa_behavioral.one(); writes --out CSV.

    RDM_RSA_DIR=$SCRATCH/rdm_behavior_1m python rdm_ceiling_rsa_behavioral_subset.py geco onestop --out $SCRATCH/rdm_ceiling_1m/rsa_ceiling_geco_onestop.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.analysis import rdm_ceiling_rsa_behavioral as M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        for m in M.R.base.DATASET_MEASURES[ds]:
            rows.append(M.one(ds, m))
            print(rows[-1], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
