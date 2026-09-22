import sys, os
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from pathlib import Path
from neuroai.models.registry import registry_model_configs

ROOT = (_P(__file__).resolve().parents[2] / "outputs_jeanzay/lpp/embedding_sweep")

FAMILIES = [
    "pythia", "olmo2", "opt", "gpt2", "gptneo", "mamba", "qwen3", "llama32",
    "mistral", "stickbreaking", "alibi", "forgetting",
    "hard_2gram", "hard_3gram", "hard_5gram", "babylm",
]

def sanitize(model_id: str) -> str:
    return model_id.replace("/", "_")

for fam in FAMILIES:
    cfgs = registry_model_configs(fam)
    print(f"\n=== {fam} ({len(cfgs)} models) ===")
    for c in cfgs:
        model_id = c["model_id"]
        san = sanitize(model_id)
        # results can live at top-level (ROOT/san) or nested (ROOT/fam/san)
        candidates = [ROOT / san, ROOT / fam / san]
        found_csvs = []
        for cand in candidates:
            found_csvs.extend(cand.rglob("lpp_checkpoint_scores.csv"))
        if not found_csvs:
            print(f"  MISSING  {model_id}")
            continue
        # expected number of checkpoints
        expected = None
        if c.get("steps"):
            expected = len(c["steps"])
        elif c.get("revisions"):
            expected = len(c["revisions"])
        else:
            expected = 1
        # count non-header rows per file, sum across checkpoint subdirs (each file is
        # usually 1 data row for a single-checkpoint run, or could be a rolled-up file)
        total_rows = 0
        for f in found_csvs:
            try:
                with open(f) as fh:
                    n = sum(1 for _ in fh) - 1
                total_rows += max(n, 0)
            except Exception:
                pass
        n_files = len(found_csvs)
        flag = "OK" if n_files >= expected else "PARTIAL"
        print(f"  {flag:8s} {model_id}: {n_files} checkpoint-dir(s) w/ scores "
              f"(expected {expected}), {total_rows} total data row(s)")
