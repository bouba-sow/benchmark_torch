#!/usr/bin/env python3
"""One model at a time: repair audited embedding caches, with explicit postchecks."""
from __future__ import annotations
import argparse
import importlib
import json
import logging
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
log = logging.getLogger("embedding_repair")


def cache_files(ds, model, revision, cache_root):
    safe = model.replace("/", "_")
    rev = revision.replace("/", "_").replace("-", "_")
    if ds == "pereira":
        return {0: cache_root / "pereira_embed_features" / f"{safe}__{rev}__sentence_mean_cbrrnn.npz"}
    if ds == "tuckute":
        return {0: ROOT / "outputs_jeanzay/tuckute/embedding_sweep/.embed_cache" / safe / revision / "embeddings.npz"}
    if ds == "szewczyk":
        files = list((cache_root / "szewczyk_features").glob(f"{safe}_{revision}_all_*.npz"))
        return {i: f for i, f in enumerate(files)}
    directory = "zuco_reading_features/nr" if ds == "zuco_reading_nr" else ds + "_features"
    roots = [cache_root / directory]
    if ds == "naturalstories":
        roots.insert(0, ROOT / ".cache/naturalstories_features")
    pattern = (f"{safe}__en__sec*__ctx48__cbrrnn.npz" if model.startswith("cclark/cbr-rnn")
               else f"{safe}__{rev}__en__sec*__all.npz")
    found = {}
    for root in roots:
        for path in root.glob(pattern):
            match = re.search(r"__sec(\d+)__", path.name)
            if match:
                found.setdefault(int(match[1]), path)
    return found


def validate(ds, entry, cache_root):
    """Require finalized readable archives and all expected sections, not a prefix hit."""
    import numpy as np
    paths = cache_files(ds, entry["model_id"], entry["revision"], cache_root)
    required = set(range(1, 362)) if ds == "ucl_reading" else (
        set(range(1, 11)) if ds == "naturalstories" else {0})
    missing = required - paths.keys()
    if missing:
        raise FileNotFoundError(f"{ds}: {len(missing)} missing sections; first={sorted(missing)[:8]}")
    for section in sorted(required):
        path = paths[section]
        with zipfile.ZipFile(path) as archive:
            with archive.open("features.npy" if ds != "tuckute" else "embeddings.npy") as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError(f"Unsupported NPY version {version}: {path}")
                header_size = member.tell()
            info = archive.getinfo("features.npy" if ds != "tuckute" else "embeddings.npy")
            if info.file_size != header_size + int(np.prod(shape)) * dtype.itemsize:
                raise ValueError(f"Truncated array: {path}")
            if len(shape) not in (2, 3) or min(shape) <= 0 or shape[-1] < 2:
                raise ValueError(f"Invalid embedding shape {shape}: {path}")
            expected_rows = {"broderick": 11419, "podcast": 5136, "zuco_reading_nr": 6588, "pereira": 627}.get(ds)
            if expected_rows is not None and shape[0] != expected_rows:
                raise ValueError(f"Incomplete rows {shape[0]} != {expected_rows}: {path}")
    return [str(paths[i]) for i in sorted(required)]


def invoke(module, argv):
    previous = sys.argv
    try:
        sys.argv = [module.__file__, *argv]
        module.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError(f"Extractor exited {exc.code}") from exc
    finally:
        sys.argv = previous


def extract(ds, entry, modules, cache_root):
    model, family, idx = entry["model_id"], entry["family"], entry["array_index"]
    reg = ["--from-registry", "--family", family, "--array-index", str(idx), "--device", "cuda"]
    if ds in {"ucl_reading", "zuco_reading_nr", "broderick"}:
        kind = "neural" if ds == "broderick" else "behavioral"
        flag = "--extract_only" if ds == "zuco_reading_nr" else "--activations_only"
        invoke(modules[ds], ["--config", f"configs/embedding/{kind}/{ds}.yaml", *reg, flag])
    elif ds == "pereira":
        mod = modules[ds]
        cfg = mod.load_config("configs/embedding/neural/pereira.yaml")
        bench = mod._build_benchmark_config(cfg, model_id=model, revision=entry["revision"],
            n_params=entry["n_params"], output_dir=cache_root / "pereira_embed_features",
            layer_stride=1, device="cuda", feature_cache_dir=str(cache_root / "pereira_embed_features"))
        mod.PereiraEmbedBenchmark(bench).extract_and_cache_hidden_states()
    elif ds == "tuckute":
        mod = modules[ds]
        cfg = mod.load_config("configs/embedding/neural/tuckute.yaml")
        sentences, _, _ = mod.TuckuteDataset(cfg["data"]["data_root"]).get_condition("all")
        mod.extract_embeddings(model, list(sentences), revision="main", device="cuda",
            n_params=entry["n_params"], cache_dir=ROOT / "outputs_jeanzay/tuckute/embedding_sweep/.embed_cache")
    else:
        raise RuntimeError(f"Previously present cache is now missing for {ds}/{model}; audit required")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="scripts/tools/rsa_embedding_repair.json")
    p.add_argument("--status", default="outputs_jeanzay/logs/rsa_embedding_repair_status.json")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cache_root = Path(os.environ["BENCHMARK_CACHE_ROOT"])
    work = json.loads(Path(a.manifest).read_text())["models"]
    modules = {ds: importlib.import_module("scripts.run.embedding." + name) for ds, name in {
        "ucl_reading": "behavioral.ucl_reading", "zuco_reading_nr": "behavioral.zuco_reading",
        "broderick": "neural.broderick", "pereira": "neural.pereira", "tuckute": "neural.tuckute"}.items()}
    from neuroai.features import llm, cbr_rnn
    clear = llm.clear_hf_model_cache
    for mod in modules.values():
        if hasattr(mod, "clear_hf_model_cache"):
            mod.clear_hf_model_cache = lambda: None
    status_path = Path(a.status)
    prior = {}
    if status_path.exists():
        try:
            prior_rows = json.loads(status_path.read_text())
            prior = {(row.get("dataset"), row.get("model_id")): row for row in prior_rows}
        except (json.JSONDecodeError, TypeError):
            log.warning("Ignoring unreadable prior status file %s", status_path)
    statuses = []
    failed = False
    for entry in work:
        log.info("MODEL %s: %s", entry["model_id"], entry["datasets"])
        try:
            for ds in entry["datasets"]:
                result = {"model_id": entry["model_id"], "dataset": ds}
                try:
                    old = prior.get((ds, entry["model_id"]), {})
                    if old.get("status") in {"already_cached", "extracted_and_verified"}:
                        try:
                            paths = validate(ds, entry, cache_root)
                            result.update(status="already_cached", files=len(paths), resumed=True)
                            statuses.append(result)
                            out = Path(a.status)
                            out.parent.mkdir(parents=True, exist_ok=True)
                            tmp = out.with_suffix(".tmp")
                            tmp.write_text(json.dumps(statuses, indent=2) + "\n")
                            os.replace(tmp, out)
                            log.info("PAIR RESULT %s", result)
                            continue
                        except (FileNotFoundError, ValueError, zipfile.BadZipFile, KeyError):
                            log.info("Previously successful cache is incomplete; repairing %s / %s", entry["model_id"], ds)
                    try:
                        paths = validate(ds, entry, cache_root)
                        result["status"] = "already_cached"
                    except (FileNotFoundError, ValueError, zipfile.BadZipFile, KeyError):
                        log.info("Extracting %s / %s", entry["model_id"], ds)
                        extract(ds, entry, modules, cache_root)
                        paths = validate(ds, entry, cache_root)
                        result["status"] = "extracted_and_verified"
                    result["files"] = len(paths)
                except Exception as exc:
                    log.exception("FAILED %s / %s", entry["model_id"], ds)
                    failed = True
                    result.update(status="failed", error=str(exc))
                statuses.append(result)
                out = Path(a.status)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                tmp.write_text(json.dumps(statuses, indent=2) + "\n")
                os.replace(tmp, out)
                log.info("PAIR RESULT %s", result)
        finally:
            clear()
            cbr_rnn._cbr_slot_model = None
            cbr_rnn._cbr_slot_tokenizer = None
            cbr_rnn._cbr_slot_key = None
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
