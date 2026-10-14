#!/usr/bin/env python3
"""Mean/std of last-layer states over ALL words of the stimulus sentences (Nieuwland 2018 item set).

The production N400 embedding vector is the critical noun's state, and the current standardization uses the mean/std of
those 160 noun vectors. Here the statistics come from every word of every sentence (context words + noun, sentence-initial
word included), using exactly the production forward pass (text = ``context + " " + noun``, same tokenizer/model loading);
a word's state is the mean over the tokens overlapping its character span (the production rule for the noun).
The noun vectors recomputed here are saved too (``noun_states``) and checked against the production cache.

Output: $SCRATCH/sentence_word_stats/nieuwland/<family>/<model_dir>/<revision>.npz (mu, sigma, n_words, noun_states_max_abs_diff).
Array task i of N handles every N-th model (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT).
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/run/embedding/neural"))

import numpy as np
import pandas as pd

from neuroai.analysis.benchmark_common import lookup_model_meta
from neuroai.benchmarks.eeg_erp import _feature_cache_path, _items_hash
from neuroai.models.registry import LPP_FAMILIES
from scripts.analysis import n400_item_mean_rsa as M
from scripts.analysis import n400_paper_rsa as P

SCR = Path(os.environ.get("SCRATCH", "."))
OUT = SCR / "sentence_word_stats/nieuwland"


def spec_for(family, model_id):
    return next(s for s in LPP_FAMILIES[family] if model_id in (s.model_id, *(s.revision_model_ids or {}).values()))


def word_states(texts, nouns, family, task, device, batch_size=8):
    """Per sentence: (n_words, D) last-layer state of each whitespace word; returns list of arrays."""
    import torch
    from neuroai.features.llm import _get_cached_causal_lm
    name, rev = task.load_model_id, (None if (task.revision or "main") == "main" else task.revision)
    if "cbr-rnn" in name.lower() or "cbr_rnn" in name.lower():
        from neuroai.features.cbr_rnn import contextual_word_embeddings_cbr_rnn
        return [np.asarray(contextual_word_embeddings_cbr_rnn(t.split(), name, device=device)) for t in texts]
    spec = spec_for(family, task.model_id)
    kw = {"trust_remote_code": True} if spec.trust_remote_code else {}
    tok, model = _get_cached_causal_lm(name, revision=rev, device=device, load_kw=kw,
                                       tokenizer_no_revision=bool(rev) and "pythia" in name.lower(), n_params=spec.n_params)
    max_len = getattr(model.config, "max_position_embeddings", 2048)
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            bt = texts[s:s + batch_size]
            enc = tok(bt, return_tensors="pt", padding=True, truncation=True, max_length=max_len, return_offsets_mapping=True)
            off = enc.pop("offset_mapping")
            enc = {k: v.to(device) for k, v in enc.items()}
            o = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            hs = None if isinstance(o, tuple) else o.hidden_states
            if hs is None:  # hook path (models that ignore output_hidden_states), as in extract_sentence_features
                got, hooks, inner = [], [], getattr(model, "model", model)
                f = lambda _m, _i, _o: got.append(_o.detach() if isinstance(_o, torch.Tensor) else _o[0].detach())  # noqa: E731
                if getattr(inner, "embeddings", None) is not None:
                    hooks.append(inner.embeddings.register_forward_hook(f))
                hooks += [l.register_forward_hook(f) for l in getattr(inner, "layers", [])]
                model(**enc, use_cache=False)
                [h.remove() for h in hooks]
                hs = tuple(got)
            last = hs[-1]
            for bi, text in enumerate(bt):
                spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
                attn = enc["attention_mask"][bi]
                rows = []
                for ws, we in spans:
                    idx = [t for t in range(off.shape[1]) if attn[t].item() and int(off[bi, t, 1]) > ws and int(off[bi, t, 0]) < we]
                    if not idx:
                        idx = [int(attn.sum().item()) - 1]
                    rows.append(last[bi, idx, :].mean(dim=0).float().cpu().numpy())
                out.append(np.stack(rows))
    return out


def main():
    device = "cuda"
    items = M.Nieuwland().items
    ctx, nouns = items["sentence_context"].tolist(), items["noun"].tolist()
    texts = [f"{c} {n}" for c, n in zip(ctx, nouns)]
    ih = _items_hash(ctx, nouns)
    tasks = P.registry_tasks()
    k, n = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    only = os.environ.get("ONLY_FAMILY")
    for family, task in tasks[k::n]:
        if only and family != only:
            continue
        rev = task.revision or "main"
        out = OUT / family / task.model_dir_safe / f"{rev.replace('/', '_')}.npz"
        if out.is_file():
            continue
        cache = _feature_cache_path(M.NIEUWLAND_CACHE, task.load_model_id, None if rev == "main" else rev, "all", ih)
        if not cache.is_file():
            continue  # model not part of the Nieuwland sweep
        t0 = time.time()
        try:
            per = word_states(texts, nouns, family, task, device)
            allw = np.concatenate(per).astype(np.float64)
            if not np.isfinite(allw).all():
                raise ValueError("non-finite states")
            noun = np.stack([p[-1] for p in per])
            f = np.load(cache)["features"]
            ref = f if f.ndim == 2 else f[:, -1, :]
            diff = float(np.abs(noun - ref).max() / (np.abs(ref).max() + 1e-12))
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, mu=allw.mean(0), sigma=allw.std(0), n_words=len(allw), noun_rel_diff=diff)
            print(f"OK {family}/{task.model_id}@{rev} words={len(allw)} noun_rel_diff={diff:.2e} {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"ERROR {family}/{task.model_id}@{rev}: {e!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
