import sys, os
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer, AutoConfig
from neuroai.lpp_llm_features import (
    load_story_text, build_lbg_word_stream, simplify_word, do_word_match,
    LBG_PHANTOM_CONTRACTION_EXPANSIONS, _ensure_pad_token_for_batches,
)
from neuroai.datasets.le_petit_prince import load_word_information_table, words_for_section
from neuroai.models.registry import registry_model_configs

BIDS_ROOT = str(_P(__file__).resolve().parents[2] / "data/lpp_fmri")
LANG = "en"
N_SECTIONS = 9
STRIDE_DELTA = 64
LBG_ZIP = str(_P(__file__).resolve().parents[2] / "data/lpp_fmri_extra/lbg_story_text/lpp_en_text.zip")

FAMILIES = [
    "gpt2", "gptneo", "opt", "mamba", "mamba2", "pythia", "olmo2", "qwen3",
    "llama32", "mistral", "stickbreaking", "alibi", "forgetting",
    "hard_2gram", "hard_3gram", "hard_5gram", "babylm",
]

words_full = load_word_information_table(BIDS_ROOT, LANG, apply_lbg_fixes=True)

def _keep(w):
    return isinstance(w, str) and w.strip() != "" and w != " "

section_word_lists = {}
section_texts = {}
for sec in range(1, N_SECTIONS + 1):
    sec_df = words_for_section(words_full, sec)
    sec_lm = sec_df.loc[sec_df["word"].map(_keep)].copy()
    wl, _, _ = build_lbg_word_stream(sec_lm, LANG)
    section_word_lists[sec] = wl
    section_texts[sec] = load_story_text(
        bids_root=BIDS_ROOT, lang=LANG, section=sec, lbg_story_text_zip=LBG_ZIP
    ).replace("\n", " ").replace("\r", " ")


def run_alignment(fulltext_run, word_list, tok, maxlen, stride, use_fix):
    enc = tok(
        fulltext_run, return_tensors="pt", return_offsets_mapping=True,
        truncation=True, padding=True, max_length=maxlen,
        return_overflowing_tokens=True, stride=stride,
    )
    om = enc["offset_mapping"]
    am = enc["attention_mask"]
    seq_len = int(enc["input_ids"].shape[1])
    n_chunks_total = int(enc["input_ids"].shape[0])

    def batch_start(bidx):
        if not use_fix:
            return stride
        row = am[bidx]
        s = int(row.sum().item())
        fr = int(row.nonzero()[0, 0].item()) if s > 0 else 0
        return stride + fr

    idx_batch, idx_token = 0, 0
    mapping = []
    idx_word_to_idx_token = []
    for idx_word, raw_w in enumerate(word_list):
        w_key = simplify_word(raw_w, LANG)
        if idx_token == seq_len:
            idx_batch += 1
            if idx_batch >= n_chunks_total:
                mapping.append(None)
                return mapping, n_chunks_total
            idx_token = batch_start(idx_batch)
        if idx_batch >= om.shape[0] or idx_token >= om.shape[1]:
            mapping.append(None)
            continue
        i_start = int(om[idx_batch, idx_token, 0].item())
        i_stop = int(om[idx_batch, idx_token, 1].item())
        n_try = 0
        max_try = 3 if w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS else 20
        ran_out = False
        while n_try < max_try and not do_word_match(w_key, fulltext_run[i_start:i_stop], LANG):
            idx_token += 1
            if idx_token == seq_len:
                idx_batch += 1
                if idx_batch >= n_chunks_total:
                    ran_out = True
                    break
                idx_token = batch_start(idx_batch)
            i_start = int(om[idx_batch, idx_token, 0].item())
            i_stop = int(om[idx_batch, idx_token, 1].item())
            n_try += 1
        if ran_out:
            mapping.append(None)
            break
        if n_try == max_try:
            if idx_word_to_idx_token and w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS:
                prev_batch, prev_token = idx_word_to_idx_token[-1]
                alias_batch, alias_token = prev_batch, prev_token + 1
                if alias_token == seq_len:
                    alias_batch += 1
                    if alias_batch >= n_chunks_total:
                        mapping.append(None)
                        break
                    alias_token = batch_start(alias_batch)
                idx_word_to_idx_token.append((alias_batch, alias_token))
                mapping.append((alias_batch, alias_token))
                idx_batch, idx_token = alias_batch, alias_token + 1
                continue
            mapping.append(("FAIL", idx_word, raw_w))
            idx_token += 1
            continue
        idx_word_to_idx_token.append((idx_batch, idx_token))
        mapping.append((idx_batch, idx_token))
        if LANG != "cn":
            idx_token += 1
    return mapping, n_chunks_total


seen = {}
results = []
for fam in FAMILIES:
    try:
        cfgs = registry_model_configs(fam)
    except Exception as e:
        print(f"FAMILY {fam}: registry error {e!r}")
        continue
    trc_by_id = {}
    for c in cfgs:
        trc_by_id[c["model_id"]] = trc_by_id.get(c["model_id"], False) or bool(c.get("trust_remote_code", False))
    for model_id in sorted(trc_by_id):
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trc_by_id[model_id])
            _ensure_pad_token_for_batches(tok)
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trc_by_id[model_id])
        except Exception as e:
            print(f"MODEL {fam}/{model_id}: load failed: {e!r}")
            continue
        maxlen = int(getattr(cfg, "max_position_embeddings", None) or 2048)
        stride = maxlen - STRIDE_DELTA
        if stride <= 0:
            print(f"MODEL {fam}/{model_id}: invalid stride (maxlen={maxlen}) -- skipped")
            continue
        key = (tok.__class__.__name__, tok.padding_side, len(tok), maxlen)
        if key in seen:
            continue
        seen[key] = model_id

        any_diff = False
        any_fail_old = False
        any_fail_new = False
        n_chunks_any = 1
        for sec, wl in section_word_lists.items():
            m_old, n_chunks = run_alignment(section_texts[sec], wl, tok, maxlen, stride, use_fix=False)
            m_new, _ = run_alignment(section_texts[sec], wl, tok, maxlen, stride, use_fix=True)
            n_chunks_any = max(n_chunks_any, n_chunks)
            if any(isinstance(x, tuple) and x and x[0] == "FAIL" for x in m_old):
                any_fail_old = True
            if any(isinstance(x, tuple) and x and x[0] == "FAIL" for x in m_new):
                any_fail_new = True
            if m_old != m_new:
                any_diff = True
                n_diff = sum(1 for a, b in zip(m_old, m_new) if a != b)
                print(f"DIVERGES fam={fam} model={model_id} section={sec} n_chunks={n_chunks} "
                      f"n_diff_words={n_diff}/{len(wl)}")
        status = "SILENT-CORRUPTION-RISK" if (any_diff and not any_fail_old) else (
            "OLD-CODE-CRASHED" if any_fail_old else "SAFE-IDENTICAL"
        )
        print(f"{status}: fam={fam} model={model_id} padding_side={tok.padding_side} "
              f"maxlen={maxlen} max_n_chunks_any_section={n_chunks_any} "
              f"diverges={any_diff} old_crashed={any_fail_old} new_crashed={any_fail_new}")

print()
print("=== unique (tokenizer,maxlen) configs tested:", len(seen), "===")
