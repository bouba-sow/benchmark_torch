import sys, os
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer
from neuroai.lpp_llm_features import (
    load_story_text, build_lbg_word_stream, simplify_word, do_word_match,
    LBG_PHANTOM_CONTRACTION_EXPANSIONS, _ensure_pad_token_for_batches,
)
from neuroai.datasets.le_petit_prince import load_word_information_table, words_for_section
from neuroai.models.registry import registry_model_configs

BIDS_ROOT = str(_P(__file__).resolve().parents[2] / "data/lpp_fmri")
LANG = "en"
N_SECTIONS = 9
MAXLEN = 2048
STRIDE_DELTA = 64
STRIDE = MAXLEN - STRIDE_DELTA
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
    try:
        sec_df = words_for_section(words_full, sec)
        sec_lm = sec_df.loc[sec_df["word"].map(_keep)].copy()
        wl, _, _ = build_lbg_word_stream(sec_lm, LANG)
        section_word_lists[sec] = wl
        section_texts[sec] = load_story_text(
            bids_root=BIDS_ROOT, lang=LANG, section=sec, lbg_story_text_zip=LBG_ZIP
        ).replace("\n", " ").replace("\r", " ")
    except Exception as e:
        print(f"SECTION {sec}: failed to build word stream/text: {e!r}")

def check_alignment(fulltext_run, word_list, tok):
    enc = tok(
        fulltext_run, return_tensors="pt", return_offsets_mapping=True,
        truncation=True, padding=True, max_length=MAXLEN,
        return_overflowing_tokens=True, stride=STRIDE,
    )
    om = enc["offset_mapping"]
    am = enc["attention_mask"]
    seq_len = int(enc["input_ids"].shape[1])

    def batch_start(bidx):
        row = am[bidx]
        s = int(row.sum().item())
        fr = int(row.nonzero()[0, 0].item()) if s > 0 else 0
        return STRIDE + fr

    idx_batch, idx_token = 0, 0
    n_matched = 0
    fails = []
    idx_word_to_idx_token = []
    n_chunks_total = int(enc["input_ids"].shape[0])
    for idx_word, raw_w in enumerate(word_list):
        w_key = simplify_word(raw_w, LANG)
        if idx_token == seq_len:
            idx_batch += 1
            if idx_batch >= n_chunks_total:
                fails.append((idx_word, raw_w, "ran past last chunk"))
                break
            idx_token = batch_start(idx_batch)
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
            fails.append((idx_word, raw_w, "ran past last chunk (inner)"))
            break
        if n_try == max_try:
            if idx_word_to_idx_token and w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS:
                prev_batch, prev_token = idx_word_to_idx_token[-1]
                alias_batch, alias_token = prev_batch, prev_token + 1
                if alias_token == seq_len:
                    alias_batch += 1
                    if alias_batch >= n_chunks_total:
                        fails.append((idx_word, raw_w, "ran past last chunk (phantom alias)"))
                        break
                    alias_token = batch_start(alias_batch)
                idx_word_to_idx_token.append((alias_batch, alias_token))
                idx_batch, idx_token = alias_batch, alias_token + 1
                n_matched += 1
                continue
            fails.append((idx_word, raw_w, "hard fail"))
            idx_token += 1
            continue
        idx_word_to_idx_token.append((idx_batch, idx_token))
        n_matched += 1
        if LANG != "cn":
            idx_token += 1
    return n_matched, len(word_list), fails, int(enc["input_ids"].shape[0])


seen_tokenizers = {}
total_bad = 0
for fam in FAMILIES:
    try:
        cfgs = registry_model_configs(fam)
    except Exception as e:
        print(f"FAMILY {fam}: could not load registry configs: {e!r}")
        continue
    trc_by_id = {}
    for c in cfgs:
        trc_by_id[c["model_id"]] = trc_by_id.get(c["model_id"], False) or bool(c.get("trust_remote_code", False))
    model_ids = sorted(trc_by_id)
    for model_id in model_ids:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trc_by_id[model_id])
            _ensure_pad_token_for_batches(tok)
        except Exception as e:
            print(f"MODEL {fam}/{model_id}: tokenizer load failed: {e!r}")
            continue
        sig = (tok.__class__.__name__, tok.padding_side, tuple(sorted(tok.get_vocab().items()))[:1])
        # Cheap dedup key: class + padding_side + vocab size (good enough; false
        # negatives just mean a little redundant work, not missed coverage).
        dedup_key = (tok.__class__.__name__, tok.padding_side, len(tok))
        if dedup_key in seen_tokenizers:
            continue
        seen_tokenizers[dedup_key] = model_id
        any_bad = False
        for sec, wl in section_word_lists.items():
            n_matched, n_total, fails, n_chunks = check_alignment(section_texts[sec], wl, tok)
            if fails:
                any_bad = True
                total_bad += 1
                print(f"BAD fam={fam} model={model_id} padding_side={tok.padding_side} "
                      f"section={sec} n_chunks={n_chunks} matched={n_matched}/{n_total} "
                      f"fails={fails[:3]}")
        status = "BAD" if any_bad else "ok"
        print(f"{status}: fam={fam} model={model_id} padding_side={tok.padding_side} "
              f"tokenizer_class={tok.__class__.__name__}")

print()
print("=== SUMMARY ===")
print("unique tokenizers tested:", len(seen_tokenizers))
print("total (model,section) failures:", total_bad)
