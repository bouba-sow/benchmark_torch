import sys, os
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
print("HF_HOME:", os.environ.get("HF_HOME"), flush=True)
from transformers import AutoTokenizer
from neuroai.lpp_llm_features import (
    load_story_text, build_lbg_word_stream, simplify_word, do_word_match,
    LBG_PHANTOM_CONTRACTION_EXPANSIONS,
)
from neuroai.datasets.le_petit_prince import load_word_information_table, words_for_section

bids_root = str(_P(__file__).resolve().parents[2] / "data/lpp_fmri")
lang = "en"
section_id = 2
model_name = "AntonV/mamba2-1.3b-hf"
maxlen = 2048
stride_delta = 64
stride = maxlen - stride_delta

tok = AutoTokenizer.from_pretrained(model_name)

lbg_zip = str(_P(__file__).resolve().parents[2] / "data/lpp_fmri_extra/lbg_story_text/lpp_en_text.zip")
fulltext_run = load_story_text(
    bids_root=bids_root, lang=lang, section=section_id, lbg_story_text_zip=lbg_zip
).replace("\n", " ").replace("\r", " ")

words_full = load_word_information_table(bids_root, lang, apply_lbg_fixes=True)
sec_df = words_for_section(words_full, section_id)
wc = sec_df["word"]
def _keep(w):
    return isinstance(w, str) and w.strip() != "" and w != " "
sec_lm = sec_df.loc[wc.map(_keep)].copy()

word_list, char_to_word_idx, n_words_after_cn = build_lbg_word_stream(sec_lm, lang)
print("n words in stream:", len(word_list))

overflow_enc = tok(
    fulltext_run,
    return_tensors="pt",
    return_offsets_mapping=True,
    truncation=True,
    padding=True,
    max_length=maxlen,
    return_overflowing_tokens=True,
    stride=stride,
)
om = overflow_enc["offset_mapping"]
am_enc = overflow_enc["attention_mask"]
seq_len = int(overflow_enc["input_ids"].shape[1])
n_chunks = int(overflow_enc["input_ids"].shape[0])
print("n_chunks:", n_chunks, "seq_len:", seq_len, "stride:", stride)

def _batch_start_token(batch_idx):
    am_row = am_enc[batch_idx]
    real_sum = int(am_row.sum().item())
    first_real = int(am_row.nonzero()[0, 0].item()) if real_sum > 0 else 0
    return stride + first_real

idx_batch = 0
idx_token = 0
idx_word_to_idx_token = []

TARGET = 1686
LO, HI = 1675, 1690

for idx_word, raw_w in enumerate(word_list):
    w_key = simplify_word(raw_w, lang)
    if idx_token == seq_len:
        idx_batch += 1
        idx_token = _batch_start_token(idx_batch)
    i_start = int(om[idx_batch, idx_token, 0].item())
    i_stop = int(om[idx_batch, idx_token, 1].item())
    n_try = 0
    verbose = LO <= idx_word <= HI
    if verbose:
        print(f"word {idx_word}: {raw_w!r} key={w_key!r} START batch={idx_batch} tok#{idx_token} "
              f"abs_off=({i_start},{i_stop}) text={fulltext_run[i_start:i_stop]!r}")
    max_try = 3 if w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS else 20
    while n_try < max_try and not do_word_match(w_key, fulltext_run[i_start:i_stop], lang):
        idx_token += 1
        if idx_token == seq_len:
            idx_batch += 1
            idx_token = _batch_start_token(idx_batch)
        i_start = int(om[idx_batch, idx_token, 0].item())
        i_stop = int(om[idx_batch, idx_token, 1].item())
        n_try += 1
        if verbose:
            print(f"   try {n_try}: batch={idx_batch} tok#{idx_token} abs_off=({i_start},{i_stop}) "
                  f"text={fulltext_run[i_start:i_stop]!r}")
    if n_try == max_try:
        print(f"HARD FAIL at word {idx_word}: {raw_w!r}")
        if idx_word_to_idx_token:
            pb, pt = idx_word_to_idx_token[-1]
            p_start = int(om[pb, pt, 0].item())
            print("raw text window around last good match:", repr(fulltext_run[max(0, p_start-80):p_start+200]))
        continue
    idx_word_to_idx_token.append((idx_batch, idx_token))
    if lang != "cn":
        idx_token += 1

print("done. total words matched:", len(idx_word_to_idx_token), "/", len(word_list))
