import sys, os
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer
from neuroai.lpp_llm_features import load_story_text

bids_root = str(_P(__file__).resolve().parents[2] / "data/lpp_fmri")
lang = "en"
section_id = 2
model_name = "AntonV/mamba2-1.3b-hf"
maxlen = 2048
stride = 1984

tok = AutoTokenizer.from_pretrained(model_name)
lbg_zip = str(_P(__file__).resolve().parents[2] / "data/lpp_fmri_extra/lbg_story_text/lpp_en_text.zip")
fulltext_run = load_story_text(
    bids_root=bids_root, lang=lang, section=section_id, lbg_story_text_zip=lbg_zip
).replace("\n", " ").replace("\r", " ")

enc = tok(
    fulltext_run,
    return_tensors="pt",
    return_offsets_mapping=True,
    truncation=True,
    padding=True,
    max_length=maxlen,
    return_overflowing_tokens=True,
    stride=stride,
)
om = enc["offset_mapping"]
ids = enc["input_ids"]
n_chunks = ids.shape[0]
print("n_chunks:", n_chunks)

for k in range(n_chunks):
    row_ids = ids[k]
    # find real (non-pad) length
    L = row_ids.shape[0]
    o0 = om[k, 0].tolist()
    o_last = om[k, L-1].tolist()
    print(f"chunk {k}: len={L} first_tok_off={o0} last_tok_off={o_last} "
          f"first_text={fulltext_run[o0[0]:o0[1]]!r} last_text={fulltext_run[o_last[0]:o_last[1]]!r}")

print()
print("=== decode each chunk's raw token id sequence around the boundary and compare ===")
for k in range(n_chunks):
    print(f"--- chunk {k} tokens[0:6] decoded: {tok.convert_ids_to_tokens(ids[k,0:6].tolist())}")
    print(f"--- chunk {k} tokens[2040:2048] decoded: {tok.convert_ids_to_tokens(ids[k,2040:2048].tolist())}")

print()
print("=== does chunk_k[64:2048] == chunk_{k+1}[0:1984] as token ids? ===")
for k in range(n_chunks - 1):
    a = ids[k, 64:2048]
    b = ids[k+1, 0:1984]
    same = bool((a == b).all())
    print(f"chunk {k}[64:2048] == chunk {k+1}[0:1984]: {same}")
    if not same:
        diffs = (a != b).nonzero().flatten().tolist()[:10]
        print("  first mismatch local idxs (within slice):", diffs)
        for d in diffs[:5]:
            print(f"    slice_idx={d} chunk{k}_tok={tok.convert_ids_to_tokens([a[d].item()])} "
                  f"chunk{k+1}_tok={tok.convert_ids_to_tokens([b[d].item()])}")
