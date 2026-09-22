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
print("padding_side:", tok.padding_side)
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
am = enc["attention_mask"]
n_chunks = ids.shape[0]

for k in range(n_chunks):
    real = int(am[k].sum().item())
    first_real = int(am[k].nonzero()[0, 0].item()) if real > 0 else -1
    print(f"chunk {k}: real_tokens={real} first_real_local_idx={first_real} seq_len={ids.shape[1]}")

# chunk1's local index 64 == global start of stride overlap it shares with chunk0.
# chunk2's first real local idx should, if left-padded, correspond to the SAME
# global token position as chunk1's local idx (64 + (2048-2048))? Let's directly
# compare token ids: does chunk2[first_real:] tail-match chunk1[X:] for some X?
c1 = ids[1]
c2 = ids[2]
fr2 = int(am[2].nonzero()[0, 0].item())
real2 = c2[fr2:]
print("chunk2 real length:", real2.shape[0])
# search for the offset X in chunk1 where chunk1[X:] == real2[: (2048-X)] (as much as fits)
found = None
for X in range(2048):
    seg1 = c1[X:]
    n = min(seg1.shape[0], real2.shape[0])
    if n == 0:
        break
    if bool((seg1[:n] == real2[:n]).all()):
        found = X
        break
print("chunk1 local index X where chunk1[X:] matches chunk2 real tail:", found)
if found is not None:
    print("chunk1[X] decoded:", tok.convert_ids_to_tokens([c1[found].item()]))
    print("chunk2[first_real] decoded:", tok.convert_ids_to_tokens([c2[fr2].item()]))
    om1 = om[1, found].tolist()
    om2 = om[2, fr2].tolist()
    print("chunk1[X] abs offset:", om1, repr(fulltext_run[om1[0]:om1[1]]))
    print("chunk2[first_real] abs offset:", om2, repr(fulltext_run[om2[0]:om2[1]]))
