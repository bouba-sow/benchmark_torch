import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained("Jtapsa/moep", trust_remote_code=True)
print("config vocab_size:", getattr(cfg, "vocab_size", None))
tok = AutoTokenizer.from_pretrained("Jtapsa/moep", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Jtapsa/moep", trust_remote_code=True)
model.eval()

emb_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Embedding)]
for n, m in emb_modules:
    print("Embedding:", n, "num_embeddings=", m.num_embeddings, "embedding_dim=", m.embedding_dim)

# Try a small forward with a known-safe input to sanity check.
ids = torch.tensor([[tok.bos_token_id, 100, 200, tok.eos_token_id]])
print("max token id in test input:", ids.max().item())
with torch.no_grad():
    try:
        out = model(input_ids=ids)
        print("forward OK on CPU, output type:", type(out))
    except Exception as e:
        print("forward FAILED on CPU:", repr(e))

# Now check: what does the naturalstories surprisal pipeline actually pass as
# input_ids -- print min/max id it would produce for a real sentence, and
# compare against the embedding table size.
text = "This is a simple test sentence for the natural stories corpus."
enc = tok(text, return_tensors="pt")
print("encoded input_ids:", enc["input_ids"])
print("max id:", enc["input_ids"].max().item(), "min id:", enc["input_ids"].min().item())
