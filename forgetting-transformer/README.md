# Constrained-Attention Transformers

Models and code for comparing attention mechanisms in language models. All trained models are on HuggingFace: https://huggingface.co/Lanni-ni

## Setup (requires CUDA)

```bash
git clone https://github.com/Lanni-ni/forgetting-transformer.git
cd forgetting-transformer
pip install torch transformers einops
pip install --no-deps --no-build-isolation git+https://github.com/sustcsonglin/flash-linear-attention.git
```

## Loading Models

```python
import sys
sys.path.insert(0, "src")
import forgetting_transformer.model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Lanni-ni/transformer_babylm_10m_2layer",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
```

## Model List

Static: transformer, alibi, forgetting_gate, stickbreaking, hard_2gram, hard_3gram, hard_5gram

Dynamic (epochs 1-10, regular + inverse): dynamic_alibi, dynamic_forgetting, dynamic_sliding_window

Training data: BabyLM-10M, BabyLM-100M, Pile. Architectures: 2-layer (2_4_256), 4-layer (4_6_384).
