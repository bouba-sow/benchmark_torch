# Handoff: logit-lens per-layer surprisal for constrained/MoE BabyLM models

## Context

Goal: extract per-layer surprisal (logit-lens method, Kuribayashi et al. 2025,
arXiv:2502.01615) for the fedorenko2016 / michaelov / tuckute benchmarks, for
the "constrained" BabyLM families (stickbreaking, alibi, forgetting,
hard_2gram/3gram/5gram — all `trust_remote_code=True`, in
`neuroai/models/registry.py`) and the BabyLM MoE model (`Jtapsa/moep`). Job
scripts already existed (`scripts/slurm/jz/*_stickbreaking.sh` etc.) but had
never actually been run — confirmed via `outputs_jeanzay/logs/`: no log files
for `*_surprisal_stickbreaking*`, `*_surprisal_alibi*`,
`*_surprisal_forgetting*`, `*_surprisal_hard_*`, and the `babylm` family's
array index 2 (moep) never ran either — only CLASS-IT (index 0) and
blm-gpt2s (index 1), both standard architectures, succeeded previously.

Root cause of never having run these: `fla` (flash-linear-attention)'s Triton
backend needs a live CUDA driver just to *import*, confirmed directly:
```
$ .venv/bin/python -c "import fla"
RuntimeError: 0 active drivers ([]). There should only be one.
```
So none of this is even importable on a CPU-only login node, let alone
Oberon (no GPU at all) — it has to be tested on an actual GPU node. Tested
directly on Jean Zay H100 (`--account=ldh@h100 --qos=qos_gpu_h100-dev
-C h100`).

**Everything below except the "Files touched" section describes code that
already exists in the repo** (the model/attention implementations, mostly
under `forgetting-transformer/src/forgetting_transformer/`) — none of it was
written as part of this session; it's a technical description of what's
already there, to explain *why* each family behaves the way it does under
logit-lens extraction. The only code I actually changed is
`neuroai/features/logit_lens.py`, detailed in "Bugs 1 & 2" below. **That file
lives outside `outputs_jeanzay/` and will NOT survive the Oberon→Jean Zay
rsync** — apply the same edits on the Oberon-side copy.

## How logit-lens surprisal extraction actually works, mechanically

For every family in this benchmark (standard HF architectures included), the
pipeline in `neuroai/features/logit_lens.py::logit_lens_surprisal_all_layers`
does, per text chunk:

1. **Tokenize** the chunk (sliding-window, `stride_delta` overlap between
   chunks for long texts — same chunking as the embeddings pipeline).
2. **Forward pass** through the model to get one hidden-state tensor per
   transformer layer. For standard HF architectures (GPT-2, Llama-style,
   Mamba, …) this is just `model(..., output_hidden_states=True)` and reading
   `out.hidden_states` — a tuple of `num_hidden_layers + 1` tensors (index 0 =
   token embeddings, then one per block).
3. **Per-layer logit-lens readout**: for every layer `l` (excluding the
   embedding layer), take that layer's hidden state, run it through the
   model's *own* final norm (RMSNorm/LayerNorm — whatever it natively uses
   before its LM head) and the model's *own* unembedding matrix, softmax, and
   read off `-log p(actual_next_token)` — i.e., "pretend this layer's output
   was already the network's final output." No extra training, no auxiliary
   probe (that would be tuned-lens, a different, heavier method).
4. Sub-word tokens making up one word get their surprisals averaged (this
   repo's existing convention).

**Why none of the constrained/BabyLM-custom families take the standard path
in step 2**: every one of them (stickbreaking, alibi, forgetting, hard n-gram,
moep) defines its own `forward()` in `trust_remote_code` model code that
either doesn't accept `output_hidden_states` at all, or accepts it but never
actually populates it — it just returns a raw hidden-state tensor (see e.g.
`StickbreakingModel.forward`, which literally does `hidden_states =
self.norm(hidden_states); return hidden_states` — no branching on
`output_hidden_states` whatsoever). `_get_cached_causal_lm`/
`_forward_hidden_states` (in `neuroai/features/llm.py`) detect this
(`out.hidden_states is None`) and fall back to **manually registering a
PyTorch forward hook on every module in `model.model.layers`**, running a
second forward pass, and collecting each hook's captured output tensor as the
per-layer hidden state instead. This hook-based capture is what makes
per-layer surprisal possible at all for these families — but (see Bug 2
below) it returns a tuple with **no embedding entry**, unlike the native path,
which is exactly what `logit_lens.py` got wrong.

## Attention mechanism per family (why each one needs what it needs)

All of these live under
`forgetting-transformer/src/forgetting_transformer/model/<family>/modeling_<family>.py`
in this repo (the actual HF Hub checkpoints under `Lanni-ni/*` bundle their
own near-identical copy of this code as `trust_remote_code`).

**stickbreaking** (`StickbreakingAttention`, ICLR 2025) — replaces softmax
attention entirely. Instead of normalizing all attention logits jointly via
softmax, it treats attention as a sequential "stick-breaking" process: each
key position gets a local "break probability" `p_i = sigmoid(logit_i)`, and
its actual attention weight is `p_i × ∏_{j<i}(1 - p_j)` — the fraction of a
unit "stick" not yet allocated to any earlier position. This gives causal,
naturally-decaying attention weights without any explicit decay parameter.
Implemented via the official Triton kernel `sb_attn` (from the standalone
`stickbreaking_attention` package) — **this is a real custom GPU kernel, not
just a plain-PyTorch reference**, so it hard-requires a CUDA-capable node to
even run, independent of the `trust_remote_code`/hook issues. Supports
optional RoPE and GQA (`num_kv_heads < num_heads`).

**alibi** (`Attention` in `modeling_alibi.py`) — plain scaled dot-product
softmax attention (`q @ k.T / sqrt(d)`, standard causal mask), with a fixed
linear positional bias added to the raw logits before softmax:
`bias[h,i,j] = slope_h × (position term)`, where `slope_h` follows the
original ALiBi geometric-sequence-of-slopes schedule (`_get_slopes`). No
learned relative-position parameters — the recency bias is entirely baked
into fixed per-head slopes. The attention math itself needs no Triton kernel,
but the surrounding model still imports `RMSNorm`/`RotaryEmbedding`/
`swiglu_linear` from `fla.modules`, so the whole file still needs the same
CUDA-capable environment just to import, even though ALiBi itself doesn't use
a custom kernel.

**forgetting** (`ForgettingAttentionLayer`, "FoX") — softmax attention plus a
**data-dependent forget gate**: a per-token, per-head scalar
`α = sigmoid(fgate_proj(hidden_state))`, log'd and cumulatively summed along
the sequence to get `log_λ[t] = Σ_{≤t} log(α)`. The bias added to raw
attention scores is `log_λ[i] - log_λ[j]` for query `i`, key `j` — i.e. a
*learned, content-dependent* generalization of ALiBi's fixed linear decay:
the model decides, per token, how fast to "forget" the past instead of using
a fixed slope. `fgate_type` can be `"full"` (this data-dependent gate),
`"bias_only"` (a single learned-but-content-independent per-head bias, no
dependence on input), or `"none"` (plain softmax, no forgetting at all).
Implemented in `forgetting_attention_std` — a plain-PyTorch *reference*
kernel (there's a faster Triton production kernel elsewhere in the FLA
ecosystem this repo doesn't use directly). Also the family with its own
bypass loader (`_load_forgetting_direct` in `llm.py`) for an unrelated
checkpoint/config-mismatch bug — unaffected by anything in this handoff.

**hard_2gram / hard_3gram / hard_5gram** (`SlidingWindowAttention`, Kuribayashi
et al. 2022, "Context Limitations Make Neural LMs More Human-Like") — the
simplest mechanism here: plain scaled dot-product softmax attention with a
**hard binary mask** restricting each query to the most recent `window_size`
tokens only (`window_size=1` → bigram/2-gram context, `=2` → trigram,
`=4` → 5-gram); everything outside the window gets `-inf` before softmax.
No decay, no learned bias, no smooth falloff — a fixed hard cutoff meant as a
baseline testing whether a hard limited context (vs. a smoothly-decaying one
like ALiBi/forgetting) alone drives more human-like behavior. Plain PyTorch,
no custom Triton kernel for the attention math itself.

**moep** (`Jtapsa/moep`, "Mixture-of-Experts Parallel" — a completely separate
codebase from the forgetting-transformer/FLA lineage above, not built on
`fla` at all) — this one is architecturally the odd one out, and it's why it
doesn't work: unlike every other family here, **its residual-stream width
changes partway through the network**. Concretely, for the tested checkpoint
(`dim=384`, `n_heads=6`, `parallel_dim=192`, `n_experts=4`, `moe_k=2`,
`n_parallel_blocks=4`, `parallel_k=2`), `self.model.layers` is:

```
layers[0]    = Layer_Block(dim=384)              — full-width, standard softmax self-attention
layers[1]    = MoE_Block(384 → 192)               — sparse top-2-of-4 LINEAR projection down (no attention at all)
layers[2..N-3] = ParallelLayer(dim=192) × (N-4)    — see below
layers[N-2]  = MoE_Block(192 → 384)               — sparse top-2-of-4 LINEAR projection back up
layers[N-1]  = Layer_Block(dim=384)              — full-width, standard softmax self-attention
```

`MoE_Block` is pure feed-forward routing (a `Top_k_Router` picks 2 of 4
per-token expert *linear layers*, weighted-sums their outputs) — despite the
name, it contains no attention at all; it's really a sparse dimension-
changing adapter, not a transformer block.

`ParallelLayer` is the genuinely novel part (the "Parallel" in "MoEP"): it
runs **4 complete, independent 192-dim `Layer_Block`s (full self-attention +
FFN each) on the same input in parallel** (all 4 always computed, not sparse
dispatch), then a `Top_k_Router` picks 2 of those 4 *complete block outputs*
per token and weighted-sums them. So the "mixture" here is over whole
transformer sub-blocks, not just FFN experts as in a standard MoE — hence
"Mixture-of-Experts Parallel."

**Why logit-lens breaks on moep specifically**: logit-lens's whole method
requires every intermediate layer's hidden state to live in the *same*
vector space as the model's final layer, so the model's own final norm
(`LayerNorm(384)`) + unembedding (sized for 384) can be applied uniformly to
every layer read out. That's true for layers 0 and N-1 (both 384-dim) but
false for the `MoE_Block`/`ParallelLayer` stretch in the middle (192-dim) —
confirmed directly:
```
RuntimeError: Given normalized_shape=[384], expected input with shape [*, 384], but got input of size[26, 192]
```
This is a real dimensionality mismatch, not a bug to patch — see "Unresolved"
below.

## Bug 1 — `trust_remote_code` silently dropped

`logit_lens_surprisal_all_layers` and `logit_lens_surprisal_trials` both
accept a `trust_remote_code: bool` parameter, but never forward it into the
`load_kw` dict passed to `_get_cached_causal_lm` (which reads
`_trc = bool(load_kw.get("trust_remote_code", False))`). Result: `_trc` is
always `False` for these two functions regardless of what the caller passes.
Every `trust_remote_code=True` family except "forgetting" (bypass loader,
doesn't even check `trust_remote_code`) hits HF's interactive "Do you wish to
run the custom code? [y/N]" prompt and crashes with `EOFError` in any
non-interactive (SLURM/batch) job.

**Fix** — in `neuroai/features/logit_lens.py`, both call sites of
`_get_cached_causal_lm` currently read:

```python
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=revision,
        device=device,
        load_kw={"output_hidden_states": True},
        tokenizer_no_revision=tokenizer_no_revision,
        n_params=n_params,
        untrained=untrained,
        untrained_seed=untrained_seed,
    )
```

Change `load_kw` to:

```python
        load_kw={"output_hidden_states": True, "trust_remote_code": trust_remote_code},
```

(Both functions already have `trust_remote_code: bool = False` in their
signature — this just threads it through.)

## Bug 2 — embedding-layer slice wrongly applied to hook-fallback output

As explained in "How logit-lens works" above: the native
`output_hidden_states=True` path returns `num_hidden_layers + 1` entries
(embedding at index 0, then one per block); the hook-based fallback used for
all the custom architectures above returns *only* the per-block captures,
with no embedding entry. `logit_lens.py` unconditionally did:

```python
            hs = _forward_hidden_states(model, ids_k, am_k, model_name=model_name)
            layer_hs = hs[1:]   # assumes hs[0] is always the embedding layer
```

For every hook-fallback family this silently discarded a **real transformer
layer** instead of an embedding. Verified directly: a `_2layer` checkpoint
(confirmed via `AutoConfig.num_hidden_layers=2` and
`len(model.model.layers)=2` — genuinely 2 blocks) was only yielding 1 layer of
logit-lens output for stickbreaking/hard_2gram/hard_3gram/hard_5gram, and
(by lucky coincidence, see below) an apparently-correct 2 for alibi.

This same native-vs-hook distinction was already correctly handled for the
*embeddings* extraction pipeline elsewhere in `llm.py` (~line 1760:
`# hidden_states: tuple ... including the token embedding (index 0), unless
the hook fallback engaged (no separate embedding-layer entry in that case)` —
that code does not blindly slice). `logit_lens.py` just never got the
equivalent treatment.

**Fix** — add this helper to `neuroai/features/logit_lens.py` (placed just
above `_find_final_norm`):

```python
def _exclude_embedding_layer(hs: tuple, model: tp.Any) -> tuple:
    """Drop the embedding-layer entry (index 0), only when it's actually there.

    The native ``output_hidden_states=True`` path returns embeddings + one
    entry per transformer block (``num_hidden_layers + 1`` total). The
    hook-based fallback in ``_forward_hidden_states`` (used for
    trust_remote_code architectures that ignore ``output_hidden_states``)
    returns only the per-block captures — no embedding entry to strip.
    Blindly slicing ``hs[1:]`` in that case silently discards a real
    transformer layer instead of the embedding.
    """
    n_hidden_layers = config_dim_attr(
        model.config, ("num_hidden_layers", "n_layer", "n_layers"),
    )
    return hs[1:] if len(hs) == n_hidden_layers + 1 else hs
```

Add `config_dim_attr` to the existing `from .llm import (...)` block at the
top of the file (already a public helper in `neuroai/features/llm.py`, used
elsewhere for reading `num_hidden_layers`-style fields across non-standard
config classes).

Then replace both occurrences of `layer_hs = hs[1:]` (one in
`logit_lens_surprisal_all_layers`, one in `logit_lens_surprisal_trials`) with:

```python
            layer_hs = _exclude_embedding_layer(hs, model)
```

**Side note on the "alibi looked correct before the fix" coincidence**: when
initially tested back-to-back with other models in one Python process, the
hook fallback appeared to fire 3 times for alibi's 2 real layers (making the
post-`[1:]` slice come out at the "right" count, 2, purely by accident).
Reproduced in isolation with a clean single-model script afterward and got
exactly 2 distinct, non-duplicate captures matching its 2 real layers — the
"3" was an artifact of the multi-model test script sharing one process
(stale/leftover hook registration from a prior model in the same loop, most
likely), not a real per-architecture issue, and not something to worry about
in production (one model per SLURM array task, always a fresh process).

## Verification (Jean Zay H100, after both fixes)

Ran `logit_lens_surprisal_all_layers` on a 24-word test sentence for each
family, `device="cuda"`, `trust_remote_code=True`:

| family | model | result | shape (words, layers, 1) |
|---|---|---|---|
| stickbreaking | Lanni-ni/stickbreaking_babylm_10m_2layer | OK | (24, 2, 1) |
| alibi | Lanni-ni/alibi_babylm_10m_2layer | OK | (24, 2, 1) |
| hard_2gram | Lanni-ni/hard_2gram_babylm_10m_2layer | OK | (24, 2, 1) |
| hard_3gram | Lanni-ni/hard_3gram_babylm_10m_2layer | OK | (24, 2, 1) |
| hard_5gram | Lanni-ni/hard_5gram_babylm_10m_2layer | OK | (24, 2, 1) |
| forgetting | Lanni-ni/forgetting_babylm_10m_2layer | OK (already worked pre-fix, via its own bypass loader) | (24, 2, 1) |

All layer counts match each checkpoint's real `config.num_hidden_layers` (2).
Surprisal values were finite for all words/layers except the expected word-0
NaN (no left context to predict it from).

## Unresolved — moep (`Jtapsa/moep`), NOT a quick fix

See the architecture breakdown above (dim=384 sandwiching a parallel_dim=192
core). Possible paths forward, none attempted:
- Skip the reduced-dim (`MoE_Block`/`ParallelLayer`) layers in the logit-lens
  sweep entirely, reporting only the two full-width layers (0 and N-1) — loses
  most of the model's depth but is at least well-defined.
- Trace exactly how the 192→384 `MoE_Block` recombines the parallel core's
  output into the main residual stream, and find/derive a projection so
  intermediate 192-dim states can be read out through *that* combination
  rather than directly through the 384-dim final norm — real design work, not
  a one-liner, and there may not be a principled projection (the up-projection
  is itself a sparse top-2-of-4 *learned* MoE linear, not a fixed invertible
  map).

## Files touched

Only `neuroai/features/logit_lens.py`:
1. Added `config_dim_attr` to the `from .llm import (...)` block.
2. Added `_exclude_embedding_layer` helper function.
3. Changed both `_get_cached_causal_lm(...)` calls' `load_kw` to include
   `"trust_remote_code": trust_remote_code`.
4. Changed both `layer_hs = hs[1:]` lines to
   `layer_hs = _exclude_embedding_layer(hs, model)`.

No other files were modified as part of this investigation. Everything
described about stickbreaking/alibi/forgetting/hard-ngram/moep's internal
architecture is pre-existing code, read (not written) during this
investigation, to explain why the two bugs manifested the way they did.
