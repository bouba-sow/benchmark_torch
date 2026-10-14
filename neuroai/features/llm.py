"""Contextual word embeddings from HuggingFace causal LMs.

Default extraction (``contextual_word_embeddings_hf``) applies a **moving
window of cumulative words**, takes the **last non-padding token** embedding,
and honours ``layer_index``.

To mirror **l-bg/llms_brain_lateralization** (full transcription, sliding tokenizer
chunks, tokenizer offsets, token-mean per stimulus word), use
``contextual_word_embeddings_hf_lbg`` with ``hf_extraction_mode=\"lbg_fulltext\"``.

Per-layer support
-----------------
Set ``layer_index="all"`` to extract all hidden states (embedding + every
transformer layer), returning shape ``(n_words, n_layers, hidden)``.  Use
``layer_index=N`` (int) to select a single specific layer (0 = token
embedding).  Default ``"last"`` returns only the final layer as before.

Disk caching
------------
Pass ``cache_dir`` and ``section_id`` to cache raw feature arrays as
``.npz`` files keyed by model / revision / language / section / hyperparams.
Subsequent calls with the same settings load from disk instantly.

l-bg story text
---------------
This module also includes helpers for loading full story text (from
password-protected zip or Praat TextGrid fallback), word matching,
and Chinese character pooling — originally from ``lpp_lbg_fulltext.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import typing as tp
import weakref
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# transformers 5.13.1 compatibility monkeypatches for trust_remote_code models
# (e.g. BabyLM-community's gpt-bert-* models) whose custom modeling code
# predates conventions this transformers version now hard-requires. Applied
# here (module import time) rather than per-script, so every entry point that
# loads models via `_get_cached_causal_lm` below (Podcast, Tuckute, NS-fMRI,
# LPP, ...) gets them, instead of needing the same patch copy-pasted into
# every sbatch/srun script that happens to load one of these models.
# ---------------------------------------------------------------------------

_orig_json_default = json.JSONEncoder.default


def _json_default_stringify_unknown(self, o):
    # config_utils.py's to_json_string() has no `default=`, so an unexpected
    # non-serializable field (e.g. a bare torch.dtype in a trust_remote_code
    # config) raises TypeError at model-load `logger.info` time. Fall back to
    # str() for anything the original default can't handle.
    try:
        return _orig_json_default(self, o)
    except TypeError:
        return str(o)


json.JSONEncoder.default = _json_default_stringify_unknown

try:
    import transformers.modeling_utils as _modeling_utils

    _orig_pretrained_model_init = _modeling_utils.PreTrainedModel.__init__

    def _pretrained_model_init_with_tied_weights_fallback(self, *args, **kwargs):
        _orig_pretrained_model_init(self, *args, **kwargs)
        # Vendor custom __init__ methods (e.g. GPTBERTForCausalLM) sometimes
        # never call self.post_init(), which is what normally sets
        # all_tied_weights_keys and coerces the ignore-key lists to sets.
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        if isinstance(getattr(self, "_keys_to_ignore_on_load_unexpected", None), list):
            self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected)
        if isinstance(getattr(self, "_keys_to_ignore_on_load_missing", None), list):
            self._keys_to_ignore_on_load_missing = set(self._keys_to_ignore_on_load_missing)

    _modeling_utils.PreTrainedModel.__init__ = _pretrained_model_init_with_tied_weights_fallback
except ImportError:
    pass

# On a CPU-only node, `import fla` (transitively pulled in by
# `transformers.dynamic_module_utils.check_imports` when eagerly validating a
# trust_remote_code model's declared imports -- e.g. the Lanni-ni
# stickbreaking/alibi/forgetting/hard-n-gram families -- REGARDLESS of
# whether that specific model actually uses fla at runtime) crashes with an
# inscrutable "RuntimeError: 0 active drivers ([]). There should only be
# one." This is `fla/ops/utils/index.py` applying `@triton.autotune(...)` at
# MODULE level, which eagerly calls `triton.runtime.driver.active.get_
# benchmarker()` -- and this triton version registers only nvidia/amd GPU
# backends, no CPU fallback, so `driver.active` (lazily initialized on first
# access) raises the moment anything touches it with zero GPUs visible.
# Pre-empt this by installing a dummy driver (satisfies the one attribute
# access `Autotuner.__init__` needs) before anything can trigger the import,
# but only when CUDA is genuinely unavailable -- a real GPU still gets
# triton's real driver. fla itself has its own graceful CPU fallback once
# past this one crash point (see its `_device.py`'s "roll back to CPU").
try:
    import torch as _torch_for_triton_check

    if not _torch_for_triton_check.cuda.is_available():
        import triton.runtime.driver as _triton_driver_singleton

        class _NullTritonBenchmarker:
            def __call__(self, *args, **kwargs):
                raise RuntimeError(
                    "Triton GPU kernels are unavailable on this CPU-only node"
                )

        class _NullTritonDriver:
            def get_benchmarker(self):
                return _NullTritonBenchmarker()

            def __getattr__(self, name):
                raise RuntimeError(
                    f"Triton GPU driver attribute {name!r} accessed on a "
                    "CPU-only node"
                )

        _triton_driver_singleton.active = _NullTritonDriver()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# l-bg story text helpers (merged from lpp_lbg_fulltext.py)
# ---------------------------------------------------------------------------

LBG_STORY_TEXT_ZIP_PASSWORD_BYTES = b"lessentielestinvisiblepourlesyeux"

PUNCT = ["-", "'", "\u2019", "\u201c", "\u201d", "«", "»", "\u2014"]
CHINESE_PUNCT = ["，", "《", "》", "。", "：", "“", "”", "；", "？", "！", "…", "、"]

# See neuroai.lpp_llm_features.LBG_PHANTOM_CONTRACTION_EXPANSIONS for the
# full rationale -- LPP's word table splits auxiliary contractions into two
# annotated words (e.g. "hasn't" -> "hasnt" + "not"), and the second piece
# has no independent surface text once the first piece's loose partial match
# has already absorbed the whole contraction. Only used as a fallback once
# the normal forward search has already exhausted every candidate in range.
LBG_PHANTOM_CONTRACTION_EXPANSIONS = frozenset({
    "not", "nt",              # n't
    "would", "will",          # 'd / 'll
    "have", "has", "had",     # 've / 's / 'd
    "am", "is", "are",        # 'm / 's / 're
})


def _annotation_subdir(lang: str) -> str:
    lg = lang.lower()
    mapping = {"en": "EN", "fr": "FR", "cn": "CN"}
    if lg not in mapping:
        raise ValueError(f"Unsupported language {lang!r}; expected en|fr|cn")
    return mapping[lg]


def story_text_zip_inner_path(lang: str, section: int) -> str:
    lg = lang.lower()
    return f"lpp_{lg}_text/text_{lg}_run{int(section)}.txt"


def _load_story_text_from_lbg_zip(zip_path: Path | str, lang: str, section: int) -> str:
    zp = Path(zip_path)
    if not zp.is_file():
        raise FileNotFoundError(f"Story text zip not found: {zp}")
    inner = story_text_zip_inner_path(lang, section)
    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open(inner, pwd=LBG_STORY_TEXT_ZIP_PASSWORD_BYTES) as fp:
            raw = fp.read()
    text = raw.decode("utf-8")
    return text.replace("\n", " ").replace("\r", " ")


def _textgrid_path(bids_root: Path | str, lang: str, section: int) -> Path:
    sub = _annotation_subdir(lang)
    return Path(bids_root) / "annotation" / sub / f"lpp{sub}_section{int(section)}.TextGrid"


def _parse_tokensalign_textgrid(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
        try:
            content = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise ValueError(f"Cannot decode TextGrid {path}: tried utf-8-sig, utf-16, utf-8, latin-1")
    tokens: list[str] = []
    interval_re = re.compile(
        r"intervals\s*\[\d+\]:\s*"
        r"xmin\s*=\s*([0-9.+-eE]+)\s*"
        r"xmax\s*=\s*([0-9.+-eE]+)\s*"
        r'text\s*=\s*"(.*?)"\s*(?=\n|\Z)',
        re.DOTALL,
    )
    for m in interval_re.finditer(content):
        t = m.group(3)
        if t is None:
            continue
        t = t.strip()
        if not t or t == "#":
            continue
        tokens.append(t)
    if not tokens:
        raise ValueError(f"No TokensAlign intervals parsed from {path}")
    return " ".join(tokens)


def _load_story_text_from_textgrid(bids_root: Path | str, lang: str, section: int) -> str:
    path = _textgrid_path(bids_root, lang, section)
    if not path.is_file():
        raise FileNotFoundError(f"TextGrid not found: {path}")
    return _parse_tokensalign_textgrid(path)


def load_story_text(
    *,
    bids_root: Path | str,
    lang: str,
    section: int,
    lbg_story_text_zip: Path | str | None = None,
) -> str:
    if lbg_story_text_zip is not None:
        return _load_story_text_from_lbg_zip(lbg_story_text_zip, lang, section)
    return _load_story_text_from_textgrid(bids_root, lang, section)


def simplify_word(word: str, lang: str) -> str:
    word = str(word).lower()
    word = word.replace(" ", "")
    for p in PUNCT:
        word = word.replace(p, "")
    if lang.lower() == "cn":
        for p in CHINESE_PUNCT:
            word = word.replace(p, "")
    return word


def do_word_match(word_in_list: str, word_in_text: str, lang: str) -> bool:
    word_in_list = simplify_word(word_in_list, lang)
    word_in_text = simplify_word(word_in_text, lang)
    if len(word_in_text) > 0 and word_in_list.startswith(word_in_text):
        return True
    min_len = 1 - (1 if lang.lower() == "cn" else 0)
    if len(word_in_list) > min_len and word_in_list in word_in_text:
        return True
    if len(word_in_text) > min_len and word_in_text in word_in_list:
        return True
    if word_in_list == "one" and word_in_text == "1":
        return True
    if word_in_list == "did" and word_in_text == "didn":
        return True
    if word_in_list == "nt" and word_in_text == "t":
        return True
    if word_in_list == "does" and word_in_text == "doesn":
        return True
    if word_in_list == "do" and word_in_text == "don":
        return True
    if word_in_list == "is" and word_in_text == "isn":
        return True
    if word_in_list in {
        "threetwofive", "threetwosix", "threetwoseven",
        "threetwoeight", "threetwonine", "threethreezero",
    } and word_in_text == "3":
        return True
    if word_in_list == "na\\ive" and word_in_text == "naive":
        return True
    if word_in_list == "repondit" and word_in_text == "répondit":
        return True
    if word_in_list == "oeuvre" and word_in_text == "œuvre":
        return True
    if word_in_list == "oeil" and word_in_text == "œil":
        return True
    if word_in_list == "a" and word_in_text == "à":
        return True
    if word_in_list == "coeur" and word_in_text == "cœur":
        return True
    return False


def build_lbg_word_stream(
    words_df: pd.DataFrame,
    lang: str,
    *,
    use_csv_row_enumerator_index: bool = False,
) -> tuple[list[str], list[int] | None, int]:
    lg = lang.lower()
    word_list: list[str] = []
    char_to_word_idx: list[int] | None = None
    if lg == "cn":
        char_to_word_idx = []
        compact_wid = 0
        for enum_ix, (_, row) in enumerate(words_df.iterrows()):
            w = row["word"]
            if not isinstance(w, str) or w == " ":
                continue
            tag_ix = enum_ix if use_csv_row_enumerator_index else compact_wid
            compact_wid += 1
            for _ch in w:
                char_to_word_idx.append(tag_ix)
                word_list.append(_ch)
        if len(char_to_word_idx) != len(word_list):
            raise RuntimeError("char_to_word_idx / word_list length mismatch")
        return word_list, char_to_word_idx, compact_wid
    for _, row in words_df.iterrows():
        w = row["word"]
        if isinstance(w, str) and w != " ":
            word_list.append(w)
    return word_list, None, len(word_list)


def aggregate_cn_char_activations(
    char_features_per_layer: list[list[np.ndarray]],
    char_to_word_idx: list[int],
    n_words: int,
) -> list[np.ndarray]:
    n_layers = len(char_features_per_layer)
    out_stacked: list[np.ndarray] = []

    idx_arr = np.array(char_to_word_idx, dtype=int)
    for idx_layer in range(n_layers):
        layer_chars = char_features_per_layer[idx_layer]
        layer_words_activations: list = []
        for idx_word in range(n_words):
            word_activations = []
            for i in np.flatnonzero(idx_arr == idx_word):
                if i < len(layer_chars) and np.asarray(layer_chars[i]).size > 1:
                    word_activations.append(np.asarray(layer_chars[i], dtype=np.float32))
            if len(word_activations) > 0:
                layer_words_activations.append(np.mean(np.stack(word_activations, axis=0), axis=0))
            else:
                layer_words_activations.append("tmp")
        next_activation = layer_words_activations[-1]
        for iw in range(n_words - 1, -1, -1):
            entry = layer_words_activations[iw]
            if isinstance(entry, str) and entry == "tmp":
                layer_words_activations[iw] = np.asarray(next_activation, dtype=np.float32)
            next_activation = layer_words_activations[iw]
        out_stacked.append(
            np.stack(layer_words_activations, axis=0).astype(np.float32, copy=False)
        )
    return out_stacked


# ---------------------------------------------------------------------------
# HF model / feature extraction
# ---------------------------------------------------------------------------

LayerIndex = tp.Union[tp.Literal["last", "all"], int]

# EleutherAI/pythia-2.8b (non-deduped) ships a stale duplicate ``model.safetensors`` on many
# checkpoint branches; transformers prefers it over sharded weights or pytorch_model.bin.
# Prefer sharded safetensors when ``model.safetensors.index.json`` exists; otherwise load
# ``pytorch_model.bin``.  Pythia-2.8b-deduped uses (cleaner Hub layout).
_PYTHIA_28B_HUB_FIX_MODELS = frozenset(
    {"EleutherAI/pythia-2.8b", "EleutherAI/pythia-2.8b-deduped"}
)
_SAFETENSORS_SHARD_INDEX = "model.safetensors.index.json"

# Single-slot in-process cache: one (model_id, revision, device) at a time so we
# forward all LPP sections without reloading weights; evicts when the key changes
# (next checkpoint / model) to limit GPU memory.
_hf_slot_key: tuple[str, str, str] | None = None
_hf_slot_tokenizer: tp.Any = None
_hf_slot_model: tp.Any = None


def clear_hf_model_cache() -> None:
    """Drop the in-memory HF model/tokenizer held by :func:`contextual_word_embeddings_hf`."""
    global _hf_slot_key, _hf_slot_tokenizer, _hf_slot_model
    _hf_slot_key = None
    _hf_slot_tokenizer = None
    if _hf_slot_model is not None:
        del _hf_slot_model
    _hf_slot_model = None
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _piece_to_single_id(tokenizer: tp.Any, piece: str) -> int | None:
    """Map one vocabulary piece to a single token id, or None if not exactly one id."""
    try:
        out = tokenizer.convert_tokens_to_ids(piece)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(out, int):
        tid = out
    elif isinstance(out, list):
        if len(out) != 1:
            return None
        tid = out[0]
    else:
        return None
    if isinstance(tid, int) and tid >= 0:
        return tid
    return None


def _infer_pad_token_id_from_vocab(tokenizer: tp.Any) -> tuple[int | None, str | None]:
    """Resolve pad id when special-token attributes are unset (e.g. ``TokenizersBackend`` + HF revision).

    Pythia checkpoint revisions (``step0``, …) often load as :class:`TokenizersBackend` with
    ``pad_token_id`` / ``eos_token_id`` all ``None`` while ``<|padding|>`` / ``<|endoftext|>``
    still map to concrete ids via :meth:`convert_tokens_to_ids`.
    """
    seen: set[str] = set()
    pieces: list[str] = []
    for p in (
        getattr(tokenizer, "pad_token", None),
        "<|padding|>",
        getattr(tokenizer, "eos_token", None),
        "<|endoftext|>",
        "<|eos|>",
        "</s>",
        "<pad>",
    ):
        if isinstance(p, str) and p and p not in seen:
            seen.add(p)
            pieces.append(p)
    for piece in pieces:
        tid = _piece_to_single_id(tokenizer, piece)
        if tid is not None:
            return tid, piece
    return None, None


def _ensure_pad_token_for_batches(tokenizer: tp.Any) -> None:
    """Ensure ``tokenizer(..., padding=True, ...)`` works for causal LMs.

    GPT-NeoX (e.g. Pythia) often omits ``pad_token`` / ``pad_token_id``.  Setting only
    ``pad_token = eos_token`` fails when ``eos_token`` is unset but ``eos_token_id``
    is valid.

    Some ``revision=`` checkpoint loads use :class:`TokenizersBackend` with **no**
    ``pad_token_id`` / ``eos_token_id`` attributes; ids must be read from the vocab
    (e.g. ``<|padding|>``).
    """
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        tokenizer.pad_token_id = eos_id
        if getattr(tokenizer, "pad_token", None) is None:
            eos = getattr(tokenizer, "eos_token", None)
            if eos is not None:
                tokenizer.pad_token = eos
        return
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None:
        tokenizer.pad_token_id = unk_id
        if getattr(tokenizer, "pad_token", None) is None:
            unk = getattr(tokenizer, "unk_token", None)
            if unk is not None:
                tokenizer.pad_token = unk
        return
    vid, vpiece = _infer_pad_token_id_from_vocab(tokenizer)
    if vid is not None:
        tokenizer.pad_token_id = vid
        if getattr(tokenizer, "pad_token", None) is None and vpiece is not None:
            tokenizer.pad_token = vpiece
        return
    raise ValueError(
        f"{type(tokenizer).__name__} has no pad/eos/unk token id; "
        "cannot use padding=True for batch encoding."
    )


def _pythia_28b_needs_hub_weight_fix(model_name: str) -> bool:
    return model_name in _PYTHIA_28B_HUB_FIX_MODELS


def _pythia_28b_use_sharded_weight_index(model_name: str, revision: str | None) -> bool:
    """True when we must load Pythia-2.8B from sharded safetensors (Hub index present)."""
    if not _pythia_28b_needs_hub_weight_fix(model_name):
        return False
    if revision in (None, "", "main"):
        return False
    try:
        from huggingface_hub import file_exists

        return file_exists(model_name, _SAFETENSORS_SHARD_INDEX, revision=revision)
    except Exception:  # noqa: BLE001
        return False


def _configure_pythia_28b_model_load(
    model_name: str,
    revision: str | None,
    model_rev_fp: dict[str, str],
    model_kw: dict[str, object],
    *,
    rev_key: str,
) -> dict[str, object]:
    """Apply Hub workarounds for Pythia-2.8B checkpoint weight loading."""
    from transformers import AutoConfig

    if not _pythia_28b_needs_hub_weight_fix(model_name):
        return model_kw
    if revision in (None, "", "main"):
        return model_kw

    out = dict(model_kw)
    if _pythia_28b_use_sharded_weight_index(model_name, revision):
        cfg = AutoConfig.from_pretrained(model_name, **model_rev_fp)
        cfg.transformers_weights = _SAFETENSORS_SHARD_INDEX
        out["config"] = cfg
        logger.info(
            "LPP HF: %s @%s — sharded safetensors via %s (stale monolithic "
            "model.safetensors also on Hub)",
            model_name,
            rev_key,
            _SAFETENSORS_SHARD_INDEX,
        )
        return out

    try:
        from huggingface_hub import file_exists

        has_pytorch = file_exists(model_name, "pytorch_model.bin", revision=revision)
        has_mono_safe = file_exists(model_name, "model.safetensors", revision=revision)
    except Exception:  # noqa: BLE001
        has_pytorch = has_mono_safe = False

    if has_pytorch and has_mono_safe:
        out["use_safetensors"] = False
        logger.info(
            "LPP HF: %s @%s — no shard index; loading pytorch_model.bin "
            "(monolithic model.safetensors on this branch is often stale)",
            model_name,
            rev_key,
        )
    elif has_mono_safe:
        logger.warning(
            "LPP HF: %s @%s — only model.safetensors on Hub (no shard index or "
            "pytorch_model.bin); weights may be wrong. Prefer EleutherAI/pythia-2.8b-deduped.",
            model_name,
            rev_key,
        )
    return out


# Ministral-3 Hub cards use model_type="mistral3" (Mistral3Config). Transformers maps
# ministral3 → Ministral3ForCausalLM but AutoModelForCausalLM does not include mistral3.
_CAUSAL_LM_CLASS_BY_MODEL_TYPE: dict[str, str] = {
    "mistral3": "Ministral3ForCausalLM",
    "ministral3": "Ministral3ForCausalLM",
}


def _model_config_type(
    model_name: str,
    rev_fp: dict[str, str],
    trust_remote_code: bool = False,
) -> str | None:
    from transformers import AutoConfig

    try:
        extra = {"trust_remote_code": True} if trust_remote_code else {}
        cfg = AutoConfig.from_pretrained(model_name, **rev_fp, **extra)
        return getattr(cfg, "model_type", None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read config for %s: %s", model_name, exc)
        return None


def _needs_fix_mistral_regex(model_name: str, model_type: str | None) -> bool:
    n = model_name.lower()
    mt = (model_type or "").lower()
    return "mistral" in n or "ministral" in n or mt in _CAUSAL_LM_CLASS_BY_MODEL_TYPE


def _resolve_causal_lm_class(
    model_name: str,
    model_rev_fp: dict[str, str],
    trust_remote_code: bool = False,
) -> type:
    """Return the causal LM class for ``model_name`` (Auto* or architecture-specific)."""
    from transformers import AutoModelForCausalLM
    import transformers as tr

    model_type = _model_config_type(model_name, model_rev_fp, trust_remote_code=trust_remote_code)
    cls_name = _CAUSAL_LM_CLASS_BY_MODEL_TYPE.get(model_type or "")
    if cls_name:
        logger.info(
            "HF LM: %s model_type=%r → %s",
            model_name,
            model_type,
            cls_name,
        )
        return getattr(tr, cls_name)
    return AutoModelForCausalLM


def _load_hf_tokenizer(
    model_name: str,
    tok_rev_fp: dict[str, str],
    trust_remote_code: bool = False,
) -> tp.Any:
    from transformers import AutoTokenizer

    if model_name.startswith(_FOX_DIRECT_PREFIX):
        # zhixuan-lin/{fox,transformer}-* ship tokenizer_class="JSONGPT2Tokenizer"
        # in tokenizer_config.json, a custom class with no auto_map pointing at
        # it (these aren't real trust_remote_code repos -- see _load_fox_direct's
        # docstring) -- AutoTokenizer.from_pretrained can't resolve the name and
        # raises "Couldn't instantiate the backend tokenizer". The vendored
        # forgetting_transformer.tokenizer module registers this class with
        # AutoTokenizer as an import side effect; just importing it here (before
        # from_pretrained looks the name up) is enough to fix the resolution.
        import sys

        _repo_root = Path(__file__).resolve().parents[2] / "forgetting-transformer" / "src"
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        import forgetting_transformer.tokenizer  # noqa: F401 -- side effect: AutoTokenizer.register(...)

    tok_kw = dict(tok_rev_fp)
    model_type = _model_config_type(model_name, tok_rev_fp, trust_remote_code=trust_remote_code)
    if _needs_fix_mistral_regex(model_name, model_type):
        tok_kw.setdefault("fix_mistral_regex", True)
    if trust_remote_code:
        tok_kw["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kw)
    # contextual_word_embeddings_hf / _fulltext extract the last real token via
    # last_pos = attention_mask.sum() - 1, which is only correct for RIGHT
    # padding. Some community checkpoints (e.g. AntonV/mamba2-*-hf) ship a
    # tokenizer_config.json with padding_side="left" — silently extracting a
    # padding-token (or wrong-word) hidden state for every sequence that isn't
    # the longest in its batch, tanking every downstream ridge score uniformly
    # across layers. Force right-padding here so the offset math is always valid.
    tokenizer.padding_side = "right"
    return tokenizer


def _model_input_device(model: tp.Any) -> str:
    """Device for batch inputs (first shard when ``device_map`` is used)."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first = next(iter(model.hf_device_map.values()))
        val = first[0] if isinstance(first, (list, tuple)) else first
        if isinstance(val, int):
            return f"cuda:{val}"
        s = str(val)
        return f"cuda:{s}" if s.isdigit() else s
    return str(next(model.parameters()).device)


def _apply_device_map_if_needed(
    model_kw: dict[str, object],
    device: str,
    *,
    n_params: float | None = None,
    headroom_gib: int = 10,
) -> bool:
    """Configure device placement for multi-GPU or large-model loading.

    Strategy
    --------
    * Model fits on one GPU  → ``device_map={"": 0}`` (everything on cuda:0).
      This avoids the inter-device tensor bugs that ``device_map="auto"``
      triggers on some architectures (e.g. GPT-Neo position embeddings landing
      on a different device than token embeddings when 4 GPUs are visible).
    * Model needs sharding   → ``device_map="auto"`` with per-GPU ``max_memory``.
    """
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return False
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        return False

    try:
        import accelerate  # noqa: F401 — required for device_map
    except ImportError:
        logger.warning(
            "HF LM: multi-GPU / large-model load needs accelerate; "
            "falling back to single-GPU .to()",
        )
        return False

    model_kw.setdefault("torch_dtype", torch.bfloat16)

    # transformers routes any `device_map` load through its offload-capable
    # path; for a checkpoint that ships only `pytorch_model.bin` (no
    # safetensors — e.g. EleutherAI/gpt-j-6b) that path raises
    # "provide an offload_folder ... weights are not in safetensors format"
    # even when the model fits entirely on the GPU and nothing is offloaded.
    # A writable scratch dir satisfies it (used only as a re-save workspace).
    if "offload_folder" not in model_kw:
        import os as _os
        import tempfile as _tf
        _base = _os.environ.get("SCRATCH") or _tf.gettempdir()
        _off = _os.path.join(_base, ".cache", "hf_offload", f"pid{_os.getpid()}")
        _os.makedirs(_off, exist_ok=True)
        model_kw["offload_folder"] = _off

    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    usable_gib = max(1, (gpu_mem - headroom_gib * (1024 ** 3)) // (1024 ** 3))

    # Estimate model size in bfloat16 (2 bytes/param).
    model_gib = (n_params * 2) / (1024 ** 3) if n_params else 0.0
    fits_on_one = model_gib <= usable_gib

    if fits_on_one:
        # Force the entire model onto cuda:0. ``device_map="auto"`` with
        # multiple visible GPUs can split small models across devices, causing
        # RuntimeError in architectures that add embeddings from two modules
        # (e.g. GPT-Neo: inputs_embeds + position_embeds on different devices).
        model_kw["device_map"] = {"": 0}
        logger.info(
            "HF LM: device_map={'':0} (fits on one GPU, %.1f GiB est.)%s",
            model_gib,
            f" (~{n_params / 1e9:.1f}B params)" if n_params else "",
        )
    else:
        # Model needs true multi-GPU sharding.
        max_per_gpu = f"{usable_gib}GiB"
        model_kw["device_map"] = "auto"
        model_kw["max_memory"] = {i: max_per_gpu for i in range(n_gpus)}
        logger.info(
            "HF LM: device_map=auto on %d GPU(s), max %s/GPU%s",
            n_gpus,
            max_per_gpu,
            f" (~{n_params / 1e9:.1f}B params)" if n_params else "",
        )
    return True


def _load_causal_lm_from_vlm_checkpoint(
    model_name: str,
    model_kw: dict,
) -> "nn.Module | None":
    """Load a CausalLM from a VLM-format checkpoint by stripping the ``language_model.`` prefix.

    VLM checkpoints for Ministral-3 store text weights as if
    ``language_model = Ministral3ForCausalLM``, giving keys like
    ``language_model.model.embed_tokens.weight`` and ``language_model.lm_head.weight``.
    However, ``AutoModel.from_config(text_config)`` instantiates ``Ministral3Model``
    (the bare base), which expects flat ``model.embed_tokens.weight`` — the prefix
    mismatch leaves ALL text weights randomly initialized.

    This function reads only the ``language_model.*`` keys from the cached safetensors
    shards, strips the prefix, and loads them into a fresh ``AutoModelForCausalLM``
    (e.g. ``Ministral3ForCausalLM``).  The resulting model has correct weights AND a
    working ``lm_head``, so surprisal can be computed directly from its logits output.

    Returns the loaded CausalLM on CPU, or ``None`` if the HF cache is unavailable.
    """
    import json

    import torch
    import torch.nn as _nn

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    cache_root = Path(hf_hub_cache or "~/.cache/huggingface/hub").expanduser()
    safe_org_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = cache_root / safe_org_name / "snapshots"
    if not snapshots_dir.exists():
        logger.warning("HF LM: VLM prefix-strip: cache not found at %s", snapshots_dir)
        return None

    # Resolve snapshot directory
    snap: Path | None = None
    revision = model_kw.get("revision")
    if revision:
        direct = snapshots_dir / revision
        if direct.is_dir():
            snap = direct
        else:
            ref_path = cache_root / safe_org_name / "refs" / str(revision)
            if ref_path.is_file():
                snap = snapshots_dir / ref_path.read_text().strip()
    if snap is None or not snap.is_dir():
        snaps = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        snap = max(snaps, key=lambda p: p.stat().st_mtime) if snaps else None
    if snap is None or not snap.is_dir():
        return None

    PREFIX = "language_model."

    try:
        from safetensors.torch import safe_open  # type: ignore[import]

        state_dict: dict[str, "torch.Tensor"] = {}

        index_path = snap / "model.safetensors.index.json"
        if index_path.is_file():
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            # Group language_model.* keys by shard to minimise file opens
            shard_to_keys: dict[str, list[str]] = {}
            for key, shard in weight_map.items():
                if key.startswith(PREFIX):
                    shard_to_keys.setdefault(shard, []).append(key)
            for shard_fname, keys in shard_to_keys.items():
                shard_path = snap / shard_fname
                if not shard_path.is_file():
                    continue
                with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                    for key in keys:
                        state_dict[key[len(PREFIX):]] = sf.get_tensor(key)
        else:
            single_sf = snap / "model.safetensors"
            if single_sf.is_file():
                with safe_open(str(single_sf), framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        if key.startswith(PREFIX):
                            state_dict[key[len(PREFIX):]] = sf.get_tensor(key)

        if not state_dict:
            logger.warning(
                "HF LM: VLM prefix-strip: no 'language_model.*' keys found for %s", model_name
            )
            return None

        # Instantiate the correct CausalLM from the text config
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg_kw = {k: v for k, v in model_kw.items() if k in ("revision", "cache_dir", "trust_remote_code")}
        full_config = AutoConfig.from_pretrained(model_name, **cfg_kw)
        text_config = getattr(full_config, "text_config", full_config)
        # from_config() defaults to fp32 regardless of the checkpoint's own dtype,
        # which for a 14B+ model means a full fp32 copy (~2x the bf16 state_dict
        # already held in `state_dict`) is materialised on CPU just to be
        # immediately overwritten by load_state_dict() below — doubling peak CPU
        # RAM and risking an OOM kill for large VLM-as-text checkpoints. Build it
        # directly in the target dtype instead.
        _dtype = model_kw.get("torch_dtype", torch.bfloat16)
        causal_model = AutoModelForCausalLM.from_config(text_config, torch_dtype=_dtype)

        missing, unexpected = causal_model.load_state_dict(state_dict, strict=False)
        if missing:
            # Tied weights (e.g. lm_head == embed_tokens) appear as "missing"
            # but are resolved automatically by tie_weights(); only warn for others.
            non_tied = [k for k in missing if "lm_head" not in k]
            if non_tied:
                logger.warning(
                    "HF LM: VLM prefix-strip for %s: %d non-lm_head missing keys: %s",
                    model_name, len(non_tied), non_tied[:5],
                )
        causal_model.tie_weights()
        causal_model.eval()
        logger.info(
            "HF LM: loaded %s via VLM prefix-strip (%d keys, %d missing, %d unexpected)",
            model_name, len(state_dict), len(missing), len(unexpected),
        )
        return causal_model

    except Exception as exc:
        logger.warning("HF LM: VLM prefix-strip failed for %s: %s", model_name, exc)
        return None


def _load_lm_head_from_hf_cache(
    model_name: str,
    revision: str | None,
    hf_hub_cache: str | None,
) -> "torch.Tensor | None":
    """Load the LM head weight from HF cache without a full model reload.

    Tries the following safetensors keys in order:
      1. ``language_model.lm_head.weight``  — explicit LM head (Ministral-3-8B, 3-14B)
      2. ``language_model.model.embed_tokens.weight``  — tied embed (Ministral-3-3B)

    Returns the weight tensor on CPU, or ``None`` if neither key is found or the
    cache cannot be read.
    """
    import json

    cache_root = Path(hf_hub_cache or "~/.cache/huggingface/hub").expanduser()
    safe_org_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = cache_root / safe_org_name / "snapshots"
    if not snapshots_dir.exists():
        logger.warning("HF LM: HF cache snapshots not found at %s", snapshots_dir)
        return None

    # Resolve snapshot directory (commit hash or branch ref)
    snap: Path | None = None
    if revision:
        direct = snapshots_dir / revision
        if direct.is_dir():
            snap = direct
        else:
            ref_path = cache_root / safe_org_name / "refs" / revision
            if ref_path.is_file():
                snap = snapshots_dir / ref_path.read_text().strip()
    if snap is None or not snap.is_dir():
        snaps = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        snap = max(snaps, key=lambda p: p.stat().st_mtime) if snaps else None

    if snap is None or not snap.is_dir():
        return None

    # Keys to try in order: explicit lm_head, then tied embed_tokens fallback
    candidate_keys = [
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
    ]

    try:
        from safetensors.torch import safe_open  # type: ignore[import]

        # Sharded safetensors (large models)
        index_path = snap / "model.safetensors.index.json"
        if index_path.is_file():
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            for target_key in candidate_keys:
                shard_fname = weight_map.get(target_key)
                if shard_fname is None:
                    continue
                shard_path = snap / shard_fname
                if not shard_path.is_file():
                    continue
                with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                    tensor = sf.get_tensor(target_key)
                logger.debug("HF LM: loaded %s from shard for %s", target_key, model_name)
                return tensor
            # No candidate found in index
            return None

        # Single-file safetensors
        single_sf = snap / "model.safetensors"
        if single_sf.is_file():
            with safe_open(str(single_sf), framework="pt", device="cpu") as sf:
                all_keys = set(sf.keys())
                for target_key in candidate_keys:
                    if target_key in all_keys:
                        logger.debug("HF LM: loaded %s from single shard for %s", target_key, model_name)
                        return sf.get_tensor(target_key)
    except Exception as exc:
        logger.debug("HF LM: could not load lm_head from cache for %s: %s", model_name, exc)

    return None


def _load_plain_lm_head_from_hf_cache(
    model_name: str,
    revision: str | None,
    hf_hub_cache: str | None,
) -> "torch.Tensor | None":
    """Load a plain (non-VLM) ``lm_head.weight`` tensor straight from the HF
    cache snapshot, without instantiating the model.

    Used to repair ``trust_remote_code`` checkpoints whose bundled config
    class has a bug that forces ``tie_word_embeddings=True`` regardless of
    what the checkpoint actually was — e.g. ``Lanni-ni/hard_{2,3,5}gram_*``
    (see ``forgetting_transformer.model.sliding_window.configuration_sliding_window``,
    whose ``super().__init__()`` never forwards ``tie_word_embeddings``, unlike
    every sibling config in that repo). Such models report ``lm_head.weight``
    as a "missing" key (silently tied to the input embedding) even when the
    checkpoint file has its own, separately-trained ``lm_head.weight`` tensor.
    Returns ``None`` if no ``lm_head.weight`` key exists in the checkpoint at
    all (i.e. the model really was saved tied) or the cache can't be read.
    """
    import json

    cache_root = Path(hf_hub_cache or "~/.cache/huggingface/hub").expanduser()
    safe_org_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = cache_root / safe_org_name / "snapshots"
    if not snapshots_dir.exists():
        return None

    snap: Path | None = None
    if revision:
        direct = snapshots_dir / revision
        if direct.is_dir():
            snap = direct
        else:
            ref_path = cache_root / safe_org_name / "refs" / revision
            if ref_path.is_file():
                snap = snapshots_dir / ref_path.read_text().strip()
    if snap is None or not snap.is_dir():
        snaps = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        snap = max(snaps, key=lambda p: p.stat().st_mtime) if snaps else None
    if snap is None or not snap.is_dir():
        return None

    try:
        from safetensors.torch import safe_open  # type: ignore[import]

        index_path = snap / "model.safetensors.index.json"
        if index_path.is_file():
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            shard_fname = weight_map.get("lm_head.weight")
            if shard_fname is None:
                return None
            shard_path = snap / shard_fname
            if not shard_path.is_file():
                return None
            with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                return sf.get_tensor("lm_head.weight")

        single_sf = snap / "model.safetensors"
        if single_sf.is_file():
            with safe_open(str(single_sf), framework="pt", device="cpu") as sf:
                if "lm_head.weight" in set(sf.keys()):
                    return sf.get_tensor("lm_head.weight")
    except Exception as exc:
        logger.debug("HF LM: could not check lm_head.weight in cache for %s: %s", model_name, exc)

    return None


_FORGETTING_DIRECT_PREFIX = "Lanni-ni/forgetting_"

_FORGETTING_CONFIG_IGNORED_KEYS = {
    "architectures", "auto_map", "transformers_version", "model_type", "torch_dtype",
}


def _load_forgetting_direct(model_name: str, model_rev_fp: dict[str, object]) -> tp.Any:
    """Load a ``Lanni-ni/forgetting_*`` checkpoint via the local (verified-
    matching) ``forgetting_transformer`` classes, bypassing the repo's
    ``trust_remote_code`` snapshot.

    That snapshot is stale: it uses an older module layout (``attn.f_proj`` +
    separate ``mlp.up_proj``/``gate_proj(hidden)``) that mismatches the actual
    saved checkpoint weights (``attn.fgate_proj`` + a single fused
    ``mlp.gate_proj(2*hidden)``, chunked at forward time). Loading via
    ``AutoModelForCausalLM(trust_remote_code=True)`` silently leaves the
    mismatched-shape layers randomly initialised instead of raising, producing
    a model that "runs" but gives near-random (very high perplexity) output.
    See ``neuroai/models/registry.py``'s ``"forgetting"`` family comment and
    a one-off parameter-counting diagnostic (2026-07, since removed) that
    this loader mirrors.
    """
    import json
    import sys

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    _repo_root = Path(__file__).resolve().parents[2] / "forgetting-transformer" / "src"
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from forgetting_transformer.model.forgetting_transformer.configuration_forgetting_transformer import (
        ForgettingTransformerConfig,
    )
    from forgetting_transformer.model.forgetting_transformer.modeling_forgetting_transformer import (
        ForgettingTransformerForCausalLM,
    )

    revision = model_rev_fp.get("revision")
    cfg_path = hf_hub_download(model_name, "config.json", revision=revision)
    with open(cfg_path) as f:
        raw_cfg = json.load(f)
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k not in _FORGETTING_CONFIG_IGNORED_KEYS}
    config = ForgettingTransformerConfig(**cfg_kwargs)

    model = ForgettingTransformerForCausalLM(config)

    try:
        weights_path = hf_hub_download(model_name, "model.safetensors", revision=revision)
        state_dict = load_file(weights_path)
    except Exception:
        import torch

        weights_path = hf_hub_download(model_name, "pytorch_model.bin", revision=revision)
        state_dict = torch.load(weights_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # lm_head.weight is expected to be missing when tie_word_embeddings=True
    # (correctly tied to the input embedding at construction time by the
    # local, non-buggy config/model classes).
    real_missing = [k for k in missing if k != "lm_head.weight"]
    if real_missing or unexpected:
        raise RuntimeError(
            f"direct load for {model_name} still mismatched: "
            f"missing={real_missing} unexpected={unexpected}"
        )
    return model


_MOLM_DIRECT_PREFIX = "ibm-research/MoLM-"

_MOLM_CONFIG_IGNORED_KEYS = {
    "architectures", "auto_map", "transformers_version", "model_type", "torch_dtype",
}


def _load_moduleformer_direct(model_name: str, model_rev_fp: dict[str, object]) -> tp.Any:
    """Load an ``ibm-research/MoLM-*`` checkpoint via the vendored ``moduleformer``
    classes (``moduleformer/moduleformer/`` at the repo root), bypassing
    ``AutoModelForCausalLM``/``AutoConfig`` entirely.

    IBM never published this as a standard HF ``trust_remote_code`` repo --
    ``config.json`` has no ``auto_map`` entry, and ``"moduleformer"`` isn't a
    registered ``AutoConfig`` model_type in this transformers version (5.13.1
    at the time this was added), so ``AutoModelForCausalLM(model_type=
    "moduleformer", trust_remote_code=True)`` raises ``ValueError:
    Transformers does not recognize this architecture`` before ever reaching
    the checkpoint's weights. Same treatment as ``_load_forgetting_direct``
    above (vendored source, import the real classes, load config + weights by
    hand) -- see ``neuroai/models/registry.py``'s ``"molm"`` family comment.
    """
    import json
    import sys

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    _repo_root = Path(__file__).resolve().parents[2] / "moduleformer"
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from moduleformer.configuration_moduleformer import ModuleFormerConfig
    from moduleformer.modeling_moduleformer import ModuleFormerForCausalLM

    revision = model_rev_fp.get("revision")
    cfg_path = hf_hub_download(model_name, "config.json", revision=revision)
    with open(cfg_path) as f:
        raw_cfg = json.load(f)
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k not in _MOLM_CONFIG_IGNORED_KEYS}
    config = ModuleFormerConfig(**cfg_kwargs)

    model = ModuleFormerForCausalLM(config)

    try:
        weights_path = hf_hub_download(model_name, "model.safetensors", revision=revision)
        state_dict = load_file(weights_path)
    except Exception:
        import torch

        try:
            weights_path = hf_hub_download(model_name, "pytorch_model.bin", revision=revision)
            state_dict = torch.load(weights_path, map_location="cpu")
        except Exception:
            # Sharded checkpoint (e.g. ibm-research/MoLM-700M-8B ships
            # pytorch_model-0000N-of-0000M.bin + an index, no single-file
            # model.safetensors/pytorch_model.bin) -- merge every shard named
            # in the index's weight_map.
            index_path = hf_hub_download(model_name, "pytorch_model.bin.index.json", revision=revision)
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            state_dict = {}
            for shard_fname in sorted(set(weight_map.values())):
                shard_path = hf_hub_download(model_name, shard_fname, revision=revision)
                state_dict.update(torch.load(shard_path, map_location="cpu"))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"direct load for {model_name} mismatched: missing={missing} unexpected={unexpected}"
        )
    return model


_FOX_DIRECT_PREFIX = "zhixuan-lin/"

_FOX_CONFIG_IGNORED_KEYS = {
    "architectures", "auto_map", "transformers_version", "model_type", "torch_dtype",
}


def _load_fox_direct(model_name: str, model_rev_fp: dict[str, object]) -> tp.Any:
    """Load a ``zhixuan-lin/{fox,transformer}-*`` checkpoint via the vendored
    (and locally patched) ``forgetting_transformer`` classes, bypassing
    ``AutoModelForCausalLM``/``AutoConfig`` entirely.

    Like ``ibm-research/MoLM-*``, these were never published as standard HF
    ``trust_remote_code`` repos -- ``config.json`` has no ``auto_map`` entry
    and ships no ``.py`` files at all, and their model_types
    (``forgetting_transformer-project_fox`` / ``transformer-project_fox``)
    aren't registered ``AutoConfig`` types -- so ``trust_remote_code=True``
    alone does nothing here; the real classes must be imported directly from
    ``forgetting-transformer/src`` (the same vendored copy
    ``_load_forgetting_direct`` uses, patched 2026-09-12 for two transformers/
    fla version-skew bugs — see ``neuroai/models/registry.py``'s ``"fox"``
    family comment for the full writeup). Which of the two model classes to
    use is picked from the checkpoint's own ``architectures`` field.
    """
    import json
    import sys

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    _repo_root = Path(__file__).resolve().parents[2] / "forgetting-transformer" / "src"
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    revision = model_rev_fp.get("revision")
    cfg_path = hf_hub_download(model_name, "config.json", revision=revision)
    with open(cfg_path) as f:
        raw_cfg = json.load(f)
    arch = raw_cfg.get("architectures", [None])[0]
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k not in _FOX_CONFIG_IGNORED_KEYS}

    if arch == "TransformerForCausalLM":
        from forgetting_transformer.model.transformer.configuration_transformer import (
            TransformerConfig,
        )
        from forgetting_transformer.model.transformer.modeling_transformer import (
            TransformerForCausalLM,
        )

        config = TransformerConfig(**cfg_kwargs)
        model = TransformerForCausalLM(config)
    else:
        from forgetting_transformer.model.forgetting_transformer.configuration_forgetting_transformer import (
            ForgettingTransformerConfig,
        )
        from forgetting_transformer.model.forgetting_transformer.modeling_forgetting_transformer import (
            ForgettingTransformerForCausalLM,
        )

        config = ForgettingTransformerConfig(**cfg_kwargs)
        model = ForgettingTransformerForCausalLM(config)

    weights_path = hf_hub_download(model_name, "model.safetensors", revision=revision)
    state_dict = load_file(weights_path)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"direct load for {model_name} mismatched: missing={missing} unexpected={unexpected}"
        )
    return model


_tied_weights_fallback_patched = False


def _patch_transformers_tied_weights_fallback() -> None:
    """Make ``get_total_byte_count`` tolerate models missing
    ``all_tied_weights_keys`` (idempotent, applied once).

    ``all_tied_weights_keys`` is set by ``PreTrainedModel.post_init()`` — a
    method every custom model class is expected to call at the end of its
    own ``__init__``. Some ``trust_remote_code`` models (e.g. BabyLM-
    community's ``gpt-bert-*`` custom modeling file) never call it — either
    an omission, or the file predates this attribute being added to
    ``post_init`` — so ``model.all_tied_weights_keys`` raises
    ``AttributeError`` the moment ``from_pretrained``'s internal
    ``caching_allocator_warmup`` step touches it. That step is a memory
    pre-allocation size estimate, not correctness-critical for the forward
    pass we actually need, so treating a missing attribute as "no known tied
    weights" (empty dict) is a safe fallback — this only changes behavior
    for models that are already missing the attribute; well-formed models
    are unaffected.
    """
    global _tied_weights_fallback_patched
    if _tied_weights_fallback_patched:
        return
    try:
        import transformers.modeling_utils as _mu

        _orig_get_total_byte_count = _mu.get_total_byte_count

        def _get_total_byte_count_safe(model, accelerator_device_map, hf_quantizer=None):
            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = {}
            return _orig_get_total_byte_count(model, accelerator_device_map, hf_quantizer)

        _mu.get_total_byte_count = _get_total_byte_count_safe
        _tied_weights_fallback_patched = True
    except Exception:
        logger.warning("Could not patch transformers tied-weights fallback", exc_info=True)


def _repair_gpt_bert_position_indices(model: tp.Any) -> None:
    """Recompute BabyLM-community gpt-bert's relative-position bucket buffers.

    Each ``Attention`` layer in the vendor's ``modeling_gpt_bert.py`` builds a
    ``position_indices`` lookup table (relative-position bucket per
    query/key-position pair) from ``config.position_bucket_size`` /
    ``config.max_position_embeddings`` via a plain ``torch.arange``-based
    formula inside ``__init__``, registered with ``persistent=False`` (so it's
    never part of the checkpoint's state_dict — it's meant to be recomputed
    fresh every time, deterministically, from config alone). That's fine under
    ``from_config`` (verified: buffer values land in the expected
    ``[0, 2*position_bucket_size - 2]`` range) — but under ``from_pretrained``'s
    meta-device low-memory init path, this custom class never calls
    ``self.post_init()`` (same omission ``_patch_transformers_tied_weights_fallback``
    works around for ``all_tied_weights_keys``), so the buffer is left holding
    whatever uninitialized memory the meta->real device materialization step
    happened to allocate — arbitrary huge int64 values instead of small bucket
    indices. Every later ``F.embedding(position_indices[...], ...)`` call then
    raises ``IndexError: index out of range in self``, deterministically, for
    every input (not data-dependent). Recomputed here in place using the exact
    same formula as the vendor's ``Attention.__init__`` / ``make_log_bucket_position``,
    rather than trying to make a class transformers wasn't written to support
    call ``post_init()``.
    """
    cfg = getattr(model, "config", None)
    bucket_size = getattr(cfg, "position_bucket_size", None)
    max_pos = getattr(cfg, "max_position_embeddings", None)
    if bucket_size is None or max_pos is None:
        return

    fixed_any = False
    for name, buf in list(model.named_buffers()):
        if not name.endswith("position_indices"):
            continue
        import math

        import torch

        rel = (
            torch.arange(max_pos, dtype=torch.long).unsqueeze(1)
            - torch.arange(max_pos, dtype=torch.long).unsqueeze(0)
        )
        sign = torch.sign(rel)
        mid = bucket_size // 2
        abs_pos = torch.where(
            (rel < mid) & (rel > -mid),
            torch.full_like(rel, mid - 1),
            rel.abs().clamp(max=max_pos - 1),
        )
        log_pos = (
            torch.ceil(torch.log(abs_pos.float() / mid) / math.log((max_pos - 1) / mid) * (mid - 1))
            .int() + mid
        )
        bucket_pos = torch.where(abs_pos <= mid, rel, log_pos * sign).long()
        fixed = (bucket_size - 1 + bucket_pos).to(device=buf.device)

        *parent_path, buf_attr = name.split(".")
        owner = model
        for p in parent_path:
            owner = getattr(owner, p)
        owner.register_buffer(buf_attr, fixed, persistent=False)
        fixed_any = True

    if fixed_any:
        logger.info(
            "HF LM: repaired GPT-BERT position_indices buffer(s) — "
            "meta-device from_pretrained init left them uninitialized "
            "(see _repair_gpt_bert_position_indices)"
        )


def _get_cached_causal_lm(
    model_name: str,
    *,
    revision: str | None,
    device: str,
    load_kw: dict[str, object],
    tokenizer_no_revision: bool = False,
    tokenizer_revision: str | None = None,
    n_params: float | None = None,
    untrained: bool = False,
    untrained_seed: int = 0,
) -> tuple[tp.Any, tp.Any]:
    """Load and LRU-cache (size-1) a causal-LM tokenizer + model pair.

    ``tokenizer_no_revision=True`` loads the tokenizer from the **base** model
    (no revision / step), matching l-bg's ``extract_llm_activations.py`` which
    always calls ``AutoTokenizer.from_pretrained(base_name)`` for Pythia.
    The *model* weights still come from ``revision``.

    When multiple GPUs are visible (or ``n_params`` ≥ 6B), uses ``device_map='auto'``
    with per-GPU ``max_memory`` headroom for activations (same idea as LPP perplexity).

    ``untrained=True`` skips weight loading entirely: the architecture and
    tokenizer are still resolved from ``model_name``/``revision`` (so
    tokenization and layer count match the trained model exactly), but the
    model is instantiated via ``from_config`` instead of ``from_pretrained``,
    giving HuggingFace's own architecture-correct random initialisation
    (rather than a naive ``torch.randn`` that would ignore the architecture's
    init scheme). Used for untrained-baseline brain scores.
    """
    _patch_transformers_tied_weights_fallback()
    global _hf_slot_key, _hf_slot_tokenizer, _hf_slot_model
    rev_key = revision if revision not in (None, "", "main") else "main"
    slot = (
        model_name,
        rev_key,
        str(device),
        tokenizer_no_revision,
        tokenizer_revision or "",
        untrained,
        untrained_seed if untrained else 0,
    )
    if _hf_slot_key == slot and _hf_slot_model is not None:
        _ensure_pad_token_for_batches(_hf_slot_tokenizer)
        return _hf_slot_tokenizer, _hf_slot_model
    clear_hf_model_cache()
    import torch

    # gpt-bert's custom DWA-accumulator code hard-codes a float32
    # torch.Tensor() inside its in-place .set_() buffer update (see
    # InPlaceSetSlice.forward in the model's remote code) — it crashes as
    # soon as the model runs in bf16. Auto-detected here (rather than relying
    # on every one of ~30 call sites across the codebase to remember to pass
    # dtype="float32") so no future caller can reintroduce this by omission.
    if "torch_dtype" not in load_kw and "gpt-bert" in model_name.lower():
        load_kw = {**load_kw, "torch_dtype": torch.float32}

    model_rev = revision if revision not in (None, "", "main") else None
    model_rev_fp = {"revision": model_rev} if model_rev is not None else {}
    if tokenizer_revision is not None:
        tok_rev_fp = {"revision": tokenizer_revision}
    elif tokenizer_no_revision:
        tok_rev_fp = {}
    else:
        tok_rev_fp = model_rev_fp
    _trc = bool(load_kw.get("trust_remote_code", False))
    tokenizer = _load_hf_tokenizer(model_name, tok_rev_fp, trust_remote_code=_trc)
    # Hidden states are requested in forward(), not here — passing
    # output_hidden_states to from_pretrained breaks GPT-NeoX when we inject a
    # custom config for sharded Pythia-2.8B weights (transformers_weights index).
    model_kw = {k: v for k, v in load_kw.items() if k != "output_hidden_states"}
    model_kw.update(model_rev_fp)
    model_kw = _configure_pythia_28b_model_load(
        model_name, model_rev, model_rev_fp, model_kw, rev_key=rev_key,
    )

    if untrained:
        # Random-init baseline: resolve config + architecture as usual, but
        # never touch pretrained weights. None of the pretrained-weight-repair
        # special cases below (VLM prefix-strip, Pythia-2.8B sharding,
        # trust_remote_code lm_head fixups) apply here.
        from transformers import AutoConfig, set_seed

        use_device_map = False
        set_seed(untrained_seed)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=_trc, **model_rev_fp)
        model_cls = _resolve_causal_lm_class(model_name, model_rev_fp, trust_remote_code=_trc)
        _dtype = model_kw.get(
            "torch_dtype", torch.bfloat16 if device.startswith("cuda") else torch.float32,
        )
        model = model_cls.from_config(config, trust_remote_code=_trc, torch_dtype=_dtype)
        _missing = []
    elif model_name.startswith(_FORGETTING_DIRECT_PREFIX):
        # Bypass the buggy trust_remote_code snapshot entirely -- see
        # _load_forgetting_direct's docstring. Forgetting checkpoints are all
        # small (27-45M params), so device_map is never needed here.
        use_device_map = False
        model = _load_forgetting_direct(model_name, model_rev_fp)
        _dtype = model_kw.get("torch_dtype", torch.bfloat16)
        model = model.to(dtype=_dtype)
        _missing: list[str] = []
    elif model_name.startswith(_MOLM_DIRECT_PREFIX):
        # Bypass the unregistered "moduleformer" AutoConfig model_type -- see
        # _load_moduleformer_direct's docstring. Always single-GPU .to(device)
        # below (like the forgetting/FoX direct-load branches) -- unlike the
        # generic AutoModelForCausalLM path, _load_moduleformer_direct never
        # implements real accelerate device_map sharding, so trusting
        # _apply_device_map_if_needed()'s True here (it returns True for
        # essentially any real GPU job) left the model silently stranded on
        # CPU while callers moved their inputs to cuda -- "Expected all
        # tensors to be on the same device" crashes in eeg_erp.py's
        # extract_sentence_features. All 3 MoLM checkpoints (up to 8B total
        # params, ~16 GiB in bf16) fit comfortably on one A100/H100/H200 GPU
        # anyway. Confirmed 2026-09-23.
        use_device_map = False
        model = _load_moduleformer_direct(model_name, model_rev_fp)
        _dtype = model_kw.get("torch_dtype", torch.bfloat16)
        model = model.to(dtype=_dtype)
        _missing: list[str] = []
    elif model_name.startswith(_FOX_DIRECT_PREFIX):
        # Bypass trust_remote_code entirely -- see _load_fox_direct's
        # docstring. All checkpoints are 760M, so device_map is never needed.
        use_device_map = False
        model = _load_fox_direct(model_name, model_rev_fp)
        _dtype = model_kw.get("torch_dtype", torch.bfloat16)
        model = model.to(dtype=_dtype)
        _missing: list[str] = []
    else:
        use_device_map = _apply_device_map_if_needed(model_kw, device, n_params=n_params)

        model_cls = _resolve_causal_lm_class(model_name, model_rev_fp, trust_remote_code=_trc)
        model, _load_info = model_cls.from_pretrained(
            model_name, output_loading_info=True, **model_kw
        )
        _missing = _load_info.get("missing_keys", [])
    # Detect multimodal checkpoints silently loaded with wrong (text-only) class:
    # the VLM checkpoint stores weights under "language_model.*" but the text-only
    # class expects flat "model.*" keys — all transformer weights end up MISSING and
    # are randomly initialised.  Fix by prefix-stripping into the correct CausalLM.
    if len(_missing) > 20 and any("layers" in k for k in _missing):
        logger.warning(
            "HF LM: %s — %d missing keys (likely VLM checkpoint loaded as text LM); "
            "retrying via prefix-strip into CausalLM",
            model_name, len(_missing),
        )
        del model
        import gc as _gc; _gc.collect()

        # Primary: strip "language_model." prefix from checkpoint and load into
        # AutoModelForCausalLM.from_config(text_config).  This correctly initialises
        # ALL weights (layers + embed_tokens + lm_head) from the cached shards.
        causal_model = _load_causal_lm_from_vlm_checkpoint(model_name, model_kw)
        if causal_model is not None:
            model = causal_model
            # _load_causal_lm_from_vlm_checkpoint always returns the model on CPU.
            # If device_map was requested, the model.to(device) branch below is
            # skipped (use_device_map=True), leaving the model on CPU and causing
            # forward passes to run at CPU speed with GPU util=0%.  Fix: move
            # explicitly to the target device/dtype here, before that branch.
            if use_device_map:
                _dtype = model_kw.get("torch_dtype", None)
                if _dtype is None:
                    _dtype = torch.bfloat16
                model = model.to(dtype=_dtype).to(device)
                # Clear the first (failed) model's GPU memory that del + gc didn't free
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(
                    "HF LM: VLM prefix-strip model moved to %s (%s)", device, _dtype,
                )
        else:
            # Fallback: load via AutoModel (backbone only), extract language_model,
            # and attach lm_head from safetensors.  All transformer weights remain
            # randomly initialised in this path — use only when cache is unavailable.
            logger.warning(
                "HF LM: prefix-strip unavailable for %s — falling back to backbone "
                "extraction (transformer weights will be randomly initialised)",
                model_name,
            )
            try:
                from transformers import AutoModelForVision2Seq as _VLMCls
            except ImportError:
                from transformers import AutoModel as _VLMCls  # type: ignore[assignment]
            _vlm = _VLMCls.from_pretrained(model_name, **model_kw)
            if not hasattr(_vlm, "language_model"):
                raise ValueError(
                    f"VLM fallback for {model_name} succeeded but has no .language_model "
                    "attribute — cannot extract text backbone"
                )
            model = _vlm.language_model
            # Attach lm_head if missing (backbone has no lm_head attribute)
            if not hasattr(model, "lm_head"):
                import torch.nn as _nn
                lm_head_w = _load_lm_head_from_hf_cache(
                    model_name,
                    revision=model_kw.get("revision"),
                    hf_hub_cache=os.environ.get("HF_HUB_CACHE"),
                )
                if lm_head_w is not None:
                    d_vocab, d_hid = lm_head_w.shape
                    _lm_head = _nn.Linear(d_hid, d_vocab, bias=False)
                    _lm_head.weight = _nn.Parameter(lm_head_w)
                    model.lm_head = _lm_head
                    logger.info(
                        "HF LM: loaded lm_head from checkpoint for %s (vocab=%d, d=%d)",
                        model_name, d_vocab, d_hid,
                    )
                elif hasattr(_vlm, "lm_head"):
                    model.lm_head = _vlm.lm_head
                    logger.info("HF LM: attached lm_head from VLM for %s", model_name)
                else:
                    _embed = (
                        getattr(model, "embed_tokens", None)
                        or getattr(getattr(model, "model", None), "embed_tokens", None)
                    )
                    if _embed is None:
                        try:
                            _embed = model.get_input_embeddings()
                        except Exception:
                            pass
                    if _embed is None:
                        try:
                            _embed = _vlm.get_input_embeddings()
                        except Exception:
                            pass
                    if _embed is not None:
                        vocab_size, d_hid = _embed.weight.shape
                        _lm_head = _nn.Linear(d_hid, vocab_size, bias=False)
                        _lm_head.weight = _embed.weight
                        model.lm_head = _lm_head
                        logger.info(
                            "HF LM: tied lm_head = embed_tokens for %s", model_name,
                        )
                    else:
                        logger.warning(
                            "HF LM: no lm_head found for %s — surprisal will fail", model_name,
                        )
            del _vlm
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("HF LM: extracted .language_model from VLM for %s", model_name)
    elif _trc and "lm_head.weight" in _missing and hasattr(model, "lm_head"):
        # Some trust_remote_code checkpoints (e.g. Lanni-ni/hard_{2,3,5}gram_*,
        # forgetting/stickbreaking siblings) bundle remote code whose
        # tie_word_embeddings handling is broken, so a "missing" lm_head.weight
        # is left as a freshly-initialised tensor and never resolved — either
        # tied to the input embedding (correct, if the checkpoint has no
        # separate lm_head saved) or loaded from a real saved tensor (if one
        # exists). Left alone, the fresh init can be all-zero/NaN, which makes
        # logits = hidden_states @ lm_head.weight.T degenerate to a constant
        # (all-zero) output for every input — i.e. a uniform softmax regardless
        # of the model's real weights. Confirmed by a one-off weight-checksum diagnostic
        # (2026-07-13, since removed): every hard_ngram model showed lm_head sum=mean=std=0 (or
        # nan) and logits_mean=logits_std=0.0 exactly.
        _real_lm_head = _load_plain_lm_head_from_hf_cache(
            model_name,
            revision=model_kw.get("revision"),
            hf_hub_cache=os.environ.get("HF_HUB_CACHE"),
        )
        import torch.nn as _nn

        if _real_lm_head is not None:
            _orig_device = model.lm_head.weight.device
            with torch.no_grad():
                new_weight = _nn.Parameter(
                    _real_lm_head.to(device=_orig_device, dtype=model.lm_head.weight.dtype)
                )
                model.lm_head.weight = new_weight
            logger.warning(
                "HF LM: %s — lm_head.weight was incorrectly tied by a buggy "
                "trust_remote_code config; loaded the real untied weight from "
                "the checkpoint instead", model_name,
            )
        else:
            # No separate lm_head tensor exists in the checkpoint at all, so
            # the freshly-initialised placeholder transformers put there
            # (whether it looks obviously degenerate -- all-zero/NaN, seen
            # with device_map -- or just ordinary random noise, seen without
            # device_map -- is *always* meaningless) must be tied to the input
            # embedding. Don't gate this on "does it look degenerate": a
            # non-device_map load can leave an innocuous-looking but equally
            # wrong random lm_head that would silently slip past a zero/NaN check.
            embed = model.get_input_embeddings()
            current = model.lm_head.weight.detach()
            if embed.weight.device != current.device or embed.weight.dtype != current.dtype:
                with torch.no_grad():
                    embed.weight = _nn.Parameter(
                        embed.weight.to(device=current.device, dtype=current.dtype)
                    )
            # True tying: assign the *same* Parameter object so downstream
            # param-counting code (sum(p.numel() for p in model.parameters()))
            # correctly deduplicates it instead of silently double-counting.
            model.lm_head.weight = embed.weight
            logger.warning(
                "HF LM: %s — lm_head.weight had no real saved tensor (buggy "
                "trust_remote_code tie_word_embeddings handling); tied it to "
                "the input embedding instead", model_name,
            )
    if type(model).__name__.lower().startswith("gptbert"):
        _repair_gpt_bert_position_indices(model)

    model.eval()

    if not use_device_map:
        model.to(device)
    _ensure_pad_token_for_batches(tokenizer)
    _hf_slot_key = slot
    _hf_slot_tokenizer = tokenizer
    _hf_slot_model = model
    logger.info(
        "HF LM: loaded & cached %s @%s (device_map=%s, tok_rev=%s)",
        model_name,
        rev_key,
        use_device_map,
        tokenizer_revision or ("main" if tokenizer_no_revision else rev_key),
    )
    return tokenizer, model


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _feature_cache_path(
    cache_dir: Path | str,
    model_name: str,
    revision: str | None,
    lang: str,
    section: int,
    context_words: int | None,
    max_length: int | None,
    layer_index: LayerIndex,
    untrained: bool = False,
    untrained_seed: int = 0,
) -> Path:
    safe_model = model_name.replace("/", "_")
    safe_rev = (revision or "main").replace("/", "_").replace("-", "_")
    if untrained:
        safe_rev = f"{safe_rev}_untrained_seed{untrained_seed}"
    layer_tag = f"layer{layer_index}" if isinstance(layer_index, int) else layer_index
    ctx_tag = "full" if context_words is None else str(context_words)
    maxlen_tag = "model" if max_length is None else str(max_length)
    filename = (
        f"{safe_model}__{safe_rev}__{lang.lower()}"
        f"__sec{section:02d}"
        f"__ctx{ctx_tag}"
        f"__maxlen{maxlen_tag}"
        f"__{layer_tag}.npz"
    )
    return Path(cache_dir) / filename


def _lbg_feature_cache_path(
    cache_dir: Path | str,
    model_name: str,
    revision: str | None,
    lang: str,
    section: int,
    maxlen: int,
    stride: int,
    layer_index: LayerIndex,
) -> Path:
    """Separate cache keys from contextual-window extraction (``.npz``)."""
    safe_model = model_name.replace("/", "_")
    safe_rev = (revision or "main").replace("/", "_").replace("-", "_")
    layer_tag = f"layer{layer_index}" if isinstance(layer_index, int) else layer_index
    filename = (
        f"{safe_model}__{safe_rev}__{lang.lower()}"
        f"__sec{section:02d}"
        f"__lbg_ml{maxlen}"
        f"__stride{stride}"
        f"__{layer_tag}.npz"
    )
    return Path(cache_dir) / filename


def _fulltext_feature_cache_path(
    cache_dir: Path | str,
    model_name: str,
    revision: str | None,
    lang: str,
    section: int,
    maxlen: int,
    stride_delta: int,
    layer_index: LayerIndex,
    untrained: bool = False,
    untrained_seed: int = 0,
) -> Path:
    """Cache key for full-text sliding-window extraction (one pass per chunk)."""
    safe_model = model_name.replace("/", "_")
    safe_rev = (revision or "main").replace("/", "_").replace("-", "_")
    if untrained:
        safe_rev = f"{safe_rev}_untrained_seed{untrained_seed}"
    layer_tag = f"layer{layer_index}" if isinstance(layer_index, int) else layer_index
    filename = (
        f"{safe_model}__{safe_rev}__{lang.lower()}"
        f"__sec{section:02d}"
        f"__fulltext_ml{maxlen}"
        f"__sdelta{stride_delta}"
        f"__{layer_tag}.npz"
    )
    return Path(cache_dir) / filename


def _load_cached_features(cache_path: Path) -> np.ndarray | None:
    if cache_path.exists():
        try:
            return np.load(str(cache_path))["features"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load feature cache %s: %s", cache_path, exc)
    return None


def _save_cached_features(cache_path: Path, features: np.ndarray) -> None:
    """Write ``features`` atomically to ``cache_path`` (``.npz`` with key ``features``).

    Uses a visible temp name (no leading dot — hidden temps are unreliable on Lustre)
    and retries ``os.replace`` briefly for distributed filesystem latency.
    """
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # NumPy appends ``.npz`` when the path does not end with ``.npz``; keep ``.part.npz``.
    tmp = cache_path.parent / f"{cache_path.stem}.part{os.getpid()}.npz"
    # For large arrays (>4 GiB) skip zlib compression: savez_compressed runs at
    # 10–50 MB/s on dense float32 data, turning an 82 GiB GPT-NeoX-20B cache save
    # into a 2-hour CPU stall.  np.savez writes at disk speed (~1 GB/s on Lustre).
    size_gib = features.nbytes / (1024 ** 3)
    save_fn = np.savez_compressed if size_gib < 4.0 else np.savez
    try:
        save_fn(str(tmp), features=features)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise OSError(f"feature cache write produced no file: {tmp}")
        last_err: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, cache_path)
                last_err = None
                break
            except (FileNotFoundError, OSError) as exc:
                last_err = exc
                if not tmp.is_file():
                    raise OSError(
                        f"feature cache temp missing before replace: {tmp}"
                    ) from exc
                time.sleep(0.25 * (attempt + 1))
        if last_err is not None:
            raise last_err
    finally:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)


def _try_save_cached_features(cache_path: Path, features: np.ndarray) -> bool:
    """Like :func:`_save_cached_features`, but log and return False instead of raising."""
    try:
        _save_cached_features(cache_path, features)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "HF word features: cache write failed for %s (%s); continuing without cache",
            cache_path,
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Layer stacking (OPT and similar: embedding dim != transformer hidden size)
# ---------------------------------------------------------------------------

def _select_uniform_hidden_layers(
    layer_tensors: list,
    *,
    model_name: str,
    warned: list[bool],
) -> list:
    """Keep only layers whose hidden size matches the modal dimension.

    OPT exposes ``word_embed_proj_dim`` (e.g. 512) at layer 0 while transformer
    blocks use ``hidden_size`` (e.g. 1024).  Stacking all layers requires dropping
    mismatched entries (same rule as :func:`contextual_word_embeddings_hf_lbg`).
    """
    dims = [int(t.shape[-1]) for t in layer_tensors]
    if len(set(dims)) <= 1:
        return layer_tensors
    from collections import Counter

    modal_dim = Counter(dims).most_common(1)[0][0]
    if not warned[0]:
        n_drop = sum(d != modal_dim for d in dims)
        logger.warning(
            "HF word features: %s has non-uniform hidden dims across %d layers — "
            "dropping %d layer(s) whose dim != %d (embedding projection mismatch)",
            model_name,
            len(layer_tensors),
            n_drop,
            modal_dim,
        )
        warned[0] = True
    kept = [t for t in layer_tensors if int(t.shape[-1]) == modal_dim]
    if not kept:
        raise ValueError(
            f"{model_name}: no layers left after hidden-dim filter; dims={sorted(set(dims))}"
        )
    return kept


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------

def _chunks(n: int, size: int) -> tp.Iterator[tuple[int, int]]:
    for i in range(0, n, size):
        yield i, min(i + size, n)


# ---------------------------------------------------------------------------
# Sliding-context helper
# ---------------------------------------------------------------------------

def _build_sliding_context_texts(
    words: list[str],
    tokenizer,
    max_length: int,
) -> list[str]:
    """Build per-word context strings using a sliding window with 50% overlap.

    Starting from the beginning of the word list, each word's context string is
    "all preceding words + current word".  When the tokenized length of that
    string would exceed ``max_length``, the context window start is advanced by
    half the current window size (50% overlap).  This ensures the new word sees
    the right half of the previous window as its left context — the standard
    approach for encoding long texts with causal LMs (Schrimpf et al. 2021,
    Goldstein et al. 2022).

    Parameters
    ----------
    words : list of str
        Surface tokens in time order.
    tokenizer :
        HuggingFace tokenizer (already loaded).
    max_length : int
        Maximum number of tokens (including special tokens) per context string.

    Returns
    -------
    list of str, length == len(words)
        Context string for each word.
    """
    texts: list[str] = []
    context_start = 0

    for i in range(len(words)):
        context = " ".join(words[context_start:i + 1])
        n_tokens = len(tokenizer.encode(context))

        if n_tokens > max_length:
            # Advance context_start by half the current window size.
            half = max(1, (i - context_start) // 2)
            context_start += half
            context = " ".join(words[context_start:i + 1])

            # Safety: if the current word alone already exceeds max_length,
            # anchor context_start to the current word.
            if len(tokenizer.encode(context)) > max_length:
                context_start = i
                context = words[i]

        texts.append(context)

    logger.debug(
        "_build_sliding_context_texts: %d words, max_length=%d, "
        "final context_start=%d (last window covers words [%d, %d])",
        len(words), max_length, context_start, context_start, len(words) - 1,
    )
    return texts


# ---------------------------------------------------------------------------
# Hook-based fallback for custom (trust_remote_code) architectures that ignore
# output_hidden_states=True and return out.hidden_states = None — observed for
# the FLA-style ALiBi/stickbreaking/forgetting checkpoints and MoE ("babylm")
# checkpoints used in the recency-bias / MoE sweeps. Shared by
# neuroai.benchmarks.fedorenko2016 and scripts/run/embedding/neural/tuckute.py.
# ---------------------------------------------------------------------------

def config_dim_attr(config: tp.Any, names: tuple[str, ...]) -> int:
    """Read the first present attribute among ``names`` from a HF config.

    Custom architectures (e.g. MoE "babylm" checkpoints' ``MoEPLMConfig``)
    don't always use the standard ``num_hidden_layers`` / ``hidden_size``
    field names — try the common aliases instead of assuming one.
    """
    for name in names:
        val = getattr(config, name, None)
        if val is not None:
            return int(val)
    raise AttributeError(
        f"{type(config).__name__} has none of the expected attributes: {names}"
    )


def _model_transformer_layers(model: tp.Any) -> list:
    """Return the model's transformer block list via a few common attribute paths."""
    for path in (
        "model.layers", "transformer.h", "model.decoder.layers",
        "decoder.layers", "layers", "model.model.layers",
    ):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    return []


# Per-model cache of (hook_handles, captures_list) once a model has been found
# to ignore output_hidden_states=True. Registered ONCE per model instance and
# reused for every subsequent call — mirrors the "if not hooks: register"
# pattern in neuroai.benchmarks.fedorenko2016. Without this, a model called
# thousands of times (e.g. fulltext chunking) would re-probe with a wasted
# forward pass and re-register/remove hooks on every single call, which is
# both slow and spams one warning log line per call. Keyed by the model object
# itself (weakly) so an evicted/GC'd model's entry disappears automatically —
# no explicit cache invalidation needed.
_hook_fallback_cache: "weakref.WeakKeyDictionary[tp.Any, tuple[list, list]]" = weakref.WeakKeyDictionary()


def _forward_hidden_states(
    model: tp.Any,
    input_ids: tp.Any,
    attention_mask: tp.Any,
    *,
    model_name: str = "",
) -> tuple:
    """Run one forward pass and return per-layer hidden states.

    Falls back to forward hooks on the model's transformer blocks when
    ``output_hidden_states=True`` is ignored (``out.hidden_states is None``) —
    the layer list is re-resolved architecturally (not batch-dependent), so the
    fallback yields the same layer count on every call, unlike a naive
    embeddings+layers hook that can pick up a variable number of firings for
    MoE-style conditional modules. The hook registration itself happens only
    once per model (see :data:`_hook_fallback_cache`).
    """
    import torch

    cached = _hook_fallback_cache.get(model)
    if cached is None:
        with torch.no_grad():
            out = model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False, return_dict=True,
            )
        # Some trust_remote_code architectures (e.g. GPT-BERT) ignore/ default
        # differently on return_dict and hand back a plain tuple instead of a
        # ModelOutput — same rationale as logit_lens.py's identical explicit
        # return_dict=True. Fall through to the hook-based path below rather
        # than crashing on `.hidden_states` access.
        if not isinstance(out, tuple) and out.hidden_states is not None:
            return out.hidden_states

        logger.warning(
            "HF LM: %s returned hidden_states=None — falling back to forward "
            "hooks (registered once, reused for subsequent calls)",
            model_name or type(model).__name__,
        )
        layers = _model_transformer_layers(model)
        if not layers:
            raise RuntimeError(
                f"{model_name or type(model).__name__} returned hidden_states=None "
                "and no transformer layers were found for hook-based extraction."
            )

        captures: list = []

        def _hook(_m: tp.Any, _i: tp.Any, output: tp.Any) -> None:
            h = output[0] if isinstance(output, tuple) else output
            captures.append(h.detach())

        hook_handles = [layer.register_forward_hook(_hook) for layer in layers]
        cached = (hook_handles, captures)
        _hook_fallback_cache[model] = cached

    _hook_handles, captures = cached
    captures.clear()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if not captures:
        raise RuntimeError(
            f"{model_name or type(model).__name__} returned hidden_states=None "
            "and hook fallback captured nothing."
        )
    return tuple(captures)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def contextual_word_embeddings_hf(
    words: list[str],
    model_name: str,
    *,
    layer_index: LayerIndex = "last",
    revision: str | None = None,
    device: str = "cpu",
    context_words: int | None = 48,
    max_length: int | None = 512,
    batch_size: int = 4,
    cache_dir: str | Path | None = None,
    section_id: int | None = None,
    lang: str = "en",
    tokenizer_no_revision: bool = False,
    n_params: float | None = None,
    trust_remote_code: bool = False,
    untrained: bool = False,
    untrained_seed: int = 0,
    dtype: "torch.dtype | str | None" = None,
) -> np.ndarray:
    """Return contextual word embeddings for a list of surface words.

    Parameters
    ----------
    words
        Surface tokens in time order (section-local).
    model_name
        HuggingFace model id (e.g. ``"EleutherAI/pythia-1b"``).
    layer_index
        ``"last"``  — last hidden layer → ``(n_words, hidden)``
        ``"all"``   — all layers (embedding + each transformer layer)
                      → ``(n_words, n_layers, hidden)``
        ``int N``   — single layer N (0 = token embedding)
                      → ``(n_words, hidden)``
    revision
        Optional HF weight revision (Pythia ``step*``, OLMo2 branch, etc.).
    device
        ``"cuda"`` or ``"cpu"``.
    context_words
        Maximum number of **previous** words prepended (plus current word).
    cache_dir
        Directory to persist/load ``.npz`` feature files.  Requires
        ``section_id`` to be set for a unique cache key.
    section_id
        Section or story index (1-based).  Only used when ``cache_dir`` is set.
    lang
        Language code (``"en"``, ``"fr"``, ``"cn"``).  Only used for cache key.
    untrained
        If ``True``, skip pretrained weight loading and use a randomly
        initialised model of the same architecture (untrained baseline).

    Returns
    -------
    np.ndarray of float32
        Shape ``(n_words, hidden)`` for scalar ``layer_index``,
        ``(n_words, n_layers, hidden)`` for ``layer_index="all"``.
    """
    if not words:
        raise ValueError("words must be non-empty")

    # --- cache look-up ---
    cache_path: Path | None = None
    if cache_dir is not None and section_id is not None:
        rev_key = revision if revision not in (None, "", "main") else None
        cache_path = _feature_cache_path(
            cache_dir, model_name, rev_key, lang, section_id,
            context_words, max_length, layer_index,
            untrained=untrained, untrained_seed=untrained_seed,
        )
        cached = _load_cached_features(cache_path)
        if cached is not None:
            logger.info(
                "HF word features: loaded from cache %s (shape %s)",
                cache_path.resolve(), cached.shape,
            )
            return cached

    import torch

    rev = revision if revision not in (None, "", "main") else None
    load_kw: dict[str, object] = {"output_hidden_states": True}
    if trust_remote_code:
        load_kw["trust_remote_code"] = True
    if dtype is not None:
        # Same gpt-bert DWA-accumulator bf16 workaround as
        # logit_lens_surprisal_all_layers — an in-place buffer .set_() inside
        # gpt-bert's custom DWA accumulator hard-codes float32, so loading
        # the model's own default bf16 weights crashes the first time the
        # model actually runs a forward pass (including here, since this
        # function also calls the model's forward() for hidden states).
        load_kw["torch_dtype"] = dtype
    elif trust_remote_code and torch.cuda.is_available():
        # Same V100 bf16-PTX-codegen workaround as
        # logit_lens_surprisal_all_layers — see that function's setup comment.
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            load_kw["torch_dtype"] = torch.float32
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=revision,
        device=device,
        load_kw=load_kw,
        tokenizer_no_revision=tokenizer_no_revision,
        n_params=n_params,
        untrained=untrained,
        untrained_seed=untrained_seed,
    )
    input_device = _model_input_device(model)

    # Effective token limit used both for sliding-window context building and
    # for the tokenizer's hard-truncation safety net.
    if max_length is not None:
        eff_max_length: int = max_length
    else:
        # Derive from model config (most reliable) or tokenizer attribute,
        # capped at 32 768 to avoid OOM on models with huge nominal contexts.
        cfg_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        tok_max = getattr(tokenizer, "model_max_length", None)
        candidates = [v for v in (cfg_max, tok_max) if isinstance(v, int) and v < 1_000_000]
        eff_max_length = min(candidates) if candidates else 2048

    # Build context strings
    if context_words is None:
        # Sliding window: grow context until eff_max_length is reached, then
        # shift forward by half the current window (50 % overlap).
        texts = _build_sliding_context_texts(words, tokenizer, eff_max_length)
    else:
        texts = [
            " ".join(words[max(0, i - context_words): i + 1])
            for i in range(len(words))
        ]

    # Determine return_all_layers flag
    return_all = layer_index == "all"
    target_layer: int | None = None
    if isinstance(layer_index, int):
        target_layer = layer_index

    out_vecs: list[np.ndarray] = []
    dim_warned = [False]

    for i0, i1 in _chunks(len(texts), batch_size):
        batch = texts[i0:i1]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=eff_max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(input_device) for k, v in enc.items()}

        hs = _forward_hidden_states(
            model, enc["input_ids"], enc["attention_mask"], model_name=model_name,
        )

        # hidden_states: tuple of (batch, seq, hidden) — one per layer
        # including the token embedding (index 0), unless the hook fallback
        # engaged (no separate embedding-layer entry in that case).
        n_model_layers = len(hs)

        for b in range(enc["input_ids"].shape[0]):
            attn = enc["attention_mask"][b]
            last_pos = int(attn.sum().item()) - 1

            if return_all:
                layer_tensors = [
                    hs[li][b, last_pos] for li in range(n_model_layers)
                ]
                layer_tensors = _select_uniform_hidden_layers(
                    layer_tensors,
                    model_name=model_name,
                    warned=dim_warned,
                )
                vec = torch.stack(layer_tensors, dim=0).float().cpu().numpy()
            elif target_layer is not None:
                if target_layer >= n_model_layers:
                    raise ValueError(
                        f"layer_index={target_layer} but model only has "
                        f"{n_model_layers} layers (0-indexed)."
                    )
                vec = hs[target_layer][b, last_pos].float().cpu().numpy()
            else:
                # "last"
                vec = hs[-1][b, last_pos].float().cpu().numpy()

            out_vecs.append(vec)

        del hs

    emb = np.stack(out_vecs, axis=0).astype(np.float32, copy=False)

    shape_str = "×".join(str(d) for d in emb.shape)
    logger.info(
        "HF word features: model=%s%s layer=%s words=%d shape=%s device=%s",
        model_name,
        f" @{rev}" if rev else "",
        layer_index,
        len(words),
        shape_str,
        device,
    )

    # --- cache write ---
    if cache_path is not None and _try_save_cached_features(cache_path, emb):
        resolved = cache_path.resolve()
        nbytes = resolved.stat().st_size if resolved.is_file() else 0
        logger.info(
            "HF word features: cached → %s (%d MB, shape %s)",
            resolved,
            nbytes // (1024 * 1024),
            shape_str,
        )

    return emb


def _fulltext_chunks_on_gpu(
    gpu_id: int,
    model_name: str,
    revision: str | None,
    n_params: float | None,
    input_ids_np: "np.ndarray",
    attn_mask_np: "np.ndarray",
    chunk_indices: list[int],
    chunk_to_words_sub: dict[int, list[int]],
    word_tok_indices: dict[int, list[int]],
    trust_remote_code: bool = False,
    untrained: bool = False,
    untrained_seed: int = 0,
    dtype: "torch.dtype | str | None" = None,
) -> dict[int, "np.ndarray"]:
    """Spawned worker: forward-pass a subset of fulltext chunks on one GPU.

    Parameters
    ----------
    chunk_to_words_sub : {chunk_k: [word_idx, ...]} only for this GPU's chunks.
    word_tok_indices   : {word_idx: [token_positions]} for every word in these chunks.

    Returns {word_idx: np.ndarray shape (n_kept_layers, D)}
    """
    import gc as _gc
    import numpy as np
    import torch
    from collections import Counter

    torch.cuda.set_device(gpu_id)
    device_str = f"cuda:{gpu_id}"

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype is None and "gpt-bert" in model_name.lower():
        # See _get_cached_causal_lm's identical auto-detection for why this
        # can't just rely on the caller passing dtype explicitly.
        dtype = torch.float32

    model_rev = revision if revision not in (None, "", "main") else None
    model_rev_fp = {"revision": model_rev} if model_rev else {}

    # Use bfloat16 + pin to this GPU (same logic as _apply_device_map_if_needed
    # for the fits-on-one-GPU case; multi-GPU extraction is only triggered when
    # the model fits on a single H100) -- unless the caller overrides dtype
    # (e.g. gpt-bert's DWA accumulator hardcodes a float32 torch.Tensor() in
    # its custom InPlaceSetSlice.apply, so it needs torch_dtype=float32 or the
    # bf16 activations can't be set_() into it -- see
    # logit_lens_surprisal_all_layers's setup comment for the full rationale).
    load_kw: dict = {"torch_dtype": dtype or torch.bfloat16, "device_map": {"": gpu_id}}
    load_kw.update(model_rev_fp)
    if trust_remote_code:
        load_kw["trust_remote_code"] = True

    # Use the same VLM-aware loading logic as _get_cached_causal_lm:
    # 1. Resolve architecture-specific class (handles mistral3 → Ministral3ForCausalLM)
    # 2. Detect VLM checkpoints (>20 missing keys with "layers") and prefix-strip
    _trc_mp = trust_remote_code
    if untrained:
        # Random-init baseline (see _get_cached_causal_lm): architecture/config
        # only, pretrained weights never loaded.
        from transformers import AutoConfig, set_seed

        set_seed(untrained_seed)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=_trc_mp, **model_rev_fp)
        model_cls = _resolve_causal_lm_class(model_name, model_rev_fp, trust_remote_code=_trc_mp)
        _dtype = load_kw.get("torch_dtype", torch.bfloat16)
        model = model_cls.from_config(config, trust_remote_code=_trc_mp, torch_dtype=_dtype)
        model = model.to(device_str)
        _missing = []
    elif model_name.startswith(_FORGETTING_DIRECT_PREFIX):
        # Bypass the buggy trust_remote_code snapshot -- see
        # _load_forgetting_direct's docstring (same fix as _get_cached_causal_lm).
        model = _load_forgetting_direct(model_name, model_rev_fp)
        _dtype = load_kw.get("torch_dtype", torch.bfloat16)
        model = model.to(dtype=_dtype).to(device_str)
        _missing: list[str] = []
    elif model_name.startswith(_MOLM_DIRECT_PREFIX):
        # Bypass the unregistered "moduleformer" AutoConfig model_type -- see
        # _load_moduleformer_direct's docstring (same fix as _get_cached_causal_lm).
        model = _load_moduleformer_direct(model_name, model_rev_fp)
        _dtype = load_kw.get("torch_dtype", torch.bfloat16)
        model = model.to(dtype=_dtype).to(device_str)
        _missing: list[str] = []
    elif model_name.startswith(_FOX_DIRECT_PREFIX):
        # Bypass trust_remote_code entirely -- see _load_fox_direct's
        # docstring (same fix as _get_cached_causal_lm).
        model = _load_fox_direct(model_name, model_rev_fp)
        _dtype = load_kw.get("torch_dtype", torch.bfloat16)
        model = model.to(dtype=_dtype).to(device_str)
        _missing: list[str] = []
    else:
        model_cls = _resolve_causal_lm_class(model_name, model_rev_fp, trust_remote_code=_trc_mp)
        model, _load_info = model_cls.from_pretrained(
            model_name, output_loading_info=True, **load_kw
        )
        _missing = _load_info.get("missing_keys", [])
    if len(_missing) > 20 and any("layers" in k for k in _missing):
        del model
        _gc.collect()
        torch.cuda.empty_cache()
        causal_model = _load_causal_lm_from_vlm_checkpoint(model_name, load_kw)
        if causal_model is not None:
            _dtype = load_kw.get("torch_dtype", torch.bfloat16)
            model = causal_model.to(dtype=_dtype).to(device_str)
        else:
            raise ValueError(
                f"Cannot load VLM checkpoint {model_name} as causal LM "
                "(HF cache unavailable for prefix-strip)"
            )
    elif _trc_mp and "lm_head.weight" in _missing and hasattr(model, "lm_head"):
        # Same repair as _get_cached_causal_lm: some trust_remote_code checkpoints
        # (e.g. Lanni-ni/hard_{2,3,5}gram_*) have a broken tie_word_embeddings
        # handling in their remote code, leaving a "missing" lm_head.weight as a
        # freshly-initialised (often all-zero/NaN) tensor instead of either
        # loading a real saved one or tying it to the input embedding.
        _real_lm_head = _load_plain_lm_head_from_hf_cache(
            model_name,
            revision=load_kw.get("revision"),
            hf_hub_cache=os.environ.get("HF_HUB_CACHE"),
        )
        import torch.nn as _nn

        if _real_lm_head is not None:
            _orig_device = model.lm_head.weight.device
            with torch.no_grad():
                model.lm_head.weight = _nn.Parameter(
                    _real_lm_head.to(device=_orig_device, dtype=model.lm_head.weight.dtype)
                )
            logger.warning(
                "HF LM: %s — lm_head.weight was incorrectly tied by a buggy "
                "trust_remote_code config; loaded the real untied weight from "
                "the checkpoint instead", model_name,
            )
        else:
            # See the analogous branch in _get_cached_causal_lm: don't gate
            # this on "does it look degenerate" -- a non-device_map load can
            # leave an innocuous-looking but equally wrong random lm_head.
            embed = model.get_input_embeddings()
            current = model.lm_head.weight.detach()
            if embed.weight.device != current.device or embed.weight.dtype != current.dtype:
                with torch.no_grad():
                    embed.weight = _nn.Parameter(
                        embed.weight.to(device=current.device, dtype=current.dtype)
                    )
            # True tying: assign the *same* Parameter object so downstream
            # param-counting code (sum(p.numel() for p in model.parameters()))
            # correctly deduplicates it instead of silently double-counting.
            model.lm_head.weight = embed.weight
            logger.warning(
                "HF LM: %s — lm_head.weight had no real saved tensor (buggy "
                "trust_remote_code tie_word_embeddings handling); tied it to "
                "the input embedding instead", model_name,
            )
    model.eval()

    keep_layers: list[int] | None = None
    result: dict[int, np.ndarray] = {}

    for k in chunk_indices:
        ids_k = torch.tensor(input_ids_np[k : k + 1], dtype=torch.long, device=device_str)
        am_k  = torch.tensor(attn_mask_np[k : k + 1], dtype=torch.long, device=device_str)
        hs = _forward_hidden_states(model, ids_k, am_k, model_name=model_name)  # tuple of (1, seq_len, D)

        if keep_layers is None:
            dims = [hs[li].shape[-1] for li in range(len(hs))]
            modal_dim = Counter(dims).most_common(1)[0][0]
            keep_layers = [li for li, d in enumerate(dims) if d == modal_dim]

        for wi in chunk_to_words_sub.get(k, []):
            tok_idx = word_tok_indices[wi]
            layer_vecs = [
                hs[li][0, tok_idx].detach().float().cpu().numpy().mean(axis=0)
                for li in keep_layers
            ]
            result[wi] = np.stack(layer_vecs, axis=0)  # (n_kept_layers, D)

        del hs
        torch.cuda.empty_cache()

    return result


def contextual_word_embeddings_hf_fulltext(
    words: list[str],
    model_name: str,
    *,
    revision: str | None = None,
    device: str = "cpu",
    layer_index: LayerIndex = "last",
    max_length: int | None = None,
    stride_delta: int = 64,
    tokenizer_no_revision: bool = True,
    cache_dir: str | Path | None = None,
    section_id: int = 0,
    lang: str = "en",
    n_params: int | None = None,
    trust_remote_code: bool = False,
    untrained: bool = False,
    untrained_seed: int = 0,
    dtype: "torch.dtype | str | None" = None,
) -> np.ndarray:
    """Full-text sliding-window LLM feature extraction.

    Feeds the entire transcript in overlapping fixed-length chunks (one forward
    pass per chunk, not one per word).  For a causal LM the triangular attention
    mask means hidden state at position *i* already encodes only tokens 0..i, so
    this is equivalent to the per-word moving-window approach but orders of
    magnitude faster: ~n_chunks passes (typically single-digit to low double-digit)
    instead of ~n_words passes with growing sequence lengths.

    Each word is assigned to the earliest chunk in which it falls inside the
    non-overlapping (new) region, which places it near the right edge of that
    chunk and therefore maximises available left context.

    Requires a **fast** tokenizer (``return_offsets_mapping``).

    Returns
    -------
    np.ndarray  shape (n_words, D) or (n_words, n_layers, D)
    """
    # CBR-RNN has a custom architecture not loadable via AutoModelForCausalLM.
    # Route to the dedicated path, which processes the flat word list directly.
    if "cbr-rnn" in model_name.lower() or "cbr_rnn" in model_name.lower():
        if untrained:
            raise NotImplementedError(
                "untrained baseline not supported for CBR-RNN (custom architecture, "
                "not loadable via AutoModelForCausalLM.from_config)."
            )
        from .cbr_rnn import contextual_word_embeddings_cbr_rnn as _cbr_emb
        cache_path_cbr: Path | None = None
        if cache_dir is not None:
            cache_path_cbr = _fulltext_feature_cache_path(
                cache_dir, model_name, revision, lang, section_id,
                256, stride_delta, layer_index,
            )
            cached = _load_cached_features(cache_path_cbr)
            if cached is not None:
                logger.info("CBR-RNN fulltext features: loaded from cache (shape %s)", cached.shape)
                return cached
        arr = _cbr_emb(words, model_name=model_name, device=device)  # (n_words, D)
        arr = arr[:, np.newaxis, :].astype(np.float32)                # (n_words, 1, D)
        if cache_path_cbr is not None:
            try:
                np.savez_compressed(str(cache_path_cbr), features=arr)
            except Exception as exc:
                logger.warning("CBR-RNN feature cache write failed: %s", exc)
        return arr

    from transformers import AutoConfig

    rev_norm = revision if revision not in (None, "", "main") else None
    cf_kw: dict[str, tp.Any] = {"revision": rev_norm} if rev_norm is not None else {}
    if trust_remote_code:
        cf_kw["trust_remote_code"] = True
    try:
        lm_cfg = AutoConfig.from_pretrained(model_name, **cf_kw)
    except Exception:
        lm_cfg = None

    if max_length is not None:
        eff_max_length: int = max_length
    else:
        cfg_max = getattr(lm_cfg, "max_position_embeddings", None) if lm_cfg is not None else None
        candidates = [v for v in (cfg_max,) if isinstance(v, int) and 0 < v < 1_000_000]
        eff_max_length = min(candidates) if candidates else 2048
        eff_max_length = min(eff_max_length, 32768)

    stride = max(1, eff_max_length - stride_delta)
    step = eff_max_length - stride  # tokens advanced per chunk

    # --- cache check (before loading model/tokenizer) ---
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _fulltext_feature_cache_path(
            cache_dir, model_name, revision, lang, section_id,
            eff_max_length, stride_delta, layer_index,
            untrained=untrained, untrained_seed=untrained_seed,
        )
        cached = _load_cached_features(cache_path)
        if cached is not None:
            logger.info(
                "HF fulltext features: loaded from cache %s (shape %s)",
                cache_path.resolve(), cached.shape,
            )
            return cached

    import torch

    # --- decide single-GPU vs multi-GPU before loading model ---
    n_gpus_avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Multi-GPU: load one model copy per GPU (in parallel subprocesses).
    # Only viable when each copy fits on one H100 (80 GB with 10 GB headroom = 70 GB,
    # i.e. models up to ~35B params in bfloat16 — practically everything except
    # 70B+ models).  We fall back to single-GPU if n_params is unknown.
    gpu_mem_gib = (
        (torch.cuda.get_device_properties(0).total_memory - 10 * 1024 ** 3) / (1024 ** 3)
        if n_gpus_avail > 0 else 0.0
    )
    est_model_gib = (n_params * 2 / (1024 ** 3)) if n_params else 0.0
    use_multigpu = (
        n_gpus_avail > 1
        and device.startswith("cuda")
        and est_model_gib > 0
        and est_model_gib <= gpu_mem_gib
    )

    if use_multigpu:
        # Load only the tokenizer in the main process; model copies are loaded
        # inside the spawned workers (one per GPU) so no GPU memory is wasted here.
        model_rev_tok = revision if revision not in (None, "", "main") else None
        tok_rev_fp = {} if tokenizer_no_revision else (
            {"revision": model_rev_tok} if model_rev_tok else {}
        )
        tokenizer = _load_hf_tokenizer(model_name, tok_rev_fp, trust_remote_code=trust_remote_code)
    else:
        _ft_load_kw: dict[str, object] = {"output_hidden_states": True}
        if trust_remote_code:
            _ft_load_kw["trust_remote_code"] = True
        if dtype is not None:
            # Same gpt-bert DWA-accumulator bf16 workaround as
            # logit_lens_surprisal_all_layers — see that function's setup
            # comment for the full rationale.
            _ft_load_kw["torch_dtype"] = dtype
        elif trust_remote_code and torch.cuda.is_available():
            # Same V100 bf16-PTX-codegen workaround as
            # logit_lens_surprisal_all_layers.
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                _ft_load_kw["torch_dtype"] = torch.float32
        tokenizer, model = _get_cached_causal_lm(
            model_name,
            revision=revision,
            device=device,
            load_kw=_ft_load_kw,
            tokenizer_no_revision=tokenizer_no_revision,
            n_params=n_params,
            untrained=untrained,
            untrained_seed=untrained_seed,
        )
        model = model  # keep reference for single-GPU path below
        input_device = _model_input_device(model)

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "contextual_word_embeddings_hf_fulltext requires a fast tokenizer "
            f"(return_offsets_mapping); got {type(tokenizer).__name__}. "
            "Fall back to contextual_word_embeddings_hf for slow tokenizers."
        )

    # Models like Stickbreaking / Alibi often have no pad_token; set it to eos.
    _ensure_pad_token_for_batches(tokenizer)

    # Build full text and exact char spans for each word
    full_text = " ".join(words)
    word_char_spans: list[tuple[int, int]] = []
    pos = 0
    for w in words:
        word_char_spans.append((pos, pos + len(w)))
        pos += len(w) + 1  # +1 for space separator

    # Tokenize the full text into overlapping fixed-length chunks
    overflow_enc = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=eff_max_length,
        return_overflowing_tokens=True,
        stride=stride,
        padding=True,
    )

    om_t = overflow_enc["offset_mapping"]  # (n_chunks, seq_len, 2)  — tensor
    n_chunks = int(overflow_enc["input_ids"].shape[0])
    seq_len = int(overflow_enc["input_ids"].shape[1])
    om_np = om_t.numpy()  # (n_chunks, seq_len, 2)  — numpy for fast boolean ops

    logger.info(
        "HF fulltext [%s%s]: %d words → %d chunks × %d tokens "
        "(stride_delta=%d, gpus=%d%s)",
        model_name, f"@{revision}" if revision else "",
        len(words), n_chunks, eff_max_length, stride_delta,
        n_gpus_avail if use_multigpu else 1,
        " parallel" if use_multigpu else "",
    )

    # --- assign each word to the first chunk where it enters the non-overlap region ---
    new_region_start = [0] + [stride] * (n_chunks - 1)

    word_assignments: list[tuple[int, list[int]]] = []
    for wi, (w_start, w_end) in enumerate(word_char_spans):
        assigned = False
        for k in range(n_chunks):
            ns = new_region_start[k]
            t_starts = om_np[k, ns:, 0]
            t_ends = om_np[k, ns:, 1]
            mask = (t_starts < w_end) & (t_ends > w_start)
            idx = np.where(mask)[0]
            if idx.size > 0:
                word_assignments.append((k, (idx + ns).tolist()))
                assigned = True
                break
        if not assigned:
            logger.warning(
                "fulltext: no token found for word %d '%s' — using last chunk end",
                wi, words[wi],
            )
            word_assignments.append((n_chunks - 1, [seq_len - 1]))

    from collections import defaultdict
    chunk_to_words: dict[int, list[int]] = defaultdict(list)
    for wi, (k, _) in enumerate(word_assignments):
        chunk_to_words[k].append(wi)

    out_vecs: list[np.ndarray | None] = [None] * len(words)
    keep_layers: list[int] | None = None
    n_layers: int | None = None

    if use_multigpu:
        # --- multi-GPU parallel forward passes ---
        # Split chunk indices across n_gpus_avail GPUs (round-robin so chunks
        # that have words assigned are distributed evenly).
        import multiprocessing as _mpp
        from concurrent.futures import ProcessPoolExecutor as _PPE

        active_chunks = sorted(chunk_to_words.keys())
        chunk_groups = [active_chunks[i::n_gpus_avail] for i in range(n_gpus_avail)]

        # Per-worker data: lightweight dicts (no model weights)
        input_ids_np  = overflow_enc["input_ids"].numpy()
        attn_mask_np  = overflow_enc["attention_mask"].numpy()
        word_tok_idx  = {wi: word_assignments[wi][1] for wi in range(len(words))}

        ctx = _mpp.get_context("spawn")
        with _PPE(max_workers=n_gpus_avail, mp_context=ctx) as pool:
            futs = [
                pool.submit(
                    _fulltext_chunks_on_gpu,
                    gpu_id,
                    model_name, revision, n_params,
                    input_ids_np, attn_mask_np,
                    chunk_groups[gpu_id],
                    {k: chunk_to_words[k] for k in chunk_groups[gpu_id]},
                    word_tok_idx,
                    trust_remote_code,
                    untrained,
                    untrained_seed,
                    dtype,
                )
                for gpu_id in range(n_gpus_avail)
            ]
            for fut in futs:
                for wi, vec in fut.result().items():
                    out_vecs[wi] = vec

        # Infer keep_layers from first word vector
        first_vec = next(v for v in out_vecs if v is not None)
        n_layers = first_vec.shape[0]
        keep_layers = list(range(n_layers))  # already filtered inside worker

    else:
        # --- single-GPU sequential forward passes (original code) ---
        from collections import Counter as _Counter

        for k in sorted(chunk_to_words.keys()):
            ids_k = overflow_enc["input_ids"][k : k + 1].to(input_device)
            am_k = overflow_enc["attention_mask"][k : k + 1].to(input_device)

            # _forward_hidden_states falls back to forward hooks on its own when
            # the model ignores output_hidden_states=True (custom architectures
            # like stickbreaking / forgetting / alibi / MoE "babylm" checkpoints).
            hs = _forward_hidden_states(model, ids_k, am_k, model_name=model_name)

            if keep_layers is None:
                dims = [hs[li].shape[-1] for li in range(len(hs))]
                modal_dim = _Counter(dims).most_common(1)[0][0]
                keep_layers = [li for li, d in enumerate(dims) if d == modal_dim]
                if len(keep_layers) < len(hs):
                    logger.warning(
                        "fulltext: non-uniform layer dims in %s — keeping %d/%d layers "
                        "(modal hidden_dim=%d)",
                        model_name, len(keep_layers), len(hs), modal_dim,
                    )
                n_layers = len(keep_layers)

            for wi in chunk_to_words[k]:
                _, tok_indices = word_assignments[wi]
                layer_vecs: list[np.ndarray] = []
                for li in keep_layers:  # type: ignore[union-attr]
                    emb_tokens = hs[li][0, tok_indices].detach().float().cpu().numpy()
                    layer_vecs.append(emb_tokens.mean(axis=0))
                out_vecs[wi] = np.stack(layer_vecs, axis=0)  # (n_kept_layers, D)

            del hs

    assert keep_layers is not None and n_layers is not None, "no chunks processed"
    assert all(v is not None for v in out_vecs), "some words not assigned"

    all_arr = np.stack(out_vecs, axis=0).astype(np.float32)  # (n_words, n_layers, D)

    # Layer selection
    if isinstance(layer_index, int):
        result = all_arr[:, layer_index, :]
    elif layer_index == "last":
        result = all_arr[:, -1, :]
    elif layer_index == "all":
        result = all_arr
    else:
        raise ValueError(f"Unknown layer_index: {layer_index!r}")

    shape_str = "×".join(str(d) for d in result.shape)
    rev = revision or ""
    logger.info(
        "HF fulltext [%s%s] layer=%s → %d words, shape %s (%d chunks, device=%s)",
        model_name, f"@{rev}" if rev else "", layer_index,
        len(words), shape_str, n_chunks, device,
    )

    if cache_path is not None and _try_save_cached_features(cache_path, result):
        resolved = cache_path.resolve()
        nbytes = resolved.stat().st_size if resolved.is_file() else 0
        logger.info(
            "HF fulltext features: cached → %s (%d MB, shape %s)",
            resolved, nbytes // (1024 * 1024), shape_str,
        )

    return result


def contextual_word_embeddings_hf_lbg(
    words_df: pd.DataFrame,
    model_name: str,
    *,
    bids_root: str | Path,
    lang: str,
    section_id: int,
    revision: str | None = None,
    device: str = "cpu",
    layer_index: LayerIndex = "last",
    lbg_story_text_zip: str | Path | None = None,
    hf_lbg_max_length: int | None = None,
    hf_lbg_stride_delta: int = 64,
    tokenizer_no_revision: bool = True,
    cache_dir: str | Path | None = None,
    trust_remote_code: bool = False,
    dtype: "torch.dtype | str | None" = None,
) -> np.ndarray:
    """LLM embeddings matching **l-bg/llms_brain_lateralization** ``extract_llm_activations.py``.

    Full transcription per section (official story zip **or** TextGrid fallback),
    sliding-window tokenizer, offset-based word anchoring, and **mean** pooling over
    all sub-word tokens spanning each stimulus word.

    Outputs the same ndarray layout as :func:`contextual_word_embeddings_hf`
    after layer selection — use with a filtered word table whose rows omit ``' '``
    spacer entries (handled in :class:`neuroai.lpp_encoding.LppFmriEncoder`).

    Requires a **Fast** tokenizer (``return_offsets_mapping``).
    """
    from transformers import AutoConfig

    lg = lang.lower()

    rev = revision if revision not in (None, "", "main") else None

    cf_kw = {"revision": rev} if rev is not None else {}
    if trust_remote_code:
        cf_kw["trust_remote_code"] = True
    try:
        lm_cfg = AutoConfig.from_pretrained(model_name, **cf_kw)
    except Exception:
        lm_cfg = None
    maxlen_pre = getattr(lm_cfg, "max_position_embeddings", None) if lm_cfg is not None else None
    maxlen0 = int(hf_lbg_max_length or maxlen_pre or 2048)
    stride0 = maxlen0 - hf_lbg_stride_delta
    cache_path: Path | None = None
    if cache_dir is not None:
        if stride0 <= 0:
            raise ValueError(
                f"Invalid l-bg stride: maxlen={maxlen0}, delta={hf_lbg_stride_delta}"
            )
        cache_path = _lbg_feature_cache_path(
            cache_dir,
            model_name,
            revision,
            lg,
            section_id,
            maxlen0,
            stride0,
            layer_index,
        )
        cached_arr = _load_cached_features(cache_path)
        if cached_arr is not None:
            logger.info(
                "HF word features (l-bg): loaded from cache %s (shape %s)",
                cache_path.resolve(),
                cached_arr.shape,
            )
            return cached_arr

    import torch

    _lbg_load_kw: dict[str, object] = {"output_hidden_states": True}
    if trust_remote_code:
        _lbg_load_kw["trust_remote_code"] = True
    if dtype is not None:
        # Same gpt-bert DWA-accumulator bf16 workaround as
        # logit_lens_surprisal_all_layers — see that function's setup comment.
        _lbg_load_kw["torch_dtype"] = dtype
    elif trust_remote_code and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            _lbg_load_kw["torch_dtype"] = torch.float32
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=revision,
        device=device,
        load_kw=_lbg_load_kw,
        tokenizer_no_revision=tokenizer_no_revision,
    )
    input_device = _model_input_device(model)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "l-bg-style extraction requires a fast tokenizer supporting "
            "return_offsets_mapping; got "
            f"{type(tokenizer).__name__}"
        )

    # Keep window hyper-params identical to cache fingerprint (``maxlen0`` / ``stride0``).
    maxlen, stride = maxlen0, stride0

    fulltext_run = load_story_text(
        bids_root=bids_root,
        lang=lg,
        section=section_id,
        lbg_story_text_zip=Path(lbg_story_text_zip)
        if lbg_story_text_zip is not None
        else None,
    ).replace("\n", " ").replace("\r", " ")

    word_list, char_to_word_idx, n_words_after_cn = build_lbg_word_stream(words_df, lg)
    if not word_list:
        raise ValueError("Empty word stream for LPP l-bg extraction")

    overflow_enc = tokenizer(
        fulltext_run,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=maxlen,
        return_overflowing_tokens=True,
        stride=stride,
    )
    om = tp.cast(torch.Tensor, overflow_enc["offset_mapping"])
    seq_len = int(overflow_enc["input_ids"].shape[1])
    idx_batch = 0
    idx_token = 0
    idx_word_to_idx_token: list[tuple[int, int]] = []

    for idx_word, raw_w in enumerate(word_list):
        w_key = simplify_word(raw_w, lg)
        if idx_token == seq_len:
            idx_batch += 1
            idx_token = stride
        i_start = int(om[idx_batch, idx_token, 0].item())
        i_stop = int(om[idx_batch, idx_token, 1].item())
        n_try = 0
        # Phantom-expansion candidates (see LBG_PHANTOM_CONTRACTION_EXPANSIONS)
        # get a much narrower search window: a genuine, independently-occurring
        # "not"/"would"/etc. sits right at (or within a token or two of) the
        # expected position, whereas do_word_match's loose substring rule can
        # find a false-positive match on a short, unrelated word (e.g. "not"
        # matching inside "No" from unrelated dialogue several tokens away) if
        # allowed to wander the full window -- which then corrupts every
        # subsequent word's alignment. A real match is never far; a phantom
        # should fail fast and get aliased rather than risk a wrong match.
        max_try = 3 if w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS else 20
        while n_try < max_try and not do_word_match(
            w_key, fulltext_run[i_start:i_stop], lg
        ):
            idx_token += 1
            if idx_token == seq_len:
                idx_batch += 1
                idx_token = stride
            i_start = int(om[idx_batch, idx_token, 0].item())
            i_stop = int(om[idx_batch, idx_token, 1].item())
            n_try += 1
        if n_try == max_try:
            if idx_word_to_idx_token and w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS:
                # Auxiliary-contraction phantom split (see
                # LBG_PHANTOM_CONTRACTION_EXPANSIONS docstring) -- this word
                # has no independent surface token. Give it the very next
                # token slot (typically a leftover punctuation/quote token
                # from the contraction) rather than literally reusing the
                # preceding word's own token index: downstream pooling
                # computes each word's embedding as the mean over the span
                # from its token to the NEXT word's token, so two words
                # sharing one index collapses that span to empty and crashes
                # np.stack. Keeping indices strictly increasing preserves
                # that invariant while still anchoring the phantom to the
                # closest real token available.
                prev_batch, prev_token = idx_word_to_idx_token[-1]
                alias_batch, alias_token = prev_batch, prev_token + 1
                if alias_token == seq_len:
                    alias_batch += 1
                    alias_token = stride
                logger.warning(
                    "l-bg tokenizer alignment: word %d ('%s', model=%s, "
                    "section=%s) has no independent surface token (UD "
                    "auxiliary-contraction split) -- aliasing onto the "
                    "next available token instead of failing.",
                    idx_word, raw_w, model_name, section_id,
                )
                idx_word_to_idx_token.append((alias_batch, alias_token))
                idx_batch, idx_token = alias_batch, alias_token + 1
                continue
            inner = ""
            try:
                if lbg_story_text_zip is not None:
                    inner = story_text_zip_inner_path(lg, section_id)
            except Exception:
                inner = "(zip)"
            raise ValueError(
                "l-bg tokenizer alignment failed "
                f"(word index {idx_word}, model={model_name}, section={section_id}, "
                f"text_backend={inner!s})"
            )
        idx_word_to_idx_token.append((idx_batch, idx_token))
        if lg != "cn":
            idx_token += 1
        elif idx_word < len(word_list) - 1 and not do_word_match(
            str(word_list[idx_word + 1]), fulltext_run[i_start:i_stop], lg
        ):
            idx_token += 1

    hidden_states_batches: list[list[np.ndarray]] = []
    nb = overflow_enc["input_ids"].shape[0]
    for k in range(int(nb)):
        ids_k = overflow_enc["input_ids"][k : k + 1].to(input_device)
        am_k = overflow_enc["attention_mask"][k : k + 1].to(input_device)
        hs = _forward_hidden_states(model, ids_k, am_k, model_name=model_name)
        hidden_states_batches.append(
            [hs[j][0].detach().float().cpu().numpy() for j in range(len(hs))]
        )

    # The hook fallback (for custom architectures ignoring
    # output_hidden_states=True) resolves the layer list once per forward
    # call; for MoE-style conditional modules this can still yield a
    # different *number* of layers, and/or a different hidden dim at the same
    # layer index, on different chunks. Keep only the layer indices whose
    # hidden dim is the per-chunk modal dim in EVERY chunk (same technique as
    # the fulltext extractor's keep_layers filter) so the per-layer stacking
    # below never sees a shape mismatch.
    from collections import Counter as _Counter

    good_layers: set[int] | None = None
    for batch in hidden_states_batches:
        dims = [arr.shape[-1] for arr in batch]
        modal_dim = _Counter(dims).most_common(1)[0][0]
        layers_here = {li for li, d in enumerate(dims) if d == modal_dim}
        good_layers = layers_here if good_layers is None else (good_layers & layers_here)
    if not good_layers:
        raise RuntimeError(
            f"l-bg: {model_name} produced no hidden-state layer with a "
            "consistent shape across all chunks — cannot extract embeddings."
        )
    keep_layers = sorted(good_layers)
    total_layers = {len(b) for b in hidden_states_batches}
    if len(keep_layers) < max(total_layers):
        logger.warning(
            "l-bg: %s produced inconsistent hidden-state layers across chunks "
            "(counts=%s) — keeping %d/%s layers with a shape consistent in every chunk",
            model_name, sorted(total_layers), len(keep_layers), max(total_layers),
        )
    hidden_states_batches = [[b[li] for li in keep_layers] for b in hidden_states_batches]

    n_layers_tpl = len(hidden_states_batches[0])
    layers_words_activations: list[list[np.ndarray]] = [[] for _ in range(n_layers_tpl)]

    for idx_word in range(len(word_list) - 1):
        idx_batch_a, idx_tok_a = idx_word_to_idx_token[idx_word]
        idx_batch_b, idx_tok_b = idx_word_to_idx_token[idx_word + 1]
        emb_layers = [[] for _ in range(n_layers_tpl)]

        def _pull(b: int, t: int, li_: int) -> np.ndarray:
            return hidden_states_batches[b][li_][t].astype(np.float32, copy=False)

        if idx_batch_a == idx_batch_b:
            for i in range(idx_tok_a, idx_tok_b):
                for li2 in range(n_layers_tpl):
                    emb_layers[li2].append(_pull(idx_batch_a, i, li2))
        else:
            for i in range(idx_tok_a, seq_len):
                for li2 in range(n_layers_tpl):
                    emb_layers[li2].append(_pull(idx_batch_a, i, li2))
            for i in range(stride, idx_tok_b):
                for li2 in range(n_layers_tpl):
                    emb_layers[li2].append(_pull(idx_batch_b, i, li2))

        for li3 in range(n_layers_tpl):
            arr = np.stack(emb_layers[li3], axis=0).astype(np.float32, copy=False)
            layers_words_activations[li3].append(arr.mean(axis=0))

    eos_id = getattr(tokenizer, "eos_token_id", None)
    # Last token span (upstream breaks at eos)
    last_i = len(word_list) - 1
    idx_batch_l, idx_tok_l = idx_word_to_idx_token[last_i]
    emb_last = [[] for _ in range(n_layers_tpl)]
    row_ids = overflow_enc["input_ids"][idx_batch_l]
    row_attn = overflow_enc["attention_mask"][idx_batch_l]
    valid_len = int(row_attn.sum().item())
    for i in range(idx_tok_l, min(valid_len, seq_len)):
        tok = int(row_ids[i].item())
        if eos_id is not None and tok == eos_id:
            break
        for li4 in range(n_layers_tpl):
            emb_last[li4].append(_pull(idx_batch_l, i, li4))
    if not emb_last[0]:
        raise ValueError(
            "l-bg extraction produced no hidden states for final word span"
        )
    for li5 in range(n_layers_tpl):
        arr = np.stack(emb_last[li5], axis=0).astype(np.float32, copy=False)
        layers_words_activations[li5].append(arr.mean(axis=0))

    if lg == "cn":
        stacked_layers = aggregate_cn_char_activations(
            layers_words_activations, tp.cast(list[int], char_to_word_idx), n_words_after_cn
        )
    else:
        stacked_layers = [
            np.stack(layers_words_activations[i], axis=0).astype(
                np.float32, copy=False
            )
            for i in range(n_layers_tpl)
        ]

    # Some models (e.g. OPT-350M) have word_embed_proj_dim != hidden_size, so
    # layer 0 (token embeddings) has a different hidden dim than transformer layers.
    # Drop any layers whose dim doesn't match the modal dim before stacking.
    if stacked_layers and len({a.shape[-1] for a in stacked_layers}) > 1:
        from collections import Counter
        modal_dim = Counter(a.shape[-1] for a in stacked_layers).most_common(1)[0][0]
        n_dropped = sum(a.shape[-1] != modal_dim for a in stacked_layers)
        logger.warning(
            "LPP HF (l-bg): %s has non-uniform hidden dims across %d layers — "
            "dropping %d layer(s) whose dim != %d (likely embedding projection mismatch)",
            model_name, len(stacked_layers), n_dropped, modal_dim,
        )
        stacked_layers = [a for a in stacked_layers if a.shape[-1] == modal_dim]

    all_arr = np.stack(stacked_layers, axis=1)
    emb: np.ndarray
    if isinstance(layer_index, int):
        emb = np.asarray(all_arr[:, layer_index, :], dtype=np.float32, copy=False)
    elif layer_index == "last":
        emb = np.asarray(all_arr[:, -1, :], dtype=np.float32, copy=False)
    elif layer_index == "all":
        emb = np.asarray(all_arr, dtype=np.float32, copy=False)
    else:
        raise ValueError(layer_index)

    shape_str = "×".join(str(d) for d in emb.shape)
    zip_note = ""
    if lbg_story_text_zip:
        zip_note = " (+zip transcript)"
    else:
        zip_note = " (TextGrid transcript)"
    logger.info(
        "HF word features (l-bg)%s: model=%s%s section=%s words=%d "
        "shape=%s maxlen=%d stride=%s device=%s",
        zip_note,
        model_name,
        f" @{rev}" if rev else "",
        section_id,
        emb.shape[0],
        shape_str,
        maxlen,
        stride,
        device,
    )
    if cache_path is not None and _try_save_cached_features(cache_path, emb):
        resolved = cache_path.resolve()
        nbytes = resolved.stat().st_size if resolved.is_file() else 0
        logger.info(
            "HF word features (l-bg): cached → %s (%d MB, shape %s)",
            resolved,
            nbytes // (1024 * 1024),
            shape_str,
        )
    return emb


# ---------------------------------------------------------------------------
# Surprisal extraction (l-bg full-text sliding window)
# ---------------------------------------------------------------------------

def contextual_word_surprisal_hf_lbg(
    words_df: "pd.DataFrame",
    model_name: str,
    *,
    bids_root: "Path | str | None" = None,
    lbg_story_text_zip: "Path | str | None" = None,
    lang: str = "en",
    section_id: "int | str | None" = None,
    revision: "str | None" = None,
    device: str = "cpu",
    hf_lbg_stride_delta: int = 64,
    hf_lbg_max_length: "int | None" = None,
    tokenizer_no_revision: bool = False,
    cache_dir: "Path | str | None" = None,
) -> "np.ndarray":
    """Word-level surprisal (-log P) from a causal LM using the l-bg sliding window.

    Uses the same full-text tokenisation and word alignment as
    ``contextual_word_embeddings_hf_lbg``, but computes token-level
    surprisal (sum over constituent tokens per word) instead of hidden states.

    Returns
    -------
    np.ndarray, shape ``(n_words,)``
        Surprisal in nats for each word.
    """
    from transformers import AutoConfig

    lg = lang.lower()
    rev = revision if revision not in (None, "", "main") else None
    cf_kw = {"revision": rev} if rev is not None else {}
    try:
        lm_cfg = AutoConfig.from_pretrained(model_name, **cf_kw)
    except Exception:
        lm_cfg = None
    maxlen_pre = getattr(lm_cfg, "max_position_embeddings", None) if lm_cfg is not None else None
    maxlen0 = int(hf_lbg_max_length or maxlen_pre or 2048)
    stride0 = maxlen0 - hf_lbg_stride_delta

    cache_path: "Path | None" = None
    if cache_dir is not None:
        cache_path = _lbg_feature_cache_path(
            cache_dir, model_name, revision, lg, section_id,
            maxlen0, stride0, "surprisal",
        )
        cached = _load_cached_features(cache_path)
        if cached is not None:
            arr = cached.squeeze()
            # words_df is already filtered (no space words) by _words_df_for_features.
            # Validate shape so stale caches are detected early.
            expected_n = len(words_df)
            if arr.shape[0] == expected_n:
                logger.info("LPP surprisal (l-bg): loaded from cache %s", cache_path.resolve())
                return arr
            logger.warning(
                "LPP surprisal (l-bg): stale cache %s (shape %s, expected %d words) "
                "— recomputing",
                cache_path.resolve(), arr.shape, expected_n,
            )

    import torch

    # For surprisal we only need logits, not hidden states — use the shared model cache.
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=revision,
        device=device,
        load_kw={},
        tokenizer_no_revision=tokenizer_no_revision,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "l-bg surprisal requires a fast tokenizer with return_offsets_mapping; "
            f"got {type(tokenizer).__name__}"
        )

    maxlen, stride = maxlen0, stride0

    fulltext_run = load_story_text(
        bids_root=bids_root, lang=lg, section=section_id,
        lbg_story_text_zip=Path(lbg_story_text_zip) if lbg_story_text_zip else None,
    ).replace("\n", " ").replace("\r", " ")

    word_list, char_to_word_idx, n_words_after_cn = build_lbg_word_stream(words_df, lg)
    if not word_list:
        raise ValueError("Empty word stream for LPP l-bg surprisal extraction")

    overflow_enc = tokenizer(
        fulltext_run,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=maxlen,
        return_overflowing_tokens=True,
        stride=stride,
    )
    om = tp.cast("torch.Tensor", overflow_enc["offset_mapping"])
    seq_len = int(overflow_enc["input_ids"].shape[1])

    # ── word → (batch, token) alignment (identical to lbg embeddings) ──────
    idx_batch = 0
    idx_token = 0
    idx_word_to_idx_token: list[tuple[int, int]] = []
    for idx_word, raw_w in enumerate(word_list):
        w_key = simplify_word(raw_w, lg)
        if idx_token == seq_len:
            idx_batch += 1
            idx_token = stride
        i_start = int(om[idx_batch, idx_token, 0].item())
        i_stop = int(om[idx_batch, idx_token, 1].item())
        n_try = 0
        # See the identical guard in contextual_word_embeddings_hf_lbg above
        # for the full rationale: phantom-expansion candidates get a much
        # narrower window to avoid a false-positive match on a short,
        # unrelated word several tokens away.
        max_try = 3 if w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS else 20
        while n_try < max_try and not do_word_match(w_key, fulltext_run[i_start:i_stop], lg):
            idx_token += 1
            if idx_token == seq_len:
                idx_batch += 1
                idx_token = stride
            i_start = int(om[idx_batch, idx_token, 0].item())
            i_stop = int(om[idx_batch, idx_token, 1].item())
            n_try += 1
        if n_try == max_try:
            if idx_word_to_idx_token and w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS:
                # Auxiliary-contraction phantom split -- see
                # LBG_PHANTOM_CONTRACTION_EXPANSIONS docstring. Give it the
                # next token slot rather than literally reusing the
                # preceding word's index, so every word keeps a distinct,
                # monotonically increasing token index (same invariant as
                # contextual_word_embeddings_hf_lbg -- kept consistent here
                # even though an empty span only zeros this word's summed
                # surprisal rather than crashing).
                prev_batch, prev_token = idx_word_to_idx_token[-1]
                alias_batch, alias_token = prev_batch, prev_token + 1
                if alias_token == seq_len:
                    alias_batch += 1
                    alias_token = stride
                logger.warning(
                    "l-bg surprisal alignment: word %d ('%s', model=%s, "
                    "section=%s) has no independent surface token (UD "
                    "auxiliary-contraction split) -- aliasing onto the "
                    "next available token instead of failing.",
                    idx_word, raw_w, model_name, section_id,
                )
                idx_word_to_idx_token.append((alias_batch, alias_token))
                idx_batch, idx_token = alias_batch, alias_token + 1
                continue
            raise ValueError(
                f"l-bg surprisal alignment failed "
                f"(word index {idx_word}, model={model_name}, section={section_id})"
            )
        idx_word_to_idx_token.append((idx_batch, idx_token))
        if lg != "cn":
            idx_token += 1
        elif idx_word < len(word_list) - 1 and not do_word_match(
            str(word_list[idx_word + 1]), fulltext_run[i_start:i_stop], lg
        ):
            idx_token += 1

    # ── forward passes: collect per-token log-probs per batch ───────────────
    nb = overflow_enc["input_ids"].shape[0]
    # logp_batches[b][t] = -log P(token_t | context) in batch b
    logp_batches: list[np.ndarray] = []
    inp_device = device

    for k in range(int(nb)):
        ids_k = overflow_enc["input_ids"][k : k + 1].to(inp_device)
        am_k = overflow_enc["attention_mask"][k : k + 1].to(inp_device)
        with torch.no_grad():
            out = model(input_ids=ids_k, attention_mask=am_k)
        logits = out.logits[0].float().cpu()             # (seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1).numpy()
        ids_cpu = overflow_enc["input_ids"][k].numpy()   # (seq_len,)
        # surprisal[t] = -log P(token_t | token_0..t-1) = -log_probs[t-1, ids[t]]
        # token 0 has no predecessor → surprisal = 0
        surp = np.zeros(seq_len, dtype=np.float32)
        surp[1:] = -log_probs[np.arange(seq_len - 1), ids_cpu[1:]]
        logp_batches.append(surp)

    # ── aggregate to word level: sum token surprisals ────────────────────────
    word_surprisals: list[float] = []

    def _tok_surp(b: int, t: int) -> float:
        return float(logp_batches[b][t])

    for idx_word in range(len(word_list) - 1):
        b_a, t_a = idx_word_to_idx_token[idx_word]
        b_b, t_b = idx_word_to_idx_token[idx_word + 1]
        s = 0.0
        if b_a == b_b:
            for t in range(t_a, t_b):
                s += _tok_surp(b_a, t)
        else:
            for t in range(t_a, seq_len):
                s += _tok_surp(b_a, t)
            for t in range(stride, t_b):
                s += _tok_surp(b_b, t)
        word_surprisals.append(s)

    # Last word: tokens from its start to end of valid sequence
    b_l, t_l = idx_word_to_idx_token[-1]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    row_ids = overflow_enc["input_ids"][b_l].numpy()
    row_attn = overflow_enc["attention_mask"][b_l].numpy()
    valid_len = int(row_attn.sum())
    s = 0.0
    for t in range(t_l, min(valid_len, seq_len)):
        if eos_id is not None and int(row_ids[t]) == eos_id:
            break
        s += _tok_surp(b_l, t)
    word_surprisals.append(s)

    result = np.array(word_surprisals, dtype=np.float32)

    # words_df is pre-filtered by _words_df_for_features (no space words).
    if result.shape[0] != len(words_df):
        raise ValueError(
            f"LPP surprisal shape mismatch: computed {result.shape[0]} surprisal values "
            f"but words_df has {len(words_df)} words "
            f"(model={model_name}, section={section_id}). "
            "The l-bg tokenizer alignment skipped or duplicated a word."
        )

    if cache_path is not None and _try_save_cached_features(cache_path, result[:, np.newaxis]):
        logger.info("LPP surprisal (l-bg): cached → %s (shape %s)", cache_path.resolve(), result.shape)

    return result


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def contextual_word_embeddings_last_layer(
    words: list[str],
    model_name: str,
    **kwargs: tp.Any,
) -> np.ndarray:
    """Shorthand: return only the last hidden layer, shape ``(n_words, hidden)``."""
    return contextual_word_embeddings_hf(words, model_name, layer_index="last", **kwargs)


def contextual_word_embeddings_all_layers(
    words: list[str],
    model_name: str,
    **kwargs: tp.Any,
) -> np.ndarray:
    """Shorthand: return all hidden layers, shape ``(n_words, n_layers, hidden)``."""
    return contextual_word_embeddings_hf(words, model_name, layer_index="all", **kwargs)


__all__ = [
    "LayerIndex",
    "clear_hf_model_cache",
    "contextual_word_embeddings_hf",
    "contextual_word_embeddings_hf_lbg",
    "contextual_word_embeddings_last_layer",
    "contextual_word_embeddings_all_layers",
]
