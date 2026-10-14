"""Contextual word embeddings for LPP (HuggingFace causal LMs).

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

import logging
import re
import typing as tp
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from .features.llm import embedding_layer_selection  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# l-bg story text helpers (merged from lpp_lbg_fulltext.py)
# ---------------------------------------------------------------------------

LBG_STORY_TEXT_ZIP_PASSWORD_BYTES = b"lessentielestinvisiblepourlesyeux"

PUNCT = ["-", "'", "\u2019", "\u201c", "\u201d", "«", "»", "\u2014"]
CHINESE_PUNCT = ["，", "《", "》", "。", "：", "“", "”", "；", "？", "！", "…", "、"]

# LPP's word table follows the UD convention of splitting auxiliary
# contractions into two annotated words (e.g. "hasn't" -> "hasnt" + "not",
# "I'd" -> "i'd" + "would"). do_word_match's loose startswith/substring rules
# let the FIRST piece absorb the whole surface contraction (e.g. "hasnt"
# matches a token spelled "hasnt" or "hasn't" in full), leaving the SECOND
# piece -- a pure grammatical expansion -- with literally no independent
# surface text anywhere in the story. Confirmed empirically across every
# non-drift l-bg alignment failure found in the LPP English corpus (sections
# 1/4/5/6/7, spanning both "n't"->not and "'d"->would splits); only used as a
# fallback once the normal forward search has already exhausted every
# candidate in range, so it can't preempt a legitimate match elsewhere.
LBG_PHANTOM_CONTRACTION_EXPANSIONS = frozenset({
    "not", "nt", "t",         # n't -- LPP's word column spells this
                              # phantom as the expanded "not", the abbreviated
                              # "nt", OR the bare trailing letter "t"
                              # (106 "t"/lemma="not" rows in the EN word
                              # table) depending on which UD split rule fired
                              # for that particular contraction; all three
                              # spellings need to hit this fallback.
    "would", "will", "d", "ll",   # 'd / 'll, with the same bare-letter
                              # spellings ("d": would/had, "ll": will) as
                              # word-column alternatives to the full lemma.
    "have", "has", "had", "ve",   # 've / 's / 'd, "ve" being the bare
                              # spelling for "have" (22 rows).
    "am", "is", "are", "m", "re",  # 'm / 's / 're, "m"/"re" bare spellings
                              # for am/are.
    "s",                      # 's, annotated with the LPP word-table's bare
                              # "s" spelling rather than the expanded "is"/
                              # "has" (lemma carries the expansion; e.g.
                              # section 1 word 238 "it's" -> word="s",
                              # lemma="is") -- same UD contraction-split
                              # phantom as "is"/"has" above, just under a
                              # different literal spelling. Confirmed via
                              # the LPP EN word table: ~200 "s"/lemma="is"
                              # rows across all 9 sections.
    "clock",                  # "o'clock" -- LPP's word column splits this
                              # (a single BPE token in most tokenizers, e.g.
                              # BLOOM's " o'clock") into "o" + "clock" rows.
                              # Not an aux-contraction like the rest of this
                              # set, but the same "no independent surface
                              # token" situation: "clock" merges entirely
                              # into the "o'clock" token that "o" itself
                              # already anchors, so it needs the same narrow-
                              # window-then-alias fallback rather than
                              # do_word_match's loose substring rule wandering
                              # forward onto an unrelated later "clock" token
                              # (confirmed on LPP EN section 7: "four o'clock"
                              # at word 941 was matching a "three o'clock"
                              # several words later instead, corrupting
                              # downstream alignment through word 944).
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
        raise ValueError(
            f"Cannot decode TextGrid {path}: tried utf-8-sig, utf-16, utf-8, latin-1"
        )
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


_APOSTROPHES = ("'", "’")


def do_word_match(word_in_list: str, word_in_text: str, lang: str) -> bool:
    raw_text_stripped = str(word_in_text).strip()
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
    if (
        len(word_in_list) == 1
        and word_in_text.startswith(word_in_list)
        and len(raw_text_stripped) > 1
        and raw_text_stripped[1] in _APOSTROPHES
        and raw_text_stripped[0].lower() == word_in_list
    ):
        # A single-character word (e.g. the pronoun "I", or the "o" of
        # "o'clock") merges with a following contraction/clitic into one BPE
        # token in some tokenizers (BLOOM: " I've"/" I'm"/" I'd"/" I'll"/
        # "o'clock" are each a single token, simplify_word stripping the
        # apostrophe -> "ive"/"im"/"id"/"ill"/"oclock"). word_in_list has
        # length 1, so the general `word_in_list in word_in_text` substring
        # rule above never fires for it (min_len=1 for non-CN languages
        # requires length > 1) -- without this, matching silently skips the
        # merged token and keeps searching forward for a standalone token
        # starting with the same letter, landing on some unrelated later
        # occurrence and corrupting all downstream alignment (confirmed on
        # LPP EN: section 1 word 366 "I" in "I've", section 7 word 940 "o" in
        # "o'clock"). Requiring the apostrophe to sit immediately after the
        # single letter in the RAW (unsimplified) token text keeps this
        # narrow -- it only fires for genuine letter+clitic merges, not
        # arbitrary short-word substring collisions (the class of bug fixed
        # separately for phantom-contraction candidates below).
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


def _get_cached_causal_lm(
    model_name: str,
    *,
    revision: str | None,
    device: str,
    load_kw: dict[str, object],
    tokenizer_no_revision: bool = False,
    trust_remote_code: bool = False,
    n_params: float | None = None,
    untrained: bool = False,
    untrained_seed: int = 0,
) -> tp.Any:
    """Load and LRU-cache (size-1) a causal-LM tokenizer + model pair.

    ``tokenizer_no_revision=True`` loads the tokenizer from the **base** model
    (no revision / step), matching l-bg's ``extract_llm_activations.py`` which
    always calls ``AutoTokenizer.from_pretrained(base_name)`` for Pythia.
    The *model* weights still come from ``revision``.

    ``n_params`` enables multi-GPU sharding (``device_map``) for large models
    via :func:`neuroai.features.llm._apply_device_map_if_needed` — without it,
    models too big for one GPU (e.g. OPT-66B) OOM even when multiple GPUs are
    visible, since a plain ``model.to(device)`` never spreads across devices.

    ``untrained=True`` skips pretrained weight loading: the architecture and
    tokenizer are still resolved from ``model_name``/``revision``, but the
    model is built via ``from_config`` (HuggingFace's own architecture-correct
    random init) instead of ``from_pretrained`` — untrained baseline.
    """
    global _hf_slot_key, _hf_slot_tokenizer, _hf_slot_model
    rev_key = revision if revision not in (None, "", "main") else "main"
    slot = (
        model_name, rev_key, str(device), tokenizer_no_revision,
        untrained, untrained_seed if untrained else 0,
    )
    if _hf_slot_key == slot and _hf_slot_model is not None:
        _ensure_pad_token_for_batches(_hf_slot_tokenizer)
        return _hf_slot_tokenizer, _hf_slot_model
    clear_hf_model_cache()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, set_seed

    from neuroai.features.llm import (
        _apply_device_map_if_needed,
        _load_hf_tokenizer,
        _resolve_causal_lm_class,
    )

    # See neuroai.features.llm._get_cached_causal_lm's identical guard: gpt-bert's
    # custom DWA-accumulator code crashes in bf16, so this is auto-detected
    # rather than relying on every caller to pass dtype="float32" explicitly.
    if "torch_dtype" not in load_kw and "gpt-bert" in model_name.lower():
        load_kw = {**load_kw, "torch_dtype": torch.float32}

    model_rev = revision if revision not in (None, "", "main") else None
    model_rev_fp = {"revision": model_rev} if model_rev is not None else {}
    tok_rev_fp = {} if tokenizer_no_revision else model_rev_fp
    if trust_remote_code:
        model_rev_fp = {**model_rev_fp, "trust_remote_code": True}
        tok_rev_fp = {**tok_rev_fp, "trust_remote_code": True}
    # Route through llm.py's _load_hf_tokenizer rather than a bare
    # AutoTokenizer.from_pretrained: zhixuan-lin/{fox,transformer}-* ship an
    # unregistered tokenizer_class ("JSONGPT2Tokenizer") that only resolves
    # after importing forgetting_transformer.tokenizer as a side effect (see
    # that function's docstring) -- calling AutoTokenizer directly here raised
    # "Couldn't instantiate the backend tokenizer" for every fox/transformer
    # model despite _load_fox_direct already handling the *model* load below.
    tokenizer = _load_hf_tokenizer(model_name, tok_rev_fp, trust_remote_code=trust_remote_code)
    # Hidden states are requested in forward(), not here — passing
    # output_hidden_states to from_pretrained breaks GPT-NeoX when we inject a
    # custom config for sharded Pythia-2.8B weights (transformers_weights index).
    model_kw = {k: v for k, v in load_kw.items() if k != "output_hidden_states"}
    model_kw.update(model_rev_fp)
    model_kw = _configure_pythia_28b_model_load(
        model_name, model_rev, model_rev_fp, model_kw, rev_key=rev_key,
    )
    # Some architectures (e.g. Mistral3, whose config class is
    # Mistral3Config) aren't resolvable via plain AutoModelForCausalLM --
    # "Unrecognized configuration class ... for this kind of AutoModel".
    # Same fix as neuroai.features.llm._get_cached_causal_lm: resolve the
    # concrete class from the model_type up front.
    from neuroai.features.llm import (
        _FORGETTING_DIRECT_PREFIX,
        _FOX_DIRECT_PREFIX,
        _MOLM_DIRECT_PREFIX,
        _load_forgetting_direct,
        _load_fox_direct,
        _load_moduleformer_direct,
    )

    if untrained:
        import torch

        set_seed(untrained_seed)
        use_device_map = False
        config = AutoConfig.from_pretrained(model_name, **model_rev_fp)
        _dtype = model_kw.get(
            "torch_dtype", torch.bfloat16 if device.startswith("cuda") else torch.float32,
        )
        model_cls = _resolve_causal_lm_class(model_name, model_rev_fp, trust_remote_code=trust_remote_code)
        model = model_cls.from_config(config, torch_dtype=_dtype)
    elif model_name.startswith(_MOLM_DIRECT_PREFIX):
        # ibm/MoLM-* ("moduleformer" architecture) isn't a registered
        # AutoConfig model_type -- bypass AutoModelForCausalLM/AutoConfig
        # entirely, same as neuroai.features.llm._get_cached_causal_lm (see
        # _load_moduleformer_direct's docstring there for the full rationale).
        # Largest checkpoint is ~8B params total, always fits on one GPU, so
        # no device_map sharding needed here either.
        use_device_map = False
        model = _load_moduleformer_direct(model_name, model_rev_fp)
        _dtype = model_kw.get("torch_dtype", torch.bfloat16)
        model = model.to(dtype=_dtype).to(device)
    elif model_name.startswith(_FORGETTING_DIRECT_PREFIX):
        # Lanni-ni/forgetting_* ship trust_remote_code that is not in the offline HF cache (and is buggy);
        # load via the vendored forgetting_transformer classes, same as neuroai.features.llm. Small (27-45M).
        use_device_map = False
        model = _load_forgetting_direct(model_name, model_rev_fp)
        _dtype = model_kw.get("torch_dtype", torch.bfloat16)
        model = model.to(dtype=_dtype).to(device)
    elif model_name.startswith(_FOX_DIRECT_PREFIX):
        # zhixuan-lin/{fox,transformer}-* aren't registered AutoConfig
        # model_types either (see _load_fox_direct's docstring) -- the
        # tokenizer fix above (_load_hf_tokenizer) only covers the
        # tokenizer; the *model* load must bypass AutoModelForCausalLM the
        # same way MoLM does just above, or from_pretrained's internal
        # AutoConfig.from_pretrained raises "Transformers does not
        # recognize this architecture" before ever reaching the weights.
        use_device_map = False
        model = _load_fox_direct(model_name, model_rev_fp)
        _dtype = model_kw.get("torch_dtype", torch.bfloat16)
        model = model.to(dtype=_dtype).to(device)
    else:
        model_cls = _resolve_causal_lm_class(model_name, model_rev_fp, trust_remote_code=trust_remote_code)
        use_device_map = _apply_device_map_if_needed(model_kw, device, n_params=n_params)
        model = model_cls.from_pretrained(model_name, **model_kw)
    model.eval()
    if not use_device_map:
        model.to(device)
    _ensure_pad_token_for_batches(tokenizer)
    _hf_slot_key = slot
    _hf_slot_tokenizer = tokenizer
    _hf_slot_model = model
    logger.info(
        "LPP HF: loaded & cached %s @%s on %s (tok_no_rev=%s, device_map=%s)",
        model_name, rev_key, device, tokenizer_no_revision, use_device_map,
    )
    return tokenizer, model


def _is_fla_model(model_name: str) -> bool:
    """forgetting-transformer / FoX checkpoints run FLA kernels that mishandle padded inputs."""
    from neuroai.features.llm import _FORGETTING_DIRECT_PREFIX, _FOX_DIRECT_PREFIX
    return model_name.startswith((_FORGETTING_DIRECT_PREFIX, _FOX_DIRECT_PREFIX))


def _forward_all_hidden_states(
    model: tp.Any,
    input_ids: tp.Any,
    attention_mask: tp.Any,
    *,
    model_name: str = "",
) -> tuple[tuple[torch.Tensor, ...], bool]:
    """Run one forward pass and return per-layer hidden states as CPU-ready tensors.

    Uses ``use_cache=False`` for custom HF models (ALiBi / stickbreaking / forgetting)
    whose remote code breaks on ``DynamicCache.to_legacy_cache()``.

    Falls back to forward hooks when ``output_hidden_states=True`` is ignored.
    """
    import torch

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    # Some trust_remote_code architectures (e.g. GPT-BERT) hand back a plain
    # tuple instead of a ModelOutput regardless of return_dict — fall through
    # to the hook-based path below rather than crashing on `.hidden_states`.
    hs = None if isinstance(out, tuple) else out.hidden_states
    if hs is not None:
        return hs, False

    if model_name:
        logger.warning(
            "LPP HF: %s returned hidden_states=None — falling back to forward hooks",
            model_name,
        )
    hs_list: list[torch.Tensor] = []
    hooks: list[tp.Any] = []
    inner = getattr(model, "model", model)
    emb = getattr(inner, "embeddings", None)
    if emb is not None:
        hooks.append(
            emb.register_forward_hook(
                lambda _m, _i, o: hs_list.append(
                    o.detach() if isinstance(o, torch.Tensor) else o[0].detach()
                )
            )
        )
    for lyr in getattr(inner, "layers", []):
        hooks.append(
            lyr.register_forward_hook(
                lambda _m, _i, o: hs_list.append(
                    o.detach() if isinstance(o, torch.Tensor) else o[0].detach()
                )
            )
        )
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    for h in hooks:
        h.remove()
    if not hs_list:
        raise RuntimeError(
            f"LPP HF: {model_name or 'model'} returned hidden_states=None and "
            "hook fallback found no layers."
        )
    return tuple(hs_list), True


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

from .features.llm import embedding_cache_tag  # noqa: E402


def _feature_cache_path(
    cache_dir: Path | str,
    model_name: str,
    revision: str | None,
    lang: str,
    section: int,
    context_words: int,
    max_length: int,
    layer_index: LayerIndex,
    untrained: bool = False,
    untrained_seed: int = 0,
) -> Path:
    safe_model = model_name.replace("/", "_")
    safe_rev = (revision or "main").replace("/", "_").replace("-", "_")
    if untrained:
        safe_rev = f"{safe_rev}_untrained_seed{untrained_seed}"
    layer_tag = f"layer{layer_index}" if isinstance(layer_index, int) else layer_index
    filename = (
        f"{safe_model}__{safe_rev}__{lang.lower()}"
        f"__sec{section:02d}"
        f"__ctx{context_words}"
        f"__maxlen{max_length}"
        f"__{layer_tag}{embedding_cache_tag(model_name)}.npz"
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
        f"__{layer_tag}{embedding_cache_tag(model_name)}.npz"
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
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic on lustre: parallel step jobs share one cache_dir. The temp name
    # must still end in ".npz" -- np.savez_compressed silently appends ".npz"
    # to any path that doesn't already end with it (e.g. a ".npz.tmp" name),
    # which previously produced a stray ".npz.tmp.npz" file that the
    # subsequent replace() could never find (FileNotFoundError).
    import os
    import uuid
    tmp = cache_path.with_name(f"{cache_path.stem}.tmp{os.getpid()}{uuid.uuid4().hex[:8]}.npz")
    np.savez_compressed(str(tmp), features=features)
    tmp.replace(cache_path)


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------

def _chunks(n: int, size: int) -> tp.Iterator[tuple[int, int]]:
    for i in range(0, n, size):
        yield i, min(i + size, n)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

@embedding_layer_selection
def contextual_word_embeddings_hf(
    words: list[str],
    model_name: str,
    *,
    layer_index: LayerIndex = "last",
    revision: str | None = None,
    device: str = "cpu",
    context_words: int = 48,
    max_length: int = 512,
    batch_size: int = 4,
    cache_dir: str | Path | None = None,
    section_id: int | None = None,
    lang: str = "en",
    tokenizer_no_revision: bool = False,
    trust_remote_code: bool = False,
    n_params: float | None = None,
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
        LPP section index (1-based).  Only used when ``cache_dir`` is set.
    lang
        Language code (``"en"``, ``"fr"``, ``"cn"``).  Only used for cache key.

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
                "LPP features: loaded from cache %s (shape %s)",
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
        # logit_lens_surprisal_all_layers — see that function's setup comment.
        load_kw["torch_dtype"] = dtype
    elif trust_remote_code and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            load_kw["torch_dtype"] = torch.float32
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=revision,
        device=device,
        load_kw=load_kw,
        tokenizer_no_revision=tokenizer_no_revision,
        trust_remote_code=trust_remote_code,
        n_params=n_params,
        untrained=untrained,
        untrained_seed=untrained_seed,
    )

    # Build context-windowed texts
    texts: list[str] = []
    for i, w in enumerate(words):
        left = words[max(0, i - context_words): i + 1]
        texts.append(" ".join(left))

    # Determine return_all_layers flag
    return_all = layer_index == "all"
    target_layer: int | None = None
    if isinstance(layer_index, int):
        target_layer = layer_index

    out_vecs: list[np.ndarray] = []

    for i0, i1 in _chunks(len(texts), batch_size):
        batch = texts[i0:i1]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            hs, _ = _forward_all_hidden_states(
                model,
                enc["input_ids"],
                enc["attention_mask"],
                model_name=model_name,
            )

        # hidden_states: tuple of (batch, seq, hidden) — one per layer
        # including the token embedding (index 0)
        n_model_layers = len(hs)

        for b in range(enc["input_ids"].shape[0]):
            attn = enc["attention_mask"][b]
            last_pos = int(attn.sum().item()) - 1

            if return_all:
                # Stack to (n_layers, hidden)
                vec = torch.stack(
                    [hs[li][b, last_pos] for li in range(n_model_layers)],
                    dim=0,
                ).float().cpu().numpy()
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
        "LPP HF features: model=%s%s layer=%s words=%d shape=%s device=%s",
        model_name,
        f" @{rev}" if rev else "",
        layer_index,
        len(words),
        shape_str,
        device,
    )

    # --- cache write ---
    if cache_path is not None:
        _save_cached_features(cache_path, emb)
        resolved = cache_path.resolve()
        nbytes = resolved.stat().st_size if resolved.is_file() else 0
        logger.info(
            "LPP features: cached → %s (%d MB, shape %s)",
            resolved,
            nbytes // (1024 * 1024),
            shape_str,
        )

    return emb


@embedding_layer_selection
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
    n_params: float | None = None,
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
    lg = lang.lower()

    rev = revision if revision not in (None, "", "main") else None

    cf_kw = {"revision": rev} if rev is not None else {}
    if trust_remote_code:
        cf_kw["trust_remote_code"] = True
    # ibm/MoLM-* ("moduleformer") and zhixuan-lin/{fox,transformer}-*
    # ("forgetting_transformer-project_fox" / "transformer-project_fox")
    # aren't registered AutoConfig model_types -- AutoConfig.from_pretrained
    # raises before ever reaching the config, even though only a single field
    # (the position-embedding limit) is needed here. Read config.json
    # directly instead, same source _load_moduleformer_direct/_load_fox_direct
    # use for the real model load (see their docstrings in
    # neuroai/features/llm.py for the full rationale) -- trust_remote_code
    # alone does not help either family (see _load_fox_direct's docstring).
    from .features.llm import _MOLM_DIRECT_PREFIX, _FOX_DIRECT_PREFIX
    if model_name.startswith(_MOLM_DIRECT_PREFIX) or model_name.startswith(_FOX_DIRECT_PREFIX):
        import json as _json
        from huggingface_hub import hf_hub_download as _hf_hub_download
        _cfg_path = _hf_hub_download(model_name, "config.json", revision=rev)
        with open(_cfg_path) as _f:
            _raw_cfg = _json.load(_f)
        maxlen_pre = (
            _raw_cfg.get("block_size") or _raw_cfg.get("n_positions")
            or _raw_cfg.get("max_position_embeddings")
        )
    else:
        from transformers import AutoConfig
        lm_cfg = AutoConfig.from_pretrained(model_name, **cf_kw)
        maxlen_pre = getattr(lm_cfg, "max_position_embeddings", None)
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
                "LPP features (l-bg): loaded from cache %s (shape %s)",
                cache_path.resolve(),
                cached_arr.shape,
            )
            return cached_arr

    import torch

    lbg_load_kw: dict[str, object] = {"output_hidden_states": True}
    if trust_remote_code:
        lbg_load_kw["trust_remote_code"] = True
    if dtype is not None:
        lbg_load_kw["torch_dtype"] = dtype
    elif trust_remote_code and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            lbg_load_kw["torch_dtype"] = torch.float32
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=revision,
        device=device,
        load_kw=lbg_load_kw,
        tokenizer_no_revision=tokenizer_no_revision,
        trust_remote_code=trust_remote_code,
        n_params=n_params,
    )
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
    am_enc = overflow_enc["attention_mask"]
    seq_len = int(overflow_enc["input_ids"].shape[1])

    def _batch_real_span(batch_idx: int) -> tuple[int, int]:
        # Every overflow chunk except (possibly) the last is a full `seq_len`
        # window with no padding, so its real tokens span [0, seq_len). The
        # *last* chunk can be shorter than `seq_len` (not enough remaining
        # text to fill a full window) and the tokenizer left-pads it (mamba2
        # and other causal-LM tokenizers default to ``padding_side="left"``),
        # which shifts its real tokens to start at some index > 0 rather than
        # 0. Blindly assuming [0, seq_len) or [0, valid_len) there reads
        # left-pad filler, or truncates the real span short, instead of the
        # actual token positions -- silently corrupting alignment and pooling.
        # Anchoring on the row's attention_mask finds the true real-token span
        # regardless of padding side.
        am_row = am_enc[batch_idx]
        real_sum = int(am_row.sum().item())
        first_real = int(am_row.nonzero()[0, 0].item()) if real_sum > 0 else 0
        return first_real, first_real + real_sum

    def _batch_start_token(batch_idx: int) -> int:
        first_real, _ = _batch_real_span(batch_idx)
        return stride + first_real

    idx_batch = 0
    idx_token = 0
    idx_word_to_idx_token: list[tuple[int, int]] = []

    for idx_word, raw_w in enumerate(word_list):
        w_key = simplify_word(raw_w, lg)
        if idx_token == seq_len:
            idx_batch += 1
            idx_token = _batch_start_token(idx_batch)
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
        is_phantom_candidate = w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS
        max_try = 3 if is_phantom_candidate else 20

        def _candidate_matches(text_span: str) -> bool:
            if is_phantom_candidate:
                # Phantom-expansion candidates (bare UD-split fragments like
                # "nt"/"t"/"d"/"ll"/"ve"/"m"/"re"/"s", and ambiguous full
                # spellings like "not"/"would"/"have"/"is") have no
                # independent surface token in the overwhelming common case
                # -- do_word_match's loose substring rule ("word_in_list in
                # word_in_text" or vice versa) can accidentally match a short
                # fragment against an unrelated token that merely CONTAINS
                # those letters (e.g. "nt" is a literal substring of "want",
                # "an" is a literal substring of "elephant") while scanning
                # the narrow post-failure window. That's worse than failing
                # fast: it silently anchors the phantom word onto the WRONG
                # token, consumes it, and corrupts every later word's search
                # start position (confirmed on LPP EN section 1: "nt" at
                # word 885 matching "want" cascaded into a real alignment
                # failure 6 words later at word 891 "boa"). Requiring exact
                # equality here still matches the rare case where the
                # annotation's phantom spelling genuinely has its own
                # standalone token (e.g. "not" as a real, unmerged word),
                # while refusing coincidental substring hits.
                return simplify_word(w_key, lg) == simplify_word(text_span, lg)
            return do_word_match(w_key, text_span, lg)

        while n_try < max_try and not _candidate_matches(
            fulltext_run[i_start:i_stop]
        ):
            idx_token += 1
            if idx_token == seq_len:
                idx_batch += 1
                idx_token = _batch_start_token(idx_batch)
            i_start = int(om[idx_batch, idx_token, 0].item())
            i_stop = int(om[idx_batch, idx_token, 1].item())
            n_try += 1
        if n_try == max_try:
            if idx_word_to_idx_token and w_key in LBG_PHANTOM_CONTRACTION_EXPANSIONS:
                # Auxiliary-contraction phantom split (see
                # LBG_PHANTOM_CONTRACTION_EXPANSIONS docstring) -- this word
                # has no independent surface token. Give it the very next
                # token slot (typically a leftover punctuation/quote token
                # from the contraction, e.g. the "'" the preceding word's
                # match skipped past) rather than literally reusing the
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
                    alias_token = _batch_start_token(alias_batch)
                # If the very next real word already anchors at this
                # candidate slot -- e.g. BLOOM merges "it"+"'s" into a
                # single " It's" token immediately followed by " true", with
                # no spare token in between at all -- grabbing prev_token+1
                # here would steal that word's only anchor, leaving it
                # unfindable by the time its own turn comes (search only
                # moves forward). Share the preceding word's own token
                # instead: the empty span this creates for the *preceding*
                # word is handled by a single-token fallback in the pooling
                # loop below, and the real next word keeps its anchor.
                if idx_word + 1 < len(word_list):
                    _next_key = simplify_word(word_list[idx_word + 1], lg)
                    _ns = int(om[alias_batch, alias_token, 0].item())
                    _ne = int(om[alias_batch, alias_token, 1].item())
                    if _next_key not in LBG_PHANTOM_CONTRACTION_EXPANSIONS and do_word_match(
                        _next_key, fulltext_run[_ns:_ne], lg
                    ):
                        alias_batch, alias_token = prev_batch, prev_token
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
        ids_k = overflow_enc["input_ids"][k : k + 1].to(device)
        am_k = overflow_enc["attention_mask"][k : k + 1].to(device)
        n_real = int(am_k.sum())
        full_len = int(ids_k.shape[1])
        # FLA kernels (forgetting / FoX) corrupt hidden states when the window is padded; forward only the real
        # tokens and pad the hidden states back (right padding only, so token positions are unchanged).
        trim = (_is_fla_model(model_name) and n_real < full_len and bool(am_k[0, :n_real].all()))
        with torch.no_grad():
            hs, _ = _forward_all_hidden_states(
                model,
                ids_k[:, :n_real] if trim else ids_k,
                am_k[:, :n_real] if trim else am_k,
                model_name=model_name,
            )
        if trim:
            hs = tuple(torch.nn.functional.pad(h, (0, 0, 0, full_len - n_real)) for h in hs)
        hidden_states_batches.append(
            [hs[j][0].detach().float().cpu().numpy() for j in range(len(hs))]
        )
        del hs

    # The forward-hook fallback in _forward_all_hidden_states (for custom
    # architectures — e.g. MoE "babylm" checkpoints — that ignore
    # output_hidden_states=True) can capture a different *number* of layers,
    # and/or a different *hidden dim at the same layer index*, on different
    # chunks (conditional/MoE modules firing hooks inconsistently). Keep only
    # the layer indices whose hidden dim is the per-chunk modal dim in EVERY
    # chunk (same technique as the fulltext extractor's keep_layers filter),
    # so the per-layer stacking below never sees a shape mismatch.
    from collections import Counter as _Counter

    good_layers: set[int] | None = None
    for batch in hidden_states_batches:
        dims = [arr.shape[-1] for arr in batch]
        modal_dim = _Counter(dims).most_common(1)[0][0]
        layers_here = {li for li, d in enumerate(dims) if d == modal_dim}
        good_layers = layers_here if good_layers is None else (good_layers & layers_here)
    if not good_layers:
        raise RuntimeError(
            f"LPP HF: {model_name} produced no hidden-state layer with a "
            "consistent shape across all chunks — cannot extract embeddings."
        )
    keep_layers = sorted(good_layers)
    total_layers = {len(b) for b in hidden_states_batches}
    if len(keep_layers) < max(total_layers):
        logger.warning(
            "LPP HF: %s produced inconsistent hidden-state layers across chunks "
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
            # A phantom-alias collision (see LBG_PHANTOM_CONTRACTION_EXPANSIONS
            # aliasing above) can leave a word sharing its token index with
            # its own predecessor, making this span empty -- fall back to
            # that single shared token rather than crashing on an empty
            # np.stack.
            hi = idx_tok_b if idx_tok_b > idx_tok_a else idx_tok_a + 1
            for i in range(idx_tok_a, hi):
                for li2 in range(n_layers_tpl):
                    emb_layers[li2].append(_pull(idx_batch_a, i, li2))
        else:
            for i in range(idx_tok_a, seq_len):
                for li2 in range(n_layers_tpl):
                    emb_layers[li2].append(_pull(idx_batch_a, i, li2))
            for i in range(_batch_start_token(idx_batch_b), idx_tok_b):
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
    _, real_end_l = _batch_real_span(idx_batch_l)
    for i in range(idx_tok_l, min(real_end_l, seq_len)):
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
        "LPP HF features (l-bg)%s: model=%s%s section=%s words=%d "
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
    if cache_path is not None:
        _save_cached_features(cache_path, emb)
        resolved = cache_path.resolve()
        nbytes = resolved.stat().st_size if resolved.is_file() else 0
        logger.info(
            "LPP features (l-bg): cached → %s (%d MB, shape %s)",
            resolved,
            nbytes // (1024 * 1024),
            shape_str,
        )
    return emb


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
