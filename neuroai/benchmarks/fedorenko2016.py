"""Fedorenko2016 iEEG encoding benchmark (word-level scalar responses).

Predicts per-word electrode responses from LLM hidden states using grouped
sentence-level ridge CV. Targets are scalar responses (no time axis), so this
benchmark uses multi-output scalar ridge (all electrodes solved jointly).
"""

from __future__ import annotations

import hashlib
import logging
import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..benchmarks.base import ModelConfig, RegressionConfig, word_predictions_dir
from ..datasets.fedorenko2016 import Fedorenko2016Dataset
from ..regressors.ridge import CVScore, cv_ridge_score_multioutput, cv_ridge_score_multioutput_gpu
from ..features.llm import embedding_layer_selection  # noqa: E402

logger = logging.getLogger(__name__)


def _safe_model_id(model_name: str) -> str:
    return model_name.replace("/", "_")


def _feature_cache_path(
    cache_dir: Path,
    model_name: str,
    revision: str,
    sentence_hash: str,
) -> Path:
    from ..features.llm import embedding_cache_tag

    return cache_dir / _safe_model_id(model_name) / revision / f"fedorenko2016_{sentence_hash}{embedding_cache_tag(model_name)}.npz"


def _sentences_hash(sentences: list[str]) -> str:
    joined = "||".join(sentences)
    return hashlib.md5(joined.encode()).hexdigest()[:12]


def _resolve_transformer_hidden_states(
    all_hidden_states: tuple[torch.Tensor, ...],
    *,
    n_layers: int,
    model_name: str,
    warned: list[bool],
) -> tuple[torch.Tensor, ...]:
    """Drop the token-embedding layer, returning exactly ``n_layers`` tensors.

    Normally ``all_hidden_states[0]`` is the embedding layer and the rest are
    the ``n_layers`` decoder outputs, all sharing ``hidden_size``. OPT-350M
    (and similar models with ``word_embed_proj_dim != hidden_size``) instead
    emit one tensor whose last dim is the projection size — at the *end* of
    the tuple in current transformers versions, not necessarily at index 0 —
    so a positional ``[1:]`` slice can keep the mismatched tensor and drop a
    real layer instead (same issue handled positionally in
    :func:`neuroai.features.llm._select_uniform_hidden_layers` and inline in
    ``eeg_erp.py``). Filter by dimension instead of position to be robust to
    either layout.
    """
    dims = [int(t.shape[-1]) for t in all_hidden_states]
    if len(set(dims)) <= 1:
        return all_hidden_states[1:]

    from collections import Counter

    modal_dim = Counter(dims).most_common(1)[0][0]
    kept = tuple(t for t in all_hidden_states if int(t.shape[-1]) == modal_dim)
    if not warned[0]:
        logger.warning(
            "%s: non-uniform hidden dims across %d hidden-state tensors — "
            "keeping %d tensor(s) with dim=%d (embedding/output projection "
            "mismatch, e.g. OPT-350M's word_embed_proj_dim != hidden_size)",
            model_name,
            len(all_hidden_states),
            len(kept),
            modal_dim,
        )
        warned[0] = True
    if len(kept) != n_layers:
        raise ValueError(
            f"{model_name}: expected {n_layers} uniform-dim hidden states after "
            f"filtering, got {len(kept)} (dims={sorted(set(dims))})"
        )
    return kept


@embedding_layer_selection
def extract_embeddings_and_perplexity_by_sentence(
    *,
    sentence_tokens: list[list[str]],
    sentence_texts: list[str],
    model_name: str,
    revision: str | None,
    device: str,
    batch_size: int,
    tokenizer_no_revision: bool,
    trust_remote_code: bool,
    n_params: float | None,
    cache_dir: str | None,
    max_length: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract word embeddings + per-sentence NLL/token counts.

    Returns
    -------
    word_emb : np.ndarray
        Shape ``(n_words_total, n_layers, hidden_size)``, sentence-concatenated.
    sentence_nll : np.ndarray
        Mean NLL per sentence.
    sentence_ntoks : np.ndarray
        Number of scored LM tokens per sentence (for corpus perplexity).
    """
    from neuroai.features.llm import (
        _get_cached_causal_lm,
        _model_input_device,
        config_dim_attr,
    )
    from neuroai.features.cbr_rnn import contextual_word_embeddings_cbr_rnn

    rev = revision or "main"
    sent_hash = _sentences_hash(sentence_texts)
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_base = Path(cache_dir)
        cache_path = _feature_cache_path(cache_base, model_name, rev, sent_hash)
        if cache_path.is_file():
            cached = np.load(cache_path, allow_pickle=True)
            cached_sentences = cached["sentences"].tolist()
            if cached_sentences == sentence_texts:
                logger.info("Feature cache hit: %s", cache_path)
                return (
                    cached["word_embeddings"],
                    cached["sentence_nll"],
                    cached["sentence_ntoks"],
                )
            logger.warning("Feature cache sentence mismatch; recomputing: %s", cache_path)

    # CBR-RNN is not an AutoModelForCausalLM; use native extractor and skip perplexity.
    if "cbr-rnn" in model_name.lower() or "cbr_rnn" in model_name.lower():
        sent_embs: list[np.ndarray] = []
        for toks in sentence_tokens:
            e = contextual_word_embeddings_cbr_rnn(
                toks,
                model_name,
                device=device,
                context_words=48,
                cache_dir=cache_dir,
                section_id=0,
                lang="en",
            )
            # Expand to "layerwise" shape with a singleton layer axis.
            sent_embs.append(e[:, None, :].astype(np.float32))
        word_embeddings = np.concatenate(sent_embs, axis=0)
        sentence_nll = np.full(len(sentence_tokens), np.nan, dtype=np.float64)
        sentence_ntoks = np.zeros(len(sentence_tokens), dtype=np.int32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                word_embeddings=word_embeddings,
                sentence_nll=sentence_nll,
                sentence_ntoks=sentence_ntoks,
                sentences=np.array(sentence_texts, dtype=object),
            )
        return word_embeddings, sentence_nll, sentence_ntoks

    load_kw: dict[str, tp.Any] = {"output_hidden_states": True}
    if trust_remote_code:
        load_kw["trust_remote_code"] = True
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=revision,
        device=device,
        load_kw=load_kw,
        tokenizer_no_revision=tokenizer_no_revision,
        n_params=n_params,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    input_device = _model_input_device(model)

    n_layers = config_dim_attr(
        model.config, ("num_hidden_layers", "n_layer", "n_layers", "num_layers")
    )
    n_sent = len(sentence_tokens)
    sentence_nll = np.full(n_sent, np.nan, dtype=np.float64)
    sentence_ntoks = np.zeros(n_sent, dtype=np.int32)
    word_emb_chunks: list[np.ndarray] = []
    dim_warned = [False]
    width_warned = [False]

    import torch.nn.functional as F

    # Hook-based fallback for FLA-style architectures (stickbreaking, forgetting, alibi)
    # that ignore output_hidden_states=True and return out.hidden_states = None.
    hook_captures: list[torch.Tensor] = []
    hooks: list = []

    def _make_hook(captures: list) -> tp.Any:
        def _hook(module: tp.Any, inp: tp.Any, output: tp.Any) -> None:
            h = output[0] if isinstance(output, tuple) else output
            captures.append(h.detach())
        return _hook

    def _find_transformer_layers(m: tp.Any) -> list:
        for path in ("model.layers", "transformer.h", "model.decoder.layers",
                     "decoder.layers", "layers", "model.model.layers"):
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if hasattr(obj, "__len__") and len(obj) > 0:
                    return list(obj)
            except AttributeError:
                continue
        raise RuntimeError(
            f"Cannot find transformer layers in {type(m).__name__} for hook-based extraction."
        )

    for start in range(0, n_sent, batch_size):
        end = min(start + batch_size, n_sent)
        toks_batch = sentence_tokens[start:end]
        enc = tokenizer(
            toks_batch,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )

        input_ids = enc["input_ids"].to(input_device)
        attention_mask = enc["attention_mask"].to(input_device)
        with torch.no_grad():
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            except TypeError:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

        # Some trust_remote_code checkpoints (notably BabyLM GPT-BERT) ignore
        # return_dict and return a tuple even when hidden states are requested.
        # Use the same hook fallback as models that explicitly return
        # hidden_states=None, rather than dereferencing `.hidden_states` here.
        output_hidden_states = (
            None if isinstance(out, (tuple, list)) else getattr(out, "hidden_states", None)
        )
        if output_hidden_states is None:
            # FLA / stickbreaking / forgetting / alibi: register hooks once, reuse every batch.
            if not hooks:
                found_layers = _find_transformer_layers(model)
                if len(found_layers) != n_layers:
                    # config-reported layer count doesn't always match the
                    # model's real module list — e.g. Jtapsa/moep's
                    # config.n_layers=12 names only its "conceptual" depth
                    # (1 large block + 10 parallel blocks + 1 large block);
                    # the real nn.ModuleList also holds 2 extra MoE
                    # shrink/grow projection modules in between, 14 total.
                    # Trust the discovered module count so the true final
                    # layer gets hooked, not a truncated config-sized prefix.
                    logger.warning(
                        "%s: found %d transformer layers via hooks, config "
                        "reports n_layers=%d — using the discovered count "
                        "(hook-based extraction reflects the real module "
                        "list, not the config's layer-count semantics)",
                        model_name, len(found_layers), n_layers,
                    )
                    n_layers = len(found_layers)
                for layer in found_layers[:n_layers]:
                    hooks.append(layer.register_forward_hook(_make_hook(hook_captures)))
            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=False, use_cache=False, return_dict=True)
                except TypeError:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=False)
            hidden_states = tuple(hook_captures[:n_layers])
            hook_captures.clear()
        else:
            hidden_states = _resolve_transformer_hidden_states(
                output_hidden_states, n_layers=n_layers, model_name=model_name, warned=dim_warned,
            )

        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, (tuple, list)) and out:
            # Causal-LM tuple outputs put logits first when labels are absent.
            logits = out[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"{model_name}: expected logits in ModelOutput or first tuple element, "
                f"got {type(out).__name__}"
            )
        logits = logits.float()

        # Per-sentence perplexity ingredients.
        shift_logits = logits[:, :-1, :]
        shift_ids = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].float()
        nll_per_tok = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_ids.reshape(-1),
            reduction="none",
        ).reshape(shift_logits.size(0), -1)
        nll_per_tok = nll_per_tok * shift_mask
        n_tok = shift_mask.sum(dim=1).clamp(min=1)
        mean_nll = (nll_per_tok.sum(dim=1) / n_tok).detach().cpu().numpy()
        sentence_nll[start:end] = mean_nll
        sentence_ntoks[start:end] = n_tok.detach().cpu().numpy().astype(np.int32)

        # Word embeddings from tokenizer word_ids map.
        for bi, toks in enumerate(toks_batch):
            word_ids = enc.word_ids(batch_index=bi)
            if word_ids is None:
                raise ValueError("Tokenizer must be fast for fedorenko2016 extraction.")
            n_words = len(toks)
            # Derive the per-layer width from the actual resolved tensors rather
            # than trusting model.config.hidden_size: some custom architectures
            # (e.g. MoE "babylm" checkpoints) report a config hidden_size that
            # doesn't match the real transformer layer output width (the inverse
            # of OPT-350M's embedding/hidden_size split) — allocating with the
            # config value there silently breaks the write below.
            #
            # Some architectures (e.g. Jtapsa/moep, a "shrink -> parallel ->
            # grow" MoE design) genuinely have DIFFERENT widths at different
            # layers by construction — most layers run at a reduced
            # `parallel_dim`, a couple run at the full residual-stream `dim`.
            # That's not a data problem, so allocate at the widest layer seen
            # and zero-pad narrower layers' unused columns (constant-zero
            # columns are harmless to the downstream ridge fit, just
            # uninformative for that layer/column range) rather than
            # asserting every layer shares one width.
            layer_widths = [int(h.shape[-1]) for h in hidden_states[:n_layers]]
            actual_hidden_size = max(layer_widths)
            if len(set(layer_widths)) > 1 and not width_warned[0]:
                logger.warning(
                    "%s: non-uniform per-layer hidden widths %s — zero-padding "
                    "narrower layers up to %d (architecture has mixed-width "
                    "layers, e.g. a shrink/grow MoE design; not a data error)",
                    model_name, layer_widths, actual_hidden_size,
                )
                width_warned[0] = True
            # NaN default (not zero) so a genuinely-unmapped word (no tok_pos,
            # `continue` below) still trips the existing non-finite -> zero
            # fallback/warning further down — only the intra-layer width
            # padding for mixed-width architectures is zero-filled explicitly.
            sent_out = np.full((n_words, n_layers, actual_hidden_size), np.nan, dtype=np.float32)
            for wi in range(n_words):
                tok_pos = [ti for ti, w in enumerate(word_ids) if w == wi]
                if not tok_pos:
                    continue
                for li in range(n_layers):
                    hs = hidden_states[li][bi, tok_pos, :]
                    v = hs.mean(dim=0).float().detach().cpu().numpy()
                    sent_out[wi, li, : v.shape[0]] = v
                    if v.shape[0] < actual_hidden_size:
                        sent_out[wi, li, v.shape[0]:] = 0.0
            word_emb_chunks.append(sent_out)

    for h in hooks:
        h.remove()

    word_embeddings = np.concatenate(word_emb_chunks, axis=0)
    if np.any(~np.isfinite(word_embeddings)):
        logger.warning("Non-finite embeddings found; replacing with zeros")
        word_embeddings = np.nan_to_num(word_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            word_embeddings=word_embeddings,
            sentence_nll=sentence_nll,
            sentence_ntoks=sentence_ntoks,
            sentences=np.array(sentence_texts, dtype=object),
        )
        logger.info("Feature cache written: %s", cache_path)

    return word_embeddings, sentence_nll, sentence_ntoks


@dataclass
class Fedorenko2016DataConfig:
    root: str
    subjects: list[int] | None = None
    electrodes: list[int] | None = None
    context_words: int | None = None
    max_length: int | None = None
    zscore_targets: bool = True
    drop_constant_electrodes: bool = True
    feature_cache_dir: str | None = None


@dataclass
class Fedorenko2016BenchmarkConfig:
    data: Fedorenko2016DataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    layerwise: bool = True
    n_splits: int = 5
    n_jobs: int = 20
    n_layer_workers: int = 8
    tokenizer_no_revision: bool = True
    regression_device: str = "cuda"  # cuda -> himalaya, cpu -> sklearn
    # Save every layer's out-of-fold predictions (n_layers, n_words,
    # n_electrodes) + the observed (z-scored) target matrix + word/electrode
    # identity to one per-checkpoint .npz, for post-hoc per-word difficulty
    # splits (e.g. DLT cost) — mirrors natural_stories_embed_gpu.py's
    # convention. return_oof_predictions=True is a free byproduct of the
    # same k-fold CV call on both cv_ridge_score_multioutput (CPU) and
    # cv_ridge_score_multioutput_gpu (GPU) — no extra regression calls, no
    # separate refit — so this is on for every layer by default, not just
    # the last one. On by default.
    save_word_predictions: bool = True
    output_dir: str | None = None


@dataclass
class FedorenkoElectrodeScore:
    electrode: int
    subject_uid: int
    score: CVScore
    layer_scores: list[CVScore] | None = None

    def best_layer_index(self, metric: str = "spearmanr") -> int | None:
        if not self.layer_scores:
            return None
        vals = np.asarray([getattr(s, metric) for s in self.layer_scores], dtype=float)
        if np.any(np.isfinite(vals)):
            return int(np.nanargmax(vals))
        return None

    def best_layer_spearmanr(self) -> float:
        if self.layer_scores:
            vals = np.asarray([s.spearmanr for s in self.layer_scores], dtype=float)
            return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else float("nan")
        return self.score.spearmanr

    def best_layer_pearsonr(self) -> float:
        if self.layer_scores:
            vals = np.asarray([s.pearsonr for s in self.layer_scores], dtype=float)
            return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else float("nan")
        return self.score.pearsonr

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            "electrode": self.electrode,
            "subject_uid": self.subject_uid,
            **self.score.to_dict(),
            "best_spearmanr": self.best_layer_spearmanr(),
            "best_pearsonr": self.best_layer_pearsonr(),
            "best_layer": self.best_layer_index("spearmanr"),
        }


@dataclass
class Fedorenko2016Result:
    model_name: str
    model_revision: str | None
    n_words: int
    n_sentences: int
    electrode_scores: list[FedorenkoElectrodeScore]
    sentence_nll: np.ndarray
    sentence_ntoks: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.electrode_scores])

    def by_subject(self) -> pd.DataFrame:
        df = self.to_dataframe()
        cols = ["best_spearmanr", "best_pearsonr", "spearmanr", "pearsonr"]
        out = (
            df.groupby("subject_uid")[cols]
            .agg(["mean", "std", "count"])
            .pipe(lambda x: x.set_axis([f"{c}_{s}" for c, s in x.columns], axis=1))
            .reset_index()
        )
        return out


class Fedorenko2016Encoder:
    """Fit layerwise ridge models for Fedorenko2016 responses."""

    def __init__(self, cfg: Fedorenko2016BenchmarkConfig) -> None:
        self.cfg = cfg

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        ds = Fedorenko2016Dataset(self.cfg.data.root)
        words_df = ds.load_word_table()
        elec_map = ds.load_electrodes()

        if self.cfg.data.electrodes is not None:
            keep_cols = {f"electrode_{int(e)}" for e in self.cfg.data.electrodes}
            always_cols = {"stimulus_id", "stimulus", "word", "word_num", "sentence_id"}
            words_df = words_df[[c for c in words_df.columns if c in keep_cols or c in always_cols]]
            elec_map = elec_map[elec_map["electrode"].isin(self.cfg.data.electrodes)]

        if self.cfg.data.subjects is not None:
            sub_keep = set(int(s) for s in self.cfg.data.subjects)
            keep_elec = set(elec_map.loc[elec_map["subject_UID"].isin(sub_keep), "electrode"].astype(int))
            keep_cols = {f"electrode_{e}" for e in keep_elec}
            always_cols = {"stimulus_id", "stimulus", "word", "word_num", "sentence_id"}
            words_df = words_df[[c for c in words_df.columns if c in keep_cols or c in always_cols]]
            elec_map = elec_map[elec_map["subject_UID"].isin(sub_keep)]

        return words_df, elec_map

    def _build_sentence_tokens(self, words_df: pd.DataFrame) -> tuple[list[list[str]], list[str]]:
        sentence_tokens: list[list[str]] = []
        sentence_texts: list[str] = []
        for _, g in words_df.sort_values(["sentence_id", "word_num"]).groupby("sentence_id"):
            toks = g["word"].astype(str).tolist()
            sentence_tokens.append(toks)
            sentence_texts.append(" ".join(toks))
        return sentence_tokens, sentence_texts

    def extract_features_only(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        words_df, _ = self._load_data()
        sentence_tokens, sentence_texts = self._build_sentence_tokens(words_df)
        emb, sentence_nll, sentence_ntoks = extract_embeddings_and_perplexity_by_sentence(
            sentence_tokens=sentence_tokens,
            sentence_texts=sentence_texts,
            model_name=self.cfg.model.model_name,
            revision=self.cfg.model.model_revision,
            device=self.cfg.model.device,
            batch_size=self.cfg.model.batch_size,
            tokenizer_no_revision=self.cfg.tokenizer_no_revision,
            trust_remote_code=self.cfg.model.trust_remote_code,
            n_params=self.cfg.model.n_params,
            cache_dir=self.cfg.data.feature_cache_dir,
            max_length=self.cfg.data.max_length,
        )
        return emb, sentence_nll, sentence_ntoks, len(words_df), len(sentence_tokens)

    def run(self) -> Fedorenko2016Result:
        words_df, elec_map = self._load_data()
        sentence_tokens, sentence_texts = self._build_sentence_tokens(words_df)
        emb, sentence_nll, sentence_ntoks = extract_embeddings_and_perplexity_by_sentence(
            sentence_tokens=sentence_tokens,
            sentence_texts=sentence_texts,
            model_name=self.cfg.model.model_name,
            revision=self.cfg.model.model_revision,
            device=self.cfg.model.device,
            batch_size=self.cfg.model.batch_size,
            tokenizer_no_revision=self.cfg.tokenizer_no_revision,
            trust_remote_code=self.cfg.model.trust_remote_code,
            n_params=self.cfg.model.n_params,
            cache_dir=self.cfg.data.feature_cache_dir,
            max_length=self.cfg.data.max_length,
        )

        # --- Build full electrode matrix (all subjects stacked as columns) ---
        electrode_cols = [c for c in words_df.columns if c.startswith("electrode_")]
        Y = words_df[electrode_cols].to_numpy(dtype=np.float64)
        if self.cfg.data.drop_constant_electrodes:
            keep_mask = np.nanstd(Y, axis=0) > 1e-8
            Y = Y[:, keep_mask]
            electrode_cols = [c for c, keep in zip(electrode_cols, keep_mask) if keep]
        if self.cfg.data.zscore_targets:
            m = np.nanmean(Y, axis=0)
            s = np.nanstd(Y, axis=0)
            s[s < 1e-8] = 1.0
            Y = (Y - m) / s

        groups = words_df["sentence_id"].astype(int).to_numpy()
        n_splits = min(self.cfg.n_splits, int(np.unique(groups).size))
        n_layers = emb.shape[1]

        # --- One joint regression call over all electrodes (himalaya is
        #     multi-output and treats each electrode independently) ---
        logger.info(
            "Running joint ridge regression: %d electrodes × %d layers (n_splits=%d, backend=%s)",
            Y.shape[1], n_layers, n_splits,
            "gpu" if (self.cfg.regression_device == "cuda" and torch.cuda.is_available()) else "cpu",
        )
        save_word_predictions = self.cfg.save_word_predictions
        if self.cfg.regression_device == "cuda" and torch.cuda.is_available():
            layer_scores, oof_by_layer = self._regress_gpu(
                emb, Y, groups, n_splits, save_word_predictions=save_word_predictions,
            )
        else:
            layer_scores, oof_by_layer = self._regress_cpu(
                emb, Y, groups, n_splits, save_word_predictions=save_word_predictions,
            )

        if save_word_predictions and oof_by_layer is not None:
            self._save_word_predictions(
                words_df, electrode_cols, elec_map, Y, oof_by_layer,
            )

        # --- Assign scores back to electrodes, then log per subject ---
        scores: list[FedorenkoElectrodeScore] = []
        for j, col in enumerate(electrode_cols):
            elec_idx = int(col.split("_")[1])
            sub_rows = elec_map[elec_map["electrode"] == elec_idx]
            subj_uid = int(sub_rows["subject_UID"].iloc[0]) if not sub_rows.empty else -1
            ls = [layer_scores[li][j] for li in range(n_layers)]
            scores.append(
                FedorenkoElectrodeScore(
                    electrode=elec_idx,
                    subject_uid=subj_uid,
                    score=ls[-1],
                    layer_scores=ls if self.cfg.layerwise else None,
                )
            )

        # Log per-subject summary (post-hoc slice — no extra regression)
        for subj_uid in sorted(elec_map["subject_UID"].unique()):
            subj_scores = [s for s in scores if s.subject_uid == int(subj_uid)]
            if not subj_scores:
                continue
            best_spearman = float(np.nanmean([s.best_layer_spearmanr() for s in subj_scores]))
            best_pearson  = float(np.nanmean([s.best_layer_pearsonr()  for s in subj_scores]))
            logger.info(
                "[P%02d] %d electrodes, mean best-layer Spearman r = %.4f, Pearson r = %.4f",
                subj_uid, len(subj_scores), best_spearman, best_pearson,
            )

        return Fedorenko2016Result(
            model_name=self.cfg.model.model_name,
            model_revision=self.cfg.model.model_revision,
            n_words=len(words_df),
            n_sentences=len(sentence_tokens),
            electrode_scores=scores,
            sentence_nll=sentence_nll,
            sentence_ntoks=sentence_ntoks,
        )

    def _regress_cpu(
        self,
        emb: np.ndarray,
        Y: np.ndarray,
        groups: np.ndarray,
        n_splits: int,
        *,
        save_word_predictions: bool = False,
    ) -> tuple[list[list[CVScore]], np.ndarray | None]:
        """Return per-layer multi-output CV scores (sklearn, sequential),
        plus every layer's out-of-fold predictions (n_layers, n_words,
        n_electrodes) when ``save_word_predictions`` — free byproduct of the
        same k-fold CV call, see config field docstring."""
        n_layers = emb.shape[1]
        out: list[list[CVScore]] = []
        oof_by_layer: list[np.ndarray] | None = [] if save_word_predictions else None
        for li in range(n_layers):
            if save_word_predictions:
                scores, oof = cv_ridge_score_multioutput(
                    emb[:, li, :],
                    Y,
                    groups=groups,
                    n_splits=n_splits,
                    alpha=self.cfg.regression.alpha,
                    scale=self.cfg.regression.scale,
                    standardize_target=True,
                    random_state=self.cfg.regression.random_state,
                    return_oof_predictions=True,
                )
                oof_by_layer.append(oof)
            else:
                scores = cv_ridge_score_multioutput(
                    emb[:, li, :],
                    Y,
                    groups=groups,
                    n_splits=n_splits,
                    alpha=self.cfg.regression.alpha,
                    scale=self.cfg.regression.scale,
                    standardize_target=True,
                    random_state=self.cfg.regression.random_state,
                )
            out.append(scores)
            logger.info("  layer %d/%d done", li + 1, n_layers)
        oof_stacked = np.stack(oof_by_layer, axis=0).astype(np.float32) if oof_by_layer is not None else None
        return out, oof_stacked

    def _regress_gpu(
        self,
        emb: np.ndarray,
        Y: np.ndarray,
        groups: np.ndarray,
        n_splits: int,
        *,
        save_word_predictions: bool = False,
    ) -> tuple[list[list[CVScore]], np.ndarray | None]:
        """Return per-layer multi-output CV scores (himalaya, parallel CUDA
        streams), plus every layer's out-of-fold predictions (n_layers,
        n_words, n_electrodes) when ``save_word_predictions`` — free
        byproduct of the same k-fold CV call, see config field docstring."""
        from himalaya.backend import set_backend as _set_backend

        _set_backend("torch_cuda")
        torch.cuda.init()

        # torch.linalg.svd (used inside cv_ridge_score_multioutput_gpu) does a
        # one-time, non-thread-safe CUDA/MAGMA backend registration on its
        # first call in the process — separate from torch.cuda.init() above.
        # If two layer-worker threads race to trigger that first call
        # concurrently, PyTorch raises "RuntimeError: lazy wrapper should be
        # called at most once" and the layer's fit is lost (same fix as
        # podcast.py / broderick.py's eigh-based GPU-batched regression).
        _svd_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
        torch.linalg.svd(_svd_warmup, full_matrices=False)
        del _svd_warmup

        n_layers = emb.shape[1]
        n_workers = min(self.cfg.n_layer_workers, n_layers)
        out: list[list[CVScore] | None] = [None] * n_layers
        oof_out: list[np.ndarray | None] = [None] * n_layers

        def _fit(li: int) -> tuple[int, list[CVScore], np.ndarray | None]:
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                if save_word_predictions:
                    scores, oof = cv_ridge_score_multioutput_gpu(
                        emb[:, li, :],
                        Y,
                        groups=groups,
                        n_splits=n_splits,
                        alpha=self.cfg.regression.alpha,
                        scale=self.cfg.regression.scale,
                        standardize_target=True,
                        random_state=self.cfg.regression.random_state,
                        return_oof_predictions=True,
                    )
                else:
                    scores = cv_ridge_score_multioutput_gpu(
                        emb[:, li, :],
                        Y,
                        groups=groups,
                        n_splits=n_splits,
                        alpha=self.cfg.regression.alpha,
                        scale=self.cfg.regression.scale,
                        standardize_target=True,
                        random_state=self.cfg.regression.random_state,
                    )
                    oof = None
            stream.synchronize()
            return li, scores, oof

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fit, li): li for li in range(n_layers)}
            for fut in as_completed(futures):
                li, scores, oof = fut.result()
                out[li] = scores
                oof_out[li] = oof
                logger.info("  layer %d/%d done", li + 1, n_layers)

        oof_stacked = (
            np.stack(oof_out, axis=0).astype(np.float32) if save_word_predictions else None
        )
        return tp.cast(list[list[CVScore]], out), oof_stacked

    def _save_word_predictions(
        self,
        words_df: pd.DataFrame,
        electrode_cols: list[str],
        elec_map: pd.DataFrame,
        Y: np.ndarray,
        oof_by_layer: np.ndarray,
    ) -> None:
        """Save every layer's out-of-fold predictions + the observed
        (z-scored) target matrix + word/electrode identity to one
        per-checkpoint ``.npz`` (mirrors ``natural_stories_embed_gpu.py``'s
        ``_save_word_predictions`` convention), for post-hoc per-word
        difficulty splits (e.g. DLT cost). One file per checkpoint — unlike
        GECO/Podcast's per-subject loop, Fedorenko2016 fits every subject's
        electrodes jointly in one regression call (see ``run()``), so there's
        no natural per-subject split point."""
        output_dir = self.cfg.output_dir
        if not output_dir:
            logger.warning(
                "save_word_predictions=True but output_dir is not set — skipping word-prediction save",
            )
            return
        out_dir = word_predictions_dir(output_dir, "fedorenko2016_word_predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (self.cfg.model.model_revision or "main").replace("/", "_")
        safe_model = self.cfg.model.model_name.replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}.npz"

        electrode_ids = np.array([int(c.split("_")[1]) for c in electrode_cols])
        subject_uids = np.array([
            int(elec_map.loc[elec_map["electrode"] == e, "subject_UID"].iloc[0])
            if (elec_map["electrode"] == e).any() else -1
            for e in electrode_ids
        ])
        np.savez_compressed(
            out_path,
            oof_predictions=oof_by_layer,              # (n_layers, n_words, n_electrodes)
            y_true=Y.astype(np.float32),                # (n_words, n_electrodes), z-scored
            electrode=electrode_ids,                    # (n_electrodes,)
            subject_uid=subject_uids,                   # (n_electrodes,)
            stimulus_id=words_df["stimulus_id"].to_numpy(),
            word=words_df["word"].astype(str).to_numpy(),
            word_num=words_df["word_num"].to_numpy(),
            sentence_id=words_df["sentence_id"].to_numpy(),
            model=np.array(self.cfg.model.model_name),
            revision=np.array(self.cfg.model.model_revision or "main"),
        )
        logger.info("  Saved word-level predictions (%s) -> %s", oof_by_layer.shape, out_path)
