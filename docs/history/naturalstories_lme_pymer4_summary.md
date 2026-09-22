# Natural Stories LME (pymer4) - Short Summary

## Goal

Evaluate whether model surprisal predicts trial-level self-paced reading times in Natural Stories, using mixed-effects models aligned with Clark et al. (COLING 2025).

Main files:

- `neuroai/benchmarks/natural_stories_lme.py`
- `neuroai/regressors/lme.py`
- `neuroai/datasets/natural_stories.py`
- `scripts/run_naturalstories_lme_sweep.py`

## Pipeline

1. Load trial RTs and apply standard SPR filtering (`min_correct`, RT bounds).
2. Build sentence metadata (`sentid`, `sentpos`) from `all_stories.tok`.
3. Optionally drop sentence-initial and -final words (Clark preprocessing).
4. Attach KenLM lexical surprisal per word (cached; see below).
5. Compute word-level LM surprisal (+ spillover) per checkpoint.
6. Fit baseline vs full LME on **train**; evaluate on **test**.

### KenLM nuisance (Clark-style baseline)

N-grams are **not** recomputed — they live in `kenlm/wikipedia/en.arpa.bin`. We score each word and cache surprisal:

```bash
.venv/bin/python scripts/compute_naturalstories_kenlm_surprisal.py \
  --kenlm-repo kenlm \
  --out .cache/naturalstories_features/kenlm_wikipedia_en_surprisal.csv
```

Uses `KenlmModel.from_repo_root("kenlm", "wikipedia", "en")` (same normalize + SentencePiece as edugp). **Jean Zay:** weights are Git LFS — run once on login node:

```bash
cd $WORK/benchmark/kenlm && git lfs install && git lfs pull
# check: file should be ~hundreds of MB, not a 130-byte pointer
ls -lh wikipedia/en.arpa.bin wikipedia/en.sp.model
```

Without cache/repo, LME falls back to corpus word-count surprisal.

## Model (default: `style: clark2025`)

Dependent variable: **raw** `fdur` (reading time in ms).

**Baseline**

- `fdur ~ sentpos_z + wlen_z + unigram_z + (1 + sentpos_z + wlen_z | subject)`

**Full**

- Adds `surprisal_z` and `prev_surprisal_z` (current + spillover LM surprisal)
- `(1 + sentpos_z + wlen_z + surprisal_z + prev_surprisal_z | subject)`

Subject-only random effects (no `word` or `sentid:subject`).

## Data split (train / test)

- `split_id = factor(subject) + factor(sentid)` mod 4
- **Train:** `{0, 1}` (~50%)
- **Test:** `{2, 3}` (~50%)

## Reported metrics per checkpoint

- `beta`, `prev_beta`, `t`, `p` — from **train** fit
- `delta_loglik_train`, `lrt_p` — LRT on train
- `delta_loglik_test` — ΔLogLik on test (primary OOS metric; also in `delta_loglik`)
- `test_pearsonr_full_model`, `test_spearmanr_full_model` — predicted vs observed RT on **test** (full LME)
- `test_pearsonr_baseline_lme`, `test_spearmanr_baseline_lme` — same on **test** (baseline LME, no LM surprisal)
- `test_delta_pearsonr`, `test_delta_spearmanr` — full minus baseline
- `n_obs` (train), `n_obs_test`

## Legacy ModelBlocks mode

Set `lme.style: modelblocks` for log-RT + `(1 | word) + (1 | sentid:subject)`.

## One-line takeaway

Train/test LME with Clark-style fixed effects and subject random effects: inferential stats on train, OOS ΔLogLik and Spearman on test.
