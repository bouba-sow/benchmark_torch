# Text-only corrections for the short (38-page) Overleaf version

These corrections need no new analysis. Each fixes a factual error, an overclaim, or a presentation problem
the reviewer flagged in the 38-page draft. The analyses added later (robust inference, step B) are listed at the
end only as material for rebuttal answers; they are not needed for these fixes.

Paths refer to the files in `paperwriting/sections/` of the 38-page version.

## A. Factual errors (fix these first)

1. **`results.tex`, last paragraph (prediction–RSA agreement).** Delete the sentence
   "Agreement is thus consistently high for neural data but variable for behavior." It contradicts Table 11.
   Replace with:
   > Surprisal comparisons vary in both domains. Among neural datasets, correlations are about .75 for Nieuwland
   > and Federmeier 2007, but close to zero or negative for Pereira (.019), MEG-MASC (.138) and Broderick (−.119).
   > Agreement between direct prediction and RSA is thus high only for embeddings on most neural datasets, and
   > even there it is modest for MEG-MASC (.275); for behavior, and for surprisal in both domains, it varies in sign.

2. **`results.tex`, RDM-regression paragraph ("Controlling for these lexical distances…").** Say what the axes show
   (values from `reports/rsa_summary/rdm_regression/rdm_pareto_scores.csv`):
   > For embeddings, the gain on the behavioral axis is small for every model (BehavScores between 0 and .005),
   > whereas most models improve neural prediction beyond the lexical RDMs (BrainScores from −.014 to .133,
   > positive for 102 of 118 models). … For surprisal the situation is reversed. Behavioral gains are larger
   > (.012 to .166), and the frontier consists entirely of BabyLM and constrained models; BabyLM models have the
   > highest median behavioral score (.147 vs .103 for Standard LLMs). On the neural axis, only 11 of 118 models
   > have a positive BrainScore. StickBreaking-Pile-2L (.042) and Forgetting-Pile-2L (.025) lead and are among
   > these few, but Standard LLMs have the least negative median (−.049) and are absent from the frontier only
   > because of their lower behavioral scores. The neural axis thus shows a ranking among models that mostly fail
   > to improve on the lexical baseline, not good neural prediction.

   Add to the Figure 4 caption: "most neural surprisal scores are negative."

3. **`discussion.tex`.** Replace "memory-constrained models perform competitively on the neural surprisal
   RDM-regression frontier" with "memory-constrained models occupy the neural end of the surprisal RDM-regression
   frontier …, although neural scores on that frontier are close to or below the lexical baseline, so this reflects
   a ranking rather than good absolute prediction."

4. **`discussion.tex`, first and last lines.** Remove the ```` ```latex ```` and ```` ``` ```` fences. LaTeX prints
   them as text, and the reviewer saw them.

## B. Overclaims in the abstract and conclusion

5. **Abstract.** Remove or soften:
   - "data and memory constraints **generally** improve behavioral alignment". In Table 2, behavioral surprisal is
     .006 (n.s.) for memory only and .000 for both constraints. Say instead that the data budget is associated
     with better behavioral prediction in both tracks, and memory constraints only with behavioral embeddings.
   - "human-like behavioral predictions do not necessarily imply human-like neural **representations**". The design
     does not test representations. Say "…need not improve the prediction of brain activity".
   - "competitiveness of small and constrained models": for the RDM regression, qualify it with point 2 above.

6. **Statistical hedging (if you keep the original Table 2).** The reported standard errors treat rows as
   independent apart from a dataset intercept. State in the text that they are likely optimistic. Do not put the
   neural memory-only surprisal effect (p = .048) in the abstract. Treat the neural surprisal coefficients as
   tentative: they depend on Federmeier 2007, Hubbard 2019, Broderick and Pereira, whose references are flagged.

## C. Facts the reviewer asked for (one sentence each)

7. **Three Standard LLMs dropped from the regressions (118 → 115).** They are the three OLMo-2 models
   (OLMo-2-0425-1B, OLMo-2-1124-7B and OLMo-2-1124-13B), excluded because their perplexities were not computed.

8. **Surprisal for GPT-BERT models.** These models combine causal and masked objectives. They are evaluated through
   their causal language-modeling head (`AutoModelForCausalLM`, native final-layer logits), so their surprisal is
   a left-to-right next-token prediction for all three variants.

9. **One benchmark name** throughout (abstract, introduction, conclusion). The draft mixes "HumanScore-Language"
   and "LLM-Human Convergence Benchmark".

10. **Discuss Bu et al. (2026)** in the main text (Related Work, memory constraints), since the
    memory-constrained checkpoints come from that work.

## D. Presentation leftovers

11. **`iclr2027_conference.bib`.** Delete the drafting `note` fields that mention "CogPlausAI review" or
    "Confirm this is the intended reference" (thamma2025humanlike, xufutrell2026strategic, rasmussen2017leftcorner,
    rambaud2025mapformer). Shorten the vanbree2025noiseceiling note to "Preprint".

12. **`appendix_rsa.tex`, "Scaling and agreement" subsection.** Remove the pipeline wording: "saved raw RSA
    summaries", "historical group assignments", "108 models", "Pareto input pipeline", and "retain their saved
    response and pair constructions". Also remove "saved" from the captions of the ceiling and score-range tables.

13. **AlKhamissi et al. (2025, EMNLP).** The citations were checked against the paper. Related Work:
    > Tracking eight Pythia models (14M to 6.7B parameters) across 34 training checkpoints on five neural
    > benchmarks, AlKhamissi et al. (2025) find that alignment with the human language network rises sharply early
    > in training, peaks after roughly 8B tokens and then saturates, following formal linguistic competence more
    > closely than functional competence. They also report that, once controlling for the number of features,
    > model size is not a reliable predictor of brain alignment, and that the correlation of brain alignment with
    > next-word prediction and with behavioral alignment fades once models surpass human language proficiency.

    Discussion (after "consistent with scaling accounts…"): use their size result as a caution, because our ridge
    encoders have more features for larger models. Use their late-training negative brain–behavior correlation as
    a parallel to our opposite size associations.

## Not in the short version: rebuttal material

These analyses were done and are kept here to answer reviewer questions:
- `scripts/analysis/robust_inference.R` gives lineage, model and corpus random effects, the joint behavioral ×
  neural model, the lineage bootstrap, and the leave-one-unit-out and exclusion refits.
- `scripts/analysis/robust_inference_stepB.R` gives the within-architecture data dose-response, the FoX matched
  pairs, and the per-mechanism contrasts.
- Run both with `sbatch scripts/slurm/jz/utils/robust_inference.sbatch`. Output goes to `reports/robust_inference/`,
  and page 5 of the matched-scale reports is built from it.

Key answers:

| Question | Answer |
|---|---|
| Embedding behavioral data effect | Robust to all checks (+.033, p = .008, ≈1 between-model SD). |
| Embedding neural memory penalty | Robust (−.031, p = .002). |
| Is the behavioral–neural difference tested? | Yes, by the joint model (p ≤ .007). |
| Embedding behavioral memory advantage | Not robust (p = .12). |
| Surprisal constraint effects | Fragile; the neural ones vanish without the flagged units. |
| Data dose-response | Within memory architectures, less data does not improve behavior. |
| Where is the neural penalty? | Forget gate and distance bias. |
