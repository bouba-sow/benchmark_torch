Results and claims draft — 14 October 2026

The strongest organizing claim is: **alignment with human language processing depends jointly on the response being predicted, the model feature used, and the evaluation method. Behavioral and neural alignment do not give interchangeable assessments of models.** The current results support specific advantages for constrained models, alongside neural advantages for larger models; they do not support the manuscript's blanket conclusion that constraints never help.

This is an evidence memo for rebuilding the paper, not a rewritten manuscript. Only final-layer results and ordinary, non-z-scored outcomes inform the claims below. “Ordinary” Pareto scores are still noise-ceiling normalized. Embedding feature standardization before RSA is a different operation from z-scoring the outcome across models.

**Evidence and scope**

The primary regression evidence is model 3 (page 3) of [embedding](../reports/matched_scale_regression_report_embedding.pdf) and [surprisal](../reports/matched_scale_regression_report_surprisal.pdf). PDF values were checked against the accompanying TeX and coefficient CSVs in `outputs_jeanzay/matched_scale_report/`. These are separate behavioral and neural REML fits with memory, data, memory × data, log10 parameters, log10 perplexity, and a dataset random intercept.

The plots examined are the two ordinary last-layer predictive Pareto fronts in `reports/paper/`, both fronts in `reports/rsa_summary/rsa_pareto/`, and the eight non-z-scored figures in `reports/rsa_summary/figures/`. The latter include RDM-regression Pareto and scaling plots as well as Pearson RSA scaling and benchmark-comparison plots. The paper draft in `paperwriting/sections/` supplies context, not evidence.

| Analysis | What the score measures | Aggregation / coverage |
| --- | --- | --- |
| Direct embedding prediction | Encoding Pearson r / correlation ceiling | Predictive Pareto uses the median across dataset units within each axis; 118 plotted models |
| Direct surprisal prediction | Held-out response ΔLL / ΔLL ceiling | Same median construction; 118 plotted models |
| Model-3 regressions | The corresponding normalized direct-prediction score | 115 models; 1,822 behavioral rows across 16 measures and 1,257 neural rows across 11 datasets |
| Pearson RSA Pareto | Model–response RDM correlation / RSA ceiling | 118 models, all with 8 neural datasets and 16 behavioral measures; neural unit ratios summarized within datasets before aggregation |
| RDM-regression Pareto in `figures/` | Held-out RDM ΔLL beyond word-length and frequency RDMs / split-half ΔLL reference | 118 models with 8 neural datasets and 16 behavioral measures |
| RSA panel figures | Raw RSA, or correlations of RSA with direct benchmark scores / log size | 118 models where available; older group assignments |
| RDM ΔLL scaling panels | Raw held-out ΔLL per pair beyond lexical confounds | 108 models per panel, Pythia excluded; some extreme points outside the plotted view |

These metrics have different meanings. Neither a Pearson-r ratio nor a ΔLL ratio is a fraction of explained variance. Values should not be compared numerically across tracks as if the units were identical. The RDM figures also have narrower neural coverage than the direct benchmark.

**1. Behavioral and neural outcomes require separate conclusions**

Paper-ready claim: “The associations of cognitive and developmental constraints with alignment differed between behavioral and neural responses. Pooling these responses obscured advantages confined to behavior and disadvantages observed for neural prediction.”

Model 3 gives the following coefficients. Parentheses contain SEs. Stars reproduce the reports' nominal tests: * p < .05, ** p < .01, *** p < .001. All coefficients are in the corresponding noise-ceiling-normalized outcome units, not IQR units.

| Term | Embedding: behavioral | Embedding: neural | Surprisal: behavioral | Surprisal: neural |
| --- | ---: | ---: | ---: | ---: |
| Memory only vs standard LLM | +.016 (.002)*** | −.030 (.007)*** | +.006 (.005) | −.161 (.082)* |
| Data only vs standard LLM | +.039 (.003)*** | −.004 (.009) | +.032 (.007)*** | −.339 (.104)** |
| Memory × data | −.043 (.004)*** | +.006 (.011) | −.038 (.008)*** | +.292 (.125)* |
| Both constraints vs standard LLM | +.013 (.003)*** | −.028 (.008)*** | +.000 (.006) | −.208 (.099)* |
| Log10 parameters | +.006 (.001)*** | +.039 (.004)*** | −.023 (.003)*** | +.391 (.048)*** |
| Log10 perplexity | −.003 (.001)* | −.004 (.004) | −.039 (.003)*** | +.069 (.041) |

“Memory only” and “data only” are conditional effects with the other constraint absent. The combined contrast is βM + βD + βMD. Rounding the displayed components need not reproduce the displayed sum exactly.

The point estimates demonstrate opposing patterns. A formal statement that two modality coefficients differ significantly would require testing their difference; significance in one fit and nonsignificance in another does not itself test that difference.

**2. Data-constrained models have a behavioral advantage in both predictor tracks**

Paper-ready claim: “Conditional on parameter count and perplexity, BabyLM models without a memory constraint showed higher behavioral prediction scores in both the embedding and surprisal analyses.”

The contrasts are +.039 for embeddings and +.032 for surprisal, both p < .001 in model 3. This directly contradicts the draft's claim that the data-budget advantage is absent for surprisal. It is also more specific than “BabyLM models are more human-like”: the result concerns behavioral prediction, this model sample, and this adjustment specification.

There is no corresponding positive neural contrast: the embedding estimate is near zero (−.004, p = .687), while the surprisal estimate is negative (−.339, p = .00112). Do not generalize the behavioral advantage to all human responses.

**3. Memory constraints have a selective behavioral advantage and neural disadvantages**

Paper-ready claim: “Memory-constrained models showed an advantage for behavioral prediction from final-layer embeddings, but lower neural prediction scores in both feature tracks.”

The embedding behavioral contrast is +.016; the neural contrast is −.030. Behavioral surprisal provides no clear memory-only advantage (+.006, p = .247). The neural surprisal contrast is −.161, but its nominal p = .0484 is borderline and should not carry the main argument. The larger, clearer inference is the absence of a universal memory-constraint advantage, together with the positive behavioral embedding result.

“Memory limits never help” is contradicted by these results. “Memory limits make models more human-like” is also too broad.

**4. The behavioral advantages of the two constraints do not add together**

Paper-ready claim: “The combination of memory and data constraints yielded less behavioral benefit than an additive combination of their separate associations.”

The behavioral interactions are negative in both tracks (−.043 embedding; −.038 surprisal; both p < .001). The combined group retains a positive embedding contrast (+.013) but has essentially zero surprisal contrast (+.000). Thus, the combined embedding benefit is reduced, not eliminated.

For neural surprisal, the interaction is positive (+.292), but the combined contrast remains negative (−.208). A positive interaction means that the combined association is less negative than the sum of the two isolated associations; it does not mean that the combined group outperforms standard LLMs. Neural embedding provides no clear interaction evidence.

The estimates put the combined behavioral group below data-only BabyLM models. A formal test of that difference requires the βM + βMD contrast and its covariance-based SE; the interaction test alone is not that comparison.

**5. Size and language-modeling quality have distinct, outcome-dependent associations**

Paper-ready claim: “Larger models had higher neural prediction scores in both tracks. Behavioral prediction showed a different pattern: parameter count was positively associated with embedding-based prediction but negatively associated with surprisal-based prediction after adjustment for perplexity.”

A tenfold increase in parameters corresponds to +.0385 neural embedding units and +.3910 neural surprisal units, versus +.0064 behavioral embedding units and −.0235 behavioral surprisal units. These are conditional associations, not experimental effects of increasing a model's size.

Lower perplexity is associated with better behavioral surprisal prediction (β = −.0390 per decade of perplexity). Its association with behavioral embedding prediction is much smaller (−.0027, p = .0258). Neither neural perplexity coefficient reaches p < .05; the positive neural surprisal estimate is not evidence that worse language modeling improves neural alignment.

Replace “embeddings follow scale, surprisal follows language-modeling quality” with this modality-specific account. In particular, size remains a substantial positive predictor of neural surprisal alignment in the requested analysis.

**6. Direct-prediction Pareto fronts favor standard LLMs, but contain meaningful exceptions**

The fronts were independently reconstructed from the current benchmark summaries using the plotting pipeline and matched the labels in the requested PDFs. They maximize both median neural and median behavioral scores.

| Track | Frontier models in increasing neural-score order |
| --- | --- |
| Embeddings | OPT-350M; BLM-GPT2s-90M; OPT-13B; OLMo2-13B; BLOOM-3B; OPT-30B; OPT-66B |
| Surprisal | BLOOM-1.1B; OPT-350M; OPT-1.3B; Ministral-3-3B; Ministral-3-14B |

The registry classifies BLOOM as Memory-Constrained. Therefore the embedding front includes five standard LLMs, one BabyLM, and one memory-constrained model; the surprisal front includes four standard LLMs and one memory-constrained model. Neither direct front contains a Constrained-BabyLM model.

The endpoints illustrate the competing objectives. Embedding OPT-350M scores (.2189 neural, .5632 behavioral), whereas OPT-66B scores (.2827, .5125). Surprisal BLOOM-1.1B scores (.1019, .2108), whereas Ministral-3-14B scores (.3423, .1167). No single evaluated model maximizes both axes in either track.

Paper-ready claim: “Standard LLMs occupied most of the direct-prediction Pareto frontier, with smaller models attaining the highest behavioral composites and larger models attaining the highest neural composites.”

Front membership is a descriptive property of estimated composite scores, not a significance test, a causal trade-off, or proof of computational efficiency. Parameter count is displayed, not optimized as a third objective. Adjusted group advantages and unadjusted frontier membership answer different questions and can coexist.

**7. Representational evaluation changes which models are competitive**

Paper-ready claim: “The models favored by direct response prediction differed from those favored by representational comparisons. Small data-constrained and memory-constrained models occupied substantial portions of the representational Pareto fronts.”

For ordinary Pearson RSA, the embedding front runs from Hard2gram-BabyLM100M-4L through MoEP-80M and several GPTBERT models to RWKV4-3B and Qwen3-8B/14B. Hard2gram-BabyLM100M-4L has the highest behavioral composite (.0697); Qwen3-14B has the highest neural composite (.0654). The Qwen3-8B neural score is nearly identical (.0652), with a higher behavioral score (.0317 versus .0151), so do not overinterpret the final frontier step.

The surprisal RSA front contains GPTBERT-10M-Mixed, GPTBERT-10M-CausalFocus, Pythia-1.4B-deduped, BLOOM-7.1B, Mamba2-780M, and RWKV4-14B. GPTBERT-10M-Mixed leads behavior (.3769); RWKV4-14B leads the neural composite (.0881).

In the confound-controlled RDM-regression figures, the embedding front includes all four groups: Hard2gram variants and MoEP at its behavioral end, GPTBERT models in between, and Pythia-12B-deduped/GPT-NeoX-20B at its neural end. Hard2gram-BabyLM100M-4L leads behavior (.0050) but has a slightly negative neural composite (−.000071).

The RDM-regression surprisal front contains no standard LLM: GPTBERT-10M-MaskedFocus, GPTBERT-10M-Mixed, GPTBERT-10M-CausalFocus, Forgetting-BabyLM10M-4L, Forgetting-Pile-2L, and StickBreaking-Pile-2L. The behavioral leader, GPTBERT-10M-MaskedFocus, has a negative neural composite (−.1993); StickBreaking-Pile-2L has the highest neural composite (.0424). Thus, frontier membership does not imply positive incremental neural fit.

This supports evaluation-dependent rankings. It does not establish that RSA is the uniquely valid definition of human-likeness, or that the ranking change is caused by lexical adjustment alone: scoring, response construction, ceiling estimation, and neural coverage also differ. Plain Pearson RSA is not itself adjusted for lexical confounds; the RDM-regression track is.

**8. Raw RSA and RDM results reject a universal “larger is better” account**

Paper-ready claim: “Scaling trends varied across datasets and across neural and behavioral responses, with particularly frequent negative size associations for behavioral embedding RSA.”

Across all models in the saved RSA summary tables, embedding RSA correlates positively with log size in seven of eight neural datasets, with MEG-MASC the exception (r = −.558). Fourteen of sixteen behavioral embedding measures have negative correlations; the two positive values, OneStop GP (.153) and Natural Stories SPR (.101), are small. Examples of negative behavioral associations include MECO FP (−.597) and ZuCo TSR (−.643).

Surprisal RSA is more heterogeneous: size associations are strongly positive for the three N400 datasets (r = .729–.795), negative for several eye-tracking measures such as GECO GP (−.669) and OneStop GP (−.813), and positive for UCL SPR (.507) and ZuCo SR/TSR (.458/.545).

The confound-controlled RDM ΔLL panels likewise show many declining behavioral size trends, particularly for surprisal in first-pass and go-past measures, but positive trends for some other measures such as ZuCo SR/TSR. Neural panels differ substantially; for example, surprisal Podcast declines while Nieuwland rises. These descriptive trends are not matched-size comparisons or established scaling laws.

**9. Predictive fit and representational similarity agree selectively**

Paper-ready claim: “Encoding performance and representational similarity were related, but their correspondence depended on the dataset and was less consistent for behavioral responses.”

Across-model correlations between embedding encoding performance and embedding RSA are positive for all eight available neural datasets, from .275 in MEG-MASC to .929 in Federmeier 2007. Behavioral embedding comparisons include positive Provo correlations (.649–.678), near-zero MECO/GECO relationships, and negative values for GECO GP (−.359), OneStop FP/SP (−.342/−.412), and ZuCo NR/SR (−.325/−.219).

For surprisal, RSA–direct-ΔLL correlations are strong for Nieuwland and Federmeier 2007 (both about .75), but near zero for Pereira (.019) and negative for Broderick (−.119). Behavioral examples range from negative GECO GP (−.304) to positive OneStop SP (.751), UCL SPR (.715), and ZuCo TSR (.694).

These numbers are pooled descriptive correlations from `rsa/summary_correlation.csv` and `rsa/summary_dll.csv`; they are not the category-specific correlations printed on the figures. Correlations between model scores can reflect shared family and size structure. They do not establish shared algorithms.

**Claims to remove or replace in the current manuscript**

| Current claim | Evidence-based replacement |
| --- | --- |
| No constraint improves alignment | Benefits depend on modality, predictor, and evaluation; behavioral advantages occur in model 3 |
| Memory limits never help | Positive behavioral embedding contrast; neural disadvantages; no clear behavioral surprisal advantage |
| The data-budget advantage is absent for surprisal | It is present for behavioral surprisal, not neural surprisal |
| Both constraints are indistinguishable from standard LLMs | Positive behavioral embedding contrast, near-zero behavioral surprisal contrast, negative neural contrasts |
| Combining constraints eliminates the embedding benefit | The combined behavioral embedding benefit remains positive but is smaller |
| Embeddings follow scale; surprisal follows perplexity | Size and perplexity associations depend on modality as well as predictor |
| All reported scores are IQR-standardized | Requested results use noise-ceiling-normalized outcomes or raw RSA/ΔLL, depending on the figure |
| SEs are clustered by model family | Requested model 3 uses conventional mixed-model SEs with a dataset random intercept |
| Every analysis contains 118 models | Pareto plots contain 118; model 3 contains 115; RDM scaling panels contain 108 |
| RSA controls word length and frequency | Plain RSA does not; held-out RDM regression evaluates improvement beyond those confounds |
| Best-layer results establish robustness | Outside the present final-layer-only evidence scope |

**Limits on how strongly to state the claims**

The regression findings above accurately describe model 3, but its nominal p-values do not account explicitly for model-family clustering or the shared participants/items of multiple reading measures. A dataset random intercept does not supply those missing dependence terms. Phrase the findings as associations in this specification; stronger inferential language needs an appropriate dependence-aware analysis. The constraint groups also differ in architectures, training, and size ranges, so adjustment is not experimental matching and can require extrapolation.

Neural surprisal coefficients are especially sensitive in principle to the scale introduced by the ceiling denominators. The saved source table ranges from about −3.44 to 6.98 for Federmeier 2007, versus about −.013 to .0069 for MEG-MASC. The model-3 report includes all 27 units. These facts justify caution about a general neural surprisal effect; this memo has not established that any particular dataset causes the estimated effect. No z-scored alternative was used to draw the conclusions here.

The RSA scaling and benchmark-comparison panels use 41 standard LLMs and 42 memory-constrained models, whereas the current Pareto registry places Mamba/Mamba2/RWKV4 among standard LLMs. Do not combine their category-specific slopes with current group-regression claims until classification is synchronized. Pooled correlations cited above avoid relying on those group labels, but the panel data and current RSA Pareto are still distinct analysis versions with different neural aggregation.

The RSA loader's `best_layer_rsa_pearson` column name initially appears inconsistent with the final-layer restriction for Provo and MECO. Inspection of their v2 raw inputs found one saved layer in each of 354 files per corpus, and the summary's selected layer matched the maximum saved layer throughout. Thus this name alone is not evidence of best-layer selection in those inputs. Layer numbering differs between model implementations; comparing it blindly with architecture depth can produce false mismatches.

The auxiliary `surprisal_lastlayer_pareto_no_pereira.pdf` has different behavioral labels/scale from the main surprisal PDF even though omitting a neural dataset should leave the common models' behavioral scores unchanged. It cannot be treated as a clean current leave-Pereira-out comparison without regeneration. The hard-DLT fronts use a different subset and only Podcast on the neural axis; they are not evidence for an all-neural-dataset conclusion, and their layer provenance was not established here. They are excluded from the central claims.

Exact frontier membership has no uncertainty assessment here. The current broad claims can be drafted now; publication-level assertions about robust group superiority, universal scaling, or causal cognitive mechanisms would go beyond these results.

**Proposed sequence for the paper**

1. Introduce behavioral and neural alignment as distinct targets and prediction/RSA/RDM regression as distinct evaluations.
2. Present the direct-prediction fronts to show the observed distribution of model performance and absence of one joint winner.
3. Present model 3 separately by modality: behavioral constraint advantages, neural penalties or null contrasts, and nonadditivity.
4. Explain the modality-specific size and perplexity associations.
5. Present representational fronts and panel results to show which conclusions generalize across evaluations and which change.
6. Discuss candidate mechanisms as hypotheses, with dependence, ceiling normalization, dataset coverage, and observational model comparisons as explicit limits.

A possible central paragraph:

“Across final-layer evaluations, alignment with human language processing depended on both the response domain and the evaluation method. In regressions adjusting for parameter count and perplexity, data-constrained models showed higher behavioral prediction scores in both feature tracks, while memory constraints were associated with higher behavioral embedding scores but lower neural scores. Larger models had higher neural prediction scores, whereas behavioral surprisal prediction declined with size conditional on perplexity. Model rankings also changed between direct prediction and representational evaluation, where small and constrained models occupied substantial portions of the Pareto fronts. These findings favor a multidimensional account of alignment over a single ranking of models by human-likeness.”

The paragraph summarizes the requested analyses; it should retain the regression and descriptive qualifications in the surrounding results and discussion.
