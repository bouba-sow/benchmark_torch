Deepseek:
# Review of "Human-Aligned Language Models Do Not Better Predict Human Language Processing"

## Summary

This paper presents a large-scale empirical investigation into whether incorporating human-inspired constraints into language models—specifically (1) memory-limited architectures and (2) developmentally plausible training data (BabyLM-scale)—improves their ability to predict human behavioral and neural responses to language. The authors evaluate 118 models across 18 datasets (self-paced reading, eye-tracking, fMRI, EEG/ERP, MEG, ECoG) using two predictor types (hidden-state embeddings via ridge regression and surprisal via mixed-effects models). Using a matched-scale regression that controls for model size and perplexity, they find no evidence that either constraint improves alignment beyond what scale and language-modeling quality already predict. The paper is positioned as a rigorous test of whether human-like inductive biases drive algorithmic convergence.

---

## Strengths

1. **Scale and scope**: The evaluation is impressively comprehensive—118 models across 4 constraint groups, 18 datasets spanning 27 dataset units, and two complementary predictor tracks. This is a substantial empirical contribution that goes well beyond typical single-corpus or single-modality studies.

2. **Methodological rigor in the matched-scale analysis**: The authors correctly recognize that naive group comparisons would confound constraint effects with model size and perplexity. The regression approach that includes log-parameter count and log-perplexity as covariates is the right way to isolate constraint effects. The use of cluster-robust standard errors by model family is also appropriate given the non-independence of models within families.

3. **Transparency about limitations**: The authors are commendably honest about the limitations of their design, particularly the limited power to separate constraint effects from family effects (since each constraint group contains only a few families) and the fact that predictive power is not mechanism.

4. **Comprehensive appendix**: The supplementary materials are thorough, with full model lists, dataset descriptions, LME formulas, noise ceiling procedures, and RSA methods. This supports reproducibility.

5. **Clear writing and organization**: The paper is well-structured, the hypotheses are clearly stated, and the results are presented transparently. The distinction between the embedding and surprisal tracks is maintained consistently.

---

## Major Weaknesses

### 1. The null result is underpowered for the question being asked

The paper's central claim—that human-inspired constraints do not improve alignment—rests on null results from a design with limited statistical power. As the authors acknowledge, each constraint group contains only a few model families. The cluster-robust standard errors, while conservative, mean that the confidence intervals around the constraint coefficients are wide. For example, the memory constraint coefficient for embeddings is +0.07 (SE 0.15), which means the 95% CI spans roughly [-0.22, +0.36]. This is compatible with a moderate positive effect, a moderate negative effect, or no effect. The paper concludes "no evidence that it does" help, but this is an absence of evidence, not evidence of absence. The authors acknowledge this in the limitations, but the abstract and conclusion overstate the certainty of the null finding.

**Recommendation**: The paper should more carefully calibrate its claims. A Bayesian analysis or equivalence testing framework would allow the authors to state what effect sizes can be ruled out, rather than simply failing to reject the null. At minimum, the abstract and conclusion should be tempered to reflect the uncertainty.

### 2. The "alignment" operationalization is purely predictive, not mechanistic

The paper repeatedly acknowledges that predictive power is not mechanism (Section 5: "Predictive power is not mechanism"). Yet the framing throughout—"algorithmic convergence," "algorithmic alignment"—implies a stronger claim about shared computations. The authors test whether model-derived features predict human data, not whether the models and humans use the same algorithms. This is a fundamental conceptual limitation.

The paper would be stronger if it either:
- (a) Reframed the contribution as being about predictive alignment specifically, avoiding language that implies mechanistic convergence, or
- (b) Included at least some mechanistic analyses (e.g., RSA on internal representations, which is mentioned in Appendix H but not integrated into the main results).

As it stands, the paper is caught between two framings: it motivates the work with questions about algorithmic convergence (Marr's levels), but the empirical test is purely predictive. The conclusion that "scale and perplexity are what predict alignment" is about predictive alignment, not algorithmic alignment, and the paper should be clearer about this distinction.

### 3. The one positive result (BabyLM on embeddings) is dismissed too quickly

The data constraint on embeddings at the last layer shows a positive coefficient of +0.55 (SE 0.13), which is substantial and statistically reliable. The authors dismiss this as likely reflecting lexical information (word length/frequency) that the ridge encoders do not control for, and note that it disappears at the best layer and for surprisal. However, this explanation is post hoc and untested. The authors did not run the embedding analysis with lexical covariates to test this directly. The representational similarity analysis (Appendix H) does control for word length and frequency, but it is not a clean test of this specific hypothesis because it uses a different scoring procedure.

**Recommendation**: The authors should either (a) run the embedding ridge regression with word length and frequency as additional predictors to test whether the BabyLM advantage persists, or (b) more clearly flag this as an unresolved finding that warrants follow-up, rather than dismissing it.

### 4. The constraints tested may not be the right ones

The paper tests two specific constraints: memory-limited architectures (various forms) and BabyLM-scale training data. These are certainly the most prominent in the literature, but they may not be the constraints most likely to induce human-like processing. For example:
- **Curriculum learning**: Humans do not learn from randomly shuffled text; they learn from developmentally ordered input. None of the models tested incorporate this.
- **Grounded, interactive input**: Human language learning is grounded in sensorimotor experience and interactive feedback. BabyLM models are trained on text alone, just less of it.
- **Inductive biases beyond memory**: Human language processing involves hierarchical structure building, syntactic transformations, and compositional semantics that may not be captured by attention constraints alone.

The paper's conclusion that "the properties of a language model that matter are scale and quality, not the human-inspired constraints we test" is appropriately scoped, but the abstract and introduction sometimes imply a broader claim. The authors should be more explicit that they test a specific, limited set of constraints and that other forms of human-like inductive bias remain untested.

### 5. The best-layer analysis undermines the primary conclusion in a way that needs more attention

The best-layer analysis (Table 8) shows that the memory constraint becomes *harmful* for embeddings (-0.31, SE 0.14), and the double constraint is clearly worse than standard LLMs (-0.49, SE 0.14). The authors note that this may reflect depth differences (best-layer selection favors deep models), but this explanation is not fully satisfying. If the best layer is where the model's representation is most aligned with human data, then a memory constraint that hurts at the best layer is a meaningful negative result, not just an artifact of depth. The paper should explore this more carefully—e.g., by comparing models at matched depth, or by examining the layerwise profiles for memory-constrained vs. standard models.

### 6. The surprisal track's dependence on perplexity needs more scrutiny

The finding that surprisal alignment is predicted by perplexity (-0.21 per decade) rather than size is interesting, but it raises a question: is this simply because perplexity is a measure of language-modeling quality, and better language models produce better surprisal estimates? If so, this is not a deep finding—it is almost tautological. The paper would benefit from a more careful analysis of *why* perplexity predicts surprisal alignment. For example, does the relationship hold when controlling for training data, architecture, or other factors? Is it driven by a few outlier models? The current analysis is correlational and does not establish a causal role for perplexity.

---

## Minor Weaknesses

1. **The paper does not report effect sizes for individual datasets**: The regression pools all dataset units, which is appropriate for the main analysis, but the reader is left wondering whether the null result holds uniformly across modalities. Are there datasets where constraints do help? A per-dataset breakdown (even in the appendix) would be informative.

2. **The noise ceiling analysis is relegated to the appendix**: The main text mentions noise ceilings only in passing, but the fact that many models are far from ceiling (e.g., 13-77% for embeddings) is important context. If models are far from ceiling, there is room for constraints to help, even if they don't in this particular evaluation.

3. **The "best layer" rule is problematic**: The authors acknowledge that selecting the best layer over a variable number of layers favors deep models. This is a serious methodological issue. A more principled approach would be to use a fixed relative depth (e.g., 50% or 75% of layers) or to use layerwise profiles to identify a consistent pattern across models. The current approach makes the best-layer results difficult to interpret.

4. **The paper does not discuss the possibility of interaction effects between constraints and dataset characteristics**: For example, do memory constraints help more on datasets with long-distance dependencies? Do BabyLM models perform better on datasets with simpler vocabulary? These questions are not addressed, but they are natural follow-ups that could reveal where constraints matter.

5. **The writing occasionally overstates the findings**: For example, the abstract says "we find no evidence that it does" (constraints help), but the conclusion should acknowledge that the study is underpowered to detect small effects. The phrase "the properties of a language model that matter are therefore its scale and its quality as a language model, not the human-inspired constraints we test" is too strong given the limitations.

6. **The paper does not engage with the broader debate about whether predictive alignment is the right target**: There is a growing literature arguing that predictive power is not sufficient for mechanistic alignment (e.g., work on adversarial examples, out-of-distribution generalization). The paper would benefit from engaging with this critique more directly.

---

## Questions for the Authors

1. **Power analysis**: Can you provide a formal power analysis or equivalence test to quantify what effect sizes your design can reliably detect? This would help readers calibrate the strength of the null finding.

2. **Lexical confound**: Did you run the embedding ridge regression with word length and frequency as additional predictors? If so, what happened to the BabyLM advantage? If not, can you do so and report the results?

3. **Best-layer analysis**: Can you compare memory-constrained and standard models at matched depth (e.g., by subsampling layers from deep models or by comparing only models with similar layer counts)? This would help disentangle the depth confound from the constraint effect.

4. **Per-dataset variation**: Are there individual datasets where constraints show a reliable positive effect? Even if the pooled analysis is null, understanding where constraints might help (or hurt) would be informative.

5. **Alternative constraints**: Have you considered testing other human-inspired constraints, such as curriculum learning, interactive training, or grounded input? If not, can you discuss why these were excluded and whether your conclusions might generalize to them?

6. **Mechanistic follow-up**: Given that your RSA analysis (Appendix H) is already implemented, can you integrate it into the main text to provide at least some evidence about representational alignment (as opposed to predictive alignment)?

---

## Overall Assessment

This is a valuable and ambitious paper that makes a significant empirical contribution to the debate about human-aligned language models. The scale of the evaluation is impressive, and the matched-scale regression is a methodologically sound approach to isolating constraint effects. However, the paper's conclusions are stronger than the evidence warrants, given the limited power to detect small effects and the purely predictive operationalization of "alignment." The dismissal of the one positive result (BabyLM on embeddings) is also too quick.

**Recommendation**: Reject with encouragement to resubmit after major revisions.

The paper has the potential to be a strong contribution, but it needs to:
1. Temper its claims to reflect the uncertainty in the null results.
2. Either integrate mechanistic analyses (RSA) into the main text or reframe the contribution as being about predictive alignment specifically.
3. Directly test the lexical confound explanation for the BabyLM embedding result.
4. Provide a more careful analysis of the best-layer results to disentangle depth from constraint effects.
5. Engage more deeply with the limitations of predictive alignment as a target for mechanistic claims.

With these revisions, the paper could make a compelling case that human-inspired constraints (at least those tested) do not improve predictive alignment beyond scale and quality, while being appropriately humble about what this means for algorithmic convergence.

Claude
# Review: "Are Human-Constrained Language Models More Human-Like? Evaluating Algorithmic Convergence Across Neural and Behavioral Measures"

## Summary

The paper evaluates 118 public checkpoints in a 2×2 design (memory-limited architecture × BabyLM-scale data) on 18 behavioral and neural datasets, split into 27 units. It uses two predictors: ridge-decoded hidden states and logit-lens surprisal. A pooled regression on robustly scaled scores, controlling for log parameters and log perplexity, finds no reliable advantage for either constraint. Scale predicts embedding alignment and perplexity predicts surprisal alignment. The authors conclude that these constraints do not help beyond scale.

## Strengths

- The question is worth asking, and a uniform pipeline across this many models and modalities is useful.
- The authors are unusually candid about several limitations: power, layer-rule sensitivity, extrapolation, and the lexical confound in the ridge track.
- Reporting both predictor tracks and both layer rules is good practice.
- Reporting noise ceilings, including their failures (the ‡ rows), is commendable.

## Major weaknesses

**1. The headline inference does not follow from the evidence.**
- "No evidence" is repeatedly rendered as "indistinguishable" and "never helps." With SE 0.22, the 95% CI for the double constraint on embeddings is roughly [−0.44, +0.42] IQR. The size effect is only 0.33 per decade, so the data cannot exclude an advantage worth more than a decade of scale, or a penalty of the same size.
- No equivalence test or region of practical equivalence is reported. "Never helps" (abstract) is not supported by βM = +0.07 (SE 0.15).
- The Limitations section concedes this, but the abstract and title do not.

**2. The study is observational, and constraint is confounded with family, depth and training data.**
- Constrained-BabyLM is 24 models from five families in one release, all with 2–4 layers and 15–48M parameters. Depth and width differ completely from the BabyLM group (13 layers). The memory × data interaction (−0.63) is therefore identified from a group with essentially no overlap in architecture, depth or size with its comparison groups. It is not a clean test of "constraint."
- The group assignments are questionable as operationalizations of "memory constraint":
  - BLOOM is classified as constrained because it uses ALiBi, which is a length-extrapolation device and not a memory bottleneck.
  - MoLM is a sparse mixture-of-experts model with no memory limit at all.
  - Mamba and RWKV, which have a fixed-size recurrent state, are put in the reference group, while CBR-RNN is put in the constrained group. That is 17 bounded-state models contaminating the baseline.
  - FoX at 760M is a mild constraint.
- "Web-scale" training for the Pile 2/4-layer models is not documented. Table 5 has blank token counts for them.
- The paper never tests whether the conclusions survive reclassification.
- The cleanest controlled contrasts in the data are never analyzed directly:
  - FoX against its matched baseline (same size and data).
  - 10M against 100M words against Pile within each Bu et al. architecture, which gives a dose-response curve.
  - 2 against 4 layers.
- The real fix is to train a small set of ablated models yourselves, with the same tokenizer, data, size and seeds. Without that, the title's causal question ("are constrained models more human-like?") is not answered.

**3. Conditioning on perplexity and size is conceptually and statistically fragile.**
- The constraints plausibly act through perplexity and size, so these are mediators. The regression estimates only a direct effect. The unadjusted table (Table 1) shows BabyLM above Standard on embeddings (0.20 vs 0.11) and is relegated to a "confounded" footnote. Both total and direct effects should be shown.
- Perplexity is not comparable across tokenizers, and it is undefined or ill-posed for masked and mixed GPT-BERT variants. How was it computed (per word or per token, and on what text)?
- Size and perplexity are strongly collinear, yet the paper draws sharp conclusions about which one matters in each track. VIFs and single-predictor fits are needed.
- The model assumes linear, common slopes across groups. The literature the paper cites (Oh & Schuler; Kuribayashi) reports non-monotonic perplexity–fit relationships. No interactions, nonlinearity or residual diagnostics are shown.
- The BabyLM groups sit in a region with almost no Standard LLMs (Pythia-14M/31M/70M, GPT-2 small). This is extrapolation. There is no common-support analysis, no matching, and no leave-one-family-out check.

**4. The statistical treatment of dependence is insufficient.**
- Clustering on 22 families, with only a few per constraint group, gives a small effective number of clusters for the constraint coefficients. Plain CR1 standard errors are anti-conservative here, so use CR2 or a wild-cluster bootstrap.
- Dependence across dataset units (shared items and participants within a corpus) is acknowledged but ignored. Two-way clustering, or a hierarchical model with family and dataset random effects, is the standard remedy.
- Rows are 3,117, not 118 × 27 = 3,186. Where do the 69 missing rows come from, and is the missingness related to model size or group?
- The reader is told that model-based standard errors would flag "many more" coefficients, but those values are only in the code repository, not the paper.

**5. Pooling modalities contradicts the paper's own related work.**
- The paper cites de Varda et al. (2026), who find opposite scaling trends for behavior (smaller is better) and brain (larger is better). Yet it estimates a single coefficient pooled over both.
- Results should be split by modality (and ideally by measure type), with per-dataset estimates and leave-one-dataset-out checks. No per-dataset result appears anywhere.
- Robust scaling gives every unit equal weight, including units where the human signal is almost pure noise (Broderick ceiling 0.108, MEG-MASC 0.097). Dividing noise by its IQR amplifies it to unit variance. Reliability weighting or exclusion should be tried.

**6. The embedding track is not interpretable as stated.**
- Ridge regression with d up to 8,192 on datasets with a few hundred items (N400 sets with 192 and 564 items, Pereira) is heavily capacity-confounded. "Scale predicts embedding alignment" may simply reflect dimensionality. The RSA control exists (Appendix H) but its results are never discussed in the main text.
- The ridge encoders receive no length or frequency covariates, while the surprisal track does. The two tracks are therefore not comparable, and the one positive finding (+0.55) is dismissed with an untested lexical conjecture. This is trivially testable: add the covariates, or residualize. Doing so is a minimum requirement.
- Treatment is asymmetric. The single positive result is explained away without testing, while the negative best-layer results are headlined in the abstract ("clearly worse") and then disclaimed in the Discussion ("we draw no conclusion about the size of the memory penalty").

**7. The layer rule drives the conclusions.**
- The coefficient for the memory constraint on embeddings moves from +0.07 to −0.31, and the data constraint from +0.55 to −0.02, depending on the rule.
- Last-layer embeddings are known to be a poor representation for large LMs, because the last layer is specialized for next-token prediction. The primary rule thus handicaps the models at the high end of the scale axis.
- Best-layer selection on held-out score is optimistically biased. It should be done inside the inner CV.
- A relative-depth rule (e.g., a fixed fraction of depth) is the obvious third option and is absent.
- Because group membership is almost perfectly collinear with depth (2–4 layers against up to 65), the best-layer "memory penalty" cannot be separated from depth, as the authors admit. Then the abstract should not assert it.

**8. The "algorithmic convergence" framing is not supported by the experiments.**
- The authors concede that predictive power does not test shared mechanism. Marr's levels are not operationalized anywhere in the analysis (Fig. 1A, 1C are rhetorical), and the claim that behavioral and neural measures "probe the algorithmic level" is a stretch.
- The memory-constraint literature cited makes phenomenon-specific claims (similarity-based interference, locality, agreement attraction). Broad naturalistic-corpus predictive power averages these away. Targeted tests on those phenomena would be a far more meaningful test of whether the constraints do what they were designed to do.
- The title and abstract should be rewritten to say what was done: predictive fit of public checkpoints, not algorithmic convergence.

**9. Figures and results are under-reported.**
- The main text has no results figure. Figures 2–5 (Pareto fronts) are in the appendix and never discussed.
- Some of them cut against the narrative and deserve discussion: constrained or small models sit on the Pareto frontier in Fig. 4 (Forgetting-Pile-2L, Hard2gram-BabyLM) and Fig. 5 (Hard2gram-BabyLM100M-4L, MoEP-80M). MoLM is on the frontier in Fig. 2.
- The frontier plots are also hard to read, with illegible labels.
- A coefficient plot with CIs per track, layer rule and modality is the figure this paper needs.

## Methodological and technical concerns

- Some surprisal ceilings are unusable. For Broderick, 59% of models exceed the ceiling, and for Federmeier 47% do. The ceiling procedure is therefore unreliable, which undermines the "far from ceiling" claim in the Discussion.
- The full ΔLL model includes extra random slopes (surprisal, previous-word surprisal) that the baseline lacks, so ΔLL conflates the predictor's value with added random-effect flexibility.
- The logit lens is poorly calibrated for early layers in several families, which affects best-layer surprisal. The tuned lens was cited but not used.
- Logit-lens surprisal for masked or mixed GPT-BERT variants is not well-defined as next-word surprisal. How was it computed?
- bfloat16 extraction can matter when ΔLL values are around 10⁻³ (see Table 3). Show a float32 check on a subset.
- Each checkpoint is a single seed, so within-architecture variance is unknown.
- "BrainScore" and "BehavScore" collide with the established Brain-Score name and are confusing.
- RSA is described as transductive (z-scoring uses test stimuli), with pair non-independence inflating ΔLL, and the RSA tracks cover only a subset of datasets. It cannot serve as a clean control, as the authors admit.

## Presentation and consistency

- Section 3.2 says 11 standard families with 14M–66B parameters. Appendix B says twelve families "from around 100 million".
- The text says "Mistral"; Table 5 lists Ministral. Llama-3.1-8B appears under "Llama-3.2".
- The Table 7 caption says OneStop has no embedding/ridge track, but OneStop appears in Table 7 and Table 4.
- The reference list contains leftover drafting notes ("Listed as … in the CogPlausAI review… Confirm this is the intended reference", "cite the arXiv preprint until a venue is confirmed", "no archival identifier available"). The "Author Contributions" and "Acknowledgments" template text is also still present. These look unprofessional and must go.
- Section 3 mixes methods with discussion in places.
- The benchmark is claimed as a contribution, but nothing about its specification, licensing, splits or interface is given.

## Questions for the authors

1. What are the FoX vs FoX-baseline contrasts and the 10M / 100M / Pile dose-response within architecture, as direct paired comparisons?
2. Do the results hold if BLOOM and MoLM are moved out of "constrained," and Mamba, RWKV and CBR-RNN are grouped together as bounded-state models?
3. What happens to the +0.55 data effect once word length and frequency are added to the ridge encoders?
4. What do the results look like separately for behavioral and neural units, and per dataset?
5. What do the RSA results say about each main finding?
6. How does the analysis change with CR2 or a bootstrap, two-way clustering, and a depth covariate?

## Assessment

**Soundness:** 2/4. The data collection is substantial, but the central inference rests on a confounded observational design with limited power, an overstated null, and an untested explanation for the one positive result.
**Significance:** moderate. A well-controlled version of this study would matter to the cognitive-modeling community.
**Novelty:** limited. The paper largely reanalyses public checkpoints (including Bu et al.'s) with an aggregate regression, and the headline "scale matters" finding is already known.
**Clarity:** 3/4 for the writing, but the main text lacks results visualization.

**Recommended score: 4 (marginally below the acceptance threshold). Confidence: 4/5.**

This could move to a 6 if the authors:
1. Add matched-pair and dose-response analyses.
2. Fix the covariate asymmetry between the two tracks.
3. Split results by modality and dataset.
4. Test sensitivity to group classification.
5. Replace "no evidence/never helps/algorithmic convergence" with calibrated claims backed by CIs or equivalence bounds.
6. Ideally train a small set of controlled models.
