## Technical Consistency Review

PoxCSAF-Net: Cross-Scale Attention Fusion for Privacy-Aware Federated Classification of Mpox and Visually Similar Skin Lesions

Source manuscript: PoxCSAF-Net_Frnt.pdf, 32 pages

Review date: 22 October 2026

Review basis: Manuscript text, equations, figures, tables, and reported statistics. No training code, dataset manifests, or experimental logs were supplied.

## Executive assessment

The main reported classification metrics are broadly arithmetically consistent. The strongest concerns involve privacy accounting, held-out-derived training weights, the meaning of external validation, the justification of a probability correction, and the interpretation of the proposed cross-scale head. Other issues are confirmed terminology or figure inconsistencies. Findings below are classified as confirmed manuscript discrepancies, methodological interpretation concerns, or items requiring implementation-level verification.

## Complete finding register

| ID | Severity | Finding | Status |
| --- | --- | --- | --- |
| 01 | High | Held-out labels used for focal-loss weights | Methodological concern |
| 02 | High | Classification-head dimensions disagree with Figure 2 | Confirmed figure inconsistency |
| 03 | Critical | DP-SGD privacy unit with offline augmented copies | Requires implementation verification |
| 04 | High | External evaluation is restricted to three classes | Confirmed protocol distinction |
| 05 | High | Prior-shift correction derived from focal weights | Mathematical justification missing |
| 06 | High | FedAvg vs FedProx described as aggregation strategies | Confirmed terminology error |
| 07 | Moderate-high | FedPer tested on classes absent from local training | Evaluation interpretation concern |
| 08 |   | Moderate-high Cross-dataset independence and simulated clients | Claims and deduplication need clarification |
| 09 | Moderate | Repeated configurations have differing results | Reproducibility check needed |
| 10 | High | Ablation does not establish CSAH performance advantage | Scientific interpretation concern |

## Detailed technical findings

## 01. Focal-loss weights derived from a held-out split

High | Methodological concern | Manuscript location: Section 3.5.6, page 11, Equations (17)-(18), Table 2

What the manuscript says. The paper fixes the class-weight vector across folds using representative held-out class counts n_c = {42, 94, 35, 116} for Chickenpox, Healthy, Measles, and Monkeypox. It states that this avoids fold-specific leakage because the vector is not recalculated from each fold's held-out labels.

Why this matters. Using held-out labels to set the training objective can introduce information from evaluation partitions into model development. A fixed vector does not automatically eliminate this concern. The severity depends on how the representative split was selected and whether it was outside all evaluated folds.

Recommended correction. Derive weights exclusively from each fold's training data before augmentation, or use a weighting scheme defined independently of evaluation labels. Re-run affected experiments if weights change. Explain the source of the weights and the exact split used.

Implementation or data check. Compare the reported fixed weights with a fold-specific training-derived scheme. The manuscript already reports a training-distribution-weighting ablation (Table 5), which can help determine whether that scheme is an appropriate primary configuration.


## 02. Figure 2 disagrees with the four-class classification head

High | Confirmed presentation inconsistency | Manuscript location: Section 3.5.4, page 10, Equation (15), Figure 2a on page 8

What the manuscript says. Equation (15) specifies W2(head) with shape 4 × 128. Figure 2a instead labels the last classification layer as Linear: 128 → 128, followed by four-class softmax output.

Why this matters. A 128-dimensional output does not directly match the four class logits specified mathematically. The narrative and diagram disagree about the architecture.

Recommended correction. If Equation (15) reflects the implementation, update the figure to Linear: 128 → 4. If the implementation includes another layer, explicitly specify that layer in the equation, text, and architecture table.

Implementation or data check. Inspect the classification head's forward pass and final weight tensor shape.

## 03. DP-SGD privacy accounting versus augmented original images

Critical | Conditional privacy concern | Manuscript location: Section 3.3, pages 6-7, Section 4.1, pages 14-15, Table 9 on page 24

What the manuscript says. Each local training set may be expanded to approximately 1,500 training images by offline augmentation. DP-SGD uses Opacus with target epsilons 8 and 12, delta = 10^-5, per-example clipping C = 1.0, and an accountant intended to persist over all communication rounds.

Why this matters. If augmented variants of one original photograph are treated as independently sampled records, the reported guarantee may apply to augmented records rather than the underlying original image. Reusing source images changes the privacy unit. Persisting the accountant across rounds is necessary for composition but does not, by itself, solve this record- grouping issue. The paper also cannot claim patient-level privacy without a patient-level unit and accounting argument.

Recommended correction. State precisely whether the privacy unit is an augmented training record, original photograph, or patient. If claiming original-image-level DP, redesign sampling and accounting so all related augmentations are handled consistently or provide a defensible group-privacy treatment. Report actual per-client and worst-case cumulative privacy spending rather than only a target.

Implementation or data check. Inspect dataset index and sampler, augmentation multiplicity, Opacus data loader and sampling assumptions, persistent accountant state for each client, and the final achieved epsilon on each evaluated fold.

## 04. Restricted three-class external validation differs from the original task

High | Confirmed evaluation protocol distinction | Manuscript location: Section 5.10, page 25

What the manuscript says. The trained model predicts Chickenpox, Healthy, Measles, and Monkeypox. MPox-Vision contains only Chickenpox, Measles, and Monkeypox. The method drops the Healthy probability, renormalizes the remaining three outputs, and then evaluates a three-class classification problem. Reported uncorrected performance is 88.1% accuracy and 0.878 macro-F1.

Why this matters. This is a valid restricted-label-set analysis when clearly labeled, but it does not directly measure accuracy of the original four-class inference rule. A case originally predicted as Healthy is reassigned to one of the three diseases before scoring.

Recommended correction. Label the result as conditional or restricted three-class external evaluation. Report the proportion of Healthy predictions on external disease-positive images. Where feasible, provide an additional strict four-output evaluation in which Healthy predictions are errors. Avoid presenting the restricted score as a direct four-class deployment result.

Implementation or data check. Export original four-way predictions and compare strict predictions with restricted three-way predictions. Retain the same external samples and model checkpoints.

## 05. Focal-loss alpha is not an established prior-shift correction

High | Mathematical justification missing | Manuscript location: Section 5.10, page 25, Equation (18) on page 11

What the manuscript says. After dropping Healthy, the manuscript divides each of the three class probabilities by its fixed focal-loss alpha and renormalizes. It calls this a prior-shift correction and states that it assumes a uniform external prior. Post- correction external accuracy is 88.8% and macro-F1 is 0.886.

Why this matters. Equation (18) uses alpha_c proportional to 1/n_c. Dividing a class score by alpha_c therefore multiplies it proportionally by n_c. Such a rule is not automatically a theoretically valid change-of-prior adjustment, particularly because


class-weighted focal loss changes the relationship between the trained softmax and an estimate of the original population posterior.

Recommended correction. Either derive the correction from explicit source and target prior assumptions and a calibrated posterior model, or call it a heuristic score adjustment. Always report uncorrected and corrected performance separately. Make clear that the external labels did not determine the adjustment.

Implementation or data check. Check whether the correction was prespecified. Compare the raw four-class scores, restricted uncorrected scores, and adjusted scores, including class-level precision, recall, and confusion matrices.

Equation-level explanation: alpha_c ∝ 1/n_c, so p_corrected,c ∝ p_c/alpha_c ∝ p_c · n_c, before normalizing over the retained classes.

## 06. FedAvg and FedProx change the local objective, not the server aggregation

High | Confirmed conceptual wording error | Manuscript location: Introduction, Section 3.6, Section 3.6.3, pages 11-13, Equations (20)-(21)

What the manuscript says. The manuscript characterizes its FedAvg versus FedProx experiment as comparing aggregation strategies and says differences reflect aggregation strategy alone. Both procedures use sample-size-weighted parameter averaging in Equation (21). FedProx differs by including a proximal term with mu = 0.01, while FedAvg has mu = 0.

Why this matters. The causal variable in this comparison is the local optimization objective. It is not a change in the server aggregation operator. This matters to claims about why one configuration performs better or worse.

Recommended correction. Replace references to different aggregation strategies with local federated optimization methods, proximal regularization, or local objectives. Describe both as using the same aggregation rule. Reserve aggregation-method language for comparisons such as trimmed mean versus weighted averaging.

Implementation or data check. No new experiment is required if the methodology reflects the implementation.

## 07. FedPer comparison may penalize locally specialized heads

Moderate-high | Evaluation interpretation concern | Manuscript location: Sections 3.3, 4.2, and 5.8, Table 8 on page 24

What the manuscript says. Dirichlet(alpha = 0.5) label-skew training can give clients zero examples of some classes, yet the paper says each client test partition contains all four classes. FedPer keeps a local head on each client. In Table 8, FedPer has mean macro-F1 0.709 versus 0.905 for the global FedProx model under label skew, and 0.896 versus 0.911 under quantity skew.

Why this matters. Locally trained heads that have not encountered particular classes can predict poorly on a balanced four- class test set. The comparison supports a conclusion about performance on that evaluation distribution, but not a universal statement that personalization performs worse across naturally heterogeneous clients.

Recommended correction. Explicitly state the test distribution and its consequences. Ideally add a client-distribution- matched evaluation, while preserving the common four-class test for comparability. Interpret the large label-skew gap in terms of unsupported local classes and distribution mismatch.

Implementation or data check. Review per-client train and test class counts and ensure the local head persists across rounds exactly as described.

## 08. Pooled datasets and simulated clients limit cross-dataset claims

Moderate-high | Scope and independence concern | Manuscript location: Sections 3.2-3.3, pages 5-7, Section 5.10, page 25, Limitations on page 26

What the manuscript says. The development pool combines MSID, MSLDv2, EMSID, and some additional public images, then creates five simulated clients with IID, quantity-skew, and Dirichlet partitions. EMSID already includes images from overlapping public sources. The manuscript describes SHA-256, perceptual hashing, and SSIM deduplication before partitioning. MPox- Vision is presented as external validation.

Why this matters. Federated heterogeneity generated from one deduplicated image pool is not the same as training on five independent institutions or dataset-specific sites. Image-level deduplication within the development pool does not itself demonstrate that MPox-Vision is overlap-free relative to all development sources. The absence of patient IDs also prevents a guarantee of patient-disjoint splits.

Recommended correction. Emphasize simulated heterogeneous clients from a cross-source pooled dataset. Avoid implying five real-world clinical institutions. Verify external-to-development duplicate and near-duplicate exclusion, and document that verification. Consider leave-one-dataset-out or source-specific-client studies if available.


Implementation or data check. Audit provenance records for all sources, manual overlap-review outcomes, cross-corpus hashes, patient identifiers when available, and train/validation/test manifests.

## 09. Equivalent-looking model configurations have different numerical summaries

Moderate | Reproducibility check needed | Manuscript location: Table 3 on page 18, Table 5 on page 21, Section 5.3 on page 20, Table 2 on page 14

What the manuscript says. Table 3 reports the main heterogeneous FedProx macro-F1 as 0.9106 ± 0.022 and a matched-seed FedProx rerun as 0.9174 ± 0.020. Its homogeneous FedProx result is 0.9247 ± 0.023, while the apparently corresponding Table 5 primary ablation value is 0.923 ± 0.011. The paper also states that seed 42 and supported deterministic settings were used.

Why this matters. The first two heterogeneous rows can legitimately differ because Section 5.3 explicitly says the methods were retrained. The homogeneous difference might also be explained by a different run or evaluation protocol. The manuscript does not sufficiently trace each result to its experiment configuration and checkpoint-selection procedure.

Recommended correction. Add a compact experiment-ID or protocol table that distinguishes the primary run, matched-seed rerun, and ablation run. Clarify fold splits, seeds, augmentation, checkpoint selection, TTA, SWA, and evaluation pipeline for each. Do not overwrite results simply to make tables agree.

Implementation or data check. Recompute the reported means and standard deviations from original fold-level logs, then reconcile all runs with their saved configuration files.

## Reported comparisons requiring reconciliation

| Configuration | Result A | Result B | Interpretation |
| --- | --- | --- | --- |
| Heterogeneous FedProx | Table 3 main: 0.9106 ± 0.022 | Table 3 matched-seed: 0.9174 ± 0.020 | Retraining is explicitly indicated. Document why outputs differ. |
| Homogeneous primary | Table 3: 0.9247 ± 0.023 | Table 5: 0.923 ± 0.011 | Potentially different runs. Identify precise protocol. |

## 10. The CSAH ablation does not establish a measurable improvement

High | Scientific claim and interpretation concern | Manuscript location: Section 5.4, Table 5 on page 21, Discussion on page 26

What the manuscript says. Under standard evaluation, the plain ConvNeXtV2-Tiny baseline has macro-F1 0.918 ± 0.022, the proposed primary architecture has 0.923 ± 0.011, and the simpler CBAM + ECA + GAP configuration has 0.931 ± 0.020. The manuscript notes that paired confidence intervals for the key architecture comparisons include zero, except that replacing GeM with average pooling lowers F1 by 0.017 with 95% CI [−0.032, −0.001].

Why this matters. The complete CSAH system is not the highest mean-F1 architecture in Table 5 and does not show a statistically distinguishable gain over simpler alternatives on these five folds. The strongest isolated result concerns GeM, not superiority of the entire cross-scale attention pathway.

Recommended correction. Keep the reported numbers unchanged and frame the architecture as a studied integrated design rather than a proven accuracy improvement over all simpler alternatives. State that the incremental contribution of CSAH is inconclusive at this sample size. If stronger novelty evidence is required, conduct larger repeated-seed, cross-dataset, or parameter-matched ablations.

Implementation or data check. Check that all ablation configurations share identical folds, training budgets, selection rules, and model initialization conventions.

## Ablation comparison from Table 5

| Configuration | Macro-F1 | Supported interpretation |
| --- | --- | --- |
| Plain ConvNeXtV2-Tiny | 0.918 ± 0.022 | Strong backbone baseline |
| Full PoxCSAF-Net primary | 0.923 ± 0.011 | Modest mean increase, not a measurable gain in paired comparison |
| CBAM + ECA + GAP | 0.931 ± 0.020 | Highest mean among the listed configurations |


## Minor but required submission fixes

## Unresolved Dirichlet citation

Where. Section 3.3, page 7

Observed. The Dirichlet(alpha = 0.5) description ends with an unresolved “(? )” reference.

Fix. Insert the intended source or remove the orphan citation.

## Figure 4 subfigure labels

Where. Figure 4, page 19

Observed. Its first two panels are labeled “Figure 5a” and “Figure 5b”, although they belong to Figure 4. Remaining panels use Figure 4c through Figure 4f.

Fix. Use a consistent Figure 4a through Figure 4f naming convention, or panel A through F.

## Old ablation names

Where. Section 5.4 and Table 5, page 21

Observed. Configuration names include “msaf primary”, “msaf gem only”, and “winning primary”, which read like internal experiment labels rather than polished PoxCSAF-Net terminology.

Fix. Replace with consistent reader-facing model names and define variants in the caption.

## Abstract remains template text

Where. Page 1

Observed. The abstract is not an actual study abstract. It still contains publisher template guidance.

Fix. Replace with a completed structured or unstructured abstract that follows the target journal requirements.

## Author and administrative placeholders

Where. Pages 1 and 27-28

Observed. The first page retains “Corresponding Author” and “email@uni.edu”. Author Contributions, Funding, Acknowledgments, and Data Availability sections also contain template guidance or placeholders, including [GENERATED/ANALYZED], [NAME OF REPOSITORY], and [LINK].

Fix. Populate all mandatory metadata, declarations, and links or remove non-applicable placeholder prose according to journal policy.

## Minor punctuation typo

Where. Section 3.1, page 5

Observed. The study overview says “model calibration, and, external validation”.

Fix. Remove the extra comma before “external validation”.

## Checks that did not reveal contradictions

These checks are important so that the report does not misrepresent the manuscript as having widespread numerical errors.

| Metric / test | Independent check | Manuscript value | Assessment |
| --- | --- | --- | --- |
| Centralized macro-F1 | Mean of class F1: 0.9135 | Table 3: 0.9134 | Consistent within rounding |
| Homogeneous FL macro-F1 Mean of class F1: 0.92475 |   | Table 3: 0.9247 | Consistent within rounding |
| Heterogeneous FL macro-F1 Mean of class F1: 0.9105 |   | Table 3: 0.9106 | Consistent within rounding |
| Dirichlet FL macro-F1 | Mean of class F1: 0.9055 | Table 3: 0.9054 | Consistent within rounding |
| Development image count | 220 + 460 + 221 + 562 = 1,463 | Section 3.2: 1,463 | Consistent |
| Class confusion | Chickenpox-Mpox errors increase across heterogeneity settings in the described Fold 1 matrices | Section 5.2 | Direction is consistent with discussion |


## Recommended correction sequence

- P0. Privacy validity. Determine the DP privacy unit, group augmented copies correctly, verify sampling assumptions and achieved per-client epsilons before claiming original-image privacy.

- P0. Training leakage. Rework the focal-loss alpha source so held-out labels do not influence training. Decide whether affected experiments must be re-run.

- P1. External validation scope. Distinguish restricted three-class evaluation from strict original four-class predictions and, if possible, report both.

- P1. Probability correction. Derive the prior correction or relabel it as a prespecified heuristic. Keep corrected and uncorrected results distinct.

- P1. Reproducibility. Map Table 3 and Table 5 results to exact run IDs, seeds, folds, checkpoint rules, SWA, and TTA.

- P1. Data independence. Verify overlap of MPox-Vision against development sources. State the simulated-client and patient- independence limitations precisely.

- P2. Architectural claims. Align results and novelty claims with the inconclusive CSAH ablation. Describe GeM evidence accurately.

- P2. Comparative framing. Correct FedAvg versus FedProx terminology and explain FedPer test-distribution mismatch.

- P2. Architecture figure. Correct the final layer dimension in Figure 2a after checking implementation.

- P3. Submission cleanup. Replace placeholders, repair Figure 4 numbering, resolve the Dirichlet citation, standardize ablation labels, and correct punctuation.

## Scope and verification note

This report preserves all major, secondary, and minor issues raised during the review of the uploaded manuscript. A confirmed manuscript discrepancy means the text, figure, or mathematical description is internally inconsistent. A methodological concern means the written protocol may be defensible only with stronger justification or a revised claim. Items marked for verification cannot be conclusively judged without training code, original fold-level outputs, data manifests, or privacy- accountant settings. No evidence of invented measurements is inferred from these observations.
