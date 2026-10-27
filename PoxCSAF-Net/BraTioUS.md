

Then we drop the human-annotation chapter entirely and **replace it with computational + independent-data proofs**. This is still doable for Q1 — just with slightly higher residual risk. Here is the revised, 100%-solo-executable plan:

# Revised Plan: No annotator, no doctor, zero manual labeling by you

## Updated dataset roles

| Dataset | Role | Annotation origin (why it's trustworthy) |
|---|---|---|
| BraTioUS (yours) | Train + internal validation only (patient-level, vendor-stratified) | Expert-refined pseudo-labels (used honestly, audited, never used as the final proof) |
| **BraTS 2023** | Raw material → **synthetic iUS benchmark** (train split + **test split**) | Manual radiologist MRI labels, transferred to synthesized US → **noise-free GT** |
| **RESECT-SEG** (public, KCL) | External test 1 — real iUS | Independently annotated by its own authors — **check its exact annotation method in its paper and cite it** |
| **REMIND** (preop transcranial US, Bern) | External test 2 (hardest domain distance) | Expert-annotated by its own team (needs data-use agreement, no work from you) |
| Faanes/Reinertsen 2024/25 2D iUS set, Angel-Raya interventional US set | External test 3/4 if accessible | Independent cohorts, different annotation pipelines |
| Imperial College cohort | External test 5 **by email request only** (the Cancers 2025 authors used it) | Their annotations, zero yours |

**Your headline metric rule: no final claim relies on BraTioUS masks.** BraTioUS masks are allowed to train, never to judge.

## How the "label bias" answer now works — 4 layers, all solo

1. **Synthetic-noise self-validation of your noise detector.**
   - Take the clean-label stream (BraTS-derived synthetic iUS).
   - Inject realistic noise into some masks (boundary jitter, region deletion/addition, teacher-style systematic errors).
   - Run your detector (multi-architecture 5-fold disagreement + CLIP confident learning + loss statistics) and show it **finds exactly the injected errors** (precision/recall of detection).
   - Apply the validated detector to real BraTioUS → report which ~10% masks are suspect. You have proven the detector works without any human label.

2. **Quantify the experts' actual corrections — from the authors, by email.**
   - The descriptor says final masks = nnU-Net pseudolabels + neurosurgeon corrections. Ask the BraTioUS authors (corresponding author of the Data in Brief / Cancers papers) for the **raw pre-correction pseudolabels** (or their correction statistics).
   - Diff(pseudo, final) = a direct, data-driven map of **where the model was wrong and the expert fixed it**. Even partial/aggregate numbers ("7.3% of masks modified, mean modified area 4%") are gold for your paper. This converts the bias from unknown to measured — with one email, no labor.
   - If they refuse: your synthetic-noise analysis + external validation still carry the argument.

3. **Independent-cohort generalization as the truth test.**
   - A model biased to BraTioUS's teacher will do well on BraTioUS masks but **should still perform normally on RESECT-SEG/REMIND if your method is genuinely good**. Consistent performance across cohorts annotated by unrelated pipelines = strongest realistic evidence against teacher-capture.
   - Also report the **teacher-affinity diagnostic**: plain nnU-Net vs independent architectures trained on BraTioUS — show the gap yourself before reviewers do.

4. **Noise-robust training by design** (unchanged): co-teaching + symmetric Dice + suspect-mask down-weighting — and show on the synthetic benchmark (clean GT) that your robust recipe ≥ standard training, even at high injected noise rates.

## Paper structure becomes

1. First large-scale **leave-one-vendor-out benchmark** on BraTioUS (baseline table, many architectures) — contribution #1
2. **Label-noise-aware training** validated on synthetic ground truth — contribution #2
3. **MRI→synthetic-iUS** as a general data-amplification trick with clean labels — contribution #3
4. Multi-cohort external validation (2–4 independent datasets) — the trust layer

## Honest risk statement (so you can decide with eyes open)

- **What survives without an expert:** everything computational — noise measurement, robust training, clean synthetic benchmark, cross-cohort transfer. This is defensible at *CBMB, CMP, Ultrasound in Med & Bio, JJBS*-level Q1, and realistically at *MedIA/TMI* if external validation is strong (3+ cohorts) and the circularity analysis is upfront.
- **What weakens:** the "true quality on real images" sentence now becomes "true quality on *other people's* manual labels (RESECT-SEG etc.)" — which is honestly nearly as good, because those are the same kind of reference everyone in the field accepts.
- **One mandatory homework:** before committing, download RESECT-SEG and REMIND and **verify in their papers** that their masks are manual/independent (I'm confident they're independent-annotated, but confirm the exact provenance yourself — same lesson as BraTioUS!).

**Bottom line:** still a complete Q1-shaped plan with zero clinical involvement. The single highest-value cheap action now is **the email to the BraTioUS authors** (pre-correction masks / correction stats) — it's the only remaining way to touch the bias question directly.
