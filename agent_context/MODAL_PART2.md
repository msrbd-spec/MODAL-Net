=====================================================================
MODAL-Net — MASTER BUILD PLAN (PART 2 of 2): phases, commands, tables, figures, tests, work packages, risks
=====================================================================
S8. ANTI-OVERFITTING RECIPE (defaults of config.yaml; ablated one factor at a time in phase2a)
  zoo evidence: train acc 93-99% vs test 77-85% (gap 14-18). Goal: train-val gap <= 8-10 points (diagnostic only, never a selection criterion).
  - 30 epochs, 2-epoch warmup, cosine, early stopping on val QWK (patience 8), best-epoch checkpoint by QWK, evaluated model = EMA (decay 0.999).
  - drop_path 0.3, head dropout 0.3, weight decay 0.05 (none on bias/norm), clip 1.0, bf16, layer-wise LR decay 0.8 (head x10), first epoch trains with the stem/stage 0 frozen.
  - Mixup 0.2 / CutMix 1.0 on 50% of batches (soft targets), ordinal label smoothing eps 0.1, medical-safe augmentation (aug=medical; no elastic/grid distortion).
  - Optional: SAM (rho 0.05), SWA; both are ablation rows only.

S9. PHASES (each phase = a results/<phase>/ folder; experiments are parallel alternatives against a reference unless stated)
  phase0_protocol        splits, dedup audit, zoo import, tests.
  phase1_zoo             imported zoo table (MaxViT-L chosen: best on every metric; MaxViT-L vs SwinV2-L acc +2.6, p~0.015, Welch on fold means, rough).
  phase2a_recipe         one factor at a time on a plain MaxViT-L baseline (reg: drop_path/dropout/LLRD; mix; ordinal smoothing; EMA; medical aug; SAM optional) + bundle of adopted/tentative factors.
  phase2b_preprocess     legacy (= reference) vs scaled (resize-first + pad-to-square + sigma = size/50) vs clahe_only vs none. Decide on OOF QWK and Messidor-DEV QWK/Mild recall jointly.
  phase3a_pretrain       ImageNet (= reference) vs EyePACS-supervised vs lesion-aware SSL vs SSL -> EyePACS-supervised.   Adopt only if OOF QWK is not significantly worse and DEV QWK or Mild recall improves with CI excluding 0.
                         Kill: if SSL does not beat EyePACS-supervised on DEV and OOF, report "no added value" and use EyePACS-supervised.
  phase3b_joint          best pretraining (= reference) vs joint APTOS + EyePACS fine-tuning (EyePACS as second labelled domain).
  phase4_label_loss      reference vs EMD loss, CORAL, nested ambiguity-aware soft labels, soft labels + EMD. Expect ~0-1 point; null results are reported.
  phase5_architecture    reference vs wavelet gate (prior ~0); ConvNeXtV2-L and EfficientNetV2-M trained with the same recipe; heterogeneous ensemble (inference-level, winner_eligible=false).
  phase6_domain_gen      source-only DG: amplitude mixing (stage pre|post x beta_max 0.03/0.10/0.20) , cross-domain MixStyle (needs joint sampler), both; labelled extras: UDA (Messidor-DEV style pool) and dev-calibrated cut points (winner_eligible=false).
                         Hyper-parameters are chosen on Messidor-DEV (primary: DEV QWK, Mild recall) with OOF QWK drop <= 0.005. If the winner uses preproc=legacy, add the experiment p6_ref (same config, preproc=scaled, no DG) because amplitude mixing needs preproc != legacy.
  phase7_feature_balance optional (S7n).
  phase8_inference       TTA variants, temperature scaling, ordinal-interval conformal, selective prediction (no training).
  phase9_final           winner re-run with seeds 1,2 (seed 0 = the winner itself) and the plain MaxViT-L baseline with seeds 1,2; seed table; then master table and figures.

S10. COMMANDS (every experiment is ONE command; it trains 5 folds, tests, evaluates Messidor DEV and TEST, writes CSVs and figures)
  Conventions: --phase names the results folder; --tag is the unique experiment name (also the checkpoints/ and logs/ folder); --inherit X starts from the saved config of X; --inherit @phaseNAME uses results/phaseNAME/WINNER.txt.
  Independent commands can run in parallel on different GPUs: CUDA_VISIBLE_DEVICES=1 python main.py ...      Never start a command that inherits a tag before that tag's command has finished.
  --- Phase 0 (once) ---
  python main.py --mode make_splits
  python main.py --mode dedup_audit
  python main.py --mode detect_lesions
  python main.py --mode import_zoo --phase phase1_zoo --csv baseline_result.csv
  pytest tests -q
  --- Phase 2a: recipe (plain baseline = the zoo recipe) ---
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_plain   --set loss=ce drop_path=0.0 head_dropout=0.0 llrd=1.0 mix_prob=0.0 ordinal_eps=0.0 ema_decay=0.0 aug=basic preproc=legacy
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_reg     --inherit p2a_plain --set drop_path=0.3 head_dropout=0.3 llrd=0.8
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_mix     --inherit p2a_plain --set mix_prob=0.5
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_osmooth --inherit p2a_plain --set ordinal_eps=0.1
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_ema     --inherit p2a_plain --set ema_decay=0.999
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_aug     --inherit p2a_plain --set aug=medical
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_sam     --inherit p2a_plain --set use_sam=true          (optional)
  python main.py --mode phase_select --phase phase2a_recipe --reference p2a_plain --design parallel
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_bundle --inherit p2a_plain --set bundle_from=phase2a_recipe
  python main.py --mode phase_select --phase phase2a_recipe --reference p2a_plain --design parallel          (final WINNER.txt of the phase, bundle included)
  --- Phase 2b: preprocessing parity (reference = the phase2a winner, no retraining) ---
  python main.py --mode experiment --phase phase2b_preprocess --tag p2b_scaled     --inherit @phase2a_recipe --set preproc=scaled
  python main.py --mode experiment --phase phase2b_preprocess --tag p2b_clahe_only --inherit @phase2a_recipe --set preproc=clahe_only
  python main.py --mode experiment --phase phase2b_preprocess --tag p2b_none       --inherit @phase2a_recipe --set preproc=none
  python main.py --mode phase_select --phase phase2b_preprocess --reference @phase2a_recipe --design parallel
  --- Phase 3a: pretraining ---
  python main.py --mode pretrain --kind eyepacs_sup --inherit @phase2b_preprocess --tag pre_eyepacs_sup
  python main.py --mode pretrain --kind ssl         --inherit @phase2b_preprocess --tag pre_ssl --set ssl_save_dir=checkpoints/pretrain/ssl_maxvit
  python main.py --mode pretrain --kind ssl_sup     --inherit @phase2b_preprocess --tag pre_ssl_sup --set ssl_init=checkpoints/pretrain/ssl_maxvit/ssl_pretrained_backbone.pth
  python main.py --mode experiment --phase phase3a_pretrain --tag p3a_sup    --inherit @phase2b_preprocess --set pretrain_path=checkpoints/pretrain/eyepacs_sup.pth
  python main.py --mode experiment --phase phase3a_pretrain --tag p3a_ssl    --inherit @phase2b_preprocess --set pretrain_path=checkpoints/pretrain/ssl_maxvit/ssl_pretrained_backbone.pth
  python main.py --mode experiment --phase phase3a_pretrain --tag p3a_sslsup --inherit @phase2b_preprocess --set pretrain_path=checkpoints/pretrain/ssl_sup.pth
  python main.py --mode phase_select --phase phase3a_pretrain --reference @phase2b_preprocess --design parallel
  --- Phase 3b: joint multi-source fine-tuning ---
  python main.py --mode experiment --phase phase3b_joint --tag p3b_joint    --inherit @phase3a_pretrain --set joint_eyepacs=true joint_w=0.5
  python main.py --mode experiment --phase phase3b_joint --tag p3b_joint_w25 --inherit @phase3a_pretrain --set joint_eyepacs=true joint_w=0.25        (optional)
  python main.py --mode phase_select --phase phase3b_joint --reference @phase3a_pretrain --design parallel
  --- Phase 4: labels and loss (soft labels are nested: 5 x (4 inner + 1 outer) trainings) ---
  python main.py --mode experiment --phase phase4_label_loss --tag p4_emd   --inherit @phase3b_joint --set loss=ce_emd lambda_emd=1.0
  python main.py --mode experiment --phase phase4_label_loss --tag p4_coral --inherit @phase3b_joint --set loss=coral head_type=coral
  python main.py --mode make_softlabels --phase phase4_label_loss --tag sl_p4 --inherit @phase3b_joint
  python main.py --mode experiment --phase phase4_label_loss --tag p4_aasl      --inherit @phase3b_joint --set soft_labels=aasl soft_label_run=sl_p4
  python main.py --mode experiment --phase phase4_label_loss --tag p4_aasl_emd  --inherit @phase3b_joint --set soft_labels=aasl soft_label_run=sl_p4 loss=ce_emd lambda_emd=1.0
  python main.py --mode phase_select --phase phase4_label_loss --reference @phase3b_joint --design parallel
  --- Phase 5: architecture and heterogeneous ensemble ---
  python main.py --mode experiment --phase phase5_architecture --tag p5_wghr --inherit @phase4_label_loss --set use_wghr=true
  python main.py --mode experiment --phase phase5_architecture --tag p5_convnextv2 --inherit @phase4_label_loss --set backbone=convnextv2_large.fcmae_ft_in22k_in1k pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
  python main.py --mode experiment --phase phase5_architecture --tag p5_effnetv2   --inherit @phase4_label_loss --set backbone=tf_efficientnetv2_m.in21k_ft_in1k pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
  python main.py --mode ensemble_experiment --phase phase5_architecture --tag p5_hetero_ens --members @phase4_label_loss,p5_convnextv2,p5_effnetv2 --set winner_eligible=false
  python main.py --mode phase_select --phase phase5_architecture --reference @phase4_label_loss --design parallel
  --- Phase 6: domain generalisation (choose on Messidor-DEV) ---
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_ref          --inherit @phase5_architecture --set preproc=scaled amp_mix=false          (only if the winner uses preproc=legacy)
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_pre_b03  --inherit @phase5_architecture --set preproc=scaled amp_mix=true amp_stage=pre  amp_beta_max=0.03
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_pre_b10  --inherit @phase5_architecture --set preproc=scaled amp_mix=true amp_stage=pre  amp_beta_max=0.10
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_pre_b20  --inherit @phase5_architecture --set preproc=scaled amp_mix=true amp_stage=pre  amp_beta_max=0.20
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_post_b10 --inherit @phase5_architecture --set preproc=scaled amp_mix=true amp_stage=post amp_beta_max=0.10
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_ms           --inherit @phase5_architecture --set preproc=scaled joint_eyepacs=true mixstyle=true ms_p=0.5 ms_alpha=0.1
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_ms       --inherit @phase5_architecture --set preproc=scaled joint_eyepacs=true mixstyle=true amp_mix=true amp_stage=<BEST_STAGE> amp_beta_max=<BEST_BETA>
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_uda          --inherit @phase5_architecture --set preproc=scaled uda_style_pool=messidor_dev amp_mix=true amp_stage=<BEST_STAGE> winner_eligible=false     (labelled UDA, not DG)
  python main.py --mode inference_experiment --phase phase6_domain_gen --tag p6_devcal --base <BEST_P6_TAG> --set recalibrate=devcut winner_eligible=false       (cut points fitted on Messidor-DEV labels, labelled "dev-calibrated")
  python main.py --mode phase_select --phase phase6_domain_gen --reference p6_ref --design parallel        (reference = @phase5_architecture if p6_ref was not needed)
  --- Phase 7 (optional) ---
  python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_none     --base @phase6_domain_gen --method none
  python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_reweight --base @phase6_domain_gen --method reweight
  python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_smote    --base @phase6_domain_gen --method smote
  python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_gauss    --base @phase6_domain_gen --method gauss
  python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_lfsd     --base @phase6_domain_gen --method lfsd
  python main.py --mode phase_select --phase phase7_feature_balance --reference p7_none --design parallel
  --- Phase 8: inference (no training) ---
  python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_none    --base @phase6_domain_gen --set tta=none
  python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_flip    --base @phase6_domain_gen --set tta=flip
  python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_fliprot --base @phase6_domain_gen --set tta=flip_rot
  python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_scale   --base @phase6_domain_gen --set tta=flip_rot tta_scales=[0.9,1.0,1.1]
  python main.py --mode inference_experiment --phase phase8_inference --tag p8_temp         --base @phase6_domain_gen --set tta=<BEST_TTA> temperature_scaling=true
  python main.py --mode calibrate_conformal   --phase phase8_inference --tag p8_conformal   --base p8_temp --set conformal_alpha=0.10
  python main.py --mode phase_select --phase phase8_inference --reference p8_tta_none --design parallel
  --- Phase 9: final (seeds) ---
  python main.py --mode inference_experiment --phase phase9_final --tag p9_final_s0 --base @phase8_inference
  python main.py --mode experiment --phase phase9_final --tag p9_final_s1 --inherit @phase6_domain_gen --set seed=1 tta=<BEST_TTA>
  python main.py --mode experiment --phase phase9_final --tag p9_final_s2 --inherit @phase6_domain_gen --set seed=2 tta=<BEST_TTA>
  python main.py --mode experiment --phase phase9_final --tag p9_base_s1  --inherit p2a_plain --set seed=1
  python main.py --mode experiment --phase phase9_final --tag p9_base_s2  --inherit p2a_plain --set seed=2
  python main.py --mode inference_experiment --phase phase9_final --tag p9_base_s0 --base p2a_plain
  python main.py --mode seed_summary --phase phase9_final
  --- Phase 10: XAI (WP10) ---
  python main.py --mode xai --phase phase9_final --tag winner [--set xai_seed=0 xai_top_k=10 xai_methods=[gradcampp,gradcam,layercam,eigencam] xai_show_lesions=true xai_split=test]
  # Example for strictly most confident correct predictions:
  python main.py --mode xai --phase phase9_final --tag winner --set xai_top_k=1
  --- End: master table and paper artefacts ---
  python main.py --mode master_table            -> results/master/master_table.csv (+ .tex, master_full.csv, master_per_class.csv, master_binary.csv, master_config.csv, master_deltas.csv)
  python main.py --mode report                  -> results/tables/table01..15 and results/figures/fig01..18
  Placeholders: <BEST_STAGE>, <BEST_BETA>, <BEST_P6_TAG>, <BEST_TTA> are read from results/<phase>/phase_table.csv; the winner tag of each phase is in results/<phase>/WINNER.txt.

S11. PAPER TABLES (built from the master table by `--mode report`; CSV + LaTeX booktabs in results/tables/; 2 dp; mean+-std over folds for single models, bootstrap 95% CI for ensembles; best bold, second underlined; each CSV also lists the experiment names used)
  table01 baseline zoo (imported)  | table02 recipe + preprocessing (phase2a/2b)  | table03 pretraining + joint-source (phase3a/3b)  | table04 label/loss (phase4)  | table05 architecture + ensemble (phase5)
  table06 domain generalisation (phase6: ext QWK, Mild recall, predicted No-DR share, ext acc, ext referable acc/AUROC/sens@spec90; UDA and dev-calibrated rows marked)  | table07 feature-level balancing (phase7)
  table08 inference (TTA, temperature scaling: ECE/NLL)  | table09 final per-class report, internal test and Messidor TEST  | table10 binary (referable) metrics: acc, prec, sens, spec, F1, AUROC, AUPRC, sens@spec90/95, internal and external
  table11 statistical tests (each adopted component vs its reference: delta, 95% CI, Holm p, McNemar p)  | table12 conformal / selective prediction  | table13 literature comparison (reproduced zoo models + literature rows ONLY from verified papers, TODO_CITE until verified)
  table14 compute (params, FLOPs, GPU hours, ms/image)  | table15 seed variance (final vs plain baseline, seeds 0-2).

S12. FIGURES (`--mode report` -> results/figures/; per-experiment figures stay inside the experiment folder)
  Bold styling rule (WP10, Step 1): All text in every figure must be bold (fontweight >= 700): titles, axis labels, tick labels, legends, legend titles, annotations, colorbars, panel/row/column labels. Enforced by set_bold_style() and save_fig() with assert weight_value(t.get_fontweight()) >= 700.
  Style: Arial 9 pt labels / 8 pt ticks, line width 1.5, grid alpha 0.25, no titles (captions in LaTeX), 300 dpi PNG + vector PDF, widths 3.5 in single / 7.2 in double.
  Colours: TRAIN '#1F77B4' (blue) and VAL '#D62728' (red) for every train/val curve. CLASS colours (Okabe-Ito): No DR '#0072B2', Mild '#E69F00', Moderate '#009E73', Severe '#D55E00', Proliferative '#CC79A7'.
  DOMAIN colours: APTOS '#332288', EyePACS '#117733', Messidor-2 '#CC6677'. Neutral '#7F7F7F'. Confusion matrices 'Blues'; difference heatmaps 'RdBu_r' centred at 0; grids 'viridis'.
  fig01 pipeline (manual)   fig02 class distributions (domain colours, % with 2 dp labels, n table)   fig03 train/val curves (mean +- std shading over folds; best-epoch line; gap annotation)
  fig04 confusion matrices (5-class and 2x2 referable; test and Messidor TEST; "n" and row-% to 2 dp)   fig05 ROC per class + macro + referable with bootstrap band and spec 90/95 markers   fig06 PR curves
  fig07 reliability diagrams before/after temperature scaling (bars '#1F77B4', diagonal grey dashed, gap hatched '#D62728', ECE in legend)   fig08 risk-coverage internal vs external (AURC in legend)
  fig09 t-SNE/UMAP of penultimate features (<=3000 points, fixed seed): by class and by domain, baseline vs final   fig10 under-grading: mean expected grade by true class (internal vs external) and predicted vs true class distribution on Messidor TEST
  fig11 ablation bars per phase with 95% CI error bars (reference grey, ADOPT '#0072B2', REJECT light grey hatched)   fig12 forest plot of paired deltas (QWK, referable AUROC, Mild recall) with Holm p
  fig13 per-class F1 and recall grouped bars (class colours), baseline vs final, internal and external   fig14 label ambiguity: histogram of alpha per class and a 4x4 grid of highest-alpha training images annotated "given -> OOF-pred"
  fig15 domain analysis: radially averaged log amplitude spectra per domain before/after amplitude mixing, and the stage x beta heatmap of Messidor-DEV QWK
  fig16 XAI grid: results/figures/fig16_xai_grid.{png,pdf} (5-row XAI Grid: classes 0..4 x columns [Original | Grad-CAM++ | Grad-CAM | LayerCAM | Eigen-CAM | Detected lesions] with shared horizontal colorbar, bold row labels with ensemble confidence, bold column headers, Arial bold via set_bold_style)
  fig17 conformal: coverage vs target and interval-width distribution (bars '#0072B2')   fig18 preprocessing parity: 8 random APTOS vs 8 Messidor images after each preproc mode (legacy vs scaled), with native sizes in the captions.

S13. WORK PACKAGES FOR THE AGENT (give Part 1 + Part 2 and ONE package at a time; each ends with the checks of S0)
  Prompt template: "Read the attached MODAL-Net master plan (Part 1 and Part 2) and the legacy_seed folder. Implement ONLY work package WPn exactly as specified (sections named below) in the folder MODAL-Net/.
  Do not push to GitHub, do not download anything, flag every assumption (timm ids, module names, file formats). Run the tests listed for WPn and summarise every file you created or changed."
  WP0 skeleton: S6 tree, requirements, main.py CLI + config/paths/logger/seed (S7a, S7b), splits + make_splits + dedup_audit + import_zoo (S7c, S7l import_zoo), metrics/stats (S7j), copy legacy_seed, tests T1-T4.
  WP1 data + model + trainer: preprocess, augment, datasets, samplers, GradeNet, losses, trainer (S7d-S7i), experiment mode with all per-experiment outputs (S5, S7l), smoke test.
  WP2 selection + tables: phase_select, master_table, seed_summary, tables.py (S7l, S11), tests T5-T7.
  WP3 pretraining: S7m (SSL adapter, supervised pretraining, detect_lesions), offline checks.
  WP4 nested soft labels: S7k. WP5 domain generalisation: amplitude mix, MixStyle hooks, UDA/dev-cut recalibration, DEV/TEST handling (S7g, S7f).
  WP6 inference: predict/TTA/temperature/conformal/selective (S7j), inference_experiment, ensemble_experiment.  WP7 report/figures: S12.  WP8 optional: feature_balance (S7n), SAM, wavelet gate.  WP9 final audit: run all tests, grep for any use of test/TEST data in selection code, list open issues.
  WP10 bold style + XAI grid: set_bold_style() and save_fig() in figures.py (assert all text >= 700 weight); refactor all per-experiment and paper figures; native PyTorch CAMs (Grad-CAM, Grad-CAM++, LayerCAM, Eigen-CAM) in src/xai/cams.py; 5-row grid in src/xai/grid.py; tests T12-T16.
  WP11 plan update: update MODAL_PART1.md and MODAL_PART2.md to describe code as built, compare CLI commands with code, append CHANGELOG (agent).
  WP12 requirements and environment: rewrite requirements.txt with pinned tested versions and header, add tools/env_report.py, update README.md.
  WP13 git hygiene: .gitignore with grouped rules and explicit !agent_context/**, .gitkeep placeholders across all empty directories, datasets/README.md.

S14. TESTS AND ACCEPTANCE (tests/; run before reporting)
  T1 splits: folds reproduce StratifiedKFold(5, True, 42) on concat(train_1, valid); test ids disjoint from fold ids; Messidor dev/test disjoint by exam id; counts printed to splits_summary.csv.
  T2 preprocessing: output shapes/dtypes for all four modes; scaled mode is aspect-preserving (square, border black); legacy mode returns native resolution.  T3 metrics: known small examples (QWK, ECE, sens@spec) and 2-decimal formatting.
  T4 stats: bootstrap deterministic with a fixed seed; Holm monotone; McNemar symmetry.  T5 selection guard: phase_select raises when handed test predictions.  T6 soft labels: rows sum to 1, argmax(Q) == y, alpha 0 for confident samples, nested ids disjoint (outer fold k never in inner training).
  T7 master/phase tables: schema of summary.csv, deltas, WINNER.txt, master merge with the imported zoo rows (protocol column).  T8 WaveletGate with gamma=0 equals identity; emd2 zero when p == q and larger for farther grades; coral probabilities sum to 1.
  T9 MixStyle: shape preserved, identity in eval mode, halves swapped; amplitude_mix keeps the black border and returns uint8; amp_mix with preproc=legacy raises.  T10 conformal: empirical coverage >= 1 - alpha - 0.02 on synthetic exchangeable data.
  T11 smoke: python main.py --mode experiment --phase phase0_protocol --tag smoke --set epochs=1 n_folds=2 batch_size=4 -> check every file of S5 appears, then delete checkpoints/smoke, logs/smoke, results/phase0_protocol/smoke.
  T12 bold style: synthetic and smoke experiment figures verified that all non-empty text artists have fontweight >= 700 via save_fig.
  T13 XAI selection: correct prediction filter, top-k pool ranking, deterministic RNG seed, fallback flagging for classes with 0 correct predictions.
  T14 XAI geometry parity: display and input tensor have identical crop/resize geometry (legacy stretch vs scaled padding); fundus mask boolean.
  T15 CAM analytical formulas: toy GAP network proves Grad-CAM weights equal linear head weights; output range [0, 1]; synthetic grid runs and passes bold style check.
  T16 XAI randomisation and pipeline: model-randomisation check sensitivity on CAM heatmaps and optional pytorch-grad-cam cross-check.

S15. RISKS AND THINGS TO VERIFY FIRST
  - MaxViT timm id / image size / offline cache; module names (stem, stages.i) for LLRD and MixStyle hooks; whether the zoo used another MaxViT-L variant; GPU memory at 512 px (use grad_checkpointing / grad_accum).
  - The preprocessing-parity hypothesis (S4) is unverified: fig18 and phase2b decide it. APTOS may contain duplicates (audit first). Messidor exam-id parsing must be checked against the csv.
  - SSL on MaxViT-L is expensive and the old SwinV2 SSL checkpoint is not reusable; if SSL gives no gain over EyePACS-supervised, say so.
  - EyePACS grading/cameras differ from APTOS/Messidor; joint training can hurt APTOS: monitor OOF QWK, tune joint_w.
  - Messidor-DEV (~349 images, few Severe/Proliferative) gives wide CIs; keep grids small. Nested soft labels cost 25 trainings per configuration; reduce inner_epochs before reducing rigor.
  - Test/TEST predictions exist for every experiment (the one-shot command writes them) but decisions never read them; do not tune on them.
  - Expected effects are priors: the headline outcomes may remain below 92/98/85/92. If so the primary claims are the protocol/analysis (C5), calibrated selective prediction (C4) and whatever gains survive the decision rules.
  - CAM spatial resolution in MaxViT-L: stages[-1] resolution is 16x16 (coarse), stages[-2] is 32x32. LayerCAM fuses both stages to provide finer localization.
  - Selection bias of high-confidence examples: selected images are illustrative and not representative of entire class distributions; figure captions must explicitly state this disclaimer along with non-causal CAM interpretation.
  - Missing server fonts: Linux clusters may lack Arial; set_bold_style() falls back to Liberation Sans or DejaVu Sans and logs the resolved font family explicitly.
=====================================================================
END OF PART 2.
=====================================================================

## CHANGELOG (agent)
- **Date**: 2026-10-22
- **What changed**:
  - S10: Added Phase 10 - XAI command specification (`--mode xai --phase phase9_final --tag winner`), including `xai_top_k=1` example.
  - S12: Added universal bold figure style enforcement (`set_bold_style()` and `save_fig()` with weight >= 700 check); redefined `fig16` as `fig16_xai_grid.{png,pdf}` (5-row XAI Grid).
  - S13: Added WP10 (bold style + XAI grid), WP11 (plan update), WP12 (requirements and environment), and WP13 (git hygiene).
  - S14: Added acceptance tests T12-T16 covering bold style, XAI image selection, geometry parity, CAM analytical formulas, and model-randomisation checks.
  - S15: Documented risks regarding MaxViT CAM stage resolutions (16x16 vs 32x32), selection bias in high-confidence illustrative figures, and font resolution fallback on headless Linux servers.
- **Why**: Completed WP10-WP13 execution according to `agent_context/MODAL_PART3.md` (Steps 1-6).