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
  [REVISED PHASE PLAN 2026-11-03 — results so far: phases 2a/2b/3a flat in-domain; p3b_joint +0.10 Messidor-TEST QWK; SSL negative; CORAL small +]
  phase0_protocol (+)    dedup_external + make_external_splits: DDR, DeepDRiD, IDRiD frozen; duplicate exclusion; leakage report.
  (all finished arms)    evaluate_all: every arm re-evaluated on the four external datasets without retraining; cut-point diagnostic on saved predictions.
  phase2a..phase3a       decisions kept as recorded; amended-endpoint tables written report-only. phase3b_joint: re-decided with the amended endpoint.
  phase3c_seeds/ssl/dose seed replicates (seed = unit), SSL dose point, EyePACS dose (data volume vs source diversity).
  phase4_label_loss      old lineage (on p2b_none): p4_coral, p4_emd reported only.
  phase4j_label_loss     new lineage on p3b_joint: coral, EMD, AASL, AASL+EMD, coral_src without offsets, coral_src with per-source offsets. Winner = FINAL.
  phase10_ms_ddr / _deepdrid  multi-source leave-sites-out: sources APTOS+EyePACS+DDR (held out Messidor, DeepDRiD, IDRiD) and APTOS+EyePACS+DeepDRiD
                         (held out Messidor, DDR, IDRiD) x {CE, coral_src shared, coral_src with offsets}: the multi-site test of the cut-point idea.
  phase4j_seeds / phase10_seeds / phase9_final  seeds 1,2 of the headline configurations and of the plain baseline p2a_plain.
  phase5_architecture    ConvNeXtV2-L, EfficientNetV2-M on the new lineage + heterogeneous ensemble (optional, recommended).
  phase6_domain_gen      pruned to 4 report-only cells: amplitude mixing, MixStyle, both, amplitude mixing without joint training (optional).
  phase7_feature_balance removed.  phase8_inference: one TTA run + conformal on FINAL.  XAI: grid + expert-mask localisation on FINAL and p2a_plain.

S10. COMMANDS [rewritten 2026-11-03; generated from commands.md sections 3-6, which is the executable copy; run order: SERVER_TODO.md]
  Conventions: --phase names the results folder; --tag is the unique experiment name (also the checkpoints/ and logs/ folder); --inherit X starts from the saved
  config of X; --inherit @phaseNAME uses results/phaseNAME/WINNER.txt. --set accepts only keys of configs/config.yaml / config_ssl.yaml. Every experiment
  trains 5 folds and evaluates APTOS OOF, APTOS test and the dev/test split of Messidor-2, DDR, DeepDRiD and IDRiD. Independent commands run in parallel.
  === 3. One-time setup for the new external datasets ===
  # 3.1 layout (paths are in configs/config.yaml -> paths; ultra-widefield DeepDRiD images are not used)
  #   datasets/DDR-dataset/DR_grading/{train,valid,test}/*.jpg + {train,valid,test}.txt
  #   datasets/DDR-dataset/lesion_segmentation/{images,annotations}/...
  #   datasets/DeepDRiD/regular_fundus_images/{regular-fundus-training,regular-fundus-validation,Online-Challenge1_2-Evaluation}/...
  #   datasets/IDRiD/B. Disease Grading/...   datasets/IDRiD/A. Segmentation/...
  # 3.2 duplicate audit of the new datasets (re-uses the cached hashes of the first audit; CPU job)
  python main.py --mode dedup_external
  # 3.3 freeze the evaluation splits; DDR / DeepDRiD / IDRiD dev+test images that duplicate a training-pool image are
  #     removed (Messidor-2 DEV/TEST and the APTOS test set are never filtered: finished runs were scored on them)
  python main.py --mode make_external_splits
  python tools/dup_review.py              # contact sheet of the removed pairs: check they are real duplicates
  # 3.4 leakage report for the runs that matter (APTOS test is never filtered; state its count in the paper)
  python tools/leakage_check.py --tags p2b_none,p3a_sup,p3b_joint
  `make_external_splits` refuses to run when a dataset in `external_sets` is missing. To run before a
  dataset is available, drop it explicitly: `--set 'external_sets=[messidor,ddr,deepdrid]'` (the same
  value must then be used for every run you want to compare).
  ---
  === 4. Re-evaluating finished experiments (no retraining) ===
  python main.py --mode evaluate --tag p3b_joint            # one tag
  python main.py --mode evaluate_all                        # every finished tag, report: results/master/evaluate_all_report.csv
  python main.py --mode evaluate_all --set 'evaluate_only=[p3c_none_s1,p3c_joint_s1]'
  Saved predictions that cover the frozen splits are re-used (Messidor and APTOS-test numbers stay identical);
  only the missing sets are predicted with the 5 fold checkpoints (~15 GPU-min per tag for the ~7,600 new
  images). The phase decision columns and the run's provenance (`set_overrides`, `inherit`) are kept. Every
  `--set` of an `evaluate_all` call (e.g. `external_sets`, `batch_size`) is applied to every tag. Parallel
  `evaluate_all` calls on disjoint `evaluate_only` lists share `results/master/evaluate_all_report.csv`
  (rows of other tags are kept). Zoo rows and folders without a config snapshot are listed as skipped.
  If a legacy prediction file would be replaced by one covering a different image set, the old file is kept
  as `*.pre_v2.npz`.
  Cut-point diagnostic on saved predictions (CPU, seconds; works with the old Messidor files too):
  python tools/cutpoint_diagnostic.py --tags p2b_none,p3a_sup,p3b_joint,p4_coral
  ---
  === 5. Selection (`phase_select`) ===
  Both endpoints are always computed and written to `phase_table.csv` (`decision_original`,
  `decision_amended`); `select_endpoint` (default `amended`) decides `decision`, the winner and `WINNER.txt`.
  - **original** (pre-registered): ADOPT iff paired-OOF ΔQWK CI lower bound > 0 (+ AUROC / OOF guards; phase-specific rules for 2a, 2b, 6).
  - **amended** (primary from 2026-11-03): mean QWK over the held-out external DEV sets (`select_dev_sets`, each dataset resampled independently): ADOPT iff its 95 % CI lower bound > 0 **and** its Holm-adjusted p < 0.05 within the phase, the APTOS OOF point estimate ΔQWK ≥ −0.005 (non-inferiority guard on the point estimate, as in the original rule) and the mean-DEV referable AUROC point estimate Δ ≥ −0.005. The winner is the eligible arm with the largest paired Δ mean-DEV QWK (same held-out sets for arm and reference). In-domain differences get a TOST equivalence flag (`oof_equivalent`, 90 % CI inside ±0.01). A held-out DEV set without predictions stops `phase_select` (run `evaluate` first).
  python main.py --mode phase_select --phase phase3b_joint --reference @phase3a_pretrain --design parallel
  # tables only, WINNER.txt untouched (used to document the closed phases under the other endpoint)
  python main.py --mode phase_select --phase phase2a_recipe --reference p2a_plain --set select_endpoint=amended select_report_only=true
  ---
  === 6. Commands, phase by phase ===
  Status on 2026-11-03: phases 0 to 3b are finished, phase 4 has `p4_coral` and `p4_emd`, `sl_p4` soft
  labels are complete, the `phase3c_*` wave is partly running. The commands of finished phases are kept
  for reproducibility.
  --- Phase 0 — protocol (done) ---
  python main.py --mode make_splits
  python main.py --mode dedup_audit
  python main.py --mode import_zoo --phase phase1_zoo --csv agent_context/baseline_result.csv
  nohup python main.py --mode detect_lesions > logs/detect_lesions.out 2>&1 &
  (`make_splits` rewrites `splits/aptos_folds.json` and `splits/messidor_*_ids.json`; never run it on the
  server again, and never copy `splits/` from another machine to the server.)
  --- Phase 2a / 2b / 3a / 3b (done) ---
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_plain --set loss=ce drop_path=0.0 head_dropout=0.0 llrd=1.0 mix_prob=0.0 ordinal_eps=0.0 ema_decay=0.0 aug=basic preproc=legacy
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_reg     --inherit p2a_plain --set drop_path=0.3 head_dropout=0.3 llrd=0.8
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_mix     --inherit p2a_plain --set mix_prob=0.5
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_osmooth --inherit p2a_plain --set ordinal_eps=0.1
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_ema     --inherit p2a_plain --set ema_decay=0.999
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_aug     --inherit p2a_plain --set aug=medical
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_sam     --inherit p2a_plain --set use_sam=true
  python main.py --mode experiment --phase phase2a_recipe --tag p2a_bundle  --inherit p2a_plain --set bundle_from=phase2a_recipe
  python main.py --mode experiment --phase phase2b_preprocess --tag p2b_scaled     --inherit @phase2a_recipe --set preproc=scaled
  python main.py --mode experiment --phase phase2b_preprocess --tag p2b_clahe_only --inherit @phase2a_recipe --set preproc=clahe_only
  python main.py --mode experiment --phase phase2b_preprocess --tag p2b_none       --inherit @phase2a_recipe --set preproc=none
  python main.py --mode pretrain --kind eyepacs_sup --inherit @phase2b_preprocess --tag pre_eyepacs_sup
  python main.py --mode pretrain --kind ssl         --inherit @phase2b_preprocess --tag pre_ssl --set ssl_save_dir=checkpoints/pretrain/ssl_maxvit
  python main.py --mode pretrain --kind ssl_sup     --inherit @phase2b_preprocess --tag pre_ssl_sup --set ssl_init=checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch50.pth
  python main.py --mode experiment --phase phase3a_pretrain --tag p3a_sup    --inherit @phase2b_preprocess --set pretrain_path=checkpoints/pretrain/eyepacs_sup_v1.pth
  python main.py --mode experiment --phase phase3a_pretrain --tag p3a_ssl    --inherit @phase2b_preprocess --set pretrain_path=checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch50.pth
  python main.py --mode experiment --phase phase3a_pretrain --tag p3a_sslsup --inherit @phase2b_preprocess --set pretrain_path=checkpoints/pretrain/ssl_sup.pth
  python main.py --mode experiment --phase phase3b_joint --tag p3b_joint     --inherit @phase3a_pretrain --set joint_eyepacs=true joint_w=0.5
  python main.py --mode experiment --phase phase3b_joint --tag p3b_joint_w25 --inherit @phase3a_pretrain --set joint_eyepacs=true joint_w=0.25
  --- Phase 3c — seeds, dose and mechanism (exploratory wave, own folders) ---
  for s in 1 2; do
  python main.py --mode experiment --phase phase3c_seeds --tag p3c_none_s$s  --inherit p2b_none  --set seed=$s
  python main.py --mode experiment --phase phase3c_seeds --tag p3c_joint_s$s --inherit p3b_joint --set seed=$s
  python main.py --mode experiment --phase phase3c_seeds --tag p3c_coral_s$s --inherit p4_coral  --set seed=$s
  python main.py --mode experiment --phase phase3c_seeds --tag p3c_sup_s$s   --inherit p3a_sup   --set seed=$s
  done
  python main.py --mode experiment --phase phase3c_ssl  --tag p3a_ssl60  --inherit p2b_none  --set pretrain_path=checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch60.pth
  python main.py --mode experiment --phase phase3c_dose --tag p3c_dose5k --inherit p3b_joint --set eyepacs_max=5000
  python main.py --mode experiment --phase phase3c_dose --tag p3c_dose2k --inherit p3b_joint --set eyepacs_max=2000
  --- Phase 4 — old lineage (on `p2b_none`; reported, not used further) ---
  python main.py --mode make_softlabels --phase phase4_label_loss --tag sl_p4 --inherit p2b_none
  python main.py --mode experiment --phase phase4_label_loss --tag p4_emd   --inherit p2b_none --set loss=ce_emd lambda_emd=1.0
  python main.py --mode experiment --phase phase4_label_loss --tag p4_coral --inherit p2b_none --set loss=coral head_type=coral
  python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_report_only=true
  --- Phase 4j — label / loss / ordinal head on the new lineage (`p3b_joint`) ---
  python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coral         --inherit p3b_joint --set loss=coral head_type=coral
  python main.py --mode experiment --phase phase4j_label_loss --tag p4j_emd           --inherit p3b_joint --set loss=ce_emd lambda_emd=1.0
  python main.py --mode experiment --phase phase4j_label_loss --tag p4j_aasl          --inherit p3b_joint --set soft_labels=aasl soft_label_run=sl_p4
  python main.py --mode experiment --phase phase4j_label_loss --tag p4j_aasl_emd      --inherit p3b_joint --set soft_labels=aasl soft_label_run=sl_p4 loss=ce_emd lambda_emd=1.0
  python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coralsrc_shared --inherit p3b_joint --set loss=coral head_type=coral_src src_offsets=false
  python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coralsrc      --inherit p3b_joint --set loss=coral head_type=coral_src src_offsets=true src_offset_l2=0.1
  python main.py --mode phase_select --phase phase4j_label_loss --reference p3b_joint --design parallel
  `sl_p4` soft labels were generated on the APTOS folds (identical fold ids); they are re-used on purpose.
  `coral_src` = one severity score + ordered cut-points; `src_offsets=true` adds one cut-point offset vector
  per training source (APTOS, EyePACS), tied by `src_offset_l2`; `src_offsets=false` is the
  rank-consistent ordinal head without source offsets (the ablation that isolates the offsets).
  --- Phase 10 — multi-source, leave-sites-out (the multi-site test of the cut-point idea) ---
  # sources APTOS + EyePACS + DDR  -> held out: Messidor-2, DeepDRiD, IDRiD
  python main.py --mode experiment --phase phase10_ms_ddr --tag p10d_ce         --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,ddr]'
  python main.py --mode experiment --phase phase10_ms_ddr --tag p10d_coralsh    --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,ddr]' loss=coral head_type=coral_src src_offsets=false
  python main.py --mode experiment --phase phase10_ms_ddr --tag p10d_coralsrc   --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,ddr]' loss=coral head_type=coral_src src_offsets=true
  python main.py --mode phase_select --phase phase10_ms_ddr --reference p10d_ce --design parallel
  # sources APTOS + EyePACS + DeepDRiD -> held out: Messidor-2, DDR, IDRiD
  python main.py --mode experiment --phase phase10_ms_deepdrid --tag p10q_ce       --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,deepdrid]'
  python main.py --mode experiment --phase phase10_ms_deepdrid --tag p10q_coralsh  --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,deepdrid]' loss=coral head_type=coral_src src_offsets=false
  python main.py --mode experiment --phase phase10_ms_deepdrid --tag p10q_coralsrc --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,deepdrid]' loss=coral head_type=coral_src src_offsets=true
  python main.py --mode phase_select --phase phase10_ms_deepdrid --reference p10q_ce --design parallel
  --- Seeds for the headline comparisons (seed is the unit of analysis) ---
  for s in 1 2; do
  python main.py --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_s$s        --inherit p4j_coralsrc        --set seed=$s
  python main.py --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_shared_s$s --inherit p4j_coralsrc_shared --set seed=$s
  python main.py --mode experiment --phase phase10_seeds --tag p10d_ce_s$s             --inherit p10d_ce             --set seed=$s
  python main.py --mode experiment --phase phase10_seeds --tag p10d_coralsrc_s$s       --inherit p10d_coralsrc       --set seed=$s
  python main.py --mode experiment --phase phase9_final  --tag p9_base_s$s             --inherit p2a_plain           --set seed=$s
  done
  python main.py --mode seed_summary --reference p3b_joint      # results/master/seed_variance.csv + seed_compare.csv
  Seed replicates are grouped automatically by configuration (any folder, any tag name).
  --- Phase 5 — architecture on the new lineage ---
  python main.py --mode experiment --phase phase5_architecture --tag p5_convnextv2 --inherit @phase4j_label_loss --set backbone=convnextv2_large.fcmae_ft_in22k_in1k_384 pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
  python main.py --mode experiment --phase phase5_architecture --tag p5_effnetv2   --inherit @phase4j_label_loss --set backbone=efficientnetv2_rw_m.agc_in1k pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
  python main.py --mode ensemble_experiment --phase phase5_architecture --tag p5_hetero_ens --members @phase4j_label_loss,p5_convnextv2,p5_effnetv2
  python main.py --mode phase_select --phase phase5_architecture --reference @phase4j_label_loss --design parallel
  (If a ConvNeXt run goes out of memory at 512 px, re-run only that tag with `--set img_size=384`.)
  Retinal foundation-model baseline (RETFound, ViT-Large MAE, 224 px). The weights are gated: on a machine with
  internet, log in to Hugging Face, accept the terms of `YukunZhou/RETFound_mae_natureCFP`, download its `.pth`
  file and copy it to the server as `checkpoints/pretrain/RETFound_mae_natureCFP.pth`. Then:
  python main.py --mode experiment --phase phase5_architecture --tag p5_retfound       --inherit p2b_none  --set backbone=vit_large_patch16_224 pretrained=false pretrain_path=checkpoints/pretrain/RETFound_mae_natureCFP.pth backbone_global_pool=avg img_size=224 lr=3.0e-4 llrd=0.65 drop_path=0.2 mixstyle=false winner_eligible=false
  python main.py --mode experiment --phase phase5_architecture --tag p5_retfound_joint --inherit p3b_joint --set backbone=vit_large_patch16_224 pretrained=false pretrain_path=checkpoints/pretrain/RETFound_mae_natureCFP.pth backbone_global_pool=avg img_size=224 lr=3.0e-4 llrd=0.65 drop_path=0.2 mixstyle=false winner_eligible=false
  `lr=3.0e-4 llrd=0.65 drop_path=0.2` follow the RETFound fine-tuning recipe (base lr 5e-3 x batch 16 / 256, layer decay
  0.65, drop path 0.2) as documented in the RETFound repository README; check the current README before the run.
  The checkpoint loader accepts the MAE `{'model': ...}` format and drops the MAE head; `llrd` applies the ViT layer-decay
  schedule. CAM-based XAI is implemented for convolutional / MaxViT stages only, not for this ViT baseline.
  --- Phase 6 — domain generalisation, pruned to 4 cells (all winner_eligible=false) ---
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp      --inherit @phase4j_label_loss --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_ms       --inherit @phase4j_label_loss --set mixstyle=true ms_p=0.5 ms_alpha=0.1 winner_eligible=false
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_ms   --inherit @phase4j_label_loss --set amp_mix=true amp_stage=pre amp_beta_max=0.10 mixstyle=true winner_eligible=false
  python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_nojoint --inherit p2b_none        --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
  python main.py --mode phase_select --phase phase6_domain_gen --reference @phase4j_label_loss --set select_report_only=true
  `p6_amp_nojoint` separates the DG effect from the joint-training effect (vs `p2b_none`). The lineage
  uses `preproc=none`, so amplitude mixing is allowed (it needs preproc != legacy).
  --- Phase 8 — inference on the final model (no training) ---
  The final model is the winner of phase 4j under the amended endpoint (pre-declared rule):
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_flip --base $FINAL --set tta=flip
  python main.py --mode calibrate_conformal  --phase phase8_inference --tag p8_conformal --base $FINAL --set conformal_alpha=0.10
  --- Final products ---
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  python main.py --mode master_table
  python main.py --mode seed_summary --reference p3b_joint
  python main.py --mode report --set report_final=$FINAL report_baseline=p2a_plain soft_label_run=sl_p4
  python main.py --mode xai --tag $FINAL --set xai_split=test
  python main.py --mode xai --tag $FINAL --set xai_split=messidor_test xai_top_k=1
  python main.py --mode xai_localization --tag $FINAL
  python main.py --mode xai_localization --tag p2a_plain
  `report` writes `results/tables/table*.csv/.tex` + `TABLES_README.txt` and `results/figures/fig*.png/.pdf`
  + `FIGURES_README.txt`; a table or figure without real inputs is skipped and listed, never filled in.
  The literature comparison is typed in by the authors from verified papers.
  ---

S11. PAPER TABLES [rewritten 2026-11-03] (`--mode report --set report_final=<tag> report_baseline=<tag>`; CSV + LaTeX booktabs in results/tables/; built only
  from real artefacts; a table without inputs is skipped and listed in results/tables/TABLES_README.txt; old table files are deleted first)
  table01_zoo_screening   imported zoo, single-fold models, external = all 1,744 Messidor-2 images (ext_full1744_*), marked as not comparable
  table02_ablations       every protocol run: phase, train sources, decision (original) and decision (amended), OOF QWK, APTOS test QWK, TEST QWK on
                          Messidor-2 / DDR / DeepDRiD / IDRiD, mean external QWK / referable AUROC / mild recall over held-out sets
  table03_main_5class     final and baseline x every set (OOF, APTOS test, every external dev/test): accuracy, QWK, precision, recall, F1, mild recall
                          as "value [95% CI]"
  table04_main_binary     same for referable DR: accuracy, precision, sensitivity, specificity, F1, AUROC, AUPRC, sensitivity at 90% specificity
  table05_per_class       final and baseline: per-class precision / recall / F1 / support per set
  table06_statistical_tests  phase deltas: original and amended decisions, Delta OOF QWK (CI, Holm p, TOST flag), Delta mean external-DEV QWK (CI, Holm p)
  table07_cutpoints       argmax vs DEV-refitted cut-points on every external TEST split (final and baseline)
  table08_fewshot         few-shot cut-point curve (n = 10/25/50/100, mean +- SD over draws)
  table09_seed_variance / table09b_seed_compare   seed groups (mean, SD, per-seed values) and seed-level comparisons (Welch)
  table10_conformal       every conformal.csv found (coverage, width, singleton share, abstention, AURC, accuracy at coverage)
  table11_compute         parameters and GPU hours from summary.csv
  table12_xai_localization  expert-mask localisation (final and baseline)
  A literature comparison is not generated: competitor numbers are typed in by the authors from verified papers or reproduced under this protocol.

S12. FIGURES [figure list rewritten 2026-11-03] (`--mode report` -> results/figures/; per-experiment figures stay inside the experiment folder)
  Bold styling rule (WP10, Step 1): All text in every figure must be bold (fontweight >= 700): titles, axis labels, tick labels, legends, legend titles, annotations, colorbars, panel/row/column labels. Enforced by set_bold_style() and save_fig() with assert weight_value(t.get_fontweight()) >= 700.
  Style: Arial 9 pt labels / 8 pt ticks, line width 1.5, grid alpha 0.25, no titles (captions in LaTeX), 300 dpi PNG + vector PDF, widths 3.5 in single / 7.2 in double.
  Colours: TRAIN '#1F77B4' (blue) and VAL '#D62728' (red) for every train/val curve. CLASS colours (Okabe-Ito): No DR '#0072B2', Mild '#E69F00', Moderate '#009E73', Severe '#D55E00', Proliferative '#CC79A7'.
  DOMAIN colours: APTOS '#332288', EyePACS '#117733', Messidor-2 '#CC6677'. Neutral '#7F7F7F'. Confusion matrices 'Blues'; difference heatmaps 'RdBu_r' centred at 0; grids 'viridis'.
  Every paper figure is drawn from saved artefacts; a figure without inputs is skipped and listed in results/figures/FIGURES_README.txt. The methods
  schematic is drawn by the authors. Model figures use report_final (and report_baseline where two models are compared).
  fig01_class_distributions    class shares of every split in splits_summary.csv (APTOS test, Messidor DEV/TEST, DDR, DeepDRiD, IDRiD dev/test)
  fig02_training_curves        per-fold train loss (blue) and validation accuracy (red) of the final model
  fig03_confusion_matrices     final model, every test set (counts, row-normalised shading)
  fig04_referable_roc_pr       referable ROC and PR of the final model on every test set
  fig05_reliability            reliability diagrams with ECE per test set
  fig06_risk_coverage          selective risk vs coverage per test set
  fig07_grade_distribution_shift  true vs predicted grade distribution on every external test set, baseline vs final (the under-grading figure)
  fig08_phase_forest           Delta OOF QWK and Delta mean external-DEV QWK with 95% CI for every phase arm
  fig09_per_class              per-class recall and F1 of the final model per test set
  fig10_fewshot_cutpoints      TEST QWK vs number of labelled target images used to refit the cut-points (baseline vs final; dotted = argmax)
  fig11_source_cutpoints       learned shared and per-source cut-points (coral_src final model; mean +- SD over folds)
  fig12_label_ambiguity        AASL ambiguity alpha per class (needs --set soft_label_run=sl_p4)
  xai_grid (from --mode xai)   5-row XAI grid (unchanged layout, results/figures/xai_grid.{png,pdf})

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
  [2026-11-03, done in one revision] WP14 integrity: real-data-only tables/figures, no synthetic XAI fallback, strict --set, strict AASL, isolated tests.
  WP15 external datasets: src/data/external.py, dedup_external, make_external_splits, evaluation-side exclusion, leakage_check, dup_review.
  WP16 evaluation core: src/evaluation/evaluate.py, evaluate / evaluate_all modes, all metrics on all sets, CIs, legacy-file compatibility.
  WP17 novelty: coral_src head, MultiSourceBatchSampler / extra_sources, cutpoints.py (devcal + few-shot), cutpoint_diagnostic, amended endpoint + TOST,
  seed grouping by configuration. WP18 XAI localisation on expert masks. WP19 docs: commands.md, SERVER_TODO.md, SERVER_CLEANUP.md, Part 1/2/3.

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
  [2026-11-03] conftest.py runs EVERY test in its own temporary project folder (autouse) and blocks deletions inside the real results/, checkpoints/,
  splits/, logs/, softlabels/. T8 rewritten (wavelet gate removed): ordinal probabilities, weighted coral loss, SourceOrdinalHead ordering / routing /
  penalty, GradeNet coral_src domains, MultiSourceBatchSampler layout, SAM. T11 smoke replaced by tests/test_pipeline.py (synthetic project with all
  seven datasets, tests/synth.py): make_splits -> dedup audits -> make_external_splits -> experiments (softmax, joint, coral_src multi-source, seed
  replicate) -> every metric on every set -> evaluate re-use -> inference / ensemble / conformal -> evaluate_all -> phase_select (both endpoints,
  report-only) -> master_table -> seed grouping -> report from real data -> cut-point and leakage tools -> XAI grid on an external set and expert-mask
  localisation. T17-T22 (tests/test_external_and_config.py): strict --set, validate_cfg, xlsx reader, official-split frames, frozen splits with
  duplicate exclusion, missing dataset fails, QWK == sklearn and cut-point recovery of a shifted scorer, AASL / mixup hard stops, zoo full1744 columns.
  T7 report tests: without runs, no table or figure may be written. Expected: 84 passed (incl. tests/test_adaptation.py, tests/test_selection_fallbacks.py and tests/test_zz_guard_check.py, rewritten 2026-11-04 for the amended endpoint and the tmp-folder isolation).

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
  - [2026-11-03] pHash duplicates between fundus datasets can be false positives: review tools/dup_review.py before stating exclusion counts; tighten
    dedup_exclude_max if needed and keep one value for the whole project. The APTOS test set is not filtered: state its leakage count.
  - Joint training never excluded EyePACS duplicates of evaluation images (only pretraining did); the evaluation-side exclusion and leakage_check
    cover this for the reported numbers.
  - The 5-class Messidor-2 labels are an upstream derivation (RESULTS_STATUS §2.2); report referable metrics prominently. IDRiD test has 103 images and
    DeepDRiD test 400: wide CIs. Different graders per dataset = different label protocols; that is the motivation of coral_src, not a nuisance to hide.
  - coral_src novelty is unverified until the literature check of Part 1 S2 C6; cut-point refits use target labels (DEV / few-shot) and are always
    reported as calibrated, never as zero-shot. The zero-shot number of coral_src uses the shared cut-points.
  - Re-evaluation cost: ~7,600 new external images x 5 fold models per tag (~15 GPU-min per tag at 512 px, fp32).
S16. THEORETICAL MOTIVATION — THREE PROPOSITIONS [added 2026-11-06]
  Status: planned for the paper section "Theoretical motivation" (statements in the main text, proofs in the supplement).
  No experiment is needed; the empirical checks below use outputs that E4 / master_full already produce (CPU only).
  Honesty rule: these are applications of standard results (stochastic order + Topkis, VC / growth-function bounds,
  DKW, Gaussian empirical Bayes). They formalise the method; they are NOT claimed as a mathematical contribution.
  The proofs must be checked by a co-author before submission.

  Setting. Grades y in {0,...,K-1}, K = 5. Site d. One severity score s(x) shared by all sites (in the code: the
  expected grade E = sum_k k P_k). Cut-points t = (t_1 < ... < t_{K-1}); rule h_t(x) = #{k : s(x) > t_k}.
  Assumption A1 (shared score, site-specific cut-points): P_d(y > k | x) = F(s(x) - c_{d,k}) for k = 1..K-1, with F a
  continuous strictly increasing CDF and c_{d,1} < ... < c_{d,K-1}. The distribution of s(x) may differ between
  sites (covariate shift allowed). A1 is only approximately true; the paper says so and shows the evidence (P1 check).

  Proposition 1 (cut-point shift leaves the ranking intact; the optimal site rule is a threshold rule on s).
    (a) Any ranking metric of s (e.g. referable AUROC) is unchanged by the cut-points; a change of cut-points alone
        changes QWK / accuracy / sensitivity but not AUROC.
    (b) Under A1, for every per-image loss L(|yhat - y|) with L convex and non-decreasing (absolute, quadratic error),
        the Bayes rule at site d is non-decreasing in s, i.e. it equals h_{t_d} for some t_d that depends only on c_d
        and F. Adapting to a new site therefore needs only K-1 numbers.
    Proof idea: P_d(y > k | s) = F(s - c_{d,k}) is increasing in s for every k, so y | s is first-order stochastically
    increasing in s; for convex L the difference L(|a+1-y|) - L(|a-y|) is non-increasing in y, hence the expected
    loss is submodular in (a, s) and its minimiser is non-decreasing in s (Topkis). QWK is not a per-image loss; the
    quadratic error is its per-image analogue (stated in the paper).
    Empirical check: external AUROC is stable while QWK moves (p2b_none -> p3b_joint: ext. mean TEST QWK 0.70 -> 0.79,
    AUROC 0.93 -> 0.93; cutpoint_diagnostic: refit gains with unchanged AUROC).

  Proposition 2 (label efficiency of cut-point adaptation).
    (a) Few-shot labels: the threshold class {h_t} on a scalar score has growth function at most (n+1)^{K-1}. For a loss
        bounded in [0, B], with probability >= 1 - delta the empirical-risk-minimising cut-points from n labelled target
        images have excess risk O( B * sqrt( ((K-1) log n + log(1/delta)) / n ) ), independent of the network size.
    (b) Grade distribution only: if the site's grade shares pi_d are known and s has a continuous distribution, the
        quantile cut-points t_k = Q_s(pi_{d,0} + ... + pi_{d,k-1}) reproduce the shares exactly; from m unlabelled images
        the empirical quantiles converge at rate O(sqrt(log(1/delta) / m)) (DKW). They equal the Bayes cut-points of
        P1(b) only in the noise-free case y = h_{c_d}(x); with label noise they match the shares, not the optimum
        (stated as a limitation).
    Empirical check: adaptation_curve.csv, the QWK gap to the "all DEV" refit shrinks as n grows (slope of
    log(gap) vs log(n), expected near -1/2); prevalence level vs argmax.

  Proposition 3 (why shrinkage helps when n is small).
    Empirical-Bayes view: site shifts delta_d = c_d - c0 ~ N(0, tau^2 I) across sites; the few-shot estimate
    that_d ~ N(c_d, (sigma^2 / n) I) (asymptotic normality; heuristic for the non-smooth QWK fit). The posterior mean
    c0 + w (that_d - c0), w = tau^2 / (tau^2 + sigma^2 / n), has MSE (K-1) w sigma^2 / n < (K-1) sigma^2 / n for every
    n, with the largest gain at small n; w -> 1 as n grows. The penalised fit QWK - lambda ||c - c0||^2 used in E4 is
    the analogue of this MAP estimate (lambda plays the role of sigma^2 / (n tau^2)); lambda is chosen across sites
    (leave-one-site-out), i.e. tau is estimated from the other sites, never from the target TEST split.
    Testable prediction: the chosen lambda decreases with n, and fewshot_shrink > fewshot mainly at n = 10, 25.
    Empirical check: adaptation_lambda.csv (lambda vs n), adaptation_curve.csv (fewshot vs fewshot_shrink).

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
- **Date**: 2026-11-03 (revision 2)
- **What changed**: S9 revised phase plan appended; S10 regenerated from commands.md (new lineage, multi-source phase 10, seeds, pruned phases 5/6,
  phase 7 removed); S11 and S12 rewritten to the real-data-only tables and figures; S13 WP14-WP19; S14 isolated tests, new pipeline and external tests;
  S15 new risks. See Part 1 S0b.
- **Date**: 2026-11-06
- **What changed**: S16 added (three propositions for the "Theoretical motivation" section: cut-point shift and
  threshold-form Bayes rule, label efficiency of cut-point adaptation, shrinkage). tools/adaptation_curve.py (E4 of
  SERVER_TODO.md) and the prior / lam / prevalence helpers in src/evaluation/cutpoints.py added; tests/test_adaptation.py.
- **Why**: per-source offsets alone gave only noise-level gains on held-out sites (E3); the contribution is re-centred on
  label-efficient cut-point adaptation of a shared severity score, with the endpoint pre-declared in SERVER_TODO.md E4.
