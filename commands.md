# MODAL-Net — Run Guide (all commands)

This is the reference for every command of the project. **What to run next, in order, is in
`SERVER_TODO.md`**; what can be deleted on the server is in `SERVER_CLEANUP.md`. This guide mirrors
section S10 of `agent_context/MODAL_PART2.md`.

**You do not edit any file between experiments.** Every variation is passed with `--set key=value`.
`--set` now refuses keys that do not exist in `configs/config.yaml` / `configs/config_ssl.yaml` (a typo
stops the run instead of silently training a copy of the parent) and tokens without `=`.

Shell note: quote every value that contains brackets, e.g. `'extra_sources=[eyepacs,ddr]'`
(zsh and some bash settings treat `[...]` as a file pattern).

---

## 1. Environment and submission

```bash
source .venv/bin/activate            # or use env/bin/python directly (see the header of requirements.txt)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python tools/env_report.py
pytest -q                            # ~4 min on CPU; every test runs in its own temporary folder and can
                                     # never touch the real splits/, results/ or checkpoints/
```

On the cluster every training / evaluation command below is wrapped in the scheduler exactly as before:

```bash
# example: the p4j_coralsrc run of section 6
phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_label_loss/p4j_coralsrc.txt -- \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
  env/bin/python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coralsrc \
  --inherit p3b_joint --set loss=coral head_type=coral_src src_offsets=true src_offset_l2=0.1
```

Rules:
- Never run two commands with the same `--tag` at the same time. Re-running a command resumes (finished folds are skipped).
- `--inherit X` needs `checkpoints/X/config_snapshot.yaml`; `--base X` / `--members X` need X **finished**.
- `@phaseNAME` = the tag in `results/phaseNAME/WINNER.txt` (exists only after `phase_select` of that phase).
- Do not delete `checkpoints/<tag>/` of any tag that is used later (`--inherit`, `--base`, `evaluate`).
- Never look at TEST columns to decide anything. Decision viewer (DEV / OOF columns only):
```bash
python -c "import pandas as pd; t=pd.read_csv('results/PHASE/phase_table.csv'); print(t[['experiment','oof_qwk','d_oof_qwk','ci_lo','oof_equivalent','d_dev_mean_qwk','d_dev_mean_qwk_ci_lo','dev_sets_used','decision_original','decision_amended','winner']].to_string(index=False))"
```

---

## 2. What every experiment now produces

Every mode that produces predictions (`experiment`, `inference_experiment`, `ensemble_experiment`,
`evaluate`, `evaluate_all`) evaluates **the same sets with the same metrics**:

| set | what | role |
|---|---|---|
| `oof` | pooled APTOS out-of-fold (3,296) | internal validation |
| `aptos_test` | APTOS held-out test (366, never filtered) | internal test (reporting) |
| `messidor_dev` / `messidor_test` | Messidor-2 exam-grouped DEV (~20 %) / TEST (~80 %) | external (DEV = decisions) |
| `ddr_dev` / `ddr_test` | DDR official valid / test (grade 5 removed) | external |
| `deepdrid_dev` / `deepdrid_test` | DeepDRiD official validation / Challenge-1 evaluation set | external |
| `idrid_dev` / `idrid_test` | IDRiD official training set / testing set | external |

Metrics per set (5-class): accuracy, QWK, macro precision / recall / F1, adjacent accuracy, MAE, mild
recall, predicted-No-DR share, per-class precision / recall / F1. Binary (referable = grades 2-4):
accuracy, precision, sensitivity, specificity, F1, AUROC, AUPRC, sensitivity at 90 / 95 % specificity.
Calibration: ECE, NLL, Brier. 95 % bootstrap CIs (over images) for the main metrics.
Per external dataset also: 4 cut-points refitted on its DEV split and the TEST metrics they give
(`cutpoints.csv`), and the few-shot cut-point curve (`fewshot.csv`, n = 10 / 25 / 50 / 100).

Files: `summary.csv` (one row; legacy columns kept: `test_*` = APTOS test, `ext_*` = Messidor-2 TEST,
`extdev_*` = Messidor-2 DEV; plus `<set>_<metric>` for every set and `ext_mean_test_*` over the held-out
external sets), `metrics_long.csv` (tidy, with CIs and an `in_source` flag), `predictions_<set>.npz`,
`per_class.csv`, `binary.csv`, `calibration.csv`, `confusion_<set>.csv`, `cutpoints.csv`, `fewshot.csv`,
`figures/`. A dataset used for training (`extra_sources`) is marked `in_source` and is excluded from
`ext_mean_*` and from the amended selection endpoint.

---

## 3. One-time setup for the new external datasets

```bash
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
```
`make_external_splits` refuses to run when a dataset in `external_sets` is missing. To run before a
dataset is available, drop it explicitly: `--set 'external_sets=[messidor,ddr,deepdrid]'` (the same
value must then be used for every run you want to compare).

---

## 4. Re-evaluating finished experiments (no retraining)

```bash
python main.py --mode evaluate --tag p3b_joint            # one tag
python main.py --mode evaluate_all                        # every finished tag, report: results/master/evaluate_all_report.csv
python main.py --mode evaluate_all --set 'evaluate_only=[p3c_none_s1,p3c_joint_s1]'
```
Saved predictions that cover the frozen splits are re-used (Messidor and APTOS-test numbers stay identical);
only the missing sets are predicted with the 5 fold checkpoints (~15 GPU-min per tag for the ~7,600 new
images). The phase decision columns and the run's provenance (`set_overrides`, `inherit`) are kept. Every
`--set` of an `evaluate_all` call (e.g. `external_sets`, `batch_size`) is applied to every tag. Parallel
`evaluate_all` calls on disjoint `evaluate_only` lists share `results/master/evaluate_all_report.csv`
(rows of other tags are kept). Zoo rows and folders without a config snapshot are listed as skipped.
If a legacy prediction file would be replaced by one covering a different image set, the old file is kept
as `*.pre_v2.npz`.

Cut-point diagnostic on saved predictions (CPU, seconds; works with the old Messidor files too):
```bash
python tools/cutpoint_diagnostic.py --tags p2b_none,p3a_sup,p3b_joint,p4_coral
```

---

## 5. Selection (`phase_select`)

Both endpoints are always computed and written to `phase_table.csv` (`decision_original`,
`decision_amended`); `select_endpoint` (default `amended`) decides `decision`, the winner and `WINNER.txt`.

- **original** (pre-registered): ADOPT iff paired-OOF ΔQWK CI lower bound > 0 (+ AUROC / OOF guards; phase-specific rules for 2a, 2b, 6).
- **amended** (primary from 2026-11-03): mean QWK over the held-out external DEV sets (`select_dev_sets`, each dataset resampled independently): ADOPT iff its 95 % CI lower bound > 0 **and** its Holm-adjusted p < 0.05 within the phase, the APTOS OOF point estimate ΔQWK ≥ −0.005 (non-inferiority guard on the point estimate, as in the original rule) and the mean-DEV referable AUROC point estimate Δ ≥ −0.005. The winner is the eligible arm with the largest paired Δ mean-DEV QWK (same held-out sets for arm and reference). In-domain differences get a TOST equivalence flag (`oof_equivalent`, 90 % CI inside ±0.01). A held-out DEV set without predictions stops `phase_select` (run `evaluate` first).

```bash
python main.py --mode phase_select --phase phase3b_joint --reference @phase3a_pretrain --design parallel
# tables only, WINNER.txt untouched (used to document the closed phases under the other endpoint)
python main.py --mode phase_select --phase phase2a_recipe --reference p2a_plain --set select_endpoint=amended select_report_only=true
```

---

## 6. Commands, phase by phase

Status on 2026-11-03: phases 0 to 3b are finished, phase 4 has `p4_coral` and `p4_emd`, `sl_p4` soft
labels are complete, the `phase3c_*` wave is partly running. The commands of finished phases are kept
for reproducibility.

### Phase 0 — protocol (done)
```bash
python main.py --mode make_splits
python main.py --mode dedup_audit
python main.py --mode import_zoo --phase phase1_zoo --csv agent_context/baseline_result.csv
nohup python main.py --mode detect_lesions > logs/detect_lesions.out 2>&1 &
```
(`make_splits` rewrites `splits/aptos_folds.json` and `splits/messidor_*_ids.json`; never run it on the
server again, and never copy `splits/` from another machine to the server.)

### Phase 2a / 2b / 3a / 3b (done)
```bash
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
```

### Phase 3c — seeds, dose and mechanism (exploratory wave, own folders)
```bash
for s in 1 2; do
python main.py --mode experiment --phase phase3c_seeds --tag p3c_none_s$s  --inherit p2b_none  --set seed=$s
python main.py --mode experiment --phase phase3c_seeds --tag p3c_joint_s$s --inherit p3b_joint --set seed=$s
python main.py --mode experiment --phase phase3c_seeds --tag p3c_coral_s$s --inherit p4_coral  --set seed=$s
python main.py --mode experiment --phase phase3c_seeds --tag p3c_sup_s$s   --inherit p3a_sup   --set seed=$s
done
python main.py --mode experiment --phase phase3c_ssl  --tag p3a_ssl60  --inherit p2b_none  --set pretrain_path=checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch60.pth
python main.py --mode experiment --phase phase3c_dose --tag p3c_dose5k --inherit p3b_joint --set eyepacs_max=5000
python main.py --mode experiment --phase phase3c_dose --tag p3c_dose2k --inherit p3b_joint --set eyepacs_max=2000
```

### Phase 4 — old lineage (on `p2b_none`; reported, not used further)
```bash
python main.py --mode make_softlabels --phase phase4_label_loss --tag sl_p4 --inherit p2b_none
python main.py --mode experiment --phase phase4_label_loss --tag p4_emd   --inherit p2b_none --set loss=ce_emd lambda_emd=1.0
python main.py --mode experiment --phase phase4_label_loss --tag p4_coral --inherit p2b_none --set loss=coral head_type=coral
python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_report_only=true
```

### Phase 4j — label / loss / ordinal head on the new lineage (`p3b_joint`)
```bash
python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coral         --inherit p3b_joint --set loss=coral head_type=coral
python main.py --mode experiment --phase phase4j_label_loss --tag p4j_emd           --inherit p3b_joint --set loss=ce_emd lambda_emd=1.0
python main.py --mode experiment --phase phase4j_label_loss --tag p4j_aasl          --inherit p3b_joint --set soft_labels=aasl soft_label_run=sl_p4
python main.py --mode experiment --phase phase4j_label_loss --tag p4j_aasl_emd      --inherit p3b_joint --set soft_labels=aasl soft_label_run=sl_p4 loss=ce_emd lambda_emd=1.0
python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coralsrc_shared --inherit p3b_joint --set loss=coral head_type=coral_src src_offsets=false
python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coralsrc      --inherit p3b_joint --set loss=coral head_type=coral_src src_offsets=true src_offset_l2=0.1
python main.py --mode phase_select --phase phase4j_label_loss --reference p3b_joint --design parallel
```
`sl_p4` soft labels were generated on the APTOS folds (identical fold ids); they are re-used on purpose.
`coral_src` = one severity score + ordered cut-points; `src_offsets=true` adds one cut-point offset vector
per training source (APTOS, EyePACS), tied by `src_offset_l2`; `src_offsets=false` is the
rank-consistent ordinal head without source offsets (the ablation that isolates the offsets).

### Phase 10 — multi-source, leave-sites-out (the multi-site test of the cut-point idea)
```bash
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
```

### Seeds for the headline comparisons (seed is the unit of analysis)
```bash
for s in 1 2; do
python main.py --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_s$s        --inherit p4j_coralsrc        --set seed=$s
python main.py --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_shared_s$s --inherit p4j_coralsrc_shared --set seed=$s
python main.py --mode experiment --phase phase10_seeds --tag p10d_ce_s$s             --inherit p10d_ce             --set seed=$s
python main.py --mode experiment --phase phase10_seeds --tag p10d_coralsrc_s$s       --inherit p10d_coralsrc       --set seed=$s
python main.py --mode experiment --phase phase9_final  --tag p9_base_s$s             --inherit p2a_plain           --set seed=$s
done
python main.py --mode seed_summary --reference p3b_joint      # results/master/seed_variance.csv + seed_compare.csv
```
Seed replicates are grouped automatically by configuration (any folder, any tag name).

### Phase 5 — architecture on the new lineage
```bash
python main.py --mode experiment --phase phase5_architecture --tag p5_convnextv2 --inherit @phase4j_label_loss --set backbone=convnextv2_large.fcmae_ft_in22k_in1k_384 pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
python main.py --mode experiment --phase phase5_architecture --tag p5_effnetv2   --inherit @phase4j_label_loss --set backbone=efficientnetv2_rw_m.agc_in1k pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
python main.py --mode ensemble_experiment --phase phase5_architecture --tag p5_hetero_ens --members @phase4j_label_loss,p5_convnextv2,p5_effnetv2
python main.py --mode phase_select --phase phase5_architecture --reference @phase4j_label_loss --design parallel
```
(If a ConvNeXt run goes out of memory at 512 px, re-run only that tag with `--set img_size=384`.)

Retinal foundation-model baseline (RETFound, ViT-Large MAE, 224 px). The weights are gated: on a machine with
internet, log in to Hugging Face, accept the terms of `YukunZhou/RETFound_mae_natureCFP`, download its `.pth`
file and copy it to the server as `checkpoints/pretrain/RETFound_mae_natureCFP.pth`. Then:
```bash
python main.py --mode experiment --phase phase5_architecture --tag p5_retfound       --inherit p2b_none  --set backbone=vit_large_patch16_224 pretrained=false pretrain_path=checkpoints/pretrain/RETFound_mae_natureCFP.pth backbone_global_pool=avg img_size=224 lr=3.0e-4 llrd=0.65 drop_path=0.2 mixstyle=false winner_eligible=false
python main.py --mode experiment --phase phase5_architecture --tag p5_retfound_joint --inherit p3b_joint --set backbone=vit_large_patch16_224 pretrained=false pretrain_path=checkpoints/pretrain/RETFound_mae_natureCFP.pth backbone_global_pool=avg img_size=224 lr=3.0e-4 llrd=0.65 drop_path=0.2 mixstyle=false winner_eligible=false
```
`lr=3.0e-4 llrd=0.65 drop_path=0.2` follow the RETFound fine-tuning recipe (base lr 5e-3 x batch 16 / 256, layer decay
0.65, drop path 0.2) as documented in the RETFound repository README; check the current README before the run.
The checkpoint loader accepts the MAE `{'model': ...}` format and drops the MAE head; `llrd` applies the ViT layer-decay
schedule. CAM-based XAI is implemented for convolutional / MaxViT stages only, not for this ViT baseline.

### Phase 6 — domain generalisation, pruned to 4 cells (all winner_eligible=false)
```bash
python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp      --inherit @phase4j_label_loss --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
python main.py --mode experiment --phase phase6_domain_gen --tag p6_ms       --inherit @phase4j_label_loss --set mixstyle=true ms_p=0.5 ms_alpha=0.1 winner_eligible=false
python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_ms   --inherit @phase4j_label_loss --set amp_mix=true amp_stage=pre amp_beta_max=0.10 mixstyle=true winner_eligible=false
python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_nojoint --inherit p2b_none        --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
python main.py --mode phase_select --phase phase6_domain_gen --reference @phase4j_label_loss --set select_report_only=true
```
`p6_amp_nojoint` separates the DG effect from the joint-training effect (vs `p2b_none`). The lineage
uses `preproc=none`, so amplitude mixing is allowed (it needs preproc != legacy).

### Phase 8 — inference on the final model (no training)
The final model is the winner of phase 4j under the amended endpoint (pre-declared rule):
```bash
FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_flip --base $FINAL --set tta=flip
python main.py --mode calibrate_conformal  --phase phase8_inference --tag p8_conformal --base $FINAL --set conformal_alpha=0.10
```

### Final products
```bash
FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
python main.py --mode master_table
python main.py --mode seed_summary --reference p3b_joint
python main.py --mode report --set report_final=$FINAL report_baseline=p2a_plain soft_label_run=sl_p4
python main.py --mode xai --tag $FINAL --set xai_split=test
python main.py --mode xai --tag $FINAL --set xai_split=messidor_test xai_top_k=1
python main.py --mode xai_localization --tag $FINAL
python main.py --mode xai_localization --tag p2a_plain
```
`report` writes `results/tables/table*.csv/.tex` + `TABLES_README.txt` and `results/figures/fig*.png/.pdf`
+ `FIGURES_README.txt`; a table or figure without real inputs is skipped and listed, never filled in.
The literature comparison is typed in by the authors from verified papers.

---

## 7. If something goes wrong

| Message / symptom | Fix |
|---|---|
| `--set key ... does not exist` | typo in the key; the message lists close matches |
| `Evaluation inputs missing` before training starts | run `make_external_splits`, or check `paths` in `configs/config.yaml` |
| `AASL soft labels for fold k not found` | wait until `softlabels/<run>/outer0..4.npz` all exist |
| `coral / coral_src heads ... set mix_prob=0` | ordinal heads cannot use Mixup/CutMix (the lineage already has `mix_prob=0`) |
| `ensemble member X has no predictions for <set>` | `python main.py --mode evaluate --tag X` first |
| out of memory | add `--set batch_size=8 grad_accum=2 grad_checkpointing=true` and re-run (finished folds are skipped) |
| `@phaseX` / `WINNER.txt` not found | run `phase_select` for that phase first |
| `phase_select` raises about test data | expected guard; do not bypass it |
| `--mode pretrain` segfaults right after `Config snapshot saved` | read `results/phase0_protocol/audit/duplicates.csv` with `dtype=str` (already done in the code) |
| foreign `/tmp/*.py` shadows a stdlib module | never launch project code from `/tmp`; run from the repo with `env/bin/python` |
| a short pretraining run overwrote `checkpoints/pretrain/eyepacs_sup.pth` | restore it from `eyepacs_sup_v1.pth` / the backup before any run that uses it |
| a results folder of a phase lost while `checkpoints/` survived | do NOT re-run `experiment` (it retrains); use 7A |

### 7A. Rebuilding a lost results folder from checkpoints (inference only)
One job per tag of the lost phase, e.g. for `p2a_reg` of `phase2a_recipe`:
```bash
phd run -p shr_gpu -ng 1 -GR H100 -l rb_p2a_reg.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 env/bin/python tools/rebuild_oof.py --tag p2a_reg --phase phase2a_recipe
```
It regenerates every `oof_foldK.npz` from `foldK.pth` (exactly the trainer's path) and resumes
`run_experiment()`, which then writes every evaluation set and metric. Re-run that phase's
`phase_select` afterwards; never write `WINNER.txt` by hand.

Back up after every phase (megabytes, not GPU hours):
```bash
mkdir -p backups && tar czf backups/results_$(date +%Y%m%d_%H%M).tgz results/ splits/ softlabels/ checkpoints/*/config_snapshot.yaml checkpoints/*/thresholds.json
```
