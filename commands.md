# MODAL-Net — Combined Run Guide (replaces commands.md and run_helper.md)

This guide mirrors section S10 of `agent_context/MODAL_PART2.md`. If the agent renamed a flag while building the code, the agent-updated S10 wins; compare once before the first run.

**Main idea: you do not edit any file between experiments.** Every variation is passed with `--set key=value`. Never change the defaults inside `configs/config.yaml` between runs, because inheritance and snapshots assume stable defaults.

---

## 1. One-time setup (do once, in this order)

### 1A. Environment
```bash
source .venv/bin/activate   (see the header of requirements.txt)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python tools/env_report.py                # compare versions with the header of requirements.txt; check the MaxViT/ConvNeXt/EfficientNet ids can be created offline
pytest -q                                 # fast checks (splits, metrics, selection guard); stop if anything fails
```

### 1B. The only places where you may need to edit a file (line numbers are unknown because the agent writes the code; use grep to find the line)

| When | File | What to find | What to change |
|---|---|---|---|
| Before the first run, if the zoo used another MaxViT-L variant | `configs/config.yaml` | `grep -n "backbone" configs/config.yaml` | set `backbone:` to the exact timm id used in the zoo (and `img_size`, `img_size_arg` only if that id needs it) |
| If you get out-of-memory | none (use a flag) | — | add `--set batch_size=8 grad_accum=2 grad_checkpointing=true` to the command (effective batch stays 16) |
| Before phase 3a (SSL) | `configs/config_ssl.yaml` | `grep -n "ssl_backbone\|ssl_batch_size\|ssl_save_dir\|ssl_epochs" configs/config_ssl.yaml` | `ssl_backbone` = same timm id as `backbone`; `ssl_batch_size` = what fits on your GPU; keep `ssl_save_dir: checkpoints/pretrain/ssl_maxvit` |
| Only if the Messidor exam grouping looks wrong (check below) | `src/data/splits.py` | `grep -n "split('_')" src/data/splits.py` (inside `make_messidor_split`) | change the expression that builds the exam id from `id_code` |
| Phase 6 and phase 8 | none | — | replace the placeholders `<BEST_STAGE>`, `<BEST_BETA>`, `<BEST_P6_TAG>`, `<BEST_TTA>` in the commands (how: section 5) |

Check the Messidor exam grouping before `make_splits` (Messidor-2 has two images per exam, so the number of groups should be about half the number of images):
```bash
python - <<'EOF'
import pandas as pd
d = pd.read_csv('datasets/Messidor_2/messidor_data.csv'); d = d[d['adjudicated_gradable'] == 1]
g = d['id_code'].astype(str).str.split('_').str[:2].str.join('_')
print(len(d), 'images ->', g.nunique(), 'groups'); print(d['id_code'].head(5).tolist())
EOF
```

---

## 2. How to launch and rules

- Long runs: use `tmux`, or `nohup <command> > logs/<name>.out 2>&1 &`. Example: `nohup python main.py --mode experiment --phase phase2a_recipe --tag p2a_plain --set ... > logs/p2a_plain.out 2>&1 &`
- Parallel on several GPUs: `CUDA_VISIBLE_DEVICES=1 nohup python main.py ... &` (one command per GPU).
- **Never run two commands with the same `--tag` at the same time.** Re-running the same command resumes: finished folds are skipped.
- A command that uses `--inherit X`, `--base X` or `--members X` needs `X` to exist: `--inherit` only needs `checkpoints/X/config_snapshot.yaml` (written when X starts); `--base` and `--members` need X **finished**.
- `@phaseNAME` means "the winner of that phase" (`results/phaseNAME/WINNER.txt`). It exists only after `phase_select` of that phase has run.
- Do not delete `checkpoints/<tag>/` of any tag used later as `--inherit` or `--base`.
- Never look at test columns to decide anything. Use the viewer below (it prints only validation / OOF / Messidor-DEV columns):
```bash
python -c "import pandas as pd; t=pd.read_csv('results/PHASE/phase_table.csv'); print(t[['experiment','oof_qwk','d_oof_qwk','ci_lo','ci_hi','p_holm','extdev_qwk','extdev_mild_recall','decision']].to_string(index=False))"
```
(replace `PHASE`; if a column name differs, print `t.columns` and use the matching names).
- Watch a run: `tail -f logs/<tag>/experiment_*.log`; folds finished: `ls checkpoints/<tag>/`.

Legend used below: **[PAR]** = these commands are independent, run them together (one per GPU) or one after another; **[WAIT]** = wait until everything above it has finished; **[SEQ]** = must run in this order.

---

## 3. Dependency map

| Phase | Needs before it starts | Inside the phase |
|---|---|---|
| 0 setup | environment | sequential (detect_lesions can run in the background) |
| 2a recipe | phase 0 | `p2a_plain` first, then the single-factor runs in parallel, then select, bundle, select |
| 2b preprocess | `WINNER` of 2a | 3 runs in parallel, then select |
| 3a pretrain | `WINNER` of 2b, `detect_lesions` finished, `dedup_audit` done | pretraining runs, then 3 experiments (each waits for its own checkpoint), then select |
| 3b joint | `WINNER` of 3a | 2 runs in parallel, then select |
| 4 labels/loss | `WINNER` of 3b | `make_softlabels` and EMD/CORAL in parallel; the two soft-label experiments wait for `make_softlabels`; then select |
| 5 architecture | `WINNER` of 4 | 3 runs in parallel; ensemble waits for the two backbone runs; then select |
| 6 domain gen | `WINNER` of 5 | amplitude/MixStyle runs in parallel, choose best, then amp+MixStyle and UDA, then dev-calibration, then select |
| 7 feature balance (optional) | `WINNER` of 6 | 5 runs in parallel, select |
| 8 inference | `WINNER` of 6 | 4 TTA runs in parallel, then temperature, then conformal, then select |
| 9 final | `WINNER` of 6 and 8, `p2a_plain` finished | 4 trainings in parallel, 2 inference runs, then summaries |
| master, report, XAI | phase 9 | sequential |

---

## 4. Commands, phase by phase

### Phase 0 — protocol (once)
```bash
python main.py --mode make_splits                                              # [SEQ] 1
pytest -q                                                                      # [SEQ] 2
python main.py --mode dedup_audit                                              # [SEQ] 3 (review results/phase0_protocol/audit/duplicates.csv before phase 3)
python main.py --mode import_zoo --phase phase1_zoo --csv agent_context/baseline_result.csv   # [SEQ] 4
nohup python main.py --mode detect_lesions > logs/detect_lesions.out 2>&1 &    # [SEQ] 5, CPU job: let it run in the background during phase 2; it must finish before pre_ssl (phase 3a)
```

### Phase 2a — recipe (plain baseline = the zoo recipe)
```bash
# [SEQ] start this first; the other runs can start as soon as checkpoints/p2a_plain/config_snapshot.yaml exists
python main.py --mode experiment --phase phase2a_recipe --tag p2a_plain --set loss=ce drop_path=0.0 head_dropout=0.0 llrd=1.0 mix_prob=0.0 ordinal_eps=0.0 ema_decay=0.0 aug=basic preproc=legacy

# [PAR] single-factor runs
python main.py --mode experiment --phase phase2a_recipe --tag p2a_reg     --inherit p2a_plain --set drop_path=0.3 head_dropout=0.3 llrd=0.8
python main.py --mode experiment --phase phase2a_recipe --tag p2a_mix     --inherit p2a_plain --set mix_prob=0.5
python main.py --mode experiment --phase phase2a_recipe --tag p2a_osmooth --inherit p2a_plain --set ordinal_eps=0.1
python main.py --mode experiment --phase phase2a_recipe --tag p2a_ema     --inherit p2a_plain --set ema_decay=0.999
python main.py --mode experiment --phase phase2a_recipe --tag p2a_aug     --inherit p2a_plain --set aug=medical
python main.py --mode experiment --phase phase2a_recipe --tag p2a_sam     --inherit p2a_plain --set use_sam=true       # optional

# [WAIT] p2a_plain and every single-factor run finished
python main.py --mode phase_select --phase phase2a_recipe --reference p2a_plain --design parallel
python main.py --mode experiment --phase phase2a_recipe --tag p2a_bundle --inherit p2a_plain --set bundle_from=phase2a_recipe
# [WAIT] p2a_bundle finished
python main.py --mode phase_select --phase phase2a_recipe --reference p2a_plain --design parallel
cat results/phase2a_recipe/WINNER.txt
```
If no factor is ADOPT/TENTATIVE, the winner is `p2a_plain`; continue anyway, it is a valid result.

### Phase 2b — preprocessing parity (reference = the 2a winner; no retraining of the reference)
```bash
# [WAIT] WINNER.txt of phase2a exists
# [PAR]
python main.py --mode experiment --phase phase2b_preprocess --tag p2b_scaled     --inherit @phase2a_recipe --set preproc=scaled
python main.py --mode experiment --phase phase2b_preprocess --tag p2b_clahe_only --inherit @phase2a_recipe --set preproc=clahe_only
python main.py --mode experiment --phase phase2b_preprocess --tag p2b_none       --inherit @phase2a_recipe --set preproc=none
# [WAIT]
python main.py --mode phase_select --phase phase2b_preprocess --reference @phase2a_recipe --design parallel
```

### Phase 3a — pretraining
```bash
# [WAIT] WINNER.txt of phase2b exists, detect_lesions finished, duplicates reviewed
# [PAR] these two pretrainings are independent (both are long and GPU heavy)
python main.py --mode pretrain --kind eyepacs_sup --inherit @phase2b_preprocess --tag pre_eyepacs_sup
python main.py --mode pretrain --kind ssl         --inherit @phase2b_preprocess --tag pre_ssl --set ssl_save_dir=checkpoints/pretrain/ssl_maxvit

# [WAIT for pre_ssl] then:
python main.py --mode pretrain --kind ssl_sup     --inherit @phase2b_preprocess --tag pre_ssl_sup --set ssl_init=checkpoints/pretrain/ssl_maxvit/ssl_pretrained_backbone.pth

# each experiment waits only for its own checkpoint (they can overlap with the pretraining of the others)
# [WAIT for checkpoints/pretrain/eyepacs_sup.pth]
python main.py --mode experiment --phase phase3a_pretrain --tag p3a_sup    --inherit @phase2b_preprocess --set pretrain_path=checkpoints/pretrain/eyepacs_sup.pth
# [WAIT for checkpoints/pretrain/ssl_maxvit/ssl_pretrained_backbone.pth]
python main.py --mode experiment --phase phase3a_pretrain --tag p3a_ssl    --inherit @phase2b_preprocess --set pretrain_path=checkpoints/pretrain/ssl_maxvit/ssl_pretrained_backbone.pth
# [WAIT for checkpoints/pretrain/ssl_sup.pth]
python main.py --mode experiment --phase phase3a_pretrain --tag p3a_sslsup --inherit @phase2b_preprocess --set pretrain_path=checkpoints/pretrain/ssl_sup.pth

# [WAIT] all three experiments finished
python main.py --mode phase_select --phase phase3a_pretrain --reference @phase2b_preprocess --design parallel
```

### Phase 3b — joint APTOS + EyePACS fine-tuning
```bash
# [WAIT] WINNER.txt of phase3a exists. [PAR]
python main.py --mode experiment --phase phase3b_joint --tag p3b_joint     --inherit @phase3a_pretrain --set joint_eyepacs=true joint_w=0.5
python main.py --mode experiment --phase phase3b_joint --tag p3b_joint_w25 --inherit @phase3a_pretrain --set joint_eyepacs=true joint_w=0.25     # optional
# [WAIT]
python main.py --mode phase_select --phase phase3b_joint --reference @phase3a_pretrain --design parallel
```

### Phase 4 — labels and loss (soft labels are nested: 5 x 4 inner trainings, the longest step of this phase)
```bash
# [WAIT] WINNER.txt of phase3b exists
# [PAR] these three are independent
python main.py --mode make_softlabels --phase phase4_label_loss --tag sl_p4 --inherit @phase3b_joint
python main.py --mode experiment --phase phase4_label_loss --tag p4_emd   --inherit @phase3b_joint --set loss=ce_emd lambda_emd=1.0
python main.py --mode experiment --phase phase4_label_loss --tag p4_coral --inherit @phase3b_joint --set loss=coral head_type=coral
# [WAIT for make_softlabels] (check that softlabels/sl_p4/outer0.npz ... outer4.npz exist) then [PAR]
python main.py --mode experiment --phase phase4_label_loss --tag p4_aasl     --inherit @phase3b_joint --set soft_labels=aasl soft_label_run=sl_p4
python main.py --mode experiment --phase phase4_label_loss --tag p4_aasl_emd --inherit @phase3b_joint --set soft_labels=aasl soft_label_run=sl_p4 loss=ce_emd lambda_emd=1.0
# [WAIT] all four finished; afterwards checkpoints/sl_p4/inner/ can be deleted
python main.py --mode phase_select --phase phase4_label_loss --reference @phase3b_joint --design parallel
```

### Phase 5 — architecture and heterogeneous ensemble
```bash
# [WAIT] WINNER.txt of phase4 exists. [PAR]
python main.py --mode experiment --phase phase5_architecture --tag p5_wghr       --inherit @phase4_label_loss --set use_wghr=true
python main.py --mode experiment --phase phase5_architecture --tag p5_convnextv2 --inherit @phase4_label_loss --set backbone=convnextv2_large.fcmae_ft_in22k_in1k_384 pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
python main.py --mode experiment --phase phase5_architecture --tag p5_effnetv2   --inherit @phase4_label_loss --set backbone=efficientnetv2_rw_m.agc_in1k pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
# [WAIT for p5_convnextv2 and p5_effnetv2]
python main.py --mode ensemble_experiment --phase phase5_architecture --tag p5_hetero_ens --members @phase4_label_loss,p5_convnextv2,p5_effnetv2 --set winner_eligible=false
# [WAIT] all finished
python main.py --mode phase_select --phase phase5_architecture --reference @phase4_label_loss --design parallel
```

### Phase 6 — domain generalisation (chosen on Messidor-DEV)
```bash
# [WAIT] WINNER.txt of phase5 exists. [PAR]
# p6_ref is needed only when the winning config still uses preproc=legacy (amplitude mixing needs preproc != legacy); check: grep -n "preproc" checkpoints/$(cut -d' ' -f1 results/phase5_architecture/WINNER.txt)/config_snapshot.yaml
python main.py --mode experiment --phase phase6_domain_gen --tag p6_ref          --inherit @phase5_architecture --set preproc=scaled amp_mix=false
python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_pre_b03  --inherit @phase5_architecture --set preproc=scaled amp_mix=true amp_stage=pre  amp_beta_max=0.03
python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_pre_b10  --inherit @phase5_architecture --set preproc=scaled amp_mix=true amp_stage=pre  amp_beta_max=0.10
python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_pre_b20  --inherit @phase5_architecture --set preproc=scaled amp_mix=true amp_stage=pre  amp_beta_max=0.20
python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_post_b10 --inherit @phase5_architecture --set preproc=scaled amp_mix=true amp_stage=post amp_beta_max=0.10
python main.py --mode experiment --phase phase6_domain_gen --tag p6_ms           --inherit @phase5_architecture --set preproc=scaled joint_eyepacs=true mixstyle=true ms_p=0.5 ms_alpha=0.1

# [WAIT] all six finished. Look at the table (section 5) and note <BEST_STAGE> and <BEST_BETA> (best amplitude run on DEV)
python main.py --mode phase_select --phase phase6_domain_gen --reference p6_ref --design parallel      # reference = @phase5_architecture if p6_ref was not needed

# [PAR] after you filled the placeholders
python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_ms --inherit @phase5_architecture --set preproc=scaled joint_eyepacs=true mixstyle=true amp_mix=true amp_stage=<BEST_STAGE> amp_beta_max=<BEST_BETA>
python main.py --mode experiment --phase phase6_domain_gen --tag p6_uda    --inherit @phase5_architecture --set preproc=scaled uda_style_pool=messidor_dev amp_mix=true amp_stage=<BEST_STAGE> amp_beta_max=<BEST_BETA> winner_eligible=false     # labelled UDA, not DG

# [WAIT for p6_amp_ms], then note <BEST_P6_TAG> (best non-UDA run on DEV)
python main.py --mode inference_experiment --phase phase6_domain_gen --tag p6_devcal --base <BEST_P6_TAG> --set recalibrate=devcut winner_eligible=false      # cut points fitted on Messidor-DEV labels, labelled "dev-calibrated"
# [WAIT] p6_uda and p6_devcal finished
python main.py --mode phase_select --phase phase6_domain_gen --reference p6_ref --design parallel
```

### Phase 7 — feature-level balancing (optional)
```bash
# [WAIT] WINNER.txt of phase6 exists. [PAR]
python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_none     --base @phase6_domain_gen --method none
python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_reweight --base @phase6_domain_gen --method reweight
python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_smote    --base @phase6_domain_gen --method smote
python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_gauss    --base @phase6_domain_gen --method gauss
python main.py --mode feature_balance --phase phase7_feature_balance --tag p7_lfsd     --base @phase6_domain_gen --method lfsd
# [WAIT]
python main.py --mode phase_select --phase phase7_feature_balance --reference p7_none --design parallel
```

### Phase 8 — inference (no training)
```bash
# [WAIT] WINNER.txt of phase6 exists. [PAR]
python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_none    --base @phase6_domain_gen --set tta=none temperature_scaling=false
python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_flip    --base @phase6_domain_gen --set tta=flip temperature_scaling=false
python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_fliprot --base @phase6_domain_gen --set tta=flip_rot temperature_scaling=false
python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_scale   --base @phase6_domain_gen --set tta=flip_rot 'tta_scales=[0.9,1.0,1.1]' temperature_scaling=false
# [WAIT] all four finished; choose <BEST_TTA> (best Messidor-DEV QWK; if the differences are tiny choose the one with fewer views)
python main.py --mode inference_experiment --phase phase8_inference --tag p8_temp --base @phase6_domain_gen --set tta=<BEST_TTA> temperature_scaling=true
# [WAIT]
python main.py --mode calibrate_conformal --phase phase8_inference --tag p8_conformal --base p8_temp --set conformal_alpha=0.10
python main.py --mode phase_select --phase phase8_inference --reference p8_tta_none --design parallel
```

### Phase 9 — final (seeds) and summaries
```bash
# [WAIT] WINNER.txt of phase6 and phase8 exist, p2a_plain finished. [PAR] four trainings (the longest part) + two inference runs
python main.py --mode experiment --phase phase9_final --tag p9_final_s1 --inherit @phase6_domain_gen --set seed=1 tta=<BEST_TTA> temperature_scaling=true
python main.py --mode experiment --phase phase9_final --tag p9_final_s2 --inherit @phase6_domain_gen --set seed=2 tta=<BEST_TTA> temperature_scaling=true
python main.py --mode experiment --phase phase9_final --tag p9_base_s1  --inherit p2a_plain --set seed=1
python main.py --mode experiment --phase phase9_final --tag p9_base_s2  --inherit p2a_plain --set seed=2
python main.py --mode inference_experiment --phase phase9_final --tag p9_final_s0 --base @phase8_inference
python main.py --mode inference_experiment --phase phase9_final --tag p9_base_s0  --base p2a_plain
# [WAIT] all six finished
python main.py --mode seed_summary --phase phase9_final
```

### Final products
```bash
python main.py --mode master_table      # [SEQ] results/master/master_table.csv (+ .tex, master_full.csv, master_per_class.csv, master_binary.csv, master_config.csv, master_deltas.csv)
python main.py --mode report            # [SEQ] results/tables/table01..15 and results/figures/fig01..18 (bold text, 300 dpi PNG + PDF)
python main.py --mode xai --phase phase9_final --tag p9_final_s0               # [SEQ] XAI grid (5 classes x [original | Grad-CAM++ | Grad-CAM | LayerCAM | Eigen-CAM | detected lesions])
python main.py --mode xai --phase phase9_final --tag p9_final_s0 --set xai_top_k=1 xai_seed=0      # optional: strictly the most confident example per class
```
(If the agent has not yet added `--mode xai`, run the follow-up prompt first.)

---

## 5. Filling the placeholders (read only validation / OOF / DEV columns)

- `<BEST_STAGE>` and `<BEST_BETA>`: print the phase 6 table (viewer in section 2). Among the `p6_amp_*` rows choose the highest `extdev_qwk` (tie: higher `extdev_mild_recall`) whose `d_oof_qwk` is not below -0.005. `p6_amp_pre_b10` means stage `pre`, beta `0.10`.
- `<BEST_P6_TAG>`: the best of `p6_ref`, `p6_amp_*`, `p6_ms`, `p6_amp_ms` by the same rule (never `p6_uda`).
- `<BEST_TTA>`: one of `none`, `flip`, `flip_rot`; choose by the phase 8 table (`extdev_qwk`); for `p8_tta_scale` use `flip_rot` together with `'tta_scales=[0.9,1.0,1.1]'`.
- Winners: `cat results/<phase>/WINNER.txt` (first word = the tag). `--inherit @phase` and `--base @phase` already read it for you.

## 6. If something goes wrong

| Message / symptom | Fix |
|---|---|
| out of memory | add `--set batch_size=8 grad_accum=2 grad_checkpointing=true` to the same command and re-run it (finished folds are skipped) |
| `@phaseX` / `WINNER.txt` not found | run `phase_select` for that phase first |
| `amp_mix needs preproc != legacy` | add `preproc=scaled` (as in the phase 6 commands) |
| fewer than 5 fold checkpoints at test time | re-run the same `experiment` command; do not edit checkpoints by hand |
| `pre_ssl` finishes immediately with "Already completed" | use a new `ssl_save_dir` for each SSL variant |
| `phase_select` raises about test data | expected guard; do not bypass it |
| two jobs with the same tag | stop one; delete nothing; re-run the survivor's command to check it resumes cleanly |
| `results/<phase>/` folder lost while `checkpoints/` survived | do NOT re-run `experiment` (it would retrain every fold); rebuild the artefacts from the checkpoints with section 6A |

### 6A. Rebuilding a lost `results/<phase>/` folder from checkpoints (inference only)

`train_fold` skips a fold only when `checkpoints/<tag>/foldK.pth`, `foldK_val_metrics.json` **and**
`results/<phase>/<tag>/oof_foldK.npz` all exist (`trainer.py:185`). If a results folder disappears,
re-running `--mode experiment` therefore silently retrains all five folds. Use `tools/rebuild_oof.py`
instead, one job per tag of that phase (~8 GPU-min each):

```bash
phd run -p shr_gpu -ng 1 -GR H100 -l rb_<tag>.txt --   env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1   env/bin/python tools/rebuild_oof.py --tag <tag> --phase <phase>
```

It loads the config from `checkpoints/<tag>/config_snapshot.yaml`, regenerates every
`oof_foldK.npz` from `foldK.pth` through the same `evaluate_loader()` path the trainer used (exact,
not approximate: the checkpoint and its OOF arrays are saved together from the same model object,
`trainer.py:434-439`), then resumes `run_experiment()`, which skips training and rewrites the
thresholds, the fitted temperature, the ensemble test/DEV predictions, `per_fold.csv`,
`summary.csv` and `config.yaml`. Every fold logs `QWK rebuilt=<x> recorded=<y> -> OK|MISMATCH`; the
merged OOF QWK must also match that tag's entry in `results/all_runs.npz`. When all tags of the
phase are back, re-run that phase's `phase_select` so `WINNER.txt` and `phase_table.csv` are
recomputed from the data - never write `WINNER.txt` by hand, later phases resolve `@phase` through it.

Back up after every phase finishes (a tarball costs megabytes, a re-run costs GPU-hours):

```bash
tar czf /tmp/results_rescue_$(date +%Y%m%d_%H%M).tgz results/ checkpoints/*/config_snapshot.yaml
```
