# SERVER TODO — what to run, in this order (state of 2026-11-03)

Tick the boxes as you go. **Mandatory** steps are needed for the paper; *optional* steps can be skipped
without breaking anything later. Every command is run from the repo root on the server, with the
`phd run ... env/bin/python` wrapper shown in `commands.md` §1 for GPU jobs. GPU-hour figures are
estimates from the measured rates (plain 5-fold arm ≈ 2.8 h, joint / multi-source arm ≈ 4.7 h,
re-evaluation ≈ 0.25 h per tag). Full reference of every command: `commands.md`.
What can be deleted afterwards: `SERVER_CLEANUP.md`.

---

## A. Stop, wait, back up (before the new code goes on the server)

- [ ] **A1 (mandatory) — stop the SSL pretraining.** Its contrastive loss has been 0.0000 for ~47 epochs;
  the remaining 33 epochs cost ≈ 52 GPU h for no expected gain. SSL is reported as a negative control.
  Cancel the `pre_ssl` job in your scheduler (it is the job that writes `logs/pre_ssl/`); if it runs as a plain
  process on the node instead:
  ```bash
  ps aux | grep -- "--tag pre_ssl" | grep -v grep     # shows the process
  pkill -f -- "--tag pre_ssl"                          # stops it
  ```
  Keep `checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch50.pth` and `..._epoch60.pth` (used by `p3a_ssl`, `p3a_ssl60`).

- [ ] **A2 (mandatory) — let the jobs that are running now finish with the old code**
  (`p3c_joint_s1`, `p3c_coral_s1`, `p3c_none_s1`, `p3c_sup_s1`, `p3c_none_s2`). Do **not** submit any new
  job until step B is done. Their results are fully compatible: step C adds the new datasets to them.

- [ ] **A3 (mandatory) — back up** the protocol state (small):
  ```bash
  mkdir -p backups && tar czf backups/proto_$(date +%Y%m%d_%H%M)_before_v2.tgz results/ splits/ softlabels/ \
      checkpoints/*/config_snapshot.yaml checkpoints/*/thresholds.json
  ```

## B. Put the new code and the new datasets on the server

- [ ] **B1 (mandatory) — copy the code from the Mac.** Copy only code and documents. **Never copy `splits/`,
  `results/`, `checkpoints/`, `softlabels/` or `logs/` from the Mac**: the Mac copies are test leftovers
  (the Mac `splits/messidor_*_ids.json` hold 24/96 synthetic ids); the real ones exist only on the server.
  On the Mac, in the MODAL-Net folder (set `SERVER` once to the login and repo path you already use for scp):
  ```bash
  SERVER=your_login@your_cluster:/path/to/MODAL-Net
  rsync -av --exclude datasets --exclude results --exclude checkpoints --exclude splits --exclude softlabels \
        --exclude logs --exclude backups --exclude outputs --exclude PoxCSAF-Net --exclude _transfer \
        --exclude .git --exclude .pytest_cache --exclude __pycache__ --exclude .DS_Store ./ "$SERVER/"
  ```
- [ ] **B2 (mandatory) — remove the four files that no longer exist in the code** (on the server):
  ```bash
  rm -f src/training/feature_balance.py src/models/wavelet_gate.py tests/test_feature_balance.py tests/test_smoke.py
  ```
- [ ] **B3 (mandatory) — copy the three new datasets** (on the Mac; only what the code reads, ≈ 16 GB):
  ```bash
  rsync -av datasets/DDR-dataset/DR_grading datasets/DDR-dataset/lesion_segmentation "$SERVER/datasets/DDR-dataset/"
  rsync -av --exclude .git --exclude ultra-widefield_images datasets/DeepDRiD/regular_fundus_images "$SERVER/datasets/DeepDRiD/"
  rsync -av "datasets/IDRiD/B. Disease Grading" "datasets/IDRiD/A. Segmentation" "$SERVER/datasets/IDRiD/"
  ```
  The DeepDRiD test folder must be called `Online-Challenge1_2-Evaluation` (as on the Mac); paths are in
  `configs/config.yaml -> paths`.
- [ ] **B4 (mandatory) — tests** (CPU, ≈ 4 min; they run in temporary folders and cannot touch real files):
  ```bash
  env/bin/python -m pytest -q          # expected: 71 passed
  ```
- [ ] **B5 (mandatory) — remove the stale generated tables and the test leftovers**: run sections 1 and 2 of
  `SERVER_CLEANUP.md` now (before step C), so that no table built by the old code survives.

## C. Freeze the new evaluation sets and evaluate every finished model on them

- [ ] **C1 (mandatory) — duplicate audit of DDR / DeepDRiD / IDRiD** (CPU, ≈ 1–2 h; re-uses the cached hashes
  of the first audit, so EyePACS is not re-hashed):
  ```bash
  env/bin/python main.py --mode dedup_external
  ```
- [ ] **C2 (mandatory) — freeze the splits** (writes `splits/external/*.csv`, `exclusions.csv`, `manifest.json`):
  ```bash
  env/bin/python main.py --mode make_external_splits
  cat splits/external/manifest.json
  env/bin/python tools/dup_review.py --max-pairs 80        # look at results/phase0_protocol/audit/dup_review_*.png
  ```
  Expected sizes before exclusions: DDR dev 2,503 / test 3,759; DeepDRiD dev 400 / test 400; IDRiD dev 413 /
  test 103. Messidor DEV 349 / TEST 1,395 and the APTOS test set are never filtered (the finished runs were
  scored on them); their overlap with training images is reported in C3. If the contact sheets show that many removed pairs are *different*
  images, re-run C2 with a stricter threshold, e.g. `--set dedup_exclude_max=2`, and keep that value for the
  whole project.
- [ ] **C3 (mandatory) — leakage report** for the paper (APTOS test is never filtered; state the count):
  ```bash
  env/bin/python tools/leakage_check.py --tags p2b_none,p3a_sup,p3b_joint,p4_coral
  ```
- [ ] **C4 (mandatory) — evaluate every finished tag on every set** (GPU, ≈ 0.25 h per tag; one job, or split
  into parallel jobs with `evaluate_only`):
  ```bash
  env/bin/python main.py --mode evaluate_all
  cat results/master/evaluate_all_report.csv            # every row must say ok (zoo rows say skipped)
  ```
  Parallel alternative (three GPU jobs):
  ```bash
  env/bin/python main.py --mode evaluate_all --set 'evaluate_only=[p2a_plain,p2a_reg,p2a_mix,p2a_osmooth,p2a_ema,p2a_aug,p2a_sam,p2a_bundle]'
  env/bin/python main.py --mode evaluate_all --set 'evaluate_only=[p2b_scaled,p2b_none,p2b_clahe_only,p3a_sup,p3a_sup_v2,p3a_sslsup,p3a_ssl,p3b_joint,p3b_joint_w25]'
  env/bin/python main.py --mode evaluate_all --set 'evaluate_only=[p4_coral,p4_emd,p3c_none_s1,p3c_none_s2,p3c_joint_s1,p3c_coral_s1,p3c_sup_s1]'
  ```
- [ ] **C5 (mandatory, free) — cut-point diagnostic**, the pre-test of the new component (CPU, seconds):
  ```bash
  env/bin/python tools/cutpoint_diagnostic.py --tags p2b_none,p3a_sup,p3b_joint,p4_coral
  ```
  The first table (`devcf_*`, cross-fitted inside each DEV split, no TEST image used) is the evidence that
  motivates `coral_src`: a positive `devcf_qwk_gain` interval on most external datasets means "the model ranks
  images well; only its cut-points are wrong for a new site". The second table (TEST, cut-points from the whole
  DEV: `qwk_gain`, `mild_recall_argmax → mild_recall_devcal`, `pred_nodr_argmax → pred_nodr_devcal`, next to
  `test_bin_auroc`) is for reporting only. Whatever the outcome, both tables go into the paper (pre-declared here).

## D. Re-issue the selection under the amended endpoint (written before any phase-4j result exists)

- [ ] **D1 (mandatory) — document the closed phases under the amended endpoint** (tables only, history untouched):
  ```bash
  env/bin/python main.py --mode phase_select --phase phase2a_recipe     --reference p2a_plain        --set select_endpoint=amended select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase2b_preprocess --reference p2a_reg          --set select_endpoint=amended select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase3a_pretrain   --reference p2b_none         --set select_endpoint=amended select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase3a_pretrain   --reference p2b_none         --set select_endpoint=original select_report_only=true
  ```
- [ ] **D2 (mandatory) — phase 3b decision under the amended endpoint** (rewrites `results/phase3b_joint/WINNER.txt`):
  ```bash
  env/bin/python main.py --mode phase_select --phase phase3b_joint --reference p2b_none --set select_endpoint=amended
  env/bin/python main.py --mode phase_select --phase phase3b_joint --reference p2b_none --set select_endpoint=original select_report_only=true
  head -n 1 results/phase3b_joint/WINNER.txt
  ```
  Expected `p3b_joint`. If another tag wins, stop here and look at `phase_table.csv` before step E (the new
  lineage below inherits `p3b_joint` by name, so a different winner means the plan must be re-discussed).
- [ ] **D3 (mandatory) — old phase 4 (on `p2b_none`) as a report-only table**:
  ```bash
  env/bin/python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_endpoint=amended select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_endpoint=original select_report_only=true
  ```
  `p4_aasl` / `p4_aasl_emd` on the old lineage are **not** run any more; they are run on the new lineage (E1).

## E. New lineage: ordinal heads and the cut-point component (GPU)

- [ ] **E1 (mandatory) — phase 4j, six arms, all in parallel** (≈ 6 × 4.7 = 28 GPU h). Commands: `commands.md` §6
  "Phase 4j": `p4j_coral`, `p4j_emd`, `p4j_aasl`, `p4j_aasl_emd`, `p4j_coralsrc_shared`, `p4j_coralsrc`.
  After each AASL run: `grep -c "AASL soft labels for fold" logs/p4j_aasl/*.log` must print 5.
- [ ] **E2 (mandatory) — phase 4j selection**:
  ```bash
  env/bin/python main.py --mode phase_select --phase phase4j_label_loss --reference p3b_joint
  env/bin/python main.py --mode phase_select --phase phase4j_label_loss --reference p3b_joint --set select_endpoint=original select_report_only=true
  head -n 1 results/phase4j_label_loss/WINNER.txt      # = the final model (FINAL) used from step H on
  ```
- [ ] **E3 (mandatory) — phase 10, multi-source leave-sites-out** (≈ 6 × 4.7 = 28 GPU h, parallel). Commands:
  `commands.md` §6 "Phase 10": `p10d_ce`, `p10d_coralsh`, `p10d_coralsrc` (sources APTOS + EyePACS + DDR) and
  `p10q_ce`, `p10q_coralsh`, `p10q_coralsrc` (APTOS + EyePACS + DeepDRiD), then the two `phase_select` lines.
  This is the multi-site test of the cut-point idea: each arm is scored only on the datasets it never saw.

## F. Seeds (seed is the unit of analysis for every headline claim)

- [ ] **F1 (mandatory) — finish the phase-3c wave with the new code** (≈ 10 GPU h):
  `p3c_joint_s2`, `p3c_coral_s2`, `p3c_sup_s2` (commands: `commands.md` §6 "Phase 3c").
- [ ] **F2 (mandatory) — seeds 1 and 2 of the new headline configurations** (≈ 10 runs, ≈ 44 GPU h):
  `p4j_coralsrc_s1/_s2`, `p4j_coralsrc_shared_s1/_s2`, `p10d_ce_s1/_s2`, `p10d_coralsrc_s1/_s2`,
  `p9_base_s1/_s2` (commands: `commands.md` §6 "Seeds"). If FINAL (step E2) is not `p4j_coralsrc`, add
  its seeds too:
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  for s in 1 2; do env/bin/python main.py --mode experiment --phase phase4j_seeds --tag ${FINAL}_s$s --inherit $FINAL --set seed=$s; done
  ```
- [ ] **F3 (mandatory) — seed tables**:
  ```bash
  env/bin/python main.py --mode seed_summary --reference p3b_joint
  ```

## G. Mechanism and secondary arms

- [ ] **G1 (mandatory) — data volume vs source diversity**: `p3c_dose5k`, `p3c_dose2k` (≈ 9.4 GPU h).
- [ ] *G2 (optional) — SSL dose point*: `p3a_ssl60` (≈ 2.8 GPU h) for the SSL negative-control paragraph.
- [ ] *G3 (optional, recommended) — phase 5 architectures on the new lineage* (≈ 12 GPU h): `p5_convnextv2`,
  `p5_effnetv2`, `p5_hetero_ens`, `phase_select` (`commands.md` §6 "Phase 5"). Reviewers ask whether the effect
  depends on the backbone.
- [ ] *G3b (optional, recommended for TMI) — retinal foundation-model baseline (RETFound)* (≈ 2 × 2 GPU h at 224 px):
  download the gated weights first (see `commands.md` §6 "Phase 5", RETFound paragraph), then run `p5_retfound` and
  `p5_retfound_joint`. Reviewers of a 2026 DR paper will ask how the model compares with a retinal foundation model.
- [ ] *G4 (optional) — phase 6 domain generalisation, 4 cells* (≈ 16 GPU h): `p6_amp`, `p6_ms`, `p6_amp_ms`,
  `p6_amp_nojoint` (report-only selection).
- Not run any more: phase 7 (feature balance, code removed), the 4-way TTA grid, `p3c_cnext_ref`,
  `p3c_effnet_ref` (superseded by G3), wavelet gate.

## H. Final model, report, XAI

- [ ] **H1 (mandatory) — inference products of FINAL** (no training):
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  env/bin/python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_flip --base $FINAL --set tta=flip
  env/bin/python main.py --mode calibrate_conformal  --phase phase8_inference --tag p8_conformal --base $FINAL --set conformal_alpha=0.10
  ```
- [ ] **H2 (mandatory) — XAI** (grid + expert-mask localisation on DDR / IDRiD lesion masks; final vs baseline):
  ```bash
  env/bin/python main.py --mode xai --tag $FINAL --set xai_split=test
  env/bin/python main.py --mode xai --tag $FINAL --set xai_split=messidor_test xai_top_k=1
  env/bin/python main.py --mode xai_localization --tag $FINAL
  env/bin/python main.py --mode xai_localization --tag p2a_plain
  ```
- [ ] **H3 (mandatory) — tables and figures** (only real data; see `results/tables/TABLES_README.txt`,
  `results/figures/FIGURES_README.txt`):
  ```bash
  env/bin/python main.py --mode master_table
  env/bin/python main.py --mode report --set report_final=$FINAL report_baseline=p2a_plain soft_label_run=sl_p4
  ```
- [ ] **H4 (mandatory) — back up** (same command as A3, new name) and then clean up with `SERVER_CLEANUP.md`
  sections 1, 3 and 4 only (section 2 would delete the final tables you just built).

---

## Total GPU budget of this list
Mandatory: C4 ≈ 7, E1 ≈ 28, E3 ≈ 28, F1 ≈ 10, F2 ≈ 44, G1 ≈ 9.4, H ≈ 2 → **≈ 128 GPU h**
(≈ 1.5–2 days of wall clock at 4–6 parallel slots). Optional G2–G4: ≈ 35 GPU h. The SSL stop (A1) saves 52.
