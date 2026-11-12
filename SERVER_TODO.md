# SERVER TODO — every step of the paper, in order (state of 2026-11-04)

This file is enough to run the whole project: every command is written here in full, so you do not need
`commands.md` (it stays as the reference with more background). What can be deleted, and when, is in
`SERVER_CLEANUP.md`.

Tick the boxes yourself as you go. **Mandatory** steps are needed for the paper; *optional* steps can be
skipped without breaking anything later.

## Daily report (upload these files to Claude once a day)

**Start:** after **D3** is done. Then submit the wave-5 jobs that do not depend on the plan (F1, F2a
`p9_base_*`, G2b, G3b), but submit **E1 / E3 only after Claude's feedback on that first report**.

**Every report, in this order (on the server):**
1. Check which runs are finished (replace the tags by the ones you are waiting for):
   ```bash
   for t in p10q_coralsrc p9_base_s1 p9_base_s2; do ls results/phase*/"$t"/summary.csv >/dev/null 2>&1 && echo "DONE     $t" || echo "not yet  $t"; done
   ```
2. Run the CPU step that belongs to each newly finished block: E2 / E3 selection lines, E4, the E4 seed re-run.
3. Refresh the master and seed tables (CPU, ≈ 1 min; this is also F3):
   ```bash
   env/bin/python main.py --mode seed_summary --reference p3b_joint && env/bin/python main.py --mode master_table
   ```
4. Upload these files:

| # | Server path | When |
|---|---|---|
| 1 | `results/master/master_full.csv` | every report (every run × every set × every metric) |
| 2 | `results/master/seed_variance.csv` and `results/master/seed_compare.csv` | every report once seed runs exist |
| 3 | `results/<phase>/phase_table.csv` | each phase selected since the last report (E2: `phase4j_label_loss`; E3: `phase10_ms_ddr`, `phase10_ms_deepdrid`) |
| 4 | `results/master/adaptation_compare.csv`, `adaptation_curve.csv` and `adaptation_lambda.csv` | after E4 |
| 5 | `results/master/adaptation_seeds/adaptation_compare.csv` and `.../adaptation_curve.csv` | after the E4 seed re-run (needs F2c) |
| 6 | `results/master/recal/`, `recal_seeds/`, `recal_backbones/` — all `*.csv` except every `recal_draws.csv` (large) | after R3 / R4 / R5 |

**Next report (when E3, F1, F2a, F2b and F2c are finished), step by step:**
1. Check:
   ```bash
   for t in p10q_coralsrc p9_base_s1 p9_base_s2 p4j_coralsrc_s1 p4j_coralsrc_s2 p4j_coralsrc_shared_s1 p4j_coralsrc_shared_s2 p10d_ce_s1 p10d_ce_s2 p10d_coralsrc_s1 p10d_coralsrc_s2; do ls results/phase*/"$t"/summary.csv >/dev/null 2>&1 && echo "DONE     $t" || echo "not yet  $t"; done
   for d in results/phase*/*/; do [ -f "$d/oof.npz" ] && [ ! -f "$d/predictions_ddr_dev.npz" ] && echo "not evaluated: $d"; done   # F1: must print nothing
   ```
2. E3 selection: the four `phase_select` lines at the end of E3.
3. E4: the test line, the main `adaptation_curve.py` line, then the seed re-run line (all in E4).
4. F2d: read the rule in F2d (with the current E2 winner `p4j_coralsrc_shared` there is nothing to run).
5. Refresh (step 3 above).
6. Upload: `master_full.csv`, `seed_variance.csv`, `seed_compare.csv`, `phase10_ms_ddr/phase_table.csv`,
   `phase10_ms_deepdrid/phase_table.csv`, `adaptation_compare.csv`, `adaptation_curve.csv`, `adaptation_lambda.csv`,
   `adaptation_seeds/adaptation_compare.csv`, `adaptation_seeds/adaptation_curve.csv`
   (and `phase4j_label_loss/phase_table.csv` if it was not sent yet).

---

## 0. How to use this file

### 0.1 Every time you log in (paste once per terminal)
```bash
cd /home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

```

### 0.2 Reading the steps
- **SEQUENTIAL** = wait until the previous step has finished (check with the `for t in ...; do ls results/phase*/"$t"/summary.csv ...` line given in the step, or the log) before you start the next one.
- **PARALLEL** = submit all lines of the block at the same time, each line is its own GPU job.
  If `phd run` keeps your terminal busy until the job ends, open one `tmux` window per line.
- GPU steps start with `phd run ...` (one line = one scheduler job; its log goes to the `-l` file under `outputs/`).
  CPU steps (plain `env/bin/python ...`) run directly in the terminal. Run long CPU steps inside `tmux`/`screen`
  so they survive a disconnect.
- Re-running exactly the same line **resumes** a run (finished folds are skipped). Never run two jobs with
  the same `--tag` at the same time.
- `--inherit X` needs `checkpoints/X/config_snapshot.yaml` (X has started once); `--base X` / `--members X`
  need X **finished**.
- `@phaseNAME` = the tag written in `results/phaseNAME/WINNER.txt` (exists only after `phase_select` of that phase).
- Never look at TEST columns to decide anything. Decisions use only OOF and DEV columns:
  ```bash
  # replace PHASE by the phase folder, e.g. phase4j_label_loss
  env/bin/python -c "import pandas as pd; t=pd.read_csv('results/PHASE/phase_table.csv'); print(t[['experiment','oof_qwk','d_oof_qwk','ci_lo','oof_equivalent','d_dev_mean_qwk','d_dev_mean_qwk_ci_lo','dev_sets_used','decision_original','decision_amended','winner']].to_string(index=False))"
  ```
- Back up after every finished section (megabytes, seconds):
  ```bash
  mkdir -p backups && tar czf backups/results_$(date +%Y%m%d_%H%M).tgz results/ splits/ softlabels/ \
      checkpoints/*/config_snapshot.yaml checkpoints/*/thresholds.json
  ```

### 0.3 The order at a glance

| Wave | Steps | Kind | Starts when |
|---|---|---|---|
| 1 | A, B (stop, copy, tests) | done / CPU | now |
| 2 | C1 → C2 → C3 | CPU, sequential | B4 passes |
| 3 | C4 (re-evaluate every finished run) | GPU, 3 jobs parallel | C2 done (C3 can run at the same time) |
| 4 | C4b → C5 → D1, D2, D3 (interim tables, diagnostic, selections) | CPU, minutes | C4 done |
| 5 | E1 (6 jobs) + E3 (6 jobs) + F1 (1 evaluate job) + `p9_base` seeds (2) + optional G2b, G3b | GPU, all parallel | D2 says `p3b_joint` |
| 6 | E2, E3 selections, then E4 (adaptation analysis) | CPU, minutes | all E1 / E3 runs DONE |
| 7 | F2 (rest of the seeds), optional G3, G4 | GPU, parallel | E2 (FINAL known) |
| 8 | F3, H1, H2 | CPU + GPU | F2 done (H1/H2 may start right after E2) |
| 8b | R0 → R1 (GPU, 9 jobs) → R2 → R3 → R4; R5 after F2c/G3/G3b | GPU + CPU | now (independent of running jobs) |
| 9 | H3 → H4 (tables, figures, backup, cleanup) | CPU | everything above done |

With 4–6 GPU slots this is ≈ 1.5–2 days of GPU wall clock after wave 4. The total GPU budget is at the end.

---

## A. Stop, wait, back up (before the new code goes on the server)

- [✅] **A1 (mandatory) — stop the SSL pretraining.** Its contrastive loss has been 0.0000 for ~47 epochs;
  the remaining 33 epochs cost ≈ 52 GPU h for no expected gain. SSL is reported as a negative control.
  Cancel the `pre_ssl` job in your scheduler (it is the job that writes `logs/pre_ssl/`); if it runs as a plain
  process on the node instead:
  ```bash
  ps aux | grep -- "--tag pre_ssl" | grep -v grep     # shows the process
  pkill -f -- "--tag pre_ssl"                          # stops it
  ```
  Keep `checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch50.pth` and `..._epoch60.pth` (used by `p3a_ssl`, `p3a_ssl60`).

- [✅] **A2 (mandatory) — let the jobs that are running now finish with the old code**
  (`p3c_joint_s1`, `p3c_coral_s1`, `p3c_none_s1`, `p3c_sup_s1`, `p3c_none_s2`). Do **not** submit any new
  job until step B is done. Their results are fully compatible: step C4 adds the new datasets to them, and
  C4b / F3 / H3 put them into the master and seed tables automatically (every `results/phase*` folder is
  discovered, `phase3c_seeds` included; nothing has to be merged by hand).

- [✅] **A3 (mandatory) — back up** the protocol state (small):
  ```bash
  mkdir -p backups && tar czf backups/proto_$(date +%Y%m%d_%H%M)_before_v2.tgz results/ splits/ softlabels/ \
      checkpoints/*/config_snapshot.yaml checkpoints/*/thresholds.json
  ```

## B. Put the new code and the new datasets on the server

- [✅] **B1 (mandatory) — copy the code from the Mac.** Copy only code and documents. **Never copy `splits/`,
  `results/`, `checkpoints/`, `softlabels/` or `logs/` from the Mac**: the Mac copies are test leftovers
  (the Mac `splits/messidor_*_ids.json` hold 24/96 synthetic ids); the real ones exist only on the server.
  On the Mac, in the MODAL-Net folder (set `SERVER` once to the login and repo path you already use for scp):
  ```bash
  SERVER=your_login@your_cluster:/path/to/MODAL-Net
  rsync -av --exclude datasets --exclude results --exclude checkpoints --exclude splits --exclude softlabels \
        --exclude logs --exclude backups --exclude outputs --exclude PoxCSAF-Net --exclude _transfer --exclude daily \
        --exclude .git --exclude .pytest_cache --exclude __pycache__ --exclude .DS_Store ./ "$SERVER/"
  ```
  This rsync only adds and updates files; it never deletes a server file that is missing on the Mac (that is
  why B2 exists).
- [✅] **B2 (mandatory) — remove the four files that no longer exist in the code** (on the server):
  ```bash
  rm -f src/training/feature_balance.py src/models/wavelet_gate.py tests/test_feature_balance.py tests/test_smoke.py
  ```
- [✅] **B3 (mandatory) — copy the three new datasets** (on the Mac; only what the code reads, ≈ 16 GB):
  ```bash
  rsync -av datasets/DDR-dataset/DR_grading datasets/DDR-dataset/lesion_segmentation "$SERVER/datasets/DDR-dataset/"
  rsync -av --exclude .git --exclude ultra-widefield_images datasets/DeepDRiD/regular_fundus_images "$SERVER/datasets/DeepDRiD/"
  rsync -av "datasets/IDRiD/B. Disease Grading" "datasets/IDRiD/A. Segmentation" "$SERVER/datasets/IDRiD/"
  ```
  The DeepDRiD test folder must be called `Online-Challenge1_2-Evaluation` (as on the Mac); paths are in
  `configs/config.yaml -> paths`. A `datasets/DDR-dataset/.cache/` folder (Hugging Face `.lock` / `.metadata`
  files) is never read by the code; it can be deleted (`SERVER_CLEANUP.md` §1).
- [✅] **B4 (mandatory) — tests** (CPU, ≈ 5–10 min; they run in temporary folders and cannot touch real files).
  The first run failed only because two old server-only test files (`tests/test_selection_fallbacks.py`,
  `tests/test_zz_guard_check.py`) were written for the old code. Both are rewritten on the Mac.
  - [✅] **B4a — copy the two rewritten test files (and the updated documents)**: run the B1 `rsync` line again
    on the Mac (it only sends what changed).
  - [✅] **B4b — run the tests again** (on the server):
    ```bash
    env/bin/python -m pytest -q -p no:warnings          # expected: 80 passed
    ```
    `80 passed` = 71 tests of the main suite + 5 in `test_selection_fallbacks.py` + 4 in `test_zz_guard_check.py`.
    A different count means `tests/` on the server is not identical to the Mac: compare with
    `ls tests/` on both machines. Do not continue to C2 with a failing test.
- [✅] **B5 (mandatory) — remove the stale generated tables and the test leftovers**: run sections 1 and 2 of
  `SERVER_CLEANUP.md` now (before step C), so that no table built by the old code survives.

## C. Freeze the new evaluation sets and evaluate every finished model on them

C1 → C2 are **SEQUENTIAL** (CPU). C3 and C4 may run at the same time once C2 is done.

- [✅] **C1 (mandatory) — duplicate audit of DDR / DeepDRiD / IDRiD** (CPU, ≈ 1–2 h; re-uses the cached hashes
  of the first audit, so EyePACS is not re-hashed):
  ```bash
  env/bin/python main.py --mode dedup_external
  ```
  Finished when the command returns without an error; its output is under `results/phase0_protocol/audit/`.
- [✅] **C2 (mandatory) — freeze the splits** (CPU, minutes; writes `splits/external/*.csv`, `exclusions.csv`,
  `manifest.json`). Needs C1.
  ```bash
  env/bin/python main.py --mode make_external_splits
  cat splits/external/manifest.json
  env/bin/python tools/dup_review.py --max-pairs 80        # look at results/phase0_protocol/audit/dup_review_*.png
  ```
  Expected sizes before exclusions: DDR dev 2,503 / test 3,759; DeepDRiD dev 400 / test 400; IDRiD dev 413 /
  test 103. Messidor DEV 349 / TEST 1,395 and the APTOS test set are never filtered (the finished runs were
  scored on them); their overlap with training images is reported in C3. If the contact sheets show that
  many removed pairs are *different* images, re-run C2 with a stricter threshold, e.g.
  `env/bin/python main.py --mode make_external_splits --set dedup_exclude_max=2`, and keep that value for the
  whole project. **Do C2 only once and before any C4 / training job**: every number of the paper is computed
  on these frozen splits.
- [✅] **C3 (mandatory) — leakage report** for the paper (CPU, minutes; APTOS test is never filtered, state the
  count). Needs C2. Can run while C4 runs.
  ```bash
  env/bin/python tools/leakage_check.py --tags p2b_none,p3a_sup,p3b_joint,p4_coral
  ```
- [✅] **C4 (mandatory) — evaluate every finished tag on every set** (GPU, ≈ 0.25 h per tag, ≈ 7 GPU h in total).
  Needs C2. **PARALLEL**: three jobs on disjoint tag lists (≈ 2.5 h wall clock):
  ```bash
  mkdir -p outputs/c4   # the -l paths write into this folder, create it first
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/c4/eval_part1.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode evaluate_all --set 'evaluate_only=[p2a_plain,p2a_reg,p2a_mix,p2a_osmooth,p2a_ema,p2a_aug,p2a_sam,p2a_bundle]'
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/c4/eval_part2.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode evaluate_all --set 'evaluate_only=[p2b_scaled,p2b_none,p2b_clahe_only,p3a_sup,p3a_sup_v2,p3a_sslsup,p3a_ssl,p3b_joint,p3b_joint_w25]'
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/c4/eval_part3.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode evaluate_all --set 'evaluate_only=[p4_coral,p4_emd,p4_aasl,p4_aasl_emd,p3c_none_s1,p3c_none_s2,p3c_joint_s1,p3c_coral_s1,p3c_sup_s1]'
  ```

  (Single-job alternative: the same `phd run ...` line with `--mode evaluate_all` and no `--set`.) An `evaluate_only` name without a results
  folder stops that job at once with the list of unknown names: remove the name from the list and re-submit.
  Check when all three jobs have ended:
  ```bash
  cat results/master/evaluate_all_report.csv            # every row must say ok (zoo rows say skipped)
  ```
  A row that is not `ok`: re-run only that tag, e.g.
  `phd run -p shr_gpu -ng 1 -GR H100 -l outputs/c4/eval_p3b_joint.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode evaluate --tag p3b_joint`.
  Then check that **no** finished run was left out of the three lists (each printed folder needs the same `--mode evaluate --tag <tag>` job):
  ```bash
  for d in results/phase*/*/; do [ -f "$d/oof.npz" ] && [ ! -f "$d/predictions_ddr_dev.npz" ] && echo "not evaluated: $d"; done
  ```
- [✅] **C4b (mandatory, CPU, seconds) — interim seed and master tables** (needs C4). They now hold every finished
  run, the A2 seed runs included, on every internal and external set:
  ```bash
  env/bin/python main.py --mode seed_summary --reference p3b_joint     # results/master/seed_variance.csv + seed_compare.csv
  env/bin/python main.py --mode master_table                           # results/master/master_table.csv/.tex, master_full.csv, master_long.csv
  ```
  These are interim tables: they are overwritten in F3 and H3, so it does not matter how often you run them.
  Run them only after C4: before C4 the DDR / DeepDRiD / IDRiD columns are empty.

- [✅] **C5 (mandatory, CPU, seconds) — cut-point diagnostic**, the pre-test of the new component (needs C4):
  ```bash
  env/bin/python tools/cutpoint_diagnostic.py --tags p2b_none,p3a_sup,p3b_joint,p4_coral
  ```
  The first table (`devcf_*`, cross-fitted inside each DEV split, no TEST image used) is the evidence that
  motivates `coral_src`: a positive `devcf_qwk_gain` interval on most external datasets means "the model ranks
  images well; only its cut-points are wrong for a new site". The second table (TEST, cut-points from the whole
  DEV: `qwk_gain`, `mild_recall_argmax → mild_recall_devcal`, `pred_nodr_argmax → pred_nodr_devcal`, next to
  `test_bin_auroc`) is for reporting only. Whatever the outcome, both tables go into the paper (pre-declared here).

## D. Re-issue the selection under the amended endpoint (written before any phase-4j result exists)

All of D is CPU and takes minutes. Needs C4 (the amended endpoint refuses to run if a DEV prediction file of
DDR / DeepDRiD / IDRiD is missing; that error means C4 is not finished for that tag). D1, D2 and D3 are
independent of each other; run them in this order anyway.

- [✅] **D1 (mandatory) — document the closed phases under the amended endpoint** (tables only, history untouched):
  ```bash
  env/bin/python main.py --mode phase_select --phase phase2a_recipe     --reference p2a_plain --set select_endpoint=amended select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase2b_preprocess --reference p2a_reg   --set select_endpoint=amended select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase3a_pretrain   --reference p2b_none  --set select_endpoint=amended select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase3a_pretrain   --reference p2b_none  --set select_endpoint=original select_report_only=true
  ```
  Output: `results/<phase>/phase_table_amended.csv` / `phase_table_original.csv`; `WINNER.txt` is not touched.
- [✅] **D2 (mandatory) — phase 3b decision under the amended endpoint** (rewrites `results/phase3b_joint/WINNER.txt`):
  ```bash
  env/bin/python main.py --mode phase_select --phase phase3b_joint --reference p2b_none --set select_endpoint=amended
  env/bin/python main.py --mode phase_select --phase phase3b_joint --reference p2b_none --set select_endpoint=original select_report_only=true
  head -n 1 results/phase3b_joint/WINNER.txt
  ```
  Expected `p3b_joint`. If another tag wins, **stop here** and look at `phase_table.csv` before step E (the new
  lineage below inherits `p3b_joint` by name, so a different winner means the plan must be re-discussed).
- [✅] **D3 (mandatory) — old phase 4 (on `p2b_none`) as a report-only table**:
  ```bash
  env/bin/python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_endpoint=amended select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_endpoint=original select_report_only=true
  ```

  No new phase-4 runs on the old lineage; AASL is tested on the new lineage (E1).
- [✅] Back up (command in 0.2).

## E. New lineage: ordinal heads and the cut-point component (GPU)

E1 and E3 are independent (both inherit `p3b_joint`): submit all twelve jobs together if you have the slots,
plus the wave-5 jobs of F1 / F2a / G2b / G3b below. Every run evaluates itself on all sets with all metrics at the end
(no separate evaluate step needed).

- [✅] **E1 (mandatory) — phase 4j, six arms, PARALLEL** (≈ 6 × 4.7 = 28 GPU h). Needs D2 = `p3b_joint`.
  ```bash
  mkdir -p outputs/phase4j_label_loss   # the -l paths go in here, create first
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_label_loss/p4j_coral.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coral --inherit p3b_joint --set loss=coral head_type=coral
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_label_loss/p4j_emd.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_label_loss --tag p4j_emd --inherit p3b_joint --set loss=ce_emd lambda_emd=1.0
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_label_loss/p4j_aasl.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_label_loss --tag p4j_aasl --inherit p3b_joint --set soft_labels=aasl soft_label_run=sl_p4
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_label_loss/p4j_aasl_emd.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_label_loss --tag p4j_aasl_emd --inherit p3b_joint --set soft_labels=aasl soft_label_run=sl_p4 loss=ce_emd lambda_emd=1.0
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_label_loss/p4j_coralsrc_shared.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coralsrc_shared --inherit p3b_joint --set loss=coral head_type=coral_src src_offsets=false
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_label_loss/p4j_coralsrc.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coralsrc --inherit p3b_joint --set loss=coral head_type=coral_src src_offsets=true src_offset_l2=0.1
  ```
  
  Checks while / after they run:
  ```bash
  grep -c "AASL soft labels for fold" logs/p4j_aasl/*.log logs/p4j_aasl_emd/*.log    # each must print 5
  for t in p4j_coral p4j_emd p4j_aasl p4j_aasl_emd p4j_coralsrc_shared p4j_coralsrc; do ls results/phase*/"$t"/summary.csv >/dev/null 2>&1 && echo "DONE     $t" || echo "not yet  $t"; done
  ```
  (`sl_p4` soft labels were made on the same APTOS folds; they are re-used on purpose. `coral_src` = one
  severity score + ordered cut-points; `src_offsets=true` adds one cut-point offset vector per training source,
  tied by `src_offset_l2`; `src_offsets=false` is the same head without offsets, the ablation that isolates them.)
- [✅] **E2 (mandatory, CPU, seconds) — phase 4j selection**. SEQUENTIAL: only when the check line shows all six E1 runs DONE.
  ```bash
  env/bin/python main.py --mode phase_select --phase phase4j_label_loss --reference p3b_joint
  env/bin/python main.py --mode phase_select --phase phase4j_label_loss --reference p3b_joint --set select_endpoint=original select_report_only=true
  head -n 1 results/phase4j_label_loss/WINNER.txt      # = the final model (FINAL) used from step F2 on
  ```
  From now on every new terminal also needs:
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt); echo $FINAL
  ```
- [✅] **E3 (mandatory) — phase 10, multi-source leave-sites-out, PARALLEL** (≈ 6 × 4.7 = 28 GPU h). Needs D2.
  This is the multi-site test of the cut-point idea: each arm is scored only on the datasets it never saw.
  ```bash
  # sources APTOS + EyePACS + DDR  -> held out: Messidor-2, DeepDRiD, IDRiD
  mkdir -p outputs/phase10_ms_ddr outputs/phase10_ms_deepdrid   # the -l log paths go in here
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_ms_ddr/p10d_ce.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_ms_ddr --tag p10d_ce       --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,ddr]'
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_ms_ddr/p10d_coralsh.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_ms_ddr --tag p10d_coralsh  --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,ddr]' loss=coral head_type=coral_src src_offsets=false
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_ms_ddr/p10d_coralsrc.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_ms_ddr --tag p10d_coralsrc --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,ddr]' loss=coral head_type=coral_src src_offsets=true
  # sources APTOS + EyePACS + DeepDRiD -> held out: Messidor-2, DDR, IDRiD
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_ms_deepdrid/p10q_ce.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_ms_deepdrid --tag p10q_ce       --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,deepdrid]'
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_ms_deepdrid/p10q_coralsh.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_ms_deepdrid --tag p10q_coralsh  --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,deepdrid]' loss=coral head_type=coral_src src_offsets=false
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_ms_deepdrid/p10q_coralsrc.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_ms_deepdrid --tag p10q_coralsrc --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,deepdrid]' loss=coral head_type=coral_src src_offsets=true
  ```
  Then, SEQUENTIAL (CPU, seconds), each line only when its three runs are DONE:
  ```bash
  for t in p10d_ce p10d_coralsh p10d_coralsrc p10q_ce p10q_coralsh p10q_coralsrc; do ls results/phase*/"$t"/summary.csv >/dev/null 2>&1 && echo "DONE     $t" || echo "not yet  $t"; done
  env/bin/python main.py --mode phase_select --phase phase10_ms_ddr      --reference p10d_ce
  env/bin/python main.py --mode phase_select --phase phase10_ms_ddr      --reference p10d_ce --set select_endpoint=original select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase10_ms_deepdrid --reference p10q_ce
  env/bin/python main.py --mode phase_select --phase phase10_ms_deepdrid --reference p10q_ce --set select_endpoint=original select_report_only=true
  ```
- [✅] **E4 (mandatory, CPU, ≈ 5–15 min) — cut-point adaptation to new sites** (no GPU, no retraining; works on
  the saved predictions). Needs E3 finished and selected. It asks how much information from a new site is needed
  to fix the grading, and whether the ordinal shared-score heads need less of it than CE: level 0 = nothing
  (argmax), level 1 = the site's grade distribution only (no image labels), level 2 = n = 10/25/50/100 labelled
  DEV images (plain refit and a refit shrunk towards the default cut-points, shrinkage strength chosen on the
  OTHER held-out sites' DEV splits), level 3 = the whole DEV split. Only held-out sites are used; cut-points are
  fitted on DEV only, TEST is only reported.
  First bring the new code to the server (on the Mac: the B1 `rsync` line again; it sends `tools/adaptation_curve.py`,
  `src/evaluation/cutpoints.py`, `tests/test_adaptation.py`), then on the server:
  ```bash
  env/bin/python -m pytest -q -p no:warnings tests/test_adaptation.py          # expected: 4 passed
  env/bin/python tools/adaptation_curve.py --groups "p10d_ce,p10d_coralsh,p10d_coralsrc;p10q_ce,p10q_coralsh,p10q_coralsrc;p3b_joint,p4j_coral,p4j_coralsrc_shared,p4j_coralsrc"
  ```
  Output: `results/master/adaptation_curve.csv`, `adaptation_compare.csv`, `adaptation_lambda.csv`,
  `results/figures/fig_adaptation.{png,pdf}`. The first tag of each `;`-group is the reference of that group.

  > Note 2026-11-09: the level formerly called `prevalence` uses the grade shares of the site's DEV LABELS — it is
  > an ORACLE upper bound, now named `prevalence_oracle`. The annotation-free version and the baselines are in step R.
  > **After R0, re-run the E4 command above once** (CPU, ≈ 10 min): the numbers stay identical, the level gets its
  > new name, and `adaptation_compare.csv` gets patient/exam-level bootstrap intervals (`bootstrap_unit` column).

  **Pre-declared rule (written 2026-11-06, before any adaptation result exists; do not change it after looking):**
  - Primary endpoint: the two phase-10 groups (truly unseen sites), level `fewshot_shrink`, n = 25, QWK, mean over
    the held-out datasets, paired difference of each ordinal head vs the `_ce` arm of its group
    (`adaptation_compare.csv`, `dataset = MEAN`, `metric = qwk`). "The ordinal shared score adapts better" is
    claimed only if `diff_lo > 0` in **both** phase-10 groups, and it must hold again in the seed re-run below.
  - Reported whatever the outcome: every level and n, mild recall and referable sensitivity (also in
    `adaptation_compare.csv`), and the phase-4j group (in-source heads, all four external sites held out).
  - After F2 (seeds), re-run on the seed tags into a separate folder (use the phase-10 seed tags that F2c ran):
    ```bash
    env/bin/python tools/adaptation_curve.py --out_dir results/master/adaptation_seeds --groups "p10d_ce_s1,p10d_coralsrc_s1;p10d_ce_s2,p10d_coralsrc_s2"
    ```
  - Theory (no run needed): the three propositions of `agent_context/MODAL_PART2.md` S16 are written into the paper
    by Claude from the E4 report files (`adaptation_curve.csv`, `adaptation_compare.csv`, `adaptation_lambda.csv`).
    Upload `results/master/adaptation_lambda.csv` with the E4 report too.
- [✅] Back up (command in 0.2).

## F. Seeds (seed is the unit of analysis for every headline claim)

Seed replicates are grouped automatically by configuration (any folder, any tag name).

- [✅] **F1 (mandatory) — evaluate the eight runs that are already trained (old code) but were not in the C4 lists**
  (GPU, one job, ≈ 2 h). `p3c_joint_s2`, `p3c_coral_s2`, `p3c_sup_s2` (seeds), `p3c_dose5k`, `p3c_dose2k` (G1),
  `p3a_ssl60` (G2) and `p3c_cnext_ref`, `p3c_effnet_ref` (backbone check on the old lineage) finished before the
  new code; they only need the new evaluation sets. **Do not retrain them.** Can start now.
  ```bash
  mkdir -p outputs/c4
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/c4/eval_part4.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode evaluate_all --set 'evaluate_only=[p3c_joint_s2,p3c_coral_s2,p3c_sup_s2,p3c_dose5k,p3c_dose2k,p3a_ssl60,p3c_cnext_ref,p3c_effnet_ref]'
  ```
  Afterwards the C4 check loop must print nothing:
  ```bash
  for d in results/phase*/*/; do [ -f "$d/oof.npz" ] && [ ! -f "$d/predictions_ddr_dev.npz" ] && echo "not evaluated: $d"; done
  ```
  (If the old F1 / G1 / G2 `--mode experiment` lines were already submitted, let them finish: a re-run skips the
  finished folds and only evaluates, so the result is the same.)
- [✅] **F2 (mandatory) — seeds 1 and 2 of the headline configurations, PARALLEL** (≈ 10 runs, ≈ 44 GPU h).
  Three groups with different start conditions:
  ```bash
  # ✅ F2a: baseline seeds, no dependency -> submit in wave 5 together with E1 / E3
  mkdir -p outputs/phase9_final outputs/phase4j_seeds outputs/phase10_seeds   # the -l log paths go in here
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase9_final/p9_base_s1.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase9_final --tag p9_base_s1 --inherit p2a_plain --set seed=1
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase9_final/p9_base_s2.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase9_final --tag p9_base_s2 --inherit p2a_plain --set seed=2

  # ✅ F2b: needs p4j_coralsrc and p4j_coralsrc_shared DONE (E1)
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_seeds/p4j_coralsrc_s1.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_s1        --inherit p4j_coralsrc        --set seed=1
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_seeds/p4j_coralsrc_s2.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_s2        --inherit p4j_coralsrc        --set seed=2
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_seeds/p4j_coralsrc_shared_s1.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_shared_s1 --inherit p4j_coralsrc_shared --set seed=1
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_seeds/p4j_coralsrc_shared_s2.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_shared_s2 --inherit p4j_coralsrc_shared --set seed=2

  # ✅ F2c: needs p10d_ce and p10d_coralsrc DONE (E3)
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_seeds/p10d_ce_s1.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_seeds --tag p10d_ce_s1       --inherit p10d_ce       --set seed=1
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_seeds/p10d_ce_s2.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_seeds --tag p10d_ce_s2       --inherit p10d_ce       --set seed=2
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_seeds/p10d_coralsrc_s1.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_seeds --tag p10d_coralsrc_s1 --inherit p10d_coralsrc --set seed=1
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase10_seeds/p10d_coralsrc_s2.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase10_seeds --tag p10d_coralsrc_s2 --inherit p10d_coralsrc --set seed=2
  ```
  [Skipped] F2d (conditional) — seeds of FINAL, only when FINAL is not already seeded in F2b. Check FINAL first:
  ```bash
  head -n 1 results/phase4j_label_loss/WINNER.txt
  ```
  - It prints `p4j_coralsrc` or `p4j_coralsrc_shared` → **skip F2d** (those seeds are already in F2b).
    With the current E2 result (`p4j_coralsrc_shared`) F2d is skipped.
  - It prints any other tag (e.g. `p4j_coral`) → **run the block below** (2 GPU jobs):
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  mkdir -p outputs/phase4j_seeds   # the -l log paths go in here
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_seeds/${FINAL}_s1.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_seeds --tag ${FINAL}_s1 --inherit $FINAL --set seed=1
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase4j_seeds/${FINAL}_s2.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase4j_seeds --tag ${FINAL}_s2 --inherit $FINAL --set seed=2
  ```
- [✅] **F3 (mandatory, CPU, seconds) — seed tables**, when every F1 / F2 run is DONE:
  ```bash
  for t in p3c_joint_s2 p3c_coral_s2 p3c_sup_s2 p9_base_s1 p9_base_s2 p4j_coralsrc_s1 p4j_coralsrc_s2 p4j_coralsrc_shared_s1 p4j_coralsrc_shared_s2 p10d_ce_s1 p10d_ce_s2 p10d_coralsrc_s1 p10d_coralsrc_s2; do ls results/phase*/"$t"/summary.csv >/dev/null 2>&1 && echo "DONE     $t" || echo "not yet  $t"; done
  env/bin/python main.py --mode seed_summary --reference p3b_joint
  ```
- [✅] Back up (command in 0.2).

## G. Mechanism and secondary arms

- [Skipped] **G1 (mandatory) — data volume vs source diversity**: `p3c_dose5k`, `p3c_dose2k` are already trained;
  they are evaluated in F1. Nothing to run here.
- [Skipped] *G2 (optional) — SSL dose point*: `p3a_ssl60` is already trained; evaluated in F1. Nothing to run here.
- [✅] *G2b (optional, recommended) — joint training + EyePACS-supervised pretraining* (≈ 4.7 GPU h, no dependency).
  Under the amended endpoint phase 3a would adopt `p3a_sup`, while the lineage (`p3b_joint`) was built without it;
  this one arm answers "why not both?". Report-only (`winner_eligible=false`), own folder, lineage unchanged:
  ```bash
  mkdir -p outputs/phase3c_jointsup
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase3c_jointsup/p3c_joint_sup.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase3c_jointsup --tag p3c_joint_sup --inherit p3b_joint --set pretrain_path=checkpoints/pretrain/eyepacs_sup_v1.pth winner_eligible=false
  ```
- [✅] *G3 (optional, recommended) — phase 5 architectures on the new lineage* (≈ 12 GPU h). Reviewers ask whether
  the effect depends on the backbone. Needs E2. The two training lines are PARALLEL; the ensemble needs both DONE;
  the selection needs the ensemble DONE:
  ```bash
  mkdir -p outputs/phase5_architecture   # the -l log paths go in here
  # ✅
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase5_architecture/p5_convnextv2.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase5_architecture --tag p5_convnextv2 --inherit @phase4j_label_loss --set backbone=convnextv2_large.fcmae_ft_in22k_in1k_384 pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
  # ✅
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase5_architecture/p5_effnetv2.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase5_architecture --tag p5_effnetv2   --inherit @phase4j_label_loss --set backbone=efficientnetv2_rw_m.agc_in1k pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false

  # SEQUENTIAL after both are DONE:
  # ✅
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase5_architecture/p5_hetero_ens.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode ensemble_experiment --phase phase5_architecture --tag p5_hetero_ens --members @phase4j_label_loss,p5_convnextv2,p5_effnetv2

  # ✅
  env/bin/python main.py --mode phase_select --phase phase5_architecture --reference @phase4j_label_loss
  ```
  If a ConvNeXt run goes out of memory at 512 px, re-run only that line with `img_size=384` added to its `--set`.
- [✅] *G3b (optional, recommended for TMI) — retinal foundation-model baseline (RETFound)* (≈ 2 × 2 GPU h at 224 px).
  Reviewers of a 2026 DR paper will ask how the model compares with a retinal foundation model. The weights are
  gated: on a machine with internet, log in to Hugging Face, accept the terms of `YukunZhou/RETFound_mae_natureCFP`,
  download its `.pth` file and copy it to the server as `checkpoints/pretrain/RETFound_mae_natureCFP.pth`. Then
  (PARALLEL, no other dependency; check the current RETFound README for the fine-tuning recipe before the run):
  ```bash
  ls -lh checkpoints/pretrain/RETFound_mae_natureCFP.pth
  mkdir -p outputs/phase5_architecture   # the -l log paths go in here
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase5_architecture/p5_retfound.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase5_architecture --tag p5_retfound       --inherit p2b_none  --set backbone=vit_large_patch16_224 pretrained=false pretrain_path=checkpoints/pretrain/RETFound_mae_natureCFP.pth backbone_global_pool=avg img_size=224 lr=3.0e-4 llrd=0.65 drop_path=0.2 mixstyle=false winner_eligible=false
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase5_architecture/p5_retfound_joint.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase5_architecture --tag p5_retfound_joint --inherit p3b_joint --set backbone=vit_large_patch16_224 pretrained=false pretrain_path=checkpoints/pretrain/RETFound_mae_natureCFP.pth backbone_global_pool=avg img_size=224 lr=3.0e-4 llrd=0.65 drop_path=0.2 mixstyle=false winner_eligible=false
  ```
  (CAM-based XAI is not available for this ViT baseline; it is a performance baseline only.)
- [Skipped] *G4 (optional) — phase 6 domain generalisation, 4 cells, PARALLEL* (≈ 16 GPU h). The first three need E2;
  `p6_amp_nojoint` has no dependency (wave 5). Report-only selection after all four are DONE:
  ```bash
  mkdir -p outputs/phase6_domain_gen   # the -l log paths go in here
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase6_domain_gen/p6_amp.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp         --inherit @phase4j_label_loss --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase6_domain_gen/p6_ms.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase6_domain_gen --tag p6_ms          --inherit @phase4j_label_loss --set mixstyle=true ms_p=0.5 ms_alpha=0.1 winner_eligible=false
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase6_domain_gen/p6_amp_ms.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_ms      --inherit @phase4j_label_loss --set amp_mix=true amp_stage=pre amp_beta_max=0.10 mixstyle=true winner_eligible=false
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase6_domain_gen/p6_amp_nojoint.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode experiment --phase phase6_domain_gen --tag p6_amp_nojoint --inherit p2b_none           --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
  # SEQUENTIAL after all four are DONE:
  env/bin/python main.py --mode phase_select --phase phase6_domain_gen --reference @phase4j_label_loss --set select_report_only=true
  ```
- Not run any more: phase 7 (feature balance, code removed), the 4-way TTA grid, wavelet gate, `p4_aasl` / `p4_aasl_emd` on the old lineage.
- [✅] Back up (command in 0.2).

## R. Reviewer-requested analyses (independent reviews of 2026-11-09)

Why: the E4 "prevalence" level used the grade shares of the site's own DEV labels, i.e. ORACLE prevalence
(now renamed `prevalence_oracle`); it is an upper bound, not a label-free result. Step R adds genuinely
annotation-free estimates, the standard baselines reviewers expect, clinical operating points, patient/exam-level
bootstrap, an in-distribution control, a controlled shift decomposition and the APTOS near-duplicate sensitivity
analysis. **Nothing that already ran has to be re-run**: R works on the saved predictions, plus a few GPU jobs that
only READ finished checkpoints. The headline model of the paper becomes the CE model `p3b_joint`
(the phase-4j winner `p4j_coralsrc_shared` stays in the paper as the pre-specified, non-replicating ordinal result).

Order: R0 → R1 (GPU, can start right away, in parallel with the jobs that are still running) → R2 → R3 → R4.
R5 waits for F2c, G3 and G3b.

- [✅] **R0 (mandatory) — bring the new code and test it.** On the Mac: the B1 `rsync` line. On the server:
  ```bash
  env/bin/python -m pytest -q -p no:warnings          # expected: 140 passed (≈ 10–15 min on CPU)
  ```
  No model download is needed for step R.

- [🔃] **R1 (mandatory, GPU, 9 jobs, PARALLEL; ≈ 0.3–1 GPU h each)** — target-site adaptation baselines and
  penultimate features, for the three CE runs. Each job only READS the finished checkpoints and writes into
  `results/<phase>/<tag>/target_adapt/` or `.../features/`; the run's own result files are never touched.
  ```bash
  mkdir -p outputs/recal
  for T in p3b_joint p10d_ce p10q_ce; do
    phd run -p shr_gpu -ng 1 -GR H100 -l outputs/recal/${T}_bn.txt   -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode target_adapt --tag $T --set target_adapt_method=bn
    phd run -p shr_gpu -ng 1 -GR H100 -l outputs/recal/${T}_tent.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode target_adapt --tag $T --set target_adapt_method=tent
    phd run -p shr_gpu -ng 1 -GR H100 -l outputs/recal/${T}_feat.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode extract_features --tag $T
  done
  ```
  - `bn` = BatchNorm statistics re-estimated on the UNLABELLED target DEV images; `tent` = TENT (entropy
    minimisation of the normalisation-layer affine parameters on the unlabelled target DEV images, Adam lr 1e-4,
    1 epoch, fixed a priori). TEST images are only predicted.
  - A `tent` job out of memory: re-submit that line with `--set target_adapt_method=tent grad_checkpointing=true`.
  - `extract_features` stops with an error if its features do not reproduce the saved predictions (safety check);
    `results/<phase>/<tag>/reproduction_check.csv` shows the comparison.
  - Check when all 9 have ended:
    ```bash
    ls results/phase*/{p3b_joint,p10d_ce,p10q_ce}/target_adapt/*/summary.csv results/phase*/{p3b_joint,p10d_ce,p10q_ce}/features/head.npz
    ```
    (6 `summary.csv` + 3 `head.npz` = done.)

- [] **R2 (mandatory, CPU, ≈ 5 min) — few-label last-layer fine-tuning baseline** (needs the 3 `features` jobs of R1):
  ```bash
  env/bin/python tools/fewshot_head.py --tags p3b_joint,p10d_ce,p10q_ce
  ```
  Output: `results/master/recal/fewshot_head_draws.csv`, `fewshot_head_lambda.csv` (same image draws as E4).

- [ ] **R3 (mandatory, CPU, ≈ 5–15 min) — the recalibration suite** (after R1 and R2; it merges their outputs —
  if it is run earlier the missing rows are only noted, re-running overwrites). The first line adds the
  duplicate-audit rows of the phase-10 runs (needed for the APTOS near-duplicate sensitivity analysis):
  ```bash
  env/bin/python tools/leakage_check.py --tags p2b_none,p3a_sup,p3b_joint,p4_coral,p10d_ce,p10q_ce,p4j_coralsrc_shared
  env/bin/python tools/recalibration_suite.py --tags p3b_joint,p10d_ce,p10q_ce,p4j_coralsrc_shared
  ```
  Output in `results/master/recal/`: `recal_summary.csv`, `recal_compare.csv`, `recal_draws.csv`,
  `prevalence_estimates.csv`, `binary_operating_points.csv`, `ordinal_discrimination.csv`, `indist_control.csv`,
  `aptos_dup_sensitivity.csv`, `recal_fitted.csv`, `recal_notes.csv`; figures `results/figures/fig_recal_*`.
  Also written: annotation-free prevalence with bootstrap uncertainty (`--B_pi 200` resamples of the unlabelled
  DEV images, patient/exam-level where groups exist) and two uncertainty-aware variants (`*_bagged_quantile`,
  `*_shrunk_quantile`); `MEAN_COMMON` rows (mean over the sites held out by every tag on the command line);
  few-label intervals are joint (draws × patient/exam-level TEST bootstrap; column `interval_source`).
  Every row has an `info` column naming the target information it used (none, unlabelled_target,
  prevalence_oracle, prevalence_other_sites, prevalence_misspecified, n_labels, all_dev_labels). Rows with
  "oracle" in the method name are upper bounds only. `recal_notes.csv` lists everything skipped and why.

- [✅] **R4 (mandatory, CPU, ≈ 5–15 min) — controlled shift decomposition** (independent of R1–R3):
  ```bash
  env/bin/python tools/shift_decomposition.py --tags p3b_joint,p10d_ce,p10q_ce
  ```
  Output: `results/master/recal/shift_decomposition.csv`, `shift_decomposition_contrasts.csv`,
  `results/figures/fig_shift_decomposition.*`. Scenario A = pure prior shift built from APTOS images,
  B = the site's own images resampled to the APTOS grade mix, C = observed, A0 = APTOS images at the APTOS grade
  mix (no shift; controls for the fact that QWK-optimal cut-points beat argmax even without any shift).
  Read the contrasts B−A0 and C−A (annotation/acquisition effect at equal grade shares) and A−A0, C−B
  (prior-shift effect). A0/A/B are semi-synthetic resamplings of saved predictions (stated in the paper).

- [✅] **R5 (mandatory, CPU) — replicate R3 on seeds and backbones** (after F2c, G3 and G3b are DONE; separate
  folders, nothing is overwritten):
  ```bash
  env/bin/python tools/recalibration_suite.py --out_dir results/master/recal_seeds --tags p3c_joint_s1,p3c_joint_s2,p10d_ce_s1,p10d_ce_s2,p10d_coralsrc_s1,p10d_coralsrc_s2
  env/bin/python tools/recalibration_suite.py --out_dir results/master/recal_backbones --tags p5_convnextv2,p5_effnetv2,p5_retfound,p5_retfound_joint
  ```
  (A tag that does not exist yet is reported and skipped; run the line again when it exists.)

  **Pre-declared rules for step R (written 2026-11-09, before any R result exists; do not change after looking):**
  - Primary 1 (annotation-free claim): method `mlls_bcts_quantile` (prior estimated by EM from UNLABELLED target
    DEV images, then quantile cut-points) vs `argmax`; QWK; `dataset = MEAN` in `recal_compare.csv`
    (patient/exam-clustered bootstrap). Claim "annotation-free ordinal recalibration improves cross-dataset
    grading" only if `diff_lo > 0` for **all three** of `p3b_joint`, `p10d_ce`, `p10q_ce`, and the sign holds in R5
    for the CE seed runs.
  - Primary 2 (beyond the standard method): `mlls_bcts_quantile` vs `mlls_bcts_bayes` (same estimated prior,
    standard Bayes correction), same rule. If it fails, the paper says the cut-point form is not better than
    standard prior correction.
  - Pre-declared secondary (added 2026-11-10, still before any R result): the uncertainty-aware variants
    `mlls_bcts_bagged_quantile` and `mlls_bcts_shrunk_quantile` (prior bootstrapped over unlabelled DEV images;
    bagged cut-points, or prior shrunk toward the source prior when the estimate is noisy) vs `mlls_bcts_quantile`,
    same rule. They become the proposed method only if they beat `mlls_bcts_quantile` with `diff_lo > 0` in all
    three tags; otherwise they are reported as variants. Few-label intervals are the JOINT ones (draws × clustered
    TEST bootstrap). Cross-tag statements use the `MEAN_COMMON` rows (sites held out by every tag).
  - Reported whatever the outcome: every method and info level (oracle rows only as upper bounds), prevalence
    estimation error, misspecification curve, few-label comparison at n = 25 (`fewshot_shrink` vs
    `fewshot_head_ft`, `fewshot_temp_cuts`, `fewshot_plain`), BN-adapt and TENT, referable sensitivity at DEV-chosen
    specificity 0.90/0.95/0.97/0.99 with referrals and missed cases per 1,000, mild recall, in-distribution control
    (net external gain), shift decomposition, APTOS near-duplicate sensitivity, seeds and backbones.
  - Framing decided by the outcome, not afterwards: Primary 1 holds → method paper (target IEEE JBHI);
    Primary 1 fails → multi-dataset benchmark/analysis paper (target Computers in Biology and Medicine).

- [ ] Back up (command in 0.2).

## H. Final model, report, XAI

FINAL = winner of phase 4j under the amended endpoint (pre-declared rule) = `p4j_coralsrc_shared`; it did not
replicate in phase 10, so after the external reviews the paper's HEADLINE model is the CE model `p3b_joint`
(step R). H1 and H2 are run for both; the headline is used for the paper tables (H3). H1 and H2 can start now;
H3 must be the very last computation.

- [🔃] **H1 (mandatory) — inference products of FINAL** (no training). The TTA job is GPU; conformal calibration
  works on saved predictions (CPU). Independent of each other:
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  mkdir -p outputs/phase8_inference   # the -l log paths go in here
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase8_inference/p8_tta_flip.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_flip --base $FINAL --set tta=flip
  env/bin/python main.py --mode calibrate_conformal --phase phase8_inference --tag p8_conformal --base $FINAL --set conformal_alpha=0.10
  ```
  Headline CE model `p3b_joint` (same two products; conformal is CPU, TTA is 1 GPU job — can be submitted together with R1):
  ```bash
  env/bin/python main.py --mode calibrate_conformal --phase phase8_inference --tag p8_conformal_ce --base p3b_joint --set conformal_alpha=0.10
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/phase8_inference/p8_tta_flip_ce.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode inference_experiment --phase phase8_inference --tag p8_tta_flip_ce --base p3b_joint --set tta=flip
  ```
- [🔃] **H2 (mandatory) — XAI** (grid + expert-mask localisation on DDR / IDRiD lesion masks; final vs baseline).
  GPU, the four jobs are PARALLEL:
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  mkdir -p outputs/xai   # the -l log paths go in here
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/xai/${FINAL}_grid_test.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode xai --tag $FINAL --set xai_split=test
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/xai/${FINAL}_grid_messidor.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode xai --tag $FINAL --set xai_split=messidor_test xai_top_k=1
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/xai/${FINAL}_loc.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode xai_localization --tag $FINAL
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/xai/p2a_plain_loc.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode xai_localization --tag p2a_plain
  # headline CE model (same jobs, PARALLEL with the ones above)
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/xai/p3b_joint_grid_test.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode xai --tag p3b_joint --set xai_split=test
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/xai/p3b_joint_loc.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode xai_localization --tag p3b_joint
  ```
  Output: `results/<phase>/<tag>/xai/` (grid + `xai_localization.csv`), copied grid in `results/figures/xai_grid.*`.
- [ ] **H3 (mandatory, CPU) — tables and figures**, SEQUENTIAL, after every run you want in the paper is DONE
  (only real data is used; a table or figure without inputs is skipped and listed in
  `results/tables/TABLES_README.txt` / `results/figures/FIGURES_README.txt`, never filled in):
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  env/bin/python main.py --mode master_table
  env/bin/python main.py --mode seed_summary --reference p3b_joint
  env/bin/python main.py --mode report --set report_final=p3b_joint report_baseline=p2a_plain soft_label_run=sl_p4
  cat results/tables/TABLES_README.txt results/figures/FIGURES_README.txt
  ```
  If a README lists a missing input that should exist, run the step that makes it and re-run H3 (it overwrites).
  The literature comparison table is typed in by the authors from verified papers.
- [ ] **H4 (mandatory) — back up and clean up**:
  ```bash
  mkdir -p backups && tar czf backups/final_$(date +%Y%m%d_%H%M).tgz results/ splits/ softlabels/ \
      checkpoints/*/config_snapshot.yaml checkpoints/*/thresholds.json
  ```
  Then `SERVER_CLEANUP.md` sections 1, 3 and 4 only (section 2 would delete the final tables you just built).

---

## If something goes wrong

| Message / symptom | Fix |
|---|---|
| `--set key ... does not exist` | typo in the key; the message lists close matches |
| `Evaluation inputs missing` before training starts | C2 not done, or a path in `configs/config.yaml -> paths` is wrong |
| `phase_select ...: DEV predictions missing for [...]` | C4 (or `--mode evaluate --tag X`) not finished for the listed tags |
| `AASL soft labels for fold k not found` | wait until `softlabels/sl_p4/outer0..4.npz` all exist |
| `coral / coral_src heads ... set mix_prob=0` | ordinal heads cannot use Mixup/CutMix (the lineage already has `mix_prob=0`) |
| `ensemble member X has no predictions for <set>` | run `--mode evaluate --tag X` first (same `phd run` line as in C4) |
| out of memory | add `batch_size=8 grad_accum=2 grad_checkpointing=true` to that line's `--set` and re-submit (finished folds are skipped) |
| `@phaseX` / `WINNER.txt` not found | run `phase_select` for that phase first |
| `phase_select` raises about test data | expected guard; do not bypass it |
| a job died (node failure, time limit) | re-submit exactly the same line; it resumes from the last finished fold |
| a results folder lost while `checkpoints/` survived | do NOT re-run `experiment` (it retrains); rebuild it from the fold checkpoints with the GPU job below, then re-run that phase's `phase_select` |

Rebuilding a lost results folder (inference only; one job per tag, e.g. `p2a_reg` of `phase2a_recipe`):
```bash
phd run -p shr_gpu -ng 1 -GR H100 -l outputs/rebuild/p2a_reg.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
  env/bin/python tools/rebuild_oof.py --tag p2a_reg --phase phase2a_recipe
```
(`mkdir -p outputs/rebuild` first.)

Never write `WINNER.txt` by hand, never run `make_splits` again on the server, never copy `splits/` or
`results/` from the Mac.

---

## Total GPU budget of this list
Mandatory: C4 ≈ 7, E1 ≈ 28, E3 ≈ 28, F1 ≈ 2 (evaluation only), F2 ≈ 44, H ≈ 2 → **≈ 111 GPU h**
(≈ 1.5–2 days of wall clock at 4–6 parallel slots). Optional G2b–G4: ≈ 37 GPU h. The SSL stop (A1) saves 52.
Step R: ≈ 5–9 GPU h (R1, 9 jobs) + CPU minutes.
