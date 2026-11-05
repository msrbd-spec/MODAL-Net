# SERVER TODO — every step of the paper, in order (state of 2026-11-04)

This file is enough to run the whole project: every command is written here in full, so you do not need
`commands.md` (it stays as the reference with more background). What can be deleted, and when, is in
`SERVER_CLEANUP.md`.

Tick the boxes yourself as you go. **Mandatory** steps are needed for the paper; *optional* steps can be
skipped without breaking anything later.

## Daily report (upload these files to Claude once a day)

**Start:** after **D3** is done. Then submit the wave-5 jobs that do not depend on the plan (F1, F2a
`p9_base_*`, G1, G2, G3b), but submit **E1 / E3 only after Claude's feedback on that first report**.

Before uploading, refresh the master tables on the server (CPU, ≈ 1 min):
```bash
env/bin/python main.py --mode seed_summary --reference p3b_joint && env/bin/python main.py --mode master_table
```

| # | Server path | When |
|---|---|---|
| 1 | `results/master/master_full.csv` | every report (every run × every set × every metric) |
| 2 | `results/master/seed_compare.csv` | every report once seed runs exist |
| 3 | `results/<phase>/phase_table.csv` | for each phase selected since the last report |
| 4 | `results/master/cutpoint_diagnostic.csv` | first report (and whenever C5 is re-run) |
| 5 | `results/master/leakage_check.csv` | first report only |

---

## 0. How to use this file

### 0.1 Every time you log in (paste once per terminal)
```bash
cd /home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# gpu <log-name> <main.py arguments...>   -> submits one GPU job; its log goes to outputs/<log-name>.txt
gpu() {
  local log="outputs/$1.txt"; shift
  mkdir -p "$(dirname "$log")"
  phd run -p shr_gpu -ng 1 -GR H100 -l "$log" -- \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    env/bin/python main.py "$@"
}

# st <tag> [<tag> ...]   -> prints DONE when the run has finished (results/<phase>/<tag>/summary.csv exists)
st() {
  for t in "$@"; do
    if ls results/phase*/"$t"/summary.csv >/dev/null 2>&1; then echo "DONE     $t"; else echo "not yet  $t"; fi
  done
}
```

### 0.2 Reading the steps
- **SEQUENTIAL** = wait until the previous step has finished (check with `st <tag>`, the log, or the
  check line given in the step) before you start the next one.
- **PARALLEL** = submit all lines of the block at the same time, each line is its own GPU job.
  If `phd run` keeps your terminal busy until the job ends, open one `tmux` window per line.
- CPU steps (no `gpu` in front) run directly in the terminal. Run long CPU steps inside `tmux`/`screen`
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
| 5 | E1 (6 jobs) + E3 (6 jobs) + F1 (3) + G1 (2) + `p9_base` seeds (2) + optional G2, G3b | GPU, all parallel | D2 says `p3b_joint` |
| 6 | E2, E3 selections | CPU, seconds | all E1 / E3 runs DONE |
| 7 | F2 (rest of the seeds), optional G3, G4 | GPU, parallel | E2 (FINAL known) |
| 8 | F3, H1, H2 | CPU + GPU | F2 done (H1/H2 may start right after E2) |
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
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/c4/eval_part3.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode evaluate_all --set 'evaluate_only=[p4_coral,p4_emd,p3c_none_s1,p3c_none_s2,p3c_joint_s1,p3c_coral_s1,p3c_sup_s1]'
  ```

  (Single-job alternative: `gpu c4/eval_all --mode evaluate_all`.) An `evaluate_only` name without a results
  folder stops that job at once with the list of unknown names: remove the name from the list and re-submit.
  Check when all three jobs have ended:
  ```bash
  cat results/master/evaluate_all_report.csv            # every row must say ok (zoo rows say skipped)
  ```
  A row that is not `ok`: re-run only that tag, e.g. `gpu c4/eval_p3b_joint --mode evaluate --tag p3b_joint`.
- [⛔] **C4b (mandatory, CPU, seconds) — interim seed and master tables** (needs C4). They now hold every finished
  run, the A2 seed runs included, on every internal and external set:
  ```bash
  env/bin/python main.py --mode seed_summary --reference p3b_joint     # results/master/seed_variance.csv + seed_compare.csv
  env/bin/python main.py --mode master_table                           # results/master/master_table.csv/.tex, master_full.csv, master_long.csv
  ```
  These are interim tables: they are overwritten in F3 and H3, so it does not matter how often you run them.
  Run them only after C4: before C4 the DDR / DeepDRiD / IDRiD columns are empty.

  ### Outputs:
  2026-11-05 16:02:43 | INFO | Mode: seed_summary  Tag: default  Phase: None
/home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR/env/lib/python3.9/site-packages/scipy/stats/_axis_nan_policy.py:531: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
  res = hypotest_fun_out(*samples, **kwds)
seed_summary: 26 configuration groups -> results/master/seed_variance.csv; comparisons vs p3b_joint -> results/master/seed_compare.csv
2026-11-05 16:02:50 | INFO | Mode: master_table  Tag: default  Phase: None
master_table: Compiled 44 experiments into results/master/

- [✅] **C5 (mandatory, CPU, seconds) — cut-point diagnostic**, the pre-test of the new component (needs C4):
  ```bash
  env/bin/python tools/cutpoint_diagnostic.py --tags p2b_none,p3a_sup,p3b_joint,p4_coral
  ```
  The first table (`devcf_*`, cross-fitted inside each DEV split, no TEST image used) is the evidence that
  motivates `coral_src`: a positive `devcf_qwk_gain` interval on most external datasets means "the model ranks
  images well; only its cut-points are wrong for a new site". The second table (TEST, cut-points from the whole
  DEV: `qwk_gain`, `mild_recall_argmax → mild_recall_devcal`, `pred_nodr_argmax → pred_nodr_devcal`, next to
  `test_bin_auroc`) is for reporting only. Whatever the outcome, both tables go into the paper (pre-declared here).

  ### Result:
        tag  dataset  n_dev  devcf_qwk_argmax  devcf_qwk_refit  devcf_qwk_gain  devcf_gain_lo  devcf_gain_hi
 p2b_none messidor    349             0.662            0.751           0.090          0.031          0.133
 p2b_none      ddr   2445             0.736            0.841           0.105          0.079          0.124
 p2b_none deepdrid    399             0.767            0.819           0.052          0.002          0.088
 p2b_none    idrid    413             0.816            0.858           0.042          0.011          0.074
  p3a_sup messidor    349             0.702            0.775           0.073          0.036          0.115
  p3a_sup      ddr   2445             0.783            0.886           0.103          0.084          0.118
  p3a_sup deepdrid    399             0.788            0.852           0.064          0.028          0.096
  p3a_sup    idrid    413             0.809            0.887           0.078          0.054          0.101
p3b_joint messidor    349             0.765            0.793           0.028         -0.017          0.061
p3b_joint      ddr   2445             0.838            0.884           0.047          0.032          0.062
p3b_joint deepdrid    399             0.849            0.868           0.019         -0.013          0.049
p3b_joint    idrid    413             0.830            0.889           0.059          0.040          0.077
 p4_coral messidor    349             0.690            0.768           0.078          0.043          0.120
 p4_coral      ddr   2445             0.755            0.857           0.102          0.083          0.124
 p4_coral deepdrid    399             0.781            0.839           0.058          0.014          0.086
 p4_coral    idrid    413             0.833            0.881           0.049          0.026          0.074

      tag  dataset  n_test  test_bin_auroc  qwk_argmax  qwk_devcal  qwk_gain  qwk_gain_ci_lo  qwk_gain_ci_hi  mild_recall_argmax  mild_recall_devcal  pred_nodr_argmax  pred_nodr_devcal  true_nodr_share
 p2b_none messidor    1395           0.936       0.698       0.792     0.094           0.069           0.118               0.104               0.335            74.265            56.559           58.423
 p2b_none      ddr    3638           0.926       0.608       0.775     0.166           0.148           0.184               0.220               0.231            67.922            55.607           49.093
 p2b_none deepdrid     399           0.964       0.833       0.867     0.034           0.011           0.057               0.457               0.400            52.882            50.125           50.125
 p2b_none    idrid     103           0.899       0.676       0.730     0.054          -0.026           0.133               0.800               0.400            12.621            27.184           33.010
  p3a_sup messidor    1395           0.946       0.741       0.811     0.070           0.050           0.091               0.283               0.528            68.172            54.050           58.423
  p3a_sup      ddr    3638           0.943       0.669       0.828     0.158           0.140           0.176               0.328               0.339            65.943            48.955           49.093
  p3a_sup deepdrid     399           0.981       0.861       0.890     0.028           0.007           0.051               0.543               0.600            52.632            44.361           50.125
  p3a_sup    idrid     103           0.930       0.682       0.798     0.115           0.043           0.191               0.600               0.200            21.359            33.010           33.010
p3b_joint messidor    1395           0.915       0.802       0.823     0.021          -0.001           0.044               0.330               0.250            68.100            62.151           58.423
p3b_joint      ddr    3638           0.906       0.731       0.814     0.082           0.067           0.098               0.446               0.172            63.579            54.233           49.093
p3b_joint deepdrid     399           0.986       0.908       0.909     0.001          -0.017           0.018               0.543               0.571            51.128            46.867           50.125
p3b_joint    idrid     103           0.927       0.710       0.781     0.072           0.002           0.147               0.800               0.200            20.388            33.010           33.010
 p4_coral messidor    1395           0.927       0.716       0.806     0.090           0.066           0.113               0.137               0.425            73.333            55.556           58.423
 p4_coral      ddr    3638           0.891       0.611       0.790     0.178           0.158           0.198               0.129               0.285            70.533            49.258           49.093
 p4_coral deepdrid     399           0.961       0.828       0.858     0.030           0.004           0.058               0.371               0.429            58.145            45.865           50.125
 p4_coral    idrid     103           0.914       0.655       0.762     0.107           0.035           0.185               0.800               0.400            15.534            30.097           33.010

written: results/master/cutpoint_diagnostic.csv


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
2026-11-05 16:51:33 | INFO | Mode: phase_select  Tag: default  Phase: phase2a_recipe
phase_select phase2a_recipe (amended, report only): best arm p2a_plain; WINNER.txt and summaries untouched -> results/phase2a_recipe/phase_table_amended.csv
2026-11-05 16:52:22 | INFO | Mode: phase_select  Tag: default  Phase: phase2b_preprocess
phase_select phase2b_preprocess (amended, report only): best arm p2b_none; WINNER.txt and summaries untouched -> results/phase2b_preprocess/phase_table_amended.csv
2026-11-05 16:52:46 | INFO | Mode: phase_select  Tag: default  Phase: phase3a_pretrain
phase_select phase3a_pretrain (amended, report only): best arm p3a_sup; WINNER.txt and summaries untouched -> results/phase3a_pretrain/phase_table_amended.csv
2026-11-05 16:53:16 | INFO | Mode: phase_select  Tag: default  Phase: phase3a_pretrain
phase_select phase3a_pretrain (original, report only): best arm p2b_none; WINNER.txt and summaries untouched -> results/phase3a_pretrain/phase_table_original.csv

- [✅] **D2 (mandatory) — phase 3b decision under the amended endpoint** (rewrites `results/phase3b_joint/WINNER.txt`):
  ```bash
  env/bin/python main.py --mode phase_select --phase phase3b_joint --reference p2b_none --set select_endpoint=amended
  env/bin/python main.py --mode phase_select --phase phase3b_joint --reference p2b_none --set select_endpoint=original select_report_only=true
  head -n 1 results/phase3b_joint/WINNER.txt
  ```
  Expected `p3b_joint`. If another tag wins, **stop here** and look at `phase_table.csv` before step E (the new
  lineage below inherits `p3b_joint` by name, so a different winner means the plan must be re-discussed).
  Output: 
  2026-11-05 17:06:42 | INFO | Mode: phase_select  Tag: default  Phase: phase3b_joint
phase_select for phase3b_joint (amended): WINNER is p3b_joint. Table written to results/phase3b_joint/phase_table.csv
2026-11-05 17:06:59 | INFO | Mode: phase_select  Tag: default  Phase: phase3b_joint
phase_select phase3b_joint (original, report only): best arm p2b_none; WINNER.txt and summaries untouched -> results/phase3b_joint/phase_table_original.csv
p3b_joint

- [⛔] **D3 (mandatory) — old phase 4 (on `p2b_none`) as a report-only table**:
  ```bash
  env/bin/python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_endpoint=amended select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_endpoint=original select_report_only=true
  ```

  ### First Output:
  2026-11-05 17:08:58 | INFO | Mode: phase_select  Tag: default  Phase: phase4_label_loss
Traceback (most recent call last):
  File "/home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR/main.py", line 188, in <module>
    main()
  File "/home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR/main.py", line 157, in main
    phase_select(args.phase, args.reference, design=getattr(args, 'design', 'parallel'), cfg=cfg)
  File "/home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR/src/evaluation/tables.py", line 300, in phase_select
    raise FileNotFoundError(
FileNotFoundError: phase_select phase4_label_loss: DEV predictions missing for ['p4_aasl:ddr', 'p4_aasl:deepdrid', 'p4_aasl:idrid']. Run --mode evaluate (or evaluate_all) for these tags first, or restrict select_dev_sets.
2026-11-05 17:09:06 | INFO | Mode: phase_select  Tag: default  Phase: phase4_label_loss
phase_select phase4_label_loss (original, report only): best arm p4_coral; WINNER.txt and summaries untouched -> results/phase4_label_loss/phase_table_original.csv

### How I fix them:
```bash
  ls results/phase4_label_loss/
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/c4/eval_p4_aasl.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode evaluate --tag p4_aasl
  phd run -p shr_gpu -ng 1 -GR H100 -l outputs/c4/eval_p4_aasl_emd.txt -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/bin/python main.py --mode evaluate --tag p4_aasl_emd
  env/bin/python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_endpoint=amended select_report_only=true
  env/bin/python -c "import pandas as pd; t=pd.read_csv('results/phase4_label_loss/phase_table_amended.csv'); print(t[['experiment','oof_qwk','d_oof_qwk','d_dev_mean_qwk','d_dev_mean_qwk_ci_lo','dev_sets_used','decision_amended']].to_string(index=False))"
  env/bin/python main.py --mode seed_summary --reference p3b_joint && env/bin/python main.py --mode master_table
```

### Final output:
env/bin/python main.py --mode phase_select --phase phase4_label_loss --reference p2b_none --set select_endpoint=amended select_report_only=true
2026-11-05 17:33:31 | INFO | Mode: phase_select  Tag: default  Phase: phase4_label_loss
phase_select phase4_label_loss (amended, report only): best arm p4_coral; WINNER.txt and summaries untouched -> results/phase4_label_loss/phase_table_amended.csv
env/bin/python -c "import pandas as pd; t=pd.read_csv('results/phase4_label_loss/phase_table_amended.csv'); print(t[['experiment','oof_qwk','d_oof_qwk','d_dev_mean_qwk','d_dev_mean_qwk_ci_lo','dev_sets_used','decision_amended']].to_string(index=False))"
 experiment  oof_qwk  d_oof_qwk  d_dev_mean_qwk  d_dev_mean_qwk_ci_lo               dev_sets_used decision_amended
    p4_aasl     0.93       0.01            0.01                 -0.00 messidor,ddr,deepdrid,idrid           REJECT
p4_aasl_emd     0.93       0.00            0.01                  0.00 messidor,ddr,deepdrid,idrid           REJECT
   p4_coral     0.93       0.01            0.02                  0.01 messidor,ddr,deepdrid,idrid            ADOPT
     p4_emd     0.93      -0.00           -0.02                 -0.04 messidor,ddr,deepdrid,idrid           REJECT


  `p4_aasl` / `p4_aasl_emd` on the old lineage are **not** run any more; they are run on the new lineage (E1).
- [✅] Back up (command in 0.2).

## E. New lineage: ordinal heads and the cut-point component (GPU)

E1 and E3 are independent (both inherit `p3b_joint`): submit all twelve jobs together if you have the slots,
plus the wave-5 jobs of F1 / G1 below. Every run evaluates itself on all sets with all metrics at the end
(no separate evaluate step needed).

- [🔃] **E1 (mandatory) — phase 4j, six arms, PARALLEL** (≈ 6 × 4.7 = 28 GPU h). Needs D2 = `p3b_joint`.
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
  st p4j_coral p4j_emd p4j_aasl p4j_aasl_emd p4j_coralsrc_shared p4j_coralsrc
  ```
  (`sl_p4` soft labels were made on the same APTOS folds; they are re-used on purpose. `coral_src` = one
  severity score + ordered cut-points; `src_offsets=true` adds one cut-point offset vector per training source,
  tied by `src_offset_l2`; `src_offsets=false` is the same head without offsets, the ablation that isolates them.)
- [ ] **E2 (mandatory, CPU, seconds) — phase 4j selection**. SEQUENTIAL: only when `st` shows all six E1 runs DONE.
  ```bash
  env/bin/python main.py --mode phase_select --phase phase4j_label_loss --reference p3b_joint
  env/bin/python main.py --mode phase_select --phase phase4j_label_loss --reference p3b_joint --set select_endpoint=original select_report_only=true
  head -n 1 results/phase4j_label_loss/WINNER.txt      # = the final model (FINAL) used from step F2 on
  ```
  From now on every new terminal also needs:
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt); echo $FINAL
  ```
- [ ] **E3 (mandatory) — phase 10, multi-source leave-sites-out, PARALLEL** (≈ 6 × 4.7 = 28 GPU h). Needs D2.
  This is the multi-site test of the cut-point idea: each arm is scored only on the datasets it never saw.
  ```bash
  # sources APTOS + EyePACS + DDR  -> held out: Messidor-2, DeepDRiD, IDRiD
  gpu phase10_ms_ddr/p10d_ce       --mode experiment --phase phase10_ms_ddr --tag p10d_ce       --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,ddr]'
  gpu phase10_ms_ddr/p10d_coralsh  --mode experiment --phase phase10_ms_ddr --tag p10d_coralsh  --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,ddr]' loss=coral head_type=coral_src src_offsets=false
  gpu phase10_ms_ddr/p10d_coralsrc --mode experiment --phase phase10_ms_ddr --tag p10d_coralsrc --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,ddr]' loss=coral head_type=coral_src src_offsets=true
  # sources APTOS + EyePACS + DeepDRiD -> held out: Messidor-2, DDR, IDRiD
  gpu phase10_ms_deepdrid/p10q_ce       --mode experiment --phase phase10_ms_deepdrid --tag p10q_ce       --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,deepdrid]'
  gpu phase10_ms_deepdrid/p10q_coralsh  --mode experiment --phase phase10_ms_deepdrid --tag p10q_coralsh  --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,deepdrid]' loss=coral head_type=coral_src src_offsets=false
  gpu phase10_ms_deepdrid/p10q_coralsrc --mode experiment --phase phase10_ms_deepdrid --tag p10q_coralsrc --inherit p3b_joint --set joint_eyepacs=false 'extra_sources=[eyepacs,deepdrid]' loss=coral head_type=coral_src src_offsets=true
  ```
  Then, SEQUENTIAL (CPU, seconds), each line only when its three runs are DONE:
  ```bash
  st p10d_ce p10d_coralsh p10d_coralsrc p10q_ce p10q_coralsh p10q_coralsrc
  env/bin/python main.py --mode phase_select --phase phase10_ms_ddr      --reference p10d_ce
  env/bin/python main.py --mode phase_select --phase phase10_ms_ddr      --reference p10d_ce --set select_endpoint=original select_report_only=true
  env/bin/python main.py --mode phase_select --phase phase10_ms_deepdrid --reference p10q_ce
  env/bin/python main.py --mode phase_select --phase phase10_ms_deepdrid --reference p10q_ce --set select_endpoint=original select_report_only=true
  ```
- [ ] Back up (command in 0.2).

## F. Seeds (seed is the unit of analysis for every headline claim)

Seed replicates are grouped automatically by configuration (any folder, any tag name).

- [ ] **F1 (mandatory) — finish the phase-3c wave with the new code, PARALLEL** (≈ 10 GPU h). Can start
  together with E1 / E3 (wave 5); needs only C2. (`p3c_none_s2` finished in A2.)
  ```bash
  gpu phase3c_seeds/p3c_joint_s2 --mode experiment --phase phase3c_seeds --tag p3c_joint_s2 --inherit p3b_joint --set seed=2
  gpu phase3c_seeds/p3c_coral_s2 --mode experiment --phase phase3c_seeds --tag p3c_coral_s2 --inherit p4_coral  --set seed=2
  gpu phase3c_seeds/p3c_sup_s2   --mode experiment --phase phase3c_seeds --tag p3c_sup_s2   --inherit p3a_sup   --set seed=2
  ```
- [ ] **F2 (mandatory) — seeds 1 and 2 of the headline configurations, PARALLEL** (≈ 10 runs, ≈ 44 GPU h).
  Three groups with different start conditions:
  ```bash
  # F2a: baseline seeds, no dependency -> submit in wave 5 together with E1 / E3
  gpu phase9_final/p9_base_s1 --mode experiment --phase phase9_final --tag p9_base_s1 --inherit p2a_plain --set seed=1
  gpu phase9_final/p9_base_s2 --mode experiment --phase phase9_final --tag p9_base_s2 --inherit p2a_plain --set seed=2

  # F2b: needs p4j_coralsrc and p4j_coralsrc_shared DONE (E1)
  gpu phase4j_seeds/p4j_coralsrc_s1        --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_s1        --inherit p4j_coralsrc        --set seed=1
  gpu phase4j_seeds/p4j_coralsrc_s2        --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_s2        --inherit p4j_coralsrc        --set seed=2
  gpu phase4j_seeds/p4j_coralsrc_shared_s1 --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_shared_s1 --inherit p4j_coralsrc_shared --set seed=1
  gpu phase4j_seeds/p4j_coralsrc_shared_s2 --mode experiment --phase phase4j_seeds --tag p4j_coralsrc_shared_s2 --inherit p4j_coralsrc_shared --set seed=2

  # F2c: needs p10d_ce and p10d_coralsrc DONE (E3)
  gpu phase10_seeds/p10d_ce_s1       --mode experiment --phase phase10_seeds --tag p10d_ce_s1       --inherit p10d_ce       --set seed=1
  gpu phase10_seeds/p10d_ce_s2       --mode experiment --phase phase10_seeds --tag p10d_ce_s2       --inherit p10d_ce       --set seed=2
  gpu phase10_seeds/p10d_coralsrc_s1 --mode experiment --phase phase10_seeds --tag p10d_coralsrc_s1 --inherit p10d_coralsrc --set seed=1
  gpu phase10_seeds/p10d_coralsrc_s2 --mode experiment --phase phase10_seeds --tag p10d_coralsrc_s2 --inherit p10d_coralsrc --set seed=2
  ```
  F2d: only if FINAL (E2) is **neither** `p4j_coralsrc` **nor** `p4j_coralsrc_shared`, add its two seeds (needs E2):
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  gpu phase4j_seeds/${FINAL}_s1 --mode experiment --phase phase4j_seeds --tag ${FINAL}_s1 --inherit $FINAL --set seed=1
  gpu phase4j_seeds/${FINAL}_s2 --mode experiment --phase phase4j_seeds --tag ${FINAL}_s2 --inherit $FINAL --set seed=2
  ```
- [ ] **F3 (mandatory, CPU, seconds) — seed tables**, when every F1 / F2 run is DONE:
  ```bash
  st p3c_joint_s2 p3c_coral_s2 p3c_sup_s2 p9_base_s1 p9_base_s2 p4j_coralsrc_s1 p4j_coralsrc_s2 \
     p4j_coralsrc_shared_s1 p4j_coralsrc_shared_s2 p10d_ce_s1 p10d_ce_s2 p10d_coralsrc_s1 p10d_coralsrc_s2
  env/bin/python main.py --mode seed_summary --reference p3b_joint
  ```
- [ ] Back up (command in 0.2).

## G. Mechanism and secondary arms

- [ ] **G1 (mandatory) — data volume vs source diversity, PARALLEL** (≈ 9.4 GPU h). Wave 5 (needs only C2):
  ```bash
  gpu phase3c_dose/p3c_dose5k --mode experiment --phase phase3c_dose --tag p3c_dose5k --inherit p3b_joint --set eyepacs_max=5000
  gpu phase3c_dose/p3c_dose2k --mode experiment --phase phase3c_dose --tag p3c_dose2k --inherit p3b_joint --set eyepacs_max=2000
  ```
- [ ] *G2 (optional) — SSL dose point* (≈ 2.8 GPU h) for the SSL negative-control paragraph. Wave 5:
  ```bash
  gpu phase3c_ssl/p3a_ssl60 --mode experiment --phase phase3c_ssl --tag p3a_ssl60 --inherit p2b_none --set pretrain_path=checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch60.pth
  ```
- [ ] *G3 (optional, recommended) — phase 5 architectures on the new lineage* (≈ 12 GPU h). Reviewers ask whether
  the effect depends on the backbone. Needs E2. The two training lines are PARALLEL; the ensemble needs both DONE;
  the selection needs the ensemble DONE:
  ```bash
  gpu phase5_architecture/p5_convnextv2 --mode experiment --phase phase5_architecture --tag p5_convnextv2 --inherit @phase4j_label_loss --set backbone=convnextv2_large.fcmae_ft_in22k_in1k_384 pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
  gpu phase5_architecture/p5_effnetv2   --mode experiment --phase phase5_architecture --tag p5_effnetv2   --inherit @phase4j_label_loss --set backbone=efficientnetv2_rw_m.agc_in1k pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false
  # SEQUENTIAL after both are DONE:
  gpu phase5_architecture/p5_hetero_ens --mode ensemble_experiment --phase phase5_architecture --tag p5_hetero_ens --members @phase4j_label_loss,p5_convnextv2,p5_effnetv2
  env/bin/python main.py --mode phase_select --phase phase5_architecture --reference @phase4j_label_loss
  ```
  If a ConvNeXt run goes out of memory at 512 px, re-run only that line with `img_size=384` added to its `--set`.
- [ ] *G3b (optional, recommended for TMI) — retinal foundation-model baseline (RETFound)* (≈ 2 × 2 GPU h at 224 px).
  Reviewers of a 2026 DR paper will ask how the model compares with a retinal foundation model. The weights are
  gated: on a machine with internet, log in to Hugging Face, accept the terms of `YukunZhou/RETFound_mae_natureCFP`,
  download its `.pth` file and copy it to the server as `checkpoints/pretrain/RETFound_mae_natureCFP.pth`. Then
  (PARALLEL, no other dependency; check the current RETFound README for the fine-tuning recipe before the run):
  ```bash
  ls -lh checkpoints/pretrain/RETFound_mae_natureCFP.pth
  gpu phase5_architecture/p5_retfound       --mode experiment --phase phase5_architecture --tag p5_retfound       --inherit p2b_none  --set backbone=vit_large_patch16_224 pretrained=false pretrain_path=checkpoints/pretrain/RETFound_mae_natureCFP.pth backbone_global_pool=avg img_size=224 lr=3.0e-4 llrd=0.65 drop_path=0.2 mixstyle=false winner_eligible=false
  gpu phase5_architecture/p5_retfound_joint --mode experiment --phase phase5_architecture --tag p5_retfound_joint --inherit p3b_joint --set backbone=vit_large_patch16_224 pretrained=false pretrain_path=checkpoints/pretrain/RETFound_mae_natureCFP.pth backbone_global_pool=avg img_size=224 lr=3.0e-4 llrd=0.65 drop_path=0.2 mixstyle=false winner_eligible=false
  ```
  (CAM-based XAI is not available for this ViT baseline; it is a performance baseline only.)
- [ ] *G4 (optional) — phase 6 domain generalisation, 4 cells, PARALLEL* (≈ 16 GPU h). The first three need E2;
  `p6_amp_nojoint` has no dependency (wave 5). Report-only selection after all four are DONE:
  ```bash
  gpu phase6_domain_gen/p6_amp         --mode experiment --phase phase6_domain_gen --tag p6_amp         --inherit @phase4j_label_loss --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
  gpu phase6_domain_gen/p6_ms          --mode experiment --phase phase6_domain_gen --tag p6_ms          --inherit @phase4j_label_loss --set mixstyle=true ms_p=0.5 ms_alpha=0.1 winner_eligible=false
  gpu phase6_domain_gen/p6_amp_ms      --mode experiment --phase phase6_domain_gen --tag p6_amp_ms      --inherit @phase4j_label_loss --set amp_mix=true amp_stage=pre amp_beta_max=0.10 mixstyle=true winner_eligible=false
  gpu phase6_domain_gen/p6_amp_nojoint --mode experiment --phase phase6_domain_gen --tag p6_amp_nojoint --inherit p2b_none           --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
  # SEQUENTIAL after all four are DONE:
  env/bin/python main.py --mode phase_select --phase phase6_domain_gen --reference @phase4j_label_loss --set select_report_only=true
  ```
- Not run any more: phase 7 (feature balance, code removed), the 4-way TTA grid, `p3c_cnext_ref`,
  `p3c_effnet_ref` (superseded by G3), wavelet gate, `p4_aasl` / `p4_aasl_emd` on the old lineage.
- [ ] Back up (command in 0.2).

## H. Final model, report, XAI

FINAL = winner of phase 4j under the amended endpoint (pre-declared rule). H1 and H2 can start as soon as E2 is
done; H3 must be the very last computation.

- [ ] **H1 (mandatory) — inference products of FINAL** (no training). The TTA job is GPU; conformal calibration
  works on saved predictions (CPU). Independent of each other:
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  gpu phase8_inference/p8_tta_flip --mode inference_experiment --phase phase8_inference --tag p8_tta_flip --base $FINAL --set tta=flip
  env/bin/python main.py --mode calibrate_conformal --phase phase8_inference --tag p8_conformal --base $FINAL --set conformal_alpha=0.10
  ```
- [ ] **H2 (mandatory) — XAI** (grid + expert-mask localisation on DDR / IDRiD lesion masks; final vs baseline).
  GPU, the four jobs are PARALLEL:
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  gpu xai/${FINAL}_grid_test     --mode xai --tag $FINAL --set xai_split=test
  gpu xai/${FINAL}_grid_messidor --mode xai --tag $FINAL --set xai_split=messidor_test xai_top_k=1
  gpu xai/${FINAL}_loc           --mode xai_localization --tag $FINAL
  gpu xai/p2a_plain_loc          --mode xai_localization --tag p2a_plain
  ```
  Output: `results/<phase>/<tag>/xai/` (grid + `xai_localization.csv`), copied grid in `results/figures/xai_grid.*`.
- [ ] **H3 (mandatory, CPU) — tables and figures**, SEQUENTIAL, after every run you want in the paper is DONE
  (only real data is used; a table or figure without inputs is skipped and listed in
  `results/tables/TABLES_README.txt` / `results/figures/FIGURES_README.txt`, never filled in):
  ```bash
  FINAL=$(head -n 1 results/phase4j_label_loss/WINNER.txt)
  env/bin/python main.py --mode master_table
  env/bin/python main.py --mode seed_summary --reference p3b_joint
  env/bin/python main.py --mode report --set report_final=$FINAL report_baseline=p2a_plain soft_label_run=sl_p4
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
| `ensemble member X has no predictions for <set>` | `gpu eval_X --mode evaluate --tag X` first |
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
Mandatory: C4 ≈ 7, E1 ≈ 28, E3 ≈ 28, F1 ≈ 10, F2 ≈ 44, G1 ≈ 9.4, H ≈ 2 → **≈ 128 GPU h**
(≈ 1.5–2 days of wall clock at 4–6 parallel slots). Optional G2–G4: ≈ 35 GPU h. The SSL stop (A1) saves 52.
