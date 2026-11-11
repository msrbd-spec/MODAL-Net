# SERVER CLEANUP — what can be deleted, where, and when

All paths are relative to the repository root on the server. Start every cleanup with:
```bash
cd /home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR
```
Each block has a `du -sh` / `ls` line that only **shows** what would go, and an `rm` line that deletes.
Run the show line first, read it, then run the `rm` line. Nothing in sections 1, 3 and 4 is needed by any
later step of `SERVER_TODO.md`.

## When to run which section

| Section | What | When (step of `SERVER_TODO.md`) | Status |
|---|---|---|---|
| 1 | big leftovers of stopped runs, test leftovers, caches | B5 (done) and again at H4 | safe to repeat any time |
| 2 | tables / figures built by the old code | **only** at B5 | **DONE — never run again** (it would delete current results) |
| 3 | feature caches of the removed phase 7 | any time | once |
| 4 | old backups | after each new backup | whenever `backups/` grows |
| 5 | checkpoints of arms not in any table | after the paper is submitted | once, optional |
| 6 | large intermediate files of the reviewer analyses (step R) | after the paper tables are final (H3) | once |

**Never delete:**
- `checkpoints/<tag>/` of any finished experiment (fold weights, `config_snapshot.yaml`, `thresholds.json`).
  `evaluate`, `--inherit`, ensembles and XAI need them.
- `splits/` (all of it, `splits/external/` included), `softlabels/sl_p4/`.
- `results/phase*/`, except the folders listed below.
- `checkpoints/pretrain/eyepacs_sup_v1.pth`, `checkpoints/pretrain/eyepacs_sup.pth`,
  `checkpoints/pretrain/ssl_sup.pth`, `checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch50.pth`,
  `.../ssl_backbone_epoch60.pth`, `checkpoints/pretrain/RETFound_mae_natureCFP.pth` (if downloaded).
- `results/phase0_protocol/audit/` (duplicate audit, hash caches, contact sheets).
- `datasets/` (except the DDR `.cache/` folder in section 1).
- `outputs/` while a job is still running (its log is written there).
- `results/<phase>/<tag>/target_adapt/` and `results/master/recal/` (small; they hold the reviewer analyses of step R).

---

## 1. Safe now (and again at H4)

Step by step; every step is independent, skip one if its show line prints nothing.

**1a. Nested soft-label inner models** (≈ 18 GB). The soft labels themselves are saved in
`softlabels/sl_p4/outer0..4.npz`; the 20 inner models are not used again.
```bash
ls softlabels/sl_p4/                      # must list outer0.npz ... outer4.npz before you delete 1a
du -sh checkpoints/sl_p4/inner
rm -rf checkpoints/sl_p4/inner
```
`checkpoints/sl_p4/` keeps its `config_snapshot.yaml` (it documents how the soft labels were made). If the
soft-label run also left fold models directly in `checkpoints/sl_p4/` (`outer*.pth`), they can go too:
```bash
ls -la checkpoints/sl_p4/
rm -f checkpoints/sl_p4/outer*.pth
```

**1b. Stopped SSL run** (≈ 4.4 GB): resume state and the unused epoch-40 backbone. SSL was stopped in A1.
```bash
du -sh checkpoints/pretrain/ssl_maxvit/ssl_training_checkpoint.pth checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch40.pth
rm -f  checkpoints/pretrain/ssl_maxvit/ssl_training_checkpoint.pth checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch40.pth
ls checkpoints/pretrain/ssl_maxvit/       # epoch50 and epoch60 must still be there
```

**1c. Leftovers of the old smoke test and of old mock runs** (small).
```bash
du -sh checkpoints/smoke results/phase0_protocol/smoke logs/smoke logs/test_fb_reweight logs/ens_test logs/p8_conf_test logs/p8_tta_flip_mock logs/default 2>/dev/null
rm -rf checkpoints/smoke results/phase0_protocol/smoke logs/smoke logs/test_fb_reweight logs/ens_test logs/p8_conf_test logs/p8_tta_flip_mock logs/default
```

**1d. Removed phase 7 folder and Python caches** (small; caches are rebuilt automatically).
```bash
rm -rf results/phase7_feature_balance .pytest_cache
find . -path ./env -prune -o -name __pycache__ -type d -prune -exec rm -rf {} +
```

**1e. Hugging Face download cache inside DDR** (≈ 78 MB of `.lock` / `.metadata` files; never read by the code,
which only reads `DR_grading/` and `lesion_segmentation/`).
```bash
du -sh datasets/DDR-dataset/.cache 2>/dev/null
rm -rf datasets/DDR-dataset/.cache
```

**1f. Test output file** (only if you saved one on the server, e.g. `test_errors.txt`):
```bash
ls -la test_errors.txt 2>/dev/null && rm -f test_errors.txt
```

## 2. Stale generated tables — ONLY once, right after the new code is deployed (SERVER_TODO B5, done)

These were built by the old code (before phase 3b and before the new datasets); `master_table` and `report`
rewrite them from the real summaries (SERVER_TODO C4b, F3, H3). Deleting them before step C avoided quoting an
old number by mistake. **Do not run this section again**: after C4b it would delete the current tables, the
leakage report and the cut-point diagnostic.
```bash
du -sh results/master results/tables results/figures
rm -f results/master/*.csv results/master/*.tex
rm -f results/tables/table*.csv results/tables/table*.tex
rm -f results/figures/fig*.png results/figures/fig*.pdf
```

## 3. Feature caches of removed code

The removed feature-balance mode cached penultimate features under `checkpoints/<owner>/extracted_features/`
(only if phase 7 was ever started). The first line shows them, the second deletes them:
```bash
find checkpoints -maxdepth 2 -type d -name extracted_features -exec du -sh {} +
find checkpoints -maxdepth 2 -type d -name extracted_features -prune -exec rm -rf {} +
```

## 4. Old backups (keep the newest two)

```bash
ls -lt backups/
# keep the two newest tarballs, delete the rest:
ls -t backups/*.tgz | tail -n +3 | xargs -r rm -f
ls -lt backups/                           # two files must remain
```
Keep the `proto_*_before_v2.tgz` of A3 as well until the paper is accepted (it is the state before the new
code); move it out of `backups/` first if you want to keep it, e.g. `mkdir -p backups_keep && mv backups/proto_*_before_v2.tgz backups_keep/`.

## 5. After the paper is submitted (optional, large)

Fold checkpoints of arms that are not in any table can be archived off the cluster. Before deleting any
`checkpoints/<tag>/`, check that no other tag uses it (via `--inherit`, `--base`, `--members` or `ckpt_from`):
```bash
TAG=p2a_sam                                                                   # example
grep -l "ckpt_from: $TAG\|base: $TAG\|inherit: $TAG\|$TAG," checkpoints/*/config_snapshot.yaml
# no output -> nothing depends on it:
du -sh checkpoints/$TAG && rm -rf checkpoints/$TAG
```
Never delete the checkpoints of `p2a_plain`, `p2b_none`, `p3a_sup`, `p3b_joint`, `p4_coral`, FINAL, the
phase-10 arms or any seed run: they are in the paper tables and the XAI figures.

## 6. After the reviewer analyses (step R) — once the paper tables are final (after H3)

**6a. Penultimate features** written by `--mode extract_features` (R1): large (≈ 0.1–0.3 GB per tag), and only
`tools/fewshot_head.py` (R2) reads them. They can be rebuilt in ≈ 0.25 GPU h per tag if ever needed.
```bash
du -sh results/phase*/*/features 2>/dev/null
rm -rf results/phase*/*/features
```

**6b. Job logs of finished runs** (after the H4 backup; the backup keeps results, not logs):
```bash
du -sh outputs logs
tar czf backups/logs_$(date +%Y%m%d).tgz outputs logs && rm -rf outputs/* logs/*
```

---

## On the Mac (`/Users/meheraj/Documents/MODAL-Net`)

The Mac copy is now a git repo that also holds **real** result files copied from the server
(`results/master/*.csv`, `results/phase*/phase_table*.csv`). Do **not** delete those. Only these are leftovers:
```bash
cd /Users/meheraj/Documents/MODAL-Net
rm -rf _transfer daily                                   # temporary transfer folder / old daily-report copies
rm -f splits/aptos_folds.json splits/messidor_dev_ids.json splits/messidor_test_ids.json   # synthetic (24/96 ids from tests)
rm -rf checkpoints/smoke results/phase0_protocol/smoke
rm -rf logs/test_fb_reweight logs/ens_test logs/p8_conf_test logs/p8_tta_flip_mock logs/default
rm -rf results/phase7_feature_balance .pytest_cache datasets/DDR-dataset/.cache
rm -f test_errors.txt E3_Status.md
find . -name __pycache__ -type d -prune -exec rm -rf {} +
find . -name .DS_Store -delete
git status --short                                       # review, then commit the deletions of tracked files
```
`results/tables/table*` and `results/figures/fig*` on the Mac that are older than 2026-11-03 were made by the old
code (test leftovers); the real ones are rebuilt on the server in H3. Delete them only if you are sure:
`ls -lt results/tables results/figures`.
The DeepDRiD `.git/` and `ultra-widefield_images/` folders on the Mac are not used by the code and are not
copied to the server (SERVER_TODO B3).
