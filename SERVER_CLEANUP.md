# SERVER CLEANUP — what can be deleted, where, and when

All paths are relative to the repository root on the server (`cd` there first). Each block starts with
a `du -sh` line that only **shows** the size; the `rm` line under it deletes. Run the `du` line first,
read it, then run the `rm` line. Nothing here is needed by any later step of `SERVER_TODO.md`.

**Never delete:** `checkpoints/<tag>/` of any finished experiment (fold weights + `config_snapshot.yaml` +
`thresholds.json`; `evaluate`, `--inherit`, ensembles and XAI need them), `splits/`, `softlabels/sl_p4/`,
`results/phase*/` (except the folders listed below), `checkpoints/pretrain/eyepacs_sup_v1.pth`,
`checkpoints/pretrain/eyepacs_sup.pth`, `checkpoints/pretrain/ssl_sup.pth`,
`checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch50.pth`, `.../ssl_backbone_epoch60.pth`,
`results/phase0_protocol/audit/` (duplicate audit + hash caches), `datasets/`.

---

## 1. Safe now

| What | Why it is not needed | Size (approx.) |
|---|---|---|
| `checkpoints/sl_p4/inner/` | 20 inner models of the nested soft-label run; the soft labels are saved in `softlabels/sl_p4/outer0..4.npz` | ≈ 18 GB |
| `checkpoints/pretrain/ssl_maxvit/ssl_training_checkpoint.pth` | SSL resume state; SSL is stopped (SERVER_TODO A1) | ≈ 3.5 GB |
| `checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch40.pth` | intermediate SSL backbone used by no arm | ≈ 0.9 GB |
| `checkpoints/smoke/`, `results/phase0_protocol/smoke/`, `logs/smoke/` | left by the old smoke test | small |
| `logs/test_fb_reweight/`, `logs/ens_test/`, `logs/p8_conf_test/`, `logs/p8_tta_flip_mock/`, `logs/default/` | logs of old test runs (mock tags) | small |
| `results/phase7_feature_balance/` | phase 7 was removed from the plan and the code | empty |
| `.pytest_cache/`, every `__pycache__/` | Python caches, rebuilt automatically | small |

```bash
du -sh checkpoints/sl_p4/inner
rm -rf checkpoints/sl_p4/inner

du -sh checkpoints/pretrain/ssl_maxvit/ssl_training_checkpoint.pth checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch40.pth
rm -f  checkpoints/pretrain/ssl_maxvit/ssl_training_checkpoint.pth checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch40.pth

du -sh checkpoints/smoke results/phase0_protocol/smoke logs/smoke logs/test_fb_reweight logs/ens_test logs/p8_conf_test logs/p8_tta_flip_mock logs/default 2>/dev/null
rm -rf checkpoints/smoke results/phase0_protocol/smoke logs/smoke logs/test_fb_reweight logs/ens_test logs/p8_conf_test logs/p8_tta_flip_mock logs/default

rm -rf results/phase7_feature_balance .pytest_cache
find . -name __pycache__ -type d -prune -exec rm -rf {} +
```

`checkpoints/sl_p4/` keeps its `config_snapshot.yaml` (it documents how the soft labels were made). If the
soft-label run also left fold models directly in `checkpoints/sl_p4/` (`outer*.pth`), they can go too:
```bash
ls -la checkpoints/sl_p4/
rm -f checkpoints/sl_p4/outer*.pth
```

## 2. Stale generated tables — ONLY once, right after the new code is deployed (SERVER_TODO B5)

These were built by the old code (before phase 3b and before the new datasets); `master_table` and `report`
rewrite them from the real summaries (SERVER_TODO H3). Deleting them before step C avoids quoting an old
number by mistake. Do not run this section after H3: it would delete the final tables, the leakage report
and the cut-point diagnostic.

```bash
du -sh results/master results/tables results/figures
rm -f results/master/*.csv results/master/*.tex
rm -f results/tables/table*.csv results/tables/table*.tex
rm -f results/figures/fig*.png results/figures/fig*.pdf
```

## 3. Feature caches of removed code

The removed feature-balance mode cached penultimate features under `checkpoints/<owner>/extracted_features/`
(only if phase 7 was ever started):
```bash
find checkpoints -maxdepth 2 -type d -name extracted_features -exec du -sh {} +
find checkpoints -maxdepth 2 -type d -name extracted_features -prune -exec rm -rf {} +
```

## 4. Old backups (keep the newest two)

```bash
ls -lt backups/
# keep the two newest tarballs, delete the rest:
ls -t backups/*.tgz | tail -n +3 | xargs -r rm -f
```

## 5. After the paper is submitted (optional, large)

Fold checkpoints of arms that are not in any table can be archived off the cluster. Before deleting any
`checkpoints/<tag>/`, check that no other tag uses it:
```bash
grep -l "ckpt_from: p2a_sam\|base: p2a_sam" checkpoints/*/config_snapshot.yaml      # example for p2a_sam
```

---

## On the Mac (`/Users/meheraj/Documents/MODAL-Net`)

These are test leftovers on the Mac copy only (the real ones are on the server):
```bash
cd /Users/meheraj/Documents/MODAL-Net
rm -rf _transfer                                        # temporary archive used to move the code
rm -f splits/aptos_folds.json splits/messidor_dev_ids.json splits/messidor_test_ids.json   # synthetic (24/96 ids)
rm -rf checkpoints/smoke results/phase0_protocol/smoke results/phase0_protocol/splits_summary.csv
rm -rf logs/test_fb_reweight logs/ens_test logs/p8_conf_test logs/p8_tta_flip_mock logs/default
rm -f results/tables/table* results/master/master_* results/figures/fig* results/all_runs.npz
rm -rf results/phase7_feature_balance .pytest_cache
find . -name __pycache__ -type d -prune -exec rm -rf {} +
find . -name .DS_Store -delete
```
The DeepDRiD `.git/` and `ultra-widefield_images/` folders on the Mac are not used by the code and are not
copied to the server (SERVER_TODO B3).
