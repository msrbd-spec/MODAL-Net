

## Answer: Not fully yet — E3 is **incomplete**, and where results exist the cut-point effect is **real but far weaker than the hypothesis wants**.

### 1. E3 is not done — do not run `phase_select` yet
Per `SERVER_TODO.md` lines 285–304, E3 = 6 jobs across two phases (`phase10_ms_ddr`, `phase10_ms_deepdrid`), then a sequential CPU `phase_select`.

| Job | State |
|---|---|
| `p10d_ce`, `p10d_coralsh`, `p10d_coralsrc` | ✅ DONE |
| `p10q_ce`, `p10q_coralsh` | ✅ DONE |
| **`p10q_coralsrc`** | 🔄 **running** — fold 3/5, epoch 15/30 (≈ mid-run) |

No `results/phase10*/phase_table.csv` and no `WINNER.txt` exist yet — correct, because the `st` gate in the file requires all six DONE before `phase_select`. So there is **no formal ADOPT/REJECT decision yet**.

### 2. Held-out numbers so far (each arm scored only on datasets it never saw)

`phase10_ms_ddr` — trained on APTOS+EyePACS+**DDR**, held out Messidor/DeepDRiD/IDRiD:

| arm | mean TEST QWK | mean AUROC | **selection DEV-mean QWK** (Δ vs `ce`) |
|---|---|---|---|
| ce | 0.800 | 0.957 | 0.8267 (ref) |
| coralsh (shared cut-points) | 0.803 | 0.960 | 0.8267 (**Δ 0.000**) |
| coralsrc (per-source offsets) | **0.810** | 0.960 | 0.830 (**Δ +0.003**) |

`phase10_ms_deepdrid` — trained on APTOS+EyePACS+**DeepDRiD**, held out Messidor/DDR/IDRiD:

| arm | mean TEST QWK | mean AUROC | **selection DEV-mean QWK** (Δ vs `ce`) |
|---|---|---|---|
| ce | 0.763 | 0.923 | 0.8233 (ref) |
| coralsh | 0.770 | 0.940 | 0.8367 (**Δ +0.013**) |
| coralsrc | — | — | pending |

### 3. Is that "as expected"?

**Qualitatively consistent, but the headline idea is not holding up cleanly:**

- **Good:** the ordinal cut-point heads never *hurt* cross-site transfer — AUROC ≥ CE and QWK ≥ CE on every fold. The head survives leaving sites out.
- **Weak:** every gain is ≤ +0.013 QWK on the selection (DEV-mean) endpoint — squarely in the noise-floor band that `RESULTS_STATUS.md` keeps flagging (the only phase-4j arm that ever cleared the gate moved +0.0065). No powered cross-site win yet.
- **Against the hypothesis:** E3's whole point is to show that **per-source cut-point offsets** (`coralsrc`, `src_offsets=true`) generalize best to unseen sites. So far that is *not* consistent: on the DDR-holdout fold `coralsrc` is marginally best (+0.003), but on the DeepDRiD-holdout fold it is the **shared** head `coralsh` (+0.013) that beats CE — the opposite ordering — and `coralsrc` there is still running. The offset-vs-shared ordering flips between the two folds, so the offset contribution is not yet a reproducible effect.
- This mirrors the in-domain result in E1/`phase4j_label_loss`: every label-loss arm was OOF-equivalent to reference and **REJECT**, and the only `ADOPT`/winner was `p4j_coralsrc_shared` (shared, no offsets) — already a hint that offsets per se aren't the driver.

### 4. One data caveat to be careful about
The stored `extdev_qwk` column reflects **only Messidor** (pre-registered external set). For `phase10_ms_deepdrid` it reads 0.76, but the true mean over the three held-out DEV sets is ~0.82–0.84. Selection must use the `phase_select` `d_dev_mean_qwk` over `dev_sets_used` (all held-out sets), not the `extdev_qwk` column — which is exactly what the `phase_select ... --reference p10d_ce/p10q_ce` lines in E3 do.

### 5. Also worth noting
Referable (mild-DR) recall drops for the ordinal heads on some held-out sets (e.g. ms_ddr mean mild recall 0.47→0.24/0.32), on small IDRiD/DDR samples — a possible clinical sensitivity cost, likely noisy but should be checked once `phase_select` runs.

### Bottom line
Results are **directionally fine (heads don't break generalization, ordinal AUROC edges CE)** but show **only near-noise QWK gains and an inconsistent offsets-vs-shared ordering**, so the cut-point hypothesis is **not yet confirmed**. Wait for `p10q_coralsrc` to finish, then run the four `phase_select` lines from E3 to get the official Δ/CI/ADOPT-REJECT before drawing any conclusion.

Want me to write a one-shot script that, once `p10q_coralsrc` finishes, runs the four `phase_select` commands and prints the two phase tables with the Δ and decision columns?
