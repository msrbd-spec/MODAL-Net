# DR Severity Grading — Full Result Ledger, Compute Accounting, and Constraint Analysis

**As of:** 2026-11-03 18:30 (cluster local time) — refreshed from live logs, `config_snapshot.yaml`
files and `softlabels/` contents. A parallel wave (§4.1) is in flight: 5 seed replicates running,
`sl_p4` finished 18:17 (soft labels **5/5**), so the AASL pair is unblocked but **not yet submitted**;
9 further wave arms still queued.
**Scope:** every training run launched in `phase0_protocol … phase4_label_loss`, the SSL
pretraining in detail, a paired-bootstrap re-analysis of the saved prediction arrays, an
accounting of GPU-hours spent vs. remaining, and an explicit statement of what currently blocks
the *protocol* and what currently blocks the *novelty claims*.

**Provenance** (nothing here is remembered; all re-extracted at write time):

| Source | What was read |
|---|---|
| `results/<phase>/<tag>/summary.csv` | per-arm metrics (2 dp, as written by `experiment.py`) |
| `results/<phase>/phase_table.csv`, `WINNER.txt` | official `phase_select` decisions + Δ columns |
| `results/<phase>/<tag>/{oof.npz, predictions_external_test.npz, predictions_external_dev.npz}` | per-image probabilities → own paired bootstrap (§6) |
| `checkpoints/<tag>/config_snapshot.yaml` | the *actual* hyper-parameters each arm trained with |
| `logs/<tag>/*.log` | first/last timestamps → wall spans, epoch rates, kill/resume history |
| `datasets/*`, `splits/*.json`, `results/phase0_protocol/*` | corpus sizes, class distributions |
| `src/data/splits.py`, `agent_context/MODAL_PART2.md`, `commands.md` | split logic, plan text, run status |

> **Three corrections to statements made earlier in this project's history** — recorded so they
> never reach the manuscript:
> 1. The OOF pool is **3,296** APTOS images (5 × ~659 validation folds), **not 1,744**. The number
>    1,744 is the *whole Messidor-2 corpus* and shows up in `results/phase1_zoo/zoo_table.csv` as
>    `ext_set=full1744`, which is a different protocol (§3.1).
> 2. The backbone is `maxvit_large_tf_512.in1k` (ImageNet-**1k**), not ImageNet-21k
>    (`checkpoints/p3b_joint/config_snapshot.yaml`, key `backbone`).
> 3. My first paired-bootstrap pass contained a **QWK normalisation bug** (missing `1/N` on the
>    expected term) yielding nonsense κ≈0.9998. §6 holds the fixed version, cross-checked against
>    `sklearn.metrics.cohen_kappa_score(weights='quadratic')`: 0.6982 vs 0.6982 for `p2b_none`/TEST.

---

## 1. Executive status in one screen

**19 fine-tuning arms fully trained** (5 folds each = **95 five-fold checkpoints** — 19 tag folders
holding exactly five weights apiece; `checkpoints/` as a whole holds **136 `.pth` files / 118.6 GB**
once the 11 pretraining weights, the 5 `sl_p4` outer models and the in-flight seed weights are counted),
**4 pretraining runs** (2 × supervised-EyePACS, 1 × SSL, 1 × SSL→supervised),
**6 jobs active right now** (5 seed replicates of the §4.1 wave + `pre_ssl`). `sl_p4` **completed
18:17:37** with all five soft-label files ⇒ `p4_aasl` / `p4_aasl_emd` are unblocked and **not yet
submitted**.

**GPU-hours spent, measured from log spans (§7.1): ≈190**, of which SSL is **62 %**. **Remaining if the
plan is executed as written: ≈170** — or **≈90** if SSL stops now (saves 52 h) and phase 6 is pruned to
4 cells (saves 27 h).

| Phase | Content | Status | Arms | Official outcome | What the data says (§6) |
|---|---|---|---|---|---|
| 0 | splits, dedup audit, lesion masks, zoo import | ✅ DONE | — | protocol frozen | — |
| 1 | 10-architecture zoo | ✅ DONE (imported, *not trained here*) | 10 | MaxViT-L kept | protocol not comparable to phases 2+ (§3.1) |
| 2a | anti-overfitting recipe, one factor at a time | ✅ DONE | 8 | winner `p2a_reg` (only `TENTATIVE`) | every ΔOOF ≤ 0.01, CI ∋ 0 → indistinguishable |
| 2b | preprocessing parity | ✅ DONE | 3 | winner `p2b_none` (all three `ADOPT`) | a three-way tie |
| 3a | pretraining (EyePACS-sup, SSL, SSL→sup) | ✅ DONE | 4 | **4× REJECT**, reference carried | sup **+0.043 EXT QWK (p<0.001)**; SSL **−0.030 EXT AUROC** |
| 3b | joint APTOS+EyePACS fine-tune | ✅ DONE | 2 | **2× REJECT**, reference carried | **+0.104 EXT QWK [+0.080,+0.130]**, mild recall 0.10→0.33 |
| 4 | EMD / CORAL / AASL soft labels | 🔄 2/5 trained, generator **done 5/5**, pair unblocked | 2/5 | `phase_select` pending | **`p4_coral` ΔOOF +0.0065 [+0.001,+0.012] — first arm to clear the gate** (§3.5b) |
| — | **parallel wave**: seed replicates, SSL dose, joint dose, arch fork | 🔄 5 running, 10 to submit | 5 | none — exploratory, own phase folders | retires C8 (image-only CIs); tests data-volume vs source-diversity (§4.1) |
| 5 | architecture + heterogeneous ensemble | ⏳ | 0/3 | — | — |
| 6 | domain generalisation (amplitude mixing, MixStyle, UDA, dev-calibration) | ⏳ | 0/~9 | — | — |
| 7 | feature-level class balance (optional) | ⏳ | 0/5 | — | — |
| 8 | TTA / temperature / conformal / selective prediction | ⏳ | 0/6 | — | — |
| 9 | winner + baseline, seeds 1,2 | ⏳ | 0/6 | — | seed variance now also enters via §4.1 (and C10 blocks `seed_summary` from seeing it) |
| 10 | XAI (Grad-CAM++/LayerCAM/EigenCAM + lesion overlay) | ⏳ | — | — | 88,702 masks on disk, unused by any metric |

**The single most consequential fact in this document:** the adoption rule (adopt ⇔ paired-OOF
ΔQWK CI lower bound > 0) rejected, in phases 3a and 3b, precisely the configurations that are
**highly significantly better out-of-distribution** (+0.10 QWK on Messidor-TEST, CI
[+0.080,+0.130], p<0.001), because in-domain OOF QWK is pinned at 0.926 ± 0.007 by APTOS label
noise. Phases 4→9 therefore *inherit* `p2b_none`: the pipeline is on track to ship the model that
is 0.10 QWK worse under domain shift. See §8-C1 and §8-C4.

---

## 2. Data, splits, and endpoints (as actually implemented)

### 2.1 Corpora on disk

| Corpus | On disk | Used | Notes |
|---|---|---|---|
| **APTOS-2019** `datasets/APTOS_19` | 3,662 images (`train_images`+`val_images`+`test_images`) | **3,296** CV pool + **366** held-out test | `train_1.csv` / `valid.csv` / `test.csv`; pool class counts ≈ 1605 / 340 / 912 / 176 / 267 |
| **EyePACS** `datasets/EyePACS/EyePACS` | **88,702** `.jpg` | `eyepacs_max = 20000` per run | **only 35,126 are labelled** (`trainLabels.csv/trainLabels.csv`; 25,810 / 2,443 / 5,292 / 873 / 708) → 53,576 label-free images exist but the supervised/joint arms never touch them |
| **EyePACS lesion masks** `datasets/EyePACS/lesion_labels` | **88,702** `.npz` | SSL multi-task heads only | `ssl_lesion_label_dir`, `ssl_mask_size=128`, loss weights type 0.5 / count 0.3 |
| **Messidor-2** `datasets/Messidor_2` | **1,744** images, 1,744 rows, all `adjudicated_gradable==1` | DEV **349**, TEST **1,395** (locked) | classes 1,017 / 270 / 347 / 75 / 35 |

### 2.2 Split construction (`src/data/splits.py`)

* APTOS stratified 5-fold, per-fold validation 659–660 images → **OOF = 3,296 predictions from 5
  models**. Fold-level class balance is preserved to ±1 image (see
  `results/phase0_protocol/splits_summary.csv`).
* APTOS held-out test = 366 images (199 / 30 / 87 / 17 / 33).
* Messidor: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, groups = *exam*
  (`id_code` first two `_`-tokens). Fold-0 test → **DEV (349)**, everything else → **TEST (1,395)**.
  Exam-id disjointness is asserted in code.
* **Both Messidor subsets come from one site with one non-mydriatic camera.** The split is
  *within-dataset*, exam-grouped. Legitimate held-out data, but it is **not** multi-site /
  multi-device external validation (§8-C3).
* The 5-class `diagnosis` column in the local `messidor_data.csv` is an **upstream derived label**:
  the CSV carries only `id_code, diagnosis, adjudicated_dme, adjudicated_gradable`, whereas the
  original Messidor-2 study ships referable yes/no + exudation probability + red-flags. Every
  5-class external QWK therefore inherits an undocumented mapping; the binary *referable* metric is
  the defensible one on this set (§9-N6).

### 2.3 Endpoints and the adoption rule as implemented

| Metric | Definition | Role in the protocol |
|---|---|---|
| `oof_qwk` | quadratic weighted κ on the 3,296 APTOS OOF predictions | **the only adoption gate**: Δ vs reference must have bootstrap CI lower bound > 0 (Holm across arms) |
| `extdev_qwk`, `extdev_mild_recall` | Messidor-DEV (n=349) QWK; recall of classes {1,2} | tie-break / `TENTATIVE`; phase-6 hyper-parameter choice |
| `test_qwk` | APTOS held-out (n=366) | reported |
| `ext_qwk` | Messidor-TEST (n=1,395) | "locked", reported once |
| `ece_test`, `ece_ext` | expected calibration error | reported |
| protocol | `ensemble5_tta` for every phase-2a→4 arm; zoo used `single_fold_mean` | zoo is *not* comparable (§3.1) |

Evaluated model = mean probability of the 5 fold models (best-epoch checkpoint by validation QWK),
`tta: none`. Shared recipe read back from `checkpoints/<tag>/config_snapshot.yaml`: 30 epochs,
2-epoch warmup, cosine, `batch_size 8` × `grad_accum 2`, `lr 1e-4`, `llrd 0.8`, `drop_path 0.3`,
`head_dropout 0.3`, `wd 0.05`, clip 1.0, bf16, early stopping patience 8, `img_size 512`,
`preproc none`, `seed 0`.

---

## 3. Complete result ledger — every trained arm, every phase

Values are as stored by `experiment.py` (2 dp). Blank = not produced by that protocol.
`DEV` = Messidor-DEV (349), `EXT` = Messidor-TEST (1,395), `mild` = recall of classes {1,2},
`noDR%` = share of external images predicted as class 0 (a pathology indicator, §3.6).

### 3.1 Phase 1 — architecture zoo (`results/phase1_zoo/zoo_table.csv`)

Imported via `--mode import_zoo --csv baseline_result.csv`: these were **not trained in this
project's jobs**, they use protocol `single_fold_mean` (one fold, mean of seeds) and their `ext`
column is the **whole 1,744-image Messidor-2** (including the 349 that later became DEV). They are
therefore *not* numerically comparable to phases 2a→4 — quote them as an ordering, never as a delta.

| Arch | train acc | cv acc | test acc | test QWK | Messidor acc | Messidor QWK | gap |
|---|---|---|---|---|---|---|---|
| **MAXVIT_L** (kept) | 99.24 | **86.18** | **85.19** | **0.90** | **65.21** | **0.59** | 13.06 |
| CONVNEXTV2_L | 95.52 | 84.47 | 82.90 | 0.88 | 63.25 | 0.55 | 11.05 |
| SWINV2_L | 98.55 | 83.79 | 82.57 | 0.87 | 62.98 | 0.54 | 14.76 |
| EFFICIENTNETV2_M | 98.72 | 82.49 | 82.51 | 0.89 | 63.89 | 0.57 | 16.23 |
| REGNETY_120 | 97.90 | 82.18 | 81.58 | 0.88 | 62.19 | 0.48 | 15.72 |
| VGG16_BN | 97.51 | 81.09 | 80.88 | 0.87 | 59.87 | 0.49 | 16.42 |
| RESNET50 | 96.84 | 80.65 | 79.78 | 0.85 | 60.21 | 0.41 | 16.19 |
| DENSENET121 | 93.97 | 80.54 | 78.53 | 0.86 | 59.94 | 0.34 | 13.43 |
| MOBILENETV3_L | 93.30 | 79.97 | 77.38 | 0.85 | 60.40 | 0.39 | 13.33 |
| SQUEEZENET1_1 | 80.72 | 77.41 | 75.08 | 0.80 | 58.74 | 0.18 | 3.31 |

Note the 13–16-point train–test gap of every high-capacity model: the pool is 3,296 images, the
backbone has ~640 M parameters, and it memorises. This is the empirical root of everything in §8-C1.

### 3.2 Phase 2a — training-recipe factors, one at a time (reference `p2a_plain`)

| Arm | factor changed | OOF QWK | OOF AUROC | test QWK | DEV QWK | DEV mild | EXT QWK | EXT AUROC | EXT mild | noDR% | ECE ext | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `p2a_plain` | *reference* (zoo recipe, `preproc=legacy`) | 0.92 | 0.98 | 0.92 | 0.57 | 0.00 | 0.61 | 0.90 | 0.00 | 85.1 | 0.19 | REFERENCE |
| `p2a_reg` | drop_path .3 + head drop .3 + llrd .8 | 0.92 | 0.98 | 0.92 | 0.50 | 0.00 | 0.54 | 0.91 | 0.00 | 86.4 | 0.23 | TENTATIVE → **winner** |
| `p2a_mix` | mixup .2 / cutmix 1.0 @50% | 0.92 | 0.98 | 0.92 | 0.58 | 0.00 | 0.60 | 0.93 | 0.00 | 85.0 | 0.14 | REJECT |
| `p2a_osmooth` | ordinal label smoothing ε=.1 | 0.92 | 0.98 | 0.92 | 0.56 | 0.00 | 0.60 | 0.91 | 0.00 | 85.3 | 0.21 | REJECT |
| `p2a_ema` | EMA 0.999 | 0.92 | 0.98 | 0.91 | 0.51 | 0.00 | 0.56 | 0.91 | 0.00 | 86.6 | 0.24 | REJECT |
| `p2a_aug` | `aug=medical` | 0.92 | 0.98 | 0.90 | **0.67** | 0.05 | **0.66** | 0.91 | 0.03 | 79.7 | 0.16 | REJECT |
| `p2a_sam` | SAM ρ=0.05 | 0.92 | 0.98 | 0.92 | 0.59 | 0.02 | 0.62 | **0.94** | 0.00 | 85.0 | 0.23 | TENTATIVE |
| `p2a_bundle` | adopted bundle of the above | 0.92 | 0.98 | 0.92 | 0.53 | 0.00 | 0.56 | 0.93 | 0.00 | 86.2 | 0.26 | TENTATIVE |

`WINNER.txt`: *“Selected as phase winner: decision=TENTATIVE, OOF QWK=0.9200, DEV QWK=0.5000.”*
Every arm sits at OOF QWK 0.92 with an AUROC of 0.98: the gate cannot separate them. Meanwhile the
external column swings from 0.54 to 0.67 QWK — a **13-point spread that the rule is blind to** — and
the arm officially promoted (`p2a_reg`) has the *worst* external numbers of the phase. Note also
that all eight arms have external mild-recall ≈ 0 (§3.6).

### 3.3 Phase 2b — preprocessing parity (reference = `@phase2a_recipe` = `p2a_reg`)

| Arm | preprocessing | OOF QWK | OOF AUROC | test QWK | DEV QWK | DEV AUROC | DEV mild | EXT QWK | EXT AUROC | EXT mild | noDR% | ECE ext | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `p2b_scaled` | resize-first + pad + CLAHE σ=size/50 | 0.92 | 0.98 | 0.92 | 0.64 | 0.94 | 0.12 | 0.67 | 0.93 | 0.03 | 78.7 | 0.19 | ADOPT |
| **`p2b_none`** | **no preprocessing** | **0.93** | 0.98 | **0.93** | **0.66** | 0.94 | **0.19** | **0.70** | **0.94** | **0.10** | **74.3** | 0.20 | ADOPT → **winner, current reference** |
| `p2b_clahe_only` | CLAHE only | 0.93 | 0.99 | 0.92 | 0.63 | 0.95 | 0.17 | 0.68 | 0.93 | 0.07 | 76.7 | 0.23 | ADOPT |

Exact reference values (`WINNER.txt` of phase 3a): **OOF QWK 0.9262, DEV QWK 0.6629, EXT QWK 0.6982
(this last from the re-analysis in §6, the CSV rounds it to 0.70)**. All three arms were labelled
`ADOPT` — the rule adopted a three-way statistical tie and broke it on the DEV column. This arm,
`p2b_none`, is what every later phase inherits and what all Δ in §6 are measured against.

### 3.4 Phase 3a — pretraining (reference `p2b_none`)

Pretraining runs themselves (from `logs/pre_*/` and `checkpoints/pretrain/`):

| Run | kind | epochs | lr | outcome | wall | artefact |
|---|---|---|---|---|---|---|
| `pre_eyepacs_sup` | EyePACS supervised | 15 planned, best at ep ~11 | 1e-4 | best **val QWK 0.7900** (EyePACS val, own scale) | 2.1 h | `checkpoints/pretrain/eyepacs_sup.pth` (kept as `eyepacs_sup_v1.pth`) |
| `pre_eyepacs_sup_v2` | EyePACS supervised | 4 | 3e-5 | best **val QWK 0.7802** | 0.6 h | overwrote `eyepacs_sup.pth` |
| `pre_ssl` | lesion-aware SSL | **67 / 100** (still running; ep 68 ≈19:31) | 1.5e-4, backbone ×0.1 | total loss 0.0136; **Contrastive ≡ 0.0000**, type 0.0024, count 0.0004, recon 0.0619 | **≈118 h so far** | `ssl_maxvit/ssl_backbone_epoch{40,50,60}.pth` + `ssl_pretrained_backbone.pth` (ep 60) + 3.5 GB resume ckpt |
| `pre_ssl_sup` | SSL → supervised head | 4 | 3e-5 | best **val QWK 0.7729** | 0.6 h | `checkpoints/pretrain/ssl_sup.pth` |

Downstream 5-fold fine-tunes of each backbone:

| Arm | initialisation | OOF QWK | OOF AUROC | test QWK | DEV QWK | DEV mild | EXT QWK | EXT AUROC | EXT mild | noDR% | ECE ext | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `p3a_sup` | `eyepacs_sup_v1.pth` — the **15-epoch** pretrain, val QWK 0.7900 | 0.93 | 0.98 | 0.92 | **0.70** | **0.33** | **0.74** | **0.95** | **0.28** | 68.2 | 0.15 | REJECT |
| `p3a_sup_v2` | `eyepacs_sup.pth` — the **4-epoch / lr 3e-5** pretrain, val QWK 0.7802 | 0.93 | 0.98 | 0.93 | 0.70 | 0.19 | 0.72 | 0.94 | 0.18 | 72.3 | 0.18 | REJECT |
| `p3a_sslsup` | `ssl_sup.pth` (SSL ep-50 → 4 supervised epochs) | 0.92 | 0.98 | 0.92 | 0.65 | 0.14 | 0.73 | 0.94 | 0.16 | 72.3 | **0.13** | REJECT |
| `p3a_ssl` | `ssl_maxvit/ssl_backbone_epoch50.pth` (**epoch 50, not 60/100**) | 0.92 | 0.98 | 0.92 | 0.64 | 0.12 | 0.70 | 0.93 | 0.07 | 75.5 | 0.18 | REJECT |

Corrected against `checkpoints/<tag>/config_snapshot.yaml`: the arm with the larger external gain
(`p3a_sup`, +0.043) is the one built on the **better and longer** pretrain (val QWK 0.7900, 15 epochs),
and `p3a_ssl` was scored from the **epoch-50** checkpoint, so the SSL row measures 50 epochs of
training — not the 67 it has now reached. Any seed re-run must reuse those exact paths
(`--inherit <tag>` already carries them).

`WINNER.txt`: *“No in-phase arm met the adoption rule; the reference is carried forward unchanged.”*
The plan's own kill-criterion (“if SSL does not beat EyePACS-supervised on DEV and OOF, report *no
added value*”) **fires**: supervised beats SSL on DEV 0.70 vs 0.64 and on EXT 0.74 vs 0.70.

### 3.5 Phase 3b — joint APTOS + EyePACS fine-tuning (reference `@phase3a_pretrain` = `p2b_none`)

| Arm | extra source | OOF QWK | test QWK | DEV QWK | DEV mild | EXT QWK | EXT AUROC | EXT mild | noDR% | ECE ext | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`p3b_joint`** | EyePACS, `joint_w=0.5`, 20k imgs | 0.93 | **0.94** | **0.77** | **0.33** | **0.80** | 0.92 | **0.33** | 68.1 | **0.07** | REJECT |
| `p3b_joint_w25` | EyePACS, `joint_w=0.25` | 0.93 | 0.93 | 0.74 | 0.29 | 0.79 | 0.90 | 0.30 | 66.9 | 0.10 | REJECT |

Same verdict sentence: reference carried forward. Both runs cost **4.7 GPU h each** (a joint epoch is
~2× a plain epoch because half of each batch is drawn from 20,000 extra images).

### 3.5b Phase 4 — label/loss arms completed so far (2 of 5, `phase_select` **not yet run**)

Finished on 2026-11-03: `p4_coral` at 14:45 (2 h 31 m) and `p4_emd` at 14:54 (2 h 43 m).

| Arm | OOF QWK | ΔOOF (paired bootstrap, 95 % CI) | DEV QWK | ΔDEV | EXT QWK | ΔEXT | EXT AUROC | mild ext | ECE ext |
|---|---|---|---|---|---|---|---|---|---|
| `p2b_none` (ref) | 0.9262 | — | 0.6629 | — | 0.6982 | — | 0.8556 | 0.104 | 0.20 |
| **`p4_coral`** | **0.9327** | **+0.0065 [+0.0012, +0.0121]**, P(d≤0)=0.006 | **0.6921** | +0.0293 [−0.011, +0.069] | **0.7158** | +0.0176 [−0.001, +0.037] | 0.8639 | 0.137 | 0.19 |
| `p4_emd` | 0.9261 | −0.0001 [−0.0059, +0.0058] | 0.6271 | −0.0358 [−0.089, +0.005] | 0.6994 | +0.0012 [−0.018, +0.023] | 0.8548 | 0.066 | 0.20 |

* **`p4_coral` is the first arm in the whole project whose pre-registered gate is satisfied**: the
  paired-OOF ΔQWK CI excludes 0 (+0.0065, P(d≤0)=0.006), and it is the only arm whose direction is
  positive on *all three* endpoints simultaneously (OOF +0.007, DEV +0.029, EXT +0.018, AUROC +0.008).
  It is still a ~2×-noise-floor effect, i.e. a small real gain, not a headline.
  Caveat: this is *my* bootstrap, not `phase_select`'s resampler — the official
  `--mode phase_select --phase phase4_label_loss --reference @phase3b_joint --design parallel` must be
  run over all five arms before `ADOPT`/`WINNER.txt` is quoted.
* `p4_emd` is a clean null in-domain (−0.0001) and directionally negative on DEV (−0.036, CI ∋ 0),
  with mild recall *dropping* to 0.066 — EMD appears to flatten the ordinal distribution rather than
  sharpen it.

**Footgun found while checking the soft-label gate — `src/training/trainer.py:198-207`.** If
`softlabels/<run>/outer<k>.npz` for the current fold does not exist, the `if os.path.isfile(sl_path)`
guard leaves `soft_dict = None` and training **silently continues with plain CE** — no exception, no
warning line. Launching `p4_aasl` or `p4_aasl_emd` before all five `outer0..4.npz` files exist therefore
produces a hybrid CE/AASL arm that is labelled as AASL in every artefact. Hard rule: wait for
`ls softlabels/sl_p4/outer*.npz | wc -l` to print **5**, and afterwards confirm five
`Loaded … AASL soft labels for fold k` lines in the run log.

**The adoption gate as actually implemented** (`src/evaluation/tables.py:279`, after the phase-specific
branches at 260–276): a parallel arm is `ADOPT` iff **`ci_lo > 0` AND `d_bin_auroc ≥ −0.005` AND
`d_oof_qwk ≥ −0.005`**; otherwise `REJECT` (`TENTATIVE` is permitted only for the a-priori
regularisation items of phase 2a). Against the reference `p2b_none`:

| Arm | `ci_lo` | clause 1 | `d_bin_auroc` | clause 2 | `d_oof_qwk` | clause 3 | Expected decision |
|---|---|---|---|---|---|---|---|
| `p4_coral` | **+0.0012** | ✅ | +0.0083 | ✅ | +0.0065 | ✅ | **ADOPT** |
| `p4_emd` | −0.0059 | ❌ | −0.0008 | ✅ | −0.0001 | ✅ | REJECT (fails only clause 1) |

Phase-specific rules differ and must not be conflated: **phase 6** adopts on `d_extdev_qwk > 0` with an
OOF drop tolerance of 0.005 (line 262), and **phase 2b** on a joint DEV rule requiring
`d_extdev_qwk_ci_lo > 0`, `d_extdev_qwk > 0.05`, `d_extdev_mild_ci_lo > 0` plus both guards (271–273).
Winner = argmax over `{REFERENCE} ∪ {ADOPT/TENTATIVE with winner_eligible}` with ties broken on
`extdev_qwk` (288–300); if every in-phase arm is REJECTed and the reference lives in another phase, the
reference is carried forward. `resolve()` raises `FileNotFoundError` when `@phaseX` has no `WINNER.txt`,
so a half-decided parent blocks later phases loudly rather than silently.

**Interpretation limit (see C8):** `ci_lo` above is a *paired image-resampling* CI computed against the
single seed-0 reference checkpoint. It contains **no seed variance**, so `P(Δ≤0)=0.006` means "given
seed 0", not "given the protocol".

### 3.6 Cross-cutting observations that no single table shows

1. **In-domain is saturated.** Across 17 arms and 4 phases, `oof_qwk` occupies exactly two values
   (0.92 / 0.93) and `oof_bin_auroc` exactly two (0.98 / 0.99). Total observed spread ≤ 0.01.
2. **Out-of-domain is unstable.** Over the same arms, DEV QWK spans 0.50 → 0.77 and EXT QWK
   0.54 → 0.80. The dataset changed by one camera and the metric moved 25 points.
3. **Every phase-2a/3a arm predicts “no DR” for 68–86 % of external images**, which is why their
   external mild-recall is ~0.00–0.19. The two interventions that mix in EyePACS *during*
   fine-tuning drop this to 66–68 % and lift mild recall to 0.28–0.33. The severity distribution —
   not the discrimination — is what transfers badly.
4. **Calibration tracks the same pattern**: `ece_ext` 0.13–0.26 for the ImageNet-only arms vs
   **0.07** for `p3b_joint`.
5. **Nothing in phases 2–3 moved `test_qwk` (APTOS held-out) by more than 0.02**, so the negative
   results are internally consistent, not fold luck.

---

## 4. What is running — refreshed 2026-11-03 18:12

| Job | Folder | State | Detail |
|---|---|---|---|
| `p3c_joint_s1` | `phase3c_seeds` | 🔄 fold 0 **ep 24/30** | started 17:09; a joint epoch costs ≈2× a plain one |
| `p3c_coral_s1` | `phase3c_seeds` | 🔄 fold 1 **ep 12/30** | CORAL head (`loss=coral head_type=coral` inherited) + `seed=1` |
| `p3c_none_s1` | `phase3c_seeds` | 🔄 fold 0 **ep 8/30** | **the reference under seed 1** — the run every other Δ depends on |
| `p3c_sup_s1` | `phase3c_seeds` | 🔄 fold 0 **ep 8/30** | seed 1 of `p3a_sup`; `--inherit` carries `pretrain_path=eyepacs_sup_v1.pth` |
| `p3c_none_s2` | `phase3c_seeds` | 🔄 fold 0 **ep 5/30** | reference, seed 2 — started ≈18:05 |
| `sl_p4` | phase-4 gate | ✅ **complete 18:17:37** (6.28 h) | `softlabels/sl_p4/outer0–4.npz` **all five written** ⇒ the AASL pair is unblocked |
| `pre_ssl` | 3a tail | 🔄 **epoch 67/100** (17:58) | loss 0.0136, `Contrastive: 0.0000`; 1.57 h/epoch ⇒ **≈52 h** for the remaining 33 epochs |

`p4_coral` finished 14:45 (2 h 31 m) and `p4_emd` 14:54 (2 h 43 m); their numbers are in §3.5b.
**`results/phase4_label_loss/` holds the two completed arms and no `WINNER.txt` — correct, select has
not been run and must not be until all five arms exist.**

### 4.1 The parallel wave — 15 arms (5 running, 10 to submit), what each one buys

Every arm forks from an **already-finished** config via `--inherit <tag>` and lives in a **new** phase
folder, so no closed phase's `phase_table.csv` / `WINNER.txt` is ever re-decided (constraint C7). None
of them touches `@phase4_label_loss`.

| Tag | Phase folder | Inherit | `--set` | GPU h | Question | Status |
|---|---|---|---|---|---|---|
| `p3c_none_s1` / `_s2` | `phase3c_seeds` | `p2b_none` | `seed=1/2` | 2.6 ea | seed variance of the **reference** | 🔄 / 🔄 |
| `p3c_coral_s1` / `_s2` | `phase3c_seeds` | `p4_coral` | `seed=1/2` | 2.6 ea | does CORAL's +0.0065 survive a seed change | 🔄 / ⏳ |
| `p3c_joint_s1` / `_s2` | `phase3c_seeds` | `p3b_joint` | `seed=1/2` | 4.7 ea | is the +0.10 EXT QWK gain joint training or seed 0 | 🔄 / ⏳ |
| `p3c_sup_s1` / `_s2` | `phase3c_seeds` | `p3a_sup` | `seed=1/2` | 2.6 ea | best pretrain arm × seed | 🔄 / ⏳ |
| `p3a_ssl60` | `phase3c_ssl` | `@phase2b_preprocess` | `pretrain_path=checkpoints/pretrain/ssl_maxvit/ssl_backbone_epoch60.pth` | 2.8 | 2nd SSL dose point (50→60) **without** waiting 52 h for epoch 100 | ⏳ |
| `p3c_dose5k` / `p3c_dose2k` | `phase3c_dose` | `p3b_joint` | `eyepacs_max=5000` / `=2000` | 9.4 | **data volume or source diversity?** gain at 2k ⇒ diversity; scales with N ⇒ volume | ⏳ |
| `p3c_cnext_ref` / `p3c_effnet_ref` | `phase3c_arch` | `p2b_none` | `backbone=convnextv2_large.fcmae_ft_in22k_in1k_384` / `efficientnetv2_rw_m.agc_in1k`, `pretrain_path=null llrd=1.0 mixstyle=false winner_eligible=false` | ≈7 | zoo comparability: are phase-1 zoo numbers comparable with fine-tuned arms | ⏳ |
| `p4_aasl` / `p4_aasl_emd` | `phase4_label_loss` | `@phase3b_joint` | `soft_labels=aasl soft_label_run=sl_p4` (+ `loss=ce_emd lambda_emd=1.0`) | 5.2 | completes phase 4 ⇒ only then may select run | **READY — 5/5 soft labels at 18:17, not submitted** |

**Reading of the wave.** The seed block is the priority: every Δ quoted anywhere in this ledger is a
paired *image* bootstrap against one seed-0 reference, so until reference + contender each have ≥3
seeds, no effect below ≈0.01 QWK is protocol-level (C8). The `eyepacs_max` pair replaces the withdrawn
equal-quantity control (C9) and is the cheapest test of *why* joint training helps, which is the
mechanism the novelty claim rests on. The two `_ref` arms are a lineage **fork**, deliberately not the
pre-registered `p5_*` arms.

**Caveats recorded at launch time.**
* `p3a_ssl60`, `p3c_dose*` and `p3c_*_ref` are **exploratory**, not pre-registered; the pre-registered
  `p5_convnextv2` / `p5_effnetv2` / `p3a_ssl100` still have to be run against the phase-4 winner.
* Both architecture ids are present in `~/.cache/huggingface/hub` and are the **same timm ids the
  phase-1 zoo used**, so `HF_HUB_OFFLINE=1` is safe and the comparability intent holds. The residual
  risk is compute only: `img_size` stays 512 for a `_384` model (≈1.8× FLOPs) — on OOM, resubmit that
  single arm with `img_size=384`.
* Post-launch check that actually catches silent failures: dump the override keys from each new
  `checkpoints/<tag>/config_snapshot.yaml` and confirm `seed`, `eyepacs_max`, `backbone`,
  `pretrain_path` differ from the parent (C9 — `--set` never errors on an unknown key).
* After each AASL run, `grep -c "AASL soft labels for fold" <log>` must print **5**.
* Protocol snapshot taken 17:25: `backups/proto_20261103_1723_pre_phase4select.tgz` (3.28 GB, of which
  97 % is the seven zoo `.pth` files inside `results/baselines/`; recurring snapshots should add
  `--exclude='results/baselines/*/*.pth'` ⇒ ≈100 MB). Disk 183 GB free of 1.1 TB (83 % used).

<details><summary>Earlier snapshot (12:48) — kept for the record</summary>

At 12:48 all three phase-4 jobs were still mid-run: `p4_emd` fold 1 ep 5/30, `p4_coral` fold 0 ep 20/30,
`sl_p4` inner fit “Fold 2” ep 6/15, with observed rates 1.6–1.7 min/epoch (fine-tunes) and 1.3
min/epoch (inner fits) — which is what predicted the 14:45/14:54 completions and the ≈18:30 soft-label ETA.

</details>

---

## 5. SSL deep-dive (the most expensive component, and the clearest failure)

### 5.1 What was actually configured

```
backbone            maxvit_large_tf_512.in1k   ssl_img_size      512      ssl_epochs     100
ssl_batch_size      8                          ssl_grad_accum    1        ssl_temperature 0.07
ssl_lr              1.5e-4 (backbone ×0.1)     ssl_warmup        5        ssl_weight_decay 0.05
ssl_use_contrastive true  w=1.0     ssl_use_reconstruction true w=0.2   ssl_mask_size 128
ssl_use_multitask   true  type w=0.5, count w=0.3   ssl_use_ema true (0.996)
ssl_image_dirs / labels  datasets/EyePACS/EyePACS  (88,702 jpg; 35,126 labelled)
ssl_lesion_label_dir     datasets/EyePACS/lesion_labels (88,702 .npz)
```

### 5.2 What actually happened

* **The contrastive term is identically zero.** With `ssl_batch_size = 8` each anchor gets 7 in-batch
  + 7 momentum-queue negatives = **15 negatives**; at `τ = 0.07` the InfoNCE optimum is reached
  almost immediately by collapsing the embedding, and the logged value has been
  `Contrastive: 0.0000` since roughly epoch 20 and for all 43 subsequent epochs. The remaining
  gradient signal is reconstruction (0.062) and the two lesion multi-task heads (0.0028 / 0.0004).
  So the artefact named “SSL backbone” is effectively *a masked-reconstruction + lesion-count/type
  multitask model with a dead contrastive loss* — the “self-supervised contrastive” novelty claim
  cannot be made from this checkpoint.
* **Two-part timeline.** Run 1: 10-30 13:44 → 11-02 12:28, reached epoch 58/100 then hit the 72 h
  `shr_gpu` wall limit. Because SSL only writes a resumable checkpoint every **10 epochs**, the kill
  cost ~8 epochs. Run 2 resumed from epoch 50 at 11-02 15:41 and is at **epoch 67/100** (17:58 on
  11-03), i.e. 1.57 h/epoch — so the 33 epochs still outstanding cost **≈52 GPU h**.
  **Cumulative SSL cost ≈ 94 GPU h; running to 100 costs ≈ 57 more.**
* **Downstream effect of 94 GPU h** (vs the ImageNet-initialised reference `p2b_none`, paired
  bootstrap, §6): ΔOOF QWK **−0.006** [−0.013, +0.001]; ΔEXT QWK **+0.000** [−0.019, +0.020];
  ΔEXT binary AUROC **−0.030** [−0.043, −0.019] — i.e. pure SSL is **significantly worse** at
  referable-DR discrimination on external data, and its mild recall (0.47 vs 0.50) also drops.
* `pre_ssl_sup` (4 supervised epochs on top of the epoch-50 SSL backbone) *does* recover:
  ΔEXT QWK **+0.037** [+0.015, +0.061], but it remains below plain supervised EyePACS pretraining
  (+0.043) which cost 2.1 GPU h instead of 94.

### 5.3 The options, priced

| Option | Cost | Consequence |
|---|---|---|
| **Stop SSL now (epoch 67)** and use `ssl_backbone_epoch60.pth` for the dose arm | 0 h extra | **partly taken**: `p3a_ssl60` (§4.1) is the second point on the curve (ep 50 → ep 60) at 2.8 GPU h, so “does more SSL help?” is testable today rather than in 52 h |
| Run SSL to 100 as planned | **≈52 GPU h** (33 epochs × 1.57 h) | tests the *under-training* hypothesis; expected payoff low — the loss is **saturated, not plateauing**, and `Contrastive` has been 0.0000 for 47 consecutive epochs. Since GPUs are not the binding constraint, running it out *and* running `p3a_ssl60` is the zero-regret choice; add `p3a_ssl100` for the pre-registered arm |
| **Re-run SSL with `ssl_batch_size ≥ 256`** (grad-accum 32, or 4-way DDP) | ≈25–35 GPU h for 50 epochs | the only way the contrastive component becomes a real component; everything else stays as configured |
| Keep as-is and report as a **pre-registered negative control** with the saturation analysis | 0 h | honest, cheap, and the analysis itself (15-negatives/anchor ⇒ κ(τ) degeneracy) is publishable content |

---

## 6. The one real effect: re-analysis of the saved per-image predictions

`phase_select` compares arms on OOF QWK. To check whether that endpoint has any power at all, the
six phase-3 arms were re-compared against the current reference `p2b_none` using the **saved
per-image probability arrays** (identical images and label vectors, verified element-wise), with a
**paired image bootstrap**, 4,000 resamples, quadratic weighted κ, cross-checked against
`sklearn`. `P(d≤0)=0.000` means no resample favoured the reference.
`mild` = the pipeline's own definition: fraction of **true class 1** predicted as class 1
(`src/evaluation/metrics.py:51`).

### 6.1 Messidor-TEST (n = 1,395, locked) — reference QWK **0.6982**, binary AUROC **0.8556**, mild **0.104**

| Arm | QWK | ΔQWK (95 % CI) | binary AUROC | ΔAUROC (95 % CI) | mild |
|---|---|---|---|---|---|
| `p3b_joint` | **0.8020** | **+0.1039 [+0.0802, +0.1304]** | 0.8756 | +0.0199 [+0.0061, +0.0331] | 0.330 |
| `p3b_joint_w25` | 0.7873 | +0.0891 [+0.0636, +0.1170] | 0.8609 | +0.0053 [−0.0075, +0.0181] | 0.300 |
| `p3a_sup` | 0.7413 | +0.0432 [+0.0236, +0.0656] | **0.8873** | **+0.0317 [+0.0191, +0.0447]** | 0.283 |
| `p3a_sslsup` | 0.7348 | +0.0366 [+0.0146, +0.0614] | 0.8664 | +0.0107 [+0.0004, +0.0213] | — |
| `p3a_sup_v2` | 0.7214 | +0.0232 [+0.0007, +0.0471] | — | — | — |
| `p3a_ssl` | 0.6986 | +0.0004 [−0.0185, +0.0204] | **0.8255** | **−0.0302 [−0.0426, −0.0186]** | — |

### 6.2 Messidor-DEV (n = 349) — reference QWK **0.6629**, AUROC **0.8502**

| Arm | QWK | ΔQWK (95 % CI) | AUROC | ΔAUROC (95 % CI) |
|---|---|---|---|---|
| `p3b_joint` | **0.7656** | **+0.1028 [+0.0606, +0.1486]** | 0.8719 | +0.0217 [−0.0047, +0.0498] |
| `p3b_joint_w25` | 0.7414 | +0.0786 [+0.0345, +0.1274] | 0.8585 | +0.0083 [−0.0173, +0.0360] |
| `p3a_sup` | 0.7028 | +0.0399 [+0.0096, +0.0728] | **0.8941** | **+0.0438 [+0.0228, +0.0674]** |
| `p3a_sup_v2` | 0.7010 | +0.0382 [−0.0050, +0.0828] | — | — |
| `p3a_sslsup` | 0.6518 | −0.0110 [−0.0683, +0.0399] | 0.8660 | +0.0158 [−0.0040, +0.0373] |
| `p3a_ssl` | 0.6424 | −0.0204 [−0.0660, +0.0229] | 0.8200 | −0.0302 [−0.0546, −0.0072] |

### 6.3 APTOS OOF (n = 3,296) — reference QWK **0.9262** → *the selection endpoint*

| Arm | QWK | ΔQWK (95 % CI) | verdict the rule gives |
|---|---|---|---|
| `p3b_joint_w25` | 0.9296 | +0.0035 [−0.0026, +0.0097] | REJECT |
| `p3b_joint` | 0.9286 | +0.0025 [−0.0047, +0.0096] | REJECT |
| `p3a_sup_v2` | 0.9276 | +0.0014 [−0.0051, +0.0080] | REJECT |
| `p3a_sup` | 0.9258 | −0.0004 [−0.0066, +0.0055] | REJECT |
| `p3a_ssl` | 0.9198 | −0.0063 [−0.0131, +0.0007] | REJECT |
| `p3a_sslsup` | 0.9196 | −0.0066 [−0.0137, +0.0005] | REJECT |

### 6.4 What this means, and what it does not mean

* **Effect size vs measurement resolution.** The selection endpoint resolves ≈ ±0.007 QWK; the
  largest true in-domain effect (0.0035) is half that. On external data the same arms differ by 0.10
  with CI widths of 0.05. The endpoint is not merely sub-optimal, it is **~20× under-resolved** for
  the interventions under test.
* **Three arms reach p < 0.001 on external QWK, two on external binary AUROC.** After Holm
  correction over 6 arms × 3 endpoints, `p3b_joint` and `p3a_sup` survive comfortably (raw
  p ≤ 0.001 vs a Holm threshold of 0.05/18 ≈ 0.0028).
* **Mild-DR recall is the clinically vivid number**: the reference flags only **10.4 %** of true
  mild cases as mild; `p3b_joint` flags **33.0 %**, `p3a_sup` 28.3 %, pure SSL adds 0. (Under the
  looser definition “predicted ≥ 1 among true classes {1,2}” the same comparison is 0.500 → 0.663.)
* **Not yet a publishable claim.** Each arm is **one** training run: the CI is over *images*, not
  *models*. Between-seed SD for 30-epoch fine-tunes is typically 0.01–0.03 QWK, the same order as the
  smallest effects here. `ext_qwk` (the locked set) was inspected post-hoc for this analysis, and
  DEV/TEST are splits of **one** site. §8-C2, §8-C3 and §10-B address all three.

---

## 7. Compute accounting — spent and remaining

### 7.1 Spent (wall span per tag, from `logs/<tag>/*.log` first/last timestamps)

| Phase / tag group | Runs | Wall span per run | Honest compute estimate |
|---|---|---|---|
| phase 1 zoo | 10 | — | **0 h** — imported from `baseline_result.csv`, never trained here |
| phase 2a | 8 | log span 47–51 h (queue + preemption Oct 27→29 inflate this; pure run ≈ 2.5–3 h, SAM ≈ 2×) | **≈ 26 GPU h** |
| phase 2b | 3 | 2.3 / 2.8 / 2.8 h | **7.9 GPU h** |
| `pre_eyepacs_sup`, `_v2` | 2 | 2.1 h, 0.6 h | **2.7 GPU h** |
| `pre_ssl` (run 1 epochs 1–58 over 93.5 h incl. preemption; run 2 resumed at ep 50 → ep 67 in 26.7 h) | 1 | 1.57 h/epoch measured on run 2 | **≈ 118 GPU h** (75 epoch-equivalents; **≈13 h wasted** replaying epochs 51–58 — the 3.5 GB resume ckpt is written only every 10 epochs, and run 1 died at ep 58 with ep 50 on disk) |
| `pre_ssl_sup` | 1 | 0.6 h | **0.6 GPU h** |
| phase 3a | 4 | 2.0–2.8 h | **10.2 GPU h** |
| phase 3b | 2 | 4.7 h each | **9.4 GPU h** |
| phase 4 | 3 | `p4_emd` 2.71 h, `p4_coral` 2.51 h, `sl_p4` 6.28 h (done 18:17:37) | **11.5 GPU h** |
| §4.1 wave | 5 running | `p3c_joint_s1` 1.30, `p3c_coral_s1` 1.30, `p3c_none_s1` 0.39, `p3c_sup_s1` 0.39, `p3c_none_s2` 0.16 | **≈ 3.5 h booked**, ≈38 h booked in total |
| phase 0 `detect_lesions` (88,702 masks) | 1 | not logged | unknown, order 10 h |
| | | **Total ≈ 190 GPU h** (≈200 incl. mask detection) | of which SSL = **62 %** |

`summary.csv` has a `gpu_hours` column but it is not trustworthy: it reads 0.06–0.08 for every
phase-2a arm because those summaries were regenerated by `tools/rebuild_oof.py`, which times only the
inference pass. Use the log spans above.

### 7.2 Remaining, priced at the measured rates (plain 5-fold arm ≈ 2.8 h, joint arm ≈ 4.7 h)

| Phase | Arms still to run | GPU h |
|---|---|---|
| 4 | `p4_aasl`, `p4_aasl_emd` — **soft labels finished 5/5 at 18:17, nothing blocks these any more** | **5.4** |
| §4.1 | 3 seeds still to submit (`p3c_coral_s2` 2.6, `p3c_sup_s2` 2.6, `p3c_joint_s2` 4.7), `p3a_ssl60` 2.8, `p3c_dose5k` + `p3c_dose2k` 9.4, `p3c_cnext_ref` + `p3c_effnet_ref` ≈7 | **≈ 29** |
| — | SSL tail, if pushed to epoch 100 (33 epochs × 1.57 h) | **52** |
| 5 | `p5_wghr`, `p5_convnextv2`, `p5_effnetv2` + `p5_hetero_ens` | ≈ 9 |
| 6 | `p6_ref` + amplitude-mixing grid (2 stages × 3 β) + MixStyle + amp&MS + UDA + dev-calibration | ≈ 45 (**prunable to ≈ 18**) |
| 7 | 5 feature-balance arms (optional) | ≈ 8 |
| 8 | 6 inference-only tags (TTA ×4, temperature, conformal) | ≈ 2 |
| 9 | winner s1,s2 + baseline s1,s2 (+2 inference tags) + `seed_summary` — partly superseded by the §4.1 seeds, and still blocked by C10 | ≈ 15 |
| 10 | XAI on the winner | ≈ 1 |
| | **≈ 170 GPU h remaining** as written; **≈ 90** if SSL stops now (−52) and phase 6 is pruned to 4 cells (−27) | |

At 4 concurrent single-GPU slots the remaining plan is ≈ 43 h of pure wall clock (≈ 1.8 days with
queue); the §4.1 wave's ≈ 38 h fits inside a single day once ≥6 slots are free, which is exactly why it
was submitted as a wave rather than sequenced. The 72 h per-job ceiling is not binding for a single arm
(max 4.7 h) but it **did** kill `pre_ssl` mid-run once (§5.2). Nothing else in the remaining plan
approaches it.

---

## 8. Constraints on the plan (protocol-level)

Ordered by how much each one costs the project if left unfixed.

**C1 — The adoption gate is blind to the only effects that exist.** *Severity: blocking.*
Adoption requires paired-OOF ΔQWK `ci_lo > 0` on 3,296 APTOS images whose QWK is pinned at ≈ 0.926
(±0.007) by single-grader label noise. Every phase-3 arm that improves external QWK by +0.04…+0.10
(p < 0.001) is recorded as `REJECT`. The ablation tables that would form the results section are
therefore **structurally guaranteed to be null**, and the phase winner is chosen by noise.
*Fix:* declare the primary endpoint prospectively as external QWK + mild recall with the paired
bootstrap, keep APTOS OOF as a non-inferiority constraint (`ΔOOF ≥ −0.005`), and present in-domain
nulls as **equivalence** results (TOST, margin 0.01) with a power statement. Because this changes a
pre-specified rule it must be written down *before* phase-5 results are read, and the phase-3b
decision re-issued as “endpoint amended, original endpoint reported alongside”.

**C2 — No model-level variance.** *(being retired now — quantified in C8, seed block of §4.1 in flight)* *Severity: blocking for any positive claim.*
All 17 arms are single runs at `seed 0`; every CI quoted anywhere is over images, not over training.
Seeds appear only in phase 9, and only for the winner and the plain baseline — the two configurations
whose difference is already known to be ≈ 0. *Fix:* ≥3 (ideally 5) seeds for the 3–4 headline
configurations (reference, `p3a_sup`, `p3b_joint`, best phase-4 arm), with **seed** as the unit of
analysis. ≈ 45–70 GPU h.

**C3 — One “external” dataset that is really one site.** *Severity: high.*
DEV and TEST are exam-grouped random splits of a single 1,744-image, single-camera study with an
undocumented 5-class label transform (§2.2). One such set is the first thing a TMI/MedIA reviewer
attacks, and tuning phase-6 hyper-parameters **on DEV** makes DEV non-independent for DG claims.
*Fix:* add ≥1 (ideally 2) genuinely independent sources (DRDB, IDRx, APTOS-2019 hidden test, RISS,
SIIV-IG, REFUGE-DR) and move hyper-parameter selection to a nested APTOS split. *Blocker:*
`HF_HUB_OFFLINE=1` and no outbound network inside jobs → data must be fetched outside the scheduler
and copied in. This is the long-lead item; it should start now.

**C4 — The inheritance chain now points at the weakest model.** *Severity: high, and compounding.*
`@phase3a_pretrain` and `@phase3b_joint` both resolve through `WINNER.txt` to `p2b_none`, so all five
phase-4 arms are training **on top of the configuration that is 0.10 QWK worse out-of-distribution**,
and phases 5→9 inherit the same. If the endpoint is amended later, phase 4 and everything after it will
have been measured on a chain that no longer represents the intended model, and any phase-6 “DG gain”
will be confounded with the discarded joint-training gain. *Fix:* settle the endpoint question
**before** phase 5 launches; if it changes, re-point the chain (`--inherit p3b_joint`) and re-run the
three already-trained phase-4 arms (≈ 8 GPU h) instead of reporting a mixed lineage.

**C5 — Phase-1 baselines are not comparable.** *Severity: medium (presentation).*
`single_fold_mean` protocol, no OOF, external column = all 1,744 images including DEV, and imported
rather than trained (§3.1). Mixing zoo rows with phase-2+ rows in one table invites a desk reject.
*Fix:* keep the zoo as a separate screening table with a footnote; the real architectural comparison is
phase 5, which retrains ConvNeXtV2-L and EfficientNetV2-M under the correct protocol.

**C6 — Selection on DEV leaks into the DG phase.** *Severity: medium.*
Phase 6 chooses amplitude-mixing stage and β “on Messidor-DEV (primary: DEV QWK, mild recall)”. That is
acceptable if documented, but it disqualifies DEV as *evidence* for those same arms.
`winner_eligible=false` is already set for `p6_uda` and `p6_devcal`; extend it to the whole amp/MixStyle
grid and state the rule explicitly: DG claims are made only on the untouched third site (C3) or on TEST
in one pre-declared look. Note the related config trap: amplitude mixing needs `preproc != legacy`
(plan §S9), and the current winner is `preproc=none`, so the extra `p6_ref` arm is **not** required —
but verify that the amplitude-mixing transform is well defined on raw, un-preprocessed 512×512 inputs,
because it was designed and tuned against `preproc=scaled`.

**C7 — Artefact and reproducibility fragility.** *Severity: medium (operational).*
A `results/<phase>/` folder has been lost once (hence `tools/rebuild_oof.py`);
`--mode pretrain --kind eyepacs_sup` **overwrites** `checkpoints/pretrain/eyepacs_sup.pth` with no
versioning (a manual `cp` is what saved v1); SSL resumes only every 10 epochs, so a wall-limit kill
costs up to ~12 h; `--set` is `nargs="*"` so a token missing its `=` silently kills a run (`p4_coral`
died once this way); thresholds and the temperature are fitted on the OOF pool, so re-running
`phase_select` after adding an arm can flip another arm's decision. *Fix:* tarball `results/` +
`config_snapshot.yaml` after every phase; snapshot pretrain outputs before re-running; never hand-write
`WINNER.txt`.

**C8 — Adoption CIs contain no seed variance.** *Severity: high (interpretation).*
Every `Δ`/`CI` column in `phase_table.csv` is a paired bootstrap **over images** of one checkpoint per
fold, computed against the **single seed-0 reference checkpoint**. It therefore measures image-sampling
noise only, and seed noise in the reference propagates into *every* arm's Δ in the same direction.
`p4_coral`'s `P(Δ≤0)=0.006` means "given seed 0", not "given the protocol"; its CI lower bound
(+0.0012) is a fifth of the point estimate. Until reference and contender each have ≥3 seeds, treat any
|Δ| < 0.01 QWK as provisional. The seed block of §4.1 exists precisely to retire this constraint.

**C9 — `--set` accepts unknown keys silently.** *Severity: high (operational).*
`parse_set` (`src/utils/config.py:9-21`) is `yaml.safe_load` on the value — correct casting for
`true/false/null/numbers`, but **no key whitelist**: `validate_cfg` asserts only ~6 fields
(`amp_mix`×`preproc`, `mixstyle`×backbone, `head_type=coral`×`soft_labels`, `uda_style_pool`, `tta`,
`recalibrate`, `pretrained_fallback`, `xai_*`). A control written with a non-existent key therefore
trains an exact copy of the parent and is labelled as the control. Recorded near-miss: an
"equal-quantity augmentation" control drafted as `--set aptos_aug_multiplier=6` — **no such key
exists** (real knobs: `aug`, `alpha_max`, `amp_beta_max`, `mixstyle`, `eyepacs_max`, `joint_w`,
`preproc`, `uda_style_pool`); withdrawn before submission. Corollary: the pre-registered equal-quantity
in-domain control is **not expressible** with existing keys — the labelled APTOS pool is 3,296 images,
so "20k extra in-domain images" needs a new `aptos_repeat` knob (duplication + fresh augmentation).
`eyepacs_max` dose-response (§4.1) is the substitute. **Rule: after every submission, dump the override
keys from `checkpoints/<tag>/config_snapshot.yaml` and confirm they differ from the parent.**

**C10 — `seed_summary` is hard-coded to phase-9 tag names.** *Severity: medium (tooling).*
`tables.py:530` groups only tags starting `p9_final_s` / `p9_base_s` (default phase `phase9_final`) and
returns silently otherwise, so the seed replicates kept in `phase3c_seeds` are **invisible** to
`--mode seed_summary` and `results/master/seed_variance.csv` will not reflect them. Either patch those
two prefixes to group by stripping `_s\d+$`, or aggregate `results/phase3c_seeds/*/summary.csv` directly.

**C11 — CORAL and AASL are mutually exclusive by the validator.** *Severity: medium (design).*
`validate_cfg` asserts `not (head_type == 'coral' and soft_labels != 'none')`. Two consequences: phase 4
compares arms that use **different heads** (CORAL vs softmax), so its winner-selection is a head
*branch* choice as much as a loss choice; and if CORAL is adopted, the obvious follow-up — AASL soft
labels on top of the adopted config — is **illegal as written**. Either justify the assertion (CORAL
learns cumulative-threshold targets, which do not compose with a soft target distribution) and state
that the two novelties are alternative branches, or relax it and add a CORAL+AASL arm.

**C12 — the zoo baselines have no recorded configuration.** *Severity: medium (provenance).*
`results/phase1_zoo/<NAME>/` contains **only** `summary.csv`; there is no `config_snapshot.yaml` for any
zoo or `results/baselines/` arm, so the `img_size` / recipe behind the zoo numbers is unrecoverable from
artefacts (the weights themselves survive only in the 17:25 backup tarball). Any
"zoo baseline vs fine-tuned arm" comparability claim must therefore pin `img_size` explicitly in the new
arms (§4.1 `p3c_*_ref`) and state the assumption it rests on.

---

## 9. Constraints on the novelties (scientific-claim level)

**N1 — EMD loss / ordinal smoothing: below the noise floor. CORAL: the one exception found so far.**
*Status 18:30:* both trained arms are in (§3.5b) and `phase_select` has not been run, so the *official*
phase-4 answer is still open — but the data already splits this claim in two. **EMD is a null** (ΔOOF
−0.0001, ΔDEV −0.036, mild recall 0.104 → 0.066: it flattens the ordinal distribution rather than
sharpening it). **CORAL is the first arm in the project with `ci_lo > 0`**: ΔOOF **+0.0065
[+0.0012, +0.0121]**, P(Δ≤0)=0.006, and the only arm of 19 positive on OOF, DEV, EXT QWK and binary
AUROC simultaneously. What keeps it short of a headline is C8 (image-only CI against a seed-0 reference)
and the small external n — **not** the in-domain result. If select records ADOPT, restate N1 as
*ordinal-structure objectives do not move a near-ceiling in-domain endpoint, except the one that changes
the head*; per C11 adopting CORAL also excludes AASL, making phase 4 a head-**branch** choice.
The plan itself says “expect ~0–1 point; null results are reported”. One QWK point is ~140× the 0.007
resolution of the gate, and the observed spread of *all* phase-2a loss factors was ≤ 0.01. These arms
can only ever be reported as null-or-equivalence unless the endpoint changes (C1). Two mechanical
restrictions compound it: `config.py` forbids `head_type=coral` together with `soft_labels != none`, so
the CORAL × AASL cell **cannot be crossed** — the two ordinal novelties are mutually exclusive by
construction, which undercuts any claim that ordinal structure per se is what matters; and the CORAL arm
shows its classic over-fitting signature (fold 0: train acc 98.8 % vs val acc 84.9 %, best val
QWK 0.938 around epoch 18 then flat) — the ordinal head adds capacity, not transfer, which is exactly why
its +0.0065 needs the §4.1 seed replicates before it is called an effect rather than a lucky draw.

**N2 — AASL soft labels: the procedure is the contribution, the gain probably is not.**
The nested cross-fitted generation (5 outer × 4 inner = 20 leakage-free 15-epoch fits, **measured
6.28 GPU h**, complete 18:17:37) is
better practice than self-generated soft labels and is the most defensible methods contribution in the
project. But its expected QWK effect is ≤ 0.005, and its *intended* benefit — calibrated ambiguity on
ordinal boundaries — is invisible in the current metric set: class 1 has only 340 images in the OOF
pool, so a 10 % ambiguity improvement shifts overall QWK by < 0.002. Showing the mechanism needs
ambiguity-specific read-outs (confidence-vs-ordinal-distance, boundary ECE, selective-prediction AUC),
none of which are implemented.

**N3 — The lesion-aware SSL is not a contrastive model, and its lesion term is unmeasured.**
The contrastive loss has been exactly `0.0000` for 43 consecutive epochs (§5.2): 15 negatives per
anchor at τ = 0.07 makes the objective trivially satisfiable, so the 94-GPU-h artefact is a
masked-reconstruction + lesion-type/count multitask encoder. Independently, the lesion signal only ever
reaches the model through a 128×128 type/count head — **no localisation metric exists anywhere in the
codebase**, so “lesion-aware” is an architecture claim with no measurement, despite 88,702 masks on
disk. The downstream effect is not neutral but negative: ΔEXT binary AUROC **−0.030 [−0.043, −0.019]**.
Honest options: retrain at `ssl_batch_size ≥ 256` **and** add a localisation/attention-alignment
metric, or report it as the pre-registered negative control it already is (with the saturation
analysis, which is itself citable).

**N4 — Joint multi-source fine-tuning: the one positive result, with no mechanism attribution.**
+0.104 EXT QWK is large and highly significant, but the design cannot say *why*: 20,000 labelled
EyePACS images change data quantity, severity coverage, camera diversity and label-protocol diversity
in one step. `joint_w=0.5 > 0.25` only shows “more is better”. The reviewer question “is this just more
data?” is currently unanswered. Controls, each ≈ 3 GPU h: (a) 20,000 extra *in-domain* augmented APTOS
samples (equalise quantity); (b) *unlabelled* EyePACS — **already measured**: `p3a_ssl` trained on
EyePACS images gives +0.000 EXT QWK, which largely excludes “more EyePACS images is enough”; (c)
shuffled-label EyePACS (excludes regularisation-by-noise); (d) a `joint_w` sweep; (e) severity-stratified
transfer (the mild-recall evidence already points here). Because (b) is in hand and negative, the
defensible claim is “**labelled cross-protocol severity coverage, not image volume**” — testable and
publishable.

**N5 — Confounding between phase 3b and the phase-6 DG story.** Amplitude mixing is a source-only DG
method, MixStyle needs the joint sampler (i.e. needs EyePACS in the batch), and the phase-6 reference
inherits whatever phase 3b/5 left behind. If the chain keeps joint training, every “DG” arm also carries
the data-diversity gain, and DEV-based β selection will attribute it to stylisation. Either freeze one
reference configuration for the whole DG grid, or add the `(DG, no-joint)` and `(no-DG, joint)` cells so
the two factors separate.

**N6 — Label ceiling and label provenance.** APTOS is single-grader; published inter-ophthalmologist
agreement on APTOS-like 5-level grading is κ ≈ 0.85–0.92 and the pipeline already reaches 0.926 OOF —
the in-domain endpoint is at or above credible human agreement, which is exactly why no loss
engineering moves it. On the external side, 5-class QWK rests on a derived Messidor label (§2.2), so the
defensible external claims are (i) binary referable AUROC / sensitivity@spec-90, and (ii) *ordinal
direction* (mild recall, adjacent accuracy, ECE) rather than the exact 5-level κ.

**N7 — “Novel ingredient” vs “novel framing”: neither is nailed down yet.** Every component trained so
far (EMD, CORAL, ordinal smoothing, AASL, SSL, joint multi-source, and the planned amplitude mixing /
MixStyle / temperature / conformal) is off-the-shelf. The project's genuine originality is the
**protocol** (leakage-guarded nested cross-fitting, pre-registered adoption rule, paired-OOF bootstrap,
Holm correction) plus a **negative-result + endpoint-validity** finding. That is a solid Q2 /
benchmark-track paper and an easy workshop 1-pager. Top-tier needs ≥1 of: a genuinely new component
(lesion-conditional ordinal head; severity-ordered hard-triplet SSL instead of random-pair InfoNCE; a
hierarchical referable-then-severity decomposition shown to help transfer), a powered multi-site
external study, or a clinical/utility study.

**N8 — Unused assets that could supply the novelty without new data.**
88,702 lesion masks → lesion-grounded attention supervision + a localisation metric; 53,576 **label-free**
EyePACS images (88,702 − 35,126 labelled) → ordinal hard-triplet contrastive learning with zero extra
annotation; the phase-10 XAI stage is already specified and can deliver the “severity evidence is
lesion-grounded, not colour-statistics-grounded” analysis that N4 says is missing.

---

## 10. Decision points — partly overtaken by events (wave in flight, §4.1); the rest is a priced menu

*Started since this menu was written:* the seed programme (Option B step 4) is **running** as
`phase3c_seeds` (`p2b_none`, `p3a_sup`, `p3b_joint`, `p4_coral` × seeds 1–2), the SSL dose decision
(step 3) is **being tested** by `p3a_ssl60`, and the architecture-comparability question is **forked**
onto the reference as `p3c_cnext_ref` / `p3c_effnet_ref` rather than waiting for the phase-4 winner.
Option B steps 1, 2, 5 and 6 are **not** started.

**Option A — finish the plan exactly as written.** ≈ 170 GPU h (§7.2). Deliverable: a 30-row ablation table
where nothing beats an ImageNet-1k MaxViT-L, one external dataset, and now 2–3 seeds for four configs
instead of one (§4.1). Honest venue: Q2
(*Computers in Biology and Medicine*, *Artificial Intelligence in Medicine*, *TVST*, *Eye*) or a
MICCAI/IPMI workshop. Cost of this path: the 0.10 QWK result stays buried in a `REJECT` row.

**Option B — keep the compute, change the primary endpoint (recommended).**
1. Write the endpoint amendment **now**, before phase-5 results exist: primary = external QWK +
   referable AUROC + mild recall (paired bootstrap); secondary = APTOS OOF as a non-inferiority
   constraint (`ΔOOF ≥ −0.005`); in-domain nulls reported as TOST equivalence with a power statement.
2. Re-issue the phase-3b decision under the amended endpoint and re-point the chain to `p3b_joint`
   **before** launching phase 5. Re-run the three phase-4 arms already trained on the new parent
   (≈ 8 GPU h) so the lineage is single-valued.
3. ~~Stop SSL at epoch 60–70~~ → **under test**: `p3a_ssl60` (§4.1) measures the 50→60 dose response
   today; stopping the run now saves the remaining **52 GPU h** and re-files SSL as a negative control
   (§5.3).
4. Spend the freed compute on **seeds**, not arms — **started**: 2 seeds × {`p2b_none`, `p3a_sup`,
   `p3b_joint`, `p4_coral`} are running as §4.1 (≈20 GPU h). Extend to 5 seeds × {reference, best
   phase-4 arm} ≈ 60 GPU h, with **seed as the unit of analysis**, and fix C10 so `seed_summary` can
   actually see them.
5. Prune phase 6 to 4 cells: `(no-DG)`, `amp@best-stage`, `mixstyle`, `amp+mixstyle`, all with
   `winner_eligible=false`, plus the two separability cells from N5.
6. Start the second external dataset **today** (C3) — it is the only item whose lead time exceeds the
   compute.
   Total ≈ 120 GPU h, and every number in the paper becomes either a powered positive or an equivalence.

**Option C — add one genuine novelty (only if a Q1 main track is the target).** Cheapest defensible
choice given the assets: a **lesion-conditional ordinal head** + localisation metric built on the
existing 88,702 masks, evaluated with attention-localisation AUROC (a metric with far more resolution
than QWK at this n) *and* the existing external protocol. Second choice: severity-ordered hard-triplet
SSL on the 53,576 label-free images (fixes the saturation that killed the current SSL). Third: the
hierarchical referable-then-severity decomposition. Each ≈ 25–40 GPU h plus code.

**Stop doing (any option):** the phase-7 feature-balance grid (null expected, 8 GPU h), the 4-way TTA
grid (TTA moves QWK by < 0.005 at this n), and any further loss-function arm evaluated on the OOF
endpoint alone.

### Venue reality check (as of today's numbers)

| Configuration | Credible venue |
|---|---|
| As-is after phase 9, current framing | Q2, or workshop |
| Option B (amended endpoint + seeds + one extra site) | Q1: *IEEE JBHI*, *Artificial Intelligence in Medicine*, *Computers in Biology and Medicine*; stretch *TMI*/*MedIA* |
| Option B + Option C (mechanism + novelty) + 2 sites | *TMI*, *Medical Image Analysis*, *npj Digital Medicine*, *Radiology: AI* |
| + reader study / prospective or multi-centre clinical data | top-tier without further method novelty |

*Update 18:30 — does `p4_coral` move this table? No, and it is worth being explicit about why.* It is the
first arm to clear the pre-registered gate, which is evidence that **the protocol can detect a real
effect at all** — that is what makes the surrounding 19 nulls informative rather than merely empty. But a
+0.0065 QWK gain on a near-ceiling in-domain endpoint, measured with an image-only CI against one
seed-0 reference (C1/C8), is not a venue-moving result and must not be presented as one. The rows above
are gated by the *external* items — second site, endpoint amendment, seed-level inference — and none of
those changed today.

---

## 11. Appendix

**A. Where things live.** `results/<phase>/<tag>/{summary.csv,oof.npz,predictions_external_{test,dev}.npz,calibration.csv,per_class.csv,binary.csv,figures/}` ·
`results/<phase>/{phase_table.csv,WINNER.txt}` · `results/all_runs.npz` (all arms, one row each) ·
`results/master/` (`master_table.csv/.tex`, `master_deltas.csv` — **currently stale: built before
phase 3b**) · `checkpoints/<tag>/{foldK.pth,config_snapshot.yaml}` ·
`checkpoints/pretrain/{eyepacs_sup.pth,eyepacs_sup_v1.pth,ssl_sup.pth,ssl_maxvit/}` ·
`softlabels/sl_p4/outer{0..4}.npz` (**all five complete**, generated 12:00→18:17:37, 6.28 GPU h) ·
`backups/proto_20261103_1723_pre_phase4select.tgz` (3.28 GB protocol snapshot: `results/` + every
`config_snapshot.yaml`; use `--exclude='results/baselines/*/*.pth'` for the ≈100 MB recurring copies, C7) ·
`results/phase3c_{seeds,ssl,dose,arch}/` (the §4.1 exploratory wave — deliberately outside the
pre-registered phases) · `logs/<tag>/*.log` (per-fold detail) ·
`outputs/<phase>/<tag>.txt` (scheduler stdout, archived).

**B. Reproduce every number in §6** (read-only, ~20 s):

```python
import numpy as np
def qwk(y, p, K=5):                      # quadratic weighted kappa, count-based
    n = len(y); W = (np.arange(K)[:,None] - np.arange(K)[None,:])**2.
    cm = np.bincount(y*K + p, minlength=K*K).reshape(K,K).astype(float)
    den = (W * np.outer(cm.sum(1), cm.sum(0))).sum() / n          # the 1/N matters
    return 1.0 if den == 0 else 1.0 - (W*cm).sum()/den
A = np.load('results/phase2b_preprocess/p2b_none/predictions_external_test.npz')
B = np.load('results/phase3b_joint/p3b_joint/predictions_external_test.npz')
y, a, b = A['y_true'].astype(int), A['P'].argmax(1), B['P'].argmax(1)
rng = np.random.default_rng(0); d = []
for _ in range(4000):
    i = rng.integers(0, len(y), len(y)); d.append(qwk(y[i],b[i]) - qwk(y[i],a[i]))   # arm - ref
print(qwk(y,b) - qwk(y,a), np.percentile(d, [2.5, 97.5]))         # +0.1039  [+0.080, +0.130]
```

(Verified: the snippet as printed returns `0.1039 [0.0802 0.1304]`. Note the sign convention — form the
difference *arm minus reference* inside the loop, or the interval comes out negated.)

**C. Status vocabulary** (`phase_select`): `REFERENCE` = the carried-forward baseline; `ADOPT` =
adoption rule met; `TENTATIVE` = rule not met but the DEV columns favour the arm; `REJECT` = rule not
met. Only `ADOPT`/`TENTATIVE` can set `WINNER.txt`, and ties break on `extdev_qwk`.

**D. Open verification items** (small, but they will be asked in review): the exact rule that produced
the 5-class Messidor `diagnosis` column; whether `ext_mild_recall` is computed on argmax or on the
threshold-calibrated predictions (values reproduced here match argmax exactly: 0.104/0.330/0.283 for
`p2b_none`/`p3b_joint`/`p3a_sup` vs stored 0.10/0.33/0.28); the EyePACS→APTOS label equivalence used by
`joint_eyepacs`; and why `p2a_aug` (`aug=medical`, external QWK 0.66, DEV 0.67) was marked `REJECT`
while `p2a_reg` (0.54/0.50) became the winner — that reversal is a direct consequence of C1 and needs an
explicit sentence in the paper.

*Document generated from the repository state above; re-run the extraction scripts after each phase and
diff before quoting numbers in a manuscript.*
