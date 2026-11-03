# DR Severity Grading — Full Result Ledger, Compute Accounting, and Constraint Analysis

**As of:** 2026-11-03 12:48 (cluster local time)
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

**17 fine-tuning arms fully trained** (5 folds each = **85 fold checkpoints, ≈72 GB**),
**4 pretraining runs** (2 × supervised-EyePACS, 1 × SSL, 1 × SSL→supervised),
**3 jobs active right now** (`p4_emd`, `p4_coral`, `sl_p4`).

| Phase | Content | Status | Arms | Official outcome | What the data says (§6) |
|---|---|---|---|---|---|
| 0 | splits, dedup audit, lesion masks, zoo import | ✅ DONE | — | protocol frozen | — |
| 1 | 10-architecture zoo | ✅ DONE (imported, *not trained here*) | 10 | MaxViT-L kept | protocol not comparable to phases 2+ (§3.1) |
| 2a | anti-overfitting recipe, one factor at a time | ✅ DONE | 8 | winner `p2a_reg` (only `TENTATIVE`) | every ΔOOF ≤ 0.01, CI ∋ 0 → indistinguishable |
| 2b | preprocessing parity | ✅ DONE | 3 | winner `p2b_none` (all three `ADOPT`) | a three-way tie |
| 3a | pretraining (EyePACS-sup, SSL, SSL→sup) | ✅ DONE | 4 | **4× REJECT**, reference carried | sup **+0.043 EXT QWK (p<0.001)**; SSL **−0.030 EXT AUROC** |
| 3b | joint APTOS+EyePACS fine-tune | ✅ DONE | 2 | **2× REJECT**, reference carried | **+0.104 EXT QWK [+0.080,+0.130]**, mild recall 0.10→0.33 |
| 4 | EMD / CORAL / AASL soft labels | 🔄 **RUNNING** | 2/5 training, 2 gated | — | unknown; expected ≤1 pt (§9) |
| 5 | architecture + heterogeneous ensemble | ⏳ | 0/3 | — | — |
| 6 | domain generalisation (amplitude mixing, MixStyle, UDA, dev-calibration) | ⏳ | 0/~9 | — | — |
| 7 | feature-level class balance (optional) | ⏳ | 0/5 | — | — |
| 8 | TTA / temperature / conformal / selective prediction | ⏳ | 0/6 | — | — |
| 9 | winner + baseline, seeds 1,2 | ⏳ | 0/6 | — | the only place seed variance enters |
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
| `pre_ssl` | lesion-aware SSL | **63 / 100** (still running) | 1.5e-4, backbone ×0.1 | total loss 0.0139; **Contrastive ≡ 0.0000**, type 0.0028, count 0.0004, recon 0.0619 | **94 h so far** | `ssl_maxvit/ssl_backbone_epoch{40,50,60}.pth` + `ssl_pretrained_backbone.pth` (ep 60) + 3.5 GB resume ckpt |
| `pre_ssl_sup` | SSL → supervised head | 4 | 3e-5 | best **val QWK 0.7729** | 0.6 h | `checkpoints/pretrain/ssl_sup.pth` |

Downstream 5-fold fine-tunes of each backbone:

| Arm | initialisation | OOF QWK | OOF AUROC | test QWK | DEV QWK | DEV mild | EXT QWK | EXT AUROC | EXT mild | noDR% | ECE ext | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `p3a_sup` | `eyepacs_sup.pth` (v2) | 0.93 | 0.98 | 0.92 | **0.70** | **0.33** | **0.74** | **0.95** | **0.28** | 68.2 | 0.15 | REJECT |
| `p3a_sup_v2` | same, lr 3e-5 variant | 0.93 | 0.98 | 0.93 | 0.70 | 0.19 | 0.72 | 0.94 | 0.18 | 72.3 | 0.18 | REJECT |
| `p3a_sslsup` | `ssl_sup.pth` (SSL ep-50 → 4 sup epochs) | 0.92 | 0.98 | 0.92 | 0.65 | 0.14 | 0.73 | 0.94 | 0.16 | 72.3 | **0.13** | REJECT |
| `p3a_ssl` | `ssl_pretrained_backbone.pth` (ep ~58) | 0.92 | 0.98 | 0.92 | 0.64 | 0.12 | 0.70 | 0.93 | 0.07 | 75.5 | 0.18 | REJECT |

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

## 4. What is running at this minute (2026-11-03 12:48)

Three jobs are writing logs whose last line is < 2 min old; a fourth (SSL) is alive but silent
because it prints once per epoch.

| Job | Tag | Started | Progress at 12:48 | Rate | ETA |
|---|---|---|---|---|---|
| `p4_emd` (EMD loss) | phase 4 | 12:11:30 | fold 0 **done** (early-stopped ep 14, val QWK best recorded), fold 1 at ep 5/30 | ~1.6–1.7 min/epoch | fold 0 took 27 min ⇒ **~2.2–2.5 h total, ≈14:30** |
| `p4_coral` (CORAL head) | phase 4 | 12:14:41 (2nd submission — the first died on a malformed `--set`) | fold 0 at ep 20/30, best val QWK 0.938 | 1.6 min/epoch | ~48 min/fold ⇒ **~4 h, ≈16:15** |
| `sl_p4` (AASL soft labels) | phase 4 | 12:00:53 | inner fit “Fold 2”, ep 6/15 | 1.3 min/epoch | 20 inner fits × 15 ep = 300 epochs ⇒ **≈6.5 h, ≈18:30** |
| `pre_ssl` (SSL pretraining) | phase 3a tail | resumed 11-02 15:41 | **epoch 63/100** (logged 11:44) | 93 min/epoch | 37 epochs left ⇒ **≈57 GPU h more** if allowed to finish |

Gate for the remaining two phase-4 arms: `softlabels/sl_p4/outer<k>.npz` does **not exist yet**
(`ls softlabels/sl_p4 | wc -l` = 0), so `p4_aasl` and `p4_aasl_emd` are correctly not submitted.
`checkpoints/sl_p4/inner/` holds the in-progress inner models; nothing is lost if the job dies
between outer folds.

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
  cost ~8 epochs. Run 2 resumed from epoch 50 at 11-02 15:41 and is at epoch 63 now.
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
| **Stop SSL at epoch 60–70** and use `ssl_backbone_epoch60.pth` for `p3a_ssl100` | 0 h extra | “does more SSL help?” answered at one point on the curve instead of two |
| Run SSL to 100 as planned | ≈57 GPU h | tests the *under-training* hypothesis; expected payoff low — the loss is saturated, not plateauing |
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
| `pre_ssl` (2 jobs, epoch 1 → 63) | 1 | 94.0 h | **94 GPU h** |
| `pre_ssl_sup` | 1 | 0.6 h | **0.6 GPU h** |
| phase 3a | 4 | 2.0–2.8 h | **10.2 GPU h** |
| phase 3b | 2 | 4.7 h each | **9.4 GPU h** |
| phase 4 (in flight) | 3 | 0.5–0.8 h so far | ≈ 0.5 h booked |
| phase 0 `detect_lesions` (88,702 masks) | 1 | not logged | unknown, order 10 h |
| | | **Total ≈ 150 GPU h** | of which SSL = **63 %** |

`summary.csv` has a `gpu_hours` column but it is not trustworthy: it reads 0.06–0.08 for every
phase-2a arm because those summaries were regenerated by `tools/rebuild_oof.py`, which times only the
inference pass. Use the log spans above.

### 7.2 Remaining, priced at the measured rates (plain 5-fold arm ≈ 2.8 h, joint arm ≈ 4.7 h)

| Phase | Arms still to run | GPU h |
|---|---|---|
| 4 | `p4_emd` (running), `p4_coral` (running), `sl_p4` (running), then `p4_aasl`, `p4_aasl_emd` | ≈ 25 total, ≈ 13 still to start |
| — | SSL tail, if pushed to epoch 100 | **57** |
| 5 | `p5_wghr`, `p5_convnextv2`, `p5_effnetv2` + `p5_hetero_ens` | ≈ 9 |
| 6 | `p6_ref` + amplitude-mixing grid (2 stages × 3 β) + MixStyle + amp&MS + UDA + dev-calibration | ≈ 45 (**prunable to ≈ 18**) |
| 7 | 5 feature-balance arms (optional) | ≈ 8 |
| 8 | 6 inference-only tags (TTA ×4, temperature, conformal) | ≈ 2 |
| 9 | winner s1,s2 + baseline s1,s2 (+2 inference tags) + `seed_summary` | ≈ 15 |
| 10 | XAI on the winner | ≈ 1 |
| | **≈ 105 GPU h remaining** (≈ 60 with phase 6 pruned and SSL stopped at 60–70) | |

At 4 concurrent single-GPU slots: ≈ 1–1.5 days wall clock for the plan as written, plus queue. The
72 h per-job ceiling is not binding for a single arm (max 4.7 h) but it **did** kill `pre_ssl`
mid-run once (§5.2). Nothing else in the remaining plan approaches it.

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

**C2 — No model-level variance.** *Severity: blocking for any positive claim.*
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

---

## 9. Constraints on the novelties (scientific-claim level)

**N1 — EMD loss / CORAL head / ordinal smoothing: effect below the noise floor, by design.**
The plan itself says “expect ~0–1 point; null results are reported”. One QWK point is ~140× the 0.007
resolution of the gate, and the observed spread of *all* phase-2a loss factors was ≤ 0.01. These arms
can only ever be reported as null-or-equivalence unless the endpoint changes (C1). Two mechanical
restrictions compound it: `config.py` forbids `head_type=coral` together with `soft_labels != none`, so
the CORAL × AASL cell **cannot be crossed** — the two ordinal novelties are mutually exclusive by
construction, which undercuts any claim that ordinal structure per se is what matters; and the running
CORAL arm already shows its classic failure mode (fold 0: train acc 98.8 % vs val acc 84.9 %, best val
QWK 0.938 around epoch 18 then flat) — the ordinal head adds capacity, not transfer.

**N2 — AASL soft labels: the procedure is the contribution, the gain probably is not.**
The nested cross-fitted generation (5 outer × 4 inner = 20 leakage-free 15-epoch fits, ≈ 6.5 GPU h) is
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

## 10. Decision points (nothing below is started — this is a menu, priced)

**Option A — finish the plan exactly as written.** ≈ 105 GPU h. Deliverable: a 30-row ablation table
where nothing beats an ImageNet-1k MaxViT-L, one external dataset, one seed. Honest venue: Q2
(*Computers in Biology and Medicine*, *Artificial Intelligence in Medicine*, *TVST*, *Eye*) or a
MICCAI/IPMI workshop. Cost of this path: the 0.10 QWK result stays buried in a `REJECT` row.

**Option B — keep the compute, change the primary endpoint (recommended).**
1. Write the endpoint amendment **now**, before phase-5 results exist: primary = external QWK +
   referable AUROC + mild recall (paired bootstrap); secondary = APTOS OOF as a non-inferiority
   constraint (`ΔOOF ≥ −0.005`); in-domain nulls reported as TOST equivalence with a power statement.
2. Re-issue the phase-3b decision under the amended endpoint and re-point the chain to `p3b_joint`
   **before** launching phase 5. Re-run the three phase-4 arms already trained on the new parent
   (≈ 8 GPU h) so the lineage is single-valued.
3. Stop SSL at epoch 60–70 (saves ≈ 50 GPU h) and re-file it as a negative control (§5.3).
4. Spend the freed compute on **seeds**, not arms: 5 seeds × {`p2b_none`, `p3a_sup`, `p3b_joint`,
   best phase-4 arm} ≈ 60 GPU h, with seed as the unit of analysis.
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

---

## 11. Appendix

**A. Where things live.** `results/<phase>/<tag>/{summary.csv,oof.npz,predictions_external_{test,dev}.npz,calibration.csv,per_class.csv,binary.csv,figures/}` ·
`results/<phase>/{phase_table.csv,WINNER.txt}` · `results/all_runs.npz` (all arms, one row each) ·
`results/master/` (`master_table.csv/.tex`, `master_deltas.csv` — **currently stale: built before
phase 3b**) · `checkpoints/<tag>/{foldK.pth,config_snapshot.yaml}` ·
`checkpoints/pretrain/{eyepacs_sup.pth,eyepacs_sup_v1.pth,ssl_sup.pth,ssl_maxvit/}` ·
`softlabels/sl_p4/outer<k>.npz` (pending) · `logs/<tag>/*.log` (per-fold detail) ·
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
