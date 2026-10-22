=====================================================================
MODAL-Net — MASTER BUILD PLAN (PART 1 of 2): identity, structure, data, outputs, code
=====================================================================
S0. NAME AND AGENT RULES
- Name: MODAL-Net = Multi-source, Ordinal, Domain-generalizable, Ambiguity-aware, Lesion-aware Network (backbone MaxViT-L). Working title: "MODAL-Net: Multi-Source Lesion-Aware
  Pretraining with Ambiguity-Aware Ordinal Learning for Domain-Generalizable Diabetic Retinopathy Grading". Check the name for collisions before submission.
- Build the repo FROM SCRATCH in a new folder MODAL-Net/. The old repo no longer exists. Four legacy modules are supplied in legacy_seed/ (S7m); copy them, do not rewrite them.
- datasets/ already exists and is READ-ONLY (only datasets/EyePACS/lesion_labels/ may be written, by detect_lesions). Keep its layout exactly (S4).
- Never push to GitHub. Cluster is offline (set HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 in main.py): pretrained weights must already be in the local timm/HF cache; flag any missing weight, never download.
- The APTOS test set (366) and the Messidor-2 TEST split are never used for early stopping, model selection, hyper-parameters or phase decisions. select/decision code must raise an error if given them.
- Fail loudly: a missing image raises FileNotFoundError (the old code silently substituted a black image). validate_cfg(cfg) (S7b) runs at start-up and rejects invalid option combinations.
- Claim hygiene: no "SOTA", "definitive", "proves", "bulletproof"; competitor numbers only from verified papers (with citation) or reproduced under this protocol; every novelty claim needs a literature check.
- After each work package (S13 in Part 2): python -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" <changed files>, run that package's tests (S14 in Part 2), list every changed file.
- All numbers written to CSV / tables / figures use 2 decimals (accuracy and recalls in percent 0-100, QWK/AUROC/F1 as 0.00-1.00).

S1. TARGETS, PRIORS, DECISION RULES
- Owner targets (STRETCH, not guaranteed): internal 5-class acc 92%, internal binary acc 98-99%, external 5-class acc 85%, external binary acc 92%. Binary = referable DR (grades 2,3,4 vs 0,1), score = P2+P3+P4.
- Evidence: zoo MaxViT-L (single fold models): acc 85.19+-0.95, QWK 0.90, bin 93.77, ext 65.21, ext QWK 0.59, ext bin 85.17; train acc 93-99 (overfit). Messidor-2 (n=1744; 1017/270/347/75/35): models predict No DR for
  ~78-82%, Mild recall ~0, Severe -> Moderate. If Mild recall is 0 the maximum external 5-class accuracy is 84.5%; with internal-level per-class recall it is ~84.6%, so 85% external needs better-than-internal Mild/Severe recall.
- Assistant priors (guesses, replace by measurements): internal 5-class 87-89 (ensemble), QWK 0.92-0.94, internal binary 95-96.5, external 5-class 68-76, external binary 88-91.
- PRIMARY metrics (pre-registered): internal = pooled-OOF QWK + referable AUROC; external = Messidor-DEV QWK, Mild recall, predicted-No-DR share, referable sens@spec90. Secondary: accuracy, macro-F1, adjacent acc, ECE, selective accuracy.
- Decision tiers per experiment vs its reference (paired bootstrap, 10,000 resamples, Holm inside the phase): ADOPT = 95% CI of the primary gain excludes 0 and guard metrics do not drop more than the tolerance
  (referable AUROC -0.005, OOF QWK -0.005); TENTATIVE = point gain >= +0.005 QWK but CI includes 0 (allowed ONLY for the a-priori regularisation items of phase2a); REJECT otherwise.
  Rejected components are never carried forward and are still reported. Experiments flagged winner_eligible=false (UDA, dev-calibrated, ensembles) never become winners.
- Overall kill: if after phase6 the external TEST QWK gain over the plain MaxViT-L baseline is < 0.05 (CI includes 0), stop adding methods; write the paper around the analysis/protocol and the calibrated selective-prediction layer.

S2. CONTRIBUTIONS (claimable) AND WHAT IS NOT NEW
 C1 multi-source lesion-aware pretraining pipeline (lesion-pseudo-label multitask SSL on EyePACS -> supervised EyePACS grading -> APTOS fine-tune; EyePACS also as a second labelled source domain). Not new: supervised EyePACS pretraining, contrastive SSL.
 C2 leak-free NESTED cross-fitted ambiguity-aware soft labels + ordinal squared-EMD loss. Not new: confident learning, label smoothing, squared-EMD ordinal loss (Hou et al. 2016, verify), self-distillation. Contribution = nested protocol + evaluation.
 C3 source-only domain generalisation with a multi-source style bank (Fourier amplitude mixing, cross-domain feature MixStyle; Messidor-2 never used for training) + preprocessing-parity analysis. Not new: FDA/FACT, MixStyle.
 C4 ordinal-interval conformal selective prediction for referable-DR screening. Not new: split conformal, selective prediction (verify ordinal-conformal literature).
 C5 evaluation protocol: dedup audit, grouped Messidor dev/test split, selection guard, cross-architecture error analysis, external under-grading analysis.
 Optional: zero-init wavelet-gated residual stem (prior ~0 gain), feature-level balancing (kept only if it beats cheap comparators).
 Literature searches before claiming: "lesion-aware self-supervised diabetic retinopathy EyePACS", "ordinal conformal prediction diabetic retinopathy", "domain generalization fundus Messidor-2 MixStyle Fourier",
 "confident learning diabetic retinopathy label noise", "squared EMD ordinal loss".

S3. ENVIRONMENT (requirements.txt; python 3.9+ / 3.12, torch cu124 installed separately)
  timm>=1.0  albumentations>=1.4  opencv-python  numpy  pandas  scipy  scikit-learn  pyyaml  tqdm  matplotlib  pytest  Pillow  (optional: umap-learn, pytorch-grad-cam, tabulate)
  Tested local environment: macOS Darwin 24.6.0 arm64, Python 3.9.6, torch 2.8.0, torchvision 0.23.0, timm 1.0.30, albumentations 2.0.8, opencv-python-headless 5.0.0.93, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1, pyyaml 6.0.3, tqdm 4.68.3, matplotlib 3.9.4, Pillow 11.3.0, pytest 8.4.2.
  CUDA and cuDNN marked "UNTESTED - recommended" in requirements.txt (CUDA 12.4 + cuDNN 9.x recommended on cluster).
  Diagnostics: python tools/env_report.py prints python/torch/CUDA/cuDNN/timm versions and verifies offline backbone instantiation.
  Check offline availability of: MaxViT-L timm id (default maxvit_large_tf_512.in21k_ft_in1k at 512; use the zoo's id if different), convnextv2_large.fcmae_ft_in22k_in1k, tf_efficientnetv2_m.in21k_ft_in1k.
  Hardware knobs in config: batch_size, grad_accum, grad_checkpointing (use if memory is short), num_workers.

S4. DATASETS (existing, keep exactly; paths are in configs/config.yaml -> paths)
  datasets/
    APTOS_19/   train_images/ val_images/ test_images/   train_1.csv valid.csv test.csv     (CSV columns: id_code, diagnosis; image file = <id_code>.png, extensions .png/.jpg/.jpeg tried in that order)
    Messidor_2/ images/   messidor_data.csv                                                  (columns: id_code, diagnosis, adjudicated_dme, adjudicated_gradable; use rows with adjudicated_gradable == 1; n = 1744)
    EyePACS/    EyePACS/ (*.jpeg)   trainLabels.csv/trainLabels.csv (a folder named trainLabels.csv containing the csv; columns: image, level)   lesion_labels/ (*.npz, created by detect_lesions)
  APTOS split (reproduce the legacy logic EXACTLY so folds match the zoo): merged = concat([train_1.csv, valid.csv], ignore_index=True) (3,296 rows); StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    .split(merged, merged['diagnosis']); fold k = k-th (train_idx, val_idx); images are searched in [train_images, val_images]. TEST = test.csv (366 rows, images in test_images), fixed, never used for selection.
  Messidor-2 split (make_splits): group rows by exam id (image id prefix, e.g. first two '_' tokens; verify against the csv, ask the owner if the pattern fails), StratifiedGroupKFold(5, shuffle=True, random_state=42) on diagnosis:
    fold 0 -> DEV (~20%), others -> TEST (~80%). Save splits/messidor_dev_ids.json, splits/messidor_test_ids.json. DEV is for hyper-parameters/DG decisions/cut-point recalibration/optional UDA pool; TEST is for final reporting.
  Old external numbers used all 1744 images: the zoo rows in the master table carry ext_set=full1744; everything new uses TEST (and DEV for decisions).
  Duplicate audit (tools/dedup_audit.py -> results/phase0_protocol/audit/duplicates.csv): 64-bit DCT pHash of the green channel of a central crop after CLAHE, Hamming <= 4 => candidate pair across/inside APTOS, EyePACS, Messidor-2.
    Fundus images look alike: review the top pairs visually. Rules: duplicates inside APTOS stay in the same fold/test (use groups); any EyePACS image duplicating an APTOS or Messidor image is excluded from EyePACS training; report counts.
  Preprocessing parity hypothesis (to be tested in phase2b, not assumed): the legacy pipeline = crop fundus bounding box -> CLAHE (LAB, clip 2.0, 8x8) -> Ben Graham with sigma=10 at NATIVE resolution -> albumentations Resize(512,512) that STRETCHES non-square crops.
    Native resolution differs by dataset (APTOS up to ~4000 px, Messidor 1440-2304 px), so sigma and aspect ratio are not comparable across domains.

S5. OUTPUT LAYOUT (created by the code; results are grouped by phase, then experiment)
  checkpoints/
    <tag>/fold0.pth ... fold4.pth  fold<k>_val_metrics.json  thresholds.json  config_snapshot.yaml  inner/outer<k>_inner<j>.pth (only for make_softlabels; deletable afterwards)
        (tag = experiment tag, globally unique, prefixed by phase code. Inference-only tags (inference_experiment, ensemble_experiment, feature_balance) contain only config_snapshot.yaml with ckpt_from: <owner tag>.)
    pretrain/eyepacs_sup.pth  pretrain/ssl_maxvit/{ssl_pretrained_backbone.pth, ssl_training_checkpoint.pth}  pretrain/ssl_sup.pth      (backbone state_dicts only)
  logs/<tag>/<mode>_<YYYYmmdd_HHMMSS>.log
  softlabels/<tag>/outer0.npz ... outer4.npz                                                          (ids, P_oof, Q, alpha)
  splits/ aptos_folds.json  messidor_dev_ids.json  messidor_test_ids.json
  results/
    all_runs.npz                                                    (legacy registry: run -> {test, external, ...} metric dicts; written by log_run)
    phase0_protocol/  audit/duplicates.csv  splits_summary.csv
    phase1_zoo/<MODEL>/summary.csv  zoo_table.csv                   (imported from baseline_result.csv)
    phase2a_recipe/<experiment>/...   phase_table.csv  phase_table.tex  WINNER.txt
    phase2b_preprocess/  phase3a_pretrain/  phase3b_joint/  phase4_label_loss/  phase5_architecture/  phase6_domain_gen/  phase7_feature_balance/  phase8_inference/  phase9_final/   (same inner layout)
    master/ master_table.csv  master_table.tex  master_full.csv  master_per_class.csv  master_binary.csv  master_config.csv  master_deltas.csv  seed_variance.csv
    tables/  table01_*.csv/.tex ... table15_*.csv/.tex          (paper tables, built from master)
    figures/ fig01_*.png/.pdf ... fig18_*.png/.pdf               (paper figures; fig16 is fig16_xai_grid.{png,pdf})
  Inside every results/<phase>/<experiment>/:
    summary.csv                one row, all key metrics (schema below)       per_fold.csv     fold rows + mean + std (single-model metrics, test and external)
    per_class.csv              rows class x {test, external}: precision, recall, F1, support
    binary.csv                 referable metrics for test, external_dev, external_test (acc, prec, sens, spec, F1, AUROC, AUPRC, sens@spec90, sens@spec95)
    confusion_test.csv  confusion_external_test.csv  (5x5 counts, rows = true)      calibration.csv   ECE, NLL, Brier, temperature (test, external_test)
    config.yaml                final config of the run        oof.npz  (ids, y, P) pooled out-of-fold CV predictions
    predictions_test.npz  predictions_external_dev.npz  predictions_external_test.npz   (keys: ids, y_true, y_pred, y_prob [ensemble], y_prob_folds [5,N,5])
    curves/fold0.csv ... fold4.csv   (epoch, lr, train_loss, train_acc, val_loss, val_acc, val_qwk, val_bin_auroc, gap)        figures/ curves.png cm_test.png cm_external.png roc.png pr.png reliability.png
    xai/                       fig_xai_grid.{png,pdf} (5-row grid), xai_selection.csv, xai_caption_notes.txt, <method>/class<c>_<id>.png, xai_metrics.csv (if xai_quant=true)
  summary.csv columns (2 dp): phase, experiment, description, set_overrides, inherit, protocol, ext_set, n_params_M, gpu_hours,
    cv_val_acc, cv_val_qwk, cv_val_bin_auroc (mean over folds of best-epoch values), oof_acc, oof_qwk, oof_bin_auroc (pooled OOF),
    test_acc, test_qwk, test_prec_macro, test_rec_macro, test_f1_macro, test_adj_acc, test_mae, test_f1_c0..c4, test_rec_c0..c4, test_acc_single_mean, test_acc_single_std, test_qwk_single_mean, test_qwk_single_std,
    test_bin_acc, test_bin_prec, test_bin_sens, test_bin_spec, test_bin_f1, test_bin_auroc, test_bin_auprc, test_sens_at_spec90,
    extdev_acc, extdev_qwk, extdev_bin_auroc, extdev_mild_recall,
    ext_acc, ext_qwk, ext_prec_macro, ext_rec_macro, ext_f1_macro, ext_adj_acc, ext_mild_recall, ext_pred_nodr_share, ext_bin_acc, ext_bin_prec, ext_bin_sens, ext_bin_spec, ext_bin_f1, ext_bin_auroc, ext_bin_auprc, ext_sens_at_spec90,
    ece_test, ece_ext, train_val_gap, decision, is_winner, notes.      (protocol = 'ensemble5_tta' for 5-fold ensembles, 'single_fold_mean' for imported zoo rows)
    set_overrides is stored with json.dumps(overrides) (so null/true/false survive and yaml.safe_load can read it back).
  phase_table.csv = all summary rows of the phase + reference, d_oof_qwk, ci_lo, ci_hi, p_holm, d_bin_auroc, d_extdev_qwk, d_extdev_mild_recall, decision (ADOPT/TENTATIVE/REJECT/REFERENCE), winner.

S6. REPOSITORY TREE (create all)
  MODAL-Net/
    main.py  requirements.txt  README.md  pytest.ini  .gitignore
    configs/ config.yaml  config_ssl.yaml
    datasets/ (README.md, existing data symlinked, images never committed)   splits/   checkpoints/   logs/   softlabels/   results/
    tools/dedup_audit.py  tools/env_report.py
    agent_context/ (tracked: MODAL_PART1.md, MODAL_PART2.md, MODAL_PART3.md, baseline_result.csv, legacy_seed/)
    src/
      utils/        config.py paths.py logger.py seed.py io_utils.py
      data/         splits.py preprocess.py datasets.py augment.py samplers.py amplitude_mix.py eyepacs_dataset.py(seed)
      preprocessing/lesion_detection.py(seed)
      models/       gradenet.py mixstyle.py wavelet_gate.py ssl_model.py(seed, adapted)
      losses/       losses.py
      training/     trainer.py ema.py softlabels.py pretrain.py ssl_trainer.py(seed) feature_balance.py
      evaluation/   metrics.py stats.py predict.py conformal.py calibration.py experiment.py tables.py figures.py
      xai/          cams.py grid.py
    tests/          test_*.py (T1-T16)

S7. CODE SPECIFICATION
S7a. configs/config.yaml (defaults; every key can be overridden with --set key=value)
    paths: {aptos: {train_images: datasets/APTOS_19/train_images, val_images: datasets/APTOS_19/val_images, test_images: datasets/APTOS_19/test_images,
                    train_csv: datasets/APTOS_19/train_1.csv, val_csv: datasets/APTOS_19/valid.csv, test_csv: datasets/APTOS_19/test.csv},
            messidor: {images: datasets/Messidor_2/images, csv: datasets/Messidor_2/messidor_data.csv},
            eyepacs: {images: datasets/EyePACS/EyePACS, labels_csv: datasets/EyePACS/trainLabels.csv/trainLabels.csv, lesion_labels: datasets/EyePACS/lesion_labels}}
    backbone: maxvit_large_tf_512.in21k_ft_in1k   img_size: 512   img_size_arg: null   num_classes: 5   head_type: softmax   preproc: legacy   aug: basic
    epochs: 30  warmup_epochs: 2  patience: 8  batch_size: 16  grad_accum: 1  num_workers: 8  grad_checkpointing: false  seed: 0  n_folds: 5  kfold_seed: 42
    lr: 1.0e-4  llrd: 0.8  head_mult: 10  weight_decay: 0.05  drop_path: 0.3  head_dropout: 0.3  grad_clip: 1.0  freeze_epochs: 1
    loss: ce  lambda_emd: 1.0  ordinal_eps: 0.1  ordinal_sigma: 0.7  mix_prob: 0.5  mixup_alpha: 0.2  cutmix_alpha: 1.0  ema_decay: 0.999  use_sam: false  sam_rho: 0.05
    soft_labels: none  soft_label_run: null  alpha_max: 0.4  inner_k: 4  inner_epochs: 15
    pretrain_path: null  joint_eyepacs: false  joint_w: 0.5  eyepacs_max: 20000
    amp_mix: false  amp_prob: 0.5  amp_beta_min: 0.01  amp_beta_max: 0.10  amp_stage: pre  style_bank_size: 2000
    mixstyle: false  ms_p: 0.5  ms_alpha: 0.1  ms_stages: [0, 1]  uda_style_pool: none  use_wghr: false
    tta: none  tta_scales: [1.0]  temperature_scaling: true  conformal_alpha: 0.10  recalibrate: none  winner_eligible: true  bundle_from: null  ckpt_from: null
    pre_epochs: 15  ssl_init: null
    xai_seed: 0  xai_top_k: 10  xai_methods: [gradcampp, gradcam, layercam, eigencam]  xai_show_lesions: true  xai_split: test  xai_alpha: 0.5  xai_quant: false  xai_quant_max_images: 400
  Defaults = the a-priori MODAL recipe (regularisation, mixup, EMA, ordinal smoothing, LLRD) except aug (basic, tested in phase2a) and preproc (legacy, tested in phase2b). The plain zoo-like baseline is created with explicit overrides in phase2a.
  tta defaults to none to keep the ~40 experiments fast (8 views x 5 models x 3 sets is slow); TTA is decided in phase8 and the final runs set it explicitly.
S7b. CLI (main.py) and config resolution (src/utils/config.py)
    python main.py --mode MODE [--phase P] [--tag T] [--inherit T|@phase] [--set k=v ...] [--fold N] [--base T|@phase] [--members a,b,c] [--kind K] [--method M] [--reference T|@phase]
                   [--design parallel] [--csv path]
    MODES: make_splits dedup_audit detect_lesions import_zoo pretrain experiment train test external_validation make_softlabels inference_experiment ensemble_experiment feature_balance
           calibrate_conformal phase_select seed_summary master_table report xai
    def parse_set(pairs): return {p.split('=', 1)[0]: yaml.safe_load(p.split('=', 1)[1]) for p in pairs}       # values: true/false/null/numbers/lists/strings
    def resolve(x):                         # '@phase2a_recipe' -> tag written in results/phase2a_recipe/WINNER.txt (first token of the file)
        return open(f'results/{x[1:]}/WINNER.txt').read().split()[0] if x and x.startswith('@') else x
    def find_exp(tag): return sorted(glob.glob(f'results/phase*/{tag}'))[0]        # results folder of an experiment, whatever phase it belongs to (cross-phase references)
    def ckpt_owner(cfg): return cfg.get('ckpt_from') or cfg['tag']                  # inference-only experiments load the checkpoints of ckpt_from
    def build_cfg(args):
        cfg = yaml.safe_load(open('configs/config.yaml'))
        base = resolve(args.inherit) or (args.tag if args.mode in {'test', 'external_validation', 'calibrate_conformal'} else None)
        if base and os.path.exists(f'checkpoints/{base}/config_snapshot.yaml'): cfg.update(yaml.safe_load(open(f'checkpoints/{base}/config_snapshot.yaml')))
        ov = parse_set(args.set); cfg.update(ov)
        if cfg.get('bundle_from'):              # merge the overrides of every ADOPT/TENTATIVE experiment of that phase (rows of its phase_table.csv, in table order)
            t = pd.read_csv(f"results/{cfg['bundle_from']}/phase_table.csv")
            for _, r in t[t.decision.isin(['ADOPT', 'TENTATIVE'])].iterrows(): cfg.update(yaml.safe_load(r['set_overrides']) or {})
            cfg.update(ov)
        cfg.update(tag=args.tag, phase=args.phase, set_overrides=json.dumps(ov), inherit=str(args.inherit)); validate_cfg(cfg); return cfg
    def validate_cfg(c):
        assert not (c['amp_mix'] and c['preproc'] == 'legacy'), 'amp_mix needs preproc != legacy (equal-size, aspect-preserving images)'
        assert not (c['mixstyle'] and 'maxvit' not in c['backbone']), 'mixstyle manual stage loop is implemented for timm MaxViT only'
        assert not (c['head_type'] == 'coral' and c['soft_labels'] != 'none'), 'coral is incompatible with soft labels'
        assert c['uda_style_pool'] in ('none', 'messidor_dev') and c['tta'] in ('none', 'flip', 'flip_rot') and c['recalibrate'] in ('none', 'devcut')
    Snapshot: train/experiment/pretrain/make_softlabels save the final cfg to checkpoints/<tag>/config_snapshot.yaml before training; inference-only modes save one with ckpt_from.
    Evaluation modes restore the snapshot of their own tag automatically, then apply --set on top. Logging: logs/<tag>/<mode>_<timestamp>.log plus console. seed_everything(cfg.seed) (python, numpy, torch).
S7c. src/data/splits.py
    def aptos_frames(cfg):
        P = cfg['paths']['aptos']; merged = pd.concat([pd.read_csv(P['train_csv']), pd.read_csv(P['val_csv'])], ignore_index=True)
        skf = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['kfold_seed'])
        return merged, list(skf.split(merged, merged['diagnosis']))
    def fold_frames(cfg, k): m, sp = aptos_frames(cfg); a, b = sp[k]; return m.iloc[a].reset_index(drop=True), m.iloc[b].reset_index(drop=True)
    def test_frame(cfg): return pd.read_csv(cfg['paths']['aptos']['test_csv'])
    def messidor_all(cfg): d = pd.read_csv(cfg['paths']['messidor']['csv']); return d[d['adjudicated_gradable'] == 1].reset_index(drop=True)
    def make_messidor_split(cfg):
        d = messidor_all(cfg); g = d['id_code'].astype(str).str.split('_').str[:2].str.join('_')        # exam id: VERIFY against the csv
        dev_idx = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d, d['diagnosis'], g)))[1]
        dev = set(d.loc[dev_idx, 'id_code'].astype(str)); json.dump(sorted(dev), open('splits/messidor_dev_ids.json', 'w'))
        json.dump(sorted(set(d['id_code'].astype(str)) - dev), open('splits/messidor_test_ids.json', 'w'))
    def messidor_frame(cfg, split):  # 'dev' | 'test'
        d = messidor_all(cfg); ids = set(json.load(open(f'splits/messidor_{split}_ids.json'))); return d[d['id_code'].astype(str).isin(ids)].reset_index(drop=True)
    def eyepacs_frame(cfg, exclude_ids=()):
        d = pd.read_csv(cfg['paths']['eyepacs']['labels_csv']).rename(columns={'image': 'id_code', 'level': 'diagnosis'}); return d[~d.id_code.isin(exclude_ids)].reset_index(drop=True)
    make_splits also writes splits/aptos_folds.json (ids per fold) and results/phase0_protocol/splits_summary.csv (counts per class per split) and asserts: test ids disjoint from all fold ids, folds partition the pool.
S7d. src/data/preprocess.py (preproc modes: legacy | scaled | clahe_only | none)
    def crop_fundus(img, thr=10, margin=5):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); _, t = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
        cs, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea)); x, y = max(0, x - margin), max(0, y - margin)
            w, h = min(img.shape[1] - x, w + 2 * margin), min(img.shape[0] - y, h + 2 * margin); img = img[y:y + h, x:x + w]
        return img
    def clahe(img):
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)); l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    def ben_graham(img, sigma): return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
    def pad_square_resize(img, size):
        h, w = img.shape[:2]; s = max(h, w); c = np.zeros((s, s, 3), img.dtype); y0, x0 = (s - h) // 2, (s - w) // 2; c[y0:y0 + h, x0:x0 + w] = img
        return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
    def preprocess(bgr, mode, size, style_rgb=None, amp=None):      # returns BGR uint8
        img = crop_fundus(bgr)
        if mode == 'legacy': return ben_graham(clahe(img), 10)        # native-resolution sigma 10; Resize(size,size) later in albumentations (zoo behaviour, kept for reproducibility)
        img = pad_square_resize(img, size)                            # aspect-preserving; fixed relative scale
        if amp and amp['stage'] == 'pre' and style_rgb is not None: img = cv2.cvtColor(amplitude_mix(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), style_rgb, amp['beta'], amp['lam']), cv2.COLOR_RGB2BGR)
        if mode == 'scaled': img = ben_graham(clahe(img), size / 50)
        elif mode == 'clahe_only': img = clahe(img)
        if amp and amp['stage'] == 'post' and style_rgb is not None: img = cv2.cvtColor(amplitude_mix(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), style_rgb, amp['beta'], amp['lam']), cv2.COLOR_RGB2BGR)
        return img
    Style bank (built once per process, fixed seed, style_bank_size images from datasets/EyePACS/EyePACS, or from Messidor-DEV for the UDA row): stage 'pre' -> crop_fundus + pad_square_resize, kept as RGB;
    stage 'post' -> the full preprocess(mode) output, converted to RGB. amp_mix with preproc=legacy is rejected by validate_cfg.
S7e. src/data/augment.py
    build_transform(cfg, train):  val/test = Compose([Resize(size,size), Normalize(ImageNet mean/std), ToTensorV2()]).
    aug='basic' (zoo-like): RandomResizedCrop(size=(s,s), scale=(0.9,1.0), ratio=(0.9,1.1), interpolation=cv2.INTER_CUBIC), RandomRotate90, HorizontalFlip, VerticalFlip, ColorJitter(0.2,0.2,0.2,0.02,p=.5),
      RandomBrightnessContrast(0.2,0.2,p=.5), CLAHE(2.0,p=.3), Sharpen(alpha=(.2,.5),lightness=(.5,1),p=.3), Affine(translate_percent=.1, scale=.9, rotate=45, p=.3),
      CoarseDropout(num_holes_range=(1,3), hole_height_range=(8,16), hole_width_range=(8,16), fill=0, p=.15), Normalize, ToTensorV2.
    aug='medical': Affine(scale=(.9,1.1), rotate=(-180,180), translate_percent=(-.05,.05), p=.8), HorizontalFlip, VerticalFlip, RandomBrightnessContrast(.2,.2,p=.5), RandomGamma((80,120),p=.3), GaussianBlur((3,3),p=.2),
      ImageCompression(quality_range=(70,100), p=.3), CoarseDropout (as above, p=.15), Normalize, ToTensorV2.   No elastic/grid distortion. Verify argument names against the installed albumentations.
S7f. src/data/datasets.py and samplers.py
    class DRDataset(Dataset):
        def __init__(self, df, dirs, cfg, train, soft=None, domain=0, style_bank=None):    # df has id_code, diagnosis; soft = {id: Q(5,)} or None
            self.df, self.dirs, self.cfg, self.train, self.soft, self.domain, self.bank = df.reset_index(drop=True), dirs, cfg, train, soft, domain, style_bank; self.tf = build_transform(cfg, train)
        def _read(self, name):
            for d in self.dirs:
                for p in ([os.path.join(d, name)] if name.lower().endswith(('.png', '.jpg', '.jpeg')) else [os.path.join(d, name + e) for e in ('.png', '.jpg', '.jpeg')]):
                    if os.path.isfile(p):
                        im = cv2.imread(p)
                        if im is not None: return im
            raise FileNotFoundError(name)
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]; name = str(r['id_code']); y = int(r['diagnosis']); amp = None; sty = None
            if self.train and self.cfg['amp_mix'] and self.bank is not None and random.random() < self.cfg['amp_prob']:
                amp = {'stage': self.cfg['amp_stage'], 'beta': random.uniform(self.cfg['amp_beta_min'], self.cfg['amp_beta_max']), 'lam': random.random()}; sty = random.choice(self.bank)
            img = cv2.cvtColor(preprocess(self._read(name), self.cfg['preproc'], self.cfg['img_size'], sty, amp), cv2.COLOR_BGR2RGB)
            x = self.tf(image=img)['image']; q = self.soft[name] if self.soft is not None else base_target(y, self.cfg)      # base_target = (1-eps)*onehot + eps*ordinal_gaussian (eps 0 => one-hot)
            return {'x': x, 'y': y, 'q': torch.as_tensor(q, dtype=torch.float32), 'domain': self.domain, 'id': name}
    Loaders: train = DataLoader(ds, batch_size, shuffle=True, drop_last=True, num_workers, pin_memory=True, persistent_workers=True); eval = shuffle False, drop_last False.
    Joint mode: DataLoader(ConcatDataset([aptos_train, eyepacs_ds]), batch_sampler=JointBatchSampler(...), num_workers=..., pin_memory=True)   (batch_sampler, not sampler; do not pass batch_size/shuffle/drop_last).
    class JointBatchSampler(Sampler):     # indices into ConcatDataset([aptos_train, eyepacs]); each batch = [B/2 APTOS | B/2 EyePACS]; EyePACS drawn with prob ~ 1/sqrt(class frequency); epoch = n_aptos // (B/2) batches
        def __init__(self, n_a, y_e, B, seed=0): self.n_a, self.B, self.y_e = n_a, B, y_e; f = np.bincount(y_e, minlength=5).astype(float); self.pe = (1 / np.sqrt(f))[y_e]; self.pe /= self.pe.sum(); self.rng = np.random.default_rng(seed)
        def __iter__(self):
            h = self.B // 2; perm = self.rng.permutation(self.n_a)
            for s in range(0, len(perm) - h + 1, h): yield [int(i) for i in perm[s:s + h]] + [int(self.n_a + j) for j in self.rng.choice(len(self.y_e), h, p=self.pe)]
        def __len__(self): return self.n_a // (self.B // 2)
    Per-sample loss weight: APTOS 1.0, EyePACS cfg.joint_w (from batch['domain']). EyePACS frame = eyepacs_frame(cfg, exclude_ids=duplicates) subsampled to eyepacs_max with a fixed seed. Evaluate only on APTOS folds/test and Messidor.
S7g. src/models/gradenet.py (MixStyle is part of the model so .train()/.eval() and deepcopy for EMA behave correctly; no forward hooks)
    class GradeNet(nn.Module):
        def __init__(self, cfg):
            super().__init__(); kw = dict(pretrained=True, num_classes=0, drop_path_rate=cfg['drop_path'])
            if cfg.get('img_size_arg'): kw['img_size'] = cfg['img_size_arg']
            self.backbone = timm.create_model(cfg['backbone'], **kw)
            if cfg['grad_checkpointing']: self.backbone.set_grad_checkpointing(True)
            self.wghr = WaveletGate() if cfg['use_wghr'] else nn.Identity()
            self.ms_idx = list(cfg['ms_stages']) if cfg['mixstyle'] else []; self.ms = nn.ModuleList([MixStyle(cfg['ms_p'], cfg['ms_alpha']) for _ in self.ms_idx])
            K = cfg['num_classes']; out = K - 1 if cfg['head_type'] == 'coral' else K
            self.head = nn.Sequential(nn.Dropout(cfg['head_dropout']), nn.Linear(self.backbone.num_features, out))
            if cfg['pretrain_path']:
                sd = torch.load(cfg['pretrain_path'], map_location='cpu'); sd = sd.get('state_dict', sd); mis, unx = self.backbone.load_state_dict(sd, strict=False)
                assert len(mis) < 0.05 * len(self.backbone.state_dict()), f'pretrain load: {len(mis)} missing, {len(unx)} unexpected'
        def forward(self, x):
            x = self.wghr(x); b = self.backbone
            if not self.ms_idx: f = b(x)
            else:                                   # manual MaxViT path (VERIFY attribute names stem / stages / norm / forward_head in the installed timm)
                f = b.stem(x)
                for i, st in enumerate(b.stages):
                    f = st(f)
                    if i in self.ms_idx: f = self.ms[self.ms_idx.index(i)](f)
                f = b.forward_head(b.norm(f))
            return self.head(f)
    def probs_from_logits(z, head_type):
        if head_type != 'coral': return F.softmax(z, 1)
        c = torch.sigmoid(z); c = torch.cat([torch.ones_like(c[:, :1]), c, torch.zeros_like(c[:, :1])], 1); return (c[:, :-1] - c[:, 1:]).clamp_min(1e-8)     # differences of P(y>=k)
    def llrd_groups(model, lr, decay, wd, head_mult):    # layer-wise LR decay over stem + stages.0..3 (timm MaxViT names; VERIFY with named_children()); if decay == 1.0 or names not found: one backbone group
        order = ['stem', 'stages.0', 'stages.1', 'stages.2', 'stages.3']; groups, used = [], set()
        for i, n in enumerate(order):
            ps = [(k, p) for k, p in model.backbone.named_parameters() if k.startswith(n)]; used |= {k for k, _ in ps}
            if ps: groups.append({'params': [p for _, p in ps], 'lr': lr * decay ** (len(order) - 1 - i), 'weight_decay': wd})
        groups.append({'params': [p for k, p in model.backbone.named_parameters() if k not in used], 'lr': lr, 'weight_decay': wd})
        groups.append({'params': list(model.head.parameters()), 'lr': lr * head_mult, 'weight_decay': wd}); return [g for g in groups if g['params']]
    (Put bias / norm parameters of every group into a twin group with weight_decay 0.)
    src/models/wavelet_gate.py (optional A1; zero-init so step 0 equals the pretrained model):
    class WaveletGate(nn.Module):
        def __init__(self, hidden=8): super().__init__(); self.att = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 1, 3, padding=1)); self.gamma = nn.Parameter(torch.zeros(1))
        def forward(self, x):                                           # H, W even
            a, b, c, d = x[..., 0::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 0::2], x[..., 1::2, 1::2]
            e = (((a + b - c - d) / 4) ** 2 + ((a - b + c - d) / 4) ** 2 + ((a - b - c + d) / 4) ** 2).sum(1, keepdim=True).sqrt()
            g = F.interpolate(torch.sigmoid(self.att(e)), size=x.shape[-2:], mode='bilinear', align_corners=False)
            return x + self.gamma * g * (x - F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode='nearest'))
    src/models/mixstyle.py (Zhou et al. 2021; cross-domain pairing for [domain A | domain B] batches):
    class MixStyle(nn.Module):
        def __init__(self, p=0.5, alpha=0.1, eps=1e-6): super().__init__(); self.p, self.eps = p, eps; self.beta = torch.distributions.Beta(alpha, alpha)
        def forward(self, x):
            if not self.training or torch.rand(1).item() > self.p: return x
            B = x.size(0); mu = x.mean((2, 3), keepdim=True).detach(); sig = (x.var((2, 3), keepdim=True, unbiased=False) + self.eps).sqrt().detach(); xn = (x - mu) / sig
            lam = self.beta.sample((B, 1, 1, 1)).to(x.device); h = B // 2
            perm = (torch.cat([torch.arange(h, B), torch.arange(0, h)]) if B % 2 == 0 else torch.randperm(B)).to(x.device)
            return xn * (lam * sig + (1 - lam) * sig[perm]) + (lam * mu + (1 - lam) * mu[perm])
    (Without the joint sampler the pairing is meaningless: mixstyle experiments always set joint_eyepacs=true.)
    src/data/amplitude_mix.py:
    def amplitude_mix(src, trg, beta, lam):                              # uint8 RGB, any size; trg is resized to src
        trg = cv2.resize(trg, (src.shape[1], src.shape[0])); s, t = src.astype(np.float32), trg.astype(np.float32)
        Fs = np.fft.fftshift(np.fft.fft2(s, axes=(0, 1)), axes=(0, 1)); Ft = np.fft.fftshift(np.fft.fft2(t, axes=(0, 1)), axes=(0, 1))
        As, Ps, At = np.abs(Fs), np.angle(Fs), np.abs(Ft); h, w = s.shape[:2]; b = max(1, int(min(h, w) * beta)); cy, cx = h // 2, w // 2
        As[cy-b:cy+b, cx-b:cx+b] = (1 - lam) * As[cy-b:cy+b, cx-b:cx+b] + lam * At[cy-b:cy+b, cx-b:cx+b]
        out = np.fft.ifft2(np.fft.ifftshift(As * np.exp(1j * Ps), axes=(0, 1)), axes=(0, 1)).real
        return np.clip(out * (s.sum(2, keepdims=True) > 15), 0, 255).astype(np.uint8)                  # keep the black border
S7h. src/losses/losses.py
    def ordinal_gaussian(y, K=5, sigma=0.7): k = torch.arange(K, device=y.device)[None].float(); g = torch.exp(-(k - y[:, None].float()) ** 2 / (2 * sigma ** 2)); return g / g.sum(1, keepdim=True)
    def soft_ce(z, q, w=None): l = -(q * F.log_softmax(z, 1)).sum(1); return (l * w).sum() / w.sum() if w is not None else l.mean()
    def emd2(z, q, w=None): p = F.softmax(z, 1); l = ((p.cumsum(1) - q.cumsum(1)) ** 2).sum(1) / z.size(1); return (l * w).sum() / w.sum() if w is not None else l.mean()
    def coral_loss(z, y, K=5): lv = (y[:, None] > torch.arange(K - 1, device=y.device)[None]).float(); return F.binary_cross_entropy_with_logits(z, lv)
    def total_loss(z, q, y, cfg, w=None):
        if cfg['head_type'] == 'coral': return coral_loss(z, y)
        L = soft_ce(z, q, w)
        return (L + cfg['lambda_emd'] * emd2(z, q, w)) if cfg['loss'] == 'ce_emd' else L
    base_target(y, cfg) = (1-eps)*onehot + eps*ordinal_gaussian(y, sigma) (eps = cfg.ordinal_eps; eps 0 => one-hot); nested soft labels Q (S7k) already include the ordinal smoothing. Mixup/CutMix mixes q exactly like the images.
S7i. src/training/trainer.py (+ ema.py): one function per fold
    class ModelEMA: __init__(model, decay): self.m = copy.deepcopy(model).eval(); self.d = decay;  update(model): for e, p in zip(self.m.parameters(), model.parameters()): e.data.mul_(self.d).add_(p.data, alpha=1 - self.d); for be, b in zip(self.m.buffers(), model.buffers()): be.copy_(b)
    def mix_batch(x, q, cfg):
        if torch.rand(1).item() > cfg['mix_prob']: return x, q
        perm = torch.randperm(x.size(0), device=x.device)
        if torch.rand(1).item() < 0.5: lam = float(np.random.beta(cfg['mixup_alpha'], cfg['mixup_alpha'])); x = lam * x + (1 - lam) * x[perm]
        else:
            lam = float(np.random.beta(cfg['cutmix_alpha'], cfg['cutmix_alpha'])); H, W = x.shape[-2:]; r = math.sqrt(1 - lam); ch, cw = int(H * r), int(W * r); cy, cx = np.random.randint(H), np.random.randint(W)
            y1, y2, x1, x2 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw // 2, 0), min(cx + cw // 2, W); x = x.clone(); x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]; lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        return x, lam * q + (1 - lam) * q[perm]
    def train_fold(cfg, k, tag, logger, train_df=None, val_df=None, out=None):      # train_df/val_df/out are overridden by the nested soft-label code; default = APTOS fold k
        1. tr, va = fold_frames(cfg, k) unless given; soft = load_softlabels(cfg, k) if cfg.soft_labels == 'aasl' (dict id -> Q; assert coverage of every train id); dirs = [train_images, val_images].
        2. datasets/loaders (S7f); joint: ConcatDataset([DRDataset(tr, dirs, ..., domain=0), DRDataset(eyepacs_frame, [eyepacs images], ..., domain=1)]) with JointBatchSampler.
        3. model = GradeNet(cfg).cuda() (MixStyle is built inside when cfg.mixstyle); ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 (created AFTER the model is complete).
           opt = AdamW(llrd_groups(model, lr, llrd, wd, head_mult)); per-step schedule: linear warmup (warmup_epochs) then cosine to 1% of base lr. Freeze the stem + stage 0 for the first freeze_epochs epochs.
        4. loop epochs: model.train(); for b in loader: x, q, y = b['x'].cuda(), b['q'].cuda(), b['y'].cuda(); x, q = mix_batch(x, q, cfg); w = where(b['domain'] == 1, joint_w, 1.0) if joint else None;
              with torch.autocast('cuda', dtype=torch.bfloat16): z = model(x); loss = total_loss(z.float(), q, y, cfg, w) / grad_accum;   loss.backward(); every grad_accum steps: clip_grad_norm_(grad_clip), opt.step(), sched.step(), zero_grad, ema.update(model).
           (optional use_sam: SAM two-pass wrapper, rho 0.05, no grad accumulation; implement last, default off.)
        5. evaluate with the EMA model (or model) in eval mode: probs on the fold's val set (no TTA) -> metrics (qwk, acc, bin_auroc); train accuracy on a fixed random 600-image subset of the training fold with eval transforms (gap diagnostic);
           append the row to results/<phase>/<tag>/curves/fold<k>.csv.
        6. early stopping on val QWK (patience cfg.patience; improvement threshold 1e-4); at every new best: save the evaluation model state_dict to checkpoints/<tag>/fold<k>.pth (out overrides the folder), fold<k>_val_metrics.json {epoch, val_acc, val_qwk, val_bin_auroc},
           and the OOF arrays results/<phase>/<tag>/oof_fold<k>.npz (ids, y, P).
        7. after all 5 folds (experiment mode): merge oof_fold*.npz -> oof.npz; thresholds.json = {'thr_spec90': threshold of the referable score at 90% specificity on the pooled OOF, 'thr_spec95': ..., 'temperature': fitted T}.
    Resume: skip a fold if checkpoints/<tag>/fold<k>.pth and fold<k>_val_metrics.json exist and the snapshot matches. Gradient checkpointing and bf16 autocast are used on A100/H100; no GradScaler needed.
S7j. src/evaluation: predict.py, metrics.py, stats.py, calibration.py, conformal.py
    predict.py:
      def apply_view(x, r, h, v): x = torch.rot90(x, r, (2, 3)); x = x.flip(3) if h else x; return x.flip(2) if v else x
      TTA = {'none': [(0, 0, 0)], 'flip': [(0, h, v) for h in (0, 1) for v in (0, 1)], 'flip_rot': [(r, h, 0) for r in range(4) for h in (0, 1)]}
      def rescale(x, s): return x if s == 1.0 else F.interpolate(F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False), size=x.shape[-2:], mode='bilinear', align_corners=False)
      @torch.no_grad() def predict_probs(models, loader, cfg, tta=None, scales=None): average softmax (probs_from_logits) over models x views x scales; returns ids, y, P (N,5) and per-model P.
      ensemble = the 5 fold checkpoints of ckpt_owner(cfg) (abort with an error if fewer than 5 exist). Also write per-fold single-model metrics (no TTA) for per_fold.csv mean/std.
    metrics.py (all functions take y (N,), P (N,5)):
      metrics5: acc (%), qwk (cohen_kappa_score weights='quadratic'), prec_macro, rec_macro, f1_macro, per-class P/R/F1/support, adj_acc (|pred-y|<=1, %), mae, mild_recall (class 1), pred_nodr_share (% predicted class 0).
      metrics_bin (referable): s = P[:, 2:].sum(1); acc/prec/sens/spec/f1 at threshold 0.5, auroc (roc_auc_score), auprc (average_precision_score), sens_at_spec90/95 = realised sensitivity when the threshold comes from checkpoints/<owner>/thresholds.json
        (fitted on pooled OOF only), never from the evaluated set.
      calibration: ece(P, y, bins=15) = sum_b |acc_b - conf_b| * frac_b ; nll ; brier ; temperature scaling: fit T on OOF log-probabilities by minimising NLL (scipy minimize_scalar), apply to test/external probabilities when cfg.temperature_scaling.
      recalibrate == 'devcut': expected grade E = sum_k k*P_k; fit E' = a*E + b on Messidor-DEV (2 parameters, maximise QWK), predictions = digitize(E', [0.5, 1.5, 2.5, 3.5]); evaluated on Messidor TEST; labelled "dev-calibrated" (winner_eligible=false).
    stats.py:
      def paired_bootstrap(y, Pa, Pb, fn, B=10000, seed=0):
          rng = np.random.default_rng(seed); n = len(y); d = np.empty(B)
          for b in range(B): i = rng.integers(0, n, n); d[b] = fn(y[i], Pa[i]) - fn(y[i], Pb[i])
          delta = fn(y, Pa) - fn(y, Pb); return delta, np.percentile(d, [2.5, 97.5]), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
      def holm(p): o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
          for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
          return adj
      def mcnemar_exact(y, pa, pb): a, b = pa == y, pb == y; n01, n10 = int((~a & b).sum()), int((a & ~b).sum()); return scipy.stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    conformal.py (ordinal-interval split conformal; calibration set = pooled OOF):
      def interval_scores(P, y):                 # minimal contiguous-interval mass around the argmax that contains the true grade
          out = np.zeros(len(y)); K = P.shape[1]
          for i in range(len(y)):
              lo = hi = int(P[i].argmax()); m = P[i, lo]
              while not (lo <= y[i] <= hi):
                  l = P[i, lo - 1] if lo > 0 else -1.0; r = P[i, hi + 1] if hi < K - 1 else -1.0
                  if l >= r: lo -= 1; m += P[i, lo]
                  else: hi += 1; m += P[i, hi]
              out[i] = m
          return out
      qhat = np.quantile(scores_cal, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method='higher')
      def interval_set(p, qhat): grow the interval greedily around the argmax until mass >= qhat; return (lo, hi)
      Report coverage, mean width, singleton share, singleton accuracy, abstention for referable decisions (interval crossing grade 1|2 => abstain), risk-coverage, AURC, accuracy at 70/80/90% coverage; on Messidor coverage may degrade (shift): report it.
S7k. src/training/softlabels.py (ambiguity-aware soft labels, NESTED)
    def build_soft_labels(P, y, K=5, eps=0.1, sigma=0.7, alpha_max=0.4):
        n = len(y); oh = np.eye(K)[y]; kk = np.arange(K)[None]; g = np.exp(-(kk - y[:, None]) ** 2 / (2 * sigma ** 2)); g /= g.sum(1, keepdims=True); base = (1 - eps) * oh + eps * g
        s = P[np.arange(n), y]; t = np.array([s[y == j].mean() if (y == j).any() else 1.0 for j in range(K)])        # class-wise confident-learning thresholds
        a = np.clip((t[y] - s) / np.maximum(t[y], 1e-6), 0, 1) * alpha_max; Q = (1 - a)[:, None] * base + a[:, None] * P; return Q / Q.sum(1, keepdims=True), a
      alpha_max 0.4 keeps >= 0.54 mass on the given label: argmax(Q) == y always; labels are softened, never dropped or flipped.
    NESTED protocol (mandatory; global OOF would leak fold-k validation labels into fold-k training targets): for outer fold k: pool T_k = the other 4 folds; inner StratifiedKFold(inner_k=4, shuffle, seed 42+k) on T_k ->
      4 inner models (same cfg but epochs=inner_epochs, mixstyle/amp off, soft_labels none) via train_fold(cfg, ..., train_df=..., val_df=..., out=checkpoints/<tag>/inner/outer<k>_inner<j>.pth) -> inner-OOF P for every image of T_k ->
      build_soft_labels -> save softlabels/<tag>/outer<k>.npz (ids, P, Q, alpha). The outer model then trains on T_k with Q and is evaluated on fold k. Cost = 5 x (4 inner + 1 outer) trainings.
      assert that no id of outer fold k occurs in any inner training set.
    Soft labels are always evaluated against the ORIGINAL hard labels (val, test, external). Never evaluate on a "cleaned" test set. Training with them: --set soft_labels=aasl soft_label_run=<tag of make_softlabels>.
S7l. src/evaluation/experiment.py (one-shot experiment) and table builders (src/evaluation/tables.py)
    run_experiment(cfg): (1) train all folds (resume-aware) (2) merge OOF, thresholds, temperature (3) predict test with the 5-fold ensemble + cfg.tta (4) predict Messidor DEV and TEST (5) write per-experiment files (S5):
      summary.csv, per_fold.csv, per_class.csv, binary.csv, confusion_*.csv, calibration.csv, config.yaml, predictions_*.npz, oof.npz, curves/, figures/ (curves, cm_test, cm_external, roc, pr, reliability) (6) results/all_runs.npz via log_run(tag, kind, metrics) (7) print the summary row.
      gpu_hours = wall-clock of training x number of GPUs, written to summary.csv; n_params_M from the model.
    inference_experiment(--base tag): no training; owner = ckpt_owner of the base (follow ckpt_from until a tag that owns fold checkpoints); writes checkpoints/<tag>/config_snapshot.yaml with ckpt_from: <owner> and the base config plus the --set changes
      (tta / tta_scales / temperature_scaling / recalibrate / conformal_alpha); writes the same result files under its own results/<phase>/<tag>/ (oof.npz copied from the owner).
    ensemble_experiment(--members a,b,c): members may be @phase references; averages the members' saved probabilities (test, external dev/test, oof where available); writes the same files; winner_eligible=false.
    feature_balance(--base, --method none|reweight|smote|gauss|lfsd): S7n; writes the same files with ckpt_from.
    import_zoo(--csv baseline_result.csv --phase phase1_zoo): columns Model,Best_Val_Acc,Final_Train_Acc,Accuracy,QWK,Precision,Recall,F1_Score,Bin_Acc,Bin_Prec,Bin_Rec,Bin_F1,Ext_Acc,Ext_QWK,Ext_Bin_Acc as "mean ± std" strings
      -> results/phase1_zoo/<MODEL>/summary.csv (cv_val_acc, train_acc, test_acc, test_qwk, test_prec_macro, test_rec_macro, test_f1_macro, test_bin_acc, test_bin_prec, test_bin_sens, test_bin_f1, ext_acc, ext_qwk, ext_bin_acc and *_std columns; other columns empty),
      protocol='single_fold_mean', ext_set='full1744' and results/phase1_zoo/zoo_table.csv.
    phase_select(--phase P --reference R --design parallel): experiment folders are located with find_exp(tag) (so a reference from another phase works); reads only oof.npz + predictions_external_dev.npz + summary columns cv_*/oof_*/extdev_* of every experiment of the phase and of the reference;
      for each experiment: paired bootstrap on pooled OOF for QWK and referable AUROC, paired bootstrap on DEV for QWK and Mild recall, Holm across the phase; decision by S1 (TENTATIVE only for phase2a);
      winner = best OOF QWK among {REFERENCE + ADOPT + TENTATIVE experiments that have winner_eligible=true} (ties: referable AUROC); phase6_domain_gen uses DEV QWK first, then OOF QWK as a guard. Writes phase_table.csv, phase_table.tex, WINNER.txt (tag on the first line, then a one-line reason),
      and sets decision/is_winner in each summary.csv. The function raises if it is asked to read any *test* predictions. The reference may be a tag from an earlier phase; then WINNER.txt can name that tag.
    master_table: scan results/phase*/*/summary.csv (+ phase_table decisions); write master/master_table.csv with columns
      phase, experiment, description, protocol, ext_set, oof_qwk, test_acc, test_qwk, test_f1_macro, test_adj_acc, test_bin_acc, test_bin_auroc, test_sens_at_spec90, ext_acc, ext_qwk, ext_mild_recall, ext_pred_nodr_share, ext_bin_acc, ext_bin_auroc,
      ece_test, gpu_hours, decision, is_winner;  plus master_full.csv (all summary columns), master_per_class.csv, master_binary.csv, master_config.csv (experiment, set_overrides, inherit, config.yaml as flat key/value), master_deltas.csv (all phase_table delta columns),
      master_table.tex (booktabs, best bold per column, second underlined). Phase order = the order of the phase folders; zoo rows first.
    seed_summary(--phase phase9_final): mean +- std over seeds for final and baseline -> master/seed_variance.csv.
S7m. LEGACY SEED ADAPTATION (files in legacy_seed/, keep the same relative paths; names verified against the files)
    src/preprocessing/lesion_detection.py: class LesionDetector (detect_microaneurysms / detect_hemorrhages / detect_exudates / detect_all) and detect_lesions_batch(image_dirs, output_dir, mask_size=128, extensions). Use as is.
      `python main.py --mode detect_lesions` calls detect_lesions_batch([paths.eyepacs.images], paths.eyepacs.lesion_labels, 128) (skips existing .npz). npz keys: mask[128,128,3], count[3], presence[3], heatmap[128,128].
    src/data/eyepacs_dataset.py: EyePACSDataset(image_dirs, lesion_label_dir, img_size=512, mask_size=128, extensions=('.jpeg', '.jpg', '.png')) returns two augmented views + lesion pseudo-labels (crop + Ben Graham sigma 10, native resolution). Use as is for SSL.
    src/models/ssl_model.py: classes ProjectionHead, LesionTypeHead, LesionCountHead, LesionDecoder, SSLModel (has save_backbone), functions info_nce_loss, SSLCombinedLoss. REWRITE only the backbone part of SSLModel for MaxViT-L:
      timm.create_model(name, pretrained=True, num_classes=0, drop_path_rate=0.1) (the same class and key names as GradeNet.backbone); get the stage-3 and stage-4 NCHW feature maps by running backbone.stem / backbone.stages manually
      (stage index 2 and 3) instead of timm features_only (features_only renames the keys and uses NHWC permutes for Swin); read the channel sizes from a dummy forward and pass them to the heads (defaults in the file are the SwinV2 values 768/1536);
      save_backbone() must save backbone.state_dict() with the num_classes=0 key names so GradeNet(pretrain_path=...) loads it.
    src/training/ssl_trainer.py: keep. Every SSL variant needs its own ssl_save_dir (default checkpoints/pretrain/ssl_maxvit): training resumes from <ssl_save_dir>/ssl_training_checkpoint.pth and prints "Already completed" when finished.
      configs/config_ssl.yaml: copy the old file, change ssl_backbone to cfg.backbone, ssl_save_dir to checkpoints/pretrain/ssl_maxvit, ssl_batch_size to what fits; main.py --mode pretrain --kind ssl merges config_ssl.yaml into cfg and then re-applies the CLI --set overrides.
      Replace torch.cuda.amp.GradScaler/autocast by torch.amp equivalents if the installed torch warns.
    If a seed file is missing, re-implement with the same output contracts.
    Supervised pretraining (src/training/pretrain.py): --kind eyepacs_sup: EyePACS frame (dedup-excluded), 2% held out for monitoring, GradeNet(cfg) from ImageNet weights, sqrt-inverse class sampler, loss soft_ce with ordinal smoothing, LLRD, EMA, pre_epochs, cfg.preproc and cfg.aug;
      save only model.backbone.state_dict() to checkpoints/pretrain/eyepacs_sup.pth. --kind ssl_sup: same but initialised with pretrain_path = cfg.ssl_init (the SSL backbone), saves checkpoints/pretrain/ssl_sup.pth.
S7n. src/training/feature_balance.py (optional stretch)
    Extract penultimate features (backbone output) of the OOF-validation images with each fold model (not of its training images, which are over-separable); train a linear/MLP head on real + synthetic minority features with methods:
    none | reweight (logit adjustment: add tau*log prior to the logits in training, plain logits at test) | smote (same-class feature interpolation) | gauss (class-conditional Gaussian, shrinkage covariance) |
    lfsd (class-conditional 1-D DDPM, MLP denoiser, T=200, epsilon prediction). Evaluate with the same experiment outputs. Keep lfsd only if it beats the best cheaper method with a CI excluding 0.

S7o. src/xai: cams.py, grid.py (XAI grid figure and evaluation)
    - Config keys: xai_seed (default 0), xai_top_k (default 10), xai_methods (default [gradcampp, gradcam, layercam, eigencam]), xai_show_lesions (default true), xai_split (test | external_test), xai_alpha (default 0.5), xai_quant (default false), xai_quant_max_images (default 400).
    - Selection rule: candidates where y_true == c AND y_pred == c, ranked descending by ensemble probability y_prob[:, c]; pool = top xai_top_k; pick ONE with np.random.default_rng(xai_seed + c). Fallback if class has 0 correct: take highest probability image of class c, marked misclassified. Writes results/<phase>/<tag>/xai/xai_selection.csv.
    - Geometry: load_display_and_input(row, cfg) ensures display image matches model input geometry (legacy: cv2.resize stretch to 512x512; scaled: pad_square_resize) without CLAHE/Ben Graham. fundus_mask = grayscale > 10 + morphological closing + largest connected component.
    - ProbWrapper: returns logits for softmax and log(probs_from_logits) for CORAL; CAM target = score of true class (or predicted class for fallback).
    - CAM formulas (native PyTorch, hooks on backbone.stages[-1] and stages[-2]):
      - gradcam: w_k = mean_ij g_k(i,j); cam = ReLU(sum_k w_k * A_k) on stages[-1]
      - gradcampp: alpha_k(i,j) = g^2 / (2*g^2 + sum_ab A_k(a,b)*g^3 + 1e-7); w_k = sum_ij alpha_k * ReLU(g_k); cam = ReLU(sum_k w_k * A_k) on stages[-1]
      - layercam: cam = ReLU(sum_k ReLU(g_k) * A_k) on stages[-2] (32x32) AND stages[-1] (16x16), each normalised and upsampled to 512x512, then averaged
      - eigencam: 1st principal component of channel-centered activations via SVD, sign aligned with mean activation, cam = ReLU(projection)
    - Target layers in MaxViT-L: stages[-2] resolution 32x32 (512 channels), stages[-1] resolution 16x16 (1024 channels).
    - Post-processing: ReLU -> min-max to [0,1] -> bilinear upsample to 512x512 -> multiply by fundus_mask -> min-max. Per image: compute map with each fold model (loaded ONE AT A TIME for memory efficiency) and average normalized maps.
    - Layout: 5 rows (classes 0..4) x columns [Original | Grad-CAM++ | Grad-CAM | LayerCAM | Eigen-CAM | Detected lesions]. Width 7.2 in, square cells, 300 dpi, bold text, shared horizontal colorbar 'Normalised attention (0-1)'.
    - Outputs: results/<phase>/<tag>/xai/fig_xai_grid.{png,pdf}, copied to results/figures/fig16_xai_grid.{png,pdf}, xai_selection.csv, xai_caption_notes.txt, single overlays class<c>_<id>.png, optional xai_metrics.csv.

=====================================================================
END OF PART 1. CONTINUE WITH PART 2 (phases, commands, tables, figures, tests, work packages, risks).
=====================================================================

## CHANGELOG (agent)
- **Date**: 2026-10-22
- **What changed**:
  - S3: Recorded tested local environment versions (Python 3.9.6, PyTorch 2.8.0, timm 1.0.30, albumentations 2.0.8, etc.), marked CUDA/cuDNN as UNTESTED - recommended, added `tools/env_report.py`.
  - S5: Added XAI outputs under `results/<phase>/<tag>/xai/` and redefined paper figure `fig16` as `fig16_xai_grid.{png,pdf}`.
  - S6: Added `src/xai/{cams.py, grid.py}`, `tools/env_report.py`, `.gitignore`, `.gitkeep` placeholders across empty folders, `datasets/README.md`, and tracking for `agent_context/`.
  - S7a: Added `xai_*` default keys to `configs/config.yaml`.
  - S7b: Added `xai` mode to CLI dispatch in `main.py`.
  - S7o: Created subsection specifying XAI CAM formulas, image selection rules, geometry parity, target layer resolutions (MaxViT stage-2 32x32, stage-3 16x16), and artifact outputs.
- **Why**: Completed WP10-WP13 execution according to `agent_context/MODAL_PART3.md` (Step 1-6).