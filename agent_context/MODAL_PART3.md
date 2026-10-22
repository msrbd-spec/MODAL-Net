Context: MODAL-Net is built and audited (WP0-WP9). The plan files are in agent_context/ at the repository root:
agent_context/MODAL_PART1.md, agent_context/MODAL_PART2.md, agent_context/legacy_seed/, agent_context/baseline_result.csv. Keep following the rules of the setup prompt (no GitHub push, no downloads,
never use test / Messidor TEST for selection, 2-decimal numbers, flag every assumption). Do the five steps below IN THIS ORDER. After each step run its checks, then continue. At the end send one report (format in step 6).

STEP 1 - BOLD FIGURE STYLE (affects every diagram, old and new)
All text in every figure must be bold: titles, axis labels, tick labels, legends, legend titles, annotations (confusion-matrix numbers), colorbar labels, panel/row/column labels, table-like text inside figures.
1.1 In src/evaluation/figures.py add:
    def set_bold_style():
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.weight': 'bold', 'axes.labelweight': 'bold',
            'axes.titleweight': 'bold', 'figure.titleweight': 'bold', 'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
            'axes.linewidth': 1.2, 'lines.linewidth': 1.5, 'axes.grid': True, 'grid.alpha': 0.25, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})
        # print once which font family was actually resolved (Arial may be missing on the server; the fallback must be reported, not silent)
    def save_fig(fig, path_stem):          # the ONLY way figures are saved
        for t in fig.findobj(matplotlib.text.Text): t.set_fontweight('bold')          # enforce on legend, ticks, annotations, colorbars
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_text().strip(): assert weight_value(t.get_fontweight()) >= 700, (path_stem, t.get_text())
        fig.savefig(path_stem + '.png', dpi=300, bbox_inches='tight'); fig.savefig(path_stem + '.pdf', bbox_inches='tight')
    (weight_value maps strings such as 'bold' via matplotlib.font_manager.weight_dict to numbers.)
1.2 Refactor EVERY existing figure function (per-experiment: curves, cm_test, cm_external, roc, pr, reliability; paper figures fig01-fig18) to call set_bold_style() once and save only through save_fig().
    Seaborn/sklearn display objects: pass annotation kwargs with fontweight='bold' and still rely on save_fig for the final enforcement. Keep the colours, sizes and layouts of the plan (train blue '#1F77B4', val red '#D62728', class and domain palettes).
1.3 Test T12: build a small figure with legend, annotation, colorbar and ticks, run save_fig, assert every non-empty Text artist is bold; also run it on one real per-experiment figure of the smoke experiment.

STEP 2 - XAI GRID FIGURE (new work package WP10)
Goal: show how the final model attends to lesions. One figure: 5 rows (one test image per class) x columns [Original | Grad-CAM++ | Grad-CAM | LayerCAM | Eigen-CAM | Detected lesions].
2.1 Mode and config. New mode: python main.py --mode xai --phase phase9_final --tag <final experiment tag> [--set xai_seed=0 xai_top_k=10 xai_methods=[gradcampp,gradcam,layercam,eigencam] xai_show_lesions=true xai_split=test]
    Add to configs/config.yaml (defaults): xai_seed: 0, xai_top_k: 10, xai_methods: [gradcampp, gradcam, layercam, eigencam], xai_show_lesions: true, xai_split: test, xai_alpha: 0.5, xai_quant: false, xai_quant_max_images: 400.
    Extend validate_cfg (methods in the allowed set, xai_split in {test, external_test}). Models = the 5 fold checkpoints of ckpt_owner(cfg), loaded ONE AT A TIME. Model in eval(), fp32, gradients enabled, no autocast. XAI is for explanation only:
    nothing it produces may feed back into model selection.
2.2 Image selection (from results/<phase>/<tag>/predictions_test.npz with ids, y_true, y_pred, y_prob; classes 0..4 = No DR, Mild, Moderate, Severe, Proliferative):
    for each class c: candidates = images with y_true == c AND y_pred == c, ranked by y_prob[:, c] descending; pool = the first xai_top_k (xai_top_k=1 means strictly the most confident); pick ONE with np.random.default_rng(xai_seed + c).
    If a class has no correct image: take the highest-probability image of that class, mark it "misclassified (pred = X)" in the row label and in the csv. Write results/<phase>/<tag>/xai/xai_selection.csv
    (class, id, y_true, y_pred, confidence, rank_in_class, pool_size, seed, fallback_flag).
2.3 Geometry. def load_display_and_input(row, cfg) returns (display_rgb uint8 512x512, input_tensor 1x3x512x512, fundus_mask 512x512 bool):
    model input = exactly the eval pipeline (preprocess + eval transform of the experiment's cfg); display image = crop_fundus -> resize with the SAME geometry as the model input (preproc=legacy: plain cv2.resize stretch to 512x512;
    other modes: pad_square_resize), without CLAHE / Ben Graham, so overlays are spatially aligned and readable. fundus_mask = display grayscale > 10 (then morphological closing + largest component).
2.3b Output wrapper: class ProbWrapper(nn.Module) returns the logits for head_type softmax and log(probs_from_logits) for coral; the CAM target is the score of class c (the class of the row; for fallback rows the predicted class).
2.4 CAM methods, implemented natively in src/xai/cams.py with forward/backward hooks on a target layer (no extra dependency; MaxViT stage outputs are NCHW so no reshape transform is needed; verify shapes and tell me the resolution of
    stages[-1] and stages[-2]). A = activations (C,H,W), g = d score / d A.
      gradcam:    w_k = mean_ij g_k(i,j);  cam = ReLU(sum_k w_k * A_k)
      gradcampp:  alpha_k(i,j) = g^2 / (2*g^2 + sum_ab A_k(a,b) * g^3 + 1e-7);  w_k = sum_ij alpha_k(i,j) * ReLU(g_k(i,j));  cam = ReLU(sum_k w_k * A_k)
      layercam:   cam = ReLU(sum_k ReLU(g_k) * A_k) computed on stages[-2] AND stages[-1], each normalised and upsampled to 512x512, then averaged (finer localisation)
      eigencam:   gradient-free: A flattened to (HW, C), centred per channel, first principal component by SVD, sign fixed so that it correlates positively with the mean activation, cam = ReLU(projection)
    Target layers: gradcam, gradcampp, eigencam -> backbone.stages[-1]; layercam -> stages[-2] and stages[-1]. If the model was built with mixstyle/WaveletGate the hooks must still work (hooks go on backbone.stages, not on GradeNet.forward).
    Post-processing for every map: ReLU -> min-max to [0,1] -> bilinear upsample to 512x512 -> multiply by fundus_mask -> min-max again. Per image: compute the map with each of the 5 fold models and AVERAGE the normalised maps, then min-max.
    If pytorch-grad-cam happens to be installed, add an optional cross-check in the tests (correlation > 0.9 on a toy CNN); do not make it a dependency.
2.5 Figure layout (src/xai/grid.py, saved with save_fig -> results/<phase>/<tag>/xai/fig_xai_grid.{png,pdf} and copied to results/figures/fig16_xai_grid.{png,pdf}):
    columns: Original | Grad-CAM++ | Grad-CAM | LayerCAM | Eigen-CAM | Detected lesions (classical LesionDetector mask contours in white/yellow on the original; skip this column when xai_show_lesions=false). Methods follow xai_methods order.
    Column titles (bold) on the top row only; row labels (bold, rotated 90 degrees) on the left: class name plus second line "p = 0.97" (ensemble confidence). No per-image titles, no axes. Overlay = (1-a)*display + a*jet(cam) with a = xai_alpha, only inside the fundus mask.
    One shared horizontal colorbar below the CAM columns, bold label "Normalised attention (0-1)". Width 7.2 in, square cells, 300 dpi, Arial bold via set_bold_style, tight layout with small gaps.
2.6 Other outputs: results/<phase>/<tag>/xai/<method>/class<c>_<id>.png (single overlays), xai_selection.csv, and xai_caption_notes.txt containing: the selection rule (most confident correct predictions, top-k pool, seed), the sentence that the examples are illustrative and not
    representative, that CAM maps show where gradients/activations are large and are not evidence of causal reliance, that the detected-lesion mask comes from a classical heuristic detector (not ground truth), and the resolution of the target layers. Use these notes in the figure caption.
2.7 Tests T13-T15 (synthetic, no checkpoints): selection logic (only correct predictions, top-k pool, deterministic for a seed, fallback flagged); geometry (display and input tensor have the same crop/resize geometry for legacy and scaled modes);
    CAM formulas on a toy network where the answer is known (global-average-pool + linear head: gradcam weights equal the head weights); output shapes and value range [0,1]; the grid function runs on a random tiny model with fake data and the result passes the bold check.
2.8 OPTIONAL after the figure works and the tests pass (skip if it takes long; implement only when xai_quant=true): (a) lesion enrichment per class and method on correct test predictions (cap xai_quant_max_images): enrichment = (CAM mass inside the dilated detected-lesion mask / total CAM mass) / (mask area / fundus area),
    mean with bootstrap 95% CI -> results/<phase>/<tag>/xai/xai_metrics.csv (values above 1 mean attention is enriched on detected lesions; say clearly that the detector is heuristic); (b) a model-randomisation sanity check: re-initialise the head and the last stage, recompute maps, report the rank correlation with the original maps (it should drop clearly).

STEP 3 - UPDATE THE PLAN FILES (in place, in agent_context/)
3.1 Update agent_context/MODAL_PART1.md and agent_context/MODAL_PART2.md so that they describe the code as it now exists. Keep section numbers stable; add new subsections instead of renumbering. Do not delete earlier content except to correct a statement that no longer matches the code.
3.2 Part 1: add S7o (XAI: config keys, selection rule, geometry, CAM formulas, outputs), update S3 (environment/requirements, step 4), S5 (results/<phase>/<tag>/xai/, results/figures/fig16_xai_grid.*), S6 (new files: src/xai/{cams.py,grid.py}, tools/env_report.py, .gitignore, .gitkeep placeholders, agent_context/, datasets/README.md), S7a (xai_* keys).
    Part 2: S10 add "Phase 10 - XAI" with the command above (and an example with xai_top_k=1), S12 add the bold-text rule and redefine fig16 as the XAI grid, S13 add WP10 (bold style + XAI), WP11 (plan update), WP12 (requirements/environment), WP13 (git hygiene), S14 add tests T12-T16, S15 add risks
    (CAM resolution of the last stage, selection bias of high-confidence examples, fonts missing on the server).
3.3 Compare every command written in S10 with the CLI you actually implemented (flag names, defaults, modes). Fix the plan text wherever the code differs, and list each fix. If you believe the CODE is wrong rather than the plan, do not change either silently: report it.
3.4 Append a section "CHANGELOG (agent)" at the end of both files: date, what changed, why.

STEP 4 - REQUIREMENTS AND ENVIRONMENT (WP12)
4.1 Rewrite requirements.txt from the environment you actually used for testing. Run and record: python --version, pip freeze (direct dependencies only go into the file), torch.__version__, torchvision.__version__, torch.version.cuda, torch.backends.cudnn.version(), nvidia-smi if available, OS.
    Pin every direct dependency with == to the tested version (torch, torchvision, timm, numpy, pandas, scipy, scikit-learn, opencv-python or opencv-python-headless, albumentations, pyyaml, tqdm, matplotlib, pytest, Pillow if used, optional ones marked clearly). Do NOT invent versions.
    Anything you could not test (for example CUDA/cuDNN when the test machine has no GPU) must be written as "UNTESTED - recommended" in a comment, never as a fact.
4.2 Start requirements.txt with a comment header, for example:
      # MODAL-Net environment. Tested on: <OS>, Python <x.y.z>, torch <v>, torchvision <v>, CUDA <v or "no GPU in the test environment">, cuDNN <v or UNTESTED>.
      # Setup (no download links in any command):
      #   1) python -m venv .venv ; source .venv/bin/activate
      #   2) Install the torch/torchvision pair listed below with the CUDA build that matches the server driver, using whatever your cluster supports (module, conda channel, internal mirror or pre-downloaded wheels). Do not edit the pinned versions casually.
      #   3) pip install -r requirements.txt
      #   4) python tools/env_report.py     # prints python/torch/CUDA/cuDNN/timm/albumentations versions; compare with this header
      #   5) pytest -q                      # fast checks (splits, metrics, selection guard); run before long experiments
      # Offline cluster: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 and pre-populate the timm/HF cache with the MaxViT-L, ConvNeXtV2-L and EfficientNetV2-M weights (ids listed in agent_context/MODAL_PART1.md S3).
    No URLs, --index-url or --extra-index-url lines anywhere in the file or in any command you write. Add tools/env_report.py (prints versions, CUDA availability, device names, cudnn version, whether timm can create the backbone ids from the config without downloading).
4.3 Update README.md briefly: environment setup (point to the header of requirements.txt), how to run an experiment, the xai command, and where the plan files live.

STEP 5 - GIT HYGIENE (WP13)
5.1 Create .gitignore (comments in English, grouped). Ignore: __pycache__/, *.pyc, .pytest_cache/, .ipynb_checkpoints/, .venv/, venv/, env/, .env, .idea/, .vscode/, .DS_Store, __MACOSX/, *.egg-info/, build/, dist/, *.swp, nohup.out, slurm-*.out, wandb/, *.log,
    *.pth, *.pt, *.ckpt, *.safetensors, *.zip, *.tar, *.tar.gz, and the CONTENT (not the folders) of: datasets/ (images and csv), logs/, checkpoints/, softlabels/, results/.
    Keep tracked: source code, configs/, tools/, tests/, splits/*.json (small and needed for identical splits on the server), README.md, requirements.txt, .gitignore, datasets/README.md, agent_context/.
5.2 tests/: do NOT ignore it. It is small, runs in seconds and protects the selection guard and the splits on the server. Add a commented line "# tests/" with a note that the owner can uncomment it to exclude tests from the repository.
5.3 agent_context/ MUST be tracked: do not ignore it, and make sure no generic pattern (*.zip, *.csv, *.md, *.py, *.log, *.pt ...) can hit anything inside it (baseline_result.csv, MODAL_PART1.md, MODAL_PART2.md, legacy_seed/**).
    Add a final explicit rule  !agent_context/**  with a comment, and verify with git: git check-ignore -v agent_context/MODAL_PART1.md agent_context/MODAL_PART2.md agent_context/baseline_result.csv agent_context/legacy_seed/src/models/ssl_model.py   -> must print nothing.
5.4 Empty folders must survive git: add a .gitkeep file to logs/, checkpoints/, checkpoints/pretrain/, softlabels/, splits/, results/ and to each of results/phase0_protocol, phase1_zoo, phase2a_recipe, phase2b_preprocess, phase3a_pretrain, phase3b_joint,
    phase4_label_loss, phase5_architecture, phase6_domain_gen, phase7_feature_balance, phase8_inference, phase9_final, master, tables, figures, xai. Make .gitignore keep these placeholders while ignoring everything else inside, for example:
      results/**
      !results/**/
      !results/**/.gitkeep
      (same pattern for logs, checkpoints, softlabels). Verify the pattern works instead of assuming it: in a TEMPORARY copy of the repo run git init, git add -A, git status --short and git ls-files; confirm that the .gitkeep files are tracked,
      that a dummy results/phase2a_recipe/x/summary.csv, checkpoints/t/fold0.pth, logs/t/a.log and datasets/APTOS_19/a.png are ignored, and that agent_context files are tracked. Do not run git init inside the real repository if it has no .git already.
5.5 Datasets: do not add placeholder subfolders for the images (a symlinked datasets/ folder on the server would conflict). Add datasets/README.md with the exact expected layout (S4 of Part 1) and the sentence that images are never committed.
    Add optional commented lines to .gitignore showing how to also track the master tables, for example  # !results/master/*.csv

STEP 6 - FINAL CHECKS AND REPORT
Run: ast syntax check on all changed files, pytest -q (all tests including T12-T16), python tools/env_report.py, one smoke experiment (epochs=1, n_folds=2, tiny batch) followed by python main.py --mode xai on it (use xai_top_k=1; if no checkpoints can be produced in your
environment, say so and rely on the synthetic tests), and the git checks of step 5. Reply with: (1) every file created or changed, (2) test and check results (include the font that was actually used for figures), (3) the exact versions written to requirements.txt and which of them were UNTESTED,
(4) differences you found between plan and code and what you changed in the plan files, (5) open questions. Do not start any other work.