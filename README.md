# MODAL-Net

**Multi-source, Ordinal, Domain-generalizable, Ambiguity-aware, Lesion-aware Network**

Backbone: MaxViT-L for Diabetic Retinopathy Grading.

Paper Title: *"MODAL-Net: Multi-Source Lesion-Aware Pretraining with Ambiguity-Aware Ordinal Learning for Domain-Generalizable Diabetic Retinopathy Grading"*

---

## 1. Environment Setup

Please follow the detailed setup instructions in the header of [`requirements.txt`](requirements.txt):
1. Create and activate a virtual environment (`python -m venv .venv; source .venv/bin/activate`).
2. Install the PyTorch/TorchVision pair matching your server driver/CUDA build.
3. Install dependencies: `pip install -r requirements.txt`.
4. Verify environment and backbones: `python tools/env_report.py`.
5. Run fast test suite: `pytest -q`.

> **Offline Cluster Rule:** Set `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and ensure model weights are pre-cached.

---

## 2. Running Experiments

Every experiment runs via `main.py` with 5-fold cross-validation, evaluation on APTOS Test and Messidor-2 DEV and TEST sets, and automated artifact generation:

```bash
# Standard training experiment
python main.py --mode experiment --phase phase2a_recipe --tag my_recipe_exp

# Override configuration parameters directly via CLI
python main.py --mode experiment --phase phase2a_recipe --tag my_recipe_exp --set lr=5.0e-5 batch_size=8

# Inference-only evaluation with existing fold checkpoints
python main.py --mode inference_experiment --phase phase8_inference --tag my_inf_exp --base <checkpoint_tag> --set tta=five_crop

# Conformal prediction calibration
python main.py --mode calibrate_conformal --phase phase8_inference --tag my_conformal_exp --base <checkpoint_tag> --set conformal_alpha=0.10

# Master table and paper figures generation
python main.py --mode report
```

---

## 3. Explainability (XAI Grid)

To generate the 5-row publication XAI grid comparing Grad-CAM++, Grad-CAM, LayerCAM, Eigen-CAM, and classical lesion detector contours:

```bash
python main.py --mode xai --phase phase9_final --tag winner --set xai_top_k=1 xai_seed=0 xai_show_lesions=true
```

Output artifacts are saved to `results/<phase>/<tag>/xai/`:
- `fig_xai_grid.png` and `fig_xai_grid.pdf` (also copied to `results/figures/fig16_xai_grid.{png,pdf}`)
- `xai_selection.csv` (selected images and confidence scores)
- `xai_caption_notes.txt` (detailed figure caption and interpretation disclaimers)
- `xai/<method>/class<c>_<id>.png` (individual CAM heatmaps overlaid on native fundus images)

---

## 4. Plan and Specifications

Comprehensive specifications, clinical priors, ablation tables, mathematical formulations, and engineering work packages are documented in:
- [`agent_context/MODAL_PART1.md`](agent_context/MODAL_PART1.md): Identity, data splits, repository layout, module specifications.
- [`agent_context/MODAL_PART2.md`](agent_context/MODAL_PART2.md): 10-phase roadmap, execution commands, paper tables, figures, test suite.
