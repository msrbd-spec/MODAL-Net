# Datasets Directory

This directory contains the clinical datasets used by MODAL-Net for diabetic retinopathy grading.

> **CRITICAL RULE**: Fundus image files and dataset CSVs are **NEVER committed to git**. On cluster nodes or local machines, populate or symlink this directory with existing pre-downloaded datasets.

---

## Expected Directory Layout

Follow the exact layout specified in `agent_context/MODAL_PART1.md` Section S4:

```text
datasets/
├── APTOS_19/
│   ├── train_images/          # Training fundus images (<id_code>.png/.jpg)
│   ├── val_images/            # Validation fundus images (<id_code>.png/.jpg)
│   ├── test_images/           # Test fundus images (<id_code>.png/.jpg)
│   ├── train_1.csv            # Columns: id_code, diagnosis
│   ├── valid.csv              # Columns: id_code, diagnosis
│   └── test.csv               # Columns: id_code, diagnosis (366 rows; strictly locked)
│
├── Messidor_2/
│   ├── images/                # External test images
│   └── messidor_data.csv      # Columns: id_code, diagnosis, adjudicated_dme, adjudicated_gradable
│                              # Filter: adjudicated_gradable == 1 (n = 1,744)
│
└── EyePACS/
    ├── EyePACS/               # Raw EyePACS training images (*.jpeg)
    ├── trainLabels.csv/       # Subdirectory containing:
    │   └── trainLabels.csv    # Columns: image, level
    └── lesion_labels/         # Precomputed classical lesion masks (*.npz from detect_lesions)
```

---

## Split Protocol

- **APTOS-2019**: Merged `train_1.csv` + `valid.csv` (3,296 images) evaluated with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. `test.csv` (366 images) is fixed and NEVER used for model selection or early stopping.
- **Messidor-2**: Split by patient/exam group (first two tokens of `id_code`) using `StratifiedGroupKFold(5, shuffle=True, random_state=42)`. Fold 0 is DEV (~20%, for cut-point recalibration and hyperparameter decisions); folds 1-4 are TEST (~80%, for external generalizability evaluation).
