# Datasets Directory

This directory contains the clinical datasets used by MODAL-Net. Images and dataset CSVs are **never committed
to git**; on the cluster the folders are copied or symlinked here. All paths can be changed in
`configs/config.yaml -> paths`.

```text
datasets/
├── APTOS_19/
│   ├── train_images/  val_images/  test_images/      # <id_code>.png
│   ├── train_1.csv  valid.csv                         # id_code, diagnosis  (CV pool: 3,296 images)
│   └── test.csv                                       # id_code, diagnosis  (366 images, held-out test)
├── Messidor_2/
│   ├── images/
│   └── messidor_data.csv                              # id_code, diagnosis, adjudicated_dme, adjudicated_gradable (gradable: 1,744)
├── EyePACS/
│   ├── EyePACS/                                       # *.jpeg (88,702; 35,126 labelled)
│   ├── trainLabels.csv/trainLabels.csv                # a FOLDER named trainLabels.csv holding the csv: image, level
│   └── lesion_labels/                                 # *.npz written by --mode detect_lesions
├── DDR-dataset/
│   ├── DR_grading/
│   │   ├── train/ valid/ test/                        # *.jpg
│   │   └── train.txt valid.txt test.txt               # "<file> <grade>", grade 5 = ungradable (dropped by the code)
│   └── lesion_segmentation/                           # expert lesion masks (XAI localisation only)
│       ├── images/{train,val,test}/*.jpg
│       └── annotations/{train,val,tet}/{EX,HE,MA,SE}/*.tif     # the official test folder is spelled "tet"
├── DeepDRiD/
│   └── regular_fundus_images/                         # ultra-widefield images are not used
│       ├── regular-fundus-training/   Images/<patient>/<image_id>.jpg + regular-fundus-training.csv
│       ├── regular-fundus-validation/ Images/<patient>/<image_id>.jpg + regular-fundus-validation.csv
│       └── Online-Challenge1_2-Evaluation/ Images/<patient>/<image_id>.jpg + Challenge1_labels.xlsx
└── IDRiD/
    ├── B. Disease Grading/
    │   ├── 1. Original Images/{a. Training Set, b. Testing Set}/IDRiD_XXX.jpg
    │   └── 2. Groundtruths/{a. IDRiD_Disease Grading_Training Labels.csv, b. IDRiD_Disease Grading_Testing Labels.csv}
    └── A. Segmentation/                               # expert lesion masks (XAI localisation only)
        ├── 1. Original Images/{a. Training Set, b. Testing Set}/IDRiD_XX.jpg
        └── 2. All Segmentation Groundtruths/{a. Training Set, b. Testing Set}/{1. Microaneurysms, 2. Haemorrhages, 3. Hard Exudates, 4. Soft Exudates}/IDRiD_XX_<MA|HE|EX|SE>.tif
```

Notes
- DeepDRiD: the Evaluation folder is called `Online-Challenge1_2-Evaluation` (renamed from `...1&2...`, which breaks
  shell commands). Labels are per IMAGE from the eye-level columns; splits are patient-disjoint.
- IDRiD: training and testing folders reuse the same file names; the code prefixes ids with `tr_` / `te_`.
- Splits used for evaluation are frozen by `python main.py --mode make_external_splits` into `splits/external/`
  (official splits; evaluation images that duplicate a training-pool image are removed and listed in
  `splits/external/exclusions.csv`).
- Cite the original dataset papers (verify the DDR citation from its DOI: the Hugging Face mirror README gives a
  different author list than the original release).
