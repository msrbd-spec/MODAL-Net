# MODAL-Net

**Multi-source, Ordinal, Domain-generalizable, Ambiguity-aware, Lesion-aware Network** — diabetic retinopathy
grading (5-class ICDR + referable DR), backbone MaxViT-L (timm `maxvit_large_tf_512.in1k`).

Where to start:
- **`SERVER_TODO.md`** — what to run next on the cluster, in order (mandatory / optional, GPU-hour estimates).
- **`commands.md`** — every command of the project (all phases, evaluation, selection, report, XAI, troubleshooting).
- **`SERVER_CLEANUP.md`** — what can be deleted on the server and on the Mac, with copy-paste commands.
- **`RESULTS_STATUS.md`** — the result ledger up to 2026-11-03.
- **`agent_context/MODAL_PART1.md` / `MODAL_PART2.md` / `MODAL_PART3.md`** — the specification (Part 1 S0b summarises the 2026-11-03 revision).

---

## 1. Environment

Follow the header of [`requirements.txt`](requirements.txt): create a virtual environment, install the torch / torchvision
pair for your CUDA build, `pip install -r requirements.txt`, then `python tools/env_report.py`.
Offline cluster: `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (weights must already be in the cache).

```bash
pytest -q        # 140 tests, ~5-15 min on CPU; every test runs in its own temporary folder
```

## 2. Data

`datasets/README.md` describes the expected layout of APTOS-2019, Messidor-2, EyePACS, DDR, DeepDRiD and IDRiD.
The external evaluation splits are frozen once with `python main.py --mode dedup_external` and
`python main.py --mode make_external_splits` (see `commands.md` §3).

## 3. Running

Every experiment is one command. It trains 5 folds and evaluates the pooled APTOS OOF predictions, the APTOS test set
and the dev / test split of Messidor-2, DDR, DeepDRiD and IDRiD with the same 5-class, referable and calibration metrics:

```bash
python main.py --mode experiment --phase phase4j_label_loss --tag p4j_coralsrc --inherit p3b_joint \
    --set loss=coral head_type=coral_src src_offsets=true
python main.py --mode evaluate --tag p3b_joint          # re-evaluate a finished run on every set (no retraining)
python main.py --mode evaluate_all                      # every finished run
python main.py --mode phase_select --phase phase4j_label_loss --reference p3b_joint
python main.py --mode report --set report_final=p4j_coralsrc report_baseline=p2a_plain
```

`--set` accepts only keys that exist in `configs/config.yaml` / `configs/config_ssl.yaml`.

## 4. Explainability

```bash
python main.py --mode xai --tag p4j_coralsrc --set xai_split=messidor_test xai_top_k=1   # 5-row CAM grid
python main.py --mode xai_localization --tag p4j_coralsrc                               # CAM vs expert lesion masks (DDR, IDRiD)
```
Outputs: `results/<phase>/<tag>/xai/` (grid, selection, caption notes, `xai_localization.csv`) and `results/figures/xai_grid.{png,pdf}`.

## 5. Integrity rules built into the code

- Selection never reads TEST predictions (selection guard); TEST splits are for reporting only.
- Paper tables and figures are built from real artefacts only; missing inputs are listed in
  `results/tables/TABLES_README.txt` and `results/figures/FIGURES_README.txt`, never filled in.
- Evaluation images that duplicate a training-pool image are removed from the external splits (`splits/external/exclusions.csv`);
  `tools/leakage_check.py` reports the remaining overlap per run (the APTOS test set is never filtered).
