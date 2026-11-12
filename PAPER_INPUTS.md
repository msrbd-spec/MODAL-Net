# PAPER INPUTS — files Claude needs to write the paper

Only result files are shared: CSV, JSON, TXT/YAML/TEX text files and figures (PNG/PDF). No backups, no
checkpoints, no `.npz`, no features. Run section 1 **after H3** (tables and figures exist only after H3).
All paths are relative to the repo root on the server; inside the zip they keep the same paths.

## 1. Collect + zip (server, repo root, ≈ 1 min)

Copy the whole block into the server terminal. Every file that is not found is printed as a WARNING with its
name and location, and the same list is saved in `paper_inputs/MISSING.txt` (also inside the zip).

```bash
cd /home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR
rm -rf paper_inputs paper_inputs.zip && mkdir -p paper_inputs && : > paper_inputs/MISSING.txt
take() {   # take <path-or-glob> ... : copy with the same relative path, warn if absent
  for p in "$@"; do
    found=0
    for f in $p; do [ -e "$f" ] && { cp --parents -r "$f" paper_inputs/; found=1; }; done
    [ $found = 1 ] || { echo "WARNING: not found: $(basename "$p")  (expected at: $p)"; echo "$p" >> paper_inputs/MISSING.txt; }
  done
}

# A. data, splits, duplicate audit
take results/phase0_protocol/splits_summary.csv splits/external/manifest.json splits/external/exclusions.csv \
     results/master/leakage_check.csv

# B. all runs x all sets x all metrics, seeds
take results/master/master_full.csv results/master/master_table.csv results/master/seed_variance.csv \
     results/master/seed_compare.csv results/master/evaluate_all_report.csv

# C. selection decision of every phase (pre-specified rule + amendment)
for P in phase2a_recipe phase2b_preprocess phase3a_pretrain phase3b_joint phase4_label_loss phase4j_label_loss \
         phase5_architecture phase6_domain_gen phase10_ms_ddr phase10_ms_deepdrid; do
  take results/$P/phase_table.csv results/$P/WINNER.txt
done
take "results/phase*/phase_table_amended.csv" "results/phase*/phase_table_original.csv"

# D. cut-point diagnostic + E4 adaptation curve (+ seed re-run)
take results/master/cutpoint_diagnostic.csv results/master/adaptation_compare.csv results/master/adaptation_curve.csv \
     results/master/adaptation_lambda.csv results/master/adaptation_seeds/adaptation_compare.csv \
     results/master/adaptation_seeds/adaptation_curve.csv results/master/adaptation_seeds/adaptation_lambda.csv

# E. step R: recalibration suite (main / seeds / backbones / domain generalisation), shift decomposition, few-label head
for D in results/master/recal results/master/recal_seeds results/master/recal_backbones results/master/recal_dg; do
  take $D/recal_summary.csv $D/recal_compare.csv $D/recal_fitted.csv $D/recal_notes.csv \
       $D/prevalence_estimates.csv $D/binary_operating_points.csv $D/ordinal_discrimination.csv \
       $D/indist_control.csv $D/aptos_dup_sensitivity.csv
done
take results/master/recal/shift_decomposition.csv results/master/recal/shift_decomposition_contrasts.csv \
     results/master/recal/shift_decomposition_notes.csv results/master/recal/fewshot_head_draws.csv \
     results/master/recal/fewshot_head_lambda.csv

# F. headline and comparison runs: per-run metrics + training configuration
for R in phase3b_joint/p3b_joint phase2a_recipe/p2a_plain phase2b_preprocess/p2b_none \
         phase4j_label_loss/p4j_coralsrc_shared phase10_ms_ddr/p10d_ce phase10_ms_deepdrid/p10q_ce \
         phase5_architecture/p5_retfound phase5_architecture/p5_retfound_joint \
         phase5_architecture/p5_convnextv2 phase5_architecture/p5_effnetv2 \
         phase6_domain_gen/p6_amp phase6_domain_gen/p6_ms phase6_domain_gen/p6_amp_ms phase6_domain_gen/p6_amp_nojoint; do
  take results/$R/summary.csv results/$R/metrics_long.csv results/$R/per_class.csv results/$R/binary.csv \
       results/$R/calibration.csv "results/$R/confusion_*.csv" results/$R/cutpoints.csv results/$R/fewshot.csv
  take checkpoints/$(basename $R)/config_snapshot.yaml
done
for R in phase3b_joint/p3b_joint phase10_ms_ddr/p10d_ce phase10_ms_deepdrid/p10q_ce; do   # step R1 outputs
  take results/$R/target_adapt/bn/summary.csv results/$R/target_adapt/tent/summary.csv \
       results/$R/target_adapt/bn/adapt_log.csv results/$R/target_adapt/tent/adapt_log.csv \
       results/$R/reproduction_check.csv
done
for R in phase3b_joint/p3b_joint phase4j_label_loss/p4j_coralsrc_shared phase2a_recipe/p2a_plain; do   # XAI
  take results/$R/xai/xai_localization.csv
done

# G. inference products (TTA, conformal) of FINAL and of the headline CE model
take "results/phase8_inference/p8_tta_flip/*.csv" "results/phase8_inference/p8_conformal/*.csv" \
     "results/phase8_inference/p8_tta_flip_ce/*.csv" "results/phase8_inference/p8_conformal_ce/*.csv"

# H. paper tables and figures (built by H3), incl. tool figures and the two READMEs
take "results/tables/*.csv" "results/tables/*.tex" results/tables/TABLES_README.txt \
     "results/figures/*.png" "results/figures/*.pdf" results/figures/FIGURES_README.txt

n=$(find paper_inputs -type f ! -name MISSING.txt | wc -l); m=$(wc -l < paper_inputs/MISSING.txt)
env/bin/python -m zipfile -c paper_inputs.zip paper_inputs/
echo "collected $n files, $m missing -> $(du -h paper_inputs.zip | cut -f1)  $(pwd)/paper_inputs.zip"
```

## 2. Server → Windows PC (WinSCP) → Mac → GitHub

1. WinSCP: connect to the server as usual, open the folder
   `/home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR`, drag `paper_inputs.zip` to a folder on the PC.
2. Move the zip to the Mac (any way you normally use), put it in the repo root, then on the Mac:
   ```bash
   cd /Users/meheraj/Documents/MODAL-Net
   unzip -o paper_inputs.zip && rm paper_inputs.zip
   git add paper_inputs && git commit -m "paper inputs" && git push
   ```
   (`paper_inputs/` is not git-ignored; `results/` is. A single file above 100 MB is rejected by GitHub —
   none of these files should be near that; if the zip is larger than ≈ 100 MB, tell Claude first.)
3. Tell Claude "paper inputs pushed" and paste the `MISSING.txt` content (or "none").

### Backup plan if the zip step fails

- **Plan B (tar instead of zip)** — on the server: `tar czf paper_inputs.tgz paper_inputs`. Download
  `paper_inputs.tgz` with WinSCP the same way; on the Mac use `tar xzf paper_inputs.tgz` instead of `unzip`.
- **Plan C (no archive at all)** — WinSCP can copy whole folders: in the right-hand (server) panel open
  `/home/sr5/imam.bokhary/project/LTA/Prepare_Data/envs/DR`, drag the folder `paper_inputs` (it was already
  filled by section 1 even if the zip failed) to the left-hand (PC) panel. WinSCP keeps the sub-folders. If a
  transfer stops half-way, drag the same folder again and choose "Overwrite" or "Newer only".
- **Plan D (one folder at a time)** — if a whole-folder copy keeps failing, copy these folders one by one from
  `paper_inputs/` with WinSCP, in this order (smallest first): `paper_inputs/results/master`,
  `paper_inputs/results/phase*` (each phase folder), `paper_inputs/results/tables`, `paper_inputs/results/figures`,
  `paper_inputs/checkpoints`, `paper_inputs/splits`, then the loose files (`SERVER_TODO.md`, ...,
  `MISSING.txt`). Keep the same folder structure on the PC (create the folders with the same names).
- Whatever plan you use, the Mac must end with a folder `MODAL-Net/paper_inputs/` that has the same sub-folders
  as on the server; then run the `git add paper_inputs ...` lines above.
- **Do NOT add the Mac's own `results/` folder to git** (`git status` may show it as untracked — that is the
  local copy; only `paper_inputs/` is shared).

## 3. What each part is used for

| Part | Paper section |
|---|---|
| A | Datasets, splits, duplicate audit, leakage statement |
| B | Main results (5-class + referable, internal + 4 external sets, CIs), seed variability |
| C | Training-side experiments, pre-specified selection rule and its documented amendment |
| D | Cut-point shift diagnostic, few-label adaptation curve, shrinkage λ vs n |
| E | Main contribution: annotation-free / few-label recalibration vs baselines, clinical operating points, shift decomposition, in-distribution control, APTOS near-duplicate sensitivity, seeds, backbones, and whether domain-generalisation training removes the cut-point shift (recal_dg) |
| F | Per-class / binary / calibration details, confusion matrices, BN-adapt / TENT, XAI localisation, exact training configuration |
| G | Conformal prediction and test-time augmentation |
| H | The final paper tables and figures |

## 4. If a file is reported missing — what to do

Find the folder of the missing path below, run the step that writes it (`SERVER_TODO.md`), then run section 1 again.

| Missing path starts with | Written by (SERVER_TODO step) | Action |
|---|---|---|
| `results/master/master_*`, `seed_*` | daily-report refresh / H3 | `env/bin/python main.py --mode seed_summary --reference p3b_joint && env/bin/python main.py --mode master_table` |
| `results/master/leakage_check.csv` | C3 / R3 | first line of R3 (`tools/leakage_check.py ...`) |
| `results/master/cutpoint_diagnostic.csv` | C5 | C5 command |
| `results/master/adaptation_*`, `adaptation_seeds/` | E4 (+ re-run after R0) | E4 command(s) |
| `results/master/recal/recal_*`, `prevalence_*`, `binary_*`, `ordinal_*`, `indist_*`, `aptos_dup_*` | R3 | R3 commands |
| `results/master/recal/shift_decomposition*` | R4 | R4 command |
| `results/master/recal/fewshot_head_*` | R2 | R2 command (needs the R1 `extract_features` jobs) |
| `results/master/recal_seeds/`, `recal_backbones/` | R5 | R5 commands |
| `results/master/recal_dg/`, `results/phase6_domain_gen/...` | G4 (TODO "TOMORROW" Step 1 + Step 3) | the G4 job of that run, then Step 3 |
| `results/<phase>/<tag>/target_adapt/`, `reproduction_check.csv` | R1 | the matching R1 job (`target_adapt` bn/tent or `extract_features`) |
| `results/<phase>/<tag>/xai/` | H2 | the matching H2 job |
| `results/phase8_inference/...` | H1 | the matching H1 line |
| `results/tables/`, `results/figures/` | H3 | H3 commands; then read `TABLES_README.txt` / `FIGURES_README.txt`, which list every table/figure that was skipped and why |
| `results/<phase>/phase_table*.csv`, `WINNER.txt` | D1–D3, E2, E3, G3 | the `phase_select` line of that phase |
| `results/<phase>/<tag>/...` of a run (summary, per_class, ...) | that run's training step | check `ls results/phase*/<tag>`; if the run exists in another phase folder, send that path to Claude |
| `checkpoints/<tag>/config_snapshot.yaml` | that run's training step | `ls checkpoints/<tag>/`; if absent the run was never started under that tag |
| `splits/external/*`, `results/phase0_protocol/*` | C2 / phase 0 | do **not** re-create splits; tell Claude |

Some files are optional and may legitimately be missing: `phase_table_amended.csv` / `phase_table_original.csv`
exist only for phases selected under both endpoints; `results/phase8_inference/p8_tta_flip*` only if H1 TTA ran;
`p5_*` runs only if G3/G3b ran; `p6_*` runs and `recal_dg/` only after G4. If unsure, send `MISSING.txt` to Claude before re-running anything.
