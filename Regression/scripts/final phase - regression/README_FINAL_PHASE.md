# Final Phase: Dual-Track Runner

This adds a strict two-track runner for the final cascade + joint-learning autofocus pipeline.

## Tracks (never mix)
- `smears`: `pbs`, `wbc`, `bma`
- `biopsy`: `focus_train`, `focus_test`

`run_track.py` enforces that datasets passed by `--datasets` must belong to the chosen track.

## Main commands
```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/smears/run_smears.py"
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/biopsy/run_biopsy.py"
```

Optional flags:
- `--skip-manifest-create`
- `--force`
- `--dry-run`
- `--datasets ...` (track-local override)

## Pipeline phases per track
1. Load per-dataset manifest/table and validate required inputs.
2. Add labels (`delta_z`, `y_sign`, `y_mag`) and ROI gating policy.
3. Build leakage-safe group split by priority:
   - `patient_id` / `slide_id`
   - else `stack_id`
   - else `image_path` parent folder
4. Train sign model (MobileNetV3-Small) and calibrate confidence threshold `tau` on validation set.
5. Train two magnitude regressors with joint loss (Huber + triplet):
   - `R_plus` for `delta_z > 0`
   - `R_minus` for `delta_z < 0`

Inference/evaluation route:
- Sign uses ROI-weighted vote and confidence gating (`tau`).
- Magnitude branch is selected by predicted sign.

## Output layout (track-specific)
Base:
- `/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/`

Per track:
- `manifests/combined_group_manifest.csv`
- `splits/split_train.csv`, `split_val.csv`, `split_test.csv`
- `models/sign/sign_model.keras`
- `models/sign/tau.json`
- `models/regress_plus/R_plus.keras`
- `models/regress_minus/R_minus.keras`
- `metrics/sign_metrics.csv`
- `metrics/reg_metrics.csv`
- `metrics/end_to_end_metrics.csv`
- `config.json`

Group summary tables:
- `/home/dineth/focus_measure/journal/Regression/tables/FINAL_PHASE/SMEARS/`
- `/home/dineth/focus_measure/journal/Regression/tables/FINAL_PHASE/BIOPSY/`

## Safeguards
- Fails loudly if required input manifest/table files are missing.
- Fails on invalid track/dataset combinations.
- Group-safe split prevents cross-split leakage.
- Output directories are track-local and separate for smears vs biopsy.
