# Phase 4 Sign Classifier + ROI-Voted Inference

Phase 4 trains Stage-A sign classification using cached Phase-3 tensors:
- `X_A = [I, D1, D2]` with shape `200x200x3`

Track boundary is strict:
- `smears`: `pbs`, `wbc`, `bma`
- `biopsy`: `focus_train`, `focus_test`

No cross-track mixing is allowed.

## Scripts
- `train_sign.py`: group-safe split + training + test evaluation + optional LODOO.
- `calibrate_tau.py`: val-split confidence-gating calibration.
- `infer_vote_sign.py`: ROI-weighted voting for FOV-level sign.
- `utils.py`: shared IO/split/metrics/tensor helpers.

## Inputs
- Phase-3 cache index:
  `/home/dineth/focus_measure/journal/Regression/data/cache_phase3/<track>/cache_index.csv`
- Phase-3 XA tensor paths from `cache_path_XA` (`.npy` or `.npz`).

If cache index is missing, scripts fail and instruct running Phase 3 first.

## Outputs (track-specific only)
`/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/sign/`

Contains:
- `models/best_model.keras`
- `splits/train.csv`, `val.csv`, `test.csv`
- `metrics/history.csv`, `eval.json`, `confusion_matrix.csv`, `per_bin_metrics.csv`
- `metrics/lodoo_results.csv` (if `--lodoo`)
- `calibration/tau_sweep.csv`, `chosen_tau.json`
- `inference/vote_sign_results.csv`
- `inference/vote_sign_per_roi.csv` (if `--save-per-roi`)

## Split safety
Splits are leakage-safe by `group_id` priority:
1. existing `group_id`
2. `slide_id`
3. `patient_id`
4. parent directory of `source_image_path` / `image_path`

A group never appears in more than one split.

## Voting logic
Per ROI:
- `p_i = model(XA_i)`
- `c_i = max(p_i, 1-p_i)`
- keep if `c_i >= tau`
- weight `w_i = roi_importance_i * c_i`

Per FOV:
- `V = sum(w_i * 1[p_i >= 0.5])`
- `W = sum(w_i)`
- sign = `1` if `V >= 0.5W` else `0`
- margin = `|V - 0.5W| / (0.5W + eps)`
- early exit when margin exceeds threshold while processing ROIs in descending importance.

## Example commands
```bash
python train_sign.py --track smears
python calibrate_tau.py --track smears
python infer_vote_sign.py --track smears --mode auto

python train_sign.py --track biopsy
python calibrate_tau.py --track biopsy
python infer_vote_sign.py --track biopsy --mode auto
```
