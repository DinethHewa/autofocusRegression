# Phase 5 Regression (Option B1)

Phase 5 trains two Stage-B magnitude regressors in **microns** using `defocus_um`:
- `R_plus` for `defocus_um > 0`
- `R_minus` for `defocus_um < 0`

Joint loss per branch:
- `L_total = 1.0 * Huber(y_mag_um, y_hat_um) + lambda2 * Triplet`

Triplets are sampled **within branch only** using magnitude bins:
- positive: same bin, different `group_id`
- negative: different bin, prefer semi-hard neighboring bins

## Required inputs
- Manifests:
  `/home/dineth/focus_measure/journal/Regression/data/manifest_<dataset>.csv`
  with columns at least `image_path`, `defocus_um`
- Phase-3 cache index:
  `/home/dineth/focus_measure/journal/Regression/data/cache_phase3/<track>/cache_index.csv`
  with at least `roi_uid`, `dataset`, `cache_path_XB`

If inputs are missing, scripts fail with actionable errors.

## Scripts
- `build_phase5_index.py`
  - joins manifest `defocus_um` onto ROI cache index
  - derives `fov_id`, `group_id` when missing
  - writes `index_phase5.csv`
- `train_regressors.py`
  - trains `R_plus_best.keras` and `R_minus_best.keras`
  - saves branch splits, training history, eval JSON, per-bin metrics
  - saves triplet config and optional LODOO metrics
- `infer_regress_and_aggregate.py`
  - loads Phase-4 voted sign + tau
  - routes by voted sign to branch model
  - applies tau-gated ROI weights `w_i = roi_importance_i * c_i`
  - aggregates per FOV using weighted median of signed predictions

## Output layout (per track)
`/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/regression/`

Contains:
- `models/R_plus_best.keras`, `models/R_minus_best.keras`
- `splits/train_plus.csv`, `val_plus.csv`, `test_plus.csv`
- `splits/train_minus.csv`, `val_minus.csv`, `test_minus.csv`
- `metrics/history_plus.csv`, `history_minus.csv`
- `metrics/eval_plus.json`, `eval_minus.json`
- `metrics/per_bin_plus.csv`, `per_bin_minus.csv`
- `triplets/triplet_config.json` and optional `sampling_stats.csv`
- `inference/roi_predictions.csv`
- `inference/fov_aggregate_predictions.csv`

## Example commands
```bash
python build_phase5_index.py --track smears
python train_regressors.py --track smears
python infer_regress_and_aggregate.py --track smears

python build_phase5_index.py --track biopsy
python train_regressors.py --track biopsy
python infer_regress_and_aggregate.py --track biopsy
```
