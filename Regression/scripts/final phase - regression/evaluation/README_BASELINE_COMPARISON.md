# Baseline Comparison Package

## Purpose
This package adds manuscript-grade baseline comparisons for the final smear/biopsy autofocus pipeline.
The comparison is designed to answer three separate questions without conflating them:

1. Architecture control: does the proposed two-stage sign-aware design outperform a simpler direct signed-distance regressor?
2. Input control: does content-aware ROI selection outperform deterministic center-only or tiled full-field proxy inputs when the downstream models are otherwise fixed?
3. Classical-method control: does the learned pipeline outperform a transparent non-deep-learning autofocus baseline based on handcrafted focus measures?

## Current Proposed Method
The current proposed method uses:
- content-aware ROI selection upstream
- a Stage A sign classifier (`phase4_sign/train_sign.py` / `infer_vote_sign.py`)
- Stage B plus/minus magnitude regressors (`phase5_regression/train_regressors.py` / `infer_regress_and_aggregate.py`)
- FOV-level sign voting, weighted-median aggregation, and uncertainty handling

The proposed method outputs already available in the repository are reused directly when present.

## Repository Interfaces Used
### Manifests / Index Files
- Phase-5 ROI index: `data/out_final_phase/<track>/regression/index_phase5.csv`
- Shared sign splits: `data/out_final_phase/<track>/sign/splits/{train,val,test}.csv`
- Baseline shared splits: `data/out_final_phase/<track>/baselines/_shared_splits/{train,val,test}.csv`

### ROI-Level Inputs
- Stage A cached tensors: `cache_path_XA`
- Stage B cached tensors: `cache_path_XB`
- Metadata used downstream includes:
  - `roi_uid`
  - `dataset`
  - `group_id`
  - `fov_id`
  - `patch_id`
  - `roi_importance`
  - `defocus_um`
  - `source_image_path`

### Current Inference Outputs
- Proposed FOV predictions: `data/out_final_phase/<track>/regression/inference/fov_aggregate_predictions.csv`
- Stage A and Stage B outputs under `data/out_final_phase/<track>/sign/` and `.../regression/`
- Standard evaluation outputs under `data/out_final_phase/<track>/evaluation/`

### Runtime Measurement Points
The package records or reuses:
- preprocessing/feature extraction latency where available
- inference latency per ROI/FOV
- aggregation latency or a transparent proxy when only per-ROI timings are available

## Fairness Rules Used Here
- Group-safe train/val/test splits are fixed across learned baselines.
- The direct single-stage regressor uses the same cached ROI tensors as the proposed pipeline.
- Input baselines are kept separate from architecture baselines.
- The classical baseline is explicitly non-deep-learning and uses only handcrafted features plus shallow regression.
- Inference-only baselines are labeled as such; retrained baselines are not silently substituted.

## Baselines Implemented
### 1. `proposed_two_stage_roi`
Family: `proposed`
Scope: full pipeline
Uses the repository's existing two-stage ROI-aware autofocus outputs.

### 2. `direct_single_stage_regression`
Family: `learned_architecture`
Scope: architecture baseline
Trains a single signed-distance regressor on the same cached ROI tensors used by the proposed Stage B path.
No sign classifier, no plus/minus branching.

### 3. `classical_focus_best`
Family: `classical`
Scope: classical handcrafted baseline
Builds handcrafted focus features from cached ROI tensors and selects the best shallow regressor / input scope on validation data.
Currently compares:
- `center_crop_proxy`
- `roi_selected_proxy`
- `full_field_tiled_proxy`
with shallow models:
- ridge regression
- Huber regression

### 4. `center_crop_inference_only`
Family: `learned_input`
Scope: input baseline, inference only
Keeps the trained proposed sign/regression models fixed and replaces content-aware ROI selection with the deterministic center-most ROI.

### 5. `full_field_tiled_proxy`
Family: `learned_input`
Scope: input baseline, proxy
Keeps the trained proposed sign/regression models fixed and uses all cached ROI tiles as a tiled full-field proxy.
This is not a true end-to-end full-image network.

## Optional / Explicitly Unavailable Rows
The runner can emit unavailable rows with `available=0` for:
- `center_crop_retrained`
- `full_image_retrained`
when those experiments are not implemented in the current code path.

## Outputs
Saved under `data/out_final_phase/<track>/evaluation/` unless `--out-dir` is provided:
- `baseline_comparison_results.csv`
- `baseline_by_dataset.csv`
- `baseline_near_focus.csv`
- `baseline_pairwise_tests.csv`
- `baseline_efficiency.csv`
- `baseline_failure_cases.csv`
- `baseline_qualitative_panel.csv`
- figures such as:
  - `baseline_efficiency_frontier.png`
  - `baseline_dataset_mae.png`
  - `baseline_near_focus.png`

Paper-table copies are written under:
- `tables/FINAL_PHASE/<TRACK>/`

## Commands
Train and evaluate the main manuscript baselines:
```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/run_baseline_comparison.py" \
  --track smears \
  --mode train_and_eval \
  --save-plots \
  --save-latex \
  --resume
```

Run only evaluation using already trained outputs:
```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/run_baseline_comparison.py" \
  --track smears \
  --baselines proposed_two_stage_roi direct_single_stage_regression classical_focus_best center_crop_inference_only full_field_tiled_proxy \
  --mode eval_only \
  --save-plots \
  --save-latex \
  --resume
```

Integrate with the existing full-evaluation runner:
```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/run_full_evaluation.py" \
  --track smears \
  --include-baseline-comparison \
  --save-latex \
  --resume
```
