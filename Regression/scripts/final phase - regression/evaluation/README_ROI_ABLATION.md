# ROI Ablation Evaluation For Fixed-Model Autofocus

## Purpose
This evaluation layer isolates the effect of ROI-selection policy on downstream autofocus-distance estimation.

The sign classifier and magnitude regressors are kept fixed. ROI policies are swapped only at inference time. This avoids conflating ROI-policy quality with retraining effects and keeps the primary analysis scientifically defensible.

## Design Summary
1. Load the Phase 5 ROI index from `regression/index_phase5.csv`.
2. Precompute fixed per-ROI model outputs once for all candidate ROIs:
   - Stage A sign probability and confidence
   - Stage B positive-branch magnitude prediction
   - Stage B negative-branch magnitude prediction
3. For each ROI policy, select a subset of ROIs per FOV.
4. Re-run FOV-level voting and weighted-median aggregation using only that selected subset.
5. Compare end-to-end autofocus-distance performance across policies.

This means the main comparison answers:

`If the downstream models are unchanged, how much does the ROI policy alone affect autofocus accuracy, robustness, and efficiency?`

## Fixed-Model Rationale
The main paper claim is about ROI selection quality, not about retraining entire downstream stacks for every selector.

Keeping Stage A and Stage B fixed provides:
- causal isolation of the ROI policy
- lower experimental variance
- lower compute cost
- a fairer ablation for publication

Optional retraining-based comparisons can still be added later as secondary experiments, but they are not the primary evidence produced by this pipeline.

## Existing Interface Audit
The current codebase exposes the following interfaces.

### ROI-side Inputs
ROI candidates are aligned to the Phase 3 cache grid and Phase 5 index.

Observed/required ROI-level columns in `regression/index_phase5.csv`:
- `roi_uid`
- `dataset`
- `group_id`
- `fov_id`
- `cache_path_XA`
- `cache_path_XB`
- `defocus_um`
- `y_sign`
- `y_mag_um`
- `patch_id`
- `source_image_path` or `image_path`
- `roi_importance` if available

### Stage A Inputs / Outputs
Current Stage A inference consumes `X_A` tensors from `cache_path_XA`.

Observed per-ROI output columns in `sign/inference/vote_sign_per_roi.csv`:
- `fov_id`
- `roi_uid`
- `dataset`
- `roi_importance`
- `p`
- `c`
- `kept`
- `w`
- `V_cum`
- `W_cum`
- `margin_cum`
- `processed_rank`

Observed FOV-level output columns in `sign/inference/vote_sign_results.csv`:
- `fov_id`
- `pred_sign`
- `pred_sign_int`
- `vote_margin`
- `V`
- `W`
- `num_rois_total`
- `num_rois_processed`
- `num_rois_kept`
- `early_exit`
- `track`
- `tau`
- `mode`

### Stage B Inputs / Outputs
Current Stage B inference consumes `X_B` tensors from `cache_path_XB`.

Observed per-ROI output columns in `regression/inference/roi_predictions.csv`:
- `roi_uid`
- `fov_id`
- `dataset`
- `voted_sign`
- `vote_margin`
- `p_sign`
- `c_sign`
- `roi_importance`
- `w`
- `y_mag_pred`
- `signed_dz_pred`
- `y_sign_pred`
- `weight`

Observed FOV-level output columns in `regression/inference/fov_aggregate_predictions.csv`:
- `fov_id`
- `voted_sign`
- `vote_margin`
- `dz_hat_um`
- `status`
- `num_rois_used`
- `num_rois_total`
- `tau`
- `top_k`
- `runtime_ms`
- `margin_threshold`

### Existing FOV Aggregation
The current final package aggregates ROI predictions to FOV predictions as follows:
- Stage A ROI confidence gating with `tau`
- ROI weight `w_i = roi_importance_i * c_i`
- Stage A hard sign voting on selected ROIs
- Stage B branch routing by FOV voted sign
- weighted median of signed ROI predictions for final `dz_hat_um`

The ROI-ablation runner reuses this logic and only changes the ROI subset.

## Standard ROI Policy Interface
The module `roi_policy_utils.py` standardizes ROI selection output.

Each selected ROI row includes at minimum:
- `fov_id`
- `roi_id`
- `dataset`
- `selection_rank`
- `selection_score`
- `roi_policy`
- `selected`

Additional diagnostics are populated where available, such as:
- `focus_score`
- `occupancy_score`
- `hybrid_score`
- `valid_by_tau_empty`
- `selection_backend`

## Supported ROI Policies
### `center_top1`
Choose the central ROI, or the closest available ROI to the center of the cached grid.

### `random_k`
Deterministic random sampling of `K` ROIs per FOV using a fixed seed.

### `focus_only_topk`
Top `K` ROIs by handcrafted focus score only.

### `occupancy_only_topk`
Top `K` ROIs by occupancy or cellness score only.

### `hybrid_proposed`
The proposed policy. This uses occupancy gating plus focus-aware ranking and diversity-aware top-k selection, reflecting the intended logic of `roi_selection_cnn_v1.py`.

### `cnn_adaptive`
Direct CNN-adaptive selection path using the same hybrid logic family.

### `legacy_adaptive`
Direct legacy adaptive ROI logic from `roi_selection_v2.py` when its calibration CSV is available.

### `all_rois`
Use all available ROIs in the FOV. This is useful as a compute-heavy baseline.

### `oracle_best_single_roi`
Analysis-only oracle. Uses ground truth to pick the single ROI with the smallest downstream error. This is not a deployable method.

## Output Files
The ROI-ablation suite writes the following outputs under `data/out_final_phase/<track>/evaluation/`.

Primary outputs:
- `roi_ablation_to_regression_performance.csv`
- `roi_ablation_by_dataset.csv`
- `roi_efficiency_tradeoff.csv`
- `roi_ablation_confidence_intervals.csv`
- `roi_policy_pairwise_tests.csv`
- `roi_robustness_vs_error.csv`
- `roi_ablation_near_focus.csv`
- `roi_failure_cases.csv`
- `roi_score_diagnostics.csv`
- `domain_gap_reduction.csv`

Supplementary outputs:
- `roi_ablation_by_magnitude_bin.csv`
- `roi_ablation_by_roi_count_bucket.csv`
- `roi_ablation_fixed_model_outputs.csv`
- `roi_ablation_fixed_model_latency.json`
- `roi_ablation_metadata.json`
- `qualitative_failure_panels/`

Reference tables are also written to:
- `evaluation/reference/tables/`
- `tables/FINAL_PHASE/<TRACK>/`

## Coverage And Uncertainty Definitions
- `coverage_pct`: fraction of FOVs with a valid end-to-end signed distance prediction under that ROI policy.
- `uncertain_pct`: fraction of FOVs without a valid prediction after ROI selection, Stage A gating, and aggregation.

These definitions are consistent with the existing final-phase gating and aggregation logic.

## Runtime Measurement
For each ROI policy the suite estimates:
- ROI selection time per FOV
- Stage A latency contribution
- Stage B latency contribution
- weighted-median aggregation time
- total latency per FOV

The Stage A and Stage B latency terms are derived from fixed-model per-ROI timing measurements on the same track.

## Robustness Analysis
The robustness analysis perturbs source FOV images, recomputes ROI selection, measures Jaccard overlap with the original selected set, and then measures the change in final autofocus error.

This supports a stronger claim than ROI-selection stability alone:

`stable ROI selection should translate into stable downstream autofocus predictions.`

## Failure Analysis
Worst-case FOVs by absolute end-to-end error are saved per policy. Where the source image is available, a qualitative panel is generated showing:
- the original FOV image
- the selected ROI boxes
- predicted signed distance
- ground-truth signed distance
- absolute error
- dataset and policy label

## Commands
Run the ROI-ablation suite directly:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/run_roi_ablation_suite.py" \
  --track smears \
  --roi-policies center_top1 random_k focus_only_topk occupancy_only_topk hybrid_proposed all_rois \
  --k-values 1 3 5 7 \
  --bootstrap 1000 \
  --bins 0.5 \
  --save-plots \
  --save-latex \
  --resume
```

Run the full evaluation pipeline including ROI ablation:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/run_full_evaluation.py" \
  --track smears \
  --with-roi-ablation \
  --save-plots \
  --save-latex \
  --resume
```

Run the sanity checks:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/sanity_check_roi_ablation.py" \
  --track smears \
  --sample-fovs 24 \
  --k-values 1 3 5 7
```

## Practical Notes
- The primary comparison is inference-time only.
- If a policy cannot be run because required scores or calibration files are missing, the suite records it as unavailable instead of crashing the entire evaluation.
- Occupancy-based policies can fall back to a deterministic dummy-threshold backend for smoke testing, but publication runs should use the intended occupancy model or predictor when available.
