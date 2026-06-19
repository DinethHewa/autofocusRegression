# Final Phase Evaluation (Phases 4 + 5)

This module evaluates the autofocus pipeline for both tracks:
- `smears`
- `biopsy`

All outputs are track-separated.

## Input assumptions

Per track (`<track>` = `smears` or `biopsy`):
- Sign outputs:  
  `/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/sign/`
- Regression outputs:  
  `/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/regression/`
- Required Phase5 index:  
  `/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/regression/index_phase5.csv`

Ground truth is taken from `defocus_um` in Phase5 index (joined from manifests).

## Output directories

Evaluation data outputs:
- `/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/evaluation/`

Paper tables:
- `/home/dineth/focus_measure/journal/Regression/tables/FINAL_PHASE/<TRACK_UPPER>/`

## Scripts

- `evaluate_stageA.py`: Stage A sign metrics + plots + `Table_StageA.csv`
- `evaluate_stageB.py`: Stage B magnitude metrics + plots + `Table_StageB.csv`
- `evaluate_end_to_end.py`: cascade end-to-end metrics + plots + `Table_EndToEnd.csv`
- `evaluate_runtime.py`: runtime/latency/model-size profiling + plots + `Table_Runtime.csv`
- `run_ablation_suite.py`: ablation summary + plot + `Table_Ablation.csv`
- `cross_dataset_validation.py`: cross-dataset (LODOO/proxy) summary + plot
- `statistical_tests.py`: Wilcoxon, McNemar, bootstrap CI, Friedman+Nemenyi
- `generate_all_figures.py`: ensures all required figures are generated
- `run_full_evaluation.py`: master runner for the full evaluation pipeline

## CLI (common)

All scripts accept:
- `--track {smears,biopsy}`
- `--save-plots`
- `--save-latex`
- `--resume` (skip when expected outputs already exist)
- `--bins 0.5`
- `--bootstrap 1000`
- `--k-values 3 5 7`

## Recommended usage

Run all evaluation for smears:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/run_full_evaluation.py" --track smears --save-latex
```

Run all evaluation for biopsy:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/run_full_evaluation.py" --track biopsy --save-latex

Resume a previously completed evaluation run:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/run_full_evaluation.py" --track smears --resume
```
```

Run only figure regeneration:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/generate_all_figures.py" --track smears --save-plots --force
```

## Generated key files

In `.../evaluation/`:
- `stageA_metrics.json`, `stageA_confusion_matrix.csv`, `stageA_bin_accuracy.csv`
- `stageB_metrics.json`, `stageB_bin_mae.csv`
- `end_to_end_metrics.json`, `catastrophic_rate.csv`
- `runtime_metrics.json`, `latency_vs_k.csv`
- `ablation_results.csv`
- `cross_dataset_results.csv`
- `statistical_tests_results.json`, `confidence_intervals.csv`
- `all_figures_index.csv`
- `evaluation_log.txt`

In `.../tables/FINAL_PHASE/<TRACK_UPPER>/`:
- `Table_StageA.csv` (+ optional `.tex`)
- `Table_StageB.csv` (+ optional `.tex`)
- `Table_EndToEnd.csv` (+ optional `.tex`)
- `Table_Ablation.csv` (+ optional `.tex`)
- `Table_Runtime.csv` (+ optional `.tex`)

## Troubleshooting

- If a script fails with missing inputs, run upstream phase scripts first (Phase4/Phase5 inference outputs are required).
- If `vote_sign_per_roi.csv` is missing, rerun Phase4 inference with `--save-per-roi` for full StageA/statistical analysis.
- If `roi_predictions.csv` or `fov_aggregate_predictions.csv` is missing, rerun Phase5 inference.
- If `index_phase5.csv` is missing, run `build_phase5_index.py` first.
