# Q1 Paper Package Builder

This utility creates a separate, publication-oriented package from the final-phase outputs.

Script:
- `build_q1_paper_package.py`

## Purpose

The script assembles:
- main-paper figures
- supplementary figures
- manuscript-ready summary tables
- copied reference evaluation plots/tables
- a paper asset manifest with file paths and suggested usage
- a concise results summary

## Default output location

Per track:
- `/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/paper_package/`

If `--track all` is used:
- `/home/dineth/focus_measure/journal/Regression/data/out_final_phase/paper_package_all/<track>/`

## What it generates

Inside each package:
- `figures/main/`
- `figures/supplementary/`
- `tables/main/`
- `tables/reference/`
- `reference/evaluation_outputs/`
- `reference/source_data/`
- `metadata/paper_asset_manifest.csv`
- `metadata/results_summary.md`
- `metadata/package_config.json`

## Inputs expected

The builder expects the final evaluation outputs to exist under:
- `.../data/out_final_phase/<track>/evaluation/`

If they are missing, the script will run:
- `run_full_evaluation.py --track <track> --save-plots --resume`

Use `--skip-evaluation` if you want it to fail instead of auto-running evaluation.

## Example commands

Build a package for smears:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/build_q1_paper_package.py" --track smears --save-latex --resume
```

Build a package for biopsy:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/build_q1_paper_package.py" --track biopsy --save-latex --resume
```

Build both tracks into a common root:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/build_q1_paper_package.py" --track all --save-latex --resume
```

Force regeneration:

```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/evaluation/build_q1_paper_package.py" --track smears --force
```
