# Final Phase Regression: Manifest Creation

## Dataset Groups (hard boundary)
- Group 1 `smears`: `pbs`, `wbc`, `bma`
- Group 2 `biopsy`: `focus_train`, `focus_test`

Use the dedicated runner for each group. Do not mix groups.

## Output Convention
For each dataset `<dataset>`:
- Manifest: `/home/dineth/focus_measure/journal/Regression/data/manifest_<dataset>.csv`
- Table: `/home/dineth/focus_measure/journal/Regression/tables/<DATASET>/<dataset>_table.csv`

Where `<DATASET>` is uppercase (examples: `PBS`, `WBC`, `BMA`, `FOCUS_TRAIN`, `FOCUS_TEST`).

## Runners
- Smears: `/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/smears/create_manifests_smears.py`
- Biopsy: `/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/biopsy/create_manifests_biopsy.py`

## Usage
```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/smears/create_manifests_smears.py"
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/biopsy/create_manifests_biopsy.py"
```

Dataset subsets:
```bash
python ".../smears/create_manifests_smears.py" --datasets pbs wbc
python ".../biopsy/create_manifests_biopsy.py" --datasets focus_train
```

Force overwrite:
```bash
python ".../smears/create_manifests_smears.py" --force
python ".../biopsy/create_manifests_biopsy.py" --force
```

Dry run:
```bash
python ".../smears/create_manifests_smears.py" --dry-run
python ".../biopsy/create_manifests_biopsy.py" --dry-run
```

## Behavior Notes
- Scripts reuse Phase-3 manifest creators in:
  `/home/dineth/focus_measure/journal/Regression/scripts/phase 3 - sign selection/manifest_creation`
- By default, existing manifest+table pairs are skipped (idempotent behavior).
- If only one output exists, the runner regenerates/repairs outputs for that dataset.
- If a creator emits a non-standard table name, the runner detects it under the dataset table folder and copies it to `<dataset>_table.csv`.
- If no table is produced (common for focus datasets), the runner creates `<dataset>_table.csv` from the manifest data.
- Missing required Phase-3 scripts are reported with expected locations and the runner exits non-zero.
