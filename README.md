# Microscopy Autofocus Regression Pipeline

This repository contains the manuscript-facing code for a two-stage microscopy autofocus pipeline. The final pipeline builds leakage-safe dataset splits, caches ROI tensors, predicts defocus direction with a Stage-A sign classifier, estimates defocus magnitude with sign-conditioned Stage-B regressors, and aggregates ROI predictions into final field-of-view autofocus estimates.

The main implementation is under:

```text
Regression/scripts/final phase - regression/
```

Older supporting code is included only where the final pipeline still depends on it:

```text
Regression/scripts/phase 1- manifest creation/
Regression/scripts/phase 2 - roi selection/
Regression/scripts/phase 3 - sign selection/manifest_creation/
```

## Scope

The final pipeline supports two strict tracks:

```text
smears: PBS, WBC, BMA
biopsy: focus_train, focus_test
```

The smear-track workflow is:

```text
create manifests
build unified manifest
make leakage-safe splits
build Phase 3 cached tensors
train Stage-A sign classifier
calibrate tau
infer ROI-voted sign
build Phase-5 regression index
train sign-conditioned regressors
infer and aggregate signed defocus
run evaluation, ROI ablation, baseline comparison, and paper packaging
```

## Large Artifacts

Large generated files are intentionally excluded from this GitHub package. That includes trained models, cached tensors, raw image data, large prediction CSVs, paper packages, and experiment-output archives.

Use Zenodo or another data repository for: https://doi.org/10.5281/zenodo.20766723

```text
*.keras
*.h5
*.npy
*.npz
cache_phase3/
out_final_phase/
paper_package/
large CSV result tables
raw microscopy images
```

## Notes

Some scripts in the research codebase still contain local absolute paths from the original workstation. Before running this package on another machine, update path configuration in `Regression/scripts/final phase - regression/common_paths.py` and related runner scripts, or refactor paths to be repository-relative.

