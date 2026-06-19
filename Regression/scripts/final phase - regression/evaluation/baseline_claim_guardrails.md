# Baseline Claim Guardrails

## Claims Supported If The Proposed Method Wins
Safe wording:
- The proposed two-stage ROI-aware autofocus pipeline outperformed the tested direct single-stage signed-distance regressor on this dataset and split.
- Content-aware ROI selection outperformed the tested center-crop input baseline under fixed downstream models.
- The proposed method achieved a stronger accuracy-efficiency tradeoff than the tested baselines in this repository.
- The proposed method reduced error on the tested smear datasets, including near-focus regimes, relative to the implemented baselines.
- The proposed method outperformed the implemented handcrafted classical-focus baseline under the tested calibration setup.

## Claims That Require Caution
Use qualified wording:
- Results demonstrate improved performance on the current smear/biopsy repository data and split protocol.
- Full-field results are based on a tiled proxy unless a true full-image retrained baseline is explicitly run.
- Center-crop retraining and full-image retraining should not be implied when only inference-only or proxy baselines are available.

## Claims Not Supported By This Package Alone
Unsafe wording:
- Universal superiority across all microscopy domains
- Superiority over every possible classical autofocus method
- Clinical generalization beyond the evaluated datasets
- Superiority of tiled full-field proxy over a true full-image end-to-end model

## Practical Writing Guidance
- Explicitly distinguish `architecture baseline`, `input baseline`, and `classical baseline` in the paper.
- Report macro dataset averages and domain-gap statistics, not only pooled numbers.
- State when a baseline is an inference-only control versus a retrained learned baseline.
- If a baseline is unavailable, say so directly and keep it marked `available=0` in tables rather than omitting it silently.
