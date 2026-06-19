# Phase 3 Preprocessing (DoG-lite + 1-level DWT E_HF)

## What this phase computes
For each ROI patch (fixed `200x200`):
1. Convert to grayscale `I`.
2. Robust normalize per ROI:
   - percentile clip (`p1`, `p99` by default)
   - rescale to `[0,1]`
   - z-score
3. DoG-lite with three Gaussian scales:
   - `B1 = G(I, sigma1)`
   - `B2 = G(I, sigma2)`
   - `B3 = G(I, sigma3)`
   - `D1 = B1 - B2`, `D2 = B2 - B3`
   - z-score normalize `D1`, `D2`
4. 1-level Haar/DB1 DWT high-frequency energy:
   - compute `LH`, `HL`, `HH`
   - `E_HF = sqrt(LH^2 + HL^2 + HH^2)`
   - upsample to `200x200` (bilinear)
   - z-score normalize `E_HF`

Outputs:
- Stage A tensor: `X_A = [I, D1, D2]` (3 channels)
- Stage B tensor: `X_B = [I, D1, D2, E_HF]` (4 channels)

## Why grayscale
Intensity-domain focus cues and high-frequency blur behavior are fundamentally luminance-driven. Using grayscale avoids color-channel instability and keeps preprocessing deterministic/fast.

## Why LL is not used
LL is the low-frequency approximation and does not directly capture blur-sensitive detail changes. Phase 3 keeps only high-frequency detail (`LH`, `HL`, `HH`) via `E_HF`.

## Track-specific sigma presets
- `smears`: `(0.7, 1.4, 2.8)`
- `biopsy`: `(1.0, 2.0, 4.0)`

## Caching
Cache builder writes deterministic per-ROI files under:
- `/home/dineth/focus_measure/journal/Regression/data/cache_phase3/smears/`
- `/home/dineth/focus_measure/journal/Regression/data/cache_phase3/biopsy/`

Per ROI cache includes:
- `*_XA.npy`
- `*_XB.npy`
- `*_meta.json`

And a global index:
- `cache_index.csv` mapping `roi_uid`, `source_image_path`, `patch_id`, `cache_path_XA`, `cache_path_XB`, track/dataset, plus available labels.

Idempotent behavior: existing cache entries are skipped unless `--force`.

## Reproducibility logs
- Config: `/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/preprocessing_config.json`
- Stats: `/home/dineth/focus_measure/journal/Regression/data/out_final_phase/<track>/metrics/preprocessing_stats.csv`

Stats include per-channel mean/std and average DoG/DWT runtime per ROI.

## Commands
Build cache:
```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/phase3_preprocessing/phase3_build_cache.py" --track smears
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/phase3_preprocessing/phase3_build_cache.py" --track biopsy
```

Preview on sample full images:
```bash
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/phase3_preprocessing/phase3_preview.py" --input "/mnt/data/defocus-350.jpg" --track biopsy
python "/home/dineth/focus_measure/journal/Regression/scripts/final phase - regression/phase3_preprocessing/phase3_preview.py" --input "/mnt/data/pos000_1_page_2.png" --track smears
```
