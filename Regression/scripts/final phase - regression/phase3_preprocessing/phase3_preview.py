#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config_phase3 import build_phase3_config
from preprocess_ops import (
    assemble_XA_XB,
    compute_dog_lite,
    compute_dwt_ehf,
    percentile_clip,
    rescale01,
    to_grayscale,
    zscore,
)
from tiling import cut_to_grid


def _load_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    if cv2 is not None:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

    with Image.open(p) as im:
        return np.asarray(im)


def _norm_to_u8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo = float(np.percentile(a, 1.0))
    hi = float(np.percentile(a, 99.0))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _save_diagnostic(path: Path, I: np.ndarray, D1: np.ndarray, D2: np.ndarray, EHF: np.ndarray) -> None:
    labels = ["I", "D1", "D2", "EHF"]
    images = [_norm_to_u8(x) for x in [I, D1, D2, EHF]]
    h, w = images[0].shape

    canvas = Image.new("L", (w * 4, h + 18), color=0)
    for i, im in enumerate(images):
        tile = Image.fromarray(im, mode="L")
        canvas.paste(tile, (i * w, 18))

    # Add simple text labels with PIL default font.
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for i, lab in enumerate(labels):
        draw.text((i * w + 4, 2), lab, fill=255)

    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Phase-3 preprocessing (DoG-lite + DWT E_HF).")
    parser.add_argument("--input", required=True, help="Input full image path or ROI path")
    parser.add_argument("--track", required=True, choices=["smears", "biopsy"])
    parser.add_argument("--num-patches", type=int, default=6, help="Number of patches to visualize")
    parser.add_argument("--resume", action="store_true", help="Skip diagnostic images that already exist.")
    args = parser.parse_args()

    cfg = build_phase3_config(args.track)
    out_dir = cfg.out_dir_default / "debug_phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = _load_image(args.input)
    roi_size = int(cfg.roi_size)
    if img.shape[0] == roi_size and img.shape[1] == roi_size:
        patches = [("r00_c00", img)]
        print("[WARN] Input appears to be ROI-sized already; previewing single patch.")
    else:
        patches = cut_to_grid(img, grid=(10, 10), roi_size=roi_size)

    n = min(max(int(args.num_patches), 1), len(patches))
    print(f"[INFO] Processing {n}/{len(patches)} patches for preview.")

    wrote = 0
    skipped = 0
    for i in range(n):
        patch_id, patch = patches[i]
        out_file = out_dir / f"preview_{Path(args.input).stem}_{patch_id}.png"
        if args.resume and out_file.is_file():
            skipped += 1
            print(f"[INFO] Resume: keeping existing preview {out_file}")
            continue

        I = to_grayscale(patch)
        I = percentile_clip(I, p_low=cfg.p_low, p_high=cfg.p_high)
        I = rescale01(I, eps=cfg.eps)
        I = zscore(I, eps=cfg.eps)

        D1, D2 = compute_dog_lite(I, cfg.sigmas, eps=cfg.eps)
        EHF = compute_dwt_ehf(I, wavelet=cfg.wavelet, roi_size=cfg.roi_size, eps=cfg.eps)
        XA, XB = assemble_XA_XB(I, D1, D2, EHF, roi_size=cfg.roi_size)
        if XA.shape != (roi_size, roi_size, 3) or XB.shape != (roi_size, roi_size, 4):
            raise ValueError(f"Unexpected XA/XB shape: {XA.shape}, {XB.shape}")

        _save_diagnostic(out_file, I, D1, D2, EHF)
        print(f"[DONE] Wrote: {out_file}")
        wrote += 1

    print(f"[INFO] Preview summary: wrote={wrote} skipped={skipped}")


if __name__ == "__main__":
    main()
