#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _resize_patch(patch: np.ndarray, roi_size: int) -> np.ndarray:
    if patch.ndim not in (2, 3):
        raise ValueError(f"Patch must be 2D or 3D, got shape={patch.shape}")
    if cv2 is not None:
        resized = cv2.resize(patch, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
        return resized

    if patch.ndim == 2:
        img = Image.fromarray(patch)
        return np.asarray(img.resize((roi_size, roi_size), resample=Image.BILINEAR))

    img = Image.fromarray(patch[:, :, :3])
    return np.asarray(img.resize((roi_size, roi_size), resample=Image.BILINEAR))


def cut_to_grid(full_image: np.ndarray, grid: tuple[int, int] = (10, 10), roi_size: int = 200) -> List[Tuple[str, np.ndarray]]:
    if full_image.ndim not in (2, 3):
        raise ValueError(f"full_image must be 2D/3D array, got shape={full_image.shape}")
    rows, cols = grid
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid must be positive, got {grid}")

    h, w = full_image.shape[:2]
    if h < rows or w < cols:
        raise ValueError(f"Image too small for grid {grid}: shape={full_image.shape}")

    row_edges = np.linspace(0, h, rows + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, cols + 1, dtype=np.int32)

    patches: List[Tuple[str, np.ndarray]] = []
    for r in range(rows):
        y0 = int(row_edges[r])
        y1 = int(row_edges[r + 1])
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        for c in range(cols):
            x0 = int(col_edges[c])
            x1 = int(col_edges[c + 1])
            if x1 <= x0:
                x1 = min(w, x0 + 1)

            patch = full_image[y0:y1, x0:x1]
            if patch.size == 0:
                raise ValueError(f"Empty patch encountered at r={r}, c={c}")
            if patch.shape[0] != roi_size or patch.shape[1] != roi_size:
                patch = _resize_patch(patch, roi_size=roi_size)
            if patch.shape[0] != roi_size or patch.shape[1] != roi_size:
                raise ValueError(f"Patch resize failed at r={r}, c={c}; shape={patch.shape}")

            patch_id = f"r{r:02d}_c{c:02d}"
            patches.append((patch_id, patch))

    return patches
