#!/usr/bin/env python3
"""
CNN occupancy helpers (framework-agnostic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


try:
    from roi_selection_v2 import TILE_SIZE, SMALL_THRESH, LARGE_THRESH
except Exception:  # pragma: no cover
    TILE_SIZE = 200
    SMALL_THRESH = 400
    LARGE_THRESH = 2000


TileBox = Tuple[int, int, int, int]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to float32 in [0,1], preserving channel count."""
    if image is None:
        raise ValueError("image is None")
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("image must have shape (H,W) or (H,W,C)")
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        max_val = float(arr.max()) if arr.size else 65535.0
        scale = max_val if max_val > 0 else 65535.0
        arr = arr.astype(np.float32) / scale
    else:
        arr = arr.astype(np.float32)
        mx = float(arr.max()) if arr.size else 1.0
        if mx > 1.5:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def ensure_channel_dim(image: np.ndarray) -> np.ndarray:
    """Ensure image has a channel dimension."""
    if image.ndim == 2:
        return image[:, :, None]
    return image


def tile_boxes_resolution_aware(image_shape: Sequence[int]) -> Tuple[List[TileBox], int, int]:
    """Compute tile boxes using the same policy as roi_selection_v2."""
    if len(image_shape) < 2:
        raise ValueError("image_shape must include H,W")
    h, w = int(image_shape[0]), int(image_shape[1])
    boxes: List[TileBox] = []
    if min(h, w) < SMALL_THRESH:
        return [(0, 0, w, h)], 1, 1
    if h > LARGE_THRESH and w > LARGE_THRESH:
        grid_h = grid_w = 10
        ph, pw = h // grid_h, w // grid_w
        for i in range(grid_h):
            for j in range(grid_w):
                y0, y1 = i * ph, (i + 1) * ph
                x0, x1 = j * pw, (j + 1) * pw
                boxes.append((x0, y0, x1, y1))
        return boxes, grid_h, grid_w
    n = int(min(h // TILE_SIZE, w // TILE_SIZE))
    if n <= 0:
        return [(0, 0, w, h)], 1, 1
    grid_h = grid_w = n
    tile_h = h // n
    tile_w = w // n
    for i in range(grid_h):
        for j in range(grid_w):
            y0, y1 = i * tile_h, (i + 1) * tile_h
            x0, x1 = j * tile_w, (j + 1) * tile_w
            boxes.append((x0, y0, x1, y1))
    return boxes, grid_h, grid_w


def compute_tile_occupancy_from_map(p_map: np.ndarray, tile_boxes: Iterable[TileBox]) -> np.ndarray:
    """Mean probability per tile from a per-pixel map."""
    p_map = np.asarray(p_map, dtype=np.float32)
    if p_map.ndim != 2:
        raise ValueError("p_map must be 2D")
    occ = []
    for x0, y0, x1, y1 in tile_boxes:
        tile = p_map[y0:y1, x0:x1]
        occ.append(float(tile.mean()) if tile.size else 0.0)
    return np.asarray(occ, dtype=np.float32)


def compute_tile_maxprob_from_map(p_map: np.ndarray, tile_boxes: Iterable[TileBox]) -> np.ndarray:
    """Max probability per tile from a per-pixel map."""
    p_map = np.asarray(p_map, dtype=np.float32)
    if p_map.ndim != 2:
        raise ValueError("p_map must be 2D")
    occ = []
    for x0, y0, x1, y1 in tile_boxes:
        tile = p_map[y0:y1, x0:x1]
        occ.append(float(tile.max()) if tile.size else 0.0)
    return np.asarray(occ, dtype=np.float32)


def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    from PIL import Image
    pil = Image.fromarray((image * 255).astype(np.uint8))
    return np.asarray(pil.resize(size, resample=Image.BILINEAR)).astype(np.float32) / 255.0


def tile_image_for_cnn(
    image: np.ndarray,
    tile_boxes: Sequence[TileBox],
    downsample: int | None = None,
) -> np.ndarray:
    """Extract tile crops as a batch for per-tile CNN inference."""
    img = normalize_image(image)
    img = ensure_channel_dim(img)
    if downsample is not None and downsample > 0:
        h, w = img.shape[:2]
        scale = downsample / float(max(h, w))
        if scale < 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = _resize_image(img, (new_w, new_h))
    tiles = []
    for x0, y0, x1, y1 in tile_boxes:
        tiles.append(img[y0:y1, x0:x1])
    return np.asarray(tiles, dtype=np.float32)


def batch_predict(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    items: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if items.size == 0:
        return np.zeros((0,), dtype=np.float32)
    outputs = []
    n = len(items)
    for start in range(0, n, batch_size):
        batch = items[start : start + batch_size]
        preds = predict_fn(batch)
        outputs.append(np.asarray(preds))
    return np.concatenate(outputs, axis=0).reshape(-1).astype(np.float32)


@dataclass
class CellnessModelInterface:
    """Framework-agnostic adapter for occupancy inference."""

    predict_map_fn: Callable[[np.ndarray], np.ndarray] | None = None
    predict_tiles_fn: Callable[[np.ndarray], np.ndarray] | None = None
    name: str = "cellness_adapter"

    @property
    def has_map(self) -> bool:
        return self.predict_map_fn is not None

    @property
    def has_tiles(self) -> bool:
        return self.predict_tiles_fn is not None

    def predict_cellness_map(self, image: np.ndarray) -> np.ndarray:
        if self.predict_map_fn is None:
            raise RuntimeError("predict_map_fn is not set")
        p_map = self.predict_map_fn(image)
        p_map = np.asarray(p_map, dtype=np.float32)
        if p_map.ndim == 3:
            p_map = p_map[:, :, 0]
        return np.clip(p_map, 0.0, 1.0)

    def predict_tile_occupancy(self, tiles: np.ndarray) -> np.ndarray:
        if self.predict_tiles_fn is None:
            raise RuntimeError("predict_tiles_fn is not set")
        occ = self.predict_tiles_fn(tiles)
        occ = np.asarray(occ, dtype=np.float32).reshape(-1)
        return np.clip(occ, 0.0, 1.0)

    @classmethod
    def from_callable(cls, predict_fn: Callable[[np.ndarray], np.ndarray], mode: str = "auto") -> "CellnessModelInterface":
        mode = (mode or "auto").lower()
        if mode == "map":
            return cls(predict_map_fn=predict_fn, name="callable_map")
        if mode == "tiles":
            return cls(predict_tiles_fn=predict_fn, name="callable_tiles")
        return cls(predict_map_fn=predict_fn, name="callable_auto")


class DummyThresholdCellness(CellnessModelInterface):
    """Simple grayscale threshold baseline for development."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        super().__init__(predict_map_fn=self._predict_map, name="dummy_threshold")

    def _predict_map(self, image: np.ndarray) -> np.ndarray:
        img = normalize_image(image)
        if img.ndim == 3:
            gray = img.mean(axis=-1)
        else:
            gray = img
        return (gray >= self.threshold).astype(np.float32)
