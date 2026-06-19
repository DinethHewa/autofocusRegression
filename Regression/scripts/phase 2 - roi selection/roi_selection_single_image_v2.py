#!/usr/bin/env python3
"""
Single-image RIN v2:
  - No diversity or border guards.
  - Selects at least 60% of tiles (by count).
  - Optional .npy export of selected tiles and visualization PNG.
  - Basic file existence check to avoid silent failures.
"""

import argparse
import os
import cv2
import numpy as np
from typing import Dict, Tuple

from roi_selection_v2 import (
    TILE_SIZE,
    load_image_rgb,
    to_gray,
    tile_image_resolution_aware,
    composite_focus_measure,
    gradient_prior,
    entropy_prior,
    robust_normalize,
    select_topk_tiles,
    visualize_tiles,
)


def compute_rin_scores_for_image(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    tiles, grid_h, grid_w = tile_image_resolution_aware(img_rgb)
    n_tiles = len(tiles)
    F_raw = np.zeros(n_tiles, dtype=np.float32)
    G_raw = np.zeros(n_tiles, dtype=np.float32)
    E_raw = np.zeros(n_tiles, dtype=np.float32)
    for idx, tile in enumerate(tiles):
        gray = to_gray(tile)
        F_raw[idx] = composite_focus_measure(gray)
        G_raw[idx] = gradient_prior(gray)
        E_raw[idx] = entropy_prior(gray)
    F_hat = robust_normalize(F_raw)
    G_hat = robust_normalize(G_raw)
    E_hat = robust_normalize(E_raw)
    T = 0.6 * G_hat + 0.4 * E_hat
    S = 0.7 * F_hat + 0.3 * T
    tiles_np = np.stack(tiles, axis=0).astype(np.float32)
    return tiles_np, S, F_hat, grid_h, grid_w


def rin_for_image_path(img_path: str) -> Dict:
    img_rgb = load_image_rgb(img_path)
    tiles, S, F_hat, gh, gw = compute_rin_scores_for_image(img_rgb)
    top_indices = select_topk_tiles(S, gh, gw)
    return {
        "tiles": tiles,
        "scores": S,
        "focus_norm": F_hat,
        "grid_h": gh,
        "grid_w": gw,
        "top_indices": np.array(top_indices, dtype=np.int32),
        "top_tiles": tiles[top_indices],
        "E_total": float(S.sum()),
        "N_tiles": len(S),
        "K_selected": len(top_indices),
    }


def main():
    ap = argparse.ArgumentParser(description="Single-image ROI selection v2 (>=60% tiles, no diversity/border guards)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--output-npy", default=None, help="Optional path to save selected tiles (.npy)")
    ap.add_argument("--viz", default=None, help="Optional path to save visualization PNG")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    res = rin_for_image_path(args.image)
    print(f"[INFO] Selected {len(res['top_tiles'])}/{len(res['tiles'])} tiles")

    if args.output_npy:
        np.save(args.output_npy, res["top_tiles"])
        print(f"[INFO] Saved tiles to {args.output_npy}")

    if args.viz:
        viz = visualize_tiles(load_image_rgb(args.image),
                              res["tiles"],
                              res["scores"],
                              res["top_indices"],
                              res["grid_h"],
                              res["grid_w"],
                              res.get("sparsity_label"),
                              res.get("sparsity_metric"),
                              res.get("sparsity_method"))
        cv2.imwrite(args.viz, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved viz to {args.viz}")


if __name__ == "__main__":
    main()
