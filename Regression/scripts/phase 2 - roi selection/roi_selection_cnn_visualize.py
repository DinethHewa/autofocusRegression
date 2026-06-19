#!/usr/bin/env python3
"""
CNN tile visualization using the best trained model.

Shows:
  - Blue: CNN-chosen tiles (occ >= tau_empty)
  - Red: Final selected tiles after ranking
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

from calibration_script_v2 import (
    infer_model_shape,
    load_keras_model,
    load_image_rgb,
    prepare_tiles_for_model,
    resolve_default_model_path,
    tile_image_resolution_aware,
)
from roi_cnn_occupancy import CellnessModelInterface
from roi_selection_cnn_v1 import ROISelectorCNNConfig, select_rois_cnn
from roi_selection_v2 import D_MIN_NON_DENSE, visualize_tiles


def _load_calibration(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("Calibration CSV is empty.")
    row = df.iloc[0].to_dict()
    calib: Dict[str, float] = {}
    for key, val in row.items():
        try:
            calib[key] = float(val)
        except (TypeError, ValueError):
            continue
    return calib


def _make_predict_tiles(model, target_hw, channels: int, batch_size: int):
    def _predict_tiles(tiles: np.ndarray) -> np.ndarray:
        if tiles is None or len(tiles) == 0:
            return np.zeros((0,), dtype=np.float32)
        arr = prepare_tiles_for_model(list(tiles), target_hw=target_hw, channels=channels)
        preds = model.predict(arr, batch_size=batch_size, verbose=0)
        return np.asarray(preds).reshape(-1).astype(np.float32)

    return _predict_tiles


def _save_viz(img_rgb: np.ndarray, res: Dict[str, object], out_path: str) -> None:
    tiles, gh, gw = tile_image_resolution_aware(img_rgb)
    tiles_np = np.asarray(tiles, dtype=np.float32)
    occ = np.asarray(res.get("occ", []), dtype=np.float32)
    tau_empty = res.get("debug", {}).get("tau_empty", 0.0)
    cnn_indices = np.where(occ >= float(tau_empty))[0] if occ.size else np.array([], dtype=np.int32)
    scores = np.asarray(res.get("score", []), dtype=np.float32)
    sumS = float(np.sum(scores[np.isfinite(scores)])) if scores.size else 0.0
    viz = visualize_tiles(
        img_rgb,
        tiles_np,
        scores,
        np.asarray(res.get("selected_indices", []), dtype=np.int32),
        gh,
        gw,
        k_selected=int(res.get("k", 0)),
        sumS=sumS,
        cnn_indices=cnn_indices,
    )
    cv2.imwrite(out_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))


def _iter_manifest_paths(manifest: str, path_col: str, limit_images: Optional[int]) -> list[str]:
    df = pd.read_csv(manifest)
    if path_col not in df.columns:
        raise ValueError(f"Manifest missing column: {path_col}")
    if limit_images:
        df = df.head(limit_images)
    return df[path_col].dropna().astype(str).tolist()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CNN visualization using best model")
    ap.add_argument("--image", default=None, help="Optional single image path")
    ap.add_argument("--manifests", nargs="+", default=None, help="Optional manifest CSV(s)")
    ap.add_argument("--path-col", default="image_path", help="Manifest column for image path")
    ap.add_argument("--viz-dir", required=True, help="Output directory for visualization PNGs")
    ap.add_argument("--model-path", default=str(resolve_default_model_path()))
    ap.add_argument("--model-input-size", type=int, default=None)
    ap.add_argument("--model-batch-size", type=int, default=64)
    ap.add_argument("--cnn-calibration-csv", default=None, help="Optional CNN calibration CSV (tau/beta)")
    ap.add_argument("--tau-empty", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=100)
    ap.add_argument("--use-sum-occ", action="store_true", default=True)
    ap.add_argument("--no-use-sum-occ", dest="use_sum_occ", action="store_false")
    ap.add_argument("--w-occ", type=float, default=0.85)
    ap.add_argument("--w-focus", type=float, default=0.15)
    ap.add_argument("--focus-on-candidates-only", action="store_true", default=True)
    ap.add_argument("--focus-on-all", dest="focus_on_candidates_only", action="store_false")
    ap.add_argument("--candidate-multiplier", type=float, default=2.0)
    ap.add_argument("--d-min-nondense", type=int, default=D_MIN_NON_DENSE)
    ap.add_argument("--dense-cap-ratio", type=float, default=0.55)
    ap.add_argument("--dense-min-ratio", type=float, default=None)
    ap.add_argument("--downsample-for-cnn", type=int, default=None)
    ap.add_argument("--limit-images", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image and not args.manifests:
        raise RuntimeError("Provide --image or --manifests.")

    os.makedirs(args.viz_dir, exist_ok=True)
    model = load_keras_model(Path(args.model_path))
    input_h, input_w, channels = infer_model_shape(model, args.model_input_size)
    target_hw = (input_h, input_w)

    predict_tiles_fn = _make_predict_tiles(model, target_hw, channels, args.model_batch_size)
    predictor = CellnessModelInterface(predict_tiles_fn=predict_tiles_fn, name="keras_tile_classifier")

    calib = _load_calibration(args.cnn_calibration_csv)
    cfg = ROISelectorCNNConfig(
        tau_empty=args.tau_empty,
        beta=args.beta,
        k_min=args.k_min,
        k_max=args.k_max,
        use_sum_occ=args.use_sum_occ,
        w_occ=args.w_occ,
        w_focus=args.w_focus,
        focus_on_candidates_only=args.focus_on_candidates_only,
        candidate_multiplier=args.candidate_multiplier,
        d_min_nondense=args.d_min_nondense,
        dense_cap_ratio=args.dense_cap_ratio,
        dense_min_ratio=args.dense_min_ratio,
        downsample_for_cnn=args.downsample_for_cnn,
        batch_size=args.model_batch_size,
    )

    if args.image:
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        img_rgb = load_image_rgb(args.image)
        res = select_rois_cnn(img_rgb, calib, cfg, predictor)
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(args.viz_dir, f"{base}_cnn.png")
        _save_viz(img_rgb, res, out_path)
        print(f"[VIZ] Saved {out_path}")
        return

    for man_path in args.manifests:
        if not os.path.isfile(man_path):
            print(f"[WARN] Manifest not found: {man_path}")
            continue
        man_base = os.path.splitext(os.path.basename(man_path))[0]
        out_dir = os.path.join(args.viz_dir, man_base)
        os.makedirs(out_dir, exist_ok=True)
        for img_path in _iter_manifest_paths(man_path, args.path_col, args.limit_images):
            if not os.path.isfile(img_path):
                print(f"[WARN] Missing image: {img_path}")
                continue
            img_rgb = load_image_rgb(img_path)
            res = select_rois_cnn(img_rgb, calib, cfg, predictor)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{base}_cnn.png")
            _save_viz(img_rgb, res, out_path)
            print(f"[VIZ] Saved {out_path}")


if __name__ == "__main__":
    main()
