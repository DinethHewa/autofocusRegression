#!/usr/bin/env python3
"""
CNN-dominant ROI selection (cellness gate + focus tie-breaker).
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import importlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from roi_cnn_occupancy import (
    CellnessModelInterface,
    DummyThresholdCellness,
    batch_predict,
    compute_tile_occupancy_from_map,
    normalize_image,
    tile_boxes_resolution_aware,
    tile_image_for_cnn,
)

from roi_selection_v2 import (
    D_MIN_NON_DENSE,
    composite_focus_measure,
    load_image_rgb,
    to_gray,
    visualize_tiles,
    select_topk_tiles_diverse,
)


logger = logging.getLogger("roi_selection_cnn")


@dataclass
class ROISelectorCNNConfig:
    tau_empty: float = 0.2
    beta: float = 0.4
    k_min: int = 1
    k_max: int = 100
    use_sum_occ: bool = True
    w_occ: float = 0.85
    w_focus: float = 0.15
    focus_on_candidates_only: bool = True
    candidate_multiplier: float = 2.0
    d_min_nondense: int = D_MIN_NON_DENSE
    dense_cap_ratio: float = 0.55
    dense_min_ratio: float | None = None
    downsample_for_cnn: int | None = None
    batch_size: int = 64


def _robust_focus_norm(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(vals)
    if not np.any(valid):
        return np.zeros_like(vals)
    v = vals[valid]
    med = float(np.median(v))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    denom = iqr if iqr > 1e-6 else 1.0
    z = (vals - med) / denom
    return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)


def _candidate_count(k: int, valid_count: int, multiplier: float | int) -> int:
    if valid_count <= 0:
        return 0
    if isinstance(multiplier, float):
        if multiplier <= 1.0:
            return max(1, int(np.ceil(valid_count * multiplier)))
        return max(1, int(np.ceil(max(1, k) * multiplier)))
    return max(1, int(multiplier))


def _downsample_image(image: np.ndarray, max_side: int | None) -> Tuple[np.ndarray, float]:
    if max_side is None:
        return image, 1.0
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return image, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    try:
        import cv2
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image
        pil = Image.fromarray((image * 255).astype(np.uint8))
        resized = np.asarray(pil.resize((new_w, new_h), resample=Image.BILINEAR)).astype(np.float32) / 255.0
    return resized, scale


def _scale_tile_boxes(tile_boxes: List[Tuple[int, int, int, int]], scale: float) -> List[Tuple[int, int, int, int]]:
    if scale == 1.0:
        return tile_boxes
    scaled = []
    for x0, y0, x1, y1 in tile_boxes:
        scaled.append(
            (
                int(round(x0 * scale)),
                int(round(y0 * scale)),
                int(round(x1 * scale)),
                int(round(y1 * scale)),
            )
        )
    return scaled


def _focus_for_tiles(image: np.ndarray, tile_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    img = normalize_image(image)
    img = img if img.ndim == 3 else np.repeat(img[:, :, None], 3, axis=2)
    focus_vals = []
    for x0, y0, x1, y1 in tile_boxes:
        tile = img[y0:y1, x0:x1]
        gray = to_gray(tile)
        focus_vals.append(float(composite_focus_measure(gray)))
    return np.asarray(focus_vals, dtype=np.float32)


def _predict_occ_from_map(
    image: np.ndarray,
    tile_boxes: List[Tuple[int, int, int, int]],
    predictor: CellnessModelInterface,
    downsample_for_cnn: int | None,
) -> np.ndarray:
    img = normalize_image(image)
    img_ds, scale = _downsample_image(img, downsample_for_cnn)
    p_map = predictor.predict_cellness_map(img_ds)
    boxes = _scale_tile_boxes(tile_boxes, scale)
    return compute_tile_occupancy_from_map(p_map, boxes)


def _predict_occ_from_tiles(
    image: np.ndarray,
    tile_boxes: List[Tuple[int, int, int, int]],
    predictor: CellnessModelInterface,
    downsample_for_cnn: int | None,
    batch_size: int,
) -> np.ndarray:
    tiles = tile_image_for_cnn(image, tile_boxes, downsample=downsample_for_cnn)
    return batch_predict(predictor.predict_tile_occupancy, tiles, batch_size=batch_size)


def select_rois_cnn(
    image: np.ndarray,
    calibration: Dict[str, float] | None,
    cfg: ROISelectorCNNConfig,
    cellness_predictor: CellnessModelInterface,
    focus_fn: Callable[[np.ndarray], float] | None = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    focus_fn = focus_fn or (lambda gray: composite_focus_measure(gray))

    tile_boxes, grid_h, grid_w = tile_boxes_resolution_aware(image.shape)
    n_tiles = len(tile_boxes)
    if n_tiles == 0:
        return {
            "selected_indices": [],
            "tile_boxes": [],
            "grid_h": 0,
            "grid_w": 0,
            "occ": np.array([], dtype=np.float32),
            "focus": np.array([], dtype=np.float32),
            "score": np.array([], dtype=np.float32),
            "k": 0,
            "n_nonempty": 0,
            "debug": {"reason": "no_tiles"},
        }

    if calibration:
        tau_empty = float(calibration.get("tau_empty", cfg.tau_empty))
        beta = float(calibration.get("beta", cfg.beta))
    else:
        tau_empty = cfg.tau_empty
        beta = cfg.beta

    t_occ_start = time.perf_counter()
    if cellness_predictor.has_map:
        occ = _predict_occ_from_map(
            image,
            tile_boxes,
            cellness_predictor,
            cfg.downsample_for_cnn,
        )
    else:
        occ = _predict_occ_from_tiles(
            image,
            tile_boxes,
            cellness_predictor,
            cfg.downsample_for_cnn,
            cfg.batch_size,
        )
    t_occ = time.perf_counter() - t_occ_start

    valid = occ >= tau_empty
    n_valid = int(valid.sum())
    if cfg.use_sum_occ:
        k_est = int(round(beta * float(occ[valid].sum()))) if n_valid else 0
    else:
        k_est = int(round(beta * n_valid)) if n_valid else 0

    k = max(cfg.k_min, min(k_est, cfg.k_max, n_valid)) if n_valid else 0
    if cfg.dense_cap_ratio and n_tiles > 0:
        k = min(k, int(np.ceil(cfg.dense_cap_ratio * n_tiles)))
    if cfg.dense_min_ratio and n_tiles > 0 and n_valid > 0:
        k = max(k, int(np.ceil(cfg.dense_min_ratio * n_tiles)))
        k = min(k, n_valid)

    if n_valid < cfg.k_min and n_tiles > 0:
        logger.warning("valid tiles < k_min; falling back to focus-only selection")
        focus_all = _focus_for_tiles(image, tile_boxes)
        order = np.argsort(-focus_all)
        k_fallback = min(max(cfg.k_min, 1), n_tiles)
        selected = order[:k_fallback].tolist()
        return {
            "selected_indices": selected,
            "tile_boxes": tile_boxes,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "occ": occ,
            "focus": focus_all,
            "score": focus_all,
            "k": k_fallback,
            "n_nonempty": n_valid,
            "debug": {
                "tau_empty": tau_empty,
                "beta": beta,
                "k_est": k_est,
                "fallback_focus": True,
            },
        }

    if n_valid == 0 or k == 0:
        return {
            "selected_indices": [],
            "tile_boxes": tile_boxes,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "occ": occ,
            "focus": np.full_like(occ, np.nan),
            "score": occ,
            "k": 0,
            "n_nonempty": n_valid,
            "debug": {"tau_empty": tau_empty, "beta": beta, "k_est": k_est},
        }

    valid_indices = np.where(valid)[0]
    occ_valid = occ[valid]
    order = valid_indices[np.argsort(-occ_valid)]
    cand_count = min(len(order), _candidate_count(k, len(order), cfg.candidate_multiplier))
    candidates = order[:cand_count]

    focus_vals = np.full(n_tiles, np.nan, dtype=np.float32)
    t_focus_start = time.perf_counter()
    if cfg.focus_on_candidates_only:
        for idx in candidates:
            x0, y0, x1, y1 = tile_boxes[int(idx)]
            tile = normalize_image(image)[y0:y1, x0:x1]
            tile = tile if tile.ndim == 3 else np.repeat(tile[:, :, None], 3, axis=2)
            focus_vals[int(idx)] = float(focus_fn(to_gray(tile)))
    else:
        focus_vals = _focus_for_tiles(image, tile_boxes)
    t_focus = time.perf_counter() - t_focus_start

    focus_norm = _robust_focus_norm(focus_vals)
    focus_norm[np.isnan(focus_vals)] = 0.0

    score = (cfg.w_occ * occ) + (cfg.w_focus * focus_norm)
    score[~valid] = -np.inf

    selected, d_min_used = select_topk_tiles_diverse(
        score,
        grid_h,
        grid_w,
        k=k,
        d_min=cfg.d_min_nondense,
    )
    t_total = time.perf_counter() - t0

    return {
        "selected_indices": selected,
        "tile_boxes": tile_boxes,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "occ": occ,
        "focus": focus_vals,
        "score": score,
        "k": k,
        "n_nonempty": n_valid,
        "debug": {
            "tau_empty": tau_empty,
            "beta": beta,
            "k_est": k_est,
            "candidates": int(cand_count),
            "d_min_used": int(d_min_used),
            "timing_s": {"occ": t_occ, "focus": t_focus, "total": t_total},
        },
    }


def _load_predictor(path: str | None, mode: str, threshold: float) -> CellnessModelInterface:
    if not path:
        logger.info("using dummy threshold predictor")
        return DummyThresholdCellness(threshold=threshold)
    module_path, _, func_name = path.partition(":")
    if not func_name:
        raise ValueError("--cellness-predictor must be in module:callable form")
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise ValueError("cellness predictor is not callable")
    return CellnessModelInterface.from_callable(fn, mode=mode)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CNN-dominant ROI selection (cellness gate + focus tie-breaker)")
    ap.add_argument("--manifests", nargs="+", required=True, help="Manifest CSV(s)")
    ap.add_argument("--path-col", default="image_path", help="Column name for image path")
    ap.add_argument("--dataset-col", default="dataset", help="Dataset column (optional)")
    ap.add_argument("--calibration-csv", default=None, help="Optional CNN-light calibration CSV")
    ap.add_argument("--cellness-predictor", default=None, help="Python callable: module.path:fn")
    ap.add_argument("--predictor-mode", default="auto", choices=["auto", "map", "tiles"])
    ap.add_argument("--dummy-threshold", type=float, default=0.5, help="Dummy predictor threshold")
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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--viz-dir", default=None, help="Optional visualization directory")
    ap.add_argument("--tiles-dir", default=None, help="Optional directory to save *_tiles.npy")
    ap.add_argument("--diag-prefix", default=None, help="Optional diagnostics CSV prefix")
    ap.add_argument("--limit-images", type=int, default=None)
    return ap.parse_args()


def _load_calibration(path: str | None) -> Dict[str, float] | None:
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


def run_cli(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    predictor = _load_predictor(args.cellness_predictor, args.predictor_mode, args.dummy_threshold)
    calib = _load_calibration(args.calibration_csv)
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
        batch_size=args.batch_size,
    )

    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
    tiles_root = args.tiles_dir or args.viz_dir
    if tiles_root:
        os.makedirs(tiles_root, exist_ok=True)

    diag_rows: List[Dict[str, Any]] = []
    for man_path in args.manifests:
        df = pd.read_csv(man_path)
        if args.limit_images:
            df = df.head(args.limit_images)

        man_base = os.path.splitext(os.path.basename(man_path))[0]
        viz_dir = os.path.join(args.viz_dir, man_base) if args.viz_dir else None
        tiles_dir = os.path.join(tiles_root, man_base) if tiles_root else None
        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
        if tiles_dir:
            os.makedirs(tiles_dir, exist_ok=True)

        for _, row in df.iterrows():
            img_path = row[args.path_col]
            if not os.path.isfile(img_path):
                logger.warning("missing image: %s", img_path)
                continue
            img = load_image_rgb(img_path)
            res = select_rois_cnn(img, calib, cfg, predictor)
            selected = res["selected_indices"]
            if tiles_dir:
                base = os.path.splitext(os.path.basename(img_path))[0]
                tiles_path = os.path.join(tiles_dir, f"{base}_tiles.npy")
                tile_boxes = res["tile_boxes"]
                tiles = []
                for idx in selected:
                    x0, y0, x1, y1 = tile_boxes[int(idx)]
                    tiles.append(img[y0:y1, x0:x1])
                np.save(tiles_path, np.asarray(tiles, dtype=np.float32))
                logger.info("saved %s", tiles_path)
            if viz_dir:
                try:
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    out_path = os.path.join(viz_dir, f"{base}_cnn.png")
                    tile_boxes, gh, gw = tile_boxes_resolution_aware(img.shape)
                    tiles = []
                    for x0, y0, x1, y1 in tile_boxes:
                        tiles.append(img[y0:y1, x0:x1])
                    tiles_np = np.asarray(tiles, dtype=np.float32)
                    scores = res["score"]
                    viz = visualize_tiles(
                        img,
                        tiles_np,
                        scores,
                        np.asarray(selected, dtype=np.int32),
                        gh,
                        gw,
                        sparsity_label=None,
                        sparsity_metric=None,
                        sparsity_method=None,
                        k_selected=res["k"],
                        sumS=float(np.sum(scores[np.isfinite(scores)])),
                    )
                    import cv2
                    cv2.imwrite(out_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    logger.info("saved %s", out_path)
                except Exception as exc:
                    logger.warning("viz failed for %s: %s", img_path, exc)
            diag_rows.append(
                {
                    "image_path": img_path,
                    "k": res["k"],
                    "n_nonempty": res["n_nonempty"],
                    "tau_empty": res["debug"].get("tau_empty"),
                    "beta": res["debug"].get("beta"),
                }
            )

    if args.diag_prefix and diag_rows:
        out_csv = f"{args.diag_prefix}_cnn.csv"
        pd.DataFrame(diag_rows).to_csv(out_csv, index=False)
        logger.info("saved diagnostics %s", out_csv)


def main() -> None:
    run_cli(parse_args())


if __name__ == "__main__":
    main()
