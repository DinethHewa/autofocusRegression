#!/usr/bin/env python3
"""
Lightweight calibration for CNN occupancy-based ROI selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform as py_platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

from roi_cnn_occupancy import (
    CellnessModelInterface,
    DummyThresholdCellness,
    compute_tile_occupancy_from_map,
    normalize_image,
    tile_boxes_resolution_aware,
    tile_image_for_cnn,
)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _hash_file_sha256(path: str | Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        _warn(f"failed to hash manifest: {exc}")
        return ""


def _find_git_root(start: Path) -> Path | None:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return None


def _get_git_commit() -> str:
    try:
        root = _find_git_root(Path(__file__).resolve())
        if root is None:
            return ""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except Exception:
        return ""


def _compute_k_from_occ(
    occ: np.ndarray,
    tau_empty: float,
    beta: float,
    k_min: int,
    k_max: int,
    use_sum_occ: bool,
) -> tuple[int, int]:
    occ = np.asarray(occ, dtype=np.float32).reshape(-1)
    if occ.size == 0:
        return 0, 0
    valid = occ >= tau_empty
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0, 0
    if use_sum_occ:
        k_est = int(round(beta * float(occ[valid].sum())))
    else:
        k_est = int(round(beta * n_valid))
    k = max(k_min, min(k_est, k_max, n_valid))
    return k, n_valid


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def _write_meta_json(
    out_path: str,
    args: argparse.Namespace,
    provenance: Dict[str, object],
    summary: Dict[str, object],
    failures: Dict[str, object],
) -> None:
    payload = {
        "args": vars(args),
        "provenance": provenance,
        "summary": summary,
        "failures": failures,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(_sanitize_for_json(payload), handle, indent=2)


def _write_diagnostics_csv(out_path: str, rows: List[Dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _load_predictor(path: str | None, mode: str, threshold: float) -> CellnessModelInterface:
    """Load a callable predictor.

    Notes:
      - "mode" applies only to callable predictors.
      - "auto" cannot be inferred reliably from a Python callable; we treat it as "map" and warn.
    """
    if not path:
        return DummyThresholdCellness(threshold=threshold)
    if mode == "auto":
        _warn(
            'predictor-mode="auto" cannot be inferred for Python callables; treating as "map". '
            "Use --predictor-mode tiles if your callable expects tile batches."
        )
        mode = "map"
    module_path, _, func_name = path.partition(":")
    if not func_name:
        raise ValueError("--cellness-predictor must be in module:callable form")
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name)
    return CellnessModelInterface.from_callable(fn, mode=mode)


def resolve_default_model_path() -> Path:
    base = Path(__file__).resolve().parents[2]
    return base / "data" / "roi_tile_benchmark" / "runs" / "cnn" / "best_model.keras"


def load_keras_model(model_path: Path):
    if tf is None:
        raise ImportError("TensorFlow is required to load a Keras model.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def infer_model_shape(model, input_size: int | None) -> tuple[int, int, int]:
    if input_size is not None:
        return input_size, input_size, 3
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) != 4:
        return 224, 224, 3
    h = shape[1] or 224
    w = shape[2] or 224
    c = shape[3] or 3
    return int(h), int(w), int(c)


def prepare_tiles_for_model(tiles: np.ndarray, target_hw: tuple[int, int], channels: int) -> np.ndarray:
    arr = tiles.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, :, None]
    if channels == 1 and arr.shape[-1] == 3:
        arr = np.mean(arr, axis=-1, keepdims=True)
    elif channels == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    h, w = target_hw
    if arr.shape[1] != h or arr.shape[2] != w:
        if tf is None:
            raise ImportError("TensorFlow is required for resizing tiles.")
        arr = tf.image.resize(arr, (h, w)).numpy()
    return arr.astype(np.float32)


def make_keras_tile_predictor(model, target_hw: tuple[int, int], channels: int, batch_size: int):
    def _predict_tiles(tiles: np.ndarray) -> np.ndarray:
        if tiles.size == 0:
            return np.zeros((0,), dtype=np.float32)
        arr = prepare_tiles_for_model(tiles, target_hw=target_hw, channels=channels)
        preds = model.predict(arr, batch_size=batch_size, verbose=0)
        return np.asarray(preds).reshape(-1)

    return _predict_tiles


def build_predictor(args: argparse.Namespace) -> tuple[CellnessModelInterface, Dict[str, str]]:
    if args.cellness_predictor:
        predictor = _load_predictor(args.cellness_predictor, args.predictor_mode, args.dummy_threshold)
        return predictor, {"predictor_source": "callable", "model_path": ""}

    model_path = Path(args.model_path).expanduser().resolve()
    model = load_keras_model(model_path)
    input_h, input_w, channels = infer_model_shape(model, args.model_input_size)
    predict_tiles = make_keras_tile_predictor(
        model,
        target_hw=(input_h, input_w),
        channels=channels,
        batch_size=args.model_batch_size,
    )
    predictor = CellnessModelInterface.from_callable(predict_tiles, mode="tiles")
    info = {
        "predictor_source": "keras_model",
        "model_path": str(model_path),
        "model_input_size": str(args.model_input_size or input_h),
        "model_batch_size": str(args.model_batch_size),
    }
    return predictor, info


def _load_calibration_images(manifest: str, path_col: str, limit: int | None) -> List[str]:
    df = pd.read_csv(manifest)
    if limit:
        df = df.head(limit)
    return df[path_col].astype(str).tolist()


def _predict_occ_for_image(
    image: np.ndarray,
    predictor: CellnessModelInterface,
    strict_probabilities: bool,
) -> np.ndarray:
    tile_boxes, _, _ = tile_boxes_resolution_aware(image.shape)
    if predictor.has_map:
        p_map = predictor.predict_cellness_map(image)
        p_map = validate_probability_map(p_map, strict=strict_probabilities)
        return compute_tile_occupancy_from_map(p_map, tile_boxes)
    if predictor.has_tiles:
        tiles = tile_image_for_cnn(image, tile_boxes)
        occ = predictor.predict_tile_occupancy(tiles)
        occ = validate_probability_map(occ, strict=strict_probabilities)
        return occ.astype(np.float32)
    raise RuntimeError("Predictor has neither map nor tile outputs.")


def validate_probability_map(p_map: np.ndarray, strict: bool = False) -> np.ndarray:
    p_map = np.asarray(p_map, dtype=np.float32)
    if not np.isfinite(p_map).all():
        raise ValueError("predictor output contains NaN/Inf")
    min_v = float(p_map.min()) if p_map.size else 0.0
    max_v = float(p_map.max()) if p_map.size else 0.0
    if min_v < -0.05 or max_v > 1.05:
        msg = f"predictor output not in [0,1] (min={min_v:.4f}, max={max_v:.4f})"
        if strict:
            raise ValueError(msg)
        _warn(msg + "; clipping to [0,1]")
        p_map = np.clip(p_map, 0.0, 1.0)
    std_v = float(p_map.std()) if p_map.size else 0.0
    if std_v < 1e-6:
        _warn("predictor output nearly constant (std < 1e-6)")
    return p_map


def calibrate_tau_empty_valley(
    occ_values: np.ndarray,
    occ_clip: float = 0.2,
    bins: int = 256,
    fallback_percentile: float = 5.0,
    min_samples: int = 500,
) -> tuple[float, str, Dict[str, float]]:
    occ = np.asarray(occ_values, dtype=np.float32).reshape(-1)
    occ = occ[np.isfinite(occ)]
    diagnostics: Dict[str, float] = {
        "occ_clip": float(occ_clip),
        "bins": float(bins),
        "occ_p1": float(np.percentile(occ, 1.0)) if occ.size else 0.0,
        "occ_p5": float(np.percentile(occ, 5.0)) if occ.size else 0.0,
        "occ_p50": float(np.percentile(occ, 50.0)) if occ.size else 0.0,
        "occ_p95": float(np.percentile(occ, 95.0)) if occ.size else 0.0,
        "occ_p99": float(np.percentile(occ, 99.0)) if occ.size else 0.0,
    }
    if occ.size < min_samples:
        tau = float(np.percentile(occ, fallback_percentile)) if occ.size else 0.0
        diagnostics["peak_bin"] = -1
        diagnostics["valley_bin"] = -1
        diagnostics["empty_rate_at_tau"] = float(np.mean(occ <= tau)) if occ.size else 0.0
        return tau, "fallback_percentile", diagnostics

    occ_clip = float(occ_clip)
    occ_c = np.clip(occ, 0.0, occ_clip)
    counts, edges = np.histogram(occ_c, bins=bins, range=(0.0, occ_clip))
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_idx = int(np.argmax(counts)) if counts.size else -1
    diagnostics["peak_bin"] = float(peak_idx)

    valley_idx = None
    min_idx = None
    min_val = None
    nondec_streak = 0
    saw_increase = False
    for i in range(peak_idx + 1, len(counts)):
        if min_idx is None:
            min_idx = i
            min_val = counts[i]
        if min_val is None or counts[i] < min_val:
            min_val = counts[i]
            min_idx = i
        if counts[i] >= counts[i - 1]:
            nondec_streak += 1
            if counts[i] > counts[i - 1]:
                saw_increase = True
        else:
            nondec_streak = 0
            saw_increase = False
        if nondec_streak >= 2 and saw_increase and min_idx is not None:
            valley_idx = min_idx
            break

    diagnostics["valley_bin"] = float(valley_idx) if valley_idx is not None else -1.0
    tau = float(centers[valley_idx]) if valley_idx is not None else -1.0
    if not (0.0 < tau < occ_clip):
        tau = float(np.percentile(occ, fallback_percentile))
        diagnostics["empty_rate_at_tau"] = float(np.mean(occ <= tau)) if occ.size else 0.0
        return tau, "fallback_percentile", diagnostics
    diagnostics["empty_rate_at_tau"] = float(np.mean(occ <= tau)) if occ.size else 0.0
    return tau, "valley", diagnostics


def calibrate_tau_empty_percentile(
    occ_values: np.ndarray,
    percentile: float = 5.0,
) -> tuple[float, str, Dict[str, float]]:
    occ = np.asarray(occ_values, dtype=np.float32).reshape(-1)
    occ = occ[np.isfinite(occ)]
    tau = float(np.percentile(occ, percentile)) if occ.size else 0.0
    diag: Dict[str, float] = {"tau_percentile": float(percentile)}
    return tau, "percentile", diag


def calibrate_tau_empty_otsu(
    occ_values: np.ndarray,
    occ_clip: float = 1.0,
    bins: int = 512,
    min_samples: int = 500,
) -> tuple[float, str, Dict[str, float]]:
    """Otsu threshold on occupancy values (unsupervised bimodal separation)."""
    occ = np.asarray(occ_values, dtype=np.float32).reshape(-1)
    occ = occ[np.isfinite(occ)]
    diagnostics: Dict[str, float] = {"occ_clip_otsu": float(occ_clip), "bins_otsu": float(bins)}
    if occ.size < min_samples:
        tau = float(np.percentile(occ, 5.0)) if occ.size else 0.0
        diagnostics["otsu_fallback_p5"] = 1.0
        return tau, "otsu_fallback_p5", diagnostics
    x = np.clip(occ, 0.0, float(occ_clip))
    hist, edges = np.histogram(x, bins=bins, range=(0.0, float(occ_clip)))
    hist = hist.astype(np.float64)
    if hist.sum() <= 0:
        return 0.0, "otsu_empty_hist", diagnostics
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (0.5 * (edges[:-1] + edges[1:])))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 1e-12] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b2)) if np.isfinite(sigma_b2).any() else 0
    tau = float(0.5 * (edges[idx] + edges[idx + 1]))
    diagnostics["otsu_idx"] = float(idx)
    return tau, "otsu", diagnostics


def calibrate_tau_empty_dispatch(
    occ_values: np.ndarray,
    method: str,
    percentile: float = 5.0,
) -> tuple[float, str, Dict[str, float]]:
    """Select tau_empty estimation method without breaking output compatibility."""
    method = (method or "valley").lower()
    if method == "valley":
        return calibrate_tau_empty_valley(occ_values, fallback_percentile=percentile)
    if method == "otsu":
        return calibrate_tau_empty_otsu(occ_values)
    if method == "percentile":
        return calibrate_tau_empty_percentile(occ_values, percentile=percentile)
    _warn(f"Unknown --tau-method={method!r}; falling back to valley.")
    return calibrate_tau_empty_valley(occ_values, fallback_percentile=percentile)


def calibrate_beta_for_target_k(
    occ_by_image: List[np.ndarray],
    tau_empty: float,
    target_avg_k: float,
    use_sum_occ: bool,
    k_min: int,
    k_max: int,
    beta_init: float,
    beta_high_max: float,
) -> tuple[float, float, bool]:
    if not occ_by_image:
        return 0.0, 0.0, False

    def _avg_k(beta_val: float) -> float:
        ks = []
        for occ in occ_by_image:
            valid = occ >= tau_empty
            if use_sum_occ:
                k_est = int(round(beta_val * float(occ[valid].sum()))) if np.any(valid) else 0
            else:
                k_est = int(round(beta_val * int(valid.sum()))) if np.any(valid) else 0
            k = max(k_min, min(k_est, k_max, int(valid.sum()))) if np.any(valid) else 0
            ks.append(k)
        return float(np.mean(ks)) if ks else 0.0

    high = float(beta_init) if beta_init > 0 else 0.1
    beta_high_max = float(beta_high_max)
    avg_k_high = _avg_k(high)
    while avg_k_high < target_avg_k and high < beta_high_max:
        high = min(high * 2.0, beta_high_max)
        avg_k_high = _avg_k(high)

    if avg_k_high < target_avg_k:
        _warn(
            "beta search saturated at beta_high_max="
            f"{beta_high_max:.3f}; avg_k={avg_k_high:.2f} < target {target_avg_k}"
        )
        return float(beta_high_max), avg_k_high, True

    low = 0.0
    for _ in range(30):
        mid = 0.5 * (low + high)
        avg_k = _avg_k(mid)
        if avg_k < target_avg_k:
            low = mid
        else:
            high = mid
    achieved = _avg_k(high)
    return float(high), achieved, False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Light calibration for CNN occupancy ROI selection")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--path-col", default="image_path")
    ap.add_argument("--cellness-predictor", default=None, help="Python callable: module.path:fn")
    ap.add_argument(
        "--predictor-mode",
        default="map",
        choices=["map", "tiles", "auto"],
        help="Prediction interface for callable predictors. Use auto to prefer map if available, else tiles.",
    )
    ap.add_argument("--model-path", default=str(resolve_default_model_path()))
    ap.add_argument("--model-input-size", type=int, default=None)
    ap.add_argument("--model-batch-size", type=int, default=64)
    ap.add_argument("--dummy-threshold", type=float, default=0.5)
    ap.add_argument("--target-avg-k", type=float, default=20.0)
    ap.add_argument("--use-sum-occ", dest="use_sum_occ", action="store_true", help="Compute k from beta * sum(occ_valid).")
    ap.add_argument(
        "--no-use-sum-occ",
        dest="use_sum_occ",
        action="store_false",
        help="Compute k from beta * N_valid instead of sum(occ_valid).",
    )
    ap.set_defaults(use_sum_occ=True)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=100)
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument(
        "--tau-method",
        default="valley",
        choices=["valley", "otsu", "percentile"],
        help="Method to estimate tau_empty from occupancy distribution.",
    )
    ap.add_argument(
        "--tau-percentile",
        type=float,
        default=5.0,
        help="Percentile used when --tau-method=percentile and as fallback for valley.",
    )
    ap.add_argument(
        "--out-methods-txt",
        default="",
        help="If set, write a Methods-appendix-ready calibration diagnostics block to this path.",
    )
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--strict-probabilities", action="store_true", default=False)
    ap.add_argument("--beta-init", type=float, default=0.1)
    ap.add_argument("--beta-high-max", type=float, default=100.0)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-split", type=float, default=0.0)
    ap.add_argument("--val-max-images", type=int, default=0)
    ap.add_argument("--out-meta-json", default="")
    ap.add_argument("--predictor-id", default="")
    ap.add_argument("--manifest-hash", action="store_true", default=False)
    ap.add_argument("--export-diagnostics-csv", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    predictor, predictor_info = build_predictor(args)
    paths = _load_calibration_images(args.manifest, args.path_col, args.limit_images)

    rng = np.random.default_rng(args.seed)
    val_paths: List[str] = []
    train_paths: List[str] = []
    if args.val_split and args.val_split > 0 and len(paths) > 0:
        idx = rng.permutation(len(paths))
        val_count = int(round(args.val_split * len(paths)))
        val_count = max(0, min(val_count, len(paths)))
        val_paths = [paths[i] for i in idx[:val_count]]
        train_paths = [paths[i] for i in idx[val_count:]]
    else:
        train_paths = paths
        val_paths = []
    if args.val_max_images and args.val_max_images > 0 and len(val_paths) > args.val_max_images:
        val_paths = val_paths[: args.val_max_images]
    if args.val_split > 0 and not val_paths:
        _warn("val_split produced empty validation set; skipping validation.")

    occ_values = []
    occ_by_image_train: List[np.ndarray] = []
    occ_by_image_val: List[np.ndarray] = []
    occ_by_path: Dict[str, np.ndarray] = {}
    split_by_path: Dict[str, str] = {}
    images_seen = 0
    images_failed = 0
    train_seen = 0
    train_used = 0
    train_failed = 0
    val_seen = 0
    val_used = 0
    val_failed = 0
    failed_examples: List[tuple[str, str]] = []
    failed_records: List[tuple[str, str, str]] = []

    def _process_paths(paths_to_process: List[str], split: str) -> None:
        nonlocal images_seen, images_failed, train_seen, train_used, train_failed
        nonlocal val_seen, val_used, val_failed
        for path in paths_to_process:
            images_seen += 1
            split_by_path[path] = split
            if split == "train":
                train_seen += 1
            else:
                val_seen += 1
            try:
                from roi_selection_v2 import load_image_rgb
                img = load_image_rgb(path)
                img = normalize_image(img)
                occ = _predict_occ_for_image(img, predictor, strict_probabilities=args.strict_probabilities)
                if args.export_diagnostics_csv:
                    occ_by_path[path] = occ
                if split == "train":
                    occ_by_image_train.append(occ)
                    occ_values.append(occ)
                    train_used += 1
                else:
                    occ_by_image_val.append(occ)
                    val_used += 1
            except Exception as exc:
                images_failed += 1
                if split == "train":
                    train_failed += 1
                else:
                    val_failed += 1
                failed_records.append((split, path, str(exc)))
                if len(failed_examples) < 10:
                    failed_examples.append((path, str(exc)))
                continue

    _process_paths(train_paths, "train")
    _process_paths(val_paths, "val")

    print(
        "[INFO] Calibration images: train_used={}, train_seen={}, train_failed={}, "
        "val_used={}, val_seen={}, val_failed={}".format(
            train_used,
            train_seen,
            train_failed,
            val_used,
            val_seen,
            val_failed,
        ),
        file=sys.stderr,
    )
    if failed_examples:
        print("[WARN] Example failures:", file=sys.stderr)
        for path, err in failed_examples:
            print(f"  - {path}: {err}", file=sys.stderr)

    min_images_required = 20 if train_seen < 200 else max(20, int(0.1 * train_seen))
    if train_used < min_images_required:
        raise RuntimeError(
            "Insufficient images for calibration; "
            f"used={train_used}, seen={train_seen}, failed={train_failed}, "
            f"required>={min_images_required}."
        )

    all_occ = np.concatenate(occ_values, axis=0) if occ_values else np.array([], dtype=np.float32)
    tau_valley, _, tau_diag_valley = calibrate_tau_empty_valley(all_occ, fallback_percentile=args.tau_percentile)
    tau_otsu, _, tau_diag_otsu = calibrate_tau_empty_otsu(all_occ)
    tau_pct, _, tau_diag_pct = calibrate_tau_empty_percentile(all_occ, percentile=args.tau_percentile)
    tau_empty, tau_method, tau_diag = calibrate_tau_empty_dispatch(
        all_occ, method=args.tau_method, percentile=args.tau_percentile
    )
    beta, achieved_avg_k, saturated_beta = calibrate_beta_for_target_k(
        occ_by_image_train,
        tau_empty=tau_empty,
        target_avg_k=args.target_avg_k,
        use_sum_occ=args.use_sum_occ,
        k_min=args.k_min,
        k_max=args.k_max,
        beta_init=args.beta_init,
        beta_high_max=args.beta_high_max,
    )

    occ_p1 = float(np.percentile(all_occ, 1.0)) if all_occ.size else 0.0
    occ_p5 = float(np.percentile(all_occ, 5.0)) if all_occ.size else 0.0
    occ_p50 = float(np.percentile(all_occ, 50.0)) if all_occ.size else 0.0
    occ_p95 = float(np.percentile(all_occ, 95.0)) if all_occ.size else 0.0
    occ_p99 = float(np.percentile(all_occ, 99.0)) if all_occ.size else 0.0
    empty_rate_at_tau = float(np.mean(all_occ <= tau_empty)) if all_occ.size else 0.0
    empty_rate_valley = float(np.mean(all_occ <= tau_valley)) if all_occ.size else 0.0
    empty_rate_otsu = float(np.mean(all_occ <= tau_otsu)) if all_occ.size else 0.0
    empty_rate_pct = float(np.mean(all_occ <= tau_pct)) if all_occ.size else 0.0

    val_metrics: Dict[str, object] = {}
    if args.val_split and args.val_split > 0:
        val_metrics["val_images_used"] = val_used
        val_metrics["val_images_failed"] = val_failed
        if val_used == 0:
            _warn("No validation images used; skipping validation metrics.")
            val_metrics.update(
                {
                    "val_empty_rate": float("nan"),
                    "val_avg_valid_tiles": float("nan"),
                    "val_avg_k_achieved": float("nan"),
                    "val_k_std": float("nan"),
                    "val_saturation_rate": float("nan"),
                }
            )
        else:
            total_tiles = 0
            total_empty = 0
            valid_counts = []
            k_values = []
            saturated = []
            for occ in occ_by_image_val:
                occ = np.asarray(occ, dtype=np.float32).reshape(-1)
                total_tiles += int(occ.size)
                total_empty += int(np.sum(occ < tau_empty))
                k, n_valid = _compute_k_from_occ(
                    occ,
                    tau_empty=tau_empty,
                    beta=beta,
                    k_min=args.k_min,
                    k_max=args.k_max,
                    use_sum_occ=args.use_sum_occ,
                )
                valid_counts.append(n_valid)
                k_values.append(k)
                saturated.append(1 if (args.k_max > 0 and k == args.k_max) else 0)
            val_metrics.update(
                {
                    "val_empty_rate": float(total_empty / total_tiles) if total_tiles else 0.0,
                    "val_avg_valid_tiles": float(np.mean(valid_counts)) if valid_counts else 0.0,
                    "val_avg_k_achieved": float(np.mean(k_values)) if k_values else 0.0,
                    "val_k_std": float(np.std(k_values)) if k_values else 0.0,
                    "val_saturation_rate": float(np.mean(saturated)) if saturated else 0.0,
                }
            )

    predictor_spec = (
        args.cellness_predictor
        if args.cellness_predictor
        else f"keras_model:{args.model_path}" if args.model_path else "DummyThresholdCellness"
    )
    manifest_path = str(Path(args.manifest).expanduser().resolve())
    manifest_sha256 = _hash_file_sha256(manifest_path) if args.manifest_hash else ""
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": py_platform.python_version(),
        "platform": py_platform.platform(),
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "predictor_spec": predictor_spec,
        "predictor_mode": args.predictor_mode,
        "predictor_id": args.predictor_id,
        "seed": int(args.seed),
        "val_split": float(args.val_split),
        "val_max_images": int(args.val_max_images),
        "script_name": Path(__file__).name,
        "git_commit": _get_git_commit(),
    }

    summary = {
        "tau_empty": tau_empty,
        "tau_empty_method": tau_method or "unknown",
        "empty_rate_at_tau": empty_rate_at_tau,
        "tau_peak_bin": tau_diag.get("peak_bin", -1.0),
        "tau_valley_bin": tau_diag.get("valley_bin", -1.0),
        "tau_occ_clip": tau_diag.get("occ_clip", 0.0),
        "tau_bins": tau_diag.get("bins", 0.0),
        "tau_method_requested": args.tau_method,
        "tau_percentile": float(args.tau_percentile),
        "tau_empty_valley": float(tau_valley),
        "tau_empty_otsu": float(tau_otsu),
        "tau_empty_percentile": float(tau_pct),
        "empty_rate_valley": float(empty_rate_valley),
        "empty_rate_otsu": float(empty_rate_otsu),
        "empty_rate_percentile": float(empty_rate_pct),
        "tau_otsu_idx": tau_diag_otsu.get("otsu_idx", -1.0),
        "beta": beta,
        "achieved_avg_k": achieved_avg_k,
        "saturated_beta": bool(saturated_beta),
        "target_avg_k": args.target_avg_k,
        "use_sum_occ": bool(args.use_sum_occ),
        "occ_mean": float(all_occ.mean()) if all_occ.size else 0.0,
        "occ_median": float(np.median(all_occ)) if all_occ.size else 0.0,
        "occ_std": float(all_occ.std()) if all_occ.size else 0.0,
        "occ_p1": occ_p1,
        "occ_p5": occ_p5,
        "occ_p95": occ_p95,
        "occ_p99": occ_p99,
        "images_used": train_used,
        "images_seen": images_seen,
        "images_failed": images_failed,
        "strict_probabilities": bool(args.strict_probabilities),
        "beta_init": float(args.beta_init),
        "beta_high_max": float(args.beta_high_max),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "predictor_source": predictor_info.get("predictor_source", ""),
        "model_path": predictor_info.get("model_path", ""),
        "model_input_size": predictor_info.get("model_input_size", ""),
        "model_batch_size": predictor_info.get("model_batch_size", ""),
        "timestamp_utc": provenance["timestamp_utc"],
        "python_version": provenance["python_version"],
        "platform": provenance["platform"],
        "manifest_path": provenance["manifest_path"],
        "manifest_sha256": provenance["manifest_sha256"],
        "predictor_spec": provenance["predictor_spec"],
        "predictor_mode": provenance["predictor_mode"],
        "predictor_id": provenance["predictor_id"],
        "val_split": provenance["val_split"],
        "val_max_images": provenance["val_max_images"],
        "script_name": provenance["script_name"],
        "git_commit": provenance["git_commit"],
    }
    if val_metrics:
        summary.update(val_metrics)

    if args.bootstrap and args.bootstrap > 0 and train_used > 0:
        if args.bootstrap > 200:
            _warn("bootstrap is large; this may be slow")
        rng = np.random.default_rng(args.seed)
        tau_samples = []
        beta_samples = []
        avgk_samples = []
        for _ in range(int(args.bootstrap)):
            idx = rng.integers(0, len(occ_by_image_train), size=len(occ_by_image_train))
            sampled = [occ_by_image_train[i] for i in idx]
            sampled_occ = np.concatenate(sampled, axis=0) if sampled else np.array([], dtype=np.float32)
            tau_b, _, _ = calibrate_tau_empty_dispatch(
                sampled_occ, method=args.tau_method, percentile=args.tau_percentile
            )
            beta_b, avg_k_b, _ = calibrate_beta_for_target_k(
                sampled,
                tau_empty=tau_b,
                target_avg_k=args.target_avg_k,
                use_sum_occ=args.use_sum_occ,
                k_min=args.k_min,
                k_max=args.k_max,
                beta_init=args.beta_init,
                beta_high_max=args.beta_high_max,
            )
            tau_samples.append(tau_b)
            beta_samples.append(beta_b)
            avgk_samples.append(avg_k_b)
        summary.update(
            {
                "tau_empty_ci_low": float(np.percentile(tau_samples, 2.5)) if tau_samples else 0.0,
                "tau_empty_ci_high": float(np.percentile(tau_samples, 97.5)) if tau_samples else 0.0,
                "beta_ci_low": float(np.percentile(beta_samples, 2.5)) if beta_samples else 0.0,
                "beta_ci_high": float(np.percentile(beta_samples, 97.5)) if beta_samples else 0.0,
                "achieved_avg_k_ci_low": float(np.percentile(avgk_samples, 2.5)) if avgk_samples else 0.0,
                "achieved_avg_k_ci_high": float(np.percentile(avgk_samples, 97.5)) if avgk_samples else 0.0,
            }
        )

    if args.export_diagnostics_csv:
        diag_rows: List[Dict[str, object]] = []
        for path in train_paths + val_paths:
            split = split_by_path.get(path, "train")
            occ = occ_by_path.get(path)
            if occ is None:
                diag_rows.append(
                    {
                        "split": split,
                        "image_path": path,
                        "n_tiles": 0,
                        "occ_mean": float("nan"),
                        "occ_median": float("nan"),
                        "occ_p95": float("nan"),
                        "n_valid": float("nan"),
                        "k_achieved": float("nan"),
                        "empty_rate_image": float("nan"),
                        "status": "failed",
                    }
                )
                continue
            occ = np.asarray(occ, dtype=np.float32).reshape(-1)
            k, n_valid = _compute_k_from_occ(
                occ,
                tau_empty=tau_empty,
                beta=beta,
                k_min=args.k_min,
                k_max=args.k_max,
                use_sum_occ=args.use_sum_occ,
            )
            empty_rate_image = float(np.mean(occ < tau_empty)) if occ.size else 0.0
            diag_rows.append(
                {
                    "split": split,
                    "image_path": path,
                    "n_tiles": int(occ.size),
                    "occ_mean": float(occ.mean()) if occ.size else 0.0,
                    "occ_median": float(np.median(occ)) if occ.size else 0.0,
                    "occ_p95": float(np.percentile(occ, 95.0)) if occ.size else 0.0,
                    "n_valid": int(n_valid),
                    "k_achieved": int(k),
                    "empty_rate_image": empty_rate_image,
                    "status": "ok",
                }
            )
        _write_diagnostics_csv(args.export_diagnostics_csv, diag_rows)
        print(f"[INFO] Saved diagnostics to {args.export_diagnostics_csv}")

    pd.DataFrame([summary]).to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved calibration to {args.out_csv}")

    methods_block = "\n".join(
        [
            "[CALIBRATION DIAGNOSTICS - CNN Occupancy ROI Gating]",
            f"manifest={provenance['manifest_path']}",
            f"predictor_spec={provenance['predictor_spec']} (mode={provenance['predictor_mode']})",
            f"train_images_used={train_used}/{train_seen} (failed={train_failed})",
            (
                f"val_images_used={val_used}/{val_seen} (failed={val_failed})"
                if args.val_split and args.val_split > 0
                else "val_split=0 (no hold-out validation)"
            ),
            "occupancy_quantiles(p1,p5,p50,p95,p99)=({:.4g},{:.4g},{:.4g},{:.4g},{:.4g})".format(
                occ_p1, occ_p5, occ_p50, occ_p95, occ_p99
            ),
            (
                "tau_candidates: valley={:.6g} (empty_rate={:.3f}), "
                "otsu={:.6g} (empty_rate={:.3f}), p{}={:.6g} (empty_rate={:.3f})"
            ).format(
                tau_valley,
                empty_rate_valley,
                tau_otsu,
                empty_rate_otsu,
                args.tau_percentile,
                tau_pct,
                empty_rate_pct,
            ),
            f"tau_selected(method={tau_method}): tau_empty={tau_empty:.6g} (empty_rate={empty_rate_at_tau:.3f})",
            (
                "beta_selected: beta={:.6g}, target_avg_k={:.3f}, achieved_avg_k={:.3f}, "
                "saturated_beta={}".format(
                    beta, float(args.target_avg_k), achieved_avg_k, bool(saturated_beta)
                )
            ),
            f"k_rule: use_sum_occ={bool(args.use_sum_occ)}, k_min={args.k_min}, k_max={args.k_max}",
            (
                "provenance: timestamp_utc={}, git_commit={}, manifest_sha256={}".format(
                    provenance["timestamp_utc"], provenance["git_commit"], provenance["manifest_sha256"]
                )
            ),
        ]
    )
    print(methods_block, file=sys.stderr)
    if args.out_methods_txt:
        Path(args.out_methods_txt).expanduser().resolve().write_text(methods_block + "\n", encoding="utf-8")
        print(f"[INFO] Saved methods diagnostics to {args.out_methods_txt}")

    if args.out_meta_json:
        failures = {
            "train_seen": train_seen,
            "train_used": train_used,
            "train_failed": train_failed,
            "val_seen": val_seen,
            "val_used": val_used,
            "val_failed": val_failed,
            "failed_examples": failed_examples,
        }
        _write_meta_json(args.out_meta_json, args, provenance, summary, failures)
        print(f"[INFO] Saved metadata to {args.out_meta_json}")


if __name__ == "__main__":
    main()
