#!/usr/bin/env python3
"""
calibration_script_v2.py

Per-dataset calibration of raw priors and sparsity thresholds.
Uses best CNN model scores as F_raw (focus proxy).

CLI:
  --manifest <csv>
  --dataset-name <str>
  --path-col <col> (default: image_path)
  --max-images <int> (optional)
  --out-csv <path> (default: <dataset-name>_calibration.csv)
  --seed <int> (default: 42)
  --model-path <path> (default: best CNN model)
  --model-input-size <int> (optional override)
  --model-batch-size <int> (default: 64)
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.fft import fft2, fftshift
from PIL import Image, ImageFile

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------
# Constants (match roi_selection_v2)
# ----------------------------
TILE_SIZE = 200
SMALL_THRESH = 400
LARGE_THRESH = 2000
HFER_R0 = 0.3
P_EPS = 1e-6
ETA_GRAD = 0.6
ALPHA_FOCUS = 0.7
LAMBDA_DARK = 0.25
DARK_MU = 0.55
DARK_P = 2.0
THRESH_ABS_GRAD = 0.2

# Hybrid sparsity knobs (kept simple; calibrated per-dataset)
CANNY_LO = 50
CANNY_HI = 150
MAX_HYBRID_DOWNSAMPLE = 512


# ----------------------------
# CNN utilities
# ----------------------------
def resolve_default_model_path() -> Path:
    base = Path(__file__).resolve().parents[2]
    return base / "data" / "roi_tile_benchmark" / "runs" / "cnn" / "best_model.keras"


def load_keras_model(model_path: Path):
    if tf is None:
        raise ImportError("TensorFlow is required to load a Keras model.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def infer_model_shape(model, input_size: Optional[int]) -> Tuple[int, int, int]:
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


def prepare_tiles_for_model(tiles: List[np.ndarray], target_hw: Tuple[int, int], channels: int) -> np.ndarray:
    arr = np.asarray(tiles, dtype=np.float32)
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


def predict_tile_probs(
    model,
    tiles: List[np.ndarray],
    target_hw: Tuple[int, int],
    channels: int,
    batch_size: int,
) -> np.ndarray:
    if not tiles:
        return np.zeros((0,), dtype=np.float32)
    arr = prepare_tiles_for_model(tiles, target_hw=target_hw, channels=channels)
    preds = model.predict(arr, batch_size=batch_size, verbose=0)
    preds = np.asarray(preds).reshape(-1).astype(np.float32)
    return np.clip(preds, 0.0, 1.0)


# ----------------------------
# Image utilities
# ----------------------------
def load_image_rgb(path: str) -> np.ndarray:
    """
    Load image as RGB float32 in [0,1], handling 8-bit and 16-bit inputs.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)
        except Exception as e:
            raise FileNotFoundError(f"Could not read image: {path}") from e
        dtype = img.dtype
        img = img.astype(np.float32)
        if dtype == np.uint8:
            img /= 255.0
        elif dtype == np.uint16:
            max_val = float(np.max(img)) if img.size else 0.0
            scale = max_val if max_val > 0 else 65535.0
            img /= scale
        else:
            mx = float(np.max(img)) if img.size else 1.0
            if mx > 1.5:
                img /= 255.0
        return np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if dtype == np.uint8:
        img /= 255.0
    elif dtype == np.uint16:
        max_val = float(np.max(img)) if img.size else 0.0
        scale = max_val if max_val > 0 else 65535.0
        img /= scale
    else:
        mx = float(np.max(img)) if img.size else 1.0
        if mx > 1.5:
            img /= 255.0
    return np.clip(img, 0.0, 1.0)


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor((img_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return g.astype(np.float32) / 255.0


def _downsample_gray_for_metrics(gray01: np.ndarray, max_side: int = MAX_HYBRID_DOWNSAMPLE) -> np.ndarray:
    """Downsample gray image for robust global metrics (speed + stability)."""
    h, w = gray01.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray01
    scale = max_side / float(m)
    nh, nw = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
    return cv2.resize(gray01, (nw, nh), interpolation=cv2.INTER_AREA)


def compute_foreground_fraction(gray01: np.ndarray) -> float:
    """
    Foreground fraction via Otsu on inverted grayscale.
    Assumption: background tends to be brighter in smear fields.
    This is calibrated per dataset and used only as a *relative* occupancy cue.
    """
    g = _downsample_gray_for_metrics(gray01)
    u8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    inv = 255 - u8
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(mask.mean() / 255.0)


def compute_edge_occupancy(gray01: np.ndarray) -> float:
    """
    Edge occupancy using Canny edges after mild blur.
    Robust across stain differences; correlates with 'busy' cellular texture.
    """
    g = _downsample_gray_for_metrics(gray01)
    u8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (3, 3), 0)
    edges = cv2.Canny(u8, CANNY_LO, CANNY_HI)
    return float(edges.mean() / 255.0)


def _iqr(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def _z(val: float, med: float, iqr: float, eps: float = 1e-8) -> float:
    return (float(val) - float(med)) / (float(iqr) + eps)


def compute_hybrid_sparse_score(
    R10: float,
    Neff_norm: float,
    FF: float,
    EO: float,
    calib_meds: Dict[str, float],
    calib_iqrs: Dict[str, float],
) -> float:
    """
    Higher score => more sparse.
    Uses robust z-scores (median/IQR) and *inverts* the terms where low implies sparse.
    """
    zR10 = _z(R10, calib_meds["R10_median"], calib_iqrs["R10_iqr"])
    zNe = _z(Neff_norm, calib_meds["Neff_median"], calib_iqrs["Neff_iqr"])
    zFF = _z(FF, calib_meds["FF_median"], calib_iqrs["FF_iqr"])
    zEO = _z(EO, calib_meds["EO_median"], calib_iqrs["EO_iqr"])
    return float(zR10 - zNe - zFF - zEO)


def tile_image_resolution_aware(img_rgb: np.ndarray) -> Tuple[List[np.ndarray], int, int]:
    H, W, _ = img_rgb.shape
    tiles: List[np.ndarray] = []
    if min(H, W) < SMALL_THRESH:
        tiles.append(cv2.resize(img_rgb, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA))
        return tiles, 1, 1
    if H > LARGE_THRESH and W > LARGE_THRESH:
        grid_h = grid_w = 10
        ph, pw = H // grid_h, W // grid_w
        for i in range(grid_h):
            for j in range(grid_w):
                y0, y1 = i * ph, (i + 1) * ph
                x0, x1 = j * pw, (j + 1) * pw
                patch = img_rgb[y0:y1, x0:x1]
                tiles.append(cv2.resize(patch, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA))
        return tiles, grid_h, grid_w
    n = int(min(H // TILE_SIZE, W // TILE_SIZE))
    if n <= 0:
        tiles.append(cv2.resize(img_rgb, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA))
        return tiles, 1, 1
    grid_h = grid_w = n
    tile_h = H // n
    tile_w = W // n
    for i in range(grid_h):
        for j in range(grid_w):
            y0, y1 = i * tile_h, (i + 1) * tile_h
            x0, x1 = j * tile_w, (j + 1) * tile_w
            patch = img_rgb[y0:y1, x0:x1]
            tiles.append(cv2.resize(patch, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA))
    return tiles, grid_h, grid_w


# ----------------------------
# Focus / tissue measures (non-F)
# ----------------------------
def radial_profile(fft_mag: np.ndarray, num_bins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    h, w = fft_mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / (r.max() + 1e-6)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    m_r = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        mask = (r_norm >= bins[i]) & (r_norm < bins[i + 1])
        m_r[i] = float(np.mean(fft_mag[mask])) if np.any(mask) else 0.0
    return r_centers, m_r


def hfer(gray: np.ndarray, r0: float = HFER_R0) -> float:
    F = fft2(gray)
    mag2 = np.abs(fftshift(F)) ** 2
    r_centers, m_r = radial_profile(mag2, num_bins=64)
    total = float(m_r.sum()) + P_EPS
    high = float(m_r[r_centers >= r0].sum())
    return high / total


def brenner_gradient(gray: np.ndarray) -> float:
    shifted = np.roll(gray, -2, axis=1)
    diff = gray[:, :-2] - shifted[:, :-2]
    return float(np.sum(diff ** 2))


def if_gt(a: float, b: float, then_v: float, else_v: float) -> float:
    """GP primitive: if a > b then then_v else else_v (NaN/Inf-safe)."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return else_v
    return then_v if a > b else else_v


def composite_focus_measure(gray: np.ndarray) -> float:
    """
    Retained for reference; not used for F_raw in v2 (CNN used instead).
    """
    tag = threshold_absolute_gradient(gray, threshold=THRESH_ABS_GRAD)
    ent = entropy_prior(gray)
    gse = gradient_squared_energy(gray)
    ftei = fourier_transform_energy_index(gray)
    denom = gse + ftei + P_EPS
    ratio = ent / denom if denom > 0 else np.nan
    roberts = roberts_focus_measure(gray)
    skew = intensity_skewness_index(gray)
    then_v = np.log1p(max(roberts, 0.0)) * skew
    else_v = edge_width_sharpness_index(gray)
    out = if_gt(tag, ratio, then_v, else_v)
    return else_v if not np.isfinite(out) else float(out)


def gradient_prior(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.8)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    return float(np.mean(energy))


def entropy_prior(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=64, range=(0.0, 1.0), density=True)
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return float(-(hist * np.log(hist + 1e-8)).sum())


def dark_object_prior(gray: np.ndarray, mu: float = DARK_MU, p: float = DARK_P) -> float:
    """A minimal dark-object prior emphasizing tiles with dark cellular material."""
    m = float(np.mean(gray))
    d = max(0.0, mu - m)
    return float(d ** p)


def gradient_squared_energy(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    squared = gx ** 2 + gy ** 2
    return float(np.sum(squared))


def fourier_transform_energy_index(gray: np.ndarray) -> float:
    spectrum = fftshift(fft2(gray))
    return float(np.sum(np.abs(spectrum) ** 2))


def roberts_focus_measure(gray: np.ndarray) -> float:
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    gx = gray[1:, 1:] - gray[:-1, :-1]
    gy = gray[1:, :-1] - gray[:-1, 1:]
    return float(np.sum(gx ** 2 + gy ** 2))


def intensity_skewness_index(gray: np.ndarray) -> float:
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if std <= 1e-8:
        return 0.0
    norm = ((gray - mean) / (std + 1e-8)) ** 3
    return float(np.mean(norm))


def edge_width_sharpness_index(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(magnitude))


def threshold_absolute_gradient(gray: np.ndarray, threshold: float) -> float:
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(np.mean(magnitude > threshold))


# ----------------------------
# Normalization helper
# ----------------------------
def dataset_normalize(x: np.ndarray, median: float, iqr: float, eps: float = 1e-6, clip: float = 5.0) -> np.ndarray:
    z = (x - median) / (iqr + eps)
    z = np.clip(z, -clip, clip)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------
# Sparsity metrics
# ----------------------------
def compute_r10(S: np.ndarray, eps: float = 1e-8) -> float:
    S_sorted = np.sort(S)[::-1]
    k = max(1, int(np.ceil(0.10 * len(S))))
    return float(S_sorted[:k].sum()) / (float(S.sum()) + eps)


def compute_neff_norm(S: np.ndarray, eps: float = 1e-8) -> float:
    p = S / (float(S.sum()) + eps)
    H = -float(np.sum(p * np.log(p + eps)))
    Neff = float(np.exp(H))
    return Neff / max(1, len(S))


# ----------------------------
# Main calibration
# ----------------------------
def calibrate(
    manifest: str,
    dataset_name: str,
    path_col: str,
    max_images: int,
    seed: int,
    out_csv: str,
    model_path: str,
    model_input_size: Optional[int],
    model_batch_size: int,
):
    df = pd.read_csv(manifest)
    rows = df.to_dict("records")
    random.seed(seed)
    if max_images is not None and len(rows) > max_images:
        random.shuffle(rows)
        rows = rows[:max_images]

    model = load_keras_model(Path(model_path))
    input_h, input_w, channels = infer_model_shape(model, model_input_size)
    target_hw = (input_h, input_w)

    F_all, G_all, E_all, D_all = [], [], [], []
    per_image_feats: List[Dict[str, np.ndarray]] = []
    n_tiles_used = 0
    for row in rows:
        img_path = row.get(path_col)
        if not img_path or not os.path.isfile(img_path):
            continue
        try:
            img = load_image_rgb(img_path)
            tiles, gh, gw = tile_image_resolution_aware(img)
        except Exception:
            continue
        if len(tiles) == 0:
            continue

        f_arr = predict_tile_probs(model, tiles, target_hw, channels, batch_size=model_batch_size)
        g_arr = np.zeros(len(tiles), dtype=np.float32)
        e_arr = np.zeros(len(tiles), dtype=np.float32)
        d_arr = np.zeros(len(tiles), dtype=np.float32)
        for i, t in enumerate(tiles):
            gray = to_gray(t)
            g_arr[i] = gradient_prior(gray)
            e_arr[i] = entropy_prior(gray)
            d_arr[i] = dark_object_prior(gray)
        F_all.append(f_arr)
        G_all.append(g_arr)
        E_all.append(e_arr)
        D_all.append(d_arr)
        n_tiles_used += len(tiles)
        gray_full = to_gray(img)
        ff_val = compute_foreground_fraction(gray_full)
        eo_val = compute_edge_occupancy(gray_full)
        per_image_feats.append(
            {
                "F_raw": f_arr,
                "G_raw": g_arr,
                "E_raw": e_arr,
                "D_raw": d_arr,
                "FF": ff_val,
                "EO": eo_val,
            }
        )

    if not F_all:
        raise RuntimeError("No tiles processed; check manifest and paths.")

    F_cat = np.concatenate(F_all)
    G_cat = np.concatenate(G_all)
    E_cat = np.concatenate(E_all)
    D_cat = np.concatenate(D_all)
    if np.allclose(np.std(F_cat), 0.0):
        print("[WARN] F_raw appears degenerate (std ~ 0). Check CNN model output.")
    if np.nanmin(E_cat) < -1.0:
        print("[WARN] E_raw contains large negative values; entropy_prior() may be miscomputed.")

    stats = {
        "F_median": float(np.median(F_cat)),
        "F_iqr": float(np.percentile(F_cat, 75) - np.percentile(F_cat, 25)),
        "G_median": float(np.median(G_cat)),
        "G_iqr": float(np.percentile(G_cat, 75) - np.percentile(G_cat, 25)),
        "E_median": float(np.median(E_cat)),
        "E_iqr": float(np.percentile(E_cat, 75) - np.percentile(E_cat, 25)),
        "D_median": float(np.median(D_cat)),
        "D_iqr": float(np.percentile(D_cat, 75) - np.percentile(D_cat, 25)),
        "F_p1": float(np.percentile(F_cat, 1)),
        "F_p99": float(np.percentile(F_cat, 99)),
        "G_p1": float(np.percentile(G_cat, 1)),
        "G_p99": float(np.percentile(G_cat, 99)),
        "E_p1": float(np.percentile(E_cat, 1)),
        "E_p99": float(np.percentile(E_cat, 99)),
        "D_p1": float(np.percentile(D_cat, 1)),
        "D_p99": float(np.percentile(D_cat, 99)),
    }
    stats["sparsity_method"] = "R10"

    R10_list = []
    Neff_list = []
    sumS_list = []
    maxS_list = []
    FF_list = []
    EO_list = []
    for feats in per_image_feats:
        F_cal = dataset_normalize(feats["F_raw"], stats["F_median"], stats["F_iqr"])
        G_cal = dataset_normalize(feats["G_raw"], stats["G_median"], stats["G_iqr"])
        E_cal = dataset_normalize(feats["E_raw"], stats["E_median"], stats["E_iqr"])
        D_cal = dataset_normalize(feats["D_raw"], stats["D_median"], stats["D_iqr"])
        T = ETA_GRAD * G_cal + (1.0 - ETA_GRAD) * E_cal
        S = ALPHA_FOCUS * F_cal + (1.0 - ALPHA_FOCUS) * T + LAMBDA_DARK * D_cal
        if S.size == 0:
            continue
        sumS = float(S.sum())
        maxS = float(S.max())
        R10 = compute_r10(S)
        Neff_norm = compute_neff_norm(S)
        R10_list.append(R10)
        Neff_list.append(Neff_norm)
        sumS_list.append(sumS)
        maxS_list.append(maxS)
        FF_list.append(feats.get("FF", 0.0))
        EO_list.append(feats.get("EO", 0.0))

    if not R10_list:
        raise RuntimeError("No sparsity metrics computed; check data.")

    R10_dense_max = float(np.percentile(R10_list, 25))
    R10_extreme_min = float(np.percentile(R10_list, 75))
    Neff_extreme_max = float(np.percentile(Neff_list, 25))
    Neff_dense_min = float(np.percentile(Neff_list, 75))
    empty_sumS_max = float(np.percentile(sumS_list, 1))
    empty_maxS_max = float(np.percentile(maxS_list, 1))

    R10_arr = np.asarray(R10_list, dtype=np.float32)
    Ne_arr = np.asarray(Neff_list, dtype=np.float32)
    FF_arr = np.asarray(FF_list, dtype=np.float32)
    EO_arr = np.asarray(EO_list, dtype=np.float32)

    meds = {
        "R10_median": float(np.median(R10_arr)),
        "Neff_median": float(np.median(Ne_arr)),
        "FF_median": float(np.median(FF_arr)),
        "EO_median": float(np.median(EO_arr)),
    }
    iqrs = {
        "R10_iqr": _iqr(R10_arr),
        "Neff_iqr": _iqr(Ne_arr),
        "FF_iqr": _iqr(FF_arr),
        "EO_iqr": _iqr(EO_arr),
    }

    hybrid_scores = [
        compute_hybrid_sparse_score(r, n, f, e, meds, iqrs)
        for (r, n, f, e) in zip(R10_list, Neff_list, FF_list, EO_list)
    ]
    hybrid_dense_max = float(np.percentile(hybrid_scores, 35))
    hybrid_extreme_min = float(np.percentile(hybrid_scores, 80))

    out_row = {
        "dataset": dataset_name,
        "F_median": stats["F_median"],
        "F_iqr": stats["F_iqr"],
        "G_median": stats["G_median"],
        "G_iqr": stats["G_iqr"],
        "E_median": stats["E_median"],
        "E_iqr": stats["E_iqr"],
        "D_median": stats["D_median"],
        "D_iqr": stats["D_iqr"],
        "R10_dense_max": R10_dense_max,
        "R10_extreme_min": R10_extreme_min,
        "Neff_extreme_max": Neff_extreme_max,
        "Neff_dense_min": Neff_dense_min,
        "empty_sumS_max": empty_sumS_max,
        "empty_maxS_max": empty_maxS_max,
        "sparsity_method": stats["sparsity_method"],
        "FF_median": meds["FF_median"],
        "FF_iqr": iqrs["FF_iqr"],
        "EO_median": meds["EO_median"],
        "EO_iqr": iqrs["EO_iqr"],
        "R10_median": meds["R10_median"],
        "R10_iqr": iqrs["R10_iqr"],
        "Neff_median": meds["Neff_median"],
        "Neff_iqr": iqrs["Neff_iqr"],
        "hybrid_dense_max": hybrid_dense_max,
        "hybrid_extreme_min": hybrid_extreme_min,
        "n_images_used": len(per_image_feats),
        "n_tiles_used": n_tiles_used,
        "alpha_focus_used": ALPHA_FOCUS,
        "eta_grad_used": ETA_GRAD,
        "cnn_model_path": str(model_path),
    }

    pd.DataFrame([out_row]).to_csv(out_csv, index=False)

    print(f"[SUMMARY] dataset={dataset_name} images={len(per_image_feats)} tiles={n_tiles_used}")
    print(f"  F median/iqr: {stats['F_median']:.4f} / {stats['F_iqr']:.4f}")
    print(f"  G median/iqr: {stats['G_median']:.4f} / {stats['G_iqr']:.4f}")
    print(f"  E median/iqr: {stats['E_median']:.4f} / {stats['E_iqr']:.4f}")
    print(f"  R10 dense_max={R10_dense_max:.4f}, extreme_min={R10_extreme_min:.4f}")
    print(f"  Neff extreme_max={Neff_extreme_max:.4f}, dense_min={Neff_dense_min:.4f}")
    print(
        f"  Hybrid: FF med/iqr={meds['FF_median']:.4f}/{iqrs['FF_iqr']:.4f}, "
        f"EO med/iqr={meds['EO_median']:.4f}/{iqrs['EO_iqr']:.4f}"
    )
    print(f"  Hybrid thresholds: dense_max={hybrid_dense_max:.4f}, extreme_min={hybrid_extreme_min:.4f}")
    print(f"  Empty thresholds: sumS_max={empty_sumS_max:.6f}, maxS_max={empty_maxS_max:.6f}")
    print(f"  Saved to: {out_csv}")


def parse_args():
    ap = argparse.ArgumentParser(description="Dataset calibration with CNN F_raw and sparsity thresholds")
    ap.add_argument("--manifest", required=True, help="Path to manifest CSV")
    ap.add_argument("--dataset-name", required=True, help="Dataset name to store in output")
    ap.add_argument("--path-col", default="image_path", help="Column name for image path")
    ap.add_argument("--max-images", type=int, default=None, help="Optional max images (sampled with seed)")
    ap.add_argument("--out-csv", default=None, help="Output CSV path (default <dataset-name>_calibration.csv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-path", default=str(resolve_default_model_path()))
    ap.add_argument("--model-input-size", type=int, default=None)
    ap.add_argument("--model-batch-size", type=int, default=64)
    return ap.parse_args()


def main():
    args = parse_args()
    out_csv = args.out_csv or f"{args.dataset_name}_calibration.csv"
    calibrate(
        args.manifest,
        args.dataset_name,
        args.path_col,
        args.max_images,
        args.seed,
        out_csv,
        model_path=args.model_path,
        model_input_size=args.model_input_size,
        model_batch_size=args.model_batch_size,
    )


if __name__ == "__main__":
    main()
