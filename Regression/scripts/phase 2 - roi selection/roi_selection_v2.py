#!/usr/bin/env python3
"""
RIN ROI selection v2 (no diversity/border guards, dense min selection ratio).

Differences vs roi_selection.py:
  - No diversity guard.
  - No border penalty.
  - Enforces a minimum selection ratio for dense images (>=45%).
"""

import argparse
import os
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
from scipy.fft import fft2, fftshift
import tensorflow as tf
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

TILE_SIZE = 200
SMALL_THRESH = 400
LARGE_THRESH = 2000

HFER_R0 = 0.3
P_EPS = 1e-6
ETA_GRAD = 0.6
ALPHA_FOCUS = 0.7
BETA_K = 0.6
MIN_DENSE_RATIO = 0.45
LAMBDA_DARK = 0.25  # weight of dark-object prior (saliency for compact dark regions)
DARK_MU = 0.55
DARK_P = 2.0
MAX_DENSE_RATIO = 0.55  # safety cap to avoid selecting most tiles even when labeled dense
DENSE_FF_Z_MIN = -0.5    # apply dense min-ratio only if global FF is not unusually low
DENSE_EO_Z_MIN = -0.5    # apply dense min-ratio only if global EO is not unusually low
THRESH_ABS_GRAD = 0.2

# ---- SparseFocus-aligned foreground gate ----
FG_MIN = 0.05   # tightened gate: reject weak/near-background tiles more aggressively
D_MIN_NON_DENSE = 1
# Hybrid sparsity knobs (must match calibration_script defaults; thresholds calibrated per dataset)
CANNY_LO = 50
CANNY_HI = 150
MAX_HYBRID_DOWNSAMPLE = 512
# Sparsity thresholds (tunable)
R10_DENSE_MAX = 0.45
R10_SPARSE_MAX = 0.85
NEFF_EXTREME_MAX = 0.10
NEFF_SPARSE_MAX = 0.45


def load_image_rgb(path: str) -> np.ndarray:
    """
    Load image as RGB float32 in [0,1], handling 8-bit and 16-bit inputs.
    Prevents 16-bit images from saturating to white.
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
    GP composite (exact semantics requested):
      if_gt(Threshold Absolute Gradient,
            (Image Entropy div (Gradient Squared Energy add Fourier Transform Energy Index)),
            (log1p(Roberts Focus Measure) mul Intensity Skewness Index),
            Edge Width Sharpness Index)
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

    # EXACT requested structure:
    #   if_gt(TAG, ratio, then_v, else_v)
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


def dark_object_prior(gray01: np.ndarray, mu: float = DARK_MU, p: float = DARK_P) -> float:
    """A minimal dark-object prior emphasizing tiles with dark cellular material."""
    m = float(np.mean(gray01))
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
    # Return fraction of pixels above threshold (0..1), not a raw count.
    return float(np.mean(magnitude > threshold))


def rin_for_image_rgb_calib(img_rgb: np.ndarray, calib: Dict[str, float], beta_override: float = None) -> Dict[str, Any]:
    """
    Same as rin_for_image_path_calib, but operates on an already-loaded RGB image in [0,1].
    This is required for robustness testing where we perturb the image in-memory.
    """
    tiles, S, F_hat, G_hat, E_hat, T_hat, gh, gw, s_label = compute_rin_scores_for_image(img_rgb, calib)
    valid_mask = S > 0.0
    if np.any(valid_mask):
        S_valid = S[valid_mask]
        r10_val = compute_R10(S_valid)
        neff_val = compute_Neff_norm(S_valid)
    else:
        r10_val = 0.0
        neff_val = 0.0
    N = len(S)
    sumS = float(S.sum()) if N else 0.0

    beta_used = BETA_K if beta_override is None else beta_override
    k_est = int(np.ceil(beta_used * sumS)) if N else 0
    k = max(1, min(k_est, N)) if N else 0

    if s_label == "dense" and N:
        # Fix A: only enforce the dense minimum-selection rule when global occupancy supports "dense".
        # This avoids inflating k on datasets where the tile score mass is reliable but global FF/EO are low
        # (e.g., thin-film blood smear backgrounds).
        dense_ok = True
        if all(k_ in calib for k_ in ("FF_median", "FF_iqr", "EO_median", "EO_iqr")):
            img_gray01 = to_gray(img_rgb)
            FF_g = compute_foreground_fraction(img_gray01)
            EO_g = compute_edge_occupancy(img_gray01)
            zFF = _z(FF_g, float(calib["FF_median"]), float(calib["FF_iqr"]))
            zEO = _z(EO_g, float(calib["EO_median"]), float(calib["EO_iqr"]))
            dense_ok = (zFF >= DENSE_FF_Z_MIN) and (zEO >= DENSE_EO_Z_MIN)

        if dense_ok:
            k = max(k, int(np.ceil(MIN_DENSE_RATIO * N)))

        # Fix B: safety cap for dense so we don't select an excessive fraction of tiles.
        k = min(k, int(np.ceil(MAX_DENSE_RATIO * N)))
        k = min(k, N)
        order = np.argsort(-S)
        top_indices = order[:k].tolist() if k > 0 else []
        d_min_used = 0
    else:
        top_indices, d_min_used = select_topk_tiles_diverse(S, gh, gw, k=k, d_min=D_MIN_NON_DENSE)

    ratio = len(top_indices) / N if N else 0.0

    # Per-image summaries
    F_mean = float(F_hat.mean()) if N else 0.0
    F_std = float(F_hat.std()) if N else 0.0
    T_mean = float(T_hat.mean()) if N else 0.0
    T_std = float(T_hat.std()) if N else 0.0
    S_mean = float(S.mean()) if N else 0.0
    S_std = float(S.std()) if N else 0.0

    corr_FS = float(np.corrcoef(F_hat, S)[0, 1]) if (N and F_std > 0 and S_std > 0) else 0.0
    corr_TS = float(np.corrcoef(T_hat, S)[0, 1]) if (N and T_std > 0 and S_std > 0) else 0.0

    return {
        "tiles": tiles,
        "scores": S,
        "focus_norm": F_hat,
        "grid_h": gh,
        "grid_w": gw,
        "top_indices": np.array(top_indices, dtype=np.int32),
        "top_tiles": tiles[top_indices],
        "E_total": sumS,
        "N_tiles": N,
        "K_selected": len(top_indices),
        "sparsity_label": s_label,
        "sparsity_metric": float(r10_val),
        "sparsity_method": "R10",
        "sparsity_debug": {"R10": float(r10_val), "Neff_norm": float(neff_val), "N": int(np.sum(valid_mask))},
        "selection_ratio": ratio,
        "k_est": k_est,
        "beta_used": beta_used,
        "d_min_used": int(d_min_used),
        "F_mean": F_mean,
        "F_std": F_std,
        "T_mean": T_mean,
        "T_std": T_std,
        "S_mean": S_mean,
        "S_std": S_std,
        "corr_FS": corr_FS,
        "corr_TS": corr_TS,
    }


def robust_normalize(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p5 = np.percentile(values, 5.0)
    p95 = np.percentile(values, 95.0)
    denom = p95 - p5 + eps
    return np.clip((values - p5) / denom, 0.0, 1.0)


def dataset_normalize(x: np.ndarray, median: float, iqr: float, eps: float = 1e-6, clip: float = 5.0) -> np.ndarray:
    """
    Calibration-based normalization (matches calibration_script.py):
      z = (x - median)/(iqr+eps), clipped, then sigmoid.
    """
    z = (x - median) / (iqr + eps)
    z = np.clip(z, -clip, clip)
    return 1.0 / (1.0 + np.exp(-z))


def classify_sparsity(E_raw: np.ndarray, G_raw: np.ndarray) -> str:
    """
    Classify sparsity using RAW priors (no normalization).
    active = tiles above 80th percentile of entropy OR gradient.
    occ = mean(active); thresholds:
      >=0.30 -> dense
      0.10..0.35 -> sparse
      else -> extremely_sparse
    """
    E_hi = np.percentile(E_raw, 80.0)
    G_hi = np.percentile(G_raw, 80.0)
    active = (E_raw >= E_hi) | (G_raw >= G_hi)
    occ = float(np.mean(active)) if active.size else 0.0
    if occ >= 0.27:
        return "dense"
    if occ >= 0.10 and occ < 0.30:
        return "sparse"
    return "extremely_sparse"


def compute_sparsity_label_from_scores(S: np.ndarray, method: str = "R10") -> Tuple[str, float, Dict[str, float]]:
    """
    Compute sparsity label from scores S using either R10 or Neff_norm.
    Returns (label, metric_value, debug_dict)
    """
    eps = 1e-8
    S = np.asarray(S, dtype=np.float32)
    N = len(S)
    if N == 0:
        return "extremely_sparse", 0.0, {"N": 0, "sumS": 0.0, "method": method}

    sumS = float(S.sum())
    dbg: Dict[str, float] = {"N": int(N), "sumS": sumS, "method": method}

    if method.lower() == "neff":
        p = S / (sumS + eps)
        H = -float(np.sum(p * np.log(p + eps)))
        Neff = float(np.exp(H))
        Neff_norm = Neff / max(1, N)
        dbg.update({"H": H, "Neff": Neff, "Neff_norm": Neff_norm})
        if Neff_norm <= NEFF_EXTREME_MAX:
            label = "extremely_sparse"
        elif Neff_norm <= NEFF_SPARSE_MAX:
            label = "sparse"
        else:
            label = "dense"
        return label, Neff_norm, dbg

    # Default: R10 concentration
    S_sorted = np.sort(S)[::-1]
    k = max(1, int(np.ceil(0.10 * N)))
    R10 = float(S_sorted[:k].sum()) / (sumS + eps)
    dbg.update({"k": int(k), "R10": R10})
    if R10 >= R10_SPARSE_MAX:
        label = "extremely_sparse"
    elif R10 >= R10_DENSE_MAX:
        label = "sparse"
    else:
        label = "dense"
    return label, R10, dbg


def compute_R10(S: np.ndarray, eps: float = 1e-8) -> float:
    S_sorted = np.sort(S)[::-1]
    k = max(1, int(np.ceil(0.10 * len(S))))
    return float(S_sorted[:k].sum()) / (float(S.sum()) + eps)


def compute_Neff_norm(S: np.ndarray, eps: float = 1e-8) -> float:
    p = S / (float(S.sum()) + eps)
    H = -float(np.sum(p * np.log(p + eps)))
    Neff = float(np.exp(H))
    return Neff / max(1, len(S))


def _downsample_gray_for_metrics(gray01: np.ndarray, max_side: int = MAX_HYBRID_DOWNSAMPLE) -> np.ndarray:
    h, w = gray01.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray01
    scale = max_side / float(m)
    nh, nw = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
    return cv2.resize(gray01, (nw, nh), interpolation=cv2.INTER_AREA)


def compute_foreground_fraction(gray01: np.ndarray) -> float:
    g = _downsample_gray_for_metrics(gray01)
    u8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    inv = 255 - u8
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(mask.mean() / 255.0)


def compute_edge_occupancy(gray01: np.ndarray) -> float:
    g = _downsample_gray_for_metrics(gray01)
    u8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (3, 3), 0)
    edges = cv2.Canny(u8, CANNY_LO, CANNY_HI)
    return float(edges.mean() / 255.0)


def compute_tile_foreground_fraction(gray01: np.ndarray) -> float:
    """
    Foreground fraction per tile using Otsu thresholding
    on inverted grayscale. Returns value in [0, 1].
    """
    u8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    inv = 255 - u8
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(mask.mean() / 255.0)


def _iqr_fallback(iqr_val: float, eps: float = 1e-8) -> float:
    return float(iqr_val) if float(iqr_val) > eps else 1.0


def _z(val: float, med: float, iqr: float, eps: float = 1e-8) -> float:
    return (float(val) - float(med)) / (_iqr_fallback(iqr) + eps)


def compute_hybrid_sparse_score(R10: float, Neff_norm: float, FF: float, EO: float, calib: Dict[str, float]) -> float:
    zR10 = _z(R10, calib.get("R10_median", 0.0), calib.get("R10_iqr", 1.0))
    zNe = _z(Neff_norm, calib.get("Neff_median", 0.0), calib.get("Neff_iqr", 1.0))
    zFF = _z(FF, calib.get("FF_median", 0.0), calib.get("FF_iqr", 1.0))
    zEO = _z(EO, calib.get("EO_median", 0.0), calib.get("EO_iqr", 1.0))
    # higher => more sparse (invert low=>sparse terms)
    return float(zR10 - zNe - zFF - zEO)


def label_sparsity(S: np.ndarray, calib: Dict[str, float], method: str = "R10",
                   img_gray01: np.ndarray = None) -> Tuple[str, float, float]:
    """
    Label sparsity.
    - If hybrid calibration fields exist and img_gray01 is provided, use hybrid score thresholds.
    - Otherwise preserve the original R10/Neff behavior.
    Returns: (label, R10, Neff_norm)
    """
    R10 = compute_R10(S)
    Neff_norm = compute_Neff_norm(S)

    has_hybrid = ("hybrid_dense_max" in calib) and ("hybrid_extreme_min" in calib) and (img_gray01 is not None)
    if has_hybrid:
        FF = compute_foreground_fraction(img_gray01)
        EO = compute_edge_occupancy(img_gray01)
        hs = compute_hybrid_sparse_score(R10, Neff_norm, FF, EO, calib)
        if hs <= float(calib.get("hybrid_dense_max", -1e9)):
            label = "dense"
        elif hs >= float(calib.get("hybrid_extreme_min", 1e9)):
            label = "extremely_sparse"
        else:
            label = "sparse"
        return label, R10, Neff_norm

    # ---- original behavior preserved ----
    if method.upper() == "R10":
        if R10 <= calib.get("R10_dense_max", R10_DENSE_MAX):
            label = "dense"
        elif R10 >= calib.get("R10_extreme_min", R10_SPARSE_MAX):
            label = "extremely_sparse"
        else:
            label = "sparse"
    else:
        ne_ext = calib.get("Neff_extreme_max", NEFF_EXTREME_MAX)
        ne_dense = calib.get("Neff_dense_min", NEFF_SPARSE_MAX)
        if Neff_norm <= ne_ext:
            label = "extremely_sparse"
        elif Neff_norm >= ne_dense:
            label = "dense"
        else:
            label = "sparse"
    return label, R10, Neff_norm


def compute_rin_scores_for_image(img_rgb: np.ndarray, calib: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, str]:
    tiles, grid_h, grid_w = tile_image_resolution_aware(img_rgb)
    n_tiles = len(tiles)
    F_raw = np.zeros(n_tiles, dtype=np.float32)
    G_raw = np.zeros(n_tiles, dtype=np.float32)
    E_raw = np.zeros(n_tiles, dtype=np.float32)
    D_raw = np.zeros(n_tiles, dtype=np.float32)
    FG = np.zeros(n_tiles, dtype=np.float32)
    for idx, tile in enumerate(tiles):
        gray = to_gray(tile)
        F_raw[idx] = composite_focus_measure(gray)
        G_raw[idx] = gradient_prior(gray)
        E_raw[idx] = entropy_prior(gray)
        D_raw[idx] = dark_object_prior(gray)
        FG[idx] = compute_foreground_fraction(gray)

    # ---- SparseFocus gate: discard blank tiles BEFORE saliency ----
    valid_mask = FG >= FG_MIN

    F_hat = dataset_normalize(F_raw, calib["F_median"], calib["F_iqr"])
    G_hat = dataset_normalize(G_raw, calib["G_median"], calib["G_iqr"])
    E_hat = dataset_normalize(E_raw, calib["E_median"], calib["E_iqr"])
    D_hat = dataset_normalize(
        D_raw,
        calib.get("D_median", float(np.median(D_raw))),
        calib.get("D_iqr", float(np.percentile(D_raw, 75) - np.percentile(D_raw, 25))),
    )
    T = ETA_GRAD * G_hat + (1.0 - ETA_GRAD) * E_hat

    # ---- Saliency only for valid tiles ----
    S = np.zeros_like(F_hat)
    S[valid_mask] = (
        ALPHA_FOCUS * F_hat[valid_mask]
        + (1.0 - ALPHA_FOCUS) * T[valid_mask]
        + LAMBDA_DARK * D_hat[valid_mask]
    )
    S[~valid_mask] = 0.0

    # ---- Sparsity label MUST be computed on valid tiles only ----
    if np.any(valid_mask):
        S_valid = S[valid_mask]
        sparsity_label, _, _ = label_sparsity(S_valid, calib, method="R10")
    else:
        sparsity_label = "extremely_sparse"

    tiles_np = np.stack(tiles, axis=0).astype(np.float32)
    return tiles_np, S, F_hat, G_hat, E_hat, T, grid_h, grid_w, sparsity_label


def select_topk_tiles_by_sparsity(S: np.ndarray, sparsity_label: str) -> Tuple[List[int], float]:
    """
    Select top tiles by score S using ratio based on sparsity label.
      dense -> 60%, sparse -> 30%, extremely_sparse -> 10%
    """
    N = len(S)
    if sparsity_label == "dense":
        ratio = 0.60
    elif sparsity_label == "sparse":
        ratio = 0.30
    else:
        ratio = 0.10
    k = int(np.ceil(ratio * N))
    k = max(1, min(k, N))
    order = np.argsort(-S)
    return list(order[:k]), ratio


def select_topk_tiles(S: np.ndarray,
                      grid_h: int,
                      grid_w: int,
                      beta: float = BETA_K,
                      k_min: int = 7,
                      k_max_ratio: float = None) -> List[int]:
    """
    SparseFocus-style continuous k selection:
      k_est = ceil(beta * sum(S))
      k_min <= k <= k_max (optional ratio cap), bounded by [1, N]
    """
    N = len(S)
    if N == 0:
        return []
    E_total = float(S.sum())
    k_est = int(np.ceil(beta * E_total))
    k_min_used = max(1, int(k_min))
    k_max_used = N if k_max_ratio is None else min(N, int(np.ceil(k_max_ratio * N)))
    k = min(max(k_est, k_min_used), k_max_used, N)
    order = np.argsort(-S)
    return list(order[:k])


def idx_to_rc(idx: int, grid_w: int) -> Tuple[int, int]:
    return idx // grid_w, idx % grid_w


def select_topk_tiles_diverse(S: np.ndarray,
                              grid_h: int,
                              grid_w: int,
                              k: int,
                              d_min: int) -> Tuple[List[int], int]:
    """
    Greedy diversity-constrained selection:
      - sort candidates by S descending
      - accept a tile if its Manhattan distance to all selected >= d_min
      - if not enough selected, relax d_min progressively down to 0
    Returns (selected_indices, d_min_used).
    """
    N = len(S)
    if N == 0 or k <= 0:
        return [], d_min
    k = min(k, N)
    order = np.argsort(-S)

    def run_with_d(d_req: int) -> List[int]:
        selected: List[int] = []
        for idx in order:
            if len(selected) >= k:
                break
            gw = grid_w if grid_w > 0 else max(1, int(np.ceil(np.sqrt(N))))
            r, c = idx_to_rc(idx, gw)
            ok = True
            for sel in selected:
                rs, cs = idx_to_rc(sel, gw)
                if abs(r - rs) + abs(c - cs) < d_req:
                    ok = False
                    break
            if ok:
                selected.append(int(idx))
        return selected

    d_used = max(0, int(d_min))
    selected = run_with_d(d_used)
    while len(selected) < k and d_used > 0:
        d_used -= 1
        selected = run_with_d(d_used)
    return selected[:k], d_used


def visualize_tiles(img_rgb: np.ndarray,
                    tiles: np.ndarray,
                    scores: np.ndarray,
                    top_indices: np.ndarray,
                    grid_h: int,
                    grid_w: int,
                    sparsity_label: str = None,
                    sparsity_metric: float = None,
                    sparsity_method: str = None,
                    r10: float = None,
                    neff_norm: float = None,
                    k_selected: int = None,
                    sumS: float = None,
                    cnn_indices: np.ndarray = None) -> np.ndarray:
    N = len(tiles)
    if grid_h * grid_w != N or grid_h <= 0 or grid_w <= 0:
        grid_w = int(np.ceil(np.sqrt(N)))
        grid_h = int(np.ceil(N / grid_w))

    canvas_h = grid_h * TILE_SIZE
    canvas_w = grid_w * TILE_SIZE
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    top_set = set(top_indices.tolist())
    cnn_set = set(np.asarray(cnn_indices).tolist()) if cnn_indices is not None else set()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, tile in enumerate(tiles):
        i, j = divmod(idx, grid_w)
        y0, y1 = i * TILE_SIZE, (i + 1) * TILE_SIZE
        x0, x1 = j * TILE_SIZE, (j + 1) * TILE_SIZE
        canvas[y0:y1, x0:x1] = tile
        color = (0, 1, 0)
        if idx in cnn_set:
            color = (0, 0, 1)
        if idx in top_set:
            color = (1, 0, 0)
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        txt = f"{scores[idx]:.2f}"
        cv2.putText(canvas, txt, (x0 + 5, y0 + 20), font, 0.5, color, 1, cv2.LINE_AA)

    canvas_uint8 = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)

    # Overlay sparsity label (top-right)
    if sparsity_label:
        r10_txt = f" R10={r10:.2f}" if r10 is not None else ""
        neff_txt = f" Neff={neff_norm:.2f}" if neff_norm is not None else ""
        text = f"{sparsity_label}{r10_txt}{neff_txt}"
        font_scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        pad = 5
        x = canvas_uint8.shape[1] - tw - 10
        y = 10 + th
        cv2.rectangle(canvas_uint8, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
        cv2.putText(canvas_uint8, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Inset 1: cumulative importance curve
    try:
        inset_w, inset_h = 200, 160
        inset1 = np.zeros((inset_h, inset_w, 3), dtype=np.uint8) + 20
        if sumS is None:
            sumS = float(scores.sum()) if len(scores) else 0.0
        if len(scores) and sumS > 0:
            S_sorted = np.sort(scores)[::-1]
            C = np.cumsum(S_sorted) / sumS
            xs = np.linspace(0, inset_w - 1, len(C))
            pts = np.stack([xs, (1.0 - C) * (inset_h - 1)], axis=1).astype(np.int32)
            for i in range(len(pts) - 1):
                cv2.line(inset1, (pts[i, 0], pts[i, 1]), (pts[i + 1, 0], pts[i + 1, 1]), (0, 255, 255), 1)
            if k_selected is None:
                k_selected = len(top_indices)
            frac = min(1.0, max(0.0, k_selected / max(1, len(scores))))
            xline = int(frac * (inset_w - 1))
            cv2.line(inset1, (xline, 0), (xline, inset_h - 1), (0, 0, 255), 1)
        cv2.rectangle(canvas_uint8, (10, 10), (10 + inset_w, 10 + inset_h), (255, 255, 255), 1)
        canvas_uint8[11:11 + inset_h, 11:11 + inset_w] = inset1
    except Exception:
        pass

    # Inset 2: heatmap of S on grid
    try:
        if len(scores):
            ghm = grid_h if grid_h > 0 else int(np.ceil(np.sqrt(len(scores))))
            gwm = grid_w if grid_w > 0 else ghm
            grid = np.zeros((ghm, gwm), dtype=np.float32)
            for idx, s in enumerate(scores):
                r, c = idx_to_rc(idx, gwm)
                if r < ghm and c < gwm:
                    grid[r, c] = s
            gnorm = grid
            if gnorm.max() > 0:
                gnorm = gnorm / gnorm.max()
            heat = cv2.applyColorMap((gnorm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.resize(heat, (200, 160), interpolation=cv2.INTER_NEAREST)
            y0 = 20 + inset_h
            x0 = 10
            cv2.rectangle(canvas_uint8, (x0, y0), (x0 + heat.shape[1], y0 + heat.shape[0]), (255, 255, 255), 1)
            canvas_uint8[y0 + 1:y0 + 1 + heat.shape[0], x0 + 1:x0 + 1 + heat.shape[1]] = heat
    except Exception:
        pass

    return canvas_uint8


def build_rin_dataset_from_manifests(manifest_paths: List[str],
                                     path_col: str = "img_path",
                                     dataset_col: str = "dataset",
                                     defocus_col: str = None,
                                     calib: Dict[str, float] = None,
                                     beta_override: float = None,
                                     diag_prefix: str = None) -> Tuple[tf.data.Dataset, pd.DataFrame]:
    if calib is None:
        raise RuntimeError("Calibration stats are required; pass calib dict.")
    meta_records = []
    all_tiles: List[np.ndarray] = []
    img_global_index = 0
    skip_counts: Dict[str, int] = {}
    diag_records: Dict[str, List[Dict[str, Any]]] = {}
    total_images = 0
    total_skipped = 0

    for man_path in manifest_paths:
        df = pd.read_csv(man_path)
        for row_idx, row in df.iterrows():
            total_images += 1
            img_path = row[path_col]
            dataset_id = row[dataset_col] if dataset_col in df.columns else "unknown"
            if dataset_id not in diag_records:
                diag_records[dataset_id] = []
            if not os.path.isfile(img_path):
                print(f"[WARN] Missing image: {img_path}")
                continue
            try:
                res = rin_for_image_path_calib(img_path, calib, beta_override=beta_override)
            except Exception as e:
                print(f"[WARN] Error processing {img_path}: {e}")
                continue

            if res.get("skip", False):
                skip_counts[dataset_id] = skip_counts.get(dataset_id, 0) + 1
                total_skipped += 1
                diag_records[dataset_id].append({
                    "image_path": img_path,
                    "dataset": dataset_id,
                    "N_tiles": res.get("N_tiles", 0),
                    "skip": True,
                    "skip_reason": res.get("skip_reason", "unknown"),
                    "sumS": res.get("sumS", 0.0),
                    "maxS": res.get("maxS", 0.0),
                    "k_mass": 0,
                    "k_selected": 0,
                    "sparsity_label": None,
                    "R10": None,
                    "Neff_norm": None,
                    "d_min_used": 0,
                    "F_mean": None,
                    "F_std": None,
                    "F_max": None,
                    "T_mean": None,
                    "T_std": None,
                    "T_max": None,
                    "S_mean": None,
                    "S_std": None,
                    "S_max": None,
                    "corr_FS": None,
                    "corr_TS": None,
                })
                continue

            top_indices = res["top_indices"]
            S = res["scores"]
            diag_records[dataset_id].append({
                "image_path": img_path,
                "dataset": dataset_id,
                "N_tiles": int(res.get("N_tiles", len(S))),
                "skip": False,
                "skip_reason": "",
                "sumS": float(res.get("E_total", float(S.sum()))),
                "maxS": float(res.get("S_max", float(S.max()))),
                "k_mass": int(res.get("k_est", 0)),
                "k_selected": int(res.get("K_selected", len(top_indices))),
                "sparsity_label": res.get("sparsity_label"),
                "R10": res.get("sparsity_metric"),
                "Neff_norm": res.get("sparsity_debug", {}).get("Neff_norm"),
                "d_min_used": res.get("d_min_used", 0),
                "F_mean": res.get("F_mean"),
                "F_std": res.get("F_std"),
                "F_max": res.get("F_max"),
                "T_mean": res.get("T_mean"),
                "T_std": res.get("T_std"),
                "T_max": res.get("T_max"),
                "S_mean": res.get("S_mean"),
                "S_std": res.get("S_std"),
                "S_max": res.get("S_max"),
                "corr_FS": res.get("corr_FS"),
                "corr_TS": res.get("corr_TS"),
            })
            for local_rank, (tile_idx, score) in enumerate(zip(top_indices, S[top_indices])):
                all_tiles.append(res["tiles"][tile_idx])
                rec = {
                    "global_tile_index": len(all_tiles) - 1,
                    "dataset": dataset_id,
                    "manifest": man_path,
                    "img_row_index": int(row_idx),
                    "img_global_index": int(img_global_index),
                    "img_path": img_path,
                    "tile_index": int(tile_idx),
                    "tile_rank_in_image": int(local_rank),
                    "score_S": float(score),
                    "sparsity_label": res.get("sparsity_label"),
                    "sparsity_metric": res.get("sparsity_metric"),
                    "sparsity_method": res.get("sparsity_method"),
                    "selection_ratio": res.get("selection_ratio"),
                    "d_min_used": res.get("d_min_used", 0),
                    "F_mean": res.get("F_mean"),
                    "F_std": res.get("F_std"),
                    "F_max": res.get("F_max"),
                    "T_mean": res.get("T_mean"),
                    "T_std": res.get("T_std"),
                    "T_max": res.get("T_max"),
                    "S_mean": res.get("S_mean"),
                    "S_std": res.get("S_std"),
                    "S_max": res.get("S_max"),
                    "corr_FS": res.get("corr_FS"),
                    "corr_TS": res.get("corr_TS"),
                }
                if defocus_col is not None and defocus_col in df.columns:
                    rec["defocus"] = row[defocus_col]
                meta_records.append(rec)
            img_global_index += 1

    if not all_tiles:
        raise RuntimeError("No tiles were produced by RIN. Check manifests/paths.")

    tiles_array = np.stack(all_tiles, axis=0).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(tiles_array).batch(64).prefetch(tf.data.AUTOTUNE)
    meta_df = pd.DataFrame(meta_records)

    if skip_counts:
        print("[INFO] Skip counts due to empty_field:")
        for ds_id, cnt in skip_counts.items():
            print(f"  {ds_id}: {cnt}")

    if diag_prefix:
        for ds_id, recs in diag_records.items():
            out_path = f"{diag_prefix}_{ds_id}_diagnostics.csv"
            pd.DataFrame(recs).to_csv(out_path, index=False)
            print(f"[INFO] Saved diagnostics to {out_path}")

    if total_images > 0:
        print(f"[INFO] Images processed: {total_images}, skipped: {total_skipped}, skip rate: {total_skipped/total_images:.3f}")

    return ds, meta_df


def rin_for_image_path(img_path: str) -> Dict[str, Any]:
    raise RuntimeError("Use rin_for_image_path_calib(img_path, calib) with calibration stats.")


def rin_for_image_path_calib(img_path: str, calib: Dict[str, float], beta_override: float = None) -> Dict[str, Any]:
    img_rgb = load_image_rgb(img_path)
    return rin_for_image_rgb_calib(img_rgb, calib, beta_override=beta_override)


def parse_args():
    ap = argparse.ArgumentParser(description="RIN ROI selection v2 (no diversity/border guards, dense min ratio)")
    ap.add_argument("--mode", default="legacy", choices=["legacy", "cnn"], help="Selection mode (default: legacy)")
    ap.add_argument("--manifests", nargs="+", help="Paths to manifest CSV(s)")
    ap.add_argument("--path-col", default="img_path", help="Column name for image path")
    ap.add_argument("--dataset-col", default="dataset", help="Optional dataset column name")
    ap.add_argument("--defocus-col", default=None, help="Optional defocus column name (stored in metadata if provided)")
    ap.add_argument("--calibration-csv", required=False, help="Calibration CSV produced by calibration_script.py")
    ap.add_argument("--viz-dir", default=None, help="If set, save per-image tile mosaic with scores/highlights to this directory")
    ap.add_argument("--tiles-dir", default=None, help="If set, save selected tiles as .npy per image to this directory (defaults to viz-dir if provided)")
    ap.add_argument("--diag-prefix", default=None, help="Prefix for per-image diagnostics CSV (e.g., /path/out_prefix)")
    ap.add_argument("--beta-k", type=float, default=None, help="Override beta for k estimation (defaults to BETA_K)")
    ap.add_argument("--cnn-calibration-csv", default=None, help="Optional CNN-light calibration CSV")
    ap.add_argument("--cellness-predictor", default=None, help="Python callable: module.path:fn")
    ap.add_argument("--predictor-mode", default="auto", choices=["auto", "map", "tiles"])
    ap.add_argument("--cnn-dummy-threshold", type=float, default=0.5)
    ap.add_argument("--cnn-tau-empty", type=float, default=0.2)
    ap.add_argument("--cnn-beta", type=float, default=0.4)
    ap.add_argument("--cnn-k-min", type=int, default=1)
    ap.add_argument("--cnn-k-max", type=int, default=100)
    ap.add_argument("--cnn-use-sum-occ", action="store_true", default=True)
    ap.add_argument("--cnn-no-use-sum-occ", dest="cnn_use_sum_occ", action="store_false")
    ap.add_argument("--cnn-w-occ", type=float, default=0.85)
    ap.add_argument("--cnn-w-focus", type=float, default=0.15)
    ap.add_argument("--cnn-focus-on-candidates-only", action="store_true", default=True)
    ap.add_argument("--cnn-focus-on-all", dest="cnn_focus_on_candidates_only", action="store_false")
    ap.add_argument("--cnn-candidate-multiplier", type=float, default=2.0)
    ap.add_argument("--cnn-d-min-nondense", type=int, default=D_MIN_NON_DENSE)
    ap.add_argument("--cnn-dense-cap-ratio", type=float, default=0.55)
    ap.add_argument("--cnn-dense-min-ratio", type=float, default=None)
    ap.add_argument("--cnn-downsample-for-cnn", type=int, default=None)
    ap.add_argument("--cnn-batch-size", type=int, default=64)
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.manifests:
        print("No manifests provided. Exiting.")
        return
    if args.mode == "cnn":
        from roi_selection_cnn_v1 import (
            ROISelectorCNNConfig,
            select_rois_cnn,
            _load_predictor as _load_cnn_predictor,
            _load_calibration as _load_cnn_calibration,
        )
        calib = _load_cnn_calibration(args.cnn_calibration_csv)
        predictor = _load_cnn_predictor(args.cellness_predictor, args.predictor_mode, args.cnn_dummy_threshold)
        cfg = ROISelectorCNNConfig(
            tau_empty=args.cnn_tau_empty,
            beta=args.cnn_beta,
            k_min=args.cnn_k_min,
            k_max=args.cnn_k_max,
            use_sum_occ=args.cnn_use_sum_occ,
            w_occ=args.cnn_w_occ,
            w_focus=args.cnn_w_focus,
            focus_on_candidates_only=args.cnn_focus_on_candidates_only,
            candidate_multiplier=args.cnn_candidate_multiplier,
            d_min_nondense=args.cnn_d_min_nondense,
            dense_cap_ratio=args.cnn_dense_cap_ratio,
            dense_min_ratio=args.cnn_dense_min_ratio,
            downsample_for_cnn=args.cnn_downsample_for_cnn,
            batch_size=args.cnn_batch_size,
        )
        os.makedirs(args.viz_dir, exist_ok=True) if args.viz_dir else None
        tiles_root = args.tiles_dir or args.viz_dir
        os.makedirs(tiles_root, exist_ok=True) if tiles_root else None
        for man_path in args.manifests:
            df = pd.read_csv(man_path)
            man_base = os.path.splitext(os.path.basename(man_path))[0]
            out_dir = os.path.join(args.viz_dir, man_base) if args.viz_dir else None
            os.makedirs(out_dir, exist_ok=True) if out_dir else None
            tiles_dir = os.path.join(tiles_root, man_base) if tiles_root else None
            os.makedirs(tiles_dir, exist_ok=True) if tiles_dir else None
            for _, row in df.iterrows():
                img_path = row[args.path_col]
                if not os.path.isfile(img_path):
                    print(f"[WARN] Missing image: {img_path}")
                    continue
                img_rgb = load_image_rgb(img_path)
                res = select_rois_cnn(img_rgb, calib, cfg, predictor)
                if tiles_dir:
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    tiles_path = os.path.join(tiles_dir, f"{base}_tiles.npy")
                    tiles = []
                    for idx in res["selected_indices"]:
                        x0, y0, x1, y1 = res["tile_boxes"][int(idx)]
                        tiles.append(img_rgb[y0:y1, x0:x1])
                    np.save(tiles_path, np.asarray(tiles, dtype=np.float32))
                    print(f"[TILES] Saved {tiles_path}")
                if out_dir:
                    try:
                        tiles = []
                        for x0, y0, x1, y1 in res["tile_boxes"]:
                            tiles.append(img_rgb[y0:y1, x0:x1])
                        tiles_np = np.asarray(tiles, dtype=np.float32)
                        tau_empty = res.get("debug", {}).get("tau_empty", args.cnn_tau_empty)
                        if tau_empty is None:
                            tau_empty = args.cnn_tau_empty
                        occ = np.asarray(res.get("occ", []), dtype=np.float32)
                        cnn_indices = np.where(occ >= float(tau_empty))[0] if occ.size else np.array([], dtype=np.int32)
                        viz = visualize_tiles(
                            img_rgb,
                            tiles_np,
                            res["score"],
                            np.asarray(res["selected_indices"], dtype=np.int32),
                            res["grid_h"],
                            res["grid_w"],
                            k_selected=res["k"],
                            sumS=float(np.sum(res["score"][np.isfinite(res["score"])])),
                            cnn_indices=cnn_indices,
                        )
                        base = os.path.splitext(os.path.basename(img_path))[0]
                        out_path = os.path.join(out_dir, f"{base}_cnn.png")
                        cv2.imwrite(out_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                        print(f"[VIZ] Saved {out_path}")
                    except Exception as e:
                        print(f"[WARN] Skipping viz for {img_path}: {e}")
        return
    if not args.calibration_csv or not os.path.isfile(args.calibration_csv):
        raise RuntimeError("Calibration CSV is required and was not found.")
    calib_df = pd.read_csv(args.calibration_csv)
    if calib_df.empty:
        raise RuntimeError("Calibration CSV is empty.")
    calib_row = calib_df.iloc[0].to_dict()
    # Robustly load all convertible values to ensure thresholds aren't dropped
    calib = {}
    for k, v in calib_row.items():
        try:
            calib[k] = float(v)
        except (ValueError, TypeError):
            continue
    
    # Optional: Print thresholds to confirm they are loaded
    print(f"[INFO] Loaded R10 Thresholds: Dense Max={calib.get('R10_dense_max', 'DEFAULT')}, Extreme Min={calib.get('R10_extreme_min', 'DEFAULT')}")

    os.makedirs(args.viz_dir, exist_ok=True) if args.viz_dir else None
    tiles_root = args.tiles_dir or args.viz_dir
    os.makedirs(tiles_root, exist_ok=True) if tiles_root else None

    if args.viz_dir:
        for man_path in args.manifests:
            df = pd.read_csv(man_path)
            man_base = os.path.splitext(os.path.basename(man_path))[0]
            out_dir = os.path.join(args.viz_dir, man_base)
            os.makedirs(out_dir, exist_ok=True)
            tiles_dir = None
            if tiles_root:
                tiles_dir = os.path.join(tiles_root, man_base)
                os.makedirs(tiles_dir, exist_ok=True)
            for _, row in df.iterrows():
                img_path = row[args.path_col]
                if not os.path.isfile(img_path):
                    print(f"[WARN] Missing image: {img_path}")
                    continue
                try:
                    res = rin_for_image_path_calib(img_path, calib)
                except Exception as e:
                    print(f"[WARN] Error processing {img_path}: {e}")
                    continue
                if res.get("skip", False):
                    print(f"[WARN] Skipping tiles/viz for {img_path}: {res.get('skip_reason', 'unknown')}")
                    continue
                img_rgb = None
                try:
                    img_rgb = load_image_rgb(img_path)
                except Exception as e:
                    print(f"[WARN] Skipping viz for {img_path}: {e}")
                if img_rgb is not None:
                    viz = visualize_tiles(img_rgb,
                                          res["tiles"],
                                          res["scores"],
                                          res["top_indices"],
                                          res["grid_h"],
                                          res["grid_w"],
                                          res.get("sparsity_label"),
                                          res.get("sparsity_metric"),
                                          res.get("sparsity_method"),
                                          r10=res.get("sparsity_metric"),
                                          neff_norm=res.get("sparsity_debug", {}).get("Neff_norm"),
                                          k_selected=res.get("K_selected"),
                                          sumS=res.get("E_total"))
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    out_path = os.path.join(out_dir, f"{base}_rin.png")
                    cv2.imwrite(out_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    print(f"[VIZ] Saved {out_path}")
                if tiles_dir:
                    tiles_path = os.path.join(tiles_dir, f"{base}_tiles.npy")
                    np.save(tiles_path, res["top_tiles"])
                    print(f"[TILES] Saved {tiles_path}")
    else:
        ds, meta = build_rin_dataset_from_manifests(args.manifests,
                                                    path_col=args.path_col,
                                                    dataset_col=args.dataset_col,
                                                    defocus_col=args.defocus_col,
                                                    calib=calib,
                                                    beta_override=args.beta_k,
                                                    diag_prefix=args.diag_prefix)
        print("Total selected tiles:", meta.shape[0])
        print(meta.head())
        for batch in ds.take(1):
            print("Batch shape:", batch.shape)


if __name__ == "__main__":
    main()
