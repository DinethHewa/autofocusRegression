#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _ensure_2d_finite(arr: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    out = arr.astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN/Inf values")
    return out


def _check_shape(arr: np.ndarray, roi_size: int, name: str) -> None:
    if arr.shape != (roi_size, roi_size):
        raise ValueError(f"{name} must be {roi_size}x{roi_size}, got {arr.shape}")


def _zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = float(arr.mean())
    std = float(arr.std())
    return (arr - mu) / (std + eps)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        rgb = img[:, :, :3].astype(np.float32)
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {img.shape}")
    return _ensure_2d_finite(gray, "I_gray")


def percentile_clip(I: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    I = _ensure_2d_finite(I, "I")
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError(f"Invalid percentile range: p_low={p_low}, p_high={p_high}")
    lo = float(np.percentile(I, p_low))
    hi = float(np.percentile(I, p_high))
    if hi <= lo:
        return np.zeros_like(I, dtype=np.float32)
    out = np.clip(I, lo, hi)
    return _ensure_2d_finite(out, "I_clip")


def rescale01(I: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    I = _ensure_2d_finite(I, "I")
    lo = float(I.min())
    hi = float(I.max())
    out = (I - lo) / (hi - lo + eps)
    return _ensure_2d_finite(out, "I01")


def zscore(I: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    I = _ensure_2d_finite(I, "I")
    out = _zscore(I, eps=eps)
    return _ensure_2d_finite(out, "Iz")


def _kernel_size(sigma: float) -> int:
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return int(2 * math.ceil(3 * sigma) + 1)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    k = _kernel_size(sigma)
    r = k // 2
    x = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g /= np.sum(g)
    return g.astype(np.float32)


def _convolve_reflect_1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = kernel.shape[0] // 2
    if axis == 1:
        padded = np.pad(arr, ((0, 0), (radius, radius)), mode="reflect")
        out = np.empty_like(arr, dtype=np.float32)
        for i in range(arr.shape[0]):
            out[i, :] = np.convolve(padded[i, :], kernel, mode="valid")
        return out
    if axis == 0:
        padded = np.pad(arr, ((radius, radius), (0, 0)), mode="reflect")
        out = np.empty_like(arr, dtype=np.float32)
        for j in range(arr.shape[1]):
            out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
        return out
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def gaussian_blur(I: np.ndarray, sigma: float) -> np.ndarray:
    I = _ensure_2d_finite(I, "I")
    k = _kernel_size(sigma)
    if cv2 is not None:
        out = cv2.GaussianBlur(I, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        return _ensure_2d_finite(out, "B")

    kernel = _gaussian_kernel_1d(sigma)
    tmp = _convolve_reflect_1d(I, kernel, axis=1)
    out = _convolve_reflect_1d(tmp, kernel, axis=0)
    return _ensure_2d_finite(out, "B")


def compute_dog_lite(I: np.ndarray, sigmas: Tuple[float, float, float], eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    I = _ensure_2d_finite(I, "I")
    sigma1, sigma2, sigma3 = sigmas
    B1 = gaussian_blur(I, sigma1)
    B2 = gaussian_blur(I, sigma2)
    B3 = gaussian_blur(I, sigma3)

    D1 = zscore(B1 - B2, eps=eps)
    D2 = zscore(B2 - B3, eps=eps)
    return D1.astype(np.float32), D2.astype(np.float32)


def _resize_bilinear(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if cv2 is not None:
        out = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def compute_dwt_ehf(I: np.ndarray, wavelet: str = "haar", roi_size: int = 200, eps: float = 1e-6) -> np.ndarray:
    I = _ensure_2d_finite(I, "I")
    if wavelet.lower() not in {"haar", "db1"}:
        raise ValueError(f"Only Haar/db1 is supported, got '{wavelet}'")

    H, W = I.shape
    if H % 2 != 0:
        I = np.pad(I, ((0, 1), (0, 0)), mode="reflect")
    if W % 2 != 0:
        I = np.pad(I, ((0, 0), (0, 1)), mode="reflect")

    a = I[0::2, 0::2]
    b = I[0::2, 1::2]
    c = I[1::2, 0::2]
    d = I[1::2, 1::2]

    LH = (a + b - c - d) * 0.5
    HL = (a - b + c - d) * 0.5
    HH = (a - b - c + d) * 0.5

    E_HF = np.sqrt(LH * LH + HL * HL + HH * HH).astype(np.float32)
    E_HF = _resize_bilinear(E_HF, roi_size, roi_size)
    E_HF = zscore(E_HF, eps=eps)
    _check_shape(E_HF, roi_size, "E_HF")
    return E_HF.astype(np.float32)


def assemble_XA_XB(I: np.ndarray, D1: np.ndarray, D2: np.ndarray, EHF: np.ndarray, roi_size: int = 200) -> tuple[np.ndarray, np.ndarray]:
    I = _ensure_2d_finite(I, "I")
    D1 = _ensure_2d_finite(D1, "D1")
    D2 = _ensure_2d_finite(D2, "D2")
    EHF = _ensure_2d_finite(EHF, "E_HF")

    _check_shape(I, roi_size, "I")
    _check_shape(D1, roi_size, "D1")
    _check_shape(D2, roi_size, "D2")
    _check_shape(EHF, roi_size, "E_HF")

    XA = np.stack([I, D1, D2], axis=-1).astype(np.float32)
    XB = np.stack([I, D1, D2, EHF], axis=-1).astype(np.float32)

    if XA.shape != (roi_size, roi_size, 3):
        raise ValueError(f"XA shape mismatch: {XA.shape}")
    if XB.shape != (roi_size, roi_size, 4):
        raise ValueError(f"XB shape mismatch: {XB.shape}")
    if not np.isfinite(XA).all() or not np.isfinite(XB).all():
        raise ValueError("XA/XB contains NaN/Inf")

    return XA, XB
