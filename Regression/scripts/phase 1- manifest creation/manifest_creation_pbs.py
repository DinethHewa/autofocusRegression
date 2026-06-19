#!/usr/bin/env python3
"""
Builds a PBS-only manifest by:
- Discovering stacks in PBS_ROOT with 9 or 15 slices (keeps all slices).
- Converting images to grayscale (in-memory).
- Computing 13 focus measures per stack, normalizing each curve to 0-1, and inverting
  curves whose peak appears as a minimum so that all curves are peak-at-focus.
- Voting for the best-focused slice via majority of per-measure maxima.
- Assigning defocus distances (µm) using 0.5 µm step relative to the voted focus slice.
- Writing:
    * Manifest CSV with columns [image_path, defocus_um] to OUT_DIR/manifest_pbs.csv
    * Per-stack focus curves CSVs to REGRESSION_CURVES_DIR/PBS/<stack_id>_curves.csv
    * Full table (stack_id, image_name, image_path, defocus_um) to REGRESSION_TABLES_DIR/PBS/pbs_table.csv

Usage:
    python manifest_creation_pbs.py [--use-gpu]
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageFile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    HAVE_CUPY = False
    cp = None  # type: ignore

try:
    import pywt
except Exception as e:  # pragma: no cover - required for runtime use
    print("pywt (PyWavelets) is required.", file=sys.stderr)
    raise

try:
    from skimage.feature import graycomatrix, graycoprops
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_ubyte
    HAVE_SKI = True
except Exception:
    HAVE_SKI = False

# ------------------------------------
# Configuration
# ------------------------------------
PBS_ROOT = "/home/dineth/focus_measure/datasets/New folder/bma pbf tfa/pbs_imgs"
OUT_DIR = "/home/dineth/focus_measure/journal/Regression/data"
REGRESSION_CURVES_DIR = "/home/dineth/focus_measure/journal/Regression/curves"
REGRESSION_TABLES_DIR = "/home/dineth/focus_measure/journal/Regression/tables"
CURVE_SUBDIR = os.path.join(REGRESSION_CURVES_DIR, "PBS")
TABLE_SUBDIR = os.path.join(REGRESSION_TABLES_DIR, "PBS")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest_pbs.csv")
TABLE_PATH = os.path.join(TABLE_SUBDIR, "pbs_table.csv")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
STEP_PBS = 0.5  # µm
ALLOWED_SLICES = {9, 15}


# ------------------------------------
# Helpers
# ------------------------------------
def is_img(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMAGE_EXTS


def ensure_dirs() -> None:
    for d in [OUT_DIR, CURVE_SUBDIR, TABLE_SUBDIR]:
        os.makedirs(d, exist_ok=True)


def sort_by_trailing_number(paths: List[str]) -> List[str]:
    def key(p):
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)$", base)
        return int(m.group(1)) if m else base

    return sorted(paths, key=key)


def imread_gray(path: str) -> np.ndarray:
    """Read as grayscale uint8."""
    # First try OpenCV
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype == np.uint16:
            img = (img / 257.0).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img
    # Fallback to PIL (tolerates some CRC issues)
    try:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("L")
            return np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        raise IOError(f"Failed to read: {path} ({e})")


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    if np.all(np.isnan(curve)):
        return np.zeros_like(curve, dtype=np.float32)
    cmin, cmax = float(np.nanmin(curve)), float(np.nanmax(curve))
    if not np.isfinite(cmin) or not np.isfinite(cmax):
        return np.zeros_like(curve, dtype=np.float32)
    if abs(cmax - cmin) < 1e-12:
        return np.zeros_like(curve, dtype=np.float32)
    return (curve - cmin) / (cmax - cmin)


def orient_curve_to_peak(curve_norm: np.ndarray, center_idx: float) -> Tuple[np.ndarray, int]:
    """Ensure curve has a maximum at focus. If the minimum is closer to center, invert."""
    idx_max = int(np.nanargmax(curve_norm))
    idx_min = int(np.nanargmin(curve_norm))
    dist_max = abs(idx_max - center_idx)
    dist_min = abs(idx_min - center_idx)
    if dist_min < dist_max:
        oriented = 1.0 - curve_norm
        best_idx = idx_min
    else:
        oriented = curve_norm
        best_idx = idx_max
    return oriented, best_idx


@dataclass
class StackInfo:
    stack_id: str
    slice_paths: List[str]


def discover_pbs_stacks(root: str) -> List[StackInfo]:
    stacks: List[StackInfo] = []
    for sd in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
        subdir = os.path.join(root, sd)
        imgs = [os.path.join(subdir, p) for p in os.listdir(subdir) if is_img(p)]
        if len(imgs) not in ALLOWED_SLICES:
            continue
        imgs_sorted = sort_by_trailing_number(imgs)
        stacks.append(StackInfo(stack_id=sd, slice_paths=imgs_sorted))
    return stacks


# ------------------------------------
# Focus measures (GPU-optional)
# ------------------------------------
class FocusMeasures:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and HAVE_CUPY
        if use_gpu and not HAVE_CUPY:
            print("CuPy not available; falling back to NumPy.", file=sys.stderr)
        self.xp = cp if self.use_gpu else np

    def to_xp(self, arr: np.ndarray, float_type="float32"):
        if self.use_gpu:
            return cp.asarray(arr, dtype=float_type)
        return arr.astype(float_type)

    def ensure_numpy(self, arr):
        return cp.asnumpy(arr) if self.use_gpu else arr

    def convert_and_prepare(self, image: np.ndarray, float_type="float32", for_glcm=False):
        if for_glcm or not self.use_gpu:
            return image.astype(float_type)
        return self.to_xp(image, float_type=float_type)

    # Brenner Gradient
    def brenner_focus_measure(self, image):
        img = self.convert_and_prepare(image)
        return self.xp.sum((img[:-2, :] - img[2:, :]) ** 2)

    # Sum Modified Laplacian
    def sum_modified_laplacian(self, image):
        img = self.convert_and_prepare(image)
        Lx = 2 * img[1:-1, 1:-1] - img[2:, 1:-1] - img[:-2, 1:-1]
        Ly = 2 * img[1:-1, 1:-1] - img[1:-1, 2:] - img[1:-1, :-2]
        total = (self.xp.abs(Lx) + self.xp.abs(Ly)) ** 2
        return self.xp.sum(total)

    # Fourier High Frequency Energy Ratio
    def fourier_high_freq_energy_ratio(self, image, threshold_ratio=0.2):
        img = self.convert_and_prepare(image, float_type="float64")
        img_np = self.ensure_numpy(img)
        fft = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(fft)
        magnitude = np.abs(fshift) ** 2
        h, w = img_np.shape
        crow, ccol = h // 2, w // 2
        r = int(min(crow, ccol) * threshold_ratio)
        Y, X = np.ogrid[:h, :w]
        mask = (X - ccol) ** 2 + (Y - crow) ** 2 <= r ** 2
        low_energy = np.sum(magnitude[mask])
        total_energy = np.sum(magnitude)
        if total_energy > 1e-8:
            return (total_energy - low_energy) / total_energy
        else:
            return 0.0

    # Wavelet W1/W2/W3
    def _wavelet_energy(self, image: np.ndarray, level: int):
        img = self.convert_and_prepare(image)
        img_np = self.ensure_numpy(img)
        if min(img_np.shape) < 2 ** level:
            return np.nan
        try:
            coeffs = pywt.wavedec2(img_np, "haar", level=level)
            detail_coeffs = coeffs[-1]
            cH, cV, cD = detail_coeffs
            return np.sum(np.abs(cH)) + np.sum(np.abs(cV)) + np.sum(np.abs(cD))
        except Exception:
            return np.nan

    def wavelet_w1(self, image):
        return self._wavelet_energy(image, level=1)

    def wavelet_w2(self, image):
        return self._wavelet_energy(image, level=2)

    def wavelet_w3(self, image):
        return self._wavelet_energy(image, level=3)

    # Curvelet Transform Sharpness Index (mocked with db1 wavelet detail energy)
    def curvelet_transform_sharpness_index(self, image):
        img = self.convert_and_prepare(image)
        img_np = self.ensure_numpy(img)
        try:
            coeffs = pywt.wavedec2(img_np, "db1", level=1)
            detail_coeffs = coeffs[1:]
            return sum(np.sum(np.abs(c)) for subband in detail_coeffs for c in subband)
        except Exception:
            return np.nan

    # Squared Gradient (alias of gradient_squared_energy)
    def squared_gradient(self, image):
        return self.gradient_squared_energy(image)

    def gradient_squared_energy(self, image):
        img = self.convert_and_prepare(image)
        img_np = self.ensure_numpy(img)
        gx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        squared = gx ** 2 + gy ** 2
        return np.sum(squared)

    def roberts_focus_measure(self, image):
        img = self.convert_and_prepare(image)
        img_np = self.ensure_numpy(img)
        gx = img_np[1:, 1:] - img_np[:-1, :-1]
        gy = img_np[1:, :-1] - img_np[:-1, 1:]
        return np.sum(gx ** 2 + gy ** 2)

    def fourier_transform_sharpness_index(self, image):
        img = self.convert_and_prepare(image, float_type="float64")
        img_np = self.ensure_numpy(img).squeeze()
        fft = np.fft.fft2(img_np)
        spectrum = np.fft.fftshift(fft)
        return float(np.mean(np.abs(spectrum)))

    def intensity_skewness_index(self, image):
        img = self.convert_and_prepare(image)
        img_np = self.ensure_numpy(img).squeeze()
        if img_np.size == 0:
            return np.nan
        mean = np.mean(img_np)
        std = np.std(img_np)
        if std < 1e-8:
            return 0.0
        norm = ((img_np - mean) / (std + 1e-8)) ** 3
        return float(np.mean(norm))

    def glcm_contrast(self, image):
        if not HAVE_SKI:
            gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            g = np.sqrt(gx * gx + gy * gy)
            return float(g.var())
        img = image
        if img.dtype != np.uint8:
            img = img_as_ubyte(rescale_intensity(img, in_range="image", out_range=(0, 255)))
        glcm = graycomatrix(img, [1], [0], levels=256, symmetric=True, normed=True)
        return float(graycoprops(glcm, "contrast")[0, 0])


def get_focus_functions(measure: FocusMeasures):
    return [
        ("GLCM Contrast", measure.glcm_contrast),
        ("Intensity Skewness Index", measure.intensity_skewness_index),
        ("Fourier Transform Sharpness Index", measure.fourier_transform_sharpness_index),
        ("Brenner Gradient", measure.brenner_focus_measure),
        ("Fourier High Frequency Energy Ratio", measure.fourier_high_freq_energy_ratio),
        ("Wavelet W1", measure.wavelet_w1),
        ("Wavelet W2", measure.wavelet_w2),
        ("Wavelet W3", measure.wavelet_w3),
        ("Curvelet Transform Sharpness Index", measure.curvelet_transform_sharpness_index),
        ("Sum Modified Laplacian", measure.sum_modified_laplacian),
        ("Squared Gradient", measure.squared_gradient),
        ("Gradient Squared Energy", measure.gradient_squared_energy),
        ("Roberts Focus Measure", measure.roberts_focus_measure),
    ]


# ------------------------------------
# Core processing
# ------------------------------------
def process_stack(stack: StackInfo, fm: FocusMeasures) -> Tuple[int, Dict[str, np.ndarray], Dict[str, int]]:
    focus_funcs = get_focus_functions(fm)
    values: Dict[str, List[float]] = {name: [] for name, _ in focus_funcs}

    images: List[np.ndarray] = []
    valid_paths: List[str] = []
    for p in stack.slice_paths:
        try:
            img = imread_gray(p)
            images.append(img)
            valid_paths.append(p)
        except Exception as e:
            print(f"Skipping slice {p} due to read error: {e}")

    if len(images) < 2:
        print(f"Skipping stack {stack.stack_id}: insufficient readable slices ({len(images)})")
        return -1, {}, {}

    # Compute curves
    for img in images:
        for name, func in focus_funcs:
            try:
                val = func(img)
            except Exception:
                val = np.nan
            values[name].append(float(val))

    # Normalize and orient curves
    center_idx = (len(images) - 1) / 2.0
    oriented_curves: Dict[str, np.ndarray] = {}
    best_indices: Dict[str, int] = {}

    for name, vals in values.items():
        curve = np.array(vals, dtype=np.float32)
        norm = normalize_curve(curve)
        oriented, best_idx = orient_curve_to_peak(norm, center_idx)
        oriented_curves[name] = oriented
        best_indices[name] = best_idx

    # Majority vote on maxima
    vote_counts = np.zeros(len(images), dtype=int)
    for idx in best_indices.values():
        vote_counts[idx] += 1
    max_votes = vote_counts.max()
    candidate_indices = np.flatnonzero(vote_counts == max_votes)

    if len(candidate_indices) == 1:
        focus_idx = int(candidate_indices[0])
    else:
        summed = np.zeros(len(images), dtype=np.float32)
        for curve in oriented_curves.values():
            summed += curve
        best_sum = summed[candidate_indices].max()
        sum_candidates = [idx for idx in candidate_indices if summed[idx] == best_sum]
        if len(sum_candidates) == 1:
            focus_idx = int(sum_candidates[0])
        else:
            center = center_idx
            dist = [abs(idx - center) for idx in sum_candidates]
            focus_idx = int(sum_candidates[int(np.argmin(dist))])

    return focus_idx, oriented_curves, best_indices, valid_paths


def build_manifest(use_gpu: bool = False, flush_every: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    stacks = discover_pbs_stacks(PBS_ROOT)
    if not stacks:
        raise RuntimeError(f"No valid stacks (9 or 15 images) found in {PBS_ROOT}")

    fm = FocusMeasures(use_gpu=use_gpu)
    # Resume support
    existing_manifest = pd.read_csv(MANIFEST_PATH) if os.path.isfile(MANIFEST_PATH) else pd.DataFrame(columns=["image_path", "defocus_um"])
    existing_table = pd.read_csv(TABLE_PATH) if os.path.isfile(TABLE_PATH) else pd.DataFrame(columns=["stack_id", "image_name", "image_path", "defocus_um"])
    processed_stack_ids = set(existing_table["stack_id"].unique()) if not existing_table.empty else set()

    manifest_rows: List[Dict] = existing_manifest.to_dict(orient="records")
    table_rows: List[Dict] = existing_table.to_dict(orient="records")
    skipped = 0

    processed_since_flush = 0

    for stack in tqdm(stacks, desc="Processing stacks"):
        if stack.stack_id in processed_stack_ids:
            skipped += 1
            continue

        focus_idx, oriented_curves, best_indices, valid_paths = process_stack(stack, fm)
        if focus_idx < 0 or not valid_paths:
            continue

        # Save curves CSV
        curve_path = os.path.join(CURVE_SUBDIR, f"{stack.stack_id}_curves.csv")
        curve_df = pd.DataFrame({"slice_index": list(range(len(valid_paths)))})
        curve_df["image_name"] = [os.path.basename(p) for p in valid_paths]
        for name, curve in oriented_curves.items():
            curve_df[name] = curve
        curve_df["best_vote"] = [1 if i == focus_idx else 0 for i in range(len(stack.slice_paths))]
        curve_df.to_csv(curve_path, index=False)

        # Save plot of curves
        plot_path = os.path.join(CURVE_SUBDIR, f"{stack.stack_id}_curves.png")
        plt.figure(figsize=(10, 6))
        xs = list(range(len(valid_paths)))
        for name, curve in oriented_curves.items():
            plt.plot(xs, curve, label=name, linewidth=1.0)
        plt.axvline(focus_idx, color="red", linestyle="--", linewidth=1.2, label=f"focus idx {focus_idx}")
        plt.xlabel("Slice index")
        plt.ylabel("Normalized focus measure (peak=focus)")
        plt.title(f"Stack {stack.stack_id} focus curves")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # Add rows to manifest/table
        for i, img_path in enumerate(valid_paths):
            defocus_um = (focus_idx - i) * STEP_PBS
            manifest_rows.append(
                {
                    "image_path": img_path,
                    "defocus_um": defocus_um,
                }
            )
            table_rows.append(
                {
                    "stack_id": stack.stack_id,
                    "image_name": os.path.basename(img_path),
                    "image_path": img_path,
                    "defocus_um": defocus_um,
                }
            )

        processed_since_flush += 1
        if flush_every > 0 and processed_since_flush >= flush_every:
            pd.DataFrame(manifest_rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True).to_csv(
                MANIFEST_PATH, index=False
            )
            pd.DataFrame(table_rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True).to_csv(
                TABLE_PATH, index=False
            )
            processed_since_flush = 0
        if fm.use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks()  # type: ignore[attr-defined]
            except Exception:
                pass

    manifest_df = pd.DataFrame(manifest_rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    table_df = pd.DataFrame(table_rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    if skipped:
        print(f"Skipped already processed stacks (resume): {skipped}")
    return manifest_df, table_df


def main():
    parser = argparse.ArgumentParser(description="Create PBS manifest with focus-curve voting.")
    parser.add_argument("--use-gpu", action="store_true", help="Use CuPy if available for select ops.")
    parser.add_argument("--flush-every", type=int, default=200, help="Flush manifest/table every N stacks.")
    args = parser.parse_args()

    manifest_df, table_df = build_manifest(use_gpu=args.use_gpu, flush_every=args.flush_every)

    manifest_df.to_csv(MANIFEST_PATH, index=False)
    table_df.to_csv(TABLE_PATH, index=False)

    print(f"Stacks processed: {table_df['stack_id'].nunique()}")
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Table written to {TABLE_PATH}")
    print(f"Curve CSVs in {CURVE_SUBDIR}")


if __name__ == "__main__":
    main()
