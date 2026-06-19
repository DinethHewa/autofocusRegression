#!/usr/bin/env python3
"""
Manifest creator for the focusTrain dataset (Windows paths).
- Scans multiple root folders (recursively) for image files.
- Extracts defocus distance from filenames like: defocus150.jpg -> 0.15, defocus-350.jpg -> -0.35, defocus4650.jpg -> 4.65, defocus-5350.jpg -> -5.35.
- Writes manifest_focus_train.csv with columns [image_path, defocus_um] to OUT_DIR.
- Resume-aware and flushes periodically to avoid losing progress.

Usage:
    python manifest_creation_focus_train.py [--flush-every N] [--roots ROOT1 ROOT2 ...]
"""

import argparse
import os
import re
from typing import List, Dict

import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DEFAULT_ROOTS = [
    r"D:\Academic\Courses\Higher Education\Design and Development of Autofocus and Auto selection Microscopy\dataset\5936881 (1)\Data_channel\dualLED_greenChannel\train_dualLED_greenChannel",
    r"D:\Academic\Courses\Higher Education\Design and Development of Autofocus and Auto selection Microscopy\dataset\5936881 (1)\Data_channel\incoherent_RGBchannels\train_incoherent_RGBChannels",
]

OUT_DIR = "/home/dineth/focus_measure/journal/Regression/data"
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest_focus_train.csv")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def normalize_root_path(path: str) -> str:
    """If given a Windows drive path and running under WSL/Linux, map to /mnt/<drive>/..."""
    if os.path.isdir(path):
        return path
    m = re.match(r"([A-Za-z]):\\(.*)", path)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        candidate = f"/mnt/{drive}/{rest}"
        if os.path.isdir(candidate):
            return candidate
    return path


def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def is_img(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def parse_defocus_from_name(filename: str):
    """
    Extract defocus value from names like defocus150, defocus-350, defocus4650, defocus-5350.
    Returns defocus in micrometers (float) or None if not found.
    """
    m = re.search(r"defocus([+-]?\d+)", filename, flags=re.IGNORECASE)
    if not m:
        return None
    val_str = m.group(1)
    for ch in ["−", "–", "—", "\u2212", "\u2013", "\u2014"]:
        val_str = val_str.replace(ch, "-")
    try:
        val_int = int(val_str)
    except ValueError:
        return None
    return val_int / 1000.0


def iter_images(roots: List[str]):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if not is_img(fname):
                    continue
                full_path = os.path.join(dirpath, fname)
                yield full_path


# --------------------------------------------------
# Core
# --------------------------------------------------
def build_manifest(roots: List[str], flush_every: int = 500) -> pd.DataFrame:
    ensure_out_dir()
    # Rebuild manifest from scratch to ensure correct sign parsing
    rows: List[Dict] = []
    seen_paths = set()

    processed = 0
    skipped_no_defocus = 0
    for img_path in iter_images(roots):
        if img_path in seen_paths:
            continue
        defocus_um = parse_defocus_from_name(os.path.basename(img_path))
        if defocus_um is None:
            skipped_no_defocus += 1
            continue
        rows.append({"image_path": img_path, "defocus_um": defocus_um})
        seen_paths.add(img_path)
        processed += 1
        if flush_every > 0 and processed % flush_every == 0:
            pd.DataFrame(rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True).to_csv(MANIFEST_PATH, index=False)

    manifest_df = pd.DataFrame(rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    print(f"Images added: {processed}")
    print(f"Skipped (no defocus pattern): {skipped_no_defocus}")
    print(f"Total rows in manifest: {len(manifest_df)}")
    print(f"Manifest written to {MANIFEST_PATH}")
    return manifest_df


def main():
    parser = argparse.ArgumentParser(description="Create focusTrain manifest by parsing defocus from filenames.")
    parser.add_argument("--flush-every", type=int, default=500, help="Flush manifest every N new rows (default 500).")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=DEFAULT_ROOTS,
        help="Override roots to scan (Windows paths allowed).",
    )
    args = parser.parse_args()
    norm_roots = [normalize_root_path(r) for r in args.roots]
    missing = [r for r in norm_roots if not os.path.isdir(r)]
    if missing:
        print("Warning: these roots do not exist and will be skipped:", missing)
    build_manifest(roots=norm_roots, flush_every=args.flush_every)


if __name__ == "__main__":
    main()
