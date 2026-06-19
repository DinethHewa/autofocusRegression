#!/usr/bin/env python3
"""
robustness_test.py

Runs ROI selection twice per image (original vs. perturbed) and measures
selection stability via Jaccard overlap.
"""

import argparse
import os
from typing import List, Dict, Any

import cv2
import numpy as np
import pandas as pd

from roi_selection_v2 import (
    load_image_rgb,
    rin_for_image_path_calib,
    rin_for_image_rgb_calib,
)


def apply_perturbation(img: np.ndarray,
                       blur_sigma: float = 0.0,
                       noise_std: float = 0.0,
                       brightness: float = 0.0,
                       contrast: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur, additive Gaussian noise, brightness/contrast."""
    out = img.copy()
    if blur_sigma > 0:
        ksize = max(1, int(blur_sigma * 3) * 2 + 1)
        out = cv2.GaussianBlur(out, (ksize, ksize), blur_sigma)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, out.shape).astype(np.float32)
        out = out + noise
    if contrast != 1.0 or brightness != 0.0:
        out = out * contrast + brightness
    return np.clip(out, 0.0, 1.0)


def jaccard(set_a: List[int], set_b: List[int]) -> float:
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def run_test(manifest: str,
             calib_csv: str,
             out_csv: str,
             blur_sigmas: List[float],
             noise_stds: List[float],
             brightness_list: List[float],
             contrast_list: List[float],
             path_col: str,
             dataset_col: str,
             mode: str = "legacy",
             cnn_calib_csv: str | None = None,
             cnn_tau_empty: float = 0.2,
             cellness_predictor: str | None = None,
             predictor_mode: str = "auto",
             dummy_threshold: float = 0.5) -> None:
    calib = pd.read_csv(calib_csv).iloc[0].to_dict()
    calib = {k: float(v) for k, v in calib.items() if isinstance(v, (int, float, np.integer, np.floating))}

    df = pd.read_csv(manifest)
    records: List[Dict[str, Any]] = []

    cnn_calib = None
    if mode == "cnn" and cnn_calib_csv:
        df_c = pd.read_csv(cnn_calib_csv)
        if not df_c.empty:
            cnn_calib = {k: float(v) for k, v in df_c.iloc[0].to_dict().items() if isinstance(v, (int, float))}
    if mode == "cnn":
        from roi_selection_cnn_v1 import ROISelectorCNNConfig, select_rois_cnn, _load_predictor
        predictor = _load_predictor(cellness_predictor, predictor_mode, dummy_threshold)

    for _, row in df.iterrows():
        img_path = row[path_col]
        dataset = row[dataset_col] if dataset_col in df.columns else "unknown"
        if not os.path.isfile(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue
        try:
            img = load_image_rgb(img_path)
            if mode == "legacy":
                # Use in-memory API so robustness uses identical codepath as perturbations.
                res_orig = rin_for_image_rgb_calib(img, calib)
                sel_orig = res_orig.get("top_indices", [])
            else:
                cfg = ROISelectorCNNConfig(
                    tau_empty=float(cnn_calib.get("tau_empty", cnn_tau_empty)) if cnn_calib else cnn_tau_empty,
                )
                res_orig = select_rois_cnn(img, cnn_calib, cfg, predictor)
                sel_orig = res_orig.get("selected_indices", [])
        except Exception as e:
            print(f"[WARN] Failed original selection {img_path}: {e}")
            continue

        for bs in blur_sigmas:
            for ns in noise_stds:
                for br in brightness_list:
                    for ct in contrast_list:
                        sel_pert = []
                        try:
                            pert = apply_perturbation(img, blur_sigma=bs, noise_std=ns, brightness=br, contrast=ct)
                            # IMPORTANT: run ROI selection on the *perturbed* image, not by reloading img_path.
                            if mode == "legacy":
                                res_pert = rin_for_image_rgb_calib(pert, calib)
                                sel_pert = res_pert.get("top_indices", [])
                            else:
                                res_pert = select_rois_cnn(pert, cnn_calib, cfg, predictor)
                                sel_pert = res_pert.get("selected_indices", [])
                            jac = jaccard(sel_orig, sel_pert)
                        except Exception as e:
                            print(f"[WARN] Perturbed selection failed {img_path}: {e}")
                            jac = np.nan
                            sel_pert = []
                        records.append({
                            "image_path": img_path,
                            "dataset": dataset,
                            "blur_sigma": bs,
                            "noise_std": ns,
                            "brightness": br,
                            "contrast": ct,
                            "jaccard": jac,
                            "k_orig": len(sel_orig),
                            "k_pert": len(sel_pert),
                        })

    if not records:
        print("No records collected; exiting.")
        return
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved robustness results to {out_csv}")
    agg = out_df.groupby("dataset")["jaccard"].mean().reset_index()
    print("[INFO] Mean Jaccard by dataset:")
    print(agg)


def parse_list_arg(s: str) -> List[float]:
    return [float(x) for x in s.split(",")] if s else []


def main():
    ap = argparse.ArgumentParser(description="ROI selection robustness test (selection stability under perturbations)")
    ap.add_argument("--manifest", required=True, help="Manifest CSV")
    ap.add_argument("--calibration-csv", required=True, help="Calibration CSV")
    ap.add_argument("--out-csv", required=True, help="Output CSV for robustness results")
    ap.add_argument("--path-col", default="img_path")
    ap.add_argument("--dataset-col", default="dataset")
    ap.add_argument("--blur-sigmas", default="0,1.0", help="Comma-separated sigmas")
    ap.add_argument("--noise-stds", default="0,0.01", help="Comma-separated stds (0-1 scale)")
    ap.add_argument("--brightness", default="0", help="Comma-separated brightness offsets")
    ap.add_argument("--contrast", default="1.0", help="Comma-separated contrasts")
    ap.add_argument("--mode", default="legacy", choices=["legacy", "cnn"])
    ap.add_argument("--cnn-calibration-csv", default=None)
    ap.add_argument("--cnn-tau-empty", type=float, default=0.2)
    ap.add_argument("--cellness-predictor", default=None)
    ap.add_argument("--predictor-mode", default="auto", choices=["auto", "map", "tiles"])
    ap.add_argument("--dummy-threshold", type=float, default=0.5)
    args = ap.parse_args()

    run_test(
        manifest=args.manifest,
        calib_csv=args.calibration_csv,
        out_csv=args.out_csv,
        blur_sigmas=parse_list_arg(args.blur_sigmas),
        noise_stds=parse_list_arg(args.noise_stds),
        brightness_list=parse_list_arg(args.brightness),
        contrast_list=parse_list_arg(args.contrast),
        path_col=args.path_col,
        dataset_col=args.dataset_col,
        mode=args.mode,
        cnn_calib_csv=args.cnn_calibration_csv,
        cnn_tau_empty=args.cnn_tau_empty,
        cellness_predictor=args.cellness_predictor,
        predictor_mode=args.predictor_mode,
        dummy_threshold=args.dummy_threshold,
    )


if __name__ == "__main__":
    main()
