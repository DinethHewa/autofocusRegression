#!/usr/bin/env python3
"""
beta_sweep.py

Run roi_selection_v2 over a manifest for a list of beta values and report
diagnostics (avg k_selected, skip rate, runtime estimate per beta).

This script reuses rin_for_image_path_calib from roi_selection_v2.py.
"""

import argparse
import os
import time
from typing import List, Dict, Any

import pandas as pd

from roi_selection_v2 import (
    load_image_rgb,
    rin_for_image_path_calib,
)


def parse_beta_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",")] if s else []


def run_beta_sweep(manifest: str,
                   calib_csv: str,
                   beta_list: List[float],
                   path_col: str,
                   dataset_col: str,
                   max_images: int = None,
                   seed: int = 42,
                   mode: str = "legacy",
                   cnn_calib_csv: str | None = None,
                   cnn_tau_empty: float = 0.2,
                   cellness_predictor: str | None = None,
                   predictor_mode: str = "auto",
                   dummy_threshold: float = 0.5,
                   use_sum_occ: bool = True) -> pd.DataFrame:
    calib = pd.read_csv(calib_csv).iloc[0].to_dict() if calib_csv else {}
    calib = {k: float(v) for k, v in calib.items() if isinstance(v, (int, float))}

    cnn_calib = None
    if mode == "cnn" and cnn_calib_csv:
        df = pd.read_csv(cnn_calib_csv)
        if not df.empty:
            cnn_calib = {k: float(v) for k, v in df.iloc[0].to_dict().items() if isinstance(v, (int, float))}

    df = pd.read_csv(manifest)
    if max_images is not None and len(df) > max_images:
        df = df.sample(n=max_images, random_state=seed)

    results: List[Dict[str, Any]] = []
    if mode == "cnn":
        from roi_selection_cnn_v1 import ROISelectorCNNConfig, select_rois_cnn, _load_predictor
        predictor = _load_predictor(cellness_predictor, predictor_mode, dummy_threshold)

    for beta in beta_list:
        total_imgs = 0
        skipped = 0
        k_sum = 0
        k_count = 0
        t0 = time.time()
        for _, row in df.iterrows():
            img_path = row[path_col]
            dataset = row[dataset_col] if dataset_col in row else "unknown"
            total_imgs += 1
            if not os.path.isfile(img_path):
                skipped += 1
                continue
            try:
                if mode == "legacy":
                    res = rin_for_image_path_calib(img_path, calib, beta_override=beta)
                else:
                    img = load_image_rgb(img_path)
                    cfg = ROISelectorCNNConfig(
                        tau_empty=float(cnn_calib.get("tau_empty", cnn_tau_empty)) if cnn_calib else cnn_tau_empty,
                        beta=beta,
                        use_sum_occ=use_sum_occ,
                    )
                    res = select_rois_cnn(img, cnn_calib, cfg, predictor)
            except Exception:
                skipped += 1
                continue
            if res.get("skip", False):
                skipped += 1
                continue
            k_sum += int(res.get("K_selected", res.get("k", 0)))
            k_count += 1
        elapsed = time.time() - t0
        results.append({
            "beta": beta,
            "total_images": total_imgs,
            "skipped": skipped,
            "skip_rate": skipped / total_imgs if total_imgs else 0.0,
            "avg_k_selected": (k_sum / k_count) if k_count else 0.0,
            "images_used": k_count,
            "runtime_sec": elapsed,
        })
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description="Beta sweep for roi_selection_v2")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--calibration-csv", required=True)
    ap.add_argument("--beta-list", required=True, help="Comma-separated beta values, e.g., 0.1,0.2,0.3")
    ap.add_argument("--path-col", default="img_path")
    ap.add_argument("--dataset-col", default="dataset")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--mode", default="legacy", choices=["legacy", "cnn"])
    ap.add_argument("--cnn-calibration-csv", default=None)
    ap.add_argument("--cnn-tau-empty", type=float, default=0.2)
    ap.add_argument("--cellness-predictor", default=None)
    ap.add_argument("--predictor-mode", default="auto", choices=["auto", "map", "tiles"])
    ap.add_argument("--dummy-threshold", type=float, default=0.5)
    ap.add_argument("--use-sum-occ", action="store_true", default=True)
    ap.add_argument("--no-use-sum-occ", dest="use_sum_occ", action="store_false")
    args = ap.parse_args()

    betas = parse_beta_list(args.beta_list)
    df = run_beta_sweep(
        manifest=args.manifest,
        calib_csv=args.calibration_csv,
        beta_list=betas,
        path_col=args.path_col,
        dataset_col=args.dataset_col,
        max_images=args.max_images,
        seed=args.seed,
        mode=args.mode,
        cnn_calib_csv=args.cnn_calibration_csv,
        cnn_tau_empty=args.cnn_tau_empty,
        cellness_predictor=args.cellness_predictor,
        predictor_mode=args.predictor_mode,
        dummy_threshold=args.dummy_threshold,
        use_sum_occ=args.use_sum_occ,
    )
    df.to_csv(args.out_csv, index=False)
    print(df)
    print(f"[INFO] Saved beta sweep results to {args.out_csv}")


if __name__ == "__main__":
    main()
