#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation_utils import (
    default_eval_cli,
    get_paths,
    load_phase5_index,
    log,
    per_bin_regression,
    regression_metrics,
    require_file,
    save_json,
    save_plot,
    save_table,
)


def _plot_error_hist(errors: np.ndarray, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=40)
    ax.set_xlabel("Absolute Error (µm)")
    ax.set_ylabel("Count")
    ax.set_title("Stage B Regression Error Histogram")
    ax.grid(True, alpha=0.25)
    save_plot(fig, out_path)


def _plot_error_vs_mag(y_mag: np.ndarray, abs_err: np.ndarray, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_mag, abs_err, s=8, alpha=0.25)
    # trend line by bins
    edges = np.linspace(float(np.nanmin(y_mag)), float(np.nanmax(y_mag) + 1e-9), 20)
    centers = []
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (y_mag >= lo) & (y_mag < hi)
        if idx.any():
            centers.append(0.5 * (lo + hi))
            means.append(float(np.mean(abs_err[idx])))
    if centers:
        ax.plot(centers, means, color="red", linewidth=2)
    ax.set_xlabel("Ground Truth |Δz| (µm)")
    ax.set_ylabel("Absolute Error (µm)")
    ax.set_title("Regression Error vs Magnitude")
    ax.grid(True, alpha=0.25)
    save_plot(fig, out_path)


def _plot_box_per_bin(y_mag: np.ndarray, abs_err: np.ndarray, bin_width: float, out_path):
    edges = np.arange(0, float(np.nanmax(y_mag) + bin_width + 1e-9), float(bin_width))
    groups = []
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (y_mag >= lo) & (y_mag < hi)
        if idx.any():
            groups.append(abs_err[idx])
            labels.append(f"{lo:.1f}-{hi:.1f}")
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(labels)), 4))
    if groups:
        ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_xlabel("|Δz| bin (µm)")
    ax.set_ylabel("Absolute Error (µm)")
    ax.set_title("Regression Error Boxplot per Magnitude Bin")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, alpha=0.25)
    save_plot(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Stage B magnitude regression")
    default_eval_cli(ap)
    args = ap.parse_args()

    paths = get_paths(args.track)
    log(paths, "Starting Stage B evaluation")
    required = [
        paths.eval_dir / "stageB_metrics.json",
        paths.eval_dir / "stageB_bin_mae.csv",
        paths.tables_dir / "Table_StageB.csv",
    ]
    if args.save_latex:
        required.append((paths.tables_dir / "Table_StageB.csv").with_suffix(".tex"))
    if args.save_plots:
        required.extend(
            [
                paths.eval_dir / "regression_error_histogram.png",
                paths.eval_dir / "regression_error_vs_magnitude.png",
                paths.eval_dir / "regression_error_boxplot_per_bin.png",
            ]
        )
    if args.resume and all(p.is_file() for p in required):
        log(paths, f"Resume: Stage B outputs already exist; skipping. ({paths.eval_dir})")
        return

    roi_path = require_file(paths.reg_dir / "inference" / "roi_predictions.csv", "Phase5 ROI predictions")
    idx = load_phase5_index(args.track)

    roi = pd.read_csv(roi_path)
    if "roi_uid" not in roi.columns:
        raise ValueError(f"roi_predictions missing roi_uid: {roi_path}")

    if "y_mag_pred" not in roi.columns:
        if "signed_dz_pred" in roi.columns:
            roi["y_mag_pred"] = pd.to_numeric(roi["signed_dz_pred"], errors="coerce").abs()
        else:
            raise ValueError(f"roi_predictions missing y_mag_pred and signed_dz_pred: {roi_path}")

    merged = roi.merge(idx[["roi_uid", "y_mag_um", "defocus_um", "dataset"]], on="roi_uid", how="left")
    merged["y_mag_pred"] = pd.to_numeric(merged["y_mag_pred"], errors="coerce")
    merged["y_mag_um"] = pd.to_numeric(merged["y_mag_um"], errors="coerce")

    valid = merged["y_mag_pred"].notna() & merged["y_mag_um"].notna()
    merged = merged[valid].copy().reset_index(drop=True)
    if merged.empty:
        raise ValueError("No valid Stage B rows after joining predictions with index_phase5")

    y_true = merged["y_mag_um"].to_numpy(dtype=float)
    y_pred = merged["y_mag_pred"].to_numpy(dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)

    m = regression_metrics(y_true, y_pred)
    near = y_true <= 2.0
    far = y_true > 5.0

    metrics = {
        **m,
        "near_focus_mae_um_le_2": float(np.mean(abs_err[near])) if near.any() else float("nan"),
        "far_focus_mae_um_gt_5": float(np.mean(abs_err[far])) if far.any() else float("nan"),
        "n_samples": int(len(merged)),
    }

    bin_df = per_bin_regression(abs_delta=y_true, abs_err=abs_err, bin_width=float(args.bins))

    save_json(metrics, paths.eval_dir / "stageB_metrics.json")
    bin_df.to_csv(paths.eval_dir / "stageB_bin_mae.csv", index=False)

    if args.save_plots:
        _plot_error_hist(abs_err, paths.eval_dir / "regression_error_histogram.png")
        _plot_error_vs_mag(y_true, abs_err, paths.eval_dir / "regression_error_vs_magnitude.png")
        _plot_box_per_bin(y_true, abs_err, float(args.bins), paths.eval_dir / "regression_error_boxplot_per_bin.png")

    table_df = pd.DataFrame(
        [
            {
                "track": args.track,
                "MAE_um": metrics["mae_um"],
                "RMSE_um": metrics["rmse_um"],
                "MedianAbsErr_um": metrics["median_abs_error_um"],
                "P95AbsErr_um": metrics["p95_abs_error_um"],
                "NearFocusMAE_|dz|<=2": metrics["near_focus_mae_um_le_2"],
                "FarFocusMAE_|dz|>5": metrics["far_focus_mae_um_gt_5"],
                "N": metrics["n_samples"],
            }
        ]
    )
    save_table(table_df, paths.tables_dir / "Table_StageB.csv", save_latex=bool(args.save_latex))

    log(paths, f"Stage B evaluation complete. Outputs: {paths.eval_dir}")


if __name__ == "__main__":
    main()
