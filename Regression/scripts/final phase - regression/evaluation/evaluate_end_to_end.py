#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation_utils import (
    default_eval_cli,
    get_paths,
    infer_dataset_for_fov,
    load_phase5_index,
    log,
    save_json,
    save_plot,
    save_table,
    weighted_median,
)


def _gt_per_fov(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fov_id, g in index_df.groupby("fov_id", sort=True):
        vals = pd.to_numeric(g["defocus_um"], errors="coerce").to_numpy(dtype=float)
        if "roi_importance" in g.columns:
            w = pd.to_numeric(g["roi_importance"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        else:
            w = np.ones((len(g),), dtype=float)
        gt = weighted_median(vals, w)
        if gt is None:
            gt = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
        rows.append({"fov_id": str(fov_id), "gt_signed_um": gt})
    return pd.DataFrame(rows)


def _plot_hist(errors: np.ndarray, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=40)
    ax.set_xlabel("Signed Error (µm)")
    ax.set_ylabel("Count")
    ax.set_title("End-to-End Signed Error Histogram")
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def _plot_cdf(abs_errors: np.ndarray, out_path):
    x = np.sort(abs_errors)
    y = np.arange(1, len(x) + 1) / max(len(x), 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.set_xlabel("Absolute Error (µm)")
    ax.set_ylabel("CDF")
    ax.set_title("End-to-End Error CDF")
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def _plot_convergence(mean_abs_error: float, out_path):
    # Simple illustrative controller curve: residual error decays exponentially.
    t = np.arange(0, 10)
    curve = mean_abs_error * np.power(0.6, t)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, curve, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Estimated Residual Error (µm)")
    ax.set_title("Convergence Simulation (Illustrative)")
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate end-to-end autofocus metrics")
    default_eval_cli(ap)
    ap.add_argument("--simulate-convergence", action="store_true")
    args = ap.parse_args()

    paths = get_paths(args.track)
    log(paths, "Starting end-to-end evaluation")
    required = [
        paths.eval_dir / "end_to_end_metrics.json",
        paths.eval_dir / "catastrophic_rate.csv",
        paths.tables_dir / "Table_EndToEnd.csv",
    ]
    if args.save_latex:
        required.append((paths.tables_dir / "Table_EndToEnd.csv").with_suffix(".tex"))
    if args.save_plots:
        required.extend([paths.eval_dir / "signed_error_histogram.png", paths.eval_dir / "error_cdf.png"])
        if args.simulate_convergence:
            required.append(paths.eval_dir / "convergence_curves.png")
    if args.resume and all(p.is_file() for p in required):
        log(paths, f"Resume: end-to-end outputs already exist; skipping. ({paths.eval_dir})")
        return

    pred_path = paths.reg_dir / "inference" / "fov_aggregate_predictions.csv"
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"Missing end-to-end prediction file: {pred_path}. Run Phase5 inference first."
        )

    preds = pd.read_csv(pred_path)
    if "fov_id" not in preds.columns or "dz_hat_um" not in preds.columns:
        raise ValueError(f"{pred_path} must contain fov_id and dz_hat_um")

    index_df = load_phase5_index(args.track)
    gt_df = _gt_per_fov(index_df)
    ds_map = infer_dataset_for_fov(index_df)

    merged = preds.merge(gt_df, on="fov_id", how="left").merge(ds_map, on="fov_id", how="left")
    merged["pred_signed_um"] = pd.to_numeric(merged["dz_hat_um"], errors="coerce")
    merged["gt_signed_um"] = pd.to_numeric(merged["gt_signed_um"], errors="coerce")

    valid = merged["pred_signed_um"].notna() & merged["gt_signed_um"].notna()
    eval_df = merged[valid].copy().reset_index(drop=True)

    if eval_df.empty:
        raise ValueError("No valid numeric end-to-end predictions found (all uncertain or missing GT).")

    err = eval_df["pred_signed_um"].to_numpy(dtype=float) - eval_df["gt_signed_um"].to_numpy(dtype=float)
    abs_err = np.abs(err)

    gt = eval_df["gt_signed_um"].to_numpy(dtype=float)
    pred = eval_df["pred_signed_um"].to_numpy(dtype=float)

    catastrophic = (np.sign(pred) != np.sign(gt)) & (~np.isclose(gt, 0.0))
    overshoot = np.abs(pred) > np.abs(gt)

    metrics = {
        "n_fov_eval": int(len(eval_df)),
        "mean_signed_error_um_bias": float(np.mean(err)),
        "mae_um": float(np.mean(abs_err)),
        "rmse_um": float(np.sqrt(np.mean(np.square(err)))),
        "pct_within_1um": float(np.mean(abs_err <= 1.0) * 100.0),
        "pct_within_2um": float(np.mean(abs_err <= 2.0) * 100.0),
        "catastrophic_wrong_direction_rate": float(np.mean(catastrophic) * 100.0),
        "overshoot_rate": float(np.mean(overshoot) * 100.0),
        "uncertain_rate": float(np.mean(merged["pred_signed_um"].isna()) * 100.0),
    }

    save_json(metrics, paths.eval_dir / "end_to_end_metrics.json")
    pd.DataFrame(
        [
            {
                "catastrophic_wrong_direction_rate_pct": metrics["catastrophic_wrong_direction_rate"],
                "n_fov": int(len(eval_df)),
            }
        ]
    ).to_csv(paths.eval_dir / "catastrophic_rate.csv", index=False)

    if args.save_plots:
        _plot_hist(err, paths.eval_dir / "signed_error_histogram.png")
        _plot_cdf(abs_err, paths.eval_dir / "error_cdf.png")
        if args.simulate_convergence:
            _plot_convergence(float(np.mean(abs_err)), paths.eval_dir / "convergence_curves.png")

    table_df = pd.DataFrame(
        [
            {
                "track": args.track,
                "Bias_um": metrics["mean_signed_error_um_bias"],
                "MAE_um": metrics["mae_um"],
                "RMSE_um": metrics["rmse_um"],
                "Within_1um_pct": metrics["pct_within_1um"],
                "Within_2um_pct": metrics["pct_within_2um"],
                "CatastrophicWrongDir_pct": metrics["catastrophic_wrong_direction_rate"],
                "Overshoot_pct": metrics["overshoot_rate"],
                "Uncertain_pct": metrics["uncertain_rate"],
                "N_FOV": metrics["n_fov_eval"],
            }
        ]
    )
    save_table(table_df, paths.tables_dir / "Table_EndToEnd.csv", save_latex=bool(args.save_latex))

    log(paths, f"End-to-end evaluation complete. Outputs: {paths.eval_dir}")


if __name__ == "__main__":
    main()
