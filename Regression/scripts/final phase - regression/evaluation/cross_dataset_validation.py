#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation_utils import (
    auroc_score,
    default_eval_cli,
    get_paths,
    infer_dataset_for_fov,
    load_phase5_index,
    log,
    save_plot,
)


def _load_stageA_dataset_metrics(paths, idx: pd.DataFrame) -> pd.DataFrame:
    lodoo = paths.sign_dir / "metrics" / "lodoo_results.csv"
    if lodoo.is_file():
        df = pd.read_csv(lodoo)
        if {"held_out_dataset", "auroc"}.issubset(df.columns):
            out = df.groupby("held_out_dataset", as_index=False)["auroc"].mean().rename(columns={"held_out_dataset": "dataset", "auroc": "stageA_auroc"})
            out["stageA_source"] = "lodoo"
            return out

    per_roi = paths.sign_dir / "inference" / "vote_sign_per_roi.csv"
    val_preds = paths.sign_dir / "calibration" / "val_roi_predictions.csv"

    if per_roi.is_file():
        r = pd.read_csv(per_roi)
        if {"roi_uid", "p"}.issubset(r.columns):
            label_cols = ["roi_uid", "y_sign"]
            if "dataset" not in r.columns:
                label_cols.append("dataset")
            merged = r.merge(idx[label_cols], on="roi_uid", how="left")
            rows = []
            dataset_col = "dataset"
            if dataset_col not in merged.columns:
                raise ValueError("Stage A per-ROI merge did not produce a dataset column.")
            for ds, g in merged.groupby(dataset_col):
                g = g.dropna(subset=["y_sign", "p"])
                if len(g) == 0:
                    auc = np.nan
                else:
                    auc = auroc_score(g["y_sign"].to_numpy(dtype=int), g["p"].to_numpy(dtype=float))
                rows.append({"dataset": ds, "stageA_auroc": auc, "stageA_source": "proxy_per_roi"})
            return pd.DataFrame(rows)

    if val_preds.is_file():
        v = pd.read_csv(val_preds)
        if {"dataset", "y_sign", "p"}.issubset(v.columns):
            rows = []
            for ds, g in v.groupby("dataset"):
                g = g.dropna(subset=["y_sign", "p"])
                auc = auroc_score(g["y_sign"].to_numpy(dtype=int), g["p"].to_numpy(dtype=float)) if len(g) else np.nan
                rows.append({"dataset": ds, "stageA_auroc": auc, "stageA_source": "proxy_val"})
            return pd.DataFrame(rows)

    return pd.DataFrame(columns=["dataset", "stageA_auroc", "stageA_source"])


def _load_stageB_dataset_metrics(paths, idx: pd.DataFrame) -> pd.DataFrame:
    lodoo = paths.reg_dir / "metrics" / "lodoo_regression_results.csv"
    if lodoo.is_file():
        df = pd.read_csv(lodoo)
        if {"held_out_dataset", "mae_um"}.issubset(df.columns):
            out = df.groupby("held_out_dataset", as_index=False)["mae_um"].mean().rename(columns={"held_out_dataset": "dataset", "mae_um": "stageB_mae_um"})
            out["stageB_source"] = "lodoo"
            return out

    roi_path = paths.reg_dir / "inference" / "roi_predictions.csv"
    if roi_path.is_file():
        r = pd.read_csv(roi_path)
        if "y_mag_pred" not in r.columns and "signed_dz_pred" in r.columns:
            r["y_mag_pred"] = pd.to_numeric(r["signed_dz_pred"], errors="coerce").abs()
        label_cols = ["roi_uid", "y_mag_um"]
        if "dataset" not in r.columns:
            label_cols.append("dataset")
        merged = r.merge(idx[label_cols], on="roi_uid", how="left")
        merged["y_mag_pred"] = pd.to_numeric(merged["y_mag_pred"], errors="coerce")
        merged["y_mag_um"] = pd.to_numeric(merged["y_mag_um"], errors="coerce")
        rows = []
        dataset_col = "dataset"
        if dataset_col not in merged.columns:
            raise ValueError("Stage B ROI merge did not produce a dataset column.")
        for ds, g in merged.groupby(dataset_col):
            g = g.dropna(subset=["y_mag_um", "y_mag_pred"])
            mae = float(np.mean(np.abs(g["y_mag_pred"].to_numpy(dtype=float) - g["y_mag_um"].to_numpy(dtype=float)))) if len(g) else np.nan
            rows.append({"dataset": ds, "stageB_mae_um": mae, "stageB_source": "proxy_roi"})
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["dataset", "stageB_mae_um", "stageB_source"])


def _load_end2end_dataset_metrics(paths, idx: pd.DataFrame) -> pd.DataFrame:
    fov_path = paths.reg_dir / "inference" / "fov_aggregate_predictions.csv"
    if not fov_path.is_file():
        return pd.DataFrame(columns=["dataset", "end_to_end_mae_um", "end_to_end_source"])

    pred = pd.read_csv(fov_path)
    pred["dz_hat_um_num"] = pd.to_numeric(pred["dz_hat_um"], errors="coerce")

    gt_rows = []
    for fov_id, g in idx.groupby("fov_id"):
        gt = float(np.nanmedian(pd.to_numeric(g["defocus_um"], errors="coerce").to_numpy(dtype=float)))
        gt_rows.append({"fov_id": str(fov_id), "gt_signed_um": gt})
    gt = pd.DataFrame(gt_rows)
    ds_map = infer_dataset_for_fov(idx)

    merged = pred.merge(gt, on="fov_id", how="left").merge(ds_map, on="fov_id", how="left")
    merged = merged.dropna(subset=["gt_signed_um", "dz_hat_um_num", "dataset"]).copy()

    rows = []
    for ds, g in merged.groupby("dataset"):
        mae = float(np.mean(np.abs(g["dz_hat_um_num"].to_numpy(dtype=float) - g["gt_signed_um"].to_numpy(dtype=float)))) if len(g) else np.nan
        rows.append({"dataset": ds, "end_to_end_mae_um": mae, "end_to_end_source": "proxy_fov"})
    return pd.DataFrame(rows)


def _plot_cross_dataset(df: pd.DataFrame, out_path):
    ds = df["dataset"].astype(str).tolist()
    x = np.arange(len(ds))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(x, df["stageA_auroc"].to_numpy(dtype=float))
    axes[0].set_title("Stage A AUROC")
    axes[0].set_xticks(x, ds, rotation=30)

    axes[1].bar(x, df["stageB_mae_um"].to_numpy(dtype=float))
    axes[1].set_title("Stage B MAE (µm)")
    axes[1].set_xticks(x, ds, rotation=30)

    axes[2].bar(x, df["end_to_end_mae_um"].to_numpy(dtype=float))
    axes[2].set_title("End-to-End MAE (µm)")
    axes[2].set_xticks(x, ds, rotation=30)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)

    save_plot(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-dataset validation report")
    default_eval_cli(ap)
    args = ap.parse_args()

    paths = get_paths(args.track)
    log(paths, "Starting cross-dataset validation")
    required = [paths.eval_dir / "cross_dataset_results.csv"]
    if args.save_plots:
        required.append(paths.eval_dir / "cross_dataset_barplot.png")
    if args.resume and all(p.is_file() for p in required):
        log(paths, f"Resume: cross-dataset outputs already exist; skipping. ({paths.eval_dir})")
        return

    idx = load_phase5_index(args.track)

    sA = _load_stageA_dataset_metrics(paths, idx)
    sB = _load_stageB_dataset_metrics(paths, idx)
    e2e = _load_end2end_dataset_metrics(paths, idx)

    if sA.empty and sB.empty and e2e.empty:
        raise FileNotFoundError(
            "No cross-dataset sources available. Expected at least one of:\n"
            f"- {paths.sign_dir / 'metrics' / 'lodoo_results.csv'}\n"
            f"- {paths.sign_dir / 'inference' / 'vote_sign_per_roi.csv'}\n"
            f"- {paths.sign_dir / 'calibration' / 'val_roi_predictions.csv'}\n"
            f"- {paths.reg_dir / 'metrics' / 'lodoo_regression_results.csv'}\n"
            f"- {paths.reg_dir / 'inference' / 'roi_predictions.csv'}\n"
            f"- {paths.reg_dir / 'inference' / 'fov_aggregate_predictions.csv'}"
        )

    datasets = sorted(idx["dataset"].astype(str).unique().tolist())
    out = pd.DataFrame({"dataset": datasets})
    out = out.merge(sA, on="dataset", how="left")
    out = out.merge(sB, on="dataset", how="left")
    out = out.merge(e2e, on="dataset", how="left")

    if out.empty:
        raise ValueError("No dataset results computed for cross-dataset validation")

    out_csv = paths.eval_dir / "cross_dataset_results.csv"
    out.to_csv(out_csv, index=False)

    if args.save_plots:
        _plot_cross_dataset(out, paths.eval_dir / "cross_dataset_barplot.png")

    log(paths, f"Cross-dataset validation complete: {out_csv}")


if __name__ == "__main__":
    main()
