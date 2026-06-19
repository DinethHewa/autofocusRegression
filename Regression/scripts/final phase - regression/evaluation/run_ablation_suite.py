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
            gt = float(np.nanmedian(vals))
        rows.append({"fov_id": str(fov_id), "gt_signed_um": gt})
    return pd.DataFrame(rows)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) else np.nan


def _compute_end_variants(roi_df: pd.DataFrame, fov_df_full: pd.DataFrame, gt_df: pd.DataFrame) -> list[dict]:
    out = []

    # Ensure numeric
    roi = roi_df.copy()
    roi["signed_dz_pred"] = pd.to_numeric(roi["signed_dz_pred"], errors="coerce")
    if "weight" in roi.columns:
        roi["weight"] = pd.to_numeric(roi["weight"], errors="coerce").fillna(0.0)
    elif "w" in roi.columns:
        roi["weight"] = pd.to_numeric(roi["w"], errors="coerce").fillna(0.0)
    else:
        roi["weight"] = pd.to_numeric(roi.get("roi_importance", 1.0), errors="coerce").fillna(1.0)

    pred_rows = []
    for fov_id, g in roi.groupby("fov_id", sort=True):
        g = g[g["signed_dz_pred"].notna()].copy()
        if g.empty:
            pred_rows.append({"fov_id": fov_id, "single": np.nan, "mean": np.nan, "wmean": np.nan, "wmedian": np.nan})
            continue

        single = float(g.sort_values("weight", ascending=False).iloc[0]["signed_dz_pred"])
        mean = float(g["signed_dz_pred"].mean())

        ww = g["weight"].to_numpy(dtype=float)
        vv = g["signed_dz_pred"].to_numpy(dtype=float)
        if np.sum(ww) > 0:
            wmean = float(np.sum(ww * vv) / np.sum(ww))
            wmed = weighted_median(vv, ww)
            wmed = float(wmed) if wmed is not None else np.nan
        else:
            wmean = np.nan
            wmed = np.nan

        pred_rows.append({"fov_id": fov_id, "single": single, "mean": mean, "wmean": wmean, "wmedian": wmed})

    pred_df = pd.DataFrame(pred_rows)
    merged = gt_df.merge(pred_df, on="fov_id", how="left")

    gt = merged["gt_signed_um"].to_numpy(dtype=float)
    for key, name in [
        ("single", "Single ROI"),
        ("mean", "Mean aggregation"),
        ("wmean", "Weighted mean"),
        ("wmedian", "Weighted median"),
    ]:
        pred = pd.to_numeric(merged[key], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(gt) & np.isfinite(pred)
        out.append(
            {
                "component": "End-to-End",
                "variant": name,
                "metric": "MAE_um",
                "value": _mae(gt[mask], pred[mask]) if mask.any() else np.nan,
                "available": int(mask.any()),
            }
        )

    if "dz_hat_um" in fov_df_full.columns:
        f = fov_df_full.copy()
        f["dz_hat_um_num"] = pd.to_numeric(f["dz_hat_um"], errors="coerce")
        mm = gt_df.merge(f[["fov_id", "dz_hat_um_num"]], on="fov_id", how="left")
        gt2 = mm["gt_signed_um"].to_numpy(dtype=float)
        pred2 = mm["dz_hat_um_num"].to_numpy(dtype=float)
        mask2 = np.isfinite(gt2) & np.isfinite(pred2)
        out.append(
            {
                "component": "End-to-End",
                "variant": "Voting + gating (full)",
                "metric": "MAE_um",
                "value": _mae(gt2[mask2], pred2[mask2]) if mask2.any() else np.nan,
                "available": int(mask2.any()),
            }
        )

    return out


def _plot_ablation(df: pd.DataFrame, out_path):
    work = df[(df["metric"] == "MAE_um") & (df["available"] == 1)].copy()
    if work.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No ablation data available", ha="center", va="center")
        ax.axis("off")
        save_plot(fig, out_path)
        return

    labels = work["variant"].astype(str).tolist()
    vals = work["value"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(labels)), 4))
    ax.bar(labels, vals)
    ax.set_ylabel("MAE (µm)")
    ax.set_title("Ablation Summary")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    save_plot(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ablation suite summarization")
    default_eval_cli(ap)
    args = ap.parse_args()

    paths = get_paths(args.track)
    log(paths, "Starting ablation suite")
    required = [paths.eval_dir / "ablation_results.csv", paths.tables_dir / "Table_Ablation.csv"]
    if args.save_latex:
        required.append((paths.tables_dir / "Table_Ablation.csv").with_suffix(".tex"))
    if args.save_plots:
        required.append(paths.eval_dir / "ablation_plot.png")
    if args.resume and all(p.is_file() for p in required):
        log(paths, f"Resume: ablation outputs already exist; skipping. ({paths.eval_dir})")
        return

    results = []

    # Stage A variants (if explicit ablation file exists)
    stageA_ab_file = paths.sign_dir / "metrics" / "ablation_stageA.csv"
    if stageA_ab_file.is_file():
        a = pd.read_csv(stageA_ab_file)
        for _, r in a.iterrows():
            results.append(
                {
                    "component": "StageA",
                    "variant": r.get("variant", "unknown"),
                    "metric": r.get("metric", "balanced_accuracy"),
                    "value": r.get("value", np.nan),
                    "available": 1,
                }
            )
    else:
        # placeholders + current full
        for v in ["I only", "I + DoG", "I + DWT", "Full input"]:
            results.append({"component": "StageA", "variant": v, "metric": "balanced_accuracy", "value": np.nan, "available": 0})

    # Stage B variants
    stageB_ab_file = paths.reg_dir / "metrics" / "ablation_stageB.csv"
    if stageB_ab_file.is_file():
        b = pd.read_csv(stageB_ab_file)
        for _, r in b.iterrows():
            results.append(
                {
                    "component": "StageB",
                    "variant": r.get("variant", "unknown"),
                    "metric": r.get("metric", "MAE_um"),
                    "value": r.get("value", np.nan),
                    "available": 1,
                }
            )
    else:
        for v in ["Regression only", "Regression + Triplet", "Full cascade"]:
            results.append({"component": "StageB", "variant": v, "metric": "MAE_um", "value": np.nan, "available": 0})

    # End-to-end variants computed from current predictions
    roi_path = paths.reg_dir / "inference" / "roi_predictions.csv"
    fov_path = paths.reg_dir / "inference" / "fov_aggregate_predictions.csv"
    idx = load_phase5_index(args.track)
    gt = _gt_per_fov(idx)

    if not roi_path.is_file() or not fov_path.is_file():
        raise FileNotFoundError(
            "Missing required Phase5 inference outputs for end-to-end ablation:\n"
            f"- {roi_path}\n"
            f"- {fov_path}\n"
            "Run infer_regress_and_aggregate.py first."
        )

    roi = pd.read_csv(roi_path)
    fov = pd.read_csv(fov_path)
    results.extend(_compute_end_variants(roi, fov, gt))

    res_df = pd.DataFrame(results)
    out_csv = paths.eval_dir / "ablation_results.csv"
    res_df.to_csv(out_csv, index=False)

    if args.save_plots:
        _plot_ablation(res_df, paths.eval_dir / "ablation_plot.png")

    save_table(res_df, paths.tables_dir / "Table_Ablation.csv", save_latex=bool(args.save_latex))
    log(paths, f"Ablation suite complete: {out_csv}")


if __name__ == "__main__":
    main()
