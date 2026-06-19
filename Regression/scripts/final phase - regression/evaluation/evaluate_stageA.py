#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation_utils import (
    EvalPaths,
    classification_metrics,
    default_eval_cli,
    get_paths,
    infer_dataset_for_fov,
    load_phase5_index,
    log,
    per_bin_classification,
    require_file,
    roc_curve_points,
    save_json,
    save_plot,
    save_table,
)


def _load_stageA_predictions(paths: EvalPaths) -> pd.DataFrame:
    idx = load_phase5_index(paths.track)

    per_roi = paths.sign_dir / "inference" / "vote_sign_per_roi.csv"
    val_preds = paths.sign_dir / "calibration" / "val_roi_predictions.csv"

    if per_roi.is_file():
        df = pd.read_csv(per_roi)
        required = ["roi_uid", "p", "c"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"StageA per-ROI file missing columns {miss}: {per_roi}")

        merged = df.merge(idx[["roi_uid", "defocus_um", "y_sign", "y_mag_um"]], on="roi_uid", how="left")
        if merged["y_sign"].isna().any():
            raise ValueError(
                f"Could not map y_sign for all roi_uid from {per_roi}. "
                f"Ensure index_phase5.csv is built and consistent."
            )
        merged = merged.rename(columns={"p": "prob", "c": "conf"})
        return merged

    if val_preds.is_file():
        df = pd.read_csv(val_preds)
        required = ["y_sign", "delta_z", "p", "c"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"StageA val predictions missing columns {miss}: {val_preds}")
        df = df.copy()
        df["defocus_um"] = pd.to_numeric(df["delta_z"], errors="coerce")
        df["y_mag_um"] = np.abs(df["defocus_um"].astype(float))
        df = df.rename(columns={"p": "prob", "c": "conf"})
        return df

    raise FileNotFoundError(
        "Missing StageA ROI prediction source. Expected either:\n"
        f"- {per_roi}\n"
        f"- {val_preds}\n"
        "Run Phase4 calibrate_tau.py and/or infer_vote_sign.py --save-per-roi first."
    )


def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fpr, tpr = roc_curve_points(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Stage A ROC")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, out_path)


def _plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("Stage A Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_plot(fig, out_path)


def _plot_bin_acc(bin_df: pd.DataFrame, out_path: Path) -> None:
    x = 0.5 * (bin_df["bin_lo"].to_numpy() + bin_df["bin_hi"].to_numpy())
    y = bin_df["accuracy"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("|Δz| bin center (µm)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Stage A Accuracy vs |Δz| bin")
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def _plot_confidence_curve(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    bins = np.linspace(0.5, 1.0, 11)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (conf >= lo) & (conf < hi)
        n = int(idx.sum())
        acc = float((y_true[idx] == y_pred[idx]).mean()) if n > 0 else np.nan
        rows.append((0.5 * (lo + hi), acc))

    pts = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pts[:, 0], pts[:, 1], marker="o")
    ax.set_xlabel("Confidence c")
    ax.set_ylabel("Accuracy")
    ax.set_title("Confidence vs Accuracy")
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Stage A sign classification")
    default_eval_cli(ap)
    args = ap.parse_args()

    paths = get_paths(args.track)
    log(paths, "Starting Stage A evaluation")
    required = [
        paths.eval_dir / "stageA_metrics.json",
        paths.eval_dir / "stageA_confusion_matrix.csv",
        paths.eval_dir / "stageA_bin_accuracy.csv",
        paths.tables_dir / "Table_StageA.csv",
    ]
    if args.save_latex:
        required.append((paths.tables_dir / "Table_StageA.csv").with_suffix(".tex"))
    if args.save_plots:
        required.extend(
            [
                paths.eval_dir / "ROC.png",
                paths.eval_dir / "confusion_matrix_heatmap.png",
                paths.eval_dir / "bin_accuracy_curve.png",
                paths.eval_dir / "confidence_vs_accuracy_curve.png",
            ]
        )
    if args.resume and all(p.is_file() for p in required):
        log(paths, f"Resume: Stage A outputs already exist; skipping. ({paths.eval_dir})")
        return

    df = _load_stageA_predictions(paths)

    df["y_sign"] = pd.to_numeric(df["y_sign"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce")
    df["y_mag_um"] = pd.to_numeric(df["y_mag_um"], errors="coerce")

    df = df.dropna(subset=["y_sign", "prob", "conf", "y_mag_um"]).copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid Stage A rows after cleaning.")

    y_true = df["y_sign"].to_numpy(dtype=int)
    y_prob = df["prob"].to_numpy(dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    conf = df["conf"].to_numpy(dtype=float)
    abs_delta = df["y_mag_um"].to_numpy(dtype=float)

    metrics = classification_metrics(y_true, y_pred, y_prob)

    near = abs_delta <= 1.0
    far = abs_delta > 5.0
    metrics["near_focus_accuracy_abs_dz_le_1um"] = float((y_true[near] == y_pred[near]).mean()) if near.any() else float("nan")
    metrics["far_focus_accuracy_abs_dz_gt_5um"] = float((y_true[far] == y_pred[far]).mean()) if far.any() else float("nan")
    metrics["n_samples"] = int(len(df))

    cm = np.array(
        [
            [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
            [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
        ],
        dtype=int,
    )

    bin_df = per_bin_classification(abs_delta=abs_delta, y_true=y_true, y_pred=y_pred, bin_width=float(args.bins))

    out_json = paths.eval_dir / "stageA_metrics.json"
    out_cm_csv = paths.eval_dir / "stageA_confusion_matrix.csv"
    out_bin_csv = paths.eval_dir / "stageA_bin_accuracy.csv"

    save_json(metrics, out_json)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(out_cm_csv)
    bin_df.to_csv(out_bin_csv, index=False)

    if args.save_plots:
        _plot_roc(y_true, y_prob, paths.eval_dir / "ROC.png")
        _plot_confusion(cm, paths.eval_dir / "confusion_matrix_heatmap.png")
        _plot_bin_acc(bin_df, paths.eval_dir / "bin_accuracy_curve.png")
        _plot_confidence_curve(conf, y_true, y_pred, paths.eval_dir / "confidence_vs_accuracy_curve.png")

    table_df = pd.DataFrame(
        [
            {
                "track": args.track,
                "AUROC": metrics.get("auroc", np.nan),
                "Balanced_Accuracy": metrics.get("balanced_accuracy", np.nan),
                "Precision": metrics.get("precision", np.nan),
                "Recall": metrics.get("recall", np.nan),
                "F1": metrics.get("f1", np.nan),
                "NearFocusAcc_|dz|<=1": metrics.get("near_focus_accuracy_abs_dz_le_1um", np.nan),
                "FarFocusAcc_|dz|>5": metrics.get("far_focus_accuracy_abs_dz_gt_5um", np.nan),
                "N": metrics.get("n_samples", 0),
            }
        ]
    )
    save_table(table_df, paths.tables_dir / "Table_StageA.csv", save_latex=bool(args.save_latex))

    log(paths, f"Stage A evaluation complete. Outputs: {paths.eval_dir}")


if __name__ == "__main__":
    main()
