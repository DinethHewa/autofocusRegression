#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_utils import (
    bootstrap_ci,
    default_eval_cli,
    friedman_nemenyi,
    get_paths,
    load_phase5_index,
    log,
    mcnemar_test,
    save_json,
    wilcoxon_signed_rank,
    weighted_median,
)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    return float(0.5 * (tpr + tnr))


def _bootstrap_balanced_acc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    seed: int = 42,
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    point = _balanced_accuracy(y_true, y_pred)
    rng = np.random.default_rng(seed)
    vals = np.empty((int(n_bootstrap),), dtype=float)

    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        vals[i] = _balanced_accuracy(y_true[idx], y_pred[idx])

    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return point, lo, hi


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
            finite = vals[np.isfinite(vals)]
            gt = float(np.median(finite)) if finite.size else np.nan
        rows.append({"fov_id": str(fov_id), "gt_signed_um": float(gt) if np.isfinite(gt) else np.nan})
    return pd.DataFrame(rows)


def _load_stageA_for_stats(paths, idx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vote_path = paths.sign_dir / "inference" / "vote_sign_results.csv"
    per_roi_path = paths.sign_dir / "inference" / "vote_sign_per_roi.csv"

    if not vote_path.is_file():
        raise FileNotFoundError(f"Missing StageA voted sign file: {vote_path}")

    vote = pd.read_csv(vote_path)
    if "fov_id" not in vote.columns:
        raise ValueError(f"StageA vote file missing fov_id: {vote_path}")

    if "pred_sign_int" in vote.columns:
        vote["pred_sign"] = pd.to_numeric(vote["pred_sign_int"], errors="coerce")
    elif "pred_sign" in vote.columns:
        vote["pred_sign"] = pd.to_numeric(vote["pred_sign"], errors="coerce")
    else:
        raise ValueError(f"StageA vote file missing pred_sign/pred_sign_int: {vote_path}")

    if not per_roi_path.is_file():
        raise FileNotFoundError(
            f"Missing StageA per-ROI file: {per_roi_path}. "
            "Run infer_vote_sign.py with --save-per-roi before statistical tests."
        )

    roi = pd.read_csv(per_roi_path)
    required = ["fov_id", "roi_uid", "p"]
    missing = [c for c in required if c not in roi.columns]
    if missing:
        raise ValueError(f"StageA per-ROI file missing columns {missing}: {per_roi_path}")

    roi = roi.merge(idx[["roi_uid", "y_sign"]], on="roi_uid", how="left")
    roi = roi.dropna(subset=["y_sign", "p"]).copy().reset_index(drop=True)
    if roi.empty:
        raise ValueError("No StageA per-ROI rows available after joining with index_phase5.")

    if "roi_importance" in roi.columns:
        roi["_rank_key"] = pd.to_numeric(roi["roi_importance"], errors="coerce").fillna(0.0)
    elif "c" in roi.columns:
        roi["_rank_key"] = pd.to_numeric(roi["c"], errors="coerce").fillna(0.0)
    else:
        roi["_rank_key"] = 0.0

    return vote, roi


def _single_roi_sign_baseline(roi_df: pd.DataFrame) -> pd.DataFrame:
    top = (
        roi_df.sort_values(["fov_id", "_rank_key"], ascending=[True, False])
        .drop_duplicates(subset=["fov_id"], keep="first")
        .copy()
    )
    top["pred_single"] = (pd.to_numeric(top["p"], errors="coerce") >= 0.5).astype(int)
    return top[["fov_id", "pred_single"]]


def _load_stageB_for_stats(paths, idx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_path = paths.reg_dir / "inference" / "roi_predictions.csv"
    fov_path = paths.reg_dir / "inference" / "fov_aggregate_predictions.csv"

    if not roi_path.is_file():
        raise FileNotFoundError(f"Missing StageB ROI prediction file: {roi_path}")
    if not fov_path.is_file():
        raise FileNotFoundError(f"Missing StageB FOV aggregate file: {fov_path}")

    roi = pd.read_csv(roi_path)
    if "roi_uid" not in roi.columns:
        raise ValueError(f"StageB ROI file missing roi_uid: {roi_path}")

    if "y_mag_pred" not in roi.columns:
        if "signed_dz_pred" in roi.columns:
            roi["y_mag_pred"] = pd.to_numeric(roi["signed_dz_pred"], errors="coerce").abs()
        else:
            raise ValueError(f"StageB ROI file missing y_mag_pred and signed_dz_pred: {roi_path}")

    if "signed_dz_pred" not in roi.columns and "y_mag_pred" in roi.columns and "y_sign_pred" in roi.columns:
        sgn = pd.to_numeric(roi["y_sign_pred"], errors="coerce")
        mag = pd.to_numeric(roi["y_mag_pred"], errors="coerce")
        roi["signed_dz_pred"] = np.where(sgn >= 0.5, mag, -mag)

    join_cols = ["roi_uid", "y_mag_um", "defocus_um"]
    if "fov_id" not in roi.columns:
        join_cols.append("fov_id")
    merged_roi = roi.merge(idx[join_cols], on="roi_uid", how="left")
    merged_roi["y_mag_pred"] = pd.to_numeric(merged_roi["y_mag_pred"], errors="coerce")
    merged_roi["y_mag_um"] = pd.to_numeric(merged_roi["y_mag_um"], errors="coerce")
    merged_roi["signed_dz_pred"] = pd.to_numeric(merged_roi["signed_dz_pred"], errors="coerce")
    if "fov_id" not in merged_roi.columns:
        raise ValueError("StageB ROI merge did not produce fov_id.")

    fov = pd.read_csv(fov_path)
    if "fov_id" not in fov.columns or "dz_hat_um" not in fov.columns:
        raise ValueError(f"StageB FOV file must contain fov_id and dz_hat_um: {fov_path}")

    fov["pred_full"] = pd.to_numeric(fov["dz_hat_um"], errors="coerce")
    return merged_roi, fov


def _aggregation_variants(roi_df: pd.DataFrame) -> pd.DataFrame:
    work = roi_df.copy()
    if "weight" in work.columns:
        work["w"] = pd.to_numeric(work["weight"], errors="coerce").fillna(0.0)
    elif "w" in work.columns:
        work["w"] = pd.to_numeric(work["w"], errors="coerce").fillna(0.0)
    elif "roi_importance" in work.columns:
        work["w"] = pd.to_numeric(work["roi_importance"], errors="coerce").fillna(1.0)
    else:
        work["w"] = 1.0

    rows = []
    for fov_id, g in work.groupby("fov_id", sort=True):
        g = g[g["signed_dz_pred"].notna()].copy()
        if g.empty:
            rows.append(
                {
                    "fov_id": str(fov_id),
                    "pred_single": np.nan,
                    "pred_mean": np.nan,
                    "pred_wmean": np.nan,
                    "pred_wmedian": np.nan,
                }
            )
            continue

        g = g.sort_values("w", ascending=False)
        single = float(g.iloc[0]["signed_dz_pred"])
        mean = float(g["signed_dz_pred"].mean())

        vals = g["signed_dz_pred"].to_numpy(dtype=float)
        w = g["w"].to_numpy(dtype=float)
        if np.sum(w) > 0:
            wmean = float(np.sum(vals * w) / np.sum(w))
            wmedian = weighted_median(vals, w)
            wmedian = float(wmedian) if wmedian is not None else np.nan
        else:
            wmean = np.nan
            wmedian = np.nan

        rows.append(
            {
                "fov_id": str(fov_id),
                "pred_single": single,
                "pred_mean": mean,
                "pred_wmean": wmean,
                "pred_wmedian": wmedian,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run statistical tests for final phase evaluation outputs")
    default_eval_cli(ap)
    args = ap.parse_args()

    paths = get_paths(args.track)
    log(paths, "Starting statistical tests")
    required = [paths.eval_dir / "statistical_tests_results.json", paths.eval_dir / "confidence_intervals.csv"]
    if args.resume and all(p.is_file() for p in required):
        log(paths, f"Resume: statistical outputs already exist; skipping. ({paths.eval_dir})")
        return

    idx = load_phase5_index(args.track)
    gt_fov = _gt_per_fov(idx)

    vote_df, stageA_roi = _load_stageA_for_stats(paths, idx)
    reg_roi, reg_fov = _load_stageB_for_stats(paths, idx)

    ci_rows: list[dict] = []

    # Bootstrap CI: Stage A balanced accuracy (per ROI)
    y_true_roi = stageA_roi["y_sign"].to_numpy(dtype=int)
    y_pred_roi = (pd.to_numeric(stageA_roi["p"], errors="coerce").to_numpy(dtype=float) >= 0.5).astype(int)
    point, lo, hi = _bootstrap_balanced_acc(y_true_roi, y_pred_roi, n_bootstrap=int(args.bootstrap), seed=42)
    ci_rows.append(
        {
            "metric": "stageA_balanced_accuracy",
            "point_estimate": point,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_samples": int(len(y_true_roi)),
            "bootstrap_samples": int(args.bootstrap),
        }
    )

    # Bootstrap CI: Stage B MAE (per ROI magnitude)
    reg_roi_valid = reg_roi.dropna(subset=["y_mag_um", "y_mag_pred"]).copy()
    abs_mag_err = np.abs(reg_roi_valid["y_mag_pred"].to_numpy(dtype=float) - reg_roi_valid["y_mag_um"].to_numpy(dtype=float))
    p2, lo2, hi2 = bootstrap_ci(abs_mag_err, lambda x: float(np.mean(x)), n_bootstrap=int(args.bootstrap), seed=42)
    ci_rows.append(
        {
            "metric": "stageB_mae_um",
            "point_estimate": p2,
            "ci95_lo": lo2,
            "ci95_hi": hi2,
            "n_samples": int(len(abs_mag_err)),
            "bootstrap_samples": int(args.bootstrap),
        }
    )

    # Bootstrap CI: End-to-end MAE (per FOV)
    fov_eval = reg_fov.merge(gt_fov, on="fov_id", how="left")
    fov_eval = fov_eval.dropna(subset=["pred_full", "gt_signed_um"]).copy()
    abs_fov_err = np.abs(fov_eval["pred_full"].to_numpy(dtype=float) - fov_eval["gt_signed_um"].to_numpy(dtype=float))
    p3, lo3, hi3 = bootstrap_ci(abs_fov_err, lambda x: float(np.mean(x)), n_bootstrap=int(args.bootstrap), seed=42)
    ci_rows.append(
        {
            "metric": "end_to_end_mae_um",
            "point_estimate": p3,
            "ci95_lo": lo3,
            "ci95_hi": hi3,
            "n_samples": int(len(abs_fov_err)),
            "bootstrap_samples": int(args.bootstrap),
        }
    )

    # McNemar: full voted sign vs single-ROI sign baseline
    fov_sign_gt = gt_fov.copy()
    fov_sign_gt = fov_sign_gt[~np.isclose(pd.to_numeric(fov_sign_gt["gt_signed_um"], errors="coerce"), 0.0)].copy()
    fov_sign_gt["y_true_sign"] = (pd.to_numeric(fov_sign_gt["gt_signed_um"], errors="coerce") > 0).astype(int)

    vote = vote_df[["fov_id", "pred_sign"]].copy()
    vote = vote.dropna(subset=["pred_sign"]).copy()
    vote["pred_sign"] = vote["pred_sign"].astype(int)

    single = _single_roi_sign_baseline(stageA_roi)
    mcn_df = fov_sign_gt[["fov_id", "y_true_sign"]].merge(vote, on="fov_id", how="inner").merge(single, on="fov_id", how="inner")
    mcn = mcnemar_test(
        y_true=mcn_df["y_true_sign"].to_numpy(dtype=int),
        y_pred_a=mcn_df["pred_sign"].to_numpy(dtype=int),
        y_pred_b=mcn_df["pred_single"].to_numpy(dtype=int),
    )
    mcn["n_samples"] = int(len(mcn_df))
    mcn["comparison"] = "voting_gating_full_vs_single_roi_baseline"

    # Wilcoxon + Friedman/Nemenyi on end-to-end variants
    variants = _aggregation_variants(reg_roi)
    all_fov = gt_fov.merge(variants, on="fov_id", how="left").merge(reg_fov[["fov_id", "pred_full"]], on="fov_id", how="left")

    # Wilcoxon: full vs weighted mean absolute errors
    w_df = all_fov.dropna(subset=["gt_signed_um", "pred_full", "pred_wmean"]).copy()
    err_full = np.abs(w_df["pred_full"].to_numpy(dtype=float) - w_df["gt_signed_um"].to_numpy(dtype=float))
    err_wmean = np.abs(w_df["pred_wmean"].to_numpy(dtype=float) - w_df["gt_signed_um"].to_numpy(dtype=float))
    wil = wilcoxon_signed_rank(err_full, err_wmean)
    wil["n_samples"] = int(len(w_df))
    wil["comparison"] = "full_weighted_median_vs_weighted_mean"

    # Friedman + Nemenyi: single/mean/wmean/wmedian/full
    model_cols = [
        ("single_roi", "pred_single"),
        ("mean", "pred_mean"),
        ("weighted_mean", "pred_wmean"),
        ("weighted_median", "pred_wmedian"),
        ("full_voting_gating", "pred_full"),
    ]
    tmp = all_fov[["fov_id", "gt_signed_um"] + [c for _, c in model_cols]].copy()
    for _, c in model_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    score_dict: dict[str, np.ndarray] = {}
    for mname, col in model_cols:
        mask = tmp["gt_signed_um"].notna() & tmp[col].notna()
        if int(mask.sum()) >= 5:
            e = np.abs(tmp.loc[mask, col].to_numpy(dtype=float) - tmp.loc[mask, "gt_signed_um"].to_numpy(dtype=float))
            score_dict[mname] = e

    if len(score_dict) >= 3:
        min_len = min(len(v) for v in score_dict.values())
        score_dict = {k: v[:min_len] for k, v in score_dict.items()}
        fried = friedman_nemenyi(score_dict)
        fried["n_samples"] = int(min_len)
        fried["models"] = list(score_dict.keys())
    else:
        fried = {
            "method": "insufficient_data",
            "friedman_stat": np.nan,
            "friedman_p": np.nan,
            "avg_ranks": {},
            "critical_difference": np.nan,
            "pairwise_rank_diff": {},
            "n_samples": 0,
            "models": list(score_dict.keys()),
        }

    out_json = {
        "track": args.track,
        "wilcoxon": wil,
        "mcnemar": mcn,
        "friedman_nemenyi": fried,
    }

    save_json(out_json, paths.eval_dir / "statistical_tests_results.json")
    pd.DataFrame(ci_rows).to_csv(paths.eval_dir / "confidence_intervals.csv", index=False)

    log(paths, f"Statistical tests complete. Outputs: {paths.eval_dir / 'statistical_tests_results.json'}")


if __name__ == "__main__":
    main()
