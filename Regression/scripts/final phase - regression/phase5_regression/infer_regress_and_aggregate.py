#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import (
    default_phase5_index_path,
    ensure_reg_output_tree,
    load_XA_tensor,
    load_XB_tensor,
    load_phase5_index,
    parse_voted_sign_value,
    resolve_reg_out_dir,
    weighted_median,
)


def _load_trusted_keras_model(path: str | Path) -> tf.keras.Model:
    # Pipeline checkpoints are generated locally and may include Lambda layers.
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase-5 inference: route by voted sign and aggregate weighted median dz_hat_um.")
    ap.add_argument("--track", required=True, choices=["smears", "biopsy"])
    ap.add_argument("--phase5-index", default=None)
    ap.add_argument("--sign-out-dir", default=None, help="Default: out_final_phase/<track>/sign")
    ap.add_argument("--reg-out-dir", default=None, help="Default: out_final_phase/<track>/regression")
    ap.add_argument("--top-k", type=int, default=7)
    ap.add_argument("--margin-threshold", type=float, default=0.30)
    ap.add_argument("--use-per-roi-sign", action="store_true")
    ap.add_argument("--save-per-roi", action="store_true")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--resume", action="store_true", help="Reuse existing inference outputs when present.")
    return ap.parse_args()


def _default_sign_out(track: str) -> Path:
    return Path(f"/home/dineth/focus_measure/journal/Regression/data/out_final_phase/{track}/sign")


def _predict_branch(model: tf.keras.Model, rows: pd.DataFrame, batch_size: int) -> np.ndarray:
    if rows.empty:
        return np.zeros((0,), dtype=np.float32)

    preds = []
    paths = rows["cache_path_XB"].astype(str).tolist()
    for i in range(0, len(paths), batch_size):
        sub = paths[i : i + batch_size]
        x = np.empty((len(sub), 200, 200, 4), dtype=np.float32)
        for j, p in enumerate(sub):
            x[j] = load_XB_tensor(p)
        y_hat, _ = model(x, training=False)
        preds.append(tf.squeeze(y_hat, axis=-1).numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def _compute_roi_conf_from_sign_model(sign_model: tf.keras.Model, rows: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    out = rows.copy().reset_index(drop=True)
    c = np.full((len(out),), np.nan, dtype=np.float32)
    p = np.full((len(out),), np.nan, dtype=np.float32)

    valid_idx = []
    valid_paths = []
    for i, path in enumerate(out.get("cache_path_XA", pd.Series([None] * len(out))).tolist()):
        if isinstance(path, str) and Path(path).is_file():
            valid_idx.append(i)
            valid_paths.append(path)

    for i in range(0, len(valid_paths), batch_size):
        sub_paths = valid_paths[i : i + batch_size]
        sub_idx = valid_idx[i : i + batch_size]
        x = np.empty((len(sub_paths), 200, 200, 3), dtype=np.float32)
        for j, path in enumerate(sub_paths):
            x[j] = load_XA_tensor(path)
        p_hat = sign_model.predict(x, verbose=0).reshape(-1)
        c_hat = np.maximum(p_hat, 1.0 - p_hat)
        for j, ridx in enumerate(sub_idx):
            p[ridx] = p_hat[j]
            c[ridx] = c_hat[j]

    out["p_sign"] = p
    out["c_sign"] = c
    return out


def _resolve_roi_conf(rows: pd.DataFrame, sign_out_dir: Path, use_per_roi_sign: bool, batch_size: int) -> pd.DataFrame:
    out = rows.copy().reset_index(drop=True)

    if use_per_roi_sign:
        per_roi_file = sign_out_dir / "inference" / "vote_sign_per_roi.csv"
        if per_roi_file.is_file():
            roi_df = pd.read_csv(per_roi_file)
            if "roi_uid" in roi_df.columns:
                c_col = "c" if "c" in roi_df.columns else ("confidence" if "confidence" in roi_df.columns else None)
                p_col = "p" if "p" in roi_df.columns else ("sign_prob" if "sign_prob" in roi_df.columns else None)
                if c_col is not None:
                    cols = ["roi_uid", c_col] + ([p_col] if p_col else [])
                    tmp = roi_df[cols].copy().rename(columns={c_col: "c_sign"})
                    if p_col:
                        tmp = tmp.rename(columns={p_col: "p_sign"})
                    out = out.merge(tmp, on="roi_uid", how="left")
                    if "p_sign" not in out.columns:
                        out["p_sign"] = np.nan
                    miss = int(out["c_sign"].isna().sum())
                    if miss > 0:
                        print(f"[WARN] Per-ROI sign file missing c for {miss} rows; attempting on-the-fly/fallback.")
                else:
                    print("[WARN] Per-ROI sign file found but no confidence column (c/confidence).")
            else:
                print("[WARN] Per-ROI sign file found but missing roi_uid column.")
        else:
            print(f"[WARN] --use-per-roi-sign set but file not found: {per_roi_file}")

    if "c_sign" not in out.columns:
        out["c_sign"] = np.nan
    if "p_sign" not in out.columns:
        out["p_sign"] = np.nan

    need_compute = out["c_sign"].isna()
    if need_compute.any() and "cache_path_XA" in out.columns:
        sign_model_path = sign_out_dir / "models" / "best_model.keras"
        if sign_model_path.is_file():
            sign_model = _load_trusted_keras_model(sign_model_path)
            computed = _compute_roi_conf_from_sign_model(sign_model, out.loc[need_compute].copy(), batch_size=batch_size)
            out.loc[need_compute, "c_sign"] = computed["c_sign"].to_numpy()
            out.loc[need_compute, "p_sign"] = computed["p_sign"].to_numpy()
        else:
            print(f"[WARN] Sign model not found for on-the-fly confidence: {sign_model_path}")

    still_missing = out["c_sign"].isna()
    if still_missing.any():
        print(f"[WARN] Falling back to c_i=1.0 for {int(still_missing.sum())} ROI rows.")
        out.loc[still_missing, "c_sign"] = 1.0
        out.loc[out["p_sign"].isna(), "p_sign"] = 0.5

    return out


def main() -> None:
    args = parse_args()

    reg_out_dir = resolve_reg_out_dir(args.track, args.reg_out_dir)
    reg_paths = ensure_reg_output_tree(reg_out_dir)
    roi_path = reg_paths["inference"] / "roi_predictions.csv"
    fov_path = reg_paths["inference"] / "fov_aggregate_predictions.csv"
    verbose_path = reg_paths["inference"] / "roi_predictions_verbose.csv"
    if args.resume:
        if args.save_per_roi and roi_path.is_file() and fov_path.is_file() and verbose_path.is_file():
            print(f"[INFO] Resume: regression inference outputs already exist; skipping.")
            return
        if (not args.save_per_roi) and roi_path.is_file() and fov_path.is_file():
            print(f"[INFO] Resume: regression inference outputs already exist; skipping.")
            return
    sign_out_dir = _default_sign_out(args.track) if args.sign_out_dir is None else Path(args.sign_out_dir)

    phase5_index = str(default_phase5_index_path(args.track) if args.phase5_index is None else args.phase5_index)
    df = load_phase5_index(args.track, phase5_index)

    # Select top-k by roi_importance per fov.
    work = df.sort_values(["fov_id", "roi_importance"], ascending=[True, False]).copy()
    if int(args.top_k) > 0:
        work["_rank"] = work.groupby("fov_id").cumcount() + 1
        work = work[work["_rank"] <= int(args.top_k)].copy().drop(columns=["_rank"]).reset_index(drop=True)

    vote_csv = sign_out_dir / "inference" / "vote_sign_results.csv"
    tau_json = sign_out_dir / "calibration" / "chosen_tau.json"
    if not vote_csv.is_file():
        raise FileNotFoundError(f"Phase4 voted sign file missing: {vote_csv}")
    if not tau_json.is_file():
        raise FileNotFoundError(f"Phase4 tau file missing: {tau_json}")

    with tau_json.open("r", encoding="utf-8") as f:
        tau = float(json.load(f)["tau"])

    vote_df = pd.read_csv(vote_csv)
    if "fov_id" not in vote_df.columns:
        raise ValueError(f"vote_sign_results missing fov_id: {vote_csv}")

    if "pred_sign_int" in vote_df.columns:
        vote_df["voted_sign"] = vote_df["pred_sign_int"].apply(parse_voted_sign_value)
    elif "pred_sign" in vote_df.columns:
        vote_df["voted_sign"] = vote_df["pred_sign"].apply(parse_voted_sign_value)
    else:
        raise ValueError(f"vote_sign_results missing pred_sign/pred_sign_int: {vote_csv}")

    if "vote_margin" not in vote_df.columns:
        vote_df["vote_margin"] = np.nan

    work = work.merge(vote_df[["fov_id", "voted_sign", "vote_margin"]], on="fov_id", how="left")

    miss_vote = int(work["voted_sign"].isna().sum())
    if miss_vote > 0:
        print(f"[WARN] Missing voted_sign for {miss_vote} ROI rows; these FOVs will be marked uncertain.")

    model_plus_path = reg_paths["models"] / "R_plus_best.keras"
    model_minus_path = reg_paths["models"] / "R_minus_best.keras"
    if not model_plus_path.is_file() or not model_minus_path.is_file():
        raise FileNotFoundError(
            f"Missing branch models. Expected: {model_plus_path}, {model_minus_path}. "
            f"Run train_regressors.py first."
        )

    model_plus = _load_trusted_keras_model(model_plus_path)
    model_minus = _load_trusted_keras_model(model_minus_path)

    work = _resolve_roi_conf(work, sign_out_dir=sign_out_dir, use_per_roi_sign=bool(args.use_per_roi_sign), batch_size=int(args.batch_size))

    # tau gating + weight
    work["w"] = work["roi_importance"].astype(float) * work["c_sign"].astype(float)
    work.loc[work["c_sign"].astype(float) < float(tau), "w"] = 0.0

    # route + predict
    work["y_mag_pred"] = np.nan
    work["signed_dz_pred"] = np.nan

    mask_plus = work["voted_sign"] == 1
    mask_minus = work["voted_sign"] == 0

    if mask_plus.any():
        y_plus = _predict_branch(model_plus, work.loc[mask_plus].copy(), batch_size=int(args.batch_size))
        work.loc[mask_plus, "y_mag_pred"] = y_plus
        work.loc[mask_plus, "signed_dz_pred"] = y_plus

    if mask_minus.any():
        y_minus = _predict_branch(model_minus, work.loc[mask_minus].copy(), batch_size=int(args.batch_size))
        work.loc[mask_minus, "y_mag_pred"] = y_minus
        work.loc[mask_minus, "signed_dz_pred"] = -y_minus

    # per-ROI output
    roi_cols = [
        "roi_uid",
        "fov_id",
        "dataset",
        "voted_sign",
        "vote_margin",
        "p_sign",
        "c_sign",
        "roi_importance",
        "w",
        "y_mag_pred",
        "signed_dz_pred",
    ]
    roi_cols = [c for c in roi_cols if c in work.columns]
    roi_out = work[roi_cols].copy()
    roi_out["y_sign_pred"] = roi_out["voted_sign"]
    roi_out["weight"] = roi_out["w"]
    roi_out.to_csv(roi_path, index=False)
    print(f"[DONE] Wrote ROI predictions: {roi_path} rows={len(roi_out)}")

    # FOV aggregation
    fov_rows = []
    for fov_id, g in work.groupby("fov_id", sort=True):
        t0 = time.perf_counter()
        vote_sign = parse_voted_sign_value(g["voted_sign"].iloc[0])
        vote_margin = g["vote_margin"].iloc[0] if "vote_margin" in g.columns else np.nan

        valid = (g["w"].astype(float) > 0) & g["signed_dz_pred"].notna()
        num_used = int(valid.sum())

        if vote_sign is None or num_used == 0:
            dz_hat = "uncertain"
            status = "uncertain"
        else:
            wm = weighted_median(
                values=g.loc[valid, "signed_dz_pred"].to_numpy(dtype=float),
                weights=g.loc[valid, "w"].to_numpy(dtype=float),
                eps=float(args.eps),
            )
            if wm is None:
                dz_hat = "uncertain"
                status = "uncertain"
            else:
                dz_hat = float(wm)
                status = "ok"

        runtime_ms = (time.perf_counter() - t0) * 1000.0
        fov_rows.append(
            {
                "fov_id": str(fov_id),
                "voted_sign": vote_sign,
                "vote_margin": vote_margin,
                "dz_hat_um": dz_hat,
                "status": status,
                "num_rois_used": num_used,
                "num_rois_total": int(len(g)),
                "tau": float(tau),
                "top_k": int(args.top_k),
                "runtime_ms": float(runtime_ms),
                "margin_threshold": float(args.margin_threshold),
            }
        )

    fov_out = pd.DataFrame(fov_rows)
    fov_out.to_csv(fov_path, index=False)
    print(f"[DONE] Wrote FOV aggregates: {fov_path} rows={len(fov_out)}")

    if args.save_per_roi:
        work.to_csv(verbose_path, index=False)
        print(f"[DONE] Wrote verbose ROI file: {verbose_path}")


if __name__ == "__main__":
    main()
