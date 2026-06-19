#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import (
    build_sign_labels,
    compute_balanced_accuracy,
    default_cache_index_path,
    default_sign_out_dir,
    ensure_output_tree,
    load_XA_tensor,
    load_cache_index,
    resolve_out_dir,
    save_json,
)


def predict_probs(model: tf.keras.Model, df: pd.DataFrame, batch_size: int = 128) -> np.ndarray:
    probs = []
    paths = df["cache_path_XA"].astype(str).tolist()
    for i in range(0, len(paths), batch_size):
        sub = paths[i : i + batch_size]
        x = np.empty((len(sub), 200, 200, 3), dtype=np.float32)
        for j, p in enumerate(sub):
            x[j] = load_XA_tensor(p)
        probs.append(model.predict(x, verbose=0).reshape(-1))
    return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calibrate confidence gate tau for ROI sign predictions.")
    ap.add_argument("--track", required=True, choices=["smears", "biopsy"])
    ap.add_argument("--out-dir", default=None, help="Output dir (must be within out_final_phase/<track>/sign)")
    ap.add_argument("--model-path", default=None, help="Model path (default: out_dir/models/best_model.keras)")
    ap.add_argument("--cache-index", default=None, help="cache_index.csv path")
    ap.add_argument("--splits-dir", default=None, help="splits directory (default: out_dir/splits)")
    ap.add_argument("--tau-min", type=float, default=0.50)
    ap.add_argument("--tau-max", type=float, default=0.90)
    ap.add_argument("--tau-step", type=float, default=0.01)
    ap.add_argument("--min-coverage", type=float, default=0.60)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--resume", action="store_true", help="Reuse existing calibration outputs when present.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_out_dir(args.track, args.out_dir)
    out_paths = ensure_output_tree(out_dir)
    cal_dir = out_paths["calibration"]
    sweep_path = cal_dir / "tau_sweep.csv"
    chosen_path = cal_dir / "chosen_tau.json"
    pred_path = cal_dir / "val_roi_predictions.csv"

    if args.resume and sweep_path.is_file() and chosen_path.is_file() and pred_path.is_file():
        print(f"[INFO] Resume: calibration outputs already exist in {cal_dir}; skipping.")
        return

    model_path = Path(args.model_path) if args.model_path else out_paths["models"] / "best_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cache_index_path = str(default_cache_index_path(args.track) if args.cache_index is None else args.cache_index)
    cache_df = load_cache_index(track=args.track, cache_index_path=cache_index_path)
    cache_df = build_sign_labels(cache_df)

    splits_dir = Path(args.splits_dir) if args.splits_dir else out_paths["splits"]
    val_csv = splits_dir / "val.csv"
    if not val_csv.is_file():
        raise FileNotFoundError(f"Validation split not found: {val_csv}. Run train_sign.py first.")

    val_split = pd.read_csv(val_csv)
    if "roi_uid" not in val_split.columns:
        raise ValueError(f"Split file missing roi_uid: {val_csv}")

    val_df = val_split[["roi_uid"]].merge(
        cache_df[["roi_uid", "cache_path_XA", "y_sign", "delta_z", "abs_delta_z", "dataset"]],
        on="roi_uid",
        how="left",
    )

    missing = int(val_df["cache_path_XA"].isna().sum())
    if missing > 0:
        raise ValueError(f"Validation split has {missing} roi_uid(s) missing in cache index.")

    model = tf.keras.models.load_model(model_path)
    probs = predict_probs(model, val_df, batch_size=args.batch_size)
    y_true = val_df["y_sign"].to_numpy(dtype=int)
    y_pred = (probs >= 0.5).astype(int)
    conf = np.maximum(probs, 1.0 - probs)

    if args.tau_step <= 0:
        raise ValueError("tau-step must be > 0")
    taus = np.arange(args.tau_min, args.tau_max + args.tau_step / 2.0, args.tau_step)

    rows = []
    for tau in taus:
        keep = conf >= float(tau)
        kept = int(keep.sum())
        coverage = float(kept / (len(conf) + args.eps))

        if kept == 0:
            wrong_rate = 1.0
            bal_acc = np.nan
            acc = np.nan
        else:
            wrong_rate = float((y_pred[keep] != y_true[keep]).mean())
            bal_acc = float(compute_balanced_accuracy(y_true[keep], y_pred[keep]))
            acc = float((y_pred[keep] == y_true[keep]).mean())

        rows.append(
            {
                "tau": float(np.round(tau, 6)),
                "kept_count": kept,
                "total_count": int(len(conf)),
                "coverage": coverage,
                "wrong_sign_rate": wrong_rate,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
            }
        )

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(sweep_path, index=False)

    candidates = sweep_df[(sweep_df["coverage"] >= float(args.min_coverage)) & (sweep_df["kept_count"] > 0)].copy()
    if not candidates.empty:
        chosen = candidates.sort_values(
            ["wrong_sign_rate", "balanced_accuracy", "coverage"],
            ascending=[True, False, False],
        ).iloc[0]
        rule_used = "min_wrong_sign_rate_subject_to_coverage"
    else:
        fallback = sweep_df[sweep_df["kept_count"] > 0].copy()
        if fallback.empty:
            chosen = pd.Series(
                {
                    "tau": float(args.tau_min),
                    "coverage": 0.0,
                    "wrong_sign_rate": 1.0,
                    "balanced_accuracy": 0.0,
                    "kept_count": 0,
                    "total_count": int(len(conf)),
                }
            )
            rule_used = "fallback_no_kept_rois"
        else:
            chosen = fallback.sort_values(
                ["balanced_accuracy", "wrong_sign_rate", "coverage"],
                ascending=[False, True, False],
            ).iloc[0]
            rule_used = "best_balanced_accuracy_no_coverage_match"

    chosen_payload = {
        "track": args.track,
        "tau": float(chosen["tau"]),
        "coverage": float(chosen["coverage"]),
        "wrong_sign_rate": float(chosen["wrong_sign_rate"]),
        "balanced_accuracy": float(chosen.get("balanced_accuracy", np.nan)),
        "kept_count": int(chosen.get("kept_count", 0)),
        "total_count": int(chosen.get("total_count", len(conf))),
        "min_coverage": float(args.min_coverage),
        "rule_used": rule_used,
        "model_path": str(model_path),
        "tau_range": [float(args.tau_min), float(args.tau_max), float(args.tau_step)],
    }

    save_json(chosen_payload, chosen_path)

    pred_df = val_df[["roi_uid", "dataset", "delta_z", "y_sign"]].copy()
    pred_df["p"] = probs
    pred_df["c"] = conf
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(pred_path, index=False)

    print(f"[DONE] Tau sweep: {sweep_path}")
    print(f"[DONE] Chosen tau: {chosen_path}")
    print(f"[DONE] Val predictions: {pred_path}")


if __name__ == "__main__":
    main()
