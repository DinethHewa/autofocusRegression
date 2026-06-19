#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import (
    default_cache_index_path,
    ensure_output_tree,
    load_XA_tensor,
    load_cache_index,
    resolve_out_dir,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ROI-voted sign inference with tau gating.")
    ap.add_argument("--track", required=True, choices=["smears", "biopsy"])
    ap.add_argument("--out-dir", default=None, help="Output dir (must be within out_final_phase/<track>/sign)")
    ap.add_argument("--model-path", default=None, help="Model path (default: out_dir/models/best_model.keras)")
    ap.add_argument("--tau-json", default=None, help="Chosen tau json (default: out_dir/calibration/chosen_tau.json)")
    ap.add_argument("--cache-index", default=None, help="cache_index.csv path")
    ap.add_argument("--mode", required=True, choices=["auto", "csv"])
    ap.add_argument("--fov-csv", default=None, help="Required when mode=csv")
    ap.add_argument("--top-k", type=int, default=7)
    ap.add_argument("--margin-threshold", type=float, default=0.30)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--save-per-roi", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--resume", action="store_true", help="Reuse existing inference outputs when present.")
    return ap.parse_args()


def predict_probs(model: tf.keras.Model, df: pd.DataFrame, batch_size: int = 128) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    probs = []
    paths = df["cache_path_XA"].astype(str).tolist()
    for i in range(0, len(paths), batch_size):
        sub = paths[i : i + batch_size]
        x = np.empty((len(sub), 200, 200, 3), dtype=np.float32)
        for j, p in enumerate(sub):
            x[j] = load_XA_tensor(p)
        probs.append(model.predict(x, verbose=0).reshape(-1))

    p = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    out = df[["roi_uid", "fov_id", "dataset", "roi_importance"]].copy()
    out["p"] = p
    out["c"] = np.maximum(p, 1.0 - p)
    return out


def _auto_group_key(df: pd.DataFrame) -> str:
    if "source_image_path" in df.columns and df["source_image_path"].notna().any():
        return "source_image_path"
    if "stack_id" in df.columns and df["stack_id"].notna().any():
        return "stack_id"
    return "group_id"


def build_groups_auto(cache_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    key = _auto_group_key(cache_df)
    work = cache_df.copy()
    work["fov_id"] = work[key].astype(str)

    if top_k > 0:
        work = work.sort_values(["fov_id", "roi_importance"], ascending=[True, False]).copy()
        work["_rank"] = work.groupby("fov_id").cumcount() + 1
        work = work[work["_rank"] <= int(top_k)].drop(columns=["_rank"]).reset_index(drop=True)
    return work


def _parse_roi_uid_list(text: str) -> list[str]:
    if pd.isna(text):
        return []
    parts = re.split(r"[,;\s]+", str(text).strip())
    return [p for p in parts if p]


def build_groups_csv(cache_df: pd.DataFrame, fov_csv: Path, top_k: int) -> pd.DataFrame:
    if not fov_csv.is_file():
        raise FileNotFoundError(f"fov-csv not found: {fov_csv}")

    src = pd.read_csv(fov_csv)
    if src.empty:
        raise ValueError(f"fov-csv is empty: {fov_csv}")

    if {"fov_id", "roi_uid"}.issubset(src.columns):
        map_df = src[["fov_id", "roi_uid"]].copy()
    elif {"fov_id", "roi_uids"}.issubset(src.columns):
        rows = []
        for _, row in src.iterrows():
            rid_list = _parse_roi_uid_list(row["roi_uids"])
            for rid in rid_list:
                rows.append({"fov_id": row["fov_id"], "roi_uid": rid})
        map_df = pd.DataFrame(rows)
    elif src.shape[1] >= 2:
        c0, c1 = src.columns[:2]
        rows = []
        for _, row in src.iterrows():
            val = str(row[c1])
            rid_list = _parse_roi_uid_list(val)
            if len(rid_list) == 1 and val == rid_list[0]:
                rows.append({"fov_id": row[c0], "roi_uid": rid_list[0]})
            else:
                for rid in rid_list:
                    rows.append({"fov_id": row[c0], "roi_uid": rid})
        map_df = pd.DataFrame(rows)
    else:
        raise ValueError("fov-csv must contain (fov_id, roi_uid) or (fov_id, roi_uids).")

    if map_df.empty:
        raise ValueError("No fov->roi mappings found in fov-csv.")

    map_df["roi_uid"] = map_df["roi_uid"].astype(str)
    merged = map_df.merge(
        cache_df,
        on="roi_uid",
        how="left",
        suffixes=("", "_cache"),
    )

    missing = int(merged["cache_path_XA"].isna().sum())
    if missing > 0:
        print(f"[WARN] {missing} roi_uids from fov-csv are missing in cache index.")

    merged = merged[merged["cache_path_XA"].notna()].copy().reset_index(drop=True)
    if merged.empty:
        raise ValueError("No valid ROI rows after joining fov-csv with cache index.")

    merged["fov_id"] = merged["fov_id"].astype(str)

    if top_k > 0:
        merged = merged.sort_values(["fov_id", "roi_importance"], ascending=[True, False]).copy()
        merged["_rank"] = merged.groupby("fov_id").cumcount() + 1
        merged = merged[merged["_rank"] <= int(top_k)].drop(columns=["_rank"]).reset_index(drop=True)

    return merged


def vote_group(
    g: pd.DataFrame,
    tau: float,
    margin_threshold: float,
    eps: float,
    save_per_roi: bool,
) -> tuple[dict, list[dict]]:
    g = g.sort_values("roi_importance", ascending=False).copy().reset_index(drop=True)

    V = 0.0
    W = 0.0
    processed = 0
    kept = 0
    early_exit = False
    per_rows: list[dict] = []

    for _, row in g.iterrows():
        processed += 1
        p = float(row["p"])
        c = float(row["c"])
        importance = float(row.get("roi_importance", 1.0))

        keep = c >= tau
        w = 0.0
        vote = 0
        if keep:
            vote = 1 if p >= 0.5 else 0
            w = importance * c
            V += w * vote
            W += w
            kept += 1

        margin = abs(V - 0.5 * W) / (0.5 * W + eps) if W > 0 else 0.0
        if save_per_roi:
            per_rows.append(
                {
                    "fov_id": str(row["fov_id"]),
                    "roi_uid": str(row["roi_uid"]),
                    "dataset": row.get("dataset", None),
                    "roi_importance": importance,
                    "p": p,
                    "c": c,
                    "kept": int(keep),
                    "w": w,
                    "V_cum": V,
                    "W_cum": W,
                    "margin_cum": margin,
                    "processed_rank": processed,
                }
            )

        if W > 0 and margin >= margin_threshold:
            early_exit = True
            break

    if W <= 0:
        result = {
            "fov_id": str(g.iloc[0]["fov_id"]),
            "pred_sign": "uncertain",
            "pred_sign_int": np.nan,
            "vote_margin": 0.0,
            "V": 0.0,
            "W": 0.0,
            "num_rois_total": int(len(g)),
            "num_rois_processed": int(processed),
            "num_rois_kept": int(kept),
            "early_exit": bool(early_exit),
        }
        return result, per_rows

    sign_int = 1 if V >= 0.5 * W else 0
    margin = abs(V - 0.5 * W) / (0.5 * W + eps)
    result = {
        "fov_id": str(g.iloc[0]["fov_id"]),
        "pred_sign": str(sign_int),
        "pred_sign_int": int(sign_int),
        "vote_margin": float(margin),
        "V": float(V),
        "W": float(W),
        "num_rois_total": int(len(g)),
        "num_rois_processed": int(processed),
        "num_rois_kept": int(kept),
        "early_exit": bool(early_exit),
    }
    return result, per_rows


def main() -> None:
    args = parse_args()

    out_dir = resolve_out_dir(args.track, args.out_dir)
    out_paths = ensure_output_tree(out_dir)
    out_csv = out_paths["inference"] / "vote_sign_results.csv"
    per_csv = out_paths["inference"] / "vote_sign_per_roi.csv"
    if args.resume:
        if args.save_per_roi and out_csv.is_file() and per_csv.is_file():
            print(f"[INFO] Resume: vote outputs already exist; skipping.")
            return
        if (not args.save_per_roi) and out_csv.is_file():
            print(f"[INFO] Resume: vote results already exist at {out_csv}; skipping.")
            return

    model_path = Path(args.model_path) if args.model_path else out_paths["models"] / "best_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    tau_json = Path(args.tau_json) if args.tau_json else out_paths["calibration"] / "chosen_tau.json"
    if not tau_json.is_file():
        raise FileNotFoundError(f"Tau file not found: {tau_json}. Run calibrate_tau.py first.")

    with tau_json.open("r", encoding="utf-8") as f:
        tau = float(json.load(f)["tau"])

    cache_index_path = str(default_cache_index_path(args.track) if args.cache_index is None else args.cache_index)
    cache_df = load_cache_index(track=args.track, cache_index_path=cache_index_path)

    if args.mode == "auto":
        infer_df = build_groups_auto(cache_df=cache_df, top_k=int(args.top_k))
    else:
        if args.fov_csv is None:
            raise ValueError("--fov-csv is required when mode=csv")
        infer_df = build_groups_csv(cache_df=cache_df, fov_csv=Path(args.fov_csv), top_k=int(args.top_k))

    if infer_df.empty:
        raise ValueError("No ROI rows to infer.")

    model = tf.keras.models.load_model(model_path)
    pred_df = predict_probs(model=model, df=infer_df, batch_size=int(args.batch_size))
    infer_df = infer_df.merge(pred_df[["roi_uid", "fov_id", "p", "c"]], on=["roi_uid", "fov_id"], how="left")

    if infer_df[["p", "c"]].isna().any().any():
        raise RuntimeError("Prediction merge failed for some ROIs.")

    result_rows = []
    per_roi_rows = []
    for fov_id, g in infer_df.groupby("fov_id", sort=True):
        result, roi_rows = vote_group(
            g,
            tau=float(tau),
            margin_threshold=float(args.margin_threshold),
            eps=float(args.eps),
            save_per_roi=bool(args.save_per_roi),
        )
        result["track"] = args.track
        result["tau"] = float(tau)
        result["mode"] = args.mode
        result_rows.append(result)
        if args.save_per_roi:
            per_roi_rows.extend(roi_rows)

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(out_csv, index=False)
    print(f"[DONE] Wrote vote results: {out_csv} rows={len(results_df)}")

    if args.save_per_roi:
        pd.DataFrame(per_roi_rows).to_csv(per_csv, index=False)
        print(f"[DONE] Wrote per-ROI details: {per_csv} rows={len(per_roi_rows)}")


if __name__ == "__main__":
    main()
