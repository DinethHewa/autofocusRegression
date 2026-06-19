#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

from evaluation_utils import (
    default_eval_cli,
    get_paths,
    load_phase5_index,
    log,
    parse_k_values,
    save_json,
    save_plot,
    save_table,
)


def _load_x_batch(paths: list[str], channels: int) -> np.ndarray:
    x = np.empty((len(paths), 200, 200, channels), dtype=np.float32)
    for i, p in enumerate(paths):
        arr = np.load(p, allow_pickle=False) if p.lower().endswith(".npy") else np.load(p, allow_pickle=False)
        if hasattr(arr, "files"):
            if channels == 3 and "XA" in arr.files:
                a = arr["XA"]
            elif channels == 4 and "XB" in arr.files:
                a = arr["XB"]
            else:
                a = arr[arr.files[0]]
        else:
            a = arr
        a = np.asarray(a, dtype=np.float32)
        if a.shape != (200, 200, channels):
            raise ValueError(f"Unexpected tensor shape {a.shape} for {p}; expected (200,200,{channels})")
        x[i] = a
    return x


def _time_model(model: tf.keras.Model, tensor_paths: list[str], channels: int, batch_size: int = 64) -> float:
    if not tensor_paths:
        return float("nan")

    total_s = 0.0
    total_n = 0
    for i in range(0, len(tensor_paths), batch_size):
        sub = tensor_paths[i : i + batch_size]
        x = _load_x_batch(sub, channels=channels)
        t0 = time.perf_counter()
        _ = model.predict(x, verbose=0)
        total_s += time.perf_counter() - t0
        total_n += len(sub)
    return 1000.0 * total_s / max(total_n, 1)


def _plot_runtime_breakdown(metrics: dict, out_path: Path) -> None:
    labels = ["Preprocess/ROI", "StageA/ROI", "StageB/ROI", "Aggregate/FOV"]
    vals = [
        metrics.get("preprocess_time_per_roi_ms", np.nan),
        metrics.get("stageA_time_per_roi_ms", np.nan),
        metrics.get("stageB_time_per_roi_ms", np.nan),
        metrics.get("aggregation_time_per_fov_ms", np.nan),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Runtime Breakdown")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    save_plot(fig, out_path)


def _plot_latency_vs_k(k_vals: list[int], latencies: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_vals, latencies, marker="o")
    ax.set_xlabel("k ROIs per FOV")
    ax.set_ylabel("Estimated latency per FOV (ms)")
    ax.set_title("Latency vs k")
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate runtime profile for the autofocus pipeline")
    default_eval_cli(ap)
    args = ap.parse_args()

    k_values = parse_k_values(args.k_values)
    paths = get_paths(args.track)
    log(paths, "Starting runtime evaluation")
    required = [
        paths.eval_dir / "runtime_metrics.json",
        paths.eval_dir / "latency_vs_k.csv",
        paths.tables_dir / "Table_Runtime.csv",
    ]
    if args.save_latex:
        required.append((paths.tables_dir / "Table_Runtime.csv").with_suffix(".tex"))
    if args.save_plots:
        required.extend([paths.eval_dir / "runtime_breakdown.png", paths.eval_dir / "latency_vs_k.png"])
    if args.resume and all(p.is_file() for p in required):
        log(paths, f"Resume: runtime outputs already exist; skipping. ({paths.eval_dir})")
        return

    idx = load_phase5_index(args.track)
    sample_df = idx.sample(n=min(128, len(idx)), random_state=42).copy().reset_index(drop=True)

    phase3_stats_path = paths.sign_dir.parent / "metrics" / "preprocessing_stats.csv"
    preprocess_ms = np.nan
    if phase3_stats_path.is_file():
        st = pd.read_csv(phase3_stats_path)
        row = st[st["metric"].astype(str) == "dog_time_per_roi_ms"]
        row2 = st[st["metric"].astype(str) == "dwt_time_per_roi_ms"]
        if not row.empty and not row2.empty:
            preprocess_ms = float(row.iloc[0]["value"]) + float(row2.iloc[0]["value"])

    # Stage A timing
    stageA_ms = np.nan
    sign_model_path = paths.sign_dir / "models" / "best_model.keras"
    if sign_model_path.is_file() and "cache_path_XA" in sample_df.columns:
        xa_paths = [p for p in sample_df["cache_path_XA"].dropna().astype(str).tolist() if Path(p).is_file()]
        if xa_paths:
            sign_model = tf.keras.models.load_model(sign_model_path)
            stageA_ms = _time_model(sign_model, xa_paths, channels=3, batch_size=64)

    # Stage B timing
    stageB_ms = np.nan
    plus_path = paths.reg_dir / "models" / "R_plus_best.keras"
    minus_path = paths.reg_dir / "models" / "R_minus_best.keras"
    xb_paths = [p for p in sample_df["cache_path_XB"].dropna().astype(str).tolist() if Path(p).is_file()]

    if plus_path.is_file() and xb_paths:
        reg_model = tf.keras.models.load_model(plus_path)
        stageB_ms = _time_model(reg_model, xb_paths, channels=4, batch_size=64)

    # Aggregation timing (weighted median op)
    agg_times = []
    for _, g in sample_df.groupby("fov_id"):
        vals = np.random.normal(size=len(g))
        w = np.maximum(np.random.rand(len(g)), 1e-6)
        t0 = time.perf_counter()
        order = np.argsort(vals)
        v = vals[order]
        ww = w[order]
        cs = np.cumsum(ww)
        _ = v[np.searchsorted(cs, 0.5 * cs[-1], side="left")]
        agg_times.append((time.perf_counter() - t0) * 1000.0)
    agg_ms = float(np.mean(agg_times)) if agg_times else np.nan

    # Model sizes
    size_sign = (sign_model_path.stat().st_size / (1024 * 1024)) if sign_model_path.is_file() else np.nan
    size_plus = (plus_path.stat().st_size / (1024 * 1024)) if plus_path.is_file() else np.nan
    size_minus = (minus_path.stat().st_size / (1024 * 1024)) if minus_path.is_file() else np.nan

    # Memory usage optional
    mem_mb = np.nan
    if psutil is not None:
        try:
            mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            mem_mb = np.nan

    latency_per_fov = []
    for k in k_values:
        total = (0.0 if np.isnan(preprocess_ms) else preprocess_ms) + (0.0 if np.isnan(stageA_ms) else stageA_ms) + (0.0 if np.isnan(stageB_ms) else stageB_ms)
        latency_per_fov.append(float(k) * total + (0.0 if np.isnan(agg_ms) else agg_ms))

    runtime_metrics = {
        "track": args.track,
        "preprocess_time_per_roi_ms": float(preprocess_ms) if not np.isnan(preprocess_ms) else np.nan,
        "stageA_time_per_roi_ms": float(stageA_ms) if not np.isnan(stageA_ms) else np.nan,
        "stageB_time_per_roi_ms": float(stageB_ms) if not np.isnan(stageB_ms) else np.nan,
        "aggregation_time_per_fov_ms": float(agg_ms) if not np.isnan(agg_ms) else np.nan,
        "total_pipeline_latency_per_fov_k7_ms_est": float(latency_per_fov[k_values.index(7)]) if 7 in k_values else float(latency_per_fov[-1]),
        "model_size_sign_mb": float(size_sign) if not np.isnan(size_sign) else np.nan,
        "model_size_r_plus_mb": float(size_plus) if not np.isnan(size_plus) else np.nan,
        "model_size_r_minus_mb": float(size_minus) if not np.isnan(size_minus) else np.nan,
        "memory_usage_mb": float(mem_mb) if not np.isnan(mem_mb) else np.nan,
        "k_values": k_values,
        "latency_vs_k_ms": latency_per_fov,
    }

    save_json(runtime_metrics, paths.eval_dir / "runtime_metrics.json")

    latency_df = pd.DataFrame({"k": k_values, "latency_ms": latency_per_fov})
    latency_df.to_csv(paths.eval_dir / "latency_vs_k.csv", index=False)

    if args.save_plots:
        _plot_runtime_breakdown(runtime_metrics, paths.eval_dir / "runtime_breakdown.png")
        _plot_latency_vs_k(k_values, latency_per_fov, paths.eval_dir / "latency_vs_k.png")

    table_df = pd.DataFrame(
        [
            {
                "track": args.track,
                "Preprocess_ms_per_ROI": runtime_metrics["preprocess_time_per_roi_ms"],
                "StageA_ms_per_ROI": runtime_metrics["stageA_time_per_roi_ms"],
                "StageB_ms_per_ROI": runtime_metrics["stageB_time_per_roi_ms"],
                "Latency_k7_ms": runtime_metrics["total_pipeline_latency_per_fov_k7_ms_est"],
                "SignModel_MB": runtime_metrics["model_size_sign_mb"],
                "Rplus_MB": runtime_metrics["model_size_r_plus_mb"],
                "Rminus_MB": runtime_metrics["model_size_r_minus_mb"],
                "Memory_MB": runtime_metrics["memory_usage_mb"],
            }
        ]
    )
    save_table(table_df, paths.tables_dir / "Table_Runtime.csv", save_latex=bool(args.save_latex))

    log(paths, f"Runtime evaluation complete. Outputs: {paths.eval_dir}")


if __name__ == "__main__":
    main()
