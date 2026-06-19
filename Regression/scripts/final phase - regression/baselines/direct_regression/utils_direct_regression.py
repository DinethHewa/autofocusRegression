#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent.parent / 'evaluation'
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from baseline_utils import (
    baseline_method_root,
    build_standard_fov_frame,
    ensure_baseline_tree,
    gt_per_fov,
    load_phase5_index_full,
    resolve_master_splits,
)
from evaluation_utils import regression_metrics, save_json

PHASE5_DIR = HERE.parent.parent / 'phase5_regression'
if str(PHASE5_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIR))

from utils import load_XB_tensor  # type: ignore


def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def default_direct_root(track: str) -> Path:
    return baseline_method_root(track, 'direct_regression')


def ensure_direct_tree(track: str) -> dict[str, Path]:
    return ensure_baseline_tree(default_direct_root(track))


def load_xb_batch(df: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch = df.iloc[indices]
    paths = batch['cache_path_XB'].astype(str).tolist()
    x = np.empty((len(paths), 200, 200, 4), dtype=np.float32)
    for i, p in enumerate(paths):
        x[i] = load_XB_tensor(p)
    y = pd.to_numeric(batch['defocus_um'], errors='coerce').to_numpy(dtype=np.float32)
    return x, y


def predict_roi(model: tf.keras.Model, df: pd.DataFrame, batch_size: int = 128) -> np.ndarray:
    preds: list[np.ndarray] = []
    n = len(df)
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n), dtype=np.int64)
        x, _ = load_xb_batch(df, idx)
        y_hat = model(x, training=False)
        preds.append(tf.squeeze(y_hat, axis=-1).numpy())
    if not preds:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32)


def aggregate_fov_predictions(roi_df: pd.DataFrame, top_k: int = 7, default_method: str = 'weighted_mean') -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fov_id, g in roi_df.groupby('fov_id', sort=True):
        work = g.sort_values(['roi_importance', 'roi_uid'], ascending=[False, True]).head(int(top_k)).copy()
        pred = pd.to_numeric(work['y_pred_signed_um'], errors='coerce').to_numpy(dtype=float)
        gt = pd.to_numeric(work['defocus_um'], errors='coerce').to_numpy(dtype=float)
        w = pd.to_numeric(work.get('roi_importance', 1.0), errors='coerce').fillna(1.0).to_numpy(dtype=float)
        valid = np.isfinite(pred)
        pred = pred[valid]
        w = w[valid]
        if pred.size == 0:
            mean_pred = weighted_pred = median_pred = np.nan
            status = 'no_valid_roi'
        else:
            mean_pred = float(np.mean(pred))
            weighted_pred = float(np.sum(pred * w) / max(float(np.sum(w)), 1e-12)) if np.sum(w) > 0 else mean_pred
            median_pred = float(np.median(pred))
            status = 'ok'
        chosen = {'mean': mean_pred, 'weighted_mean': weighted_pred, 'median': median_pred}.get(default_method, weighted_pred)
        rows.append(
            {
                'fov_id': str(fov_id),
                'dataset': str(work['dataset'].iloc[0]),
                'y_true_signed_um': float(np.nanmedian(gt)) if np.isfinite(gt).any() else np.nan,
                'pred_mean_um': mean_pred,
                'pred_weighted_mean_um': weighted_pred,
                'pred_median_um': median_pred,
                'y_pred_signed_um': chosen,
                'num_inputs_used': int(len(work)),
                'status': status,
            }
        )
    return pd.DataFrame(rows)


def per_bin_fov_metrics(fov_df: pd.DataFrame, bin_width_um: float = 0.5) -> pd.DataFrame:
    work = fov_df.copy()
    gt = np.abs(pd.to_numeric(work['y_true_signed_um'], errors='coerce').to_numpy(dtype=float))
    abs_err = np.abs(pd.to_numeric(work['y_pred_signed_um'], errors='coerce').to_numpy(dtype=float) - pd.to_numeric(work['y_true_signed_um'], errors='coerce').to_numpy(dtype=float))
    valid = np.isfinite(gt) & np.isfinite(abs_err)
    gt = gt[valid]
    abs_err = abs_err[valid]
    if gt.size == 0:
        return pd.DataFrame(columns=['bin_lo', 'bin_hi', 'count', 'mae_um'])
    vmax = max(float(np.nanmax(gt)), float(bin_width_um))
    edges = np.arange(0.0, vmax + float(bin_width_um) + 1e-12, float(bin_width_um))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (gt >= lo) & (gt < hi)
        rows.append({'bin_lo': float(lo), 'bin_hi': float(hi), 'count': int(np.sum(mask)), 'mae_um': float(np.mean(abs_err[mask])) if np.any(mask) else np.nan})
    return pd.DataFrame(rows)


def evaluate_direct_predictions(roi_df: pd.DataFrame, fov_df: pd.DataFrame) -> dict[str, Any]:
    roi_true = pd.to_numeric(roi_df['defocus_um'], errors='coerce').to_numpy(dtype=float)
    roi_pred = pd.to_numeric(roi_df['y_pred_signed_um'], errors='coerce').to_numpy(dtype=float)
    roi_mask = np.isfinite(roi_true) & np.isfinite(roi_pred)
    f_true = pd.to_numeric(fov_df['y_true_signed_um'], errors='coerce').to_numpy(dtype=float)
    f_pred = pd.to_numeric(fov_df['y_pred_signed_um'], errors='coerce').to_numpy(dtype=float)
    f_mask = np.isfinite(f_true) & np.isfinite(f_pred)
    roi_metrics = regression_metrics(roi_true[roi_mask], roi_pred[roi_mask]) if np.any(roi_mask) else {k: np.nan for k in ['mae_um', 'rmse_um', 'median_abs_error_um', 'p95_abs_error_um']}
    fov_metrics = regression_metrics(f_true[f_mask], f_pred[f_mask]) if np.any(f_mask) else {k: np.nan for k in ['mae_um', 'rmse_um', 'median_abs_error_um', 'p95_abs_error_um']}
    return {
        'n_roi_eval': int(np.sum(roi_mask)),
        'n_fov_eval': int(np.sum(f_mask)),
        'roi_mae_um': float(roi_metrics['mae_um']) if np.isfinite(roi_metrics['mae_um']) else np.nan,
        'roi_rmse_um': float(roi_metrics['rmse_um']) if np.isfinite(roi_metrics['rmse_um']) else np.nan,
        'fov_mae_um': float(fov_metrics['mae_um']) if np.isfinite(fov_metrics['mae_um']) else np.nan,
        'fov_rmse_um': float(fov_metrics['rmse_um']) if np.isfinite(fov_metrics['rmse_um']) else np.nan,
        'fov_median_abs_error_um': float(fov_metrics['median_abs_error_um']) if np.isfinite(fov_metrics['median_abs_error_um']) else np.nan,
        'fov_p95_abs_error_um': float(fov_metrics['p95_abs_error_um']) if np.isfinite(fov_metrics['p95_abs_error_um']) else np.nan,
    }


def write_split_files(paths: dict[str, Path], train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    keep_cols = [c for c in ['roi_uid', 'dataset', 'group_id', 'fov_id', 'defocus_um', 'roi_importance', 'cache_path_XB', 'cache_path_XA', 'patch_id', 'source_image_path'] if c in train_df.columns]
    train_df[keep_cols].to_csv(paths['splits'] / 'train.csv', index=False)
    val_df[keep_cols].to_csv(paths['splits'] / 'val.csv', index=False)
    test_df[keep_cols].to_csv(paths['splits'] / 'test.csv', index=False)


def load_or_create_direct_splits(track: str, seed: int = 42, split: tuple[float, float, float] = (0.70, 0.15, 0.15), resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    index_df = load_phase5_index_full(track)
    train_df, val_df, test_df, meta = resolve_master_splits(track, index_df, seed=seed, split=split, resume=resume)
    return train_df, val_df, test_df, meta, index_df


def save_eval_json(payload: dict[str, Any], out_path: Path) -> None:
    save_json(payload, out_path)
