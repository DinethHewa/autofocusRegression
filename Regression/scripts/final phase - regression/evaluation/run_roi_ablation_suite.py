#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from evaluation_utils import (
    TRACKS,
    auroc_score,
    classification_metrics,
    default_eval_cli,
    get_paths,
    log,
    require_file,
    roc_curve_points,
    save_json,
    save_plot,
    save_table,
    weighted_median,
    wilcoxon_signed_rank,
)
from roi_policy_utils import (
    K_POLICIES,
    ROIPolicyContext,
    expand_policy_jobs,
    policy_label,
    sanitize_policy_label,
    select_rois_for_policy,
    summarize_context_warnings,
)
import roi_policy_utils as rpu

HERE = Path(__file__).resolve().parent
PHASE5_DIR = HERE.parent / 'phase5_regression'
PHASE3_DIR = HERE.parent / 'phase3_preprocessing'
if str(PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE3_DIR))

from preprocess_ops import assemble_XA_XB, compute_dog_lite, compute_dwt_ehf, percentile_clip, rescale01, to_grayscale, zscore  # type: ignore

DEFAULT_POLICIES = [
    'center_top1',
    'random_k',
    'focus_only_topk',
    'occupancy_only_topk',
    'hybrid_proposed',
    'all_rois',
]

PRIMARY_COMPARE_BASELINES = {'center_top1', 'random_k', 'focus_only_topk', 'occupancy_only_topk', 'all_rois'}
NEAR_FOCUS_BINS = [
    ('<=0.5', 0.0, 0.5),
    ('0.5-1.0', 0.5, 1.0),
    ('1.0-2.0', 1.0, 2.0),
    ('>2.0', 2.0, np.inf),
]
ROI_COUNT_BUCKETS = [
    ('1', 1, 1),
    ('2-3', 2, 3),
    ('4-7', 4, 7),
    ('8+', 8, np.inf),
]
PERTURBATIONS = [
    ('blur', 0.8),
    ('blur', 1.2),
    ('noise', 0.01),
    ('noise', 0.02),
    ('brightness', 0.05),
    ('brightness', -0.05),
    ('contrast', 0.90),
    ('contrast', 1.10),
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module {name} from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE5_UTILS = _load_module('phase5_reg_utils_eval', PHASE5_DIR / 'utils.py')
load_XA_tensor = PHASE5_UTILS.load_XA_tensor
load_XB_tensor = PHASE5_UTILS.load_XB_tensor
parse_voted_sign_value = PHASE5_UTILS.parse_voted_sign_value


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _load_trusted_keras_model(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)


def _load_phase5_index_full(track: str) -> pd.DataFrame:
    p = require_file(get_paths(track).reg_dir / 'index_phase5.csv', 'phase5 index')
    df = pd.read_csv(p, low_memory=False)
    required = ['roi_uid', 'dataset', 'fov_id', 'defocus_um', 'y_sign', 'y_mag_um', 'cache_path_XA', 'cache_path_XB']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Phase5 index missing required columns {missing}: {p}')
    if 'patch_id' not in df.columns:
        df['patch_id'] = 'r00_c00'
    if 'roi_importance' not in df.columns:
        df['roi_importance'] = 1.0
    df['roi_importance'] = pd.to_numeric(df['roi_importance'], errors='coerce').fillna(1.0).astype(float)
    for col in ['defocus_um', 'y_mag_um']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'source_image_path' not in df.columns:
        if 'image_path' in df.columns:
            df['source_image_path'] = df['image_path'].astype(str)
        else:
            df['source_image_path'] = df['fov_id'].astype(str)
    df['fov_id'] = df['fov_id'].astype(str)
    df['roi_uid'] = df['roi_uid'].astype(str)
    df['dataset'] = df['dataset'].astype(str)
    df['patch_id'] = df['patch_id'].astype(str)
    return df


def _gt_per_fov(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fov_id, g in index_df.groupby('fov_id', sort=True):
        vals = pd.to_numeric(g['defocus_um'], errors='coerce').to_numpy(dtype=float)
        w = pd.to_numeric(g.get('roi_importance', 1.0), errors='coerce').fillna(1.0).to_numpy(dtype=float)
        gt = weighted_median(vals, w)
        if gt is None:
            finite = vals[np.isfinite(vals)]
            gt = float(np.median(finite)) if finite.size else np.nan
        rows.append({'fov_id': str(fov_id), 'gt_signed_um': gt})
    return pd.DataFrame(rows)


def _infer_dataset_for_fov(index_df: pd.DataFrame) -> pd.DataFrame:
    tmp = (
        index_df.groupby(['fov_id', 'dataset']).size().reset_index(name='n').sort_values(['fov_id', 'n'], ascending=[True, False])
    )
    tmp = tmp.drop_duplicates(subset=['fov_id'], keep='first').reset_index(drop=True)
    return tmp[['fov_id', 'dataset']]


def _load_tau(paths) -> float:
    tau_path = require_file(paths.sign_dir / 'calibration' / 'chosen_tau.json', 'chosen tau json')
    with tau_path.open('r', encoding='utf-8') as f:
        return float(json.load(f)['tau'])


def _predict_sign_all(model: tf.keras.Model, df: pd.DataFrame, batch_size: int) -> tuple[np.ndarray, float]:
    probs = []
    start = time.perf_counter()
    paths = df['cache_path_XA'].astype(str).tolist()
    for i in range(0, len(paths), batch_size):
        sub = paths[i:i + batch_size]
        x = np.empty((len(sub), 200, 200, 3), dtype=np.float32)
        for j, p in enumerate(sub):
            x[j] = load_XA_tensor(p)
        probs.append(model.predict(x, verbose=0).reshape(-1))
    elapsed = time.perf_counter() - start
    arr = np.concatenate(probs, axis=0).astype(np.float32) if probs else np.zeros((0,), dtype=np.float32)
    return arr, elapsed


def _predict_reg_all(model: tf.keras.Model, df: pd.DataFrame, batch_size: int) -> tuple[np.ndarray, float]:
    preds = []
    start = time.perf_counter()
    paths = df['cache_path_XB'].astype(str).tolist()
    for i in range(0, len(paths), batch_size):
        sub = paths[i:i + batch_size]
        x = np.empty((len(sub), 200, 200, 4), dtype=np.float32)
        for j, p in enumerate(sub):
            x[j] = load_XB_tensor(p)
        y_hat, _ = model(x, training=False)
        preds.append(tf.squeeze(y_hat, axis=-1).numpy())
    elapsed = time.perf_counter() - start
    arr = np.concatenate(preds, axis=0).astype(np.float32) if preds else np.zeros((0,), dtype=np.float32)
    return arr, elapsed


def _prepare_fixed_model_outputs(index_df: pd.DataFrame, paths, batch_size: int, resume: bool) -> tuple[pd.DataFrame, dict[str, float]]:
    out_csv = paths.eval_dir / 'roi_ablation_fixed_model_outputs.csv'
    latency_json = paths.eval_dir / 'roi_ablation_fixed_model_latency.json'
    if resume and out_csv.is_file() and latency_json.is_file():
        cached = pd.read_csv(out_csv)
        needed = {'roi_uid', 'p_sign', 'c_sign', 'mag_plus', 'mag_minus'}
        if needed.issubset(cached.columns):
            cached['roi_uid'] = cached['roi_uid'].astype(str)
            with latency_json.open('r', encoding='utf-8') as f:
                lat = json.load(f)
            return cached[['roi_uid', 'p_sign', 'c_sign', 'mag_plus', 'mag_minus']], lat

    sign_model_path = require_file(paths.sign_dir / 'models' / 'best_model.keras', 'StageA sign model')
    plus_model_path = require_file(paths.reg_dir / 'models' / 'R_plus_best.keras', 'StageB plus model')
    minus_model_path = require_file(paths.reg_dir / 'models' / 'R_minus_best.keras', 'StageB minus model')

    sign_model = _load_trusted_keras_model(sign_model_path)
    plus_model = _load_trusted_keras_model(plus_model_path)
    minus_model = _load_trusted_keras_model(minus_model_path)

    p_sign, sign_time_s = _predict_sign_all(sign_model, index_df, batch_size=batch_size)
    mag_plus, plus_time_s = _predict_reg_all(plus_model, index_df, batch_size=batch_size)
    mag_minus, minus_time_s = _predict_reg_all(minus_model, index_df, batch_size=batch_size)
    c_sign = np.maximum(p_sign, 1.0 - p_sign)

    out = pd.DataFrame(
        {
            'roi_uid': index_df['roi_uid'].astype(str).tolist(),
            'p_sign': p_sign,
            'c_sign': c_sign,
            'mag_plus': mag_plus,
            'mag_minus': mag_minus,
        }
    )
    out.to_csv(out_csv, index=False)

    n = max(int(len(index_df)), 1)
    latency = {
        'sign_total_time_s': float(sign_time_s),
        'r_plus_total_time_s': float(plus_time_s),
        'r_minus_total_time_s': float(minus_time_s),
        'sign_ms_per_roi': 1000.0 * float(sign_time_s) / n,
        'stageB_ms_per_roi': 1000.0 * float(0.5 * (plus_time_s + minus_time_s)) / n,
        'r_plus_ms_per_roi': 1000.0 * float(plus_time_s) / n,
        'r_minus_ms_per_roi': 1000.0 * float(minus_time_s) / n,
        'n_rois': int(len(index_df)),
        'batch_size': int(batch_size),
    }
    save_json(latency, latency_json)
    return out, latency


def _subset_index_by_fov(index_df: pd.DataFrame, max_fovs: int | None, seed: int) -> pd.DataFrame:
    if max_fovs is None:
        return index_df
    fov_ids = np.array(sorted(index_df['fov_id'].astype(str).unique().tolist()), dtype=object)
    n = min(int(max_fovs), len(fov_ids))
    rng = np.random.default_rng(int(seed))
    keep = set(rng.choice(fov_ids, size=n, replace=False).tolist())
    return index_df[index_df['fov_id'].isin(keep)].copy().reset_index(drop=True)


def _init_policy_arrays(master_df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(master_df)
    return {
        'pred_signed_um': np.full((n,), np.nan, dtype=np.float32),
        'stageA_pred_sign': np.full((n,), np.nan, dtype=np.float32),
        'stageA_score': np.full((n,), np.nan, dtype=np.float32),
        'vote_margin': np.full((n,), np.nan, dtype=np.float32),
        'num_selected_rois': np.zeros((n,), dtype=np.int32),
        'num_stageA_kept': np.zeros((n,), dtype=np.int32),
        'selection_time_ms': np.zeros((n,), dtype=np.float32),
        'aggregation_time_ms': np.zeros((n,), dtype=np.float32),
        'coverage_mask': np.zeros((n,), dtype=bool),
    }


def _bootstrap_policy_ci(abs_err: np.ndarray, signed_err: np.ndarray, catastrophic: np.ndarray, n_bootstrap: int, seed: int, max_fovs: int) -> list[dict[str, Any]]:
    valid = np.isfinite(abs_err) & np.isfinite(signed_err)
    abs_err = abs_err[valid]
    signed_err = signed_err[valid]
    catastrophic = catastrophic[valid]
    n = len(abs_err)
    if n == 0:
        return []
    if n > int(max_fovs):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=int(max_fovs), replace=False)
        abs_err = abs_err[idx]
        signed_err = signed_err[idx]
        catastrophic = catastrophic[idx]
        n = len(abs_err)
    rng = np.random.default_rng(seed)
    rows = []
    metrics = {
        'mae_um': lambda i: float(np.mean(abs_err[i])),
        'rmse_um': lambda i: float(np.sqrt(np.mean(np.square(signed_err[i])))),
        'within_1um_pct': lambda i: float(np.mean(abs_err[i] <= 1.0) * 100.0),
        'catastrophic_wrong_direction_pct': lambda i: float(np.mean(catastrophic[i]) * 100.0),
    }
    base = np.arange(n)
    for name, fn in metrics.items():
        point = fn(base)
        samples = np.empty((int(n_bootstrap),), dtype=np.float32)
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            samples[b] = fn(idx)
        rows.append(
            {
                'metric': name,
                'point_estimate': float(point),
                'ci95_lo': float(np.percentile(samples, 2.5)),
                'ci95_hi': float(np.percentile(samples, 97.5)),
                'n_fov_bootstrap': int(n),
                'n_bootstrap': int(n_bootstrap),
            }
        )
    return rows


def _holm_correction(pvals: list[float]) -> list[bool]:
    m = len(pvals)
    order = np.argsort(np.asarray(pvals, dtype=float))
    sig = [False] * m
    prev_reject = True
    for rank, idx in enumerate(order, start=1):
        thresh = 0.05 / (m - rank + 1)
        reject = prev_reject and (float(pvals[idx]) <= thresh)
        sig[int(idx)] = bool(reject)
        prev_reject = reject
    return sig


def _effect_size_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float('nan')
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float('nan')
    if not np.isfinite(sd) or sd <= 1e-12:
        return float('nan')
    return float(np.mean(diff) / sd)


def _save_table_bundle(df: pd.DataFrame, filename: str, paths, save_latex_flag: bool, best_rules: dict[str, str] | None = None) -> None:
    ref_dir = paths.eval_dir / 'reference' / 'tables'
    ref_dir.mkdir(parents=True, exist_ok=True)
    save_table(df, ref_dir / filename, save_latex=save_latex_flag and best_rules is None)
    save_table(df, paths.tables_dir / filename, save_latex=save_latex_flag and best_rules is None)
    if save_latex_flag and best_rules is not None:
        tex = _df_to_latex_bold_best(df, best_rules)
        for dst in [(ref_dir / filename).with_suffix('.tex'), (paths.tables_dir / filename).with_suffix('.tex')]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(tex, encoding='utf-8')


def _format_cell(val: Any) -> str:
    if pd.isna(val):
        return '--'
    if isinstance(val, (float, np.floating)):
        return f'{float(val):.4f}'
    return str(val)


def _df_to_latex_bold_best(df: pd.DataFrame, best_rules: dict[str, str]) -> str:
    work = df.copy()
    display = work.copy().astype(object)
    for col, rule in best_rules.items():
        if col not in work.columns:
            continue
        vals = pd.to_numeric(work[col], errors='coerce')
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            continue
        best = vals.min() if rule == 'min' else vals.max()
        mask = np.asarray(
            np.isclose(pd.to_numeric(work[col], errors='coerce'), float(best), equal_nan=False),
            dtype=bool,
        )
        for i in range(len(display)):
            display.at[i, col] = _format_cell(work.iloc[i][col])
            if bool(mask[i]):
                display.at[i, col] = '\\textbf{' + str(display.at[i, col]) + '}'
    for col in display.columns:
        if col not in best_rules:
            display[col] = display[col].map(_format_cell)
    return display.to_latex(index=False, escape=False)


def _backfill_report_tables(paths, save_latex_flag: bool) -> None:
    table_specs = [
        (
            paths.eval_dir / 'roi_ablation_to_regression_performance.csv',
            'Table_ROI_Ablation_Regression.csv',
            {
                'mae_um': 'min',
                'rmse_um': 'min',
                'median_abs_error_um': 'min',
                'p95_abs_error_um': 'min',
                'latency_ms_per_fov': 'min',
                'coverage_pct': 'max',
                'within_0.5um_pct': 'max',
                'within_1um_pct': 'max',
                'within_2um_pct': 'max',
                'stageA_balanced_acc': 'max',
                'stageA_auroc': 'max',
                'stageB_mae_um': 'min',
            },
        ),
        (
            paths.eval_dir / 'roi_ablation_by_dataset.csv',
            'Table_ROI_Ablation_ByDataset.csv',
            None,
        ),
        (
            paths.eval_dir / 'roi_ablation_near_focus.csv',
            'Table_ROI_NearFocus.csv',
            None,
        ),
        (
            paths.eval_dir / 'domain_gap_reduction.csv',
            'Table_DomainGapReduction.csv',
            None,
        ),
    ]
    for src_csv, out_name, best_rules in table_specs:
        if not src_csv.is_file():
            continue
        df = pd.read_csv(src_csv)
        _save_table_bundle(df, out_name, paths, save_latex_flag=save_latex_flag, best_rules=best_rules)


def _plot_pareto(tradeoff_df: pd.DataFrame, out_path: Path) -> None:
    work = tradeoff_df[tradeoff_df['available'] == 1].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if work.empty:
        ax.text(0.5, 0.5, 'No ROI-policy tradeoff data available', ha='center', va='center')
        ax.axis('off')
        save_plot(fig, out_path)
        return
    ax.scatter(work['latency_ms_per_fov'], work['mae_um'], s=60, color='#0f766e')
    for _, r in work.iterrows():
        ax.annotate(str(r['roi_policy']), (r['latency_ms_per_fov'], r['mae_um']), fontsize=8, xytext=(4, 4), textcoords='offset points')
    pts = work[['latency_ms_per_fov', 'mae_um', 'roi_policy']].sort_values(['latency_ms_per_fov', 'mae_um']).to_numpy()
    frontier = []
    best_y = np.inf
    for x, y, label in pts:
        if float(y) <= float(best_y):
            frontier.append((float(x), float(y)))
            best_y = float(y)
    if len(frontier) >= 2:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, color='#b45309', linewidth=2, label='Pareto frontier')
        ax.legend(frameon=False)
    ax.set_xlabel('Latency per FOV (ms)')
    ax.set_ylabel('MAE (µm)')
    ax.set_title('ROI Policy Accuracy-Efficiency Tradeoff')
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def _plot_score_correlation(diag_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    work = diag_df.copy()
    work = work[np.isfinite(pd.to_numeric(work.get('selection_score'), errors='coerce'))]
    work = work[np.isfinite(pd.to_numeric(work.get('roi_abs_error_um'), errors='coerce'))]
    if work.empty:
        ax.text(0.5, 0.5, 'No score diagnostics available', ha='center', va='center')
        ax.axis('off')
        save_plot(fig, out_path)
        return
    for policy, g in work.groupby('roi_policy_label'):
        gg = g.head(5000)
        ax.scatter(pd.to_numeric(gg['selection_score']), pd.to_numeric(gg['roi_abs_error_um']), s=8, alpha=0.18, label=str(policy))
    ax.set_xlabel('ROI selection score')
    ax.set_ylabel('ROI absolute autofocus error (µm)')
    ax.set_title('ROI Selection Score vs Downstream ROI Error')
    ax.grid(True, alpha=0.3)
    if work['roi_policy_label'].nunique() <= 8:
        ax.legend(frameon=False, fontsize=8)
    save_plot(fig, out_path)


def _plot_robustness(robust_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    work = robust_df.copy()
    if work.empty:
        ax.text(0.5, 0.5, 'No robustness data available', ha='center', va='center')
        ax.axis('off')
        save_plot(fig, out_path)
        return
    for policy, g in work.groupby('roi_policy'):
        gg = g.head(2000)
        ax.scatter(gg['jaccard'], gg['delta_abs_error_um'], s=14, alpha=0.25, label=str(policy))
    ax.set_xlabel('ROI selection Jaccard stability')
    ax.set_ylabel('Change in absolute FOV error (µm)')
    ax.set_title('ROI Stability vs Downstream Error Sensitivity')
    ax.axhline(0.0, color='black', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    if work['roi_policy'].nunique() <= 8:
        ax.legend(frameon=False, fontsize=8)
    save_plot(fig, out_path)


def _apply_perturbation(img: np.ndarray, perturbation_type: str, level: float) -> np.ndarray:
    import cv2

    out = np.asarray(img, dtype=np.float32).copy()
    if perturbation_type == 'blur':
        sigma = float(level)
        ksize = max(3, int(round(sigma * 3)) * 2 + 1)
        out = cv2.GaussianBlur(out, (ksize, ksize), sigma)
    elif perturbation_type == 'noise':
        rng = np.random.default_rng(42)
        out = out + rng.normal(0.0, float(level), size=out.shape).astype(np.float32)
    elif perturbation_type == 'brightness':
        out = out + float(level)
    elif perturbation_type == 'contrast':
        out = (out - 0.5) * float(level) + 0.5
    return np.clip(out, 0.0, 1.0)


def _prep_I(patch: np.ndarray) -> np.ndarray:
    I = to_grayscale(patch)
    I = percentile_clip(I, p_low=1.0, p_high=99.0)
    I = rescale01(I, eps=1e-6)
    I = zscore(I, eps=1e-6)
    return I.astype(np.float32)


def _patch_to_xa_xb(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    I = _prep_I(patch)
    D1, D2 = compute_dog_lite(I, (0.8, 1.6, 2.4), eps=1e-6)
    EHF = compute_dwt_ehf(I, wavelet='haar', roi_size=200, eps=1e-6)
    XA, XB = assemble_XA_XB(I, D1, D2, EHF, roi_size=200)
    return XA.astype(np.float32), XB.astype(np.float32)


def _select_policy_on_image(fov_df: pd.DataFrame, policy_name: str, ctx: ROIPolicyContext, image: np.ndarray, k: int | None, oracle_error_by_patch: dict[str, float] | None = None) -> list[str]:
    work = fov_df.copy().reset_index(drop=True)
    patch_ids = work['patch_id'].astype(str).tolist()
    if len(work) <= 1 or policy_name in {'all_rois'}:
        return patch_ids
    if policy_name == 'center_top1':
        coords = {pid: rpu.parse_patch_rc(pid) for pid in patch_ids}
        valid = {pid: rc for pid, rc in coords.items() if rc is not None}
        if not valid:
            return [patch_ids[0]]
        max_r = max(v[0] for v in valid.values())
        max_c = max(v[1] for v in valid.values())
        ctr = (max_r / 2.0, max_c / 2.0)
        return [min(valid.keys(), key=lambda pid: ((valid[pid][0] - ctr[0]) ** 2 + (valid[pid][1] - ctr[1]) ** 2, pid))]
    if policy_name == 'random_k':
        rng = rpu.stable_rng(ctx.seed, policy_name, work.iloc[0]['fov_id'])
        kk = min(int(k or 1), len(work))
        idx = rng.permutation(len(work))[:kk]
        return work.iloc[idx]['patch_id'].astype(str).tolist()
    if policy_name == 'oracle_best_single_roi' and oracle_error_by_patch:
        finite = {pid: oracle_error_by_patch.get(pid, np.nan) for pid in patch_ids}
        finite = {k0: v0 for k0, v0 in finite.items() if np.isfinite(v0)}
        if finite:
            return [min(finite.keys(), key=lambda pid: (finite[pid], pid))]
        return [patch_ids[0]]

    grid_h, grid_w = rpu._grid_shape_from_fov(work)
    boxes = rpu._fixed_grid_boxes(image.shape, grid_h, grid_w)
    patches = rpu._extract_grid_patches(image, grid_h, grid_w, roi_size=200)

    if policy_name == 'focus_only_topk':
        scores = {pid: float(rpu.composite_focus_measure(rpu.to_gray(patches[pid]))) for pid in patch_ids}
        kk = min(int(k or 1), len(work))
        return [pid for pid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:kk]]

    predictor, _ = ctx.predictor()
    if predictor.has_map:
        occ = predictor.predict_cellness_map(rpu.normalize_image(image))
        occ_arr = rpu.compute_tile_occupancy_from_map(occ, [boxes[pid] for pid in sorted(boxes.keys())])
    else:
        arr = np.stack([patches[pid] for pid in sorted(patches.keys())], axis=0).astype(np.float32)
        occ_arr = predictor.predict_tile_occupancy(arr)
    occ_scores = {pid: float(val) for pid, val in zip(sorted(boxes.keys()), occ_arr)}

    if policy_name == 'occupancy_only_topk':
        kk = min(int(k or 1), len(work))
        return [pid for pid, _ in sorted(((pid, occ_scores.get(pid, np.nan)) for pid in patch_ids), key=lambda kv: (-kv[1], kv[0]))[:kk]]

    if policy_name in {'hybrid_proposed', 'cnn_adaptive'}:
        focus_scores = {pid: float(rpu.composite_focus_measure(rpu.to_gray(patches[pid]))) for pid in patch_ids}
        row_order = sorted(patch_ids)
        occ_vec = np.asarray([occ_scores.get(pid, np.nan) for pid in row_order], dtype=np.float32)
        focus_vec = np.asarray([focus_scores.get(pid, np.nan) for pid in row_order], dtype=np.float32)
        focus_norm = rpu._robust_focus_norm(focus_vec)
        calib = ctx.cnn_calibration() or {}
        tau_empty = float(calib.get('tau_empty', ctx.tau_empty))
        beta = float(calib.get('beta', ctx.beta))
        valid = occ_vec >= tau_empty
        n_valid = int(np.sum(valid))
        if ctx.use_sum_occ:
            k_est = int(round(beta * float(np.nansum(occ_vec[valid])))) if n_valid else 0
        else:
            k_est = int(round(beta * n_valid)) if n_valid else 0
        kk = max(int(ctx.k_min), min(int(ctx.k_max), int(k_est), int(n_valid))) if n_valid else 0
        if ctx.dense_cap_ratio and len(row_order) > 0:
            kk = min(kk, int(math.ceil(float(ctx.dense_cap_ratio) * len(row_order))))
        if n_valid < int(ctx.k_min) and len(row_order) > 0:
            order = np.argsort(-np.nan_to_num(focus_vec, nan=-np.inf))
            kk = min(max(int(ctx.k_min), 1), len(row_order))
            return [row_order[int(i)] for i in order[:kk].tolist()]
        if kk <= 0:
            return []
        score = (float(ctx.w_occ) * occ_vec) + (float(ctx.w_focus) * focus_norm)
        score[~valid] = -np.inf
        valid_idx = np.where(valid)[0]
        occ_valid = occ_vec[valid]
        order = valid_idx[np.argsort(-occ_valid)]
        cand_count = min(len(order), rpu._candidate_count(kk, len(order), ctx.candidate_multiplier))
        candidate_ids = order[:cand_count].tolist()
        if ctx.focus_on_candidates_only:
            tmp_score = np.full_like(score, -np.inf, dtype=np.float32)
            for idx in candidate_ids:
                tmp_score[int(idx)] = score[int(idx)]
            score_use = tmp_score
        else:
            score_use = score
        selected_idx, _ = rpu.select_topk_tiles_diverse(score_use, grid_h, grid_w, k=kk, d_min=int(ctx.d_min_nondense))
        return [row_order[int(i)] for i in selected_idx]

    if policy_name == 'legacy_adaptive':
        calib = ctx.legacy_calibration()
        if not calib:
            return []
        res = rpu.rin_for_image_rgb_calib(image, calib, beta_override=None)
        gw = int(res.get('grid_w', 0))
        return [rpu.patch_id_from_index(int(idx), gw) for idx in np.asarray(res.get('top_indices', []), dtype=np.int32).tolist()]
    return patch_ids


def _run_robustness_analysis(index_df: pd.DataFrame, ctx: ROIPolicyContext, policy_jobs: list[tuple[str, int | None]], policy_arrays: dict[str, dict[str, np.ndarray]], master_df: pd.DataFrame, args) -> pd.DataFrame:
    multi = index_df.groupby('fov_id').size()
    fov_ids = multi[multi > 1].index.astype(str).tolist()
    if not fov_ids or int(args.robustness_max_fovs) <= 0:
        return pd.DataFrame(columns=['roi_policy', 'perturbation_type', 'perturbation_level', 'jaccard', 'delta_abs_error_um', 'dataset', 'fov_id'])
    rng = np.random.default_rng(int(args.seed))
    take = min(int(args.robustness_max_fovs), len(fov_ids))
    chosen = sorted(rng.choice(np.array(fov_ids, dtype=object), size=take, replace=False).tolist())
    sign_model = _load_trusted_keras_model(require_file(get_paths(args.track).sign_dir / 'models' / 'best_model.keras', 'StageA sign model'))
    plus_model = _load_trusted_keras_model(require_file(get_paths(args.track).reg_dir / 'models' / 'R_plus_best.keras', 'StageB plus model'))
    minus_model = _load_trusted_keras_model(require_file(get_paths(args.track).reg_dir / 'models' / 'R_minus_best.keras', 'StageB minus model'))
    tau = _load_tau(get_paths(args.track))
    gt_map = dict(zip(master_df['fov_id'].astype(str), master_df['gt_signed_um'].astype(float)))
    ds_map = dict(zip(master_df['fov_id'].astype(str), master_df['dataset'].astype(str)))
    rows = []
    for fov_id in chosen:
        g = index_df[index_df['fov_id'] == fov_id].copy().reset_index(drop=True)
        source_path = str(g['source_image_path'].iloc[0])
        if not Path(source_path).is_file():
            continue
        image = rpu.load_image_rgb(source_path)
        gt = float(gt_map.get(fov_id, np.nan))
        for policy_name, k in policy_jobs:
            if policy_name in {'legacy_adaptive', 'oracle_best_single_roi'}:
                continue
            label = policy_label(policy_name, k)
            orig_sel = _select_policy_on_image(g, policy_name, ctx, image, k)
            for pert_type, level in PERTURBATIONS:
                pert = _apply_perturbation(image, pert_type, level)
                pert_sel = _select_policy_on_image(g, policy_name, ctx, pert, k)
                a = set(orig_sel)
                b = set(pert_sel)
                jac = 1.0 if (not a and not b) else (len(a & b) / max(len(a | b), 1))
                if not pert_sel:
                    delta_abs = np.nan
                else:
                    grid_h, grid_w = rpu._grid_shape_from_fov(g)
                    patches = rpu._extract_grid_patches(pert, grid_h, grid_w, roi_size=200)
                    xa_list = []
                    xb_list = []
                    valid_rows = []
                    for pid in pert_sel:
                        if pid not in patches:
                            continue
                        xa, xb = _patch_to_xa_xb(patches[pid])
                        xa_list.append(xa)
                        xb_list.append(xb)
                        valid_rows.append(g[g['patch_id'].astype(str) == pid].iloc[0])
                    if not xa_list:
                        delta_abs = np.nan
                    else:
                        xa_batch = np.stack(xa_list, axis=0)
                        xb_batch = np.stack(xb_list, axis=0)
                        p = sign_model.predict(xa_batch, verbose=0).reshape(-1)
                        c = np.maximum(p, 1.0 - p)
                        imp = np.asarray([float(getattr(r, 'roi_importance', 1.0)) if not isinstance(r, pd.Series) else float(r.get('roi_importance', 1.0)) for r in valid_rows], dtype=float)
                        w = imp * c
                        w[c < tau] = 0.0
                        keep = w > 0
                        if not np.any(keep):
                            dz_hat = np.nan
                        else:
                            vote = (p >= 0.5).astype(float)
                            V = float(np.sum(w * vote))
                            W = float(np.sum(w))
                            pred_sign = 1 if V >= 0.5 * W else 0
                            if pred_sign == 1:
                                mag, _ = plus_model(xb_batch, training=False)
                                signed = tf.squeeze(mag, axis=-1).numpy().astype(float)
                            else:
                                mag, _ = minus_model(xb_batch, training=False)
                                signed = -tf.squeeze(mag, axis=-1).numpy().astype(float)
                            wm = weighted_median(signed[keep], w[keep])
                            dz_hat = float(wm) if wm is not None else np.nan
                        base_pred = policy_arrays[label]['pred_signed_um'][policy_arrays[label]['fov_pos_map'][fov_id]]
                        delta_abs = float(abs(dz_hat - gt) - abs(base_pred - gt)) if np.isfinite(dz_hat) and np.isfinite(base_pred) else np.nan
                rows.append(
                    {
                        'roi_policy': label,
                        'perturbation_type': pert_type,
                        'perturbation_level': float(level),
                        'jaccard': float(jac),
                        'delta_abs_error_um': delta_abs,
                        'dataset': ds_map.get(fov_id, 'unknown'),
                        'fov_id': fov_id,
                    }
                )
    robust_df = pd.DataFrame(rows)
    if not robust_df.empty:
        robust_df['delta_mae_um'] = robust_df['delta_abs_error_um']
    return robust_df


def _generate_failure_panel(source_path: str, selected_patch_ids: list[str], meta: dict[str, Any], out_path: Path) -> None:
    import matplotlib.patches as mpatches

    img = rpu.load_image_rgb(source_path)
    grid_h = int(meta.get('grid_h', 1))
    grid_w = int(meta.get('grid_w', 1))
    boxes = rpu._fixed_grid_boxes(img.shape, grid_h, grid_w)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.clip(img, 0.0, 1.0))
    for pid, box in boxes.items():
        x0, y0, x1, y1 = box
        color = '#9f1239' if pid in selected_patch_ids else '#94a3b8'
        lw = 2.2 if pid in selected_patch_ids else 0.5
        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=lw, alpha=0.9)
        ax.add_patch(rect)
    title = (
        f"{meta['roi_policy_label']} | {meta['dataset']}\n"
        f"pred={meta['pred_signed_um']:.3f} µm, gt={meta['gt_signed_um']:.3f} µm, abs err={meta['abs_error_um']:.3f} µm"
    )
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description='Run fixed-model ROI-ablation-to-regression evaluation suite')
    ap.add_argument('--track', required=True, choices=TRACKS)
    ap.add_argument('--roi-policies', nargs='+', default=DEFAULT_POLICIES)
    ap.add_argument('--k-values', nargs='+', default=['1', '3', '5', '7'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--bins', type=float, default=0.5)
    ap.add_argument('--save-plots', action='store_true')
    ap.add_argument('--save-latex', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--legacy-calibration-csv', default=None)
    ap.add_argument('--cnn-calibration-csv', default=None)
    ap.add_argument('--cellness-predictor', default=None)
    ap.add_argument('--cellness-model-path', default=None)
    ap.add_argument('--predictor-mode', default='auto', choices=['auto', 'map', 'tiles'])
    ap.add_argument('--dummy-threshold', type=float, default=0.5)
    ap.add_argument('--bootstrap-max-fovs', type=int, default=10000)
    ap.add_argument('--pairwise-max-fovs', type=int, default=50000)
    ap.add_argument('--robustness-max-fovs', type=int, default=24)
    ap.add_argument('--failure-topn', type=int, default=10)
    ap.add_argument('--max-fovs', type=int, default=None, help='Optional debug subset of FOVs')
    args = ap.parse_args()

    paths = get_paths(args.track)
    log(paths, 'Starting ROI ablation suite')
    required = [
        paths.eval_dir / 'roi_ablation_to_regression_performance.csv',
        paths.eval_dir / 'roi_ablation_by_dataset.csv',
        paths.eval_dir / 'roi_efficiency_tradeoff.csv',
        paths.eval_dir / 'roi_ablation_confidence_intervals.csv',
        paths.eval_dir / 'roi_policy_pairwise_tests.csv',
        paths.eval_dir / 'roi_robustness_vs_error.csv',
        paths.eval_dir / 'roi_ablation_near_focus.csv',
        paths.eval_dir / 'roi_failure_cases.csv',
        paths.eval_dir / 'roi_score_diagnostics.csv',
        paths.eval_dir / 'domain_gap_reduction.csv',
        paths.eval_dir / 'roi_ablation_by_magnitude_bin.csv',
        paths.eval_dir / 'roi_ablation_by_roi_count_bucket.csv',
        paths.eval_dir / 'reference' / 'tables' / 'Table_ROI_Ablation_Regression.csv',
        paths.eval_dir / 'reference' / 'tables' / 'Table_ROI_Ablation_ByDataset.csv',
        paths.eval_dir / 'reference' / 'tables' / 'Table_ROI_NearFocus.csv',
        paths.eval_dir / 'reference' / 'tables' / 'Table_DomainGapReduction.csv',
        paths.tables_dir / 'Table_ROI_Ablation_Regression.csv',
        paths.tables_dir / 'Table_ROI_Ablation_ByDataset.csv',
        paths.tables_dir / 'Table_ROI_NearFocus.csv',
        paths.tables_dir / 'Table_DomainGapReduction.csv',
    ]
    if args.save_plots:
        required.extend([
            paths.eval_dir / 'roi_efficiency_pareto.png',
            paths.eval_dir / 'roi_robustness_vs_error.png',
            paths.eval_dir / 'roi_score_correlation.png',
        ])
    if args.save_latex:
        required.extend([
            paths.eval_dir / 'reference' / 'tables' / 'Table_ROI_Ablation_Regression.tex',
            paths.eval_dir / 'reference' / 'tables' / 'Table_ROI_Ablation_ByDataset.tex',
            paths.eval_dir / 'reference' / 'tables' / 'Table_ROI_NearFocus.tex',
            paths.eval_dir / 'reference' / 'tables' / 'Table_DomainGapReduction.tex',
            paths.tables_dir / 'Table_ROI_Ablation_Regression.tex',
            paths.tables_dir / 'Table_ROI_Ablation_ByDataset.tex',
            paths.tables_dir / 'Table_ROI_NearFocus.tex',
            paths.tables_dir / 'Table_DomainGapReduction.tex',
        ])
    if args.resume:
        _backfill_report_tables(paths, save_latex_flag=bool(args.save_latex))
    if args.resume and all(p.is_file() for p in required):
        log(paths, f'Resume: ROI ablation outputs already exist; skipping. ({paths.eval_dir})')
        return

    k_values = [int(v) for v in args.k_values]
    policy_jobs = expand_policy_jobs(args.roi_policies, k_values)

    index_df = _load_phase5_index_full(args.track)
    index_df = _subset_index_by_fov(index_df, args.max_fovs, args.seed)
    if index_df.empty:
        raise ValueError('No Phase5 index rows available after subset filtering.')
    counts = index_df.groupby('fov_id').size().rename('num_candidate_rois')
    index_df = index_df.merge(counts, on='fov_id', how='left')

    master_df = _gt_per_fov(index_df).merge(_infer_dataset_for_fov(index_df), on='fov_id', how='left')
    master_df = master_df.sort_values('fov_id').reset_index(drop=True)
    master_df['gt_abs_um'] = np.abs(pd.to_numeric(master_df['gt_signed_um'], errors='coerce'))
    master_df['gt_sign'] = (pd.to_numeric(master_df['gt_signed_um'], errors='coerce') > 0).astype(int)
    fov_pos_map = {str(fid): i for i, fid in enumerate(master_df['fov_id'].astype(str).tolist())}
    tau = _load_tau(paths)

    fixed_model_df, latency_ref = _prepare_fixed_model_outputs(index_df, paths, batch_size=int(args.batch_size), resume=bool(args.resume))
    fixed_model_df['roi_uid'] = fixed_model_df['roi_uid'].astype(str)
    index_df = index_df.merge(fixed_model_df, on='roi_uid', how='left')
    if index_df[['p_sign', 'c_sign', 'mag_plus', 'mag_minus']].isna().any().any():
        raise RuntimeError('Fixed-model ROI output merge failed for some rows.')

    ctx = ROIPolicyContext(
        seed=int(args.seed),
        legacy_calibration_csv=args.legacy_calibration_csv,
        cnn_calibration_csv=args.cnn_calibration_csv,
        cellness_predictor=args.cellness_predictor,
        cellness_model_path=args.cellness_model_path,
        predictor_mode=args.predictor_mode,
        dummy_threshold=float(args.dummy_threshold),
    )

    single_df = index_df[index_df['num_candidate_rois'] == 1].copy().reset_index(drop=True)
    multi_groups = list(index_df[index_df['num_candidate_rois'] > 1].groupby('fov_id', sort=True))

    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    roi_count_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    diag_rows: list[pd.DataFrame] = []
    policy_arrays: dict[str, dict[str, Any]] = {}
    unavailable_rows: list[dict[str, Any]] = []
    qualitative_root = paths.eval_dir / 'qualitative_failure_panels'
    qualitative_root.mkdir(parents=True, exist_ok=True)

    # Oracle errors by patch if needed.
    oracle_maps: dict[str, dict[str, float]] = {}
    if any(p == 'oracle_best_single_roi' for p, _ in policy_jobs):
        tmp = index_df[['fov_id', 'patch_id', 'defocus_um', 'mag_plus', 'mag_minus']].copy()
        tmp['gt_sign'] = (pd.to_numeric(tmp['defocus_um'], errors='coerce') > 0).astype(int)
        tmp['oracle_signed_pred'] = np.where(tmp['gt_sign'] == 1, tmp['mag_plus'], -tmp['mag_minus'])
        tmp['oracle_abs_error_um'] = np.abs(pd.to_numeric(tmp['oracle_signed_pred'], errors='coerce') - pd.to_numeric(tmp['defocus_um'], errors='coerce'))
        for fov_id, g in tmp.groupby('fov_id'):
            oracle_maps[str(fov_id)] = dict(zip(g['patch_id'].astype(str), pd.to_numeric(g['oracle_abs_error_um'], errors='coerce')))

    gt_signed = master_df['gt_signed_um'].to_numpy(dtype=float)
    gt_abs = master_df['gt_abs_um'].to_numpy(dtype=float)
    gt_sign = master_df['gt_sign'].to_numpy(dtype=int)
    dataset_arr = master_df['dataset'].astype(str).to_numpy(dtype=object)

    for policy_name, k in policy_jobs:
        label = policy_label(policy_name, k)
        log(paths, f'Evaluating ROI policy: {label}')

        if policy_name == 'legacy_adaptive' and ctx.legacy_calibration() is None:
            row = {'roi_policy': label, 'available': 0, 'warning': 'legacy calibration unavailable'}
            unavailable_rows.append(row)
            summary_rows.append(
                {
                    'roi_policy': label,
                    'available': 0,
                    'selection_backend': 'unavailable',
                    'K_effective_mean': np.nan,
                    'latency_ms_per_fov': np.nan,
                    'coverage_pct': np.nan,
                    'uncertain_pct': np.nan,
                    'catastrophic_wrong_direction_pct': np.nan,
                    'bias_um': np.nan,
                    'mae_um': np.nan,
                    'rmse_um': np.nan,
                    'median_abs_error_um': np.nan,
                    'p95_abs_error_um': np.nan,
                    'within_0.5um_pct': np.nan,
                    'within_1um_pct': np.nan,
                    'within_2um_pct': np.nan,
                    'stageA_balanced_acc': np.nan,
                    'stageA_auroc': np.nan,
                    'stageB_mae_um': np.nan,
                    'n_fov': int(len(master_df)),
                }
            )
            continue

        arrays = _init_policy_arrays(master_df)
        arrays['fov_pos_map'] = fov_pos_map
        backend_set: set[str] = set()
        selected_parts = []
        multi_candidate_parts = []

        if not single_df.empty:
            single_selected = single_df.copy()
            single_selected['roi_id'] = single_selected['patch_id'].astype(str)
            single_selected['roi_policy'] = policy_name
            single_selected['roi_policy_label'] = label
            single_selected['selected'] = 1
            single_selected['selection_rank'] = 1
            single_selected['selection_score'] = 1.0
            single_selected['selection_backend'] = 'single_roi'
            selected_parts.append(single_selected)
            backend_set.add('single_roi')

        for fov_id, g in multi_groups:
            oracle_map = oracle_maps.get(str(fov_id)) if policy_name == 'oracle_best_single_roi' else None
            result = select_rois_for_policy(g, policy_name=policy_name, ctx=ctx, k=k, oracle_error_by_patch=oracle_map)
            pos = fov_pos_map.get(str(fov_id))
            if pos is not None:
                arrays['selection_time_ms'][pos] = float(result.selection_time_ms)
            if result.backend:
                backend_set.add(str(result.backend))
            if result.available and not result.selected_df.empty:
                selected_parts.append(result.selected_df)
                if not result.candidate_df.empty:
                    multi_candidate_parts.append(result.candidate_df)
            else:
                if pos is not None:
                    arrays['selection_time_ms'][pos] = float(result.selection_time_ms)

        if selected_parts:
            selected_df = pd.concat(selected_parts, ignore_index=True)
        else:
            selected_df = pd.DataFrame(columns=index_df.columns.tolist() + ['roi_policy', 'roi_policy_label'])

        if selected_df.empty:
            summary_rows.append(
                {
                    'roi_policy': label,
                    'available': 0,
                    'selection_backend': ','.join(sorted(backend_set)) if backend_set else 'none',
                    'K_effective_mean': 0.0,
                    'latency_ms_per_fov': np.nan,
                    'coverage_pct': 0.0,
                    'uncertain_pct': 100.0,
                    'catastrophic_wrong_direction_pct': np.nan,
                    'bias_um': np.nan,
                    'mae_um': np.nan,
                    'rmse_um': np.nan,
                    'median_abs_error_um': np.nan,
                    'p95_abs_error_um': np.nan,
                    'within_0.5um_pct': np.nan,
                    'within_1um_pct': np.nan,
                    'within_2um_pct': np.nan,
                    'stageA_balanced_acc': np.nan,
                    'stageA_auroc': np.nan,
                    'stageB_mae_um': np.nan,
                    'n_fov': int(len(master_df)),
                }
            )
            continue

        selected_df['fov_id'] = selected_df['fov_id'].astype(str)
        selected_df['num_candidate_rois'] = pd.to_numeric(selected_df['num_candidate_rois'], errors='coerce').fillna(1).astype(int)
        sel = selected_df.copy().reset_index(drop=True)
        sel['stageA_weight'] = pd.to_numeric(sel['roi_importance'], errors='coerce').fillna(1.0) * pd.to_numeric(sel['c_sign'], errors='coerce').fillna(0.0)
        sel.loc[pd.to_numeric(sel['c_sign'], errors='coerce').fillna(0.0) < float(tau), 'stageA_weight'] = 0.0
        sel['vote_binary'] = (pd.to_numeric(sel['p_sign'], errors='coerce').fillna(0.0) >= 0.5).astype(int)

        single_sel = sel[sel['num_candidate_rois'] == 1].copy()
        if not single_sel.empty:
            for _, row in single_sel.iterrows():
                pos = fov_pos_map[str(row['fov_id'])]
                w = float(row['stageA_weight'])
                arrays['num_selected_rois'][pos] = 1
                arrays['selection_time_ms'][pos] = 0.0
                if w > 0:
                    pred_sign = 1 if float(row['vote_binary']) >= 1 else 0
                    arrays['stageA_pred_sign'][pos] = float(pred_sign)
                    arrays['stageA_score'][pos] = float(row['p_sign'])
                    arrays['vote_margin'][pos] = 1.0
                    arrays['num_stageA_kept'][pos] = 1
                    pred_signed = float(row['mag_plus']) if pred_sign == 1 else -float(row['mag_minus'])
                    arrays['pred_signed_um'][pos] = pred_signed
                    arrays['coverage_mask'][pos] = True
                arrays['aggregation_time_ms'][pos] = 0.0

        multi_sel = sel[sel['num_candidate_rois'] > 1].copy()
        fov_pred_sign_map: dict[str, int | None] = {}
        fov_pred_ok_map: dict[str, bool] = {}
        if not multi_sel.empty:
            for fov_id, g in multi_sel.groupby('fov_id', sort=True):
                pos = fov_pos_map[str(fov_id)]
                arrays['num_selected_rois'][pos] = int(len(g))
                weights = pd.to_numeric(g['stageA_weight'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
                p = pd.to_numeric(g['p_sign'], errors='coerce').fillna(0.5).to_numpy(dtype=float)
                votes = g['vote_binary'].to_numpy(dtype=float)
                kept = weights > 0
                arrays['num_stageA_kept'][pos] = int(np.sum(kept))
                if not np.any(kept):
                    fov_pred_sign_map[str(fov_id)] = None
                    fov_pred_ok_map[str(fov_id)] = False
                    arrays['aggregation_time_ms'][pos] = 0.0
                    continue
                V = float(np.sum(weights * votes))
                W = float(np.sum(weights))
                pred_sign = 1 if V >= 0.5 * W else 0
                arrays['stageA_pred_sign'][pos] = float(pred_sign)
                arrays['stageA_score'][pos] = float(np.sum(weights * p) / W)
                arrays['vote_margin'][pos] = float(abs(V - 0.5 * W) / (0.5 * W + 1e-6))
                fov_pred_sign_map[str(fov_id)] = pred_sign
                if pred_sign == 1:
                    signed = pd.to_numeric(g['mag_plus'], errors='coerce').to_numpy(dtype=float)
                    mag_pred = signed.copy()
                else:
                    signed = -pd.to_numeric(g['mag_minus'], errors='coerce').to_numpy(dtype=float)
                    mag_pred = pd.to_numeric(g['mag_minus'], errors='coerce').to_numpy(dtype=float)
                start_agg = time.perf_counter()
                wm = weighted_median(signed[kept], weights[kept])
                arrays['aggregation_time_ms'][pos] = 1000.0 * (time.perf_counter() - start_agg)
                if wm is None or not np.isfinite(wm):
                    fov_pred_ok_map[str(fov_id)] = False
                else:
                    arrays['pred_signed_um'][pos] = float(wm)
                    arrays['coverage_mask'][pos] = True
                    fov_pred_ok_map[str(fov_id)] = True

        # StageB selected ROI MAE using voted sign route.
        sel['policy_mag_pred'] = np.nan
        for fov_id, g in sel.groupby('fov_id', sort=False):
            pred_sign = arrays['stageA_pred_sign'][fov_pos_map[str(fov_id)]]
            if not np.isfinite(pred_sign):
                continue
            idx = g.index.to_numpy()
            if int(pred_sign) == 1:
                sel.loc[idx, 'policy_mag_pred'] = pd.to_numeric(g['mag_plus'], errors='coerce').to_numpy(dtype=float)
            else:
                sel.loc[idx, 'policy_mag_pred'] = pd.to_numeric(g['mag_minus'], errors='coerce').to_numpy(dtype=float)
        stageB_valid = sel['policy_mag_pred'].notna() & sel['y_mag_um'].notna()
        stageB_mae = float(np.mean(np.abs(pd.to_numeric(sel.loc[stageB_valid, 'policy_mag_pred'], errors='coerce').to_numpy(dtype=float) - pd.to_numeric(sel.loc[stageB_valid, 'y_mag_um'], errors='coerce').to_numpy(dtype=float)))) if stageB_valid.any() else np.nan

        # Candidate diagnostics for scored policies.
        if multi_candidate_parts:
            cand = pd.concat(multi_candidate_parts, ignore_index=True)
            cand['fov_id'] = cand['fov_id'].astype(str)
            pred_sign_map = {fid: arrays['stageA_pred_sign'][fov_pos_map[fid]] for fid in cand['fov_id'].unique().tolist() if fid in fov_pos_map}
            cand['policy_stageA_pred_sign'] = cand['fov_id'].map(pred_sign_map)
            merge_value_cols = [c for c in ['defocus_um', 'mag_plus', 'mag_minus'] if c not in cand.columns]
            if merge_value_cols:
                cand = cand.merge(index_df[['roi_uid', 'fov_id', 'patch_id', *merge_value_cols]], on=['roi_uid', 'fov_id', 'patch_id'], how='left')
            for base_col in ['defocus_um', 'mag_plus', 'mag_minus']:
                if base_col not in cand.columns:
                    for alt_col in [f'{base_col}_x', f'{base_col}_y']:
                        if alt_col in cand.columns:
                            cand[base_col] = cand[alt_col]
                            break
            cand['roi_signed_pred_um'] = np.where(pd.to_numeric(cand['policy_stageA_pred_sign'], errors='coerce') >= 0.5, pd.to_numeric(cand['mag_plus'], errors='coerce'), -pd.to_numeric(cand['mag_minus'], errors='coerce'))
            cand['roi_signed_error_um'] = pd.to_numeric(cand['roi_signed_pred_um'], errors='coerce') - pd.to_numeric(cand['defocus_um'], errors='coerce')
            cand['roi_abs_error_um'] = np.abs(pd.to_numeric(cand['roi_signed_error_um'], errors='coerce'))
            cand['score_rank'] = cand.groupby('fov_id')['selection_score'].rank(method='first', ascending=False)
            diag_rows.append(cand)

        pred_signed = arrays['pred_signed_um'].astype(float)
        coverage = np.isfinite(pred_signed)
        abs_err = np.abs(pred_signed - gt_signed)
        signed_err = pred_signed - gt_signed
        catastrophic = (np.sign(pred_signed) != np.sign(gt_signed)) & np.isfinite(pred_signed) & (~np.isclose(gt_signed, 0.0))
        arrays['abs_error_um'] = abs_err.astype(np.float32)
        arrays['signed_error_um'] = signed_err.astype(np.float32)
        arrays['catastrophic'] = catastrophic.astype(bool)
        arrays['gt_signed_um'] = gt_signed.astype(np.float32)
        arrays['gt_sign'] = gt_sign.astype(np.int8)
        arrays['dataset'] = dataset_arr.copy()
        policy_arrays[label] = arrays

        valid = coverage
        if np.any(valid):
            cls = classification_metrics(gt_sign[np.isfinite(arrays['stageA_pred_sign'])], arrays['stageA_pred_sign'][np.isfinite(arrays['stageA_pred_sign'])].astype(int), arrays['stageA_score'][np.isfinite(arrays['stageA_score'])])
            bias = float(np.mean(signed_err[valid]))
            mae = float(np.mean(abs_err[valid]))
            rmse = float(np.sqrt(np.mean(np.square(signed_err[valid]))))
            medae = float(np.median(abs_err[valid]))
            p95 = float(np.percentile(abs_err[valid], 95))
            within05 = float(np.mean(abs_err[valid] <= 0.5) * 100.0)
            within1 = float(np.mean(abs_err[valid] <= 1.0) * 100.0)
            within2 = float(np.mean(abs_err[valid] <= 2.0) * 100.0)
            cat_pct = float(np.mean(catastrophic[valid]) * 100.0)
        else:
            cls = {'balanced_accuracy': np.nan, 'auroc': np.nan}
            bias = mae = rmse = medae = p95 = within05 = within1 = within2 = cat_pct = np.nan

        mean_selected = float(np.mean(arrays['num_selected_rois']))
        latency_ms = float(
            np.mean(arrays['selection_time_ms']) +
            np.mean(arrays['num_selected_rois']) * float(latency_ref.get('sign_ms_per_roi', np.nan)) +
            np.mean(arrays['num_selected_rois']) * float(latency_ref.get('stageB_ms_per_roi', np.nan)) +
            np.mean(arrays['aggregation_time_ms'])
        )

        summary_rows.append(
            {
                'roi_policy': label,
                'available': 1,
                'selection_backend': ','.join(sorted(backend_set)) if backend_set else 'none',
                'K_effective_mean': mean_selected,
                'latency_ms_per_fov': latency_ms,
                'coverage_pct': float(np.mean(coverage) * 100.0),
                'uncertain_pct': float(np.mean(~coverage) * 100.0),
                'catastrophic_wrong_direction_pct': cat_pct,
                'bias_um': bias,
                'mae_um': mae,
                'rmse_um': rmse,
                'median_abs_error_um': medae,
                'p95_abs_error_um': p95,
                'within_0.5um_pct': within05,
                'within_1um_pct': within1,
                'within_2um_pct': within2,
                'stageA_balanced_acc': cls.get('balanced_accuracy', np.nan),
                'stageA_auroc': cls.get('auroc', np.nan),
                'stageB_mae_um': stageB_mae,
                'n_fov': int(len(master_df)),
            }
        )

        for ds in sorted(set(dataset_arr.tolist())):
            mask_ds = dataset_arr == ds
            valid_ds = valid & mask_ds
            dataset_rows.append(
                {
                    'roi_policy': label,
                    'dataset': ds,
                    'mae_um': float(np.mean(abs_err[valid_ds])) if np.any(valid_ds) else np.nan,
                    'rmse_um': float(np.sqrt(np.mean(np.square(signed_err[valid_ds])))) if np.any(valid_ds) else np.nan,
                    'within_1um_pct': float(np.mean(abs_err[valid_ds] <= 1.0) * 100.0) if np.any(valid_ds) else np.nan,
                    'catastrophic_wrong_direction_pct': float(np.mean(catastrophic[valid_ds]) * 100.0) if np.any(valid_ds) else np.nan,
                    'n_fov': int(np.sum(mask_ds)),
                }
            )

        # Per-magnitude bin and near-focus analysis.
        max_abs = float(np.nanmax(gt_abs)) if len(gt_abs) else float(args.bins)
        edges = np.arange(0.0, max(max_abs, float(args.bins)) + float(args.bins) + 1e-12, float(args.bins))
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask_bin = (gt_abs >= lo) & (gt_abs < hi)
            valid_bin = valid & mask_bin
            bin_rows.append(
                {
                    'roi_policy': label,
                    'bin_lo': float(lo),
                    'bin_hi': float(hi),
                    'mae_um': float(np.mean(abs_err[valid_bin])) if np.any(valid_bin) else np.nan,
                    'rmse_um': float(np.sqrt(np.mean(np.square(signed_err[valid_bin])))) if np.any(valid_bin) else np.nan,
                    'within_1um_pct': float(np.mean(abs_err[valid_bin] <= 1.0) * 100.0) if np.any(valid_bin) else np.nan,
                    'n_fov': int(np.sum(mask_bin)),
                }
            )

        for bucket, lo, hi in ROI_COUNT_BUCKETS:
            mask_bucket = (arrays['num_selected_rois'] >= lo) & (arrays['num_selected_rois'] <= hi if np.isfinite(hi) else True)
            valid_bucket = valid & mask_bucket
            roi_count_rows.append(
                {
                    'roi_policy': label,
                    'roi_count_bucket': bucket,
                    'mae_um': float(np.mean(abs_err[valid_bucket])) if np.any(valid_bucket) else np.nan,
                    'n_fov': int(np.sum(mask_bucket)),
                }
            )

        stageA_valid = np.isfinite(arrays['stageA_pred_sign'])
        for bin_label, lo, hi in NEAR_FOCUS_BINS:
            if np.isinf(hi):
                mask_nf = gt_abs > lo
            elif lo <= 0:
                mask_nf = gt_abs <= hi
            else:
                mask_nf = (gt_abs > lo) & (gt_abs <= hi)
            valid_nf = valid & mask_nf
            stageA_mask = stageA_valid & mask_nf
            bal_acc_nf = classification_metrics(gt_sign[stageA_mask], arrays['stageA_pred_sign'][stageA_mask].astype(int))['balanced_accuracy'] if np.any(stageA_mask) else np.nan
            near_rows.append(
                {
                    'roi_policy': label,
                    'focus_bin': bin_label,
                    'stageA_balanced_acc': bal_acc_nf,
                    'end_to_end_mae_um': float(np.mean(abs_err[valid_nf])) if np.any(valid_nf) else np.nan,
                    'within_0.5um_pct': float(np.mean(abs_err[valid_nf] <= 0.5) * 100.0) if np.any(valid_nf) else np.nan,
                    'within_1um_pct': float(np.mean(abs_err[valid_nf] <= 1.0) * 100.0) if np.any(valid_nf) else np.nan,
                    'n': int(np.sum(mask_nf)),
                }
            )

        policy_ci = _bootstrap_policy_ci(abs_err, signed_err, catastrophic, int(args.bootstrap), int(args.seed), int(args.bootstrap_max_fovs))
        for row in policy_ci:
            row['roi_policy'] = label
        ci_rows.extend(policy_ci)

        # Failure cases.
        if np.any(valid):
            order = np.argsort(np.where(valid, abs_err, -np.inf))[::-1][: int(args.failure_topn)]
            for pos in order:
                if not valid[pos]:
                    continue
                fov_id = str(master_df.iloc[pos]['fov_id'])
                source_path = str(index_df[index_df['fov_id'] == fov_id]['source_image_path'].iloc[0])
                g = index_df[index_df['fov_id'] == fov_id].copy().reset_index(drop=True)
                oracle_map = oracle_maps.get(fov_id) if policy_name == 'oracle_best_single_roi' else None
                result = select_rois_for_policy(g, policy_name=policy_name, ctx=ctx, k=k, oracle_error_by_patch=oracle_map)
                selected_patch_ids = result.selected_df['patch_id'].astype(str).tolist() if not result.selected_df.empty else []
                failure_meta = {
                    'roi_policy': label,
                    'roi_policy_label': label,
                    'fov_id': fov_id,
                    'dataset': str(master_df.iloc[pos]['dataset']),
                    'gt_signed_um': float(gt_signed[pos]),
                    'pred_signed_um': float(pred_signed[pos]),
                    'abs_error_um': float(abs_err[pos]),
                    'grid_h': int(g['patch_id'].astype(str).map(lambda s: rpu.parse_patch_rc(s)[0] if rpu.parse_patch_rc(s) else 0).max() + 1) if not g.empty else 1,
                    'grid_w': int(g['patch_id'].astype(str).map(lambda s: rpu.parse_patch_rc(s)[1] if rpu.parse_patch_rc(s) else 0).max() + 1) if not g.empty else 1,
                }
                panel_path = qualitative_root / sanitize_policy_label(label) / f'{sanitize_policy_label(Path(fov_id).stem)}.png'
                try:
                    _generate_failure_panel(source_path, selected_patch_ids, failure_meta, panel_path)
                    panel_rel = str(panel_path)
                except Exception as exc:
                    panel_rel = ''
                    log(paths, f'[WARN] Failed to build qualitative panel for {label} / {fov_id}: {exc}')
                failure_rows.append(
                    {
                        'roi_policy': label,
                        'fov_id': fov_id,
                        'dataset': str(master_df.iloc[pos]['dataset']),
                        'gt_signed_um': float(gt_signed[pos]),
                        'pred_signed_um': float(pred_signed[pos]),
                        'abs_error_um': float(abs_err[pos]),
                        'qualitative_panel_path': panel_rel,
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(['available', 'mae_um'], ascending=[False, True], na_position='last').reset_index(drop=True)
    dataset_df = pd.DataFrame(dataset_rows)
    bin_df = pd.DataFrame(bin_rows)
    roi_count_df = pd.DataFrame(roi_count_rows)
    near_df = pd.DataFrame(near_rows)
    ci_df = pd.DataFrame(ci_rows)
    fail_df = pd.DataFrame(failure_rows)
    diag_df = pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame(columns=['roi_policy_label'])

    if not dataset_df.empty:
        gap_rows = []
        for policy, g in dataset_df.groupby('roi_policy'):
            vals = pd.to_numeric(g['mae_um'], errors='coerce').dropna().to_numpy(dtype=float)
            if vals.size == 0:
                gap_rows.append({'roi_policy': policy, 'best_dataset_mae': np.nan, 'worst_dataset_mae': np.nan, 'mean_dataset_mae': np.nan, 'gap_worst_minus_best': np.nan})
            else:
                gap_rows.append(
                    {
                        'roi_policy': policy,
                        'best_dataset_mae': float(np.min(vals)),
                        'worst_dataset_mae': float(np.max(vals)),
                        'mean_dataset_mae': float(np.mean(vals)),
                        'gap_worst_minus_best': float(np.max(vals) - np.min(vals)),
                    }
                )
        gap_df = pd.DataFrame(gap_rows)
    else:
        gap_df = pd.DataFrame(columns=['roi_policy', 'best_dataset_mae', 'worst_dataset_mae', 'mean_dataset_mae', 'gap_worst_minus_best'])

    tradeoff_df = summary_df[['roi_policy', 'available', 'K_effective_mean', 'latency_ms_per_fov', 'mae_um', 'rmse_um']].rename(columns={'K_effective_mean': 'avg_num_selected_rois'})

    # Pairwise tests vs proposed.
    proposed_label = None
    for cand in ['hybrid_proposed', 'cnn_adaptive']:
        matching = [lbl for lbl in policy_arrays.keys() if lbl.startswith(cand)]
        if matching:
            proposed_label = matching[0]
            break
    if proposed_label is not None:
        pvals = []
        tmp_rows = []
        ref = policy_arrays[proposed_label]
        for label, arrays in policy_arrays.items():
            base_name = label.split('@', 1)[0]
            if label == proposed_label or base_name not in PRIMARY_COMPARE_BASELINES:
                continue
            mask = np.isfinite(ref['abs_error_um']) & np.isfinite(arrays['abs_error_um'])
            n = int(np.sum(mask))
            if n == 0:
                continue
            idx = np.where(mask)[0]
            if n > int(args.pairwise_max_fovs):
                rng = np.random.default_rng(int(args.seed))
                idx = rng.choice(idx, size=int(args.pairwise_max_fovs), replace=False)
            a = ref['abs_error_um'][idx].astype(float)
            b = arrays['abs_error_um'][idx].astype(float)
            test = wilcoxon_signed_rank(a, b)
            diff = a - b
            tmp_rows.append(
                {
                    'policy_a': proposed_label,
                    'policy_b': label,
                    'metric': 'abs_error_um',
                    'delta_mean': float(np.mean(diff)),
                    'p_value': float(test.get('pvalue', np.nan)),
                    'effect_size': _effect_size_dz(diff),
                    'n_pairs': int(len(idx)),
                }
            )
            pvals.append(float(test.get('pvalue', np.nan)))
        sig = _holm_correction([p if np.isfinite(p) else 1.0 for p in pvals]) if pvals else []
        for row, is_sig in zip(tmp_rows, sig):
            row['significant_after_correction'] = int(is_sig)
            row['correction_method'] = 'holm'
        pairwise_rows.extend(tmp_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    robust_df = _run_robustness_analysis(index_df, ctx, policy_jobs, policy_arrays, master_df, args)

    # Persist outputs.
    summary_csv = paths.eval_dir / 'roi_ablation_to_regression_performance.csv'
    dataset_csv = paths.eval_dir / 'roi_ablation_by_dataset.csv'
    tradeoff_csv = paths.eval_dir / 'roi_efficiency_tradeoff.csv'
    ci_csv = paths.eval_dir / 'roi_ablation_confidence_intervals.csv'
    pairwise_csv = paths.eval_dir / 'roi_policy_pairwise_tests.csv'
    robust_csv = paths.eval_dir / 'roi_robustness_vs_error.csv'
    near_csv = paths.eval_dir / 'roi_ablation_near_focus.csv'
    fail_csv = paths.eval_dir / 'roi_failure_cases.csv'
    diag_csv = paths.eval_dir / 'roi_score_diagnostics.csv'
    gap_csv = paths.eval_dir / 'domain_gap_reduction.csv'
    bin_csv = paths.eval_dir / 'roi_ablation_by_magnitude_bin.csv'
    roi_count_csv = paths.eval_dir / 'roi_ablation_by_roi_count_bucket.csv'

    summary_df.to_csv(summary_csv, index=False)
    dataset_df.to_csv(dataset_csv, index=False)
    tradeoff_df.to_csv(tradeoff_csv, index=False)
    ci_df.to_csv(ci_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)
    robust_df.to_csv(robust_csv, index=False)
    near_df.to_csv(near_csv, index=False)
    fail_df.to_csv(fail_csv, index=False)
    diag_df.to_csv(diag_csv, index=False)
    gap_df.to_csv(gap_csv, index=False)
    bin_df.to_csv(bin_csv, index=False)
    roi_count_df.to_csv(roi_count_csv, index=False)

    if args.save_plots:
        _plot_pareto(tradeoff_df, paths.eval_dir / 'roi_efficiency_pareto.png')
        _plot_robustness(robust_df, paths.eval_dir / 'roi_robustness_vs_error.png')
        _plot_score_correlation(diag_df, paths.eval_dir / 'roi_score_correlation.png')

    _save_table_bundle(
        summary_df,
        'Table_ROI_Ablation_Regression.csv',
        paths,
        save_latex_flag=bool(args.save_latex),
        best_rules={
            'mae_um': 'min',
            'rmse_um': 'min',
            'median_abs_error_um': 'min',
            'p95_abs_error_um': 'min',
            'latency_ms_per_fov': 'min',
            'coverage_pct': 'max',
            'within_0.5um_pct': 'max',
            'within_1um_pct': 'max',
            'within_2um_pct': 'max',
            'stageA_balanced_acc': 'max',
            'stageA_auroc': 'max',
            'stageB_mae_um': 'min',
        },
    )
    _save_table_bundle(dataset_df, 'Table_ROI_Ablation_ByDataset.csv', paths, save_latex_flag=bool(args.save_latex))
    _save_table_bundle(near_df, 'Table_ROI_NearFocus.csv', paths, save_latex_flag=bool(args.save_latex))
    _save_table_bundle(gap_df, 'Table_DomainGapReduction.csv', paths, save_latex_flag=bool(args.save_latex))

    meta = {
        'track': args.track,
        'policy_jobs': [policy_label(p, k) for p, k in policy_jobs],
        'seed': int(args.seed),
        'tau': float(tau),
        'fixed_model_latency': latency_ref,
        'warnings': summarize_context_warnings(ctx),
        'n_fov': int(len(master_df)),
        'n_roi': int(len(index_df)),
    }
    with (paths.eval_dir / 'roi_ablation_metadata.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)

    log(paths, f'ROI ablation suite complete: {summary_csv}')


if __name__ == '__main__':
    main()
