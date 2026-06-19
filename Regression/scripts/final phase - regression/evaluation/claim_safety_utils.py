#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation_utils import (
    DATA_OUT,
    ROOT,
    TABLES_ROOT,
    bootstrap_ci,
    get_paths,
    infer_dataset_for_fov,
    save_plot,
    save_table,
    wilcoxon_signed_rank,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_ROOT = SCRIPT_DIR.parent
BASELINES_ROOT = FINAL_ROOT / 'baselines'

METHOD_ORDER = [
    'proposed_two_stage_roi',
    'direct_single_stage_regression',
    'classical_focus_best',
    'center_crop_inference_only',
    'full_field_tiled_proxy',
]

MAIN_PAPER_PRIORITY = [
    'Table_EndToEnd.csv',
    'Table_FairArchitectureComparison.csv',
    'Table_InputProbeComparison.csv',
    'Table_ROI_Ablation_Regression.csv',
    'Table_ROI_Ablation_ByDataset.csv',
    'Table_DomainGapSummary.csv',
    'Table_SafetySummary.csv',
    'domain_gap_reduction_plot.png',
    'safety_vs_mae_frontier.png',
    'Table_PairedArchitectureComparison.csv',
]

SUPPLEMENT_PRIORITY = [
    'Table_StageA.csv',
    'Table_StageB.csv',
    'Table_ROI_NearFocus.csv',
    'Table_Baseline_ByDataset.csv',
    'Table_Baseline_NearFocus.csv',
    'Table_Baseline_Stats.csv',
    'Table_Runtime.csv',
    'Table_MacroVsWeighted.csv',
    'Table_PaperMainVsSupplementRouting.csv',
]

QUARANTINE_PRIORITY = [
    'Table_Baseline_Comparison.csv',
    'baseline_comparison_results.csv',
    'baseline_by_dataset.csv',
    'baseline_near_focus.csv',
    'baseline_pairwise_tests.csv',
    'baseline_efficiency.csv',
    'Table_Ablation.csv',
    'ablation_results.csv',
]


@dataclass
class ManuscriptPaths:
    track: str
    eval_dir: Path
    tables_dir: Path
    paper_package_dir: Path


def manuscript_paths(track: str) -> ManuscriptPaths:
    paths = get_paths(track)
    return ManuscriptPaths(
        track=track,
        eval_dir=paths.eval_dir,
        tables_dir=paths.tables_dir,
        paper_package_dir=DATA_OUT / track / 'paper_package',
    )


def method_meta(method_name: str) -> dict[str, Any]:
    meta = {
        'proposed_two_stage_roi': {
            'family': 'proposed',
            'evaluation_scope': 'full_pipeline',
            'input_scope': 'content_aware_roi',
            'training_required': 1,
            'scope_label': 'pooled_full_pipeline',
            'notes': 'Existing two-stage ROI-aware autofocus pipeline.',
        },
        'direct_single_stage_regression': {
            'family': 'learned_architecture',
            'evaluation_scope': 'architecture_baseline',
            'input_scope': 'shared_roi_tensors',
            'training_required': 1,
            'scope_label': 'held_out_test_subset',
            'notes': 'Direct single-stage signed-distance regressor on shared ROI tensors.',
        },
        'classical_focus_best': {
            'family': 'classical',
            'evaluation_scope': 'classical_handcrafted_regression',
            'input_scope': 'full_field_tiled_proxy',
            'training_required': 1,
            'scope_label': 'held_out_test_subset',
            'notes': 'Classical focus-measure baseline with handcrafted features and shallow regression.',
        },
        'center_crop_inference_only': {
            'family': 'learned_input',
            'evaluation_scope': 'inference_only',
            'input_scope': 'center_crop_proxy',
            'training_required': 0,
            'scope_label': 'pooled_inference_probe',
            'notes': 'Inference-only input probe using the center-most ROI with fixed downstream models.',
        },
        'full_field_tiled_proxy': {
            'family': 'learned_input',
            'evaluation_scope': 'inference_only_proxy',
            'input_scope': 'full_field_tiled_proxy',
            'training_required': 0,
            'scope_label': 'pooled_proxy_probe',
            'notes': 'Inference-only tiled full-field proxy using fixed downstream models, not a true full-image learner.',
        },
    }
    if method_name not in meta:
        raise KeyError(f'Unknown method metadata requested: {method_name}')
    return meta[method_name].copy()


def read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return path


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None and len(df) > int(max_rows):
        work = df.head(int(max_rows)).copy()
    else:
        work = df.copy()
    if work.empty:
        return '| empty |\n|---|\n| no rows |'
    cols = [str(c) for c in work.columns.tolist()]
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    lines = [header, sep]
    for _, row in work.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append('' if not np.isfinite(v) else f'{v:.6g}')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    if max_rows is not None and len(df) > int(max_rows):
        lines.append(f'\nShowing first {int(max_rows)} of {len(df)} rows.')
    return '\n'.join(lines)


def write_markdown(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')
    return path


def save_eval_and_table(df: pd.DataFrame, eval_path: Path, tables_path: Path | None = None, save_latex_flag: bool = False) -> None:
    save_table(df, eval_path, save_latex=False)
    if tables_path is not None:
        save_table(df, tables_path, save_latex=save_latex_flag)


def gt_per_fov(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if index_df.empty:
        return pd.DataFrame(columns=['fov_id', 'y_true_signed_um'])
    for fov_id, g in index_df.groupby('fov_id', sort=True):
        vals = pd.to_numeric(g['defocus_um'], errors='coerce').to_numpy(dtype=float)
        w = pd.to_numeric(g.get('roi_importance', 1.0), errors='coerce').fillna(1.0).to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if not np.any(finite):
            gt = np.nan
        else:
            vals = vals[finite]
            w = w[finite]
            order = np.argsort(vals)
            vals = vals[order]
            w = w[order]
            csum = np.cumsum(w)
            cutoff = 0.5 * float(np.sum(w))
            gt = float(vals[np.searchsorted(csum, cutoff, side='left')])
        rows.append({'fov_id': str(fov_id), 'y_true_signed_um': gt})
    return pd.DataFrame(rows)


def model_size_mb(paths: str | Path | Sequence[str | Path] | None) -> float:
    if paths is None:
        return float('nan')
    if isinstance(paths, (str, Path)):
        items = [Path(paths)]
    else:
        items = [Path(p) for p in paths]
    total = 0
    any_found = False
    for path in items:
        if path.is_file():
            total += path.stat().st_size
            any_found = True
    if not any_found:
        return float('nan')
    return float(total / (1024.0 * 1024.0))


def holm_correction(pvals: Sequence[float]) -> list[bool]:
    pvals = np.asarray(list(pvals), dtype=float)
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(np.where(np.isfinite(pvals), pvals, np.inf))
    reject = np.zeros((n,), dtype=bool)
    running = True
    for rank, idx in enumerate(order):
        p = pvals[idx]
        threshold = 0.05 / max(n - rank, 1)
        if running and np.isfinite(p) and p <= threshold:
            reject[idx] = True
        else:
            running = False
    return reject.tolist()


def _coerce_standard_columns(df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    work = df.copy()
    if 'fov_id' in work.columns:
        work['fov_id'] = work['fov_id'].astype(str)
    if 'dataset' in work.columns:
        work['dataset'] = work['dataset'].astype(str)
    for col in ['y_true_signed_um', 'y_pred_signed_um', 'abs_error_um', 'signed_error_um', 'runtime_ms_per_fov', 'model_size_mb', 'mean_inputs_used_per_fov', 'num_inputs_used', 'uncertain']:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors='coerce')
    if 'baseline_name' not in work.columns:
        work['baseline_name'] = method_name
    return work


def _load_index_minimal(track: str) -> pd.DataFrame:
    path = DATA_OUT / track / 'regression' / 'index_phase5.csv'
    needed = {'fov_id', 'dataset', 'defocus_um', 'roi_importance'}
    df = pd.read_csv(path, usecols=lambda c: c in needed, low_memory=False)
    if 'roi_importance' not in df.columns:
        df['roi_importance'] = 1.0
    for col in ['fov_id', 'dataset']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ['defocus_um', 'roi_importance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _load_direct_frame(track: str, index_df: pd.DataFrame) -> pd.DataFrame:
    root = DATA_OUT / track / 'baselines' / 'direct_regression'
    fov_path = root / 'inference' / 'fov_predictions.csv'
    eval_path = root / 'metrics' / 'eval.json'
    model_path = root / 'models' / 'best_model.keras'
    if not fov_path.is_file():
        raise FileNotFoundError(fov_path)
    keep = {
        'fov_id', 'dataset', 'y_pred_signed_um', 'pred_mean_um', 'runtime_ms_per_fov',
        'num_inputs_used', 'signed_error_um', 'abs_error_um', 'y_true_signed_um',
        'status', 'mean_inputs_used_per_fov', 'model_size_mb',
    }
    raw = pd.read_csv(fov_path, usecols=lambda c: c in keep, low_memory=False)
    gt_df = gt_per_fov(index_df)
    ds_df = infer_dataset_for_fov(index_df)
    work = raw.copy()
    work['fov_id'] = work['fov_id'].astype(str)
    if 'runtime_ms_per_fov' not in work.columns:
        runtime_ms_per_roi = np.nan
        if eval_path.is_file():
            runtime_ms_per_roi = float(read_json(eval_path).get('runtime_ms_per_roi', np.nan))
        work['runtime_ms_per_fov'] = pd.to_numeric(work.get('num_inputs_used', np.nan), errors='coerce') * float(runtime_ms_per_roi)
    if 'y_pred_signed_um' not in work.columns:
        work['y_pred_signed_um'] = pd.to_numeric(work.get('pred_mean_um', np.nan), errors='coerce')
    work = work.drop(columns=['dataset'], errors='ignore').merge(ds_df, on='fov_id', how='left')
    if 'y_true_signed_um' not in work.columns:
        work = work.merge(gt_df, on='fov_id', how='left')
    else:
        work['y_true_signed_um'] = pd.to_numeric(work['y_true_signed_um'], errors='coerce')
    work['signed_error_um'] = pd.to_numeric(work.get('signed_error_um', np.nan), errors='coerce')
    mask = work['signed_error_um'].isna()
    work.loc[mask, 'signed_error_um'] = pd.to_numeric(work.loc[mask, 'y_pred_signed_um'], errors='coerce') - pd.to_numeric(work.loc[mask, 'y_true_signed_um'], errors='coerce')
    work['abs_error_um'] = pd.to_numeric(work.get('abs_error_um', np.nan), errors='coerce')
    mask = work['abs_error_um'].isna()
    work.loc[mask, 'abs_error_um'] = work.loc[mask, 'signed_error_um'].abs()
    if 'uncertain' not in work.columns:
        if 'status' in work.columns:
            work['uncertain'] = (work['status'].astype(str) != 'ok').astype(int)
        else:
            work['uncertain'] = work['y_pred_signed_um'].isna().astype(int)
    work['mean_inputs_used_per_fov'] = pd.to_numeric(work.get('num_inputs_used', np.nan), errors='coerce')
    work['model_size_mb'] = model_size_mb(model_path)
    meta = method_meta('direct_single_stage_regression')
    work['family'] = meta['family']
    work['evaluation_scope'] = meta['evaluation_scope']
    work['input_scope'] = meta['input_scope']
    work['training_required'] = meta['training_required']
    work['available'] = 1
    work['notes'] = meta['notes']
    work['baseline_name'] = 'direct_single_stage_regression'
    return _coerce_standard_columns(work, 'direct_single_stage_regression')


def _load_proposed_frame(track: str, index_df: pd.DataFrame) -> pd.DataFrame:
    paths = get_paths(track)
    pred_path = DATA_OUT / track / 'regression' / 'inference' / 'fov_aggregate_predictions.csv'
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    keep = {'fov_id', 'dz_hat_um', 'runtime_ms', 'num_rois_used', 'status'}
    preds = pd.read_csv(pred_path, usecols=lambda c: c in keep, low_memory=False)
    gt_df = gt_per_fov(index_df)
    ds_df = infer_dataset_for_fov(index_df)
    work = pd.DataFrame({
        'fov_id': preds['fov_id'].astype(str),
        'y_pred_signed_um': pd.to_numeric(preds.get('dz_hat_um', np.nan), errors='coerce'),
        'uncertain': (preds.get('status', pd.Series(['ok'] * len(preds))).astype(str) != 'ok').astype(int),
        'runtime_ms_per_fov': pd.to_numeric(preds.get('runtime_ms', np.nan), errors='coerce'),
        'num_inputs_used': pd.to_numeric(preds.get('num_rois_used', np.nan), errors='coerce'),
    })
    work = work.merge(gt_df, on='fov_id', how='left').merge(ds_df, on='fov_id', how='left')
    work['signed_error_um'] = pd.to_numeric(work['y_pred_signed_um'], errors='coerce') - pd.to_numeric(work['y_true_signed_um'], errors='coerce')
    work['abs_error_um'] = work['signed_error_um'].abs()
    work['mean_inputs_used_per_fov'] = pd.to_numeric(work.get('num_inputs_used', np.nan), errors='coerce')
    work['model_size_mb'] = model_size_mb([
        paths.sign_dir / 'models' / 'best_model.keras',
        paths.reg_dir / 'models' / 'R_plus_best.keras',
        paths.reg_dir / 'models' / 'R_minus_best.keras',
    ])
    meta = method_meta('proposed_two_stage_roi')
    work['family'] = meta['family']
    work['evaluation_scope'] = meta['evaluation_scope']
    work['input_scope'] = meta['input_scope']
    work['training_required'] = meta['training_required']
    work['available'] = 1
    work['notes'] = meta['notes']
    work['baseline_name'] = 'proposed_two_stage_roi'
    return _coerce_standard_columns(work, 'proposed_two_stage_roi')


def _load_existing_standard_frame(track: str, method_name: str) -> pd.DataFrame:
    root = DATA_OUT / track / 'baselines' / method_name
    fov_path = root / 'inference' / 'fov_predictions.csv'
    if not fov_path.is_file():
        raise FileNotFoundError(fov_path)
    keep = {
        'fov_id', 'dataset', 'y_pred_signed_um', 'pred_mean_um', 'runtime_ms_per_fov', 'runtime_ms',
        'num_inputs_used', 'signed_error_um', 'abs_error_um', 'y_true_signed_um', 'status',
        'mean_inputs_used_per_fov', 'model_size_mb', 'family', 'evaluation_scope',
        'input_scope', 'training_required', 'available', 'notes', 'baseline_name',
    }
    work = pd.read_csv(fov_path, usecols=lambda c: c in keep, low_memory=False)
    meta = method_meta(method_name)
    if 'family' not in work.columns:
        work['family'] = meta['family']
    if 'evaluation_scope' not in work.columns:
        work['evaluation_scope'] = meta['evaluation_scope']
    if 'input_scope' not in work.columns:
        work['input_scope'] = meta['input_scope']
    if 'training_required' not in work.columns:
        work['training_required'] = meta['training_required']
    if 'available' not in work.columns:
        work['available'] = 1
    if 'notes' not in work.columns:
        work['notes'] = meta['notes']
    if 'baseline_name' not in work.columns:
        work['baseline_name'] = method_name
    if 'mean_inputs_used_per_fov' not in work.columns and 'num_inputs_used' in work.columns:
        work['mean_inputs_used_per_fov'] = pd.to_numeric(work['num_inputs_used'], errors='coerce')
    return _coerce_standard_columns(work, method_name)


def load_method_frames(track: str, methods: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    index_df = _load_index_minimal(track)
    frames: dict[str, pd.DataFrame] = {}
    wanted = list(methods) if methods else list(METHOD_ORDER)
    for method in wanted:
        try:
            if method == 'proposed_two_stage_roi':
                frames[method] = _load_proposed_frame(track, index_df)
            elif method == 'direct_single_stage_regression':
                frames[method] = _load_direct_frame(track, index_df)
            elif method in {'classical_focus_best', 'center_crop_inference_only', 'full_field_tiled_proxy'}:
                frames[method] = _load_existing_standard_frame(track, method)
        except FileNotFoundError:
            continue
    return frames


def load_core_outputs(
    track: str,
    *,
    include_large: bool = False,
    json_names: Sequence[str] | None = None,
    csv_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    mp = manuscript_paths(track)
    out: dict[str, Any] = {'paths': mp}
    json_files = list(json_names) if json_names is not None else [
        'stageA_metrics.json',
        'stageB_metrics.json',
        'end_to_end_metrics.json',
        'runtime_metrics.json',
        'statistical_tests_results.json',
    ]
    csv_files = list(csv_names) if csv_names is not None else [
        'cross_dataset_results.csv',
        'ablation_results.csv',
        'confidence_intervals.csv',
        'roi_ablation_to_regression_performance.csv',
        'roi_ablation_by_dataset.csv',
        'roi_ablation_near_focus.csv',
        'roi_policy_pairwise_tests.csv',
        'domain_gap_reduction.csv',
        'baseline_comparison_results.csv',
        'baseline_by_dataset.csv',
        'baseline_near_focus.csv',
        'baseline_pairwise_tests.csv',
        'baseline_efficiency.csv',
        'baseline_failure_cases.csv',
        'baseline_qualitative_panel.csv',
    ]
    if include_large:
        csv_files.extend([
            'roi_robustness_vs_error.csv',
            'roi_score_diagnostics.csv',
        ])
    for name in json_files:
        p = mp.eval_dir / name
        if p.is_file():
            out[name] = read_json(p)
    for name in csv_files:
        p = mp.eval_dir / name
        if p.is_file():
            out[name] = pd.read_csv(p, low_memory=False)
    return out


def largest_dataset_share_pct(fov_df: pd.DataFrame) -> float:
    if fov_df.empty or 'dataset' not in fov_df.columns:
        return float('nan')
    counts = fov_df['dataset'].astype(str).value_counts(dropna=False)
    if counts.empty:
        return float('nan')
    return float(counts.iloc[0] / max(float(counts.sum()), 1.0) * 100.0)


def _catastrophic_rate(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(gt) & (~np.isclose(gt, 0.0))
    if not np.any(mask):
        return float('nan')
    cat = (np.sign(pred[mask]) != np.sign(gt[mask]))
    return float(np.mean(cat) * 100.0)


def compute_frame_metrics(fov_df: pd.DataFrame) -> dict[str, float]:
    pred = pd.to_numeric(fov_df.get('y_pred_signed_um', np.nan), errors='coerce').to_numpy(dtype=float)
    gt = pd.to_numeric(fov_df.get('y_true_signed_um', np.nan), errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(pred) & np.isfinite(gt)
    if not np.any(valid):
        return {
            'n_fov': int(len(fov_df)),
            'mae_um': np.nan,
            'rmse_um': np.nan,
            'bias_um': np.nan,
            'median_abs_error_um': np.nan,
            'p95_abs_error_um': np.nan,
            'within_1um_pct': np.nan,
            'within_0p5um_pct': np.nan,
            'within_2um_pct': np.nan,
            'catastrophic_wrong_direction_pct': np.nan,
            'uncertain_pct': np.nan,
        }
    err = pred[valid] - gt[valid]
    abs_err = np.abs(err)
    if 'uncertain' in fov_df.columns:
        uncertain = pd.to_numeric(fov_df['uncertain'], errors='coerce').fillna(0).to_numpy(dtype=float)
    else:
        uncertain = np.zeros((len(fov_df),), dtype=float)
    return {
        'n_fov': int(np.sum(valid)),
        'mae_um': float(np.mean(abs_err)),
        'rmse_um': float(np.sqrt(np.mean(np.square(err)))),
        'bias_um': float(np.mean(err)),
        'median_abs_error_um': float(np.median(abs_err)),
        'p95_abs_error_um': float(np.percentile(abs_err, 95)),
        'within_0p5um_pct': float(np.mean(abs_err <= 0.5) * 100.0),
        'within_1um_pct': float(np.mean(abs_err <= 1.0) * 100.0),
        'within_2um_pct': float(np.mean(abs_err <= 2.0) * 100.0),
        'catastrophic_wrong_direction_pct': _catastrophic_rate(pred[valid], gt[valid]),
        'uncertain_pct': float(np.mean(uncertain > 0) * 100.0),
    }


def compute_dataset_metrics(fov_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if fov_df.empty or 'dataset' not in fov_df.columns:
        return pd.DataFrame(columns=['dataset', 'n_fov', 'mae_um', 'rmse_um', 'within_1um_pct', 'catastrophic_wrong_direction_pct'])
    for dataset, g in fov_df.groupby('dataset', sort=True):
        m = compute_frame_metrics(g)
        rows.append({
            'dataset': str(dataset),
            'n_fov': int(m['n_fov']),
            'mae_um': m['mae_um'],
            'rmse_um': m['rmse_um'],
            'within_1um_pct': m['within_1um_pct'],
            'catastrophic_wrong_direction_pct': m['catastrophic_wrong_direction_pct'],
        })
    return pd.DataFrame(rows)


def macro_weighted_summary(fov_df: pd.DataFrame) -> dict[str, float]:
    ds = compute_dataset_metrics(fov_df)
    if ds.empty:
        return {'weighted_mae_um': np.nan, 'macro_dataset_mae_um': np.nan, 'domain_gap_mae': np.nan, 'largest_dataset_share_pct': np.nan}
    mae = pd.to_numeric(ds['mae_um'], errors='coerce').to_numpy(dtype=float)
    w = pd.to_numeric(ds['n_fov'], errors='coerce').fillna(0).to_numpy(dtype=float)
    return {
        'weighted_mae_um': float(np.nansum(mae * w) / max(float(np.nansum(w)), 1.0)),
        'macro_dataset_mae_um': float(np.nanmean(mae)),
        'domain_gap_mae': float(np.nanmax(mae) - np.nanmin(mae)),
        'largest_dataset_share_pct': largest_dataset_share_pct(fov_df),
    }


def compute_safety_fields(fov_df: pd.DataFrame, near_threshold: float = 1.0, far_threshold: float = 2.0) -> dict[str, float]:
    pred = pd.to_numeric(fov_df.get('y_pred_signed_um', np.nan), errors='coerce').to_numpy(dtype=float)
    gt = pd.to_numeric(fov_df.get('y_true_signed_um', np.nan), errors='coerce').to_numpy(dtype=float)
    abs_gt = np.abs(gt)
    near_mask = np.isfinite(abs_gt) & (abs_gt <= float(near_threshold))
    far_mask = np.isfinite(abs_gt) & (abs_gt > float(far_threshold))
    return {
        'near_focus_wrong_direction_pct': _catastrophic_rate(pred[near_mask], gt[near_mask]),
        'far_focus_wrong_direction_pct': _catastrophic_rate(pred[far_mask], gt[far_mask]),
    }


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_bootstrap: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return float('nan'), float('nan'), float('nan')
    diff = a - b
    point = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    stats = np.empty((int(n_bootstrap),), dtype=np.float64)
    idx = np.arange(len(diff))
    for i in range(int(n_bootstrap)):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        stats[i] = float(np.mean(diff[sample_idx]))
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def load_wide_prediction_frame(track: str, methods: Sequence[str] | None = None) -> pd.DataFrame:
    index_df = _load_index_minimal(track)
    wide = gt_per_fov(index_df).merge(infer_dataset_for_fov(index_df), on='fov_id', how='left')
    wide['fov_id'] = wide['fov_id'].astype(str)
    wide['y_true_signed_um'] = pd.to_numeric(wide['y_true_signed_um'], errors='coerce')
    frames = load_method_frames(track, methods)
    for method, frame in frames.items():
        pred = frame[['fov_id', 'y_pred_signed_um']].copy()
        pred['fov_id'] = pred['fov_id'].astype(str)
        pred = pred.rename(columns={'y_pred_signed_um': method})
        wide = wide.merge(pred, on='fov_id', how='left')
        wide[method] = pd.to_numeric(wide[method], errors='coerce')
    return wide


def _fair_summary_text(a_name: str, b_name: str, delta_mae_um: float, delta_wrong_direction_pct: float) -> str:
    if np.isnan(delta_mae_um) or np.isnan(delta_wrong_direction_pct):
        return 'Insufficient shared support for a fair claim.'
    mae_eps = 1e-6
    wrong_eps = 1e-6
    if delta_mae_um < -mae_eps and delta_wrong_direction_pct < -wrong_eps:
        return f'{a_name} is better on both MAE and wrong-direction risk on the shared subset.'
    if delta_mae_um > mae_eps and delta_wrong_direction_pct > wrong_eps:
        return f'{b_name} is better on both MAE and wrong-direction risk on the shared subset.'
    if delta_mae_um > mae_eps and delta_wrong_direction_pct < -wrong_eps:
        return f'Mixed tradeoff: {b_name} has lower MAE, while {a_name} has lower wrong-direction risk on the shared subset.'
    if delta_mae_um < -mae_eps and delta_wrong_direction_pct > wrong_eps:
        return f'Mixed tradeoff: {a_name} has lower MAE, while {b_name} has lower wrong-direction risk on the shared subset.'
    return 'Methods are effectively tied on the shared subset within the observed metrics.'


def paired_comparison_rows(track: str, pair_specs: Sequence[tuple[str, str]], bootstrap: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = sorted({m for pair in pair_specs for m in pair})
    wide = load_wide_prediction_frame(track, methods)
    pred_rows = []
    test_rows = []
    pvals = []
    for method_a, method_b in pair_specs:
        cols = ['fov_id', 'dataset', 'y_true_signed_um', method_a, method_b]
        sub = wide[cols].copy()
        sub = sub.dropna(subset=['y_true_signed_um', method_a, method_b]).reset_index(drop=True)
        if sub.empty:
            continue
        sub['method_a'] = method_a
        sub['method_b'] = method_b
        sub['abs_error_a_um'] = (sub[method_a] - sub['y_true_signed_um']).abs()
        sub['abs_error_b_um'] = (sub[method_b] - sub['y_true_signed_um']).abs()
        sub['signed_error_a_um'] = sub[method_a] - sub['y_true_signed_um']
        sub['signed_error_b_um'] = sub[method_b] - sub['y_true_signed_um']
        sub['wrong_direction_a'] = ((sub[method_a] * sub['y_true_signed_um']) < 0) & (~np.isclose(sub['y_true_signed_um'], 0.0))
        sub['wrong_direction_b'] = ((sub[method_b] * sub['y_true_signed_um']) < 0) & (~np.isclose(sub['y_true_signed_um'], 0.0))
        pred_rows.append(sub)

        a_abs = sub['abs_error_a_um'].to_numpy(dtype=float)
        b_abs = sub['abs_error_b_um'].to_numpy(dtype=float)
        a_wrong = float(np.mean(sub['wrong_direction_a'].to_numpy(dtype=bool)) * 100.0)
        b_wrong = float(np.mean(sub['wrong_direction_b'].to_numpy(dtype=bool)) * 100.0)
        wil = wilcoxon_signed_rank(a_abs, b_abs)
        delta, lo, hi = paired_bootstrap_delta(a_abs, b_abs, n_bootstrap=bootstrap, seed=seed)
        row = {
            'method_a': method_a,
            'method_b': method_b,
            'shared_support_n': int(len(sub)),
            'baseline_value': float(np.mean(a_abs)),
            'comparison_value': float(np.mean(b_abs)),
            'delta_mae_um': delta,
            'delta_wrong_direction_pct': float(a_wrong - b_wrong),
            'ci_low': lo,
            'ci_high': hi,
            'p_value': float(wil.get('pvalue', np.nan)),
            'significant_after_correction': 0,
            'fair_claim_summary': _fair_summary_text(method_a, method_b, delta, float(a_wrong - b_wrong)),
        }
        test_rows.append(row)
        pvals.append(row['p_value'])
    corrected = holm_correction(pvals)
    for row, sig in zip(test_rows, corrected):
        row['significant_after_correction'] = int(sig)
    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    test_df = pd.DataFrame(test_rows)
    return pred_df, test_df


def fair_method_rows(track: str, methods: Sequence[str], required_common: bool = True) -> tuple[pd.DataFrame, int]:
    wide = load_wide_prediction_frame(track, methods)
    subset = wide[['fov_id', 'dataset', 'y_true_signed_um', *methods]].copy()
    if required_common:
        subset = subset.dropna(subset=['y_true_signed_um', *methods]).reset_index(drop=True)
    rows = []
    for method in methods:
        cur = subset[['fov_id', 'dataset', 'y_true_signed_um', method]].dropna().rename(columns={method: 'y_pred_signed_um'})
        cur['y_true_signed_um'] = pd.to_numeric(cur['y_true_signed_um'], errors='coerce')
        cur['y_pred_signed_um'] = pd.to_numeric(cur['y_pred_signed_um'], errors='coerce')
        cur['signed_error_um'] = cur['y_pred_signed_um'] - cur['y_true_signed_um']
        cur['abs_error_um'] = cur['signed_error_um'].abs()
        metrics = compute_frame_metrics(cur)
        meta = method_meta(method)
        rows.append({
            'method': method,
            'scope_label': 'paired_shared_subset' if required_common else meta['scope_label'],
            'n_fov': int(metrics['n_fov']),
            'mae_um': metrics['mae_um'],
            'rmse_um': metrics['rmse_um'],
            'catastrophic_wrong_direction_pct': metrics['catastrophic_wrong_direction_pct'],
            'bias_um': metrics['bias_um'],
            'within_1um_pct': metrics['within_1um_pct'],
            'note': meta['notes'],
        })
    return pd.DataFrame(rows), int(len(subset))


def source_asset_inventory(track: str) -> list[dict[str, Any]]:
    mp = manuscript_paths(track)
    eval_dir = mp.eval_dir
    tables_dir = mp.tables_dir
    items = []
    for path in [
        eval_dir / 'stageA_metrics.json',
        eval_dir / 'stageB_metrics.json',
        eval_dir / 'end_to_end_metrics.json',
        eval_dir / 'runtime_metrics.json',
        eval_dir / 'cross_dataset_results.csv',
        eval_dir / 'ablation_results.csv',
        eval_dir / 'confidence_intervals.csv',
        eval_dir / 'roi_ablation_to_regression_performance.csv',
        eval_dir / 'roi_ablation_by_dataset.csv',
        eval_dir / 'roi_ablation_near_focus.csv',
        eval_dir / 'roi_policy_pairwise_tests.csv',
        eval_dir / 'domain_gap_reduction.csv',
        eval_dir / 'baseline_comparison_results.csv',
        eval_dir / 'baseline_by_dataset.csv',
        eval_dir / 'baseline_near_focus.csv',
        eval_dir / 'baseline_pairwise_tests.csv',
        eval_dir / 'baseline_efficiency.csv',
        eval_dir / 'baseline_efficiency_frontier.png',
        eval_dir / 'baseline_dataset_mae.png',
        eval_dir / 'baseline_near_focus.png',
        eval_dir / 'claim_evidence_matrix.csv',
        eval_dir / 'claim_guardrails.md',
        eval_dir / 'output_audit_report.csv',
        eval_dir / 'output_audit_report.md',
        eval_dir / 'manuscript_whitelist.json',
        eval_dir / 'manuscript_blacklist.json',
        eval_dir / 'fair_architecture_comparison.csv',
        eval_dir / 'input_probe_comparison.csv',
        eval_dir / 'macro_vs_weighted.csv',
        eval_dir / 'paper_main_vs_supplement_routing.csv',
        eval_dir / 'domain_gap_summary.csv',
        eval_dir / 'domain_gap_reduction_plot.png',
        eval_dir / 'domain_gap_waterfall.png',
        eval_dir / 'safety_summary.csv',
        eval_dir / 'safety_vs_mae_frontier.png',
        eval_dir / 'paired_subset_predictions.csv',
        eval_dir / 'paired_subset_metrics.csv',
        eval_dir / 'paired_subset_tests.csv',
        eval_dir / 'paired_architecture_comparison.csv',
        eval_dir / 'asset_curation.csv',
        eval_dir / 'asset_curation.md',
        eval_dir / 'results_writing_support.json',
        eval_dir / 'results_writing_support.md',
        eval_dir / 'discussion_guardrails.md',
        eval_dir / 'abstract_guardrails.md',
        eval_dir / 'caption_bank.csv',
        eval_dir / 'caption_bank.md',
        eval_dir / 'pipeline_diagram_description.md',
        eval_dir / 'script_runbook.csv',
        eval_dir / 'script_runbook.md',
        tables_dir / 'Table_StageA.csv',
        tables_dir / 'Table_StageB.csv',
        tables_dir / 'Table_EndToEnd.csv',
        tables_dir / 'Table_Runtime.csv',
        tables_dir / 'Table_Ablation.csv',
        tables_dir / 'Table_Baseline_Comparison.csv',
        tables_dir / 'Table_Baseline_ByDataset.csv',
        tables_dir / 'Table_Baseline_NearFocus.csv',
        tables_dir / 'Table_Baseline_Stats.csv',
        tables_dir / 'Table_Baseline_Efficiency.csv',
        tables_dir / 'Table_ROI_Ablation_Regression.csv',
        tables_dir / 'Table_ROI_Ablation_ByDataset.csv',
        tables_dir / 'Table_ROI_NearFocus.csv',
        tables_dir / 'Table_DomainGapReduction.csv',
        tables_dir / 'Table_ClaimEvidence.csv',
        tables_dir / 'Table_FairArchitectureComparison.csv',
        tables_dir / 'Table_InputProbeComparison.csv',
        tables_dir / 'Table_MacroVsWeighted.csv',
        tables_dir / 'Table_PaperMainVsSupplementRouting.csv',
        tables_dir / 'Table_DomainGapSummary.csv',
        tables_dir / 'Table_SafetySummary.csv',
        tables_dir / 'Table_PairedArchitectureComparison.csv',
        tables_dir / 'Table_AssetCuration.csv',
    ]:
        items.append({
            'asset_name': path.name,
            'path': str(path),
            'exists': int(path.is_file()),
            'suffix': path.suffix.lower(),
            'kind': 'figure' if path.suffix.lower() in {'.png', '.jpg', '.jpeg'} else 'table_or_data',
        })
    return items


def stale_paper_package_paths(track: str) -> list[Path]:
    package_root = DATA_OUT / track / 'paper_package'
    if not package_root.exists():
        return []
    return sorted([p for p in package_root.rglob('*') if p.is_file()])


def simple_line_plot(x: Sequence[float], y: Sequence[float], out_path: Path, title: str, xlabel: str, ylabel: str, labels: Sequence[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker='o')
    if labels is not None:
        for xx, yy, label in zip(x, y, labels):
            ax.annotate(str(label), (xx, yy), xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)
