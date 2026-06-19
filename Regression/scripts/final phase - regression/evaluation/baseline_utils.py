#!/usr/bin/env python3
from __future__ import annotations

import abc
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation_utils import (
    TRACKS,
    classification_metrics,
    get_paths,
    infer_dataset_for_fov,
    log,
    require_file,
    save_json,
    save_plot,
    save_table,
    weighted_median,
    wilcoxon_signed_rank,
)

HERE = Path(__file__).resolve().parent
FINAL_ROOT = HERE.parent
PHASE4_DIR = FINAL_ROOT / 'phase4_sign'
PHASE5_DIR = FINAL_ROOT / 'phase5_regression'
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module {name} from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PHASE5_UTILS = _load_module('baseline_phase5_utils', PHASE5_DIR / 'utils.py')
PHASE4_UTILS = _load_module('baseline_phase4_utils', PHASE4_DIR / 'utils.py')
ROI_ABLATION = _load_module('baseline_roi_ablation_helpers', HERE / 'run_roi_ablation_suite.py')
ROI_POLICY = _load_module('baseline_roi_policy_helpers', HERE / 'roi_policy_utils.py')

load_phase5_index_raw = PHASE5_UTILS.load_phase5_index
phase5_group_split = PHASE5_UTILS.group_split
load_XB_tensor = PHASE5_UTILS.load_XB_tensor
load_XA_tensor = PHASE5_UTILS.load_XA_tensor
load_phase4_xa_tensor = PHASE4_UTILS.load_XA_tensor

ROIPolicyContext = ROI_POLICY.ROIPolicyContext
select_rois_for_policy = ROI_POLICY.select_rois_for_policy

load_tau = ROI_ABLATION._load_tau
prepare_fixed_model_outputs = ROI_ABLATION._prepare_fixed_model_outputs

BASELINE_FAMILIES = {'proposed', 'learned_architecture', 'learned_input', 'classical'}
DEFAULT_BASELINES = [
    'proposed_two_stage_roi',
    'direct_single_stage_regression',
    'classical_focus_best',
    'center_crop_inference_only',
    'full_field_tiled_proxy',
]
NEAR_FOCUS_BINS = [
    ('|dz|<=0.5', 0.0, 0.5),
    ('0.5<|dz|<=1', 0.5, 1.0),
    ('1<|dz|<=2', 1.0, 2.0),
    ('>2', 2.0, np.inf),
]


@dataclass
class BaselinePredictionBundle:
    baseline_name: str
    family: str
    evaluation_scope: str
    input_scope: str
    training_required: int
    available: int
    fov_df: pd.DataFrame
    roi_df: pd.DataFrame | None = None
    metrics: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    notes: str = ''


class BaselineMethod(abc.ABC):
    baseline_name: str
    baseline_family: str
    input_scope: str
    evaluation_scope: str
    training_required: bool
    available: bool

    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def predict_fov(self, *args, **kwargs) -> BaselinePredictionBundle:
        raise NotImplementedError

    @abc.abstractmethod
    def runtime_profile(self, *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        raise NotImplementedError


def baseline_root(track: str) -> Path:
    return get_paths(track).sign_dir.parent / 'baselines'


def baseline_method_root(track: str, method_name: str) -> Path:
    root = baseline_root(track) / method_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_baseline_tree(root: Path) -> dict[str, Path]:
    paths = {
        'root': root,
        'models': root / 'models',
        'metrics': root / 'metrics',
        'inference': root / 'inference',
        'splits': root / 'splits',
        'cache': root / 'cache',
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_phase5_index_full(track: str) -> pd.DataFrame:
    df = load_phase5_index_raw(track)
    p = get_paths(track).reg_dir / 'index_phase5.csv'
    df = pd.read_csv(p, low_memory=False)
    required = ['roi_uid', 'dataset', 'group_id', 'fov_id', 'cache_path_XB', 'defocus_um', 'y_sign', 'y_mag_um']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Phase5 index missing required columns {missing}: {p}')
    if 'cache_path_XA' not in df.columns:
        df['cache_path_XA'] = np.nan
    if 'patch_id' not in df.columns:
        df['patch_id'] = 'r00_c00'
    if 'roi_importance' not in df.columns:
        df['roi_importance'] = 1.0
    if 'source_image_path' not in df.columns:
        if 'image_path' in df.columns:
            df['source_image_path'] = df['image_path'].astype(str)
        else:
            df['source_image_path'] = np.nan
    for col in ['roi_uid', 'dataset', 'group_id', 'fov_id', 'patch_id']:
        df[col] = df[col].astype(str)
    for col in ['roi_importance', 'defocus_um', 'y_sign', 'y_mag_um']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _validate_disjoint_groups(split_map: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for name, df in split_map.items():
        for gid in df['group_id'].astype(str).unique().tolist():
            if gid in seen and seen[gid] != name:
                raise ValueError(f'Group leakage detected: group_id={gid} in {seen[gid]} and {name}')
            seen[gid] = name


def resolve_master_splits(
    track: str,
    index_df: pd.DataFrame,
    seed: int = 42,
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_root = baseline_root(track) / '_shared_splits'
    split_root.mkdir(parents=True, exist_ok=True)
    split_files = {name: split_root / f'{name}.csv' for name in ['train', 'val', 'test']}
    config_path = split_root / 'split_config.json'

    if resume and all(p.is_file() for p in split_files.values()) and config_path.is_file():
        frames = {}
        for name, path in split_files.items():
            tmp = pd.read_csv(path, low_memory=False)
            if 'roi_uid' not in tmp.columns:
                raise ValueError(f'Master split file missing roi_uid: {path}')
            frames[name] = index_df[index_df['roi_uid'].isin(tmp['roi_uid'].astype(str))].copy().reset_index(drop=True)
        _validate_disjoint_groups(frames)
        with config_path.open('r', encoding='utf-8') as f:
            meta = json.load(f)
        return frames['train'], frames['val'], frames['test'], meta

    source = 'phase5_group_split'
    sign_split_dir = get_paths(track).sign_dir / 'splits'
    use_sign = all((sign_split_dir / f'{name}.csv').is_file() for name in ['train', 'val', 'test'])
    frames: dict[str, pd.DataFrame] = {}
    if use_sign:
        frames = {}
        coverage = 0
        for name in ['train', 'val', 'test']:
            src = pd.read_csv(sign_split_dir / f'{name}.csv', low_memory=False)
            if 'roi_uid' not in src.columns:
                use_sign = False
                break
            cur = index_df[index_df['roi_uid'].isin(src['roi_uid'].astype(str))].copy().reset_index(drop=True)
            frames[name] = cur
            coverage += len(cur)
        if use_sign and coverage == len(index_df):
            source = 'phase4_sign_splits'
        else:
            frames = {}
            use_sign = False

    if not use_sign:
        train_df, val_df, test_df = phase5_group_split(index_df, split=split, seed=seed)
        frames = {'train': train_df, 'val': val_df, 'test': test_df}

    _validate_disjoint_groups(frames)
    meta = {
        'track': track,
        'seed': int(seed),
        'split': [float(x) for x in split],
        'source': source,
        'n_train': int(len(frames['train'])),
        'n_val': int(len(frames['val'])),
        'n_test': int(len(frames['test'])),
        'n_groups_train': int(frames['train']['group_id'].astype(str).nunique()),
        'n_groups_val': int(frames['val']['group_id'].astype(str).nunique()),
        'n_groups_test': int(frames['test']['group_id'].astype(str).nunique()),
    }
    keep_cols = [c for c in ['roi_uid', 'dataset', 'group_id', 'fov_id', 'defocus_um', 'y_sign', 'y_mag_um', 'roi_importance', 'cache_path_XA', 'cache_path_XB', 'patch_id', 'source_image_path'] if c in index_df.columns]
    for name, df in frames.items():
        df[keep_cols].to_csv(split_files[name], index=False)
    save_json(meta, config_path)
    return frames['train'], frames['val'], frames['test'], meta


def gt_per_fov(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fov_id, g in index_df.groupby('fov_id', sort=True):
        vals = pd.to_numeric(g['defocus_um'], errors='coerce').to_numpy(dtype=float)
        if 'roi_importance' in g.columns:
            w = pd.to_numeric(g['roi_importance'], errors='coerce').fillna(1.0).to_numpy(dtype=float)
        else:
            w = np.ones((len(g),), dtype=float)
        gt = weighted_median(vals, w)
        if gt is None:
            finite = vals[np.isfinite(vals)]
            gt = float(np.median(finite)) if finite.size else np.nan
        rows.append({'fov_id': str(fov_id), 'y_true_signed_um': gt})
    return pd.DataFrame(rows)


def build_standard_fov_frame(
    pred_df: pd.DataFrame,
    index_df: pd.DataFrame,
    baseline_name: str,
    family: str,
    evaluation_scope: str,
    input_scope: str,
    training_required: int,
    available: int = 1,
    notes: str = '',
) -> pd.DataFrame:
    gt_df = gt_per_fov(index_df)
    ds_df = infer_dataset_for_fov(index_df)
    work = pred_df.copy()
    if 'fov_id' not in work.columns:
        raise ValueError('Prediction dataframe must contain fov_id')
    if 'y_pred_signed_um' not in work.columns:
        raise ValueError('Prediction dataframe must contain y_pred_signed_um')
    work['fov_id'] = work['fov_id'].astype(str)
    work = work.merge(gt_df, on='fov_id', how='left').merge(ds_df, on='fov_id', how='left')
    work['y_pred_signed_um'] = pd.to_numeric(work['y_pred_signed_um'], errors='coerce')
    work['y_true_signed_um'] = pd.to_numeric(work['y_true_signed_um'], errors='coerce')
    work['signed_error_um'] = work['y_pred_signed_um'] - work['y_true_signed_um']
    work['abs_error_um'] = np.abs(work['signed_error_um'])
    work['baseline_name'] = baseline_name
    work['family'] = family
    work['evaluation_scope'] = evaluation_scope
    work['input_scope'] = input_scope
    work['training_required'] = int(training_required)
    work['available'] = int(available)
    work['notes'] = str(notes)
    if 'uncertain' not in work.columns:
        work['uncertain'] = work['y_pred_signed_um'].isna().astype(int)
    if 'runtime_ms_per_fov' not in work.columns:
        work['runtime_ms_per_fov'] = np.nan
    if 'model_size_mb' not in work.columns:
        work['model_size_mb'] = np.nan
    if 'mean_inputs_used_per_fov' not in work.columns:
        if 'num_inputs_used' in work.columns:
            work['mean_inputs_used_per_fov'] = pd.to_numeric(work['num_inputs_used'], errors='coerce')
        else:
            work['mean_inputs_used_per_fov'] = np.nan
    cols = [
        'dataset', 'fov_id', 'y_true_signed_um', 'y_pred_signed_um', 'abs_error_um', 'signed_error_um',
        'baseline_name', 'family', 'evaluation_scope', 'input_scope', 'training_required', 'available',
        'uncertain', 'runtime_ms_per_fov', 'model_size_mb', 'mean_inputs_used_per_fov', 'notes'
    ]
    extra = [c for c in work.columns if c not in cols]
    return work[cols + extra].copy()


def summarize_fov_frame(fov_df: pd.DataFrame, method_meta: dict[str, Any]) -> dict[str, Any]:
    work = fov_df.copy()
    pred = pd.to_numeric(work['y_pred_signed_um'], errors='coerce')
    gt = pd.to_numeric(work['y_true_signed_um'], errors='coerce')
    valid = pred.notna() & gt.notna()
    if valid.any():
        err = pred[valid].to_numpy(dtype=float) - gt[valid].to_numpy(dtype=float)
        abs_err = np.abs(err)
        catastrophic = (np.sign(pred[valid].to_numpy(dtype=float)) != np.sign(gt[valid].to_numpy(dtype=float))) & (~np.isclose(gt[valid].to_numpy(dtype=float), 0.0))
        bias = float(np.mean(err))
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(np.square(err))))
        med = float(np.median(abs_err))
        p95 = float(np.percentile(abs_err, 95))
        within05 = float(np.mean(abs_err <= 0.5) * 100.0)
        within1 = float(np.mean(abs_err <= 1.0) * 100.0)
        within2 = float(np.mean(abs_err <= 2.0) * 100.0)
        catastrophic_pct = float(np.mean(catastrophic) * 100.0)
    else:
        bias = mae = rmse = med = p95 = within05 = within1 = within2 = catastrophic_pct = np.nan
    return {
        'method': method_meta['method'],
        'family': method_meta['family'],
        'evaluation_scope': method_meta['evaluation_scope'],
        'available': int(method_meta.get('available', 1)),
        'training_required': int(method_meta.get('training_required', 0)),
        'input_scope': method_meta['input_scope'],
        'mae_um': mae,
        'rmse_um': rmse,
        'median_abs_err_um': med,
        'p95_abs_err_um': p95,
        'bias_um': bias,
        'within_0p5um_pct': within05,
        'within_1um_pct': within1,
        'within_2um_pct': within2,
        'catastrophic_wrong_direction_pct': catastrophic_pct,
        'uncertain_pct': float(np.mean(pd.to_numeric(work['uncertain'], errors='coerce').fillna(0).to_numpy(dtype=float) > 0) * 100.0),
        'runtime_ms_per_fov': float(np.nanmean(pd.to_numeric(work['runtime_ms_per_fov'], errors='coerce').to_numpy(dtype=float))),
        'model_size_mb': float(method_meta.get('model_size_mb', np.nan)),
        'mean_inputs_used_per_fov': float(np.nanmean(pd.to_numeric(work['mean_inputs_used_per_fov'], errors='coerce').to_numpy(dtype=float))),
        'n_fov': int(len(work)),
        'notes': str(method_meta.get('notes', '')),
    }


def dataset_summary_rows(fov_df: pd.DataFrame, method_meta: dict[str, Any]) -> pd.DataFrame:
    rows = []
    work = fov_df.copy()
    for dataset, g in work.groupby('dataset', sort=True):
        pred = pd.to_numeric(g['y_pred_signed_um'], errors='coerce')
        gt = pd.to_numeric(g['y_true_signed_um'], errors='coerce')
        valid = pred.notna() & gt.notna()
        if valid.any():
            err = pred[valid].to_numpy(dtype=float) - gt[valid].to_numpy(dtype=float)
            abs_err = np.abs(err)
            catastrophic = (np.sign(pred[valid].to_numpy(dtype=float)) != np.sign(gt[valid].to_numpy(dtype=float))) & (~np.isclose(gt[valid].to_numpy(dtype=float), 0.0))
            mae = float(np.mean(abs_err))
            rmse = float(np.sqrt(np.mean(np.square(err))))
            within1 = float(np.mean(abs_err <= 1.0) * 100.0)
            catastrophic_pct = float(np.mean(catastrophic) * 100.0)
        else:
            mae = rmse = within1 = catastrophic_pct = np.nan
        rows.append({
            'method': method_meta['method'],
            'dataset': str(dataset),
            'n_fov': int(len(g)),
            'mae_um': mae,
            'rmse_um': rmse,
            'within_1um_pct': within1,
            'catastrophic_wrong_direction_pct': catastrophic_pct,
            'runtime_ms_per_fov': float(np.nanmean(pd.to_numeric(g['runtime_ms_per_fov'], errors='coerce').to_numpy(dtype=float))),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    maes = pd.to_numeric(out['mae_um'], errors='coerce')
    weights = pd.to_numeric(out['n_fov'], errors='coerce').fillna(0)
    macro = float(np.nanmean(maes.to_numpy(dtype=float))) if len(out) else np.nan
    weighted = float(np.nansum(maes.to_numpy(dtype=float) * weights.to_numpy(dtype=float)) / max(float(np.nansum(weights.to_numpy(dtype=float))), 1.0))
    domain_gap = float(np.nanmax(maes.to_numpy(dtype=float)) - np.nanmin(maes.to_numpy(dtype=float))) if maes.notna().any() else np.nan
    out['macro_dataset_mae_um'] = macro
    out['weighted_dataset_mae_um'] = weighted
    out['domain_gap_mae'] = domain_gap
    return out


def near_focus_rows(fov_df: pd.DataFrame, method_meta: dict[str, Any]) -> pd.DataFrame:
    rows = []
    gt_abs = np.abs(pd.to_numeric(fov_df['y_true_signed_um'], errors='coerce').to_numpy(dtype=float))
    pred = pd.to_numeric(fov_df['y_pred_signed_um'], errors='coerce').to_numpy(dtype=float)
    gt = pd.to_numeric(fov_df['y_true_signed_um'], errors='coerce').to_numpy(dtype=float)
    for bin_name, lo, hi in NEAR_FOCUS_BINS:
        if np.isinf(hi):
            mask = gt_abs > lo
        elif lo <= 0.0:
            mask = gt_abs <= hi
        else:
            mask = (gt_abs > lo) & (gt_abs <= hi)
        valid = mask & np.isfinite(pred) & np.isfinite(gt)
        if np.any(valid):
            err = pred[valid] - gt[valid]
            abs_err = np.abs(err)
            catastrophic = (np.sign(pred[valid]) != np.sign(gt[valid])) & (~np.isclose(gt[valid], 0.0))
            mae = float(np.mean(abs_err))
            rmse = float(np.sqrt(np.mean(np.square(err))))
            within05 = float(np.mean(abs_err <= 0.5) * 100.0)
            within1 = float(np.mean(abs_err <= 1.0) * 100.0)
            catastrophic_pct = float(np.mean(catastrophic) * 100.0)
        else:
            mae = rmse = within05 = within1 = catastrophic_pct = np.nan
        rows.append({
            'method': method_meta['method'],
            'bin_name': bin_name,
            'n': int(np.sum(mask)),
            'mae_um': mae,
            'rmse_um': rmse,
            'within_0p5um_pct': within05,
            'within_1um_pct': within1,
            'catastrophic_wrong_direction_pct': catastrophic_pct,
        })
    return pd.DataFrame(rows)


def model_size_mb(paths: str | Path | Sequence[str | Path] | None) -> float:
    if paths is None:
        return float('nan')
    if isinstance(paths, (str, Path)):
        items = [Path(paths)]
    else:
        items = [Path(p) for p in paths]
    total = 0
    found = False
    for p in items:
        if p.is_file():
            total += p.stat().st_size
            found = True
    return float(total / (1024.0 * 1024.0)) if found else float('nan')


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
    stats = np.empty((int(n_bootstrap),), dtype=np.float32)
    idx_base = np.arange(len(diff))
    for i in range(int(n_bootstrap)):
        idx = rng.choice(idx_base, size=len(diff), replace=True)
        stats[i] = float(np.mean(diff[idx]))
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def holm_correction(pvals: list[float]) -> list[bool]:
    if not pvals:
        return []
    arr = np.asarray([1.0 if not np.isfinite(x) else float(x) for x in pvals], dtype=float)
    order = np.argsort(arr)
    out = [False] * len(arr)
    prev = True
    m = len(arr)
    for rank, idx in enumerate(order, start=1):
        thresh = 0.05 / (m - rank + 1)
        reject = prev and (arr[idx] <= thresh)
        out[int(idx)] = bool(reject)
        prev = reject
    return out


def effect_size_dz(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return float('nan')
    sd = float(np.std(diff, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return float('nan')
    return float(np.mean(diff) / sd)


def pairwise_rows_against_proposed(
    proposed_df: pd.DataFrame,
    baseline_frames: dict[str, pd.DataFrame],
    compare_names: Sequence[str],
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    pvals = []
    for name in compare_names:
        if name not in baseline_frames:
            continue
        other = baseline_frames[name]
        merged = proposed_df[['fov_id', 'abs_error_um']].merge(
            other[['fov_id', 'abs_error_um']], on='fov_id', suffixes=('_a', '_b'), how='inner'
        )
        a = pd.to_numeric(merged['abs_error_um_a'], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(merged['abs_error_um_b'], errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        if a.size == 0:
            continue
        test = wilcoxon_signed_rank(a, b)
        delta, lo, hi = paired_bootstrap_delta(a, b, n_bootstrap=bootstrap, seed=seed)
        row = {
            'method_a': 'proposed_two_stage_roi',
            'method_b': name,
            'metric': 'abs_error_um',
            'baseline_value': float(np.mean(a)),
            'comparison_value': float(np.mean(b)),
            'delta': delta,
            'ci_low': lo,
            'ci_high': hi,
            'p_value': float(test.get('pvalue', np.nan)),
            'effect_size': effect_size_dz(a, b),
            'correction_method': 'holm',
            'significant_after_correction': 0,
        }
        rows.append(row)
        pvals.append(row['p_value'])
    corrected = holm_correction(pvals)
    for row, sig in zip(rows, corrected):
        row['significant_after_correction'] = int(sig)
    return pd.DataFrame(rows)


def save_table_bundle(df: pd.DataFrame, csv_path: Path, tables_dir: Path, save_latex_flag: bool) -> None:
    save_table(df, csv_path, save_latex=bool(save_latex_flag))
    save_table(df, tables_dir / csv_path.name, save_latex=bool(save_latex_flag))


def plot_efficiency_frontier(df: pd.DataFrame, out_path: Path) -> None:
    work = df[pd.to_numeric(df['available'], errors='coerce').fillna(0) > 0].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if work.empty:
        ax.text(0.5, 0.5, 'No baseline efficiency data available', ha='center', va='center')
        ax.axis('off')
        save_plot(fig, out_path)
        return
    ax.scatter(pd.to_numeric(work['runtime_ms_per_fov']), pd.to_numeric(work['mae_um']), s=70, color='#1d4ed8')
    for _, row in work.iterrows():
        color = '#b91c1c' if row['method'] == 'proposed_two_stage_roi' else '#0f172a'
        ax.annotate(str(row['method']), (row['runtime_ms_per_fov'], row['mae_um']), fontsize=8, color=color, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Runtime per FOV (ms)')
    ax.set_ylabel('MAE (µm)')
    ax.set_title('Baseline Accuracy-Efficiency Frontier')
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def collect_proposed_baseline(track: str, index_df: pd.DataFrame) -> BaselinePredictionBundle:
    paths = get_paths(track)
    pred_path = require_file(paths.reg_dir / 'inference' / 'fov_aggregate_predictions.csv', 'proposed FOV predictions')
    preds = pd.read_csv(pred_path, low_memory=False)
    work = pd.DataFrame({
        'fov_id': preds['fov_id'].astype(str),
        'y_pred_signed_um': pd.to_numeric(preds['dz_hat_um'], errors='coerce'),
        'uncertain': (preds.get('status', pd.Series(['ok'] * len(preds))).astype(str) != 'ok').astype(int),
        'runtime_ms_per_fov': pd.to_numeric(preds.get('runtime_ms', np.nan), errors='coerce'),
        'num_inputs_used': pd.to_numeric(preds.get('num_rois_used', np.nan), errors='coerce'),
        'model_size_mb': model_size_mb([
            paths.sign_dir / 'models' / 'best_model.keras',
            paths.reg_dir / 'models' / 'R_plus_best.keras',
            paths.reg_dir / 'models' / 'R_minus_best.keras',
        ]),
    })
    fov_df = build_standard_fov_frame(
        work,
        index_df=index_df,
        baseline_name='proposed_two_stage_roi',
        family='proposed',
        evaluation_scope='full_pipeline',
        input_scope='content_aware_roi',
        training_required=1,
        available=1,
        notes='Existing two-stage ROI-aware autofocus pipeline.',
    )
    return BaselinePredictionBundle(
        baseline_name='proposed_two_stage_roi',
        family='proposed',
        evaluation_scope='full_pipeline',
        input_scope='content_aware_roi',
        training_required=1,
        available=1,
        fov_df=fov_df,
        roi_df=None,
        metrics={},
        runtime={'runtime_ms_per_fov': float(np.nanmean(pd.to_numeric(work['runtime_ms_per_fov'], errors='coerce')))},
        notes='Existing two-stage ROI-aware autofocus pipeline.',
    )


def run_fixed_policy_baseline(
    track: str,
    index_df: pd.DataFrame,
    method_name: str,
    family: str,
    evaluation_scope: str,
    input_scope: str,
    policy_name: str,
    k: int | None = None,
    seed: int = 42,
    batch_size: int = 128,
    note: str = '',
) -> BaselinePredictionBundle:
    paths = get_paths(track)
    tau = float(load_tau(paths))
    fixed_df, latency = prepare_fixed_model_outputs(index_df, paths, batch_size=batch_size, resume=True)
    work = index_df.merge(fixed_df, on='roi_uid', how='left')
    ctx = ROIPolicyContext(seed=int(seed))

    fov_rows = []
    roi_rows = []
    model_size = model_size_mb([
        paths.sign_dir / 'models' / 'best_model.keras',
        paths.reg_dir / 'models' / 'R_plus_best.keras',
        paths.reg_dir / 'models' / 'R_minus_best.keras',
    ])

    for fov_id, g in work.groupby('fov_id', sort=True):
        result = select_rois_for_policy(g, policy_name=policy_name, ctx=ctx, k=k)
        selected = result.selected_df.copy() if result.available and not result.selected_df.empty else pd.DataFrame()
        if selected.empty:
            fov_rows.append({
                'fov_id': str(fov_id),
                'y_pred_signed_um': np.nan,
                'uncertain': 1,
                'runtime_ms_per_fov': float(result.selection_time_ms),
                'num_inputs_used': 0,
                'selection_time_ms': float(result.selection_time_ms),
                'aggregation_time_ms': 0.0,
                'model_size_mb': model_size,
                'status': 'no_selected_roi',
            })
            continue

        selected['stageA_weight'] = pd.to_numeric(selected.get('roi_importance', 1.0), errors='coerce').fillna(1.0) * pd.to_numeric(selected['c_sign'], errors='coerce').fillna(0.0)
        selected.loc[pd.to_numeric(selected['c_sign'], errors='coerce').fillna(0.0) < tau, 'stageA_weight'] = 0.0
        weights = pd.to_numeric(selected['stageA_weight'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        p = pd.to_numeric(selected['p_sign'], errors='coerce').fillna(0.5).to_numpy(dtype=float)
        votes = (p >= 0.5).astype(float)
        keep = weights > 0
        agg_ms = 0.0
        if not np.any(keep):
            y_pred = np.nan
            status = 'all_gated'
            uncertain = 1
        else:
            V = float(np.sum(weights * votes))
            W = float(np.sum(weights))
            pred_sign = 1 if V >= 0.5 * W else 0
            if pred_sign == 1:
                signed = pd.to_numeric(selected['mag_plus'], errors='coerce').to_numpy(dtype=float)
            else:
                signed = -pd.to_numeric(selected['mag_minus'], errors='coerce').to_numpy(dtype=float)
            t0 = time.perf_counter()
            wm = weighted_median(signed[keep], weights[keep])
            agg_ms = 1000.0 * (time.perf_counter() - t0)
            y_pred = float(wm) if wm is not None else np.nan
            status = 'ok' if np.isfinite(y_pred) else 'agg_failed'
            uncertain = 0 if np.isfinite(y_pred) else 1
        fov_rows.append({
            'fov_id': str(fov_id),
            'y_pred_signed_um': y_pred,
            'uncertain': int(uncertain),
            'runtime_ms_per_fov': float(result.selection_time_ms) + float(len(selected)) * float(latency.get('sign_ms_per_roi', np.nan)) + float(len(selected)) * float(latency.get('stageB_ms_per_roi', np.nan)) + float(agg_ms),
            'num_inputs_used': int(len(selected)),
            'selection_time_ms': float(result.selection_time_ms),
            'aggregation_time_ms': float(agg_ms),
            'model_size_mb': model_size,
            'status': status,
        })
        tmp = selected.copy()
        tmp['baseline_name'] = method_name
        tmp['y_pred_signed_um'] = np.where((p >= 0.5), pd.to_numeric(tmp['mag_plus'], errors='coerce'), -pd.to_numeric(tmp['mag_minus'], errors='coerce'))
        tmp['runtime_ms_per_roi'] = float(latency.get('sign_ms_per_roi', np.nan)) + float(latency.get('stageB_ms_per_roi', np.nan))
        roi_rows.append(tmp)

    pred_df = pd.DataFrame(fov_rows)
    fov_df = build_standard_fov_frame(
        pred_df,
        index_df=index_df,
        baseline_name=method_name,
        family=family,
        evaluation_scope=evaluation_scope,
        input_scope=input_scope,
        training_required=0,
        available=1,
        notes=note,
    )
    roi_df = pd.concat(roi_rows, ignore_index=True) if roi_rows else pd.DataFrame()
    return BaselinePredictionBundle(
        baseline_name=method_name,
        family=family,
        evaluation_scope=evaluation_scope,
        input_scope=input_scope,
        training_required=0,
        available=1,
        fov_df=fov_df,
        roi_df=roi_df,
        metrics={},
        runtime=latency,
        notes=note,
    )


def unavailable_bundle(method_name: str, family: str, evaluation_scope: str, input_scope: str, n_fov: int, note: str = '') -> BaselinePredictionBundle:
    fov_df = pd.DataFrame({
        'dataset': [np.nan] * n_fov,
        'fov_id': [np.nan] * n_fov,
        'y_true_signed_um': [np.nan] * n_fov,
        'y_pred_signed_um': [np.nan] * n_fov,
        'abs_error_um': [np.nan] * n_fov,
        'signed_error_um': [np.nan] * n_fov,
        'baseline_name': [method_name] * n_fov,
        'family': [family] * n_fov,
        'evaluation_scope': [evaluation_scope] * n_fov,
        'input_scope': [input_scope] * n_fov,
        'training_required': [0] * n_fov,
        'available': [0] * n_fov,
        'uncertain': [1] * n_fov,
        'runtime_ms_per_fov': [np.nan] * n_fov,
        'model_size_mb': [np.nan] * n_fov,
        'mean_inputs_used_per_fov': [np.nan] * n_fov,
        'notes': [note] * n_fov,
    })
    return BaselinePredictionBundle(method_name, family, evaluation_scope, input_scope, 0, 0, fov_df, notes=note)
