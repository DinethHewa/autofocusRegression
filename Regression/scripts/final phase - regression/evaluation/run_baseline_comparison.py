#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline_utils import (
    DEFAULT_BASELINES,
    BaselinePredictionBundle,
    baseline_method_root,
    collect_proposed_baseline,
    dataset_summary_rows,
    gt_per_fov,
    load_phase5_index_full,
    model_size_mb,
    near_focus_rows,
    pairwise_rows_against_proposed,
    plot_efficiency_frontier,
    save_table_bundle,
    summarize_fov_frame,
    unavailable_bundle,
)
from evaluation_utils import get_paths, log, save_plot, save_table

SCRIPT_ROOT = Path(__file__).resolve().parent
FINAL_ROOT = SCRIPT_ROOT.parent
BASELINE_ROOT = FINAL_ROOT / 'baselines'
DIRECT_DIR = BASELINE_ROOT / 'direct_regression'
CLASSICAL_DIR = BASELINE_ROOT / 'classical_focus'
INPUT_DIR = BASELINE_ROOT / 'input_baselines'

SUPPORTED_BASELINES = [
    'proposed_two_stage_roi',
    'direct_single_stage_regression',
    'classical_focus_best',
    'center_crop_inference_only',
    'center_crop_retrained',
    'full_field_tiled_proxy',
    'full_image_retrained',
    'all_rois_proxy',
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run the manuscript baseline-comparison package')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--baselines', nargs='*', default=list(DEFAULT_BASELINES))
    ap.add_argument('--mode', choices=['eval_only', 'train_and_eval', 'full'], default='train_and_eval')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--bins', type=float, default=0.5)
    ap.add_argument('--k-values', nargs='*', default=None)
    ap.add_argument('--save-plots', action='store_true')
    ap.add_argument('--save-latex', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--out-dir', default='')
    return ap.parse_args()


def _comparison_out_dir(track: str, out_dir: str) -> Path:
    if out_dir:
        p = Path(out_dir)
    else:
        p = get_paths(track).eval_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run(cmd: list[str]) -> None:
    print('[INFO] Running baseline command: ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def _method_meta(bundle: BaselinePredictionBundle, method_name: str, training_required: int | None = None) -> dict[str, Any]:
    fov_df = bundle.fov_df
    model_mb = float(np.nanmean(pd.to_numeric(fov_df.get('model_size_mb', np.nan), errors='coerce').to_numpy(dtype=float))) if 'model_size_mb' in fov_df.columns else float('nan')
    return {
        'method': method_name,
        'family': bundle.family,
        'evaluation_scope': bundle.evaluation_scope,
        'available': int(bundle.available),
        'training_required': int(bundle.training_required if training_required is None else training_required),
        'input_scope': bundle.input_scope,
        'model_size_mb': model_mb,
        'notes': bundle.notes,
    }


def _standardize_existing_fov(
    fov_raw: pd.DataFrame,
    index_df: pd.DataFrame,
    method_name: str,
    family: str,
    evaluation_scope: str,
    input_scope: str,
    training_required: int,
    notes: str,
    model_size_mb_value: float,
) -> pd.DataFrame:
    work = fov_raw.copy()
    gt_df = gt_per_fov(index_df)
    ds_df = (
        index_df[['fov_id', 'dataset']]
        .drop_duplicates(subset=['fov_id'], keep='first')
        .assign(fov_id=lambda d: d['fov_id'].astype(str), dataset=lambda d: d['dataset'].astype(str))
    )
    work['fov_id'] = work['fov_id'].astype(str)
    work = work.drop(columns=['dataset', 'y_true_signed_um'], errors='ignore')
    work = work.merge(ds_df, on='fov_id', how='left').merge(gt_df, on='fov_id', how='left')
    work['y_pred_signed_um'] = pd.to_numeric(work['y_pred_signed_um'], errors='coerce')
    work['y_true_signed_um'] = pd.to_numeric(work['y_true_signed_um'], errors='coerce')
    work['signed_error_um'] = work['y_pred_signed_um'] - work['y_true_signed_um']
    work['abs_error_um'] = work['signed_error_um'].abs()
    if 'uncertain' not in work.columns:
        if 'status' in work.columns:
            work['uncertain'] = (work['status'].astype(str) != 'ok').astype(int)
        else:
            work['uncertain'] = work['y_pred_signed_um'].isna().astype(int)
    if 'runtime_ms_per_fov' not in work.columns:
        work['runtime_ms_per_fov'] = np.nan
    if 'mean_inputs_used_per_fov' not in work.columns:
        if 'num_inputs_used' in work.columns:
            work['mean_inputs_used_per_fov'] = pd.to_numeric(work['num_inputs_used'], errors='coerce')
        else:
            work['mean_inputs_used_per_fov'] = np.nan
    work['baseline_name'] = method_name
    work['family'] = family
    work['evaluation_scope'] = evaluation_scope
    work['input_scope'] = input_scope
    work['training_required'] = int(training_required)
    work['available'] = 1
    work['model_size_mb'] = float(model_size_mb_value)
    work['notes'] = str(notes)
    return work


def _collect_direct_bundle(track: str, index_df: pd.DataFrame, mode: str, seed: int, resume: bool) -> BaselinePredictionBundle:
    train_script = DIRECT_DIR / 'train_direct_regression.py'
    infer_script = DIRECT_DIR / 'infer_direct_regression.py'
    root = baseline_method_root(track, 'direct_regression')
    model_path = root / 'models' / 'best_model.keras'
    fov_path = root / 'inference' / 'fov_predictions.csv'
    roi_path = root / 'inference' / 'roi_predictions.csv'
    eval_path = root / 'metrics' / 'eval.json'
    if mode in {'train_and_eval', 'full'} and (not model_path.is_file() or not resume):
        cmd = [sys.executable, str(train_script), '--track', track, '--seed', str(seed)]
        if resume:
            cmd.append('--resume')
        _run(cmd)
    if mode in {'train_and_eval', 'full'} and (not fov_path.is_file() or not roi_path.is_file() or not resume):
        cmd = [sys.executable, str(infer_script), '--track', track, '--split-name', 'test']
        if resume:
            cmd.append('--resume')
        _run(cmd)
    if not (model_path.is_file() and fov_path.is_file()):
        raise FileNotFoundError(f'Direct single-stage regression outputs are unavailable under {root}')

    fov_raw = pd.read_csv(fov_path, low_memory=False)
    roi_raw = pd.read_csv(roi_path, low_memory=False) if roi_path.is_file() else None
    runtime_ms_per_roi = np.nan
    if eval_path.is_file():
        with eval_path.open('r', encoding='utf-8') as f:
            runtime_ms_per_roi = float(json.load(f).get('runtime_ms_per_roi', np.nan))
    if 'runtime_ms_per_fov' not in fov_raw.columns:
        fov_raw['runtime_ms_per_fov'] = pd.to_numeric(fov_raw.get('num_inputs_used', np.nan), errors='coerce') * float(runtime_ms_per_roi)
    model_mb = model_size_mb(model_path)
    note = 'Direct single-stage signed-distance regressor on the same cached ROI tensors used by the proposed method.'
    fov_df = _standardize_existing_fov(
        fov_raw,
        index_df=index_df,
        method_name='direct_single_stage_regression',
        family='learned_architecture',
        evaluation_scope='architecture_baseline',
        input_scope='shared_roi_tensors',
        training_required=1,
        notes=note,
        model_size_mb_value=model_mb,
    )
    return BaselinePredictionBundle(
        baseline_name='direct_single_stage_regression',
        family='learned_architecture',
        evaluation_scope='architecture_baseline',
        input_scope='shared_roi_tensors',
        training_required=1,
        available=1,
        fov_df=fov_df,
        roi_df=roi_raw,
        metrics={},
        runtime={'runtime_ms_per_roi': runtime_ms_per_roi},
        notes=note,
    )


def _collect_classical_bundle(track: str, mode: str, seed: int, resume: bool) -> BaselinePredictionBundle:
    fit_script = CLASSICAL_DIR / 'fit_classical_focus.py'
    eval_script = CLASSICAL_DIR / 'eval_classical_focus.py'
    root = baseline_method_root(track, 'classical_focus_best')
    fov_path = root / 'inference' / 'fov_predictions.csv'
    eval_path = root / 'metrics' / 'eval.json'
    if mode in {'train_and_eval', 'full'} and (not fov_path.is_file() or not resume):
        cmd = [sys.executable, str(fit_script), '--track', track, '--seed', str(seed)]
        if resume:
            cmd.append('--resume')
        _run(cmd)
    if mode in {'train_and_eval', 'full'} and (not fov_path.is_file() or not resume):
        cmd = [sys.executable, str(eval_script), '--track', track, '--seed', str(seed)]
        if resume:
            cmd.append('--resume')
        _run(cmd)
    if not fov_path.is_file():
        raise FileNotFoundError(f'Classical focus outputs are unavailable under {root}')
    fov_df = pd.read_csv(fov_path, low_memory=False)
    metrics = {}
    if eval_path.is_file():
        with eval_path.open('r', encoding='utf-8') as f:
            metrics = json.load(f)
    return BaselinePredictionBundle(
        baseline_name='classical_focus_best',
        family='classical',
        evaluation_scope='classical_handcrafted_regression',
        input_scope=str(fov_df['input_scope'].dropna().iloc[0]) if 'input_scope' in fov_df.columns and fov_df['input_scope'].notna().any() else 'classical_focus_selected_input',
        training_required=1,
        available=1,
        fov_df=fov_df,
        roi_df=None,
        metrics=metrics,
        runtime={'runtime_ms_per_fov': metrics.get('runtime_ms_per_fov', np.nan)},
        notes='Classical focus-measure baseline using handcrafted ROI features and shallow signed-distance regression.',
    )


def _collect_center_bundle(track: str, seed: int, resume: bool) -> BaselinePredictionBundle:
    if str(INPUT_DIR) not in sys.path:
        sys.path.insert(0, str(INPUT_DIR))
    from center_crop_baseline import CenterCropBaseline

    baseline = CenterCropBaseline(track=track, seed=seed, resume=resume)
    return baseline.predict_fov()


def _collect_full_field_bundle(track: str, seed: int, resume: bool, method_name: str = 'full_field_tiled_proxy') -> BaselinePredictionBundle:
    if str(INPUT_DIR) not in sys.path:
        sys.path.insert(0, str(INPUT_DIR))
    from full_image_baseline import FullFieldTiledProxyBaseline

    baseline = FullFieldTiledProxyBaseline(track=track, seed=seed, resume=resume)
    bundle = baseline.predict_fov()
    if method_name == 'all_rois_proxy':
        bundle.fov_df = bundle.fov_df.copy()
        bundle.fov_df['baseline_name'] = 'all_rois_proxy'
        bundle.fov_df['notes'] = 'Alias of the full-field tiled proxy using all ROI tiles.'
        bundle.baseline_name = 'all_rois_proxy'
        bundle.notes = 'Alias of the full-field tiled proxy using all ROI tiles.'
    return bundle


def _unavailable_requested(method_name: str, n_fov: int, note: str, family: str = 'learned_input', evaluation_scope: str = 'not_available', input_scope: str = 'not_available') -> BaselinePredictionBundle:
    return unavailable_bundle(method_name, family=family, evaluation_scope=evaluation_scope, input_scope=input_scope, n_fov=n_fov, note=note)


def _method_order(method: str) -> int:
    order = {name: i for i, name in enumerate(SUPPORTED_BASELINES)}
    return order.get(method, 999)


def _plot_dataset_mae(df: pd.DataFrame, out_path: Path) -> None:
    work = df[pd.to_numeric(df['mae_um'], errors='coerce').notna()].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if work.empty:
        ax.text(0.5, 0.5, 'No dataset-wise baseline results available', ha='center', va='center')
        ax.axis('off')
        save_plot(fig, out_path)
        return
    pivot = work.pivot(index='dataset', columns='method', values='mae_um').sort_index()
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel('MAE (µm)')
    ax.set_title('Dataset-Wise Baseline Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    save_plot(fig, out_path)


def _plot_near_focus(df: pd.DataFrame, out_path: Path) -> None:
    work = df[pd.to_numeric(df['mae_um'], errors='coerce').notna()].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if work.empty:
        ax.text(0.5, 0.5, 'No near-focus baseline results available', ha='center', va='center')
        ax.axis('off')
        save_plot(fig, out_path)
        return
    bin_order = ['|dz|<=0.5', '0.5<|dz|<=1', '1<|dz|<=2', '>2']
    xpos = np.arange(len(bin_order), dtype=float)
    for method, g in work.groupby('method', sort=False):
        g = g.set_index('bin_name').reindex(bin_order)
        ax.plot(xpos, pd.to_numeric(g['mae_um'], errors='coerce'), marker='o', label=method)
    ax.set_xticks(xpos)
    ax.set_xticklabels(bin_order)
    ax.set_ylabel('MAE (µm)')
    ax.set_title('Near-Focus Baseline Behavior')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_plot(fig, out_path)


def _find_failure_cases(baseline_frames: dict[str, pd.DataFrame], index_df: pd.DataFrame, out_dir: Path, save_plots_flag: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    panel_rows = []
    source_map = (
        index_df[['fov_id', 'dataset', 'source_image_path']]
        .dropna(subset=['fov_id'])
        .drop_duplicates(subset=['fov_id'], keep='first')
        .assign(fov_id=lambda d: d['fov_id'].astype(str))
        .set_index('fov_id')
    )

    def add_pair_category(a_name: str, b_name: str, category: str, descending: bool = True, limit: int = 8):
        if a_name not in baseline_frames or b_name not in baseline_frames:
            return
        a = baseline_frames[a_name][['fov_id', 'dataset', 'y_true_signed_um', 'y_pred_signed_um', 'abs_error_um']].rename(columns={'y_pred_signed_um': 'a_pred', 'abs_error_um': 'a_abs'})
        b = baseline_frames[b_name][['fov_id', 'y_pred_signed_um', 'abs_error_um']].rename(columns={'y_pred_signed_um': 'b_pred', 'abs_error_um': 'b_abs'})
        merged = a.merge(b, on='fov_id', how='inner')
        if merged.empty:
            return
        merged['delta_error_um'] = pd.to_numeric(merged['b_abs'], errors='coerce') - pd.to_numeric(merged['a_abs'], errors='coerce')
        merged = merged.sort_values('delta_error_um', ascending=not descending).head(limit)
        for _, row in merged.iterrows():
            payload = {
                'category': category,
                'dataset': row.get('dataset', np.nan),
                'fov_id': str(row['fov_id']),
                'method': a_name,
                'other_competing_method': b_name,
                'y_true_signed_um': row.get('y_true_signed_um', np.nan),
                'y_pred_signed_um': row.get('a_pred', np.nan),
                'other_y_pred_signed_um': row.get('b_pred', np.nan),
                'abs_error_um': row.get('a_abs', np.nan),
                'other_abs_error_um': row.get('b_abs', np.nan),
                'delta_error_um': row.get('delta_error_um', np.nan),
                'source_image_path': source_map.loc[str(row['fov_id']), 'source_image_path'] if str(row['fov_id']) in source_map.index else np.nan,
            }
            rows.append(payload)

    add_pair_category('proposed_two_stage_roi', 'direct_single_stage_regression', 'proposed_beats_direct', descending=True)
    add_pair_category('direct_single_stage_regression', 'proposed_two_stage_roi', 'direct_beats_proposed', descending=True)
    add_pair_category('proposed_two_stage_roi', 'center_crop_inference_only', 'center_crop_fails_roi_succeeds', descending=True)
    add_pair_category('proposed_two_stage_roi', 'full_field_tiled_proxy', 'full_field_proxy_fails_roi_succeeds', descending=True)

    if 'classical_focus_best' in baseline_frames:
        classical = baseline_frames['classical_focus_best'].copy()
        classical = classical.sort_values('abs_error_um', ascending=False).head(8)
        for _, row in classical.iterrows():
            rows.append({
                'category': 'classical_focus_fails',
                'dataset': row.get('dataset', np.nan),
                'fov_id': str(row['fov_id']),
                'method': 'classical_focus_best',
                'other_competing_method': '',
                'y_true_signed_um': row.get('y_true_signed_um', np.nan),
                'y_pred_signed_um': row.get('y_pred_signed_um', np.nan),
                'other_y_pred_signed_um': np.nan,
                'abs_error_um': row.get('abs_error_um', np.nan),
                'other_abs_error_um': np.nan,
                'delta_error_um': np.nan,
                'source_image_path': source_map.loc[str(row['fov_id']), 'source_image_path'] if str(row['fov_id']) in source_map.index else np.nan,
            })

    if 'proposed_two_stage_roi' in baseline_frames:
        for dataset_name in ['bma', 'pbs']:
            subset = baseline_frames['proposed_two_stage_roi']
            subset = subset[subset['dataset'].astype(str) == dataset_name].sort_values('abs_error_um', ascending=False).head(5)
            for _, row in subset.iterrows():
                rows.append({
                    'category': f'hard_case_{dataset_name}',
                    'dataset': row.get('dataset', np.nan),
                    'fov_id': str(row['fov_id']),
                    'method': 'proposed_two_stage_roi',
                    'other_competing_method': '',
                    'y_true_signed_um': row.get('y_true_signed_um', np.nan),
                    'y_pred_signed_um': row.get('y_pred_signed_um', np.nan),
                    'other_y_pred_signed_um': np.nan,
                    'abs_error_um': row.get('abs_error_um', np.nan),
                    'other_abs_error_um': np.nan,
                    'delta_error_um': np.nan,
                    'source_image_path': source_map.loc[str(row['fov_id']), 'source_image_path'] if str(row['fov_id']) in source_map.index else np.nan,
                })

    failure_df = pd.DataFrame(rows).drop_duplicates(subset=['category', 'fov_id', 'method', 'other_competing_method']).reset_index(drop=True)
    panel_dir = out_dir / 'qualitative_baseline_panels'
    panel_dir.mkdir(parents=True, exist_ok=True)
    if save_plots_flag and not failure_df.empty:
        for i, row in failure_df.iterrows():
            src = Path(str(row.get('source_image_path', '')))
            panel_path = panel_dir / f"{i:03d}_{str(row['category']).replace('/', '_')}_{str(row['fov_id']).replace('/', '_')}.png"
            if src.is_file():
                try:
                    img = plt.imread(str(src))
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(img)
                    ax.set_title(
                        f"{row['category']}\nGT={row['y_true_signed_um']:.3f} µm | {row['method']}={row['y_pred_signed_um']:.3f} µm\n"
                        f"abs={row['abs_error_um']:.3f} µm | other={row['other_competing_method']} | delta={row['delta_error_um']:.3f}"
                    )
                    ax.axis('off')
                    save_plot(fig, panel_path)
                    panel_rows.append({**row.to_dict(), 'panel_path': str(panel_path)})
                except Exception:
                    panel_rows.append({**row.to_dict(), 'panel_path': ''})
            else:
                panel_rows.append({**row.to_dict(), 'panel_path': ''})
    else:
        panel_rows = [{**row.to_dict(), 'panel_path': ''} for _, row in failure_df.iterrows()]
    return failure_df, pd.DataFrame(panel_rows)


def main() -> None:
    args = parse_args()
    invalid = [b for b in args.baselines if b not in SUPPORTED_BASELINES]
    if invalid:
        raise ValueError(f'Unsupported baselines requested: {invalid}. Supported: {SUPPORTED_BASELINES}')

    paths = get_paths(args.track)
    out_dir = _comparison_out_dir(args.track, args.out_dir)
    log(paths, 'Starting baseline comparison runner')

    index_df = load_phase5_index_full(args.track)
    n_fov = int(index_df['fov_id'].astype(str).nunique())

    requested = list(dict.fromkeys(['proposed_two_stage_roi'] + list(args.baselines)))
    bundles: dict[str, BaselinePredictionBundle] = {}

    for method_name in requested:
        try:
            if method_name == 'proposed_two_stage_roi':
                bundles[method_name] = collect_proposed_baseline(args.track, index_df)
            elif method_name == 'direct_single_stage_regression':
                bundles[method_name] = _collect_direct_bundle(args.track, index_df, mode=args.mode, seed=args.seed, resume=args.resume)
            elif method_name == 'classical_focus_best':
                bundles[method_name] = _collect_classical_bundle(args.track, mode=args.mode, seed=args.seed, resume=args.resume)
            elif method_name == 'center_crop_inference_only':
                bundles[method_name] = _collect_center_bundle(args.track, seed=args.seed, resume=args.resume)
            elif method_name == 'full_field_tiled_proxy':
                bundles[method_name] = _collect_full_field_bundle(args.track, seed=args.seed, resume=args.resume, method_name=method_name)
            elif method_name == 'all_rois_proxy':
                bundles[method_name] = _collect_full_field_bundle(args.track, seed=args.seed, resume=args.resume, method_name=method_name)
            elif method_name == 'center_crop_retrained':
                bundles[method_name] = _unavailable_requested(method_name, n_fov, 'Not implemented in this patch. Use center_crop_inference_only as the fixed-model input baseline.', input_scope='center_crop_retrained')
            elif method_name == 'full_image_retrained':
                bundles[method_name] = _unavailable_requested(method_name, n_fov, 'True full-image retraining is infeasible in the current code path. Use full_field_tiled_proxy instead.', input_scope='full_image_retrained')
            else:
                bundles[method_name] = _unavailable_requested(method_name, n_fov, 'Unknown baseline request.')
        except Exception as exc:
            family = 'learned_architecture' if method_name == 'direct_single_stage_regression' else 'classical' if method_name == 'classical_focus_best' else 'learned_input'
            input_scope = 'shared_roi_tensors' if method_name == 'direct_single_stage_regression' else 'classical_focus_selected_input' if method_name == 'classical_focus_best' else method_name
            bundles[method_name] = _unavailable_requested(method_name, n_fov, f'Unavailable: {exc}', family=family, input_scope=input_scope)
            log(paths, f'[WARN] baseline {method_name} unavailable: {exc}')

    summary_rows = []
    dataset_frames = []
    near_frames = []
    available_frames: dict[str, pd.DataFrame] = {}

    for method_name, bundle in bundles.items():
        meta = _method_meta(bundle, method_name)
        summary_rows.append(summarize_fov_frame(bundle.fov_df, meta))
        dataset_frames.append(dataset_summary_rows(bundle.fov_df, meta))
        near_frames.append(near_focus_rows(bundle.fov_df, meta))
        if int(meta['available']) > 0:
            available_frames[method_name] = bundle.fov_df.copy()

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['mae_um', 'catastrophic_wrong_direction_pct', 'runtime_ms_per_fov', 'method'], na_position='last').reset_index(drop=True)
    dataset_df = pd.concat(dataset_frames, ignore_index=True) if dataset_frames else pd.DataFrame()
    near_df = pd.concat(near_frames, ignore_index=True) if near_frames else pd.DataFrame()

    pairwise_df = pd.DataFrame()
    if 'proposed_two_stage_roi' in available_frames:
        compare_names = [m for m in requested if m != 'proposed_two_stage_roi' and m in available_frames]
        pairwise_df = pairwise_rows_against_proposed(
            available_frames['proposed_two_stage_roi'],
            available_frames,
            compare_names=compare_names,
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        )

    efficiency_df = summary_df[['method', 'available', 'runtime_ms_per_fov', 'mae_um', 'rmse_um', 'mean_inputs_used_per_fov']].copy()
    efficiency_df['mae_per_ms'] = pd.to_numeric(efficiency_df['mae_um'], errors='coerce') / pd.to_numeric(efficiency_df['runtime_ms_per_fov'], errors='coerce')
    efficiency_df['mae_per_input'] = pd.to_numeric(efficiency_df['mae_um'], errors='coerce') / pd.to_numeric(efficiency_df['mean_inputs_used_per_fov'], errors='coerce')

    failure_df, panel_df = _find_failure_cases(available_frames, index_df, out_dir, save_plots_flag=bool(args.save_plots))

    save_table(summary_df, out_dir / 'baseline_comparison_results.csv', save_latex=False)
    save_table(summary_df, paths.tables_dir / 'Table_Baseline_Comparison.csv', save_latex=bool(args.save_latex))

    save_table(dataset_df, out_dir / 'baseline_by_dataset.csv', save_latex=False)
    save_table(dataset_df, paths.tables_dir / 'Table_Baseline_ByDataset.csv', save_latex=bool(args.save_latex))

    save_table(near_df, out_dir / 'baseline_near_focus.csv', save_latex=False)
    save_table(near_df, paths.tables_dir / 'Table_Baseline_NearFocus.csv', save_latex=bool(args.save_latex))

    save_table(pairwise_df, out_dir / 'baseline_pairwise_tests.csv', save_latex=False)
    save_table(pairwise_df, paths.tables_dir / 'Table_Baseline_Stats.csv', save_latex=bool(args.save_latex))

    save_table(efficiency_df, out_dir / 'baseline_efficiency.csv', save_latex=False)
    save_table(efficiency_df, paths.tables_dir / 'Table_Baseline_Efficiency.csv', save_latex=bool(args.save_latex))

    save_table(failure_df, out_dir / 'baseline_failure_cases.csv', save_latex=False)
    save_table(panel_df, out_dir / 'baseline_qualitative_panel.csv', save_latex=False)

    if args.save_plots:
        plot_efficiency_frontier(efficiency_df, out_dir / 'baseline_efficiency_frontier.png')
        _plot_dataset_mae(dataset_df, out_dir / 'baseline_dataset_mae.png')
        _plot_near_focus(near_df, out_dir / 'baseline_near_focus.png')

    log(paths, f'Baseline comparison complete. Outputs: {out_dir}')


if __name__ == '__main__':
    main()
