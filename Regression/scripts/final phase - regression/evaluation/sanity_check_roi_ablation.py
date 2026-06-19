#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_utils import TRACKS, get_paths, log
from roi_policy_utils import ROIPolicyContext, expand_policy_jobs, policy_label, select_rois_for_policy


EXPECTED_SELECTION_COLUMNS = {
    'fov_id',
    'roi_id',
    'dataset',
    'selection_rank',
    'selection_score',
    'roi_policy',
    'selected',
}

EXPECTED_EVAL_COLUMNS = {
    'roi_policy',
    'K_effective_mean',
    'latency_ms_per_fov',
    'coverage_pct',
    'uncertain_pct',
    'catastrophic_wrong_direction_pct',
    'bias_um',
    'mae_um',
    'rmse_um',
    'median_abs_error_um',
    'p95_abs_error_um',
    'within_0.5um_pct',
    'within_1um_pct',
    'within_2um_pct',
    'n_fov',
}

DEFAULT_POLICIES = [
    'center_top1',
    'random_k',
    'focus_only_topk',
    'occupancy_only_topk',
    'hybrid_proposed',
    'all_rois',
]


def _load_index(track: str) -> pd.DataFrame:
    paths = get_paths(track)
    p = paths.reg_dir / 'index_phase5.csv'
    if not p.is_file():
        raise FileNotFoundError(f'Missing Phase 5 index: {p}')
    df = pd.read_csv(p, low_memory=False)
    required = ['roi_uid', 'fov_id', 'dataset']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Phase 5 index missing required columns {missing}: {p}')
    if 'patch_id' not in df.columns:
        df['patch_id'] = 'r00_c00'
    if 'source_image_path' not in df.columns and 'image_path' in df.columns:
        df['source_image_path'] = df['image_path'].astype(str)
    if 'source_image_path' not in df.columns:
        df['source_image_path'] = np.nan
    df['fov_id'] = df['fov_id'].astype(str)
    df['patch_id'] = df['patch_id'].astype(str)
    return df


def _choose_sample_fovs(df: pd.DataFrame, sample_fovs: int, seed: int) -> list[str]:
    counts = df.groupby('fov_id').size().rename('n')
    multi = counts[counts > 1].index.astype(str).tolist()
    single = counts[counts == 1].index.astype(str).tolist()
    rng = np.random.default_rng(int(seed))
    chosen: list[str] = []
    if multi:
        n_multi = min(len(multi), max(sample_fovs - 1, 1))
        chosen.extend(sorted(rng.choice(np.array(multi, dtype=object), size=n_multi, replace=False).tolist()))
    if single:
        chosen.extend(sorted(rng.choice(np.array(single, dtype=object), size=min(1, len(single)), replace=False).tolist()))
    if not chosen:
        chosen = sorted(df['fov_id'].astype(str).unique().tolist())[:sample_fovs]
    return chosen[:sample_fovs]


def _validate_eval_outputs(paths) -> list[str]:
    errors: list[str] = []
    summary_path = paths.eval_dir / 'roi_ablation_to_regression_performance.csv'
    if not summary_path.is_file():
        errors.append(f'missing ROI ablation summary CSV: {summary_path}')
        return errors
    summary_df = pd.read_csv(summary_path)
    missing = sorted(EXPECTED_EVAL_COLUMNS.difference(summary_df.columns))
    if missing:
        errors.append(f'ROI ablation summary missing columns: {missing}')
    if 'available' in summary_df.columns:
        avail = summary_df[pd.to_numeric(summary_df['available'], errors='coerce').fillna(0) > 0]
        if not avail.empty:
            if not np.isfinite(pd.to_numeric(avail['mae_um'], errors='coerce')).all():
                errors.append('available ROI policies contain non-finite mae_um values')
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description='Lightweight validation checks for ROI ablation evaluation')
    ap.add_argument('--track', required=True, choices=TRACKS)
    ap.add_argument('--sample-fovs', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--roi-policies', nargs='+', default=DEFAULT_POLICIES)
    ap.add_argument('--k-values', nargs='+', default=['1', '3', '5', '7'])
    ap.add_argument('--legacy-calibration-csv', default=None)
    ap.add_argument('--cnn-calibration-csv', default=None)
    ap.add_argument('--cellness-predictor', default=None)
    ap.add_argument('--cellness-model-path', default=None)
    ap.add_argument('--predictor-mode', default='auto', choices=['auto', 'map', 'tiles'])
    ap.add_argument('--dummy-threshold', type=float, default=0.5)
    ap.add_argument('--require-evaluation-outputs', action='store_true')
    args = ap.parse_args()

    paths = get_paths(args.track)
    log(paths, 'Starting ROI ablation sanity checks')
    index_df = _load_index(args.track)
    sample_fovs = _choose_sample_fovs(index_df, int(args.sample_fovs), int(args.seed))
    sample_df = index_df[index_df['fov_id'].isin(sample_fovs)].copy().reset_index(drop=True)
    if sample_df.empty:
        raise ValueError('No FOVs available for ROI ablation sanity checks.')

    ctx = ROIPolicyContext(
        seed=int(args.seed),
        legacy_calibration_csv=args.legacy_calibration_csv,
        cnn_calibration_csv=args.cnn_calibration_csv,
        cellness_predictor=args.cellness_predictor,
        cellness_model_path=args.cellness_model_path,
        predictor_mode=args.predictor_mode,
        dummy_threshold=float(args.dummy_threshold),
    )

    jobs = expand_policy_jobs(args.roi_policies, [int(v) for v in args.k_values])
    checks: list[dict[str, object]] = []
    errors: list[str] = []
    coverage_counts: dict[str, list[int]] = {}

    for policy_name, k in jobs:
        label = policy_label(policy_name, k)
        coverage_counts[label] = []
        for fov_id, g in sample_df.groupby('fov_id', sort=True):
            result = select_rois_for_policy(g, policy_name=policy_name, ctx=ctx, k=k)
            available = int(result.available)
            n_avail = int(len(g))
            n_sel = int(len(result.selected_df)) if result.available else 0
            coverage_counts[label].append(int(n_sel > 0))
            checks.append(
                {
                    'roi_policy': label,
                    'fov_id': str(fov_id),
                    'available': available,
                    'n_candidate': n_avail,
                    'n_selected': n_sel,
                    'warning': result.warning or '',
                    'backend': result.backend or '',
                }
            )
            if not result.available:
                continue
            missing_cols = sorted(EXPECTED_SELECTION_COLUMNS.difference(result.selected_df.columns))
            if missing_cols:
                errors.append(f'{label} missing selection columns on {fov_id}: {missing_cols}')
            if policy_name == 'center_top1' and n_sel != 1:
                errors.append(f'center_top1 selected {n_sel} ROIs on {fov_id}; expected 1')
            if policy_name == 'all_rois' and n_sel != n_avail:
                errors.append(f'all_rois selected {n_sel}/{n_avail} ROIs on {fov_id}')
            if policy_name == 'random_k':
                expected = min(int(k or 1), n_avail)
                if n_sel != expected:
                    errors.append(f'{label} selected {n_sel} ROIs on {fov_id}; expected {expected}')
            if result.selected_df['selected'].nunique() != 1 or int(result.selected_df['selected'].iloc[0]) != 1:
                errors.append(f'{label} returned malformed selected flags on {fov_id}')

    summary_rows = []
    for label, vals in coverage_counts.items():
        rate = float(np.mean(vals)) if vals else np.nan
        summary_rows.append({'roi_policy': label, 'sampled_fovs': int(len(vals)), 'selection_success_rate': rate})
        if np.isfinite(rate) and rate < 0.8 and not label.startswith('oracle_best_single_roi'):
            errors.append(f'{label} selected at least one ROI for only {rate:.1%} of sampled FOVs')

    if args.require_evaluation_outputs:
        errors.extend(_validate_eval_outputs(paths))

    checks_df = pd.DataFrame(checks)
    summary_df = pd.DataFrame(summary_rows)
    out_csv = paths.eval_dir / 'sanity_check_roi_ablation.csv'
    out_json = paths.eval_dir / 'sanity_check_roi_ablation.json'
    checks_df.to_csv(out_csv, index=False)
    payload = {
        'track': args.track,
        'sample_fovs': sample_fovs,
        'summary': summary_rows,
        'errors': errors,
    }
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    if errors:
        raise SystemExit('ROI ablation sanity checks failed:\n- ' + '\n- '.join(errors))

    log(paths, f'ROI ablation sanity checks passed: {out_csv}')


if __name__ == '__main__':
    main()
