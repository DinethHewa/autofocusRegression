#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_utils import get_paths


REQUIRED_MAIN_COLS = {
    'method', 'family', 'evaluation_scope', 'available', 'training_required', 'input_scope',
    'mae_um', 'rmse_um', 'median_abs_err_um', 'p95_abs_err_um', 'bias_um', 'within_0p5um_pct',
    'within_1um_pct', 'within_2um_pct', 'catastrophic_wrong_direction_pct', 'uncertain_pct',
    'runtime_ms_per_fov', 'model_size_mb', 'mean_inputs_used_per_fov', 'n_fov', 'notes',
}


def main() -> None:
    ap = argparse.ArgumentParser(description='Sanity checks for baseline-comparison outputs')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--out-dir', default='')
    args = ap.parse_args()

    paths = get_paths(args.track)
    out_dir = Path(args.out_dir) if args.out_dir else paths.eval_dir

    required_files = [
        out_dir / 'baseline_comparison_results.csv',
        out_dir / 'baseline_by_dataset.csv',
        out_dir / 'baseline_near_focus.csv',
        out_dir / 'baseline_pairwise_tests.csv',
        out_dir / 'baseline_efficiency.csv',
        out_dir / 'baseline_failure_cases.csv',
        out_dir / 'baseline_qualitative_panel.csv',
    ]
    missing = [str(p) for p in required_files if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing baseline outputs:\n' + '\n'.join(missing))

    main_df = pd.read_csv(out_dir / 'baseline_comparison_results.csv', low_memory=False)
    missing_cols = REQUIRED_MAIN_COLS.difference(main_df.columns)
    if missing_cols:
        raise ValueError(f'baseline_comparison_results.csv missing columns: {sorted(missing_cols)}')

    if 'proposed_two_stage_roi' not in set(main_df['method'].astype(str)):
        raise ValueError('proposed_two_stage_roi row is missing from baseline comparison results')

    avail = main_df[pd.to_numeric(main_df['available'], errors='coerce').fillna(0) > 0].copy()
    if avail.empty:
        raise ValueError('No available baseline rows found in baseline_comparison_results.csv')

    finite_mae = pd.to_numeric(avail['mae_um'], errors='coerce')
    if not np.isfinite(finite_mae).any():
        raise ValueError('No finite MAE values found for available baseline rows')

    shared_split_dir = paths.sign_dir.parent / 'baselines' / '_shared_splits'
    direct_split_dir = paths.sign_dir.parent / 'baselines' / 'direct_regression' / 'splits'
    if shared_split_dir.is_dir() and direct_split_dir.is_dir():
        for split_name in ['train', 'val', 'test']:
            shared_path = shared_split_dir / f'{split_name}.csv'
            direct_path = direct_split_dir / f'{split_name}.csv'
            if shared_path.is_file() and direct_path.is_file():
                shared = set(pd.read_csv(shared_path, usecols=['roi_uid'])['roi_uid'].astype(str))
                direct = set(pd.read_csv(direct_path, usecols=['roi_uid'])['roi_uid'].astype(str))
                if shared != direct:
                    raise ValueError(f'Split mismatch for {split_name}: shared={len(shared)} direct={len(direct)} differing={len(shared.symmetric_difference(direct))}')

    by_dataset = pd.read_csv(out_dir / 'baseline_by_dataset.csv', low_memory=False)
    if not {'method', 'dataset', 'mae_um', 'n_fov'}.issubset(by_dataset.columns):
        raise ValueError('baseline_by_dataset.csv is missing required columns')

    near_df = pd.read_csv(out_dir / 'baseline_near_focus.csv', low_memory=False)
    if not {'method', 'bin_name', 'mae_um', 'n'}.issubset(near_df.columns):
        raise ValueError('baseline_near_focus.csv is missing required columns')

    print(f'[DONE] Baseline sanity checks passed for track={args.track}: {out_dir}')


if __name__ == '__main__':
    main()
