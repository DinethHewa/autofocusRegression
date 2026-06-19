#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from claim_safety_utils import load_wide_prediction_frame, manuscript_paths, save_eval_and_table


PAIR_SPECS = [
    ('proposed_two_stage_roi', 'direct_single_stage_regression'),
    ('proposed_two_stage_roi', 'classical_focus_best'),
    ('proposed_two_stage_roi', 'center_crop_inference_only'),
    ('proposed_two_stage_roi', 'full_field_tiled_proxy'),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build paired-subset fairness package for method comparisons.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--save-latex', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def _metrics_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred_df.empty:
        return pd.DataFrame()
    for (method_a, method_b), g in pred_df.groupby(['method_a', 'method_b'], sort=False):
        for side, pred_col, abs_col, signed_col, wrong_col in [
            (method_a, method_a, 'abs_error_a_um', 'signed_error_a_um', 'wrong_direction_a'),
            (method_b, method_b, 'abs_error_b_um', 'signed_error_b_um', 'wrong_direction_b'),
        ]:
            abs_err = pd.to_numeric(g[abs_col], errors='coerce').to_numpy(dtype=float)
            signed = pd.to_numeric(g[signed_col], errors='coerce').to_numpy(dtype=float)
            rows.append({
                'pair_label': f'{method_a}__vs__{method_b}',
                'method': side,
                'shared_support_n': int(len(g)),
                'mae_um': float(np.nanmean(abs_err)),
                'rmse_um': float(np.sqrt(np.nanmean(np.square(signed)))),
                'bias_um': float(np.nanmean(signed)),
                'within_1um_pct': float(np.nanmean(abs_err <= 1.0) * 100.0),
                'catastrophic_wrong_direction_pct': float(np.nanmean(pd.to_numeric(g[wrong_col], errors='coerce').fillna(0).to_numpy(dtype=float) > 0) * 100.0),
            })
    return pd.DataFrame(rows)


def _build_prediction_rows(track: str) -> pd.DataFrame:
    methods = sorted({m for pair in PAIR_SPECS for m in pair})
    wide = load_wide_prediction_frame(track, methods)
    pred_rows = []
    for method_a, method_b in PAIR_SPECS:
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
    return pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def _claim_text(delta_mae_um: float, delta_wrong_direction_pct: float) -> str:
    if np.isnan(delta_mae_um) or np.isnan(delta_wrong_direction_pct):
        return 'Insufficient shared support for a fair paired claim.'
    if delta_mae_um > 0 and delta_wrong_direction_pct < 0:
        return 'Mixed tradeoff: comparison method has lower MAE, while proposed has lower wrong-direction risk.'
    if delta_mae_um < 0 and delta_wrong_direction_pct > 0:
        return 'Mixed tradeoff: proposed has lower MAE, while comparison method has lower wrong-direction risk.'
    if delta_mae_um < 0 and delta_wrong_direction_pct <= 0:
        return 'Proposed is better on both MAE and wrong-direction risk on the shared subset.'
    if delta_mae_um > 0 and delta_wrong_direction_pct >= 0:
        return 'Comparison method is better on both MAE and wrong-direction risk on the shared subset.'
    return 'Methods are effectively tied on the shared subset.'


def _tests_from_predictions(mp, pred_df: pd.DataFrame) -> pd.DataFrame:
    baseline_pairwise = mp.eval_dir / 'baseline_pairwise_tests.csv'
    if baseline_pairwise.is_file():
        pairwise_df = pd.read_csv(baseline_pairwise, low_memory=False)
    else:
        pairwise_df = pd.DataFrame(columns=['method_a', 'method_b', 'delta', 'ci_low', 'ci_high', 'p_value', 'significant_after_correction'])
    rows = []
    for (method_a, method_b), g in pred_df.groupby(['method_a', 'method_b'], sort=False):
        match = pairwise_df[(pairwise_df['method_a'] == method_a) & (pairwise_df['method_b'] == method_b)].copy()
        delta_mae = float(match['delta'].iloc[0]) if not match.empty else float(np.nanmean(g['abs_error_a_um']) - np.nanmean(g['abs_error_b_um']))
        ci_low = float(match['ci_low'].iloc[0]) if not match.empty else np.nan
        ci_high = float(match['ci_high'].iloc[0]) if not match.empty else np.nan
        p_value = float(match['p_value'].iloc[0]) if not match.empty else np.nan
        significant = int(match['significant_after_correction'].iloc[0]) if not match.empty else 0
        wrong_a = float(np.nanmean(pd.to_numeric(g['wrong_direction_a'], errors='coerce').fillna(0).to_numpy(dtype=float) > 0) * 100.0)
        wrong_b = float(np.nanmean(pd.to_numeric(g['wrong_direction_b'], errors='coerce').fillna(0).to_numpy(dtype=float) > 0) * 100.0)
        delta_wrong = wrong_a - wrong_b
        rows.append({
            'method_a': method_a,
            'method_b': method_b,
            'shared_support_n': int(len(g)),
            'delta_mae_um': delta_mae,
            'delta_wrong_direction_pct': delta_wrong,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_value,
            'significant_after_correction': significant,
            'scope_label': 'paired_shared_subset',
            'support_relation': 'shared_intersection_only',
            'fair_claim_summary': _claim_text(delta_mae, delta_wrong),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    pred_df = _build_prediction_rows(args.track)
    metrics_df = _metrics_from_predictions(pred_df)
    test_df = _tests_from_predictions(mp, pred_df)
    save_eval_and_table(pred_df, mp.eval_dir / 'paired_subset_predictions.csv', None, save_latex_flag=False)
    save_eval_and_table(metrics_df, mp.eval_dir / 'paired_subset_metrics.csv', None, save_latex_flag=False)
    save_eval_and_table(test_df, mp.eval_dir / 'paired_subset_tests.csv', None, save_latex_flag=False)
    save_eval_and_table(
        test_df,
        mp.eval_dir / 'paired_architecture_comparison.csv',
        mp.tables_dir / 'Table_PairedArchitectureComparison.csv',
        save_latex_flag=bool(args.save_latex),
    )


if __name__ == '__main__':
    main()
