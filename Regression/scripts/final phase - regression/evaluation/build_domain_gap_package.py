#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from claim_safety_utils import manuscript_paths, save_eval_and_table
from evaluation_utils import save_plot


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build a manuscript-first domain-gap package from ROI-ablation outputs.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--save-latex', action='store_true')
    return ap.parse_args()


def _summary_df(track: str) -> pd.DataFrame:
    mp = manuscript_paths(track)
    perf = pd.read_csv(mp.eval_dir / 'roi_ablation_to_regression_performance.csv', low_memory=False)
    by_ds = pd.read_csv(mp.eval_dir / 'roi_ablation_by_dataset.csv', low_memory=False)
    rows = []
    center_gap = float(by_ds[by_ds['roi_policy'].astype(str) == 'center_top1'].groupby('roi_policy')['mae_um'].agg(lambda s: float(np.nanmax(s) - np.nanmin(s))).iloc[0])
    all_gap = float(by_ds[by_ds['roi_policy'].astype(str) == 'all_rois'].groupby('roi_policy')['mae_um'].agg(lambda s: float(np.nanmax(s) - np.nanmin(s))).iloc[0])
    for policy, g in by_ds.groupby('roi_policy', sort=True):
        mae = pd.to_numeric(g['mae_um'], errors='coerce').to_numpy(dtype=float)
        perf_row = perf[perf['roi_policy'].astype(str) == str(policy)]
        weighted = float(pd.to_numeric(perf_row['mae_um'], errors='coerce').iloc[0]) if not perf_row.empty else np.nan
        mean_inputs = float(pd.to_numeric(perf_row['K_effective_mean'], errors='coerce').iloc[0]) if not perf_row.empty else np.nan
        latency = float(pd.to_numeric(perf_row['latency_ms_per_fov'], errors='coerce').iloc[0]) if not perf_row.empty else np.nan
        gap = float(np.nanmax(mae) - np.nanmin(mae)) if len(mae) else np.nan
        rows.append({
            'roi_policy': str(policy),
            'weighted_mae_um': weighted,
            'macro_mae_um': float(np.nanmean(mae)) if len(mae) else np.nan,
            'best_dataset_mae': float(np.nanmin(mae)) if len(mae) else np.nan,
            'worst_dataset_mae': float(np.nanmax(mae)) if len(mae) else np.nan,
            'domain_gap_mae': gap,
            'mean_inputs_used_per_fov': mean_inputs,
            'latency_proxy_ms_per_fov': latency,
            'relative_gap_reduction_vs_center_top1_pct': float((center_gap - gap) / center_gap * 100.0) if np.isfinite(center_gap) and center_gap > 0 else np.nan,
            'relative_gap_reduction_vs_all_rois_pct': float((all_gap - gap) / all_gap * 100.0) if np.isfinite(all_gap) and all_gap > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(['domain_gap_mae', 'weighted_mae_um', 'latency_proxy_ms_per_fov']).reset_index(drop=True)


def _plot_reduction(df: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    work = df.copy().sort_values('domain_gap_mae')
    ax.scatter(work['weighted_mae_um'], work['domain_gap_mae'], s=70, color='#0f766e')
    for _, row in work.iterrows():
        color = '#991b1b' if row['roi_policy'] == 'hybrid_proposed' else '#0f172a'
        ax.annotate(str(row['roi_policy']), (row['weighted_mae_um'], row['domain_gap_mae']), fontsize=8, color=color, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Weighted MAE (µm)')
    ax.set_ylabel('Domain gap MAE (worst - best)')
    ax.set_title('ROI policy: pooled error vs domain gap')
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def _plot_waterfall(df: pd.DataFrame, out_path):
    work = df.copy().sort_values('relative_gap_reduction_vs_center_top1_pct', ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ['#991b1b' if p == 'hybrid_proposed' else '#0369a1' if p == 'cnn_adaptive' else '#475569' for p in work['roi_policy']]
    ax.bar(work['roi_policy'], work['relative_gap_reduction_vs_center_top1_pct'], color=colors)
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_ylabel('Relative domain-gap reduction vs center_top1 (%)')
    ax.set_title('Adaptive ROI policies reduce cross-domain gap more than naive center selection')
    ax.tick_params(axis='x', rotation=50)
    ax.grid(True, axis='y', alpha=0.3)
    save_plot(fig, out_path)


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    summary = _summary_df(args.track)
    save_eval_and_table(
        summary,
        mp.eval_dir / 'domain_gap_summary.csv',
        mp.tables_dir / 'Table_DomainGapSummary.csv',
        save_latex_flag=bool(args.save_latex),
    )
    _plot_reduction(summary, mp.eval_dir / 'domain_gap_reduction_plot.png')
    _plot_waterfall(summary, mp.eval_dir / 'domain_gap_waterfall.png')


if __name__ == '__main__':
    main()
