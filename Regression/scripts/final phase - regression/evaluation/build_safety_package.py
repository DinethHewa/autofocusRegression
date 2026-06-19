#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from claim_safety_utils import (
    compute_frame_metrics,
    compute_safety_fields,
    load_method_frames,
    manuscript_paths,
    method_meta,
    save_eval_and_table,
)
from evaluation_utils import save_plot


METHODS = [
    'proposed_two_stage_roi',
    'direct_single_stage_regression',
    'classical_focus_best',
    'center_crop_inference_only',
    'full_field_tiled_proxy',
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build a manuscript-first safety package.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--save-latex', action='store_true')
    return ap.parse_args()


def _summary_df(track: str) -> pd.DataFrame:
    frames = load_method_frames(track, METHODS)
    rows = []
    for method in METHODS:
        if method not in frames:
            continue
        frame = frames[method]
        m = compute_frame_metrics(frame)
        s = compute_safety_fields(frame, near_threshold=1.0, far_threshold=2.0)
        meta = method_meta(method)
        rows.append({
            'method': method,
            'scope_label': meta['scope_label'],
            'mae_um': m['mae_um'],
            'catastrophic_wrong_direction_pct': m['catastrophic_wrong_direction_pct'],
            'bias_um': m['bias_um'],
            'uncertain_pct': m['uncertain_pct'],
            'near_focus_wrong_direction_pct': s['near_focus_wrong_direction_pct'],
            'far_focus_wrong_direction_pct': s['far_focus_wrong_direction_pct'],
            'note': meta['notes'],
        })
    return pd.DataFrame(rows).sort_values(['catastrophic_wrong_direction_pct', 'mae_um']).reset_index(drop=True)


def _plot_frontier(df: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(df['catastrophic_wrong_direction_pct'], df['mae_um'], s=75, color='#0f766e')
    for _, row in df.iterrows():
        color = '#991b1b' if row['method'] == 'proposed_two_stage_roi' else '#0f172a'
        ax.annotate(str(row['method']), (row['catastrophic_wrong_direction_pct'], row['mae_um']), fontsize=8, color=color, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Catastrophic wrong-direction rate (%)')
    ax.set_ylabel('MAE (µm)')
    ax.set_title('Safety vs MAE tradeoff across available methods')
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    summary = _summary_df(args.track)
    save_eval_and_table(
        summary,
        mp.eval_dir / 'safety_summary.csv',
        mp.tables_dir / 'Table_SafetySummary.csv',
        save_latex_flag=bool(args.save_latex),
    )
    _plot_frontier(summary, mp.eval_dir / 'safety_vs_mae_frontier.png')


if __name__ == '__main__':
    main()
