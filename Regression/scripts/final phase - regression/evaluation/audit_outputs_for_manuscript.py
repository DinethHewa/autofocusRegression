#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from claim_safety_utils import (
    MAIN_PAPER_PRIORITY,
    QUARANTINE_PRIORITY,
    SUPPLEMENT_PRIORITY,
    load_core_outputs,
    manuscript_paths,
    markdown_table,
    source_asset_inventory,
    stale_paper_package_paths,
    write_json,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Audit saved outputs and classify them for manuscript safety.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    return ap.parse_args()


def _classify_asset(path: str, core: dict, asset_name: str) -> tuple[str, str, str]:
    p = Path(path)
    if not p.is_file():
        return 'stale_or_inconsistent', 'missing output', 'File is missing and cannot be treated as manuscript evidence.'

    if asset_name in {'Table_Ablation.csv', 'ablation_results.csv'}:
        ablation_df = core.get('ablation_results.csv', pd.DataFrame())
        if not ablation_df.empty and 'available' in ablation_df.columns:
            if (pd.to_numeric(ablation_df['available'], errors='coerce').fillna(0) == 0).any():
                return 'stale_or_inconsistent', 'contains unavailable Stage A/Stage B ablations', 'This table includes unavailable ablation rows and should not be cited as a complete component ablation summary.'

    if asset_name in {'Table_Baseline_Comparison.csv', 'baseline_comparison_results.csv'}:
        return 'paired_scope_only', 'mixed baseline scopes', 'This overview mixes pooled proposed/input-probe rows with held-out learned-baseline rows. Use paired-subset tables for architecture claims.'

    if asset_name in {'baseline_by_dataset.csv', 'Table_Baseline_ByDataset.csv', 'baseline_near_focus.csv', 'Table_Baseline_NearFocus.csv', 'baseline_pairwise_tests.csv', 'Table_Baseline_Stats.csv'}:
        return 'supplementary_only', 'mixed or restricted baseline scope', 'These baseline outputs are still useful context, but they should not carry the main architecture claim without paired-scope tables.'

    if asset_name in {'baseline_efficiency.csv', 'Table_Baseline_Efficiency.csv', 'baseline_efficiency_frontier.png'}:
        return 'proxy_only', 'contains proxy and inference-only runtime rows', 'Efficiency output includes proxy/inference-only methods and latency proxies that need explicit scope labeling.'

    if asset_name in {'baseline_dataset_mae.png', 'baseline_near_focus.png'}:
        return 'supplementary_only', 'baseline visualization with mixed scopes', 'Useful for supplement only; main claims should rely on fair-scope tables.'

    if asset_name in {'Table_ROI_Ablation_Regression.csv', 'roi_ablation_to_regression_performance.csv', 'Table_ROI_Ablation_ByDataset.csv', 'roi_ablation_by_dataset.csv', 'Table_ROI_NearFocus.csv', 'roi_ablation_near_focus.csv', 'Table_DomainGapReduction.csv', 'domain_gap_reduction.csv'}:
        return 'manuscript_safe', 'ROI outputs match current evidence posture', 'ROI outputs are safe when framed around domain-gap reduction and small pooled-MAE differences.'

    if asset_name in {'Table_EndToEnd.csv', 'end_to_end_metrics.json'}:
        return 'manuscript_safe', 'core end-to-end evidence', 'Primary evidence for overall accuracy, low bias, and low catastrophic wrong-direction rate.'

    if asset_name in {'Table_StageA.csv', 'stageA_metrics.json', 'Table_StageB.csv', 'stageB_metrics.json', 'Table_Runtime.csv', 'runtime_metrics.json', 'cross_dataset_results.csv'}:
        return 'supplementary_only', 'important supporting output', 'Use as supporting evidence or supplement, not as the sole headline claim.'

    return 'manuscript_safe', 'no audit issue detected', 'No specific manuscript-safety issue detected for this asset.'


def _build_report(track: str) -> pd.DataFrame:
    core = load_core_outputs(track)
    rows = []
    for item in source_asset_inventory(track):
        cls, issue, note = _classify_asset(item['path'], core, item['asset_name'])
        rows.append({
            'asset_name': item['asset_name'],
            'path': item['path'],
            'classification': cls,
            'issue': issue,
            'note': note,
        })
    for stale_path in stale_paper_package_paths(track):
        rows.append({
            'asset_name': stale_path.name,
            'path': str(stale_path),
            'classification': 'stale_or_inconsistent',
            'issue': 'pre-existing paper_package asset',
            'note': 'Existing paper-package asset predates the manuscript-safety layer and should be rebuilt under the upgraded package builder.',
        })
    return pd.DataFrame(rows).drop_duplicates(subset=['path']).sort_values(['classification', 'asset_name']).reset_index(drop=True)


def _build_lists(report_df: pd.DataFrame) -> tuple[dict, dict]:
    whitelist_items = []
    blacklist_items = []
    for _, row in report_df.iterrows():
        payload = {
            'asset_name': row['asset_name'],
            'path': row['path'],
            'classification': row['classification'],
            'note': row['note'],
        }
        if row['classification'] in {'manuscript_safe', 'supplementary_only', 'paired_scope_only', 'proxy_only'}:
            whitelist_items.append(payload)
        else:
            blacklist_items.append(payload)
    whitelist = {'allowed_assets': whitelist_items}
    blacklist = {'quarantined_assets': blacklist_items}
    return whitelist, blacklist


def _report_md(track: str, df: pd.DataFrame) -> str:
    lines = [
        f'# Output Audit Report: {track}',
        '',
        'This audit classifies saved outputs by manuscript safety. Files are not deleted; they are either whitelisted with scope restrictions or quarantined.',
        '',
    ]
    for cls in ['manuscript_safe', 'supplementary_only', 'paired_scope_only', 'proxy_only', 'stale_or_inconsistent']:
        sub = df[df['classification'] == cls].copy()
        lines.extend([
            f'## {cls}',
            '',
            markdown_table(sub[['asset_name', 'issue', 'note']], max_rows=40) if not sub.empty else 'No assets in this category.',
            '',
        ])
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    report_df = _build_report(args.track)
    report_path = mp.eval_dir / 'output_audit_report.csv'
    report_df.to_csv(report_path, index=False)
    write_markdown(mp.eval_dir / 'output_audit_report.md', _report_md(args.track, report_df))
    whitelist, blacklist = _build_lists(report_df)
    write_json(mp.eval_dir / 'manuscript_whitelist.json', whitelist)
    write_json(mp.eval_dir / 'manuscript_blacklist.json', blacklist)


if __name__ == '__main__':
    main()
