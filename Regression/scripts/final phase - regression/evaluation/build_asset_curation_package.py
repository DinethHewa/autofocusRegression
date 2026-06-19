#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from claim_safety_utils import manuscript_paths, markdown_table, write_markdown
from evaluation_utils import save_table


CURATION = {
    'Table_EndToEnd.csv': ('main', 'Primary end-to-end quantitative evidence.'),
    'Table_FairArchitectureComparison.csv': ('main', 'Fair shared-subset architecture comparison.'),
    'Table_InputProbeComparison.csv': ('main', 'Input-policy probe table with scope-correct labeling.'),
    'Table_ROI_Ablation_Regression.csv': ('main', 'Primary ROI-policy overview.'),
    'Table_ROI_Ablation_ByDataset.csv': ('main', 'Shows ROI-policy impact where PBS/BMA are hardest.'),
    'Table_DomainGapSummary.csv': ('main', 'Strongest ROI-selection claim support: reduced domain gap.'),
    'Table_SafetySummary.csv': ('main', 'Strongest safety claim support: low directional failure risk.'),
    'Table_PairedArchitectureComparison.csv': ('main', 'Required to keep architecture claims fair.'),
    'domain_gap_reduction_plot.png': ('main', 'Figure version of the domain-gap result.'),
    'safety_vs_mae_frontier.png': ('main', 'Figure version of the MAE-vs-safety tradeoff.'),
    'Table_StageA.csv': ('supplement', 'Supports the sign-classification component.'),
    'Table_StageB.csv': ('supplement', 'Supports the branch-wise regression component.'),
    'Table_Runtime.csv': ('supplement', 'Useful runtime detail, but not a headline manuscript table.'),
    'Table_ROI_NearFocus.csv': ('supplement', 'Important near-focus context.'),
    'Table_Baseline_ByDataset.csv': ('supplement', 'Contextual baseline detail; fair architecture claim is elsewhere.'),
    'Table_Baseline_NearFocus.csv': ('supplement', 'Contextual baseline near-focus detail.'),
    'Table_Baseline_Stats.csv': ('supplement', 'Raw baseline stats table; fair paired table is safer for the main paper.'),
    'Table_MacroVsWeighted.csv': ('supplement', 'Forces transparent reporting of WBC dominance.'),
    'Table_ClaimEvidence.csv': ('supplement', 'Useful appendix-style claim audit.'),
    'Table_AssetCuration.csv': ('supplement', 'Package-level provenance and routing table.'),
    'Table_Baseline_Comparison.csv': ('exclude', 'Mixed-scope overview table; should not be the main fairness table.'),
    'Table_Ablation.csv': ('exclude', 'Contains unavailable Stage A/Stage B ablation rows.'),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build main-paper vs supplement asset curation package.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--save-latex', action='store_true')
    return ap.parse_args()


def _load_assets(track: str) -> pd.DataFrame:
    mp = manuscript_paths(track)
    audit_df = pd.read_csv(mp.eval_dir / 'output_audit_report.csv', low_memory=False)
    rows = []
    for _, row in audit_df.iterrows():
        name = str(row['asset_name'])
        section, why = CURATION.get(name, ('appendix', 'Useful metadata or support file not intended for the main narrative.'))
        if str(row['classification']) == 'stale_or_inconsistent':
            section = 'exclude'
            why = str(row['note'])
        rows.append({
            'asset_name': name,
            'path': str(row['path']),
            'source_classification': str(row['classification']),
            'recommended_section': section,
            'justification_for_section': why,
        })
    return pd.DataFrame(rows).sort_values(['recommended_section', 'asset_name']).reset_index(drop=True)


def _md(df: pd.DataFrame, track: str) -> str:
    lines = [
        f'# Asset Curation: {track}',
        '',
        'This file routes generated assets into main-paper, supplement, appendix, or exclude buckets.',
        '',
    ]
    for section in ['main', 'supplement', 'appendix', 'exclude']:
        sub = df[df['recommended_section'] == section].copy()
        lines.extend([
            f'## {section}',
            '',
            markdown_table(sub[['asset_name', 'source_classification', 'justification_for_section']], max_rows=60) if not sub.empty else 'No assets routed here.',
            '',
        ])
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    df = _load_assets(args.track)
    save_table(df, mp.eval_dir / 'asset_curation.csv', save_latex=False)
    save_table(df, mp.tables_dir / 'Table_AssetCuration.csv', save_latex=bool(args.save_latex))
    write_markdown(mp.eval_dir / 'asset_curation.md', _md(df, args.track))


if __name__ == '__main__':
    main()
