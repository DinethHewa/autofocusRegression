#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from claim_safety_utils import (
    fair_method_rows,
    load_method_frames,
    macro_weighted_summary,
    manuscript_paths,
    method_meta,
    save_eval_and_table,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build fair-scope comparison tables for manuscript use.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--save-latex', action='store_true')
    return ap.parse_args()


def _pairwise_df(track: str) -> pd.DataFrame:
    mp = manuscript_paths(track)
    path = mp.eval_dir / 'baseline_pairwise_tests.csv'
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if 'delta_mae_um' not in df.columns and 'delta' in df.columns:
        df = df.copy()
        df['delta_mae_um'] = pd.to_numeric(df['delta'], errors='coerce')
    return df


def _fair_claim_summary(delta_mae_um: float, delta_wrong_direction_pct: float) -> str:
    if np.isnan(delta_mae_um) or np.isnan(delta_wrong_direction_pct):
        return 'No paired comparison available.'
    if delta_mae_um > 0 and delta_wrong_direction_pct < 0:
        return 'Mixed tradeoff: direct regression has lower MAE, while the proposed method has lower wrong-direction risk.'
    if delta_mae_um < 0 and delta_wrong_direction_pct > 0:
        return 'Mixed tradeoff: the proposed method has lower MAE, while the comparison has lower wrong-direction risk.'
    if delta_mae_um < 0 and delta_wrong_direction_pct <= 0:
        return 'The proposed method is better on both MAE and wrong-direction risk on the shared subset.'
    if delta_mae_um > 0 and delta_wrong_direction_pct >= 0:
        return 'The comparison method is better on both MAE and wrong-direction risk on the shared subset.'
    return 'Methods are effectively tied on the shared subset.'


def _fair_architecture_table(track: str) -> pd.DataFrame:
    rows, shared_n = fair_method_rows(track, ['proposed_two_stage_roi', 'direct_single_stage_regression'], required_common=True)
    pair_df = _pairwise_df(track)
    match = pair_df[
        (pair_df['method_a'] == 'proposed_two_stage_roi') &
        (pair_df['method_b'] == 'direct_single_stage_regression')
    ] if not pair_df.empty else pd.DataFrame()
    delta_mae = float(match['delta_mae_um'].iloc[0]) if not match.empty else np.nan
    proposed = rows[rows['method'] == 'proposed_two_stage_roi']
    direct = rows[rows['method'] == 'direct_single_stage_regression']
    delta_wrong = np.nan
    if not proposed.empty and not direct.empty:
        delta_wrong = float(proposed['catastrophic_wrong_direction_pct'].iloc[0] - direct['catastrophic_wrong_direction_pct'].iloc[0])
    fair_note = _fair_claim_summary(delta_mae, delta_wrong)
    rows['scope_label'] = f'paired_shared_subset_n={shared_n}'
    rows.loc[rows['method'] == 'proposed_two_stage_roi', 'note'] = rows.loc[rows['method'] == 'proposed_two_stage_roi', 'note'] + ' ' + fair_note
    rows.loc[rows['method'] == 'direct_single_stage_regression', 'note'] = 'Fair shared-subset row for direct regression; do not compare with pooled proposed rows outside this table.'
    return rows


def _input_probe_table(track: str) -> pd.DataFrame:
    rows, shared_n = fair_method_rows(track, ['proposed_two_stage_roi', 'center_crop_inference_only', 'full_field_tiled_proxy'], required_common=True)
    pair_df = _pairwise_df(track)
    sig_map = {row['method_b']: int(row['significant_after_correction']) for _, row in pair_df.iterrows()}
    type_map = {
        'proposed_two_stage_roi': 'full_pipeline',
        'center_crop_inference_only': 'inference_only_probe',
        'full_field_tiled_proxy': 'proxy_probe',
    }
    rows = rows.rename(columns={'note': 'base_note'})
    rows['method_type'] = rows['method'].map(type_map).fillna('context')
    rows['scope_label'] = f'paired_common_support_n={shared_n}'
    rows['significant_vs_proposed'] = rows['method'].map(sig_map).fillna(0).astype(int)
    note_map = {
        'proposed_two_stage_roi': 'Reference row for the fixed downstream pipeline.',
        'center_crop_inference_only': 'Inference-only input probe using the center-most ROI. Not a retrained architecture replacement.',
        'full_field_tiled_proxy': 'Proxy baseline using all tiled ROIs with fixed downstream models. Not a true full-image learner.',
    }
    rows['note'] = rows['method'].map(note_map)
    cols = ['method', 'method_type', 'scope_label', 'n_fov', 'mae_um', 'rmse_um', 'catastrophic_wrong_direction_pct', 'significant_vs_proposed', 'note']
    return rows[cols].copy()


def _macro_weighted_table(track: str) -> pd.DataFrame:
    frames = load_method_frames(track)
    rows = []
    for method in ['proposed_two_stage_roi', 'direct_single_stage_regression', 'classical_focus_best', 'center_crop_inference_only', 'full_field_tiled_proxy']:
        if method not in frames:
            continue
        summary = macro_weighted_summary(frames[method])
        rows.append({
            'method': method,
            'scope_label': method_meta(method)['scope_label'],
            'weighted_mae_um': summary['weighted_mae_um'],
            'macro_dataset_mae_um': summary['macro_dataset_mae_um'],
            'domain_gap_mae': summary['domain_gap_mae'],
            'largest_dataset_share_pct': summary['largest_dataset_share_pct'],
        })
    return pd.DataFrame(rows)


def _routing_table() -> pd.DataFrame:
    rows = [
        {'asset_name': 'Table_EndToEnd.csv', 'recommended_section': 'main', 'reason': 'Primary pooled performance, bias, and catastrophic wrong-direction evidence.'},
        {'asset_name': 'Table_FairArchitectureComparison.csv', 'recommended_section': 'main', 'reason': 'Fair shared-subset architecture comparison between proposed and direct regression.'},
        {'asset_name': 'Table_InputProbeComparison.csv', 'recommended_section': 'main', 'reason': 'Scope-correct comparison against center-crop and full-field proxy input probes.'},
        {'asset_name': 'Table_ROI_Ablation_Regression.csv', 'recommended_section': 'main', 'reason': 'Primary ROI-policy overview table; should be read with domain-gap framing.'},
        {'asset_name': 'Table_ROI_Ablation_ByDataset.csv', 'recommended_section': 'main', 'reason': 'Shows that ROI-policy value is strongest on PBS/BMA rather than only on pooled MAE.'},
        {'asset_name': 'Table_DomainGapSummary.csv', 'recommended_section': 'main', 'reason': 'Directly supports the strongest ROI-selection claim: improved cross-domain consistency.'},
        {'asset_name': 'Table_SafetySummary.csv', 'recommended_section': 'main', 'reason': 'Elevates directional safety and low catastrophic failure risk into a first-class result.'},
        {'asset_name': 'Table_PairedArchitectureComparison.csv', 'recommended_section': 'main', 'reason': 'Required to prevent misuse of mixed-scope baseline overview rows.'},
        {'asset_name': 'Table_StageA.csv', 'recommended_section': 'supplement', 'reason': 'Important supporting evidence for the sign classifier but not a main-paper table.'},
        {'asset_name': 'Table_StageB.csv', 'recommended_section': 'supplement', 'reason': 'Important supporting evidence for branch-wise regression but not a headline table.'},
        {'asset_name': 'Table_Runtime.csv', 'recommended_section': 'supplement', 'reason': 'Useful supporting efficiency detail for the full pipeline.'},
        {'asset_name': 'Table_Baseline_ByDataset.csv', 'recommended_section': 'supplement', 'reason': 'Contextual baseline table with mixed baseline scopes; fair architecture claims belong elsewhere.'},
        {'asset_name': 'Table_Baseline_NearFocus.csv', 'recommended_section': 'supplement', 'reason': 'Useful for context, but architecture claims should rely on shared-subset safety/fairness tables.'},
        {'asset_name': 'Table_Baseline_Stats.csv', 'recommended_section': 'supplement', 'reason': 'Contains mixed-scope paired tests; the fair paired architecture table is the safer main-paper asset.'},
        {'asset_name': 'Table_Baseline_Comparison.csv', 'recommended_section': 'exclude', 'reason': 'Mixed-scope overview table; safe as context only after curation, not as a primary fairness table.'},
        {'asset_name': 'Table_Ablation.csv', 'recommended_section': 'exclude', 'reason': 'Contains unavailable Stage A/Stage B ablation rows and should not be cited as a complete component ablation summary.'},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    fair_arch = _fair_architecture_table(args.track)
    save_eval_and_table(
        fair_arch,
        mp.eval_dir / 'fair_architecture_comparison.csv',
        mp.tables_dir / 'Table_FairArchitectureComparison.csv',
        save_latex_flag=bool(args.save_latex),
    )

    input_probe = _input_probe_table(args.track)
    save_eval_and_table(
        input_probe,
        mp.eval_dir / 'input_probe_comparison.csv',
        mp.tables_dir / 'Table_InputProbeComparison.csv',
        save_latex_flag=bool(args.save_latex),
    )

    macro_df = _macro_weighted_table(args.track)
    save_eval_and_table(
        macro_df,
        mp.eval_dir / 'macro_vs_weighted.csv',
        mp.tables_dir / 'Table_MacroVsWeighted.csv',
        save_latex_flag=bool(args.save_latex),
    )

    routing_df = _routing_table()
    save_eval_and_table(
        routing_df,
        mp.eval_dir / 'paper_main_vs_supplement_routing.csv',
        mp.tables_dir / 'Table_PaperMainVsSupplementRouting.csv',
        save_latex_flag=bool(args.save_latex),
    )


if __name__ == '__main__':
    main()
