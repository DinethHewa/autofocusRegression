#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from claim_safety_utils import (
    load_core_outputs,
    load_wide_prediction_frame,
    manuscript_paths,
    markdown_table,
    save_eval_and_table,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build claim evidence matrix and claim guardrails from saved outputs.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--save-latex', action='store_true')
    return ap.parse_args()


def _claim_rows(track: str) -> pd.DataFrame:
    mp = manuscript_paths(track)
    core = load_core_outputs(track)
    end_m = core.get('end_to_end_metrics.json', {})
    cross_df = core.get('cross_dataset_results.csv', pd.DataFrame())
    roi_gap_df = core.get('domain_gap_reduction.csv', pd.DataFrame())
    ci_df = core.get('confidence_intervals.csv', pd.DataFrame())
    pair_tests_path = mp.eval_dir / 'baseline_pairwise_tests.csv'
    pair_tests = pd.read_csv(pair_tests_path, low_memory=False) if pair_tests_path.is_file() else pd.DataFrame()
    if not pair_tests.empty and 'delta_mae_um' not in pair_tests.columns and 'delta' in pair_tests.columns:
        pair_tests = pair_tests.copy()
        pair_tests['delta_mae_um'] = pd.to_numeric(pair_tests['delta'], errors='coerce')
    near_wide = load_wide_prediction_frame(track, ['proposed_two_stage_roi', 'direct_single_stage_regression'])

    macro_gap_note = ''
    if not cross_df.empty:
        end_mae = pd.to_numeric(cross_df['end_to_end_mae_um'], errors='coerce')
        if end_mae.notna().any():
            best_idx = end_mae.idxmin()
            worst_idx = end_mae.idxmax()
            macro_gap_note = f"Best dataset={cross_df.loc[best_idx, 'dataset']} ({end_mae.loc[best_idx]:.4f} um), worst dataset={cross_df.loc[worst_idx, 'dataset']} ({end_mae.loc[worst_idx]:.4f} um)."

    rows = []
    rows.append({
        'claim_id': 'C1',
        'claim_text': 'The smear-track system achieves strong overall autofocus-distance accuracy.',
        'claim_category': 'overall_accuracy',
        'support_status': 'supported' if end_m else 'unsupported',
        'primary_evidence_files': '; '.join([str(mp.eval_dir / 'end_to_end_metrics.json'), str(mp.tables_dir / 'Table_EndToEnd.csv')]),
        'required_scope': 'pooled_full_pipeline with macro caveat',
        'allowed_wording': 'The proposed system achieved strong overall autofocus-distance estimation performance on the smear track.',
        'forbidden_wording': 'The proposed system is universally superior across all microscopy settings.',
        'recommended_table_or_figure': 'Table_EndToEnd.csv',
        'notes': f"End-to-end MAE={end_m.get('mae_um', np.nan):.4f} um, RMSE={end_m.get('rmse_um', np.nan):.4f} um. Weighted pooled result should be paired with macro reporting.",
    })
    rows.append({
        'claim_id': 'C2',
        'claim_text': 'The system has very low catastrophic wrong-direction failure rate.',
        'claim_category': 'safety',
        'support_status': 'supported' if end_m else 'unsupported',
        'primary_evidence_files': '; '.join([str(mp.eval_dir / 'end_to_end_metrics.json'), str(mp.tables_dir / 'Table_EndToEnd.csv')]),
        'required_scope': 'pooled_full_pipeline',
        'allowed_wording': 'The proposed system maintained a very low catastrophic wrong-direction rate on the smear track.',
        'forbidden_wording': 'The system eliminates wrong-direction failures.',
        'recommended_table_or_figure': 'Table_EndToEnd.csv',
        'notes': f"Catastrophic wrong-direction rate={end_m.get('catastrophic_wrong_direction_rate', np.nan):.4f}%.",
    })
    rows.append({
        'claim_id': 'C3',
        'claim_text': 'WBC is the easiest smear domain, while PBS and BMA remain harder; macro reporting is necessary.',
        'claim_category': 'domain_generalization',
        'support_status': 'supported' if not cross_df.empty else 'unsupported',
        'primary_evidence_files': '; '.join([str(mp.eval_dir / 'cross_dataset_results.csv'), str(mp.tables_dir / 'Table_Baseline_ByDataset.csv')]),
        'required_scope': 'dataset-wise and macro-vs-weighted',
        'allowed_wording': 'Performance was strongest on WBC, whereas PBS and BMA remained harder, so macro and domain-gap reporting are necessary.',
        'forbidden_wording': 'The weighted pooled smear result alone demonstrates uniform cross-domain generalization.',
        'recommended_table_or_figure': 'Table_MacroVsWeighted.csv',
        'notes': macro_gap_note,
    })

    hybrid_row = roi_gap_df[roi_gap_df['roi_policy'].astype(str) == 'hybrid_proposed'] if not roi_gap_df.empty else pd.DataFrame()
    center_row = roi_gap_df[roi_gap_df['roi_policy'].astype(str) == 'center_top1'] if not roi_gap_df.empty else pd.DataFrame()
    support_roi_gap = (not hybrid_row.empty) and (not center_row.empty) and float(hybrid_row['gap_worst_minus_best'].iloc[0]) < float(center_row['gap_worst_minus_best'].iloc[0])
    rows.append({
        'claim_id': 'C4',
        'claim_text': 'Adaptive ROI policy matters mainly by reducing cross-domain consistency loss rather than by producing large pooled-MAE gains.',
        'claim_category': 'roi_selection',
        'support_status': 'supported' if support_roi_gap else 'mixed',
        'primary_evidence_files': '; '.join([str(mp.eval_dir / 'domain_gap_reduction.csv'), str(mp.tables_dir / 'Table_DomainGapReduction.csv')]),
        'required_scope': 'ROI policy comparison with domain-gap emphasis',
        'allowed_wording': 'Adaptive ROI selection reduced domain gap and matched all-ROI pooled error with fewer effective inputs.',
        'forbidden_wording': 'Adaptive ROI selection produced a large pooled-MAE improvement over all simpler policies.',
        'recommended_table_or_figure': 'Table_DomainGapSummary.csv',
        'notes': '' if hybrid_row.empty or center_row.empty else f"Hybrid gap={float(hybrid_row['gap_worst_minus_best'].iloc[0]):.4f}, center_top1 gap={float(center_row['gap_worst_minus_best'].iloc[0]):.4f}.",
    })

    classical_pair = pair_tests[(pair_tests['method_a'] == 'proposed_two_stage_roi') & (pair_tests['method_b'] == 'classical_focus_best')]
    rows.append({
        'claim_id': 'C5',
        'claim_text': 'The classical autofocus baseline is clearly inferior to the learned pipeline.',
        'claim_category': 'baseline_comparison',
        'support_status': 'supported' if not classical_pair.empty else 'mixed',
        'primary_evidence_files': '; '.join([str(mp.eval_dir / 'baseline_comparison_results.csv'), str(mp.tables_dir / 'Table_PairedArchitectureComparison.csv')]),
        'required_scope': 'paired shared-subset or clearly labeled held-out baseline scope',
        'allowed_wording': 'The classical focus baseline was clearly inferior under the tested smear-track setup.',
        'forbidden_wording': 'All classical autofocus methods are inferior in all microscopy domains.',
        'recommended_table_or_figure': 'Table_PairedArchitectureComparison.csv',
        'notes': '' if classical_pair.empty else f"Paired delta MAE={float(classical_pair['delta_mae_um'].iloc[0]):.4f} (proposed-classical); proposed is much better.",
    })

    direct_pair = pair_tests[(pair_tests['method_a'] == 'proposed_two_stage_roi') & (pair_tests['method_b'] == 'direct_single_stage_regression')]
    direct_status = 'mixed'
    direct_note = 'Fair shared-subset comparison required.'
    if not direct_pair.empty:
        d = direct_pair.iloc[0]
        direct_note = f"Shared-subset delta MAE={float(d['delta_mae_um']):.4f} (proposed-direct), delta wrong-direction={float(d['delta_wrong_direction_pct']):.4f} pp."
        direct_status = 'mixed'
    rows.append({
        'claim_id': 'C6',
        'claim_text': 'The proposed two-stage architecture is superior to direct single-stage regression.',
        'claim_category': 'architecture_comparison',
        'support_status': direct_status,
        'primary_evidence_files': '; '.join([str(mp.tables_dir / 'Table_FairArchitectureComparison.csv'), str(mp.tables_dir / 'Table_PairedArchitectureComparison.csv')]),
        'required_scope': 'paired shared-subset only',
        'allowed_wording': 'Compared with direct regression, the proposed system showed a mixed tradeoff between MAE and directional safety on the fair shared subset.',
        'forbidden_wording': 'The proposed architecture outperformed direct regression.',
        'recommended_table_or_figure': 'Table_FairArchitectureComparison.csv',
        'notes': direct_note,
    })

    center_pair = pair_tests[(pair_tests['method_a'] == 'proposed_two_stage_roi') & (pair_tests['method_b'] == 'center_crop_inference_only')]
    rows.append({
        'claim_id': 'C7',
        'claim_text': 'Content-aware ROI selection is superior to a fixed center-crop input policy on pooled error.',
        'claim_category': 'input_probe',
        'support_status': 'unsupported' if not center_pair.empty else 'mixed',
        'primary_evidence_files': '; '.join([str(mp.tables_dir / 'Table_InputProbeComparison.csv'), str(mp.eval_dir / 'baseline_pairwise_tests.csv')]),
        'required_scope': 'inference-only input probe; not an architecture comparison',
        'allowed_wording': 'Compared with a center-crop probe, content-aware ROI selection showed a small, scope-limited difference that should be interpreted as an input-policy probe rather than an architecture replacement.',
        'forbidden_wording': 'Content-aware ROI selection clearly outperformed center crop.',
        'recommended_table_or_figure': 'Table_InputProbeComparison.csv',
        'notes': '' if center_pair.empty else f"Paired delta MAE={float(center_pair['delta_mae_um'].iloc[0]):.4f}; current evidence does not support a clear pooled-MAE win.",
    })

    full_pair = pair_tests[(pair_tests['method_a'] == 'proposed_two_stage_roi') & (pair_tests['method_b'] == 'full_field_tiled_proxy')]
    rows.append({
        'claim_id': 'C8',
        'claim_text': 'Content-aware ROI selection is superior to a full-field tiled proxy on pooled error.',
        'claim_category': 'input_probe',
        'support_status': 'unsupported' if not full_pair.empty else 'mixed',
        'primary_evidence_files': '; '.join([str(mp.tables_dir / 'Table_InputProbeComparison.csv'), str(mp.eval_dir / 'baseline_pairwise_tests.csv')]),
        'required_scope': 'proxy baseline only; not a true end-to-end full-image learner',
        'allowed_wording': 'The full-field tiled proxy should be interpreted as a proxy comparison rather than a true full-image network, and the current evidence does not support a clear pooled-MAE advantage for the proposed ROI policy over that proxy.',
        'forbidden_wording': 'The proposed ROI policy clearly outperformed full-image input.',
        'recommended_table_or_figure': 'Table_InputProbeComparison.csv',
        'notes': '' if full_pair.empty else f"Paired delta MAE={float(full_pair['delta_mae_um'].iloc[0]):.4f}; proxy scope must be stated explicitly.",
    })

    # Near-focus safety story from shared subset direct comparison
    near_pair = near_wide.dropna(
        subset=['y_true_signed_um', 'proposed_two_stage_roi', 'direct_single_stage_regression']
    ).copy()
    near_pair = near_pair[np.abs(pd.to_numeric(near_pair['y_true_signed_um'], errors='coerce')) <= 1.0]
    near_note = 'Near-focus regime requires explicit mention.'
    if not near_pair.empty:
        truth = pd.to_numeric(near_pair['y_true_signed_um'], errors='coerce').to_numpy(dtype=float)
        prop = pd.to_numeric(near_pair['proposed_two_stage_roi'], errors='coerce').to_numpy(dtype=float)
        direct = pd.to_numeric(near_pair['direct_single_stage_regression'], errors='coerce').to_numpy(dtype=float)
        nonzero = ~np.isclose(truth, 0.0)
        prop_wrong = float(np.mean(np.sign(prop[nonzero]) != np.sign(truth[nonzero])) * 100.0)
        direct_wrong = float(np.mean(np.sign(direct[nonzero]) != np.sign(truth[nonzero])) * 100.0)
        near_note = f"Near-focus shared-subset wrong-direction: proposed={prop_wrong:.4f}%, direct={direct_wrong:.4f}%."
    rows.append({
        'claim_id': 'C9',
        'claim_text': 'Near focus remains the hardest regime, but the proposed system retains a low directional failure rate.',
        'claim_category': 'near_focus',
        'support_status': 'supported' if not near_pair.empty else 'mixed',
        'primary_evidence_files': '; '.join([str(mp.tables_dir / 'Table_ROI_NearFocus.csv'), str(mp.tables_dir / 'Table_SafetySummary.csv')]),
        'required_scope': 'near-focus binning plus explicit safety framing',
        'allowed_wording': 'Near focus remained the hardest regime, while directional failures remained uncommon under the proposed system.',
        'forbidden_wording': 'The method solved near-focus ambiguity.',
        'recommended_table_or_figure': 'Table_SafetySummary.csv',
        'notes': near_note,
    })

    ci_note = ''
    if not ci_df.empty:
        ci_note = '; '.join(
            f"{row['metric']}: [{row['ci95_lo']:.4f}, {row['ci95_hi']:.4f}]" for _, row in ci_df.iterrows()
        )
    rows.append({
        'claim_id': 'C10',
        'claim_text': 'The main performance estimates are stable and reproducible rather than being driven by a noisy single run.',
        'claim_category': 'reproducibility',
        'support_status': 'supported' if not ci_df.empty else 'unsupported',
        'primary_evidence_files': '; '.join([str(mp.eval_dir / 'confidence_intervals.csv')]),
        'required_scope': 'bootstrap CI reporting',
        'allowed_wording': 'Bootstrap confidence intervals indicate that the main performance estimates are stable.',
        'forbidden_wording': 'The reported performance is exact and variability-free.',
        'recommended_table_or_figure': 'confidence_intervals.csv',
        'notes': ci_note,
    })
    return pd.DataFrame(rows)


def _guardrails_md(df: pd.DataFrame, track: str) -> str:
    supported = df[df['support_status'] == 'supported']
    mixed = df[df['support_status'] == 'mixed']
    unsupported = df[df['support_status'] == 'unsupported']
    lines = [
        f'# Claim Guardrails: {track}',
        '',
        'This file maps manuscript claims to the exact saved evidence and constrains the wording to what the current smear-track outputs actually support.',
        '',
        '## Supported Claims',
        '',
        markdown_table(supported[['claim_id', 'claim_text', 'allowed_wording', 'recommended_table_or_figure']]) if not supported.empty else 'No supported claims recorded.',
        '',
        '## Mixed Claims',
        '',
        markdown_table(mixed[['claim_id', 'claim_text', 'allowed_wording', 'forbidden_wording', 'notes']]) if not mixed.empty else 'No mixed claims recorded.',
        '',
        '## Unsupported Claims',
        '',
        markdown_table(unsupported[['claim_id', 'claim_text', 'forbidden_wording', 'notes']]) if not unsupported.empty else 'No unsupported claims recorded.',
        '',
        '## Usage Rule',
        '',
        'Use paired-scope tables for architecture claims, domain-gap tables for ROI-selection claims, and avoid turning proxy or inference-only probes into full architecture claims.',
    ]
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    df = _claim_rows(args.track)
    save_eval_and_table(
        df,
        mp.eval_dir / 'claim_evidence_matrix.csv',
        mp.tables_dir / 'Table_ClaimEvidence.csv',
        save_latex_flag=bool(args.save_latex),
    )
    write_markdown(mp.eval_dir / 'claim_guardrails.md', _guardrails_md(df, args.track))


if __name__ == '__main__':
    main()
