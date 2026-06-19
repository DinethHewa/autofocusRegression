#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from claim_safety_utils import manuscript_paths, markdown_table, write_markdown


CAPTIONS = [
    {
        'asset_name': 'Table_EndToEnd.csv',
        'concise_caption': 'End-to-end smear-track autofocus performance of the proposed pipeline.',
        'long_caption': 'End-to-end autofocus-distance estimation performance for the smear-track pipeline, including pooled MAE, RMSE, bias, within-threshold rates, and catastrophic wrong-direction rate.',
        'one_sentence_takeaway': 'The full pipeline achieved strong pooled performance with very low bias and rare catastrophic wrong-direction failures.',
        'scope_caveat': 'Weighted pooled smear performance should be interpreted alongside macro and dataset-wise reporting because WBC dominates the support.',
        'recommended_manuscript_section': 'Results',
    },
    {
        'asset_name': 'Table_Baseline_Comparison.csv',
        'concise_caption': 'Overview baseline table with explicit scope labels.',
        'long_caption': 'Overview comparison across the proposed method, direct regression, classical focus baseline, and input-policy probes. This table mixes pooled and held-out scopes and should be used as context, not as the primary fair architecture comparison.',
        'one_sentence_takeaway': 'The overview shows relative method positions, but fair architecture claims must come from the paired-subset tables.',
        'scope_caveat': 'Do not use this table alone to claim superiority over direct regression, center-crop, or the full-field proxy.',
        'recommended_manuscript_section': 'Supplement',
    },
    {
        'asset_name': 'Table_ROI_Ablation_Regression.csv',
        'concise_caption': 'ROI-policy comparison with fixed downstream models.',
        'long_caption': 'ROI-policy ablation results obtained by keeping the trained sign and regression models fixed while varying only the inference-time ROI policy.',
        'one_sentence_takeaway': 'ROI-policy differences are statistically real but small in pooled MAE.',
        'scope_caveat': 'Use this table together with the domain-gap summary; the strongest ROI story is improved cross-domain consistency rather than a large pooled-MAE gain.',
        'recommended_manuscript_section': 'Results',
    },
    {
        'asset_name': 'Table_ROI_Ablation_ByDataset.csv',
        'concise_caption': 'Dataset-wise ROI-ablation results across smear domains.',
        'long_caption': 'Per-dataset ROI-ablation results for the smear domains, showing that ROI-policy effects are strongest on the harder PBS and BMA domains while WBC remains comparatively stable.',
        'one_sentence_takeaway': 'Adaptive ROI selection matters most on the hard domains rather than on the already-strong WBC pool.',
        'scope_caveat': 'Do not summarize this table with only a pooled average.',
        'recommended_manuscript_section': 'Results',
    },
    {
        'asset_name': 'Table_DomainGapReduction.csv',
        'concise_caption': 'Domain-gap reduction across ROI policies.',
        'long_caption': 'Cross-domain gap summary for ROI policies, reported as the difference between the worst and best dataset MAE, to highlight whether a selector improves consistency across smear domains.',
        'one_sentence_takeaway': 'Adaptive ROI policies reduced domain gap more clearly than they reduced pooled MAE.',
        'scope_caveat': 'This table is the strongest support for the ROI-selection claim and should be foregrounded accordingly.',
        'recommended_manuscript_section': 'Results',
    },
    {
        'asset_name': 'Table_PairedArchitectureComparison.csv',
        'concise_caption': 'Paired shared-subset architecture comparison.',
        'long_caption': 'Fair paired shared-subset comparison between the proposed system and comparator methods, including delta MAE, delta wrong-direction rate, paired tests, and claim-safe summaries.',
        'one_sentence_takeaway': 'This table is the correct basis for architecture claims because it forces all methods onto shared support.',
        'scope_caveat': 'Use this table, not the mixed-scope overview table, for any claim about direct regression.',
        'recommended_manuscript_section': 'Results',
    },
    {
        'asset_name': 'safety_vs_mae_frontier.png',
        'concise_caption': 'MAE versus catastrophic wrong-direction tradeoff.',
        'long_caption': 'Safety frontier comparing method-level pooled MAE against catastrophic wrong-direction rate to expose tradeoffs between numerical accuracy and directional reliability.',
        'one_sentence_takeaway': 'The proposed pipeline remains publishable because it combines strong error performance with very low directional failure risk.',
        'scope_caveat': 'Method rows still retain their original scopes; use the paired-subset tables for direct architecture fairness claims.',
        'recommended_manuscript_section': 'Results',
    },
    {
        'asset_name': 'domain_gap_reduction_plot.png',
        'concise_caption': 'Pooled error versus domain-gap tradeoff for ROI policies.',
        'long_caption': 'Scatter plot of ROI-policy weighted MAE against domain-gap MAE to show that the main benefit of adaptive selection is improved cross-domain consistency at nearly unchanged pooled error.',
        'one_sentence_takeaway': 'The adaptive selector earns its value primarily through lower domain gap, not a large average-MAE jump.',
        'scope_caveat': 'This figure should be interpreted together with the dataset-wise ROI-ablation table.',
        'recommended_manuscript_section': 'Results',
    },
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build caption bank for manuscript tables and figures.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    return ap.parse_args()


def _md(df: pd.DataFrame, track: str) -> str:
    lines = [f'# Caption Bank: {track}', '', markdown_table(df)]
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    df = pd.DataFrame(CAPTIONS)
    df.to_csv(mp.eval_dir / 'caption_bank.csv', index=False)
    write_markdown(mp.eval_dir / 'caption_bank.md', _md(df, args.track))


if __name__ == '__main__':
    main()
