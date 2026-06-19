#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from claim_safety_utils import manuscript_paths, markdown_table, write_markdown


RUNBOOK_ROWS = [
    {
        'script_name': 'create_manifests_smears.py',
        'stage': 'manifests',
        'input_files': 'raw smear image folders',
        'output_files': 'manifest_pbs.csv; manifest_wbc.csv; manifest_bma.csv',
        'failure_modes': 'missing source paths; malformed labels; inconsistent manifest schema',
        'manuscript_role': 'creates the source manifests for the smear-track pipeline',
        'claim_dependency': 'all downstream claims depend on these manifests',
    },
    {
        'script_name': 'build_manifest_smears_all.py',
        'stage': 'unified_manifest',
        'input_files': 'manifest_pbs.csv; manifest_wbc.csv; manifest_bma.csv',
        'output_files': 'manifest_smears_all.csv',
        'failure_modes': 'missing dataset tags; column mismatch; missing defocus labels',
        'manuscript_role': 'builds a unified smear-track manifest',
        'claim_dependency': 'dataset-wise reporting and macro-vs-weighted analysis',
    },
    {
        'script_name': 'make_smears_splits.py',
        'stage': 'leakage_safe_split',
        'input_files': 'manifest_smears_all.csv',
        'output_files': 'train.csv; val.csv; test.csv; split_config.json',
        'failure_modes': 'group leakage; unstable grouping keys; split imbalance',
        'manuscript_role': 'creates the group-safe smear splits',
        'claim_dependency': 'fair learned-baseline and main-pipeline evaluation',
    },
    {
        'script_name': 'phase3_build_cache.py',
        'stage': 'phase3_cache',
        'input_files': 'split manifests; raw images',
        'output_files': 'cache_phase3/<track>/*_XA.npy; *_XB.npy; cache_index.csv',
        'failure_modes': 'cache corruption; missing tensors; preprocessing mismatch',
        'manuscript_role': 'creates ROI-level cached tensors for sign and regression stages',
        'claim_dependency': 'all learned downstream stages',
    },
    {
        'script_name': 'train_sign.py',
        'stage': 'phase4_sign_train',
        'input_files': 'cache_index.csv; train/val splits',
        'output_files': 'sign model; history.csv; sign metrics',
        'failure_modes': 'class imbalance; missing ROI tensors; checkpoint mismatch',
        'manuscript_role': 'trains the sign classifier',
        'claim_dependency': 'Stage A sign-performance and directional safety story',
    },
    {
        'script_name': 'calibrate_tau.py',
        'stage': 'tau_calibration',
        'input_files': 'sign model; val ROI predictions',
        'output_files': 'tau_sweep.csv; chosen_tau.json',
        'failure_modes': 'invalid confidence sweep; missing val predictions',
        'manuscript_role': 'calibrates the confidence gate for sign voting',
        'claim_dependency': 'safety and low catastrophic wrong-direction rate',
    },
    {
        'script_name': 'infer_vote_sign.py',
        'stage': 'sign_inference',
        'input_files': 'sign model; chosen_tau.json; cache tensors',
        'output_files': 'vote_sign_results.csv; vote_sign_per_roi.csv',
        'failure_modes': 'all-gated FOVs; missing ROI votes',
        'manuscript_role': 'produces FOV-level sign decisions',
        'claim_dependency': 'Stage A inference outputs and end-to-end sign routing',
    },
    {
        'script_name': 'build_phase5_index.py',
        'stage': 'phase5_index',
        'input_files': 'cache_index.csv; manifests',
        'output_files': 'index_phase5.csv',
        'failure_modes': 'missing roi_uid; label join mismatch',
        'manuscript_role': 'creates the canonical regression index',
        'claim_dependency': 'branch-regressor training and paired baseline support',
    },
    {
        'script_name': 'train_regressors.py',
        'stage': 'phase5_regression_train',
        'input_files': 'index_phase5.csv; train/val splits',
        'output_files': 'R_plus_best.keras; R_minus_best.keras; history_plus.csv; history_minus.csv',
        'failure_modes': 'checkpoint divergence; branch imbalance; triplet sampling errors',
        'manuscript_role': 'trains the plus/minus regressors',
        'claim_dependency': 'Stage B and end-to-end results',
    },
    {
        'script_name': 'infer_regress_and_aggregate.py',
        'stage': 'phase5_inference',
        'input_files': 'branch regressors; voted sign; chosen_tau.json',
        'output_files': 'roi_predictions.csv; fov_aggregate_predictions.csv',
        'failure_modes': 'missing sign outputs; invalid aggregation weights',
        'manuscript_role': 'runs final signed-distance inference and FOV aggregation',
        'claim_dependency': 'main autofocus results and paired-subset comparison',
    },
    {
        'script_name': 'run_full_evaluation.py',
        'stage': 'standard_evaluation',
        'input_files': 'saved sign/regression outputs',
        'output_files': 'stageA_metrics.json; stageB_metrics.json; end_to_end_metrics.json; runtime_metrics.json; tables',
        'failure_modes': 'missing saved outputs; stale metrics; plot generation mismatch',
        'manuscript_role': 'builds the main evaluation outputs',
        'claim_dependency': 'overall accuracy, safety, CI stability',
    },
    {
        'script_name': 'run_roi_ablation_suite.py',
        'stage': 'roi_ablation',
        'input_files': 'saved final models; ROI policy assets',
        'output_files': 'ROI ablation CSVs; domain gap tables; ROI plots',
        'failure_modes': 'missing calibration assets; mixed selector backends; resume misrouting',
        'manuscript_role': 'tests whether ROI policy matters with downstream models fixed',
        'claim_dependency': 'ROI-selection and domain-gap claims',
    },
    {
        'script_name': 'run_baseline_comparison.py',
        'stage': 'baseline_comparison',
        'input_files': 'saved proposed outputs; baseline models and inference outputs',
        'output_files': 'baseline comparison CSVs; baseline plots',
        'failure_modes': 'mixed-scope comparisons; unavailable baselines; stale inference rows',
        'manuscript_role': 'builds the baseline comparison package',
        'claim_dependency': 'classical baseline claim and architecture/input-probe context',
    },
    {
        'script_name': 'build_q1_paper_package.py',
        'stage': 'paper_packaging',
        'input_files': 'evaluation outputs; manuscript-safety layer outputs',
        'output_files': 'paper_package/main_paper; paper_package/supplement; paper_package/excluded_or_quarantined',
        'failure_modes': 'stale whitelist/blacklist; missing curated assets; inconsistent package routing',
        'manuscript_role': 'assembles the claim-safe manuscript package',
        'claim_dependency': 'final citable paper asset set',
    },
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build pipeline diagram description and runbook for manuscript support.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    return ap.parse_args()


def _diagram_md(track: str) -> str:
    return '\n'.join([
        f'# Pipeline Diagram Description: {track}',
        '',
        '1. Create per-dataset manifests for PBS, WBC, and BMA.',
        '2. Merge them into one unified smear-track manifest.',
        '3. Create leakage-safe train/val/test splits using grouping keys.',
        '4. Build Phase-3 ROI caches containing `X_A` and `X_B` tensors.',
        '5. Train the sign classifier on `X_A`.',
        '6. Calibrate the voting threshold `tau` on the validation set.',
        '7. Infer ROI votes and aggregate them into FOV-level sign decisions.',
        '8. Build the Phase-5 regression index from cached ROIs and labels.',
        '9. Train the plus and minus branch regressors on `X_B`.',
        '10. Run final inference and aggregate ROI predictions into signed FOV autofocus distances.',
        '11. Run the standard evaluation stack.',
        '12. Run the ROI-ablation suite with fixed downstream models.',
        '13. Run the baseline comparison package.',
        '14. Build the claim-safe manuscript package.',
    ])


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    runbook = pd.DataFrame(RUNBOOK_ROWS)
    runbook.to_csv(mp.eval_dir / 'script_runbook.csv', index=False)
    write_markdown(mp.eval_dir / 'script_runbook.md', '\n'.join([f'# Script Runbook: {args.track}', '', markdown_table(runbook)]))
    write_markdown(mp.eval_dir / 'pipeline_diagram_description.md', _diagram_md(args.track))


if __name__ == '__main__':
    main()
