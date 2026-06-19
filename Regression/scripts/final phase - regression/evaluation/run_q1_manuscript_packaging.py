#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evaluation_utils import get_paths, log


SCRIPT_DIR = Path(__file__).resolve().parent
STEPS = [
    'audit_outputs_for_manuscript.py',
    'build_claim_evidence_matrix.py',
    'build_fair_scope_tables.py',
    'build_domain_gap_package.py',
    'build_safety_package.py',
    'build_paired_subset_package.py',
    'build_asset_curation_package.py',
    'build_results_writing_support.py',
    'build_caption_package.py',
    'build_pipeline_doc_support.py',
]

STEP_OUTPUTS = {
    'audit_outputs_for_manuscript.py': ['output_audit_report.csv', 'output_audit_report.md', 'manuscript_whitelist.json', 'manuscript_blacklist.json'],
    'build_claim_evidence_matrix.py': ['claim_evidence_matrix.csv', 'claim_guardrails.md'],
    'build_fair_scope_tables.py': ['fair_architecture_comparison.csv', 'input_probe_comparison.csv', 'macro_vs_weighted.csv'],
    'build_domain_gap_package.py': ['domain_gap_summary.csv', 'domain_gap_reduction_plot.png', 'domain_gap_waterfall.png'],
    'build_safety_package.py': ['safety_summary.csv', 'safety_vs_mae_frontier.png'],
    'build_paired_subset_package.py': ['paired_subset_predictions.csv', 'paired_subset_metrics.csv', 'paired_subset_tests.csv'],
    'build_asset_curation_package.py': ['asset_curation.csv', 'asset_curation.md'],
    'build_results_writing_support.py': ['results_writing_support.json', 'results_writing_support.md', 'discussion_guardrails.md', 'abstract_guardrails.md'],
    'build_caption_package.py': ['caption_bank.csv', 'caption_bank.md'],
    'build_pipeline_doc_support.py': ['pipeline_diagram_description.md', 'script_runbook.csv', 'script_runbook.md'],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run the full Q1 manuscript-packaging upgrade layer.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--save-latex', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--skip-paper-package-build', action='store_true')
    return ap.parse_args()


def _run(cmd: list[str], paths) -> None:
    log(paths, 'RUN: ' + ' '.join(cmd))
    print('[INFO] Running: ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def _step_is_complete(script_name: str, paths) -> bool:
    expected = STEP_OUTPUTS.get(script_name, [])
    if not expected:
        return False
    return all((paths.eval_dir / rel).is_file() for rel in expected)


def main() -> None:
    args = parse_args()
    paths = get_paths(args.track)
    log(paths, 'Starting Q1 manuscript-packaging runner')

    for script_name in STEPS:
        if not args.force and _step_is_complete(script_name, paths):
            log(paths, f'Resume: manuscript-support step already exists; skipping {script_name}. ({paths.eval_dir})')
            continue
        cmd = [sys.executable, str(SCRIPT_DIR / script_name), '--track', args.track]
        if args.save_latex and script_name in {
            'build_claim_evidence_matrix.py',
            'build_fair_scope_tables.py',
            'build_domain_gap_package.py',
            'build_safety_package.py',
            'build_paired_subset_package.py',
            'build_asset_curation_package.py',
        }:
            cmd.append('--save-latex')
        _run(cmd, paths)

    if not args.skip_paper_package_build:
        cmd = [sys.executable, str(SCRIPT_DIR / 'build_q1_paper_package.py'), '--track', args.track]
        if args.save_latex:
            cmd.append('--save-latex')
        if args.resume:
            cmd.append('--resume')
        else:
            cmd.append('--force')
        _run(cmd, paths)

    log(paths, 'Q1 manuscript-packaging runner finished successfully')


if __name__ == '__main__':
    main()
