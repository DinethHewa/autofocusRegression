#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evaluation_utils import default_eval_cli, get_paths, log


ORDERED_SCRIPTS = [
    "evaluate_stageA.py",
    "evaluate_stageB.py",
    "evaluate_end_to_end.py",
    "evaluate_runtime.py",
    "run_ablation_suite.py",
    "cross_dataset_validation.py",
    "statistical_tests.py",
]

CLAIM_SAFETY_SCRIPTS = [
    "audit_outputs_for_manuscript.py",
    "build_claim_evidence_matrix.py",
]

FAIR_SCOPE_SCRIPTS = [
    "build_fair_scope_tables.py",
    "build_domain_gap_package.py",
    "build_safety_package.py",
    "build_paired_subset_package.py",
]

WRITING_SUPPORT_SCRIPTS = [
    "build_results_writing_support.py",
    "build_caption_package.py",
    "build_pipeline_doc_support.py",
]

PAPER_CURATION_SCRIPTS = [
    "build_asset_curation_package.py",
]


def _run(cmd: list[str], paths) -> None:
    log(paths, f"RUN: {' '.join(cmd)}")
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_support_script(script_dir: Path, script_name: str, args, paths) -> None:
    script_path = script_dir / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing manuscript-support script: {script_path}")
    cmd = [sys.executable, str(script_path), "--track", args.track]
    if args.save_latex and script_name in {
        "build_claim_evidence_matrix.py",
        "build_fair_scope_tables.py",
        "build_domain_gap_package.py",
        "build_safety_package.py",
        "build_paired_subset_package.py",
        "build_asset_curation_package.py",
    }:
        cmd.append("--save-latex")
    _run(cmd, paths)


def main() -> None:
    ap = argparse.ArgumentParser(description="Master runner for full final-phase evaluation")
    default_eval_cli(ap)
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    ap.add_argument("--force-figures", action="store_true", help="Force regenerate all figures")
    ap.add_argument(
        "--with-roi-ablation",
        action="store_true",
        help="Run the fixed-model ROI-ablation-to-regression evaluation layer.",
    )
    ap.add_argument(
        "--include-baseline-comparison",
        action="store_true",
        help="Run the manuscript baseline-comparison package after the standard evaluation stack.",
    )
    ap.add_argument(
        "--include-claim-safety",
        action="store_true",
        help="Build the manuscript claim-evidence matrix and output audit after the standard evaluation stack.",
    )
    ap.add_argument(
        "--include-fair-scope-tables",
        action="store_true",
        help="Build fair-scope comparison, domain-gap, safety, and paired-subset manuscript tables.",
    )
    ap.add_argument(
        "--include-writing-support",
        action="store_true",
        help="Build manuscript-writing support, captions, and pipeline documentation assets.",
    )
    ap.add_argument(
        "--include-paper-curation",
        action="store_true",
        help="Build asset-curation outputs and refresh the manuscript package with whitelist/blacklist routing.",
    )
    args = ap.parse_args()
    if args.resume and args.force_figures:
        raise ValueError("--resume and --force-figures cannot be used together.")

    script_dir = Path(__file__).resolve().parent
    paths = get_paths(args.track)
    paper_package_dir = paths.sign_dir.parent / "paper_package"

    log(paths, "Starting full evaluation runner")
    resume_targets = [
        paths.eval_dir / "stageA_metrics.json",
        paths.eval_dir / "stageB_metrics.json",
        paths.eval_dir / "end_to_end_metrics.json",
        paths.eval_dir / "runtime_metrics.json",
        paths.eval_dir / "ablation_results.csv",
        paths.eval_dir / "cross_dataset_results.csv",
        paths.eval_dir / "statistical_tests_results.json",
    ]
    if args.with_roi_ablation:
        resume_targets.extend(
            [
                paths.eval_dir / "roi_ablation_to_regression_performance.csv",
                paths.eval_dir / "roi_ablation_by_dataset.csv",
                paths.eval_dir / "roi_policy_pairwise_tests.csv",
            ]
        )
    if args.include_baseline_comparison:
        resume_targets.extend(
            [
                paths.eval_dir / "baseline_comparison_results.csv",
                paths.eval_dir / "baseline_by_dataset.csv",
                paths.eval_dir / "baseline_pairwise_tests.csv",
            ]
        )
    if args.include_claim_safety:
        resume_targets.extend(
            [
                paths.eval_dir / "output_audit_report.csv",
                paths.eval_dir / "claim_evidence_matrix.csv",
                paths.eval_dir / "manuscript_whitelist.json",
                paths.eval_dir / "manuscript_blacklist.json",
            ]
        )
    if args.include_fair_scope_tables:
        resume_targets.extend(
            [
                paths.eval_dir / "fair_architecture_comparison.csv",
                paths.eval_dir / "input_probe_comparison.csv",
                paths.eval_dir / "macro_vs_weighted.csv",
                paths.eval_dir / "domain_gap_summary.csv",
                paths.eval_dir / "safety_summary.csv",
                paths.eval_dir / "paired_subset_tests.csv",
            ]
        )
    if args.include_writing_support:
        resume_targets.extend(
            [
                paths.eval_dir / "results_writing_support.json",
                paths.eval_dir / "caption_bank.csv",
                paths.eval_dir / "pipeline_diagram_description.md",
                paths.eval_dir / "script_runbook.csv",
            ]
        )
    if args.include_paper_curation:
        resume_targets.extend(
            [
                paths.eval_dir / "asset_curation.csv",
                paper_package_dir / "paper_package_manifest.csv",
                paper_package_dir / "paper_package_summary.md",
            ]
        )
    if not args.no_plots:
        resume_targets.append(paths.eval_dir / "all_figures_index.csv")
    if args.resume and all(p.is_file() for p in resume_targets):
        log(paths, f"Resume: full evaluation outputs already exist; skipping. ({paths.eval_dir})")
        return

    for script_name in ORDERED_SCRIPTS:
        script_path = script_dir / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing required evaluation script: {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "--track",
            args.track,
            "--bins",
            str(args.bins),
            "--bootstrap",
            str(args.bootstrap),
            "--k-values",
            *[str(x) for x in args.k_values],
        ]
        if args.save_latex:
            cmd.append("--save-latex")
        if not args.no_plots:
            cmd.append("--save-plots")
        if args.resume:
            cmd.append("--resume")

        _run(cmd, paths)

    if args.with_roi_ablation:
        roi_script = script_dir / "run_roi_ablation_suite.py"
        if not roi_script.is_file():
            raise FileNotFoundError(f"Missing ROI ablation script: {roi_script}")
        roi_cmd = [
            sys.executable,
            str(roi_script),
            "--track",
            args.track,
            "--bins",
            str(args.bins),
            "--bootstrap",
            str(args.bootstrap),
            "--k-values",
            *[str(x) for x in args.k_values],
        ]
        if args.save_latex:
            roi_cmd.append("--save-latex")
        if not args.no_plots:
            roi_cmd.append("--save-plots")
        if args.resume:
            roi_cmd.append("--resume")
        _run(roi_cmd, paths)

    if args.include_baseline_comparison:
        baseline_script = script_dir / "run_baseline_comparison.py"
        if not baseline_script.is_file():
            raise FileNotFoundError(f"Missing baseline comparison script: {baseline_script}")
        baseline_cmd = [
            sys.executable,
            str(baseline_script),
            "--track",
            args.track,
            "--mode",
            "train_and_eval",
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            "42",
        ]
        if args.save_latex:
            baseline_cmd.append("--save-latex")
        if not args.no_plots:
            baseline_cmd.append("--save-plots")
        if args.resume:
            baseline_cmd.append("--resume")
        _run(baseline_cmd, paths)

    if not args.no_plots:
        fig_script = script_dir / "generate_all_figures.py"
        if not fig_script.is_file():
            raise FileNotFoundError(f"Missing figure generator script: {fig_script}")

        fig_cmd = [
            sys.executable,
            str(fig_script),
            "--track",
            args.track,
            "--save-plots",
            "--bins",
            str(args.bins),
            "--bootstrap",
            str(args.bootstrap),
            "--k-values",
            *[str(x) for x in args.k_values],
        ]
        if args.save_latex:
            fig_cmd.append("--save-latex")
        if args.force_figures:
            fig_cmd.append("--force")
        if args.resume:
            fig_cmd.append("--resume")
        if args.with_roi_ablation:
            fig_cmd.append("--with-roi-ablation")
        if args.include_baseline_comparison:
            fig_cmd.append("--include-baseline-comparison")

        _run(fig_cmd, paths)

    if args.include_claim_safety:
        for script_name in CLAIM_SAFETY_SCRIPTS:
            _run_support_script(script_dir, script_name, args, paths)

    if args.include_fair_scope_tables:
        for script_name in FAIR_SCOPE_SCRIPTS:
            _run_support_script(script_dir, script_name, args, paths)

    if args.include_writing_support:
        for script_name in WRITING_SUPPORT_SCRIPTS:
            _run_support_script(script_dir, script_name, args, paths)

    if args.include_paper_curation:
        if not args.include_claim_safety:
            for script_name in CLAIM_SAFETY_SCRIPTS:
                _run_support_script(script_dir, script_name, args, paths)
        if not args.include_fair_scope_tables:
            for script_name in FAIR_SCOPE_SCRIPTS:
                _run_support_script(script_dir, script_name, args, paths)
        if not args.include_writing_support:
            for script_name in WRITING_SUPPORT_SCRIPTS:
                _run_support_script(script_dir, script_name, args, paths)
        for script_name in PAPER_CURATION_SCRIPTS:
            _run_support_script(script_dir, script_name, args, paths)
        package_cmd = [
            sys.executable,
            str(script_dir / "build_q1_paper_package.py"),
            "--track",
            args.track,
        ]
        if args.save_latex:
            package_cmd.append("--save-latex")
        if args.resume:
            package_cmd.append("--resume")
        else:
            package_cmd.append("--force")
        _run(package_cmd, paths)

    log(paths, "Full evaluation runner finished successfully")


if __name__ == "__main__":
    main()
