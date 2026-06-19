#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from evaluation_utils import default_eval_cli, get_paths, log


SCRIPT_TO_FIGURES = {
    "evaluate_stageA.py": [
        "ROC.png",
        "confusion_matrix_heatmap.png",
        "bin_accuracy_curve.png",
        "confidence_vs_accuracy_curve.png",
    ],
    "evaluate_stageB.py": [
        "regression_error_histogram.png",
        "regression_error_vs_magnitude.png",
        "regression_error_boxplot_per_bin.png",
    ],
    "evaluate_end_to_end.py": [
        "signed_error_histogram.png",
        "error_cdf.png",
    ],
    "evaluate_runtime.py": [
        "runtime_breakdown.png",
        "latency_vs_k.png",
    ],
    "run_ablation_suite.py": [
        "ablation_plot.png",
    ],
    "cross_dataset_validation.py": [
        "cross_dataset_barplot.png",
    ],
}

ROI_SCRIPT_FIGURES = {
    "run_roi_ablation_suite.py": [
        "roi_efficiency_pareto.png",
        "roi_robustness_vs_error.png",
        "roi_score_correlation.png",
    ],
}

BASELINE_SCRIPT_FIGURES = {
    "run_baseline_comparison.py": [
        "baseline_efficiency_frontier.png",
        "baseline_dataset_mae.png",
        "baseline_near_focus.png",
    ],
}


def _run_script(script_path: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(script_path),
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
        cmd.append("--save-latex")
    if script_path.name == "run_baseline_comparison.py":
        cmd.extend(["--mode", "eval_only"])
    if args.resume:
        cmd.append("--resume")

    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate all evaluation figures for a track")
    default_eval_cli(ap)
    ap.add_argument("--force", action="store_true", help="Regenerate figures even if they already exist")
    ap.add_argument(
        "--with-roi-ablation",
        action="store_true",
        help="Include ROI-ablation figures in the figure index and regenerate them if needed.",
    )
    ap.add_argument(
        "--include-baseline-comparison",
        action="store_true",
        help="Include baseline-comparison figures in the figure index and regenerate them if needed.",
    )
    args = ap.parse_args()
    if args.force and args.resume:
        raise ValueError("--force and --resume cannot be used together.")

    script_dir = Path(__file__).resolve().parent
    paths = get_paths(args.track)
    log(paths, "Starting figure generation")
    script_to_figures = dict(SCRIPT_TO_FIGURES)
    if args.with_roi_ablation:
        script_to_figures.update(ROI_SCRIPT_FIGURES)
    if args.include_baseline_comparison:
        script_to_figures.update(BASELINE_SCRIPT_FIGURES)

    all_fig_paths = [paths.eval_dir / fig for figs in script_to_figures.values() for fig in figs]
    index_path = paths.eval_dir / "all_figures_index.csv"
    if args.resume and all(p.is_file() for p in all_fig_paths) and index_path.is_file():
        log(paths, f"Resume: all figures already exist; skipping. ({paths.eval_dir})")
        return

    for script_name, fig_names in script_to_figures.items():
        script_path = script_dir / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing evaluation script required for figures: {script_path}")

        needs_run = bool(args.force)
        if not needs_run:
            for fig_name in fig_names:
                if not (paths.eval_dir / fig_name).is_file():
                    needs_run = True
                    break

        if needs_run:
            _run_script(script_path, args)

    records = []
    missing = []
    for script_name, fig_names in script_to_figures.items():
        for fig_name in fig_names:
            fig_path = paths.eval_dir / fig_name
            exists = fig_path.is_file()
            size_bytes = fig_path.stat().st_size if exists else 0
            records.append(
                {
                    "script": script_name,
                    "figure": fig_name,
                    "path": str(fig_path),
                    "exists": int(exists),
                    "size_bytes": int(size_bytes),
                }
            )
            if not exists:
                missing.append(str(fig_path))

    pd.DataFrame(records).to_csv(index_path, index=False)

    if missing:
        raise FileNotFoundError("Some required figures are still missing after generation:\n" + "\n".join(missing))

    log(paths, f"Figure generation complete. Index: {index_path}")


if __name__ == "__main__":
    main()
