#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_utils import (
    DATA_OUT,
    ROOT,
    TRACKS,
    get_paths,
    infer_dataset_for_fov,
    require_file,
    save_table,
    weighted_median,
)

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent

COLORS = {
    "blue": "#1f4e79",
    "teal": "#0f766e",
    "green": "#4d7c0f",
    "gold": "#b45309",
    "red": "#9f1239",
    "purple": "#6d28d9",
    "gray": "#475569",
    "light_gray": "#cbd5e1",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )


def _save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _sample_df(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= int(max_rows):
        return df
    return df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _log(log_path: Path, message: str) -> None:
    line = message.strip()
    print(line)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _panel(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _register(
    manifest: list[dict],
    *,
    track: str,
    tier: str,
    category: str,
    label: str,
    title: str,
    path: Path,
    section: str,
    description: str,
    source: str,
    status: str = "generated",
) -> None:
    manifest.append(
        {
            "track": track,
            "tier": tier,
            "category": category,
            "label": label,
            "title": title,
            "path": str(path),
            "section": section,
            "description": description,
            "source": source,
            "status": status,
        }
    )


def _copy_reference_file(src: Path, dst: Path, manifest: list[dict], track: str, label: str, title: str, description: str) -> None:
    if not src.is_file():
        _register(
            manifest,
            track=track,
            tier="reference",
            category="reference",
            label=label,
            title=title,
            path=dst,
            section="Supplementary",
            description=description,
            source=str(src),
            status="missing",
        )
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    _register(
        manifest,
        track=track,
        tier="reference",
        category="reference",
        label=label,
        title=title,
        path=dst,
        section="Supplementary",
        description=description,
        source=str(src),
        status="copied",
    )


def _load_phase5_index_local(track: str) -> pd.DataFrame:
    p = DATA_OUT / track / "regression" / "index_phase5.csv"
    require_file(p, "phase5 index")
    df = pd.read_csv(p, low_memory=False)
    required = ["roi_uid", "dataset", "fov_id", "defocus_um", "y_sign", "y_mag_um"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Phase5 index missing required columns {missing}: {p}")
    return df


def _gt_per_fov(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fov_id, g in index_df.groupby("fov_id", sort=True):
        vals = pd.to_numeric(g["defocus_um"], errors="coerce").to_numpy(dtype=float)
        if "roi_importance" in g.columns:
            w = pd.to_numeric(g["roi_importance"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        else:
            w = np.ones((len(g),), dtype=float)
        gt = weighted_median(vals, w)
        if gt is None:
            finite = vals[np.isfinite(vals)]
            gt = float(np.median(finite)) if finite.size else np.nan
        rows.append({"fov_id": str(fov_id), "gt_signed_um": float(gt) if np.isfinite(gt) else np.nan})
    return pd.DataFrame(rows)


def _ensure_evaluation_outputs(track: str, save_latex: bool, skip_evaluation: bool, log_path: Path) -> None:
    paths = get_paths(track)
    required = [
        paths.eval_dir / "stageA_metrics.json",
        paths.eval_dir / "stageB_metrics.json",
        paths.eval_dir / "end_to_end_metrics.json",
        paths.eval_dir / "runtime_metrics.json",
        paths.eval_dir / "ablation_results.csv",
        paths.eval_dir / "cross_dataset_results.csv",
        paths.eval_dir / "confidence_intervals.csv",
        paths.eval_dir / "statistical_tests_results.json",
    ]
    missing = [p for p in required if not p.is_file()]
    if not missing:
        return
    if skip_evaluation:
        raise FileNotFoundError(
            "Missing evaluation outputs required for paper package:\n" + "\n".join(str(p) for p in missing)
        )

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_full_evaluation.py"),
        "--track",
        track,
        "--save-plots",
        "--resume",
    ]
    if save_latex:
        cmd.append("--save-latex")
    _log(log_path, "[INFO] Missing evaluation outputs detected. Running full evaluation first:")
    _log(log_path, "[INFO] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    still_missing = [p for p in required if not p.is_file()]
    if still_missing:
        raise FileNotFoundError(
            "Evaluation did not produce all required outputs for paper package:\n" + "\n".join(str(p) for p in still_missing)
        )


def _ensure_manuscript_support_outputs(track: str, save_latex: bool, log_path: Path) -> None:
    eval_dir = get_paths(track).eval_dir
    required = [
        eval_dir / "output_audit_report.csv",
        eval_dir / "claim_evidence_matrix.csv",
        eval_dir / "fair_architecture_comparison.csv",
        eval_dir / "input_probe_comparison.csv",
        eval_dir / "macro_vs_weighted.csv",
        eval_dir / "domain_gap_summary.csv",
        eval_dir / "safety_summary.csv",
        eval_dir / "paired_subset_tests.csv",
        eval_dir / "asset_curation.csv",
        eval_dir / "results_writing_support.json",
        eval_dir / "caption_bank.csv",
        eval_dir / "pipeline_diagram_description.md",
        eval_dir / "script_runbook.csv",
        eval_dir / "manuscript_whitelist.json",
        eval_dir / "manuscript_blacklist.json",
    ]
    missing = [p for p in required if not p.is_file()]
    if not missing:
        return
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_q1_manuscript_packaging.py"),
        "--track",
        track,
        "--skip-paper-package-build",
    ]
    if save_latex:
        cmd.append("--save-latex")
    _log(log_path, "[INFO] Missing manuscript-support outputs detected. Running manuscript-packaging helpers:")
    _log(log_path, "[INFO] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _safe_relpath(src: Path) -> Path:
    try:
        return src.resolve().relative_to(ROOT)
    except Exception:
        return Path(src.name)


def _copy_with_structure(src: Path, dst_root: Path) -> Path:
    rel = _safe_relpath(src)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _load_json_optional(path: Path) -> dict | list:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _curated_summary_md(track: str, claim_df: pd.DataFrame, audit_df: pd.DataFrame, manifest_df: pd.DataFrame) -> str:
    supported = claim_df[claim_df["support_status"].astype(str) == "supported"]["claim_text"].tolist()
    mixed = claim_df[claim_df["support_status"].astype(str) == "mixed"]["claim_text"].tolist()
    unsupported = claim_df[claim_df["support_status"].astype(str) == "unsupported"]["claim_text"].tolist()
    proxy_assets = audit_df[audit_df["classification"].astype(str) == "proxy_only"]["asset_name"].astype(str).tolist()
    paired_assets = audit_df[audit_df["classification"].astype(str) == "paired_scope_only"]["asset_name"].astype(str).tolist()
    lines = [
        f"# Paper Package Summary: {track}",
        "",
        "## Strongest Supported Claims",
        "",
    ]
    if supported:
        lines.extend([f"- {x}" for x in supported])
    else:
        lines.append("- No supported claims were detected.")
    lines.extend([
        "",
        "## Mixed Or Unsupported Claims",
        "",
    ])
    if mixed:
        lines.extend([f"- Mixed: {x}" for x in mixed])
    if unsupported:
        lines.extend([f"- Unsupported: {x}" for x in unsupported])
    if not mixed and not unsupported:
        lines.append("- None.")
    lines.extend([
        "",
        "## Key Caveats",
        "",
        "- Weighted smear results are dominated by WBC and must be read alongside macro and domain-gap reporting.",
        "- Direct-regression architecture claims must come from paired shared-subset tables, not the mixed-scope overview table.",
        "- Center-crop and full-field rows are input probes; the full-field method is a tiled proxy, not a true full-image learner.",
        "",
        "## Proxy-Only Comparisons",
        "",
    ])
    if proxy_assets:
        lines.extend([f"- {x}" for x in proxy_assets])
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Paired-Scope-Only Assets",
        "",
    ])
    if paired_assets:
        lines.extend([f"- {x}" for x in paired_assets])
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Package Inventory",
        "",
        f"- Copied assets: {len(manifest_df)}",
        f"- Main-paper assets: {int((manifest_df['section'] == 'main_paper').sum())}",
        f"- Supplement assets: {int((manifest_df['section'] == 'supplement').sum())}",
        f"- Quarantined assets: {int((manifest_df['section'] == 'excluded_or_quarantined').sum())}",
    ])
    return "\n".join(lines) + "\n"


def _build_curated_package(track: str, package_root: Path, log_path: Path) -> None:
    eval_dir = get_paths(track).eval_dir
    main_dir = package_root / "main_paper"
    supp_dir = package_root / "supplement"
    excl_dir = package_root / "excluded_or_quarantined"
    for p in [main_dir, supp_dir, excl_dir]:
        p.mkdir(parents=True, exist_ok=True)

    whitelist = _load_json_optional(eval_dir / "manuscript_whitelist.json")
    blacklist = _load_json_optional(eval_dir / "manuscript_blacklist.json")
    audit_df = pd.read_csv(eval_dir / "output_audit_report.csv", low_memory=False)
    claim_df = pd.read_csv(eval_dir / "claim_evidence_matrix.csv", low_memory=False)
    curation_df = pd.read_csv(eval_dir / "asset_curation.csv", low_memory=False)

    manifest_rows: list[dict] = []
    blacklisted_paths = {str(x.get("path", "")) for x in blacklist.get("quarantined_assets", [])}

    # Start from curated manuscript assets.
    for _, row in curation_df.iterrows():
        src = Path(str(row["path"]))
        if not src.is_file():
            continue
        section = str(row["recommended_section"])
        if str(src) in blacklisted_paths or section == "exclude":
            dst = _copy_with_structure(src, excl_dir)
            final_section = "excluded_or_quarantined"
        elif section == "main":
            dst = _copy_with_structure(src, main_dir)
            final_section = "main_paper"
        else:
            dst = _copy_with_structure(src, supp_dir)
            final_section = "supplement"
        manifest_rows.append(
            {
                "asset_name": str(row["asset_name"]),
                "source_path": str(src),
                "destination_path": str(dst),
                "section": final_section,
                "classification": str(row.get("source_classification", "")),
                "justification": str(row.get("justification_for_section", "")),
            }
        )

    # Always include manuscript-support metadata files in the supplement.
    support_files = [
        eval_dir / "claim_evidence_matrix.csv",
        eval_dir / "claim_guardrails.md",
        eval_dir / "asset_curation.csv",
        eval_dir / "asset_curation.md",
        eval_dir / "caption_bank.csv",
        eval_dir / "caption_bank.md",
        eval_dir / "results_writing_support.json",
        eval_dir / "results_writing_support.md",
        eval_dir / "discussion_guardrails.md",
        eval_dir / "abstract_guardrails.md",
        eval_dir / "pipeline_diagram_description.md",
        eval_dir / "script_runbook.csv",
        eval_dir / "script_runbook.md",
        eval_dir / "output_audit_report.csv",
        eval_dir / "output_audit_report.md",
        eval_dir / "manuscript_whitelist.json",
        eval_dir / "manuscript_blacklist.json",
    ]
    seen_sources = {row["source_path"] for row in manifest_rows}
    for src in support_files:
        if not src.is_file() or str(src) in seen_sources:
            continue
        dst = _copy_with_structure(src, supp_dir)
        manifest_rows.append(
            {
                "asset_name": src.name,
                "source_path": str(src),
                "destination_path": str(dst),
                "section": "supplement",
                "classification": "metadata",
                "justification": "Manuscript-support metadata and writing guardrails.",
            }
        )

    # Copy any explicitly blacklisted assets into quarantine for traceability.
    for item in blacklist.get("quarantined_assets", []):
        src = Path(str(item.get("path", "")))
        if not src.is_file():
            continue
        if str(src) in seen_sources:
            continue
        dst = _copy_with_structure(src, excl_dir)
        manifest_rows.append(
            {
                "asset_name": src.name,
                "source_path": str(src),
                "destination_path": str(dst),
                "section": "excluded_or_quarantined",
                "classification": str(item.get("classification", "stale_or_inconsistent")),
                "justification": str(item.get("note", "")),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["section", "asset_name"]).reset_index(drop=True)
    manifest_path = package_root / "paper_package_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    summary_path = package_root / "paper_package_summary.md"
    summary_path.write_text(_curated_summary_md(track, claim_df, audit_df, manifest_df), encoding="utf-8")
    _log(log_path, f"[DONE] Curated manuscript package created under {package_root}")


def _load_stageA_df(paths, index_df: pd.DataFrame) -> pd.DataFrame:
    per_roi = paths.sign_dir / "inference" / "vote_sign_per_roi.csv"
    val_preds = paths.sign_dir / "calibration" / "val_roi_predictions.csv"

    if per_roi.is_file():
        df = pd.read_csv(per_roi, low_memory=False)
        required = ["roi_uid", "p", "c"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"StageA per-ROI file missing columns {missing}: {per_roi}")

        join_cols = ["roi_uid", "defocus_um", "y_sign", "y_mag_um"]
        if "dataset" not in df.columns:
            join_cols.append("dataset")
        merged = df.merge(index_df[join_cols], on="roi_uid", how="left")
        merged = merged.rename(columns={"p": "prob", "c": "conf"})
        merged["source"] = "vote_sign_per_roi"
        return merged

    if val_preds.is_file():
        df = pd.read_csv(val_preds, low_memory=False)
        required = ["y_sign", "delta_z", "p", "c"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"StageA validation prediction file missing columns {missing}: {val_preds}")
        df = df.copy()
        df["defocus_um"] = pd.to_numeric(df["delta_z"], errors="coerce")
        df["y_mag_um"] = np.abs(df["defocus_um"].astype(float))
        df["prob"] = pd.to_numeric(df["p"], errors="coerce")
        df["conf"] = pd.to_numeric(df["c"], errors="coerce")
        if "dataset" not in df.columns:
            df["dataset"] = "unknown"
        df["source"] = "val_roi_predictions"
        return df

    raise FileNotFoundError(
        f"Missing StageA prediction inputs:\n- {per_roi}\n- {val_preds}"
    )


def _load_stageB_roi_df(paths, index_df: pd.DataFrame) -> pd.DataFrame:
    roi_path = require_file(paths.reg_dir / "inference" / "roi_predictions.csv", "Phase5 ROI predictions")
    roi = pd.read_csv(roi_path, low_memory=False)
    if "roi_uid" not in roi.columns:
        raise ValueError(f"ROI prediction file missing roi_uid: {roi_path}")

    if "y_mag_pred" not in roi.columns:
        if "signed_dz_pred" in roi.columns:
            roi["y_mag_pred"] = pd.to_numeric(roi["signed_dz_pred"], errors="coerce").abs()
        else:
            raise ValueError(f"ROI prediction file missing y_mag_pred and signed_dz_pred: {roi_path}")

    join_cols = ["roi_uid", "y_mag_um", "defocus_um"]
    for col in ["dataset", "fov_id"]:
        if col not in roi.columns:
            join_cols.append(col)
    merged = roi.merge(index_df[join_cols], on="roi_uid", how="left")
    merged["y_mag_pred"] = pd.to_numeric(merged["y_mag_pred"], errors="coerce")
    if "signed_dz_pred" in merged.columns:
        merged["signed_dz_pred"] = pd.to_numeric(merged["signed_dz_pred"], errors="coerce")
    else:
        sgn = pd.to_numeric(merged.get("y_sign_pred", np.nan), errors="coerce")
        mag = pd.to_numeric(merged["y_mag_pred"], errors="coerce")
        merged["signed_dz_pred"] = np.where(sgn >= 0.5, mag, -mag)
    merged["y_mag_um"] = pd.to_numeric(merged["y_mag_um"], errors="coerce")
    merged["defocus_um"] = pd.to_numeric(merged["defocus_um"], errors="coerce")
    merged["abs_err_um"] = np.abs(merged["y_mag_pred"] - merged["y_mag_um"])
    return merged


def _load_fov_eval_df(paths, index_df: pd.DataFrame) -> pd.DataFrame:
    pred_path = require_file(paths.reg_dir / "inference" / "fov_aggregate_predictions.csv", "Phase5 FOV predictions")
    preds = pd.read_csv(pred_path, low_memory=False)
    if "fov_id" not in preds.columns or "dz_hat_um" not in preds.columns:
        raise ValueError(f"FOV aggregate prediction file missing fov_id or dz_hat_um: {pred_path}")

    gt_df = _gt_per_fov(index_df)
    ds_map = infer_dataset_for_fov(index_df)
    merged = preds.merge(gt_df, on="fov_id", how="left").merge(ds_map, on="fov_id", how="left")
    merged["pred_signed_um"] = pd.to_numeric(merged["dz_hat_um"], errors="coerce")
    merged["gt_signed_um"] = pd.to_numeric(merged["gt_signed_um"], errors="coerce")
    merged["signed_error_um"] = merged["pred_signed_um"] - merged["gt_signed_um"]
    merged["abs_error_um"] = np.abs(merged["signed_error_um"])
    return merged


def _roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(y_prob)[::-1]
    pos = max(int((y_true == 1).sum()), 1)
    neg = max(int((y_true == 0).sum()), 1)
    tpr = [0.0]
    fpr = [0.0]
    for tau in thresholds:
        pred = (y_prob >= tau).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        tpr.append(tp / pos)
        fpr.append(fp / neg)
    tpr.append(1.0)
    fpr.append(1.0)
    return np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)


def _build_main_results_table(track: str, package_tables: Path, metrics: dict, manifest: list[dict], save_latex_flag: bool) -> Path:
    stage_a = metrics["stageA"]
    stage_b = metrics["stageB"]
    end2end = metrics["end_to_end"]
    runtime = metrics["runtime"]

    table = pd.DataFrame(
        [
            {
                "Track": track,
                "StageA_AUROC": stage_a.get("auroc", np.nan),
                "StageA_BalancedAccuracy": stage_a.get("balanced_accuracy", np.nan),
                "StageA_F1": stage_a.get("f1", np.nan),
                "StageB_MAE_um": stage_b.get("mae_um", np.nan),
                "StageB_RMSE_um": stage_b.get("rmse_um", np.nan),
                "EndToEnd_MAE_um": end2end.get("mae_um", np.nan),
                "EndToEnd_RMSE_um": end2end.get("rmse_um", np.nan),
                "Within_1um_pct": end2end.get("pct_within_1um", np.nan),
                "Within_2um_pct": end2end.get("pct_within_2um", np.nan),
                "CatastrophicWrongDirection_pct": end2end.get("catastrophic_wrong_direction_rate", np.nan),
                "Latency_k7_ms": runtime.get("total_pipeline_latency_per_fov_k7_ms_est", np.nan),
                "SignModel_MB": runtime.get("model_size_sign_mb", np.nan),
                "Rplus_MB": runtime.get("model_size_r_plus_mb", np.nan),
                "Rminus_MB": runtime.get("model_size_r_minus_mb", np.nan),
            }
        ]
    )
    out_path = package_tables / "Table_01_MainResults.csv"
    save_table(table, out_path, save_latex=save_latex_flag)
    _register(
        manifest,
        track=track,
        tier="main",
        category="table",
        label="Table 1",
        title="Main quantitative results",
        path=out_path,
        section="Results",
        description="Core Stage A, Stage B, end-to-end, and efficiency metrics for the selected track.",
        source="evaluation JSON summaries",
    )
    return out_path


def _build_dataset_table(track: str, cross_df: pd.DataFrame, package_tables: Path, manifest: list[dict], save_latex_flag: bool) -> Path:
    table = cross_df.rename(
        columns={
            "dataset": "Dataset",
            "stageA_auroc": "StageA_AUROC",
            "stageB_mae_um": "StageB_MAE_um",
            "end_to_end_mae_um": "EndToEnd_MAE_um",
            "stageA_source": "StageA_Source",
            "stageB_source": "StageB_Source",
            "end_to_end_source": "EndToEnd_Source",
        }
    )
    out_path = package_tables / "Table_02_DatasetGeneralization.csv"
    save_table(table, out_path, save_latex=save_latex_flag)
    _register(
        manifest,
        track=track,
        tier="main",
        category="table",
        label="Table 2",
        title="Dataset-level generalization summary",
        path=out_path,
        section="Results",
        description="Per-dataset Stage A, Stage B, and end-to-end performance summaries.",
        source="cross_dataset_results.csv",
    )
    return out_path


def _build_statistical_table(track: str, stats_json: dict, ci_df: pd.DataFrame, package_tables: Path, manifest: list[dict], save_latex_flag: bool) -> Path:
    rows = []
    for _, r in ci_df.iterrows():
        rows.append(
            {
                "Category": "Bootstrap CI",
                "Name": str(r.get("metric", "")),
                "PointEstimate": r.get("point_estimate", np.nan),
                "CI95_Lo": r.get("ci95_lo", np.nan),
                "CI95_Hi": r.get("ci95_hi", np.nan),
                "Statistic": np.nan,
                "PValue": np.nan,
                "Notes": f"n={int(r.get('n_samples', 0))}",
            }
        )

    wil = stats_json.get("wilcoxon", {})
    rows.append(
        {
            "Category": "Hypothesis test",
            "Name": "Wilcoxon",
            "PointEstimate": np.nan,
            "CI95_Lo": np.nan,
            "CI95_Hi": np.nan,
            "Statistic": wil.get("statistic", np.nan),
            "PValue": wil.get("pvalue", np.nan),
            "Notes": wil.get("comparison", ""),
        }
    )

    mcn = stats_json.get("mcnemar", {})
    rows.append(
        {
            "Category": "Hypothesis test",
            "Name": "McNemar",
            "PointEstimate": np.nan,
            "CI95_Lo": np.nan,
            "CI95_Hi": np.nan,
            "Statistic": mcn.get("chi2", np.nan),
            "PValue": mcn.get("pvalue", np.nan),
            "Notes": mcn.get("comparison", ""),
        }
    )

    fried = stats_json.get("friedman_nemenyi", {})
    rows.append(
        {
            "Category": "Hypothesis test",
            "Name": "Friedman + Nemenyi",
            "PointEstimate": np.nan,
            "CI95_Lo": np.nan,
            "CI95_Hi": np.nan,
            "Statistic": fried.get("friedman_stat", np.nan),
            "PValue": fried.get("friedman_p", np.nan),
            "Notes": f"CD={fried.get('critical_difference', np.nan)}",
        }
    )

    table = pd.DataFrame(rows)
    out_path = package_tables / "Table_03_StatisticalSummary.csv"
    save_table(table, out_path, save_latex=save_latex_flag)
    _register(
        manifest,
        track=track,
        tier="main",
        category="table",
        label="Table 3",
        title="Statistical summary and confidence intervals",
        path=out_path,
        section="Results",
        description="Bootstrap confidence intervals and non-parametric significance tests for the main comparisons.",
        source="statistical_tests_results.json + confidence_intervals.csv",
    )
    return out_path


def _build_calibration_table(track: str, tau_json: dict | None, preproc_stats: pd.DataFrame | None, package_tables: Path, manifest: list[dict], save_latex_flag: bool) -> Path:
    rows = []
    if tau_json is not None:
        rows.extend(
            [
                {"Metric": "chosen_tau", "Value": tau_json.get("tau", np.nan)},
                {"Metric": "tau_coverage", "Value": tau_json.get("coverage", np.nan)},
                {"Metric": "tau_wrong_sign_rate", "Value": tau_json.get("wrong_sign_rate", np.nan)},
                {"Metric": "tau_balanced_accuracy", "Value": tau_json.get("balanced_accuracy", np.nan)},
                {"Metric": "tau_rule_used", "Value": tau_json.get("rule_used", "")},
            ]
        )
    if preproc_stats is not None and not preproc_stats.empty:
        wanted = [
            "channel_mean_I",
            "channel_std_I",
            "channel_mean_D1",
            "channel_std_D1",
            "channel_mean_D2",
            "channel_std_D2",
            "channel_mean_E_HF",
            "channel_std_E_HF",
            "dog_time_per_roi_ms",
            "dwt_time_per_roi_ms",
            "roi_count",
            "manifest_rows",
            "total_elapsed_s",
        ]
        sub = preproc_stats[preproc_stats["metric"].astype(str).isin(wanted)].copy()
        for _, r in sub.iterrows():
            rows.append({"Metric": str(r["metric"]), "Value": r["value"]})

    out_path = package_tables / "Table_04_CalibrationAndPreprocessing.csv"
    table = pd.DataFrame(rows if rows else [{"Metric": "unavailable", "Value": np.nan}])
    save_table(table, out_path, save_latex=save_latex_flag)
    _register(
        manifest,
        track=track,
        tier="supplementary",
        category="table",
        label="Table 4",
        title="Calibration and preprocessing diagnostics",
        path=out_path,
        section="Methods / Supplementary",
        description="Tau calibration summary and preprocessing normalization/runtime statistics.",
        source="chosen_tau.json + preprocessing_stats.csv",
    )
    return out_path


def _plot_stageA_composite(track: str, df: pd.DataFrame, package_figs: Path, manifest: list[dict], dpi: int) -> Path:
    y_true = pd.to_numeric(df["y_sign"], errors="coerce").to_numpy(dtype=float)
    y_prob = pd.to_numeric(df["prob"], errors="coerce").to_numpy(dtype=float)
    conf = pd.to_numeric(df["conf"], errors="coerce").to_numpy(dtype=float)
    abs_dz = pd.to_numeric(df["y_mag_um"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob) & np.isfinite(conf) & np.isfinite(abs_dz)
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]
    conf = conf[valid]
    abs_dz = abs_dz[valid]
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr = _roc_points(y_true, y_prob)
    cm = np.array(
        [
            [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
            [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
        ]
    )

    bins = np.arange(0.0, max(float(np.nanmax(abs_dz)), 0.5) + 0.5, 0.5)
    bin_centers = []
    bin_acc = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (abs_dz >= lo) & (abs_dz < hi)
        if idx.any():
            bin_centers.append(0.5 * (lo + hi))
            bin_acc.append(float((y_true[idx] == y_pred[idx]).mean()))

    conf_bins = np.linspace(0.5, 1.0, 11)
    conf_centers = []
    conf_acc = []
    for lo, hi in zip(conf_bins[:-1], conf_bins[1:]):
        idx = (conf >= lo) & (conf < hi)
        if idx.any():
            conf_centers.append(0.5 * (lo + hi))
            conf_acc.append(float((y_true[idx] == y_pred[idx]).mean()))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=COLORS["blue"], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.grid(True)
    _panel(ax, "A")

    ax = axes[0, 1]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred -", "Pred +"])
    ax.set_yticks([0, 1], labels=["True -", "True +"])
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _panel(ax, "B")

    ax = axes[1, 0]
    ax.plot(bin_centers, bin_acc, marker="o", color=COLORS["teal"], linewidth=2)
    ax.set_xlabel(r"$|\Delta z|$ bin center (µm)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs defocus magnitude")
    ax.grid(True)
    _panel(ax, "C")

    ax = axes[1, 1]
    ax.plot(conf_centers, conf_acc, marker="o", color=COLORS["gold"], linewidth=2)
    ax.set_xlabel("Classifier confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Confidence reliability")
    ax.grid(True)
    _panel(ax, "D")

    fig.suptitle(f"Stage A sign classification performance ({track})", fontsize=13, y=1.02)
    out_path = package_figs / "Figure_01_StageA_SignPerformance.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="main",
        category="figure",
        label="Figure 1",
        title="Stage A sign classification performance",
        path=out_path,
        section="Results",
        description="ROC, confusion matrix, performance by defocus magnitude, and confidence reliability for the Stage A sign classifier.",
        source="vote_sign_per_roi.csv / val_roi_predictions.csv",
    )
    return out_path


def _plot_stageB_composite(track: str, roi_df: pd.DataFrame, cross_df: pd.DataFrame | None, package_figs: Path, manifest: list[dict], dpi: int) -> Path:
    work = roi_df.dropna(subset=["y_mag_um", "y_mag_pred"]).copy()
    plot_work = _sample_df(work, max_rows=80000, seed=42)
    y_true = plot_work["y_mag_um"].to_numpy(dtype=float)
    abs_err = np.abs(plot_work["y_mag_pred"].to_numpy(dtype=float) - y_true)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.hist(abs_err, bins=40, color=COLORS["green"], edgecolor="white")
    ax.set_xlabel("Absolute error (µm)")
    ax.set_ylabel("Count")
    ax.set_title("Error histogram")
    ax.grid(True)
    _panel(ax, "A")

    ax = axes[0, 1]
    ax.scatter(y_true, abs_err, s=8, alpha=0.18, color=COLORS["blue"])
    edges = np.linspace(float(np.nanmin(y_true)), float(np.nanmax(y_true) + 1e-9), 18)
    centers = []
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (y_true >= lo) & (y_true < hi)
        if idx.any():
            centers.append(0.5 * (lo + hi))
            means.append(float(np.mean(abs_err[idx])))
    if centers:
        ax.plot(centers, means, color=COLORS["red"], linewidth=2)
    ax.set_xlabel(r"Ground-truth $|\Delta z|$ (µm)")
    ax.set_ylabel("Absolute error (µm)")
    ax.set_title("Error as a function of defocus magnitude")
    ax.grid(True)
    _panel(ax, "B")

    ax = axes[1, 0]
    bin_width = 0.5
    edges = np.arange(0, float(np.nanmax(y_true) + bin_width + 1e-9), bin_width)
    groups = []
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (y_true >= lo) & (y_true < hi)
        if idx.any():
            groups.append(abs_err[idx])
            labels.append(f"{lo:.1f}-{hi:.1f}")
    if groups:
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_xlabel(r"$|\Delta z|$ bin (µm)")
    ax.set_ylabel("Absolute error (µm)")
    ax.set_title("Distribution of error by magnitude bin")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True)
    _panel(ax, "C")

    ax = axes[1, 1]
    if cross_df is not None and not cross_df.empty and "stageB_mae_um" in cross_df.columns:
        tmp = cross_df.copy().sort_values("dataset")
        ax.bar(tmp["dataset"].astype(str), pd.to_numeric(tmp["stageB_mae_um"], errors="coerce"), color=COLORS["purple"])
        ax.set_ylabel("MAE (µm)")
        ax.set_title("Dataset-specific Stage B MAE")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "Dataset-level Stage B summary unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "D")

    fig.suptitle(f"Stage B magnitude regression performance ({track})", fontsize=13, y=1.02)
    out_path = package_figs / "Figure_02_StageB_MagnitudeRegression.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="main",
        category="figure",
        label="Figure 2",
        title="Stage B magnitude regression performance",
        path=out_path,
        section="Results",
        description="Absolute error distribution, error-versus-magnitude behavior, bin-wise spread, and dataset-level magnitude regression summary.",
        source="roi_predictions.csv + cross_dataset_results.csv",
    )
    return out_path


def _plot_end_to_end_composite(track: str, fov_df: pd.DataFrame, package_figs: Path, manifest: list[dict], dpi: int) -> Path:
    work = fov_df.dropna(subset=["pred_signed_um", "gt_signed_um"]).copy()
    plot_work = _sample_df(work, max_rows=80000, seed=42)
    err = plot_work["pred_signed_um"].to_numpy(dtype=float) - plot_work["gt_signed_um"].to_numpy(dtype=float)
    abs_err = np.abs(err)
    gt = plot_work["gt_signed_um"].to_numpy(dtype=float)
    pred = plot_work["pred_signed_um"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.hist(err, bins=40, color=COLORS["red"], edgecolor="white")
    ax.set_xlabel("Signed error (µm)")
    ax.set_ylabel("Count")
    ax.set_title("Signed error histogram")
    ax.grid(True)
    _panel(ax, "A")

    ax = axes[0, 1]
    x = np.sort(abs_err)
    y = np.arange(1, len(x) + 1) / max(len(x), 1)
    ax.plot(x, y, color=COLORS["blue"], linewidth=2)
    ax.axvline(1.0, linestyle="--", color=COLORS["gray"], linewidth=1)
    ax.axvline(2.0, linestyle="--", color=COLORS["gray"], linewidth=1)
    ax.set_xlabel("Absolute error (µm)")
    ax.set_ylabel("CDF")
    ax.set_title("Error cumulative distribution")
    ax.grid(True)
    _panel(ax, "B")

    ax = axes[1, 0]
    ax.scatter(gt, pred, s=12, alpha=0.25, color=COLORS["teal"])
    lim = float(np.nanmax(np.abs(np.concatenate([gt, pred])))) if len(gt) else 1.0
    lim = max(lim, 1.0)
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color=COLORS["gray"], linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Ground-truth signed defocus (µm)")
    ax.set_ylabel("Predicted signed defocus (µm)")
    ax.set_title("Prediction-versus-ground-truth scatter")
    ax.grid(True)
    _panel(ax, "C")

    ax = axes[1, 1]
    mean_pair = 0.5 * (gt + pred)
    diff = pred - gt
    bias = float(np.mean(diff))
    sd = float(np.std(diff))
    loa_hi = bias + 1.96 * sd
    loa_lo = bias - 1.96 * sd
    ax.scatter(mean_pair, diff, s=12, alpha=0.22, color=COLORS["gold"])
    ax.axhline(bias, color=COLORS["red"], linewidth=2, label="Bias")
    ax.axhline(loa_hi, color=COLORS["gray"], linestyle="--", linewidth=1, label="95% LoA")
    ax.axhline(loa_lo, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("Mean of prediction and reference (µm)")
    ax.set_ylabel("Prediction - reference (µm)")
    ax.set_title("Bland–Altman analysis")
    ax.legend(loc="upper right")
    ax.grid(True)
    _panel(ax, "D")

    fig.suptitle(f"End-to-end autofocus accuracy ({track})", fontsize=13, y=1.02)
    out_path = package_figs / "Figure_03_EndToEnd_Autofocus.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="main",
        category="figure",
        label="Figure 3",
        title="End-to-end autofocus accuracy",
        path=out_path,
        section="Results",
        description="Signed error distribution, cumulative error profile, prediction-versus-reference scatter, and Bland–Altman analysis at the FOV level.",
        source="fov_aggregate_predictions.csv + index_phase5.csv",
    )
    return out_path


def _plot_system_composite(track: str, runtime_json: dict, latency_df: pd.DataFrame, ablation_df: pd.DataFrame, cross_df: pd.DataFrame | None, package_figs: Path, manifest: list[dict], dpi: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    labels = ["Preprocess", "Stage A", "Stage B", "Aggregate"]
    vals = [
        runtime_json.get("preprocess_time_per_roi_ms", np.nan),
        runtime_json.get("stageA_time_per_roi_ms", np.nan),
        runtime_json.get("stageB_time_per_roi_ms", np.nan),
        runtime_json.get("aggregation_time_per_fov_ms", np.nan),
    ]
    ax.bar(labels, vals, color=[COLORS["gray"], COLORS["blue"], COLORS["green"], COLORS["gold"]])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Runtime breakdown")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y")
    _panel(ax, "A")

    ax = axes[0, 1]
    ax.plot(pd.to_numeric(latency_df["k"], errors="coerce"), pd.to_numeric(latency_df["latency_ms"], errors="coerce"), marker="o", color=COLORS["red"], linewidth=2)
    ax.set_xlabel("k ROIs per FOV")
    ax.set_ylabel("Latency per FOV (ms)")
    ax.set_title("Latency sensitivity to ROI budget")
    ax.grid(True)
    _panel(ax, "B")

    ax = axes[1, 0]
    end_ab = ablation_df[(ablation_df["component"].astype(str) == "End-to-End") & (ablation_df["metric"].astype(str) == "MAE_um")].copy()
    end_ab["value"] = pd.to_numeric(end_ab["value"], errors="coerce")
    end_ab = end_ab[end_ab["available"].astype(int) == 1]
    if not end_ab.empty:
        ax.bar(end_ab["variant"].astype(str), end_ab["value"], color=COLORS["teal"])
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("MAE (µm)")
        ax.set_title("Aggregation ablation")
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "Ablation data unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "C")

    ax = axes[1, 1]
    if cross_df is not None and not cross_df.empty and "end_to_end_mae_um" in cross_df.columns:
        tmp = cross_df.copy().sort_values("dataset")
        ax.bar(tmp["dataset"].astype(str), pd.to_numeric(tmp["end_to_end_mae_um"], errors="coerce"), color=COLORS["purple"])
        ax.set_ylabel("End-to-end MAE (µm)")
        ax.set_title("Cross-dataset end-to-end generalization")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "Cross-dataset summary unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "D")

    fig.suptitle(f"Efficiency, ablation, and robustness summary ({track})", fontsize=13, y=1.02)
    out_path = package_figs / "Figure_04_Efficiency_Ablation_Generalization.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="main",
        category="figure",
        label="Figure 4",
        title="Efficiency, ablation, and generalization summary",
        path=out_path,
        section="Results",
        description="Runtime decomposition, latency-versus-k profile, end-to-end aggregation ablation, and dataset-level generalization summary.",
        source="runtime_metrics.json + ablation_results.csv + cross_dataset_results.csv",
    )
    return out_path


def _plot_tau_calibration(track: str, tau_df: pd.DataFrame | None, tau_json: dict | None, package_figs: Path, manifest: list[dict], dpi: int) -> Path | None:
    if tau_df is None or tau_json is None or tau_df.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = pd.to_numeric(tau_df["tau"], errors="coerce").to_numpy(dtype=float)
    chosen_tau = float(tau_json.get("tau", np.nan))

    specs = [
        ("coverage", "Coverage", COLORS["blue"]),
        ("wrong_sign_rate", "Wrong-sign rate", COLORS["red"]),
        ("balanced_accuracy", "Balanced accuracy", COLORS["green"]),
    ]
    for i, (col, title, color) in enumerate(specs):
        y = pd.to_numeric(tau_df[col], errors="coerce").to_numpy(dtype=float)
        ax = axes[i]
        ax.plot(x, y, color=color, linewidth=2)
        if np.isfinite(chosen_tau):
            ax.axvline(chosen_tau, linestyle="--", color=COLORS["gray"], linewidth=1)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True)
        _panel(ax, chr(ord("A") + i))

    fig.suptitle(f"Tau calibration diagnostics ({track})", fontsize=13, y=1.05)
    out_path = package_figs / "Figure_S01_TauCalibration.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="supplementary",
        category="figure",
        label="Figure S1",
        title="Tau calibration diagnostics",
        path=out_path,
        section="Supplementary / Calibration",
        description="Coverage, wrong-sign rate, and balanced accuracy as functions of the gating threshold tau, with the selected operating point marked.",
        source="tau_sweep.csv + chosen_tau.json",
    )
    return out_path


def _plot_training_dynamics(track: str, sign_hist: pd.DataFrame | None, reg_plus: pd.DataFrame | None, reg_minus: pd.DataFrame | None, package_figs: Path, manifest: list[dict], dpi: int) -> Path | None:
    if sign_hist is None and reg_plus is None and reg_minus is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    if sign_hist is not None and not sign_hist.empty:
        sign = sign_hist.copy().reset_index(drop=True)
        sign["epoch"] = np.arange(1, len(sign) + 1)
        if "auc" in sign.columns:
            ax.plot(sign["epoch"], pd.to_numeric(sign["auc"], errors="coerce"), label="Train AUC", color=COLORS["blue"])
        if "val_auc" in sign.columns:
            ax.plot(sign["epoch"], pd.to_numeric(sign["val_auc"], errors="coerce"), label="Val AUC", color=COLORS["teal"])
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title("Sign model discrimination")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Sign history unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "A")

    ax = axes[0, 1]
    if sign_hist is not None and not sign_hist.empty:
        sign = sign_hist.copy().reset_index(drop=True)
        sign["epoch"] = np.arange(1, len(sign) + 1)
        if "loss" in sign.columns:
            ax.plot(sign["epoch"], pd.to_numeric(sign["loss"], errors="coerce"), label="Train loss", color=COLORS["gold"])
        if "val_loss" in sign.columns:
            ax.plot(sign["epoch"], pd.to_numeric(sign["val_loss"], errors="coerce"), label="Val loss", color=COLORS["red"])
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Sign model optimization")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Sign history unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "B")

    ax = axes[1, 0]
    plotted = False
    if reg_plus is not None and not reg_plus.empty:
        ax.plot(reg_plus["epoch"], pd.to_numeric(reg_plus["val_mae_um"], errors="coerce"), label="R+ val MAE", color=COLORS["green"])
        plotted = True
    if reg_minus is not None and not reg_minus.empty:
        ax.plot(reg_minus["epoch"], pd.to_numeric(reg_minus["val_mae_um"], errors="coerce"), label="R- val MAE", color=COLORS["purple"])
        plotted = True
    if plotted:
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE (µm)")
        ax.set_title("Regression validation error")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Regression history unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "C")

    ax = axes[1, 1]
    plotted = False
    if reg_plus is not None and not reg_plus.empty:
        ax.plot(reg_plus["epoch"], pd.to_numeric(reg_plus["train_triplet_loss"], errors="coerce"), label="R+ triplet", color=COLORS["green"])
        plotted = True
    if reg_minus is not None and not reg_minus.empty:
        ax.plot(reg_minus["epoch"], pd.to_numeric(reg_minus["train_triplet_loss"], errors="coerce"), label="R- triplet", color=COLORS["purple"])
        plotted = True
    if plotted:
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Triplet loss")
        ax.set_title("Embedding regularization dynamics")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Regression history unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "D")

    fig.suptitle(f"Training dynamics ({track})", fontsize=13, y=1.02)
    out_path = package_figs / "Figure_S02_TrainingDynamics.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="supplementary",
        category="figure",
        label="Figure S2",
        title="Training dynamics",
        path=out_path,
        section="Supplementary / Optimization",
        description="Optimization dynamics for the sign classifier and branch regressors, including validation MAE and triplet regularization trends.",
        source="history.csv + history_plus.csv + history_minus.csv",
    )
    return out_path


def _plot_dataset_error_profile(track: str, roi_df: pd.DataFrame, fov_df: pd.DataFrame, package_figs: Path, manifest: list[dict], dpi: int) -> Path | None:
    if ("dataset" not in roi_df.columns) and ("dataset" not in fov_df.columns):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    if "dataset" in roi_df.columns:
        tmp = _sample_df(roi_df.dropna(subset=["dataset", "abs_err_um"]).copy(), max_rows=80000, seed=42)
        order = sorted(tmp["dataset"].astype(str).unique().tolist())
        groups = [tmp[tmp["dataset"].astype(str) == ds]["abs_err_um"].to_numpy(dtype=float) for ds in order]
        if groups:
            ax.boxplot(groups, tick_labels=order, showfliers=False)
            ax.set_ylabel("ROI absolute error (µm)")
            ax.set_title("Stage B ROI error by dataset")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No ROI dataset data", ha="center", va="center")
            ax.axis("off")
    _panel(ax, "A")

    ax = axes[0, 1]
    if "dataset" in fov_df.columns:
        tmp = _sample_df(fov_df.dropna(subset=["dataset", "abs_error_um"]).copy(), max_rows=80000, seed=42)
        order = sorted(tmp["dataset"].astype(str).unique().tolist())
        groups = [tmp[tmp["dataset"].astype(str) == ds]["abs_error_um"].to_numpy(dtype=float) for ds in order]
        if groups:
            ax.boxplot(groups, tick_labels=order, showfliers=False)
            ax.set_ylabel("FOV absolute error (µm)")
            ax.set_title("End-to-end error by dataset")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No FOV dataset data", ha="center", va="center")
            ax.axis("off")
    _panel(ax, "B")

    ax = axes[1, 0]
    if "dataset" in fov_df.columns:
        tmp = fov_df.copy()
        tmp["uncertain"] = tmp["pred_signed_um"].isna().astype(int)
        bars = tmp.groupby("dataset", as_index=False)["uncertain"].mean()
        bars["uncertain"] *= 100.0
        ax.bar(bars["dataset"].astype(str), bars["uncertain"], color=COLORS["gold"])
        ax.set_ylabel("Uncertain predictions (%)")
        ax.set_title("Uncertainty rate by dataset")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y")
    _panel(ax, "C")

    ax = axes[1, 1]
    if "dataset" in fov_df.columns:
        tmp = fov_df.dropna(subset=["dataset", "pred_signed_um", "gt_signed_um"]).copy()
        tmp["catastrophic"] = ((np.sign(tmp["pred_signed_um"]) != np.sign(tmp["gt_signed_um"])) & (~np.isclose(tmp["gt_signed_um"], 0.0))).astype(int)
        bars = tmp.groupby("dataset", as_index=False)["catastrophic"].mean()
        bars["catastrophic"] *= 100.0
        ax.bar(bars["dataset"].astype(str), bars["catastrophic"], color=COLORS["red"])
        ax.set_ylabel("Catastrophic wrong-direction rate (%)")
        ax.set_title("Wrong-direction rate by dataset")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y")
    _panel(ax, "D")

    fig.suptitle(f"Dataset-stratified error profile ({track})", fontsize=13, y=1.02)
    out_path = package_figs / "Figure_S03_DatasetErrorProfiles.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="supplementary",
        category="figure",
        label="Figure S3",
        title="Dataset-stratified error profile",
        path=out_path,
        section="Supplementary / Robustness",
        description="ROI-level and FOV-level error distributions, uncertainty rate, and catastrophic wrong-direction rate stratified by dataset.",
        source="roi_predictions.csv + fov_aggregate_predictions.csv + index_phase5.csv",
    )
    return out_path


def _plot_roi_diagnostics(track: str, sign_roi_df: pd.DataFrame, reg_roi_df: pd.DataFrame, fov_df: pd.DataFrame, package_figs: Path, manifest: list[dict], dpi: int) -> Path | None:
    if sign_roi_df is None or sign_roi_df.empty:
        return None

    sign_plot = _sample_df(sign_roi_df, max_rows=100000, seed=42)
    reg_plot = _sample_df(reg_roi_df, max_rows=100000, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    if "conf" in sign_plot.columns:
        ax.hist(pd.to_numeric(sign_plot["conf"], errors="coerce"), bins=40, color=COLORS["blue"], edgecolor="white")
    ax.set_xlabel("ROI confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution")
    ax.grid(True)
    _panel(ax, "A")

    ax = axes[0, 1]
    if "kept" in sign_roi_df.columns:
        kept = int(pd.to_numeric(sign_roi_df["kept"], errors="coerce").fillna(0).astype(int).sum())
        total = int(len(sign_roi_df))
        dropped = max(total - kept, 0)
        ax.bar(["Kept", "Dropped"], [kept, dropped], color=[COLORS["green"], COLORS["red"]])
        ax.set_ylabel("ROI count")
        ax.set_title("Tau-gating retention")
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "kept column unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "B")

    ax = axes[1, 0]
    if "vote_margin" in fov_df.columns:
        ax.hist(pd.to_numeric(fov_df["vote_margin"], errors="coerce"), bins=30, color=COLORS["purple"], edgecolor="white")
        ax.set_xlabel("Vote margin")
        ax.set_ylabel("Count")
        ax.set_title("FOV vote-margin distribution")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "vote_margin unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "C")

    ax = axes[1, 1]
    if ("c_sign" in reg_plot.columns) and ("weight" in reg_plot.columns):
        x = pd.to_numeric(reg_plot["c_sign"], errors="coerce")
        y = pd.to_numeric(reg_plot["weight"], errors="coerce")
        mask = x.notna() & y.notna()
        ax.scatter(x[mask], y[mask], s=8, alpha=0.2, color=COLORS["gold"])
        ax.set_xlabel("Sign confidence")
        ax.set_ylabel("Final ROI weight")
        ax.set_title("Weight assignment versus confidence")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "weight/confidence diagnostics unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "D")

    fig.suptitle(f"ROI voting and routing diagnostics ({track})", fontsize=13, y=1.02)
    out_path = package_figs / "Figure_S04_ROIVotingDiagnostics.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="supplementary",
        category="figure",
        label="Figure S4",
        title="ROI voting and routing diagnostics",
        path=out_path,
        section="Supplementary / Inference diagnostics",
        description="ROI confidence distribution, tau-gating retention, FOV vote margins, and final regression aggregation weights.",
        source="vote_sign_per_roi.csv + roi_predictions.csv + fov_aggregate_predictions.csv",
    )
    return out_path


def _plot_triplet_preproc(track: str, sampling_stats: pd.DataFrame | None, preproc_stats: pd.DataFrame | None, package_figs: Path, manifest: list[dict], dpi: int) -> Path | None:
    if (sampling_stats is None or sampling_stats.empty) and (preproc_stats is None or preproc_stats.empty):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    if sampling_stats is not None and not sampling_stats.empty:
        plot_df = sampling_stats.copy()
        cols = ["anchors_seen", "anchors_skipped_zero", "anchors_no_positive", "anchors_no_negative", "triplets_built"]
        width = 0.14
        x = np.arange(len(plot_df))
        for i, col in enumerate(cols):
            vals = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ax.bar(x + i * width, vals, width=width, label=col)
        ax.set_xticks(x + width * 2, plot_df["branch"].astype(str).tolist())
        ax.set_ylabel("Count")
        ax.set_title("Triplet sampling diagnostics")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "Triplet sampling stats unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "A")

    ax = axes[0, 1]
    if preproc_stats is not None and not preproc_stats.empty:
        wanted = ["dog_time_per_roi_ms", "dwt_time_per_roi_ms"]
        sub = preproc_stats[preproc_stats["metric"].astype(str).isin(wanted)].copy()
        ax.bar(sub["metric"].astype(str), pd.to_numeric(sub["value"], errors="coerce"), color=[COLORS["blue"], COLORS["teal"]])
        ax.set_ylabel("ms per ROI")
        ax.set_title("Preprocessing time components")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "Preprocessing stats unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "B")

    ax = axes[1, 0]
    if preproc_stats is not None and not preproc_stats.empty:
        means = []
        stds = []
        labels = []
        for channel in ["I", "D1", "D2", "E_HF"]:
            m = preproc_stats.loc[preproc_stats["metric"].astype(str) == f"channel_mean_{channel}", "value"]
            s = preproc_stats.loc[preproc_stats["metric"].astype(str) == f"channel_std_{channel}", "value"]
            if not m.empty and not s.empty:
                labels.append(channel)
                means.append(float(m.iloc[0]))
                stds.append(float(s.iloc[0]))
        x = np.arange(len(labels))
        ax.bar(x - 0.15, means, width=0.3, label="Mean", color=COLORS["gold"])
        ax.bar(x + 0.15, stds, width=0.3, label="Std", color=COLORS["green"])
        ax.set_xticks(x, labels)
        ax.set_ylabel("Value")
        ax.set_title("Post-normalization channel statistics")
        ax.legend()
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "Channel statistics unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "C")

    ax = axes[1, 1]
    if preproc_stats is not None and not preproc_stats.empty:
        info = []
        for metric in ["roi_count", "manifest_rows", "total_elapsed_s"]:
            val = preproc_stats.loc[preproc_stats["metric"].astype(str) == metric, "value"]
            if not val.empty:
                info.append(f"{metric}: {float(val.iloc[0]):,.3f}")
        ax.text(0.02, 0.98, "\n".join(info) if info else "No summary statistics", va="top", ha="left")
        ax.set_title("Preprocessing run summary")
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Run summary unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "D")

    fig.suptitle(f"Triplet and preprocessing diagnostics ({track})", fontsize=13, y=1.02)
    out_path = package_figs / "Figure_S05_TripletAndPreprocessingDiagnostics.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="supplementary",
        category="figure",
        label="Figure S5",
        title="Triplet and preprocessing diagnostics",
        path=out_path,
        section="Supplementary / Methods diagnostics",
        description="Triplet sampler utilization and preprocessing normalization/runtime diagnostics.",
        source="sampling_stats.csv + preprocessing_stats.csv",
    )
    return out_path


def _plot_statistical_summary(track: str, ci_df: pd.DataFrame, stats_json: dict, package_figs: Path, manifest: list[dict], dpi: int) -> Path | None:
    if ci_df is None or ci_df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    work = ci_df.copy()
    work = work.reset_index(drop=True)
    y = np.arange(len(work))
    point = pd.to_numeric(work["point_estimate"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(work["ci95_lo"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(work["ci95_hi"], errors="coerce").to_numpy(dtype=float)
    err = np.vstack([point - lo, hi - point])
    ax.errorbar(point, y, xerr=err, fmt="o", color=COLORS["blue"], ecolor=COLORS["gray"], capsize=4)
    ax.set_yticks(y, work["metric"].astype(str).tolist())
    ax.set_xlabel("Estimate and 95% CI")
    ax.set_title("Bootstrap confidence intervals")
    ax.grid(True, axis="x")
    _panel(ax, "A")

    ax = axes[1]
    fried = stats_json.get("friedman_nemenyi", {})
    ranks = fried.get("avg_ranks", {})
    if ranks:
        items = sorted(ranks.items(), key=lambda kv: kv[1])
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        ax.barh(labels, vals, color=COLORS["purple"])
        cd = fried.get("critical_difference", np.nan)
        ax.set_xlabel("Average rank (lower is better)")
        ax.set_title("Model ranking (Friedman/Nemenyi)")
        if np.isfinite(cd):
            ax.text(0.98, 0.05, f"CD = {cd:.4f}", transform=ax.transAxes, ha="right", va="bottom")
        ax.grid(True, axis="x")
    else:
        ax.text(0.5, 0.5, "Rank summary unavailable", ha="center", va="center")
        ax.axis("off")
    _panel(ax, "B")

    fig.suptitle(f"Statistical evidence summary ({track})", fontsize=13, y=1.03)
    out_path = package_figs / "Figure_S06_StatisticalSummary.png"
    _save_fig(fig, out_path, dpi=dpi)
    _register(
        manifest,
        track=track,
        tier="supplementary",
        category="figure",
        label="Figure S6",
        title="Statistical evidence summary",
        path=out_path,
        section="Supplementary / Statistical validation",
        description="Bootstrap confidence intervals for key metrics and average-rank summary from Friedman/Nemenyi analysis.",
        source="confidence_intervals.csv + statistical_tests_results.json",
    )
    return out_path


def _copy_reference_assets(track: str, paths, package_root: Path, manifest: list[dict]) -> None:
    ref_eval = package_root / "reference" / "evaluation_outputs"
    ref_tables = package_root / "reference" / "tables"
    ref_sources = package_root / "reference" / "source_data"
    ref_eval.mkdir(parents=True, exist_ok=True)
    ref_tables.mkdir(parents=True, exist_ok=True)
    ref_sources.mkdir(parents=True, exist_ok=True)

    for pattern in ["*.png", "*.csv", "*.json"]:
        for src in sorted(paths.eval_dir.glob(pattern)):
            _copy_reference_file(
                src,
                ref_eval / src.name,
                manifest,
                track,
                label=f"Reference::{src.name}",
                title=src.name,
                description="Raw evaluation output copied into the paper package.",
            )

    for pattern in ["Table_*.csv", "Table_*.tex"]:
        for src in sorted(paths.tables_dir.glob(pattern)):
            _copy_reference_file(
                src,
                ref_tables / src.name,
                manifest,
                track,
                label=f"Reference::{src.name}",
                title=src.name,
                description="Raw evaluation table copied into the paper package.",
            )

    source_files = [
        paths.sign_dir / "calibration" / "tau_sweep.csv",
        paths.sign_dir / "calibration" / "chosen_tau.json",
        paths.sign_dir / "metrics" / "history.csv",
        paths.reg_dir / "metrics" / "history_plus.csv",
        paths.reg_dir / "metrics" / "history_minus.csv",
        paths.reg_dir / "triplets" / "sampling_stats.csv",
        paths.sign_dir.parent / "metrics" / "preprocessing_stats.csv",
        paths.sign_dir / "inference" / "vote_sign_results.csv",
    ]
    for src in source_files:
        _copy_reference_file(
            src,
            ref_sources / src.name,
            manifest,
            track,
            label=f"Source::{src.name}",
            title=src.name,
            description="Upstream CSV/JSON source copied to make the paper package self-contained.",
        )


def _write_summary_md(track: str, package_meta: Path, metrics: dict, manifest_df: pd.DataFrame) -> Path:
    stage_a = metrics["stageA"]
    stage_b = metrics["stageB"]
    end2end = metrics["end_to_end"]
    runtime = metrics["runtime"]

    content = f"""# Paper Package Summary: {track}

## Recommended Main Results

- Stage A AUROC: {stage_a.get("auroc", np.nan):.4f}
- Stage A balanced accuracy: {stage_a.get("balanced_accuracy", np.nan):.4f}
- Stage B MAE: {stage_b.get("mae_um", np.nan):.4f} µm
- Stage B RMSE: {stage_b.get("rmse_um", np.nan):.4f} µm
- End-to-end MAE: {end2end.get("mae_um", np.nan):.4f} µm
- End-to-end RMSE: {end2end.get("rmse_um", np.nan):.4f} µm
- End-to-end within ±1 µm: {end2end.get("pct_within_1um", np.nan):.2f} %
- Catastrophic wrong-direction rate: {end2end.get("catastrophic_wrong_direction_rate", np.nan):.2f} %
- Estimated latency at k=7: {runtime.get("total_pipeline_latency_per_fov_k7_ms_est", np.nan):.2f} ms

## Suggested Main-Paper Assets

- Figure 1: Stage A sign classification performance
- Figure 2: Stage B magnitude regression performance
- Figure 3: End-to-end autofocus accuracy
- Figure 4: Efficiency, ablation, and generalization summary
- Table 1: Main quantitative results
- Table 2: Dataset-level generalization summary
- Table 3: Statistical summary and confidence intervals

## Package Inventory

- Total assets indexed: {len(manifest_df)}
- Main-tier assets: {int((manifest_df['tier'] == 'main').sum())}
- Supplementary assets: {int((manifest_df['tier'] == 'supplementary').sum())}
- Reference copies: {int((manifest_df['tier'] == 'reference').sum())}
"""
    out_path = package_meta / "results_summary.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def _write_package_config(track: str, package_meta: Path, package_root: Path, save_latex_flag: bool, manifest: list[dict]) -> Path:
    payload = {
        "track": track,
        "package_root": str(package_root),
        "save_latex": bool(save_latex_flag),
        "n_assets": int(len(manifest)),
        "script": str(Path(__file__).resolve()),
    }
    out_path = package_meta / "package_config.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _build_track_package(track: str, out_root: Path, args: argparse.Namespace) -> None:
    paths = get_paths(track)
    package_root = out_root
    figs_main = package_root / "figures" / "main"
    figs_supp = package_root / "figures" / "supplementary"
    tables_main = package_root / "tables" / "main"
    tables_ref = package_root / "tables" / "reference"
    meta_dir = package_root / "metadata"
    for p in [figs_main, figs_supp, tables_main, tables_ref, meta_dir]:
        p.mkdir(parents=True, exist_ok=True)

    log_path = meta_dir / "paper_package_log.txt"
    required_resume = [
        figs_main / "Figure_01_StageA_SignPerformance.png",
        figs_main / "Figure_02_StageB_MagnitudeRegression.png",
        figs_main / "Figure_03_EndToEnd_Autofocus.png",
        figs_main / "Figure_04_Efficiency_Ablation_Generalization.png",
        tables_main / "Table_01_MainResults.csv",
        tables_main / "Table_02_DatasetGeneralization.csv",
        tables_main / "Table_03_StatisticalSummary.csv",
        meta_dir / "paper_asset_manifest.csv",
        meta_dir / "results_summary.md",
        package_root / "main_paper",
        package_root / "supplement",
        package_root / "excluded_or_quarantined",
        package_root / "paper_package_manifest.csv",
        package_root / "paper_package_summary.md",
    ]

    if args.force and args.resume:
        raise ValueError("--force and --resume cannot be used together.")

    if args.resume and all(p.is_file() for p in required_resume):
        _log(log_path, f"[INFO] Resume: paper package already exists for track={track}; skipping. ({package_root})")
        return

    _log(log_path, f"[INFO] Building Q1 paper package for track={track}")
    _ensure_evaluation_outputs(track, save_latex=args.save_latex, skip_evaluation=args.skip_evaluation, log_path=log_path)
    _ensure_manuscript_support_outputs(track, save_latex=args.save_latex, log_path=log_path)

    manifest: list[dict] = []
    index_df = _load_phase5_index_local(track)

    stage_a_metrics = _read_json(require_file(paths.eval_dir / "stageA_metrics.json", "Stage A metrics"))
    stage_b_metrics = _read_json(require_file(paths.eval_dir / "stageB_metrics.json", "Stage B metrics"))
    end_metrics = _read_json(require_file(paths.eval_dir / "end_to_end_metrics.json", "end-to-end metrics"))
    runtime_metrics = _read_json(require_file(paths.eval_dir / "runtime_metrics.json", "runtime metrics"))
    cross_df = pd.read_csv(require_file(paths.eval_dir / "cross_dataset_results.csv", "cross-dataset results"))
    ablation_df = pd.read_csv(require_file(paths.eval_dir / "ablation_results.csv", "ablation results"))
    ci_df = pd.read_csv(require_file(paths.eval_dir / "confidence_intervals.csv", "confidence intervals"))
    stats_json = _read_json(require_file(paths.eval_dir / "statistical_tests_results.json", "statistical tests"))
    latency_df = pd.read_csv(require_file(paths.eval_dir / "latency_vs_k.csv", "latency profile"))

    tau_df = None
    tau_json = None
    tau_path = paths.sign_dir / "calibration" / "tau_sweep.csv"
    tau_json_path = paths.sign_dir / "calibration" / "chosen_tau.json"
    if tau_path.is_file():
        tau_df = pd.read_csv(tau_path)
    if tau_json_path.is_file():
        tau_json = _read_json(tau_json_path)

    sign_hist = None
    if (paths.sign_dir / "metrics" / "history.csv").is_file():
        sign_hist = pd.read_csv(paths.sign_dir / "metrics" / "history.csv")

    reg_plus = None
    reg_minus = None
    if (paths.reg_dir / "metrics" / "history_plus.csv").is_file():
        reg_plus = pd.read_csv(paths.reg_dir / "metrics" / "history_plus.csv")
    if (paths.reg_dir / "metrics" / "history_minus.csv").is_file():
        reg_minus = pd.read_csv(paths.reg_dir / "metrics" / "history_minus.csv")

    sampling_stats = None
    if (paths.reg_dir / "triplets" / "sampling_stats.csv").is_file():
        sampling_stats = pd.read_csv(paths.reg_dir / "triplets" / "sampling_stats.csv")

    preproc_stats = None
    if (paths.sign_dir.parent / "metrics" / "preprocessing_stats.csv").is_file():
        preproc_stats = pd.read_csv(paths.sign_dir.parent / "metrics" / "preprocessing_stats.csv")

    stage_a_df = _load_stageA_df(paths, index_df)
    stage_b_roi_df = _load_stageB_roi_df(paths, index_df)
    fov_df = _load_fov_eval_df(paths, index_df)

    metrics = {
        "stageA": stage_a_metrics,
        "stageB": stage_b_metrics,
        "end_to_end": end_metrics,
        "runtime": runtime_metrics,
    }

    _build_main_results_table(track, tables_main, metrics, manifest, args.save_latex)
    _build_dataset_table(track, cross_df, tables_main, manifest, args.save_latex)
    _build_statistical_table(track, stats_json, ci_df, tables_main, manifest, args.save_latex)
    _build_calibration_table(track, tau_json, preproc_stats, tables_main, manifest, args.save_latex)

    _plot_stageA_composite(track, stage_a_df, figs_main, manifest, args.dpi)
    _plot_stageB_composite(track, stage_b_roi_df, cross_df, figs_main, manifest, args.dpi)
    _plot_end_to_end_composite(track, fov_df, figs_main, manifest, args.dpi)
    _plot_system_composite(track, runtime_metrics, latency_df, ablation_df, cross_df, figs_main, manifest, args.dpi)
    _plot_tau_calibration(track, tau_df, tau_json, figs_supp, manifest, args.dpi)
    _plot_training_dynamics(track, sign_hist, reg_plus, reg_minus, figs_supp, manifest, args.dpi)
    _plot_dataset_error_profile(track, stage_b_roi_df, fov_df, figs_supp, manifest, args.dpi)
    _plot_roi_diagnostics(track, stage_a_df, stage_b_roi_df, fov_df, figs_supp, manifest, args.dpi)
    _plot_triplet_preproc(track, sampling_stats, preproc_stats, figs_supp, manifest, args.dpi)
    _plot_statistical_summary(track, ci_df, stats_json, figs_supp, manifest, args.dpi)

    _copy_reference_assets(track, paths, package_root, manifest)

    manifest_df = pd.DataFrame(manifest).sort_values(["tier", "category", "label", "title"]).reset_index(drop=True)
    manifest_path = meta_dir / "paper_asset_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary_path = _write_summary_md(track, meta_dir, metrics, manifest_df)
    config_path = _write_package_config(track, meta_dir, package_root, args.save_latex, manifest)

    _register(
        manifest,
        track=track,
        tier="metadata",
        category="metadata",
        label="Manifest",
        title="Paper asset manifest",
        path=manifest_path,
        section="Metadata",
        description="Index of all generated and copied assets in the paper package.",
        source="paper package builder",
        status="generated",
    )
    _register(
        manifest,
        track=track,
        tier="metadata",
        category="metadata",
        label="Summary",
        title="Results summary",
        path=summary_path,
        section="Metadata",
        description="Concise manuscript-oriented summary of the main quantitative findings.",
        source="paper package builder",
        status="generated",
    )
    _register(
        manifest,
        track=track,
        tier="metadata",
        category="metadata",
        label="Config",
        title="Package config",
        path=config_path,
        section="Metadata",
        description="Execution metadata for the generated paper package.",
        source="paper package builder",
        status="generated",
    )

    manifest_df = pd.DataFrame(manifest).sort_values(["tier", "category", "label", "title"]).reset_index(drop=True)
    manifest_df.to_csv(manifest_path, index=False)

    _build_curated_package(track, package_root, log_path)

    _log(log_path, f"[DONE] Paper package ready: {package_root}")
    _log(log_path, f"[DONE] Main figures: {figs_main}")
    _log(log_path, f"[DONE] Supplementary figures: {figs_supp}")
    _log(log_path, f"[DONE] Main tables: {tables_main}")
    _log(log_path, f"[DONE] Asset manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a publication-oriented Q1 paper package from final-phase outputs."
    )
    ap.add_argument("--track", required=True, choices=["smears", "biopsy", "all"])
    ap.add_argument("--out-dir", default=None, help="Optional package output directory. For --track all, this is treated as the common root.")
    ap.add_argument("--save-latex", action="store_true", help="Export newly generated package tables as LaTeX alongside CSV.")
    ap.add_argument("--resume", action="store_true", help="Skip rebuilding a package when its core assets already exist.")
    ap.add_argument("--force", action="store_true", help="Overwrite package assets even when they already exist.")
    ap.add_argument("--skip-evaluation", action="store_true", help="Do not auto-run run_full_evaluation.py when evaluation outputs are missing.")
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def main() -> None:
    _set_style()
    args = parse_args()

    if args.track == "all":
        root = Path(args.out_dir).resolve() if args.out_dir else (DATA_OUT / "paper_package_all").resolve()
        for track in TRACKS:
            _build_track_package(track, root / track, args)
    else:
        out_root = Path(args.out_dir).resolve() if args.out_dir else (DATA_OUT / args.track / "paper_package").resolve()
        _build_track_package(args.track, out_root, args)


if __name__ == "__main__":
    main()
