#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils_smears import dump_json, normalize_columns, safe_mkdir

DATA_DIR = Path("/home/dineth/focus_measure/journal/Regression/data")
TABLES_DIR = Path("/home/dineth/focus_measure/journal/Regression/tables/FINAL_PHASE/SMEARS")

MANIFESTS = {
    "pbs": DATA_DIR / "manifest_pbs.csv",
    "wbc": DATA_DIR / "manifest_wbc.csv",
    "bma": DATA_DIR / "manifest_bma.csv",
}

REQUIRED = ["image_path", "defocus_um", "dataset"]
OPTIONAL = ["slide_id", "patient_id", "group_id", "fov_id", "stack_id", "roi_importance"]


def _load_one(dataset: str, path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required manifest: {path}")

    raw = pd.read_csv(path)
    original_cols = [str(c) for c in raw.columns]
    df = normalize_columns(raw)

    missing_required = [c for c in ["image_path", "defocus_um"] if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Manifest {path} missing required columns: {missing_required}. "
            f"Fix manifest creation for dataset '{dataset}'."
        )

    created_cols: list[str] = []

    # Enforce dataset field to source dataset (prevents accidental mixing).
    if "dataset" not in df.columns:
        created_cols.append("dataset")
    df["dataset"] = dataset

    for c in OPTIONAL:
        if c not in df.columns:
            created_cols.append(c)
            if c == "roi_importance":
                df[c] = 1.0
            else:
                df[c] = np.nan

    before = len(df)
    df["image_path"] = df["image_path"].astype(str).str.strip()
    empty_path = df["image_path"].eq("") | df["image_path"].isna()
    dropped_path = int(empty_path.sum())
    if dropped_path:
        df = df[~empty_path].copy()

    df["defocus_um"] = pd.to_numeric(df["defocus_um"], errors="coerce")
    bad_defocus = df["defocus_um"].isna()
    dropped_defocus = int(bad_defocus.sum())
    if dropped_defocus:
        df = df[~bad_defocus].copy()

    after = len(df)

    report = {
        "dataset": dataset,
        "path": str(path),
        "original_columns": original_cols,
        "normalized_columns": [str(c) for c in df.columns],
        "created_columns": created_cols,
        "rows_before": int(before),
        "rows_after": int(after),
        "dropped_missing_image_path": int(dropped_path),
        "dropped_invalid_defocus_um": int(dropped_defocus),
        "defocus_um_min": float(df["defocus_um"].min()) if after else None,
        "defocus_um_max": float(df["defocus_um"].max()) if after else None,
    }

    # keep all columns but ensure required+optional lead
    col_order = REQUIRED + [c for c in OPTIONAL if c not in REQUIRED]
    extra_cols = [c for c in df.columns if c not in col_order]
    df = df[col_order + extra_cols].copy()

    return df.reset_index(drop=True), report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build unified SMEARS manifest from PBS/WBC/BMA manifests.")
    ap.add_argument(
        "--output-manifest",
        default=str(DATA_DIR / "manifest_smears_all.csv"),
        help="Output path for unified manifest",
    )
    ap.add_argument(
        "--report-json",
        default=str(TABLES_DIR / "manifest_smears_report.json"),
        help="Output path for manifest report json",
    )
    ap.add_argument("--resume", action="store_true", help="Reuse existing manifest/report outputs when present.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_manifest = Path(args.output_manifest)
    report_path = Path(args.report_json)
    if args.resume and out_manifest.is_file() and report_path.is_file():
        print(f"[INFO] Resume: unified manifest/report already exist; skipping.")
        print(f"[INFO] Existing manifest: {out_manifest}")
        print(f"[INFO] Existing report: {report_path}")
        return

    frames = []
    reports = []
    for ds, path in MANIFESTS.items():
        df, rep = _load_one(ds, path)
        frames.append(df)
        reports.append(rep)

    all_cols = sorted({c for f in frames for c in f.columns})
    aligned = []
    missing_summary: dict[str, list[str]] = {}
    for rep, df in zip(reports, frames):
        ds = rep["dataset"]
        missing = [c for c in all_cols if c not in df.columns]
        missing_summary[ds] = missing
        for c in missing:
            if c == "roi_importance":
                df[c] = 1.0
            else:
                df[c] = np.nan
        aligned.append(df[all_cols].copy())

    merged = pd.concat(aligned, ignore_index=True)

    safe_mkdir(out_manifest.parent)
    merged.to_csv(out_manifest, index=False)

    report_payload = {
        "output_manifest": str(out_manifest),
        "total_rows": int(len(merged)),
        "row_counts_by_dataset": {d: int((merged["dataset"] == d).sum()) for d in MANIFESTS.keys()},
        "defocus_um_min_by_dataset": {
            d: (float(merged.loc[merged["dataset"] == d, "defocus_um"].min()) if (merged["dataset"] == d).any() else None)
            for d in MANIFESTS.keys()
        },
        "defocus_um_max_by_dataset": {
            d: (float(merged.loc[merged["dataset"] == d, "defocus_um"].max()) if (merged["dataset"] == d).any() else None)
            for d in MANIFESTS.keys()
        },
        "per_manifest": reports,
        "missing_column_summary_after_alignment": missing_summary,
        "final_columns": [str(c) for c in merged.columns],
    }

    safe_mkdir(report_path.parent)
    dump_json(report_payload, report_path)

    print(f"[DONE] Wrote unified manifest: {out_manifest} rows={len(merged)}")
    print(f"[DONE] Wrote manifest report: {report_path}")


if __name__ == "__main__":
    main()
