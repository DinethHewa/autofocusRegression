#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils_smears import derive_group_id_from_path, dump_json, group_safe_split, normalize_columns, safe_mkdir

DEFAULT_MANIFEST = Path("/home/dineth/focus_measure/journal/Regression/data/manifest_smears_all.csv")
DEFAULT_SPLITS_DIR = Path("/home/dineth/focus_measure/journal/Regression/data/out_final_phase/smears/splits")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create leakage-safe SMEARS train/val/test splits.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Input unified smears manifest CSV")
    ap.add_argument("--out-dir", default=str(DEFAULT_SPLITS_DIR), help="Output splits directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--resume", action="store_true", help="Reuse existing split CSVs/config when present.")
    return ap.parse_args()


def _prepare_group_id(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if "group_id" in out.columns and out["group_id"].notna().any():
        out["group_id"] = out["group_id"].astype(str)
        missing = out["group_id"].isna() | out["group_id"].eq("")
        if missing.any():
            out.loc[missing, "group_id"] = out.loc[missing, "image_path"].astype(str).map(derive_group_id_from_path)
            return out, "group_id_filled_with_parent_for_missing"
        return out, "group_id_existing"

    out["group_id"] = out["image_path"].astype(str).map(derive_group_id_from_path)
    return out, "group_id_derived_from_image_path_parent"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Unified manifest missing: {manifest_path}. Run build_manifest_smears_all.py first."
        )

    out_dir = safe_mkdir(args.out_dir)
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    cfg_path = out_dir / "split_config.json"
    if args.resume and train_path.is_file() and val_path.is_file() and test_path.is_file() and cfg_path.is_file():
        print(f"[INFO] Resume: split outputs already exist under {out_dir}; skipping.")
        return

    df = pd.read_csv(manifest_path)
    df = normalize_columns(df)

    required = ["image_path", "defocus_um", "dataset"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Unified manifest missing required columns: {missing}. "
            f"Fix in build_manifest_smears_all.py generation."
        )

    before = len(df)
    df["image_path"] = df["image_path"].astype(str).str.strip()
    df["defocus_um"] = pd.to_numeric(df["defocus_um"], errors="coerce")

    keep = df["image_path"].ne("") & df["defocus_um"].notna()
    dropped = int((~keep).sum())
    if dropped:
        df = df[keep].copy()

    df, group_rule = _prepare_group_id(df)

    split = (float(args.split[0]), float(args.split[1]), float(args.split[2]))
    train_df, val_df, test_df = group_safe_split(
        df,
        group_col="group_id",
        split=split,
        seed=int(args.seed),
    )

    cols_preferred = [
        "image_path",
        "dataset",
        "defocus_um",
        "group_id",
        "fov_id",
        "stack_id",
        "roi_importance",
    ]
    cols = [c for c in cols_preferred if c in df.columns]

    train_df[cols].to_csv(train_path, index=False)
    val_df[cols].to_csv(val_path, index=False)
    test_df[cols].to_csv(test_path, index=False)

    cfg = {
        "seed": int(args.seed),
        "split": list(split),
        "group_id_rule": group_rule,
        "input_manifest": str(manifest_path),
        "rows_before_filter": int(before),
        "rows_after_filter": int(len(df)),
        "rows_dropped_missing_image_or_defocus": int(dropped),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(train_df["group_id"].astype(str).nunique()),
        "val_groups": int(val_df["group_id"].astype(str).nunique()),
        "test_groups": int(test_df["group_id"].astype(str).nunique()),
        "datasets": sorted(df["dataset"].astype(str).unique().tolist()),
    }

    dump_json(cfg, cfg_path)

    print(f"[DONE] Wrote split file: {train_path} rows={len(train_df)}")
    print(f"[DONE] Wrote split file: {val_path} rows={len(val_df)}")
    print(f"[DONE] Wrote split file: {test_path} rows={len(test_df)}")
    print(f"[DONE] Wrote split config: {cfg_path}")


if __name__ == "__main__":
    main()
