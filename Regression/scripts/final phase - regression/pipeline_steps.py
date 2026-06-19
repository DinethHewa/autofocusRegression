#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common_paths import manifest_path, table_path


def _required_cols(df: pd.DataFrame, cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def _read_csv_or_fail(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    return pd.read_csv(path)


def load_and_validate_manifests(datasets: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ds in datasets:
        man_path = Path(manifest_path(ds))
        tab_path = Path(table_path(ds))
        manifest_df = _read_csv_or_fail(man_path)
        table_df = _read_csv_or_fail(tab_path)

        _required_cols(manifest_df, ["image_path", "defocus_um"], str(man_path))
        _required_cols(table_df, ["image_path"], str(tab_path))

        manifest_df = manifest_df.copy()
        table_df = table_df.copy()

        manifest_df["image_path"] = manifest_df["image_path"].astype(str)
        table_df["image_path"] = table_df["image_path"].astype(str)

        merged = manifest_df.merge(table_df, on="image_path", how="left", suffixes=("", "_table"))
        merged["dataset"] = ds

        missing_meta = int(merged.isna().all(axis=1).sum())
        if missing_meta > 0:
            print(f"[WARN] {ds}: {missing_meta} rows have no metadata in table merge.")

        frames.append(merged)

    if not frames:
        raise ValueError("No datasets were loaded.")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "image_path"]).reset_index(drop=True)
    df["defocus_um"] = pd.to_numeric(df["defocus_um"], errors="coerce")
    bad = int(df["defocus_um"].isna().sum())
    if bad:
        raise ValueError(f"Found {bad} rows with invalid defocus_um.")

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_z"] = pd.to_numeric(out["defocus_um"], errors="coerce")
    if out["delta_z"].isna().any():
        raise ValueError("delta_z contains NaN after conversion.")

    out["y_sign"] = np.sign(out["delta_z"]).astype(int)
    out["y_mag"] = np.abs(out["delta_z"]).astype(float)
    out["is_zero_sign"] = (out["delta_z"] == 0).astype(int)
    return out


def add_group_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "patient_id" in out.columns and out["patient_id"].notna().any():
        out["group_id"] = out["patient_id"].astype(str)
        out["group_source"] = "patient_id"
        return out

    if "slide_id" in out.columns and out["slide_id"].notna().any():
        out["group_id"] = out["slide_id"].astype(str)
        out["group_source"] = "slide_id"
        return out

    if "stack_id" in out.columns and out["stack_id"].notna().any():
        out["group_id"] = out["stack_id"].astype(str)
        out["group_source"] = "stack_id"
        return out

    out["group_id"] = out["image_path"].astype(str).map(lambda p: str(Path(p).parent))
    out["group_source"] = "image_path_parent"
    print("[WARN] Using image_path parent folder as group_id (fallback).")
    return out


def _choose_fov_key(df: pd.DataFrame) -> str:
    for col in ["roi_fov_id", "fov_id", "stack_id", "group_id", "slide_id", "patient_id"]:
        if col in df.columns:
            return col
    raise ValueError("Could not determine FOV key for ROI gating.")


def _derive_fov_from_image_path(path: str) -> str:
    """
    Deterministic fallback FOV id from image path.
    Prefer parent folder identity; for flat folders, collapse common frame suffixes.
    """
    p = Path(str(path))
    parent = str(p.parent)
    stem = p.stem

    # Common autofocus naming patterns:
    # - *_page_<n>
    # - *_<index>
    # - *defocus-<value>
    stem = re.sub(r"_page_\d+$", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    stem = re.sub(r"defocus[-_]\d+(?:\.\d+)?$", "defocus", stem)

    return f"{parent}::{stem}"


def roi_gate(df: pd.DataFrame, k: int, min_rois_per_fov: int = 2) -> pd.DataFrame:
    out = df.copy()
    fov_key = _choose_fov_key(out)
    if fov_key in out.columns:
        uniq_ratio = float(out[fov_key].astype(str).nunique()) / max(len(out), 1)
    else:
        uniq_ratio = 1.0

    # If chosen key is effectively row-unique, derive a stable fallback from image_path.
    if uniq_ratio > 0.95 and "image_path" in out.columns:
        out["_derived_fov_id"] = out["image_path"].astype(str).map(_derive_fov_from_image_path)
        fov_key = "_derived_fov_id"
        print("[WARN] FOV key was near-unique; using derived fallback from image_path.")

    if "roi_importance" not in out.columns:
        print("[WARN] roi_importance column missing. Defaulting to uniform importance=1.0.")
        out["roi_importance"] = 1.0
    out["roi_importance"] = pd.to_numeric(out["roi_importance"], errors="coerce").fillna(0.0)

    counts = out.groupby(fov_key)[fov_key].transform("count")
    sparse_mask = counts < int(min_rois_per_fov)
    sparse_rows = int(sparse_mask.sum())
    if sparse_rows > 0:
        print(f"[WARN] Dropping sparse ROI rows: {sparse_rows} (min_rois_per_fov={min_rois_per_fov}).")
        out = out.loc[~sparse_mask].copy()

    if k <= 0:
        raise ValueError("k must be > 0 for ROI gating.")

    out = out.sort_values([fov_key, "roi_importance"], ascending=[True, False]).copy()
    out["_roi_rank"] = out.groupby(fov_key).cumcount() + 1
    drop_cols = ["_roi_rank"]
    if "_derived_fov_id" in out.columns:
        drop_cols.append("_derived_fov_id")
    gated = out[out["_roi_rank"] <= int(k)].drop(columns=drop_cols).reset_index(drop=True)

    if gated.empty:
        raise ValueError("ROI gating removed all rows. Check k/min_rois_per_fov and input data.")

    print(f"[INFO] ROI gating kept {len(gated)} / {len(df)} rows with k={k}.")
    return gated


def make_group_splits(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    work = df.copy()
    _required_cols(work, ["group_id"], "split input")

    groups = work["group_id"].dropna().astype(str).unique().tolist()
    if not groups:
        raise ValueError("No valid group_id values for splitting.")

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n >= 3 else 1
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    def _slice(group_set: set[str]) -> pd.DataFrame:
        return work[work["group_id"].astype(str).isin(group_set)].copy().reset_index(drop=True)

    train_df = _slice(train_groups)
    val_df = _slice(val_groups)
    test_df = _slice(test_groups)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Invalid split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

    leakage = (set(train_df["group_id"]) & set(val_df["group_id"])) | (set(train_df["group_id"]) & set(test_df["group_id"])) | (set(val_df["group_id"]) & set(test_df["group_id"]))
    if leakage:
        raise RuntimeError(f"Group leakage detected across splits: {sorted(leakage)[:5]}")

    print(
        f"[INFO] Group-safe splits: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"groups=({len(train_groups)},{len(val_groups)},{len(test_groups)})"
    )
    return train_df, val_df, test_df


def write_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "split_train.csv"
    val_path = out_dir / "split_val.csv"
    test_path = out_dir / "split_test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[DONE] Wrote split files to {out_dir}")
    return {"train": str(train_path), "val": str(val_path), "test": str(test_path)}


def write_combined_manifest(df: pd.DataFrame, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DONE] Wrote combined manifest: {out_path}")
    return str(out_path)


def write_config_json(config_dict: dict, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[DONE] Wrote config: {out_path}")
    return str(out_path)
