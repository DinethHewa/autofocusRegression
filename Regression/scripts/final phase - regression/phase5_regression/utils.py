#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_paths import DATA_DIR, manifest_path

TRACK_DATASETS = {
    "smears": ["pbs", "wbc", "bma"],
    "biopsy": ["focus_train", "focus_test"],
}


def fail_missing_columns(df: pd.DataFrame, required: list[str], where: str, fix_hint: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {where}: {missing}. "
            f"Fix in {fix_hint}."
        )


def save_json(payload: dict, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def default_phase5_out_dir(track: str) -> Path:
    return Path(DATA_DIR) / "out_final_phase" / track / "regression"


def default_cache_index_path(track: str) -> Path:
    return Path(DATA_DIR) / "cache_phase3" / track / "cache_index.csv"


def default_phase5_index_path(track: str) -> Path:
    return default_phase5_out_dir(track) / "index_phase5.csv"


def resolve_reg_out_dir(track: str, out_dir: str | None = None) -> Path:
    base = default_phase5_out_dir(track).resolve()
    if out_dir is None:
        return base
    cand = Path(out_dir).resolve()
    if cand == base or base in cand.parents:
        return cand
    raise ValueError(f"out-dir must be {base} or its subdirectory.")


def ensure_reg_output_tree(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "models": out_dir / "models",
        "splits": out_dir / "splits",
        "metrics": out_dir / "metrics",
        "triplets": out_dir / "triplets",
        "inference": out_dir / "inference",
        "calibration": out_dir / "calibration",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _infer_group_id(df: pd.DataFrame) -> pd.Series:
    for c in ["group_id", "slide_id", "patient_id", "stack_id"]:
        if c in df.columns and df[c].notna().any():
            return df[c].astype(str)
    if "source_image_path" in df.columns and df["source_image_path"].notna().any():
        return df["source_image_path"].astype(str).map(lambda x: Path(x).parent.name)
    if "image_path" in df.columns and df["image_path"].notna().any():
        return df["image_path"].astype(str).map(lambda x: Path(x).parent.name)
    return df["roi_uid"].astype(str).map(lambda x: x.split("_")[0])


def _infer_fov_id(df: pd.DataFrame) -> pd.Series:
    for c in ["fov_id", "source_image_path", "stack_id", "group_id"]:
        if c in df.columns and df[c].notna().any():
            return df[c].astype(str)
    if "image_path" in df.columns and df["image_path"].notna().any():
        return df["image_path"].astype(str)
    return df["roi_uid"].astype(str)


def load_manifest(dataset: str) -> pd.DataFrame:
    p = Path(manifest_path(dataset))
    if not p.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {p}. Fix by running manifest creation for dataset '{dataset}'."
        )
    df = pd.read_csv(p, low_memory=False)
    fail_missing_columns(
        df,
        required=["image_path", "defocus_um"],
        where=str(p),
        fix_hint=f"manifest creation script for {dataset}",
    )
    out = df[["image_path", "defocus_um"]].copy()
    out["image_path"] = out["image_path"].astype(str)
    out["defocus_um"] = pd.to_numeric(out["defocus_um"], errors="coerce")
    bad = int(out["defocus_um"].isna().sum())
    if bad > 0:
        raise ValueError(
            f"Manifest {p} has {bad} invalid defocus_um values. "
            f"Fix in manifest generation source."
        )
    out = out.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return out


def load_cache_index(track: str, cache_index_path: str | None = None) -> pd.DataFrame:
    if track not in TRACK_DATASETS:
        raise ValueError(f"Unsupported track: {track}")

    p = default_cache_index_path(track) if cache_index_path is None else Path(cache_index_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Run Phase 3 to generate X_B cache before Phase 5. Missing: {p}"
        )

    df = pd.read_csv(p)
    fail_missing_columns(
        df,
        required=["roi_uid", "dataset", "cache_path_XB"],
        where=str(p),
        fix_hint="Phase 3 cache index generation",
    )

    allowed = set(TRACK_DATASETS[track])
    ds = set(df["dataset"].dropna().astype(str).unique().tolist())
    if not ds.issubset(allowed):
        raise ValueError(f"Track mixing detected in cache index: {sorted(ds - allowed)}")

    df = df[df["dataset"].astype(str).isin(allowed)].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for track={track} in {p}")

    if "track" in df.columns and df["track"].notna().any():
        bad = df[df["track"].astype(str) != track]
        if not bad.empty:
            raise ValueError(
                f"cache_index contains other track values: {bad['track'].unique().tolist()}"
            )

    df["roi_uid"] = df["roi_uid"].astype(str)
    if "roi_importance" not in df.columns:
        df["roi_importance"] = 1.0
    df["roi_importance"] = pd.to_numeric(df["roi_importance"], errors="coerce").fillna(1.0)

    return df


def _merge_one_dataset(cache_ds: pd.DataFrame, manifest_ds: pd.DataFrame, dataset: str) -> pd.DataFrame:
    cache = cache_ds.copy()
    man = manifest_ds.copy()

    join_used = None
    if "image_path" in cache.columns:
        cache["image_path"] = cache["image_path"].astype(str)
        merged = cache.merge(man, on="image_path", how="left", suffixes=("", "_manifest"))
        join_used = "cache.image_path == manifest.image_path"
    elif "source_image_path" in cache.columns:
        cache["source_image_path"] = cache["source_image_path"].astype(str)
        merged = cache.merge(
            man.rename(columns={"image_path": "source_image_path"}),
            on="source_image_path",
            how="left",
            suffixes=("", "_manifest"),
        )
        if "image_path" not in merged.columns:
            merged["image_path"] = merged["source_image_path"]
        join_used = "cache.source_image_path == manifest.image_path"
    elif "parent_image_id" in cache.columns:
        cache["parent_image_id"] = cache["parent_image_id"].astype(str)
        man2 = man.copy()
        man2["parent_image_id"] = man2["image_path"].map(lambda p: Path(p).name)
        merged = cache.merge(man2[["parent_image_id", "defocus_um", "image_path"]], on="parent_image_id", how="left")
        join_used = "cache.parent_image_id == basename(manifest.image_path)"
    else:
        raise ValueError(
            f"Cannot join cache to manifest for dataset={dataset}. "
            f"Need one of cache columns: image_path, source_image_path, parent_image_id."
        )

    miss = int(merged["defocus_um"].isna().sum())
    if miss > 0:
        sample_cols = [c for c in ["roi_uid", "image_path", "source_image_path", "parent_image_id"] if c in merged.columns]
        sample = merged[merged["defocus_um"].isna()][sample_cols].head(5).to_dict(orient="records")
        raise ValueError(
            f"Failed to map defocus_um for dataset={dataset}: {miss}/{len(merged)} rows unmatched. "
            f"Join used: {join_used}. Sample unmatched: {sample}"
        )

    return merged


def build_phase5_index_from_sources(track: str, cache_df: pd.DataFrame) -> pd.DataFrame:
    datasets = TRACK_DATASETS[track]
    parts: list[pd.DataFrame] = []

    for ds in datasets:
        cds = cache_df[cache_df["dataset"].astype(str) == ds].copy()
        if cds.empty:
            continue
        mds = load_manifest(ds)
        merged = _merge_one_dataset(cds, mds, ds)
        parts.append(merged)

    if not parts:
        raise ValueError(f"No rows available after joining cache and manifests for track={track}")

    out = pd.concat(parts, ignore_index=True)
    out["group_id"] = _infer_group_id(out)
    out["fov_id"] = _infer_fov_id(out)

    out["defocus_um"] = pd.to_numeric(out["defocus_um"], errors="coerce")
    bad = int(out["defocus_um"].isna().sum())
    if bad > 0:
        raise ValueError(f"Phase5 index has {bad} invalid defocus_um rows after join.")

    out["y_sign"] = (out["defocus_um"] > 0).astype(int)
    out["y_mag_um"] = np.abs(out["defocus_um"]).astype(float)

    required = ["roi_uid", "dataset", "group_id", "fov_id", "cache_path_XB", "roi_importance", "defocus_um", "y_sign", "y_mag_um"]
    fail_missing_columns(out, required, where="phase5 merged index", fix_hint="manifest/cache index generation")

    # Validate XB path existence
    path_exists = out["cache_path_XB"].astype(str).map(lambda p: Path(p).is_file())
    miss_paths = int((~path_exists).sum())
    if miss_paths > 0:
        sample = out.loc[~path_exists, "cache_path_XB"].astype(str).head(5).tolist()
        raise FileNotFoundError(
            f"Missing cache_path_XB files: {miss_paths}. Sample: {sample}. "
            f"Run Phase 3 to generate X_B cache before Phase 5."
        )

    keep_cols = [
        "roi_uid",
        "dataset",
        "group_id",
        "fov_id",
        "cache_path_XB",
        "roi_importance",
        "defocus_um",
        "y_sign",
        "y_mag_um",
        "image_path",
        "source_image_path",
        "stack_id",
        "patch_id",
        "cache_path_XA",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]

    out = out[keep_cols].drop_duplicates(subset=["roi_uid"]).reset_index(drop=True)
    return out


def load_phase5_index(track: str, phase5_index_path: str | None = None) -> pd.DataFrame:
    p = default_phase5_index_path(track) if phase5_index_path is None else Path(phase5_index_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Phase5 index not found: {p}. Run build_phase5_index.py first."
        )

    df = pd.read_csv(p)
    required = [
        "roi_uid",
        "dataset",
        "group_id",
        "fov_id",
        "cache_path_XB",
        "roi_importance",
        "defocus_um",
        "y_sign",
        "y_mag_um",
    ]
    fail_missing_columns(
        df,
        required=required,
        where=str(p),
        fix_hint="build_phase5_index.py",
    )

    allowed = set(TRACK_DATASETS[track])
    ds = set(df["dataset"].dropna().astype(str).unique().tolist())
    if not ds.issubset(allowed):
        raise ValueError(f"Track mixing in phase5 index: {sorted(ds - allowed)}")

    df["roi_uid"] = df["roi_uid"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    df["fov_id"] = df["fov_id"].astype(str)
    df["roi_importance"] = pd.to_numeric(df["roi_importance"], errors="coerce").fillna(1.0)
    df["defocus_um"] = pd.to_numeric(df["defocus_um"], errors="coerce")
    df["y_mag_um"] = pd.to_numeric(df["y_mag_um"], errors="coerce")
    df["y_sign"] = pd.to_numeric(df["y_sign"], errors="coerce")

    bad = int(df[["defocus_um", "y_mag_um", "y_sign"]].isna().any(axis=1).sum())
    if bad > 0:
        raise ValueError(f"Phase5 index contains {bad} invalid label rows")

    return df


def group_split(
    df: pd.DataFrame,
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr, vr, te = split
    if not np.isclose(tr + vr + te, 1.0):
        raise ValueError(f"split must sum to 1.0, got {split}")
    if "group_id" not in df.columns:
        raise ValueError("group_id is required for leakage-safe split")

    groups = np.array(sorted(df["group_id"].astype(str).unique().tolist()), dtype=object)
    if len(groups) < 3:
        raise ValueError(f"Need >=3 groups for split. Got {len(groups)}")

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = max(1, int(round(n * tr)))
    n_val = max(1, int(round(n * vr)))
    n_test = max(1, n - n_train - n_val)

    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    while n_train + n_val + n_test < n:
        n_train += 1

    g_train = set(groups[:n_train].tolist())
    g_val = set(groups[n_train : n_train + n_val].tolist())
    g_test = set(groups[n_train + n_val : n_train + n_val + n_test].tolist())

    train_df = df[df["group_id"].astype(str).isin(g_train)].copy().reset_index(drop=True)
    val_df = df[df["group_id"].astype(str).isin(g_val)].copy().reset_index(drop=True)
    test_df = df[df["group_id"].astype(str).isin(g_test)].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            f"Empty split detected: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

    leak = (
        set(train_df["group_id"].astype(str)) & set(val_df["group_id"].astype(str))
    ) | (
        set(train_df["group_id"].astype(str)) & set(test_df["group_id"].astype(str))
    ) | (
        set(val_df["group_id"].astype(str)) & set(test_df["group_id"].astype(str))
    )
    if leak:
        raise RuntimeError(f"Group leakage detected: {sorted(leak)[:10]}")

    return train_df, val_df, test_df


def split_train_val_by_group(df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.array(sorted(df["group_id"].astype(str).unique().tolist()), dtype=object)
    if len(groups) < 2:
        raise ValueError("Need >=2 groups for train/val split")

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_val = max(1, int(round(val_ratio * len(groups))))
    n_val = min(n_val, len(groups) - 1)

    g_val = set(groups[:n_val].tolist())
    g_train = set(groups[n_val:].tolist())

    train_df = df[df["group_id"].astype(str).isin(g_train)].copy().reset_index(drop=True)
    val_df = df[df["group_id"].astype(str).isin(g_val)].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise RuntimeError("Train/val split produced empty subset")
    return train_df, val_df


def _load_array_from_npy_or_npz(path: Path, preferred_key: str | None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr)

    if path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=False)
        if preferred_key and preferred_key in npz.files:
            return np.asarray(npz[preferred_key])
        if npz.files:
            return np.asarray(npz[npz.files[0]])
        raise ValueError(f"NPZ has no arrays: {path}")

    raise ValueError(f"Unsupported tensor extension: {path.suffix} ({path})")


def load_XB_tensor(path: str | Path, expected_shape: tuple[int, int, int] = (200, 200, 4)) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"XB tensor missing: {p}")
    arr = _load_array_from_npy_or_npz(p, preferred_key="XB")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != expected_shape:
        raise ValueError(f"XB shape mismatch at {p}: got {arr.shape}, expected {expected_shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"XB contains NaN/Inf: {p}")
    return arr


def load_XA_tensor(path: str | Path, expected_shape: tuple[int, int, int] = (200, 200, 3)) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"XA tensor missing: {p}")
    arr = _load_array_from_npy_or_npz(p, preferred_key="XA")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != expected_shape:
        raise ValueError(f"XA shape mismatch at {p}: got {arr.shape}, expected {expected_shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"XA contains NaN/Inf: {p}")
    return arr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mae = float(np.mean(np.abs(y_true - y_pred))) if y_true.size else np.nan
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred)))) if y_true.size else np.nan
    return {"mae_um": mae, "rmse_um": rmse}


def per_bin_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, bin_width_um: float = 0.5) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.size == 0:
        return pd.DataFrame(columns=["bin_id", "range_um", "count", "mae_um", "rmse_um"])

    bin_ids = np.floor(y_true / float(bin_width_um)).astype(int)
    rows = []
    for b in sorted(np.unique(bin_ids).tolist()):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue
        lo = b * float(bin_width_um)
        hi = (b + 1) * float(bin_width_um)
        m = regression_metrics(y_true[idx], y_pred[idx])
        rows.append(
            {
                "bin_id": int(b),
                "range_um": f"[{lo:.3f},{hi:.3f})",
                "count": int(idx.sum()),
                "mae_um": m["mae_um"],
                "rmse_um": m["rmse_um"],
            }
        )
    return pd.DataFrame(rows)


def weighted_median(values: np.ndarray, weights: np.ndarray, eps: float = 1e-6) -> float | None:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return None

    v = v[valid]
    w = w[valid]
    order = np.argsort(v)
    v = v[order]
    w = w[order]

    csum = np.cumsum(w)
    cutoff = 0.5 * (csum[-1] + eps)
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def parse_voted_sign_value(value) -> int | None:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"1", "pos", "plus", "positive", "true"}:
        return 1
    if s in {"0", "neg", "minus", "negative", "false"}:
        return 0
    if s in {"uncertain", "none", "nan", ""}:
        return None
    try:
        f = float(s)
        if np.isclose(f, 1.0):
            return 1
        if np.isclose(f, 0.0):
            return 0
    except Exception:
        pass
    return None
