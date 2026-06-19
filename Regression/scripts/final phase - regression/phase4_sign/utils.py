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

from common_paths import DATA_DIR

TRACK_DATASETS = {
    "smears": ["pbs", "wbc", "bma"],
    "biopsy": ["focus_train", "focus_test"],
}


def default_cache_index_path(track: str) -> Path:
    return Path(DATA_DIR) / "cache_phase3" / track / "cache_index.csv"


def default_sign_out_dir(track: str) -> Path:
    return Path(DATA_DIR) / "out_final_phase" / track / "sign"


def resolve_out_dir(track: str, out_dir: str | None = None) -> Path:
    base = default_sign_out_dir(track).resolve()
    if out_dir is None:
        return base
    candidate = Path(out_dir).resolve()
    if candidate == base or base in candidate.parents:
        return candidate
    raise ValueError(f"out-dir must be {base} or a subdirectory under it.")


def ensure_output_tree(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "models": out_dir / "models",
        "splits": out_dir / "splits",
        "metrics": out_dir / "metrics",
        "calibration": out_dir / "calibration",
        "inference": out_dir / "inference",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(data: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _infer_group_id(df: pd.DataFrame) -> pd.Series:
    if "group_id" in df.columns and df["group_id"].notna().any():
        return df["group_id"].astype(str)
    if "slide_id" in df.columns and df["slide_id"].notna().any():
        return df["slide_id"].astype(str)
    if "patient_id" in df.columns and df["patient_id"].notna().any():
        return df["patient_id"].astype(str)
    if "source_image_path" in df.columns and df["source_image_path"].notna().any():
        return df["source_image_path"].astype(str).map(lambda p: Path(p).parent.name)
    if "image_path" in df.columns and df["image_path"].notna().any():
        return df["image_path"].astype(str).map(lambda p: Path(p).parent.name)
    return df["cache_path_XA"].astype(str).map(lambda p: Path(p).parent.name)


def _require_cols(df: pd.DataFrame, cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def load_cache_index(track: str, cache_index_path: str | None = None) -> pd.DataFrame:
    if track not in TRACK_DATASETS:
        raise ValueError(f"Unsupported track: {track}")

    path = default_cache_index_path(track) if cache_index_path is None else Path(cache_index_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cache index not found: {path}. Run Phase-3 first to build cache_phase3/{track}/cache_index.csv"
        )

    df = pd.read_csv(path)
    _require_cols(df, ["roi_uid", "cache_path_XA", "dataset"], str(path))

    if "track" in df.columns:
        bad = df[(df["track"].notna()) & (df["track"].astype(str) != track)]
        if not bad.empty:
            raise ValueError(
                f"Cache index has rows from another track ({bad['track'].unique().tolist()}); refusing to mix tracks."
            )

    allowed = set(TRACK_DATASETS[track])
    ds_values = set(df["dataset"].dropna().astype(str).unique().tolist())
    if not ds_values.issubset(allowed):
        raise ValueError(f"Datasets not allowed for track={track}: {sorted(ds_values - allowed)}")

    df = df[df["dataset"].astype(str).isin(allowed)].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No cache rows left for track={track} after dataset filtering.")

    df["roi_uid"] = df["roi_uid"].astype(str)
    dup_count = int(df["roi_uid"].duplicated().sum())
    if dup_count > 0:
        print(f"[WARN] Duplicate roi_uid rows found: {dup_count}. Keeping first occurrence.")
        df = df.drop_duplicates(subset=["roi_uid"], keep="first").reset_index(drop=True)

    if "roi_importance" not in df.columns:
        df["roi_importance"] = 1.0
    df["roi_importance"] = pd.to_numeric(df["roi_importance"], errors="coerce").fillna(1.0).astype(float)

    df["group_id"] = _infer_group_id(df)
    if df["group_id"].isna().any():
        raise ValueError("Could not infer group_id for all rows.")

    # Preserve path as string for deterministic joins and loading
    df["cache_path_XA"] = df["cache_path_XA"].astype(str)
    return df


def _find_first_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_sign_labels(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    z_idx_col = _find_first_col(work, ["z_index", "z", "z_idx"])
    z_best_col = _find_first_col(work, ["z_best", "z_star", "z*", "focus_index", "best_z"])

    if z_idx_col is not None and z_best_col is not None:
        z_idx = pd.to_numeric(work[z_idx_col], errors="coerce")
        z_best = pd.to_numeric(work[z_best_col], errors="coerce")
        delta = z_idx - z_best
    elif "delta_z" in work.columns:
        delta = pd.to_numeric(work["delta_z"], errors="coerce")
    elif "defocus_um" in work.columns:
        # If z-index labels are unavailable, allow signed defocus value as proxy.
        delta = pd.to_numeric(work["defocus_um"], errors="coerce")
    else:
        raise ValueError(
            "Cannot build sign labels. Need (z_index and z_best) or delta_z or defocus_um in cache index."
        )

    work["delta_z"] = delta
    bad = int(work["delta_z"].isna().sum())
    if bad > 0:
        print(f"[WARN] Dropping rows with invalid delta_z: {bad}")
        work = work[work["delta_z"].notna()].copy()

    work = work[work["delta_z"] != 0].copy()
    if work.empty:
        raise ValueError("No rows left after dropping delta_z == 0 for sign training.")

    work["y_sign"] = (work["delta_z"] > 0).astype(int)
    work["abs_delta_z"] = np.abs(work["delta_z"].astype(float))
    return work.reset_index(drop=True)


def group_split(
    df: pd.DataFrame,
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_r, val_r, test_r = split
    total = train_r + val_r + test_r
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {split}")

    if "group_id" not in df.columns:
        raise ValueError("group_id column is required for leakage-safe group split.")

    groups = sorted(df["group_id"].astype(str).unique().tolist())
    if len(groups) < 3:
        raise ValueError(f"Need at least 3 unique groups for train/val/test split, got {len(groups)}")

    rng = np.random.default_rng(seed)
    groups_arr = np.array(groups, dtype=object)
    rng.shuffle(groups_arr)
    groups = groups_arr.tolist()

    n = len(groups)
    n_train = int(round(train_r * n))
    n_val = int(round(val_r * n))
    n_test = n - n_train - n_val

    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = max(1, n_test)

    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    while n_train + n_val + n_test < n:
        n_train += 1

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val : n_train + n_val + n_test])

    train_df = df[df["group_id"].astype(str).isin(train_groups)].copy().reset_index(drop=True)
    val_df = df[df["group_id"].astype(str).isin(val_groups)].copy().reset_index(drop=True)
    test_df = df[df["group_id"].astype(str).isin(test_groups)].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            f"Invalid split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
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


def split_train_val_by_group(df: pd.DataFrame, val_ratio: float = 0.1765, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "group_id" not in df.columns:
        raise ValueError("group_id required")
    groups = sorted(df["group_id"].astype(str).unique().tolist())
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for train/val split")

    rng = np.random.default_rng(seed)
    arr = np.array(groups, dtype=object)
    rng.shuffle(arr)
    groups = arr.tolist()

    n_val = max(1, int(round(val_ratio * len(groups))))
    n_val = min(n_val, len(groups) - 1)

    val_groups = set(groups[:n_val])
    train_groups = set(groups[n_val:])

    train_df = df[df["group_id"].astype(str).isin(train_groups)].copy().reset_index(drop=True)
    val_df = df[df["group_id"].astype(str).isin(val_groups)].copy().reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Train/val split failed due to empty subset")
    return train_df, val_df


def load_XA_tensor(path: str | Path, expected_shape: tuple[int, int, int] = (200, 200, 3)) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"XA tensor not found: {p}")

    if p.suffix.lower() == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif p.suffix.lower() == ".npz":
        npz = np.load(p, allow_pickle=False)
        if "XA" in npz.files:
            arr = npz["XA"]
        elif npz.files:
            arr = npz[npz.files[0]]
        else:
            raise ValueError(f"NPZ file has no arrays: {p}")
    else:
        raise ValueError(f"Unsupported tensor extension: {p.suffix} ({p})")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != expected_shape:
        raise ValueError(f"XA shape mismatch for {p}: got {arr.shape}, expected {expected_shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"XA tensor has NaN/Inf: {p}")
    return arr


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    pos = y_true == 1
    neg = y_true == 0
    tpr = float((y_pred[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((y_pred[neg] == 0).mean()) if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def compute_per_bin_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abs_delta: np.ndarray,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    abs_delta = np.asarray(abs_delta, dtype=float)

    rows: list[dict] = []

    if bins is None:
        if np.allclose(abs_delta, np.round(abs_delta), atol=1e-6):
            values = sorted(np.unique(abs_delta.astype(int)).tolist())
            for v in values:
                idx = abs_delta.astype(int) == v
                if idx.sum() == 0:
                    continue
                acc = float((y_pred[idx] == y_true[idx]).mean())
                bal = compute_balanced_accuracy(y_true[idx], y_pred[idx])
                rows.append(
                    {
                        "bin": f"|dz|={v}",
                        "count": int(idx.sum()),
                        "accuracy": acc,
                        "balanced_accuracy": bal,
                    }
                )
            return pd.DataFrame(rows)

        bins = [0.0, 0.5, 1.0, 2.0, np.inf]

    if bins is None or len(bins) < 2:
        raise ValueError("bins must contain at least two edges")

    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (abs_delta > lo) & (abs_delta <= hi)
        if idx.sum() == 0:
            rows.append(
                {
                    "bin": f"({lo},{hi}]",
                    "count": 0,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                }
            )
            continue
        acc = float((y_pred[idx] == y_true[idx]).mean())
        bal = compute_balanced_accuracy(y_true[idx], y_pred[idx])
        rows.append(
            {
                "bin": f"({lo},{hi}]",
                "count": int(idx.sum()),
                "accuracy": acc,
                "balanced_accuracy": bal,
            }
        )

    return pd.DataFrame(rows)
