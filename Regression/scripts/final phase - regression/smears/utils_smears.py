#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def derive_group_id_from_path(path: str) -> str:
    p = Path(str(path))
    return p.parent.name if p.parent.name else str(p)


def group_safe_split(
    df: pd.DataFrame,
    group_col: str,
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr, va, te = split
    if not np.isclose(tr + va + te, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {split}")

    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    groups = np.array(sorted(df[group_col].astype(str).dropna().unique().tolist()), dtype=object)
    if len(groups) < 3:
        raise ValueError(f"Need at least 3 unique groups, got {len(groups)}")

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_tr = max(1, int(round(tr * n)))
    n_va = max(1, int(round(va * n)))
    n_te = max(1, n - n_tr - n_va)

    while n_tr + n_va + n_te > n:
        if n_tr > 1:
            n_tr -= 1
        elif n_va > 1:
            n_va -= 1
        else:
            n_te -= 1
    while n_tr + n_va + n_te < n:
        n_tr += 1

    g_tr = set(groups[:n_tr].tolist())
    g_va = set(groups[n_tr : n_tr + n_va].tolist())
    g_te = set(groups[n_tr + n_va : n_tr + n_va + n_te].tolist())

    train_df = df[df[group_col].astype(str).isin(g_tr)].copy().reset_index(drop=True)
    val_df = df[df[group_col].astype(str).isin(g_va)].copy().reset_index(drop=True)
    test_df = df[df[group_col].astype(str).isin(g_te)].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            f"Split produced empty subset: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

    leak = (
        set(train_df[group_col].astype(str)) & set(val_df[group_col].astype(str))
    ) | (
        set(train_df[group_col].astype(str)) & set(test_df[group_col].astype(str))
    ) | (
        set(val_df[group_col].astype(str)) & set(test_df[group_col].astype(str))
    )
    if leak:
        raise RuntimeError(f"Group leakage detected: {sorted(leak)[:10]}")

    return train_df, val_df, test_df


def command_runner(cmd: list[str], log_path: str | Path | None = None) -> None:
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"[RUN] {cmd_str}")

    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(cmd_str + "\n")

    subprocess.run(cmd, check=True)


def dump_json(payload: dict, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
